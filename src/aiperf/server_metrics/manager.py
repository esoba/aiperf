# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import CommAddress, CommandType
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_command, on_stop
from aiperf.common.messages import (
    ProfileCancelCommand,
    ProfileCompleteCommand,
    ProfileConfigureCommand,
    ProfileStartCommand,
    ServerMetricsRecordMessage,
    ServerMetricsStatusMessage,
)
from aiperf.common.metric_utils import normalize_metrics_endpoint_url
from aiperf.common.models import ErrorDetails, ServerMetricsRecord
from aiperf.common.protocols import PushClientProtocol
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServerMetricsCollectorType
from aiperf.server_metrics.data_collector import ServerMetricsDataCollector
from aiperf.server_metrics.protocols import (
    SYSTEM_METRICS_SOURCE_IDENTIFIER,
    ServerMetricsCollectorProtocol,
)


class ServerMetricsManager(BaseComponentService):
    """Coordinates server metrics collectors for server-side metrics collection.

    Supports multiple collector types via the plugin system, running simultaneously:
    - prometheus (default): Scrapes Prometheus-compatible HTTP endpoints
    - system (additive): Collects system-wide CPU, memory, and per-process GPU VRAM

    This service:
    - Manages lifecycle of collector instances (any ServerMetricsCollectorProtocol)
    - Sends ServerMetricsRecordsMessage to RecordsManager via message system
    - Handles errors gracefully with ErrorDetails

    Args:
        service_config: Service-level configuration (logging, communication, etc.)
        user_config: User-provided configuration including server_metrics endpoints
        service_id: Optional unique identifier for this service instance
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
        )

        self.records_push_client: PushClientProtocol = self.comms.create_push_client(
            CommAddress.RECORDS,
        )

        self._collectors: dict[str, ServerMetricsCollectorProtocol] = {}
        self._server_metrics_disabled = user_config.server_metrics_disabled
        self._system_metrics_enabled = user_config.system_metrics_enabled

        # Collect metrics from all endpoint URLs (for multi-URL load balancing)
        self._server_metrics_endpoints: list[str] = []
        for url in user_config.endpoint.urls:
            normalized_url = normalize_metrics_endpoint_url(url)
            if normalized_url not in self._server_metrics_endpoints:
                self._server_metrics_endpoints.append(normalized_url)
        self.info(
            f"Server Metrics: Discovered {len(self._server_metrics_endpoints)} endpoints: {self._server_metrics_endpoints}"
        )

        # Add user-specified URLs if provided
        if user_config.server_metrics_urls:
            for url in user_config.server_metrics_urls:
                normalized_url = normalize_metrics_endpoint_url(url)
                if normalized_url not in self._server_metrics_endpoints:
                    self._server_metrics_endpoints.append(normalized_url)

        self._collection_interval = Environment.SERVER_METRICS.COLLECTION_INTERVAL

        # Task for delayed shutdown, created when no endpoints are reachable
        self._shutdown_task: asyncio.Task[None] | None = None

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure server metrics collectors (additive: Prometheus + optional system).

        Always attempts Prometheus endpoint discovery. Additionally configures the
        system-level collector if --server-metrics system was specified.

        Args:
            message: Profile configuration command from SystemController
        """
        if self._server_metrics_disabled:
            await self._send_server_metrics_status(
                enabled=False,
                reason="disabled via --no-server-metrics",
                endpoints_configured=[],
                endpoints_reachable=[],
            )
            return

        self._collectors.clear()

        await self._configure_prometheus_collectors()

        if self._system_metrics_enabled:
            await self._configure_system_collector()

        all_configured = list(self._server_metrics_endpoints)
        if self._system_metrics_enabled:
            all_configured.append(SYSTEM_METRICS_SOURCE_IDENTIFIER)

        if not self._collectors:
            await self._send_server_metrics_status(
                enabled=False,
                reason="no collectors available",
                endpoints_configured=all_configured,
                endpoints_reachable=[],
            )
            return

        # Capture baseline metrics before profiling starts
        self.info("Server Metrics: Capturing baseline metrics...")
        for endpoint_url, collector in self._collectors.items():
            try:
                await collector.initialize()
                await collector.collect_and_process_metrics()
                self.debug(
                    lambda url=endpoint_url: f"Server Metrics: Captured baseline from {url}"
                )
            except Exception as e:
                self.warning(
                    f"Server Metrics: Failed to capture baseline from {endpoint_url}: {e}"
                )

        await self._send_server_metrics_status(
            enabled=True,
            reason=None,
            endpoints_configured=all_configured,
            endpoints_reachable=list(self._collectors.keys()),
        )

    async def _configure_prometheus_collectors(self) -> None:
        """Configure Prometheus-based collectors for each endpoint URL."""
        for endpoint_url in self._server_metrics_endpoints:
            self.debug(
                lambda url=endpoint_url: f"Server Metrics: Testing reachability of {url}"
            )
            collector = ServerMetricsDataCollector(
                endpoint_url=endpoint_url,
                collection_interval=self._collection_interval,
                record_callback=self._on_server_metrics_records,
                error_callback=self._on_server_metrics_error,
                collector_id=endpoint_url,
            )

            try:
                is_reachable = await collector.is_url_reachable()
                if is_reachable:
                    self._collectors[endpoint_url] = collector
                    self.debug(
                        lambda url=endpoint_url: f"Server Metrics: Prometheus endpoint {url} is reachable"
                    )
                else:
                    self.debug(
                        lambda url=endpoint_url: f"Server Metrics: Prometheus endpoint {url} is not reachable"
                    )
            except Exception as e:
                self.error(f"Server Metrics: Exception testing {endpoint_url}: {e}")

    async def _configure_system_collector(self) -> None:
        """Configure the system-level metrics collector (psutil + optional pynvml)."""
        collector_cls = plugins.get_class(
            PluginType.SERVER_METRICS_COLLECTOR,
            ServerMetricsCollectorType.SYSTEM,
        )
        collector = collector_cls(
            collection_interval=self._collection_interval,
            record_callback=self._on_server_metrics_records,
            error_callback=self._on_server_metrics_error,
        )

        try:
            is_reachable = await collector.is_url_reachable()
            if not is_reachable:
                self.warning(
                    "Server Metrics: System metrics collector is not available"
                )
                return
        except Exception as e:
            self.error(f"Server Metrics: Exception testing system collector: {e}")
            return

        self._collectors[collector.endpoint_url] = collector
        self.info("Server Metrics: System metrics collector ready")

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message: ProfileStartCommand) -> None:
        """Start all server metrics collectors for profiling phase.

        Initializes and starts background collection tasks for each configured
        collector. Handles partial failures gracefully - continues profiling if
        at least one collector starts successfully, only shuts down if all fail.

        If no collectors exist (all endpoints were unreachable during configuration),
        performs graceful shutdown.

        Args:
            message: Profile start command from SystemController signaling
                    that profiling phase is beginning
        """
        if not self._collectors:
            # Server metrics disabled status already sent in _profile_configure_command, only shutdown here
            self._shutdown_task = asyncio.create_task(self._delayed_shutdown())
            return

        started_count = 0
        for endpoint_url, collector in self._collectors.items():
            try:
                await collector.start()
                started_count += 1
            except Exception as e:
                self.error(f"Failed to start collector for {endpoint_url}: {e}")

        total_collectors = len(self._collectors)
        if started_count == 0:
            self.warning("No server metrics collectors successfully started")
            all_configured = list(self._server_metrics_endpoints)
            if self._system_metrics_enabled:
                all_configured.append(SYSTEM_METRICS_SOURCE_IDENTIFIER)
            await self._send_server_metrics_status(
                enabled=False,
                reason="all collectors failed to start",
                endpoints_configured=all_configured,
                endpoints_reachable=[],
            )
            self._shutdown_task = asyncio.create_task(self._delayed_shutdown())
            return
        elif started_count < total_collectors:
            self.warning(
                f"Partial collector startup: {started_count}/{total_collectors} collectors started successfully"
            )
        else:
            self.info(
                f"Server Metrics: Started {started_count} collector(s) successfully"
            )

    @on_command(CommandType.PROFILE_COMPLETE)
    async def _handle_profile_complete_command(
        self, message: ProfileCompleteCommand
    ) -> None:
        """Trigger final scrape when profiling completes.

        Performs one final metrics collection from all endpoints to capture
        the end state immediately after profiling finishes. This ensures we
        have metrics that cover the entire profiling period, including any
        counter/histogram changes that occurred during the final seconds.

        Critical for accurate delta calculations on counters and histograms,
        where missing the final state would undercount the actual activity.

        Idempotent: Can be called multiple times safely (e.g., if multiple
        RecordsManager instances send the command). Subsequent calls are no-ops.

        Args:
            message: Profile complete command from RecordsManager signaling that
                    all client request records have been processed
        """
        # Idempotent check - skip if already stopped or no collectors
        if not self._collectors:
            self.debug("Server Metrics: Already stopped, skipping final scrape")
            return

        self.info("Server Metrics: Profiling complete, capturing final metrics...")

        # Trigger final scrape from all collectors
        for endpoint_url, collector in list(self._collectors.items()):
            try:
                await collector.collect_and_process_metrics()
                self.debug(
                    lambda url=endpoint_url: f"Server Metrics: Captured final state from {url}"
                )
            except Exception as e:
                self.warning(
                    f"Server Metrics: Failed to capture final state from {endpoint_url}: {e}"
                )

        # Stop all collectors after final scrape
        await self._stop_all_collectors()

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        """Stop all server metrics collectors when profiling is cancelled.

        Called when user cancels profiling or an error occurs during profiling.
        Waits for flush period to allow metrics to finalize, then stops collectors.

        Args:
            message: Profile cancel command from SystemController
        """
        await self._stop_all_collectors()

    @on_stop
    async def _server_metrics_manager_stop(self) -> None:
        """Stop all server metrics collectors during service shutdown.

        Called automatically by BaseComponentService lifecycle management via @on_stop hook.
        Ensures all collectors are properly stopped and cleaned up even if shutdown
        command was not received.
        """
        await self._stop_all_collectors()

    async def _stop_all_collectors(self) -> None:
        """Stop all server metrics collectors.

        Attempts to stop each collector gracefully, logging errors but continuing with
        remaining collectors to ensure all resources are released. Does nothing if no
        collectors are configured.

        Errors during individual collector shutdown do not prevent other collectors
        from being stopped.
        """
        if not self._collectors:
            return

        # Copy the collectors to a list to avoid modifying the dictionary while iterating
        # Also enabled idempotent check to avoid stopping collectors multiple times
        collectors = list(self._collectors.items())
        self._collectors.clear()

        for endpoint_url, collector in collectors:
            try:
                await collector.stop()
            except Exception as e:
                self.error(f"Failed to stop collector for {endpoint_url}: {e}")

    async def _delayed_shutdown(self) -> None:
        """Shutdown service after a delay to allow command response to be sent.

        Waits before calling stop() to ensure the command response
        has time to be published and transmitted to the SystemController.
        """
        await asyncio.sleep(Environment.SERVER_METRICS.SHUTDOWN_DELAY)
        await asyncio.shield(self.stop())

    async def _on_server_metrics_records(
        self, records: list[ServerMetricsRecord], collector_id: str
    ) -> None:
        """Async callback for receiving server metrics records from collectors.

        Called by collector instances when they successfully collect metrics.
        Forwards records to RecordsManager via ZMQ push socket, preserving all
        metadata for hierarchical storage and processing.

        Args:
            records: List of ServerMetricsRecord objects from a collection cycle
            collector_id: Unique identifier of the collector
        """
        if not records:
            return

        for record in records:
            try:
                message = ServerMetricsRecordMessage(
                    service_id=self.service_id,
                    collector_id=collector_id,
                    record=record,
                    error=None,
                )

                await self.records_push_client.push(message)

            except Exception as e:
                self.error(
                    f"Failed to send server metrics record from {collector_id}: {e}"
                )
                # Send error message to RecordsManager to track the failure
                try:
                    error_message = ServerMetricsRecordMessage(
                        service_id=self.service_id,
                        collector_id=collector_id,
                        record=None,
                        error=ErrorDetails.from_exception(e),
                    )
                    await self.records_push_client.push(error_message)
                except Exception as nested_error:
                    self.error(
                        f"Failed to send error message after record send failure: {nested_error}"
                    )

    async def _on_server_metrics_error(
        self, error: ErrorDetails, collector_id: str
    ) -> None:
        """Async callback for receiving server metrics errors from collectors.

        Called by collectors when collection fails. Forwards error to RecordsManager
        for tracking and reporting.

        Args:
            error: ErrorDetails describing the collection error
            collector_id: Unique identifier of the collector
        """
        try:
            error_message = ServerMetricsRecordMessage(
                service_id=self.service_id,
                collector_id=collector_id,
                record=None,
                error=error,
            )

            await self.records_push_client.push(error_message)

        except Exception as e:
            self.error(f"Failed to send server metrics error message: {e}")

    async def _send_server_metrics_status(
        self,
        enabled: bool,
        reason: str | None = None,
        endpoints_configured: list[str] | None = None,
        endpoints_reachable: list[str] | None = None,
    ) -> None:
        """Send server metrics status message to SystemController.

        Args:
            enabled: Whether server metrics collection is enabled/available
            reason: Optional human-readable reason for status
            endpoints_configured: List of endpoint URLs configured
            endpoints_reachable: List of endpoint URLs that are accessible
        """
        try:
            status_message = ServerMetricsStatusMessage(
                service_id=self.service_id,
                enabled=enabled,
                reason=reason,
                endpoints_configured=endpoints_configured or [],
                endpoints_reachable=endpoints_reachable or [],
            )

            await self.publish(status_message)

        except Exception as e:
            self.error(f"Failed to send server metrics status message: {e}")
