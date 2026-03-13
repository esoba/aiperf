# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import os
import sys
import time
import traceback
import uuid
from collections.abc import Iterable

import orjson
from msgspec import Struct
from rich.console import Console
from rich.panel import Panel

from aiperf.cli_utils import (
    print_developer_mode_warning,
    warn_osl_without_ignore_eos,
)
from aiperf.common.base_service import BaseService
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.config.zmq_config import ZMQDualBindConfig
from aiperf.common.control_structs import (
    Command,
    CommandAck,
    CommandErr,
    CommandOk,
    CommandResponse,
    ControllerBoundMessage,
    Heartbeat,
    MemoryReport,
    Registration,
    RegistrationAck,
    ServerMetricsStatus,
    StatusUpdate,
    TelemetryStatus,
)
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    LifecycleState,
    MessageType,
)
from aiperf.common.environment import Environment
from aiperf.common.error_queue import (
    ErrorCollector,
    cleanup_global_error_queue,
)
from aiperf.common.exceptions import (
    LifecycleOperationError,
    ServiceRegistrationTimeoutError,
)
from aiperf.common.hooks import (
    AIPerfHook,
    on_command,
    on_init,
    on_message,
    on_start,
    on_stop,
)
from aiperf.common.logging import cleanup_global_log_queue, get_global_log_queue
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.memory_tracker import (
    MemoryPhase,
    MemoryReading,
    MemoryTracker,
    read_pss_self,
)
from aiperf.common.messages import (
    BenchmarkCompleteMessage,
    ProcessRecordsResultMessage,
    ProcessServerMetricsResultMessage,
    ProcessTelemetryResultMessage,
)
from aiperf.common.models import (
    ErrorDetails,
    ProcessRecordsResult,
)
from aiperf.common.models.error_models import ExitErrorInfo
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.types import ServiceTypeT
from aiperf.controller.controller_utils import print_exit_errors
from aiperf.controller.protocols import ServiceManagerProtocol
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.controller.system_mixins import SignalHandlerMixin
from aiperf.credit.messages import CreditsCompleteMessage
from aiperf.exporters.exporter_manager import ExporterManager
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceRunType, ServiceType, UIType
from aiperf.ui.protocols import AIPerfUIProtocol
from aiperf.workers.scaling import (
    calculate_record_processor_count,
    calculate_worker_count,
)
from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient


class SystemController(SignalHandlerMixin, BaseService):
    """System Controller service.

    This service is responsible for managing the lifecycle of all other services.
    It will start, stop, and configure all other services.
    """

    def __init__(
        self,
        config=None,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            service_id=service_id,
            **kwargs,
        )
        self.debug("Creating System Controller")
        if Environment.DEV.MODE:
            # Print a warning message to the console if developer mode is enabled, once at load time
            print_developer_mode_warning()

        # EOS may cause server to stop early, producing misleading OSL results
        if self._should_warn_osl_without_ignore_eos():
            warn_osl_without_ignore_eos()

        self._was_cancelled = False
        is_k8s_mode = self.service_config.service_run_type == ServiceRunType.KUBERNETES

        self.required_services: dict[ServiceTypeT, int] = {
            ServiceType.TIMING_MANAGER: 1,
            ServiceType.WORKER_MANAGER: 1,
            ServiceType.RECORDS_MANAGER: 1,
        }

        num_workers = calculate_worker_count(self.config, self.config)
        if self.service_config.record_processor_service_count is not None:
            num_record_processors = self.service_config.record_processor_service_count
        else:
            num_record_processors = calculate_record_processor_count(num_workers)
        self.required_services[ServiceType.WORKER] = num_workers
        self.required_services[ServiceType.RECORD_PROCESSOR] = num_record_processors

        self.proxy_manager: ProxyManager = ProxyManager(
            config=self.config,
            enable_event_bus=True,
            enable_dataset_manager=True,
            enable_raw_inference=not is_k8s_mode,
        )

        # Control ROUTER lives outside the comms lifecycle so it stays
        # alive after comms.stop() — child processes still need it during
        # their own shutdown sequence.
        additional_bind: str | None = None
        comm_config = self.service_config.comm_config
        if (
            isinstance(comm_config, ZMQDualBindConfig)
            and not comm_config.controller_host
        ):
            additional_bind = comm_config.control_tcp_bind_address

        control_address = self.comms.get_address(CommAddress.CONTROL)
        self.info(
            f"Creating control ROUTER client: "
            f"address={control_address}, additional_bind={additional_bind}"
        )
        import zmq as _zmq

        self.control_router = ZMQStreamingRouterClient(
            address=control_address,
            bind=True,
            additional_bind_address=additional_bind,
            decode_type=ControllerBoundMessage,
            socket_ops={_zmq.ROUTER_MANDATORY: 1},
        )

        ServiceManagerClass = plugins.get_class(
            PluginType.SERVICE_MANAGER, self.service_config.service_run_type
        )

        using_dashboard = self.service_config.ui_type == UIType.DASHBOARD
        log_queue = get_global_log_queue() if using_dashboard else None
        self._error_collector = ErrorCollector(
            logger=self, exit_errors=self._exit_errors
        )

        self.service_manager: ServiceManagerProtocol = ServiceManagerClass(
            required_services=self.required_services,
            config=self.config,
            log_queue=log_queue,
            error_queue=self._error_collector.error_queue,
        )
        UIClass = plugins.get_class(PluginType.UI, self.service_config.ui_type)
        self.ui: AIPerfUIProtocol = UIClass(
            service_config=self.service_config,
            user_config=self.user_config,
            log_queue=log_queue,
            controller=self,
        )
        self.attach_child_lifecycle(self.ui)
        self._profile_results: ProcessRecordsResult | None = None
        self._exit_errors: list[ExitErrorInfo] = []
        self._telemetry_results: TelemetryExportData | None = None
        self._server_metrics_results: ServerMetricsResults | None = None
        self._profile_results_received = False
        self._should_wait_for_telemetry = False
        self._should_wait_for_server_metrics = False

        self._shutdown_triggered = False
        self._shutdown_lock = asyncio.Lock()
        self._memory_tracker = MemoryTracker()

        # Configure-on-register: when enabled, each service receives
        # PROFILE_CONFIGURE immediately upon registration instead of
        # waiting for all services to register first.
        self._auto_configure: bool = False
        self._configuring_ids: set[str] = set()
        self._configured_ids: set[str] = set()
        self._all_configured_event: asyncio.Event = asyncio.Event()
        self._configure_errors: list[CommandResponse | ErrorDetails] = []
        self._configure_scheduler: LoopScheduler | None = None

        self._telemetry_endpoints_configured: list[str] = []
        self._telemetry_endpoints_reachable: list[str] = []
        self._server_metrics_endpoints_configured: list[str] = []
        self._server_metrics_endpoints_reachable: list[str] = []
        self._pod_failure_watcher_task: asyncio.Task | None = None
        self.debug("System Controller created")

    def _should_warn_osl_without_ignore_eos(self) -> bool:
        """Check if --osl is used without ignore_eos or min_tokens in extra inputs."""
        osl_mean = self.user_config.input.prompt.output_tokens.mean
        if osl_mean is None:
            return False

        extra_inputs = self.user_config.input.extra
        if not extra_inputs:
            return True

        # Check if ignore_eos or min_tokens is set with a truthy value
        extra_dict = dict(extra_inputs)
        return not (extra_dict.get("ignore_eos") or extra_dict.get("min_tokens"))

    async def request_realtime_metrics(self) -> None:
        """Request real-time metrics from the RecordsManager."""
        rm_ids = [
            s.service_id
            for s in ServiceRegistry.get_services(ServiceType.RECORDS_MANAGER)
        ]
        for sid in rm_ids:
            await self._send_control_command(
                sid, CommandType.REALTIME_METRICS, timeout=5.0
            )

    async def start_realtime_telemetry(self) -> None:
        """Send START_REALTIME_TELEMETRY command to RecordsManager(s)."""
        rm_ids = [
            s.service_id
            for s in ServiceRegistry.get_services(ServiceType.RECORDS_MANAGER)
        ]
        for sid in rm_ids:
            await self._send_control_command(
                sid, CommandType.START_REALTIME_TELEMETRY, timeout=5.0
            )

    async def initialize(self) -> None:
        """We need to override the initialize method to run the proxy manager before the base service initialize.
        This is because the proxies need to be running before we can subscribe to the message bus.
        """
        self.debug("Running ZMQ Proxy Manager Before Initialize")
        await self.proxy_manager.initialize_and_start()
        # Once the proxies are running, call the original initialize method
        await super().initialize()

    @on_init
    async def _initialize_system_controller(self) -> None:
        self.debug("Initializing System Controller")

        # Register the unified receiver that dispatches by message type.
        self.control_router.register_receiver(self._handle_control_message)

        # Initialize and start the control ROUTER independently of comms.
        self.info("Initializing control ROUTER client")
        await self.control_router.initialize()
        self.info(
            f"Control ROUTER initialized (state={self.control_router.state}), starting..."
        )
        await self.control_router.start()
        self.info(f"Control ROUTER started (state={self.control_router.state})")

        self.setup_signal_handlers(self._handle_signal)
        self.debug("Setup signal handlers")

        async with self.try_operation_or_stop("Initialize Service Manager"):
            await self.service_manager.initialize()

        self.debug("System Controller initialized successfully")

    @on_start
    async def _start_services(self) -> None:
        """Bootstrap the system services.

        Services are configured immediately upon registration rather than
        waiting for all services to register first. This overlaps the
        registration and configuration phases for faster startup.
        """
        self.debug("System Controller is bootstrapping services")
        self._controller_pss_at_start = read_pss_self()

        # Enable auto-configure so that each service receives
        # PROFILE_CONFIGURE as soon as it registers.
        self._configure_scheduler = LoopScheduler()
        self._auto_configure = True

        # Collect optional services to spawn alongside required services
        optional_services: list[ServiceTypeT] = []
        if not self.user_config.gpu_telemetry_disabled:
            optional_services.append(ServiceType.GPU_TELEMETRY_MANAGER)
        else:
            self.info("GPU telemetry disabled via --no-gpu-telemetry")
            self._should_wait_for_telemetry = False

        if not self.user_config.server_metrics_disabled:
            optional_services.append(ServiceType.SERVER_METRICS_MANAGER)
        else:
            self.info("Server metrics disabled via --no-server-metrics")
            self._should_wait_for_server_metrics = False

        api_port = self.service_config.api_port or Environment.API_SERVER.PORT
        api_host = self.service_config.api_host or Environment.API_SERVER.HOST
        if api_port is not None and api_host is not None:
            self.info(f"Starting AIPerf API server at http://{api_host}:{api_port}/")
            optional_services.append(ServiceType.API)

        total_services = (
            1 + sum(self.required_services.values()) + len(optional_services)
        )
        types_summary = f"{ServiceType.DATASET_MANAGER}: 1, " + ", ".join(
            f"{st}: {n}" for st, n in self.required_services.items()
        )
        if optional_services:
            types_summary += ", " + ", ".join(f"{st}: 1" for st in optional_services)
        self.info(f"Spawning {total_services} services ({types_summary})")
        spawn_start = time.perf_counter()

        # Spawn dataset manager first so it can begin its heavy
        # configuration (tokenizer loading, dataset prep) while
        # the remaining services are still starting up.
        await self.service_manager.run_service(ServiceType.DATASET_MANAGER)

        async with self.try_operation_or_stop("Start Service Manager"):
            await self.service_manager.start()

        if optional_services:
            await asyncio.gather(
                *[self.service_manager.run_service(st) for st in optional_services]
            )

        spawn_elapsed = time.perf_counter() - spawn_start
        self.info(f"All {total_services} services spawned in {spawn_elapsed:.2f}s")

        # Enable pod monitoring early so failed pods are detected during
        # registration/configuration rather than waiting for timeout.
        self.service_manager.activate_pod_monitoring()

        self.info("AIPerf System is CONFIGURING")
        async with self.try_operation_or_stop("Configure Services"):
            await self._wait_for_all_configured(
                timeout=Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT,
            )
        self.info("AIPerf System is CONFIGURED")
        self._auto_configure = False
        self.service_manager.activate_heartbeat_monitoring()

        # Verify pod health before starting profiling. A pod could have
        # registered its services but since crashed (e.g. OOMKilled).
        async with self.try_operation_or_stop("Pod Health Check"):
            await self.service_manager.check_pods_healthy()

        await self._start_profiling_all_services()
        self.info("AIPerf System is PROFILING")

        # Watch for pod failure threshold breach during profiling
        self._pod_failure_watcher_task = asyncio.create_task(
            self._watch_pod_failure_abort()
        )

    async def _configure_single_service(self, service_id: str) -> None:
        """Send PROFILE_CONFIGURE to a single service and track completion.

        Called as a fire-and-forget task from the registration handler so that
        each service begins configuration immediately upon registering.
        """
        self._configuring_ids.add(service_id)
        self.debug(lambda: f"Sending PROFILE_CONFIGURE to '{service_id}'")
        try:
            response = await self._send_control_command(
                service_id,
                CommandType.PROFILE_CONFIGURE,
                timeout=Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT,
            )
            if isinstance(response, CommandErr):
                self._configure_errors.append(response)
                self._all_configured_event.set()
                return
        except Exception as e:
            self.error(f"PROFILE_CONFIGURE to '{service_id}' failed: {e}")
            self._configure_errors.append(ErrorDetails.from_exception(e))
            self._all_configured_event.set()
            return

        self._configured_ids.add(service_id)
        total = len(self._configuring_ids)
        self.info(f"Configured '{service_id}' ({len(self._configured_ids)}/{total})")
        if self._all_expected_configured():
            self._all_configured_event.set()

    def _all_expected_configured(self) -> bool:
        """Check if every expected service has been configured.

        Verifies both:
        - All individually-expected service IDs (from expect_service) are configured
        - All type-count expectations (from expect_services) are met
        """
        expected_ids = ServiceRegistry.expected_ids
        expected_by_type = ServiceRegistry.expected_by_type
        if not expected_ids and not expected_by_type:
            return False
        if not expected_ids.issubset(self._configured_ids):
            return False
        for stype, expected_count in expected_by_type.items():
            configured_count = sum(
                1
                for sid in self._configured_ids
                if sid in ServiceRegistry.services
                and ServiceRegistry.services[sid].service_type == stype
            )
            if configured_count < expected_count:
                return False
        return True

    def _get_pending_type_counts(self) -> dict[str, str]:
        """Get type counts that haven't reached their expected configured count."""
        pending: dict[str, str] = {}
        for stype, expected_count in ServiceRegistry.expected_by_type.items():
            configured_count = sum(
                1
                for sid in self._configured_ids
                if sid in ServiceRegistry.services
                and ServiceRegistry.services[sid].service_type == stype
            )
            if configured_count < expected_count:
                pending[str(stype)] = f"{configured_count}/{expected_count}"
        return pending

    def _cancel_configure_tasks(self) -> None:
        """Cancel in-flight configure tasks and clear tracking."""
        if self._configure_scheduler is not None:
            self._configure_scheduler.cancel_all()

    async def _wait_for_all_configured(self, timeout: float) -> None:
        """Wait until all expected services have been configured.

        Uses fail-fast: if any service returns an error during configuration,
        we abort immediately.

        The _all_configured_event is set by:
        - _configure_single_service: on success (all expected done) or error
        - _cancel_profiling: on Ctrl+C signal
        - ServiceRegistry.fail_service wakes this via _failure_event
        """
        begin = time.perf_counter()

        if not self._all_expected_configured():
            # Ensure ServiceRegistry has a failure event we can watch
            if ServiceRegistry._failure_event is None:
                ServiceRegistry._failure_event = asyncio.Event()
            failure_event = ServiceRegistry._failure_event

            progress_task = asyncio.create_task(
                self._log_configure_progress(begin, timeout)
            )
            try:
                # Wait for any of: all configured, service failure, or timeout
                config_waiter = asyncio.create_task(self._all_configured_event.wait())
                failure_waiter = asyncio.create_task(failure_event.wait())
                try:
                    done, pending = await asyncio.wait(
                        {config_waiter, failure_waiter},
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    config_waiter.cancel()
                    failure_waiter.cancel()

                if not done:
                    # Timeout — neither event fired
                    self._cancel_configure_tasks()
                    pending_ids = ServiceRegistry.expected_ids - self._configured_ids
                    pending_types = self._get_pending_type_counts()
                    raise ServiceRegistrationTimeoutError(
                        f"Timed out waiting for services to configure "
                        f"({len(self._configured_ids)} configured). "
                        f"Pending IDs: {pending_ids}, "
                        f"Pending types: {pending_types}",
                        missing={},
                    ) from None

                # Something woke us — check what
                self._cancel_configure_tasks()

                # Cancellation (Ctrl+C)
                if self._was_cancelled:
                    raise asyncio.CancelledError(
                        "Configuration interrupted by shutdown"
                    )

                # Service process died
                ServiceRegistry._raise_on_failure()

                # Configure task returned an error
                self._parse_control_responses_for_errors(
                    self._configure_errors, "Configure Profiling"
                )

                # Verify all expected services are actually configured.
                if not self._all_expected_configured():
                    pending_ids = ServiceRegistry.expected_ids - self._configured_ids
                    pending_types = self._get_pending_type_counts()
                    raise ServiceRegistrationTimeoutError(
                        f"Configuration wait ended but not all services "
                        f"configured. Pending IDs: {pending_ids}, "
                        f"Pending types: {pending_types}",
                        missing={},
                    ) from None

            finally:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task
        else:
            self._cancel_configure_tasks()
            self._parse_control_responses_for_errors(
                self._configure_errors, "Configure Profiling"
            )

        self.info(
            f"All services configured in {time.perf_counter() - begin:.2f} seconds"
        )

        if not Environment.HTTP.SSL_VERIFY:
            self.warning(
                "SSL certificate verification is DISABLED - this is insecure. "
                "This should only be used for testing in a trusted environment."
            )

    async def _log_configure_progress(self, begin: float, timeout: float) -> None:
        """Log periodic progress during configuration wait."""
        interval = 5.0
        while True:
            await asyncio.sleep(interval)
            elapsed = time.perf_counter() - begin
            pending_types = self._get_pending_type_counts()
            pending_ids = ServiceRegistry.expected_ids - self._configured_ids
            configured = len(self._configured_ids)
            total = ServiceRegistry._total_expected
            msg = (
                f"Waiting for configuration: {configured}/{total} "
                f"({elapsed:.1f}s elapsed). "
                f"Pending IDs: {pending_ids}, Pending types: {pending_types}"
            )
            pod_summary = self.service_manager.get_pod_summary()
            if pod_summary:
                msg += f", Pod states: {pod_summary}"
            self.info(msg)

    async def _start_profiling_all_services(self) -> None:
        """Tell all services to start profiling.

        Uses fail-fast behavior: if any service returns an error,
        we abort immediately without waiting for the remaining services.
        """
        self.debug("Sending PROFILE_START command to all services")
        responses = await self._send_control_command_to_all_fail_fast(
            CommandType.PROFILE_START,
            list(ServiceRegistry.get_all_registered_ids()),
            timeout=Environment.SERVICE.PROFILE_START_TIMEOUT,
        )
        self._parse_control_responses_for_errors(responses, "Start Profiling")
        self.info("All services started profiling successfully")

    def _parse_control_responses_for_errors(
        self,
        responses: list[CommandResponse | ErrorDetails],
        operation: str,
    ) -> None:
        """Parse control channel command responses for errors."""
        for response in responses:
            if isinstance(response, ErrorDetails):
                self._exit_errors.append(
                    ExitErrorInfo(
                        error_details=response, operation=operation, service_id=None
                    )
                )
            elif isinstance(response, CommandErr):
                self._exit_errors.append(
                    ExitErrorInfo(
                        error_details=ErrorDetails(
                            type="CommandError",
                            message=response.error,
                            traceback=response.traceback or None,
                        ),
                        operation=operation,
                        service_id=response.sid,
                    )
                )
        if self._exit_errors:
            raise LifecycleOperationError(
                operation=operation,
                original_exception=None,
                lifecycle_id=self.id,
            )

    @on_message(MessageType.CREDITS_COMPLETE)
    async def _process_credits_complete_message(
        self, message: CreditsCompleteMessage
    ) -> None:
        """Log receipt of credits complete message from a service.

        Args:
            message: The credits complete message to process
        """
        service_id = message.service_id
        self.info(f"Received credits complete from '{service_id}'")

    @on_command(CommandType.PROFILE_COMPLETE)
    async def _handle_profile_complete_relay(self, message: Command) -> None:
        """Relay PROFILE_COMPLETE from RecordsManager to GPU telemetry and server metrics services."""
        target_types = [
            ServiceType.GPU_TELEMETRY_MANAGER,
            ServiceType.SERVER_METRICS_MANAGER,
            ServiceType.WORKER_MANAGER,
        ]
        target_ids = []
        for stype in target_types:
            target_ids.extend(s.service_id for s in ServiceRegistry.get_services(stype))

        if target_ids:
            await self._send_control_command_to_all(
                CommandType.PROFILE_COMPLETE, target_ids, timeout=10.0
            )

    @on_message(MessageType.PROCESS_RECORDS_RESULT)
    async def _on_process_records_result_message(
        self, message: ProcessRecordsResultMessage
    ) -> None:
        """Handle a profile results message."""
        self.trace_or_debug(
            lambda: f"Received profile results message: {message}",
            lambda: (
                f"Received profile results message: {len(message.results.results.records) if message.results.results else 0} records"
            ),
        )
        if message.results.errors:
            self.error(
                f"Received process records result message with errors: {message.results.errors}"
            )

        self.debug(
            lambda: (
                f"Error summary: {message.results.results.error_summary if message.results.results else 'N/A'}"
            )
        )

        self._profile_results = message.results

        if not message.results.results:
            self.error(
                f"Received process records result message with no records: {message.results.results}"
            )

        self._profile_results_received = True
        # Coordinate with telemetry results before shutdown
        await self._check_and_trigger_shutdown()

    @on_message(MessageType.PROCESS_TELEMETRY_RESULT)
    async def _on_process_telemetry_result_message(
        self, message: ProcessTelemetryResultMessage
    ) -> None:
        """Handle a telemetry results message."""
        try:
            self.trace_or_debug(
                lambda: f"Received telemetry results message: {message}",
                lambda: (
                    f"Received telemetry results message: {len(message.telemetry_result.results.endpoints) if message.telemetry_result.results else 0} endpoints"
                ),
            )

            telemetry_results = message.telemetry_result.results
            if not telemetry_results:
                self.error(
                    f"Received process telemetry result message with no records: {telemetry_results}"
                )
            else:
                # Update endpoint info in the summary (TelemetryExportData structure)
                telemetry_results.summary.endpoints_configured = (
                    self._telemetry_endpoints_configured
                )
                telemetry_results.summary.endpoints_successful = (
                    self._telemetry_endpoints_reachable
                )

            self._telemetry_results = telemetry_results
        except Exception as e:
            self.exception(f"Error processing telemetry results message: {e!r}")
        finally:
            self._should_wait_for_telemetry = False
            await self._check_and_trigger_shutdown()

    @on_message(MessageType.PROCESS_SERVER_METRICS_RESULT)
    async def _on_process_server_metrics_result_message(
        self, message: ProcessServerMetricsResultMessage
    ) -> None:
        """Handle a server metrics results message."""
        try:
            self.trace_or_debug(
                lambda: f"Received server metrics results message: {message}",
                lambda: (
                    f"Received server metrics results message: {len(message.server_metrics_result.results.endpoint_summaries or {}) if message.server_metrics_result.results else 0} endpoints"
                ),
            )

            self.debug(
                lambda: (
                    f"Server metrics error summary: {message.server_metrics_result.results.error_summary if message.server_metrics_result.results else 'N/A'}"
                )
            )

            server_metrics_results = message.server_metrics_result.results

            if not server_metrics_results:
                self.debug(
                    f"Received process server metrics result message with no results: {server_metrics_results}"
                )
            else:
                server_metrics_results.endpoints_configured = (
                    self._server_metrics_endpoints_configured
                )
                server_metrics_results.endpoints_successful = (
                    self._server_metrics_endpoints_reachable
                )

            self._server_metrics_results = server_metrics_results
        except Exception as e:
            self.exception(f"Error processing server metrics results message: {e!r}")
        finally:
            self._should_wait_for_server_metrics = False
            await self._check_and_trigger_shutdown()

    async def _check_and_trigger_shutdown(self) -> None:
        """Check if all required results are received and trigger unified export + shutdown.

        Coordination logic:
        1. Always wait for profile results (ProcessRecordsResultMessage)
        2. If telemetry disabled OR telemetry results received → proceed
        3. If server metrics disabled OR server metrics results received → proceed
        4. Otherwise → wait (results arrive nearly simultaneously and will call this method again)

        Thread safety:
        Uses self._shutdown_lock to prevent race conditions when ProcessRecordsResultMessage,
        ProcessTelemetryResultMessage, and ProcessServerMetricsResultMessage arrive concurrently.
        The lock ensures atomic check-and-set of _shutdown_triggered, preventing double-triggering of stop().
        """
        self.debug(
            lambda: f"_check_and_trigger_shutdown: profile_received={self._profile_results_received}, "
            f"wait_telemetry={self._should_wait_for_telemetry}, telemetry_results={self._telemetry_results is not None}, "
            f"wait_server_metrics={self._should_wait_for_server_metrics}, server_metrics_results={self._server_metrics_results is not None}, "
            f"shutdown_triggered={self._shutdown_triggered}"
        )
        # Check if we should trigger shutdown (with lock protection)
        should_shutdown = False
        async with self._shutdown_lock:
            if self._shutdown_triggered:
                self.debug(
                    "_check_and_trigger_shutdown: shutdown already triggered, returning"
                )
                return

            if not self._profile_results_received:
                self.debug(
                    "_check_and_trigger_shutdown: profile results not received yet"
                )
                return

            telemetry_ready_for_shutdown = (
                not self._should_wait_for_telemetry
                or self._telemetry_results is not None
            )

            server_metrics_ready_for_shutdown = (
                not self._should_wait_for_server_metrics
                or self._server_metrics_results is not None
            )

            if telemetry_ready_for_shutdown and server_metrics_ready_for_shutdown:
                self._shutdown_triggered = True
                should_shutdown = True
                self.info("All results received, initiating shutdown")
            else:
                if not telemetry_ready_for_shutdown:
                    self.info("Waiting for telemetry results...")
                if not server_metrics_ready_for_shutdown:
                    self.info("Waiting for server metrics results...")

        # Call stop() OUTSIDE the lock to prevent deadlock
        if should_shutdown:
            self.debug("Calling self.stop()...")
            await asyncio.shield(self.stop())
            self.debug("self.stop() completed")

    async def _handle_signal(self, sig: int) -> None:
        """Handle received signals with two-stage cancellation.

        First Ctrl+C: Graceful cancel - stops issuing new credits, cancels
        in-flight requests, and writes results to files.

        Second Ctrl+C: Force quit - immediately terminates all processes.
        Results may be incomplete or not written.

        Args:
            sig: The signal number received
        """
        if self._was_cancelled:
            # SECOND Ctrl+C - Force quit immediately
            self._print_force_quit_warning()
            self.warning(f"Force quit requested (signal {sig})")
            await self._kill()
            return

        # FIRST Ctrl+C - Graceful cancel with warning
        self._print_cancel_warning()
        self.warning(f"Graceful shutdown requested (signal {sig})")
        await self._cancel_profiling()

    def _print_cancel_warning(self) -> None:
        """Print prominent warning panel on first Ctrl+C.

        Informs user that the benchmark is being cancelled gracefully and
        results are being processed. Also instructs how to force quit.

        Uses stderr to ensure visibility even when stdout is redirected or
        captured by the UI.
        """
        console = Console(file=sys.stderr, force_terminal=True)
        console.print()
        console.print(
            Panel(
                "[bold yellow]⚠️  BENCHMARK CANCELLED[/bold yellow]\n\n"
                "Stopping credit issuance and cancelling in-flight requests...\n"
                "Results will be written to files.\n\n"
                "[dim]Press Ctrl+C again to force quit immediately[/dim]\n"
                "[dim](results may be incomplete or not written)[/dim]",
                border_style="yellow",
                padding=(1, 2),
                title="[bold yellow]Cancellation in Progress[/bold yellow]",
            )
        )
        console.print()
        console.file.flush()

    def _print_force_quit_warning(self) -> None:
        """Print warning panel on second Ctrl+C (force quit).

        Warns user that results may be incomplete due to immediate termination.

        Uses stderr to ensure visibility even when stdout is redirected or
        captured by the UI.
        """
        console = Console(file=sys.stderr, force_terminal=True)
        console.print()
        console.print(
            Panel(
                "[bold red]🛑 FORCE QUIT[/bold red]\n\n"
                "Terminating all processes immediately.\n"
                "Results may be incomplete or not written to files.",
                border_style="red",
                padding=(1, 2),
                title="[bold red]Force Quit[/bold red]",
            )
        )
        console.print()
        console.file.flush()

    async def _watch_pod_failure_abort(self) -> None:
        """Watch for pod failure threshold breach and cancel profiling."""
        await self.service_manager.pod_failure_abort_event.wait()
        if self._was_cancelled or self._shutdown_triggered:
            return
        reason = self.service_manager.pod_failure_abort_reason
        self.error(f"Aborting benchmark: {reason}")
        await self._cancel_profiling()

    async def _cancel_profiling(self) -> None:
        self.debug("Cancelling profiling of all services")
        self._was_cancelled = True
        if self._pod_failure_watcher_task and not self._pod_failure_watcher_task.done():
            self._pod_failure_watcher_task.cancel()
        self._cancel_configure_tasks()
        self._all_configured_event.set()
        self.service_manager.notify_shutdown()

        # Mark shutdown as triggered FIRST to prevent _check_and_trigger_shutdown()
        # from also calling stop() when results arrive during cancellation.
        should_call_stop = False
        async with self._shutdown_lock:
            if not self._shutdown_triggered:
                self._shutdown_triggered = True
                should_call_stop = True
            else:
                self.debug("Shutdown already triggered, skipping stop() call")

        # Send cancel to all registered services. Wait only for RecordsManager
        # response since it returns ProcessRecordsResult.
        all_ids = list(ServiceRegistry.get_all_registered_ids())
        records_manager_ids = {
            s.service_id
            for s in ServiceRegistry.get_services(ServiceType.RECORDS_MANAGER)
        }
        self.debug(
            f"Sending cancel to {len(all_ids)} services, waiting for {len(records_manager_ids)} RecordsManager(s)"
        )

        try:
            # Fire-and-forget cancel to non-RecordsManager services
            non_rm_ids = [sid for sid in all_ids if sid not in records_manager_ids]
            for sid in non_rm_ids:
                with contextlib.suppress(Exception):
                    await self.control_router.send_to(
                        sid,
                        Command(
                            cid=uuid.uuid4().hex,
                            cmd=CommandType.PROFILE_CANCEL,
                        ),
                    )

            # Wait for RecordsManager responses (they return ProcessRecordsResult)
            responses = await self._send_control_command_to_all(
                CommandType.PROFILE_CANCEL,
                list(records_manager_ids),
                timeout=Environment.SERVICE.PROFILE_CANCEL_TIMEOUT,
            )

            for response in responses:
                if isinstance(response, ErrorDetails):
                    self.warning(
                        f"Cancel command error (timeout or service unavailable): {response}"
                    )
                elif isinstance(response, CommandErr):
                    self.warning(
                        f"Cancel command failed from {response.sid}: {response.error}"
                    )

            # Extract ProcessRecordsResult from RecordsManager's CommandOk response
            for response in responses:
                if isinstance(response, CommandOk) and response.payload:
                    try:
                        data = orjson.loads(response.payload)
                        result = ProcessRecordsResult.model_validate(data)
                        self.debug(
                            f"Received ProcessRecordsResult from cancel command: {result}"
                        )
                        self._profile_results = result
                        self._profile_results_received = True
                        break
                    except Exception as e:
                        self.warning(f"Failed to parse cancel response payload: {e}")
        except Exception as e:
            self.warning(f"Exception during cancel command (proceeding to stop): {e!r}")

        if should_call_stop:
            self.debug("Stopping system controller after profiling cancelled")
            await asyncio.shield(self.stop())

    @on_stop
    async def _stop_system_controller(self) -> None:
        """Stop the system controller and all running services."""
        # Check if we're in Kubernetes mode with API enabled
        is_k8s_mode = self.service_config.service_run_type == ServiceRunType.KUBERNETES
        keep_api_running = is_k8s_mode and self.service_config.api_port

        if keep_api_running:
            # In Kubernetes mode with API: signal benchmark completion to API service
            # so it can continue serving results after other services shut down
            await self.publish(
                BenchmarkCompleteMessage(
                    service_id=self.service_id,
                    was_cancelled=self._was_cancelled,
                )
            )

        # Suppress heartbeat/process monitors before broadcasting shutdown
        self.service_manager.notify_shutdown()

        # Send shutdown command to all registered services via ROUTER (fire-and-forget)
        all_ids = list(ServiceRegistry.get_all_registered_ids())
        if keep_api_running:
            api_ids = {
                s.service_id for s in ServiceRegistry.get_services(ServiceType.API)
            }
            all_ids = [sid for sid in all_ids if sid not in api_ids]
        for sid in all_ids:
            try:
                await self.control_router.send_to(
                    sid,
                    Command(
                        cid=uuid.uuid4().hex,
                        cmd=CommandType.SHUTDOWN,
                    ),
                )
            except Exception as e:
                self.debug(f"Failed to send shutdown to {sid}: {e}")

        # Brief delay for messages to propagate before tearing down services
        await asyncio.sleep(Environment.SERVICE.SHUTDOWN_PROPAGATION_DELAY)

        await self.service_manager.shutdown_all_services()

        # In K8s mode with RAW export, wait for worker pods to upload raw records
        # to the API before stopping comms. Workers upload during their shutdown
        # sequence (after flushing RecordProcessor buffers to disk).
        if is_k8s_mode and self._should_wait_for_raw_records():
            await self._wait_for_raw_record_uploads()

        await self.comms.stop()
        await self.proxy_manager.stop()
        self.info(f"Stopping control ROUTER client (state={self.control_router.state})")
        await self.control_router.stop()
        self.info("Control ROUTER client stopped")

        # Drain subprocess errors reported via the error queue backchannel
        self._error_collector.drain_into()

        # Post-benchmark reporting (after services and comms are stopped)
        await self.ui.stop()
        await self.ui.wait_for_tasks()
        await asyncio.sleep(0.1)

        if not self._exit_errors:
            if self._profile_results and self._profile_results.results.records:
                await self._print_post_benchmark_info_and_metrics()
            elif self._was_cancelled:
                self.warning("Benchmark was cancelled before results were collected")
            else:
                self.error("No profile results to export")
                self._exit_errors.append(
                    ExitErrorInfo(
                        error_details=ErrorDetails(
                            type="NO_RESULTS",
                            message="No profile results to export",
                        ),
                        operation="profile",
                    )
                )
                self._print_exit_errors_and_log_file()
        else:
            self._print_exit_errors_and_log_file()

        self._print_process_memory_summary()

        if Environment.DEV.MODE:
            print_developer_mode_warning()

        # Clean up global queues to prevent semaphore leaks
        await cleanup_global_log_queue()
        await cleanup_global_error_queue()

        is_k8s_mode = self.service_config.service_run_type == ServiceRunType.KUBERNETES
        keep_api_running = is_k8s_mode and self.service_config.api_port

        if keep_api_running:
            # In Kubernetes mode with API running, block until API subprocess terminates.
            # This keeps the container alive so the operator can fetch results.
            # The API subprocess serves /api/metrics, /api/results/* endpoints.
            self.info("Kubernetes mode: API service continues running to serve results")
            await self.service_manager.wait_for_api_subprocess()
            self.info("API service has stopped, exiting")
            self._force_exit(0)

        self._force_exit(1 if self._exit_errors else 0)

    async def _handle_control_message(
        self, identity: str, message: ControllerBoundMessage
    ) -> Struct | None:
        """Dispatch control channel messages from child services.

        Returns a Struct response for request-reply patterns (Registration, Command).
        Returns None for fire-and-forget messages (Heartbeat, StatusUpdate, etc.).
        """
        match message:
            case Registration():
                self.debug(
                    lambda: f"Received registration from {message.stype} service: {message.sid}"
                )
                already_configuring = message.sid in self._configuring_ids
                ServiceRegistry.register(
                    service_id=message.sid,
                    service_type=message.stype,
                    first_seen_ns=time.time_ns(),
                    state=LifecycleState(message.state),
                    pod_name=message.pod_name,
                    pod_index=message.pod_index,
                )
                if (
                    not already_configuring
                    and message.num_workers is not None
                    and message.num_record_processors is not None
                ):
                    self.info(
                        f"Pod '{message.sid}' reports capacity: "
                        f"{message.num_workers} workers, "
                        f"{message.num_record_processors} record processors"
                    )
                if self._auto_configure and not already_configuring:
                    self._configure_scheduler.execute_async(
                        self._configure_single_service(message.sid)
                    )
                return RegistrationAck(rid=message.rid)
            case Heartbeat():
                self.debug(
                    lambda msg=message: f"Received heartbeat from {msg.stype} service: {msg.sid}"
                )
                ServiceRegistry.update_service(
                    service_id=message.sid,
                    service_type=message.stype,
                    last_seen_ns=time.time_ns(),
                    state=LifecycleState(message.state),
                )
                return None
            case StatusUpdate():
                self.debug(
                    lambda msg=message: f"Received status from {msg.stype} service: {msg.sid}"
                )
                ServiceRegistry.update_service(
                    service_id=message.sid,
                    service_type=message.stype,
                    last_seen_ns=time.time_ns(),
                    state=LifecycleState(message.state),
                )
                return None
            case MemoryReport():
                self._memory_tracker.record(
                    label=message.sid,
                    group=message.stype,
                    pid=message.pid,
                    phase=MemoryPhase(message.phase),
                    reading=MemoryReading(
                        pss=message.pss_bytes,
                        rss=message.rss_bytes,
                        uss=message.uss_bytes,
                        shared=message.shared_bytes,
                    ),
                )
                return None
            case TelemetryStatus():
                self._telemetry_endpoints_configured = list(
                    message.endpoints_configured
                )
                self._telemetry_endpoints_reachable = list(message.endpoints_reachable)
                self._should_wait_for_telemetry = message.enabled
                if not message.enabled:
                    reason_msg = f" - {message.reason}" if message.reason else ""
                    self.info(f"GPU telemetry disabled{reason_msg}")
                    ServiceRegistry.forget(message.sid)
                else:
                    self.info(
                        f"GPU telemetry enabled - {len(message.endpoints_reachable)}/{len(message.endpoints_configured)} endpoint(s) reachable"
                    )
                await self._check_and_trigger_shutdown()
                return None
            case ServerMetricsStatus():
                self._server_metrics_endpoints_configured = list(
                    message.endpoints_configured
                )
                self._server_metrics_endpoints_reachable = list(
                    message.endpoints_reachable
                )
                self._should_wait_for_server_metrics = message.enabled
                if not message.enabled:
                    reason_msg = f" - {message.reason}" if message.reason else ""
                    self.info(f"Server metrics disabled{reason_msg}")
                    ServiceRegistry.forget(message.sid)
                else:
                    self.info(
                        f"Server metrics enabled - {len(message.endpoints_reachable)}/{len(message.endpoints_configured)} endpoint(s) reachable."
                    )
                    unreachable = set(message.endpoints_configured) - set(
                        message.endpoints_reachable
                    )
                    if unreachable:
                        self.warning(f"Unreachable endpoints: {', '.join(unreachable)}")
                await self._check_and_trigger_shutdown()
                return None
            case Command():
                return await self._dispatch_control_command(identity, message)
            case CommandAck() | CommandOk() | CommandErr():
                # Responses to pending requests are handled by _pending_requests
                # matching in the ROUTER receive loop. If we get here, it's
                # an unexpected response.
                self.debug(
                    f"Unexpected command response from {identity}: {type(message).__name__}"
                )
                return None

    # -------------------------------------------------------------------------
    # Control channel: command dispatch and sending helpers
    # -------------------------------------------------------------------------

    async def _dispatch_control_command(
        self, identity: str, message: Command
    ) -> Struct | None:
        """Dispatch an incoming Command from a service to local @on_command hooks.

        Returns a CommandAck/CommandOk/CommandErr response struct.
        """
        for hook in self.get_hooks(AIPerfHook.ON_COMMAND):
            resolved = hook.resolve_params(self)
            if isinstance(resolved, Iterable) and message.cmd in resolved:
                try:
                    result = await hook.func(message)
                    if result is None:
                        return CommandAck(cid=message.cid, sid=self.service_id)
                    from pydantic import BaseModel

                    if isinstance(result, BaseModel):
                        payload = result.model_dump_json().encode()
                    elif isinstance(result, bytes):
                        payload = result
                    elif isinstance(result, dict):
                        payload = orjson.dumps(result)
                    else:
                        payload = orjson.dumps(result)
                    return CommandOk(
                        cid=message.cid, sid=self.service_id, payload=payload
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    self.error(
                        f"Failed to handle command {message.cmd} from {identity}: {e}"
                    )
                    return CommandErr(
                        cid=message.cid,
                        sid=self.service_id,
                        error=str(e),
                        traceback=tb,
                    )

        self.debug(f"No handler for command {message.cmd} from {identity}")
        return CommandAck(cid=message.cid, sid=self.service_id)

    async def _send_control_command(
        self,
        identity: str,
        cmd: str,
        payload: bytes = b"",
        timeout: float = Environment.SERVICE.COMMAND_RESPONSE_TIMEOUT,
    ) -> CommandResponse:
        """Send a command to a specific service via ROUTER and wait for response."""
        command = Command(cid=uuid.uuid4().hex, cmd=cmd, payload=payload)
        return await self.control_router.request_to(identity, command, timeout)

    async def _send_control_command_to_all(
        self,
        cmd: str,
        service_ids: list[str],
        payload: bytes = b"",
        timeout: float = Environment.SERVICE.COMMAND_RESPONSE_TIMEOUT,
    ) -> list[CommandResponse | ErrorDetails]:
        """Send a command to all specified services and wait for all responses."""
        tasks = {
            sid: asyncio.create_task(
                self._send_control_command(sid, cmd, payload, timeout)
            )
            for sid in service_ids
        }
        results: list[CommandResponse | ErrorDetails] = []
        for sid, task in tasks.items():
            try:
                results.append(await task)
            except asyncio.TimeoutError:
                results.append(
                    ErrorDetails(
                        type="TimeoutError",
                        message=f"Command {cmd} timed out for {sid}",
                    )
                )
            except Exception as e:
                results.append(ErrorDetails.from_exception(e))
        return results

    async def _send_control_command_to_all_fail_fast(
        self,
        cmd: str,
        service_ids: list[str],
        payload: bytes = b"",
        timeout: float = Environment.SERVICE.COMMAND_RESPONSE_TIMEOUT,
    ) -> list[CommandResponse | ErrorDetails]:
        """Send command to all services, aborting on first error."""
        tasks = {
            sid: asyncio.create_task(
                self._send_control_command(sid, cmd, payload, timeout)
            )
            for sid in service_ids
        }
        results: list[CommandResponse | ErrorDetails] = []
        try:
            for coro in asyncio.as_completed(tasks.values()):
                try:
                    response = await coro
                    results.append(response)
                    if isinstance(response, CommandErr):
                        self.debug(
                            f"Received error from {response.sid}, aborting wait for "
                            f"remaining {len(service_ids) - len(results)} service(s)"
                        )
                        break
                except asyncio.TimeoutError:
                    results.append(
                        ErrorDetails(
                            type="TimeoutError", message=f"Command {cmd} timed out"
                        )
                    )
                    break
                except Exception as e:
                    results.append(ErrorDetails.from_exception(e))
                    break
        finally:
            for task in tasks.values():
                task.cancel()
        return results

    @staticmethod
    def _force_exit(code: int) -> None:
        """Flush stdio and exit. Falls back to os._exit if sys.exit hangs
        (e.g. ZMQ context blocking in atexit)."""
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

    def _should_wait_for_raw_records(self) -> bool:
        """Check if we need to wait for raw record uploads from worker pods."""
        from aiperf.common.enums import ExportLevel

        return self.user_config.output.export_level == ExportLevel.RAW

    async def _wait_for_raw_record_uploads(self) -> None:
        """Wait for worker pods to upload raw record files to the API.

        Polls the raw_records subdirectory until we have at least one file
        per WorkerPodManager, or the timeout expires.
        """
        raw_records_dir = (
            self.user_config.output.artifact_directory
            / OutputDefaults.RAW_RECORDS_FOLDER
        )
        timeout = Environment.SERVICE.RAW_RECORD_UPLOAD_TIMEOUT
        poll_interval = 1.0
        deadline = time.monotonic() + timeout

        wpm_count = len(ServiceRegistry.get_services(ServiceType.WORKER_POD_MANAGER))
        if wpm_count == 0:
            self.debug("No WorkerPodManagers registered, skipping raw record wait")
            return

        self.info(f"Waiting for raw record uploads from {wpm_count} worker pod(s)...")

        while time.monotonic() < deadline:
            if raw_records_dir.exists():
                files = list(raw_records_dir.glob("raw_records_*.jsonl"))
                if len(files) >= wpm_count:
                    self.info(
                        f"Received {len(files)} raw record file(s) from "
                        f"{wpm_count} pod(s), proceeding with export"
                    )
                    return
                if files:
                    self.debug(
                        f"Have {len(files)}/{wpm_count} raw record file(s), "
                        "waiting for remaining pods..."
                    )
            await asyncio.sleep(poll_interval)

        # Check what we got before warning
        actual = 0
        if raw_records_dir.exists():
            actual = len(list(raw_records_dir.glob("raw_records_*.jsonl")))
        if actual > 0:
            self.warning(
                f"Timed out after {timeout}s: received {actual}/{wpm_count} "
                "raw record file(s). Proceeding with partial data."
            )
        else:
            self.warning(
                f"Timed out waiting for raw record uploads after {timeout}s. "
                "Raw records may be missing from export."
            )

    def _print_exit_errors_and_log_file(self) -> None:
        """Print post exit errors and log file info to the console."""
        console = Console()
        print_exit_errors(self._exit_errors, console=console)
        self._print_log_file_info(console)
        console.print()
        console.file.flush()

    async def _print_post_benchmark_info_and_metrics(self) -> None:
        """Print post benchmark info and metrics to the console."""
        console = Console()
        if console.width < 100:
            console.width = 100

        exporter_manager = ExporterManager(
            results=self._profile_results.results,
            config=self.config,
            telemetry_results=self._telemetry_results,
            server_metrics_results=self._server_metrics_results,
        )

        # Export data files (CSV, JSON) with complete dataset including telemetry
        await exporter_manager.export_data()

        # Export console output with complete dataset including telemetry
        await exporter_manager.export_console(console=console)

        console.print()
        self._print_cli_command(console)
        self._print_benchmark_duration(console)
        self._print_exported_file_infos(exporter_manager, console)
        self._print_log_file_info(console)
        if self._was_cancelled:
            console.print(
                "[italic yellow]The profile run was cancelled early. Results shown may be incomplete or inaccurate.[/italic yellow]"
            )

        console.print()
        console.file.flush()

    def _print_log_file_info(self, console: Console) -> None:
        """Print the log file info."""
        log_file = (
            self.user_config.output.artifact_directory
            / OutputDefaults.LOG_FOLDER
            / OutputDefaults.LOG_FILE
        )
        console.print(
            f"[bold green]Log File:[/bold green] [cyan]{log_file.resolve()}[/cyan]"
        )

    def _print_exported_file_infos(
        self, exporter_manager: ExporterManager, console: Console
    ) -> None:
        """Print the exported file infos."""
        file_infos = exporter_manager.get_exported_file_infos()
        for file_info in file_infos:
            console.print(
                f"[bold green]{file_info.export_type}[/bold green]: [cyan]{file_info.file_path.resolve()}[/cyan]"
            )

    def _print_cli_command(self, console: Console) -> None:
        """Print the CLI command that was used to run the benchmark."""
        console.print(
            f"[bold green]CLI Command:[/bold green] [italic]{self.user_config.cli_command}[/italic]"
        )

    def _print_benchmark_duration(self, console: Console) -> None:
        """Print the duration of the benchmark."""
        from aiperf.metrics.types.benchmark_duration_metric import (
            BenchmarkDurationMetric,
        )

        # Metrics are already in display units from summarize()
        duration = self._profile_results.get(BenchmarkDurationMetric.tag)
        if duration:
            duration_str = f"[bold green]{BenchmarkDurationMetric.header}[/bold green]: {duration.avg:.2f} {duration.unit}"
            if self._was_cancelled:
                duration_str += " [italic yellow](cancelled early)[/italic yellow]"
            console.print(duration_str)

    def _print_process_memory_summary(self) -> None:
        """Print memory summary for all AIPerf processes."""
        controller_pss_start = getattr(self, "_controller_pss_at_start", None)
        if controller_pss_start is not None:
            self._memory_tracker.record(
                label="SystemController",
                group="controller",
                pid=os.getpid(),
                phase=MemoryPhase.STARTUP,
                reading=MemoryReading(pss=controller_pss_start),
            )
        self._memory_tracker.capture(
            label="SystemController",
            group="controller",
            pid=os.getpid(),
            phase=MemoryPhase.SHUTDOWN,
        )

        self._memory_tracker.print_summary(title="AIPerf Process Memory")

    async def _kill(self) -> None:
        """Kill the system controller."""
        try:
            await self.service_manager.kill_all_services()
        except Exception as e:
            raise self._service_error("Failed to stop all services") from e

        await super()._kill()


def main() -> None:
    """Main entry point for the system controller."""

    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.SYSTEM_CONTROLLER)


if __name__ == "__main__":
    main()
