# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

from aiperf.common.accumulator_protocols import (
    AccumulatorProtocol,
    AnalyzerProtocol,
    ExportContext,
    StreamExporterProtocol,
    SummaryContext,
)
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    MessageType,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PluginDisabled
from aiperf.common.hooks import background_task, on_command, on_message, on_pull_message
from aiperf.common.messages import (
    AllRecordsReceivedMessage,
    MetricRecordsMessage,
    ProcessAllResultsMessage,
    ProcessRecordsCommand,
    ProfileCancelCommand,
    ProfileCompleteCommand,
    RealtimeMetricsCommand,
    RealtimeMetricsMessage,
    RecordsProcessingStatsMessage,
    ServerMetricsRecordMessage,
    StartRealtimeTelemetryCommand,
    TelemetryRecordsMessage,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    MetricResult,
    PhaseRecordsStats,
    ProcessRecordsResult,
    ProfileResults,
    TelemetryExportData,
    WorkerProcessingStats,
)
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.common.types import MetricTagT
from aiperf.common.utils import yield_to_event_loop
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditsCompleteMessage,
)
from aiperf.exporters.exporter_config import FileExportInfo
from aiperf.exporters.exporter_manager import ExporterManager
from aiperf.gpu_telemetry.protocols import GPUTelemetryAccumulatorProtocol
from aiperf.metrics.accumulator import MetricsSummary
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    AccumulatorType,
    AnalyzerType,
    PluginType,
    StreamExporterType,
    UIType,
)
from aiperf.post_processors.steady_state_analyzer import SteadyStateSummary
from aiperf.records.error_tracker import ErrorTracker
from aiperf.records.records_tracker import RecordsTracker


@dataclass
class ErrorTrackingState:
    """Base class for tracking errors with counts and thread-safe access.

    Provides common error tracking functionality for all metrics subsystems
    (telemetry, server metrics, regular metrics).
    """

    error_counts: dict[ErrorDetails, int] = field(
        default_factory=lambda: defaultdict(int)
    )


class RecordsManager(PullClientMixin, BaseComponentService):
    """Collects and processes benchmark results from workers.

    The RecordsManager receives metric records from workers and accumulates them
    for final processing. The timing manager is the ground truth for what requests
    completed within the benchmark window - when it signals phase completion with
    a final_completed_count, the RecordsManager waits until it has processed that
    many records before finalizing results.
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
            pull_client_address=CommAddress.RECORDS,
            pull_client_bind=True,
            pull_client_max_concurrency=Environment.ZMQ.PULL_MAX_CONCURRENCY,
        )

        self._records_tracker = RecordsTracker()
        self._error_tracker = ErrorTracker()

        self._previous_realtime_records: int | None = None

        self._telemetry_state = ErrorTrackingState()
        self._server_metrics_state = ErrorTrackingState()
        self._metric_state = ErrorTrackingState()

        # Role-based accumulator and stream exporter collections
        self._accumulators: dict[AccumulatorType, AccumulatorProtocol] = {}
        self._stream_exporters: dict[StreamExporterType, StreamExporterProtocol] = {}

        # Specific accumulator reference for realtime telemetry command
        self._gpu_telemetry_accumulator: GPUTelemetryAccumulatorProtocol | None = None

        for entry in chain(
            plugins.iter_entries(PluginType.ACCUMULATOR),
            plugins.iter_entries(PluginType.STREAM_EXPORTER),
        ):
            try:
                PluginClass = plugins.get_class(entry.category, entry.name)
                processor = PluginClass(
                    service_id=self.service_id,
                    service_config=self.service_config,
                    user_config=self.user_config,
                    pub_client=self.pub_client,
                )
                self.attach_child_lifecycle(processor)

                if entry.category == PluginType.ACCUMULATOR:
                    acc_type = AccumulatorType(entry.name)
                    self._accumulators[acc_type] = processor
                    if acc_type == AccumulatorType.GPU_TELEMETRY:
                        self._gpu_telemetry_accumulator = processor
                elif entry.category == PluginType.STREAM_EXPORTER:
                    self._stream_exporters[StreamExporterType(entry.name)] = processor

                self.debug(
                    f"Created {entry.category} plugin: {entry.name}: {processor.__class__.__name__}"
                )
            except PluginDisabled:
                self.debug(
                    f"{entry.category} {entry.name} is disabled and will not be used"
                )
            except Exception as e:
                self.error(f"Failed to create {entry.category} {entry.name}: {e}")

        # Analyzers: derive results from accumulators at summarize time
        self._analyzers: dict[AnalyzerType, AnalyzerProtocol] = {}
        for entry in plugins.iter_entries(PluginType.ANALYZER):
            try:
                PluginClass = plugins.get_class(entry.category, entry.name)
                analyzer = PluginClass(user_config=self.user_config)
                self._analyzers[AnalyzerType(entry.name)] = analyzer
                self.debug(
                    f"Created analyzer plugin: {entry.name}: {analyzer.__class__.__name__}"
                )
            except PluginDisabled:
                self.debug(f"Analyzer {entry.name} is disabled and will not be used")
            except Exception as e:
                self.error(f"Failed to create analyzer {entry.name}: {e}")

        # Build record_type string → handler routing table from plugin metadata
        self._routing_table = self._build_routing_table()
        self._log_routing_table()

        # Accumulators that handle metric records (for summarization and realtime metrics)
        self._metric_accumulators: list[AccumulatorProtocol] = [
            self._accumulators[AccumulatorType(entry.name)]
            for entry in plugins.iter_entries(PluginType.ACCUMULATOR)
            if "metric_records" in entry.metadata.get("record_types", [])
            and AccumulatorType(entry.name) in self._accumulators
        ]

    @on_pull_message(MessageType.METRIC_RECORDS)
    async def _on_metric_records(self, message: MetricRecordsMessage) -> None:
        """Handle a metric records message."""
        if self.is_trace_enabled:
            self.trace(f"Received metric records: {message}")

        if message.metadata.benchmark_phase != CreditPhase.PROFILING:
            self.debug(
                lambda: f"Skipping non-profiling record: {message.metadata.benchmark_phase}"
            )
            return

        record_data = message.to_data()

        await self._dispatch_record(record_data)

        self._records_tracker.update_from_record_data(record_data)
        if record_data.error:
            self._error_tracker.increment_error_count_for_phase(
                record_data.metadata.benchmark_phase, record_data.error
            )

        if self._records_tracker.check_and_set_all_records_received_for_phase(
            record_data.metadata.benchmark_phase
        ):
            await self._handle_all_records_received(
                record_data.metadata.benchmark_phase
            )

    @on_pull_message(MessageType.TELEMETRY_RECORDS)
    async def _on_telemetry_records(self, message: TelemetryRecordsMessage) -> None:
        """Handle telemetry records message from Telemetry Manager.
        The RecordsManager acts as the central hub for all record processing,
        whether inference metrics or GPU telemetry.

        Args:
            message: Batch of telemetry records from a DCGM collector
        """
        if message.valid:
            for record in message.records:
                errors = await self._dispatch_record(record)
                for e in errors:
                    self._telemetry_state.error_counts[
                        ErrorDetails.from_exception(e)
                    ] += 1
        else:
            if message.error:
                self._telemetry_state.error_counts[message.error] += 1

    @on_pull_message(MessageType.SERVER_METRICS_RECORD)
    async def _on_server_metrics_records(
        self, message: ServerMetricsRecordMessage
    ) -> None:
        """Handle server metrics record message from Server Metrics Manager.

        Forwards full record to accumulators and stream exporters via dispatch.

        Args:
            message: Server metrics record from a Prometheus collector
        """
        if message.valid:
            errors = await self._dispatch_record(message.record)
            for e in errors:
                self._server_metrics_state.error_counts[
                    ErrorDetails.from_exception(e)
                ] += 1
        else:
            if message.error:
                self._server_metrics_state.error_counts[message.error] += 1

    async def _handle_all_records_received(self, phase: CreditPhase) -> None:
        """Handle the case where all records have been received."""
        if phase != CreditPhase.PROFILING:
            self.debug(lambda: f"Skipping non-profiling phase: {phase}")
            return

        phase_stats = self._records_tracker.create_stats_for_phase(phase)
        self.info(
            lambda: f"Processed {phase_stats.success_records} valid requests and {phase_stats.error_records} errors ({phase_stats.total_records} total)."
        )

        self.info("Received all records, processing now...")
        self.execute_async(
            self._finalize_and_process_results(
                phase=phase,
                cancelled=self._records_tracker.was_phase_cancelled(phase),
            )
        )
        await yield_to_event_loop()

    async def _finalize_and_process_results(
        self, phase: CreditPhase, cancelled: bool
    ) -> None:
        """Finalize server metrics collection and process results.

        This runs as a background task to avoid blocking the message pump.
        """
        phase_stats = self._records_tracker.create_stats_for_phase(phase)

        # Send a message to the event bus to signal that we received all the records
        await self.publish(
            AllRecordsReceivedMessage(
                service_id=self.service_id,
                request_ns=time.time_ns(),
                final_processing_stats=phase_stats,
            )
        )

        # Trigger final server metrics scrape and wait for completion
        # This ensures final metrics are pushed before we export results
        response = await self.send_command_and_wait_for_response(
            ProfileCompleteCommand(service_id=self.service_id), timeout=10.0
        )

        if isinstance(response, ErrorDetails):
            self.warning(f"Server metrics final scrape timed out or failed: {response}")
        else:
            self.debug("Server metrics final scrape completed")

        self.debug("Waiting for server metrics flush period...")
        # Wait for server metrics flush period to allow final metrics to be collected
        # This ensures metrics that are still being processed by the server are captured
        flush_period = Environment.SERVER_METRICS.COLLECTION_FLUSH_PERIOD
        phase_stats = self._records_tracker.create_stats_for_phase(
            CreditPhase.PROFILING
        )
        flush_end_ns = (phase_stats.requests_end_ns or time.time_ns()) + (
            (flush_period or 0) * NANOS_PER_SECOND
        )
        sleep_dur_sec = (flush_end_ns - time.time_ns()) / NANOS_PER_SECOND
        if sleep_dur_sec > 0:
            self.info(
                f"Waiting {sleep_dur_sec:.1f}s for server metrics flush period..."
            )
            await asyncio.sleep(sleep_dur_sec)

        self.debug("Server metrics flush period complete, processing now...")
        await self._process_results(phase=phase, cancelled=cancelled)
        self.info("_finalize_and_process_results completed")

    # -------------------------------------------------------------------------
    # Record dispatch routing table
    # -------------------------------------------------------------------------

    def _build_routing_table(self) -> dict[str, list[Any]]:
        """Build record_type string → handler mapping from plugin metadata.

        Uses record_types metadata from plugins.yaml to build a dispatch table.
        When a record arrives, _dispatch_record() uses record.record_type for
        O(1) string lookup — no isinstance calls needed.

        Returns:
            Dict mapping record_type strings to lists of handlers that accept them.
        """
        table: dict[str, list[Any]] = {}
        for entry in chain(
            plugins.iter_entries(PluginType.ACCUMULATOR),
            plugins.iter_entries(PluginType.STREAM_EXPORTER),
        ):
            if entry.category == PluginType.ACCUMULATOR:
                proc = self._accumulators.get(AccumulatorType(entry.name))
            else:
                proc = self._stream_exporters.get(StreamExporterType(entry.name))

            if proc is None:
                continue  # Accumulator/exporter was disabled or failed to init

            for record_type in entry.metadata.get("record_types", []):
                table.setdefault(record_type, []).append(proc)

        return table

    async def _dispatch_record(self, record: Any) -> list[BaseException]:
        """Dispatch a record to all matching handlers via the routing table.

        Uses record.record_type for O(1) string lookup. All matching handlers
        are called concurrently via asyncio.gather. Exceptions are caught
        per-handler and logged without aborting other handlers.

        Args:
            record: Any record object with a record_type ClassVar.

        Returns:
            List of exceptions from failed handlers (empty if all succeeded).
        """
        handlers = self._routing_table.get(record.record_type, [])
        if not handlers:
            self.warning(
                f"No handlers registered for record type: {record.record_type}"
            )
            return []

        results = await asyncio.gather(
            *[handler.process_record(record) for handler in handlers],
            return_exceptions=True,
        )
        errors: list[BaseException] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                self.error(
                    f"Handler {handlers[i].__class__.__name__} failed "
                    f"for {record.record_type}: {result!r}"
                )
                errors.append(result)
        return errors

    def _log_routing_table(self) -> None:
        """Log the routing table for startup validation."""
        self.debug(
            lambda: (
                f"Routing table: {len(self._accumulators)} accumulators, "
                f"{len(self._stream_exporters)} stream exporters, "
                f"{len(self._routing_table)} record types"
            )
        )
        for record_type, handlers in self._routing_table.items():
            handler_names = [h.__class__.__name__ for h in handlers]
            self.debug(lambda rt=record_type, hn=handler_names: (f"  {rt} -> {hn}"))

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(
        self, phase_start_msg: CreditPhaseStartMessage
    ) -> None:
        """Handle a credit phase start message in order to track the total number of expected requests."""
        self._records_tracker.update_phase_info(phase_start_msg.stats)
        self.info(f"Credit phase start: {phase_start_msg.config.phase}")

    @on_message(MessageType.CREDIT_PHASE_SENDING_COMPLETE)
    async def _on_credit_phase_sending_complete(
        self, message: CreditPhaseSendingCompleteMessage
    ) -> None:
        """Handle a credit phase sending complete message in order to track the final request count."""
        if message.stats.phase == CreditPhase.PROFILING:
            self.info(
                f"Sent {message.stats.final_requests_sent:,} requests. Waiting for all to complete..."
            )
        self._records_tracker.update_phase_info(message.stats)

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(
        self, message: CreditPhaseCompleteMessage
    ) -> None:
        """Handle a credit phase complete message in order to track the end time, and check if all records have been received."""
        self._records_tracker.update_phase_info(message.stats)
        if message.stats.phase == CreditPhase.PROFILING:
            phase_stats = self._records_tracker.create_stats_for_phase(
                message.stats.phase
            )
            # TODO
            self.info(
                lambda: f"Received CREDIT_PHASE_COMPLETE message, Phase complete: {phase_stats!r}"
            )
            self.notice(
                f"All requests have completed, please wait for the results to be processed "
                f"(currently {phase_stats.total_records:,} of {phase_stats.final_requests_completed:,} records processed)..."
            )

        # This check is to prevent a race condition where the records manager processes
        # all records before the timing manager has sent the final completed count.
        if self._records_tracker.check_and_set_all_records_received_for_phase(
            message.stats.phase
        ):
            await self._handle_all_records_received(message.stats.phase)

    @on_message(MessageType.CREDITS_COMPLETE)
    async def _on_credits_complete(self, message: CreditsCompleteMessage) -> None:
        """Handle a credits complete message in order to track the end time, and check if all records have been received."""
        self.info(
            "All credits complete, please wait for the results to be processed..."
        )
        # This check is to prevent a race condition where the records manager processes
        # all records before the timing manager has sent the final completed count.
        if self._records_tracker.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING
        ):
            await self._handle_all_records_received(CreditPhase.PROFILING)

    @background_task(
        interval=Environment.RECORD.PROGRESS_REPORT_INTERVAL, immediate=False
    )
    async def _report_records_task(self) -> None:
        """Report the records processing stats."""
        active_phase_stats = self._records_tracker.create_stats_for_phase(
            CreditPhase.PROFILING
        )
        if active_phase_stats.total_records == 0:
            return  # TODO: What about worker stats?
        overall_worker_stats = self._records_tracker.create_overall_worker_stats()
        await self._publish_processing_stats(active_phase_stats, overall_worker_stats)

    async def _publish_processing_stats(
        self,
        phase_stats: PhaseRecordsStats,
        worker_stats: dict[str, WorkerProcessingStats],
    ) -> None:
        """Publish the profile processing stats."""
        message = RecordsProcessingStatsMessage(
            service_id=self.service_id,
            request_ns=time.time_ns(),
            processing_stats=phase_stats,
            worker_stats=worker_stats,
        )
        await self.publish(message)

    @on_command(CommandType.PROCESS_RECORDS)
    async def _on_process_records_command(
        self, message: ProcessRecordsCommand
    ) -> ProcessRecordsResult:
        """Handle the process records command by summarizing all accumulators and returning the results."""
        self.debug(lambda: f"Received process records command: {message}")
        return await self._process_results(
            phase=CreditPhase.PROFILING, cancelled=message.cancelled
        )

    @on_command(CommandType.PROFILE_CANCEL)
    async def _on_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> ProcessRecordsResult:
        """Handle the profile cancel command by processing current results.

        This marks the phase as cancelled in the records tracker and processes
        all currently received records. Called when user presses Ctrl+C.
        """
        self.warning(f"Received profile cancel command: {message}")

        # Mark the phase as cancelled in the tracker
        self._records_tracker.mark_phase_cancelled(CreditPhase.PROFILING)

        return await self._process_results(phase=CreditPhase.PROFILING, cancelled=True)

    @background_task(interval=None, immediate=True)
    async def _report_realtime_inference_metrics_task(self) -> None:
        """Report inference metrics at regular intervals (dashboard only)."""
        if (
            self.service_config.ui_type != UIType.DASHBOARD
            and not Environment.UI.REALTIME_METRICS_ENABLED
        ):
            return

        while not self.stop_requested:
            await asyncio.sleep(Environment.UI.REALTIME_METRICS_INTERVAL)
            phase_stats = self._records_tracker.create_stats_for_phase(
                CreditPhase.PROFILING
            )
            if phase_stats.total_records == self._previous_realtime_records:
                continue  # No new records have been processed, so no need to update the metrics
            self._previous_realtime_records = phase_stats.total_records
            await self._report_realtime_metrics()

    @on_command(CommandType.START_REALTIME_TELEMETRY)
    async def _on_start_realtime_telemetry_command(
        self, message: StartRealtimeTelemetryCommand
    ) -> None:
        """Handle command to start the realtime telemetry background task.

        This is called when the user dynamically enables the telemetry dashboard
        by pressing the telemetry option in the UI without having passed the 'dashboard' parameter
        at startup.
        """
        if self._gpu_telemetry_accumulator:
            self._gpu_telemetry_accumulator.start_realtime_telemetry()
        else:
            self.error(
                "GPU telemetry accumulator not found, cannot start realtime telemetry"
            )

    @on_command(CommandType.REALTIME_METRICS)
    async def _on_realtime_metrics_command(
        self, message: RealtimeMetricsCommand
    ) -> None:
        """Handle a real-time metrics command."""
        await self._report_realtime_metrics()

    async def _report_realtime_metrics(self) -> None:
        """Report inference metrics (used by command handler)."""
        metrics = await self._generate_realtime_metrics()
        if metrics:
            await self.publish(
                RealtimeMetricsMessage(
                    service_id=self.service_id,
                    metrics=metrics,
                )
            )

    async def _generate_realtime_metrics(self) -> list[MetricResult]:
        """Generate the real-time metrics for the profile run."""
        results = await asyncio.gather(
            *[
                asyncio.wait_for(
                    accumulator.summarize(),
                    timeout=30.0,  # Shorter timeout for realtime updates
                )
                for accumulator in self._metric_accumulators
            ],
            return_exceptions=True,
        )

        # Extract MetricResult values from typed summaries (dict[MetricTagT, MetricResult])
        metric_results: list[MetricResult] = []
        for result in results:
            if hasattr(result, "results") and isinstance(result.results, dict):
                metric_results.extend(result.results.values())

        return metric_results

    async def _process_results(
        self, phase: CreditPhase, cancelled: bool
    ) -> ProcessRecordsResult:
        """Process all accumulator results through a unified pipeline.

        1. Build per-accumulator ExportContext with appropriate error summaries
        2. Call export_results() on all accumulators concurrently
        3. Finalize stream exporters
        4. Build ProcessRecordsResult from metric accumulator results
        5. Export data files (CSV, JSON)
        6. Publish single ProcessAllResultsMessage
        """
        self.debug(lambda: f"Processing records (cancelled: {cancelled})")
        self.info("Processing records results...")

        phase_stats = self._records_tracker.create_stats_for_phase(phase)

        # Build per-accumulator ExportContext.
        # Each accumulator type needs different time windows and error sources:
        # - GPU_TELEMETRY: no end_ns (includes post-profiling final scrape), own error state
        # - SERVER_METRICS: fallback timestamps (needs valid range for stats), own error state
        # - Others (METRIC_RESULTS): standard phase window, error tracker
        export_contexts: dict[AccumulatorType, ExportContext] = {}
        for acc_type in self._accumulators:
            if acc_type == AccumulatorType.GPU_TELEMETRY:
                error_summary = [
                    ErrorDetailsCount(error_details=ed, count=c)
                    for ed, c in self._telemetry_state.error_counts.items()
                ]
                export_contexts[acc_type] = ExportContext(
                    start_ns=phase_stats.start_ns,
                    error_summary=error_summary,
                    cancelled=cancelled,
                )
            elif acc_type == AccumulatorType.SERVER_METRICS:
                error_summary = [
                    ErrorDetailsCount(error_details=ed, count=c)
                    for ed, c in self._server_metrics_state.error_counts.items()
                ]
                export_contexts[acc_type] = ExportContext(
                    start_ns=phase_stats.start_ns or time.time_ns(),
                    end_ns=phase_stats.requests_end_ns or time.time_ns(),
                    error_summary=error_summary,
                    cancelled=cancelled,
                )
            else:
                export_contexts[acc_type] = ExportContext(
                    start_ns=phase_stats.start_ns,
                    end_ns=phase_stats.requests_end_ns,
                    error_summary=self._error_tracker.get_error_summary_for_phase(
                        phase
                    ),
                    cancelled=cancelled,
                )

        # Export results from all accumulators concurrently
        acc_types = list(self._accumulators.keys())
        acc_list = list(self._accumulators.values())
        self.debug(
            lambda: (
                f"Exporting results from {len(acc_list)} accumulators: "
                f"{[a.__class__.__name__ for a in acc_list]}"
            )
        )

        raw_results = await asyncio.gather(
            *[
                asyncio.wait_for(
                    acc.export_results(export_contexts[acc_type]),
                    timeout=Environment.RECORD.PROCESS_RECORDS_TIMEOUT,
                )
                for acc_type, acc in zip(acc_types, acc_list, strict=True)
            ],
            return_exceptions=True,
        )

        # Index results by AccumulatorType
        export_outputs: dict[AccumulatorType, Any] = {}
        for acc_type, result in zip(acc_types, raw_results, strict=True):
            if isinstance(result, BaseException):
                self.error(f"export_results failed for {acc_type}: {result!r}")
            else:
                export_outputs[acc_type] = result

        # Finalize stream exporters — flush any buffered data (JSONL writers, etc.)
        await self._finalize_stream_exporters()

        # Run analyzers with SummaryContext (dependency-ordered)
        summary_ctx = SummaryContext(
            accumulators=dict(self._accumulators),
            accumulator_outputs={str(k): v for k, v in export_outputs.items()},
            start_ns=phase_stats.start_ns or 0,
            end_ns=phase_stats.requests_end_ns or 0,
            cancelled=cancelled,
        )
        analyzer_outputs: dict[AnalyzerType, Any] = {}
        for analyzer_name, analyzer in self._analyzers.items():
            try:
                result = await analyzer.summarize(summary_ctx)
                analyzer_outputs[analyzer_name] = result
                summary_ctx.accumulator_outputs[str(analyzer_name)] = result
            except PluginDisabled as e:
                self.debug(f"Analyzer {analyzer_name} disabled: {e}")
            except Exception as e:
                self.error(f"Analyzer {analyzer_name} failed: {e!r}")

        # Build ProcessRecordsResult from metric accumulator outputs
        records_results: dict[MetricTagT, MetricResult] = {}
        timeslice_metric_results: dict[str, dict[MetricTagT, MetricResult]] = {}
        timeslice_windows: dict[str, Any] = {}
        error_results: list[ErrorDetails] = []

        for _acc_type, output in export_outputs.items():
            if isinstance(output, MetricsSummary):
                records_results.update(output.results)
                if output.timeslices is not None:
                    timeslice_metric_results.update(output.timeslices)
                if output.timeslice_windows is not None:
                    timeslice_windows.update(output.timeslice_windows)

        # Collect errors from failed accumulator exports
        for _acc_type, result in zip(acc_types, raw_results, strict=True):
            if isinstance(result, BaseException):
                error_results.append(ErrorDetails.from_exception(result))

        process_records_result = ProcessRecordsResult(
            results=ProfileResults(
                records=records_results,
                timeslice_metric_results=timeslice_metric_results,
                timeslice_windows=timeslice_windows or None,
                completed=len(records_results),
                start_ns=phase_stats.start_ns or time.time_ns(),
                end_ns=phase_stats.requests_end_ns or time.time_ns(),
                error_summary=self._error_tracker.get_error_summary_for_phase(phase),
                was_cancelled=cancelled,
            ),
            errors=error_results,
        )

        # Extract typed domain results (None if accumulator was disabled or failed)
        telemetry_results: TelemetryExportData | None = export_outputs.get(
            AccumulatorType.GPU_TELEMETRY
        )
        server_metrics_results: ServerMetricsResults | None = export_outputs.get(
            AccumulatorType.SERVER_METRICS
        )

        # Export data files and publish artifacts — single ExporterManager instance
        steady_state = analyzer_outputs.get(AnalyzerType.STEADY_STATE)
        exported_artifacts = await self._export_and_publish(
            process_records_result.results,
            telemetry_results,
            server_metrics_results,
            steady_state_results=steady_state,
        )

        # Publish unified results message (includes artifact paths for SystemController display)
        self.debug("Publishing ProcessAllResultsMessage...")
        await self.publish(
            ProcessAllResultsMessage(
                service_id=self.service_id,
                results=process_records_result,
                telemetry_results=telemetry_results,
                server_metrics_results=server_metrics_results,
                steady_state_results=analyzer_outputs.get(AnalyzerType.STEADY_STATE),
                exported_artifacts=exported_artifacts,
            )
        )
        self.debug("ProcessAllResultsMessage published")
        return process_records_result

    async def _export_and_publish(
        self,
        profile_results: ProfileResults,
        telemetry_results: TelemetryExportData | None,
        server_metrics_results: ServerMetricsResults | None,
        steady_state_results: SteadyStateSummary | None = None,
    ) -> dict[str, FileExportInfo]:
        """Export data files, collect all artifact paths, and publish to remote storage.

        Single ExporterManager instance handles data export and artifact publishing.
        Stream exporter file infos are collected from already-finalized instances.
        Console export remains on the SystemController side.

        Returns:
            Dict mapping exporter class name to FileExportInfo.
            Sent over ZMQ on ProcessAllResultsMessage so SystemController
            can display file paths without creating its own ExporterManager.
        """
        try:
            exporter_manager = ExporterManager(
                results=profile_results,
                user_config=self.user_config,
                service_config=self.service_config,
                telemetry_results=telemetry_results,
                server_metrics_results=server_metrics_results,
                steady_state_results=steady_state_results,
            )
            await exporter_manager.export_data()
        except Exception as e:
            self.exception(f"Failed to export data files: {e!r}")
            return {}

        # Collect all artifact file infos: stream exporters + data exporters
        artifacts: dict[str, FileExportInfo] = {}

        for exporter in self._stream_exporters.values():
            try:
                info = exporter.get_export_info()
                if info.file_path.exists():
                    artifacts[exporter.__class__.__name__] = info
            except Exception as e:
                self.debug(f"Could not get export info from stream exporter: {e!r}")

        artifacts.update(exporter_manager.exported_file_infos)

        if not artifacts:
            return {}

        # Publish to remote storage (S3, GCS, etc.) — no-op when no publishers registered
        try:
            await exporter_manager.publish_artifacts(list(artifacts.values()))
        except Exception as e:
            self.exception(f"Failed to publish artifacts: {e!r}")

        return artifacts

    async def _finalize_stream_exporters(self) -> None:
        """Finalize all stream exporters to flush buffered data.

        Called after all records are processed and accumulators are summarized.
        Each exporter is finalized concurrently; errors are logged per-exporter
        without aborting the others.
        """
        if not self._stream_exporters:
            return

        results = await asyncio.gather(
            *[exporter.finalize() for exporter in self._stream_exporters.values()],
            return_exceptions=True,
        )
        for (exp_type, _), result in zip(
            self._stream_exporters.items(), results, strict=True
        ):
            if isinstance(result, BaseException):
                self.error(f"Stream exporter {exp_type} finalize failed: {result!r}")


def main() -> None:
    """Main entry point for the records manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.RECORDS_MANAGER)


if __name__ == "__main__":
    main()
