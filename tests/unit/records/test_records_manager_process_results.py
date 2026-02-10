# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RecordsManager._process_results() unified pipeline.

Tests the _process_results() method using a mock RecordsManager with
stub accumulators, following the same pattern as test_records_manager_routing.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import CreditPhase
from aiperf.common.models import MetricResult
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.plugin.enums import AccumulatorType, StreamExporterType
from aiperf.post_processors.metrics_accumulator import MetricsSummary

# ---------------------------------------------------------------------------
# Stub results
# ---------------------------------------------------------------------------

_STUB_METRIC_RESULT = MetricResult(
    tag="request_latency",
    header="Request Latency",
    unit="ms",
    avg=100.0,
    count=10,
)


def _make_metrics_summary() -> MetricsSummary:
    return MetricsSummary(results={_STUB_METRIC_RESULT.tag: _STUB_METRIC_RESULT})


def _make_telemetry_export_data() -> TelemetryExportData:
    """Minimal TelemetryExportData for testing."""
    from datetime import datetime, timezone

    from aiperf.common.models import TelemetryHierarchy, TelemetrySummary

    now = datetime.now(tz=timezone.utc)
    return TelemetryExportData(
        hierarchy=TelemetryHierarchy(),
        summary=TelemetrySummary(start_time=now, end_time=now),
        endpoints={},
    )


def _make_server_metrics_results() -> ServerMetricsResults:
    """Minimal ServerMetricsResults for testing."""
    return ServerMetricsResults(start_ns=1_000_000_000, end_ns=2_000_000_000)


# ---------------------------------------------------------------------------
# Mock manager builder
# ---------------------------------------------------------------------------


def _make_manager_mock(
    accumulators: dict[AccumulatorType, Any] | None = None,
    stream_exporters: dict[StreamExporterType, Any] | None = None,
    start_ns: int = 1_000_000_000,
    requests_end_ns: int = 2_000_000_000,
) -> MagicMock:
    """Create a mock RecordsManager with _process_results bound."""
    from aiperf.records.records_manager import RecordsManager

    mgr = MagicMock()
    mgr._accumulators = accumulators or {}
    mgr._stream_exporters = stream_exporters or {}

    # Records tracker returns phase stats
    phase_stats = MagicMock()
    phase_stats.start_ns = start_ns
    phase_stats.requests_end_ns = requests_end_ns
    mgr._records_tracker.create_stats_for_phase.return_value = phase_stats

    # Error tracker
    mgr._error_tracker.get_error_summary_for_phase.return_value = []

    # Telemetry/server metrics error states
    mgr._telemetry_state.error_counts = {}
    mgr._server_metrics_state.error_counts = {}

    # Service identity
    mgr.service_id = "test_records_manager"

    # Logging
    mgr.debug = MagicMock()
    mgr.info = MagicMock()
    mgr.error = MagicMock()
    mgr.exception = MagicMock()

    # Publish
    mgr.publish = AsyncMock()

    # Bind real methods
    mgr._process_results = RecordsManager._process_results.__get__(mgr)
    mgr._finalize_stream_exporters = RecordsManager._finalize_stream_exporters.__get__(
        mgr
    )
    mgr._export_and_publish = RecordsManager._export_and_publish.__get__(mgr)

    # Mock user_config/service_config for ExporterManager
    mgr.user_config = MagicMock()
    mgr.service_config = MagicMock()

    return mgr


def _make_stub_accumulator(export_result: Any = None) -> MagicMock:
    """Create a stub accumulator with an async export_results."""
    acc = MagicMock()
    acc.export_results = AsyncMock(return_value=export_result)
    acc.__class__.__name__ = "StubAccumulator"
    return acc


def _make_stub_stream_exporter() -> MagicMock:
    """Create a stub stream exporter."""
    exp = MagicMock()
    exp.finalize = AsyncMock()
    return exp


# ---------------------------------------------------------------------------
# Tests: _process_results pipeline
# ---------------------------------------------------------------------------


class TestProcessResults:
    """Test the unified _process_results() pipeline."""

    @pytest.mark.asyncio
    async def test_calls_export_results_on_all_accumulators(self) -> None:
        """export_results() is called on every registered accumulator."""
        acc_metrics = _make_stub_accumulator(_make_metrics_summary())
        acc_telemetry = _make_stub_accumulator(_make_telemetry_export_data())

        mgr = _make_manager_mock(
            accumulators={
                AccumulatorType.METRIC_RESULTS: acc_metrics,
                AccumulatorType.GPU_TELEMETRY: acc_telemetry,
            },
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        acc_metrics.export_results.assert_called_once()
        acc_telemetry.export_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_publishes_process_all_results_message(self) -> None:
        """A single ProcessAllResultsMessage is published."""
        from aiperf.common.messages import ProcessAllResultsMessage

        acc = _make_stub_accumulator(_make_metrics_summary())
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        mgr.publish.assert_called_once()
        msg = mgr.publish.call_args[0][0]
        assert isinstance(msg, ProcessAllResultsMessage)

    @pytest.mark.asyncio
    async def test_message_contains_exported_artifacts(self) -> None:
        """ProcessAllResultsMessage carries exported_artifacts dict keyed by class name."""
        from pathlib import Path

        from aiperf.common.messages import ProcessAllResultsMessage
        from aiperf.exporters.exporter_config import FileExportInfo

        csv_info = FileExportInfo(
            export_type="CSV Results", file_path=Path("/tmp/results.csv")
        )
        acc = _make_stub_accumulator(_make_metrics_summary())
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
        )

        with patch(
            "aiperf.records.records_manager.ExporterManager"
        ) as MockExporterManager:
            mock_instance = MagicMock()
            mock_instance.export_data = AsyncMock()
            mock_instance.publish_artifacts = AsyncMock()
            mock_instance.exported_file_infos = {"CsvDataExporter": csv_info}
            MockExporterManager.return_value = mock_instance

            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        msg: ProcessAllResultsMessage = mgr.publish.call_args[0][0]
        assert isinstance(msg.exported_artifacts, dict)
        assert "CsvDataExporter" in msg.exported_artifacts
        assert msg.exported_artifacts["CsvDataExporter"] is csv_info

    @pytest.mark.asyncio
    async def test_stream_exporter_artifacts_included(self) -> None:
        """Stream exporter file infos appear in exported_artifacts keyed by class name."""
        from pathlib import Path

        from aiperf.common.messages import ProcessAllResultsMessage
        from aiperf.exporters.exporter_config import FileExportInfo

        # Stream exporter with a file that "exists"
        exp = _make_stub_stream_exporter()
        exp.__class__ = type("RecordExportJSONLWriter", (), {})
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        exp.get_export_info.return_value = FileExportInfo(
            export_type="Record Export JSONL", file_path=mock_path
        )

        acc = _make_stub_accumulator(_make_metrics_summary())
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            stream_exporters={StreamExporterType.RECORD_EXPORT: exp},
        )

        with patch(
            "aiperf.records.records_manager.ExporterManager"
        ) as MockExporterManager:
            mock_instance = MagicMock()
            mock_instance.export_data = AsyncMock()
            mock_instance.publish_artifacts = AsyncMock()
            mock_instance.exported_file_infos = {}
            MockExporterManager.return_value = mock_instance

            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        msg: ProcessAllResultsMessage = mgr.publish.call_args[0][0]
        assert "RecordExportJSONLWriter" in msg.exported_artifacts
        assert (
            msg.exported_artifacts["RecordExportJSONLWriter"].export_type
            == "Record Export JSONL"
        )

    @pytest.mark.asyncio
    async def test_message_contains_typed_results(self) -> None:
        """ProcessAllResultsMessage carries telemetry and server_metrics results."""
        from aiperf.common.messages import ProcessAllResultsMessage

        telemetry_data = _make_telemetry_export_data()
        server_data = _make_server_metrics_results()

        mgr = _make_manager_mock(
            accumulators={
                AccumulatorType.METRIC_RESULTS: _make_stub_accumulator(
                    _make_metrics_summary()
                ),
                AccumulatorType.GPU_TELEMETRY: _make_stub_accumulator(telemetry_data),
                AccumulatorType.SERVER_METRICS: _make_stub_accumulator(server_data),
            },
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        msg: ProcessAllResultsMessage = mgr.publish.call_args[0][0]
        assert msg.telemetry_results is telemetry_data
        assert msg.server_metrics_results is server_data

    @pytest.mark.asyncio
    async def test_returns_process_records_result(self) -> None:
        """_process_results returns a ProcessRecordsResult with metric results."""
        from aiperf.common.models import ProcessRecordsResult

        mgr = _make_manager_mock(
            accumulators={
                AccumulatorType.METRIC_RESULTS: _make_stub_accumulator(
                    _make_metrics_summary()
                ),
            },
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            result = await mgr._process_results(
                phase=CreditPhase.PROFILING, cancelled=False
            )

        assert isinstance(result, ProcessRecordsResult)
        assert len(result.results.records) == 1
        assert "request_latency" in result.results.records

    @pytest.mark.asyncio
    async def test_accumulator_failure_logged_and_in_errors(self) -> None:
        """When an accumulator's export_results raises, error is logged and added to result.errors."""
        failing_acc = _make_stub_accumulator()
        failing_acc.export_results.side_effect = RuntimeError("export boom")

        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: failing_acc},
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            result = await mgr._process_results(
                phase=CreditPhase.PROFILING, cancelled=False
            )

        # Error logged
        mgr.error.assert_called()
        error_msg = mgr.error.call_args_list[0][0][0]
        assert "export boom" in error_msg

        # Error in result
        assert len(result.errors) == 1
        assert "export boom" in result.errors[0].message

    @pytest.mark.asyncio
    async def test_accumulator_timeout_treated_as_failure(self) -> None:
        """When an accumulator's export_results exceeds the timeout, it's treated as a failure."""
        import asyncio as _asyncio

        slow_acc = _make_stub_accumulator()
        slow_acc.export_results.side_effect = _asyncio.TimeoutError()

        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: slow_acc},
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            result = await mgr._process_results(
                phase=CreditPhase.PROFILING, cancelled=False
            )

        mgr.error.assert_called()
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_finalizes_stream_exporters(self) -> None:
        """Stream exporters are finalized after accumulator export."""
        exp = _make_stub_stream_exporter()
        mgr = _make_manager_mock(
            accumulators={
                AccumulatorType.METRIC_RESULTS: _make_stub_accumulator(
                    _make_metrics_summary()
                ),
            },
            stream_exporters={StreamExporterType.RECORD_EXPORT: exp},
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        exp.finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_exports_data_files(self) -> None:
        """ExporterManager.export_data() is called for data file export."""
        mgr = _make_manager_mock(
            accumulators={
                AccumulatorType.METRIC_RESULTS: _make_stub_accumulator(
                    _make_metrics_summary()
                ),
            },
        )

        with patch(
            "aiperf.records.records_manager.ExporterManager"
        ) as MockExporterManager:
            mock_instance = MagicMock()
            mock_instance.export_data = AsyncMock()
            mock_instance.publish_artifacts = AsyncMock()
            mock_instance.exported_file_infos = {}
            MockExporterManager.return_value = mock_instance

            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

            MockExporterManager.assert_called_once()
            mock_instance.export_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_export_failure_logged_not_raised(self) -> None:
        """ExporterManager failure is logged but doesn't crash _process_results."""
        from aiperf.common.messages import ProcessAllResultsMessage

        mgr = _make_manager_mock(
            accumulators={
                AccumulatorType.METRIC_RESULTS: _make_stub_accumulator(
                    _make_metrics_summary()
                ),
            },
        )

        with patch(
            "aiperf.records.records_manager.ExporterManager"
        ) as MockExporterManager:
            MockExporterManager.side_effect = RuntimeError("export init failed")

            result = await mgr._process_results(
                phase=CreditPhase.PROFILING, cancelled=False
            )

        # Should still return results
        assert result.results.records is not None
        # Exception logged
        mgr.exception.assert_called_once()
        # exported_artifacts should be empty on failure
        msg: ProcessAllResultsMessage = mgr.publish.call_args[0][0]
        assert msg.exported_artifacts == {}

    @pytest.mark.asyncio
    async def test_empty_accumulators_produces_empty_results(self) -> None:
        """With no accumulators, result has empty records and None for telemetry/server_metrics."""
        from aiperf.common.messages import ProcessAllResultsMessage

        mgr = _make_manager_mock(accumulators={})

        with patch("aiperf.records.records_manager.ExporterManager"):
            result = await mgr._process_results(
                phase=CreditPhase.PROFILING, cancelled=False
            )

        assert result.results.records == {}
        msg: ProcessAllResultsMessage = mgr.publish.call_args[0][0]
        assert msg.telemetry_results is None
        assert msg.server_metrics_results is None

    @pytest.mark.asyncio
    async def test_timeslice_results_merged(self) -> None:
        """Timeslice results from MetricsSummary are merged into ProfileResults."""
        summary = MetricsSummary(
            results={_STUB_METRIC_RESULT.tag: _STUB_METRIC_RESULT},
            timeslices={
                "0": {_STUB_METRIC_RESULT.tag: _STUB_METRIC_RESULT},
                "1": {_STUB_METRIC_RESULT.tag: _STUB_METRIC_RESULT},
            },
        )
        mgr = _make_manager_mock(
            accumulators={
                AccumulatorType.METRIC_RESULTS: _make_stub_accumulator(summary),
            },
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            result = await mgr._process_results(
                phase=CreditPhase.PROFILING, cancelled=False
            )

        assert result.results.timeslice_metric_results is not None
        assert len(result.results.timeslice_metric_results) == 2

    @pytest.mark.asyncio
    async def test_exporter_manager_receives_correct_domain_results(self) -> None:
        """ExporterManager is constructed with the correct typed domain results."""
        telemetry_data = _make_telemetry_export_data()
        server_data = _make_server_metrics_results()

        mgr = _make_manager_mock(
            accumulators={
                AccumulatorType.METRIC_RESULTS: _make_stub_accumulator(
                    _make_metrics_summary()
                ),
                AccumulatorType.GPU_TELEMETRY: _make_stub_accumulator(telemetry_data),
                AccumulatorType.SERVER_METRICS: _make_stub_accumulator(server_data),
            },
        )

        with patch(
            "aiperf.records.records_manager.ExporterManager"
        ) as MockExporterManager:
            mock_instance = MagicMock()
            mock_instance.export_data = AsyncMock()
            MockExporterManager.return_value = mock_instance

            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

            # Verify ExporterManager was constructed with correct domain results
            call_kwargs = MockExporterManager.call_args[1]
            assert call_kwargs["telemetry_results"] is telemetry_data
            assert call_kwargs["server_metrics_results"] is server_data


class TestProcessResultsExportContext:
    """Test ExportContext construction per accumulator type."""

    @pytest.mark.asyncio
    async def test_gpu_telemetry_context_has_no_end_ns(self) -> None:
        """GPU telemetry ExportContext should have no end_ns (includes final scrape)."""
        acc = _make_stub_accumulator(_make_telemetry_export_data())
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.GPU_TELEMETRY: acc},
            start_ns=100,
            requests_end_ns=200,
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        ctx: ExportContext = acc.export_results.call_args[0][0]
        assert ctx.start_ns == 100
        assert ctx.end_ns is None

    @pytest.mark.asyncio
    async def test_server_metrics_context_has_start_and_end(self) -> None:
        """Server metrics ExportContext should have both start and end timestamps."""
        acc = _make_stub_accumulator(_make_server_metrics_results())
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.SERVER_METRICS: acc},
            start_ns=100,
            requests_end_ns=200,
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        ctx: ExportContext = acc.export_results.call_args[0][0]
        assert ctx.start_ns is not None
        assert ctx.end_ns is not None

    @pytest.mark.asyncio
    async def test_metric_results_context_uses_error_tracker(self) -> None:
        """Metric results ExportContext uses the error tracker's phase summary."""
        from aiperf.common.models import ErrorDetails, ErrorDetailsCount

        acc = _make_stub_accumulator(_make_metrics_summary())
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
        )

        expected_summary = [
            ErrorDetailsCount(
                error_details=ErrorDetails(message="test error", code=500), count=3
            )
        ]
        mgr._error_tracker.get_error_summary_for_phase.return_value = expected_summary

        with patch("aiperf.records.records_manager.ExporterManager"):
            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        ctx: ExportContext = acc.export_results.call_args[0][0]
        assert ctx.error_summary == expected_summary

    @pytest.mark.asyncio
    async def test_cancelled_flag_propagated_to_all_contexts(self) -> None:
        """cancelled=True propagates to all accumulator ExportContexts."""
        acc_metrics = _make_stub_accumulator(_make_metrics_summary())
        acc_telemetry = _make_stub_accumulator(_make_telemetry_export_data())
        acc_server = _make_stub_accumulator(_make_server_metrics_results())

        mgr = _make_manager_mock(
            accumulators={
                AccumulatorType.METRIC_RESULTS: acc_metrics,
                AccumulatorType.GPU_TELEMETRY: acc_telemetry,
                AccumulatorType.SERVER_METRICS: acc_server,
            },
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=True)

        for acc in [acc_metrics, acc_telemetry, acc_server]:
            ctx: ExportContext = acc.export_results.call_args[0][0]
            assert ctx.cancelled is True

    @pytest.mark.asyncio
    async def test_cancelled_false_propagated(self) -> None:
        """cancelled=False propagates to ExportContexts."""
        acc = _make_stub_accumulator(_make_metrics_summary())
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        ctx: ExportContext = acc.export_results.call_args[0][0]
        assert ctx.cancelled is False

    @pytest.mark.asyncio
    async def test_cancelled_result_has_was_cancelled_flag(self) -> None:
        """ProcessRecordsResult.results.was_cancelled reflects the cancelled param."""
        mgr = _make_manager_mock(
            accumulators={
                AccumulatorType.METRIC_RESULTS: _make_stub_accumulator(
                    _make_metrics_summary()
                ),
            },
        )

        with patch("aiperf.records.records_manager.ExporterManager"):
            result = await mgr._process_results(
                phase=CreditPhase.PROFILING, cancelled=True
            )

        assert result.results.was_cancelled is True
