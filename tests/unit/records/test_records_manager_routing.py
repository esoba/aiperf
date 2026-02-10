# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RecordsManager routing table infrastructure.

Tests the _build_routing_table() and _dispatch_record() methods using
stub processors without instantiating the full RecordsManager service.

The routing table uses record_type strings (matching ClassVars on record
classes and record_types metadata in plugins.yaml) for O(1) dispatch.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.accumulator_protocols import (
    AccumulatorProtocol,
    AccumulatorResult,
    ExportContext,
    StreamExporterProtocol,
    SummaryContext,
)
from aiperf.plugin.enums import AccumulatorType, StreamExporterType

# ---------------------------------------------------------------------------
# Mapping from processor enum to record_type strings for test routing
# ---------------------------------------------------------------------------

_ACCUMULATOR_RECORD_TYPES: dict[AccumulatorType, list[str]] = {
    AccumulatorType.METRIC_RESULTS: ["fake_record_a"],
    AccumulatorType.GPU_TELEMETRY: ["fake_record_b"],
}

_STREAM_EXPORTER_RECORD_TYPES: dict[StreamExporterType, list[str]] = {
    StreamExporterType.RECORD_EXPORT: ["fake_record_a"],
}

# ---------------------------------------------------------------------------
# Stub record types for testing
# ---------------------------------------------------------------------------


class StubAccumulatorResult:
    """Minimal AccumulatorResult for testing."""

    def to_json(self) -> Any:
        return {}

    def to_csv(self) -> list[dict[str, Any]]:
        return []


class FakeRecordA:
    """Stub record type A (e.g., metric records)."""

    record_type: ClassVar[str] = "fake_record_a"


class FakeRecordB:
    """Stub record type B (e.g., telemetry records)."""

    record_type: ClassVar[str] = "fake_record_b"


class FakeRecordC:
    """Stub record type C (unregistered)."""

    record_type: ClassVar[str] = "fake_record_c"


# ---------------------------------------------------------------------------
# Stub processors for testing
# ---------------------------------------------------------------------------


class StubAccumulatorA:
    """Accumulator that handles FakeRecordA."""

    def __init__(self) -> None:
        self.process_record = AsyncMock()

    async def summarize(
        self, ctx: SummaryContext | None = None
    ) -> StubAccumulatorResult:
        return StubAccumulatorResult()

    def query_time_range(self, start_ns: int, end_ns: int) -> list[Any]:
        return []

    async def export_results(self, ctx: ExportContext) -> StubAccumulatorResult:
        return StubAccumulatorResult()


class StubAccumulatorB:
    """Accumulator that handles FakeRecordB."""

    def __init__(self) -> None:
        self.process_record = AsyncMock()

    async def summarize(
        self, ctx: SummaryContext | None = None
    ) -> StubAccumulatorResult:
        return StubAccumulatorResult()

    def query_time_range(self, start_ns: int, end_ns: int) -> list[Any]:
        return []

    async def export_results(self, ctx: ExportContext) -> StubAccumulatorResult:
        return StubAccumulatorResult()


class StubStreamExporter:
    """Stream exporter that handles FakeRecordA."""

    def __init__(self) -> None:
        self.process_record = AsyncMock()
        self.finalize = AsyncMock()
        self.get_export_info = MagicMock()


# ---------------------------------------------------------------------------
# Helper: build routing table from processor lists
# ---------------------------------------------------------------------------


def build_routing_table(
    accumulators: dict, stream_exporters: dict
) -> dict[str, list[Any]]:
    """Reproduce RecordsManager._build_routing_table() logic for testing.

    Uses string-based record_type mappings (mirroring plugins.yaml metadata)
    rather than ClassVars on the processors themselves.
    """
    table: dict[str, list[Any]] = {}
    for key, proc in accumulators.items():
        for rt in _ACCUMULATOR_RECORD_TYPES.get(key, []):
            table.setdefault(rt, []).append(proc)
    for key, proc in stream_exporters.items():
        for rt in _STREAM_EXPORTER_RECORD_TYPES.get(key, []):
            table.setdefault(rt, []).append(proc)
    return table


# ---------------------------------------------------------------------------
# Tests: Routing table construction
# ---------------------------------------------------------------------------


class TestBuildRoutingTable:
    def test_single_accumulator(self) -> None:
        acc = StubAccumulatorA()
        table = build_routing_table({AccumulatorType.METRIC_RESULTS: acc}, {})
        assert "fake_record_a" in table
        assert table["fake_record_a"] == [acc]
        assert "fake_record_b" not in table

    def test_accumulator_and_stream_exporter_fan_out(self) -> None:
        """Both accumulator and exporter handle the same record type."""
        acc = StubAccumulatorA()
        exp = StubStreamExporter()
        table = build_routing_table(
            {AccumulatorType.METRIC_RESULTS: acc},
            {StreamExporterType.RECORD_EXPORT: exp},
        )
        assert "fake_record_a" in table
        assert acc in table["fake_record_a"]
        assert exp in table["fake_record_a"]
        assert len(table["fake_record_a"]) == 2

    def test_different_record_types(self) -> None:
        acc_a = StubAccumulatorA()
        acc_b = StubAccumulatorB()
        table = build_routing_table(
            {
                AccumulatorType.METRIC_RESULTS: acc_a,
                AccumulatorType.GPU_TELEMETRY: acc_b,
            },
            {},
        )
        assert table["fake_record_a"] == [acc_a]
        assert table["fake_record_b"] == [acc_b]

    def test_empty_routing_table(self) -> None:
        table = build_routing_table({}, {})
        assert table == {}

    def test_handler_order_accumulators_before_exporters(self) -> None:
        """Accumulators appear before stream exporters in handler list."""
        acc = StubAccumulatorA()
        exp = StubStreamExporter()
        table = build_routing_table(
            {AccumulatorType.METRIC_RESULTS: acc},
            {StreamExporterType.RECORD_EXPORT: exp},
        )
        handlers = table["fake_record_a"]
        assert handlers[0] is acc
        assert handlers[1] is exp


# ---------------------------------------------------------------------------
# Tests: Record dispatch
# ---------------------------------------------------------------------------


class TestDispatchRecord:
    """Test _dispatch_record logic using a mock RecordsManager."""

    def _make_manager_mock(
        self,
        accumulators: dict,
        stream_exporters: dict,
    ) -> MagicMock:
        """Create a mock with routing table and dispatch method wired up."""
        from aiperf.records.records_manager import RecordsManager

        mgr = MagicMock()
        mgr._accumulators = accumulators
        mgr._stream_exporters = stream_exporters
        mgr._routing_table = build_routing_table(accumulators, stream_exporters)
        mgr.warning = MagicMock()
        mgr.error = MagicMock()
        # Bind the real _dispatch_record method to our mock
        mgr._dispatch_record = RecordsManager._dispatch_record.__get__(mgr)
        return mgr

    @pytest.mark.asyncio
    async def test_dispatch_calls_all_matching_handlers(self) -> None:
        acc = StubAccumulatorA()
        exp = StubStreamExporter()
        mgr = self._make_manager_mock(
            {AccumulatorType.METRIC_RESULTS: acc},
            {StreamExporterType.RECORD_EXPORT: exp},
        )

        record = FakeRecordA()
        errors = await mgr._dispatch_record(record)

        acc.process_record.assert_called_once_with(record)
        exp.process_record.assert_called_once_with(record)
        assert errors == []

    @pytest.mark.asyncio
    async def test_dispatch_only_matching_type(self) -> None:
        acc_a = StubAccumulatorA()
        acc_b = StubAccumulatorB()
        mgr = self._make_manager_mock(
            {
                AccumulatorType.METRIC_RESULTS: acc_a,
                AccumulatorType.GPU_TELEMETRY: acc_b,
            },
            {},
        )

        await mgr._dispatch_record(FakeRecordA())
        acc_a.process_record.assert_called_once()
        acc_b.process_record.assert_not_called()

        await mgr._dispatch_record(FakeRecordB())
        acc_b.process_record.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_unknown_record_type_warns(self) -> None:
        mgr = self._make_manager_mock({}, {})

        errors = await mgr._dispatch_record(FakeRecordC())

        mgr.warning.assert_called_once()
        assert "fake_record_c" in mgr.warning.call_args[0][0]
        assert errors == []

    @pytest.mark.asyncio
    async def test_dispatch_handler_exception_logged(self) -> None:
        """One handler raising does not prevent other handlers from running."""
        acc = StubAccumulatorA()
        acc.process_record.side_effect = RuntimeError("boom")
        exp = StubStreamExporter()
        mgr = self._make_manager_mock(
            {AccumulatorType.METRIC_RESULTS: acc},
            {StreamExporterType.RECORD_EXPORT: exp},
        )

        errors = await mgr._dispatch_record(FakeRecordA())

        # Exporter should still be called despite accumulator failure
        exp.process_record.assert_called_once()
        # Error should be logged
        mgr.error.assert_called_once()
        assert "boom" in mgr.error.call_args[0][0]
        # Error should be returned
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    @pytest.mark.asyncio
    async def test_dispatch_multiple_handler_exceptions(self) -> None:
        """Multiple handler failures are each logged independently."""
        acc = StubAccumulatorA()
        acc.process_record.side_effect = RuntimeError("acc error")
        exp = StubStreamExporter()
        exp.process_record.side_effect = ValueError("exp error")
        mgr = self._make_manager_mock(
            {AccumulatorType.METRIC_RESULTS: acc},
            {StreamExporterType.RECORD_EXPORT: exp},
        )

        errors = await mgr._dispatch_record(FakeRecordA())

        assert mgr.error.call_count == 2
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# Tests: Protocol conformance of stubs
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_stub_accumulator_matches_protocol(self) -> None:
        assert isinstance(StubAccumulatorA(), AccumulatorProtocol)

    def test_stub_stream_exporter_matches_protocol(self) -> None:
        assert isinstance(StubStreamExporter(), StreamExporterProtocol)

    def test_stub_result_matches_accumulator_result(self) -> None:
        assert isinstance(StubAccumulatorResult(), AccumulatorResult)


# ---------------------------------------------------------------------------
# Tests: Stream exporter finalize
# ---------------------------------------------------------------------------


class TestFinalizeStreamExporters:
    """Test _finalize_stream_exporters logic using a mock RecordsManager."""

    def _make_manager_mock(
        self,
        stream_exporters: dict,
    ) -> MagicMock:
        """Create a mock with _stream_exporters and _finalize_stream_exporters wired up."""
        from aiperf.records.records_manager import RecordsManager

        mgr = MagicMock()
        mgr._stream_exporters = stream_exporters
        mgr.debug = MagicMock()
        mgr.error = MagicMock()
        mgr._finalize_stream_exporters = (
            RecordsManager._finalize_stream_exporters.__get__(mgr)
        )
        return mgr

    @pytest.mark.asyncio
    async def test_finalize_calls_all_exporters(self) -> None:
        exp1 = StubStreamExporter()
        exp2 = StubStreamExporter()
        mgr = self._make_manager_mock(
            {
                StreamExporterType.RECORD_EXPORT: exp1,
                StreamExporterType.GPU_TELEMETRY_JSONL_WRITER: exp2,
            },
        )

        await mgr._finalize_stream_exporters()

        exp1.finalize.assert_called_once()
        exp2.finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_finalize_empty_exporters_noop(self) -> None:
        mgr = self._make_manager_mock({})
        await mgr._finalize_stream_exporters()
        # No error, no crash

    @pytest.mark.asyncio
    async def test_finalize_error_logged_per_exporter(self) -> None:
        """One exporter failing does not prevent others from finalizing."""
        exp1 = StubStreamExporter()
        exp1.finalize.side_effect = RuntimeError("flush failed")
        exp2 = StubStreamExporter()
        mgr = self._make_manager_mock(
            {
                StreamExporterType.RECORD_EXPORT: exp1,
                StreamExporterType.GPU_TELEMETRY_JSONL_WRITER: exp2,
            },
        )

        await mgr._finalize_stream_exporters()

        # Both should be called (gather runs all concurrently)
        exp1.finalize.assert_called_once()
        exp2.finalize.assert_called_once()
        # Error logged for the failing one
        mgr.error.assert_called_once()
        assert "flush failed" in mgr.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_finalize_multiple_errors(self) -> None:
        exp1 = StubStreamExporter()
        exp1.finalize.side_effect = RuntimeError("error 1")
        exp2 = StubStreamExporter()
        exp2.finalize.side_effect = ValueError("error 2")
        mgr = self._make_manager_mock(
            {
                StreamExporterType.RECORD_EXPORT: exp1,
                StreamExporterType.GPU_TELEMETRY_JSONL_WRITER: exp2,
            },
        )

        await mgr._finalize_stream_exporters()

        assert mgr.error.call_count == 2
