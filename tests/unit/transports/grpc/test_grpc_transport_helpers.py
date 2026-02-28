# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for extracted GrpcTransport helper methods:
_process_stream_chunk, _finalize_stream, _run_with_cancel_timer.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.models import RequestRecord
from aiperf.transports.grpc.grpc_client import _GrpcCallBase
from aiperf.transports.grpc.grpc_transport import GrpcTransport
from aiperf.transports.grpc.stream_chunk import StreamChunk
from aiperf.transports.grpc.trace_data import GrpcTraceData


# ---------------------------------------------------------------------------
# _process_stream_chunk
# ---------------------------------------------------------------------------
class TestProcessStreamChunk:
    """Tests for GrpcTransport._process_stream_chunk."""

    @pytest.fixture
    def record(self) -> RequestRecord:
        return RequestRecord(start_perf_ns=1000)

    @pytest.fixture
    def trace_data(self) -> GrpcTraceData:
        return GrpcTraceData()

    def test_success_chunk_returns_true(
        self, record: RequestRecord, trace_data: GrpcTraceData
    ) -> None:
        """Successful chunk should return True (continue processing)."""
        chunk = StreamChunk(
            error_message=None, response_dict={"text": "hello"}, response_size=42
        )

        result = GrpcTransport._process_stream_chunk(chunk, record, trace_data)

        assert result is True

    def test_success_chunk_appends_response(
        self, record: RequestRecord, trace_data: GrpcTraceData
    ) -> None:
        """Successful chunk should append a TextResponse to record."""
        chunk = StreamChunk(
            error_message=None, response_dict={"text": "hello"}, response_size=42
        )

        GrpcTransport._process_stream_chunk(chunk, record, trace_data)

        assert len(record.responses) == 1
        assert '"text"' in record.responses[0].text
        assert record.responses[0].content_type == "application/json"

    def test_success_chunk_updates_trace_data(
        self, record: RequestRecord, trace_data: GrpcTraceData
    ) -> None:
        """Successful chunk should append to trace_data.response_chunks."""
        chunk = StreamChunk(
            error_message=None, response_dict={"data": "x"}, response_size=99
        )

        GrpcTransport._process_stream_chunk(chunk, record, trace_data)

        assert len(trace_data.response_chunks) == 1
        perf_ns, size = trace_data.response_chunks[0]
        assert size == 99
        assert perf_ns > 0

    def test_first_chunk_sets_recv_start(
        self, record: RequestRecord, trace_data: GrpcTraceData
    ) -> None:
        """First chunk should set recv_start_perf_ns on record and trace_data."""
        chunk = StreamChunk(
            error_message=None, response_dict={"x": 1}, response_size=10
        )

        GrpcTransport._process_stream_chunk(chunk, record, trace_data)

        assert trace_data.response_receive_start_perf_ns is not None
        assert trace_data.response_headers_received_perf_ns is not None
        assert record.recv_start_perf_ns is not None
        assert record.recv_start_perf_ns == trace_data.response_receive_start_perf_ns

    def test_second_chunk_does_not_overwrite_recv_start(
        self, record: RequestRecord, trace_data: GrpcTraceData
    ) -> None:
        """Subsequent chunks should not overwrite recv_start_perf_ns."""
        chunk1 = StreamChunk(
            error_message=None, response_dict={"x": 1}, response_size=10
        )
        chunk2 = StreamChunk(
            error_message=None, response_dict={"x": 2}, response_size=20
        )

        GrpcTransport._process_stream_chunk(chunk1, record, trace_data)
        first_recv = record.recv_start_perf_ns

        GrpcTransport._process_stream_chunk(chunk2, record, trace_data)

        assert record.recv_start_perf_ns == first_recv

    def test_error_chunk_returns_false(
        self, record: RequestRecord, trace_data: GrpcTraceData
    ) -> None:
        """Error chunk should return False (stop processing)."""
        chunk = StreamChunk(
            error_message="bad things happened", response_dict=None, response_size=5
        )

        result = GrpcTransport._process_stream_chunk(chunk, record, trace_data)

        assert result is False

    def test_error_chunk_sets_record_error(
        self, record: RequestRecord, trace_data: GrpcTraceData
    ) -> None:
        """Error chunk should populate record.error and set status 500."""
        chunk = StreamChunk(
            error_message="stream error", response_dict=None, response_size=5
        )

        GrpcTransport._process_stream_chunk(chunk, record, trace_data)

        assert record.error is not None
        assert record.error.type == "gRPC:STREAM_ERROR"
        assert record.error.message == "stream error"
        assert record.error.code == 500
        assert record.status == 500
        assert record.end_perf_ns is not None
        assert trace_data.error_timestamp_perf_ns is not None

    def test_error_chunk_does_not_append_response(
        self, record: RequestRecord, trace_data: GrpcTraceData
    ) -> None:
        """Error chunk should not append to record.responses."""
        chunk = StreamChunk(error_message="error", response_dict=None, response_size=5)

        GrpcTransport._process_stream_chunk(chunk, record, trace_data)

        assert len(record.responses) == 0


# ---------------------------------------------------------------------------
# _finalize_stream
# ---------------------------------------------------------------------------
class TestFinalizeStream:
    """Tests for GrpcTransport._finalize_stream."""

    @pytest.fixture
    def mock_call(self) -> _GrpcCallBase:
        call = MagicMock(spec=_GrpcCallBase)
        call.trailing_metadata = AsyncMock(return_value=())
        return call

    def test_sets_end_ns(self, mock_call: _GrpcCallBase) -> None:
        """Should set end_perf_ns on record."""
        record = RequestRecord(start_perf_ns=1000)
        trace_data = GrpcTraceData()

        GrpcTransport._finalize_stream(mock_call, record, trace_data)

        assert record.end_perf_ns is not None
        assert trace_data.response_receive_end_perf_ns is not None

    def test_sets_status_200_on_success(self, mock_call: _GrpcCallBase) -> None:
        """Should set status 200 when no error."""
        record = RequestRecord(start_perf_ns=1000)
        trace_data = GrpcTraceData()

        GrpcTransport._finalize_stream(mock_call, record, trace_data)

        assert record.status == 200
        assert trace_data.grpc_status_code == 0
        assert trace_data.response_status_code == 200
        assert trace_data.response_reason == "OK"

    def test_preserves_existing_end_ns(self, mock_call: _GrpcCallBase) -> None:
        """Should not overwrite end_perf_ns if already set (e.g., by error)."""
        record = RequestRecord(start_perf_ns=1000)
        record.end_perf_ns = 5000
        trace_data = GrpcTraceData()

        GrpcTransport._finalize_stream(mock_call, record, trace_data)

        assert record.end_perf_ns == 5000

    def test_does_not_set_status_on_error(self, mock_call: _GrpcCallBase) -> None:
        """Should not overwrite error status to 200."""
        from aiperf.common.models import ErrorDetails

        record = RequestRecord(start_perf_ns=1000)
        record.error = ErrorDetails(type="gRPC:STREAM_ERROR", message="err", code=500)
        record.status = 500
        trace_data = GrpcTraceData()

        GrpcTransport._finalize_stream(mock_call, record, trace_data)

        assert record.status == 500
        assert trace_data.grpc_status_code is None


# ---------------------------------------------------------------------------
# _run_with_cancel_timer
# ---------------------------------------------------------------------------
class TestRunWithCancelTimer:
    """Tests for GrpcTransport._run_with_cancel_timer."""

    @pytest.fixture
    def transport(self) -> GrpcTransport:
        """Create a minimal transport (not fully initialized, just for helpers)."""
        from aiperf.common.enums import ModelSelectionStrategy
        from aiperf.common.models.model_endpoint_info import (
            EndpointInfo,
            ModelEndpointInfo,
            ModelInfo,
            ModelListInfo,
        )
        from aiperf.plugin.enums import EndpointType

        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="m")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.KSERVE_V2_INFER,
                base_url="grpc://localhost:8001",
            ),
        )
        return GrpcTransport(model_endpoint=model_endpoint)

    @pytest.mark.asyncio
    async def test_completes_normally(self, transport: GrpcTransport) -> None:
        """Coroutine completing before timeout should not set cancellation."""
        record = RequestRecord(start_perf_ns=1000)

        async def fast_coro() -> None:
            pass

        await transport._run_with_cancel_timer(
            fast_coro(), cancel_after_ns=10 * 10**9, record=record
        )

        assert record.cancellation_perf_ns is None
        assert record.error is None

    @pytest.mark.asyncio
    async def test_timeout_sets_cancellation(self, transport: GrpcTransport) -> None:
        """Coroutine exceeding timeout should set cancellation error."""
        record = RequestRecord(start_perf_ns=1000)

        async def slow_coro() -> None:
            await asyncio.sleep(10)

        await transport._run_with_cancel_timer(
            slow_coro(), cancel_after_ns=1, record=record
        )

        assert record.cancellation_perf_ns is not None
        assert record.end_perf_ns is not None
        assert record.error is not None
        assert record.error.type == "RequestCancellationError"
        assert record.error.code == 499

    @pytest.mark.asyncio
    async def test_timeout_cancels_grpc_call(self, transport: GrpcTransport) -> None:
        """Timeout should also cancel the gRPC call wrapper."""
        record = RequestRecord(start_perf_ns=1000)
        mock_call = MagicMock()
        mock_call.cancel = MagicMock(return_value=True)

        async def slow_coro() -> None:
            await asyncio.sleep(10)

        await transport._run_with_cancel_timer(
            slow_coro(), cancel_after_ns=1, record=record, call=mock_call
        )

        mock_call.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_call_no_cancel(self, transport: GrpcTransport) -> None:
        """When call is None, timeout should still work without error."""
        record = RequestRecord(start_perf_ns=1000)

        async def slow_coro() -> None:
            await asyncio.sleep(10)

        await transport._run_with_cancel_timer(
            slow_coro(), cancel_after_ns=1, record=record, call=None
        )

        assert record.error is not None
        assert record.error.type == "RequestCancellationError"

    @pytest.mark.asyncio
    async def test_label_in_debug_message(self, transport: GrpcTransport) -> None:
        """Custom label should appear in the cancellation error message context."""
        record = RequestRecord(start_perf_ns=1000)

        async def slow_coro() -> None:
            await asyncio.sleep(10)

        # Just verify it doesn't error with a custom label
        await transport._run_with_cancel_timer(
            slow_coro(), cancel_after_ns=1, record=record, label="custom-label"
        )

        assert record.error is not None
