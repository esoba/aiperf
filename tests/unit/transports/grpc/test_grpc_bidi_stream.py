# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GrpcBidiStreamCall."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from aiperf.transports.grpc.grpc_client import GrpcBidiStreamCall


class TestGrpcBidiStreamCall:
    @pytest.fixture
    def mock_call(self) -> AsyncMock:
        call = AsyncMock()
        call.write = AsyncMock()
        call.done_writing = AsyncMock()
        call.initial_metadata = AsyncMock(return_value={})
        call.trailing_metadata = AsyncMock(return_value={})
        call.cancel = MagicMock(return_value=True)
        return call

    @pytest.fixture
    def bidi_call(self, mock_call: AsyncMock) -> GrpcBidiStreamCall:
        return GrpcBidiStreamCall(mock_call)

    @pytest.mark.asyncio
    async def test_write(
        self, bidi_call: GrpcBidiStreamCall, mock_call: AsyncMock
    ) -> None:
        await bidi_call.write(b"test data")
        mock_call.write.assert_awaited_once_with(b"test data")

    @pytest.mark.asyncio
    async def test_write_multiple(
        self, bidi_call: GrpcBidiStreamCall, mock_call: AsyncMock
    ) -> None:
        """Multiple writes should all be forwarded."""
        await bidi_call.write(b"chunk1")
        await bidi_call.write(b"chunk2")
        await bidi_call.write(b"chunk3")
        assert mock_call.write.await_count == 3

    @pytest.mark.asyncio
    async def test_write_empty_bytes(
        self, bidi_call: GrpcBidiStreamCall, mock_call: AsyncMock
    ) -> None:
        """Writing empty bytes should be valid."""
        await bidi_call.write(b"")
        mock_call.write.assert_awaited_once_with(b"")

    @pytest.mark.asyncio
    async def test_done_writing(
        self, bidi_call: GrpcBidiStreamCall, mock_call: AsyncMock
    ) -> None:
        await bidi_call.done_writing()
        mock_call.done_writing.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initial_metadata(
        self, bidi_call: GrpcBidiStreamCall, mock_call: AsyncMock
    ) -> None:
        result = await bidi_call.initial_metadata()
        mock_call.initial_metadata.assert_awaited_once()
        assert result == {}

    @pytest.mark.asyncio
    async def test_trailing_metadata(
        self, bidi_call: GrpcBidiStreamCall, mock_call: AsyncMock
    ) -> None:
        result = await bidi_call.trailing_metadata()
        mock_call.trailing_metadata.assert_awaited_once()
        assert result == {}

    def test_cancel(self, bidi_call: GrpcBidiStreamCall, mock_call: AsyncMock) -> None:
        result = bidi_call.cancel()
        mock_call.cancel.assert_called_once()
        assert result is True

    def test_cancel_returns_false_when_already_cancelled(
        self, mock_call: AsyncMock
    ) -> None:
        """Cancel should return False when the call was already cancelled."""
        mock_call.cancel = MagicMock(return_value=False)
        bidi_call = GrpcBidiStreamCall(mock_call)
        assert bidi_call.cancel() is False

    @pytest.mark.asyncio
    async def test_async_iteration(self) -> None:
        """Should iterate over response chunks from the underlying call."""
        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        async def async_gen():
            for c in chunks:
                yield c

        mock_call = AsyncMock()
        mock_call.__aiter__ = lambda self: async_gen()

        bidi_call = GrpcBidiStreamCall(mock_call)
        received = []
        async for chunk in bidi_call:
            received.append(chunk)

        assert received == chunks

    @pytest.mark.asyncio
    async def test_async_iteration_empty(self) -> None:
        """Empty stream should produce no chunks."""

        async def async_gen():
            return
            yield  # noqa: RET504

        mock_call = AsyncMock()
        mock_call.__aiter__ = lambda self: async_gen()

        bidi_call = GrpcBidiStreamCall(mock_call)
        received = []
        async for chunk in bidi_call:
            received.append(chunk)

        assert received == []

    @pytest.mark.asyncio
    async def test_write_propagates_exception(self, mock_call: AsyncMock) -> None:
        """Exceptions from write should propagate."""
        mock_call.write = AsyncMock(
            side_effect=grpc.aio.AioRpcError(
                code=grpc.StatusCode.UNAVAILABLE,
                initial_metadata=grpc.aio.Metadata(),
                trailing_metadata=grpc.aio.Metadata(),
                details="Connection lost",
            )
        )
        bidi_call = GrpcBidiStreamCall(mock_call)

        with pytest.raises(grpc.aio.AioRpcError):
            await bidi_call.write(b"data")

    @pytest.mark.asyncio
    async def test_done_writing_propagates_exception(
        self, mock_call: AsyncMock
    ) -> None:
        """Exceptions from done_writing should propagate."""
        mock_call.done_writing = AsyncMock(
            side_effect=grpc.aio.AioRpcError(
                code=grpc.StatusCode.INTERNAL,
                initial_metadata=grpc.aio.Metadata(),
                trailing_metadata=grpc.aio.Metadata(),
                details="Internal error",
            )
        )
        bidi_call = GrpcBidiStreamCall(mock_call)

        with pytest.raises(grpc.aio.AioRpcError):
            await bidi_call.done_writing()
