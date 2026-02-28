# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for _GrpcCallBase shared base class and its subclasses."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.transports.grpc.grpc_client import (
    GrpcBidiStreamCall,
    GrpcStreamCall,
    _GrpcCallBase,
)


class TestGrpcCallBaseInheritance:
    """Verify GrpcStreamCall and GrpcBidiStreamCall inherit from _GrpcCallBase."""

    def test_stream_call_is_base(self) -> None:
        assert issubclass(GrpcStreamCall, _GrpcCallBase)

    def test_bidi_stream_call_is_base(self) -> None:
        assert issubclass(GrpcBidiStreamCall, _GrpcCallBase)


class TestGrpcCallBaseSharedMethods:
    """Test shared methods via both GrpcStreamCall and GrpcBidiStreamCall."""

    @pytest.fixture(params=[GrpcStreamCall, GrpcBidiStreamCall])
    def call_wrapper(self, request: pytest.FixtureRequest) -> _GrpcCallBase:
        """Create a call wrapper of each subclass type with a mock call."""
        mock_call = AsyncMock()
        mock_call.initial_metadata = AsyncMock(return_value={"key": "value"})
        mock_call.trailing_metadata = AsyncMock(return_value={"trail": "data"})
        mock_call.cancel = MagicMock(return_value=True)
        return request.param(mock_call)

    @pytest.mark.asyncio
    async def test_initial_metadata(self, call_wrapper: _GrpcCallBase) -> None:
        """initial_metadata should delegate to underlying call."""
        result = await call_wrapper.initial_metadata()
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_trailing_metadata(self, call_wrapper: _GrpcCallBase) -> None:
        """trailing_metadata should delegate to underlying call."""
        result = await call_wrapper.trailing_metadata()
        assert result == {"trail": "data"}

    def test_cancel(self, call_wrapper: _GrpcCallBase) -> None:
        """cancel should delegate to underlying call."""
        result = call_wrapper.cancel()
        assert result is True

    @pytest.mark.asyncio
    async def test_async_iteration(self) -> None:
        """Both wrapper types should support async iteration identically."""
        chunks = [b"a", b"b", b"c"]

        for cls in (GrpcStreamCall, GrpcBidiStreamCall):

            async def async_gen():
                for c in chunks:
                    yield c

            mock_call = AsyncMock()
            mock_call.__aiter__ = lambda self: async_gen()

            wrapper = cls(mock_call)
            received = []
            async for chunk in wrapper:
                received.append(chunk)

            assert received == chunks, f"Failed for {cls.__name__}"

    @pytest.mark.asyncio
    async def test_async_iteration_empty_stream(self) -> None:
        """Both wrapper types should handle empty streams."""
        for cls in (GrpcStreamCall, GrpcBidiStreamCall):

            async def empty_gen():
                return
                yield  # noqa: RET504

            mock_call = AsyncMock()
            mock_call.__aiter__ = lambda self: empty_gen()

            wrapper = cls(mock_call)
            received = []
            async for chunk in wrapper:
                received.append(chunk)

            assert received == [], f"Failed for {cls.__name__}"


class TestGrpcBidiStreamCallWriteMethods:
    """Test bidi-specific write methods not present on GrpcStreamCall."""

    def test_stream_call_has_no_write(self) -> None:
        """GrpcStreamCall should NOT have write or done_writing."""
        assert not hasattr(GrpcStreamCall, "write")
        assert not hasattr(GrpcStreamCall, "done_writing")

    def test_bidi_call_has_write(self) -> None:
        """GrpcBidiStreamCall should have write and done_writing."""
        assert hasattr(GrpcBidiStreamCall, "write")
        assert hasattr(GrpcBidiStreamCall, "done_writing")

    @pytest.mark.asyncio
    async def test_write_delegates(self) -> None:
        mock_call = AsyncMock()
        mock_call.write = AsyncMock()
        bidi = GrpcBidiStreamCall(mock_call)

        await bidi.write(b"data")
        mock_call.write.assert_awaited_once_with(b"data")

    @pytest.mark.asyncio
    async def test_done_writing_delegates(self) -> None:
        mock_call = AsyncMock()
        mock_call.done_writing = AsyncMock()
        bidi = GrpcBidiStreamCall(mock_call)

        await bidi.done_writing()
        mock_call.done_writing.assert_awaited_once()
