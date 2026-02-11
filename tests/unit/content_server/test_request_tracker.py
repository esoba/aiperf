# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for content server request tracker and tracking middleware."""

import time
from unittest.mock import AsyncMock

import pytest

from aiperf.content_server.models import ContentRequestRecord
from aiperf.content_server.request_tracker import (
    RequestTracker,
    TrackingMiddleware,
    _extract_headers,
)


def _record(tracker: RequestTracker, **overrides) -> None:
    """Helper to add a minimal record to the tracker."""
    defaults = dict(
        timestamp_ns=time.time_ns(),
        method="GET",
        path="/",
        status_code=200,
    )
    defaults.update(overrides)
    tracker.record(ContentRequestRecord(**defaults))


# ---------------------------------------------------------------------------
# _extract_headers
# ---------------------------------------------------------------------------


class TestExtractHeaders:
    def test_basic_headers(self) -> None:
        raw = [(b"Content-Type", b"image/png"), (b"X-Custom", b"value")]
        result = _extract_headers(raw)
        assert result == {"content-type": "image/png", "x-custom": "value"}

    def test_duplicate_headers_joined(self) -> None:
        raw = [(b"set-cookie", b"a=1"), (b"Set-Cookie", b"b=2")]
        result = _extract_headers(raw)
        assert result == {"set-cookie": "a=1, b=2"}

    def test_empty(self) -> None:
        assert _extract_headers([]) == {}


# ---------------------------------------------------------------------------
# RequestTracker unit tests
# ---------------------------------------------------------------------------


class TestRequestTracker:
    def test_initial_state(self) -> None:
        tracker = RequestTracker()
        assert tracker.total_requests == 0
        assert tracker.total_bytes_served == 0
        snapshot = tracker.snapshot()
        assert snapshot.total_requests == 0
        assert snapshot.records == []

    def test_record_single_request(self) -> None:
        tracker = RequestTracker()
        now = time.time_ns()
        tracker.record(
            ContentRequestRecord(
                timestamp_ns=now,
                method="GET",
                path="test.png",
                status_code=200,
                body_bytes=1024,
                latency_ns=5000,
                content_type="image/png",
            )
        )
        assert tracker.total_requests == 1
        assert tracker.total_bytes_served == 1024

        snapshot = tracker.snapshot()
        assert len(snapshot.records) == 1
        rec = snapshot.records[0]
        assert rec.path == "test.png"
        assert rec.status_code == 200
        assert rec.timestamp_ns == now

    def test_record_multiple_requests(self) -> None:
        tracker = RequestTracker()
        for i in range(5):
            _record(tracker, path=f"file_{i}", body_bytes=100)
        assert tracker.total_requests == 5
        assert tracker.total_bytes_served == 500

    def test_bounded_eviction(self) -> None:
        tracker = RequestTracker(max_records=3)
        for i in range(5):
            _record(tracker, path=f"file_{i}", body_bytes=10)
        snapshot = tracker.snapshot()
        assert len(snapshot.records) == 3
        assert snapshot.records[0].path == "file_2"
        assert snapshot.records[2].path == "file_4"

    def test_counters_survive_eviction(self) -> None:
        """Running counters reflect lifetime totals even after eviction."""
        tracker = RequestTracker(max_records=2)
        for i in range(10):
            _record(tracker, path=f"file_{i}", body_bytes=100)
        assert tracker.total_requests == 10
        assert tracker.total_bytes_served == 1000
        snapshot = tracker.snapshot()
        assert snapshot.total_requests == 10
        assert snapshot.total_bytes_served == 1000
        assert len(snapshot.records) == 2

    @pytest.mark.parametrize(
        "status_code,body_bytes",
        [(200, 1024), (403, 0), (404, 0)],
    )
    def test_record_various_status_codes(
        self, status_code: int, body_bytes: int
    ) -> None:
        tracker = RequestTracker()
        _record(tracker, status_code=status_code, body_bytes=body_bytes)
        assert tracker.snapshot().records[0].status_code == status_code


# ---------------------------------------------------------------------------
# TrackingMiddleware tests
# ---------------------------------------------------------------------------


def _scope(
    *,
    method: str = "GET",
    path: str = "/content/test.png",
    query_string: bytes = b"",
    http_version: str = "1.1",
    client: tuple[str, int] | None = ("127.0.0.1", 54321),
    headers: list[tuple[bytes, bytes]] | None = None,
) -> dict:
    """Build a minimal ASGI HTTP scope."""
    return {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query_string,
        "http_version": http_version,
        "client": client,
        "headers": headers
        or [
            (b"host", b"localhost:8090"),
            (b"user-agent", b"test-client/1.0"),
            (b"accept", b"*/*"),
        ],
    }


class TestTrackingMiddleware:
    async def test_captures_full_request_response(self) -> None:
        """Verifies every field we capture in a normal 200 response."""
        tracker = RequestTracker()

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"image/png"),
                        (b"content-length", b"400"),
                        (b"etag", b'"abc123"'),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b"\x89PNG" * 100})

        mw = TrackingMiddleware(app, tracker)
        await mw(
            _scope(
                path="/content/img.png",
                query_string=b"w=100&h=100",
                http_version="1.1",
                client=("10.0.0.5", 9999),
                headers=[
                    (b"host", b"myhost:8090"),
                    (b"user-agent", b"LLM-Server/2.0"),
                    (b"accept", b"image/*"),
                    (b"x-request-id", b"req-001"),
                ],
            ),
            AsyncMock(),
            AsyncMock(),
        )

        snapshot = tracker.snapshot()
        assert snapshot.total_requests == 1
        rec = snapshot.records[0]

        # Identity
        assert rec.method == "GET"
        assert rec.path == "/content/img.png"
        assert rec.query_string == "w=100&h=100"
        assert rec.http_version == "1.1"

        # Client
        assert rec.client_host == "10.0.0.5"
        assert rec.client_port == 9999

        # Request headers
        assert rec.request_headers["host"] == "myhost:8090"
        assert rec.request_headers["user-agent"] == "LLM-Server/2.0"
        assert rec.request_headers["accept"] == "image/*"
        assert rec.request_headers["x-request-id"] == "req-001"

        # Response
        assert rec.status_code == 200
        assert rec.content_type == "image/png"
        assert rec.response_headers["content-length"] == "400"
        assert rec.response_headers["etag"] == '"abc123"'

        # Transfer
        assert rec.body_bytes == 400
        assert rec.body_chunk_count == 1

        # Timing
        assert rec.latency_ns > 0
        assert rec.time_to_first_byte_ns > 0
        assert rec.time_to_first_body_byte_ns > 0
        assert rec.time_to_first_byte_ns <= rec.time_to_first_body_byte_ns
        assert rec.time_to_first_body_byte_ns <= rec.latency_ns

        # No error
        assert rec.error is None

    async def test_captures_404_response(self) -> None:
        tracker = RequestTracker()

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"Not Found"})

        mw = TrackingMiddleware(app, tracker)
        await mw(_scope(path="/content/missing"), AsyncMock(), AsyncMock())

        rec = tracker.snapshot().records[0]
        assert rec.status_code == 404
        assert rec.content_type == "text/plain"
        assert rec.body_bytes == 9

    async def test_accumulates_chunked_body(self) -> None:
        """body_bytes and body_chunk_count accumulate across multiple body messages."""
        tracker = RequestTracker()

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"aaa"})
            await send({"type": "http.response.body", "body": b"bbbb"})
            await send({"type": "http.response.body", "body": b"cc"})

        mw = TrackingMiddleware(app, tracker)
        await mw(_scope(), AsyncMock(), AsyncMock())

        rec = tracker.snapshot().records[0]
        assert rec.body_bytes == 9
        assert rec.body_chunk_count == 3
        assert rec.transfer_duration_ns >= 0

    async def test_skips_empty_body_chunks(self) -> None:
        """Empty body chunks should not increment chunk count."""
        tracker = RequestTracker()

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            await send({"type": "http.response.body", "body": b"data"})
            await send({"type": "http.response.body", "body": b""})

        mw = TrackingMiddleware(app, tracker)
        await mw(_scope(), AsyncMock(), AsyncMock())

        rec = tracker.snapshot().records[0]
        assert rec.body_bytes == 4
        assert rec.body_chunk_count == 1

    async def test_skips_non_http_scopes(self) -> None:
        """Websocket and lifespan scopes pass through untracked."""
        tracker = RequestTracker()
        inner_called = False

        async def app(scope, receive, send):
            nonlocal inner_called
            inner_called = True

        mw = TrackingMiddleware(app, tracker)
        await mw({"type": "websocket"}, AsyncMock(), AsyncMock())

        assert inner_called
        assert tracker.total_requests == 0

    async def test_records_on_handler_exception(self) -> None:
        """If the handler raises, the request is still recorded with error field."""
        tracker = RequestTracker()

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 500, "headers": []})
            raise ValueError("something broke")

        mw = TrackingMiddleware(app, tracker)
        with pytest.raises(ValueError, match="something broke"):
            await mw(_scope(path="/content/fail"), AsyncMock(), AsyncMock())

        assert tracker.total_requests == 1
        rec = tracker.snapshot().records[0]
        assert rec.status_code == 500
        assert rec.error == "ValueError: something broke"
        assert rec.latency_ns > 0

    async def test_wall_clock_timestamp_at_arrival(self) -> None:
        """timestamp_ns should reflect when the request arrived, not completed."""
        tracker = RequestTracker()
        before = time.time_ns()

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = TrackingMiddleware(app, tracker)
        await mw(_scope(), AsyncMock(), AsyncMock())
        after = time.time_ns()

        rec = tracker.snapshot().records[0]
        assert before <= rec.timestamp_ns <= after

    async def test_timing_ordering(self) -> None:
        """TTFB <= TTFBB <= latency must always hold."""
        tracker = RequestTracker()

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"hello"})

        mw = TrackingMiddleware(app, tracker)
        await mw(_scope(), AsyncMock(), AsyncMock())

        rec = tracker.snapshot().records[0]
        assert rec.time_to_first_byte_ns > 0
        assert rec.time_to_first_byte_ns <= rec.time_to_first_body_byte_ns
        assert rec.time_to_first_body_byte_ns <= rec.latency_ns

    async def test_no_client_info(self) -> None:
        """Handles scope with no client tuple gracefully."""
        tracker = RequestTracker()

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = TrackingMiddleware(app, tracker)
        await mw(_scope(client=None), AsyncMock(), AsyncMock())

        rec = tracker.snapshot().records[0]
        assert rec.client_host == ""
        assert rec.client_port == 0

    async def test_query_string_captured(self) -> None:
        tracker = RequestTracker()

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = TrackingMiddleware(app, tracker)
        await mw(
            _scope(query_string=b"format=png&quality=90"),
            AsyncMock(),
            AsyncMock(),
        )

        assert tracker.snapshot().records[0].query_string == "format=png&quality=90"

    async def test_request_headers_captured(self) -> None:
        tracker = RequestTracker()

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = TrackingMiddleware(app, tracker)
        await mw(
            _scope(
                headers=[
                    (b"Host", b"example.com"),
                    (b"Authorization", b"Bearer tok123"),
                    (b"Accept-Encoding", b"gzip, deflate"),
                    (b"X-Forwarded-For", b"1.2.3.4"),
                ]
            ),
            AsyncMock(),
            AsyncMock(),
        )

        h = tracker.snapshot().records[0].request_headers
        assert h["host"] == "example.com"
        assert h["authorization"] == "Bearer tok123"
        assert h["accept-encoding"] == "gzip, deflate"
        assert h["x-forwarded-for"] == "1.2.3.4"

    async def test_response_headers_captured(self) -> None:
        tracker = RequestTracker()

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"audio/wav"),
                        (b"content-length", b"44100"),
                        (b"cache-control", b"public, max-age=3600"),
                        (b"x-served-by", b"content-server-01"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b"\x00" * 100})

        mw = TrackingMiddleware(app, tracker)
        await mw(_scope(), AsyncMock(), AsyncMock())

        h = tracker.snapshot().records[0].response_headers
        assert h["content-type"] == "audio/wav"
        assert h["content-length"] == "44100"
        assert h["cache-control"] == "public, max-age=3600"
        assert h["x-served-by"] == "content-server-01"
