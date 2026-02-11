# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request tracking for the content server.

Tracks HTTP requests via ASGI middleware so that timestamps and latencies
cover the full request-response lifecycle, including the actual file
transfer (sendfile), not just route-handler execution.

Uses two clocks:
- time.time_ns()          wall-clock for timestamps (correlates with other services)
- time.perf_counter_ns()  monotonic for interval measurements (no NTP drift)
"""

import time
from collections import deque

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from aiperf.content_server.models import (
    ContentRequestRecord,
    RequestTrackerSnapshot,
)


class RequestTracker:
    """Tracks HTTP request records in a bounded deque for metrics integration.

    Running counters (total_requests, total_bytes_served) survive eviction
    from the bounded deque, providing accurate lifetime totals.
    """

    def __init__(self, max_records: int = 10000) -> None:
        self._records: deque[ContentRequestRecord] = deque(maxlen=max_records)
        self._total_requests: int = 0
        self._total_bytes_served: int = 0

    def record(self, entry: ContentRequestRecord) -> None:
        """Append a completed request record."""
        self._total_requests += 1
        self._total_bytes_served += entry.body_bytes
        self._records.append(entry)

    def snapshot(self) -> RequestTrackerSnapshot:
        """Return a snapshot of the current tracker state."""
        return RequestTrackerSnapshot(
            total_requests=self._total_requests,
            total_bytes_served=self._total_bytes_served,
            records=list(self._records),
        )

    @property
    def total_requests(self) -> int:
        """Total number of requests served (lifetime)."""
        return self._total_requests

    @property
    def total_bytes_served(self) -> int:
        """Total bytes served (lifetime)."""
        return self._total_bytes_served


def _decode(value: bytes | str) -> str:
    """Decode bytes to str, passthrough str."""
    return value.decode("latin-1") if isinstance(value, bytes) else value


def _extract_headers(raw_headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    """Convert ASGI header list to a lowercase-key dict.

    For duplicate headers the values are joined with ', ' per HTTP spec.
    """
    result: dict[str, str] = {}
    for name_raw, value_raw in raw_headers:
        key = _decode(name_raw).lower()
        val = _decode(value_raw)
        if key in result:
            result[key] = result[key] + ", " + val
        else:
            result[key] = val
    return result


class TrackingMiddleware:
    """ASGI middleware that instruments every HTTP request with detailed timing.

    For each request it captures:
    - All request headers and client info from the ASGI scope
    - All response headers from http.response.start
    - Per-chunk body byte accumulation from http.response.body
    - Four monotonic timing intervals (TTFB, TTFBB, transfer, total latency)
    - Any exception that propagated from the handler

    Non-HTTP scopes (websocket, lifespan) pass through untracked.
    """

    def __init__(self, app: ASGIApp, tracker: RequestTracker) -> None:
        self._app = app
        self._tracker = tracker

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        # ── Wall-clock + monotonic at arrival ──
        arrival_wall_ns = time.time_ns()
        arrival_mono_ns = time.perf_counter_ns()

        # ── Request info from ASGI scope ──
        method: str = scope.get("method", "")
        path: str = scope.get("path", "")
        query_string_raw: bytes = scope.get("query_string", b"")
        query_string = query_string_raw.decode("latin-1") if query_string_raw else ""
        http_version: str = scope.get("http_version", "1.1")
        client: tuple[str, int] | None = scope.get("client")
        client_host = client[0] if client else ""
        client_port = client[1] if client else 0
        request_headers = _extract_headers(scope.get("headers", []))

        # ── Mutable state captured during send ──
        status_code = 0
        response_headers: dict[str, str] = {}
        content_type = "application/octet-stream"
        body_bytes = 0
        body_chunk_count = 0
        first_byte_mono_ns = 0
        first_body_mono_ns = 0
        last_body_mono_ns = 0
        error_msg: str | None = None

        async def tracking_send(message: Message) -> None:
            nonlocal status_code, response_headers, content_type
            nonlocal body_bytes, body_chunk_count
            nonlocal first_byte_mono_ns, first_body_mono_ns, last_body_mono_ns

            now_ns = time.perf_counter_ns()

            if message["type"] == "http.response.start":
                first_byte_mono_ns = now_ns
                status_code = message.get("status", 0)
                raw = message.get("headers", [])
                response_headers = _extract_headers(raw)
                content_type = response_headers.get(
                    "content-type", "application/octet-stream"
                )

            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                chunk_len = len(chunk)
                if chunk_len > 0:
                    body_chunk_count += 1
                    body_bytes += chunk_len
                    if first_body_mono_ns == 0:
                        first_body_mono_ns = now_ns
                    last_body_mono_ns = now_ns

            await send(message)

        try:
            await self._app(scope, receive, tracking_send)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            end_mono_ns = time.perf_counter_ns()

            ttfb = (first_byte_mono_ns - arrival_mono_ns) if first_byte_mono_ns else 0
            ttfbb = (first_body_mono_ns - arrival_mono_ns) if first_body_mono_ns else 0
            transfer = (
                (last_body_mono_ns - first_body_mono_ns)
                if first_body_mono_ns and last_body_mono_ns
                else 0
            )

            self._tracker.record(
                ContentRequestRecord(
                    # Identity
                    timestamp_ns=arrival_wall_ns,
                    method=method,
                    path=path,
                    query_string=query_string,
                    http_version=http_version,
                    # Client
                    client_host=client_host,
                    client_port=client_port,
                    # Request
                    request_headers=request_headers,
                    # Response
                    status_code=status_code,
                    content_type=content_type,
                    response_headers=response_headers,
                    # Transfer
                    body_bytes=body_bytes,
                    body_chunk_count=body_chunk_count,
                    # Timing
                    latency_ns=end_mono_ns - arrival_mono_ns,
                    time_to_first_byte_ns=ttfb,
                    time_to_first_body_byte_ns=ttfbb,
                    transfer_duration_ns=transfer,
                    # Error
                    error=error_msg,
                )
            )
