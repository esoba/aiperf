# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for the content server."""

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


class ContentRequestRecord(AIPerfBaseModel):
    """Record of a single HTTP request served by the content server.

    All timing fields use nanoseconds. Wall-clock timestamps use time.time_ns()
    for correlation with other services. Latency intervals use time.perf_counter_ns()
    for monotonic precision.

    Timing diagram::

        arrival_ns                                                    completion_ns
        │                                                             │
        ├──── ttfb_ns ────┤                                           │
        │                 │                                           │
        │    routing +    http.response.start   first body   last body│
        │    handler      (status + headers)    chunk sent   chunk sent
        │                 │                     │            │        │
        │                 ├── ttfb_ns ──────────┤            │        │
        │                 │   (first body byte) │            │        │
        │                 │                     ├─ xfer_ns ──┤        │
        │                 │                     │            │        │
        ├─────────────────────── latency_ns ─────────────────────────►│
    """

    # ── Identity ──
    timestamp_ns: int = Field(
        description="Wall-clock arrival time in nanoseconds (time.time_ns)"
    )
    method: str = Field(description="HTTP method (GET, HEAD, etc.)")
    path: str = Field(description="URL path (e.g. /content/images/cat.png)")
    query_string: str = Field(default="", description="Raw query string (without ?)")
    http_version: str = Field(default="1.1", description="HTTP version (1.0, 1.1, 2)")

    # ── Client ──
    client_host: str = Field(default="", description="Client IP address")
    client_port: int = Field(default=0, description="Client ephemeral port")

    # ── Request headers ──
    request_headers: dict[str, str] = Field(
        default_factory=dict,
        description="All request headers as lowercase-key dict",
    )

    # ── Response ──
    status_code: int = Field(default=0, description="HTTP response status code")
    content_type: str = Field(
        default="application/octet-stream",
        description="Response Content-Type header value",
    )
    response_headers: dict[str, str] = Field(
        default_factory=dict,
        description="All response headers as lowercase-key dict",
    )

    # ── Transfer ──
    body_bytes: int = Field(
        default=0, description="Actual bytes sent in response body chunks"
    )
    body_chunk_count: int = Field(
        default=0, description="Number of HTTP response body chunks sent"
    )

    # ── Timing (nanoseconds, monotonic via perf_counter_ns) ──
    latency_ns: int = Field(
        default=0,
        description="Total latency from ASGI arrival to final byte sent (perf_counter_ns)",
    )
    time_to_first_byte_ns: int = Field(
        default=0,
        description="Time from arrival to http.response.start sent to client (perf_counter_ns)",
    )
    time_to_first_body_byte_ns: int = Field(
        default=0,
        description="Time from arrival to first http.response.body chunk sent (perf_counter_ns)",
    )
    transfer_duration_ns: int = Field(
        default=0,
        description="Time from first body chunk to last body chunk (perf_counter_ns)",
    )

    # ── Error ──
    error: str | None = Field(
        default=None,
        description="Exception message if the handler raised during this request",
    )


class ContentServerStatus(AIPerfBaseModel):
    """Status information for the content server."""

    enabled: bool = Field(description="Whether the content server is enabled")
    base_url: str = Field(
        default="",
        description="Base URL of the content server (e.g. http://host:port)",
    )
    content_dir: str = Field(default="", description="Directory being served")
    reason: str | None = Field(
        default=None,
        description="Reason why the server is disabled (if enabled=False)",
    )


class RequestTrackerSnapshot(AIPerfBaseModel):
    """Snapshot of the request tracker state for metrics export."""

    total_requests: int = Field(
        default=0, description="Total number of requests served"
    )
    total_bytes_served: int = Field(
        default=0, description="Total bytes served across all requests"
    )
    records: list[ContentRequestRecord] = Field(
        default_factory=list,
        description="Recent request records from the tracking buffer",
    )
