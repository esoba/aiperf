# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from typing import Any

import orjson

from aiperf.common.enums import ConnectionReuseStrategy
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.models import (
    ErrorDetails,
    RequestInfo,
    RequestRecord,
)
from aiperf.common.redact import redact_headers
from aiperf.plugin.enums import TransportType
from aiperf.transports.base_http_transport import BaseHTTPTransport
from aiperf.transports.base_transports import (
    FirstTokenCallback,
    TransportMetadata,
)
from aiperf.transports.httpcore_client import HttpCoreClient


class HttpCoreTransport(BaseHTTPTransport):
    """HTTP/2 transport implementation using httpcore.

    Provides high-performance HTTP transport with HTTP/2 multiplexing support,
    offering significantly higher concurrency than HTTP/1.1-only implementations.

    Key Features:
        - HTTP/2 multiplexing: ~100 streams per connection
        - Connection pooling: 25 connections (configurable)
        - Total capacity: 2,500+ concurrent requests
        - Automatic HTTP/1.1 fallback for compatibility
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize HTTP/2 transport.

        Args:
            **kwargs: Additional arguments passed to parent BaseTransport
        """
        super().__init__(**kwargs)
        self.httpcore_client: HttpCoreClient | None = None

    @on_init
    async def _init_httpcore_client(self) -> None:
        """Initialize the HttpCoreClient with HTTP/2 support."""
        self.httpcore_client = HttpCoreClient(
            timeout=self.run.cfg.endpoint.timeout,
        )

    @on_stop
    async def _close_httpcore_client(self) -> None:
        """Cleanup hook to close httpcore connection pool on stop."""
        if self.httpcore_client:
            client = self.httpcore_client
            self.httpcore_client = None
            await client.close()

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return HTTP/2 transport metadata."""
        return TransportMetadata(
            transport_type=TransportType.HTTP2,
            url_schemes=["http", "https"],
        )

    async def send_request(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any],
        *,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send HTTP POST request with JSON payload via httpcore HTTP/2 client.

        Connection behavior depends on the configured connection_reuse strategy:
        - POOLED: Uses the shared connection pool (default)
        - NEVER: Creates a single-connection ephemeral client, closed after use
        - STICKY_USER_SESSIONS: HTTP/2 multiplexing makes per-session connections
          redundant; logs a warning and falls back to POOLED

        Args:
            request_info: Request context and metadata
            payload: JSON-serializable request payload
            first_token_callback: Optional callback fired on first SSE message with ttft_ns

        Returns:
            Request record with responses, timing, and any errors
        """
        if self.httpcore_client is None:
            raise NotInitializedError(
                "HttpCoreTransport not initialized. Call initialize() before send_request()."
            )

        start_perf_ns = time.perf_counter_ns()
        headers = None
        reuse_strategy = self.run.cfg.endpoint.connection_reuse

        try:
            url = self.build_url(request_info)
            headers = self.build_headers(request_info)
            json_bytes = orjson.dumps(payload)

            match reuse_strategy:
                case ConnectionReuseStrategy.NEVER:
                    client = HttpCoreClient.create_ephemeral(
                        timeout=self.run.cfg.endpoint.timeout
                    )
                    try:
                        record = await client.post_request(
                            url,
                            json_bytes,
                            headers,
                            cancel_after_ns=request_info.cancel_after_ns,
                            first_token_callback=first_token_callback,
                        )
                    finally:
                        await client.close()

                case ConnectionReuseStrategy.STICKY_USER_SESSIONS:
                    self.warning(
                        "STICKY_USER_SESSIONS is not applicable for HTTP/2 "
                        "(multiplexing inherently shares connections); falling back to POOLED"
                    )
                    record = await self.httpcore_client.post_request(
                        url,
                        json_bytes,
                        headers,
                        cancel_after_ns=request_info.cancel_after_ns,
                        first_token_callback=first_token_callback,
                    )

                case _:
                    record = await self.httpcore_client.post_request(
                        url,
                        json_bytes,
                        headers,
                        cancel_after_ns=request_info.cancel_after_ns,
                        first_token_callback=first_token_callback,
                    )

            record.request_headers = redact_headers(headers)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            record = RequestRecord(
                request_headers=redact_headers(
                    headers or request_info.endpoint_headers
                ),
                start_perf_ns=start_perf_ns,
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails.from_exception(e),
            )
            self.exception(f"HTTP/2 request failed: {e!r}")

        return record
