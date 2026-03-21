# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ssl
import time
from typing import Any

import httpcore

from aiperf.common.environment import Environment
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    ErrorDetails,
    RequestRecord,
    TextResponse,
)
from aiperf.transports.http_defaults import HttpCoreDefaults, SocketDefaults
from aiperf.transports.sse_utils import AsyncSSEStreamReader


class HttpCoreClient(AIPerfLoggerMixin):
    """High-performance HTTP client using httpcore with HTTP/2 multiplexing support.

    Supports multiple concurrent streams over a single TCP connection, with automatic
    protocol negotiation (HTTP/2 or fallback to HTTP/1.1).
    """

    def __init__(self, timeout: float | None = None, **kwargs: Any) -> None:
        """Initialize the httpcore client with HTTP/2 support and connection pooling.

        Args:
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to parent AIPerfLoggerMixin
        """
        super().__init__(**kwargs)

        max_connections = HttpCoreDefaults.calculate_max_connections()

        self.debug(
            lambda: (
                f"Initializing httpcore client: {max_connections} connections, "
                f"~{max_connections * HttpCoreDefaults.STREAMS_PER_CONNECTION} stream capacity"
            )
        )

        ssl_context = ssl.create_default_context()
        if not Environment.HTTP.SSL_VERIFY:
            self.warning("TLS certificate verification is DISABLED - this is insecure!")
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        self.pool = httpcore.AsyncConnectionPool(
            http1=HttpCoreDefaults.HTTP1,
            http2=HttpCoreDefaults.HTTP2,
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
            keepalive_expiry=HttpCoreDefaults.KEEPALIVE_EXPIRY,
            retries=HttpCoreDefaults.RETRIES,
            socket_options=SocketDefaults.build_socket_options(),
            ssl_context=ssl_context,
        )

        self.timeout_seconds = timeout or 300.0
        self.debug(lambda: "httpcore client initialized successfully")

    async def close(self) -> None:
        """Close the connection pool and cleanup all resources."""
        if self.pool:
            self.debug(lambda: "Closing httpcore connection pool")
            await self.pool.aclose()
            self.pool = None
            self.debug(lambda: "httpcore connection pool closed")

    async def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: bytes | None = None,
        **kwargs: Any,
    ) -> RequestRecord:
        """Execute HTTP requests with nanosecond-precision timing and error handling.

        Automatically detects and handles SSE streams. All exceptions are caught and
        converted to ErrorDetails in the returned RequestRecord.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL with scheme
            headers: Request headers dict
            data: Optional request body bytes
            **kwargs: Additional arguments for future extension

        Returns:
            RequestRecord with status, timing data, responses, and optional error
        """
        self.debug(lambda: f"Sending {method} request to {url}")

        record = RequestRecord(start_perf_ns=time.perf_counter_ns())

        try:
            is_sse_request = headers.get("Accept", "").startswith("text/event-stream")

            httpcore_headers = [
                (name.encode("utf-8"), value.encode("utf-8"))
                for name, value in headers.items()
            ]

            extensions: dict[str, Any] = {
                "timeout": {
                    "connect": self.timeout_seconds,
                    "read": self.timeout_seconds,
                    "write": self.timeout_seconds,
                    "pool": 60.0,
                }
            }

            async with self.pool.stream(
                method=method.encode("utf-8"),
                url=url.encode("utf-8"),
                headers=httpcore_headers,
                content=data,
                extensions=extensions,
            ) as response:
                record.status = response.status
                record.recv_start_perf_ns = time.perf_counter_ns()

                self.debug(
                    lambda: (
                        f"Response status: {record.status}, "
                        f"HTTP version: HTTP/{response.extensions.get('http_version', b'').decode()}"
                    )
                )

                if record.status != 200:
                    error_body = bytearray()
                    async for chunk in response.aiter_stream():
                        error_body.extend(chunk)

                    error_text = error_body.decode("utf-8", errors="replace")
                    self.debug(
                        lambda: f"HTTP error {record.status}: {error_text[:100]}"
                    )
                    record.error = ErrorDetails(
                        code=record.status,
                        type=f"HTTP {record.status}",
                        message=error_text or f"HTTP {record.status} error",
                    )
                    record.end_perf_ns = time.perf_counter_ns()
                    return record

                response_headers = {
                    name.decode("utf-8").lower(): value.decode("utf-8")
                    for name, value in response.headers
                }
                content_type = response_headers.get("content-type", "")

                if is_sse_request and content_type.startswith("text/event-stream"):
                    self.debug("Processing SSE stream")
                    async for message in AsyncSSEStreamReader(response.aiter_stream()):
                        record.responses.append(message)
                    self.debug(lambda: f"Parsed {len(record.responses)} SSE messages")
                else:
                    self.debug("Processing regular response")
                    response_body = bytearray()
                    async for chunk in response.aiter_stream():
                        response_body.extend(chunk)

                    response_text = response_body.decode("utf-8", errors="replace")
                    record.end_perf_ns = time.perf_counter_ns()
                    record.responses.append(
                        TextResponse(
                            perf_ns=record.end_perf_ns,
                            content_type=content_type,
                            text=response_text,
                        )
                    )
                    self.debug(lambda: f"Response complete: {len(response_text)} bytes")

                if not record.end_perf_ns:
                    record.end_perf_ns = time.perf_counter_ns()

        except httpcore.ConnectTimeout as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Connection timeout: {e!r}")
            record.error = ErrorDetails(
                type="ConnectTimeout",
                message=f"Connection to {url} timed out after {self.timeout_seconds}s",
            )

        except httpcore.ReadTimeout as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Read timeout: {e!r}")
            record.error = ErrorDetails(
                type="ReadTimeout",
                message=f"Reading response from {url} timed out after {self.timeout_seconds}s",
            )

        except httpcore.WriteTimeout as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Write timeout: {e!r}")
            record.error = ErrorDetails(
                type="WriteTimeout",
                message=f"Sending request to {url} timed out after {self.timeout_seconds}s",
            )

        except httpcore.PoolTimeout as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Pool timeout: {e!r}")
            record.error = ErrorDetails(
                type="PoolTimeout",
                message=(
                    f"No available connection in pool after 60s. "
                    f"Consider increasing AIPERF_HTTP_CONNECTION_LIMIT (current: {Environment.HTTP.CONNECTION_LIMIT})"
                ),
            )

        except httpcore.TimeoutException as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Request timeout: {e!r}")
            record.error = ErrorDetails(
                type="TimeoutError",
                message=f"Request to {url} timed out: {e!r}",
            )

        except httpcore.ConnectError as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Connection error: {e!r}")
            record.error = ErrorDetails(
                type="ConnectError",
                message=f"Failed to connect to {url}: {e!r}",
            )

        except httpcore.RemoteProtocolError as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Remote protocol error: {e!r}")
            record.error = ErrorDetails(
                type="RemoteProtocolError",
                message=f"Server sent invalid HTTP/2 frames: {e!r}",
            )

        except httpcore.LocalProtocolError as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Local protocol error: {e!r}")
            record.error = ErrorDetails(
                type="LocalProtocolError",
                message=f"Client attempted invalid HTTP/2 operation: {e!r}",
            )

        except httpcore.ProtocolError as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Protocol error: {e!r}")
            record.error = ErrorDetails(
                type="ProtocolError",
                message=f"HTTP/2 protocol error: {e!r}",
            )

        except Exception as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Unexpected error in HTTP request: {e!r}")
            record.error = ErrorDetails.from_exception(e)

        return record

    async def post_request(
        self,
        url: str,
        payload: bytes,
        headers: dict[str, str],
        **kwargs: Any,
    ) -> RequestRecord:
        """Send an HTTP POST request with optional SSE streaming support.

        Args:
            url: Target URL with scheme
            payload: Request body bytes
            headers: HTTP headers dict
            **kwargs: Additional arguments passed to _request()

        Returns:
            RequestRecord with status, timing data, responses, and optional error
        """
        return await self._request("POST", url, headers, data=payload, **kwargs)

    async def get_request(
        self, url: str, headers: dict[str, str], **kwargs: Any
    ) -> RequestRecord:
        """Send an HTTP GET request.

        Args:
            url: Target URL with scheme
            headers: HTTP headers dict
            **kwargs: Additional arguments passed to _request()

        Returns:
            RequestRecord with status, timing data, responses, and optional error
        """
        return await self._request("GET", url, headers, **kwargs)
