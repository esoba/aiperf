# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base HTTP transport providing shared URL construction and header logic."""

from __future__ import annotations

from aiperf.common.models import RequestInfo
from aiperf.plugin import plugins
from aiperf.transports.base_transports import BaseTransport


def _append_path_deduped(base_url: str, path: str) -> str:
    """Append *path* to *base_url*, deduplicating when the URL already contains the path.

    Handles common user mistakes like providing ``http://server:8000/v1/chat/completions``
    as the base URL when the endpoint path is ``v1/chat/completions``.
    """
    if base_url.endswith(f"/{path}"):
        return base_url
    # Handle /v1 base URL with v1/ path prefix to avoid /v1/v1/...
    if base_url.endswith("/v1") and path.startswith("v1/"):
        path = path.removeprefix("v1/")
    return f"{base_url}/{path}"


class BaseHTTPTransport(BaseTransport):
    """Base class for HTTP-based transports (HTTP/1.1 and HTTP/2).

    Provides shared URL construction, header building, and streaming path resolution.
    Subclasses implement client lifecycle and send_request().
    """

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Build HTTP-specific headers based on streaming mode.

        Args:
            request_info: Request context with endpoint configuration

        Returns:
            HTTP headers (Content-Type and Accept)
        """
        accept = (
            "text/event-stream"
            if request_info.config.endpoint.streaming
            else "application/json"
        )
        return {"Content-Type": "application/json", "Accept": accept}

    def get_url(self, request_info: RequestInfo) -> str:
        """Build HTTP URL from base_url and endpoint path.

        Constructs the full URL by combining the base URL with the endpoint path
        from metadata or custom endpoint. Adds http:// scheme if missing.

        When multiple URLs are configured, uses request_info.url_index to select
        the appropriate URL for load balancing.

        Args:
            request_info: Request context with model endpoint info

        Returns:
            Complete HTTP URL with scheme and endpoint path
        """
        endpoint_info = request_info.config.endpoint

        # Start with base URL - use url_index for multi-URL load balancing
        url_index = request_info.url_index if request_info.url_index is not None else 0
        base_url = endpoint_info.urls[url_index % len(endpoint_info.urls)].rstrip("/")

        # Determine the endpoint path
        if endpoint_info.path:
            # Use custom endpoint path if provided
            path = endpoint_info.path.lstrip("/")
        else:
            # Get endpoint path from endpoint metadata
            endpoint_metadata = plugins.get_endpoint_metadata(endpoint_info.type)
            path = endpoint_metadata.endpoint_path
            if (
                self.run.cfg.endpoint.streaming
                and endpoint_metadata.streaming_path is not None
            ):
                path = endpoint_metadata.streaming_path

        if not path:
            url = base_url
        else:
            path = path.lstrip("/")
            url = _append_path_deduped(base_url, path)
        # Add scheme if missing for proper parsing
        return url if url.startswith("http") else f"http://{url}"
