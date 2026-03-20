# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endpoint readiness checker.

Validates that the inference endpoint is ready by sending a real inference
request with a canned prompt. Unlike a /health check, this confirms the
model is loaded and can generate output.
"""

from __future__ import annotations

import asyncio
import logging
import time

import aiohttp

from aiperf.transports.aiohttp_client import create_tcp_connector
from aiperf.transports.http_defaults import AioHttpDefaults

logger = logging.getLogger(__name__)

# "Lo" — the first message ever sent over a network. On Oct 29, 1969,
# UCLA tried to transmit "login" over the ARPANET but the system crashed
# after two characters. History's first network message was just "Lo".
_READINESS_PROMPT = "Lo"

# Canned payloads keyed by endpoint type. Each sends a minimal real
# inference request so the response proves the model is loaded.
_CANNED_PAYLOADS: dict[str, dict] = {
    "chat": {
        "model": "default",
        "messages": [{"role": "user", "content": _READINESS_PROMPT}],
        "max_tokens": 1,
    },
    "completions": {
        "model": "default",
        "prompt": _READINESS_PROMPT,
        "max_tokens": 1,
    },
    "embeddings": {
        "model": "default",
        "input": _READINESS_PROMPT,
    },
}

# Default paths per endpoint type (OpenAI-compatible).
_DEFAULT_PATHS: dict[str, str] = {
    "chat": "/v1/chat/completions",
    "completions": "/v1/completions",
    "embeddings": "/v1/embeddings",
}

_RETRY_INTERVAL = 5.0


async def wait_for_endpoint(
    url: str,
    *,
    endpoint_type: str = "chat",
    path: str | None = None,
    timeout: float = 600.0,
    api_key: str | None = None,
    model: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> None:
    """Block until the endpoint responds to a real inference request.

    Sends a canned prompt matching the endpoint type and retries until
    a successful response (status < 500) is received or the timeout
    expires.

    Args:
        url: Base server URL (e.g. ``http://localhost:8000``).
        endpoint_type: One of chat, completions, embeddings.
        path: Override the default API path for the endpoint type.
        timeout: Maximum seconds to wait (0 = skip).
        api_key: Bearer token for Authorization header.
        model: Model name to include in the payload.
        extra_headers: Additional headers to include in requests.

    Raises:
        TimeoutError: If the endpoint does not respond within *timeout*.
    """
    if timeout <= 0:
        return

    from urllib.parse import urlparse

    parsed = urlparse(url)
    has_path = parsed.path not in ("", "/")
    if has_path:
        request_url = url.rstrip("/")
    else:
        endpoint_path = path or _DEFAULT_PATHS.get(
            endpoint_type, "/v1/chat/completions"
        )
        request_url = url.rstrip("/") + endpoint_path

    payload = dict(_CANNED_PAYLOADS.get(endpoint_type, _CANNED_PAYLOADS["chat"]))
    if model:
        payload["model"] = model

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    deadline = time.perf_counter() + timeout
    last_error: str = ""

    logger.info(
        "Waiting for endpoint readiness: %s (timeout=%ss)", request_url, timeout
    )

    connector = create_tcp_connector()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        connector=connector,
        trust_env=AioHttpDefaults.TRUST_ENV,
    ) as session:
        while time.perf_counter() < deadline:
            try:
                async with session.post(
                    request_url, json=payload, headers=headers
                ) as resp:
                    if resp.status < 500:
                        elapsed = timeout - (deadline - time.perf_counter())
                        logger.info(
                            "Endpoint ready after %.1fs (status=%d)",
                            elapsed,
                            resp.status,
                        )
                        return
                    last_error = f"HTTP {resp.status}"
                    logger.warning("Endpoint not ready: %s %s", request_url, last_error)
            except aiohttp.ClientConnectorError:
                # Server not yet reachable — silent retry.
                pass
            except aiohttp.ClientError as exc:
                last_error = str(exc)
                logger.warning("Endpoint not ready: %s %s", request_url, last_error)

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            await asyncio.sleep(min(_RETRY_INTERVAL, remaining))

    raise TimeoutError(
        f"Endpoint {request_url} not ready after {timeout:.0f}s: {last_error}"
    )
