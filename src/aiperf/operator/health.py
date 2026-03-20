# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endpoint health checking for the operator."""

from __future__ import annotations

import aiohttp

from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.models import HealthCheckResult


async def check_endpoint_health(
    url: str, timeout: float = OperatorEnvironment.ENDPOINT_CHECK_TIMEOUT
) -> HealthCheckResult:
    """Check if LLM endpoint is reachable.

    Tries a single canonical health path first, falling back to alternatives
    only if the first fails.

    Args:
        url: Endpoint URL to check.
        timeout: Per-request timeout in seconds.

    Returns:
        HealthCheckResult with reachability status and error message.
    """
    from aiperf.transports.aiohttp_client import create_tcp_connector

    health_paths = ["/health", "/v1/health", "/v1/models", "/"]

    connector = create_tcp_connector()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout), connector=connector
    ) as session:
        for path in health_paths:
            try:
                check_url = url.rstrip("/") + path
                async with session.get(check_url) as response:
                    if response.status < 500:
                        return HealthCheckResult(reachable=True, error="")
            except aiohttp.ClientError:
                continue
            except Exception as e:
                return HealthCheckResult(
                    reachable=False, error=f"Unexpected error: {e}"
                )

    return HealthCheckResult(reachable=False, error="All health endpoints unreachable")
