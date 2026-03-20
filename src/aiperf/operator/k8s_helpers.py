# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable helpers for Kubernetes resource creation and metadata."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import kr8s

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry_with_backoff(
    coro_factory: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 30.0,
    backoff_multiplier: float = 2.0,
    description: str = "operation",
) -> T:
    """Retry an async operation with exponential backoff and jitter.

    Args:
        coro_factory: Zero-arg callable returning an awaitable (called each attempt).
        max_retries: Maximum number of retry attempts after the first failure.
        initial_delay: Seconds to wait before the first retry.
        max_delay: Maximum backoff cap in seconds.
        backoff_multiplier: Multiplier applied to the delay after each retry.
        description: Human-readable label for log messages.

    Returns:
        The result of the first successful call.

    Raises:
        The exception from the final attempt if all retries are exhausted.
    """
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception:
            if attempt >= max_retries:
                raise
            jittered_delay = delay * random.uniform(0.8, 1.2)
            logger.debug(
                "%s attempt %d/%d failed, retrying in %.1fs",
                description,
                attempt + 1,
                max_retries + 1,
                jittered_delay,
            )
            await asyncio.sleep(jittered_delay)
            delay = min(delay * backoff_multiplier, max_delay)

    # Unreachable, but satisfies the type checker
    raise RuntimeError(
        f"{description} failed after {max_retries + 1} attempts"
    )  # pragma: no cover


async def create_idempotent(
    resource_class: type, manifest: dict[str, Any], api: kr8s.Api
) -> None:
    """Create a K8s resource, ignoring AlreadyExists (409)."""
    try:
        await resource_class(manifest, api=api).create()
    except kr8s.ServerError as e:
        if not (e.response and e.response.status_code == 409):
            raise
