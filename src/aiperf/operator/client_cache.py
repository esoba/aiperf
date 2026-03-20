# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-job ProgressClient cache with LRU eviction.

Serializes concurrent access with an asyncio.Lock to prevent
interleaving between the None-check and dict assignment (which
contains an ``await``).
"""

from __future__ import annotations

import asyncio

from aiperf.operator.progress_client import ProgressClient

_MAX_CACHE_SIZE = 200

# Per-job ProgressClient cache keyed by namespace/job_id.
# Avoids creating a new aiohttp session every monitor tick.
_progress_clients: dict[str, ProgressClient] = {}
_client_cache_lock = asyncio.Lock()

# Tracks (pod_name, restart_count) pairs already warned about per job.
# Prevents emitting the same pod restart event every monitor tick.
_warned_pod_restarts: dict[str, set[tuple[str, int]]] = {}

# Tracks jobs where shutdown has already been sent to avoid duplicate signals.
_shutdown_sent: set[str] = set()


def job_key(namespace: str, job_id: str) -> str:
    """Create a unique cache key scoped to namespace.

    CRs in different namespaces can share the same name, so cache keys
    and results directories must be namespace-scoped.
    """
    return f"{namespace}/{job_id}"


async def get_or_create_progress_client(key: str) -> ProgressClient:
    """Get a cached ProgressClient for a job, creating one if needed.

    Serialized by _client_cache_lock to prevent concurrent interleaving
    between the None check and dict assignment (which includes an await).
    """
    async with _client_cache_lock:
        client = _progress_clients.get(key)
        if client is None:
            while len(_progress_clients) >= _MAX_CACHE_SIZE:
                oldest_key = next(iter(_progress_clients))
                await _close_unlocked(oldest_key)
            client = ProgressClient()
            await client.__aenter__()
            _progress_clients[key] = client
        return client


async def close_progress_client(key: str) -> None:
    """Close and remove a cached ProgressClient and dedup state for a job."""
    async with _client_cache_lock:
        await _close_unlocked(key)


async def _close_unlocked(key: str) -> None:
    """Close a cached ProgressClient without acquiring the lock (caller holds it)."""
    client = _progress_clients.pop(key, None)
    if client is not None:
        await client.__aexit__(None, None, None)
    _warned_pod_restarts.pop(key, None)
    _shutdown_sent.discard(key)


def _reset_for_testing() -> None:
    """Clear all cached state. For use in tests only."""
    _progress_clients.clear()
    _warned_pod_restarts.clear()
    _shutdown_sent.clear()
