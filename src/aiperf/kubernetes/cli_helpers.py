# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI helper functions for Kubernetes operations.

Job resolution, user confirmation, and formatting utilities shared
across multiple CLI commands.

kr8s-backed operations live in AIPerfKubeClient (aiperf.kubernetes.client).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from aiperf.kubernetes.console import (
    clear_last_benchmark,
    get_last_benchmark,
    print_action,
    print_error,
    print_info,
)

if TYPE_CHECKING:
    from aiperf.kubernetes.client import AIPerfKubeClient
    from aiperf.kubernetes.models import JobSetInfo


class ResolvedJobSet:
    """Result of resolving a job ID to a JobSet with an AIPerfKubeClient."""

    __slots__ = ("job_id", "jobset_info", "client")

    def __init__(
        self, job_id: str, jobset_info: JobSetInfo, client: AIPerfKubeClient
    ) -> None:
        self.job_id = job_id
        self.jobset_info = jobset_info
        self.client = client

    def __iter__(self):
        return iter((self.job_id, self.jobset_info, self.client))


def format_age(created: str) -> str:
    """Format a Kubernetes timestamp as a human-readable age string.

    Args:
        created: ISO timestamp from Kubernetes (e.g., "2024-01-15T10:30:00Z").

    Returns:
        Age string like "5s", "10m", or "2h".
    """
    if not created:
        return "Unknown"
    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
    age_seconds = max((datetime.now(timezone.utc) - created_dt).total_seconds(), 0)
    if age_seconds < 60:
        return f"{int(age_seconds)}s"
    if age_seconds < 3600:
        return f"{int(age_seconds / 60)}m"
    if age_seconds < 86400:
        return f"{int(age_seconds / 3600)}h"
    return f"{int(age_seconds / 86400)}d"


def resolve_job_id_and_namespace(
    job_id: str | None, namespace: str | None
) -> tuple[str, str | None] | None:
    """Resolve job_id and namespace, using last benchmark if not specified.

    Returns (job_id, namespace) tuple if resolved, None if not found.
    """
    if job_id is not None:
        return (job_id, namespace)

    last = get_last_benchmark()
    if last is None:
        print_error("No job_id specified and no previous benchmark found")
        print_action("Run 'aiperf kube profile' first or specify a job_id")
        return None

    print_info(f"Using last benchmark: {last.job_id} in {last.namespace}")
    return (last.job_id, namespace or last.namespace)


async def resolve_jobset(
    job_id: str | None,
    namespace: str | None = None,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> ResolvedJobSet | None:
    """Resolve job_id to a JobSetInfo with an AIPerfKubeClient.

    Combines job ID resolution, client init, and JobSet lookup into
    a single call. Most kube commands need all three.
    """
    from aiperf.kubernetes.client import AIPerfKubeClient

    resolved = resolve_job_id_and_namespace(job_id, namespace)
    if not resolved:
        return None
    job_id, namespace = resolved

    client = await AIPerfKubeClient.create(
        kubeconfig=kubeconfig, kube_context=kube_context
    )

    jobset_info = await client.find_jobset(job_id, namespace)
    if not jobset_info:
        print_error(f"No AIPerf job found with ID: {job_id}")
        if namespace:
            print_info(f"Searched namespace: {namespace}")
        else:
            print_info("Searched all namespaces")
        print_action("Run 'aiperf kube status' to see available jobs")
        return None

    return ResolvedJobSet(job_id, jobset_info, client)


async def confirm_action(msg: str) -> bool:
    """Prompt user for confirmation. Returns True if confirmed, False if aborted."""
    response = await asyncio.to_thread(input, f"{msg} [y/N] ")
    if response.lower() != "y":
        print_info("Aborted")
        return False
    return True


def clear_last_benchmark_if_matches(job_id: str) -> None:
    """Clear stored last benchmark info if it matches the given job_id."""
    last = get_last_benchmark()
    if last and last.job_id == job_id:
        clear_last_benchmark()
