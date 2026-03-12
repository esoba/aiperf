# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube list command: list benchmark jobs and their status."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from aiperf.common.config.kube_config import KubeManageOptions

app = App(name="list")


@app.default
async def list_jobs(
    job_id: str | None = None,
    all_namespaces: Annotated[bool, Parameter(name=["-A", "--all-namespaces"])] = True,
    running: Annotated[bool, Parameter(name=["--running"])] = False,
    completed: Annotated[bool, Parameter(name=["--completed"])] = False,
    failed: Annotated[bool, Parameter(name=["--failed"])] = False,
    wide: Annotated[bool, Parameter(name=["-w", "--wide"])] = False,
    *,
    manage_options: KubeManageOptions | None = None,
) -> None:
    """List AIPerf benchmark jobs and their status.

    Lists running and completed AIPerf JobSets with their status.
    By default searches all namespaces. Use --namespace to limit scope.

    Examples:
        # List all AIPerf jobs (all namespaces)
        aiperf kube list

        # List only running jobs
        aiperf kube list --running

        # List jobs in a specific namespace
        aiperf kube list --namespace aiperf-bench

        # Get status of a specific job
        aiperf kube list abc123

    Args:
        job_id: Specific job ID to check (optional).
        all_namespaces: Search in all namespaces (default: True).
        running: Show only running jobs.
        completed: Show only completed jobs.
        failed: Show only failed jobs.
        wide: Show additional columns (job-id, custom-name, endpoint).
        manage_options: Kubernetes management options (kubeconfig, namespace).
    """
    from aiperf import cli_utils
    from aiperf.kubernetes import client
    from aiperf.kubernetes import console as kube_console

    manage_options = manage_options or KubeManageOptions()

    active_filters = [
        ("Running", running),
        ("Completed", completed),
        ("Failed", failed),
    ]
    selected = [name for name, active in active_filters if active]
    if len(selected) > 1:
        kube_console.print_error(
            f"Only one status filter can be used at a time, got: {', '.join(f'--{s.lower()}' for s in selected)}"
        )
        raise SystemExit(1)
    status_filter = selected[0] if selected else None

    search_all = all_namespaces and manage_options.namespace is None

    with cli_utils.exit_on_error(title="Error Listing Kubernetes Jobs"):
        import asyncio

        kube_client = await client.AIPerfKubeClient.create(
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
        )

        jobs = await kube_client.list_jobsets(
            manage_options.namespace,
            search_all,
            job_id,
            status_filter,
        )
        if not jobs:
            filter_msg = f" with status '{status_filter}'" if status_filter else ""
            kube_console.print_info(f"No AIPerf jobs found{filter_msg}")
            return

        pod_summary_tasks = {
            job.name: kube_client.get_pod_summary(job.name, job.namespace)
            for job in jobs
        }
        pod_summary_results = await asyncio.gather(*pod_summary_tasks.values())
        pod_summaries = dict(
            zip(pod_summary_tasks.keys(), pod_summary_results, strict=True)
        )

        kube_console.print_jobs_table(jobs, pod_summaries=pod_summaries, wide=wide)
