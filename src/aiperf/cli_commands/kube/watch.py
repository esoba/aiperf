# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live monitoring of Kubernetes benchmark deployments."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

app = App(name="watch")


@app.default
async def watch(
    *,
    namespace: Annotated[str | None, Parameter(name=["-n", "--namespace"])] = None,
    job_id: Annotated[str | None, Parameter(name=["-j", "--job-id"])] = None,
    interval: Annotated[int, Parameter(name=["-i", "--interval"])] = 10,
    timeout: Annotated[int, Parameter(name=["-t", "--timeout"])] = 0,
    kubeconfig: Annotated[str | None, Parameter(name="--kubeconfig")] = None,
    context: Annotated[str | None, Parameter(name="--context")] = None,
    all_namespaces: Annotated[bool, Parameter(name=["-A", "--all-namespaces"])] = False,
) -> None:
    """Watch live status of benchmark pods, events, and resource usage.

    Monitors a benchmark deployment in real-time, showing pod status transitions,
    Kubernetes events, and detected problems. Press Ctrl+C to stop.

    Examples:
        aiperf kube watch -n my-benchmark
        aiperf kube watch --job-id abc123
        aiperf kube watch -A --interval 5

    Args:
        namespace: Kubernetes namespace to watch.
        job_id: AIPerf job ID to watch.
        interval: Polling interval in seconds.
        timeout: Maximum watch duration in seconds (0 for unlimited).
        kubeconfig: Path to kubeconfig file.
        context: Kubernetes context name.
        all_namespaces: Watch all AIPerf namespaces.
    """
    from aiperf import cli_utils

    with cli_utils.exit_on_error(title="Error Watching Benchmark"):
        import asyncio
        import time

        from aiperf.kubernetes import console as kube_console
        from aiperf.kubernetes.client import AIPerfKubeClient
        from aiperf.kubernetes.enums import JobSetStatus

        kube_client = await AIPerfKubeClient.create(
            kubeconfig=kubeconfig,
            kube_context=context,
        )

        resolved_ns = namespace
        resolved_job_id = job_id

        if not namespace and not job_id and not all_namespaces:
            from aiperf.kubernetes.console import get_last_benchmark

            last = get_last_benchmark()
            if last:
                resolved_job_id = last.job_id
                resolved_ns = last.namespace
                kube_console.print_info(
                    f"Watching last benchmark: {resolved_job_id} "
                    f"in namespace {resolved_ns}"
                )
            else:
                kube_console.print_warning(
                    "No matching namespaces found. "
                    "Use -n, -j, or -A to specify targets."
                )
                return

        jobs = await kube_client.list_jobsets(
            namespace=resolved_ns,
            all_namespaces=all_namespaces,
            job_id=resolved_job_id,
        )

        if not jobs:
            kube_console.print_warning("No AIPerf benchmarks found.")
            return

        kube_console.print_header("Watching AIPerf Benchmarks")
        kube_console.print_info(f"Polling every {interval}s. Press Ctrl+C to stop.")

        start_time = time.monotonic()

        try:
            while True:
                jobs = await kube_client.list_jobsets(
                    namespace=resolved_ns,
                    all_namespaces=all_namespaces,
                    job_id=resolved_job_id,
                )

                pod_summaries = {}
                for job in jobs:
                    summary = await kube_client.get_pod_summary(job.name, job.namespace)
                    pod_summaries[job.name] = summary

                kube_console.print_jobs_table(jobs, pod_summaries=pod_summaries)

                terminal_statuses = {
                    JobSetStatus.COMPLETED,
                    JobSetStatus.FAILED,
                }
                all_terminal = jobs and all(
                    JobSetStatus.from_str(j.status) in terminal_statuses for j in jobs
                )
                if all_terminal:
                    kube_console.print_info("All benchmarks have finished.")
                    break

                if timeout > 0:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        kube_console.print_warning(
                            f"Watch timeout reached ({timeout}s)."
                        )
                        break

                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            kube_console.print_info("Watch stopped.")
