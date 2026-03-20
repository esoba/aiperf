# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube logs command: retrieve logs from benchmark pods."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

app = App(name="logs")


@app.default
async def logs(
    job_id: Annotated[
        str | None,
        Parameter(
            help="The AIPerf job ID to get logs from (default: last deployed job)."
        ),
    ] = None,
    *,
    manage_options: KubeManageOptions | None = None,
    container: Annotated[
        str | None, Parameter(help="Specific container name to get logs from.")
    ] = None,
    follow: Annotated[
        bool, Parameter(name=["-f", "--follow"], help="Follow log output in real-time.")
    ] = False,
    tail: Annotated[
        int | None, Parameter(help="Number of lines from the end to show.")
    ] = None,
    output: Annotated[
        Path | None,
        Parameter(
            name=["-o", "--output"],
            help="Directory to save log files (one per pod). Prints to stdout if not set.",
        ),
    ] = None,
) -> None:
    """Get logs from AIPerf benchmark pods.

    Shows logs from all pods and containers associated with the job.
    If no job_id is specified, uses the last deployed benchmark.

    Use --output to save logs to a directory instead of printing to stdout.
    Each pod's logs are saved as {pod-name}.log.

    Examples:
        # Get logs from last deployed job
        aiperf kube logs

        # Get logs from a specific job
        aiperf kube logs abc123

        # Get logs from a specific container
        aiperf kube logs --container control-plane

        # Follow logs in real-time
        aiperf kube logs -f

        # Get last 100 lines
        aiperf kube logs --tail 100

        # Save logs to a directory
        aiperf kube logs --output ./my-logs
    """
    from aiperf import cli_utils

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Getting Logs"):
        import kr8s

        from aiperf.kubernetes import (
            cli_helpers,
            client,
        )
        from aiperf.kubernetes import (
            console as kube_console,
        )
        from aiperf.kubernetes import (
            logs as kube_logs,
        )

        resolved = cli_helpers.resolve_job_id_and_namespace(
            job_id, manage_options.namespace
        )
        if not resolved:
            return
        job_id, namespace = resolved

        kube_client = await client.AIPerfKubeClient.create(
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
        )

        if output:
            output.mkdir(parents=True, exist_ok=True)
            await kube_logs.save_pod_logs(
                job_id,
                namespace,
                output,
                kube_client,
                kubeconfig=manage_options.kubeconfig,
                kube_context=manage_options.kube_context,
            )
            kube_console.print_success(f"Logs saved to {output}/logs/")
            return

        pods = await kube_client.get_pods(namespace, kube_client.job_selector(job_id))

        if not pods:
            kube_console.print_warning(f"No pods found for job ID: {job_id}")
            return

        # Build list of (pod, container_name) targets
        targets: list[tuple[Any, str]] = []
        for pod in pods:
            raw_containers = pod.raw.get("spec", {}).get("containers", [])
            container_names = [c["name"] for c in raw_containers]
            target_containers = [container] if container else container_names
            for cont in target_containers:
                if cont in container_names:
                    targets.append((pod, cont))

        if not targets:
            kube_console.print_warning("No matching containers found")
            return

        if follow and len(targets) > 1:
            kube_console.print_warning(
                f"Follow mode streams one container at a time. "
                f"Showing {targets[0][0].name}/{targets[0][1]} "
                f"({len(targets)} targets total). "
                f"Use --container to select a specific container."
            )

        for pod, cont in targets:
            kube_console.print_header(f"{pod.name}/{cont}")

            try:
                kwargs: dict[str, Any] = {}
                if tail is not None:
                    kwargs["tail_lines"] = tail

                if follow:
                    async for line in pod.logs(container=cont, follow=True, **kwargs):
                        print(line)
                    break  # Only follow one target
                else:
                    lines = [line async for line in pod.logs(container=cont, **kwargs)]
                    print("\n".join(lines))
            except kr8s.ServerError as e:
                kube_console.print_error(f"Error getting logs: {e}")
