# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube attach command: connect to a running benchmark."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from aiperf.common.config.kube_config import KubeManageOptions

app = App(name="attach")


@app.default
async def attach(
    job_id: str | None = None,
    *,
    manage_options: KubeManageOptions | None = None,
    port: Annotated[int, Parameter(name=["-p", "--port"])] = 0,
) -> None:
    """Attach to a running AIPerf benchmark and stream progress.

    Connects to a running benchmark via port-forward and streams real-time
    progress updates via WebSocket. Press Ctrl+C to disconnect.

    If no job_id is specified, uses the last deployed benchmark.

    Examples:
        # Attach to the last deployed benchmark
        aiperf kube attach

        # Attach to a specific job
        aiperf kube attach abc123

        # Attach to a job in a specific namespace
        aiperf kube attach abc123 --namespace aiperf-bench

        # Use a different local port
        aiperf kube attach abc123 --port 9091

    Args:
        job_id: The AIPerf job ID to attach to (default: last deployed job).
        manage_options: Kubernetes management options (kubeconfig, namespace).
        port: Local port for port-forward (default: 0 = ephemeral).
    """
    from aiperf import cli_utils
    from aiperf.kubernetes import attach as kube_attach
    from aiperf.kubernetes import cli_helpers

    manage_options = manage_options or KubeManageOptions()
    resolved = cli_helpers.resolve_job_id_and_namespace(
        job_id, manage_options.namespace
    )
    if not resolved:
        return
    job_id, namespace = resolved

    with cli_utils.exit_on_error(title="Error Attaching to Benchmark"):
        await kube_attach.attach_to_benchmark(
            job_id,
            namespace,
            port,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
        )
