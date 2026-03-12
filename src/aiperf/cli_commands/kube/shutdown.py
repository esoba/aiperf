# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube shutdown command: signal API service shutdown."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from aiperf.common.config.kube_config import KubeManageOptions

app = App(name="shutdown")


@app.default
async def shutdown_benchmark(
    job_id: str | None = None,
    *,
    manage_options: KubeManageOptions | None = None,
    port: Annotated[int, Parameter(name="--port")] = 0,
) -> None:
    """Shut down the controller pod's API service after a benchmark finishes.

    After a benchmark completes, the controller pod keeps its API service
    running so you can retrieve results. This command signals the API to
    shut down gracefully, allowing the controller pod to exit and release
    resources. The request is rejected if the benchmark is still running.

    Equivalent to 'aiperf kube results --shutdown' without downloading results.

    If no job_id is specified, uses the last deployed benchmark.

    Examples:
        # Shut down the API for the last deployed benchmark
        aiperf kube shutdown

        # Shut down the API for a specific benchmark
        aiperf kube shutdown abc123

    Args:
        job_id: The AIPerf job ID (default: last deployed job).
        manage_options: Kubernetes management options (kubeconfig, namespace).
        port: Local port for API port-forward (default: 0 = ephemeral).
    """
    from aiperf import cli_utils
    from aiperf.kubernetes import cli_helpers
    from aiperf.kubernetes import results as kube_results

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Shutting Down API Service"):
        result = await cli_helpers.resolve_jobset(
            job_id,
            manage_options.namespace,
            manage_options.kubeconfig,
            manage_options.kube_context,
        )
        if not result:
            return
        job_id, jobset_info, client = result

        await kube_results.shutdown_api_service(
            job_id,
            jobset_info.namespace,
            client,
            port,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
        )
