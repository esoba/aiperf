# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube results command: retrieve benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from aiperf.common.config.kube_config import KubeManageOptions

app = App(name="results")


@app.default
async def results(
    job_id: str | None = None,
    *,
    manage_options: KubeManageOptions | None = None,
    output: Path | None = None,
    from_pod: Annotated[bool, Parameter(name="--from-pod")] = False,
    all_artifacts: Annotated[
        bool, Parameter(name=["--all", "-a"], negative="--summary-only")
    ] = True,
    shutdown: Annotated[bool, Parameter(name="--shutdown")] = False,
    port: Annotated[int, Parameter(name="--port")] = 0,
) -> None:
    """Retrieve results from an AIPerf benchmark.

    By default, downloads all artifacts (per-record metrics, detailed exports,
    and all files in the /results directory) via the controller pod's API
    service. Falls back to kubectl cp if the API is unavailable.

    Use --summary-only to download only the summary results instead of all
    artifacts. Use --from-pod to copy directly from the pod's /results
    directory using kubectl cp.

    Use --shutdown to shut down the API service after downloading results,
    allowing the controller pod to exit cleanly.

    If no job_id is specified, uses the last deployed benchmark.

    Examples:
        # Get results for last deployed job (downloads all artifacts)
        aiperf kube results

        # Get results for a specific job
        aiperf kube results abc123

        # Save results to a specific directory
        aiperf kube results --output ./my-results

        # Download only summary results
        aiperf kube results --summary-only

        # Get results directly from pod using kubectl cp
        aiperf kube results --from-pod

        # Download results and shut down the API service
        aiperf kube results --shutdown

    Args:
        job_id: The AIPerf job ID to get results from (default: last deployed job).
        manage_options: Kubernetes management options (kubeconfig, namespace).
        output: Output directory for results (default: ./artifacts/{name}).
        from_pod: Copy results directly from pod using kubectl cp instead of API.
        all_artifacts: Download all artifacts via API (default: True).
            Use --summary-only to download only summary results.
        shutdown: Shut down the API service after downloading results.
            Only takes effect when results are retrieved via API (not --from-pod).
        port: Local port for API port-forward (default: 0 = ephemeral).
    """
    from aiperf import cli_utils

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Retrieving Results"):
        from aiperf.kubernetes import (
            cli_helpers,
            client,
        )
        from aiperf.kubernetes import (
            console as kube_console,
        )
        from aiperf.kubernetes import (
            results as kube_results,
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

        jobset_info = await kube_client.find_jobset(job_id, namespace)
        ns = jobset_info.namespace if jobset_info else (namespace or "default")

        artifact_name = (
            jobset_info.custom_name
            if jobset_info and jobset_info.custom_name
            else job_id
        )
        output_dir = output or Path(f"./artifacts/{artifact_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        retrieval_success = False
        used_api = False

        kube_creds = {
            "kubeconfig": manage_options.kubeconfig,
            "kube_context": manage_options.kube_context,
        }

        if from_pod:
            if shutdown:
                kube_console.print_warning(
                    "--shutdown is ignored with --from-pod "
                    "(requires API connection to shut down service)"
                )
            retrieval_success = await kube_results.retrieve_results_from_pod(
                job_id, ns, output_dir, jobset_info, kube_client, **kube_creds
            )
        elif all_artifacts:
            retrieval_success = await kube_results.retrieve_all_artifacts(
                job_id, ns, output_dir, jobset_info, kube_client, port, **kube_creds
            )
            used_api = True
            if retrieval_success:
                kube_console.print_results_summary(str(output_dir))
        else:
            # --summary-only: try API first, fall back to kubectl cp
            retrieval_success = await kube_results.retrieve_results_from_api(
                job_id, ns, output_dir, jobset_info, kube_client, port, **kube_creds
            )
            used_api = True
            if retrieval_success:
                kube_console.print_results_summary(str(output_dir))
            else:
                kube_console.print_warning(
                    "Could not retrieve results from API. Trying kubectl cp..."
                )
                used_api = False
                if jobset_info:
                    retrieval_success = await kube_results.retrieve_results_from_pod(
                        job_id, ns, output_dir, jobset_info, kube_client, **kube_creds
                    )

                if not retrieval_success:
                    kube_console.print_error(f"No results found for job: {job_id}")
                    kube_console.print_info(
                        "Results may not have been generated yet, or the job was deleted."
                    )
                    kube_console.print_info(
                        "If using the operator, results are stored in the AIPerfJob status."
                    )

        if shutdown and used_api and retrieval_success:
            await kube_results.shutdown_api_service(
                job_id, ns, kube_client, port, **kube_creds
            )
