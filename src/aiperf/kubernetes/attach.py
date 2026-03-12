# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attach and auto-attach workflows for kube commands."""

from __future__ import annotations

from pathlib import Path

from aiperf.kubernetes.client import AIPerfKubeClient
from aiperf.kubernetes.console import (
    logger,
    print_action,
    print_benchmark_complete,
    print_error,
    print_info,
    print_interrupt_info,
    print_results_summary,
    print_success,
    print_warning,
    status_log,
)
from aiperf.kubernetes.constants import Containers
from aiperf.kubernetes.enums import JobSetStatus, PodPhase
from aiperf.kubernetes.logs import save_pod_logs
from aiperf.kubernetes.port_forward import port_forward_with_status
from aiperf.kubernetes.results import (
    retrieve_all_artifacts,
    stream_controller_logs,
)
from aiperf.kubernetes.ui_dispatch import API_WS_PATH, stream_progress


async def attach_to_benchmark(
    job_id: str,
    namespace: str | None,
    local_port: int,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> None:
    """Attach to a running benchmark and stream progress."""
    kube_creds = {"kubeconfig": kubeconfig, "kube_context": kube_context}
    client = await AIPerfKubeClient.create(**kube_creds)

    jobset_info = await client.find_jobset(job_id, namespace)
    if not jobset_info:
        print_error(f"No running job found with ID: {job_id}")
        return

    if jobset_info.status == JobSetStatus.COMPLETED:
        print_warning(f"Job {job_id} has already completed.")
        print_action("Use 'aiperf kube results' to retrieve results.")
        return
    if jobset_info.status == JobSetStatus.FAILED:
        print_error(f"Job {job_id} has failed.")
        print_action("Use 'aiperf kube logs' to investigate.")
        return

    pod_info = await client.find_controller_pod(jobset_info.namespace, job_id)
    if not pod_info:
        print_warning(
            f"No controller pod found for job {job_id}. The benchmark may still be starting."
        )
        return

    pod_name, pod_phase = pod_info
    if pod_phase != PodPhase.RUNNING:
        print_warning(f"Controller pod {pod_name} is not ready (status: {pod_phase})")
        return

    print_info(
        f"Attaching to JobSet/{jobset_info.name} in namespace {jobset_info.namespace}"
    )
    print_success(f"Controller pod: {pod_name}")

    try:
        async with port_forward_with_status(
            jobset_info.namespace, pod_name, local_port, **kube_creds
        ) as port:
            ws_url = f"ws://localhost:{port}{API_WS_PATH}"
            await stream_progress(ws_url)

    except KeyboardInterrupt:
        print_interrupt_info(job_id, jobset_info.namespace)


async def auto_attach_workflow(
    job_id: str,
    namespace: str,
    attach_port: int,
    wait_for_ready: bool = True,
    stream_ws: bool = False,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> None:
    """Execute auto-attach workflow after deployment.

    Waits for controller pod, then streams progress via WebSocket or
    controller logs to the terminal. Retrieves results on completion.
    """
    kube_creds = {"kubeconfig": kubeconfig, "kube_context": kube_context}
    client = await AIPerfKubeClient.create(**kube_creds)

    if wait_for_ready:
        with status_log("Waiting for controller pod to be ready..."):
            pod_name = await client.wait_for_controller_pod_ready(
                namespace, job_id, timeout=300
            )
        print_success(f"Controller pod ready: {pod_name}")
    else:
        result = await client.find_controller_pod(namespace, job_id)
        if not result:
            raise RuntimeError(
                f"No controller pod found for job {job_id}. "
                f"Remove --no-wait to wait for pod readiness."
            )
        pod_name, _ = result

    if stream_ws:
        async with port_forward_with_status(
            namespace, pod_name, attach_port, **kube_creds
        ) as port:
            ws_url = f"ws://localhost:{port}{API_WS_PATH}"
            await stream_progress(ws_url)
    else:
        logger.info("")
        await stream_controller_logs(
            namespace, pod_name, container=Containers.CONTROL_PLANE, **kube_creds
        )

    print_benchmark_complete()
    with status_log("Retrieving results..."):
        await retrieve_and_display_results(job_id, namespace, client, **kube_creds)


async def retrieve_and_display_results(
    job_id: str,
    namespace: str,
    client: AIPerfKubeClient,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> None:
    """Retrieve all artifacts from API service and display summary."""
    jobset_info = await client.find_jobset(job_id, namespace)

    artifact_name = (
        jobset_info.custom_name if jobset_info and jobset_info.custom_name else job_id
    )
    output_dir = Path(f"./artifacts/{artifact_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    success = await retrieve_all_artifacts(
        job_id,
        namespace,
        output_dir,
        jobset_info,
        client,
        local_port=0,
        kubeconfig=kubeconfig,
        kube_context=kube_context,
    )

    await save_pod_logs(
        job_id,
        namespace,
        output_dir,
        client,
        kubeconfig=kubeconfig,
        kube_context=kube_context,
    )

    if success:
        print_results_summary(str(output_dir))
    else:
        print_warning("Results not yet available from API")
        print_action(f"Try: aiperf kube results {job_id} --namespace {namespace}")
