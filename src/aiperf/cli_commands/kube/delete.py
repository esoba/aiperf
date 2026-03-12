# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube delete command: remove benchmark jobs and resources."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from aiperf.common.config.kube_config import KubeManageOptions

app = App(name="delete")


@app.default
async def delete(
    job_id: str | None = None,
    *,
    manage_options: KubeManageOptions | None = None,
    force: Annotated[bool, Parameter(name=["-f", "--force"])] = False,
    delete_namespace: Annotated[bool, Parameter(name="--delete-namespace")] = False,
) -> None:
    """Delete an AIPerf benchmark job and clean up resources.

    Removes the JobSet, ConfigMap, RBAC resources, and pods. Unlike 'cancel',
    this is intended for cleaning up completed or failed jobs. Optionally
    deletes the namespace if it was auto-generated (aiperf-{job_id} format).

    If no job_id is specified, uses the last deployed benchmark.

    Examples:
        # Delete the last deployed job (with confirmation)
        aiperf kube delete

        # Delete a specific job
        aiperf kube delete abc123

        # Force delete without confirmation
        aiperf kube delete --force

        # Also delete the auto-generated namespace
        aiperf kube delete --delete-namespace

    Args:
        job_id: The AIPerf job ID to delete (default: last deployed job).
        manage_options: Kubernetes management options (kubeconfig, namespace).
        force: Skip confirmation prompt.
        delete_namespace: Also delete the namespace if it was auto-generated.
    """
    from aiperf import cli_utils
    from aiperf.kubernetes import cli_helpers
    from aiperf.kubernetes import console as kube_console

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Deleting Job"):
        result = await cli_helpers.resolve_jobset(
            job_id,
            manage_options.namespace,
            manage_options.kubeconfig,
            manage_options.kube_context,
        )
        if not result:
            return
        job_id, jobset_info, client = result

        is_auto_namespace = jobset_info.namespace == f"aiperf-{job_id}"

        if not force:
            msg = f"Delete JobSet {jobset_info.name} in namespace {jobset_info.namespace}?"
            if delete_namespace and is_auto_namespace:
                msg = f"Delete JobSet {jobset_info.name} AND namespace {jobset_info.namespace}?"
            if not await cli_helpers.confirm_action(msg):
                return

        await client.delete_jobset(jobset_info.name, jobset_info.namespace)
        kube_console.print_success(f"Job {job_id} deleted successfully")

        cli_helpers.clear_last_benchmark_if_matches(job_id)

        if delete_namespace:
            if is_auto_namespace:
                await client.delete_namespace(jobset_info.namespace)
            else:
                kube_console.print_info(
                    f"Namespace {jobset_info.namespace} was not auto-generated, skipping deletion"
                )
