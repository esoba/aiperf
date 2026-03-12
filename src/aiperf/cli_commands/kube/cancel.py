# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube cancel command: cancel a running benchmark."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from aiperf.common.config.kube_config import KubeManageOptions

app = App(name="cancel")


@app.default
async def cancel(
    job_id: str | None = None,
    *,
    manage_options: KubeManageOptions | None = None,
    force: Annotated[bool, Parameter(name=["-f", "--force"])] = False,
) -> None:
    """Cancel a running AIPerf benchmark.

    Deletes the JobSet and all associated resources. Results are lost once
    cancelled. Run 'aiperf kube results' first to save partial results.

    Use 'aiperf kube delete' instead to clean up already-completed jobs,
    optionally removing the auto-generated namespace.

    If no job_id is specified, uses the last deployed benchmark.

    Examples:
        # Cancel the last deployed job (with confirmation)
        aiperf kube cancel

        # Cancel a specific job
        aiperf kube cancel abc123

        # Force cancel without confirmation
        aiperf kube cancel --force

    Args:
        job_id: The AIPerf job ID to cancel (default: last deployed job).
        manage_options: Kubernetes management options (kubeconfig, namespace).
        force: Skip confirmation prompt.
    """
    from aiperf import cli_utils
    from aiperf.kubernetes import cli_helpers
    from aiperf.kubernetes import console as kube_console

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Cancelling Job"):
        result = await cli_helpers.resolve_jobset(
            job_id,
            manage_options.namespace,
            manage_options.kubeconfig,
            manage_options.kube_context,
        )
        if not result:
            return
        job_id, jobset_info, client = result

        # Check current status
        is_completed = jobset_info.status in ("Completed", "Failed")
        if is_completed:
            kube_console.print_warning(
                f"Job {job_id} has already {jobset_info.status.lower()}."
            )
            if not force and not await cli_helpers.confirm_action("Delete anyway?"):
                return

        if (
            not is_completed
            and not force
            and not await cli_helpers.confirm_action(
                f"Cancel running job {job_id} in namespace {jobset_info.namespace}?"
            )
        ):
            return

        kube_console.print_step(f"Cancelling job {job_id}...")
        await client.delete_jobset(jobset_info.name, jobset_info.namespace)

        cli_helpers.clear_last_benchmark_if_matches(job_id)

        kube_console.print_success(f"Job {job_id} cancelled and resources deleted.")
