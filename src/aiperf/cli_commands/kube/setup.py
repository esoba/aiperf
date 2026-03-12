# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube setup command: cluster preparation for AIPerf benchmarks."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from aiperf.common.config.kube_config import KubeManageOptions

app = App(name="setup")


@app.default
async def setup(
    *,
    jobset_version: Annotated[str | None, Parameter(name="--jobset-version")] = None,
    dry_run: Annotated[bool, Parameter(name="--dry-run")] = False,
    skip_jobset: Annotated[bool, Parameter(name="--skip-jobset")] = False,
    operator: Annotated[bool, Parameter(name="--operator")] = False,
    operator_namespace: Annotated[
        str, Parameter(name="--operator-namespace")
    ] = "aiperf-system",
    operator_chart: Annotated[
        str, Parameter(name="--operator-chart")
    ] = "oci://nvcr.io/nvidia/aiperf-operator",
    operator_values: Annotated[str | None, Parameter(name="--operator-values")] = None,
    operator_set: Annotated[list[str] | None, Parameter(name="--operator-set")] = None,
    manage_options: KubeManageOptions | None = None,
) -> None:
    """Set up a Kubernetes cluster for AIPerf benchmarks.

    Installs prerequisites and validates the cluster is ready to run benchmarks.
    Safe to run multiple times -- already-installed components are skipped.

    Setup steps:
    - Verify cluster connectivity
    - Install JobSet CRD and controller (unless --skip-jobset)
    - Install AIPerf operator via Helm (if --operator)
    - Create namespace (if --namespace specified and doesn't exist)

    Examples:
        # Basic setup (install latest JobSet CRD)
        aiperf kube setup

        # Install JobSet + AIPerf operator
        aiperf kube setup --operator

        # Operator with custom image tag
        aiperf kube setup --operator --operator-set image.tag=v1.2.3

        # Operator from a local chart
        aiperf kube setup --operator --operator-chart ./deploy/helm/aiperf-operator

        # Pin a specific JobSet version
        aiperf kube setup --jobset-version v0.5.2

        # Setup with a dedicated benchmark namespace
        aiperf kube setup --namespace aiperf-bench

        # Preview what would be done
        aiperf kube setup --dry-run

    Args:
        jobset_version: Specific JobSet version tag to install. Queries GitHub for latest if omitted.
        dry_run: Print planned actions without executing them.
        skip_jobset: Skip JobSet CRD installation.
        operator: Install the AIPerf Kubernetes operator via Helm.
        operator_namespace: Namespace for the operator deployment.
        operator_chart: Helm chart reference (OCI URL or local path).
        operator_values: Path to a Helm values.yaml override file.
        operator_set: Helm --set key=value overrides. Can be repeated.
        manage_options: Kubernetes management options (kubeconfig, context).
    """
    from aiperf import cli_utils

    with cli_utils.exit_on_error(title="Error Running Setup"):
        from aiperf.kubernetes import setup as kube_setup

        manage_options = manage_options or KubeManageOptions()

        passed = await kube_setup.run_setup(
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
            namespace=manage_options.namespace,
            jobset_version=jobset_version,
            dry_run=dry_run,
            skip_jobset=skip_jobset,
            operator=operator,
            operator_namespace=operator_namespace,
            operator_chart=operator_chart,
            operator_values=operator_values,
            operator_set=operator_set,
        )
        if not passed:
            raise SystemExit(1)
