# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube preflight command: pre-deployment validation checks."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from aiperf.common.config.kube_config import KubeManageOptions

app = App(name="preflight")


@app.default
async def preflight(
    *,
    manage_options: KubeManageOptions | None = None,
    image: Annotated[str | None, Parameter(name="--image")] = None,
    image_pull_secret: Annotated[
        list[str] | None, Parameter(name="--image-pull-secret")
    ] = None,
    secret: Annotated[list[str] | None, Parameter(name="--secret")] = None,
    endpoint: Annotated[str | None, Parameter(name="--endpoint")] = None,
    workers: Annotated[int, Parameter(name="--workers-max")] = 1,
) -> None:
    """Run comprehensive pre-flight checks for Kubernetes deployment.

    Validates cluster connectivity, permissions, resources, and configuration
    before deploying a benchmark. Useful for debugging cluster setup issues.

    Checks performed:
    - Cluster connectivity and Kubernetes version
    - Namespace availability
    - RBAC permissions for all required operations
    - JobSet CRD installation and controller health
    - Resource quotas and node capacity
    - Secret existence (if specified)
    - Image pull information
    - Network policies
    - DNS resolution (CoreDNS)
    - Endpoint connectivity (if specified)

    Examples:
        # Run basic pre-flight checks
        aiperf kube preflight

        # Check a specific namespace with resource estimation
        aiperf kube preflight --namespace aiperf-bench --workers-max 10

        # Verify all deployment requirements
        aiperf kube preflight \\
            --image myregistry.io/aiperf:latest \\
            --image-pull-secret my-registry-creds \\
            --secret api-key \\
            --endpoint http://my-llm:8000/v1

    Args:
        manage_options: Kubernetes management options (kubeconfig, namespace).
        image: Container image to verify pull access for.
        image_pull_secret: Image pull secret names to verify exist.
        secret: Secret names to verify exist (for env vars or mounts).
        endpoint: LLM endpoint URL to verify connectivity.
        workers: Planned number of workers for resource estimation.
    """
    from aiperf import cli_utils
    from aiperf.kubernetes import console as kube_console

    with cli_utils.exit_on_error(title="Error Running Pre-flight Checks"):
        from aiperf.kubernetes import preflight as kube_preflight

        kube_console.logger.info(
            "[bold cyan]Running AIPerf Kubernetes Pre-flight Checks[/bold cyan]"
        )
        kube_console.logger.info("")
        manage_options = manage_options or KubeManageOptions()
        checker = kube_preflight.PreflightChecker(
            namespace=manage_options.namespace or "default",
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
            image=image,
            image_pull_secrets=list(image_pull_secret or []),
            secrets=list(secret or []),
            endpoint_url=endpoint,
            workers=workers,
        )

        results = await checker.run_all_checks()

        if not results.passed:
            raise SystemExit(1)
