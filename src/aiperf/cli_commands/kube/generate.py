# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube generate command: output Kubernetes YAML manifests."""

from __future__ import annotations

from cyclopts import App

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.config.kube_config import KubeOptions

app = App(name="generate")


@app.default
async def generate(
    user_config: UserConfig,
    kube_options: KubeOptions,
    service_config: ServiceConfig | None = None,
) -> None:
    """Generate Kubernetes YAML manifests for an AIPerf benchmark.

    Outputs YAML manifests to stdout for GitOps workflows or manual review.
    Use 'aiperf kube profile' to deploy directly to a cluster.

    Examples:
        # Generate manifests with CLI options
        aiperf kube generate --model Qwen/Qwen3-0.6B --url localhost:8000 --image aiperf:latest

        # Generate and save to file
        aiperf kube generate --user-config benchmark.yaml --image aiperf:latest > k8s-deployment.yaml

    Args:
        user_config: User configuration for the benchmark (same as 'aiperf profile').
        kube_options: Kubernetes-specific deployment options.
        service_config: Service configuration options (optional).
    """
    from aiperf import cli_utils

    with cli_utils.exit_on_error(title="Error Generating Kubernetes Manifests"):
        from aiperf.common.config import load_service_config
        from aiperf.kubernetes import runner

        service_config = service_config or load_service_config()
        await runner.run_kubernetes_deployment(
            user_config, service_config, kube_options, dry_run=True
        )
