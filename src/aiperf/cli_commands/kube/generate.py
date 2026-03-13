# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube generate command: output Kubernetes YAML manifests."""

from __future__ import annotations

from cyclopts import App

from aiperf.common.config.kube_config import KubeOptions
from aiperf.config.cli_builder import CLIModel

app = App(name="generate")


@app.default
async def generate(
    cli: CLIModel,
    kube_options: KubeOptions,
) -> None:
    """Generate Kubernetes YAML manifests for an AIPerf benchmark.

    Outputs YAML manifests to stdout for GitOps workflows or manual review.
    Use 'aiperf kube profile' to deploy directly to a cluster.

    Examples:
        # Generate manifests with CLI options
        aiperf kube generate --model Qwen/Qwen3-0.6B --url localhost:8000 --image aiperf:latest

        # Generate and save to file
        aiperf kube generate --model Qwen/Qwen3-0.6B --url localhost:8000 --image aiperf:latest > k8s-deployment.yaml

    Args:
        cli: Benchmark configuration (parsed from CLI flags).
        kube_options: Kubernetes-specific deployment options.
    """
    from aiperf import cli_utils

    with cli_utils.exit_on_error(title="Error Generating Kubernetes Manifests"):
        from aiperf.config.cli_builder import build_aiperf_config
        from aiperf.kubernetes import runner

        config = build_aiperf_config(cli)
        await runner.run_kubernetes_deployment(
            config,
            kube_options,
            dry_run=True,
        )
