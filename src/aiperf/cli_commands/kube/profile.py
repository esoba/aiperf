# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube profile command: deploy and run a benchmark in Kubernetes."""

from __future__ import annotations

import asyncio
from typing import Annotated

from cyclopts import App, Parameter

from aiperf.common.config.kube_config import KubeOptions
from aiperf.config.cli_builder import CLIModel

app = App(name="profile")


@app.default
async def profile(
    cli: CLIModel,
    kube_options: KubeOptions,
    detach: Annotated[bool, Parameter(name=["-d", "--detach"])] = False,
    no_wait: Annotated[bool, Parameter(name="--no-wait", negative=())] = False,
    attach_port: Annotated[int, Parameter(name="--attach-port")] = 0,
    skip_endpoint_check: Annotated[
        bool, Parameter(name="--skip-endpoint-check", negative=())
    ] = False,
    skip_preflight: Annotated[
        bool, Parameter(name="--skip-preflight", negative=())
    ] = False,
) -> None:
    """Run a benchmark in Kubernetes.

    By default, blocks and streams controller logs until completion.
    Use --detach for fire-and-forget deployment.

    Before deploying, validates that the LLM endpoint is reachable.
    Use --skip-endpoint-check to bypass this validation.

    Examples:
        # Stream controller logs (default)
        aiperf kube profile --model Qwen/Qwen3-0.6B \\
            --url http://server:8000 --image aiperf:latest --workers-max 10

        # CI/CD: deploy and exit immediately
        aiperf kube profile --model Qwen/Qwen3-0.6B \\
            --url http://server:8000 --image aiperf:latest --detach

    Args:
        cli: Benchmark configuration (parsed from CLI flags).
        kube_options: Kubernetes-specific deployment options.
        detach: Exit immediately after deploying (don't wait for completion).
            Automatically enabled in non-interactive environments (pipes, CI/CD).
        no_wait: Don't wait for pods to be ready before attaching (advanced).
        attach_port: Local port for API port-forward (default: 0 = ephemeral).
        skip_endpoint_check: Skip endpoint health validation before deploying.
        skip_preflight: Skip automatic pre-flight checks before deploying.
            For comprehensive checks, run 'aiperf kube preflight' separately.
    """
    import sys

    from aiperf import cli_utils
    from aiperf.kubernetes import console as kube_console

    with cli_utils.exit_on_error(title="Error Running Kubernetes Benchmark"):
        from aiperf.config.cli_builder import build_aiperf_config
        from aiperf.config.reverse_converter import convert_to_legacy_configs
        from aiperf.kubernetes import runner

        aiperf_config = build_aiperf_config(cli)
        user_config, service_config = convert_to_legacy_configs(aiperf_config)

        if not skip_preflight:
            from aiperf.kubernetes import preflight as kube_preflight

            endpoint_url = (
                aiperf_config.endpoint.urls[0]
                if not skip_endpoint_check and aiperf_config.endpoint.urls
                else None
            )
            preflight_ns = kube_options.namespace or "aiperf-preflight-check"
            checker = kube_preflight.PreflightChecker(
                namespace=preflight_ns,
                kubeconfig=kube_options.kubeconfig,
                kube_context=kube_options.kube_context,
                endpoint_url=endpoint_url,
            )
            preflight_results = await checker.run_quick_checks(show_progress=True)
            if preflight_results.passed:
                kube_console.print_success("Pre-flight checks passed")
            else:
                for check in preflight_results.checks:
                    if check.status == kube_preflight.CheckStatus.FAIL:
                        kube_console.print_error(f"{check.name}: {check.message}")
                        for hint in check.hints:
                            kube_console.print_info(f"  {hint}")
                kube_console.print_info(
                    "Use --skip-preflight to bypass, or run "
                    "'aiperf kube preflight' for detailed diagnostics"
                )
                raise SystemExit(1)
        job_id, namespace = await runner.run_kubernetes_deployment(
            user_config,
            service_config,
            kube_options,
            dry_run=False,
            aiperf_config=aiperf_config,
        )

        kube_console.save_last_benchmark(job_id, namespace, name=kube_options.name)

        is_interactive = sys.stdout.isatty()
        should_detach = detach or not is_interactive

        if not is_interactive and not detach:
            kube_console.print_warning(
                "Non-interactive environment detected, using detach mode"
            )

        if should_detach:
            kube_console.print_detach_info(job_id, namespace, name=kube_options.name)
            return

        try:
            from aiperf.kubernetes import attach as kube_attach

            await kube_attach.auto_attach_workflow(
                job_id,
                namespace,
                attach_port,
                wait_for_ready=not no_wait,
                kubeconfig=kube_options.kubeconfig,
                kube_context=kube_options.kube_context,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            kube_console.print_interrupt_info(job_id, namespace)
            return
