# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for running individual AIPerf services."""

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from aiperf.plugin.enums import ServiceType

app = App(name="service")


@app.default
def service(
    *,
    service_type: Annotated[
        ServiceType,
        Parameter(
            name="--type",
            show_env_var=False,
            negative=False,
            help="Service type to run.",
        ),
    ],
    benchmark_run_file: Annotated[
        Path | None,
        Parameter(
            name="--benchmark-run",
            show_env_var=False,
            negative=False,
            help="Path to a BenchmarkRun JSON file. "
            "The service bootstraps with a fully-built BenchmarkRun "
            "including metadata, variation, and trial info.",
        ),
    ] = None,
    service_id: Annotated[
        str | None,
        Parameter(
            show_env_var=False,
            negative=False,
            help="Unique identifier for the service instance. "
            "Useful when running multiple instances of the same service type.",
        ),
    ] = None,
    health_host: Annotated[
        str | None,
        Parameter(
            show_env_var=False,
            negative=False,
            help="Host to bind the health server to. "
            "Falls back to AIPERF_SERVICE_HEALTH_HOST environment variable.",
        ),
    ] = None,
    health_port: Annotated[
        int | None,
        Parameter(
            show_env_var=False,
            negative=False,
            help="HTTP port for health endpoints (/healthz, /readyz). "
            "Required for Kubernetes liveness and readiness probes. "
            "Falls back to AIPERF_SERVICE_HEALTH_PORT environment variable.",
        ),
    ] = None,
    api_port: Annotated[
        int | None,
        Parameter(
            show_env_var=False,
            negative=False,
            help="HTTP port for API endpoints (e.g., /api/dataset, /api/progress). "
            "Only used by services that expose HTTP APIs.",
        ),
    ] = None,
) -> None:
    """Run an AIPerf service in a single process.

    _Advanced use only — intended for developers and Kubernetes/distributed
    deployments where services run in separate containers or nodes._

    For standard single-node benchmarking, use the `aiperf profile` command instead.
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title=f"Error Running AIPerf Service {service_type}"):
        from aiperf.common.bootstrap import bootstrap_and_run_service
        from aiperf.common.environment import Environment

        run = None
        config = None

        if benchmark_run_file is not None:
            import orjson

            from aiperf.config.benchmark import BenchmarkRun

            run = BenchmarkRun.model_validate(
                orjson.loads(benchmark_run_file.read_bytes())
            )

        if health_host is not None:
            Environment.SERVICE.HEALTH_ENABLED = True
            Environment.SERVICE.HEALTH_HOST = health_host

        if health_port is not None:
            Environment.SERVICE.HEALTH_ENABLED = True
            Environment.SERVICE.HEALTH_PORT = health_port

        bootstrap_and_run_service(
            service_type=service_type,
            run=run,
            config=config,
            service_id=service_id,
            api_port=api_port,
        )
