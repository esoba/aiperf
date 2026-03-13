# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for running individual AIPerf services."""

from pathlib import Path
from typing import Annotated

from cyclopts import App

from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.plugin.enums import ServiceType

app = App(name="service")


@app.default
def service(
    service_type: Annotated[
        ServiceType, CLIParameter(name="--type", help="Service type to run.")
    ],
    user_config_file: Annotated[
        Path | None,
        CLIParameter(
            help="Path to the user configuration file (JSON or YAML). "
            "Falls back to AIPERF_CONFIG_USER_FILE environment variable."
        ),
    ] = None,
    service_config_file: Annotated[
        Path | None,
        CLIParameter(
            help="Path to the service configuration file (JSON or YAML). "
            "Falls back to AIPERF_CONFIG_SERVICE_FILE environment variable, "
            "then to default ServiceConfig if neither is set."
        ),
    ] = None,
    service_id: Annotated[
        str | None,
        CLIParameter(
            help="Unique identifier for the service instance. "
            "Useful when running multiple instances of the same service type."
        ),
    ] = None,
    health_host: Annotated[
        str | None,
        CLIParameter(
            help="Host to bind the health server to. "
            "Falls back to AIPERF_SERVICE_HEALTH_HOST environment variable."
        ),
    ] = None,
    health_port: Annotated[
        int | None,
        CLIParameter(
            help="HTTP port for health endpoints (/healthz, /readyz). "
            "Required for Kubernetes liveness and readiness probes. "
            "Falls back to AIPERF_SERVICE_HEALTH_PORT environment variable."
        ),
    ] = None,
    api_port: Annotated[
        int | None,
        CLIParameter(
            help="HTTP port for API endpoints (e.g., /api/dataset, /api/progress). "
            "Only used by services that expose HTTP APIs."
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
        from aiperf.config.loader import load_config

        # Load unified AIPerfConfig from the config file or AIPERF_CONFIG_FILE env var.
        # The legacy user_config_file / service_config_file parameters are accepted
        # for backward compatibility; prefer user_config_file if provided.
        config_file = user_config_file or service_config_file
        config = None
        if config_file is not None:
            config = load_config(config_file)

        if health_host is not None:
            Environment.SERVICE.HEALTH_ENABLED = True
            Environment.SERVICE.HEALTH_HOST = health_host

        if health_port is not None:
            Environment.SERVICE.HEALTH_ENABLED = True
            Environment.SERVICE.HEALTH_PORT = health_port

        bootstrap_and_run_service(
            service_type=service_type,
            config=config,
            service_id=service_id,
            api_port=api_port,
        )
