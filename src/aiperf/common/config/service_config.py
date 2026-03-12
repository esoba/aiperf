# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.base_config import ADD_TO_TEMPLATE, BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter, DisableCLI
from aiperf.common.config.config_defaults import ServiceDefaults
from aiperf.common.config.groups import Groups
from aiperf.common.config.worker_config import WorkersConfig
from aiperf.common.config.zmq_config import (
    BaseZMQCommunicationConfig,
    ZMQDualBindConfig,
    ZMQIPCConfig,
    ZMQTCPConfig,
)
from aiperf.common.enums import AIPerfLogLevel
from aiperf.common.utils import is_tty
from aiperf.plugin.enums import ServiceRunType, UIType

_logger = AIPerfLogger(__name__)


class ServiceConfig(BaseConfig):
    """Base configuration for all services. It will be provided to all services during their __init__ function."""

    _CLI_GROUP = Groups.SERVICE
    _comm_config: BaseZMQCommunicationConfig | None = None

    @model_validator(mode="after")
    def validate_log_level_from_verbose_flags(self) -> Self:
        """Set log level based on verbose flags."""
        if self.extra_verbose:
            self.log_level = AIPerfLogLevel.TRACE
        elif self.verbose:
            self.log_level = AIPerfLogLevel.DEBUG
        return self

    @model_validator(mode="after")
    def validate_ui_type(self) -> Self:
        """Set UI type based on explicit user choice, TTY detection, and verbose flags.

        Priority: explicit --ui-type > non-TTY (none) > verbose flags (simple) > default (dashboard).
        """
        if "ui_type" in self.model_fields_set:
            return self

        if not is_tty():
            self.ui_type = UIType.NONE
        elif self.verbose or self.extra_verbose:
            self.ui_type = UIType.SIMPLE
        return self

    @model_validator(mode="after")
    def validate_comm_config(self) -> Self:
        """Initialize the comm_config based on the zmq_tcp, zmq_ipc, or zmq_dual config.

        For zmq_dual (Kubernetes deployments):
        - Reads AIPERF_K8S_ZMQ_CONTROLLER_HOST env var to set controller_host
        - If env var is set: worker mode (connect via TCP to controller)
        - If env var is not set: controller mode (use IPC for local services)
        """
        _logger.debug(
            f"Validating comm_config: tcp: {self.zmq_tcp}, ipc: {self.zmq_ipc}, dual: {self.zmq_dual}"
        )

        # Count how many ZMQ configs are set
        configs_set = sum(
            1 for cfg in [self.zmq_tcp, self.zmq_ipc, self.zmq_dual] if cfg is not None
        )
        if configs_set > 1:
            raise ValueError(
                "Cannot use multiple ZMQ configurations at the same time. "
                "Choose one of: zmq_tcp, zmq_ipc, or zmq_dual"
            )

        if self.zmq_tcp is not None:
            _logger.info("Using ZMQ TCP configuration")
            self._comm_config = self.zmq_tcp
        elif self.zmq_ipc is not None:
            _logger.info("Using ZMQ IPC configuration")
            self._comm_config = self.zmq_ipc
        elif self.zmq_dual is not None:
            # For dual-bind mode, check for controller host from environment
            # Lazy import to avoid circular dependency (kubernetes package imports common.config)
            from aiperf.kubernetes.environment import K8sEnvironment

            controller_host = K8sEnvironment.ZMQ.CONTROLLER_HOST
            if controller_host:
                _logger.info(
                    f"Using ZMQ dual-bind configuration (worker mode, connecting to {controller_host})"
                )
                self.zmq_dual.controller_host = controller_host
            else:
                _logger.info(
                    "Using ZMQ dual-bind configuration (controller mode, using IPC)"
                )
            self._comm_config = self.zmq_dual
        else:
            _logger.info("Using default ZMQ IPC configuration")
            self._comm_config = ZMQIPCConfig()
        return self

    service_run_type: Annotated[
        ServiceRunType,
        Field(
            description="Type of service run (multiprocessing, kubernetes)",
        ),
        DisableCLI(reason="Only single support for now"),
    ] = ServiceDefaults.SERVICE_RUN_TYPE

    zmq_tcp: Annotated[
        ZMQTCPConfig | None,
        Field(
            description="ZMQ TCP configuration",
        ),
        DisableCLI(reason="Use config file for ZMQ settings"),
    ] = None

    zmq_ipc: Annotated[
        ZMQIPCConfig | None,
        Field(
            description="ZMQ IPC configuration",
        ),
        DisableCLI(reason="Use config file for ZMQ settings"),
    ] = None

    zmq_dual: Annotated[
        ZMQDualBindConfig | None,
        Field(
            description="ZMQ dual-bind configuration for Kubernetes. Proxies bind to both IPC "
            "(for co-located controller services) and TCP (for remote workers). Workers set "
            "AIPERF_K8S_ZMQ_CONTROLLER_HOST env var to connect via TCP.",
        ),
        DisableCLI(reason="Use config file for ZMQ settings"),
    ] = None

    workers: Annotated[
        WorkersConfig,
        Field(
            description="Worker configuration",
        ),
    ] = WorkersConfig()

    log_level: Annotated[
        AIPerfLogLevel,
        Field(
            description="Set the logging verbosity level. Controls the amount of output displayed during benchmark execution. "
            "Use `TRACE` for debugging ZMQ messages, `DEBUG` for detailed operation logs, or `INFO` (default) for standard progress updates.",
        ),
        CLIParameter(
            name=("--log-level"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.LOG_LEVEL

    verbose: Annotated[
        bool,
        Field(
            description="Equivalent to `--log-level DEBUG`. Enables detailed logging output showing function calls and state transitions. "
            "Also automatically switches UI to `simple` mode for better console visibility. Does not include raw ZMQ message logging.",
            json_schema_extra={ADD_TO_TEMPLATE: False},
        ),
        CLIParameter(
            name=("--verbose", "-v"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.VERBOSE

    extra_verbose: Annotated[
        bool,
        Field(
            description="Equivalent to `--log-level TRACE`. Enables the most verbose logging possible, including all ZMQ messages, "
            "internal state changes, and low-level operations. Also switches UI to `simple` mode. Use for deep debugging.",
            json_schema_extra={ADD_TO_TEMPLATE: False},
        ),
        CLIParameter(
            name=("--extra-verbose", "-vv"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.EXTRA_VERBOSE

    record_processor_service_count: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of `RecordProcessor` services to spawn for parallel metric computation. "
            "Higher request rates require more processors to keep up with incoming records. "
            "If not specified, automatically determined based on worker count (typically 1-2 processors per 8 workers).",
        ),
        CLIParameter(
            name=("--record-processor-service-count", "--record-processors"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.RECORD_PROCESSOR_SERVICE_COUNT

    ui_type: Annotated[
        UIType,
        Field(
            description="Select the user interface type for displaying benchmark progress. "
            "`dashboard` shows real-time metrics in a Textual TUI, `simple` uses TQDM progress bars, "
            "`none` disables UI completely. Defaults to `dashboard` in interactive terminals, "
            "`none` when not a TTY (e.g., piped or redirected output). "
            "Automatically set to `simple` when using `--verbose` or `--extra-verbose` in a TTY.",
        ),
        CLIParameter(
            name=("--ui-type", "--ui"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.UI_TYPE

    api_port: Annotated[
        int | None,
        Field(
            ge=1,
            le=65535,
            description="AIPerf API port (enables HTTP + WebSocket endpoints)",
        ),
        CLIParameter(
            name="--api-port",
            group=_CLI_GROUP,
        ),
    ] = None

    api_host: Annotated[
        str | None,
        Field(
            description="AIPerf API host (requires --api-port or AIPERF_API_SERVER_PORT to be set)",
        ),
        CLIParameter(
            name="--api-host",
            group=_CLI_GROUP,
        ),
    ] = None

    dataset_api_base_url: Annotated[
        str | None,
        Field(
            description="Base URL for dataset API endpoints (for Kubernetes workers to download datasets). "
            "Example: http://controller-0-0.jobset.namespace:9090/api/dataset",
        ),
        DisableCLI(reason="Set via environment variable in Kubernetes mode"),
    ] = None

    workers_per_pod: Annotated[
        int | None,
        Field(
            ge=1,
            le=100,
            description="Number of worker subprocesses per Kubernetes worker pod. "
            "Each pod downloads the dataset once and shares it across workers via mmap. "
            "Higher values reduce network overhead but increase per-pod resource requirements.",
        ),
        CLIParameter(
            name="--workers-per-pod",
            group=_CLI_GROUP,
        ),
    ] = None

    record_processors_per_pod: Annotated[
        int | None,
        Field(
            ge=1,
            le=100,
            description="Number of record processor subprocesses per Kubernetes worker pod. "
            "If not specified, defaults to max(1, workers_per_pod / 4).",
        ),
        CLIParameter(
            name="--record-processors-per-pod",
            group=_CLI_GROUP,
        ),
    ] = None

    cors_origins: Annotated[
        list[str] | None,
        Field(
            description="List of allowed CORS origins for the API server. "
            "If not specified, uses AIPERF_API_SERVER_CORS_ORIGINS environment variable.",
        ),
        DisableCLI(reason="Set via environment variable"),
    ] = None

    @model_validator(mode="after")
    def validate_api_host_requires_port(self) -> Self:
        """Validate that --api-host is not set without --api-port."""
        if self.api_host is not None and self.api_port is None:
            from aiperf.common.environment import Environment

            if Environment.API_SERVER.PORT is None:
                raise ValueError(
                    "--api-host requires --api-port or AIPERF_API_SERVER_PORT to be set"
                )
        return self

    @property
    def api_enabled(self) -> bool:
        """Whether the API server is enabled (port configured via CLI or environment)."""
        from aiperf.common.environment import Environment

        return (self.api_port or Environment.API_SERVER.PORT) is not None

    @property
    def comm_config(self) -> BaseZMQCommunicationConfig:
        """Get the communication configuration."""
        if not self._comm_config:
            raise ValueError(
                "Communication configuration is not set. Please provide a valid configuration."
            )
        return self._comm_config
