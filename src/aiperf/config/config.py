# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Root Configuration Model

This module defines the root AIPerfConfig model that brings together
all configuration sections into a single, validated configuration object.

The AIPerfConfig class is the primary entry point for loading and
working with AIPerf YAML configuration files.

Example Usage:
    >>> from config_v2 import load_config
    >>> config = load_config("benchmark.yaml")
    >>> print(config.models)
    >>> for name, phase in config.load.items():
    ...     print(f"{name}: {phase.dataset}")

    Or programmatically:
    >>> from config_v2 import AIPerfConfig
    >>> config = AIPerfConfig(
    ...     models=["llama-3-8b"],
    ...     endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    ...     datasets={"main": {"type": "synthetic", "count": 1000, "prompts": {"isl": 512}}},
    ...     load={"profiling": {"type": "concurrency", "dataset": "main", "requests": 100, "concurrency": 8}}
    ... )
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from aiperf.common.config.endpoint_config import (
        EndpointConfig as LegacyEndpointConfig,
    )
    from aiperf.common.config.input_config import InputConfig
    from aiperf.common.config.loadgen_config import LoadGeneratorConfig
    from aiperf.common.config.zmq_config import BaseZMQCommunicationConfig
    from aiperf.common.enums import AIPerfLogLevel, GPUTelemetryMode
    from aiperf.plugin.enums import ServiceRunType, TimingMode, UIType

from aiperf.config.artifacts import (
    ArtifactsConfig,
    GpuTelemetryConfig,
    ServerMetricsConfig,
)
from aiperf.config.dataset import (
    DatasetConfig,
)
from aiperf.config.endpoint import (
    EndpointConfig,
)
from aiperf.config.models import (
    AccuracyConfig,
    LoggingConfig,
    ModelsAdvanced,
    MultiRunConfig,
    RuntimeConfig,
    SLOsConfig,
    TokenizerConfig,
)
from aiperf.config.phases import (
    PhaseConfig,
)

__all__ = [
    "AIPerfConfig",
]


class AIPerfConfig(BaseModel):
    """
    Root configuration model for AIPerf YAML configuration.

    This is the primary configuration class that validates and contains
    all benchmark settings. It supports loading from YAML files with
    environment variable substitution.

    Required Sections:
        - models: Model(s) to benchmark
        - endpoint: Server connection settings
        - datasets: Named data sources
        - load: Benchmark load configuration (single or named phases)

    Optional Sections:
        - artifacts: Export and console settings
        - slos: SLO-based quality metrics (generic dict)
        - tokenizer: Token counting configuration
        - gpu_telemetry: GPU metrics from DCGM endpoints
        - server_metrics: Server metrics from Prometheus endpoints
        - runtime: Worker and communication settings
        - logging: Logging and debug settings

    Global Settings:
        - random_seed: Global seed for reproducibility

    Attributes:
        models: Model configuration. Can be a simple list of model names
            or an advanced configuration with selection strategy and weights.

        endpoint: Endpoint configuration for connecting to the inference
            server(s). Includes URLs, API type, authentication, and
            connection settings.

        datasets: Named dataset configurations. Keys are dataset names,
            values are dataset configurations (synthetic, file, public,
            or composed).

        load: Benchmark load configuration. Can be a single load config (with 'type' key)
            or named phases (dict of phase configs). Each phase defines
            a distinct testing stage with its own load profile and settings.

        artifacts: Artifacts configuration for benchmark output including
            directory, console format, and export formats.

        slos: SLO (Service Level Objectives) configuration as a generic dict.
            Maps metric names to threshold values for "good" requests.

        tokenizer: Tokenizer configuration for token counting.
            Used for ISL/OSL enforcement and metrics.

        gpu_telemetry: GPU telemetry configuration for DCGM metrics.
            Collects GPU metrics from DCGM exporter endpoints.

        server_metrics: Server metrics configuration for Prometheus scraping.
            Collects operational metrics from inference server endpoints.

        runtime: Runtime configuration for workers and communication.

        logging: Logging configuration for verbosity and debug settings.

        random_seed: Global random seed for reproducibility.
            Can be overridden per-dataset.

    Note:
        When a phase doesn't specify a dataset, the first dataset defined
        in the datasets section is used as the default.

    YAML Example (simple - single load config):
        models:
          - meta-llama/Llama-3.1-8B-Instruct

        endpoint:
          urls:
            - http://localhost:8000/v1/chat/completions

        datasets:
          main:
            type: synthetic
            entries: 1000
            prompts:
              isl: 512
              osl: 128

        load:
          type: concurrency
          requests: 100
          concurrency: 8

    YAML Example (advanced - named phases):
        models:
          - meta-llama/Llama-3.1-8B-Instruct

        endpoint:
          urls:
            - http://localhost:8000/v1/chat/completions
          type: chat
          streaming: true

        datasets:
          main:                    # First dataset = default
            type: synthetic
            entries: 5000
            prompts:
              isl: {mean: 550, stddev: 50}
              osl: {mean: 150, stddev: 25}

          warmup:
            type: synthetic
            entries: 100
            prompts:
              isl: 256
              osl: 64

        load:
          warmup:
            type: concurrency
            dataset: warmup
            exclude: true
            requests: 100
            concurrency: 8

          profiling:
            type: gamma
            dataset: main
            duration: 300
            rate: 50.0
            concurrency: 64
            smoothness: 1.5

        slos:
          time_to_first_token: 100
          inter_token_latency: 10

        artifacts:
          dir: ./results
          console: table
          summary: [json, yaml]
          records: [jsonl, csv]

        random_seed: 42

    Validation:
        The configuration is validated in several stages:
        1. Individual field validation (types, ranges, formats)
        2. Dataset reference validation (load configs reference existing datasets)
        3. Cross-field validation (mutual exclusivity, dependencies)

    Environment Variables:
        Values can reference environment variables using ${VAR} syntax.
        Optional defaults: ${VAR:default_value}

        Example:
            api_key: ${OPENAI_API_KEY}
            api_key: ${OPENAI_API_KEY:sk-default}
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={
            "title": "AIPerf Configuration Schema v2.0",
            "description": "Complete configuration schema for AIPerf LLM benchmarking tool",
            "examples": [
                {
                    "models": ["meta-llama/Llama-3.1-8B-Instruct"],
                    "endpoint": {
                        "urls": ["http://localhost:8000/v1/chat/completions"],
                        "type": "chat",
                        "streaming": True,
                    },
                    "datasets": {
                        "main": {
                            "type": "synthetic",
                            "entries": 1000,
                            "prompts": {"isl": 512, "osl": 128},
                        }
                    },
                    "load": {
                        "type": "concurrency",
                        "dataset": "main",
                        "requests": 100,
                        "concurrency": 8,
                    },
                }
            ],
        },
    )

    # ==========================================================================
    # REQUIRED SECTIONS
    # ==========================================================================

    models: Annotated[
        ModelsAdvanced,
        Field(
            description="Model configuration. Accepts a single model name string, "
            "a list of model names, or an advanced configuration with strategy "
            "and weighted items. All forms are normalized to ModelsAdvanced.",
        ),
    ]

    endpoint: Annotated[
        EndpointConfig,
        Field(
            description="Endpoint configuration for connecting to inference server(s). "
            "Includes URLs, API type, authentication, timeout, and connection settings.",
        ),
    ]

    datasets: Annotated[
        dict[str, DatasetConfig],
        Field(
            min_length=1,
            description="Named dataset configurations. Keys are dataset names that can be "
            "referenced in phases. Values are dataset configs (synthetic, file, public, "
            "or composed with source+augment).",
        ),
    ]

    load: Annotated[
        dict[str, PhaseConfig],
        Field(
            min_length=1,
            description="Benchmark load configuration. Can be a single load config "
            "(with 'type' key) or named phases (dict of phase configs). "
            "Single config is normalized to {'default': config}. "
            "Order is preserved (Python 3.7+) for execution sequence.",
        ),
    ]

    # ==========================================================================
    # OPTIONAL SECTIONS
    # ==========================================================================

    artifacts: Annotated[
        ArtifactsConfig,
        Field(
            default_factory=ArtifactsConfig,
            description="Artifacts configuration for benchmark output. "
            "Controls output directory, console format, and export formats.",
        ),
    ]

    slos: Annotated[
        SLOsConfig | None,
        Field(
            default=None,
            description="SLO (Service Level Objectives) configuration as a generic dict. "
            "Maps metric names to threshold values. "
            "A request is counted as good only if it meets ALL specified thresholds.",
        ),
    ]

    tokenizer: Annotated[
        TokenizerConfig,
        Field(
            default_factory=TokenizerConfig,
            description="Tokenizer configuration for token counting. "
            "Used for ISL/OSL enforcement and accurate metrics. "
            "If not specified, uses the first model name.",
        ),
    ]

    gpu_telemetry: Annotated[
        GpuTelemetryConfig,
        Field(
            default_factory=GpuTelemetryConfig,
            description="GPU telemetry configuration for DCGM metrics collection. "
            "Collects GPU metrics (power, utilization, temperature) from DCGM endpoints. "
            "Enabled by default. Set enabled: false to disable.",
        ),
    ]

    server_metrics: Annotated[
        ServerMetricsConfig,
        Field(
            default_factory=ServerMetricsConfig,
            description="Server metrics configuration for Prometheus scraping. "
            "Collects operational metrics (queue depth, KV cache, batch sizes) "
            "from inference server Prometheus endpoints. "
            "Enabled by default. Set enabled: false to disable.",
        ),
    ]

    runtime: Annotated[
        RuntimeConfig,
        Field(
            default_factory=RuntimeConfig,
            description="Runtime configuration for worker processes and "
            "inter-process communication.",
        ),
    ]

    logging: Annotated[
        LoggingConfig,
        Field(
            default_factory=LoggingConfig,
            description="Logging configuration for verbosity and debug settings. "
            "Supports per-service overrides and raw Python logging.dictConfig.",
        ),
    ]

    multi_run: Annotated[
        MultiRunConfig,
        Field(
            default_factory=MultiRunConfig,
            description="Multi-run benchmarking configuration for statistical reporting. "
            "When num_runs > 1, executes multiple runs and computes aggregate statistics.",
        ),
    ]

    accuracy: Annotated[
        AccuracyConfig | None,
        Field(
            default=None,
            description="Accuracy benchmarking configuration. "
            "When set, enables accuracy evaluation alongside performance profiling.",
        ),
    ]

    # ==========================================================================
    # GLOBAL SETTINGS
    # ==========================================================================

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Global random seed for reproducibility. "
            "Can be overridden per-dataset. "
            "If not set, uses system entropy.",
        ),
    ]

    # ==========================================================================
    # VALIDATORS
    # ==========================================================================

    @model_validator(mode="before")
    @classmethod
    def normalize_before_validation(cls, data: Any) -> Any:
        """Normalize input data before Pydantic validation.

        Handles:
            - model → models (singular to plural)
            - dataset → datasets (singular to plural, wrapped with key 'default')
            - load: single config (has 'type') → {'default': config}
            - models: str/list[str] → ModelsAdvanced dict format
        """
        if not isinstance(data, dict):
            return data

        # model → models (singular to plural)
        if "model" in data and "models" not in data:
            data["models"] = data.pop("model")

        # Normalize models to ModelsAdvanced dict format
        if "models" in data:
            models = data["models"]
            if isinstance(models, str):
                # Single string: "llama" → {"items": [{"name": "llama"}]}
                data["models"] = {"items": [{"name": models}]}
            elif isinstance(models, list) and models and isinstance(models[0], str):
                # List of strings: ["llama", "mistral"] → {"items": [...]}
                data["models"] = {"items": [{"name": name} for name in models]}
            # If it's already a dict (ModelsAdvanced format), pass through

        # dataset → datasets (single dataset becomes 'default')
        if "dataset" in data and "datasets" not in data:
            data["datasets"] = {"default": data.pop("dataset")}

        # Normalize load: single config (has 'type' key) → named phases
        if "load" in data:
            load = data["load"]
            if isinstance(load, dict) and "type" in load:
                # Single load config with 'type' key - wrap with key "default"
                data["load"] = {"default": load}
            # Otherwise it's already a dict of named phases

        # Coerce tokenizer=None to default (matches legacy UserConfig behavior)
        if data.get("tokenizer") is None:
            data.pop("tokenizer", None)

        return data

    @field_validator("load", mode="before")
    @classmethod
    def parse_load(cls, v: Any) -> dict[str, Any]:
        """
        Parse load configurations from dict format.

        Injects the phase name from the dict key into each phase config.
        """
        if not isinstance(v, dict):
            raise ValueError("load must be a dictionary with phase names as keys")

        result = {}
        for name, config in v.items():
            if isinstance(config, PhaseConfig):
                # Already a PhaseConfig, inject name
                config._name = name
                result[name] = config
            elif isinstance(config, dict):
                # Dict will be validated into PhaseConfig, name injected after
                result[name] = config
            else:
                raise ValueError(f"Load config '{name}' must be a dictionary")

        return result

    @field_validator("datasets", mode="before")
    @classmethod
    def parse_datasets(cls, v: Any) -> dict[str, Any]:
        """
        Parse dataset configurations, handling composed datasets.

        Composed datasets don't have a 'type' field but have 'source' and 'augment'.
        This validator ensures they're properly recognized.

        Also accepts already-constructed Pydantic models for programmatic use.
        """
        from aiperf.config.dataset import (
            ComposedDataset,
            FileDataset,
            PublicDataset,
            SyntheticDataset,
        )

        dataset_types = (SyntheticDataset, FileDataset, PublicDataset, ComposedDataset)

        if not isinstance(v, dict):
            raise ValueError("datasets must be a dictionary")

        result = {}
        for name, config in v.items():
            # Accept already-constructed Pydantic models (for programmatic use)
            if isinstance(config, dataset_types):
                result[name] = config
                continue

            if not isinstance(config, dict):
                raise ValueError(
                    f"Dataset '{name}' configuration must be a dictionary or Pydantic model"
                )

            # Check if this is a composed dataset (has source + augment, no type)
            if "source" in config and "augment" in config and "type" not in config:
                # This is a composed dataset - leave as-is for Pydantic to validate
                result[name] = config
            elif "type" not in config:
                # Default to synthetic if no type specified and not composed
                config["type"] = "synthetic"
                result[name] = config
            else:
                result[name] = config

        return result

    @model_validator(mode="after")
    def inject_endpoint_compat(self) -> AIPerfConfig:
        """Inject model names and strategy into endpoint for backward compat."""
        self.endpoint._model_names = self.get_model_names()
        self.endpoint._model_selection_strategy = self.models.strategy
        return self

    @model_validator(mode="after")
    def inject_phase_names(self) -> AIPerfConfig:
        """Inject phase names from dict keys into PhaseConfig objects."""
        for name, phase in self.load.items():
            phase._name = name
        return self

    @model_validator(mode="after")
    def validate_dataset_references(self) -> AIPerfConfig:
        """
        Validate that all dataset references in load configs exist.

        Ensures that every load config references an existing dataset by name.
        If a phase doesn't specify a dataset, the first dataset is used.
        """
        dataset_names = set(self.datasets.keys())

        for name, phase in self.load.items():
            if phase.dataset is not None and phase.dataset not in dataset_names:
                raise ValueError(
                    f"Load config '{name}' references undefined dataset '{phase.dataset}'. "
                    f"Available datasets: {sorted(dataset_names)}"
                )
            # If no dataset specified, first dataset will be used (see get_default_dataset)

        return self

    @model_validator(mode="after")
    def validate_seamless_not_on_first_phase(self) -> AIPerfConfig:
        """Ensure seamless is not enabled on the first load config."""
        if self.load:
            first_name = next(iter(self.load.keys()))
            first_phase = self.load[first_name]
            if first_phase.seamless:
                raise ValueError(
                    f"Load config '{first_name}' cannot have seamless=True because it is first. "
                    "Seamless transitions only apply to subsequent load configs."
                )
        return self

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    def get_model_names(self) -> list[str]:
        """
        Get list of all model names from the configuration.

        Returns:
            List of model name strings.
        """
        return [item.name for item in self.models.items]

    def get_dataset(self, name: str) -> DatasetConfig:
        """
        Get a dataset by name.

        Args:
            name: Dataset name.

        Returns:
            The dataset configuration.

        Raises:
            KeyError: If dataset not found.
        """
        if name not in self.datasets:
            raise KeyError(
                f"Dataset '{name}' not found. Available: {sorted(self.datasets.keys())}"
            )
        return self.datasets[name]

    def get_default_dataset_name(self) -> str:
        """
        Get the default dataset name (first dataset in the datasets dict).

        Returns:
            The name of the first dataset.
        """
        return next(iter(self.datasets.keys()))

    def get_default_dataset(self) -> DatasetConfig:
        """
        Get the default dataset (first dataset in the datasets dict).

        Returns:
            The first dataset configuration.
        """
        return next(iter(self.datasets.values()))

    def get_phase_dataset(self, phase: PhaseConfig) -> DatasetConfig:
        """
        Get the dataset for a specific phase.

        If the phase doesn't specify a dataset, uses the first dataset
        (the default).

        Args:
            phase: The phase configuration.

        Returns:
            The dataset configuration for the phase.
        """
        dataset_name = phase.dataset or self.get_default_dataset_name()
        return self.get_dataset(dataset_name)

    def get_profiling_phases(self) -> dict[str, PhaseConfig]:
        """
        Get load configs that should be included in results.

        Returns:
            Dict of load configs with exclude=False.
        """
        return {name: phase for name, phase in self.load.items() if not phase.exclude}

    def get_warmup_phases(self) -> dict[str, PhaseConfig]:
        """
        Get warmup load configs (excluded from results).

        Returns:
            Dict of load configs with exclude=True.
        """
        return {name: phase for name, phase in self.load.items() if phase.exclude}

    # ==========================================================================
    # BACKWARD COMPATIBILITY PROPERTIES
    # These properties provide access to common fields using patterns that
    # mirror the old ServiceConfig/UserConfig structure, enabling a gradual
    # migration from UserConfig+ServiceConfig to unified AIPerfConfig.
    # ==========================================================================

    # -- Profiling/warmup phase helpers (used by facade properties) --

    @property
    def _profiling_phase(self) -> PhaseConfig | None:
        """Get the first non-excluded (profiling) phase."""
        return next((p for p in self.load.values() if not p.exclude), None)

    @property
    def _warmup_phase(self) -> PhaseConfig | None:
        """Get the first excluded (warmup) phase."""
        return next((p for p in self.load.values() if p.exclude), None)

    # -- Legacy sub-config facades --

    @cached_property
    def input(self) -> InputConfig:
        """Build legacy InputConfig from AIPerfConfig data.

        Provides backward-compatible access to input.* fields.
        """
        from aiperf.config.reverse_converter import _build_input

        return _build_input(self, self._profiling_phase)

    @cached_property
    def loadgen(self) -> LoadGeneratorConfig:
        """Build legacy LoadGeneratorConfig from AIPerfConfig data.

        Provides backward-compatible access to loadgen.* fields.
        """
        from aiperf.config.reverse_converter import _build_loadgen

        return _build_loadgen(self, self._profiling_phase)

    @cached_property
    def legacy_endpoint(self) -> LegacyEndpointConfig:
        """Build legacy EndpointConfig from AIPerfConfig data.

        Use when code needs the legacy EndpointConfig type specifically.
        Most code should use config.endpoint directly (new EndpointConfig
        has backward compat properties for model_names, timeout_seconds, etc.).
        """
        from aiperf.config.reverse_converter import _build_user_config

        uc = _build_user_config(self)
        return uc.endpoint

    # -- ServiceConfig fields --

    @property
    def comm_config(self) -> BaseZMQCommunicationConfig:
        """Get the ZMQ communication configuration."""
        from aiperf.common.config.zmq_config import ZMQIPCConfig, ZMQTCPConfig
        from aiperf.common.enums import CommunicationType

        comm = self.runtime.communication
        if comm is None:
            return ZMQIPCConfig()

        if comm.type == CommunicationType.IPC:
            return ZMQIPCConfig(path=comm.path)

        if comm.type == CommunicationType.TCP:
            return ZMQTCPConfig(
                host=comm.host,
                records_push_pull_port=comm.records_port,
                credit_router_port=comm.credit_router_port,
            )

        return ZMQIPCConfig()

    @property
    def ui_type(self) -> UIType:
        """Get the UI type."""
        return self.runtime.ui

    @ui_type.setter
    def ui_type(self, value: UIType) -> None:
        """Set the UI type."""
        self.runtime.ui = value

    @cached_property
    def workers(self) -> object:
        """Facade returning an object with .max and .min for backward compat with WorkersConfig."""
        from types import SimpleNamespace

        return SimpleNamespace(max=self.runtime.workers, min=None)

    @property
    def workers_max(self) -> int | None:
        """Get the maximum number of workers."""
        return self.runtime.workers

    @property
    def record_processor_service_count(self) -> int | None:
        """Get the number of record processor services."""
        return self.runtime.record_processors

    @record_processor_service_count.setter
    def record_processor_service_count(self, value: int | None) -> None:
        """Set the number of record processor services."""
        self.runtime.record_processors = value

    @property
    def log_level(self) -> AIPerfLogLevel:
        """Get the logging level."""
        return self.logging.level

    @property
    def verbose(self) -> bool:
        """Check if verbose logging is enabled."""
        from aiperf.common.enums import AIPerfLogLevel

        return self.logging.level in (AIPerfLogLevel.DEBUG, AIPerfLogLevel.TRACE)

    @property
    def extra_verbose(self) -> bool:
        """Check if extra verbose (trace) logging is enabled."""
        from aiperf.common.enums import AIPerfLogLevel

        return self.logging.level == AIPerfLogLevel.TRACE

    @property
    def api_port(self) -> int | None:
        """Get the API port."""
        return self.runtime.api_port

    @api_port.setter
    def api_port(self, value: int | None) -> None:
        """Set the API port."""
        self.runtime.api_port = value

    @property
    def api_host(self) -> str | None:
        """Get the API host."""
        return self.runtime.api_host

    @api_host.setter
    def api_host(self, value: str | None) -> None:
        """Set the API host."""
        self.runtime.api_host = value

    @property
    def api_enabled(self) -> bool:
        """Whether the API server is enabled."""
        from aiperf.common.environment import Environment

        return (self.runtime.api_port or Environment.API_SERVER.PORT) is not None

    @property
    def service_run_type(self) -> ServiceRunType:
        """Get the service run type."""
        return self.runtime.service_run_type

    @service_run_type.setter
    def service_run_type(self, value: ServiceRunType) -> None:
        """Set the service run type."""
        self.runtime.service_run_type = value

    @property
    def workers_per_pod(self) -> int | None:
        """Get workers per pod for Kubernetes."""
        return self.runtime.workers_per_pod

    @workers_per_pod.setter
    def workers_per_pod(self, value: int | None) -> None:
        """Set workers per pod."""
        self.runtime.workers_per_pod = value

    @property
    def record_processors_per_pod(self) -> int | None:
        """Get record processors per pod for Kubernetes."""
        return self.runtime.record_processors_per_pod

    @property
    def dataset_api_base_url(self) -> str | None:
        """Get the dataset API base URL for Kubernetes."""
        return self.runtime.dataset_api_base_url

    @dataset_api_base_url.setter
    def dataset_api_base_url(self, value: str | None) -> None:
        """Set the dataset API base URL."""
        self.runtime.dataset_api_base_url = value

    @property
    def cors_origins(self) -> list[str] | None:
        """Get CORS origins for the API server."""
        return self.runtime.cors_origins

    # -- UserConfig fields --

    @property
    def timing_mode(self) -> TimingMode:
        """Get the timing mode based on the profiling phase type."""
        from aiperf.config.phases import PhaseType
        from aiperf.plugin.enums import TimingMode

        prof = self._profiling_phase
        if prof is None:
            return TimingMode.REQUEST_RATE
        if prof.type == PhaseType.FIXED_SCHEDULE:
            return TimingMode.FIXED_SCHEDULE
        if prof.type == PhaseType.USER_CENTRIC:
            return TimingMode.USER_CENTRIC_RATE
        return TimingMode.REQUEST_RATE

    @property
    def gpu_telemetry_disabled(self) -> bool:
        """Check if GPU telemetry collection is disabled."""
        return not self.gpu_telemetry.enabled

    @property
    def gpu_telemetry_mode(self) -> GPUTelemetryMode:
        """Get the GPU telemetry display mode."""
        return self.gpu_telemetry.mode

    @gpu_telemetry_mode.setter
    def gpu_telemetry_mode(self, value: GPUTelemetryMode) -> None:
        """Set the GPU telemetry display mode."""
        self.gpu_telemetry.mode = value

    @property
    def gpu_telemetry_urls(self) -> list[str]:
        """Get the GPU telemetry DCGM endpoint URLs."""
        return list(self.gpu_telemetry.urls) if self.gpu_telemetry.enabled else []

    @property
    def gpu_telemetry_metrics_file(self):
        """Get the path to custom GPU metrics CSV file."""
        return self.gpu_telemetry.metrics_file if self.gpu_telemetry.enabled else None

    @property
    def gpu_telemetry_collector_type(self):
        """Get the GPU telemetry collector type."""
        from aiperf.plugin.enums import GPUTelemetryCollectorType

        return GPUTelemetryCollectorType.DCGM

    @property
    def output(self) -> ArtifactsConfig:
        """Alias for artifacts config.

        Provides convenience access via config.output.* for file paths.
        """
        return self.artifacts

    @property
    def server_metrics_disabled(self) -> bool:
        """Check if server metrics collection is disabled."""
        return not self.server_metrics.enabled

    @property
    def server_metrics_formats(self) -> list:
        """Get the server metrics export formats."""
        return self.server_metrics.formats

    @property
    def server_metrics_urls(self) -> list[str]:
        """Get the server metrics endpoint URLs."""
        return list(self.server_metrics.urls) if self.server_metrics.enabled else []

    @property
    def benchmark_id(self) -> str:
        """Get the benchmark ID."""
        return self.artifacts.benchmark_id

    @property
    def cli_command(self) -> str | None:
        """Get the CLI command."""
        return self.artifacts.cli_command

    @cli_command.setter
    def cli_command(self, value: str | None) -> None:
        """Set the CLI command."""
        self.artifacts.cli_command = value
