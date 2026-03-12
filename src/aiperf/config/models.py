# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

This module contains all Pydantic models for the AIPerf YAML configuration
system. Models are organized by configuration section and include comprehensive
documentation, validation, and examples.

Sections:
    1. Models - Model names and selection strategies
    2. SLOs - Service Level Objectives (goodput thresholds)
    3. Tokenizer - HuggingFace tokenizer configuration
    4. Runtime - Worker and communication settings
    5. Logging - Logging and debug configuration
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from aiperf.common.enums import (
    AIPerfLogLevel,
    CommunicationType,
    ModelSelectionStrategy,
)
from aiperf.plugin.enums import UIType

__all__ = [
    # Section 1: Models
    "TokenizerOverride",
    "ModelItem",
    "ModelsAdvanced",
    # Section 2: SLOs (replaces GoodputConfig)
    "SLOsConfig",
    # Section 3: Tokenizer
    "TokenizerConfig",
    # Section 4: Runtime
    "TcpProxyConfig",
    "IpcCommunicationConfig",
    "TcpCommunicationConfig",
    "DualBindCommunicationConfig",
    "CommunicationConfig",
    "RuntimeConfig",
    # Section 5: Logging
    "LoggingConfig",
    # Section 6: Multi-Run
    "MultiRunConfig",
    # Section 7: Accuracy
    "AccuracyConfig",
]

# =============================================================================
# SECTION 1: MODELS CONFIGURATION
# =============================================================================


class TokenizerOverride(BaseModel):
    """
    Per-model tokenizer override configuration.

    Allows specifying a different tokenizer for a specific model,
    useful when models require specialized tokenization.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str,
        Field(description="HuggingFace tokenizer identifier or local filesystem path."),
    ]


class ModelItem(BaseModel):
    """
    Configuration for a single model in advanced models configuration.

    Used when the models section uses the advanced format with
    explicit items, weights, and per-model settings.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str,
        Field(description="Model name or identifier as known to the inference server."),
    ]

    weight: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Selection weight for weighted strategy (0.0-1.0). "
            "Weights are normalized across all models. "
            "Example: weight=0.7 means ~70%% of requests to this model.",
        ),
    ]

    lora: Annotated[
        str | None,
        Field(
            default=None,
            description="LoRA adapter name to load with this model. "
            "Server must support dynamic LoRA adapter loading.",
        ),
    ]

    modalities: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="List of input modalities this model supports. "
            "Used with modality_aware selection strategy. "
            "Valid values: 'text', 'image', 'audio', 'video'.",
        ),
    ]

    tokenizer: Annotated[
        TokenizerOverride | None,
        Field(
            default=None,
            description="Per-model tokenizer override. "
            "Use when this model requires a different tokenizer than global config.",
        ),
    ]


class ModelsAdvanced(BaseModel):
    """
    Advanced models configuration with selection strategy and item details.

    Use this format when you need weighted routing, LoRA adapters,
    modality-aware selection, or per-model tokenizer overrides.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Annotated[
        ModelSelectionStrategy,
        Field(
            default=ModelSelectionStrategy.ROUND_ROBIN,
            description="Strategy for selecting models when multiple are configured. "
            "round_robin cycles through models, random selects randomly, "
            "weighted uses configured weights, modality_aware routes by input type.",
        ),
    ]

    items: Annotated[
        list[ModelItem],
        Field(
            min_length=1,
            description="List of model configurations. At least one model required.",
        ),
    ]

    @model_validator(mode="after")
    def validate_weights_for_weighted_strategy(self) -> ModelsAdvanced:
        """Ensure weights are provided when using weighted strategy."""
        if self.strategy == ModelSelectionStrategy.WEIGHTED:
            if not all(item.weight is not None for item in self.items):
                raise ValueError(
                    "All models must have weights specified when using weighted strategy"
                )
            total_weight = sum(
                item.weight for item in self.items if item.weight is not None
            )
            if not (0.99 <= total_weight <= 1.01):
                raise ValueError(f"Model weights must sum to 1.0, got {total_weight}")
        return self


# =============================================================================
# SECTION 2: SLOs CONFIGURATION (replaces GoodputConfig)
# =============================================================================

# SLOs is a generic dict allowing any metric name with a threshold value.
# Common metrics: request_latency, time_to_first_token, inter_token_latency, tokens_per_second
SLOsConfig = dict[str, float]
"""
SLOs (Service Level Objectives) configuration as a generic dict.

Maps metric names to threshold values (in milliseconds for latency metrics).
A request is counted as "good" only if it meets ALL specified thresholds.

Example:
    slos:
      request_latency: 500       # max 500ms end-to-end latency
      time_to_first_token: 100   # max 100ms TTFT
      inter_token_latency: 15    # max 15ms between tokens
      tokens_per_second: 50      # min 50 tokens/second
"""


# =============================================================================
# SECTION 6: TOKENIZER CONFIGURATION
# =============================================================================


class TokenizerConfig(BaseModel):
    """
    Tokenizer configuration for token counting and prompt generation.

    AIPerf uses a HuggingFace tokenizer for accurate token counting,
    which is essential for ISL/OSL enforcement and metrics calculation.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str | None,
        Field(
            default=None,
            description="HuggingFace tokenizer identifier or local filesystem path. "
            "Should match the model's tokenizer for accurate token counts. "
            "Example: 'meta-llama/Llama-3.1-8B-Instruct'",
        ),
    ]

    revision: Annotated[
        str,
        Field(
            default="main",
            description="Model revision to use: branch name, tag, or commit hash. "
            "Use for version pinning to ensure reproducibility.",
        ),
    ]

    trust_remote_code: Annotated[
        bool,
        Field(
            default=False,
            description="Allow execution of custom tokenizer code from the repository. "
            "Required for some models but poses security risk. "
            "Only enable for trusted sources.",
        ),
    ]


# =============================================================================
# SECTION 4: RUNTIME CONFIGURATION
# =============================================================================


class TcpProxyConfig(BaseModel):
    """TCP proxy port configuration for a single ZMQ proxy."""

    model_config = ConfigDict(extra="forbid")

    frontend_port: Annotated[
        int,
        Field(ge=1, le=65535, description="TCP port for proxy frontend."),
    ]

    backend_port: Annotated[
        int,
        Field(ge=1, le=65535, description="TCP port for proxy backend."),
    ]


class IpcCommunicationConfig(BaseModel):
    """
    IPC (Unix socket) communication configuration.

    For single-machine deployments with lowest latency.
    Uses Unix domain sockets for all inter-service communication.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[CommunicationType.IPC],
        Field(description="Communication type. Must be 'ipc'."),
    ]

    path: Annotated[
        str,
        Field(
            default="/tmp/aiperf",
            description="Directory for IPC socket files. "
            "AIPerf creates multiple socket files in this directory.",
        ),
    ]


class TcpCommunicationConfig(BaseModel):
    """
    TCP socket communication configuration.

    For distributed deployments across machines.
    Provides detailed port configuration for all ZMQ proxies.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[CommunicationType.TCP],
        Field(description="Communication type. Must be 'tcp'."),
    ]

    host: Annotated[
        str,
        Field(
            default="127.0.0.1",
            description="Host address for TCP communication. "
            "Use 0.0.0.0 to listen on all interfaces.",
        ),
    ]

    # Core communication ports
    records_port: Annotated[
        int,
        Field(
            default=5557,
            ge=1,
            le=65535,
            description="Port for records push/pull communication.",
        ),
    ]

    credit_router_port: Annotated[
        int,
        Field(
            default=5564,
            ge=1,
            le=65535,
            description="Port for credit router (ROUTER-DEALER).",
        ),
    ]

    # Proxy configurations
    event_bus_proxy: Annotated[
        TcpProxyConfig,
        Field(
            default_factory=lambda: TcpProxyConfig(
                frontend_port=5663, backend_port=5664
            ),
            description="Event bus proxy ports (XPUB/XSUB).",
        ),
    ]

    dataset_manager_proxy: Annotated[
        TcpProxyConfig,
        Field(
            default_factory=lambda: TcpProxyConfig(
                frontend_port=5661, backend_port=5662
            ),
            description="Dataset manager proxy ports (DEALER/ROUTER).",
        ),
    ]

    raw_inference_proxy: Annotated[
        TcpProxyConfig,
        Field(
            default_factory=lambda: TcpProxyConfig(
                frontend_port=5665, backend_port=5666
            ),
            description="Raw inference proxy ports (PUSH/PULL).",
        ),
    ]


class DualBindCommunicationConfig(BaseModel):
    """
    Dual-bind (IPC + TCP) communication configuration.

    For Kubernetes deployments where controller services connect via IPC
    (co-located in same pod) and worker pods connect via TCP.

    When controller_host is None, services use IPC (local mode).
    When controller_host is set, services use TCP to that host (remote mode).
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        Literal[CommunicationType.DUAL],
        Field(description="Communication type. Must be 'dual'."),
    ]

    ipc_path: Annotated[
        str,
        Field(
            default="/tmp/aiperf",
            description="Directory for IPC socket files (local services).",
        ),
    ]

    tcp_host: Annotated[
        str,
        Field(
            default="0.0.0.0",
            description="TCP bind host for proxies. "
            "Use 0.0.0.0 to listen on all interfaces.",
        ),
    ]

    controller_host: Annotated[
        str | None,
        Field(
            default=None,
            description="Controller host for remote TCP connections. "
            "When set, services connect via TCP to this host instead of IPC. "
            "In Kubernetes, set via JobSet DNS (e.g., controller.namespace.svc).",
        ),
    ]

    # Core communication ports
    records_port: Annotated[
        int,
        Field(
            default=5557,
            ge=1,
            le=65535,
            description="Port for records push/pull communication.",
        ),
    ]

    credit_router_port: Annotated[
        int,
        Field(
            default=5564,
            ge=1,
            le=65535,
            description="Port for credit router (ROUTER-DEALER).",
        ),
    ]

    # Proxy configurations (TCP ports, IPC uses path-based naming)
    event_bus_proxy: Annotated[
        TcpProxyConfig,
        Field(
            default_factory=lambda: TcpProxyConfig(
                frontend_port=5663, backend_port=5664
            ),
            description="Event bus proxy ports (XPUB/XSUB).",
        ),
    ]

    dataset_manager_proxy: Annotated[
        TcpProxyConfig,
        Field(
            default_factory=lambda: TcpProxyConfig(
                frontend_port=5661, backend_port=5662
            ),
            description="Dataset manager proxy ports (DEALER/ROUTER).",
        ),
    ]

    raw_inference_proxy: Annotated[
        TcpProxyConfig,
        Field(
            default_factory=lambda: TcpProxyConfig(
                frontend_port=5665, backend_port=5666
            ),
            description="Raw inference proxy ports (PUSH/PULL).",
        ),
    ]


# Union for communication configs using string discriminator
CommunicationConfig = Annotated[
    IpcCommunicationConfig | TcpCommunicationConfig | DualBindCommunicationConfig,
    Field(discriminator="type"),
]


class RuntimeConfig(BaseModel):
    """Runtime configuration for benchmark execution."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    ui: Annotated[
        UIType,
        Field(
            default=UIType.DASHBOARD,
            description="User interface mode. "
            "dashboard: rich interactive UI, "
            "simple: text progress, "
            "none: silent operation.",
        ),
    ]

    workers: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Maximum worker processes. "
            "null = auto-detect based on CPU cores.",
        ),
    ]

    record_processors: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Number of parallel record processors. "
            "null = auto-detect based on CPU cores.",
        ),
    ]

    communication: Annotated[
        CommunicationConfig | None,
        Field(
            default=None,
            description="Inter-process communication configuration. "
            "Defaults to IPC for single-machine operation.",
        ),
    ]


# =============================================================================
# SECTION 5: LOGGING CONFIGURATION
# =============================================================================


class LoggingConfig(BaseModel):
    """Logging configuration for verbosity and debug settings."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    level: Annotated[
        AIPerfLogLevel,
        Field(
            default=AIPerfLogLevel.INFO,
            description="Global logging verbosity level. "
            "trace: most verbose, error: least verbose.",
        ),
    ]

    config: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description="Raw Python logging.dictConfig configuration. "
            "Overrides all other logging settings when provided.",
        ),
    ]


# =============================================================================
# SECTION 6: MULTI-RUN CONFIGURATION
# =============================================================================


class MultiRunConfig(BaseModel):
    """Configuration for multi-run benchmarking with statistical reporting.

    When num_runs > 1, AIPerf executes multiple benchmark runs and computes
    aggregate statistics (mean, std, confidence intervals) across runs.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    num_runs: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            le=10,
            description="Number of profile runs to execute for confidence reporting. "
            "When 1, runs a single benchmark. "
            "When >1, computes aggregate statistics across runs.",
        ),
    ]

    cooldown_seconds: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            description="Cooldown duration in seconds between profile runs. "
            "Allows the system to stabilize between runs.",
        ),
    ]

    confidence_level: Annotated[
        float,
        Field(
            default=0.95,
            gt=0,
            lt=1,
            description="Confidence level for computing confidence intervals (0-1). "
            "Common values: 0.90 (90%%), 0.95 (95%%), 0.99 (99%%).",
        ),
    ]

    disable_warmup_after_first: Annotated[
        bool,
        Field(
            default=True,
            description="Disable warmup for runs after the first. "
            "When true, only the first run includes warmup for steady-state measurement.",
        ),
    ]

    set_consistent_seed: Annotated[
        bool,
        Field(
            default=True,
            description="Automatically set random seed for consistent workloads across runs. "
            "When true, sets random_seed=42 if not specified, ensuring identical "
            "workloads for valid statistical comparison.",
        ),
    ]

    @model_validator(mode="after")
    def validate_multi_run_params(self) -> MultiRunConfig:
        """Validate multi-run specific parameters are only meaningful when num_runs > 1."""
        if self.num_runs == 1:
            non_default = set()
            defaults = {
                "cooldown_seconds": 0.0,
                "confidence_level": 0.95,
                "disable_warmup_after_first": True,
                "set_consistent_seed": True,
            }
            for field_name, default_val in defaults.items():
                if field_name in self.model_fields_set:
                    val = getattr(self, field_name)
                    if val != default_val:
                        non_default.add(field_name)
            if non_default:
                raise ValueError(
                    f"Multi-run fields {non_default} are only meaningful when "
                    "num_runs > 1. Either increase num_runs or remove these fields."
                )
        return self


# =============================================================================
# SECTION 7: ACCURACY CONFIGURATION
# =============================================================================


class AccuracyConfig(BaseModel):
    """Configuration for accuracy benchmarking mode.

    When benchmark is set, enables accuracy evaluation alongside
    performance profiling using standard benchmarks (MMLU, AIME, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    benchmark: Annotated[
        str | None,
        Field(
            default=None,
            description="Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). "
            "When set, enables accuracy benchmarking mode.",
        ),
    ]

    tasks: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Specific tasks or subtasks within the benchmark to evaluate "
            "(e.g., specific MMLU subjects). If not set, all tasks are included.",
        ),
    ]

    n_shots: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            le=8,
            description="Number of few-shot examples to include in the prompt. "
            "0 means zero-shot evaluation. Maximum 8.",
        ),
    ]

    enable_cot: Annotated[
        bool,
        Field(
            default=False,
            description="Enable chain-of-thought prompting for accuracy evaluation.",
        ),
    ]

    grader: Annotated[
        str | None,
        Field(
            default=None,
            description="Override the default grader for the selected benchmark "
            "(e.g., exact_match, math, multiple_choice, code_execution).",
        ),
    ]

    system_prompt: Annotated[
        str | None,
        Field(
            default=None,
            description="Custom system prompt to use for accuracy evaluation. "
            "Overrides any benchmark-specific system prompt.",
        ),
    ]

    verbose: Annotated[
        bool,
        Field(
            default=False,
            description="Enable verbose output for accuracy evaluation, "
            "showing per-problem grading details.",
        ),
    ]
