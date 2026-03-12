# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Kubernetes environment configuration.

All settings can be configured via environment variables with the AIPERF_K8S_ prefix.
Resource settings per container type use AIPERF_K8S_{SERVICE}_{FIELD} naming.

Examples:
    AIPERF_K8S_CONTROLLER_CPU_REQUEST=800m
    AIPERF_K8S_WORKER_MEMORY_LIMIT=2Gi
    AIPERF_K8S_HEALTH_INITIAL_DELAY_SECONDS=10
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "K8sEnvironment",
]


# ---------------------------------------------------------------------------
# Resource settings: one base class, instances created via _resource_settings()
# ---------------------------------------------------------------------------


class ResourceSettings(BaseSettings):
    """Container resource settings (CPU/memory requests and limits)."""

    CPU_REQUEST: str = Field(description="CPU request")
    CPU_LIMIT: str = Field(description="CPU limit")
    MEMORY_REQUEST: str = Field(description="Memory request")
    MEMORY_LIMIT: str = Field(description="Memory limit")

    def to_k8s_resources(self) -> dict[str, dict[str, str]]:
        """Convert to Kubernetes resource spec."""
        return {
            "requests": {"cpu": self.CPU_REQUEST, "memory": self.MEMORY_REQUEST},
            "limits": {"cpu": self.CPU_LIMIT, "memory": self.MEMORY_LIMIT},
        }


def _resource_settings(
    env_prefix: str,
    cpu_request: str,
    cpu_limit: str,
    memory_request: str,
    memory_limit: str,
) -> ResourceSettings:
    """Create a ResourceSettings instance with the given env prefix and defaults.

    Each instance reads from AIPERF_K8S_{env_prefix}_{FIELD} environment
    variables, falling back to the provided defaults.
    """
    # Dynamically create a subclass with the correct env_prefix and defaults.
    # This replaces 8 nearly-identical boilerplate classes.
    cls = type(
        f"_{env_prefix.rstrip('_')}Settings",
        (ResourceSettings,),
        {
            "__annotations__": {
                "CPU_REQUEST": str,
                "CPU_LIMIT": str,
                "MEMORY_REQUEST": str,
                "MEMORY_LIMIT": str,
            },
            "model_config": SettingsConfigDict(env_prefix=f"AIPERF_K8S_{env_prefix}"),
            "CPU_REQUEST": Field(default=cpu_request, description="CPU request"),
            "CPU_LIMIT": Field(default=cpu_limit, description="CPU limit"),
            "MEMORY_REQUEST": Field(
                default=memory_request, description="Memory request"
            ),
            "MEMORY_LIMIT": Field(default=memory_limit, description="Memory limit"),
        },
    )
    return cls()


# ---------------------------------------------------------------------------
# Non-resource settings
# ---------------------------------------------------------------------------


class _HealthProbeSettings(BaseSettings):
    """Health probe configuration for all containers."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_HEALTH_")

    INITIAL_DELAY_SECONDS: int = Field(
        default=5,
        ge=0,
        le=300,
        description="Seconds before starting probes after container starts",
    )
    PERIOD_SECONDS: int = Field(
        default=10,
        ge=1,
        le=300,
        description="Interval in seconds between probe checks",
    )
    TIMEOUT_SECONDS: int = Field(
        default=5, ge=1, le=60, description="Seconds before probe times out"
    )
    FAILURE_THRESHOLD: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Consecutive failures before container is restarted/marked unready",
    )
    SUCCESS_THRESHOLD: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Consecutive successes before container is marked healthy",
    )


class _ZMQSettings(BaseSettings):
    """ZMQ communication settings for Kubernetes deployments."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_ZMQ_")

    CONTROLLER_HOST: str | None = Field(
        default=None,
        description="Controller hostname for ZMQ dual-bind mode. "
        "Set on worker pods to connect via TCP to controller. "
        "When None, services use IPC (controller mode).",
    )
    IPC_PATH: str = Field(
        default="/aiperf/ipc", description="Path for IPC socket files in pods"
    )


class _PortSettings(BaseSettings):
    """Container port assignments."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_PORT_")

    # Controller pod ports
    SYSTEM_CONTROLLER_HEALTH: int = Field(
        default=8080, ge=1, le=65535, description="System controller health port"
    )
    WORKER_MANAGER_HEALTH: int = Field(
        default=8081, ge=1, le=65535, description="Worker manager health port"
    )
    TIMING_MANAGER_HEALTH: int = Field(
        default=8082, ge=1, le=65535, description="Timing manager health port"
    )
    DATASET_MANAGER_HEALTH: int = Field(
        default=8083, ge=1, le=65535, description="Dataset manager health port"
    )
    RECORDS_MANAGER_HEALTH: int = Field(
        default=8084, ge=1, le=65535, description="Records manager health port"
    )
    API_SERVICE: int = Field(
        default=9090, ge=1, le=65535, description="API service port"
    )
    API_SERVICE_HEALTH: int = Field(
        default=8085, ge=1, le=65535, description="API service health port"
    )
    GPU_TELEMETRY_MANAGER_HEALTH: int = Field(
        default=8086, ge=1, le=65535, description="GPU telemetry manager health port"
    )
    SERVER_METRICS_MANAGER_HEALTH: int = Field(
        default=8087, ge=1, le=65535, description="Server metrics manager health port"
    )

    # Worker pod ports
    WORKER_HEALTH: int = Field(
        default=8080, ge=1, le=65535, description="Worker health port"
    )
    RECORD_PROCESSOR_HEALTH: int = Field(
        default=8081, ge=1, le=65535, description="Record processor health port"
    )


class _JobSetSettings(BaseSettings):
    """JobSet-level configuration."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_JOBSET_")

    TTL_SECONDS_AFTER_FINISHED: int | None = Field(
        default=300,
        ge=0,
        description="Seconds to keep JobSet after completion (None to disable)",
    )
    CONTROLLER_BACKOFF_LIMIT: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Job backoff limit for controller (0 = no retries)",
    )
    WORKER_BACKOFF_LIMIT: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Job backoff limit for workers (allows retries for transient failures)",
    )
    CONFIG_MOUNT_PATH: str = Field(
        default="/etc/aiperf", description="Path to mount ConfigMap with configs"
    )
    DATASETS_PATH: str = Field(
        default="/aiperf/datasets",
        description="Shared path for dataset files (dataset-manager writes, API serves)",
    )


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------


class _K8sEnvironment(BaseSettings):
    """Root Kubernetes environment configuration.

    Loads configuration from environment variables with the AIPERF_K8S_ prefix.
    Resource settings per container type are created via _resource_settings()
    with service-specific env prefixes and defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_K8S_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # Resource settings per container type
    # fmt: off
    CONTROLLER: ResourceSettings = Field(default_factory=lambda: _resource_settings("CONTROLLER_", "600m", "3000m", "1Gi", "4Gi"), description="Control-plane container resources")
    WORKER: ResourceSettings = Field(default_factory=lambda: _resource_settings("WORKER_", "350m", "500m", "1Gi", "2Gi"), description="Worker container resources")
    TIMING_MANAGER: ResourceSettings = Field(default_factory=lambda: _resource_settings("TIMING_MANAGER_", "500m", "500m", "512Mi", "512Mi"), description="Timing manager resources")
    DATASET_MANAGER: ResourceSettings = Field(default_factory=lambda: _resource_settings("DATASET_MANAGER_", "200m", "1000m", "512Mi", "1Gi"), description="Dataset manager resources")
    RECORDS_MANAGER: ResourceSettings = Field(default_factory=lambda: _resource_settings("RECORDS_MANAGER_", "600m", "3000m", "896Mi", "2304Mi"), description="Records manager resources")
    RECORD_PROCESSOR: ResourceSettings = Field(default_factory=lambda: _resource_settings("RECORD_PROCESSOR_", "100m", "250m", "512Mi", "1Gi"), description="Record processor resources")
    GPU_TELEMETRY_MANAGER: ResourceSettings = Field(default_factory=lambda: _resource_settings("GPU_TELEMETRY_MANAGER_", "100m", "500m", "256Mi", "512Mi"), description="GPU telemetry manager resources")
    SERVER_METRICS_MANAGER: ResourceSettings = Field(default_factory=lambda: _resource_settings("SERVER_METRICS_MANAGER_", "100m", "500m", "256Mi", "512Mi"), description="Server metrics manager resources")
    # fmt: on

    # Non-resource settings
    HEALTH: _HealthProbeSettings = Field(
        default_factory=_HealthProbeSettings,
        description="Health probe configuration",
    )
    PORTS: _PortSettings = Field(
        default_factory=_PortSettings,
        description="Container port assignments",
    )
    ZMQ: _ZMQSettings = Field(
        default_factory=_ZMQSettings,
        description="ZMQ communication settings",
    )
    JOBSET: _JobSetSettings = Field(
        default_factory=_JobSetSettings,
        description="JobSet-level configuration",
    )


# Global singleton instance
K8sEnvironment = _K8sEnvironment()
