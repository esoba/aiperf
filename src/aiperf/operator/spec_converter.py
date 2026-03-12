# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert AIPerfJob CRD spec to AIPerf configuration objects.

This module provides the translation layer between Kubernetes-native AIPerfJob
custom resource specs and AIPerf's AIPerfConfig model. Legacy UserConfig/ServiceConfig
are produced via the reverse converter for backward compatibility with services.
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Any

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.config.config import AIPerfConfig
from aiperf.config.reverse_converter import convert_to_legacy_configs
from aiperf.kubernetes.jobset import PodCustomization, SecretMount

# Default connections per worker for auto-scaling calculation
DEFAULT_CONNECTIONS_PER_WORKER = 500

# AIPerfConfig fields that can appear in the CRD spec
CONFIG_FIELDS = {
    "models",
    "endpoint",
    "datasets",
    "load",
    "artifacts",
    "slos",
    "tokenizer",
    "gpu_telemetry",
    "server_metrics",
    "runtime",
    "logging",
    "multi_run",
    "accuracy",
    "random_seed",
}


@dataclass(slots=True)
class AIPerfJobSpecConverter:
    """Converts AIPerfJob CRD spec to AIPerf configuration objects.

    The CRD spec directly maps to AIPerfConfig schema. The spec fields
    (models, endpoint, datasets, load, etc.) are passed to AIPerfConfig
    for validation. Legacy UserConfig/ServiceConfig are produced via
    the reverse converter for backward compatibility.

    Example:
        >>> converter = AIPerfJobSpecConverter(spec, "my-job", "default")
        >>> config = converter.to_aiperf_config()
        >>> user_config, service_config = converter.to_legacy_configs()

    Attributes:
        spec: The AIPerfJob CR spec dict.
        name: The AIPerfJob CR name.
        namespace: The Kubernetes namespace.
        job_id: The unique job ID for this benchmark run (defaults to name).
    """

    spec: dict[str, Any]
    name: str
    namespace: str
    job_id: str | None = field(default=None)

    def __post_init__(self) -> None:
        """Set job_id to name if not explicitly provided."""
        if self.job_id is None:
            self.job_id = self.name

    def to_aiperf_config(self) -> AIPerfConfig:
        """Convert AIPerfJob spec to AIPerfConfig.

        The CRD spec contains the AIPerfConfig fields directly (models,
        endpoint, datasets, load, etc.). We extract them and validate
        via AIPerfConfig.model_validate().

        Returns:
            AIPerfConfig populated from the AIPerfJob spec.
        """
        config_dict = copy.deepcopy(self._get_config_dict())

        # Set cli_command for traceability
        config_dict.setdefault("artifacts", {})
        config_dict["artifacts"].setdefault(
            "cli_command", f"kubectl apply -f aiperfjob/{self.name}"
        )

        # Set output directory for results
        config_dict["artifacts"].setdefault("dir", "/results")

        return AIPerfConfig.model_validate(config_dict)

    def to_legacy_configs(self) -> tuple[UserConfig, ServiceConfig]:
        """Convert AIPerfJob spec to legacy UserConfig and ServiceConfig.

        Uses the reverse converter to bridge AIPerfConfig to the legacy
        config objects that services still expect.

        Returns:
            Tuple of (UserConfig, ServiceConfig).
        """
        config = self.to_aiperf_config()
        return convert_to_legacy_configs(config)

    def to_pod_customization(self) -> PodCustomization:
        """Convert podTemplate spec to PodCustomization.

        Returns:
            PodCustomization with node placement, secrets, and metadata.
        """
        pod_spec = self.spec.get("podTemplate", {})

        # Convert env list to dict formats
        env_list = pod_spec.get("env", [])
        env_vars = self._convert_env_vars(env_list)
        env_from_secrets = self._convert_env_from_secrets(env_list)

        # Convert volume mounts to secret mounts where applicable
        secret_mounts = self._convert_secret_mounts(
            pod_spec.get("volumes", []),
            pod_spec.get("volumeMounts", []),
        )

        return PodCustomization(
            node_selector=pod_spec.get("nodeSelector", {}),
            tolerations=pod_spec.get("tolerations", []),
            annotations=pod_spec.get("annotations", {}),
            labels=pod_spec.get("labels", {}),
            image_pull_secrets=pod_spec.get("imagePullSecrets", []),
            env_vars=env_vars,
            env_from_secrets=env_from_secrets,
            secret_mounts=secret_mounts,
            service_account=pod_spec.get("serviceAccountName"),
        )

    def to_scheduling_config(self) -> dict[str, str | None]:
        """Extract Kueue scheduling configuration from CRD spec.

        Returns:
            Dict with queue_name and priority_class keys.
        """
        scheduling = self.spec.get("scheduling", {})
        return {
            "queue_name": scheduling.get("queueName"),
            "priority_class": scheduling.get("priorityClass"),
        }

    def calculate_workers(self) -> int:
        """Calculate optimal worker count based on concurrency.

        Uses the formula: workers = ceil(concurrency / connectionsPerWorker)

        Returns:
            Number of worker pods needed.
        """
        concurrency = 1
        config = self.to_aiperf_config()
        for phase in config.load.values():
            if not phase.exclude and phase.concurrency is not None:
                concurrency = phase.concurrency
                break

        connections_per_worker = self.spec.get(
            "connectionsPerWorker", DEFAULT_CONNECTIONS_PER_WORKER
        )

        return max(1, math.ceil(concurrency / connections_per_worker))

    def _get_config_dict(self) -> dict[str, Any]:
        """Extract the AIPerfConfig-compatible dict from the CRD spec.

        The CRD spec contains AIPerfConfig fields directly at the top level.
        Non-config fields (image, podTemplate, scheduling, etc.) are excluded.

        Returns:
            Dict suitable for AIPerfConfig.model_validate().
        """
        return {k: v for k, v in self.spec.items() if k in CONFIG_FIELDS}

    def _convert_env_vars(self, env_list: list[dict[str, Any]]) -> dict[str, str]:
        """Convert Kubernetes env list to direct env vars dict.

        Args:
            env_list: List of Kubernetes EnvVar objects.

        Returns:
            Dict of env var name to value (only direct values).
        """
        return {e["name"]: e["value"] for e in env_list if "value" in e}

    def _convert_env_from_secrets(
        self, env_list: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Convert Kubernetes env list to secret references.

        Args:
            env_list: List of Kubernetes EnvVar objects.

        Returns:
            Dict of env var name to "secretName/key" format.
        """
        result = {}
        for e in env_list:
            if "valueFrom" in e and "secretKeyRef" in e["valueFrom"]:
                ref = e["valueFrom"]["secretKeyRef"]
                result[e["name"]] = f"{ref['name']}/{ref['key']}"
        return result

    def _convert_secret_mounts(
        self,
        volumes: list[dict[str, Any]],
        volume_mounts: list[dict[str, Any]],
    ) -> list[SecretMount]:
        """Convert volume/volumeMount pairs to SecretMount objects.

        Only converts volumes that are secret-backed.

        Args:
            volumes: List of Kubernetes Volume objects.
            volume_mounts: List of Kubernetes VolumeMount objects.

        Returns:
            List of SecretMount objects for secret-backed volumes.
        """
        # Build a map of volume name to secret name
        secret_volumes: dict[str, str] = {}
        for vol in volumes:
            if "secret" in vol:
                secret_volumes[vol["name"]] = vol["secret"]["secretName"]

        # Find mounts that reference secret volumes
        secret_mounts = []
        for mount in volume_mounts:
            vol_name = mount.get("name")
            if vol_name in secret_volumes:
                secret_mounts.append(
                    SecretMount(
                        name=secret_volumes[vol_name],
                        mount_path=mount["mountPath"],
                        sub_path=mount.get("subPath"),
                    )
                )

        return secret_mounts
