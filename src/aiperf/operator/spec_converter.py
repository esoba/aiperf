# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert AIPerfJob CRD spec to AIPerf configuration objects.

This module provides the translation layer between Kubernetes-native AIPerfJob
custom resource specs and AIPerf's internal UserConfig/ServiceConfig models.
"""

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.config.zmq_config import ZMQDualBindConfig
from aiperf.common.enums import AIPerfLogLevel
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import PodCustomization, SecretMount
from aiperf.plugin.enums import ServiceRunType, UIType

# Default connections per worker for auto-scaling calculation
DEFAULT_CONNECTIONS_PER_WORKER = 500


@dataclass(slots=True)
class AIPerfJobSpecConverter:
    """Converts AIPerfJob CRD spec to AIPerf configuration objects.

    This class translates the Kubernetes-native AIPerfJob spec into
    UserConfig and ServiceConfig objects that AIPerf services expect.

    The CRD spec.userConfig maps directly to UserConfig schema, so we
    use Pydantic's model_validate() for direct validation. This ensures
    exclude_unset works correctly since only fields in the YAML are set.

    Example:
        >>> converter = AIPerfJobSpecConverter(spec, "my-job", "default")
        >>> user_config = converter.to_user_config()
        >>> service_config = converter.to_service_config()

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

    def to_user_config(self) -> UserConfig:
        """Convert AIPerfJob spec to UserConfig.

        The spec.userConfig is passed directly to UserConfig.model_validate(),
        which means only fields present in the YAML are set on the model.
        This ensures exclude_unset=True works correctly when serializing.

        Returns:
            UserConfig populated from the AIPerfJob spec.
        """
        user_config_dict = copy.deepcopy(self.spec.get("userConfig", {}))

        # Set cli_command for traceability
        user_config_dict.setdefault(
            "cli_command", f"kubectl apply -f aiperfjob/{self.name}"
        )

        # Set output directory for results
        user_config_dict.setdefault("output", {})
        user_config_dict["output"].setdefault("artifact_directory", "/results")

        return UserConfig.model_validate(user_config_dict)

    def to_service_config(self) -> ServiceConfig:
        """Convert AIPerfJob spec to ServiceConfig for Kubernetes deployment.

        Returns:
            ServiceConfig configured for Kubernetes with ZMQ dual-bind.

        The ZMQDualBindConfig enables:
        - Controller pod services: use IPC (fast, same pod communication)
        - Worker pods: use TCP to connect to controller (via AIPERF_K8S_ZMQ_CONTROLLER_HOST env var)

        The controller_host is NOT set here - it's read from the AIPERF_K8S_ZMQ_CONTROLLER_HOST
        environment variable at runtime by ServiceConfig. This allows the same config file
        to be used by both controller and worker pods.

        Setting service_run_type=KUBERNETES tells SystemController to:
        - Spawn control-plane services (dataset_manager, timing_manager, etc.) as subprocesses
        - Treat workers and record processors as external Kubernetes pods managed by JobSet

        This enables the control-plane to run as a single container that loads, runs,
        and fails as a single unit.
        """
        # Create dual-bind config for Kubernetes deployment
        # controller_host=None means controller mode (IPC)
        # Workers will have AIPERF_K8S_ZMQ_CONTROLLER_HOST env var set by the JobSet,
        # which ServiceConfig reads to switch to TCP mode
        #
        # ipc_path uses K8sEnvironment.ZMQ.IPC_PATH (default: /aiperf/ipc) because
        # all containers in the controller pod share the 'ipc' emptyDir volume mounted
        # at this path. This allows IPC socket files to be shared between containers.
        #
        zmq_config = ZMQDualBindConfig(
            ipc_path=Path(K8sEnvironment.ZMQ.IPC_PATH), tcp_host="0.0.0.0"
        )

        # Build controller DNS for dataset API URL
        # Workers download datasets via HTTP from the API service on the controller pod
        # JobSet name is "aiperf-{job_id}" (see resources.py:jobset_name)
        # DNS format: {pod-name}.{headless-service}.{namespace}.svc.cluster.local
        api_port = K8sEnvironment.PORTS.API_SERVICE
        jobset_name = f"aiperf-{self.job_id}"
        controller_dns = f"{jobset_name}-controller-0-0.{jobset_name}.{self.namespace}.svc.cluster.local"
        dataset_api_base_url = f"http://{controller_dns}:{api_port}/api/dataset"

        # Use model_construct() to bypass ServiceConfig's nested ZMQDualBindConfig
        # validation (which would try to mkdir the ipc_path locally).
        # Explicitly set _fields_set so exclude_unset=True works during serialization.
        return ServiceConfig.model_construct(
            _fields_set={
                "zmq_dual",
                "ui_type",
                "service_run_type",
                "api_port",
                "api_host",
                "dataset_api_base_url",
                "log_level",
            },
            zmq_dual=zmq_config,
            ui_type=UIType.SIMPLE,
            service_run_type=ServiceRunType.KUBERNETES,
            api_port=api_port,
            api_host="0.0.0.0",  # Bind to all interfaces for pod networking
            dataset_api_base_url=dataset_api_base_url,
            log_level=AIPerfLogLevel.INFO,  # Production default; override via service config
        )

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
        # Get concurrency from userConfig.loadgen
        user_config = self.spec.get("userConfig", {})
        loadgen = user_config.get("loadgen", {})
        concurrency = loadgen.get("concurrency", 1)

        connections_per_worker = self.spec.get(
            "connectionsPerWorker", DEFAULT_CONNECTIONS_PER_WORKER
        )

        return max(1, math.ceil(concurrency / connections_per_worker))

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
