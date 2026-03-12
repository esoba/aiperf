# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JobSet specification generation for Kubernetes deployments.

This module generates JobSet YAML for deploying AIPerf as a distributed
benchmark across multiple pods. All resource and port settings are configurable
via environment variables through K8sEnvironment.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from aiperf.common.models import AIPerfBaseModel
from aiperf.kubernetes.constants import Containers, KueueLabels, Labels
from aiperf.kubernetes.enums import ImagePullPolicy, RestartPolicy
from aiperf.kubernetes.environment import K8sEnvironment, ResourceSettings
from aiperf.kubernetes.utils import parse_cpu, parse_memory_mib

# Control-plane services and their K8sEnvironment resource settings.
# When adding a new control-plane service, add its resource config here.
# Services without dedicated settings use CONTROLLER as a baseline.
CONTROL_PLANE_RESOURCE_MAP: list[tuple[str, ResourceSettings]] = [
    ("system_controller", K8sEnvironment.CONTROLLER),
    ("worker_manager", K8sEnvironment.CONTROLLER),
    ("timing_manager", K8sEnvironment.TIMING_MANAGER),
    ("dataset_manager", K8sEnvironment.DATASET_MANAGER),
    ("records_manager", K8sEnvironment.RECORDS_MANAGER),
    ("api", K8sEnvironment.CONTROLLER),
    ("gpu_telemetry_manager", K8sEnvironment.GPU_TELEMETRY_MANAGER),
    ("server_metrics_manager", K8sEnvironment.SERVER_METRICS_MANAGER),
]

# WorkerPodManager overhead: the manager process itself consumes resources
# beyond the workers/record-processors it spawns as subprocesses.
_WPM_CPU_REQUEST = 0.1  # 100m
_WPM_CPU_LIMIT = 0.2  # 200m
_WPM_MEMORY_REQUEST_MIB = 128
_WPM_MEMORY_LIMIT_MIB = 256


def _format_k8s_resource_totals(
    cpu_request: float,
    cpu_limit: float,
    memory_request_mib: int,
    memory_limit_mib: int,
) -> dict[str, dict[str, str]]:
    """Format aggregated resource totals into a K8s resource spec dict.

    Args:
        cpu_request: Total CPU request in cores.
        cpu_limit: Total CPU limit in cores.
        memory_request_mib: Total memory request in MiB.
        memory_limit_mib: Total memory limit in MiB.

    Returns:
        Kubernetes-style resource dict with requests and limits.
    """
    return {
        "requests": {
            "cpu": f"{int(cpu_request * 1000)}m",
            "memory": f"{memory_request_mib}Mi",
        },
        "limits": {
            "cpu": f"{int(cpu_limit * 1000)}m",
            "memory": f"{memory_limit_mib}Mi",
        },
    }


@dataclass(frozen=True, slots=True)
class JobSetAPIConfig:
    """JobSet API configuration constants."""

    group: str = "jobset.x-k8s.io"
    version: str = "v1alpha2"
    plural: str = "jobsets"

    @property
    def api_version(self) -> str:
        """Get the full apiVersion string for manifests."""
        return f"{self.group}/{self.version}"


# Shared JobSet API constants
JOBSET_API = JobSetAPIConfig()

# Known-good fallback version for JobSet CRD installation
JOBSET_FALLBACK_VERSION = "v0.5.2"
JOBSET_GITHUB_REPO = "kubernetes-sigs/jobset"


def get_jobset_manifest_url(version: str | None = None) -> str:
    """Build the JobSet manifest URL for a given version.

    Args:
        version: JobSet release tag (e.g. "v0.5.2"). If None, uses the fallback.

    Returns:
        URL to the JobSet manifests.yaml for kubectl apply.
    """
    v = version or JOBSET_FALLBACK_VERSION
    return (
        f"https://github.com/{JOBSET_GITHUB_REPO}/releases/download/{v}/manifests.yaml"
    )


async def get_latest_jobset_version() -> str | None:
    """Query GitHub API for the latest JobSet release tag.

    Returns:
        Latest release tag (e.g. "v0.7.1"), or None if the lookup fails.
    """
    import aiohttp
    import orjson

    from aiperf.transports.aiohttp_client import create_tcp_connector

    url = f"https://api.github.com/repos/{JOBSET_GITHUB_REPO}/releases/latest"
    headers = {"Accept": "application/vnd.github+json"}
    try:
        connector = create_tcp_connector()
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5), connector=connector
            ) as session,
            session.get(url, headers=headers) as resp,
        ):
            data = orjson.loads(await resp.read())
            tag = data.get("tag_name")
            return tag if isinstance(tag, str) else None
    except (aiohttp.ClientError, orjson.JSONDecodeError, TimeoutError):
        return None


def get_jobset_install_hint(version: str | None = None) -> str:
    """Get a user-facing hint for installing JobSet CRD.

    Args:
        version: Specific version tag, or None for fallback.

    Returns:
        Formatted install command string.
    """
    url = get_jobset_manifest_url(version)
    return f"Install JobSet: kubectl apply --server-side -f {url}"


def controller_dns_name(jobset_name: str, namespace: str) -> str:
    """Build the controller pod DNS hostname for a JobSet.

    JobSet with enableDNSHostnames creates a headless service with the same name
    as the JobSet, and pods get DNS names like:
    {jobset-name}-{job-name}-{job-index}-{pod-index}.{jobset-name}.{namespace}.svc.cluster.local

    Since we have exactly 1 controller replica with 1 pod, indices are always 0-0.

    Args:
        jobset_name: The JobSet resource name.
        namespace: Kubernetes namespace.

    Returns:
        Fully qualified DNS hostname for the controller pod.
    """
    return f"{jobset_name}-controller-0-0.{jobset_name}.{namespace}.svc.cluster.local"


class SecretMount(AIPerfBaseModel):
    """Configuration for mounting a Kubernetes secret as a volume."""

    name: str = Field(description="Secret name in Kubernetes")
    mount_path: str = Field(description="Path to mount the secret")
    sub_path: str | None = Field(
        default=None, description="Specific key to mount (optional)"
    )


class PodCustomization(AIPerfBaseModel):
    """Customization options for pod specifications.

    This allows users to customize node placement, add secrets, set environment
    variables, and add metadata to the generated Kubernetes resources.
    """

    # Node placement
    node_selector: dict[str, str] = Field(
        default_factory=dict,
        description="Node selector labels (e.g., {'nvidia.com/gpu': 'true'})",
    )
    tolerations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Pod tolerations for scheduling on tainted nodes",
    )

    # Metadata
    annotations: dict[str, str] = Field(
        default_factory=dict, description="Additional pod annotations"
    )
    labels: dict[str, str] = Field(
        default_factory=dict, description="Additional pod labels"
    )

    # Secrets and environment
    image_pull_secrets: list[str] = Field(
        default_factory=list, description="Image pull secret names"
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Additional environment variables (e.g., {'HF_TOKEN': 'xxx'})",
    )
    env_from_secrets: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables from secrets (env_name: secret_name/key)",
    )
    secret_mounts: list[SecretMount] = Field(
        default_factory=list, description="Secrets to mount as volumes"
    )

    # Service account
    service_account: str | None = Field(
        default=None, description="Service account name (uses 'default' if not set)"
    )

    def get_env_vars(self) -> list[dict[str, Any]]:
        """Get environment variables as Kubernetes env spec."""
        direct = [
            {"name": name, "value": value} for name, value in self.env_vars.items()
        ]
        secrets = [
            {
                "name": env_name,
                "valueFrom": {
                    "secretKeyRef": {
                        "name": secret_ref.split("/", 1)[0]
                        if "/" in secret_ref
                        else secret_ref,
                        "key": secret_ref.split("/", 1)[1]
                        if "/" in secret_ref
                        else env_name,
                    },
                },
            }
            for env_name, secret_ref in self.env_from_secrets.items()
        ]
        return direct + secrets

    def get_volumes(self) -> list[dict[str, Any]]:
        """Get volume definitions for secret mounts."""
        return [
            {
                "name": f"secret-{secret.name}",
                "secret": {"secretName": secret.name},
            }
            for secret in self.secret_mounts
        ]

    def get_volume_mounts(self) -> list[dict[str, Any]]:
        """Get volume mounts for secrets."""
        return [
            {
                "name": f"secret-{secret.name}",
                "mountPath": secret.mount_path,
                "readOnly": True,
                **({"subPath": secret.sub_path} if secret.sub_path else {}),
            }
            for secret in self.secret_mounts
        ]


class ContainerSpec(AIPerfBaseModel):
    """Specification for a container within a pod."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    name: str = Field(description="Container name")
    image: str = Field(description="Container image")
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None,
        description="Image pull policy (Always, Never, IfNotPresent). "
        "Defaults to Always for :latest tags, IfNotPresent otherwise.",
    )
    command: list[str] = Field(default_factory=list, description="Command to run")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: list[dict[str, Any]] = Field(
        default_factory=list, description="Environment variables"
    )
    resources: dict[str, dict[str, str]] = Field(
        default_factory=dict, description="Resource requests and limits"
    )
    volume_mounts: list[dict[str, Any]] = Field(
        default_factory=list, description="Volume mounts"
    )
    ports: list[dict[str, Any]] = Field(
        default_factory=list, description="Container ports"
    )
    startup_probe: dict[str, Any] | None = Field(
        default=None, description="Startup probe configuration"
    )
    liveness_probe: dict[str, Any] | None = Field(
        default=None, description="Liveness probe configuration"
    )
    readiness_probe: dict[str, Any] | None = Field(
        default=None, description="Readiness probe configuration"
    )
    security_context: dict[str, Any] | None = Field(
        default=None, description="Container security context"
    )

    def to_k8s_spec(self) -> dict[str, Any]:
        """Convert to Kubernetes container spec."""
        return self.model_dump(
            by_alias=True, exclude_unset=True, exclude_none=True, mode="json"
        )


class ReplicatedJobSpec(AIPerfBaseModel):
    """Specification for a replicated job within a JobSet."""

    name: str = Field(description="Replicated job name")
    replicas: int = Field(default=1, description="Number of replicas")
    containers: list[ContainerSpec] = Field(
        default_factory=list, description="Containers in the pod"
    )
    volumes: list[dict[str, Any]] = Field(
        default_factory=list, description="Pod volumes"
    )
    restart_policy: RestartPolicy = Field(
        default=RestartPolicy.ON_FAILURE, description="Pod restart policy"
    )
    backoff_limit: int = Field(default=0, description="Job backoff limit for retries")
    pod_customization: PodCustomization | None = Field(
        default=None, description="Pod customization options"
    )
    job_id: str | None = Field(default=None, description="Job ID for pod labeling")

    def to_k8s_spec(self) -> dict[str, Any]:
        """Convert to Kubernetes replicatedJob spec."""
        pod_spec: dict[str, Any] = {
            "restartPolicy": str(self.restart_policy),
            "containers": [c.to_k8s_spec() for c in self.containers],
            "volumes": self.volumes,
            # Pod-level security context
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 1000,
                "runAsGroup": 1000,
                "fsGroup": 1000,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
        }

        # Apply pod customizations
        if self.pod_customization:
            custom = self.pod_customization
            if custom.node_selector:
                pod_spec["nodeSelector"] = custom.node_selector
            if custom.tolerations:
                pod_spec["tolerations"] = custom.tolerations
            if custom.image_pull_secrets:
                pod_spec["imagePullSecrets"] = [
                    {"name": name} for name in custom.image_pull_secrets
                ]
            if custom.service_account:
                pod_spec["serviceAccountName"] = custom.service_account

        # Build metadata with annotations and labels
        pod_metadata: dict[str, Any] = {}
        if self.pod_customization and self.pod_customization.annotations:
            pod_metadata["annotations"] = self.pod_customization.annotations

        # Build pod labels: base AIPerf labels + custom labels
        pod_labels: dict[str, str] = {Labels.APP_KEY: Labels.APP_VALUE}
        if self.job_id:
            pod_labels[Labels.JOB_ID] = self.job_id
        if self.pod_customization and self.pod_customization.labels:
            pod_labels.update(self.pod_customization.labels)
        pod_metadata["labels"] = pod_labels

        pod_template: dict[str, Any] = {"spec": pod_spec}
        if pod_metadata:
            pod_template["metadata"] = pod_metadata

        return {
            "name": self.name,
            "replicas": self.replicas,
            "template": {
                "spec": {
                    "parallelism": 1,
                    "completions": 1,
                    "completionMode": "Indexed",
                    "backoffLimit": self.backoff_limit,
                    "template": pod_template,
                }
            },
        }


class JobSetSpec(AIPerfBaseModel):
    """Specification for a complete JobSet deployment.

    Resource settings, ports, and health probe configuration are loaded from
    K8sEnvironment and can be customized via AIPERF_K8S_* environment variables.
    """

    name: str = Field(description="JobSet name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    job_id: str = Field(description="Unique benchmark job ID")
    image: str = Field(description="AIPerf container image")
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None,
        description="Image pull policy for all containers (Always, Never, IfNotPresent). "
        "Set to 'Never' for local development with minikube.",
    )
    worker_replicas: int = Field(default=1, description="Number of worker pods")
    workers_per_pod: int | None = Field(
        default=None,
        description="Actual workers per pod (used for resource calculation). "
        "Defaults to Environment.WORKER.DEFAULT_WORKERS_PER_POD if not set.",
    )
    ttl_seconds: int | None = Field(
        default=None, description="TTL after finished (uses K8sEnvironment default)"
    )

    # Pod customization
    pod_customization: PodCustomization = Field(
        default_factory=PodCustomization, description="Pod customization options"
    )

    # Optional metadata for discovery
    name_label: str | None = Field(
        default=None, description="Human-readable name label for the JobSet"
    )
    extra_annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Additional annotations for the JobSet metadata",
    )

    # Kueue gang-scheduling
    queue_name: str | None = Field(
        default=None,
        description="Kueue LocalQueue name. When set, enables gang-scheduling via Kueue admission.",
    )
    priority_class: str | None = Field(
        default=None,
        description="Kueue WorkloadPriorityClass name for scheduling priority",
    )

    def _create_security_context(self) -> dict[str, Any]:
        """Create a security context for containers.

        Applies security best practices:
        - Run as non-root user
        - Drop all capabilities
        - Read-only root filesystem (writable emptyDir volumes for data/ipc/results)
        """
        return {
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "runAsGroup": 1000,
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]},
            "seccompProfile": {"type": "RuntimeDefault"},
        }

    def _create_health_probe(self, port: int, path: str = "/healthz") -> dict[str, Any]:
        """Create a health probe configuration from K8sEnvironment settings."""
        health = K8sEnvironment.HEALTH
        return {
            "httpGet": {"path": path, "port": port},
            "initialDelaySeconds": health.INITIAL_DELAY_SECONDS,
            "periodSeconds": health.PERIOD_SECONDS,
            "timeoutSeconds": health.TIMEOUT_SECONDS,
            "failureThreshold": health.FAILURE_THRESHOLD,
            "successThreshold": health.SUCCESS_THRESHOLD,
        }

    def _create_startup_probe(
        self, port: int, path: str = "/healthz"
    ) -> dict[str, Any]:
        """Create a startup probe for slow-starting containers.

        Startup probes allow containers more time to initialize before
        liveness/readiness probes take over. Uses more lenient settings
        to accommodate initialization time.
        """
        return {
            "httpGet": {"path": path, "port": port},
            "initialDelaySeconds": 0,
            "periodSeconds": 5,
            "timeoutSeconds": 5,
            "failureThreshold": 30,  # 30 * 5s = 150s max startup time
        }

    def _create_env_vars(
        self, controller_host: str | None = None
    ) -> list[dict[str, Any]]:
        """Create environment variables for a container."""
        jobset_config = K8sEnvironment.JOBSET
        config_path = jobset_config.CONFIG_MOUNT_PATH
        datasets_path = jobset_config.DATASETS_PATH
        env: list[dict[str, Any]] = [
            {
                "name": "AIPERF_CONFIG_USER_FILE",
                "value": f"{config_path}/user_config.json",
            },
            {
                "name": "AIPERF_CONFIG_SERVICE_FILE",
                "value": f"{config_path}/service_config.json",
            },
            # Shared dataset path: dataset-manager writes mmap files here,
            # API service serves them to workers via HTTP
            {"name": "AIPERF_DATASET_MMAP_BASE_PATH", "value": datasets_path},
            # Job ID and namespace for the benchmark
            {"name": "AIPERF_JOB_ID", "value": self.job_id},
            {"name": "AIPERF_NAMESPACE", "value": self.namespace},
            # Health server must bind to 0.0.0.0 so K8s probes can reach it via the pod IP
            {"name": "AIPERF_SERVICE_HEALTH_ENABLED", "value": "true"},
            {"name": "AIPERF_SERVICE_HEALTH_HOST", "value": "0.0.0.0"},
            # HF cache must be writable (readOnlyRootFilesystem)
            {"name": "HF_HOME", "value": "/tmp/hf_home"},
            # Expose the JobSet job-index as a unique pod identifier.
            # JOB_COMPLETION_INDEX is always 0 because each replicated job has
            # completions=1; the JobSet job-index label is the true replica index.
            {
                "name": "AIPERF_POD_INDEX",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.labels['jobset.sigs.k8s.io/job-index']",
                    }
                },
            },
        ]

        if controller_host:
            env.append(
                {"name": "AIPERF_K8S_ZMQ_CONTROLLER_HOST", "value": controller_host}
            )

        # Add custom environment variables from pod customization
        env.extend(self.pod_customization.get_env_vars())
        return env

    def _get_volume_mounts(self) -> list[dict[str, Any]]:
        """Get all volume mounts including config and IPC."""
        config_path = K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH
        ipc_path = K8sEnvironment.ZMQ.IPC_PATH
        datasets_path = K8sEnvironment.JOBSET.DATASETS_PATH
        mounts: list[dict[str, Any]] = [
            {"name": "config", "mountPath": config_path, "readOnly": True},
            {"name": "ipc", "mountPath": ipc_path},
            {"name": "results", "mountPath": "/results"},
            # Shared dataset volume: dataset-manager writes, API serves to workers
            {"name": "datasets", "mountPath": datasets_path},
            {"name": "tmp", "mountPath": "/tmp"},
        ]
        mounts.extend(self.pod_customization.get_volume_mounts())
        return mounts

    def _create_container(
        self,
        name: str,
        service_type: str,
        health_port: int,
        resources: dict[str, dict[str, str]],
        api_port: int | None = None,
        controller_host: str | None = None,
        extra_env: list[dict[str, Any]] | None = None,
        skip_readiness_probe: bool = False,
    ) -> ContainerSpec:
        """Create a container spec with standard AIPerf configuration.

        Args:
            name: Container name.
            service_type: AIPerf service type(s), comma-separated.
            health_port: Health check port.
            resources: Kubernetes resource requests/limits.
            api_port: Optional API port for services that expose APIs.
            controller_host: Controller DNS for worker containers.
            extra_env: Additional environment variables for this container.
            skip_readiness_probe: If True, don't add a readiness probe.
        """
        args = ["service", "--type", service_type, "--health-port", str(health_port)]
        if api_port:
            args.extend(["--api-port", str(api_port)])

        ports: list[dict[str, Any]] = [{"containerPort": health_port, "name": "health"}]
        if api_port:
            ports.append({"containerPort": api_port, "name": "api"})

        env = self._create_env_vars(controller_host=controller_host)
        if extra_env:
            env.extend(extra_env)

        # Configure probes - startup probe allows slow initialization,
        # then liveness/readiness take over for ongoing health monitoring
        startup_probe = self._create_startup_probe(health_port)
        liveness_probe = self._create_health_probe(health_port)
        readiness_probe = (
            None
            if skip_readiness_probe
            else self._create_health_probe(health_port, path="/readyz")
        )

        return ContainerSpec(
            name=name,
            image=self.image,
            image_pull_policy=self.image_pull_policy,
            command=["aiperf"],
            args=args,
            env=env,
            resources=resources,
            volume_mounts=self._get_volume_mounts(),
            ports=ports,
            startup_probe=startup_probe,
            liveness_probe=liveness_probe,
            readiness_probe=readiness_probe,
            security_context=self._create_security_context(),
        )

    def _create_controller_containers(self) -> list[ContainerSpec]:
        """Create a single container for the control-plane pod.

        The control-plane runs as a single container where SystemController
        spawns all other control-plane services as subprocesses. This enables
        the control-plane to load, run, and fail as a single unit.

        Services spawned as subprocesses:
        - worker_manager, timing_manager, dataset_manager, records_manager
        - api, gpu_telemetry_manager, server_metrics_manager

        Workers and RecordProcessors are external pods managed by JobSet.
        """
        ports = K8sEnvironment.PORTS

        # Aggregate resource requests/limits for all control-plane services
        # Each subprocess needs resources, so sum them up
        control_plane_resources = self._aggregate_control_plane_resources()

        return [
            self._create_container(
                name=Containers.CONTROL_PLANE,
                service_type="system_controller",
                health_port=ports.SYSTEM_CONTROLLER_HEALTH,
                resources=control_plane_resources,
                api_port=ports.API_SERVICE,  # API is spawned as subprocess but needs port exposed
                skip_readiness_probe=True,  # System controller manages its own lifecycle
                # Enable realtime metrics since we don't use DASHBOARD UI
                extra_env=[
                    {"name": "AIPERF_UI_REALTIME_METRICS_ENABLED", "value": "true"}
                ],
            ),
        ]

    def _aggregate_control_plane_resources(self) -> dict[str, dict[str, str]]:
        """Aggregate resource requests/limits for all control-plane services.

        Since all control-plane services run as subprocesses in a single container,
        we need to sum their resource requirements.

        Returns:
            Combined Kubernetes resource requests and limits.
        """
        # Sum up CPU and memory
        total_cpu_request = 0.0
        total_cpu_limit = 0.0
        total_memory_request = 0
        total_memory_limit = 0

        for _name, res in CONTROL_PLANE_RESOURCE_MAP:
            k8s_res = res.to_k8s_resources()
            requests = k8s_res.get("requests", {})
            limits = k8s_res.get("limits", {})

            # Parse CPU values (handle 'm' suffix for millicores)
            cpu_req = requests.get("cpu", "0")
            cpu_lim = limits.get("cpu", "0")
            total_cpu_request += parse_cpu(cpu_req)
            total_cpu_limit += parse_cpu(cpu_lim)

            # Parse memory values (handle Mi, Gi suffixes)
            mem_req = requests.get("memory", "0")
            mem_lim = limits.get("memory", "0")
            total_memory_request += parse_memory_mib(mem_req)
            total_memory_limit += parse_memory_mib(mem_lim)

        return _format_k8s_resource_totals(
            total_cpu_request, total_cpu_limit, total_memory_request, total_memory_limit
        )

    def _create_worker_containers(self, controller_dns: str) -> list[ContainerSpec]:
        """Create a single container for worker pods using WorkerPodManager.

        The WorkerPodManager runs as the main process and spawns multiple workers
        and record processors as subprocesses. This reduces network overhead by
        downloading the dataset once per pod and sharing it via mmap.
        """
        ports = K8sEnvironment.PORTS

        # Aggregate resources for all workers and record processors in the pod
        worker_pod_resources = self._aggregate_worker_pod_resources()

        return [
            self._create_container(
                name=Containers.WORKER_POD_MANAGER,
                service_type="worker_pod_manager",
                health_port=ports.WORKER_HEALTH,
                resources=worker_pod_resources,
                controller_host=controller_dns,
            ),
        ]

    def _aggregate_worker_pod_resources(self) -> dict[str, dict[str, str]]:
        """Aggregate resource requests/limits for all services in a worker pod.

        Since WorkerPodManager spawns multiple workers and record processors as
        subprocesses, we need to sum their resource requirements.

        Uses environment defaults for workers_per_pod and calculates record
        processors based on the PROCESSOR_SCALE_FACTOR.

        Returns:
            Combined Kubernetes resource requests and limits.
        """
        from aiperf.common.environment import Environment

        workers_per_pod = (
            self.workers_per_pod or Environment.WORKER.DEFAULT_WORKERS_PER_POD
        )
        record_processors_per_pod = max(
            1, workers_per_pod // Environment.RECORD.PROCESSOR_SCALE_FACTOR
        )

        # Get per-service resources
        worker_res = K8sEnvironment.WORKER.to_k8s_resources()
        rp_res = K8sEnvironment.RECORD_PROCESSOR.to_k8s_resources()

        # Parse resource values once
        worker_requests = worker_res.get("requests", {})
        worker_limits = worker_res.get("limits", {})
        rp_requests = rp_res.get("requests", {})
        rp_limits = rp_res.get("limits", {})

        # Calculate totals using multiplication instead of loops
        total_cpu_request = (
            parse_cpu(worker_requests.get("cpu", "0")) * workers_per_pod
            + parse_cpu(rp_requests.get("cpu", "0")) * record_processors_per_pod
            + _WPM_CPU_REQUEST
        )
        total_cpu_limit = (
            parse_cpu(worker_limits.get("cpu", "0")) * workers_per_pod
            + parse_cpu(rp_limits.get("cpu", "0")) * record_processors_per_pod
            + _WPM_CPU_LIMIT
        )
        total_memory_request = (
            parse_memory_mib(worker_requests.get("memory", "0")) * workers_per_pod
            + parse_memory_mib(rp_requests.get("memory", "0"))
            * record_processors_per_pod
            + _WPM_MEMORY_REQUEST_MIB
        )
        total_memory_limit = (
            parse_memory_mib(worker_limits.get("memory", "0")) * workers_per_pod
            + parse_memory_mib(rp_limits.get("memory", "0")) * record_processors_per_pod
            + _WPM_MEMORY_LIMIT_MIB
        )

        return _format_k8s_resource_totals(
            total_cpu_request, total_cpu_limit, total_memory_request, total_memory_limit
        )

    def to_k8s_manifest(self) -> dict[str, Any]:
        """Generate the complete JobSet Kubernetes manifest."""
        controller_dns = controller_dns_name(self.name, self.namespace)
        jobset_config = K8sEnvironment.JOBSET

        # Common volumes
        volumes: list[dict[str, Any]] = [
            {"name": "config", "configMap": {"name": f"{self.name}-config"}},
            {"name": "ipc", "emptyDir": {}},
            {"name": "results", "emptyDir": {}},
            # Shared dataset volume for controller containers (dataset-manager creates, API serves)
            {"name": "datasets", "emptyDir": {}},
            {"name": "tmp", "emptyDir": {}},
        ]
        volumes.extend(self.pod_customization.get_volumes())

        # Controller replicated job
        controller_job = ReplicatedJobSpec(
            name="controller",
            replicas=1,
            containers=self._create_controller_containers(),
            volumes=volumes,
            restart_policy=RestartPolicy.NEVER,
            backoff_limit=jobset_config.CONTROLLER_BACKOFF_LIMIT,
            pod_customization=self.pod_customization,
            job_id=self.job_id,
        )

        # Worker replicated job
        worker_job = ReplicatedJobSpec(
            name="workers",
            replicas=self.worker_replicas,
            containers=self._create_worker_containers(controller_dns),
            volumes=volumes,
            restart_policy=RestartPolicy.ON_FAILURE,
            backoff_limit=jobset_config.WORKER_BACKOFF_LIMIT,
            pod_customization=self.pod_customization,
            job_id=self.job_id,
        )

        # Build JobSet manifest
        labels: dict[str, str] = {
            Labels.APP_KEY: Labels.APP_VALUE,
            Labels.JOB_ID: self.job_id,
        }
        if self.name_label:
            labels[Labels.NAME] = self.name_label
        if self.queue_name:
            labels[KueueLabels.QUEUE_NAME] = self.queue_name
        if self.priority_class:
            labels[KueueLabels.PRIORITY_CLASS] = self.priority_class

        metadata: dict[str, Any] = {
            "name": self.name,
            "namespace": self.namespace,
            "labels": labels,
        }
        if self.extra_annotations:
            metadata["annotations"] = self.extra_annotations

        manifest: dict[str, Any] = {
            "apiVersion": JOBSET_API.api_version,
            "kind": "JobSet",
            "metadata": metadata,
            "spec": {
                # Enable DNS hostnames for pod-to-pod communication
                # This creates a headless service with the same name as the JobSet,
                # allowing pods to have DNS names like:
                # {jobset-name}-{job-name}-{job-index}-{pod-index}.{jobset-name}.{namespace}.svc.cluster.local
                "network": {
                    "enableDNSHostnames": True,
                },
                "successPolicy": {
                    "operator": "All",
                    "targetReplicatedJobs": ["controller"],
                },
                "replicatedJobs": [
                    controller_job.to_k8s_spec(),
                    worker_job.to_k8s_spec(),
                ],
            },
        }

        # Kueue requires JobSets to start suspended; it unsuspends after admission
        if self.queue_name:
            manifest["spec"]["suspend"] = True

        # Add TTL (use instance value if set, otherwise use environment default)
        ttl = (
            self.ttl_seconds
            if self.ttl_seconds is not None
            else jobset_config.TTL_SECONDS_AFTER_FINISHED
        )
        if ttl is not None:
            manifest["spec"]["ttlSecondsAfterFinished"] = ttl

        return manifest
