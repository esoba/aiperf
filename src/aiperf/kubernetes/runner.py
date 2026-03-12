# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes deployment runner for AIPerf."""

import asyncio
import math
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

import kr8s
import orjson
import ruamel.yaml
from kr8s.asyncio.objects import (
    ConfigMap,
    Namespace,
    Role,
    RoleBinding,
)

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.config.kube_config import KubeOptions
from aiperf.common.config.zmq_config import ZMQDualBindConfig
from aiperf.common.environment import Environment
from aiperf.kubernetes.console import print_deployment_summary
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import (
    PodCustomization,
    SecretMount,
    controller_dns_name,
    get_jobset_install_hint,
)
from aiperf.kubernetes.kr8s_resources import AsyncJobSet
from aiperf.kubernetes.resources import KubernetesDeployment
from aiperf.plugin.enums import ServiceRunType, UIType

# Retry configuration
_MAX_RETRIES = 3
_INITIAL_BACKOFF_SEC = 1.0
_BACKOFF_MULTIPLIER = 2.0

# Retryable HTTP status codes (transient failures)
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

T = TypeVar("T")

# Map resource kinds to kr8s classes for manifest application
_KIND_TO_CLASS: dict[str, type] = {
    "Namespace": Namespace,
    "ConfigMap": ConfigMap,
    "Role": Role,
    "RoleBinding": RoleBinding,
    "JobSet": AsyncJobSet,
}


class K8sDeploymentError(Exception):
    """User-friendly error for Kubernetes deployment failures."""


_ERROR_SUGGESTIONS: dict[int, str] = {
    401: (
        "Authentication failed. Check your kubeconfig or "
        "ensure your credentials haven't expired."
    ),
    403: (
        "Check that your kubeconfig has the required permissions. "
        "The service account may need additional RBAC rules."
    ),
    422: (
        "The resource definition is invalid. "
        "Check the manifest with 'aiperf kube generate'."
    ),
}

_JOBSET_CRD_HINT = (
    "The JobSet CRD is not installed. Install it with:\n"
    f"  {get_jobset_install_hint()}\n"
    "Or run: aiperf kube setup"
)


def _format_api_error(
    e: kr8s.ServerError,
    kind: str,
    name: str,
    namespace: str | None = None,
) -> str:
    """Format a kr8s ServerError into a user-friendly error message.

    Args:
        e: The ServerError from kr8s.
        kind: Resource kind (e.g., "JobSet", "ConfigMap").
        name: Resource name.
        namespace: Resource namespace (optional).

    Returns:
        User-friendly error message with suggestions.
    """
    response = e.response
    status = response.status_code if response else 0

    resource_ref = f"{kind}/{name}"
    if namespace:
        resource_ref += f" in namespace '{namespace}'"

    detail = ""
    if response is not None:
        try:
            body = orjson.loads(response.text)
            detail = body.get("message", "")
        except (orjson.JSONDecodeError, TypeError):
            detail = response.text[:200] if response.text else ""

    if status == 404:
        suggestion = (
            _JOBSET_CRD_HINT
            if kind == "JobSet"
            else f"The {kind} resource or namespace was not found."
        )
    elif "admission webhook" in detail.lower() and "too long" in detail.lower():
        suggestion = (
            "Resource names are too long. Use a shorter model name or "
            "use --name to set a custom job name."
        )
    else:
        reason = response.reason_phrase if response else "Unknown"
        suggestion = _ERROR_SUGGESTIONS.get(status, f"HTTP {status}: {reason}")

    msg = f"Failed to create {resource_ref}.\n\n"
    if detail:
        msg += f"Error: {detail}\n\n"
    msg += f"Suggestion: {suggestion}"
    return msg


async def _with_retry(
    operation: Callable[[], Awaitable[T]],
    logger: AIPerfLogger,
    operation_name: str,
) -> T:
    """Execute an async operation with exponential backoff retry on transient failures.

    Args:
        operation: Async callable that performs the K8s API operation.
        logger: Logger for retry messages.
        operation_name: Human-readable name for logging (e.g., "create ConfigMap").

    Returns:
        Result of the operation.

    Raises:
        kr8s.ServerError: If all retries are exhausted or error is not retryable.
    """
    last_exception: kr8s.ServerError | None = None
    backoff = _INITIAL_BACKOFF_SEC

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await operation()
        except kr8s.ServerError as e:
            last_exception = e
            status = e.response.status_code if e.response else 0

            if status not in _RETRYABLE_STATUS_CODES:
                raise

            if attempt >= _MAX_RETRIES:
                reason = e.response.reason_phrase if e.response else "Unknown"
                logger.error(
                    f"Failed to {operation_name} after {_MAX_RETRIES + 1} attempts: "
                    f"{status} {reason}"
                )
                raise

            reason = e.response.reason_phrase if e.response else "Unknown"
            logger.warning(
                f"Transient error during {operation_name} "
                f"(attempt {attempt + 1}/{_MAX_RETRIES + 1}): "
                f"{status} {reason}. Retrying in {backoff:.1f}s..."
            )
            await asyncio.sleep(backoff)
            backoff *= _BACKOFF_MULTIPLIER

    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected retry loop exit")


def _kube_options_to_pod_customization(kube_options: KubeOptions) -> PodCustomization:
    """Convert KubeOptions to PodCustomization model."""
    # Convert SecretMountConfig to SecretMount
    secret_mounts = [
        SecretMount(name=s.name, mount_path=s.mount_path, sub_path=s.sub_path)
        for s in kube_options.secret_mounts
    ]
    return PodCustomization(
        node_selector=kube_options.node_selector,
        tolerations=kube_options.tolerations,
        annotations=kube_options.annotations,
        labels=kube_options.labels,
        image_pull_secrets=kube_options.image_pull_secrets,
        env_vars=kube_options.env_vars,
        env_from_secrets=kube_options.env_from_secrets,
        secret_mounts=secret_mounts,
        service_account=kube_options.service_account,
    )


def _setup_zmq_dual_bind(service_config: ServiceConfig) -> None:
    """Configure ZMQ dual-bind for Kubernetes (IPC for controller, TCP for workers).

    Directly assigns fields and updates model_fields_set to bypass validators
    that would try to create the IPC directory (which doesn't exist on the CLI
    machine). Also updates _comm_config since Pydantic model validators only
    run during initialization.

    Args:
        service_config: Service configuration to modify in-place.
    """
    if service_config.zmq_dual is not None:
        return
    ipc_path = Path(K8sEnvironment.ZMQ.IPC_PATH)
    service_config.zmq_dual = ZMQDualBindConfig(ipc_path=ipc_path, tcp_host="0.0.0.0")
    service_config.model_fields_set.add("zmq_dual")
    # Critical: without updating _comm_config, comm_config returns the default
    # ZMQIPCConfig instead of ZMQDualBindConfig, causing workers to use wrong addresses
    service_config._comm_config = service_config.zmq_dual


def _setup_api_service(
    service_config: ServiceConfig, job_id: str, namespace: str
) -> None:
    """Configure API service for dataset serving to workers.

    Sets API port/host and builds the dataset download URL that workers use
    to fetch data from the controller pod's API service.

    Args:
        service_config: Service configuration to modify in-place.
        job_id: Job identifier for building the controller DNS name.
        namespace: Kubernetes namespace for DNS resolution.
    """
    if service_config.api_port is None:
        service_config.api_port = K8sEnvironment.PORTS.API_SERVICE
        service_config.api_host = "0.0.0.0"  # Listen on all interfaces for K8s
        service_config.model_fields_set.add("api_port")
        service_config.model_fields_set.add("api_host")

    if service_config.dataset_api_base_url is None:
        api_port = K8sEnvironment.PORTS.API_SERVICE
        jobset_name = f"aiperf-{job_id}"
        dns = controller_dns_name(jobset_name, namespace)
        service_config.dataset_api_base_url = f"http://{dns}:{api_port}/api/dataset"
        service_config.model_fields_set.add("dataset_api_base_url")


def _calculate_worker_pods(
    total_workers: int, service_config: ServiceConfig
) -> tuple[int, int]:
    """Calculate number of worker pods and workers per pod.

    If requesting fewer workers than default per pod, uses a single pod.
    Otherwise divides evenly with ceil rounding.

    Args:
        total_workers: Total workers requested by user (--workers-max).
        service_config: Service configuration (modified in-place to set workers_per_pod).

    Returns:
        Tuple of (num_pods, workers_per_pod).
    """
    default_workers_per_pod = (
        service_config.workers_per_pod or Environment.WORKER.DEFAULT_WORKERS_PER_POD
    )

    if total_workers <= default_workers_per_pod:
        workers_per_pod = total_workers
        num_pods = 1
    else:
        workers_per_pod = default_workers_per_pod
        num_pods = math.ceil(total_workers / workers_per_pod)

    service_config.workers_per_pod = workers_per_pod
    service_config.model_fields_set.add("workers_per_pod")

    return num_pods, workers_per_pod


def _generate_benchmark_name(user_config: UserConfig) -> str:
    """Generate a descriptive benchmark name matching artifact directory naming.

    Uses the same model, service kind, and stimulus components as the
    artifact directory to produce a human-readable name for the benchmark.
    """
    names = [
        user_config._get_artifact_model_name(),
        user_config._get_artifact_service_kind(),
        user_config._get_artifact_stimulus(),
    ]
    return "-".join(names)


async def run_kubernetes_deployment(
    user_config: UserConfig,
    service_config: ServiceConfig,
    kube_options: KubeOptions,
    *,
    dry_run: bool = False,
) -> tuple[str, str]:
    """Run AIPerf benchmark in Kubernetes mode.

    Args:
        user_config: User configuration for the benchmark.
        service_config: Service configuration.
        kube_options: Kubernetes-specific deployment options.
        dry_run: If True, output YAML manifests instead of applying to cluster.

    Returns:
        Tuple of (job_id, namespace) for the deployed benchmark.
    """
    logger = AIPerfLogger(__name__)

    job_id = uuid.uuid4().hex[:8]
    namespace = kube_options.namespace or f"aiperf-{job_id}"

    # Auto-populate name from config when not explicitly provided
    name = kube_options.name or _generate_benchmark_name(user_config)

    # Set service_run_type for Kubernetes mode
    # Must add to model_fields_set so it's serialized (exclude_unset=True in ConfigMap)
    service_config.service_run_type = ServiceRunType.KUBERNETES
    service_config.model_fields_set.add("service_run_type")

    # Always disable UI in Kubernetes pods (UI runs locally via attach mode)
    # The user's ui_type config is used by the CLI to determine attach mode
    service_config.ui_type = UIType.NONE
    service_config.model_fields_set.add("ui_type")

    # Override artifact directory to writable /results volume (readOnlyRootFilesystem)
    user_config.output.artifact_directory = Path("/results")
    user_config.output.model_fields_set.add("artifact_directory")
    user_config.model_fields_set.add("output")

    _setup_zmq_dual_bind(service_config)
    _setup_api_service(service_config, job_id, namespace)
    num_pods, workers_per_pod = _calculate_worker_pods(
        kube_options.workers, service_config
    )

    # Store total worker count so the controller can set proper expectations
    total_workers = num_pods * workers_per_pod
    if total_workers != kube_options.workers:
        from aiperf.kubernetes.console import print_warning

        print_warning(
            f"Requested {kube_options.workers} workers, but {num_pods} pods x "
            f"{workers_per_pod} workers/pod = {total_workers} workers"
        )
    service_config.workers.max = total_workers
    service_config.workers.model_fields_set.add("max")
    service_config.model_fields_set.add("workers")

    # Align record processor expectation with actual per-pod spawning.
    # Pods calculate: max(1, workers_per_pod // PROCESSOR_SCALE_FACTOR)
    # Controller calculates globally: total_workers // PROCESSOR_SCALE_FACTOR
    # These can differ, so set the exact count the pods will actually spawn.
    rp_per_pod = max(1, workers_per_pod // Environment.RECORD.PROCESSOR_SCALE_FACTOR)
    service_config.record_processor_service_count = rp_per_pod * num_pods
    service_config.model_fields_set.add("record_processor_service_count")

    # Build endpoint URL for annotations
    endpoint_url = user_config.endpoint.url if user_config.endpoint.urls else None

    # Create the deployment specification with the same job_id
    deployment = KubernetesDeployment(
        job_id=job_id,
        namespace=kube_options.namespace,
        image=kube_options.image,
        image_pull_policy=kube_options.image_pull_policy,
        worker_replicas=num_pods,
        workers_per_pod=workers_per_pod,
        ttl_seconds=kube_options.ttl_seconds,
        user_config=user_config,
        service_config=service_config,
        pod_customization=_kube_options_to_pod_customization(kube_options),
        queue_name=kube_options.queue_name,
        priority_class=kube_options.priority_class,
        name=name,
        model_names=user_config.endpoint.model_names,
        endpoint_url=endpoint_url,
    )

    # Print deployment summary to console (use stderr when dry_run to avoid polluting YAML output)
    print_deployment_summary(
        job_id=job_id,
        namespace=namespace,
        image=kube_options.image,
        workers=num_pods * workers_per_pod,
        num_pods=num_pods,
        workers_per_pod=workers_per_pod,
        ttl_seconds=kube_options.ttl_seconds,
        endpoint_url=endpoint_url,
        model_names=user_config.endpoint.model_names,
        to_stderr=dry_run,
        name=name,
    )

    # Generate all manifests
    manifests = deployment.get_all_manifests()

    if dry_run:
        # Output YAML to stdout
        _output_manifests_yaml(manifests)
        return (job_id, namespace)

    # Apply manifests to cluster
    await _apply_manifests(
        manifests, logger, kube_options.kubeconfig, kube_options.kube_context
    )

    return (job_id, namespace)


def _output_manifests_yaml(manifests: list[dict[str, Any]]) -> None:
    """Output manifests as YAML to stdout.

    Args:
        manifests: List of Kubernetes resource manifests.
    """
    yaml = ruamel.yaml.YAML()
    yaml.default_flow_style = False

    for i, manifest in enumerate(manifests):
        if i > 0:
            sys.stdout.write("---\n")
        yaml.dump(manifest, sys.stdout)


async def _apply_manifests(
    manifests: list[dict[str, Any]],
    logger: AIPerfLogger,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> None:
    """Apply manifests to the Kubernetes cluster using kr8s.

    Args:
        manifests: List of Kubernetes resource manifests.
        logger: Logger instance.
        kubeconfig: Optional path to kubeconfig file.
        kube_context: Optional Kubernetes context name.
    """
    from aiperf.kubernetes.client import get_api

    api = await get_api(kubeconfig=kubeconfig, kube_context=kube_context)

    for manifest in manifests:
        kind = manifest["kind"]
        name = manifest["metadata"]["name"]
        namespace = manifest["metadata"].get("namespace")

        cls = _KIND_TO_CLASS.get(kind)
        if cls is None:
            logger.warning(f"Unknown resource kind: {kind}")
            continue

        obj = cls(manifest, api=api)

        try:
            await _with_retry(obj.create, logger, f"create {kind}/{name}")
            ns_suffix = f" in {namespace}" if namespace else ""
            logger.info(f"Created {kind}/{name}{ns_suffix}")
        except kr8s.ServerError as e:
            status = e.response.status_code if e.response else 0
            if status == 409:
                if kind == "Namespace":
                    logger.debug(f"{kind}/{name} already exists")
                else:
                    logger.warning(f"{kind}/{name} already exists, skipping")
            else:
                error_msg = _format_api_error(e, kind, name, namespace)
                logger.error(error_msg)
                raise K8sDeploymentError(error_msg) from e
