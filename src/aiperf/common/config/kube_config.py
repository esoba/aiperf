# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes deployment configuration."""

from typing import Annotated, Any

from cyclopts import Group
from pydantic import Field, field_validator

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.kubernetes.enums import ImagePullPolicy


class _KubeGroups:
    """Groups for Kubernetes CLI options."""

    KUBERNETES = Group.create_ordered("Kubernetes")
    K8S_NODE_PLACEMENT = Group.create_ordered("Kubernetes Node Placement")
    K8S_SCHEDULING = Group.create_ordered("Kubernetes Scheduling")
    K8S_SECRETS = Group.create_ordered("Kubernetes Secrets")
    K8S_METADATA = Group.create_ordered("Kubernetes Metadata")


class SecretMountConfig(BaseConfig):
    """Configuration for mounting a Kubernetes secret as a volume."""

    name: str = Field(description="Secret name in Kubernetes")
    mount_path: str = Field(description="Path to mount the secret")
    sub_path: str | None = Field(
        default=None, description="Specific key to mount (optional)"
    )


class KubeManageOptions(BaseConfig):
    """Common options for Kubernetes job management commands.

    This config contains the kubeconfig and namespace options shared by
    management commands (status, logs, delete, attach, results, cancel, preflight).

    Example CLI usage:
        aiperf kube status --kubeconfig ~/.kube/prod-config --namespace benchmarks
        aiperf kube logs abc123 --namespace aiperf-bench
    """

    kubeconfig: Annotated[
        str | None,
        Field(
            description="Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env)"
        ),
        CLIParameter(name="--kubeconfig", group=_KubeGroups.KUBERNETES),
    ] = None

    kube_context: Annotated[
        str | None,
        Field(
            description="Kubernetes context to use (defaults to current context in kubeconfig)"
        ),
        CLIParameter(name="--kube-context", group=_KubeGroups.KUBERNETES),
    ] = None

    namespace: Annotated[
        str | None,
        Field(description="Kubernetes namespace"),
        CLIParameter(name="--namespace", group=_KubeGroups.KUBERNETES),
    ] = None


class KubeOptions(KubeManageOptions):
    """Kubernetes-specific deployment options.

    This config contains the Kubernetes deployment settings (not benchmark config).
    Inherits kubeconfig and namespace from KubeManageOptions.
    Use with UserConfig and ServiceConfig for the complete deployment specification.

    Example YAML:
        ```yaml
        image: aiperf:latest
        namespace: benchmarks
        workers: 10
        ttl_seconds: 300
        node_selector:
          gpu: "true"
        tolerations:
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
        ```
    """

    # Optional: Human-readable name
    name: Annotated[
        str | None,
        Field(
            default=None,
            description="Human-readable name for the benchmark job (DNS label, max 40 chars)",
        ),
        CLIParameter(name="--name", group=_KubeGroups.KUBERNETES),
    ] = None

    # Required: Container image
    image: Annotated[
        str,
        Field(description="AIPerf container image to use for Kubernetes deployment"),
        CLIParameter(name="--image", group=_KubeGroups.KUBERNETES),
    ]

    image_pull_policy: Annotated[
        ImagePullPolicy | None,
        Field(
            default=None,
            description="Image pull policy (Always, IfNotPresent, Never). "
            "Use 'Never' for minikube (or local clusters) with locally loaded images.",
        ),
        CLIParameter(name="--image-pull-policy", group=_KubeGroups.KUBERNETES),
    ] = None

    workers: Annotated[
        int,
        Field(
            description="Total number of workers. Automatically distributed across pods "
            "based on --workers-per-pod (default 10). E.g., --workers-max 50 = 5 pods × 10 workers."
        ),
        CLIParameter(name="--workers-max", group=_KubeGroups.KUBERNETES),
    ] = 10

    ttl_seconds: Annotated[
        int | None,
        Field(
            description="Seconds to keep pods after completion (None to disable TTL)"
        ),
        CLIParameter(name="--ttl-seconds", group=_KubeGroups.KUBERNETES),
    ] = 300

    # Node placement
    node_selector: Annotated[
        dict[str, str],
        Field(description="Node selector labels (e.g., {'gpu': 'true'})"),
        CLIParameter(name="--node-selector", group=_KubeGroups.K8S_NODE_PLACEMENT),
    ] = {}

    tolerations: Annotated[
        list[dict[str, Any]],
        Field(description="Pod tolerations for scheduling on tainted nodes"),
        CLIParameter(name="--tolerations", group=_KubeGroups.K8S_NODE_PLACEMENT),
    ] = []

    # Scheduling / Kueue
    queue_name: Annotated[
        str | None,
        Field(
            default=None,
            description="Kueue LocalQueue name for gang-scheduling. When set, the JobSet "
            "is submitted to Kueue for quota-managed admission.",
        ),
        CLIParameter(name="--queue-name", group=_KubeGroups.K8S_SCHEDULING),
    ] = None

    priority_class: Annotated[
        str | None,
        Field(
            default=None,
            description="Kueue WorkloadPriorityClass name for scheduling priority",
        ),
        CLIParameter(name="--priority-class", group=_KubeGroups.K8S_SCHEDULING),
    ] = None

    # Metadata
    annotations: Annotated[
        dict[str, str],
        Field(description="Additional pod annotations"),
        CLIParameter(name="--annotations", group=_KubeGroups.K8S_METADATA),
    ] = {}

    labels: Annotated[
        dict[str, str],
        Field(description="Additional pod labels"),
        CLIParameter(name="--labels", group=_KubeGroups.K8S_METADATA),
    ] = {}

    # Secrets and credentials
    image_pull_secrets: Annotated[
        list[str],
        Field(description="Image pull secret names"),
        CLIParameter(name="--image-pull-secrets", group=_KubeGroups.K8S_SECRETS),
    ] = []

    env_vars: Annotated[
        dict[str, str],
        Field(description="Extra environment variables (key: value)"),
        CLIParameter(name="--env-vars", group=_KubeGroups.K8S_SECRETS),
    ] = {}

    env_from_secrets: Annotated[
        dict[str, str],
        Field(
            description="Environment variables from secrets (ENV_NAME: secret_name/key)"
        ),
        CLIParameter(name="--env-from-secrets", group=_KubeGroups.K8S_SECRETS),
    ] = {}

    secret_mounts: Annotated[
        list[SecretMountConfig],
        Field(description="Secret volume mounts"),
        CLIParameter(name="--secret-mounts", group=_KubeGroups.K8S_SECRETS),
    ] = []

    service_account: Annotated[
        str | None,
        Field(description="Service account name for pods"),
        CLIParameter(name="--service-account", group=_KubeGroups.K8S_SECRETS),
    ] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        """Validate name is a valid DNS label (max 40 chars)."""
        if v is not None:
            from aiperf.kubernetes.resources import validate_dns_label

            validate_dns_label(v, "name", max_length=40)
        return v
