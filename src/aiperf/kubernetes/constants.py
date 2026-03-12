# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes label, annotation, and container-name constants.

Defined in this dependency-free module so both manifest-generation code
(jobset.py, resources.py) and CLI code (cli_helpers.py) can import them
without circular deps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobSetLabels:
    """Label keys from the JobSet controller (jobset.sigs.k8s.io)."""

    POD_INDEX: str = "jobset.sigs.k8s.io/job-index"
    JOBSET_NAME: str = "jobset.sigs.k8s.io/jobset-name"
    REPLICATED_JOB_NAME: str = "jobset.sigs.k8s.io/replicatedjob-name"


@dataclass(frozen=True)
class Labels:
    """Label keys and values used to identify AIPerf resources."""

    APP_KEY: str = "app"
    APP_VALUE: str = "aiperf"
    JOB_ID: str = "aiperf.nvidia.com/job-id"
    NAME: str = "aiperf.nvidia.com/name"
    SELECTOR: str = "app=aiperf"


@dataclass(frozen=True)
class Annotations:
    """Annotation keys used on AIPerf Kubernetes resources."""

    MODEL: str = "aiperf.nvidia.com/model"
    ENDPOINT: str = "aiperf.nvidia.com/endpoint"


@dataclass(frozen=True)
class ProgressAnnotations:
    """Progress annotations patched onto the JobSet during benchmark execution.

    External tools can observe benchmark progress without connecting to the
    controller pod's API.
    """

    PHASE: str = "aiperf.nvidia.com/progress-phase"
    PERCENT: str = "aiperf.nvidia.com/progress-percent"
    REQUESTS: str = "aiperf.nvidia.com/progress-requests"
    STATUS: str = "aiperf.nvidia.com/progress-status"


@dataclass(frozen=True)
class Containers:
    """Container names used in pod specs and CLI commands."""

    CONTROL_PLANE: str = "control-plane"
    WORKER_POD_MANAGER: str = "worker-pod-manager"


@dataclass(frozen=True)
class KueueLabels:
    """Label keys for Kueue queue integration."""

    QUEUE_NAME: str = "kueue.x-k8s.io/queue-name"
    PRIORITY_CLASS: str = "kueue.x-k8s.io/priority-class"
