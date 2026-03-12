# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for Kubernetes module tests."""

import copy
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import kr8s
import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.config.config import AIPerfConfig
from aiperf.kubernetes.client import AIPerfKubeClient
from aiperf.kubernetes.jobset import PodCustomization, SecretMount
from aiperf.plugin.enums import CommunicationBackend, ServiceRunType

# =============================================================================
# Mock kr8s API Fixtures
# =============================================================================


@pytest.fixture
def mock_kr8s_api():
    """Mock kr8s async API client."""
    api = MagicMock(spec=kr8s.Api)
    api.async_get = AsyncMock(return_value=[])
    return api


@pytest.fixture
def mock_kube_client(mock_kr8s_api):
    """Mock AIPerfKubeClient wrapping a mock kr8s API."""
    return AIPerfKubeClient(mock_kr8s_api)


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_jobset() -> dict[str, Any]:
    """Create a sample JobSet dict for testing."""
    return {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "name": "aiperf-test-job",
            "namespace": "default",
            "creationTimestamp": "2026-01-15T10:30:00Z",
            "labels": {
                "app": "aiperf",
                "aiperf.nvidia.com/job-id": "test-job-123",
            },
        },
        "status": {
            "conditions": [],
            "ready": 0,
        },
    }


@pytest.fixture
def sample_running_jobset(sample_jobset) -> dict[str, Any]:
    """Create a sample running JobSet."""
    jobset = copy.deepcopy(sample_jobset)
    jobset["status"]["ready"] = 1
    return jobset


@pytest.fixture
def sample_completed_jobset(sample_jobset) -> dict[str, Any]:
    """Create a sample completed JobSet."""
    jobset = copy.deepcopy(sample_jobset)
    jobset["status"]["conditions"] = [
        {"type": "Completed", "status": "True", "reason": "JobsCompleted"}
    ]
    return jobset


@pytest.fixture
def sample_failed_jobset(sample_jobset) -> dict[str, Any]:
    """Create a sample failed JobSet."""
    jobset = copy.deepcopy(sample_jobset)
    jobset["status"]["conditions"] = [
        {"type": "Failed", "status": "True", "reason": "BackoffLimitExceeded"}
    ]
    return jobset


@pytest.fixture
def sample_pod() -> dict[str, Any]:
    """Create a sample Pod dict (kr8s .raw format) for testing."""
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "aiperf-test-job-controller-0-0",
            "namespace": "default",
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {
                    "name": "system-controller",
                    "ready": True,
                    "restartCount": 0,
                }
            ],
        },
        "spec": {
            "containers": [{"name": "system-controller"}],
        },
    }


@pytest.fixture
def sample_pending_pod(sample_pod) -> dict[str, Any]:
    """Create a sample pending Pod."""
    pod = copy.deepcopy(sample_pod)
    pod["status"]["phase"] = "Pending"
    pod["status"]["containerStatuses"][0]["ready"] = False
    return pod


@pytest.fixture
def sample_succeeded_pod(sample_pod) -> dict[str, Any]:
    """Create a sample succeeded Pod."""
    pod = copy.deepcopy(sample_pod)
    pod["status"]["phase"] = "Succeeded"
    return pod


# =============================================================================
# Config Fixtures
# =============================================================================


@pytest.fixture
def sample_aiperf_config():
    """Create a minimal AIPerfConfig for testing."""
    return AIPerfConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets={
            "default": {"type": "synthetic", "entries": 100, "prompts": {"isl": 128}}
        },
        load={"default": {"type": "concurrency", "concurrency": 1, "requests": 10}},
    )


@pytest.fixture
def sample_user_config():
    """Create a minimal user config for testing."""
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))


@pytest.fixture
def sample_service_config():
    """Create a minimal service config for testing."""
    return ServiceConfig(
        service_run_type=ServiceRunType.MULTIPROCESSING,
        comm_backend=CommunicationBackend.ZMQ_IPC,
    )


@pytest.fixture
def sample_pod_customization():
    """Create a sample PodCustomization for testing."""
    return PodCustomization(
        node_selector={"gpu": "true"},
        tolerations=[
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
        ],
        annotations={"custom/annotation": "value"},
        labels={"custom-label": "value"},
        image_pull_secrets=["my-registry-secret"],
        env_vars={"CUSTOM_VAR": "custom_value"},
        env_from_secrets={"API_KEY": "my-secret/api-key"},
        secret_mounts=[SecretMount(name="my-secret", mount_path="/etc/secrets")],
        service_account="my-service-account",
    )


# =============================================================================
# Helper Functions
# =============================================================================


def create_jobset_list_response(jobsets: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a JobSet list API response."""
    return {"items": jobsets}


def create_server_error(status_code: int, reason: str = "Error") -> kr8s.ServerError:
    """Create a kr8s ServerError for testing error handling."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.reason_phrase = reason
    mock_response.text = f'{{"message": "{reason}"}}'
    return kr8s.ServerError(reason, response=mock_response)


def create_not_found_error(name: str = "resource") -> kr8s.NotFoundError:
    """Create a kr8s NotFoundError for testing."""
    return kr8s.NotFoundError(f"{name} not found")


def make_kr8s_object(raw: dict[str, Any]) -> MagicMock:
    """Create a mock kr8s object with a .raw attribute from a dict."""
    obj = MagicMock()
    obj.raw = raw
    obj.name = raw.get("metadata", {}).get("name", "")
    obj.namespace = raw.get("metadata", {}).get("namespace", "")
    return obj


async def async_list(items: list) -> Any:
    """Create an async generator from a list (for mocking api.async_get)."""
    for item in items:
        yield item
