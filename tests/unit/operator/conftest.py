# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for operator module tests."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# Kubernetes Resource Body Fixtures
# =============================================================================


@pytest.fixture
def sample_body() -> dict[str, Any]:
    """Create a sample Kubernetes resource body for event testing."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {
            "name": "test-job",
            "namespace": "default",
            "uid": "abc-123",
        },
        "spec": {
            "image": "aiperf:test",
            "models": ["test-model"],
            "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
            "datasets": {
                "main": {"type": "synthetic", "entries": 100, "prompts": {"isl": 128}}
            },
            "load": {
                "default": {"type": "concurrency", "concurrency": 1, "requests": 10}
            },
        },
    }


# =============================================================================
# AIPerfJob Spec Fixtures
# =============================================================================


@pytest.fixture
def minimal_aiperfjob_spec() -> dict[str, Any]:
    """Create a minimal AIPerfJob spec for testing."""
    return {
        "models": ["test-model"],
        "endpoint": {
            "urls": ["http://localhost:8000/v1/chat/completions"],
        },
        "datasets": {
            "main": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128},
            },
        },
        "load": {
            "default": {
                "type": "concurrency",
                "concurrency": 1,
                "requests": 10,
            },
        },
    }


@pytest.fixture
def full_aiperfjob_spec() -> dict[str, Any]:
    """Create a complete AIPerfJob spec with all options."""
    return {
        "image": "aiperf:test",
        "imagePullPolicy": "Always",
        "connectionsPerWorker": 100,
        "models": ["meta-llama/Llama-3.1-8B-Instruct"],
        "endpoint": {
            "urls": ["http://api.example.com/v1/chat/completions"],
            "type": "chat",
            "streaming": True,
        },
        "datasets": {
            "main": {
                "type": "synthetic",
                "entries": 1000,
                "prompts": {"isl": {"mean": 512, "stddev": 50}, "osl": 128},
            },
        },
        "load": {
            "profiling": {
                "type": "concurrency",
                "dataset": "main",
                "requests": 1000,
                "concurrency": 500,
            },
        },
        "podTemplate": {
            "nodeSelector": {"gpu": "true"},
            "tolerations": [
                {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
            ],
            "annotations": {"prometheus.io/scrape": "true"},
            "labels": {"team": "ml-platform"},
            "imagePullSecrets": ["my-registry-secret"],
            "serviceAccountName": "aiperf-sa",
            "env": [
                {"name": "DEBUG", "value": "true"},
                {
                    "name": "API_KEY",
                    "valueFrom": {
                        "secretKeyRef": {"name": "api-secrets", "key": "api-key"}
                    },
                },
            ],
            "volumes": [
                {"name": "creds", "secret": {"secretName": "my-creds"}},
            ],
            "volumeMounts": [
                {"name": "creds", "mountPath": "/etc/creds"},
            ],
        },
    }


@pytest.fixture
def aiperfjob_spec_high_concurrency() -> dict[str, Any]:
    """Create an AIPerfJob spec with high concurrency for worker scaling tests."""
    return {
        "connectionsPerWorker": 100,
        "models": ["test-model"],
        "endpoint": {
            "urls": ["http://localhost:8000/v1/chat/completions"],
        },
        "datasets": {
            "main": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128},
            },
        },
        "load": {
            "default": {
                "type": "concurrency",
                "concurrency": 1000,
                "requests": 10000,
            },
        },
    }


# =============================================================================
# Progress API Response Fixtures
# =============================================================================


@pytest.fixture
def progress_api_response_running() -> dict[str, Any]:
    """Create a progress API response for a running job."""
    return {
        "phases": {
            "warmup": {
                "phase": "warmup",
                "start_ns": 1000,
                "requests_completed": 50,
                "total_expected_requests": 50,
                "requests_per_second": 10.5,
                "requests_progress_percent": 100.0,
                "is_requests_complete": True,
            },
            "profiling": {
                "phase": "profiling",
                "start_ns": 2000,
                "requests_completed": 250,
                "total_expected_requests": 1000,
                "requests_per_second": 25.0,
                "requests_progress_percent": 25.0,
                "is_requests_complete": False,
                "requests_eta_sec": 30.0,
            },
        },
    }


@pytest.fixture
def progress_api_response_complete() -> dict[str, Any]:
    """Create a progress API response for a completed job."""
    return {
        "phases": {
            "profiling": {
                "phase": "profiling",
                "start_ns": 1000,
                "requests_completed": 1000,
                "total_expected_requests": 1000,
                "requests_per_second": 50.0,
                "requests_progress_percent": 100.0,
                "is_requests_complete": True,
            },
        },
    }


@pytest.fixture
def progress_api_response_with_error() -> dict[str, Any]:
    """Create a progress API response with an error."""
    return {
        "error": "Connection refused to endpoint",
        "phases": {
            "profiling": {
                "phase": "profiling",
                "start_ns": 1000,
                "requests_completed": 100,
                "total_expected_requests": 1000,
                "requests_per_second": 0.0,
                "requests_progress_percent": 10.0,
                "is_requests_complete": False,
            },
        },
    }


# =============================================================================
# Condition Fixtures
# =============================================================================


@pytest.fixture
def sample_conditions_list() -> list[dict[str, Any]]:
    """Create a sample conditions list from Kubernetes status."""
    return [
        {
            "type": "ConfigValid",
            "status": "True",
            "reason": "ConfigParsed",
            "message": "Config is valid",
            "lastTransitionTime": "2026-01-15T10:00:00Z",
        },
        {
            "type": "ResourcesCreated",
            "status": "True",
            "reason": "Created",
            "message": "Resources created successfully",
            "lastTransitionTime": "2026-01-15T10:00:05Z",
        },
        {
            "type": "WorkersReady",
            "status": "False",
            "reason": "WorkersStarting",
            "message": "2/5 workers ready",
            "lastTransitionTime": "2026-01-15T10:00:10Z",
        },
    ]


# =============================================================================
# Mock HTTP Session Fixtures
# =============================================================================


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp ClientSession."""
    session = MagicMock()
    session.get = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_http_response():
    """Create a mock HTTP response."""

    def _create_response(json_data: dict[str, Any], status: int = 200) -> AsyncMock:
        response = AsyncMock()
        response.status = status
        response.json = AsyncMock(return_value=json_data)
        response.raise_for_status = MagicMock()
        if status >= 400:
            from aiohttp import ClientResponseError

            response.raise_for_status.side_effect = ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=status,
            )
        return response

    return _create_response
