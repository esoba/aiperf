# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the operator web UI jobs API router."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from aiperf.operator.routers.jobs import create_jobs_router


def _make_app(kube_client=None):
    """Create a minimal FastAPI app with the jobs router for testing."""
    from fastapi import FastAPI

    app = FastAPI()
    holder = [kube_client]
    router = create_jobs_router(holder)
    app.include_router(router)
    return app


def _mock_job_info(**overrides):
    """Create a mock AIPerfJobInfo-like dict."""
    base = {
        "name": "test-bench",
        "namespace": "aiperf-benchmarks",
        "phase": "Running",
        "jobId": "test-bench",
        "jobsetName": "aiperf-test-bench",
        "workersReady": 1,
        "workersTotal": 1,
        "currentPhase": "profiling",
        "error": None,
        "startTime": "2026-03-19T18:00:00Z",
        "completionTime": None,
        "created": "2026-03-19T18:00:00Z",
        "progressPercent": 67.0,
        "throughputRps": 1.8,
        "latencyP99Ms": 13376.0,
        "model": "Qwen/Qwen3-0.6B",
        "endpoint": "http://vllm-server:8000/v1",
    }
    base.update(overrides)
    return base


class TestListJobs:
    @pytest.mark.asyncio
    async def test_list_jobs_returns_jobs(self):
        mock_client = AsyncMock()
        mock_client.list_jobs.return_value = [
            AsyncMock(model_dump=lambda by_alias=True: _mock_job_info())
        ]
        app = _make_app(mock_client)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/jobs")

        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["name"] == "test-bench"

    @pytest.mark.asyncio
    async def test_list_jobs_no_client_returns_503(self):
        app = _make_app(kube_client=None)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/jobs")

        assert resp.status_code == 503


class TestGetJob:
    @pytest.mark.asyncio
    async def test_get_job_found(self):
        mock_client = AsyncMock()
        mock_client.find_job.return_value = AsyncMock(
            model_dump=lambda by_alias=True: _mock_job_info(),
        )
        mock_client.get_raw_status.return_value = {"conditions": [], "phases": {}}
        mock_client.get_pods.return_value = []
        app = _make_app(mock_client)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/jobs/aiperf-benchmarks/test-bench")

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_job_not_found(self):
        mock_client = AsyncMock()
        mock_client.find_job.return_value = None
        app = _make_app(mock_client)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/jobs/aiperf-benchmarks/nonexistent")

        assert resp.status_code == 404


class TestCluster:
    @pytest.mark.asyncio
    async def test_cluster_info(self):
        mock_client = AsyncMock()
        mock_node = AsyncMock()
        mock_node.raw = {
            "metadata": {"name": "node1"},
            "status": {
                "allocatable": {"nvidia.com/gpu": "1"},
            },
        }

        async def _mock_get(*args, **kwargs):
            yield mock_node

        mock_client.api = AsyncMock()
        mock_client.api.get = _mock_get
        mock_client.version.return_value = {"gitVersion": "v1.33.1"}
        app = _make_app(mock_client)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/cluster")

        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == 1
        assert data["gpus"] >= 0


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_job(self):
        mock_client = AsyncMock()
        mock_client.cancel_job.return_value = None
        app = _make_app(mock_client)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/v1/jobs/aiperf-benchmarks/test-bench/cancel")

        assert resp.status_code == 200
        mock_client.cancel_job.assert_called_once_with(
            "test-bench", "aiperf-benchmarks"
        )
