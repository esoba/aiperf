# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MetricsRouterComponent."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.api.routers.metrics import MetricsRouterComponent
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.messages import RealtimeMetricsMessage
from tests.unit.api.conftest import make_latency_metric, make_metric_result


@pytest.fixture
def metrics_component(
    mock_zmq, component_service_config: ServiceConfig, component_user_config: UserConfig
) -> MetricsRouterComponent:
    """Create a MetricsRouterComponent for testing."""
    return MetricsRouterComponent(
        service_config=component_service_config,
        user_config=component_user_config,
    )


@pytest.fixture
def metrics_client(metrics_component: MetricsRouterComponent) -> TestClient:
    """Create a TestClient wired to the metrics component router."""
    app = FastAPI()
    app.state.metrics = metrics_component
    app.include_router(metrics_component.get_router())
    return TestClient(app)


class TestPrometheusMetricsEndpoint:
    """Test the /metrics endpoint."""

    def test_empty_metrics(
        self, metrics_client: TestClient, metrics_component: MetricsRouterComponent
    ) -> None:
        metrics_component._metrics = []
        response = metrics_client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_with_metrics(
        self, metrics_client: TestClient, metrics_component: MetricsRouterComponent
    ) -> None:
        metrics_component._metrics = [make_latency_metric(avg=100.0)]
        response = metrics_client.get("/metrics")
        assert response.status_code == 200
        assert "aiperf_latency" in response.text


class TestJsonMetricsEndpoint:
    """Test the /api/metrics endpoint."""

    def test_empty_metrics(
        self, metrics_client: TestClient, metrics_component: MetricsRouterComponent
    ) -> None:
        metrics_component._metrics = []
        response = metrics_client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["metrics"] == {}

    def test_with_data(
        self, metrics_client: TestClient, metrics_component: MetricsRouterComponent
    ) -> None:
        metrics_component._metrics = [make_latency_metric(avg=100.0)]
        response = metrics_client.get("/api/metrics")
        data = response.json()
        assert data["metrics"]["latency"]["avg"] == 100.0

    def test_multiple_metrics(
        self, metrics_client: TestClient, metrics_component: MetricsRouterComponent
    ) -> None:
        metrics_component._metrics = [
            make_latency_metric(avg=100.0),
            make_metric_result(
                tag="throughput", header="Throughput", unit="req/s", avg=50.0
            ),
        ]
        response = metrics_client.get("/api/metrics")
        data = response.json()
        assert "latency" in data["metrics"]
        assert "throughput" in data["metrics"]


class TestInfoLabelsCache:
    """Test the info labels caching behavior."""

    def test_get_info_labels_creates_and_caches(
        self, metrics_component: MetricsRouterComponent
    ) -> None:
        assert metrics_component._info_labels is None

        labels1 = metrics_component.get_info_labels()
        assert labels1 is not None
        assert metrics_component._info_labels is not None

        labels2 = metrics_component.get_info_labels()
        assert labels1 is labels2


class TestRealtimeMetricsHandler:
    """Test the @on_message handler from RealtimeMetricsMixin."""

    @pytest.mark.asyncio
    async def test_on_realtime_metrics_updates_state(
        self, metrics_component: MetricsRouterComponent
    ) -> None:
        metrics_component.run_hooks = AsyncMock()

        metric = make_latency_metric(avg=42.0)
        message = RealtimeMetricsMessage(service_id="test", metrics=[metric])
        await metrics_component._on_realtime_metrics(message)

        assert len(metrics_component._metrics) == 1
        assert metrics_component._metrics[0].avg == 42.0
