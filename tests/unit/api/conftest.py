# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for API tests.

Provides reusable test utilities for testing the AIPerf API module including:
- Mock WebSocket creation
- Mock service factories
- MetricResult builders
- UserConfig builders with common variations
"""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient
from starlette.websockets import WebSocketState

from aiperf.api.api_service import FastAPIService
from aiperf.api.routers.api import api_router
from aiperf.api.routers.core import core_router
from aiperf.api.routers.websocket import WebSocketManager
from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.models import MetricResult
from aiperf.common.models.record_models import ProcessRecordsResult, ProfileResults
from aiperf.config.config import AIPerfConfig

# -----------------------------------------------------------------------------
# WebSocket Mock Helpers
# -----------------------------------------------------------------------------


def make_mock_websocket(
    closed: bool = False,
    send_side_effect: Exception | None = None,
) -> AsyncMock:
    """Create a mock WebSocket with configurable behavior.

    Args:
        closed: Whether the WebSocket should report as closed.
        send_side_effect: Optional exception to raise on send_text/send_str.

    Returns:
        Configured AsyncMock WebSocket.
    """
    ws = AsyncMock()
    ws.closed = closed
    if send_side_effect:
        ws.send_text.side_effect = send_side_effect
        ws.send_str.side_effect = send_side_effect
    return ws


def make_mock_fastapi_websocket(
    client_state: WebSocketState | None = None,
) -> AsyncMock:
    """Create a mock FastAPI WebSocket with Starlette state.

    Args:
        client_state: The WebSocketState enum value (defaults to CONNECTED).

    Returns:
        Configured AsyncMock WebSocket for FastAPI.
    """
    ws = AsyncMock()
    ws.client_state = (
        client_state if client_state is not None else WebSocketState.CONNECTED
    )
    return ws


# -----------------------------------------------------------------------------
# MetricResult Builders
# -----------------------------------------------------------------------------


def make_metric_result(
    tag: str = "test_metric",
    header: str = "Test Metric",
    unit: str = "ms",
    avg: float | None = None,
    min: float | None = None,
    max: float | None = None,
    sum: float | None = None,
    p50: float | None = None,
    p95: float | None = None,
    p99: float | None = None,
    std: float | None = None,
    **kwargs,
) -> MetricResult:
    """Create a MetricResult with sensible defaults.

    Args:
        tag: Metric tag/identifier.
        header: Human-readable header.
        unit: Unit of measurement.
        avg: Average value.
        min: Minimum value.
        max: Maximum value.
        sum: Sum/total value.
        p50: 50th percentile.
        p95: 95th percentile.
        p99: 99th percentile.
        std: Standard deviation.
        **kwargs: Additional MetricResult fields.

    Returns:
        Configured MetricResult.
    """
    return MetricResult(
        tag=tag,
        header=header,
        unit=unit,
        avg=avg,
        min=min,
        max=max,
        sum=sum,
        p50=p50,
        p95=p95,
        p99=p99,
        std=std,
        **kwargs,
    )


def make_latency_metric(
    avg: float = 100.0,
    min: float = 50.0,
    max: float = 200.0,
    p50: float = 95.0,
    p95: float = 180.0,
    p99: float = 195.0,
) -> MetricResult:
    """Create a typical latency metric for testing.

    Args:
        avg: Average latency.
        min: Minimum latency.
        max: Maximum latency.
        p50: Median latency.
        p95: 95th percentile latency.
        p99: 99th percentile latency.

    Returns:
        MetricResult configured as a latency metric.
    """
    return make_metric_result(
        tag="latency",
        header="Latency",
        unit="ms",
        avg=avg,
        min=min,
        max=max,
        p50=p50,
        p95=p95,
        p99=p99,
    )


def make_throughput_metric(
    avg: float = 50.0,
    sum: float = 5000.0,
) -> MetricResult:
    """Create a typical throughput metric for testing.

    Args:
        avg: Average throughput.
        sum: Total throughput.

    Returns:
        MetricResult configured as a throughput metric.
    """
    return make_metric_result(
        tag="throughput",
        header="Throughput",
        unit="req/s",
        avg=avg,
        sum=sum,
    )


# -----------------------------------------------------------------------------
# UserConfig Builders
# -----------------------------------------------------------------------------


def make_user_config(
    benchmark_id: str | None = None,
    model_names: list[str] | None = None,
    endpoint_type: str = "chat",
    streaming: bool = False,
    concurrency: int | None = None,
    request_rate: float | None = None,
) -> UserConfig:
    """Create a UserConfig with common test variations.

    Args:
        benchmark_id: Optional benchmark identifier.
        model_names: List of model names (defaults to ["test-model"]).
        endpoint_type: Endpoint type (chat, completions, etc.).
        streaming: Whether streaming is enabled.
        concurrency: Optional concurrency setting.
        request_rate: Optional request rate setting.

    Returns:
        Configured UserConfig.
    """
    model_names = model_names or ["test-model"]
    loadgen_kwargs = {}
    if concurrency is not None:
        loadgen_kwargs["concurrency"] = concurrency
    if request_rate is not None:
        loadgen_kwargs["request_rate"] = request_rate

    config_kwargs = {
        "benchmark_id": benchmark_id,
        "endpoint": EndpointConfig(
            model_names=model_names,
            type=endpoint_type,
            streaming=streaming,
        ),
    }
    if loadgen_kwargs:
        config_kwargs["loadgen"] = LoadGeneratorConfig(**loadgen_kwargs)

    return UserConfig(**config_kwargs)


# -----------------------------------------------------------------------------
# Service Config Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def api_service_config() -> ServiceConfig:
    """Create a ServiceConfig for API service testing."""
    return ServiceConfig(api_port=9999, api_host="127.0.0.1")


@pytest.fixture
def api_user_config() -> UserConfig:
    """Create a UserConfig for API service testing."""
    return UserConfig(
        benchmark_id="test-bench",
        endpoint=EndpointConfig(model_names=["test-model"]),
    )


@pytest.fixture
def mock_fastapi_service(mock_zmq, aiperf_config: AIPerfConfig) -> FastAPIService:
    """Create a FastAPIService instance for testing without starting the server."""
    aiperf_config.api_port = 9999
    aiperf_config.api_host = "127.0.0.1"
    svc = FastAPIService(
        config=aiperf_config,
        service_id="api-test-1",
    )
    return svc


def create_test_app(service: FastAPIService | None = None) -> FastAPI:
    """Create a FastAPI app for testing with optional service injection.

    Args:
        service: Optional service instance. If None, routes requiring
                 service will raise RuntimeError.

    Returns:
        Configured FastAPI app for testing.
    """
    app = FastAPI(default_response_class=ORJSONResponse)
    app.state.service = service
    app.include_router(api_router)
    app.include_router(core_router)
    return app


# -----------------------------------------------------------------------------
# HTTP Client Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def api_test_client(mock_fastapi_service: FastAPIService) -> TestClient:
    """Create a synchronous TestClient for HTTP testing."""
    return TestClient(mock_fastapi_service.app)


@pytest.fixture
async def api_async_client(mock_fastapi_service: FastAPIService) -> AsyncClient:
    """Create an asynchronous AsyncClient for HTTP testing."""
    transport = ASGITransport(app=mock_fastapi_service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# -----------------------------------------------------------------------------
# WebSocket Manager Fixture
# -----------------------------------------------------------------------------


@pytest.fixture
def websocket_manager() -> WebSocketManager:
    """Create a fresh WebSocketManager for testing."""
    return WebSocketManager()


# -----------------------------------------------------------------------------
# Info Labels Builders
# -----------------------------------------------------------------------------


def make_info_labels(
    model: str = "test-model",
    endpoint_type: str = "chat",
    streaming: str = "false",
    benchmark_id: str | None = None,
    concurrency: str | None = None,
    request_rate: str | None = None,
    config: dict | None = None,
) -> dict[str, str]:
    """Create info labels dict for Prometheus/JSON metrics testing.

    Args:
        model: Model name(s).
        endpoint_type: Endpoint type.
        streaming: Streaming enabled flag as string.
        benchmark_id: Optional benchmark ID.
        concurrency: Optional concurrency as string.
        request_rate: Optional request rate as string.
        config: Optional full config dict.

    Returns:
        Info labels dict.
    """
    labels = {
        "model": model,
        "endpoint_type": endpoint_type,
        "streaming": streaming,
    }
    if benchmark_id:
        labels["benchmark_id"] = benchmark_id
    if concurrency:
        labels["concurrency"] = concurrency
    if request_rate:
        labels["request_rate"] = request_rate
    if config:
        labels["config"] = config
    return labels


# -----------------------------------------------------------------------------
# ProcessRecordsResult Builders
# -----------------------------------------------------------------------------


def make_profile_results(
    records: list[MetricResult] | None = None,
    completed: int = 100,
    start_ns: int = 1000000000,
    end_ns: int = 2000000000,
    was_cancelled: bool = False,
) -> ProfileResults:
    """Create a ProfileResults with sensible defaults.

    Args:
        records: List of metric results.
        completed: Number of completed requests.
        start_ns: Start time in nanoseconds.
        end_ns: End time in nanoseconds.
        was_cancelled: Whether the profile was cancelled.

    Returns:
        Configured ProfileResults.
    """
    if records is None:
        records = [make_latency_metric(), make_throughput_metric()]
    return ProfileResults(
        records=records,
        completed=completed,
        start_ns=start_ns,
        end_ns=end_ns,
        was_cancelled=was_cancelled,
    )


def make_process_records_result(
    records: list[MetricResult] | None = None,
    completed: int = 100,
    was_cancelled: bool = False,
) -> ProcessRecordsResult:
    """Create a ProcessRecordsResult with sensible defaults.

    Args:
        records: List of metric results for ProfileResults.
        completed: Number of completed requests.
        was_cancelled: Whether the profile was cancelled.

    Returns:
        Configured ProcessRecordsResult.
    """
    profile_results = make_profile_results(
        records=records,
        completed=completed,
        was_cancelled=was_cancelled,
    )
    return ProcessRecordsResult(results=profile_results)
