# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FastAPIService lifecycle, init, CORS, start/stop, and main."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.api.api_service import FastAPIService, get_service
from aiperf.api.routers.core import core_router
from aiperf.api.routers.static import static_router
from aiperf.common.config import ServiceConfig


def create_test_app(service: FastAPIService | None = None) -> FastAPI:
    """Create a FastAPI app for testing with only plain routers."""
    app = FastAPI()
    app.state.service = service
    for router in (
        core_router,
        static_router,
    ):
        app.include_router(router)
    return app


# =============================================================================
# Test app factory and dependency injection
# =============================================================================


class TestCreateTestApp:
    """Test the create_test_app factory and dependency injection patterns."""

    def test_create_test_app_with_mock_service(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test create_test_app creates a working app with injected service."""
        app = create_test_app(mock_fastapi_service)
        client = TestClient(app)

        response = client.get("/healthz")
        assert response.status_code == 200

    def test_dependency_overrides_pattern(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that dependency_overrides works for mocking service."""
        app = create_test_app(None)
        app.dependency_overrides[get_service] = lambda: mock_fastapi_service

        client = TestClient(app)
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_create_test_app_without_service_raises(self) -> None:
        """Test that service-dependent endpoints fail gracefully without a service."""
        app = create_test_app(None)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/config")
        assert response.status_code == 500


# =============================================================================
# Compression encoding selection
# =============================================================================


class TestSelectEncoding:
    """Test compression encoding selection."""

    @pytest.mark.parametrize(
        "accept_encoding,expected",
        [
            pytest.param("zstd, gzip", "zstd", id="prefers-zstd"),
            pytest.param("gzip", "gzip", id="fallback-gzip"),
            pytest.param("deflate, br", "identity", id="unknown-identity-fallback"),
            pytest.param(None, "gzip", id="none-fallback-gzip"),
            pytest.param("", "gzip", id="empty-fallback-gzip"),
            pytest.param("ZSTD, GZIP", "zstd", id="case-insensitive"),
            pytest.param("  zstd  ,  gzip  ", "zstd", id="whitespace-handling"),
        ],
    )  # fmt: skip
    def test_select_encoding(self, accept_encoding: str | None, expected: str) -> None:
        """Test encoding selection based on Accept-Encoding header."""
        from aiperf.common.compression import CompressionEncoding, select_encoding

        result = select_encoding(accept_encoding)
        assert result == CompressionEncoding(expected)


# =============================================================================
# Service properties
# =============================================================================


class TestServiceBaseUrl:
    """Test the _base_url property."""

    def test_base_url_format(self, mock_fastapi_service: FastAPIService) -> None:
        """Test _base_url returns correct format."""
        mock_fastapi_service.api_host = "0.0.0.0"
        mock_fastapi_service.api_port = 8080

        assert mock_fastapi_service._base_url == "http://0.0.0.0:8080"

    def test_base_url_localhost(self, mock_fastapi_service: FastAPIService) -> None:
        """Test _base_url with localhost."""
        mock_fastapi_service.api_host = "127.0.0.1"
        mock_fastapi_service.api_port = 9999

        assert mock_fastapi_service._base_url == "http://127.0.0.1:9999"


# =============================================================================
# FastAPIService lifecycle tests (init, start, stop, main)
# =============================================================================


@pytest.fixture
def api_service(mock_zmq, api_service_config, api_user_config) -> FastAPIService:
    """Create a real FastAPIService via direct instantiation (ZMQ mocked)."""
    api_service_config.api_port = 9999
    api_service_config.api_host = "127.0.0.1"
    return FastAPIService(
        service_config=api_service_config,
        user_config=api_user_config,
        service_id="api-test",
    )


class TestFastAPIServiceInit:
    """Test FastAPIService.__init__ via direct instantiation."""

    def test_init_sets_host_port_from_config(self, api_service) -> None:
        assert api_service.api_host == "127.0.0.1"
        assert api_service.api_port == 9999

    def test_init_creates_app(self, api_service) -> None:
        assert api_service.app is not None
        assert api_service.app.title == "AIPerf API"

    def test_init_defaults_server_to_none(self, api_service) -> None:
        assert api_service._server is None
        assert api_service._server_task is None

    def test_init_loads_routers(self, api_service) -> None:
        assert len(api_service._routers) > 0

    def test_init_with_custom_host(self, mock_zmq, api_user_config) -> None:
        sc = ServiceConfig(api_port=8080, api_host="0.0.0.0")
        service = FastAPIService(
            service_config=sc,
            user_config=api_user_config,
            service_id="api-custom",
        )
        assert service.api_host == "0.0.0.0"
        assert service.api_port == 8080


class TestFastAPIServiceCORSMiddleware:
    """Test CORS middleware is added when cors_origins is set."""

    def test_cors_middleware_added_when_origins_set(
        self, mock_zmq, api_user_config, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "aiperf.common.environment.Environment.API_SERVER",
            type(
                "_Fake",
                (),
                {"HOST": "127.0.0.1", "PORT": 8080, "CORS_ORIGINS": ["*"]},
            )(),
        )
        sc = ServiceConfig(api_port=8080)
        service = FastAPIService(
            service_config=sc,
            user_config=api_user_config,
            service_id="api-cors",
        )
        middleware_names = [m.cls.__name__ for m in service.app.user_middleware]
        assert "CORSMiddleware" in middleware_names

    def test_no_cors_middleware_when_origins_empty(
        self, mock_zmq, api_user_config, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "aiperf.common.environment.Environment.API_SERVER",
            type(
                "_Fake", (), {"HOST": "127.0.0.1", "PORT": 8080, "CORS_ORIGINS": []}
            )(),
        )
        sc = ServiceConfig(api_port=8080)
        service = FastAPIService(
            service_config=sc,
            user_config=api_user_config,
            service_id="api-no-cors",
        )
        middleware_names = [m.cls.__name__ for m in service.app.user_middleware]
        assert "CORSMiddleware" not in middleware_names


class TestFastAPIServiceStartStop:
    """Test _start_api_server and _stop_api_server."""

    @pytest.mark.asyncio
    async def test_start_creates_server_and_task(self, api_service) -> None:
        mock_server = MagicMock()
        mock_server.serve = AsyncMock()

        with (
            patch("aiperf.api.api_service.uvicorn.Config"),
            patch("aiperf.api.api_service.uvicorn.Server", return_value=mock_server),
        ):
            await api_service._start_api_server()

        assert api_service._server is mock_server
        assert api_service._server_task is not None

        api_service._server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await api_service._server_task

    @pytest.mark.asyncio
    async def test_stop_sets_should_exit_and_waits(self, api_service) -> None:
        mock_server = MagicMock()
        completed = asyncio.Event()

        async def fake_serve():
            await completed.wait()

        task = asyncio.create_task(fake_serve())
        api_service._server = mock_server
        api_service._server_task = task

        completed.set()
        await api_service._stop_api_server()

        assert mock_server.should_exit is True

    @pytest.mark.asyncio
    async def test_stop_cancels_on_timeout(self, api_service) -> None:
        mock_server = MagicMock()

        async def hang_forever():
            await asyncio.Future()

        task = asyncio.create_task(hang_forever())
        api_service._server = mock_server
        api_service._server_task = task

        real_wait_for = asyncio.wait_for
        call_count = 0

        async def first_call_times_out(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError
            return await real_wait_for(*args, **kwargs)

        with patch(
            "aiperf.api.api_service.asyncio.wait_for",
            side_effect=first_call_times_out,
        ):
            await api_service._stop_api_server()

        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_handles_no_server(self, api_service) -> None:
        api_service._server = None
        api_service._server_task = None
        await api_service._stop_api_server()

    @pytest.mark.asyncio
    async def test_stop_handles_cancelled_error(self, api_service) -> None:
        """Test _stop_api_server handles CancelledError from wait_for."""
        mock_server = MagicMock()
        api_service._server = mock_server
        api_service._server_task = asyncio.create_task(asyncio.sleep(100))

        with patch(
            "aiperf.api.api_service.asyncio.wait_for",
            side_effect=asyncio.CancelledError,
        ):
            await api_service._stop_api_server()

        assert mock_server.should_exit is True

    async def test_on_server_task_done_schedules_stop_on_exception(
        self, api_service
    ) -> None:
        """Test _on_server_task_done schedules stop when server task fails."""
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("server crashed")

        with patch.object(api_service, "stop", new_callable=AsyncMock) as mock_stop:
            api_service._on_server_task_done(task)
            assert api_service._stop_task is not None
            await asyncio.sleep(0)
            mock_stop.assert_called_once()

    def test_on_server_task_done_ignores_cancelled(self, api_service) -> None:
        """Test _on_server_task_done does nothing for cancelled tasks."""
        task = MagicMock()
        task.cancelled.return_value = True
        api_service._on_server_task_done(task)
        task.exception.assert_not_called()
        assert api_service._stop_task is None

    def test_on_server_task_done_no_exception(self, api_service) -> None:
        """Test _on_server_task_done does nothing when task succeeds."""
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        api_service._on_server_task_done(task)
        assert api_service._stop_task is None


class TestFastAPIServiceLifespan:
    """Test FastAPI lifespan hooks."""

    def test_lifespan_logs_startup_and_shutdown(self, api_service) -> None:
        """Test that lifespan logs on startup and shutdown."""
        api_service.info = MagicMock()

        with TestClient(api_service.app):
            pass

        info_calls = [call[0][0] for call in api_service.info.call_args_list]
        assert any("FastAPI starting" in msg for msg in info_calls)
        assert any("FastAPI stopped" in msg for msg in info_calls)


class TestFastAPIServiceMain:
    """Test the main() entry point."""

    def test_main_calls_bootstrap(self) -> None:
        from aiperf.api.api_service import main
        from aiperf.plugin.enums import ServiceType

        with patch(
            "aiperf.api.api_service.bootstrap_and_run_service"
        ) as mock_bootstrap:
            main()
            mock_bootstrap.assert_called_once_with(ServiceType.API)
