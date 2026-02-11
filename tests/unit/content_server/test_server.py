# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ContentServer service."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from aiperf.content_server.request_tracker import RequestTracker
from aiperf.content_server.server import ContentServer

# ---------------------------------------------------------------------------
# Helpers to build a FastAPI test client from a ContentServer instance
# ---------------------------------------------------------------------------


def _make_server(
    content_dir: Path,
    enabled: bool = True,
    is_subprocess: bool = False,
) -> ContentServer:
    """Create a ContentServer with mocked service infrastructure."""
    with (
        patch("aiperf.content_server.server.Environment") as mock_env,
        patch.object(ContentServer, "__init__", lambda self, *a, **kw: None),
    ):
        server = ContentServer.__new__(ContentServer)
        # Manually set the attributes that __init__ would set
        settings = MagicMock()
        settings.ENABLED = enabled
        settings.HOST = "127.0.0.1"
        settings.PORT = 8090
        settings.CONTENT_DIR = str(content_dir)
        settings.MAX_TRACKED_RECORDS = 10000
        mock_env.CONTENT_SERVER = settings

        server._settings = settings
        server._app = None
        server._uvicorn_server = None
        server._content_dir = None
        server._temp_dir = None
        server._tracker = RequestTracker(max_records=10000)
        server._base_url = ""

        # Mock service methods
        server.service_id = "test-content-server"
        server.info = MagicMock()
        server.debug = MagicMock()
        server.error = MagicMock()
        server.warning = MagicMock()
        server.publish = AsyncMock()
        server.execute_async = MagicMock()

    return server


def _mock_uvicorn_server() -> MagicMock:
    """Create a mock uvicorn.Server that reports as started immediately."""
    mock = MagicMock()
    mock.started = True
    mock.should_exit = False
    mock.serve = AsyncMock()
    return mock


async def _init_server(server: ContentServer) -> None:
    """Run the _initialize hook on a ContentServer."""
    # Patch parent_process to simulate main process
    with patch("aiperf.content_server.server.parent_process", return_value=None):
        await server._initialize()


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestContentServerInit:
    async def test_init_disabled(self, content_dir: Path) -> None:
        server = _make_server(content_dir, enabled=False)
        await _init_server(server)
        assert server._app is None

    async def test_init_enabled(self, content_dir: Path) -> None:
        server = _make_server(content_dir, enabled=True)
        await _init_server(server)
        assert server._app is not None
        assert server._content_dir == content_dir

    async def test_init_skipped_in_subprocess(self, content_dir: Path) -> None:
        server = _make_server(content_dir, enabled=True)
        with patch(
            "aiperf.content_server.server.parent_process",
            return_value=MagicMock(),
        ):
            await server._initialize()
        assert server._app is None

    async def test_init_invalid_content_dir(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist"
        server = _make_server(nonexistent, enabled=True)
        await _init_server(server)
        assert server._app is None
        server.error.assert_called_once()

    async def test_init_empty_content_dir_creates_temp(self, tmp_path: Path) -> None:
        """When CONTENT_DIR is empty, a temp directory should be created."""
        server = _make_server(tmp_path, enabled=True)
        server._settings.CONTENT_DIR = ""
        with patch("aiperf.content_server.server.parent_process", return_value=None):
            await server._initialize()
        assert server._app is not None
        assert server._temp_dir is not None
        assert server._content_dir is not None


# ---------------------------------------------------------------------------
# Tests: File Serving (via Starlette TestClient)
# ---------------------------------------------------------------------------


class TestContentServerFileServing:
    @pytest.fixture
    async def test_client(self, content_dir: Path) -> TestClient:
        """Create a Starlette TestClient backed by the ContentServer's FastAPI app."""
        server = _make_server(content_dir, enabled=True)
        await _init_server(server)
        assert server._app is not None
        return TestClient(server._app)

    def test_serve_existing_file(
        self, test_client: TestClient, content_dir: Path
    ) -> None:
        response = test_client.get("/content/readme.txt")
        assert response.status_code == 200
        assert response.text == "test content"

    def test_serve_nested_file(
        self, test_client: TestClient, content_dir: Path
    ) -> None:
        response = test_client.get("/content/images/test.png")
        assert response.status_code == 200
        assert "image/png" in response.headers["content-type"]

    def test_serve_jpeg(self, test_client: TestClient, content_dir: Path) -> None:
        response = test_client.get("/content/images/photo.jpeg")
        assert response.status_code == 200
        assert "jpeg" in response.headers["content-type"]

    def test_serve_wav(self, test_client: TestClient, content_dir: Path) -> None:
        response = test_client.get("/content/audio/clip.wav")
        assert response.status_code == 200

    def test_not_found(self, test_client: TestClient) -> None:
        response = test_client.get("/content/nonexistent.txt")
        assert response.status_code == 404

    def test_path_traversal_blocked(self, test_client: TestClient) -> None:
        response = test_client.get("/content/../../../etc/passwd")
        assert response.status_code in (403, 404)

    def test_health_endpoint(self, test_client: TestClient) -> None:
        response = test_client.get("/healthz")
        assert response.status_code == 200
        assert response.text == "ok"


# ---------------------------------------------------------------------------
# Tests: Request Tracking
# ---------------------------------------------------------------------------


class TestContentServerTracking:
    async def test_requests_tracked(self, content_dir: Path) -> None:
        server = _make_server(content_dir, enabled=True)
        await _init_server(server)
        assert server._app is not None

        client = TestClient(server._app)
        client.get("/content/readme.txt")
        client.get("/content/nonexistent.txt")

        snapshot = server._tracker.snapshot()
        assert snapshot.total_requests == 2
        # One 200, one 404
        statuses = {r.status_code for r in snapshot.records}
        assert 200 in statuses
        assert 404 in statuses


# ---------------------------------------------------------------------------
# Tests: Lifecycle (start / stop)
# ---------------------------------------------------------------------------


class TestContentServerLifecycle:
    async def test_start_publishes_status_when_disabled(
        self, content_dir: Path
    ) -> None:
        server = _make_server(content_dir, enabled=False)
        await _init_server(server)
        await server._start_http_server()
        server.publish.assert_called_once()
        msg = server.publish.call_args[0][0]
        assert msg.status.enabled is False

    async def test_start_publishes_status_when_enabled(self, content_dir: Path) -> None:
        server = _make_server(content_dir, enabled=True)
        await _init_server(server)
        with patch("aiperf.content_server.server.uvicorn") as mock_uvicorn:
            mock_uvicorn.Config.return_value = MagicMock()
            mock_uvicorn.Server.return_value = _mock_uvicorn_server()
            await server._start_http_server()
        server.publish.assert_called_once()
        msg = server.publish.call_args[0][0]
        assert msg.status.enabled is True
        assert "8090" in msg.status.base_url

    async def test_stop_cleans_up(self, content_dir: Path) -> None:
        server = _make_server(content_dir, enabled=True)
        await _init_server(server)
        with patch("aiperf.content_server.server.uvicorn") as mock_uvicorn:
            mock_uvicorn.Config.return_value = MagicMock()
            mock_uvicorn.Server.return_value = _mock_uvicorn_server()
            await server._start_http_server()
        assert server._uvicorn_server is not None
        await server._stop_http_server()
        assert server._uvicorn_server is None

    async def test_stop_cleans_temp_dir(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, enabled=True)
        server._settings.CONTENT_DIR = ""
        with patch("aiperf.content_server.server.parent_process", return_value=None):
            await server._initialize()
        assert server._temp_dir is not None
        await server._stop_http_server()
        assert server._temp_dir is None
