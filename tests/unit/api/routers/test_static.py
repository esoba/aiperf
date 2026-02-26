# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the static file router (index, dashboard, path traversal)."""

from unittest.mock import AsyncMock, patch

import pytest
from pytest import param
from starlette.testclient import TestClient

from aiperf.api.api_service import FastAPIService
from aiperf.api.routers.static import _read_static


class TestStaticFileServing:
    """Test static file serving with path traversal protection."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filename",
        [
            param("../secret.txt", id="parent-dir"),
            param("../../etc/passwd", id="etc-passwd"),
            param("static/../../../secret.txt", id="nested-traversal"),
            param("foo/../../../etc/passwd", id="deep-traversal"),
        ],
    )  # fmt: skip
    async def test_path_traversal_blocked(self, filename: str) -> None:
        """Test that path traversal attempts are blocked with 400."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await _read_static(filename)
        assert exc_info.value.status_code == 400
        assert "Invalid filename" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_404(self) -> None:
        """Test that non-existent files return 404."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await _read_static("nonexistent.html")
        assert exc_info.value.status_code == 404


class TestStaticPageEndpoints:
    """Test the static page serving endpoints."""

    def test_index_page_returns_html(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test index page serves HTML."""
        with patch(
            "aiperf.api.routers.static._read_static",
            new_callable=AsyncMock,
            return_value="<html>Index</html>",
        ):
            response = api_test_client.get("/")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_dashboard_page_returns_html(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test dashboard page serves HTML."""
        with patch(
            "aiperf.api.routers.static._read_static",
            new_callable=AsyncMock,
            return_value="<html>Dashboard</html>",
        ):
            response = api_test_client.get("/dashboard")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
