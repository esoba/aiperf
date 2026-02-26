# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static router for AIPerf API.

Provides endpoints for serving static HTML files for the dashboard and index pages.
"""

from __future__ import annotations

import pathlib

import aiofiles
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from aiperf.api.routers.base_router import BaseRouter

static_router = APIRouter(tags=["Static"])

_STATIC_DIR = (pathlib.Path(__file__).parent.parent / "static").resolve()


class StaticRouter(BaseRouter):
    """Static HTML file serving for dashboard and index pages."""

    def get_router(self) -> APIRouter:
        return static_router


async def _read_static(filename: str) -> str:
    """Read a static file with path traversal protection."""
    file_path = (_STATIC_DIR / filename).resolve()
    if not file_path.is_relative_to(_STATIC_DIR):
        raise HTTPException(400, "Invalid filename")

    try:
        async with aiofiles.open(file_path, encoding="utf-8") as f:
            return await f.read()
    except FileNotFoundError:
        raise HTTPException(404, f"{filename} not found") from None


@static_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve the index page."""
    return HTMLResponse(await _read_static("index.html"))


@static_router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """Serve the dashboard page."""
    return HTMLResponse(await _read_static("dashboard.html"))
