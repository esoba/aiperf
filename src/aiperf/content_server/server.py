# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content server service for serving multimodal files over HTTP.

Serves images, audio, and video files via HTTP so target LLM servers can
fetch content by URL instead of requiring base64-encoded payloads.
"""

from __future__ import annotations

import asyncio
import mimetypes
import tempfile
from multiprocessing import parent_process
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import FileResponse, PlainTextResponse, Response

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_init, on_start, on_stop
from aiperf.common.messages import ContentServerStatusMessage
from aiperf.content_server.models import ContentServerStatus
from aiperf.content_server.request_tracker import RequestTracker, TrackingMiddleware


class ContentServer(BaseComponentService):
    """HTTP file server for multimodal content.

    Serves files from a configured directory over HTTP using FastAPI/uvicorn,
    allowing LLM inference servers to fetch images, audio, and video by URL.
    Publishes ContentServerStatusMessage on start so other services know
    the base URL.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
        )
        self._settings = Environment.CONTENT_SERVER
        self._app: FastAPI | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._content_dir: Path | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._tracker: RequestTracker = RequestTracker(
            max_records=self._settings.MAX_TRACKED_RECORDS
        )
        self._base_url: str = ""

    @on_init
    async def _initialize(self) -> None:
        """Create FastAPI app, configure routes, validate content_dir."""
        if not self._settings.ENABLED:
            self.info("Content server is disabled")
            return

        # Skip in spawned child processes to avoid port conflicts
        if parent_process() is not None:
            self.debug("Content server skipped in subprocess")
            return

        # Resolve content directory
        content_dir_str = self._settings.CONTENT_DIR
        if content_dir_str:
            self._content_dir = Path(content_dir_str).resolve()
            if not self._content_dir.is_dir():
                self.error(
                    f"Content directory does not exist: {self._content_dir}. "
                    "Set AIPERF_CONTENT_SERVER_CONTENT_DIR to a valid directory."
                )
                return
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="aiperf_content_")
            self._content_dir = Path(self._temp_dir.name)
            self.info(f"Content server using temporary directory: {self._content_dir}")

        # Create FastAPI app with tracking middleware
        self._app = FastAPI(
            title="AIPerf Content Server",
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )
        self._app.add_middleware(TrackingMiddleware, tracker=self._tracker)
        self._app.add_api_route("/healthz", self._health, methods=["GET"])
        self._app.add_api_route(
            "/content/{file_path:path}", self._serve_file, methods=["GET"]
        )
        self.debug("Content server initialized")

    @on_start
    async def _start_http_server(self) -> None:
        """Start uvicorn server and publish status message."""
        if self._app is None:
            # Disabled or init failed — publish disabled status
            await self._publish_status(enabled=False, reason="not initialized")
            return

        host = self._settings.HOST
        port = self._settings.PORT
        self._base_url = f"http://{host}:{port}"

        config = uvicorn.Config(
            app=self._app,
            host=host,
            port=port,
            log_level="warning",
        )
        self._uvicorn_server = uvicorn.Server(config)

        self.execute_async(self._uvicorn_server.serve())

        # Wait for uvicorn to bind the socket before publishing readiness
        for _ in range(100):  # up to 5 seconds (100 × 50ms)
            if self._uvicorn_server.started:
                break
            await asyncio.sleep(0.05)

        if not self._uvicorn_server.started:
            self.error("Content server failed to start within timeout")
            await self._publish_status(enabled=False, reason="start timeout")
            return

        self.info(f"Content server started on {self._base_url}")
        await self._publish_status(enabled=True)

    @on_stop
    async def _stop_http_server(self) -> None:
        """Signal uvicorn to exit and clean up temp directory."""
        if self._uvicorn_server is not None:
            self.debug("Stopping content server...")
            self._uvicorn_server.should_exit = True
            self._uvicorn_server = None

        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

        self.debug("Content server stopped")

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _serve_file(self, request: Request, file_path: str) -> Response:
        """Serve a file from the content directory with path traversal prevention."""
        if self._content_dir is None:
            return PlainTextResponse("Not configured", status_code=500)

        resolved = (self._content_dir / file_path).resolve()

        # Path traversal prevention — is_relative_to handles edge cases like
        # content_dir="/tmp/content" vs resolved="/tmp/content_evil/..."
        if not resolved.is_relative_to(self._content_dir):
            return PlainTextResponse("Forbidden", status_code=403)

        if not resolved.is_file():
            return PlainTextResponse("Not Found", status_code=404)

        content_type, _ = mimetypes.guess_type(str(resolved))
        content_type = content_type or "application/octet-stream"

        return FileResponse(
            path=str(resolved),
            media_type=content_type,
        )

    async def _health(self, request: Request) -> Response:
        """Simple health check endpoint."""
        return PlainTextResponse("ok")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _publish_status(self, enabled: bool, reason: str | None = None) -> None:
        """Publish content server status to the message bus."""
        status = ContentServerStatus(
            enabled=enabled,
            base_url=self._base_url,
            content_dir=str(self._content_dir or ""),
            reason=reason,
        )
        await self.publish(
            ContentServerStatusMessage(
                service_id=self.service_id,
                status=status,
            )
        )
