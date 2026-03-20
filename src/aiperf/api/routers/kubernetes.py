# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes router for AIPerf API.

Provides endpoints for Kubernetes health and readiness probes.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from aiperf.api.depends import ServiceDep

kubernetes_router = APIRouter(tags=["Kubernetes"], include_in_schema=False)


@kubernetes_router.get("/healthz")
async def healthz(svc: ServiceDep) -> Response:
    """Kubernetes liveness probe.

    Returns 200 if the service is alive and not deadlocked.
    Returns 503 if the service is in a FAILED state and should be restarted.
    """
    if svc.is_healthy():
        return Response(status_code=200, content="ok")
    return Response(status_code=503, content="unhealthy")


@kubernetes_router.get("/readyz")
async def readyz(svc: ServiceDep) -> Response:
    """Kubernetes readiness probe.

    Returns 200 if the service is ready to accept traffic (RUNNING state).
    Returns 503 if the service is not yet ready (still initializing).
    """
    if svc.is_ready():
        return Response(status_code=200, content="ok")
    return Response(status_code=503, content="not ready")
