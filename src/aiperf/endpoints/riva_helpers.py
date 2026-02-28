# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Riva endpoint implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.endpoints.base_endpoint import BaseEndpoint


def get_extra(endpoint: BaseEndpoint) -> dict[str, Any]:
    """Extract extra params from endpoint config as a plain dict."""
    return (
        dict(endpoint.model_endpoint.endpoint.extra)
        if endpoint.model_endpoint.endpoint.extra
        else {}
    )
