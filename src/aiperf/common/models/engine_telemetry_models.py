# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


class EngineIterationStats(AIPerfBaseModel):
    """A single telemetry snapshot from an in-engine transport's runtime.

    Captures per-iteration engine metrics (batch size, token counts, queue depth)
    alongside engine-specific raw data for downstream analysis and export.
    """

    timestamp_ns: int = Field(description="Wall clock timestamp in nanoseconds.")
    batch_size: int | None = Field(
        default=None, description="Number of requests in current batch."
    )
    num_tokens: int | None = Field(
        default=None, description="Total tokens processed in iteration."
    )
    queue_depth: int | None = Field(
        default=None, description="Requests waiting in engine queue."
    )
    raw: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific raw stats dict."
    )
