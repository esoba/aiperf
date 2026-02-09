# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.models import ErrorDetails, TelemetryRecord

if TYPE_CHECKING:
    from aiperf.gpu_telemetry.accumulator import TelemetryMetricsSummary


@runtime_checkable
class GPUTelemetryCollectorProtocol(Protocol):
    """Protocol for GPU telemetry collectors.

    Defines the interface for collectors that gather GPU metrics from various sources
    (DCGM HTTP endpoints, pynvml library, etc.) and deliver them via callbacks.
    """

    @property
    def id(self) -> str:
        """Get the collector's unique identifier."""
        ...

    @property
    def endpoint_url(self) -> str:
        """Get the source identifier (URL for DCGM, 'pynvml://localhost' for pynvml)."""
        ...

    async def initialize(self) -> None:
        """Initialize the collector resources."""
        ...

    async def start(self) -> None:
        """Start the background collection task."""
        ...

    async def stop(self) -> None:
        """Stop the collector and clean up resources."""
        ...

    async def is_url_reachable(self) -> bool:
        """Check if the collector source is available.

        For DCGM: Tests HTTP endpoint reachability.
        For pynvml: Tests NVML library initialization.

        Returns:
            True if the source is available and ready for collection.
        """
        ...


# Type aliases for callbacks
TRecordCallback = Callable[[list[TelemetryRecord], str], Awaitable[None]]
TErrorCallback = Callable[[ErrorDetails, str], Awaitable[None]]


@runtime_checkable
class GPUTelemetryAccumulatorProtocol(Protocol):
    """Protocol for GPU telemetry accumulator methods beyond AccumulatorProtocol.

    export_results() is now on AccumulatorProtocol. This protocol captures
    telemetry-specific methods needed by RecordsManager for realtime dashboard.
    """

    def start_realtime_telemetry(self) -> None:
        """Start the realtime telemetry background task.

        This is called when the user dynamically enables the telemetry dashboard
        by pressing the telemetry option in the UI without having passed the 'dashboard' parameter
        at startup.
        """

    async def summarize(self) -> TelemetryMetricsSummary:
        """Generate telemetry summary with hierarchical tags for telemetry data.

        Returns:
            TelemetryMetricsSummary containing MetricResult objects with hierarchical
            tags that preserve dcgm_url -> gpu_uuid grouping structure for dashboard filtering.
        """
        ...
