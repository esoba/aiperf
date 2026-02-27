# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.models import ErrorDetails

if TYPE_CHECKING:
    from aiperf.common.models import (
        ErrorDetailsCount,
        MetricResult,
        ServerMetricsRecord,
        ServerMetricsResults,
        TimeRangeFilter,
    )

# Source identifier for system metrics collector
SYSTEM_METRICS_SOURCE_IDENTIFIER = "system://localhost"


# =============================================================================
# Collector Protocol (for ServerMetricsManager to use interchangeably)
# =============================================================================


@runtime_checkable
class ServerMetricsCollectorProtocol(Protocol):
    """Protocol for server metrics collectors.

    Defines the interface for collectors that gather server-side metrics from various
    sources (Prometheus HTTP endpoints, system-level psutil/pynvml, etc.) and deliver
    them via callbacks as ServerMetricsRecord objects.
    """

    @property
    def id(self) -> str:
        """Get the collector's unique identifier."""
        ...

    @property
    def endpoint_url(self) -> str:
        """Get the source identifier (URL for Prometheus, 'system://localhost' for system)."""
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

        For Prometheus: Tests HTTP endpoint reachability.
        For system: Tests psutil/pynvml availability.

        Returns:
            True if the source is available and ready for collection.
        """
        ...

    async def collect_and_process_metrics(self) -> None:
        """Collect metrics and process them into records.

        Called by the manager for baseline capture and final scrape.
        """
        ...


# Type aliases for callbacks
TServerMetricsRecordCallback = Callable[
    [list["ServerMetricsRecord"], str], Awaitable[None]
]
TServerMetricsErrorCallback = Callable[[ErrorDetails, str], Awaitable[None]]


# =============================================================================
# Processor Protocols (for RecordsManager pipeline)
# =============================================================================


@runtime_checkable
class ServerMetricsProcessorProtocol(Protocol):
    """Protocol for server metrics results processors that handle ServerMetricsRecord objects.

    This protocol is separate from ResultsProcessorProtocol because server metrics data
    has fundamentally different structure (hierarchical Prometheus snapshots) compared
    to inference metrics (flat key-value pairs).
    """

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        """Process individual server metrics record with complete Prometheus snapshot.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics snapshot and metadata
        """
        ...

    async def summarize(self) -> list[MetricResult]: ...


@runtime_checkable
class ServerMetricsAccumulatorProtocol(ServerMetricsProcessorProtocol, Protocol):
    """Protocol for server metrics accumulators that accumulate server metrics data and export aggregated results.

    Extends ServerMetricsProcessorProtocol to provide result export functionality with time filtering
    and error summary support. Implementations should accumulate Prometheus snapshot data and compute
    aggregated statistics (mean, p50, p90, p95, p99) for configured metrics across collection windows.
    """

    async def export_results(
        self,
        start_ns: int,
        end_ns: int,
        time_filter: TimeRangeFilter | None = None,
        error_summary: list[ErrorDetailsCount] | None = None,
    ) -> ServerMetricsResults | None:
        """Export accumulated server metrics as results.

        Args:
            start_ns: Start time of collection in nanoseconds
            end_ns: End time of collection in nanoseconds
            time_filter: Optional time filter for aggregation (excludes warmup/buffer)
            error_summary: Optional list of error counts

        Returns:
            ServerMetricsResults if data was collected, None otherwise
        """
        ...
