# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady-state detection and windowed metric computation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import Field

from aiperf.analysis.ramp_detection import (
    detect_steady_state_window,
    manual_steady_state_window,
)
from aiperf.analysis.sweep import compute_time_weighted_stats, concurrency_sweep
from aiperf.common.config import UserConfig
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PluginDisabled
from aiperf.common.models import MetricResult
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.types import MetricTagT

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import SummaryContext
    from aiperf.post_processors.metrics_accumulator import MetricsAccumulator

logger = logging.getLogger(__name__)


class SteadyStateWindowMetadata(AIPerfBaseModel, frozen=True):
    """Diagnostic metadata about the detected steady-state window."""

    ramp_up_end_ns: float = Field(description="Timestamp (ns) when ramp-up ends")
    ramp_down_start_ns: float = Field(
        description="Timestamp (ns) when ramp-down starts"
    )
    steady_state_duration_ns: float = Field(
        description="Duration of the steady-state window in nanoseconds"
    )
    total_requests: int = Field(description="Total requests in the benchmark")
    steady_state_requests: int = Field(
        description="Requests within the steady-state window"
    )
    detection_method: str = Field(description="Method used to detect steady state")


class SteadyStateSummary(AIPerfBaseModel):
    """Typed result from SteadyStateAnalyzer.summarize()."""

    results: dict[MetricTagT, MetricResult] = Field(
        description="Metric results within the steady-state window"
    )
    effective_concurrency: MetricResult = Field(
        description="Time-weighted concurrency statistics during steady state"
    )
    window_metadata: SteadyStateWindowMetadata = Field(
        description="Metadata about the detected steady-state window"
    )

    def to_json(self) -> dict[str, Any]:
        return {
            "results": [r.to_json_result().model_dump() for r in self.results.values()],
            "effective_concurrency": self.effective_concurrency.to_json_result().model_dump(),
            "window_metadata": {
                "ramp_up_end_ns": self.window_metadata.ramp_up_end_ns,
                "ramp_down_start_ns": self.window_metadata.ramp_down_start_ns,
                "steady_state_duration_ns": self.window_metadata.steady_state_duration_ns,
                "total_requests": self.window_metadata.total_requests,
                "steady_state_requests": self.window_metadata.steady_state_requests,
                "detection_method": self.window_metadata.detection_method,
            },
        }

    def to_csv(self) -> list[dict[str, Any]]:
        return [r.model_dump(exclude={"current"}) for r in self.results.values()]


class SteadyStateAnalyzer:
    """Event-based steady-state detection and windowed metric computation.

    Implements AnalyzerProtocol. No record ingestion — reads columnar
    arrays from MetricsAccumulator at summarize time.
    """

    required_accumulators: ClassVar[set[str]] = {"metric_results"}
    summary_dependencies: ClassVar[list[str]] = ["metric_results"]

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        ss_config = user_config.output.steady_state
        if not ss_config.enabled:
            raise PluginDisabled("Steady-state analysis is disabled")

        env_ss = Environment.STEADY_STATE
        self._stability_fraction = (
            ss_config.stability_fraction
            if "stability_fraction" in ss_config.model_fields_set
            else env_ss.STABILITY_FRACTION
        )
        self._sustained_window_pct = (
            ss_config.sustained_window_pct
            if "sustained_window_pct" in ss_config.model_fields_set
            else env_ss.SUSTAINED_WINDOW_PCT
        )
        self._min_window_pct = (
            ss_config.min_window_pct
            if "min_window_pct" in ss_config.model_fields_set
            else env_ss.MIN_WINDOW_PCT
        )
        self._start_pct = ss_config.start_pct
        self._end_pct = ss_config.end_pct

    async def summarize(self, ctx: SummaryContext) -> SteadyStateSummary:
        """Detect steady-state window and compute windowed metrics."""
        from aiperf.plugin.enums import AccumulatorType
        from aiperf.post_processors.metrics_accumulator import MetricsAccumulator

        metrics_acc: MetricsAccumulator | None = ctx.get_accumulator(
            AccumulatorType.METRIC_RESULTS
        )
        if metrics_acc is None or not isinstance(metrics_acc, MetricsAccumulator):
            raise PluginDisabled("MetricsAccumulator not available")

        store = metrics_acc.column_store
        n = store.count
        if n == 0:
            raise PluginDisabled("No records available for steady-state detection")

        start_ns = store.start_ns[:n]
        end_ns = store.end_ns[:n]
        filled = ~np.isnan(start_ns) & ~np.isnan(end_ns)

        if not filled.any():
            raise PluginDisabled("No valid records for steady-state detection")

        # Concurrency curve (needed for both detection and stats)
        sorted_c_ts, concurrency = concurrency_sweep(start_ns, end_ns)

        # User override or automatic detection
        if self._start_pct is not None and self._end_pct is not None:
            min_ts = float(np.nanmin(start_ns[filled]))
            max_ts = float(np.nanmax(end_ns[filled]))
            window_start, window_end = manual_steady_state_window(
                min_ts, max_ts, self._start_pct, self._end_pct
            )
            detection_method = "user_override"
        else:
            window_start, window_end = detect_steady_state_window(
                sorted_c_ts,
                concurrency,
                stability_fraction=self._stability_fraction,
                sustained_window_pct=self._sustained_window_pct,
                min_window_pct=self._min_window_pct,
            )
            detection_method = "concurrency_threshold"

        # Time-weighted concurrency statistics within the window
        conc_stats = compute_time_weighted_stats(
            sorted_c_ts, concurrency, window_start, window_end
        )

        # Steady-state mask: request started AND ended within window
        ss_mask = filled & (start_ns >= window_start) & (end_ns <= window_end)

        windowed_results = metrics_acc.compute_results_for_mask(ss_mask)

        return SteadyStateSummary(
            results=windowed_results,
            effective_concurrency=MetricResult(
                tag="effective_concurrency",
                header="Effective Concurrency",
                unit="requests",
                avg=conc_stats["avg"],
                min=conc_stats["min"],
                max=conc_stats["max"],
                p50=conc_stats["p50"],
                p90=conc_stats["p90"],
                p95=conc_stats["p95"],
                p99=conc_stats["p99"],
                std=conc_stats["std"],
            ),
            window_metadata=SteadyStateWindowMetadata(
                ramp_up_end_ns=window_start,
                ramp_down_start_ns=window_end,
                steady_state_duration_ns=window_end - window_start,
                total_requests=int(filled.sum()),
                steady_state_requests=int(ss_mask.sum()),
                detection_method=detection_method,
            ),
        )
