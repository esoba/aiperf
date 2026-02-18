# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady-state detection and windowed metric computation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import Field

from aiperf.analysis.ramp_detection import (
    cusum_steady_state_window,
    detect_steady_state_window,
    manual_steady_state_window,
    mser5_boundary_ns,
)
from aiperf.analysis.stationarity import batch_means_trend_test
from aiperf.analysis.sweepline import (
    SweepLineCurves,
    concurrency_sweep_line,
    divide_step_functions,
    prefill_throughput_sweep_line,
    throughput_sweep_line,
    total_throughput_sweep_line,
)
from aiperf.common.config import UserConfig
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PluginDisabled
from aiperf.common.models import MetricResult
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.types import MetricTagT

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import SummaryContext
    from aiperf.metrics.accumulator import MetricsAccumulator
    from aiperf.plugin.enums import AccumulatorType

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
    fraction_retained: float = Field(
        description="Fraction of total requests retained in the steady-state window"
    )

    # Sample quality
    variance_inflation_factor: float = Field(
        description="Approximate variance inflation from truncation: total_requests / steady_state_requests",
    )
    effective_p99_sample_size: int = Field(
        description="Approximate observations contributing to p99: int(steady_state_requests * 0.01)",
    )
    sample_size_warning: bool = Field(
        default=False,
        description="True if effective p99 sample size < 10 (p99 estimate unreliable)",
    )

    # Stationarity validation
    trend_correlation: float | None = Field(
        default=None,
        description="Spearman rank correlation of batch means (latency trend test)",
    )
    trend_p_value: float | None = Field(
        default=None,
        description="P-value of the batch means trend test",
    )
    stationarity_warning: bool = Field(
        default=False,
        description="True if windowed latency shows a statistically significant trend",
    )

    # Per-signal boundaries (diagnostic)
    cusum_ramp_up_end_ns: float | None = Field(
        default=None, description="CUSUM-detected ramp-up end timestamp (ns)"
    )
    cusum_ramp_down_start_ns: float | None = Field(
        default=None, description="CUSUM-detected ramp-down start timestamp (ns)"
    )
    mser5_latency_ramp_up_end_ns: float | None = Field(
        default=None,
        description="MSER-5 latency-detected ramp-up end timestamp (ns)",
    )
    mser5_latency_ramp_down_start_ns: float | None = Field(
        default=None,
        description="MSER-5 latency-detected ramp-down start timestamp (ns)",
    )
    mser5_ttft_ramp_up_end_ns: float | None = Field(
        default=None,
        description="MSER-5 TTFT-detected ramp-up end timestamp (ns)",
    )
    mser5_ttft_ramp_down_start_ns: float | None = Field(
        default=None,
        description="MSER-5 TTFT-detected ramp-down start timestamp (ns)",
    )
    cusum_throughput_ramp_up_end_ns: float | None = Field(
        default=None,
        description="CUSUM throughput-detected ramp-up end timestamp (ns)",
    )
    cusum_throughput_ramp_down_start_ns: float | None = Field(
        default=None,
        description="CUSUM throughput-detected ramp-down start timestamp (ns)",
    )

    # Bootstrap confidence intervals (optional, only when bootstrap_iterations is set)
    bootstrap_ci_ramp_up_ns: tuple[float, float] | None = Field(
        default=None,
        description="Bootstrap confidence interval for ramp-up boundary (ns)",
    )
    bootstrap_ci_ramp_down_ns: tuple[float, float] | None = Field(
        default=None,
        description="Bootstrap confidence interval for ramp-down boundary (ns)",
    )
    bootstrap_ci_mean_latency: tuple[float, float] | None = Field(
        default=None,
        description="Bootstrap confidence interval for mean latency within window",
    )
    bootstrap_ci_p99_latency: tuple[float, float] | None = Field(
        default=None,
        description="Bootstrap confidence interval for p99 latency within window",
    )
    bootstrap_n_iterations: int | None = Field(
        default=None,
        description="Number of bootstrap iterations performed",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to structured JSON-ready dictionary."""
        data: dict[str, Any] = {
            "detection_method": self.detection_method,
            "ramp_up_end_ns": self.ramp_up_end_ns,
            "ramp_down_start_ns": self.ramp_down_start_ns,
            "steady_state_duration_ns": self.steady_state_duration_ns,
            "total_requests": self.total_requests,
            "steady_state_requests": self.steady_state_requests,
            "quality": {
                "fraction_retained": self.fraction_retained,
                "variance_inflation_factor": self.variance_inflation_factor,
                "effective_p99_sample_size": self.effective_p99_sample_size,
                "sample_size_warning": self.sample_size_warning,
            },
            "stationarity": {
                "trend_correlation": self.trend_correlation,
                "trend_p_value": self.trend_p_value,
                "stationarity_warning": self.stationarity_warning,
            },
            "cross_validation": {
                "cusum_ramp_up_end_ns": self.cusum_ramp_up_end_ns,
                "cusum_ramp_down_start_ns": self.cusum_ramp_down_start_ns,
                "mser5_latency_ramp_up_end_ns": self.mser5_latency_ramp_up_end_ns,
                "mser5_latency_ramp_down_start_ns": self.mser5_latency_ramp_down_start_ns,
                "mser5_ttft_ramp_up_end_ns": self.mser5_ttft_ramp_up_end_ns,
                "mser5_ttft_ramp_down_start_ns": self.mser5_ttft_ramp_down_start_ns,
                "cusum_throughput_ramp_up_end_ns": self.cusum_throughput_ramp_up_end_ns,
                "cusum_throughput_ramp_down_start_ns": self.cusum_throughput_ramp_down_start_ns,
            },
        }
        if self.bootstrap_n_iterations is not None:
            data["bootstrap"] = {
                "n_iterations": self.bootstrap_n_iterations,
                "ci_ramp_up_ns": self.bootstrap_ci_ramp_up_ns,
                "ci_ramp_down_ns": self.bootstrap_ci_ramp_down_ns,
                "ci_mean_latency": self.bootstrap_ci_mean_latency,
                "ci_p99_latency": self.bootstrap_ci_p99_latency,
            }
        return data


class SteadyStateSummary(AIPerfBaseModel):
    """Typed result from SteadyStateAnalyzer.summarize()."""

    results: dict[MetricTagT, MetricResult] = Field(
        description="Metric results within the steady-state window"
    )
    effective_concurrency: MetricResult = Field(
        description="Time-weighted concurrency statistics during steady state"
    )
    effective_throughput: MetricResult = Field(
        description="Time-weighted throughput statistics during steady state"
    )
    effective_prefill_throughput: MetricResult = Field(
        description="Time-weighted prefill throughput statistics during steady state"
    )
    effective_generation_concurrency: MetricResult = Field(
        description="Time-weighted generation-phase concurrency statistics during steady state"
    )
    effective_prefill_concurrency: MetricResult = Field(
        description="Time-weighted prefill-phase concurrency statistics during steady state"
    )
    effective_total_throughput: MetricResult = Field(
        description="Time-weighted total throughput (prefill + generation) during steady state"
    )
    effective_throughput_per_user: MetricResult = Field(
        description="Time-weighted per-user throughput statistics during steady state"
    )
    effective_prefill_throughput_per_user: MetricResult = Field(
        description="Time-weighted per-user prefill throughput statistics during steady state"
    )
    tokens_in_flight: MetricResult = Field(
        description="Time-weighted tokens in flight (GPU memory/compute pressure) during steady state"
    )
    window_metadata: SteadyStateWindowMetadata = Field(
        description="Metadata about the detected steady-state window"
    )

    @property
    def sweep_metrics(self) -> dict[str, MetricResult]:
        """Return all sweep MetricResults keyed by tag."""
        return {
            self.effective_concurrency.tag: self.effective_concurrency,
            self.effective_throughput.tag: self.effective_throughput,
            self.effective_prefill_throughput.tag: self.effective_prefill_throughput,
            self.effective_generation_concurrency.tag: self.effective_generation_concurrency,
            self.effective_prefill_concurrency.tag: self.effective_prefill_concurrency,
            self.effective_total_throughput.tag: self.effective_total_throughput,
            self.effective_throughput_per_user.tag: self.effective_throughput_per_user,
            self.effective_prefill_throughput_per_user.tag: self.effective_prefill_throughput_per_user,
            self.tokens_in_flight.tag: self.tokens_in_flight,
        }

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "results": [r.to_json_result().model_dump() for r in self.results.values()],
            "effective_concurrency": self.effective_concurrency.to_json_result().model_dump(),
            "effective_throughput": self.effective_throughput.to_json_result().model_dump(),
            "effective_prefill_throughput": self.effective_prefill_throughput.to_json_result().model_dump(),
            "effective_generation_concurrency": self.effective_generation_concurrency.to_json_result().model_dump(),
            "effective_prefill_concurrency": self.effective_prefill_concurrency.to_json_result().model_dump(),
            "effective_total_throughput": self.effective_total_throughput.to_json_result().model_dump(),
            "effective_throughput_per_user": self.effective_throughput_per_user.to_json_result().model_dump(),
            "effective_prefill_throughput_per_user": self.effective_prefill_throughput_per_user.to_json_result().model_dump(),
            "tokens_in_flight": self.tokens_in_flight.to_json_result().model_dump(),
            "window_metadata": self.window_metadata.to_dict(),
        }
        return data

    def to_csv(self) -> list[dict[str, Any]]:
        return [r.model_dump(exclude={"current"}) for r in self.results.values()]


class SteadyStateAnalyzer:
    """Event-based steady-state detection and windowed metric computation.

    Implements AnalyzerProtocol. No record ingestion — reads columnar
    arrays from MetricsAccumulator at summarize time.
    """

    required_accumulators: ClassVar[set[AccumulatorType]] = {"metric_results"}
    summary_dependencies: ClassVar[list[AccumulatorType]] = ["metric_results"]

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        ss_config = user_config.output.steady_state
        if not ss_config.enabled:
            raise PluginDisabled("Steady-state analysis is disabled")

        env_ss = Environment.STEADY_STATE
        self._min_window_pct = (
            ss_config.min_window_pct
            if "min_window_pct" in ss_config.model_fields_set
            else env_ss.MIN_WINDOW_PCT
        )
        self._start_pct = ss_config.start_pct
        self._end_pct = ss_config.end_pct
        self._bootstrap_iterations = (
            ss_config.bootstrap_iterations
            if "bootstrap_iterations" in ss_config.model_fields_set
            else env_ss.BOOTSTRAP_ITERATIONS
        )

    async def summarize(self, ctx: SummaryContext) -> SteadyStateSummary:
        """Detect steady-state window and compute windowed metrics."""
        from aiperf.metrics.accumulator import MetricsAccumulator
        from aiperf.plugin.enums import AccumulatorType

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

        # Metric columns for MSER-5
        latency = store.numeric("request_latency")
        ttft = store.numeric("time_to_first_token")

        # Concurrency curve (needed for both detection and stats)
        sorted_c_ts, concurrency = concurrency_sweep_line(start_ns, end_ns)

        # Throughput curve (needed for Signal 4: CUSUM on throughput)
        generation_start_ns = store.generation_start_ns[:n]
        output_tokens = store.numeric("output_tokens")
        sorted_t_ts, tput = throughput_sweep_line(
            generation_start_ns, end_ns, output_tokens
        )

        # Per-signal diagnostics
        cusum_start_ns: float | None = None
        cusum_end_ns: float | None = None
        lat_start_ns: float | None = None
        lat_end_ns: float | None = None
        ttft_start_ns: float | None = None
        ttft_end_ns: float | None = None
        tput_start_ns: float | None = None
        tput_end_ns: float | None = None

        # User override or automatic detection
        if self._start_pct is not None and self._end_pct is not None:
            min_ts = float(np.nanmin(start_ns[filled]))
            max_ts = float(np.nanmax(end_ns[filled]))
            window_start, window_end = manual_steady_state_window(
                min_ts, max_ts, self._start_pct, self._end_pct
            )
            detection_method = "user_override"
        else:
            window_start, window_end, detection_method = detect_steady_state_window(
                sorted_c_ts,
                concurrency,
                start_ns,
                end_ns,
                latency,
                ttft,
                min_window_pct=self._min_window_pct,
                sorted_tput_ts=sorted_t_ts if len(sorted_t_ts) > 0 else None,
                throughput=tput if len(sorted_t_ts) > 0 else None,
            )
            # Collect per-signal boundaries for diagnostics
            cusum_start_ns, cusum_end_ns = cusum_steady_state_window(
                sorted_c_ts, concurrency, min_window_pct=0.0
            )
            lat_start_ns, lat_end_ns = mser5_boundary_ns(
                latency, start_ns, end_ns, filled
            )
            ttft_start_ns, ttft_end_ns = mser5_boundary_ns(
                ttft, start_ns, end_ns, filled
            )
            if len(sorted_t_ts) > 0:
                tput_start_ns, tput_end_ns = cusum_steady_state_window(
                    sorted_t_ts, tput, min_window_pct=0.0
                )

        # Prefer ICL-aware throughput when SSE chunk timing is available
        tput_ts, tput_vals = MetricsAccumulator._icl_aware_throughput(
            store, generation_start_ns, end_ns, output_tokens
        )

        input_tokens = store.numeric("input_sequence_length")
        sorted_p_ts, prefill_tput = prefill_throughput_sweep_line(
            start_ns, generation_start_ns, input_tokens
        )

        # Phase-specific concurrency (also used as divisor for per-user metrics)
        gen_conc_ts, gen_conc = concurrency_sweep_line(generation_start_ns, end_ns)
        pre_conc_ts, pre_conc = concurrency_sweep_line(start_ns, generation_start_ns)

        total_ts, total_tput = total_throughput_sweep_line(
            start_ns, generation_start_ns, end_ns, input_tokens, output_tokens
        )
        tpu_ts, tpu_vals = divide_step_functions(
            tput_ts, tput_vals, gen_conc_ts, gen_conc
        )
        ptpu_ts, ptpu_vals = divide_step_functions(
            sorted_p_ts, prefill_tput, pre_conc_ts, pre_conc
        )
        tif_ts, tif_vals = MetricsAccumulator._icl_aware_tokens_in_flight(
            store, start_ns, generation_start_ns, end_ns, input_tokens, output_tokens
        )

        # Build unified sweep curves and compute all metrics in one call
        sweeps = SweepLineCurves(
            concurrency_ts=sorted_c_ts,
            concurrency=concurrency,
            throughput_ts=tput_ts,
            throughput=tput_vals,
            prefill_throughput_ts=sorted_p_ts,
            prefill_throughput=prefill_tput,
            generation_concurrency_ts=gen_conc_ts,
            generation_concurrency=gen_conc,
            prefill_concurrency_ts=pre_conc_ts,
            prefill_concurrency=pre_conc,
            total_throughput_ts=total_ts,
            total_throughput=total_tput,
            throughput_per_user_ts=tpu_ts,
            throughput_per_user=tpu_vals,
            prefill_throughput_per_user_ts=ptpu_ts,
            prefill_throughput_per_user=ptpu_vals,
            tokens_in_flight_ts=tif_ts,
            tokens_in_flight=tif_vals,
        )
        sweep_results = sweeps.compute_metrics(window_start, window_end)

        # Steady-state mask: request started AND ended within window
        ss_mask = filled & (start_ns >= window_start) & (end_ns <= window_end)

        total_requests = int(filled.sum())
        steady_state_requests = int(ss_mask.sum())
        fraction_retained = (
            steady_state_requests / total_requests if total_requests > 0 else 0.0
        )

        # Sample quality
        variance_inflation_factor = (
            total_requests / steady_state_requests
            if steady_state_requests > 0
            else float("inf")
        )
        effective_p99_sample_size = int(steady_state_requests * 0.01)
        sample_size_warning = effective_p99_sample_size < 10

        # Stationarity validation on windowed latency
        trend_rho: float | None = None
        trend_p: float | None = None
        stationarity_warning = False

        windowed_latency = latency[ss_mask]
        valid_latency = windowed_latency[~np.isnan(windowed_latency)]
        if len(valid_latency) >= 10:
            trend_rho, trend_p = batch_means_trend_test(valid_latency)
            stationarity_warning = abs(trend_rho) > 0.65 and trend_p < 0.05

        # Optional bootstrap confidence intervals
        boot_ci_ramp_up: tuple[float, float] | None = None
        boot_ci_ramp_down: tuple[float, float] | None = None
        boot_ci_mean_lat: tuple[float, float] | None = None
        boot_ci_p99_lat: tuple[float, float] | None = None
        boot_n: int | None = None

        if self._bootstrap_iterations is not None and self._bootstrap_iterations > 0:
            from aiperf.analysis.bootstrap import bootstrap_detection

            boot = bootstrap_detection(
                start_ns,
                end_ns,
                latency,
                ttft,
                n_iterations=self._bootstrap_iterations,
                min_window_pct=self._min_window_pct,
                generation_start_ns=generation_start_ns,
                output_tokens=output_tokens,
            )
            boot_ci_ramp_up = boot.ci_ramp_up_ns
            boot_ci_ramp_down = boot.ci_ramp_down_ns
            boot_ci_mean_lat = boot.ci_mean_latency
            boot_ci_p99_lat = boot.ci_p99_latency
            boot_n = boot.n_iterations

        windowed_results = metrics_acc.compute_results_for_mask(
            ss_mask,
            window_start_ns=int(window_start),
            window_end_ns=int(window_end),
        )

        return SteadyStateSummary(
            results=windowed_results,
            effective_concurrency=sweep_results["effective_concurrency"],
            effective_throughput=sweep_results["effective_throughput"],
            effective_prefill_throughput=sweep_results["effective_prefill_throughput"],
            effective_generation_concurrency=sweep_results[
                "effective_generation_concurrency"
            ],
            effective_prefill_concurrency=sweep_results[
                "effective_prefill_concurrency"
            ],
            effective_total_throughput=sweep_results["effective_total_throughput"],
            effective_throughput_per_user=sweep_results[
                "effective_throughput_per_user"
            ],
            effective_prefill_throughput_per_user=sweep_results[
                "effective_prefill_throughput_per_user"
            ],
            tokens_in_flight=sweep_results["tokens_in_flight"],
            window_metadata=SteadyStateWindowMetadata(
                ramp_up_end_ns=window_start,
                ramp_down_start_ns=window_end,
                steady_state_duration_ns=window_end - window_start,
                total_requests=total_requests,
                steady_state_requests=steady_state_requests,
                detection_method=detection_method,
                fraction_retained=fraction_retained,
                variance_inflation_factor=variance_inflation_factor,
                effective_p99_sample_size=effective_p99_sample_size,
                sample_size_warning=sample_size_warning,
                trend_correlation=trend_rho,
                trend_p_value=trend_p,
                stationarity_warning=stationarity_warning,
                cusum_ramp_up_end_ns=cusum_start_ns,
                cusum_ramp_down_start_ns=cusum_end_ns,
                mser5_latency_ramp_up_end_ns=lat_start_ns,
                mser5_latency_ramp_down_start_ns=lat_end_ns,
                mser5_ttft_ramp_up_end_ns=ttft_start_ns,
                mser5_ttft_ramp_down_start_ns=ttft_end_ns,
                cusum_throughput_ramp_up_end_ns=tput_start_ns,
                cusum_throughput_ramp_down_start_ns=tput_end_ns,
                bootstrap_ci_ramp_up_ns=boot_ci_ramp_up,
                bootstrap_ci_ramp_down_ns=boot_ci_ramp_down,
                bootstrap_ci_mean_latency=boot_ci_mean_lat,
                bootstrap_ci_p99_latency=boot_ci_p99_lat,
                bootstrap_n_iterations=boot_n,
            ),
        )
