# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-accumulator energy efficiency metrics analyzer."""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from aiperf.common.config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.exceptions import PluginDisabled
from aiperf.common.models import MetricResult

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import SummaryContext
    from aiperf.common.models.telemetry_models import (
        GpuMetricTimeSeries,
        TimeRangeFilter,
    )
    from aiperf.gpu_telemetry.accumulator import GPUTelemetryAccumulator
    from aiperf.metrics.accumulator import MetricsSummary
    from aiperf.plugin.enums import AccumulatorType

logger = logging.getLogger(__name__)


class EnergySource(str, enum.Enum):
    """How total GPU energy was determined."""

    DCGM_COUNTER = "dcgm_counter"
    POWER_INTEGRATION = "power_integration"
    UNAVAILABLE = "unavailable"


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Return numerator/denominator, or None if either operand is None or denominator <= 0."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


@dataclass
class EnergyEfficiencySummary:
    """Typed result from EnergyEfficiencyAnalyzer.summarize()."""

    # Source data
    total_gpu_energy_j: float = 0.0
    average_gpu_power_w: float = 0.0
    gpu_count: int = 0
    energy_source: EnergySource = EnergySource.UNAVAILABLE

    # Tier 1
    energy_per_output_token_mj: float | None = None
    energy_per_request_j: float | None = None

    # Tier 2
    energy_per_total_token_mj: float | None = None
    performance_per_watt: float | None = None
    output_tps_per_watt: float | None = None
    goodput_per_watt: float | None = None

    # Metric results for export pipeline
    metric_results: dict[str, MetricResult] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialize to JSON-compatible structure."""
        data: dict[str, Any] = {
            "source": {
                "total_gpu_energy_j": self.total_gpu_energy_j,
                "average_gpu_power_w": self.average_gpu_power_w,
                "gpu_count": self.gpu_count,
                "energy_source": self.energy_source.value,
            },
            "metrics": {},
        }
        metrics = data["metrics"]
        if self.energy_per_output_token_mj is not None:
            metrics["energy_per_output_token_mj"] = self.energy_per_output_token_mj
        if self.energy_per_request_j is not None:
            metrics["energy_per_request_j"] = self.energy_per_request_j
        if self.energy_per_total_token_mj is not None:
            metrics["energy_per_total_token_mj"] = self.energy_per_total_token_mj
        if self.performance_per_watt is not None:
            metrics["performance_per_watt"] = self.performance_per_watt
        if self.output_tps_per_watt is not None:
            metrics["output_tps_per_watt"] = self.output_tps_per_watt
        if self.goodput_per_watt is not None:
            metrics["goodput_per_watt"] = self.goodput_per_watt
        if self.metric_results:
            data["results"] = [
                r.to_json_result().model_dump() for r in self.metric_results.values()
            ]
        return data

    def to_csv(self) -> list[dict[str, Any]]:
        """Serialize to list of CSV-compatible row dicts."""
        return [r.model_dump(exclude={"current"}) for r in self.metric_results.values()]


class EnergyEfficiencyAnalyzer:
    """Cross-accumulator energy efficiency metrics.

    Implements AnalyzerProtocol. No record ingestion — reads GPU telemetry
    hierarchy and metrics summary at summarize time to compute energy
    efficiency metrics (energy/token, tokens/joule, performance/watt, etc.).
    """

    required_accumulators: ClassVar[set[AccumulatorType]] = {
        "gpu_telemetry",
        "metric_results",
    }
    summary_dependencies: ClassVar[list[AccumulatorType]] = [
        "gpu_telemetry",
        "metric_results",
    ]

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        if user_config.gpu_telemetry_disabled:
            raise PluginDisabled("Energy efficiency metrics require GPU telemetry")

    async def summarize(self, ctx: SummaryContext) -> EnergyEfficiencySummary:
        """Compute energy efficiency metrics from GPU telemetry and inference metrics."""
        from aiperf.gpu_telemetry.accumulator import GPUTelemetryAccumulator
        from aiperf.metrics.accumulator import MetricsSummary
        from aiperf.plugin.enums import AccumulatorType

        gpu_acc: GPUTelemetryAccumulator | None = ctx.get_accumulator(
            AccumulatorType.GPU_TELEMETRY
        )
        if gpu_acc is None or not isinstance(gpu_acc, GPUTelemetryAccumulator):
            raise PluginDisabled("GPUTelemetryAccumulator not available")

        metrics_output: MetricsSummary | None = ctx.get_output(
            str(AccumulatorType.METRIC_RESULTS)
        )
        if metrics_output is None or not isinstance(metrics_output, MetricsSummary):
            raise PluginDisabled("MetricsSummary output not available")

        # Extract energy from GPU telemetry hierarchy
        duration_s = (
            (ctx.end_ns - ctx.start_ns) / NANOS_PER_SECOND
            if ctx.end_ns > ctx.start_ns
            else 0.0
        )
        time_filter = _make_time_filter(ctx.start_ns, ctx.end_ns)
        total_energy_j, avg_power_w, gpu_count, source = self._extract_energy(
            gpu_acc, duration_s, time_filter
        )

        if source == EnergySource.UNAVAILABLE or total_energy_j <= 0:
            raise PluginDisabled("No GPU energy data available")

        # Extract token/throughput data from metrics summary
        total_output_tokens = _get_metric_value(metrics_output, "total_osl")
        total_input_tokens = _get_metric_value(metrics_output, "total_isl")
        request_count = _get_metric_value(metrics_output, "request_count")
        request_throughput = _get_metric_value(metrics_output, "request_throughput")
        output_token_throughput = _get_metric_value(
            metrics_output, "output_token_throughput"
        )
        goodput = _get_metric_value(metrics_output, "goodput")

        # Compute metrics
        energy_per_output_token_mj = _safe_div(
            total_energy_j * 1000, total_output_tokens
        )
        energy_per_request_j = _safe_div(total_energy_j, request_count)

        total_tokens = (total_input_tokens or 0) + (total_output_tokens or 0)
        energy_per_total_token_mj = (
            _safe_div(total_energy_j * 1000, total_tokens) if total_tokens > 0 else None
        )

        performance_per_watt = (
            _safe_div(request_throughput, avg_power_w)
            if request_throughput is not None and avg_power_w > 0
            else None
        )

        output_tps_per_watt = (
            _safe_div(output_token_throughput, avg_power_w)
            if output_token_throughput is not None and avg_power_w > 0
            else None
        )

        goodput_per_watt = (
            _safe_div(goodput, avg_power_w)
            if goodput is not None and avg_power_w > 0
            else None
        )

        # Build MetricResult objects for the export pipeline
        metric_results = _build_metric_results(
            total_energy_j=total_energy_j,
            avg_power_w=avg_power_w,
            energy_per_output_token_mj=energy_per_output_token_mj,
            energy_per_request_j=energy_per_request_j,
            energy_per_total_token_mj=energy_per_total_token_mj,
            performance_per_watt=performance_per_watt,
            output_tps_per_watt=output_tps_per_watt,
            goodput_per_watt=goodput_per_watt,
        )

        return EnergyEfficiencySummary(
            total_gpu_energy_j=total_energy_j,
            average_gpu_power_w=avg_power_w,
            gpu_count=gpu_count,
            energy_source=source,
            energy_per_output_token_mj=energy_per_output_token_mj,
            energy_per_request_j=energy_per_request_j,
            energy_per_total_token_mj=energy_per_total_token_mj,
            performance_per_watt=performance_per_watt,
            output_tps_per_watt=output_tps_per_watt,
            goodput_per_watt=goodput_per_watt,
            metric_results=metric_results,
        )

    @staticmethod
    def _extract_energy(
        gpu_acc: GPUTelemetryAccumulator,
        duration_s: float,
        time_filter: TimeRangeFilter | None,
    ) -> tuple[float, float, int, EnergySource]:
        """Sum energy and power across all GPUs in the telemetry hierarchy."""
        from aiperf.common.models.telemetry_models import NoMetricValue

        total_energy_j = 0.0
        total_power_w = 0.0
        gpu_count = 0
        has_counter = False

        for gpus in gpu_acc._hierarchy.dcgm_endpoints.values():
            for gpu_data in gpus.values():
                gpu_count += 1
                ts: GpuMetricTimeSeries = gpu_data.time_series

                # Prefer DCGM energy counter delta (MJ -> J)
                try:
                    energy_result = ts.to_metric_result_filtered(
                        "energy_consumption",
                        "energy_consumption",
                        "Energy Consumption",
                        "MJ",
                        time_filter=time_filter,
                        is_counter=True,
                    )
                    energy_mj = energy_result.avg  # counter delta stored as avg
                    if energy_mj > 0:
                        total_energy_j += energy_mj * 1e6  # MJ -> J
                        has_counter = True
                except (NoMetricValue, Exception):
                    pass

                # Always accumulate power for avg/fallback
                try:
                    power_result = ts.to_metric_result_filtered(
                        "gpu_power_usage",
                        "gpu_power_usage",
                        "GPU Power Usage",
                        "W",
                        time_filter=time_filter,
                        is_counter=False,
                    )
                    total_power_w += power_result.avg
                except (NoMetricValue, Exception):
                    pass

        if has_counter:
            source = EnergySource.DCGM_COUNTER
            avg_power_w = total_energy_j / max(duration_s, 1e-9)
        elif total_power_w > 0 and duration_s > 0:
            # Fallback: power integration
            total_energy_j = total_power_w * duration_s
            avg_power_w = total_power_w
            source = EnergySource.POWER_INTEGRATION
        else:
            avg_power_w = 0.0
            source = EnergySource.UNAVAILABLE

        return total_energy_j, avg_power_w, gpu_count, source


def _make_time_filter(start_ns: int, end_ns: int) -> TimeRangeFilter | None:
    """Create a TimeRangeFilter from context timestamps."""
    from aiperf.common.models.telemetry_models import TimeRangeFilter

    if start_ns > 0:
        return TimeRangeFilter(start_ns=start_ns, end_ns=end_ns if end_ns > 0 else None)
    return None


def _get_metric_value(summary: MetricsSummary, tag: str) -> float | None:
    """Extract avg value for a metric tag from MetricsSummary."""
    result = summary.results.get(tag)
    if result is None:
        return None
    val = result.avg
    if val is None or val <= 0:
        return None
    return val


def _build_metric_results(
    *,
    total_energy_j: float,
    avg_power_w: float,
    energy_per_output_token_mj: float | None,
    energy_per_request_j: float | None,
    energy_per_total_token_mj: float | None,
    performance_per_watt: float | None,
    output_tps_per_watt: float | None,
    goodput_per_watt: float | None,
) -> dict[str, MetricResult]:
    """Build MetricResult objects for each energy metric."""
    results: dict[str, MetricResult] = {}

    def _add(
        tag: str, header: str, unit: str, value: float | None, display_order: int
    ) -> None:
        if value is None:
            return
        results[tag] = MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            avg=value,
        )

    _add(
        "energy_per_output_token",
        "Energy Per Output Token",
        "mJ/token",
        energy_per_output_token_mj,
        750,
    )
    _add("energy_per_request", "Energy Per Request", "J/req", energy_per_request_j, 740)
    _add("total_gpu_energy", "Total GPU Energy", "J", total_energy_j, 730)
    _add("average_gpu_power", "Average GPU Power", "W", avg_power_w, 720)
    _add(
        "energy_per_total_token",
        "Energy Per Total Token",
        "mJ/token",
        energy_per_total_token_mj,
        745,
    )
    _add(
        "performance_per_watt",
        "Performance Per Watt",
        "req/s/W",
        performance_per_watt,
        710,
    )
    _add(
        "output_tps_per_watt",
        "Output Tokens Per Second Per Watt",
        "tps/W",
        output_tps_per_watt,
        755,
    )
    _add(
        "goodput_per_watt",
        "Goodput Per Watt",
        "good-req/s/W",
        goodput_per_watt,
        705,
    )

    return results
