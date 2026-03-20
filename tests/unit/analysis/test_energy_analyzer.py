# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EnergyEfficiencyAnalyzer.

Focuses on:
- Initialization with telemetry enable/disable
- Energy extraction from DCGM counters vs power integration fallback
- Metric formula correctness (energy/token, tokens/joule, etc.)
- Edge cases: missing accumulators, zero tokens, multi-GPU summation
- Serialization (to_json, to_csv)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiperf.analysis.energy_analyzer import (
    EnergyEfficiencyAnalyzer,
    EnergyEfficiencySummary,
    EnergySource,
    _build_metric_results,
    _safe_div,
)
from aiperf.common.accumulator_protocols import SummaryContext
from aiperf.common.config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.exceptions import PluginDisabled
from aiperf.common.models.record_models import MetricResult
from aiperf.common.models.telemetry_models import (
    GpuMetadata,
    GpuMetricTimeSeries,
    GpuTelemetryData,
    TelemetryHierarchy,
)
from aiperf.gpu_telemetry.accumulator import GPUTelemetryAccumulator
from aiperf.metrics.accumulator import MetricsSummary
from aiperf.plugin.enums import AccumulatorType

# ============================================================
# Helpers
# ============================================================


def _make_user_config(*, no_gpu_telemetry: bool = False) -> UserConfig:
    """Create a minimal UserConfig for energy analyzer tests."""
    return UserConfig(
        endpoint={
            "model_names": ["test-model"],
            "type": "completions",
            "streaming": False,
        },
        no_gpu_telemetry=no_gpu_telemetry,
    )


def _make_time_series(
    *,
    power_values: list[float],
    energy_values: list[float] | None = None,
    timestamps_ns: list[int] | None = None,
) -> GpuMetricTimeSeries:
    """Build a GpuMetricTimeSeries with power (and optionally energy counter) data."""
    ts = GpuMetricTimeSeries()
    if timestamps_ns is None:
        timestamps_ns = [i * NANOS_PER_SECOND for i in range(len(power_values))]
    for i, t_ns in enumerate(timestamps_ns):
        metrics: dict[str, float] = {"gpu_power_usage": power_values[i]}
        if energy_values is not None:
            metrics["energy_consumption"] = energy_values[i]
        ts.append_snapshot(metrics, t_ns)
    return ts


def _make_gpu_telemetry_data(
    *,
    gpu_uuid: str = "GPU-0000",
    power_values: list[float],
    energy_values: list[float] | None = None,
    timestamps_ns: list[int] | None = None,
) -> GpuTelemetryData:
    """Build GpuTelemetryData for a single GPU."""
    return GpuTelemetryData(
        metadata=GpuMetadata(
            gpu_index=0,
            gpu_uuid=gpu_uuid,
            gpu_model_name="Test GPU",
            hostname="test-host",
        ),
        time_series=_make_time_series(
            power_values=power_values,
            energy_values=energy_values,
            timestamps_ns=timestamps_ns,
        ),
    )


def _make_gpu_accumulator(
    hierarchy: TelemetryHierarchy,
) -> GPUTelemetryAccumulator:
    """Create a mock GPUTelemetryAccumulator backed by a real hierarchy."""
    acc = MagicMock(spec=GPUTelemetryAccumulator)
    acc._hierarchy = hierarchy
    return acc


def _make_metrics_summary(
    *,
    total_osl: float | None = 1000.0,
    total_isl: float | None = 500.0,
    request_count: float | None = 100.0,
    request_throughput: float | None = 10.0,
    output_token_throughput: float | None = 100.0,
    goodput: float | None = None,
) -> MetricsSummary:
    """Build a MetricsSummary with common inference metrics."""
    results: dict[str, MetricResult] = {}
    if total_osl is not None:
        results["total_osl"] = MetricResult(
            tag="total_osl", header="Total OSL", unit="tokens", avg=total_osl
        )
    if total_isl is not None:
        results["total_isl"] = MetricResult(
            tag="total_isl", header="Total ISL", unit="tokens", avg=total_isl
        )
    if request_count is not None:
        results["request_count"] = MetricResult(
            tag="request_count",
            header="Request Count",
            unit="requests",
            avg=request_count,
        )
    if request_throughput is not None:
        results["request_throughput"] = MetricResult(
            tag="request_throughput",
            header="Request Throughput",
            unit="req/s",
            avg=request_throughput,
        )
    if output_token_throughput is not None:
        results["output_token_throughput"] = MetricResult(
            tag="output_token_throughput",
            header="Output Token Throughput",
            unit="tokens/sec",
            avg=output_token_throughput,
        )
    if goodput is not None:
        results["goodput"] = MetricResult(
            tag="goodput",
            header="Goodput",
            unit="req/s",
            avg=goodput,
        )
    return MetricsSummary(results=results)


def _make_summary_context(
    *,
    gpu_acc: GPUTelemetryAccumulator | None = None,
    metrics_summary: MetricsSummary | None = None,
    start_ns: int = 0,
    end_ns: int = 10 * NANOS_PER_SECOND,
) -> SummaryContext:
    """Build a SummaryContext with accumulators and outputs wired up."""
    accumulators: dict[AccumulatorType, object] = {}
    accumulator_outputs: dict[str, object] = {}

    if gpu_acc is not None:
        accumulators[AccumulatorType.GPU_TELEMETRY] = gpu_acc
    if metrics_summary is not None:
        accumulator_outputs[str(AccumulatorType.METRIC_RESULTS)] = metrics_summary

    return SummaryContext(
        accumulators=accumulators,
        accumulator_outputs=accumulator_outputs,
        start_ns=start_ns,
        end_ns=end_ns,
    )


def _single_gpu_hierarchy(
    *,
    power_values: list[float],
    energy_values: list[float] | None = None,
    timestamps_ns: list[int] | None = None,
    gpu_uuid: str = "GPU-0000",
    endpoint_url: str = "http://node1:9401/metrics",
) -> TelemetryHierarchy:
    """Build a TelemetryHierarchy with one GPU."""
    hierarchy = TelemetryHierarchy()
    gpu_data = _make_gpu_telemetry_data(
        gpu_uuid=gpu_uuid,
        power_values=power_values,
        energy_values=energy_values,
        timestamps_ns=timestamps_ns,
    )
    hierarchy.dcgm_endpoints[endpoint_url] = {gpu_uuid: gpu_data}
    return hierarchy


# ============================================================
# Init Tests
# ============================================================


class TestEnergyEfficiencyAnalyzerInit:
    """Verify constructor enable/disable behavior."""

    def test_init_gpu_telemetry_disabled_raises_plugin_disabled(self) -> None:
        config = _make_user_config(no_gpu_telemetry=True)
        with pytest.raises(PluginDisabled, match="GPU telemetry"):
            EnergyEfficiencyAnalyzer(user_config=config)

    def test_init_gpu_telemetry_enabled_succeeds(self) -> None:
        config = _make_user_config(no_gpu_telemetry=False)
        analyzer = EnergyEfficiencyAnalyzer(user_config=config)
        assert analyzer is not None


# ============================================================
# Summarize Tests
# ============================================================


class TestSummarizeDCGMCounter:
    """Verify energy extraction from DCGM cumulative counters (MJ -> J delta)."""

    @pytest.mark.asyncio
    async def test_summarize_dcgm_counter_energy(self) -> None:
        """DCGM counter path: energy_consumption delta in MJ converted to J."""
        # 10 snapshots over 10 seconds
        # Energy counter goes from 1.000 MJ to 1.001 MJ => delta = 0.001 MJ = 1000 J
        # Power gauge averages ~200 W
        timestamps_ns = [i * NANOS_PER_SECOND for i in range(10)]
        power_values = [200.0] * 10
        energy_values = [1.0 + i * 0.0001 for i in range(10)]  # 1.0000 to 1.0009 MJ

        hierarchy = _single_gpu_hierarchy(
            power_values=power_values,
            energy_values=energy_values,
            timestamps_ns=timestamps_ns,
        )
        gpu_acc = _make_gpu_accumulator(hierarchy)
        metrics_summary = _make_metrics_summary(goodput=8.0)
        ctx = _make_summary_context(
            gpu_acc=gpu_acc,
            metrics_summary=metrics_summary,
            start_ns=0,
            end_ns=9 * NANOS_PER_SECOND,
        )

        analyzer = EnergyEfficiencyAnalyzer(user_config=_make_user_config())
        result = await analyzer.summarize(ctx)

        assert result.energy_source == EnergySource.DCGM_COUNTER
        assert result.gpu_count == 1

        # Delta = (1.0009 - 1.0000) MJ = 0.0009 MJ = 900 J
        expected_energy_j = 0.0009 * 1e6  # 900 J
        assert result.total_gpu_energy_j == pytest.approx(expected_energy_j, rel=1e-6)

        # All 9 metric results present (including goodput_per_watt)
        assert len(result.metric_results) == 8
        assert "energy_per_output_token" in result.metric_results
        assert "energy_per_request" in result.metric_results
        assert "total_gpu_energy" in result.metric_results
        assert "average_gpu_power" in result.metric_results
        assert "energy_per_total_token" in result.metric_results
        assert "performance_per_watt" in result.metric_results
        assert "output_tps_per_watt" in result.metric_results
        assert "goodput_per_watt" in result.metric_results


class TestSummarizePowerIntegrationFallback:
    """Verify fallback to power * duration when energy counter is unavailable."""

    @pytest.mark.asyncio
    async def test_summarize_power_integration_fallback(self) -> None:
        """No energy_consumption metric => falls back to power * duration."""
        timestamps_ns = [i * NANOS_PER_SECOND for i in range(10)]
        power_values = [250.0] * 10  # 250 W constant

        hierarchy = _single_gpu_hierarchy(
            power_values=power_values,
            timestamps_ns=timestamps_ns,
        )
        gpu_acc = _make_gpu_accumulator(hierarchy)
        metrics_summary = _make_metrics_summary()
        duration_s = 10.0
        ctx = _make_summary_context(
            gpu_acc=gpu_acc,
            metrics_summary=metrics_summary,
            start_ns=0,
            end_ns=int(duration_s * NANOS_PER_SECOND),
        )

        analyzer = EnergyEfficiencyAnalyzer(user_config=_make_user_config())
        result = await analyzer.summarize(ctx)

        assert result.energy_source == EnergySource.POWER_INTEGRATION
        expected_energy_j = 250.0 * duration_s  # 2500 J
        assert result.total_gpu_energy_j == pytest.approx(expected_energy_j, rel=1e-6)
        assert result.average_gpu_power_w == pytest.approx(250.0, rel=1e-6)


class TestSummarizeErrors:
    """Verify PluginDisabled raised for missing prerequisites."""

    @pytest.mark.asyncio
    async def test_summarize_no_gpu_data_raises_plugin_disabled(self) -> None:
        """Empty hierarchy with no GPUs raises PluginDisabled."""
        hierarchy = TelemetryHierarchy()
        gpu_acc = _make_gpu_accumulator(hierarchy)
        metrics_summary = _make_metrics_summary()
        ctx = _make_summary_context(gpu_acc=gpu_acc, metrics_summary=metrics_summary)

        analyzer = EnergyEfficiencyAnalyzer(user_config=_make_user_config())
        with pytest.raises(PluginDisabled, match="No GPU energy data"):
            await analyzer.summarize(ctx)

    @pytest.mark.asyncio
    async def test_summarize_no_metrics_output_raises_plugin_disabled(self) -> None:
        """Missing MetricsSummary output raises PluginDisabled."""
        hierarchy = _single_gpu_hierarchy(power_values=[200.0] * 5)
        gpu_acc = _make_gpu_accumulator(hierarchy)
        ctx = _make_summary_context(gpu_acc=gpu_acc, metrics_summary=None)

        analyzer = EnergyEfficiencyAnalyzer(user_config=_make_user_config())
        with pytest.raises(PluginDisabled, match="MetricsSummary"):
            await analyzer.summarize(ctx)

    @pytest.mark.asyncio
    async def test_summarize_no_accumulator_raises_plugin_disabled(self) -> None:
        """Missing GPUTelemetryAccumulator raises PluginDisabled."""
        metrics_summary = _make_metrics_summary()
        ctx = _make_summary_context(gpu_acc=None, metrics_summary=metrics_summary)

        analyzer = EnergyEfficiencyAnalyzer(user_config=_make_user_config())
        with pytest.raises(PluginDisabled, match="GPUTelemetryAccumulator"):
            await analyzer.summarize(ctx)


# ============================================================
# Metric Formula Tests
# ============================================================


class TestMetricFormulas:
    """Verify individual metric computations produce correct values."""

    @pytest.mark.asyncio
    async def test_energy_per_output_token_formula(self) -> None:
        """energy_per_output_token_mj = energy_j * 1000 / total_output_tokens."""
        energy_j = 500.0
        total_osl = 2000.0
        expected_mj = energy_j * 1000 / total_osl  # 250.0 mJ/token

        result = await self._summarize_with(energy_j=energy_j, total_osl=total_osl)
        assert result.energy_per_output_token_mj == pytest.approx(expected_mj, rel=1e-6)

    @pytest.mark.asyncio
    async def test_energy_per_request_formula(self) -> None:
        """energy_per_request_j = energy_j / request_count."""
        energy_j = 500.0
        request_count = 50.0
        expected = energy_j / request_count  # 10.0 J/req

        result = await self._summarize_with(
            energy_j=energy_j, request_count=request_count
        )
        assert result.energy_per_request_j == pytest.approx(expected, rel=1e-6)

    @pytest.mark.asyncio
    async def test_energy_per_total_token_formula(self) -> None:
        """energy_per_total_token_mj = energy_j * 1000 / (input + output tokens)."""
        energy_j = 500.0
        total_osl = 2000.0
        total_isl = 1000.0
        total_tokens = total_osl + total_isl
        expected_mj = energy_j * 1000 / total_tokens  # ~166.67 mJ/token

        result = await self._summarize_with(
            energy_j=energy_j, total_osl=total_osl, total_isl=total_isl
        )
        assert result.energy_per_total_token_mj == pytest.approx(expected_mj, rel=1e-6)

    @pytest.mark.asyncio
    async def test_performance_per_watt_formula(self) -> None:
        """performance_per_watt = request_throughput / avg_power_w."""
        # Power integration: energy_j = power * duration, so power = energy_j / duration
        # Helper uses duration_s=10, so power = 3000 / 10 = 300 W
        energy_j = 3000.0
        request_throughput = 15.0
        avg_power_w = energy_j / 10.0  # 300 W
        expected = request_throughput / avg_power_w  # 0.05 req/s/W

        result = await self._summarize_with(
            energy_j=energy_j, request_throughput=request_throughput
        )
        assert result.performance_per_watt == pytest.approx(expected, rel=1e-6)

    @pytest.mark.asyncio
    async def test_output_tps_per_watt_formula(self) -> None:
        """output_tps_per_watt = output_token_throughput / avg_power_w."""
        energy_j = 3000.0
        output_token_throughput = 500.0
        avg_power_w = energy_j / 10.0  # 300 W
        expected = output_token_throughput / avg_power_w

        result = await self._summarize_with(
            energy_j=energy_j, output_token_throughput=output_token_throughput
        )
        assert result.output_tps_per_watt == pytest.approx(expected, rel=1e-6)

    @pytest.mark.asyncio
    async def test_goodput_per_watt_formula(self) -> None:
        """goodput_per_watt = goodput / avg_power_w."""
        energy_j = 3000.0
        goodput = 8.0  # good req/s
        avg_power_w = energy_j / 10.0  # 300 W
        expected = goodput / avg_power_w

        result = await self._summarize_with(energy_j=energy_j, goodput=goodput)
        assert result.goodput_per_watt == pytest.approx(expected, rel=1e-6)

    @pytest.mark.asyncio
    async def test_zero_tokens_returns_none(self) -> None:
        """When total_osl is 0, energy_per_output_token is None."""
        result = await self._summarize_with(total_osl=0.0)
        assert result.energy_per_output_token_mj is None

    async def _summarize_with(
        self,
        *,
        energy_j: float = 500.0,
        power_w: float = 250.0,
        total_osl: float | None = 1000.0,
        total_isl: float | None = 500.0,
        request_count: float | None = 100.0,
        request_throughput: float | None = 10.0,
        output_token_throughput: float | None = 100.0,
        goodput: float | None = None,
    ) -> EnergyEfficiencySummary:
        """Run summarize with power integration to get a known energy_j."""
        duration_s = 10.0
        constant_power = energy_j / duration_s  # power * duration = energy_j

        timestamps_ns = [i * NANOS_PER_SECOND for i in range(int(duration_s))]
        hierarchy = _single_gpu_hierarchy(
            power_values=[constant_power] * int(duration_s),
            timestamps_ns=timestamps_ns,
        )
        gpu_acc = _make_gpu_accumulator(hierarchy)
        metrics_summary = _make_metrics_summary(
            total_osl=total_osl,
            total_isl=total_isl,
            request_count=request_count,
            request_throughput=request_throughput,
            output_token_throughput=output_token_throughput,
            goodput=goodput,
        )
        ctx = _make_summary_context(
            gpu_acc=gpu_acc,
            metrics_summary=metrics_summary,
            start_ns=0,
            end_ns=int(duration_s * NANOS_PER_SECOND),
        )
        analyzer = EnergyEfficiencyAnalyzer(user_config=_make_user_config())
        return await analyzer.summarize(ctx)


# ============================================================
# Helper Function Tests
# ============================================================


class TestSafeDiv:
    """Verify _safe_div edge cases."""

    @pytest.mark.parametrize(
        "numerator,denominator,expected",
        [
            (10.0, 2.0, 5.0),
            (100.0, 3.0, 100.0 / 3.0),
            (0.0, 5.0, 0.0),
        ],
    )  # fmt: skip
    def test_safe_div_normal(
        self, numerator: float, denominator: float, expected: float
    ) -> None:
        assert _safe_div(numerator, denominator) == pytest.approx(expected)

    def test_safe_div_zero_denominator_returns_none(self) -> None:
        assert _safe_div(10.0, 0.0) is None

    def test_safe_div_negative_denominator_returns_none(self) -> None:
        assert _safe_div(10.0, -1.0) is None

    def test_safe_div_none_denominator_returns_none(self) -> None:
        assert _safe_div(10.0, None) is None

    def test_safe_div_none_numerator_returns_none(self) -> None:
        assert _safe_div(None, 5.0) is None


# ============================================================
# Serialization Tests
# ============================================================


class TestEnergyEfficiencySummarySerialization:
    """Verify to_json and to_csv produce correct output."""

    def _make_summary_with_metrics(self) -> EnergyEfficiencySummary:
        """Create a populated summary for serialization tests."""
        metric_results = _build_metric_results(
            total_energy_j=1000.0,
            avg_power_w=200.0,
            energy_per_output_token_mj=500.0,
            energy_per_request_j=10.0,
            energy_per_total_token_mj=333.0,
            performance_per_watt=0.05,
            output_tps_per_watt=10.0,
            goodput_per_watt=0.04,
        )
        return EnergyEfficiencySummary(
            total_gpu_energy_j=1000.0,
            average_gpu_power_w=200.0,
            gpu_count=2,
            energy_source=EnergySource.DCGM_COUNTER,
            energy_per_output_token_mj=500.0,
            energy_per_request_j=10.0,
            energy_per_total_token_mj=333.0,
            performance_per_watt=0.05,
            output_tps_per_watt=10.0,
            goodput_per_watt=0.04,
            metric_results=metric_results,
        )

    def test_to_json_includes_all_fields(self) -> None:
        summary = self._make_summary_with_metrics()
        data = summary.to_json()

        assert "source" in data
        assert data["source"]["total_gpu_energy_j"] == 1000.0
        assert data["source"]["average_gpu_power_w"] == 200.0
        assert data["source"]["gpu_count"] == 2
        assert data["source"]["energy_source"] == "dcgm_counter"

        assert "metrics" in data
        assert "energy_per_output_token_mj" in data["metrics"]
        assert "energy_per_request_j" in data["metrics"]
        assert "energy_per_total_token_mj" in data["metrics"]
        assert "performance_per_watt" in data["metrics"]
        assert "output_tps_per_watt" in data["metrics"]
        assert "goodput_per_watt" in data["metrics"]

        assert "results" in data
        assert len(data["results"]) == 8

    def test_to_csv_returns_metric_results(self) -> None:
        summary = self._make_summary_with_metrics()
        rows = summary.to_csv()

        assert isinstance(rows, list)
        assert len(rows) == 8
        tags = {row["tag"] for row in rows}
        assert "energy_per_output_token" in tags
        assert "total_gpu_energy" in tags

    def test_to_json_empty_metrics_omits_optional_fields(self) -> None:
        summary = EnergyEfficiencySummary()
        data = summary.to_json()

        assert data["source"]["energy_source"] == "unavailable"
        assert data["metrics"] == {}
        assert "results" not in data


# ============================================================
# Multi-GPU Tests
# ============================================================


class TestMultiGPU:
    """Verify energy summation across multiple GPUs."""

    @pytest.mark.asyncio
    async def test_summarize_multi_gpu_sums_energy(self) -> None:
        """Energy from 2 GPUs on different endpoints is summed correctly."""
        timestamps_ns = [i * NANOS_PER_SECOND for i in range(10)]
        # GPU 0: counter goes 1.000 -> 1.001 MJ => delta 0.001 MJ = 1000 J
        energy_0 = [1.0 + i * 0.0001 for i in range(10)]
        # GPU 1: counter goes 2.000 -> 2.002 MJ => delta 0.002 MJ = 2000 J
        energy_1 = [2.0 + i * 0.0002 for i in range(10)]

        hierarchy = TelemetryHierarchy()

        gpu0 = _make_gpu_telemetry_data(
            gpu_uuid="GPU-0000",
            power_values=[200.0] * 10,
            energy_values=energy_0,
            timestamps_ns=timestamps_ns,
        )
        hierarchy.dcgm_endpoints["http://node1:9401/metrics"] = {"GPU-0000": gpu0}

        gpu1 = _make_gpu_telemetry_data(
            gpu_uuid="GPU-1111",
            power_values=[300.0] * 10,
            energy_values=energy_1,
            timestamps_ns=timestamps_ns,
        )
        hierarchy.dcgm_endpoints["http://node2:9401/metrics"] = {"GPU-1111": gpu1}

        gpu_acc = _make_gpu_accumulator(hierarchy)
        metrics_summary = _make_metrics_summary()
        ctx = _make_summary_context(
            gpu_acc=gpu_acc,
            metrics_summary=metrics_summary,
            start_ns=0,
            end_ns=9 * NANOS_PER_SECOND,
        )

        analyzer = EnergyEfficiencyAnalyzer(user_config=_make_user_config())
        result = await analyzer.summarize(ctx)

        assert result.gpu_count == 2
        assert result.energy_source == EnergySource.DCGM_COUNTER
        # GPU-0: delta = (1.0009 - 1.0) * 1e6 = 900 J
        # GPU-1: delta = (2.0018 - 2.0) * 1e6 = 1800 J
        # Total = 2700 J
        expected_total = (energy_0[-1] - energy_0[0]) * 1e6 + (
            energy_1[-1] - energy_1[0]
        ) * 1e6
        assert result.total_gpu_energy_j == pytest.approx(expected_total, rel=1e-6)
