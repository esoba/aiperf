# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for EnergyEfficiencyProcessor derived metrics.

Focuses on:
- compute() happy path with varying GPU counts
- Graceful skipping when telemetry/metrics are absent or zero
- Multi-endpoint GPU collection
- Pathological inputs (NaN, inf, negative energy deltas)
- Unit tag serialization for new GenericMetricUnit values
"""

import math
from datetime import datetime

import pytest
from pytest import param

from aiperf.common.enums import EnergyMetricUnit, GenericMetricUnit, PowerMetricUnit
from aiperf.common.models import (
    EndpointData,
    GpuSummary,
    JsonMetricResult,
    MetricResult,
    TelemetryExportData,
    TelemetrySummary,
)
from aiperf.post_processors.energy_efficiency_processor import compute

# ============================================================
# Helpers
# ============================================================


def _make_gpu_summary(
    gpu_index: int,
    energy_mj: float | None = None,
    power_watts: float | None = None,
) -> GpuSummary:
    """Build a GpuSummary with optional energy and power metrics."""
    metrics: dict[str, JsonMetricResult] = {}
    if energy_mj is not None:
        metrics["energy_consumption"] = JsonMetricResult(
            unit=EnergyMetricUnit.MEGAJOULE.info.tag,
            avg=energy_mj,
        )
    if power_watts is not None:
        metrics["gpu_power_usage"] = JsonMetricResult(
            unit=PowerMetricUnit.WATT.info.tag,
            avg=power_watts,
        )
    return GpuSummary(
        gpu_index=gpu_index,
        gpu_name=f"Test GPU {gpu_index}",
        gpu_uuid=f"GPU-{gpu_index:04d}-aaaa-bbbb-cccc",
        hostname="node1",
        metrics=metrics,
    )


def _make_telemetry(
    gpu_summaries: list[GpuSummary],
    endpoint_name: str = "localhost:9401",
) -> TelemetryExportData:
    """Wrap GpuSummary list into TelemetryExportData under a single endpoint."""
    gpus = {f"gpu_{g.gpu_index}": g for g in gpu_summaries}
    return TelemetryExportData(
        summary=TelemetrySummary(
            start_time=datetime(2025, 1, 1),
            end_time=datetime(2025, 1, 1, 0, 1),
        ),
        endpoints={endpoint_name: EndpointData(gpus=gpus)},
    )


def _make_multi_endpoint_telemetry(
    endpoints: dict[str, list[GpuSummary]],
) -> TelemetryExportData:
    """Build TelemetryExportData with multiple endpoints, each containing its own GPUs."""
    endpoint_data = {}
    for endpoint_name, gpu_summaries in endpoints.items():
        gpus = {f"gpu_{g.gpu_index}": g for g in gpu_summaries}
        endpoint_data[endpoint_name] = EndpointData(gpus=gpus)
    return TelemetryExportData(
        summary=TelemetrySummary(
            start_time=datetime(2025, 1, 1),
            end_time=datetime(2025, 1, 1, 0, 1),
        ),
        endpoints=endpoint_data,
    )


def _make_inference_results(
    throughput: float | None = 1000.0,
    total_tokens: float | None = 60000.0,
) -> list[MetricResult]:
    """Build minimal inference MetricResult list with output_token_throughput and total_output_tokens."""
    results: list[MetricResult] = []
    if throughput is not None:
        results.append(
            MetricResult(
                tag="output_token_throughput",
                header="Output Token Throughput",
                unit="tokens/sec",
                avg=throughput,
            )
        )
    if total_tokens is not None:
        results.append(
            MetricResult(
                tag="total_output_tokens",
                header="Total Output Tokens",
                unit="tokens",
                avg=total_tokens,
            )
        )
    return results


DURATION_NS = 60_000_000_000  # 60 seconds


# ============================================================
# Happy Path Tests
# ============================================================


class TestEnergyEfficiencyCompute:
    """Tests for the compute() function happy path and basic metric skipping."""

    def test_no_telemetry_returns_empty(self) -> None:
        results = compute(
            inference_results=_make_inference_results(),
            telemetry=None,
            benchmark_duration_ns=DURATION_NS,
        )
        assert results == []

    def test_empty_endpoints_returns_empty(self) -> None:
        telemetry = TelemetryExportData(
            summary=TelemetrySummary(
                start_time=datetime(2025, 1, 1),
                end_time=datetime(2025, 1, 1, 0, 1),
            ),
            endpoints={},
        )
        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )
        assert results == []

    @pytest.mark.parametrize(
        "num_gpus",
        [
            param(1, id="1-gpu"),
            param(2, id="2-gpus"),
            param(4, id="4-gpus"),
            param(8, id="8-gpus"),
        ],
    )  # fmt: skip
    def test_all_four_metrics_computed(self, num_gpus: int) -> None:
        energy_per_gpu_mj = 0.001  # 0.001 MJ = 1000 J each
        power_per_gpu_w = 250.0
        total_tokens = 60000.0
        throughput = 1000.0

        gpus = [
            _make_gpu_summary(
                i, energy_mj=energy_per_gpu_mj, power_watts=power_per_gpu_w
            )
            for i in range(num_gpus)
        ]
        telemetry = _make_telemetry(gpus)
        inference = _make_inference_results(
            throughput=throughput, total_tokens=total_tokens
        )

        results = compute(
            inference_results=inference,
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        assert tags == {
            "total_energy_joules",
            "tokens_per_joule",
            "avg_gpu_power_watts",
            "tokens_per_second_per_watt",
        }

        by_tag = {r.tag: r for r in results}

        # total_energy_joules = num_gpus * 0.001 MJ * 1_000_000 J/MJ = num_gpus * 1000 J
        expected_energy_j = num_gpus * 1000.0
        assert by_tag["total_energy_joules"].avg == pytest.approx(expected_energy_j)
        assert by_tag["total_energy_joules"].unit == EnergyMetricUnit.JOULE.info.tag

        # tokens_per_joule = total_tokens / total_energy_joules
        assert by_tag["tokens_per_joule"].avg == pytest.approx(
            total_tokens / expected_energy_j
        )
        assert (
            by_tag["tokens_per_joule"].unit
            == GenericMetricUnit.TOKENS_PER_JOULE.info.tag
        )

        # avg_gpu_power_watts = mean of per-GPU power
        assert by_tag["avg_gpu_power_watts"].avg == pytest.approx(power_per_gpu_w)
        assert by_tag["avg_gpu_power_watts"].unit == PowerMetricUnit.WATT.info.tag

        # tokens_per_second_per_watt = throughput / avg_power
        assert by_tag["tokens_per_second_per_watt"].avg == pytest.approx(
            throughput / power_per_gpu_w
        )
        assert (
            by_tag["tokens_per_second_per_watt"].unit
            == GenericMetricUnit.TOKENS_PER_SECOND_PER_WATT.info.tag
        )

    def test_no_energy_metric_skips_energy_and_tokens_per_joule(self) -> None:
        gpus = [_make_gpu_summary(0, energy_mj=None, power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        assert "total_energy_joules" not in tags
        assert "tokens_per_joule" not in tags
        # Power metrics should still be present
        assert "avg_gpu_power_watts" in tags
        assert "tokens_per_second_per_watt" in tags

    def test_no_power_metric_skips_power_results(self) -> None:
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=None)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        assert "avg_gpu_power_watts" not in tags
        assert "tokens_per_second_per_watt" not in tags
        # Energy metrics should still be present
        assert "total_energy_joules" in tags
        assert "tokens_per_joule" in tags

    def test_zero_energy_guards_against_division(self) -> None:
        gpus = [_make_gpu_summary(0, energy_mj=0.0, power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        # Zero energy means no energy/tokens_per_joule results
        assert "total_energy_joules" not in tags
        assert "tokens_per_joule" not in tags
        # Power metrics still present
        assert "avg_gpu_power_watts" in tags

    def test_zero_power_guards_against_division(self) -> None:
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=0.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        assert "avg_gpu_power_watts" not in tags
        assert "tokens_per_second_per_watt" not in tags

    def test_no_throughput_skips_tokens_per_second_per_watt(self) -> None:
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(throughput=None),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        assert "avg_gpu_power_watts" in tags
        assert "tokens_per_second_per_watt" not in tags

    def test_no_total_tokens_skips_tokens_per_joule(self) -> None:
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(total_tokens=None),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        assert "total_energy_joules" in tags
        assert "tokens_per_joule" not in tags

    def test_correct_headers(self) -> None:
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        by_tag = {r.tag: r for r in results}
        assert by_tag["total_energy_joules"].header == "Total Energy Consumption"
        assert by_tag["tokens_per_joule"].header == "Tokens Per Joule"
        assert by_tag["avg_gpu_power_watts"].header == "Average GPU Power"
        assert (
            by_tag["tokens_per_second_per_watt"].header == "Tokens Per Second Per Watt"
        )

    @pytest.mark.parametrize(
        "num_gpus",
        [
            param(2, id="2-gpus"),
            param(4, id="4-gpus"),
            param(8, id="8-gpus"),
        ],
    )  # fmt: skip
    def test_multi_gpu_energy_sums_correctly(self, num_gpus: int) -> None:
        energy_per_gpu_mj = 0.002  # 2000 J each
        gpus = [
            _make_gpu_summary(i, energy_mj=energy_per_gpu_mj, power_watts=300.0)
            for i in range(num_gpus)
        ]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        by_tag = {r.tag: r for r in results}
        expected_j = num_gpus * 2000.0
        assert by_tag["total_energy_joules"].avg == pytest.approx(expected_j)

    @pytest.mark.parametrize(
        "num_gpus,powers",
        [
            param(2, [200.0, 400.0], id="2-gpus-varied"),
            param(4, [100.0, 200.0, 300.0, 400.0], id="4-gpus-varied"),
        ],
    )  # fmt: skip
    def test_multi_gpu_power_averages_correctly(
        self, num_gpus: int, powers: list[float]
    ) -> None:
        gpus = [
            _make_gpu_summary(i, energy_mj=0.001, power_watts=powers[i])
            for i in range(num_gpus)
        ]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        by_tag = {r.tag: r for r in results}
        expected_avg = sum(powers) / len(powers)
        assert by_tag["avg_gpu_power_watts"].avg == pytest.approx(expected_avg)


# ============================================================
# Multi-Endpoint GPU Collection
# ============================================================


class TestMultiEndpointCollection:
    """Verify _collect_gpu_summaries aggregates GPUs across multiple endpoints."""

    def test_gpus_from_two_endpoints_summed_for_energy(self) -> None:
        """Energy should sum across GPUs from different endpoints."""
        telemetry = _make_multi_endpoint_telemetry(
            {
                "endpoint-a:9401": [
                    _make_gpu_summary(0, energy_mj=0.001, power_watts=200.0)
                ],
                "endpoint-b:9401": [
                    _make_gpu_summary(1, energy_mj=0.002, power_watts=300.0)
                ],
            }
        )

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        by_tag = {r.tag: r for r in results}
        # 0.001 MJ + 0.002 MJ = 0.003 MJ = 3000 J
        assert by_tag["total_energy_joules"].avg == pytest.approx(3000.0)
        # Power averaged across both GPUs: (200 + 300) / 2 = 250
        assert by_tag["avg_gpu_power_watts"].avg == pytest.approx(250.0)

    def test_gpus_from_three_endpoints_all_contribute(self) -> None:
        """All endpoints contribute GPUs regardless of endpoint count."""
        telemetry = _make_multi_endpoint_telemetry(
            {
                "ep1:9401": [_make_gpu_summary(0, energy_mj=0.001, power_watts=100.0)],
                "ep2:9401": [
                    _make_gpu_summary(1, energy_mj=0.001, power_watts=200.0),
                    _make_gpu_summary(2, energy_mj=0.001, power_watts=300.0),
                ],
                "ep3:9401": [_make_gpu_summary(3, energy_mj=0.001, power_watts=400.0)],
            }
        )

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        by_tag = {r.tag: r for r in results}
        # 4 GPUs * 0.001 MJ * 1e6 = 4000 J
        assert by_tag["total_energy_joules"].avg == pytest.approx(4000.0)
        # Mean power: (100 + 200 + 300 + 400) / 4 = 250
        assert by_tag["avg_gpu_power_watts"].avg == pytest.approx(250.0)

    def test_endpoint_with_no_gpus_ignored(self) -> None:
        """An endpoint with zero GPUs does not affect results."""
        telemetry = TelemetryExportData(
            summary=TelemetrySummary(
                start_time=datetime(2025, 1, 1),
                end_time=datetime(2025, 1, 1, 0, 1),
            ),
            endpoints={
                "ep-with-gpus:9401": EndpointData(
                    gpus={
                        "gpu_0": _make_gpu_summary(
                            0, energy_mj=0.001, power_watts=250.0
                        ),
                    }
                ),
                "ep-empty:9401": EndpointData(gpus={}),
            },
        )

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        by_tag = {r.tag: r for r in results}
        assert by_tag["total_energy_joules"].avg == pytest.approx(1000.0)
        assert by_tag["avg_gpu_power_watts"].avg == pytest.approx(250.0)


# ============================================================
# Edge Cases: Zero Throughput/Tokens
# ============================================================


class TestZeroInferenceMetrics:
    """Verify zero-valued inference metrics are treated as absent (skip derived metrics)."""

    def test_zero_throughput_skips_tokens_per_second_per_watt(self) -> None:
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(throughput=0.0),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        assert "avg_gpu_power_watts" in tags
        assert "tokens_per_second_per_watt" not in tags

    def test_zero_total_tokens_skips_tokens_per_joule(self) -> None:
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(total_tokens=0.0),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        assert "total_energy_joules" in tags
        assert "tokens_per_joule" not in tags


# ============================================================
# Pathological Inputs: NaN / Inf / Negative
# ============================================================


class TestPathologicalInputs:
    """Verify behavior with NaN, inf, and negative telemetry values."""

    def test_nan_energy_treated_as_present_but_produces_nan_result(self) -> None:
        """NaN energy passes the `is not None` check but NaN > 0 is False, so skipped."""
        gpus = [_make_gpu_summary(0, energy_mj=float("nan"), power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        # NaN > 0 is False, so total_energy_joules is skipped
        assert "total_energy_joules" not in tags
        assert "tokens_per_joule" not in tags
        # Power metrics still computed
        assert "avg_gpu_power_watts" in tags

    def test_nan_power_treated_as_present_but_produces_nan_result(self) -> None:
        """NaN power passes the `is not None` check. NaN > 0 is False, so avg_gpu_power skipped."""
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=float("nan"))]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        # NaN mean > 0 is False
        assert "avg_gpu_power_watts" not in tags
        assert "tokens_per_second_per_watt" not in tags
        # Energy metrics still computed
        assert "total_energy_joules" in tags

    def test_inf_energy_produces_inf_result(self) -> None:
        """Inf energy passes > 0 check, so metrics are produced with inf values."""
        gpus = [_make_gpu_summary(0, energy_mj=float("inf"), power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        by_tag = {r.tag: r for r in results}
        assert math.isinf(by_tag["total_energy_joules"].avg)
        # tokens_per_joule = 60000 / inf = 0.0
        assert by_tag["tokens_per_joule"].avg == pytest.approx(0.0)

    def test_inf_power_produces_inf_avg_and_zero_efficiency(self) -> None:
        """Inf power leads to inf avg and tokens_per_second_per_watt = throughput / inf = 0."""
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=float("inf"))]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        by_tag = {r.tag: r for r in results}
        assert math.isinf(by_tag["avg_gpu_power_watts"].avg)
        assert by_tag["tokens_per_second_per_watt"].avg == pytest.approx(0.0)

    def test_negative_energy_delta_skipped(self) -> None:
        """Negative energy (counter reset) sums to negative total, which fails > 0 guard."""
        gpus = [_make_gpu_summary(0, energy_mj=-0.005, power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        # Negative energy fails the > 0 guard
        assert "total_energy_joules" not in tags
        assert "tokens_per_joule" not in tags
        # Power metrics unaffected
        assert "avg_gpu_power_watts" in tags

    def test_negative_power_skipped(self) -> None:
        """Negative power fails the > 0 guard."""
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=-100.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        assert "avg_gpu_power_watts" not in tags
        assert "tokens_per_second_per_watt" not in tags

    def test_mixed_valid_and_nan_gpus_only_sums_valid(self) -> None:
        """When some GPUs have NaN energy and others have valid, only valid GPUs contribute."""
        gpus = [
            _make_gpu_summary(0, energy_mj=0.001, power_watts=200.0),
            _make_gpu_summary(1, energy_mj=float("nan"), power_watts=300.0),
        ]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        by_tag = {r.tag: r for r in results}
        # NaN + 0.001 MJ = NaN, and NaN > 0 is False
        assert "total_energy_joules" not in by_tag
        # Power averages: (200 + 300) / 2 = 250 (NaN does not affect power)
        assert by_tag["avg_gpu_power_watts"].avg == pytest.approx(250.0)


# ============================================================
# Unit Tag Serialization
# ============================================================


class TestUnitTagSerialization:
    """Verify GenericMetricUnit.TOKENS_PER_JOULE and TOKENS_PER_SECOND_PER_WATT serialize correctly."""

    def test_tokens_per_joule_unit_tag_value(self) -> None:
        assert GenericMetricUnit.TOKENS_PER_JOULE.info.tag == "tokens/J"

    def test_tokens_per_second_per_watt_unit_tag_value(self) -> None:
        assert GenericMetricUnit.TOKENS_PER_SECOND_PER_WATT.info.tag == "tokens/sec/W"

    def test_energy_results_use_correct_unit_strings(self) -> None:
        """Verify the actual unit strings on MetricResult objects match expected tags."""
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        by_tag = {r.tag: r for r in results}
        assert by_tag["total_energy_joules"].unit == "J"
        assert by_tag["tokens_per_joule"].unit == "tokens/J"
        assert by_tag["avg_gpu_power_watts"].unit == "W"
        assert by_tag["tokens_per_second_per_watt"].unit == "tokens/sec/W"

    def test_metric_result_roundtrips_through_json(self) -> None:
        """MetricResult with energy units survives JSON serialization/deserialization."""
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        for result in results:
            json_str = result.model_dump_json()
            restored = MetricResult.model_validate_json(json_str)
            assert restored.tag == result.tag
            assert restored.unit == result.unit
            assert restored.avg == pytest.approx(result.avg)


# ============================================================
# Empty / Degenerate Inference Results
# ============================================================


class TestEmptyInferenceResults:
    """Verify behavior when inference_results list has no matching metrics."""

    def test_empty_inference_results_skips_derived_metrics(self) -> None:
        """No output_token_throughput or total_output_tokens means only base energy/power emitted."""
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=[],
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        assert "total_energy_joules" in tags
        assert "avg_gpu_power_watts" in tags
        # Derived metrics require throughput/tokens which are missing
        assert "tokens_per_joule" not in tags
        assert "tokens_per_second_per_watt" not in tags

    def test_unrelated_metrics_in_inference_results_ignored(self) -> None:
        """Metrics with tags other than output_token_throughput/total_output_tokens are ignored."""
        gpus = [_make_gpu_summary(0, energy_mj=0.001, power_watts=200.0)]
        telemetry = _make_telemetry(gpus)

        unrelated = [
            MetricResult(
                tag="inter_token_latency",
                header="Inter Token Latency",
                unit="ms",
                avg=5.0,
            ),
        ]

        results = compute(
            inference_results=unrelated,
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        tags = {r.tag for r in results}
        assert "total_energy_joules" in tags
        assert "avg_gpu_power_watts" in tags
        assert "tokens_per_joule" not in tags
        assert "tokens_per_second_per_watt" not in tags

    def test_gpus_with_no_metrics_returns_empty(self) -> None:
        """GPUs with empty metrics dict produce no results."""
        gpu = GpuSummary(
            gpu_index=0,
            gpu_name="Test GPU 0",
            gpu_uuid="GPU-0000-aaaa-bbbb-cccc",
            hostname="node1",
            metrics={},
        )
        telemetry = _make_telemetry([gpu])

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        assert results == []


# ============================================================
# Partial GPU Metrics
# ============================================================


class TestPartialGpuMetrics:
    """Verify GPUs with only some metrics contribute only to their relevant computations."""

    def test_some_gpus_missing_energy_only_valid_summed(self) -> None:
        """Only GPUs with energy_consumption contribute to total energy."""
        gpus = [
            _make_gpu_summary(0, energy_mj=0.002, power_watts=300.0),
            _make_gpu_summary(1, energy_mj=None, power_watts=200.0),
        ]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        by_tag = {r.tag: r for r in results}
        # Only GPU 0 has energy: 0.002 MJ = 2000 J
        assert by_tag["total_energy_joules"].avg == pytest.approx(2000.0)
        # Power averaged over both GPUs that have power
        assert by_tag["avg_gpu_power_watts"].avg == pytest.approx(250.0)

    def test_some_gpus_missing_power_only_valid_averaged(self) -> None:
        """Only GPUs with gpu_power_usage contribute to average power."""
        gpus = [
            _make_gpu_summary(0, energy_mj=0.001, power_watts=400.0),
            _make_gpu_summary(1, energy_mj=0.001, power_watts=None),
        ]
        telemetry = _make_telemetry(gpus)

        results = compute(
            inference_results=_make_inference_results(),
            telemetry=telemetry,
            benchmark_duration_ns=DURATION_NS,
        )

        by_tag = {r.tag: r for r in results}
        # Energy sums both: 0.002 MJ = 2000 J
        assert by_tag["total_energy_joules"].avg == pytest.approx(2000.0)
        # Only GPU 0 has power
        assert by_tag["avg_gpu_power_watts"].avg == pytest.approx(400.0)
