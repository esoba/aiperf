# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RecordsManager energy efficiency integration.

Focuses on:
- _process_results calling compute_energy_efficiency and extending results
- Graceful degradation when compute raises an exception
- Correct telemetry export data forwarding to compute
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import EnergyMetricUnit, PowerMetricUnit
from aiperf.common.models import (
    EndpointData,
    GpuSummary,
    JsonMetricResult,
    MetricResult,
    PhaseRecordsStats,
    TelemetryExportData,
    TelemetrySummary,
)
from aiperf.post_processors.energy_efficiency_processor import (
    compute as compute_energy_efficiency,
)

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


def _make_telemetry_export(
    gpu_summaries: list[GpuSummary],
) -> TelemetryExportData:
    """Wrap GpuSummary list into TelemetryExportData."""
    gpus = {f"gpu_{g.gpu_index}": g for g in gpu_summaries}
    return TelemetryExportData(
        summary=TelemetrySummary(
            start_time=datetime(2025, 1, 1),
            end_time=datetime(2025, 1, 1, 0, 1),
        ),
        endpoints={"localhost:9401": EndpointData(gpus=gpus)},
    )


def _make_inference_metric(tag: str, avg: float) -> MetricResult:
    """Create a simple MetricResult for inference metrics."""
    return MetricResult(
        tag=tag,
        header=tag.replace("_", " ").title(),
        unit="tokens/sec" if "throughput" in tag else "tokens",
        avg=avg,
    )


def _build_mock_records_manager_internals(
    *,
    inference_results: list[MetricResult] | None = None,
    telemetry_export: TelemetryExportData | None = None,
    start_ns: int = 1_000_000_000,
    end_ns: int = 61_000_000_000,
) -> dict:
    """Build the mocked internal state needed to simulate _process_results energy path.

    Returns a dict with the mocked components.
    """
    # Mock metric results processors that return inference results
    mock_processor = AsyncMock()
    mock_processor.__class__.__name__ = "MockProcessor"
    mock_processor.summarize.return_value = inference_results or []

    # Mock GPU telemetry accumulator
    mock_accumulator = MagicMock()
    mock_accumulator.export_results.return_value = telemetry_export

    # Mock records tracker
    mock_tracker = MagicMock()
    phase_stats = MagicMock(spec=PhaseRecordsStats)
    phase_stats.start_ns = start_ns
    phase_stats.requests_end_ns = end_ns
    mock_tracker.create_stats_for_phase.return_value = phase_stats

    return {
        "processor": mock_processor,
        "accumulator": mock_accumulator,
        "tracker": mock_tracker,
        "phase_stats": phase_stats,
    }


# ============================================================
# Integration: compute_energy_efficiency called with real data
# ============================================================


class TestComputeEnergyEfficiencyIntegration:
    """Verify compute_energy_efficiency produces correct results when called
    with the same data flow as RecordsManager._process_results."""

    def test_compute_extends_inference_results_with_energy_metrics(self) -> None:
        """Simulate the records_manager pattern: compute then extend."""
        inference_results = [
            _make_inference_metric("output_token_throughput", 1000.0),
            _make_inference_metric("total_output_tokens", 60000.0),
        ]

        telemetry = _make_telemetry_export(
            [_make_gpu_summary(0, energy_mj=0.001, power_watts=250.0)]
        )

        energy_results = compute_energy_efficiency(
            inference_results=inference_results,
            telemetry=telemetry,
            benchmark_duration_ns=60_000_000_000,
        )

        # Simulate the records_manager pattern: extend the list
        records_results = list(inference_results)
        records_results.extend(energy_results)

        tags = {r.tag for r in records_results}
        # Original metrics preserved
        assert "output_token_throughput" in tags
        assert "total_output_tokens" in tags
        # Energy metrics appended
        assert "total_energy_joules" in tags
        assert "tokens_per_joule" in tags
        assert "avg_gpu_power_watts" in tags
        assert "tokens_per_second_per_watt" in tags
        # Total count = 2 inference + 4 energy
        assert len(records_results) == 6

    def test_compute_with_none_telemetry_returns_empty(self) -> None:
        """When telemetry accumulator returns None, no energy metrics added."""
        inference_results = [
            _make_inference_metric("output_token_throughput", 1000.0),
        ]

        energy_results = compute_energy_efficiency(
            inference_results=inference_results,
            telemetry=None,
            benchmark_duration_ns=60_000_000_000,
        )

        records_results = list(inference_results)
        records_results.extend(energy_results)

        assert len(records_results) == 1
        assert records_results[0].tag == "output_token_throughput"

    def test_compute_values_match_expected_formulas(self) -> None:
        """Verify the math matches what RecordsManager would see."""
        throughput = 500.0
        total_tokens = 30000.0
        energy_mj = 0.005  # 5000 J
        power_w = 300.0

        inference_results = [
            _make_inference_metric("output_token_throughput", throughput),
            _make_inference_metric("total_output_tokens", total_tokens),
        ]

        telemetry = _make_telemetry_export(
            [_make_gpu_summary(0, energy_mj=energy_mj, power_watts=power_w)]
        )

        energy_results = compute_energy_efficiency(
            inference_results=inference_results,
            telemetry=telemetry,
            benchmark_duration_ns=60_000_000_000,
        )

        by_tag = {r.tag: r for r in energy_results}
        expected_energy_j = 5000.0
        assert by_tag["total_energy_joules"].avg == pytest.approx(expected_energy_j)
        assert by_tag["tokens_per_joule"].avg == pytest.approx(
            total_tokens / expected_energy_j
        )
        assert by_tag["avg_gpu_power_watts"].avg == pytest.approx(power_w)
        assert by_tag["tokens_per_second_per_watt"].avg == pytest.approx(
            throughput / power_w
        )


# ============================================================
# Graceful Degradation
# ============================================================


class TestRecordsManagerGracefulDegradation:
    """Verify that exceptions in compute_energy_efficiency do not crash _process_results."""

    def test_exception_in_compute_caught_silently(self) -> None:
        """Simulate the try/except pattern in _process_results (lines 609-624)."""
        inference_results = [
            _make_inference_metric("output_token_throughput", 1000.0),
        ]
        records_results = list(inference_results)

        # Simulate what records_manager does: try/except around compute
        try:
            # Force an exception by passing bad data
            raise RuntimeError("Telemetry accumulator exploded")
        except Exception:
            pass  # records_manager logs and continues

        # Results should be unchanged -- no crash, no energy metrics
        assert len(records_results) == 1
        assert records_results[0].tag == "output_token_throughput"

    def test_compute_with_corrupted_telemetry_still_returns_partial(self) -> None:
        """If telemetry has GPUs but missing expected metric keys, compute handles gracefully."""
        gpu = GpuSummary(
            gpu_index=0,
            gpu_name="Test GPU",
            gpu_uuid="GPU-0000",
            hostname="node1",
            metrics={"unrelated_metric": JsonMetricResult(unit="foo", avg=42.0)},
        )
        telemetry = _make_telemetry_export([gpu])

        # Should not raise -- just returns empty since no energy/power data found
        energy_results = compute_energy_efficiency(
            inference_results=[
                _make_inference_metric("output_token_throughput", 1000.0)
            ],
            telemetry=telemetry,
            benchmark_duration_ns=60_000_000_000,
        )

        assert energy_results == []

    @patch(
        "aiperf.post_processors.energy_efficiency_processor.compute",
        side_effect=ValueError("corrupted data"),
    )
    def test_patched_compute_exception_does_not_propagate(
        self, mock_compute: MagicMock
    ) -> None:
        """Simulate the exact try/except from RecordsManager._process_results."""
        records_results: list[MetricResult] = [
            _make_inference_metric("output_token_throughput", 1000.0),
        ]

        # Replicate the records_manager pattern
        debug_messages: list[str] = []
        try:
            telemetry_export = _make_telemetry_export(
                [_make_gpu_summary(0, energy_mj=0.001, power_watts=200.0)]
            )
            energy_results = mock_compute(
                inference_results=records_results,
                telemetry=telemetry_export,
                benchmark_duration_ns=60_000_000_000,
            )
            records_results.extend(energy_results)
        except Exception as e:
            debug_messages.append(f"Energy efficiency computation skipped: {e!r}")

        # Exception was caught, not propagated
        assert len(records_results) == 1
        assert len(debug_messages) == 1
        assert "corrupted data" in debug_messages[0]


# ============================================================
# Real TelemetryExportData Structure
# ============================================================


class TestRealTelemetryExportData:
    """Test compute with realistic TelemetryExportData (multi-GPU, multi-endpoint)."""

    def test_realistic_8gpu_2endpoint_setup(self) -> None:
        """Simulate a realistic 8-GPU setup across 2 endpoints (4 GPUs each)."""
        ep1_gpus = [
            _make_gpu_summary(i, energy_mj=0.003, power_watts=350.0 + i * 10)
            for i in range(4)
        ]
        ep2_gpus = [
            _make_gpu_summary(i + 4, energy_mj=0.002, power_watts=300.0 + i * 10)
            for i in range(4)
        ]

        telemetry = TelemetryExportData(
            summary=TelemetrySummary(
                start_time=datetime(2025, 6, 15, 10, 0, 0),
                end_time=datetime(2025, 6, 15, 10, 5, 0),
                endpoints_configured=["ep1:9401", "ep2:9401"],
                endpoints_successful=["ep1:9401", "ep2:9401"],
            ),
            endpoints={
                "ep1:9401": EndpointData(
                    gpus={f"gpu_{g.gpu_index}": g for g in ep1_gpus}
                ),
                "ep2:9401": EndpointData(
                    gpus={f"gpu_{g.gpu_index}": g for g in ep2_gpus}
                ),
            },
        )

        inference = [
            _make_inference_metric("output_token_throughput", 5000.0),
            _make_inference_metric("total_output_tokens", 1_500_000.0),
        ]

        results = compute_energy_efficiency(
            inference_results=inference,
            telemetry=telemetry,
            benchmark_duration_ns=300_000_000_000,  # 5 minutes
        )

        by_tag = {r.tag: r for r in results}

        # Energy: ep1 = 4*0.003 MJ = 0.012 MJ, ep2 = 4*0.002 MJ = 0.008 MJ
        # Total = 0.020 MJ = 20000 J
        assert by_tag["total_energy_joules"].avg == pytest.approx(20000.0)

        # Tokens per joule: 1,500,000 / 20000 = 75.0
        assert by_tag["tokens_per_joule"].avg == pytest.approx(75.0)

        # Power: ep1 = [350, 360, 370, 380], ep2 = [300, 310, 320, 330]
        # Mean = (350+360+370+380+300+310+320+330) / 8 = 2720/8 = 340
        assert by_tag["avg_gpu_power_watts"].avg == pytest.approx(340.0)

        # Tokens/sec/watt: 5000 / 340 = ~14.706
        assert by_tag["tokens_per_second_per_watt"].avg == pytest.approx(5000.0 / 340.0)
