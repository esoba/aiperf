# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Energy efficiency metrics derived from GPU telemetry and inference results."""

from aiperf.common.enums import (
    EnergyMetricUnit,
    GenericMetricUnit,
    PowerMetricUnit,
)
from aiperf.common.models import (
    GpuSummary,
    MetricResult,
    TelemetryExportData,
)


def _extract_output_token_throughput(
    inference_results: list[MetricResult],
) -> float | None:
    """Find the output_token_throughput metric avg value from inference results.

    Args:
        inference_results: List of inference MetricResult objects

    Returns:
        The avg output token throughput (tokens/sec), or None if not found
    """
    for result in inference_results:
        if result.tag == "output_token_throughput" and result.avg is not None:
            return result.avg
    return None


def _extract_total_output_tokens(
    inference_results: list[MetricResult],
) -> float | None:
    """Find the total_output_tokens metric avg value from inference results.

    Args:
        inference_results: List of inference MetricResult objects

    Returns:
        The total output tokens count, or None if not found
    """
    for result in inference_results:
        if result.tag == "total_output_tokens" and result.avg is not None:
            return result.avg
    return None


def _collect_gpu_summaries(
    telemetry: TelemetryExportData,
) -> list[GpuSummary]:
    """Collect all GpuSummary objects across all endpoints.

    Args:
        telemetry: Telemetry export data with endpoint hierarchy

    Returns:
        Flat list of GpuSummary objects across all endpoints
    """
    summaries: list[GpuSummary] = []
    for endpoint_data in telemetry.endpoints.values():
        summaries.extend(endpoint_data.gpus.values())
    return summaries


def compute(
    inference_results: list[MetricResult],
    telemetry: TelemetryExportData | None,
    benchmark_duration_ns: int,
) -> list[MetricResult]:
    """Compute energy efficiency metrics from inference results and GPU telemetry.

    Derives four metrics:
    1. total_energy_joules - Total energy consumed across all GPUs (J)
    2. tokens_per_joule - Output tokens per joule of energy
    3. avg_gpu_power_watts - Mean power draw across all GPUs (W)
    4. tokens_per_second_per_watt - Token throughput per watt of power

    Args:
        inference_results: List of inference MetricResult objects (must contain output_token_throughput)
        telemetry: GPU telemetry export data, or None if telemetry was disabled
        benchmark_duration_ns: Benchmark duration in nanoseconds

    Returns:
        List of energy efficiency MetricResult objects (may be empty)
    """
    if telemetry is None:
        return []

    gpu_summaries = _collect_gpu_summaries(telemetry)
    if not gpu_summaries:
        return []

    results: list[MetricResult] = []

    # Compute total energy (joules) across all GPUs
    # energy_consumption is reported in MJ (megajoules) as a delta value in avg field
    total_energy_joules = _compute_total_energy_joules(gpu_summaries)
    if total_energy_joules is not None and total_energy_joules > 0:
        results.append(
            MetricResult(
                tag="total_energy_joules",
                header="Total Energy Consumption",
                unit=EnergyMetricUnit.JOULE.info.tag,
                avg=total_energy_joules,
            )
        )

        # tokens_per_joule requires total output tokens
        total_output_tokens = _extract_total_output_tokens(inference_results)
        if total_output_tokens is not None and total_output_tokens > 0:
            results.append(
                MetricResult(
                    tag="tokens_per_joule",
                    header="Tokens Per Joule",
                    unit=GenericMetricUnit.TOKENS_PER_JOULE.info.tag,
                    avg=total_output_tokens / total_energy_joules,
                )
            )

    # Compute average GPU power (watts) across all GPUs
    avg_power_watts = _compute_avg_gpu_power_watts(gpu_summaries)
    if avg_power_watts is not None and avg_power_watts > 0:
        results.append(
            MetricResult(
                tag="avg_gpu_power_watts",
                header="Average GPU Power",
                unit=PowerMetricUnit.WATT.info.tag,
                avg=avg_power_watts,
            )
        )

        # tokens_per_second_per_watt requires output token throughput
        throughput = _extract_output_token_throughput(inference_results)
        if throughput is not None and throughput > 0:
            results.append(
                MetricResult(
                    tag="tokens_per_second_per_watt",
                    header="Tokens Per Second Per Watt",
                    unit=GenericMetricUnit.TOKENS_PER_SECOND_PER_WATT.info.tag,
                    avg=throughput / avg_power_watts,
                )
            )

    return results


def _compute_total_energy_joules(
    gpu_summaries: list[GpuSummary],
) -> float | None:
    """Sum energy_consumption deltas across all GPUs, converting MJ to J.

    Args:
        gpu_summaries: List of GpuSummary objects

    Returns:
        Total energy in joules, or None if no energy data available
    """
    total_mj: float = 0.0
    found = False
    for gpu in gpu_summaries:
        energy_metric = gpu.metrics.get("energy_consumption")
        if energy_metric is not None and energy_metric.avg is not None:
            total_mj += energy_metric.avg
            found = True
    if not found:
        return None
    # Convert MJ to J
    return EnergyMetricUnit.MEGAJOULE.info.convert_to(EnergyMetricUnit.JOULE, total_mj)


def _compute_avg_gpu_power_watts(
    gpu_summaries: list[GpuSummary],
) -> float | None:
    """Compute mean gpu_power_usage avg across all GPUs.

    Args:
        gpu_summaries: List of GpuSummary objects

    Returns:
        Mean power in watts, or None if no power data available
    """
    power_values: list[float] = []
    for gpu in gpu_summaries:
        power_metric = gpu.metrics.get("gpu_power_usage")
        if power_metric is not None and power_metric.avg is not None:
            power_values.append(power_metric.avg)
    if not power_values:
        return None
    return sum(power_values) / len(power_values)
