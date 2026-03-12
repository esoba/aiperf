# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PerGPUOutputThroughputMetric.

Focuses on:
- Per-GPU division: throughput / world_size
- Edge cases: large world sizes, fractional results
- Error handling: zero/negative world_size, missing dependencies
"""

import pytest
from pytest import param

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.output_token_throughput_metrics import (
    OutputTokenThroughputMetric,
)
from aiperf.metrics.types.per_gpu_output_throughput_metric import (
    PerGPUOutputThroughputMetric,
)
from aiperf.metrics.types.world_size_metric import WorldSizeMetric

# ============================================================
# Happy Path Tests
# ============================================================


class TestPerGPUOutputThroughputHappyPath:
    """Verify per-GPU throughput division."""

    @pytest.mark.parametrize(
        "throughput,world_size,expected",
        [
            (1000.0, 1, 1000.0),
            (1000.0, 2, 500.0),
            (1000.0, 4, 250.0),
            (1000.0, 8, 125.0),
            (500.0, 2, 250.0),
        ],
    )  # fmt: skip
    def test_derive_value_calculates_per_gpu_throughput(
        self,
        throughput: float,
        world_size: int,
        expected: float,
    ) -> None:
        metric = PerGPUOutputThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[OutputTokenThroughputMetric.tag] = throughput
        metric_results[WorldSizeMetric.tag] = world_size

        result = metric.derive_value(metric_results)
        assert result == pytest.approx(expected)


# ============================================================
# Edge Cases
# ============================================================


class TestPerGPUOutputThroughputEdgeCases:
    """Verify boundary conditions."""

    @pytest.mark.parametrize(
        "throughput,world_size,expected",
        [
            param(10000.0, 64, 156.25, id="large-cluster-64-gpus"),
            param(10000.0, 128, 78.125, id="large-cluster-128-gpus"),
            param(0.0, 4, 0.0, id="zero-throughput"),
            param(1.5, 1, 1.5, id="fractional-throughput-single-gpu"),
            param(100.0, 3, pytest.approx(33.333333, rel=1e-5), id="non-power-of-two-gpus"),
        ],
    )  # fmt: skip
    def test_derive_value_edge_cases(
        self,
        throughput: float,
        world_size: int,
        expected: float,
    ) -> None:
        metric = PerGPUOutputThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[OutputTokenThroughputMetric.tag] = throughput
        metric_results[WorldSizeMetric.tag] = world_size

        result = metric.derive_value(metric_results)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_world_size_from_pre_seeded_value(self) -> None:
        """Simulate world_size being pre-seeded by the processor (not from derive)."""
        metric = PerGPUOutputThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[OutputTokenThroughputMetric.tag] = 2400.0
        # Pre-seeded world_size=4 (as MetricResultsProcessor would do)
        metric_results[WorldSizeMetric.tag] = 4

        result = metric.derive_value(metric_results)
        assert result == pytest.approx(600.0)

    def test_required_metrics_includes_world_size_and_throughput(self) -> None:
        assert WorldSizeMetric.tag in PerGPUOutputThroughputMetric.required_metrics
        assert (
            OutputTokenThroughputMetric.tag
            in PerGPUOutputThroughputMetric.required_metrics
        )


# ============================================================
# Error Handling
# ============================================================


class TestPerGPUOutputThroughputErrors:
    """Verify proper error handling."""

    def test_derive_value_zero_world_size_raises(self) -> None:
        metric = PerGPUOutputThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[OutputTokenThroughputMetric.tag] = 1000.0
        metric_results[WorldSizeMetric.tag] = 0

        with pytest.raises(NoMetricValue, match="positive"):
            metric.derive_value(metric_results)

    def test_derive_value_negative_world_size_raises(self) -> None:
        metric = PerGPUOutputThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[OutputTokenThroughputMetric.tag] = 1000.0
        metric_results[WorldSizeMetric.tag] = -1

        with pytest.raises(NoMetricValue, match="positive"):
            metric.derive_value(metric_results)

    def test_derive_value_missing_throughput_raises(self) -> None:
        metric = PerGPUOutputThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[WorldSizeMetric.tag] = 2

        with pytest.raises(NoMetricValue):
            metric.derive_value(metric_results)

    def test_derive_value_missing_world_size_raises(self) -> None:
        metric = PerGPUOutputThroughputMetric()

        metric_results = MetricResultsDict()
        metric_results[OutputTokenThroughputMetric.tag] = 1000.0

        with pytest.raises(NoMetricValue):
            metric.derive_value(metric_results)
