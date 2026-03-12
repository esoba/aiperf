# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AvgConcurrentRequestsMetric.

Focuses on:
- Little's Law calculation: sum(latencies) / duration
- Edge cases: single request, very high concurrency, fractional results
- Error handling: zero duration, missing dependencies
"""

import pytest
from pytest import param

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.avg_concurrent_requests_metric import (
    AvgConcurrentRequestsMetric,
)
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from tests.unit.metrics.conftest import create_metric_array

# ============================================================
# Happy Path Tests
# ============================================================


class TestAvgConcurrentRequestsHappyPath:
    """Verify Little's Law calculation under normal conditions."""

    @pytest.mark.parametrize(
        "latencies_ns,duration_ns,expected",
        [
            # 10 requests * 1s each / 10s = concurrency 1.0
            ([1_000_000_000] * 10, 10_000_000_000, 1.0),
            # 10 requests * 2s each / 10s = concurrency 2.0
            ([2_000_000_000] * 10, 10_000_000_000, 2.0),
            # 4 requests * 5s each / 10s = concurrency 2.0
            ([5_000_000_000] * 4, 10_000_000_000, 2.0),
            # 1 request * 500ms / 1s = concurrency 0.5
            ([500_000_000], 1_000_000_000, 0.5),
        ],
    )  # fmt: skip
    def test_derive_value_calculates_avg_concurrency(
        self,
        latencies_ns: list[int],
        duration_ns: int,
        expected: float,
    ) -> None:
        metric = AvgConcurrentRequestsMetric()

        metric_results = MetricResultsDict()
        metric_results[RequestLatencyMetric.tag] = create_metric_array(latencies_ns)
        metric_results[BenchmarkDurationMetric.tag] = duration_ns

        result = metric.derive_value(metric_results)
        assert result == pytest.approx(expected)


# ============================================================
# Edge Cases
# ============================================================


class TestAvgConcurrentRequestsEdgeCases:
    """Verify boundary conditions and realistic latency distributions."""

    @pytest.mark.parametrize(
        "latencies_ns,duration_ns,expected",
        [
            param(
                [100_000_000, 500_000_000, 2_000_000_000, 50_000_000, 800_000_000],
                5_000_000_000,
                0.69,
                id="mixed-latency-distribution",
            ),
            param(
                [10_000_000_000] * 100,
                10_000_000_000,
                100.0,
                id="high-concurrency-100-overlapping",
            ),
            param(
                [1],
                1_000_000_000,
                1e-9,
                id="near-zero-latency",
            ),
            param(
                [999_999_999_999],
                1_000_000_000_000,
                pytest.approx(1.0, abs=1e-9),
                id="single-request-nearly-full-duration",
            ),
        ],
    )  # fmt: skip
    def test_derive_value_edge_cases(
        self,
        latencies_ns: list[int],
        duration_ns: int,
        expected: float,
    ) -> None:
        metric = AvgConcurrentRequestsMetric()

        metric_results = MetricResultsDict()
        metric_results[RequestLatencyMetric.tag] = create_metric_array(latencies_ns)
        metric_results[BenchmarkDurationMetric.tag] = duration_ns

        result = metric.derive_value(metric_results)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_derive_value_realistic_multi_request_burst(self) -> None:
        """Simulate a burst of requests with varying latencies typical of LLM inference."""
        # 20 requests: mix of fast (200ms) and slow (3s) completions over 10s
        fast_latencies = [200_000_000] * 15  # 15 fast requests: 200ms each
        slow_latencies = [3_000_000_000] * 5  # 5 slow requests: 3s each
        all_latencies = fast_latencies + slow_latencies
        duration_ns = 10_000_000_000  # 10s

        # Expected: (15*0.2e9 + 5*3e9) / 10e9 = (3e9 + 15e9) / 10e9 = 1.8
        metric = AvgConcurrentRequestsMetric()
        metric_results = MetricResultsDict()
        metric_results[RequestLatencyMetric.tag] = create_metric_array(all_latencies)
        metric_results[BenchmarkDurationMetric.tag] = duration_ns

        result = metric.derive_value(metric_results)
        assert result == pytest.approx(1.8)


# ============================================================
# Error Handling
# ============================================================


class TestAvgConcurrentRequestsErrors:
    """Verify proper error handling."""

    def test_derive_value_zero_duration_raises(self) -> None:
        metric = AvgConcurrentRequestsMetric()

        metric_results = MetricResultsDict()
        metric_results[RequestLatencyMetric.tag] = create_metric_array([1_000_000_000])
        metric_results[BenchmarkDurationMetric.tag] = 0

        with pytest.raises(NoMetricValue, match="duration is zero"):
            metric.derive_value(metric_results)

    def test_derive_value_missing_latency_raises(self) -> None:
        metric = AvgConcurrentRequestsMetric()

        metric_results = MetricResultsDict()
        metric_results[BenchmarkDurationMetric.tag] = 10_000_000_000

        with pytest.raises(NoMetricValue):
            metric.derive_value(metric_results)

    def test_derive_value_missing_duration_raises(self) -> None:
        metric = AvgConcurrentRequestsMetric()

        metric_results = MetricResultsDict()
        metric_results[RequestLatencyMetric.tag] = create_metric_array([1_000_000_000])

        with pytest.raises(NoMetricValue):
            metric.derive_value(metric_results)
