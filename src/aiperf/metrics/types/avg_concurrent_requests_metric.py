# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


class AvgConcurrentRequestsMetric(BaseDerivedMetric[float]):
    """
    Metric for average number of concurrent requests during the benchmark.

    Uses Little's Law:
        Avg Concurrent Requests = Sum(Request Latencies) / Benchmark Duration
    """

    tag = "avg_concurrent_requests"
    header = "Avg Concurrent Requests"
    short_header = "Avg Concurrency"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    display_order = 950
    flags = MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        RequestLatencyMetric.tag,
        BenchmarkDurationMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        request_latency_array = metric_results.get_or_raise(RequestLatencyMetric)
        benchmark_duration = metric_results.get_or_raise(BenchmarkDurationMetric)

        if benchmark_duration == 0:
            raise NoMetricValue(
                "Benchmark duration is zero, cannot calculate avg concurrent requests"
            )

        return request_latency_array.sum / benchmark_duration  # type: ignore
