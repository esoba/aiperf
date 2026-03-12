# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.output_token_throughput_metrics import (
    OutputTokenThroughputMetric,
)
from aiperf.metrics.types.world_size_metric import WorldSizeMetric


class PerGPUOutputThroughputMetric(BaseDerivedMetric[float]):
    """
    Metric for per-GPU output token throughput.

    Formula:
        Per-GPU Output Token Throughput = Output Token Throughput / World Size
    """

    tag = "per_gpu_output_token_throughput"
    header = "Per-GPU Output Token Throughput"
    short_header = "Per-GPU TPS"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND
    display_order = 810
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        OutputTokenThroughputMetric.tag,
        WorldSizeMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        output_throughput = metric_results.get_or_raise(OutputTokenThroughputMetric)
        world_size = metric_results.get_or_raise(WorldSizeMetric)

        if world_size <= 0:  # type: ignore
            raise NoMetricValue(
                "World size must be positive to calculate per-GPU throughput"
            )

        return output_throughput / world_size  # type: ignore
