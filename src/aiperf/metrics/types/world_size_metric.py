# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict


class WorldSizeMetric(BaseDerivedMetric[int]):
    """
    Metric for the world size (number of GPUs) used in the benchmark.

    Returns 1 by default; overridden via pre-seeding in MetricResultsProcessor.
    """

    tag = "world_size"
    header = "World Size"
    short_header = "GPUs"
    unit = GenericMetricUnit.COUNT
    display_order = 0
    flags = MetricFlags.INTERNAL | MetricFlags.NO_CONSOLE
    required_metrics = None

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> int:
        return 1
