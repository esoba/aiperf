# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import (
    GenericMetricUnit,
    MetricFlags,
    MetricOverTimeUnit,
    MetricTimeUnit,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict


class BaseSessionMetric(BaseDerivedMetric[float]):
    """Base for session metrics whose values are injected by SessionMetricsResultsProcessor."""

    __is_abstract__ = True
    required_metrics = None
    flags = MetricFlags.NO_INDIVIDUAL_RECORDS

    def __init_subclass__(cls, **kwargs: object) -> None:
        cls.__is_abstract__ = False
        super().__init_subclass__(**kwargs)

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        raise NoMetricValue("Provided by SessionMetricsResultsProcessor")


class SessionDurationMetric(BaseSessionMetric):
    """Time from first request start to last request end per session."""

    tag = "session_duration"
    header = "Session Duration"
    short_header = "Sess Dur"
    unit = MetricTimeUnit.MILLISECONDS
    display_order = 1200


class SessionTurnsMetric(BaseSessionMetric):
    """Number of request-response turns per session."""

    tag = "session_turns"
    header = "Turns Per Session"
    short_header = "Turns/Sess"
    short_header_hide_unit = True
    unit = GenericMetricUnit.COUNT
    display_order = 1210


class SessionCountMetric(BaseSessionMetric):
    """Total number of completed sessions."""

    tag = "session_count"
    header = "Session Count"
    short_header = "Sessions"
    short_header_hide_unit = True
    unit = GenericMetricUnit.COUNT
    display_order = 1220
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.NO_INDIVIDUAL_RECORDS


class SessionThroughputMetric(BaseSessionMetric):
    """Sessions completed per second over the benchmark duration."""

    tag = "session_throughput"
    header = "Session Throughput"
    short_header = "Sess/sec"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.REQUESTS_PER_SECOND
    display_order = 1230
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.NO_INDIVIDUAL_RECORDS
