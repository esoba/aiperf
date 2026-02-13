# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import ClassVar, Generic

from aiperf.common.enums import AggregationKind, MetricType, MetricValueTypeVarT
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class BaseAggregateMetric(
    Generic[MetricValueTypeVarT], BaseMetric[MetricValueTypeVarT], ABC
):
    """A base class for aggregate metrics that produce a single scalar from many records.

    Each subclass declares an ``aggregation_kind`` ClassVar (SUM, MAX, MIN) and implements
    ``_parse_record`` to extract the per-record value. Aggregation is performed externally
    by ``MetricsAccumulator`` using vectorized numpy operations — the metric class itself
    is stateless after ``parse_record`` returns.

    Examples:
    ```python
    class RequestCountMetric(BaseAggregateMetric[int]):
        aggregation_kind = AggregationKind.SUM

        @classmethod
        def _parse_record(cls, record: ParsedResponseRecord, record_metrics: MetricRecordDict) -> int:
            return 1
    ```
    """

    type = MetricType.AGGREGATE
    aggregation_kind: ClassVar[AggregationKind]

    @classmethod
    def parse_record(
        cls, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> MetricValueTypeVarT:
        """Parse the record and return the individual value.

        Raises:
            ValueError: If the metric cannot be computed for the given inputs.
        """
        cls._require_valid_record(record)
        cls._check_metrics(record_metrics)
        return cls._parse_record(record, record_metrics)

    @classmethod
    @abstractmethod
    def _parse_record(
        cls, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> MetricValueTypeVarT:
        """Parse the record and *return* the individual value based on this record alone.

        This method is called after the required metrics are checked, so it can assume that the required metrics are available.
        This method is called after the record is checked, so it can assume that the record is valid.

        Raises:
            ValueError: If the metric cannot be computed for the given inputs.
        """
        raise NotImplementedError("Subclasses must implement this method")
