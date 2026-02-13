# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Generic

from aiperf.common.enums import MetricType, MetricValueTypeVarT
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class BaseRecordMetric(
    Generic[MetricValueTypeVarT], BaseMetric[MetricValueTypeVarT], ABC
):
    """A base class for record-based metrics. These metrics are computed for each record,
    and are independent of other records. The final results will be a list of values, one for each record.

    NOTE: Set the generic type to be the type of the individual values, and NOT a list, unless the metric produces
    a list *for every record*. In that case, the result will be a list of lists.

    Examples:
    ```python
    class InputSequenceLengthMetric(BaseRecordMetric[int]):
        # ... Metric attributes ...
        # ... Input validation ...

        @classmethod
        def _parse_record(
            cls,
            record: ParsedResponseRecord,
            record_metrics: MetricRecordDict,
        ) -> int:
            return record.token_counts.input
    ```
    """

    type = MetricType.RECORD

    @classmethod
    def parse_record(
        cls, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> MetricValueTypeVarT:
        """Parse a single record and return the metric value."""
        cls._require_valid_record(record)
        cls._check_metrics(record_metrics)
        return cls._parse_record(record, record_metrics)

    @classmethod
    @abstractmethod
    def _parse_record(
        cls, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> MetricValueTypeVarT:
        """Parse a single record and return the metric value. This method is implemented by subclasses.
        This method is called after the required metrics are checked, so it can assume that the required metrics are available.
        This method is called after the record is checked, so it can assume that the record is valid.

        Raises:
            ValueError: If the metric cannot be computed for the given inputs.
        """
        raise NotImplementedError("Subclasses must implement this method")
