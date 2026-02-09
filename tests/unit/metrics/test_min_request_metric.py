# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from aiperf.common.enums import AggregationKind
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric
from aiperf.post_processors.metrics_accumulator import _AGGREGATE_FUNCS
from tests.unit.metrics.conftest import create_record


class TestMinRequestTimestampMetric:
    def test_min_request_timestamp(self):
        """Test min request timestamp extraction"""
        record = create_record(start_ns=1500)

        metric = MinRequestTimestampMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 1500  # Uses timestamp_ns which equals start_ns

    def test_min_request_aggregation(self):
        """Test aggregation finds minimum request timestamp"""
        records = [
            create_record(start_ns=2000),  # timestamp: 2000
            create_record(start_ns=1000),  # timestamp: 1000 (minimum)
            create_record(start_ns=3000),  # timestamp: 3000
        ]

        metric = MinRequestTimestampMetric()
        values = [metric.parse_record(record, MetricRecordDict()) for record in records]

        result = _AGGREGATE_FUNCS[AggregationKind.MIN](np.array(values))
        assert result == 1000
