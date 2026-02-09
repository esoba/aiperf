# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MetricsAccumulator."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from aiperf.common.config import OutputConfig, UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import AggregationKind, MetricType
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import MetricResult
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from aiperf.post_processors.metrics_accumulator import (
    _AGGREGATE_FUNCS,
    MetricsAccumulator,
    MetricsSummary,
)
from tests.unit.post_processors.conftest import (
    create_metric_records_message,
)


class TestMetricsAccumulator:
    """Test cases for MetricsAccumulator."""

    def test_initialization(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processor initialization sets up necessary data structures."""
        processor = MetricsAccumulator(mock_user_config)

        assert isinstance(processor._derive_funcs, dict)
        assert isinstance(processor._time_series, dict)
        assert isinstance(processor._tags_to_types, dict)
        assert isinstance(processor._aggregation_kinds, dict)
        assert isinstance(processor._records, list)

    @pytest.mark.asyncio
    async def test_process_record_record_metric(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processing record metric stores values in time series."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        message = create_metric_records_message(
            x_request_id="test-1",
            results=[{"test_record": 42.0}],
        )
        await processor.process_record(message.to_data())

        assert "test_record" in processor._time_series
        assert len(processor._time_series["test_record"]) == 1
        assert list(processor._time_series["test_record"].values) == [42.0]

        # New data should expand the time series
        message2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=1_000_000_001,
            results=[{"test_record": 84.0}],
        )
        await processor.process_record(message2.to_data())
        assert list(processor._time_series["test_record"].values) == [42.0, 84.0]

    @pytest.mark.asyncio
    async def test_process_record_record_metric_list_values(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processing record metric with list values extends the time series."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        message = create_metric_records_message(
            x_request_id="test-1",
            results=[{"test_record": [10.0, 20.0, 30.0]}],
        )
        await processor.process_record(message.to_data())

        assert "test_record" in processor._time_series
        assert list(processor._time_series["test_record"].values) == [10.0, 20.0, 30.0]

    @pytest.mark.asyncio
    async def test_process_record_aggregate_metric(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processing aggregate metric stores values in time series."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }

        message1 = create_metric_records_message(
            x_request_id="test-1",
            results=[{RequestCountMetric.tag: 5}],
        )
        await processor.process_record(message1.to_data())

        # Values stored in time series, not in running instances
        assert RequestCountMetric.tag in processor._time_series
        assert list(processor._time_series[RequestCountMetric.tag].values) == [5.0]

        message2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=1_000_000_001,
            results=[{RequestCountMetric.tag: 3}],
        )
        await processor.process_record(message2.to_data())
        assert list(processor._time_series[RequestCountMetric.tag].values) == [5.0, 3.0]

    @pytest.mark.asyncio
    async def test_aggregate_sum_computed_at_summary_time(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test aggregate SUM values are computed vectorized from stored values."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }
        processor._metric_classes = {RequestCountMetric.tag: RequestCountMetric()}

        for i in range(3):
            msg = create_metric_records_message(
                x_request_id=f"test-{i}",
                request_start_ns=1_000_000_000 + i,
                results=[{RequestCountMetric.tag: 5}],
            )
            await processor.process_record(msg.to_data())

        results = processor._build_results_dict()
        assert results[RequestCountMetric.tag] == 15.0

    @pytest.mark.asyncio
    async def test_record_count_and_iter_requests(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test record_count and iter_requests work correctly."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {}

        msg1 = create_metric_records_message(x_request_id="test-1")
        msg2 = create_metric_records_message(
            x_request_id="test-2", request_start_ns=1_000_000_001
        )
        data1 = msg1.to_data()
        data2 = msg2.to_data()

        await processor.process_record(data1)
        await processor.process_record(data2)

        assert processor.record_count == 2
        assert list(processor.iter_requests()) == [data1, data2]


class TestAggregationKind:
    """Test AggregationKind enum and vectorized aggregate functions."""

    def test_sum(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert _AGGREGATE_FUNCS[AggregationKind.SUM](values) == 10.0

    def test_max(self) -> None:
        values = np.array([1.0, 4.0, 2.0, 3.0])
        assert _AGGREGATE_FUNCS[AggregationKind.MAX](values) == 4.0

    def test_min(self) -> None:
        values = np.array([3.0, 1.0, 4.0, 2.0])
        assert _AGGREGATE_FUNCS[AggregationKind.MIN](values) == 1.0

    def test_aggregate_kind_on_request_count(self) -> None:
        assert RequestCountMetric.aggregation_kind == AggregationKind.SUM

    def test_aggregate_kind_on_min_request_timestamp(self) -> None:
        from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric

        assert MinRequestTimestampMetric.aggregation_kind == AggregationKind.MIN

    def test_aggregate_kind_on_max_response_timestamp(self) -> None:
        from aiperf.metrics.types.max_response_metric import (
            MaxResponseTimestampMetric,
        )

        assert MaxResponseTimestampMetric.aggregation_kind == AggregationKind.MAX


class TestQueryTimeRange:
    @pytest.mark.asyncio
    async def test_empty(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        processor = MetricsAccumulator(mock_user_config)
        assert processor.query_time_range(0, 10_000) == []

    @pytest.mark.asyncio
    async def test_single_record_inside(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {}
        record = create_metric_records_message(
            x_request_id="test-1", request_start_ns=5_000
        ).to_data()
        await processor.process_record(record)
        assert processor.query_time_range(0, 10_000) == [record]

    @pytest.mark.asyncio
    async def test_single_record_outside(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {}
        record = create_metric_records_message(
            x_request_id="test-1", request_start_ns=15_000
        ).to_data()
        await processor.process_record(record)
        assert processor.query_time_range(0, 10_000) == []

    @pytest.mark.asyncio
    async def test_boundary_inclusive_start_exclusive_end(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {}
        record1 = create_metric_records_message(
            x_request_id="test-1", request_start_ns=1_000
        ).to_data()
        record2 = create_metric_records_message(
            x_request_id="test-2", request_start_ns=2_000
        ).to_data()
        await processor.process_record(record1)
        await processor.process_record(record2)
        # [1_000, 2_000) should include 1_000 but exclude 2_000
        assert processor.query_time_range(1_000, 2_000) == [record1]

    @pytest.mark.asyncio
    async def test_multiple_records_filtering(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {}
        records = []
        for i, ts in enumerate([100, 200, 300, 400, 500]):
            r = create_metric_records_message(
                x_request_id=f"test-{i}", request_start_ns=ts
            ).to_data()
            records.append(r)
            await processor.process_record(r)

        result = processor.query_time_range(200, 400)
        assert result == [records[1], records[2]]


class TestSummarize:
    @pytest.mark.asyncio
    async def test_summarize_returns_metrics_summary(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test summarize returns MetricsSummary wrapping MetricResult objects."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {RequestLatencyMetric.tag: MetricType.RECORD}
        processor._metric_classes = {RequestLatencyMetric.tag: RequestLatencyMetric()}

        # Manually add data to time series
        from aiperf.post_processors.inference_time_series import InferenceTimeSeries

        ts = InferenceTimeSeries()
        ts.append(0, 42.0)
        processor._time_series[RequestLatencyMetric.tag] = ts

        summary = await processor.summarize()

        assert isinstance(summary, MetricsSummary)
        assert len(summary.results) == 1
        assert RequestLatencyMetric.tag in summary.results
        assert isinstance(summary.results[RequestLatencyMetric.tag], MetricResult)
        assert summary.timeslices is None

    @pytest.mark.asyncio
    async def test_summarize_with_derived_metrics(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test derived metrics are computed during summarize."""

        def mock_derive_func(results_dict: MetricResultsDict) -> float:
            return 100.0

        processor = MetricsAccumulator(mock_user_config)
        processor._derive_funcs = {RequestThroughputMetric.tag: mock_derive_func}
        processor._metric_classes = {
            RequestThroughputMetric.tag: RequestThroughputMetric()
        }

        summary = await processor.summarize()

        assert isinstance(summary, MetricsSummary)
        assert RequestThroughputMetric.tag in summary.results

    @pytest.mark.asyncio
    async def test_summarize_derived_handles_no_metric_value(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test derived metrics gracefully handle NoMetricValue."""

        def failing_derive_func(results_dict: MetricResultsDict) -> float:
            raise NoMetricValue("Cannot derive value")

        processor = MetricsAccumulator(mock_user_config)
        processor._derive_funcs = {RequestThroughputMetric.tag: failing_derive_func}
        processor._metric_classes = {}

        with patch.object(processor, "debug") as mock_debug:
            summary = await processor.summarize()
            assert RequestThroughputMetric.tag not in summary.results
            mock_debug.assert_called()

    @pytest.mark.asyncio
    async def test_summarize_derived_handles_value_error(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test derived metrics gracefully handle ValueError."""

        def failing_derive_func(results_dict: MetricResultsDict) -> float:
            raise ValueError("Calculation error")

        processor = MetricsAccumulator(mock_user_config)
        processor._derive_funcs = {RequestThroughputMetric.tag: failing_derive_func}
        processor._metric_classes = {}

        with patch.object(processor, "warning") as mock_warning:
            summary = await processor.summarize()
            assert RequestThroughputMetric.tag not in summary.results
            mock_warning.assert_called()


class TestTimesliceSummarize:
    @pytest.mark.asyncio
    async def test_summarize_with_timeslices(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test summarize produces timeslice results when slice_duration is set."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}
        processor._metric_classes = {"test_record": RequestLatencyMetric()}

        # Process records in two different 1-second windows
        msg1 = create_metric_records_message(
            x_request_id="test-1",
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            results=[{"test_record": 42.0}],
        )
        await processor.process_record(msg1.to_data())

        msg2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=int(1.5 * NANOS_PER_SECOND),
            results=[{"test_record": 84.0}],
        )
        await processor.process_record(msg2.to_data())

        summary = await processor.summarize()

        assert isinstance(summary, MetricsSummary)
        assert summary.timeslices is not None
        assert len(summary.timeslices) == 2
        assert 0 in summary.timeslices
        assert 1 in summary.timeslices
        # Each timeslice should have results
        assert len(summary.timeslices[0]) > 0
        assert len(summary.timeslices[1]) > 0

    @pytest.mark.asyncio
    async def test_summarize_no_timeslices_without_config(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test summarize returns None timeslices when slice_duration is not set."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}
        processor._metric_classes = {"test_record": RequestLatencyMetric()}

        msg = create_metric_records_message(
            x_request_id="test-1",
            results=[{"test_record": 42.0}],
        )
        await processor.process_record(msg.to_data())

        summary = await processor.summarize()
        assert summary.timeslices is None

    @pytest.mark.asyncio
    async def test_timeslice_accumulation(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test that values within same timeslice are accumulated."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}
        processor._metric_classes = {"test_record": RequestLatencyMetric()}

        # Two records in same 1-second window
        msg1 = create_metric_records_message(
            x_request_id="test-1",
            request_start_ns=int(0.3 * NANOS_PER_SECOND),
            results=[{"test_record": 10.0}],
        )
        await processor.process_record(msg1.to_data())

        msg2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{"test_record": 20.0}],
        )
        await processor.process_record(msg2.to_data())

        summary = await processor.summarize()
        assert summary.timeslices is not None
        # Both should be in the same timeslice
        assert len(summary.timeslices) == 1
        assert 0 in summary.timeslices

    @pytest.mark.asyncio
    async def test_timeslice_aggregate_metrics(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test aggregate metrics use vectorized AggregationKind per timeslice."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }
        processor._metric_classes = {RequestCountMetric.tag: RequestCountMetric()}

        # First timeslice: 5 + 3 = 8
        msg1 = create_metric_records_message(
            x_request_id="test-1",
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            results=[{RequestCountMetric.tag: 5}],
        )
        await processor.process_record(msg1.to_data())

        msg2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{RequestCountMetric.tag: 3}],
        )
        await processor.process_record(msg2.to_data())

        # Second timeslice: 7
        msg3 = create_metric_records_message(
            x_request_id="test-3",
            request_start_ns=int(1.5 * NANOS_PER_SECOND),
            results=[{RequestCountMetric.tag: 7}],
        )
        await processor.process_record(msg3.to_data())

        summary = await processor.summarize()
        assert summary.timeslices is not None
        assert len(summary.timeslices) == 2

        # Each timeslice should have aggregated separately via SUM
        ts0_results = summary.timeslices[0]
        ts1_results = summary.timeslices[1]
        assert ts0_results[RequestCountMetric.tag].avg == 8  # 5 + 3
        assert ts1_results[RequestCountMetric.tag].avg == 7

    @pytest.mark.asyncio
    async def test_timeslice_max_aggregate(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test MAX aggregation per timeslice."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {"max_ts": MetricType.AGGREGATE}
        processor._aggregation_kinds = {"max_ts": AggregationKind.MAX}
        processor._metric_classes = {"max_ts": RequestLatencyMetric()}

        msg1 = create_metric_records_message(
            x_request_id="test-1",
            request_start_ns=int(0.3 * NANOS_PER_SECOND),
            results=[{"max_ts": 100}],
        )
        await processor.process_record(msg1.to_data())

        msg2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{"max_ts": 300}],
        )
        await processor.process_record(msg2.to_data())

        summary = await processor.summarize()
        assert summary.timeslices is not None
        ts0_results = summary.timeslices[0]
        assert ts0_results["max_ts"].avg == 300.0  # MAX of 100, 300

    @pytest.mark.asyncio
    async def test_timeslice_min_aggregate(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test MIN aggregation per timeslice."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {"min_ts": MetricType.AGGREGATE}
        processor._aggregation_kinds = {"min_ts": AggregationKind.MIN}
        processor._metric_classes = {"min_ts": RequestLatencyMetric()}

        msg1 = create_metric_records_message(
            x_request_id="test-1",
            request_start_ns=int(0.3 * NANOS_PER_SECOND),
            results=[{"min_ts": 500}],
        )
        await processor.process_record(msg1.to_data())

        msg2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{"min_ts": 200}],
        )
        await processor.process_record(msg2.to_data())

        summary = await processor.summarize()
        assert summary.timeslices is not None
        ts0_results = summary.timeslices[0]
        assert ts0_results["min_ts"].avg == 200.0  # MIN of 500, 200


class TestMetricsSummary:
    def test_to_json(self) -> None:
        summary = MetricsSummary(
            results={
                "test": MetricResult(
                    tag="test", header="Test", unit="ms", avg=42.0, count=1
                )
            }
        )
        json_data = summary.to_json()
        assert "results" in json_data
        assert len(json_data["results"]) == 1

    def test_to_json_with_timeslices(self) -> None:
        summary = MetricsSummary(
            results={},
            timeslices={
                0: {
                    "test": MetricResult(
                        tag="test", header="Test", unit="ms", avg=42.0, count=1
                    )
                }
            },
        )
        json_data = summary.to_json()
        assert "timeslices" in json_data
        assert "0" in json_data["timeslices"]

    def test_to_csv(self) -> None:
        summary = MetricsSummary(
            results={
                "test": MetricResult(
                    tag="test", header="Test", unit="ms", avg=42.0, count=1
                )
            }
        )
        csv_data = summary.to_csv()
        assert len(csv_data) == 1

    def test_to_csv_with_timeslices(self) -> None:
        summary = MetricsSummary(
            results={
                "test": MetricResult(
                    tag="test", header="Test", unit="ms", avg=42.0, count=1
                )
            },
            timeslices={
                0: {
                    "ts_test": MetricResult(
                        tag="ts_test", header="TS Test", unit="ms", avg=10.0, count=1
                    )
                }
            },
        )
        csv_data = summary.to_csv()
        # 1 overall result + 1 timeslice result
        assert len(csv_data) == 2
        assert csv_data[1]["timeslice"] == 0


class TestProtocolConformance:
    def test_satisfies_accumulator_protocol(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        from aiperf.common.accumulator_protocols import AccumulatorProtocol

        processor = MetricsAccumulator(mock_user_config)
        assert isinstance(processor, AccumulatorProtocol)

    def test_summary_satisfies_accumulator_result(self) -> None:
        from aiperf.common.accumulator_protocols import AccumulatorResult

        summary = MetricsSummary(results={})
        assert isinstance(summary, AccumulatorResult)


class TestFullMetrics:
    @pytest.mark.asyncio
    async def test_full_metrics_with_derived(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test full_metrics returns the complete results dict including derived metrics."""

        def mock_derive_func(results_dict: MetricResultsDict) -> float:
            return 200.0

        processor = MetricsAccumulator(mock_user_config)
        processor._derive_funcs = {RequestThroughputMetric.tag: mock_derive_func}
        processor._metric_classes = {}

        full_results = await processor.full_metrics()
        assert RequestThroughputMetric.tag in full_results
        assert full_results[RequestThroughputMetric.tag] == 200.0
