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
from aiperf.common.types import TimesliceWindow
from aiperf.metrics.accumulator import (
    _AGGREGATE_FUNCS,
    MetricsAccumulator,
    MetricsSummary,
)
from aiperf.metrics.column_store import ColumnStore
from aiperf.metrics.metric_dicts import MetricResultsDict, metric_result_from_array
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from tests.unit.post_processors.conftest import (
    create_accumulator_with_metrics,
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
        assert isinstance(processor._column_store, ColumnStore)
        assert isinstance(processor._tags_to_types, dict)
        assert isinstance(processor._aggregation_kinds, dict)

    @pytest.mark.asyncio
    async def test_process_record_record_metric(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processing record metric stores values in column store."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        message = create_metric_records_message(
            x_request_id="test-1",
            session_num=0,
            results=[{"test_record": 42.0}],
        )
        await processor.process_record(message.to_data())

        assert "test_record" in processor._column_store.numeric_tags()
        values = processor._column_store.numeric("test_record")
        assert list(values[~np.isnan(values)]) == [42.0]

        # New data should expand the column store
        message2 = create_metric_records_message(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=1_000_000_001,
            results=[{"test_record": 84.0}],
        )
        await processor.process_record(message2.to_data())
        values = processor._column_store.numeric("test_record")
        assert list(values[~np.isnan(values)]) == [42.0, 84.0]

    @pytest.mark.asyncio
    async def test_process_record_record_metric_list_values(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processing record metric with list values stores in ragged series."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        message = create_metric_records_message(
            x_request_id="test-1",
            session_num=0,
            results=[{"test_record": [10.0, 20.0, 30.0]}],
        )
        await processor.process_record(message.to_data())

        assert "test_record" in processor._column_store.ragged_tags()
        ragged = processor._column_store.ragged("test_record")
        assert list(ragged.values) == [10.0, 20.0, 30.0]

    @pytest.mark.asyncio
    async def test_process_record_aggregate_metric(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processing aggregate metric stores values in column store."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }

        message1 = create_metric_records_message(
            x_request_id="test-1",
            session_num=0,
            results=[{RequestCountMetric.tag: 5}],
        )
        await processor.process_record(message1.to_data())

        assert RequestCountMetric.tag in processor._column_store.numeric_tags()
        values = processor._column_store.numeric(RequestCountMetric.tag)
        assert list(values[~np.isnan(values)]) == [5.0]

        message2 = create_metric_records_message(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=1_000_000_001,
            results=[{RequestCountMetric.tag: 3}],
        )
        await processor.process_record(message2.to_data())
        values = processor._column_store.numeric(RequestCountMetric.tag)
        assert list(values[~np.isnan(values)]) == [5.0, 3.0]

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
                session_num=i,
                request_start_ns=1_000_000_000 + i,
                results=[{RequestCountMetric.tag: 5}],
            )
            await processor.process_record(msg.to_data())

        results = processor._compute_results()
        assert results[RequestCountMetric.tag].avg == 15.0

    @pytest.mark.asyncio
    async def test_record_count(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test record_count derives from column store."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {}

        msg1 = create_metric_records_message(x_request_id="test-1", session_num=0)
        msg2 = create_metric_records_message(
            x_request_id="test-2", session_num=1, request_start_ns=1_000_000_001
        )

        await processor.process_record(msg1.to_data())
        await processor.process_record(msg2.to_data())

        assert processor.record_count == 2


class TestComputeResultsWindowBounds:
    """Test that _compute_results propagates window bounds to derived metrics."""

    @pytest.mark.asyncio
    async def test_window_bounds_set_on_scalar_dict(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Window bounds passed to _compute_results reach the derived-metric scalar dict."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }
        processor._metric_classes = {RequestCountMetric.tag: RequestCountMetric()}

        captured: list[MetricResultsDict] = []

        def spy_derive(results_dict: MetricResultsDict) -> float:
            captured.append(results_dict)
            return 42.0

        processor._derive_funcs = {RequestThroughputMetric.tag: spy_derive}
        processor._metric_classes[RequestThroughputMetric.tag] = (
            RequestThroughputMetric()
        )

        msg = create_metric_records_message(
            x_request_id="test-1",
            session_num=0,
            results=[{RequestCountMetric.tag: 10}],
        )
        await processor.process_record(msg.to_data())

        processor._compute_results(
            window_start_ns=1_000_000_000, window_end_ns=5_000_000_000
        )

        assert len(captured) == 1
        assert captured[0].window_start_ns == 1_000_000_000
        assert captured[0].window_end_ns == 5_000_000_000

    @pytest.mark.asyncio
    async def test_compute_results_for_mask_forwards_window_bounds(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """compute_results_for_mask forwards window bounds to _compute_results."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }
        processor._metric_classes = {RequestCountMetric.tag: RequestCountMetric()}

        captured: list[MetricResultsDict] = []

        def spy_derive(results_dict: MetricResultsDict) -> float:
            captured.append(results_dict)
            return 42.0

        processor._derive_funcs = {RequestThroughputMetric.tag: spy_derive}
        processor._metric_classes[RequestThroughputMetric.tag] = (
            RequestThroughputMetric()
        )

        msg = create_metric_records_message(
            x_request_id="test-1",
            session_num=0,
            results=[{RequestCountMetric.tag: 10}],
        )
        await processor.process_record(msg.to_data())

        mask = np.ones(processor._column_store.count, dtype=bool)
        processor.compute_results_for_mask(
            mask, window_start_ns=2_000_000_000, window_end_ns=8_000_000_000
        )

        assert len(captured) == 1
        assert captured[0].window_start_ns == 2_000_000_000
        assert captured[0].window_end_ns == 8_000_000_000


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
        mask = processor.query_time_range(0, 10_000)
        assert len(mask) == 0

    @pytest.mark.asyncio
    async def test_single_record_inside(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {}
        record = create_metric_records_message(
            x_request_id="test-1", session_num=0, request_start_ns=5_000
        ).to_data()
        await processor.process_record(record)
        mask = processor.query_time_range(0, 10_000)
        assert mask.sum() == 1

    @pytest.mark.asyncio
    async def test_single_record_outside(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {}
        record = create_metric_records_message(
            x_request_id="test-1", session_num=0, request_start_ns=15_000
        ).to_data()
        await processor.process_record(record)
        mask = processor.query_time_range(0, 10_000)
        assert mask.sum() == 0

    @pytest.mark.asyncio
    async def test_boundary_inclusive_start_exclusive_end(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {}
        record1 = create_metric_records_message(
            x_request_id="test-1", session_num=0, request_start_ns=1_000
        ).to_data()
        record2 = create_metric_records_message(
            x_request_id="test-2", session_num=1, request_start_ns=2_000
        ).to_data()
        await processor.process_record(record1)
        await processor.process_record(record2)
        # [1_000, 2_000) should include 1_000 but exclude 2_000
        mask = processor.query_time_range(1_000, 2_000)
        assert mask.sum() == 1
        assert mask[0] is np.True_
        assert mask[1] is np.False_

    @pytest.mark.asyncio
    async def test_multiple_records_filtering(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {}
        for i, ts in enumerate([100, 200, 300, 400, 500]):
            r = create_metric_records_message(
                x_request_id=f"test-{i}", session_num=i, request_start_ns=ts
            ).to_data()
            await processor.process_record(r)

        mask = processor.query_time_range(200, 400)
        assert mask.sum() == 2
        np.testing.assert_array_equal(np.where(mask)[0], [1, 2])


class TestSummarize:
    @pytest.mark.asyncio
    async def test_summarize_returns_metrics_summary(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test summarize returns MetricsSummary wrapping MetricResult objects."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {RequestLatencyMetric.tag: MetricType.RECORD}
        processor._metric_classes = {RequestLatencyMetric.tag: RequestLatencyMetric()}

        # Inject data via process_record
        msg = create_metric_records_message(
            x_request_id="test-1",
            session_num=0,
            results=[{RequestLatencyMetric.tag: 42.0}],
        )
        await processor.process_record(msg.to_data())

        summary = await processor.summarize()

        assert isinstance(summary, MetricsSummary)
        assert RequestLatencyMetric.tag in summary.results
        # Also includes effective_concurrency + effective_throughput from sweep injection
        assert len(summary.results) >= 1
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
            session_num=0,
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            results=[{"test_record": 42.0}],
        )
        await processor.process_record(msg1.to_data())

        msg2 = create_metric_records_message(
            x_request_id="test-2",
            session_num=1,
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
            session_num=0,
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
            session_num=0,
            request_start_ns=int(0.3 * NANOS_PER_SECOND),
            results=[{"test_record": 10.0}],
        )
        await processor.process_record(msg1.to_data())

        msg2 = create_metric_records_message(
            x_request_id="test-2",
            session_num=1,
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
            session_num=0,
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            results=[{RequestCountMetric.tag: 5}],
        )
        await processor.process_record(msg1.to_data())

        msg2 = create_metric_records_message(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{RequestCountMetric.tag: 3}],
        )
        await processor.process_record(msg2.to_data())

        # Second timeslice: 7
        msg3 = create_metric_records_message(
            x_request_id="test-3",
            session_num=2,
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
            session_num=0,
            request_start_ns=int(0.3 * NANOS_PER_SECOND),
            results=[{"max_ts": 100}],
        )
        await processor.process_record(msg1.to_data())

        msg2 = create_metric_records_message(
            x_request_id="test-2",
            session_num=1,
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
            session_num=0,
            request_start_ns=int(0.3 * NANOS_PER_SECOND),
            results=[{"min_ts": 500}],
        )
        await processor.process_record(msg1.to_data())

        msg2 = create_metric_records_message(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{"min_ts": 200}],
        )
        await processor.process_record(msg2.to_data())

        summary = await processor.summarize()
        assert summary.timeslices is not None
        ts0_results = summary.timeslices[0]
        assert ts0_results["min_ts"].avg == 200.0  # MIN of 500, 200

    @pytest.mark.asyncio
    async def test_compute_timeslices_populates_timeslice_windows(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test _compute_timeslices returns timeslice_windows with correct boundaries."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}
        processor._metric_classes = {"test_record": RequestLatencyMetric()}

        msg1 = create_metric_records_message(
            x_request_id="test-1",
            session_num=0,
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            results=[{"test_record": 42.0}],
        )
        await processor.process_record(msg1.to_data())

        msg2 = create_metric_records_message(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=int(1.5 * NANOS_PER_SECOND),
            results=[{"test_record": 84.0}],
        )
        await processor.process_record(msg2.to_data())

        summary = await processor.summarize()

        assert summary.timeslice_windows is not None
        assert len(summary.timeslice_windows) == 2
        assert 0 in summary.timeslice_windows
        assert 1 in summary.timeslice_windows

        w0 = summary.timeslice_windows[0]
        w1 = summary.timeslice_windows[1]
        assert isinstance(w0, TimesliceWindow)
        assert isinstance(w1, TimesliceWindow)

        # Windows should be consecutive 1-second bins
        assert w0.end_ns == w1.start_ns
        assert w1.end_ns - w1.start_ns == NANOS_PER_SECOND
        # is_complete defaults to None (complete)
        assert w0.is_complete is None
        assert w1.is_complete is None

    @pytest.mark.asyncio
    async def test_timeslice_windows_none_without_config(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test timeslice_windows is None when slice_duration is not set."""
        processor = MetricsAccumulator(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}
        processor._metric_classes = {"test_record": RequestLatencyMetric()}

        msg = create_metric_records_message(
            x_request_id="test-1",
            session_num=0,
            results=[{"test_record": 42.0}],
        )
        await processor.process_record(msg.to_data())

        summary = await processor.summarize()
        assert summary.timeslice_windows is None


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
        processor._metric_classes = {
            RequestThroughputMetric.tag: RequestThroughputMetric()
        }

        full_results = await processor.full_metrics()
        assert RequestThroughputMetric.tag in full_results
        assert isinstance(full_results[RequestThroughputMetric.tag], MetricResult)
        assert full_results[RequestThroughputMetric.tag].avg == 200.0


class TestMetricResultFromArray:
    """Test metric_result_from_array computes correct statistics."""

    def test_single_value(self) -> None:
        """Single-element array: all stats equal the value."""
        arr = np.array([5.0], dtype=np.float64)
        r = metric_result_from_array("test", "Test", "ms", arr, 5.0)
        assert r.tag == "test"
        assert r.header == "Test"
        assert r.unit == "ms"
        assert r.count == 1
        assert r.min == 5.0
        assert r.max == 5.0
        assert r.avg == 5.0
        assert r.std == 0.0
        assert r.p50 == 5.0

    def test_five_values(self) -> None:
        """Five evenly-spaced values: known min/max/avg/p50."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        r = metric_result_from_array("t", "T", "u", arr, 15.0)
        assert r.count == 5
        assert r.min == 1.0
        assert r.max == 5.0
        assert r.avg == 3.0
        assert r.p50 == 3.0
        np.testing.assert_allclose(r.std, np.std([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_hundred_values(self) -> None:
        """1..100: verify percentile interpolation on a larger dataset."""
        values = list(range(1, 101))
        arr = np.array(values, dtype=np.float64)
        r = metric_result_from_array("t", "T", "u", arr, float(sum(values)))
        assert r.count == 100
        assert r.min == 1.0
        assert r.max == 100.0
        assert r.avg == 50.5
        assert r.p50 == 50.5
        np.testing.assert_allclose(r.p1, 1.99)
        np.testing.assert_allclose(r.p99, 99.01)

    def test_sorts_in_place(self) -> None:
        """Verify the function sorts the input array in-place."""
        arr = np.array([5.0, 1.0, 3.0], dtype=np.float64)
        metric_result_from_array("t", "T", "u", arr, 9.0)
        np.testing.assert_array_equal(arr, [1.0, 3.0, 5.0])


# ---------------------------------------------------------------------------
# Helpers for timeslice sweep metric tests
# ---------------------------------------------------------------------------


def _make_sweep_metric_classes():
    """Create minimal metric classes needed for sweep-based timeslice tests."""
    from aiperf.common.enums import MetricType

    class FakeLatency:
        tag = "request_latency"
        type = MetricType.RECORD
        header = "Request Latency"
        unit = "ms"

    class FakeOutputTokens:
        tag = "output_tokens"
        type = MetricType.RECORD
        header = "Output Tokens"
        unit = "tokens"

    class FakeTTFT:
        tag = "time_to_first_token"
        type = MetricType.RECORD
        header = "Time To First Token"
        unit = "ns"

    class FakeISL:
        tag = "input_sequence_length"
        type = MetricType.RECORD
        header = "Input Sequence Length"
        unit = "tokens"

    return FakeLatency, FakeOutputTokens, FakeTTFT, FakeISL


class TestTimesliceSweepMetrics:
    """Tests for sweep-based effective_concurrency and effective_throughput in timeslices."""

    @pytest.mark.asyncio
    async def test_timeslice_has_effective_concurrency_and_throughput(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """All sweep metrics are present in every timeslice with correct tag/unit."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_user_config, latency_cls, output_cls, ttft_cls
        )

        # One request: 0.5s start, 0.8s end, 10 output tokens, 50ms TTFT
        msg = create_metric_records_message(
            session_num=0,
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            request_end_ns=int(0.8 * NANOS_PER_SECOND),
            results=[
                {
                    "request_latency": 300_000_000.0,
                    "output_tokens": 10.0,
                    "time_to_first_token": 50_000_000.0,
                }
            ],
        )
        await acc.process_record(msg.to_data())

        summary = await acc.summarize()
        assert summary.timeslices is not None
        for ts_results in summary.timeslices.values():
            assert "effective_concurrency" in ts_results
            assert "effective_throughput" in ts_results
            assert "effective_prefill_throughput" in ts_results
            ec = ts_results["effective_concurrency"]
            et = ts_results["effective_throughput"]
            ept = ts_results["effective_prefill_throughput"]
            assert ec.tag == "effective_concurrency"
            assert ec.unit == "requests"
            assert et.tag == "effective_throughput"
            assert et.unit == "tokens/sec"
            assert ept.tag == "effective_prefill_throughput"
            assert ept.unit == "tokens/sec"

    @pytest.mark.asyncio
    async def test_timeslice_effective_concurrency_overlapping_requests(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Overlapping requests in a timeslice produce avg concurrency > 1."""
        mock_user_config.output = OutputConfig(slice_duration=2.0)
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_user_config, latency_cls, output_cls, ttft_cls
        )

        # Two overlapping requests within the same 2s timeslice
        # Request A: [0.1s, 1.5s)  Request B: [0.5s, 1.8s)
        for i, (start, end) in enumerate(
            [(0.1, 1.5), (0.5, 1.8)],
        ):
            msg = create_metric_records_message(
                session_num=i,
                request_start_ns=int(start * NANOS_PER_SECOND),
                request_end_ns=int(end * NANOS_PER_SECOND),
                results=[
                    {
                        "request_latency": (end - start) * NANOS_PER_SECOND,
                        "output_tokens": 5.0,
                        "time_to_first_token": 10_000_000.0,
                    }
                ],
            )
            await acc.process_record(msg.to_data())

        summary = await acc.summarize()
        assert summary.timeslices is not None
        ts0 = summary.timeslices[0]
        assert ts0["effective_concurrency"].avg > 1.0

    @pytest.mark.asyncio
    async def test_timeslice_effective_throughput_nonzero(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Records with output_tokens and TTFT produce nonzero throughput."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_user_config, latency_cls, output_cls, ttft_cls
        )

        msg = create_metric_records_message(
            session_num=0,
            request_start_ns=int(0.1 * NANOS_PER_SECOND),
            request_end_ns=int(0.9 * NANOS_PER_SECOND),
            results=[
                {
                    "request_latency": 800_000_000.0,
                    "output_tokens": 100.0,
                    "time_to_first_token": 50_000_000.0,
                }
            ],
        )
        await acc.process_record(msg.to_data())

        summary = await acc.summarize()
        assert summary.timeslices is not None
        ts0 = summary.timeslices[0]
        assert ts0["effective_throughput"].avg > 0.0

    @pytest.mark.asyncio
    async def test_timeslice_sweep_metrics_zero_throughput_without_tokens(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Without output_tokens, throughput avg is 0 but concurrency is nonzero."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        latency_cls, _, _, _ = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(mock_user_config, latency_cls)

        msg = create_metric_records_message(
            session_num=0,
            request_start_ns=int(0.2 * NANOS_PER_SECOND),
            request_end_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{"request_latency": 500_000_000.0}],
        )
        await acc.process_record(msg.to_data())

        summary = await acc.summarize()
        assert summary.timeslices is not None
        ts0 = summary.timeslices[0]
        assert ts0["effective_throughput"].avg == 0.0
        assert ts0["effective_concurrency"].avg > 0.0

    @pytest.mark.asyncio
    async def test_timeslice_sweep_metrics_multiple_slices(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Records across 3 slices each have distinct sweep metric values."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_user_config, latency_cls, output_cls, ttft_cls
        )

        # 3 non-overlapping requests, one per 1s slice
        records = [
            (0, 0.1, 0.9, 800e6, 10.0, 50e6),
            (1, 1.1, 1.9, 800e6, 20.0, 50e6),
            (2, 2.1, 2.9, 800e6, 30.0, 50e6),
        ]
        for session_num, start, end, latency, tokens, ttft in records:
            msg = create_metric_records_message(
                session_num=session_num,
                request_start_ns=int(start * NANOS_PER_SECOND),
                request_end_ns=int(end * NANOS_PER_SECOND),
                results=[
                    {
                        "request_latency": latency,
                        "output_tokens": tokens,
                        "time_to_first_token": ttft,
                    }
                ],
            )
            await acc.process_record(msg.to_data())

        summary = await acc.summarize()
        assert summary.timeslices is not None
        assert len(summary.timeslices) == 3

        # Each slice should have its own sweep metrics
        for ts_idx in range(3):
            ts = summary.timeslices[ts_idx]
            assert "effective_concurrency" in ts
            assert "effective_throughput" in ts
            assert ts["effective_concurrency"].avg > 0.0
            assert ts["effective_throughput"].avg > 0.0

        # Throughput should scale with token count (more tokens → higher throughput)
        # Since request durations are identical, throughput is proportional to tokens
        t0 = summary.timeslices[0]["effective_throughput"].avg
        t1 = summary.timeslices[1]["effective_throughput"].avg
        t2 = summary.timeslices[2]["effective_throughput"].avg
        assert t1 > t0
        assert t2 > t1


class TestOverallSweepMetrics:
    """Tests for sweep-based effective_concurrency and effective_throughput in overall results."""

    @pytest.mark.asyncio
    async def test_overall_has_effective_concurrency_and_throughput(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """All sweep metrics are present in the overall results with correct tag/unit."""
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_user_config, latency_cls, output_cls, ttft_cls
        )

        msg = create_metric_records_message(
            session_num=0,
            request_start_ns=int(0.1 * NANOS_PER_SECOND),
            request_end_ns=int(0.9 * NANOS_PER_SECOND),
            results=[
                {
                    "request_latency": 800_000_000.0,
                    "output_tokens": 50.0,
                    "time_to_first_token": 50_000_000.0,
                }
            ],
        )
        await acc.process_record(msg.to_data())

        summary = await acc.summarize()
        assert "effective_concurrency" in summary.results
        assert "effective_throughput" in summary.results
        assert "effective_prefill_throughput" in summary.results
        ec = summary.results["effective_concurrency"]
        et = summary.results["effective_throughput"]
        ept = summary.results["effective_prefill_throughput"]
        assert ec.tag == "effective_concurrency"
        assert ec.unit == "requests"
        assert et.tag == "effective_throughput"
        assert et.unit == "tokens/sec"
        assert ept.tag == "effective_prefill_throughput"
        assert ept.unit == "tokens/sec"

    @pytest.mark.asyncio
    async def test_overall_effective_concurrency_overlapping_requests(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Overlapping requests produce avg concurrency > 1 in overall results."""
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_user_config, latency_cls, output_cls, ttft_cls
        )

        for i, (start, end) in enumerate([(0.1, 1.5), (0.5, 1.8)]):
            msg = create_metric_records_message(
                session_num=i,
                request_start_ns=int(start * NANOS_PER_SECOND),
                request_end_ns=int(end * NANOS_PER_SECOND),
                results=[
                    {
                        "request_latency": (end - start) * NANOS_PER_SECOND,
                        "output_tokens": 5.0,
                        "time_to_first_token": 10_000_000.0,
                    }
                ],
            )
            await acc.process_record(msg.to_data())

        summary = await acc.summarize()
        assert summary.results["effective_concurrency"].avg > 1.0

    @pytest.mark.asyncio
    async def test_overall_effective_throughput_nonzero(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Records with output_tokens and TTFT produce nonzero overall throughput."""
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_user_config, latency_cls, output_cls, ttft_cls
        )

        msg = create_metric_records_message(
            session_num=0,
            request_start_ns=int(0.1 * NANOS_PER_SECOND),
            request_end_ns=int(0.9 * NANOS_PER_SECOND),
            results=[
                {
                    "request_latency": 800_000_000.0,
                    "output_tokens": 100.0,
                    "time_to_first_token": 50_000_000.0,
                }
            ],
        )
        await acc.process_record(msg.to_data())

        summary = await acc.summarize()
        assert summary.results["effective_throughput"].avg > 0.0

    @pytest.mark.asyncio
    async def test_overall_zero_throughput_without_tokens(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Without output_tokens, throughput avg is 0 but concurrency is nonzero."""
        latency_cls, _, _, _ = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(mock_user_config, latency_cls)

        msg = create_metric_records_message(
            session_num=0,
            request_start_ns=int(0.2 * NANOS_PER_SECOND),
            request_end_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{"request_latency": 500_000_000.0}],
        )
        await acc.process_record(msg.to_data())

        summary = await acc.summarize()
        assert summary.results["effective_throughput"].avg == 0.0
        assert summary.results["effective_concurrency"].avg > 0.0

    @pytest.mark.asyncio
    async def test_overall_sweep_metrics_not_present_when_empty(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """No sweep metrics when no records have been ingested."""
        latency_cls, _, _, _ = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(mock_user_config, latency_cls)

        summary = await acc.summarize()
        assert "effective_concurrency" not in summary.results
        assert "effective_throughput" not in summary.results
        assert "effective_prefill_throughput" not in summary.results

    @pytest.mark.asyncio
    async def test_overall_effective_prefill_throughput_nonzero(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Records with ISL and TTFT produce nonzero prefill throughput."""
        latency_cls, output_cls, ttft_cls, isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_user_config, latency_cls, output_cls, ttft_cls, isl_cls
        )

        msg = create_metric_records_message(
            session_num=0,
            request_start_ns=int(0.1 * NANOS_PER_SECOND),
            request_end_ns=int(0.9 * NANOS_PER_SECOND),
            results=[
                {
                    "request_latency": 800_000_000.0,
                    "output_tokens": 100.0,
                    "time_to_first_token": 50_000_000.0,
                    "input_sequence_length": 200.0,
                }
            ],
        )
        await acc.process_record(msg.to_data())

        summary = await acc.summarize()
        assert summary.results["effective_prefill_throughput"].avg > 0.0

    @pytest.mark.asyncio
    async def test_overall_zero_prefill_throughput_without_isl(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Without input_sequence_length metric, prefill throughput avg is 0."""
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_user_config, latency_cls, output_cls, ttft_cls
        )

        msg = create_metric_records_message(
            session_num=0,
            request_start_ns=int(0.2 * NANOS_PER_SECOND),
            request_end_ns=int(0.7 * NANOS_PER_SECOND),
            results=[
                {
                    "request_latency": 500_000_000.0,
                    "output_tokens": 50.0,
                    "time_to_first_token": 50_000_000.0,
                }
            ],
        )
        await acc.process_record(msg.to_data())

        summary = await acc.summarize()
        assert summary.results["effective_prefill_throughput"].avg == 0.0
