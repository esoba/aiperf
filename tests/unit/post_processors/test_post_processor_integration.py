# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration unit tests for post-processing pipeline."""

from unittest.mock import Mock

import pytest

from aiperf.common.config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import AggregationKind, MetricType
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from aiperf.post_processors.metric_record_processor import MetricRecordProcessor
from aiperf.post_processors.metrics_accumulator import MetricsAccumulator
from tests.unit.post_processors.conftest import (
    create_metric_records_message,
    create_results_processor_with_metrics,
    setup_mock_registry_sequences,
)

TEST_LATENCY_VALUES = [100.0, 150.0, 200.0]
TEST_REQUEST_COUNT = 100
TEST_DURATION_SECONDS = 10
EXPECTED_THROUGHPUT = TEST_REQUEST_COUNT / TEST_DURATION_SECONDS


@pytest.mark.asyncio
class TestPostProcessorIntegration:
    """Integration tests focusing on key processor handoffs."""

    async def test_record_to_results_data_flow(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
    ) -> None:
        """Test data flows correctly from record processor to results processor."""
        results_processor = create_results_processor_with_metrics(
            mock_user_config, RequestLatencyMetric, RequestCountMetric
        )
        message = create_metric_records_message(
            x_request_id="test-1",
            results=[{RequestLatencyMetric.tag: 100.0, RequestCountMetric.tag: 1}],
        )

        await results_processor.process_record(message.to_data())

        assert RequestLatencyMetric.tag in results_processor._time_series
        assert list(
            results_processor._time_series[RequestLatencyMetric.tag].values
        ) == [100.0]

        # Aggregate metrics also stored in time series
        assert RequestCountMetric.tag in results_processor._time_series
        assert list(results_processor._time_series[RequestCountMetric.tag].values) == [
            1.0
        ]

    async def test_multiple_batches_accumulation(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
    ) -> None:
        """Test accumulation across multiple record batches."""
        results_processor = create_results_processor_with_metrics(
            mock_user_config, RequestLatencyMetric
        )

        for idx, value in enumerate(TEST_LATENCY_VALUES):
            message = create_metric_records_message(
                x_request_id=f"test-{idx}",
                request_start_ns=1_000_000_000 + idx,
                x_correlation_id=f"test-correlation-{idx}",
                results=[{RequestLatencyMetric.tag: value}],
            )
            await results_processor.process_record(message.to_data())

        assert RequestLatencyMetric.tag in results_processor._time_series
        accumulated_data = list(
            results_processor._time_series[RequestLatencyMetric.tag].values
        )
        assert accumulated_data == TEST_LATENCY_VALUES

    async def test_error_metrics_isolation(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
        error_parsed_record: ParsedResponseRecord,
    ) -> None:
        """Test that error and valid metrics are processed separately."""
        setup_mock_registry_sequences(
            mock_metric_registry, [], [ErrorRequestCountMetric]
        )

        record_processor = MetricRecordProcessor(mock_user_config)

        assert len(record_processor.error_parse_funcs) == 1
        assert len(record_processor.valid_parse_funcs) == 0

        from tests.unit.post_processors.conftest import create_metric_metadata

        metadata = create_metric_metadata()
        result = await record_processor.process_record(error_parsed_record, metadata)
        assert ErrorRequestCountMetric.tag in result
        assert result[ErrorRequestCountMetric.tag] == 1

    async def test_derived_metrics_computation(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
    ) -> None:
        """Test derived metrics are computed from accumulated results."""
        setup_mock_registry_sequences(
            mock_metric_registry, [RequestThroughputMetric], []
        )

        results_processor = MetricsAccumulator(mock_user_config)

        # Set up RequestCount as aggregate with values in time series
        from aiperf.post_processors.inference_time_series import InferenceTimeSeries

        results_processor._tags_to_types = {
            RequestCountMetric.tag: MetricType.AGGREGATE,
        }
        results_processor._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }
        ts = InferenceTimeSeries()
        ts.append(0, float(TEST_REQUEST_COUNT))
        results_processor._time_series[RequestCountMetric.tag] = ts

        # BenchmarkDuration is a DERIVED metric; add it as a derive func that
        # returns a constant, then let RequestThroughput compute from both.
        benchmark_ns = TEST_DURATION_SECONDS * NANOS_PER_SECOND

        def benchmark_duration_derive(results_dict: MetricResultsDict) -> int:
            return benchmark_ns

        results_processor._derive_funcs = {
            BenchmarkDurationMetric.tag: benchmark_duration_derive,
            **results_processor._derive_funcs,
        }

        full_results = await results_processor.full_metrics()

        assert RequestThroughputMetric.tag in full_results
        assert full_results[RequestThroughputMetric.tag] == EXPECTED_THROUGHPUT

    async def test_complete_pipeline_summary(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
    ) -> None:
        """Test complete pipeline produces proper summary results."""
        results_processor = create_results_processor_with_metrics(
            mock_user_config, RequestLatencyMetric
        )

        # Add data via the time series
        from aiperf.post_processors.inference_time_series import InferenceTimeSeries

        ts = InferenceTimeSeries()
        for i, v in enumerate(TEST_LATENCY_VALUES):
            ts.append(i, v)
        results_processor._time_series[RequestLatencyMetric.tag] = ts

        summary = await results_processor.summarize()

        assert hasattr(summary, "results")
        assert isinstance(summary.results, list)
        assert all(hasattr(result, "tag") for result in summary.results)
        assert all(hasattr(result, "avg") for result in summary.results)
        assert all(hasattr(result, "count") for result in summary.results)
