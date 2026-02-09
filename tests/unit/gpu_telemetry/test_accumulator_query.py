# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPUTelemetryAccumulator.process_record() and query_time_range()."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from aiperf.common.accumulator_protocols import AccumulatorProtocol
from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.gpu_telemetry.accumulator import GPUTelemetryAccumulator
from aiperf.plugin.enums import EndpointType
from tests.unit.post_processors.conftest import make_telemetry_record


@pytest.fixture
def accumulator(mock_metric_registry) -> GPUTelemetryAccumulator:
    user_config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            streaming=False,
        )
    )
    mock_pub = Mock()
    mock_pub.publish = AsyncMock()
    return GPUTelemetryAccumulator(
        user_config=user_config,
        service_config=ServiceConfig(),
        pub_client=mock_pub,
    )


class TestGPUTelemetryAccumulatorProtocol:
    def test_satisfies_accumulator_protocol(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        assert isinstance(accumulator, AccumulatorProtocol)


class TestProcessRecord:
    @pytest.mark.asyncio
    async def test_process_record_stores_and_adds_to_hierarchy(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        record = make_telemetry_record(timestamp_ns=1_000)
        await accumulator.process_record(record)

        assert len(accumulator._raw_records) == 1
        assert accumulator._raw_timestamps_ns == [1_000]
        assert len(accumulator._hierarchy.dcgm_endpoints) > 0

    @pytest.mark.asyncio
    async def test_process_telemetry_record_delegates_to_process_record(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        record = make_telemetry_record(timestamp_ns=2_000)
        await accumulator.process_telemetry_record(record)

        assert len(accumulator._raw_records) == 1
        assert accumulator._raw_timestamps_ns == [2_000]


class TestQueryTimeRange:
    @pytest.mark.asyncio
    async def test_empty(self, accumulator: GPUTelemetryAccumulator) -> None:
        assert accumulator.query_time_range(0, 10_000) == []

    @pytest.mark.asyncio
    async def test_single_record_inside(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        record = make_telemetry_record(timestamp_ns=5_000)
        await accumulator.process_record(record)
        assert accumulator.query_time_range(0, 10_000) == [record]

    @pytest.mark.asyncio
    async def test_single_record_outside(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        await accumulator.process_record(make_telemetry_record(timestamp_ns=15_000))
        assert accumulator.query_time_range(0, 10_000) == []

    @pytest.mark.asyncio
    async def test_boundary_inclusive_start(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        record = make_telemetry_record(timestamp_ns=1_000)
        await accumulator.process_record(record)
        assert accumulator.query_time_range(1_000, 2_000) == [record]

    @pytest.mark.asyncio
    async def test_boundary_exclusive_end(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        await accumulator.process_record(make_telemetry_record(timestamp_ns=2_000))
        assert accumulator.query_time_range(1_000, 2_000) == []

    @pytest.mark.asyncio
    async def test_multiple_records_filtering(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        timestamps = [100, 200, 300, 400, 500]
        records = [make_telemetry_record(timestamp_ns=ts) for ts in timestamps]
        for r in records:
            await accumulator.process_record(r)

        result = accumulator.query_time_range(200, 400)
        assert result == [records[1], records[2]]

    @pytest.mark.asyncio
    async def test_equal_start_end_returns_empty(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        await accumulator.process_record(make_telemetry_record(timestamp_ns=100))
        assert accumulator.query_time_range(100, 100) == []
