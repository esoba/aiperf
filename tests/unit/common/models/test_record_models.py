# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel, Field, SerializeAsAny
from pytest import param

from aiperf.common.enums import SSEFieldType
from aiperf.common.models import MetricResult, ProfileResults, SSEMessage
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.common.models.record_models import InEngineResponse


class TestProfileResults:
    """Test cases for ProfileResults model."""

    def test_profile_results_with_timeslice_metric_results(self):
        """Test ProfileResults can store timeslice metric results."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        timeslice_results = {
            0: [metric_result],
            1: [metric_result],
        }

        profile_results = ProfileResults(
            records=[metric_result],
            timeslice_metric_results=timeslice_results,
            completed=1,
            start_ns=1000000000,
            end_ns=2000000000,
        )

        assert profile_results.timeslice_metric_results is not None
        assert 0 in profile_results.timeslice_metric_results
        assert 1 in profile_results.timeslice_metric_results
        assert len(profile_results.timeslice_metric_results[0]) == 1
        assert len(profile_results.timeslice_metric_results[1]) == 1

    def test_profile_results_without_timeslice_metric_results(self):
        """Test ProfileResults works without timeslice metric results."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        profile_results = ProfileResults(
            records=[metric_result],
            completed=1,
            start_ns=1000000000,
            end_ns=2000000000,
        )

        assert profile_results.timeslice_metric_results is None

    def test_profile_results_with_empty_timeslice_dict(self):
        """Test ProfileResults with empty timeslice results dict."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        profile_results = ProfileResults(
            records=[metric_result],
            timeslice_metric_results={},
            completed=1,
            start_ns=1000000000,
            end_ns=2000000000,
        )

        assert profile_results.timeslice_metric_results is not None
        assert len(profile_results.timeslice_metric_results) == 0

    def test_profile_results_with_multiple_timeslices_and_metrics(self):
        """Test ProfileResults with multiple timeslices containing multiple metrics."""
        latency_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        throughput_result = MetricResult(
            tag="request_throughput",
            header="Request Throughput",
            unit="requests/sec",
            avg=50.0,
            count=1,
        )

        timeslice_results = {
            0: [latency_result, throughput_result],
            1: [latency_result, throughput_result],
            2: [latency_result, throughput_result],
        }

        profile_results = ProfileResults(
            records=[latency_result, throughput_result],
            timeslice_metric_results=timeslice_results,
            completed=2,
            start_ns=1000000000,
            end_ns=3000000000,
        )

        assert profile_results.timeslice_metric_results is not None
        assert len(profile_results.timeslice_metric_results) == 3
        for i in range(3):
            assert i in profile_results.timeslice_metric_results
            assert len(profile_results.timeslice_metric_results[i]) == 2


class TestSSEMessageDataclass:
    """Test that SSEMessage dataclass works correctly."""

    def test_parse_produces_valid_message(self) -> None:
        """parse() produces a fully usable SSEMessage."""
        msg = SSEMessage.parse("data: hello\nevent: message", perf_ns=42)
        assert msg.perf_ns == 42
        assert len(msg.packets) == 2
        assert msg.packets[0].name == SSEFieldType.DATA
        assert msg.packets[0].value == "hello"
        assert msg.packets[1].name == SSEFieldType.EVENT
        assert msg.packets[1].value == "message"

    def test_parse_returns_sse_message_instance(self) -> None:
        """parse() returns an SSEMessage instance."""
        msg = SSEMessage.parse("data: test", perf_ns=1)
        assert isinstance(msg, SSEMessage)

    def test_parse_empty_produces_no_packets(self) -> None:
        """Empty input yields zero packets."""
        msg = SSEMessage.parse("", perf_ns=0)
        assert msg.packets == []

    def test_parse_bytes_input(self) -> None:
        """parse() handles bytes input."""
        msg = SSEMessage.parse(b"data: from_bytes", perf_ns=99)
        assert msg.packets[0].value == "from_bytes"

    def test_pydantic_serialization_roundtrip(self) -> None:
        """SSEMessage roundtrips through Pydantic when inside a model field."""

        class Wrapper(BaseModel):
            responses: SerializeAsAny[list[SSEMessage]] = Field(default_factory=list)

        msg = SSEMessage.parse("data: roundtrip\nevent: test", perf_ns=123)
        wrapper = Wrapper(responses=[msg])
        json_bytes = wrapper.model_dump_json().encode()
        restored = Wrapper.model_validate_json(json_bytes)
        assert restored.responses[0].perf_ns == 123
        assert len(restored.responses[0].packets) == 2
        assert restored.responses[0].packets[0].value == "roundtrip"

    def test_parse_get_text_and_get_json(self) -> None:
        """Protocol methods work on dataclass instances."""
        msg = SSEMessage.parse('data: {"key": "value"}', perf_ns=1)
        assert msg.get_text() == '{"key": "value"}'
        json_obj = msg.get_json()
        assert json_obj == {"key": "value"}


class TestMetricResultSumField:
    """Test the sum field on MetricResult."""

    def test_sum_field_stored(self) -> None:
        result = MetricResult(
            tag="total_tokens",
            header="Total Tokens",
            unit="tokens",
            avg=100.0,
            sum=5000.0,
            count=50,
        )
        assert result.sum == 5000.0

    def test_sum_field_defaults_to_none(self) -> None:
        result = MetricResult(tag="latency", header="Latency", unit="ms", avg=10.0)
        assert result.sum is None

    @pytest.mark.parametrize(
        "sum_value",
        [0, 42, 1_000_000, 3.14, -1.0],
        ids=["zero", "int", "large", "float", "negative"],
    )
    def test_sum_accepts_numeric_types(self, sum_value) -> None:
        result = MetricResult(tag="metric", header="M", unit="u", sum=sum_value)
        assert result.sum == sum_value


class TestMetricResultToJsonResult:
    """Test that to_json_result excludes sum."""

    def test_sum_excluded_from_json_result(self) -> None:
        result = MetricResult(
            tag="throughput",
            header="Throughput",
            unit="req/s",
            avg=50.0,
            min=10.0,
            max=90.0,
            sum=5000.0,
            count=100,
        )
        json_result = result.to_json_result()

        assert isinstance(json_result, JsonMetricResult)
        assert not hasattr(json_result, "sum")
        assert json_result.avg == 50.0
        assert json_result.min == 10.0
        assert json_result.max == 90.0

    def test_stat_keys_preserved_in_json_result(self) -> None:
        result = MetricResult(
            tag="latency",
            header="Latency",
            unit="ms",
            avg=100.0,
            p50=95.0,
            p99=200.0,
            min=10.0,
            max=300.0,
            std=25.0,
        )
        json_result = result.to_json_result()

        assert json_result.avg == 100.0
        assert json_result.p50 == 95.0
        assert json_result.p99 == 200.0
        assert json_result.min == 10.0
        assert json_result.max == 300.0
        assert json_result.std == 25.0


# ============================================================
# InEngineResponse — Speculative Decoding Fields
# ============================================================


class TestInEngineResponseSpecDecodeFields:
    """Verify InEngineResponse dataclass with decode_iterations/max_draft_len."""

    def test_construct_with_spec_decode_fields(self) -> None:
        response = InEngineResponse(
            perf_ns=1000,
            text="Hello",
            input_tokens=10,
            output_tokens=5,
            decode_iterations=3,
            max_draft_len=5,
        )
        assert response.decode_iterations == 3
        assert response.max_draft_len == 5

    def test_defaults_to_none(self) -> None:
        response = InEngineResponse(
            perf_ns=1000,
            text="Hello",
            input_tokens=10,
            output_tokens=5,
        )
        assert response.decode_iterations is None
        assert response.max_draft_len is None

    def test_serialization_round_trip_with_spec_decode(self) -> None:
        """Verify spec decode fields survive Pydantic serialization when nested in a model."""
        from dataclasses import asdict

        original = InEngineResponse(
            perf_ns=1000,
            text="Hello",
            input_tokens=10,
            output_tokens=5,
            decode_iterations=7,
            max_draft_len=3,
            output_token_ids=[1, 2, 3, 4, 5],
        )

        data = asdict(original)
        restored = InEngineResponse(**data)

        assert restored.decode_iterations == 7
        assert restored.max_draft_len == 3
        assert restored.output_token_ids == [1, 2, 3, 4, 5]
        assert restored.text == "Hello"
        assert restored.input_tokens == 10
        assert restored.output_tokens == 5

    def test_serialization_round_trip_without_spec_decode(self) -> None:
        """Verify None fields roundtrip correctly."""
        from dataclasses import asdict

        original = InEngineResponse(
            perf_ns=1000,
            text="Hello",
            input_tokens=10,
            output_tokens=5,
        )

        data = asdict(original)
        restored = InEngineResponse(**data)

        assert restored.decode_iterations is None
        assert restored.max_draft_len is None
        assert restored.output_token_ids is None

    @pytest.mark.parametrize(
        "decode_iterations,max_draft_len",
        [
            (0, 0),
            (0, 5),
            (100, 10),
            param(0, None, id="zero-iters-no-draft"),
            param(None, 5, id="no-iters-with-draft"),
        ],
    )  # fmt: skip
    def test_boundary_values(
        self, decode_iterations: int | None, max_draft_len: int | None
    ) -> None:
        response = InEngineResponse(
            perf_ns=1000,
            text="text",
            input_tokens=10,
            output_tokens=5,
            decode_iterations=decode_iterations,
            max_draft_len=max_draft_len,
        )
        assert response.decode_iterations == decode_iterations
        assert response.max_draft_len == max_draft_len

    def test_get_json_returns_none(self) -> None:
        """InEngineResponse.get_json always returns None (no JSON round-trip)."""
        response = InEngineResponse(
            perf_ns=1000,
            text="Hello",
            input_tokens=10,
            output_tokens=5,
        )
        assert response.get_json() is None

    def test_get_text_returns_text(self) -> None:
        response = InEngineResponse(
            perf_ns=1000,
            text="Generated text",
            input_tokens=10,
            output_tokens=5,
        )
        assert response.get_text() == "Generated text"

    def test_get_raw_returns_text(self) -> None:
        response = InEngineResponse(
            perf_ns=1000,
            text="Generated text",
            input_tokens=10,
            output_tokens=5,
        )
        assert response.get_raw() == "Generated text"
