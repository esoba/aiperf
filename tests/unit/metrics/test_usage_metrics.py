# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponse, ParsedResponseRecord, RequestRecord
from aiperf.common.models.record_models import TextResponseData, TokenCounts
from aiperf.common.models.usage_models import Usage
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.usage_metrics import (
    TotalUsageAcceptedPredictionTokensMetric,
    TotalUsageCompletionAudioTokensMetric,
    TotalUsagePromptAudioTokensMetric,
    TotalUsagePromptCachedTokensMetric,
    TotalUsageReasoningTokensMetric,
    TotalUsageRejectedPredictionTokensMetric,
    UsageAcceptedPredictionTokensMetric,
    UsageCompletionAudioTokensMetric,
    UsagePromptAudioTokensMetric,
    UsagePromptCachedTokensMetric,
    UsageReasoningTokensMetric,
    UsageRejectedPredictionTokensMetric,
)


def create_record_with_usage(
    start_ns: int = 100,
    completion_tokens_details: dict | None = None,
    prompt_tokens_details: dict | None = None,
    streaming: bool = False,
) -> ParsedResponseRecord:
    """Create a test record with usage details dicts."""
    request = RequestRecord(
        conversation_id="test-conversation",
        turn_index=0,
        model_name="test-model",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=start_ns + 100,
    )

    usage_dict: dict = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
    if completion_tokens_details is not None:
        usage_dict["completion_tokens_details"] = completion_tokens_details
    if prompt_tokens_details is not None:
        usage_dict["prompt_tokens_details"] = prompt_tokens_details

    usage = Usage(usage_dict)

    if streaming:
        # Simulate streaming: first chunk has no usage, last chunk has usage
        responses = [
            ParsedResponse(
                perf_ns=start_ns + 25,
                data=TextResponseData(text="chunk1"),
                usage=None,
            ),
            ParsedResponse(
                perf_ns=start_ns + 50,
                data=TextResponseData(text="chunk2"),
                usage=usage,
            ),
        ]
    else:
        responses = [
            ParsedResponse(
                perf_ns=start_ns + 50,
                data=TextResponseData(text="test"),
                usage=usage,
            ),
        ]

    return ParsedResponseRecord(
        request=request,
        responses=responses,
        token_counts=TokenCounts(input=100, output=50, reasoning=0),
    )


class TestUsagePromptCachedTokensMetric:
    """Tests for UsagePromptCachedTokensMetric."""

    def test_extracts_cached_tokens(self):
        record = create_record_with_usage(
            prompt_tokens_details={"cached_tokens": 42},
        )
        metric = UsagePromptCachedTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 42

    def test_returns_zero_cached_tokens(self):
        record = create_record_with_usage(
            prompt_tokens_details={"cached_tokens": 0},
        )
        metric = UsagePromptCachedTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_raises_when_missing(self):
        record = create_record_with_usage()
        metric = UsagePromptCachedTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_streaming_takes_last_non_none(self):
        record = create_record_with_usage(
            prompt_tokens_details={"cached_tokens": 77},
            streaming=True,
        )
        metric = UsagePromptCachedTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 77

    def test_metadata(self):
        assert UsagePromptCachedTokensMetric.tag == "usage_prompt_cached_tokens"
        assert UsagePromptCachedTokensMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert UsagePromptCachedTokensMetric.has_flags(MetricFlags.LARGER_IS_BETTER)
        assert UsagePromptCachedTokensMetric.missing_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert UsagePromptCachedTokensMetric.missing_flags(
            MetricFlags.SUPPORTS_AUDIO_ONLY
        )


class TestUsagePromptAudioTokensMetric:
    """Tests for UsagePromptAudioTokensMetric."""

    def test_extracts_prompt_audio_tokens(self):
        record = create_record_with_usage(
            prompt_tokens_details={"audio_tokens": 30},
        )
        metric = UsagePromptAudioTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 30

    def test_returns_zero_audio_tokens(self):
        record = create_record_with_usage(
            prompt_tokens_details={"audio_tokens": 0},
        )
        metric = UsagePromptAudioTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_raises_when_missing(self):
        record = create_record_with_usage()
        metric = UsagePromptAudioTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_streaming_takes_last_non_none(self):
        record = create_record_with_usage(
            prompt_tokens_details={"audio_tokens": 55},
            streaming=True,
        )
        metric = UsagePromptAudioTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 55

    def test_metadata(self):
        assert UsagePromptAudioTokensMetric.tag == "usage_prompt_audio_tokens"
        assert UsagePromptAudioTokensMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert UsagePromptAudioTokensMetric.has_flags(MetricFlags.LARGER_IS_BETTER)
        assert UsagePromptAudioTokensMetric.has_flags(MetricFlags.SUPPORTS_AUDIO_ONLY)
        assert UsagePromptAudioTokensMetric.missing_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )


class TestUsageCompletionAudioTokensMetric:
    """Tests for UsageCompletionAudioTokensMetric."""

    def test_extracts_completion_audio_tokens(self):
        record = create_record_with_usage(
            completion_tokens_details={"audio_tokens": 20},
        )
        metric = UsageCompletionAudioTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 20

    def test_returns_zero_audio_tokens(self):
        record = create_record_with_usage(
            completion_tokens_details={"audio_tokens": 0},
        )
        metric = UsageCompletionAudioTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_raises_when_missing(self):
        record = create_record_with_usage()
        metric = UsageCompletionAudioTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_streaming_takes_last_non_none(self):
        record = create_record_with_usage(
            completion_tokens_details={"audio_tokens": 88},
            streaming=True,
        )
        metric = UsageCompletionAudioTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 88

    def test_metadata(self):
        assert UsageCompletionAudioTokensMetric.tag == "usage_completion_audio_tokens"
        assert UsageCompletionAudioTokensMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert UsageCompletionAudioTokensMetric.has_flags(MetricFlags.LARGER_IS_BETTER)
        assert UsageCompletionAudioTokensMetric.has_flags(
            MetricFlags.SUPPORTS_AUDIO_ONLY
        )
        assert UsageCompletionAudioTokensMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )


class TestUsageAcceptedPredictionTokensMetric:
    """Tests for UsageAcceptedPredictionTokensMetric."""

    def test_extracts_accepted_prediction_tokens(self):
        record = create_record_with_usage(
            completion_tokens_details={"accepted_prediction_tokens": 15},
        )
        metric = UsageAcceptedPredictionTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 15

    def test_returns_zero_accepted_prediction_tokens(self):
        record = create_record_with_usage(
            completion_tokens_details={"accepted_prediction_tokens": 0},
        )
        metric = UsageAcceptedPredictionTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_raises_when_missing(self):
        record = create_record_with_usage()
        metric = UsageAcceptedPredictionTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_streaming_takes_last_non_none(self):
        record = create_record_with_usage(
            completion_tokens_details={"accepted_prediction_tokens": 99},
            streaming=True,
        )
        metric = UsageAcceptedPredictionTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 99

    def test_metadata(self):
        assert (
            UsageAcceptedPredictionTokensMetric.tag
            == "usage_accepted_prediction_tokens"
        )
        assert UsageAcceptedPredictionTokensMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert UsageAcceptedPredictionTokensMetric.has_flags(
            MetricFlags.LARGER_IS_BETTER
        )
        assert UsageAcceptedPredictionTokensMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert UsageAcceptedPredictionTokensMetric.missing_flags(
            MetricFlags.SUPPORTS_AUDIO_ONLY
        )


class TestUsageRejectedPredictionTokensMetric:
    """Tests for UsageRejectedPredictionTokensMetric."""

    def test_extracts_rejected_prediction_tokens(self):
        record = create_record_with_usage(
            completion_tokens_details={"rejected_prediction_tokens": 5},
        )
        metric = UsageRejectedPredictionTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 5

    def test_returns_zero_rejected_prediction_tokens(self):
        record = create_record_with_usage(
            completion_tokens_details={"rejected_prediction_tokens": 0},
        )
        metric = UsageRejectedPredictionTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_raises_when_missing(self):
        record = create_record_with_usage()
        metric = UsageRejectedPredictionTokensMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_streaming_takes_last_non_none(self):
        record = create_record_with_usage(
            completion_tokens_details={"rejected_prediction_tokens": 12},
            streaming=True,
        )
        metric = UsageRejectedPredictionTokensMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 12

    def test_metadata(self):
        assert (
            UsageRejectedPredictionTokensMetric.tag
            == "usage_rejected_prediction_tokens"
        )
        assert UsageRejectedPredictionTokensMetric.has_flags(MetricFlags.NO_CONSOLE)
        assert UsageRejectedPredictionTokensMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert UsageRejectedPredictionTokensMetric.missing_flags(
            MetricFlags.LARGER_IS_BETTER
        )
        assert UsageRejectedPredictionTokensMetric.missing_flags(
            MetricFlags.SUPPORTS_AUDIO_ONLY
        )


class TestTotalUsageDerivedSumMetrics:
    """Tests for Total* derived sum metrics wiring."""

    @pytest.mark.parametrize(
        "total_cls,record_cls",
        [
            (TotalUsageReasoningTokensMetric, UsageReasoningTokensMetric),
            (TotalUsagePromptCachedTokensMetric, UsagePromptCachedTokensMetric),
            (TotalUsagePromptAudioTokensMetric, UsagePromptAudioTokensMetric),
            (TotalUsageCompletionAudioTokensMetric, UsageCompletionAudioTokensMetric),
            (
                TotalUsageAcceptedPredictionTokensMetric,
                UsageAcceptedPredictionTokensMetric,
            ),
            (
                TotalUsageRejectedPredictionTokensMetric,
                UsageRejectedPredictionTokensMetric,
            ),
        ],
    )
    def test_derived_sum_wiring(self, total_cls, record_cls):
        assert total_cls.record_metric_type is record_cls
        assert total_cls.required_metrics == {record_cls.tag}
        assert total_cls.unit == record_cls.unit
        assert total_cls.flags == record_cls.flags
