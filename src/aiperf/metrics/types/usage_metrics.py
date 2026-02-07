# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API usage field token metrics.

These metrics track token counts as reported in the API response's usage field.
"""

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.derived_sum_metric import DerivedSumMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class UsagePromptTokensMetric(BaseRecordMetric[int]):
    """
    API usage field prompt token count metric.

    This represents the number of prompt (input) tokens as reported in the
    API response's usage field. Recorded for reference and comparison.

    Formula:
        Usage Prompt Tokens = response.usage.prompt_tokens (last non-None)
    """

    tag = "usage_prompt_tokens"
    header = "Usage Prompt Tokens"
    short_header = "Usage Prompt"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported prompt token count from the record.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide prompt token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                if prompt_tokens is not None:
                    return prompt_tokens

        raise NoMetricValue("Usage prompt token count is not available in the record.")


class UsageCompletionTokensMetric(BaseRecordMetric[int]):
    """
    API usage field completion token count metric.

    This represents the number of completion (output) tokens as reported in the
    API response's usage field. Recorded for reference and comparison.

    Formula:
        Usage Completion Tokens = response.usage.completion_tokens (last non-None)
    """

    tag = "usage_completion_tokens"
    header = "Usage Completion Tokens"
    short_header = "Usage Completion"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported completion token count from the record.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide completion token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                completion_tokens = response.usage.completion_tokens
                if completion_tokens is not None:
                    return completion_tokens

        raise NoMetricValue(
            "Usage completion token count is not available in the record."
        )


class UsageTotalTokensMetric(BaseRecordMetric[int]):
    """
    API usage field total token count metric.

    This represents the total number of tokens (prompt + completion) as reported
    in the API response's usage field. Recorded for reference and comparison.

    Formula:
        Usage Total Tokens = response.usage.total_tokens (last non-None)
    """

    tag = "usage_total_tokens"
    header = "Usage Total Tokens"
    short_header = "Usage Total"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported total token count from the record.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide total token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                total_tokens = response.usage.total_tokens
                if total_tokens is not None:
                    return total_tokens

        raise NoMetricValue("Usage total token count is not available in the record.")


class UsageReasoningTokensMetric(BaseRecordMetric[int]):
    """
    API usage field reasoning token count metric.

    This represents the number of reasoning tokens as reported in the
    API response's usage field (for models that support reasoning).
    Recorded for reference and comparison.

    Formula:
        Usage Reasoning Tokens = response.usage.completion_tokens_details.reasoning_tokens (last non-None)
    """

    tag = "usage_reasoning_tokens"
    header = "Usage Reasoning Tokens"
    short_header = "Usage Reasoning"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported reasoning token count from the record.

        Reasoning tokens are nested in completion_tokens_details.reasoning_tokens
        (or output_tokens_details.reasoning_tokens) per the official OpenAI spec.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide reasoning token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                reasoning = response.usage.reasoning_tokens
                if reasoning is not None:
                    return reasoning

        raise NoMetricValue(
            "Usage reasoning token count is not available in the record."
        )


class UsagePromptCachedTokensMetric(BaseRecordMetric[int]):
    """
    API usage field prompt cached token count metric.

    This represents the number of cached tokens from prompt_tokens_details
    as reported in the API response's usage field.

    Formula:
        Usage Prompt Cached Tokens = response.usage.prompt_tokens_details.cached_tokens (last non-None)
    """

    tag = "usage_prompt_cached_tokens"
    header = "Usage Prompt Cached Tokens"
    short_header = "Usage Prompt Cached"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.NO_CONSOLE | MetricFlags.LARGER_IS_BETTER
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported prompt cached token count from the record.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide prompt cached token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                cached = response.usage.prompt_cached_tokens
                if cached is not None:
                    return cached

        raise NoMetricValue(
            "Usage prompt cached token count is not available in the record."
        )


class UsagePromptAudioTokensMetric(BaseRecordMetric[int]):
    """
    API usage field prompt audio token count metric.

    This represents the number of audio tokens from prompt_tokens_details
    as reported in the API response's usage field.

    Formula:
        Usage Prompt Audio Tokens = response.usage.prompt_tokens_details.audio_tokens (last non-None)
    """

    tag = "usage_prompt_audio_tokens"
    header = "Usage Prompt Audio Tokens"
    short_header = "Usage Prompt Audio"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.NO_CONSOLE
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.SUPPORTS_AUDIO_ONLY
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported prompt audio token count from the record.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide prompt audio token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                audio = response.usage.prompt_audio_tokens
                if audio is not None:
                    return audio

        raise NoMetricValue(
            "Usage prompt audio token count is not available in the record."
        )


class UsageCompletionAudioTokensMetric(BaseRecordMetric[int]):
    """
    API usage field completion audio token count metric.

    This represents the number of audio tokens from completion_tokens_details
    as reported in the API response's usage field.

    Formula:
        Usage Completion Audio Tokens = response.usage.completion_tokens_details.audio_tokens (last non-None)
    """

    tag = "usage_completion_audio_tokens"
    header = "Usage Completion Audio Tokens"
    short_header = "Usage Completion Audio"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.NO_CONSOLE
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.SUPPORTS_AUDIO_ONLY
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported completion audio token count from the record.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide completion audio token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                audio = response.usage.completion_audio_tokens
                if audio is not None:
                    return audio

        raise NoMetricValue(
            "Usage completion audio token count is not available in the record."
        )


class UsageAcceptedPredictionTokensMetric(BaseRecordMetric[int]):
    """
    API usage field accepted prediction token count metric.

    This represents the number of accepted prediction tokens from
    completion_tokens_details as reported in the API response's usage field.

    Formula:
        Usage Accepted Prediction Tokens = response.usage.completion_tokens_details.accepted_prediction_tokens (last non-None)
    """

    tag = "usage_accepted_prediction_tokens"
    header = "Usage Accepted Prediction Tokens"
    short_header = "Usage Accepted Pred"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.NO_CONSOLE
        | MetricFlags.LARGER_IS_BETTER
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported accepted prediction token count from the record.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide accepted prediction token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                accepted = response.usage.accepted_prediction_tokens
                if accepted is not None:
                    return accepted

        raise NoMetricValue(
            "Usage accepted prediction token count is not available in the record."
        )


class UsageRejectedPredictionTokensMetric(BaseRecordMetric[int]):
    """
    API usage field rejected prediction token count metric.

    This represents the number of rejected prediction tokens from
    completion_tokens_details as reported in the API response's usage field.

    Formula:
        Usage Rejected Prediction Tokens = response.usage.completion_tokens_details.rejected_prediction_tokens (last non-None)
    """

    tag = "usage_rejected_prediction_tokens"
    header = "Usage Rejected Prediction Tokens"
    short_header = "Usage Rejected Pred"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.NO_CONSOLE
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported rejected prediction token count from the record.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide rejected prediction token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                rejected = response.usage.rejected_prediction_tokens
                if rejected is not None:
                    return rejected

        raise NoMetricValue(
            "Usage rejected prediction token count is not available in the record."
        )


class TotalUsagePromptTokensMetric(DerivedSumMetric[int, UsagePromptTokensMetric]):
    """
    Total API-reported prompt tokens across all requests.

    Formula:
        ```
        Total Usage Prompt Tokens = Sum(Usage Prompt Tokens)
        ```
    """

    tag = "total_usage_prompt_tokens"
    header = "Total Usage Prompt Tokens"
    short_header = "Total Usage Prompt"
    short_header_hide_unit = True


class TotalUsageCompletionTokensMetric(
    DerivedSumMetric[int, UsageCompletionTokensMetric]
):
    """
    Total API-reported completion tokens across all requests.

    Formula:
        ```
        Total Usage Completion Tokens = Sum(Usage Completion Tokens)
        ```
    """

    tag = "total_usage_completion_tokens"
    header = "Total Usage Completion Tokens"
    short_header = "Total Usage Completion"
    short_header_hide_unit = True


class TotalUsageTokensMetric(DerivedSumMetric[int, UsageTotalTokensMetric]):
    """
    Total API-reported total tokens across all requests.

    Formula:
        ```
        Total Usage Total Tokens = Sum(Usage Total Tokens)
        ```
    """

    tag = "total_usage_total_tokens"
    header = "Total Usage Total Tokens"
    short_header = "Total Usage Total"
    short_header_hide_unit = True


class TotalUsageReasoningTokensMetric(
    DerivedSumMetric[int, UsageReasoningTokensMetric]
):
    """
    Total API-reported reasoning tokens across all requests.

    Formula:
        ```
        Total Usage Reasoning Tokens = Sum(Usage Reasoning Tokens)
        ```
    """

    tag = "total_usage_reasoning_tokens"
    header = "Total Usage Reasoning Tokens"
    short_header = "Total Usage Reasoning"
    short_header_hide_unit = True


class TotalUsagePromptCachedTokensMetric(
    DerivedSumMetric[int, UsagePromptCachedTokensMetric]
):
    """
    Total API-reported prompt cached tokens across all requests.

    Formula:
        ```
        Total Usage Prompt Cached Tokens = Sum(Usage Prompt Cached Tokens)
        ```
    """

    tag = "total_usage_prompt_cached_tokens"
    header = "Total Usage Prompt Cached Tokens"
    short_header = "Total Usage Prompt Cached"
    short_header_hide_unit = True


class TotalUsagePromptAudioTokensMetric(
    DerivedSumMetric[int, UsagePromptAudioTokensMetric]
):
    """
    Total API-reported prompt audio tokens across all requests.

    Formula:
        ```
        Total Usage Prompt Audio Tokens = Sum(Usage Prompt Audio Tokens)
        ```
    """

    tag = "total_usage_prompt_audio_tokens"
    header = "Total Usage Prompt Audio Tokens"
    short_header = "Total Usage Prompt Audio"
    short_header_hide_unit = True


class TotalUsageCompletionAudioTokensMetric(
    DerivedSumMetric[int, UsageCompletionAudioTokensMetric]
):
    """
    Total API-reported completion audio tokens across all requests.

    Formula:
        ```
        Total Usage Completion Audio Tokens = Sum(Usage Completion Audio Tokens)
        ```
    """

    tag = "total_usage_completion_audio_tokens"
    header = "Total Usage Completion Audio Tokens"
    short_header = "Total Usage Comp Audio"
    short_header_hide_unit = True


class TotalUsageAcceptedPredictionTokensMetric(
    DerivedSumMetric[int, UsageAcceptedPredictionTokensMetric]
):
    """
    Total API-reported accepted prediction tokens across all requests.

    Formula:
        ```
        Total Usage Accepted Prediction Tokens = Sum(Usage Accepted Prediction Tokens)
        ```
    """

    tag = "total_usage_accepted_prediction_tokens"
    header = "Total Usage Accepted Prediction Tokens"
    short_header = "Total Usage Accepted Pred"
    short_header_hide_unit = True


class TotalUsageRejectedPredictionTokensMetric(
    DerivedSumMetric[int, UsageRejectedPredictionTokensMetric]
):
    """
    Total API-reported rejected prediction tokens across all requests.

    Formula:
        ```
        Total Usage Rejected Prediction Tokens = Sum(Usage Rejected Prediction Tokens)
        ```
    """

    tag = "total_usage_rejected_prediction_tokens"
    header = "Total Usage Rejected Prediction Tokens"
    short_header = "Total Usage Rejected Pred"
    short_header_hide_unit = True
