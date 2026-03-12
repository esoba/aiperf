# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speculative decoding metrics for TRT-LLM in-engine transport.

These metrics capture per-request and aggregate speculative decoding statistics
when decode_iterations and max_draft_len metadata are available from TRT-LLM.
"""

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseDerivedMetric, BaseRecordMetric
from aiperf.metrics.derived_sum_metric import DerivedSumMetric
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
    TotalOutputSequenceLengthMetric,
)

_SPEC_DECODE_FLAGS = (
    MetricFlags.IN_ENGINE_SPEC_DECODE | MetricFlags.PRODUCES_TOKENS_ONLY
)


def _get_spec_decode_metadata(
    record: ParsedResponseRecord,
) -> tuple[int, int]:
    """Extract decode_iterations and max_draft_len from the last content response metadata.

    Args:
        record: Parsed response record with metadata on responses.

    Returns:
        Tuple of (decode_iterations, max_draft_len).

    Raises:
        NoMetricValue: If speculative decoding metadata is not available.
    """
    for response in reversed(record.responses):
        if "decode_iterations" in response.metadata:
            decode_iterations = response.metadata["decode_iterations"]
            max_draft_len = response.metadata.get("max_draft_len", 0)
            return int(decode_iterations), int(max_draft_len)

    raise NoMetricValue(
        "Speculative decoding metadata (decode_iterations) not available."
    )


# ---- Per-Request Record Metrics -----------------------------------------------


class DecodeIterationCountMetric(BaseRecordMetric[int]):
    """Number of decode iterations for a request (speculative decoding).

    Formula:
        Decode Iteration Count = decode_iterations + 1

    The +1 accounts for the verification step in speculative decoding.
    """

    tag = "decode_iteration_count"
    header = "Decode Iteration Count"
    short_header = "Decode Iters"
    unit = GenericMetricUnit.COUNT
    display_order = 900
    flags = _SPEC_DECODE_FLAGS | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        decode_iterations, _ = _get_spec_decode_metadata(record)
        return decode_iterations + 1


class DraftTokenCountMetric(BaseRecordMetric[int]):
    """Total draft tokens proposed for a request (speculative decoding).

    Formula:
        Draft Token Count = max_draft_len x (decode_iterations + 1)
    """

    tag = "draft_token_count"
    header = "Draft Token Count"
    short_header = "Drafted"
    unit = GenericMetricUnit.TOKENS
    display_order = 901
    flags = _SPEC_DECODE_FLAGS | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        decode_iterations, max_draft_len = _get_spec_decode_metadata(record)
        if max_draft_len <= 0:
            raise NoMetricValue("max_draft_len is zero or not configured.")
        return max_draft_len * (decode_iterations + 1)


class AcceptedDraftTokenCountMetric(BaseRecordMetric[int]):
    """Number of draft tokens accepted by the verifier (speculative decoding).

    Formula:
        Accepted Draft Tokens = output_tokens - decode_iterations - 1

    Each decode iteration produces at least one token from the verifier,
    so the accepted drafts are the surplus tokens beyond the verifier tokens.
    """

    tag = "accepted_draft_token_count"
    header = "Accepted Draft Token Count"
    short_header = "Accepted"
    unit = GenericMetricUnit.TOKENS
    display_order = 902
    flags = _SPEC_DECODE_FLAGS | MetricFlags.NO_CONSOLE
    required_metrics = {OutputSequenceLengthMetric.tag}

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        decode_iterations, _ = _get_spec_decode_metadata(record)
        osl = record_metrics.get_or_raise(OutputSequenceLengthMetric)
        accepted = osl - decode_iterations - 1  # type: ignore
        if accepted < 0:
            raise NoMetricValue(
                f"Accepted draft count is negative ({accepted}), data may be inconsistent."
            )
        return accepted


class DraftAcceptanceRateMetric(BaseRecordMetric[float]):
    """Fraction of drafted tokens that were accepted (speculative decoding).

    Formula:
        Draft Acceptance Rate = accepted_draft_tokens / draft_token_count
    """

    tag = "draft_acceptance_rate"
    header = "Draft Acceptance Rate"
    short_header = "Accept Rate"
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    display_order = 903
    flags = _SPEC_DECODE_FLAGS
    required_metrics = {
        AcceptedDraftTokenCountMetric.tag,
        DraftTokenCountMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        accepted = record_metrics.get_or_raise(AcceptedDraftTokenCountMetric)
        drafted = record_metrics.get_or_raise(DraftTokenCountMetric)
        if drafted == 0:  # type: ignore
            raise NoMetricValue(
                "Draft token count is zero, cannot compute acceptance rate."
            )
        return accepted / drafted  # type: ignore


class AcceptanceLengthMetric(BaseRecordMetric[float]):
    """Average number of tokens accepted per decode iteration (speculative decoding).

    Formula:
        Acceptance Length = output_tokens / (decode_iterations + 1)
    """

    tag = "acceptance_length"
    header = "Acceptance Length"
    short_header = "Accept Len"
    unit = GenericMetricUnit.TOKENS
    display_order = 904
    flags = _SPEC_DECODE_FLAGS
    required_metrics = {
        OutputSequenceLengthMetric.tag,
        DecodeIterationCountMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        osl = record_metrics.get_or_raise(OutputSequenceLengthMetric)
        decode_iter_count = record_metrics.get_or_raise(DecodeIterationCountMetric)
        if decode_iter_count == 0:  # type: ignore
            raise NoMetricValue("Decode iteration count is zero.")
        return osl / decode_iter_count  # type: ignore


# ---- Derived Aggregate Metrics ------------------------------------------------


class TotalDecodeIterationsMetric(DerivedSumMetric[int, DecodeIterationCountMetric]):
    """Total decode iterations across all requests.

    Formula:
        Total Decode Iterations = Sum(Decode Iteration Counts)
    """

    tag = "total_decode_iterations"
    header = "Total Decode Iterations"
    short_header = "Total Iters"
    short_header_hide_unit = True
    flags = _SPEC_DECODE_FLAGS | MetricFlags.NO_CONSOLE


class TotalDraftTokensMetric(DerivedSumMetric[int, DraftTokenCountMetric]):
    """Total draft tokens proposed across all requests.

    Formula:
        Total Draft Tokens = Sum(Draft Token Counts)
    """

    tag = "total_draft_tokens"
    header = "Total Draft Tokens"
    short_header = "Total Drafted"
    short_header_hide_unit = True
    flags = _SPEC_DECODE_FLAGS | MetricFlags.NO_CONSOLE


class TotalAcceptedDraftTokensMetric(
    DerivedSumMetric[int, AcceptedDraftTokenCountMetric]
):
    """Total accepted draft tokens across all requests.

    Formula:
        Total Accepted Draft Tokens = Sum(Accepted Draft Token Counts)
    """

    tag = "total_accepted_draft_tokens"
    header = "Total Accepted Draft Tokens"
    short_header = "Total Accepted"
    short_header_hide_unit = True
    flags = _SPEC_DECODE_FLAGS | MetricFlags.NO_CONSOLE


class OverallDraftAcceptanceRateMetric(BaseDerivedMetric[float]):
    """Overall draft acceptance rate across all requests.

    Formula:
        Overall Draft Acceptance Rate = total_accepted_draft_tokens / total_draft_tokens
    """

    tag = "overall_draft_acceptance_rate"
    header = "Overall Draft Acceptance Rate"
    short_header = "Overall Accept Rate"
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    display_order = 910
    flags = _SPEC_DECODE_FLAGS
    required_metrics = {
        TotalAcceptedDraftTokensMetric.tag,
        TotalDraftTokensMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        total_accepted = metric_results.get_or_raise(TotalAcceptedDraftTokensMetric)
        total_drafted = metric_results.get_or_raise(TotalDraftTokensMetric)
        if total_drafted == 0:  # type: ignore
            raise NoMetricValue(
                "Total draft tokens is zero, cannot compute overall acceptance rate."
            )
        return total_accepted / total_drafted  # type: ignore


class OverallAcceptanceLengthMetric(BaseDerivedMetric[float]):
    """Overall average acceptance length across all requests.

    Formula:
        Overall Acceptance Length = total_osl / total_decode_iterations
    """

    tag = "overall_acceptance_length"
    header = "Overall Acceptance Length"
    short_header = "Overall Accept Len"
    unit = GenericMetricUnit.TOKENS
    display_order = 911
    flags = _SPEC_DECODE_FLAGS
    required_metrics = {
        TotalOutputSequenceLengthMetric.tag,
        TotalDecodeIterationsMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        total_osl = metric_results.get_or_raise(TotalOutputSequenceLengthMetric)
        total_decode_iters = metric_results.get_or_raise(TotalDecodeIterationsMetric)
        if total_decode_iters == 0:  # type: ignore
            raise NoMetricValue(
                "Total decode iterations is zero, cannot compute overall acceptance length."
            )
        return total_osl / total_decode_iters  # type: ignore
