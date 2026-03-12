# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for speculative decoding metrics (TRT-LLM in-engine transport).

Focuses on:
- Per-request record metrics (decode iteration count, draft/accepted counts, rates, length)
- Derived aggregate metrics (totals, overall rates/lengths)
- _get_spec_decode_metadata helper extraction from response metadata
- Edge cases: missing metadata, zero values, negative accepted counts
- End-to-end pipeline with multi-request realistic data
"""

import pytest
from pytest import approx

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
    TotalOutputSequenceLengthMetric,
)
from aiperf.metrics.types.speculative_decoding_metrics import (
    AcceptanceLengthMetric,
    AcceptedDraftTokenCountMetric,
    DecodeIterationCountMetric,
    DraftAcceptanceRateMetric,
    DraftTokenCountMetric,
    OverallAcceptanceLengthMetric,
    OverallDraftAcceptanceRateMetric,
    TotalAcceptedDraftTokensMetric,
    TotalDecodeIterationsMetric,
    TotalDraftTokensMetric,
    _get_spec_decode_metadata,
)
from tests.unit.metrics.conftest import (
    create_metric_array,
    create_record,
    run_simple_metrics_pipeline,
)


def _create_spec_decode_record(
    output_tokens: int,
    decode_iterations: int,
    max_draft_len: int,
    input_tokens: int = 100,
) -> ParsedResponseRecord:
    """Create a ParsedResponseRecord with speculative decoding metadata."""
    record = create_record(
        responses=[200],
        input_tokens=input_tokens,
        output_tokens_per_response=output_tokens,
    )
    # Inject metadata into the last response
    record.responses[-1].metadata["decode_iterations"] = decode_iterations
    record.responses[-1].metadata["max_draft_len"] = max_draft_len
    return record


def _create_record_no_spec_metadata(
    output_tokens: int = 10,
) -> ParsedResponseRecord:
    """Create a record without speculative decoding metadata."""
    return create_record(
        responses=[200],
        output_tokens_per_response=output_tokens,
    )


# ---- DecodeIterationCountMetric -----------------------------------------------


class TestDecodeIterationCountMetric:
    @pytest.mark.parametrize(
        "decode_iterations, expected",
        [
            (0, 1),
            (5, 6),
            (10, 11),
            (99, 100),
        ],
    )
    def test_basic_computation(self, decode_iterations: int, expected: int) -> None:
        """Test decode iteration count = decode_iterations + 1."""
        record = _create_spec_decode_record(
            output_tokens=20, decode_iterations=decode_iterations, max_draft_len=5
        )
        metric = DecodeIterationCountMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == expected

    def test_no_metadata_raises(self) -> None:
        """Test that missing metadata raises NoMetricValue."""
        record = _create_record_no_spec_metadata()
        metric = DecodeIterationCountMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_metric_metadata(self) -> None:
        """Test metric flags and tag."""
        assert DecodeIterationCountMetric.tag == "decode_iteration_count"
        assert DecodeIterationCountMetric.has_flags(MetricFlags.IN_ENGINE_SPEC_DECODE)
        assert DecodeIterationCountMetric.has_flags(MetricFlags.PRODUCES_TOKENS_ONLY)
        assert DecodeIterationCountMetric.has_flags(MetricFlags.NO_CONSOLE)


# ---- DraftTokenCountMetric ---------------------------------------------------


class TestDraftTokenCountMetric:
    @pytest.mark.parametrize(
        "decode_iterations, max_draft_len, expected",
        [
            (0, 5, 5),  # 5 * (0 + 1) = 5
            (4, 5, 25),  # 5 * (4 + 1) = 25
            (9, 3, 30),  # 3 * (9 + 1) = 30
            (0, 1, 1),  # 1 * (0 + 1) = 1
        ],
    )
    def test_basic_computation(
        self, decode_iterations: int, max_draft_len: int, expected: int
    ) -> None:
        """Test draft token count = max_draft_len * (decode_iterations + 1)."""
        record = _create_spec_decode_record(
            output_tokens=20,
            decode_iterations=decode_iterations,
            max_draft_len=max_draft_len,
        )
        metric = DraftTokenCountMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == expected

    def test_zero_max_draft_len_raises(self) -> None:
        """Test that max_draft_len=0 raises NoMetricValue."""
        record = _create_spec_decode_record(
            output_tokens=20, decode_iterations=5, max_draft_len=0
        )
        metric = DraftTokenCountMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_no_metadata_raises(self) -> None:
        """Test that missing metadata raises NoMetricValue."""
        record = _create_record_no_spec_metadata()
        metric = DraftTokenCountMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())


# ---- AcceptedDraftTokenCountMetric -------------------------------------------


class TestAcceptedDraftTokenCountMetric:
    @pytest.mark.parametrize(
        "output_tokens, decode_iterations, expected",
        [
            (20, 4, 15),  # 20 - 4 - 1 = 15
            (10, 0, 9),  # 10 - 0 - 1 = 9
            (1, 0, 0),  # 1 - 0 - 1 = 0
            (50, 9, 40),  # 50 - 9 - 1 = 40
        ],
    )
    def test_basic_computation(
        self, output_tokens: int, decode_iterations: int, expected: int
    ) -> None:
        """Test accepted = output_tokens - decode_iterations - 1."""
        record = _create_spec_decode_record(
            output_tokens=output_tokens,
            decode_iterations=decode_iterations,
            max_draft_len=5,
        )
        metric = AcceptedDraftTokenCountMetric()
        osl_metric = OutputSequenceLengthMetric()
        record_metrics = MetricRecordDict()
        record_metrics[OutputSequenceLengthMetric.tag] = osl_metric.parse_record(
            record, record_metrics
        )
        result = metric.parse_record(record, record_metrics)
        assert result == expected

    def test_negative_accepted_raises(self) -> None:
        """Test that negative accepted count raises NoMetricValue."""
        record = _create_spec_decode_record(
            output_tokens=1, decode_iterations=5, max_draft_len=5
        )
        metric = AcceptedDraftTokenCountMetric()
        osl_metric = OutputSequenceLengthMetric()
        record_metrics = MetricRecordDict()
        record_metrics[OutputSequenceLengthMetric.tag] = osl_metric.parse_record(
            record, record_metrics
        )
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, record_metrics)


# ---- DraftAcceptanceRateMetric -----------------------------------------------


class TestDraftAcceptanceRateMetric:
    @pytest.mark.parametrize(
        "accepted, drafted, expected",
        [
            (15, 25, 0.6),
            (0, 10, 0.0),
            (10, 10, 1.0),
            (5, 20, 0.25),
        ],
    )
    def test_basic_computation(
        self, accepted: int, drafted: int, expected: float
    ) -> None:
        """Test acceptance rate = accepted / drafted."""
        metric = DraftAcceptanceRateMetric()
        record_metrics = MetricRecordDict()
        record_metrics[AcceptedDraftTokenCountMetric.tag] = accepted
        record_metrics[DraftTokenCountMetric.tag] = drafted
        # Need a valid record for parse_record
        record = _create_spec_decode_record(
            output_tokens=20, decode_iterations=5, max_draft_len=5
        )
        result = metric.parse_record(record, record_metrics)
        assert result == approx(expected)

    def test_zero_drafted_raises(self) -> None:
        """Test that zero drafted tokens raises NoMetricValue."""
        metric = DraftAcceptanceRateMetric()
        record_metrics = MetricRecordDict()
        record_metrics[AcceptedDraftTokenCountMetric.tag] = 0
        record_metrics[DraftTokenCountMetric.tag] = 0
        record = _create_spec_decode_record(
            output_tokens=20, decode_iterations=5, max_draft_len=5
        )
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, record_metrics)


# ---- AcceptanceLengthMetric --------------------------------------------------


class TestAcceptanceLengthMetric:
    @pytest.mark.parametrize(
        "output_tokens, decode_iterations, expected",
        [
            (20, 4, 4.0),  # 20 / (4 + 1) = 4.0
            (10, 0, 10.0),  # 10 / (0 + 1) = 10.0
            (15, 4, 3.0),  # 15 / (4 + 1) = 3.0
            (7, 2, 7 / 3),  # 7 / (2 + 1) = 2.333...
        ],
    )
    def test_basic_computation(
        self, output_tokens: int, decode_iterations: int, expected: float
    ) -> None:
        """Test acceptance length = osl / decode_iteration_count."""
        metric = AcceptanceLengthMetric()
        record_metrics = MetricRecordDict()
        record_metrics[OutputSequenceLengthMetric.tag] = output_tokens
        record_metrics[DecodeIterationCountMetric.tag] = decode_iterations + 1
        record = _create_spec_decode_record(
            output_tokens=output_tokens,
            decode_iterations=decode_iterations,
            max_draft_len=5,
        )
        result = metric.parse_record(record, record_metrics)
        assert result == approx(expected)


# ---- Derived Sum Metrics -----------------------------------------------------


class TestTotalDecodeIterationsMetric:
    @pytest.mark.parametrize(
        "values, expected_sum",
        [
            ([6, 11, 4], 21),
            ([1], 1),
            ([], 0),
            ([0, 0], 0),
        ],
    )
    def test_sum_calculation(self, values: list[int], expected_sum: int) -> None:
        """Test total decode iterations sums correctly."""
        metric = TotalDecodeIterationsMetric()
        metric_results = MetricResultsDict()
        metric_results[DecodeIterationCountMetric.tag] = create_metric_array(values)
        result = metric.derive_value(metric_results)
        assert result == expected_sum

    def test_metric_metadata(self) -> None:
        assert TotalDecodeIterationsMetric.tag == "total_decode_iterations"
        assert TotalDecodeIterationsMetric.has_flags(MetricFlags.IN_ENGINE_SPEC_DECODE)
        assert TotalDecodeIterationsMetric.has_flags(MetricFlags.NO_CONSOLE)


class TestTotalDraftTokensMetric:
    @pytest.mark.parametrize(
        "values, expected_sum",
        [
            ([25, 30, 5], 60),
            ([100], 100),
            ([], 0),
        ],
    )
    def test_sum_calculation(self, values: list[int], expected_sum: int) -> None:
        """Test total draft tokens sums correctly."""
        metric = TotalDraftTokensMetric()
        metric_results = MetricResultsDict()
        metric_results[DraftTokenCountMetric.tag] = create_metric_array(values)
        result = metric.derive_value(metric_results)
        assert result == expected_sum


class TestTotalAcceptedDraftTokensMetric:
    @pytest.mark.parametrize(
        "values, expected_sum",
        [
            ([15, 20, 3], 38),
            ([0], 0),
            ([], 0),
        ],
    )
    def test_sum_calculation(self, values: list[int], expected_sum: int) -> None:
        """Test total accepted draft tokens sums correctly."""
        metric = TotalAcceptedDraftTokensMetric()
        metric_results = MetricResultsDict()
        metric_results[AcceptedDraftTokenCountMetric.tag] = create_metric_array(values)
        result = metric.derive_value(metric_results)
        assert result == expected_sum


# ---- Overall Derived Metrics -------------------------------------------------


class TestOverallDraftAcceptanceRateMetric:
    @pytest.mark.parametrize(
        "total_accepted, total_drafted, expected",
        [
            (38, 60, 38 / 60),
            (0, 100, 0.0),
            (50, 50, 1.0),
        ],
    )
    def test_basic_computation(
        self, total_accepted: int, total_drafted: int, expected: float
    ) -> None:
        """Test overall acceptance rate = total_accepted / total_drafted."""
        metric = OverallDraftAcceptanceRateMetric()
        metric_results = MetricResultsDict()
        metric_results[TotalAcceptedDraftTokensMetric.tag] = total_accepted
        metric_results[TotalDraftTokensMetric.tag] = total_drafted
        result = metric.derive_value(metric_results)
        assert result == approx(expected)

    def test_zero_drafted_raises(self) -> None:
        """Test that zero total drafted raises NoMetricValue."""
        metric = OverallDraftAcceptanceRateMetric()
        metric_results = MetricResultsDict()
        metric_results[TotalAcceptedDraftTokensMetric.tag] = 0
        metric_results[TotalDraftTokensMetric.tag] = 0
        with pytest.raises(NoMetricValue):
            metric.derive_value(metric_results)

    def test_metric_metadata(self) -> None:
        assert OverallDraftAcceptanceRateMetric.tag == "overall_draft_acceptance_rate"
        assert OverallDraftAcceptanceRateMetric.has_flags(
            MetricFlags.IN_ENGINE_SPEC_DECODE
        )
        assert OverallDraftAcceptanceRateMetric.has_flags(
            MetricFlags.PRODUCES_TOKENS_ONLY
        )
        assert OverallDraftAcceptanceRateMetric.missing_flags(MetricFlags.NO_CONSOLE)


class TestOverallAcceptanceLengthMetric:
    @pytest.mark.parametrize(
        "total_osl, total_decode_iters, expected",
        [
            (100, 20, 5.0),
            (50, 10, 5.0),
            (7, 3, 7 / 3),
        ],
    )
    def test_basic_computation(
        self, total_osl: int, total_decode_iters: int, expected: float
    ) -> None:
        """Test overall acceptance length = total_osl / total_decode_iterations."""
        metric = OverallAcceptanceLengthMetric()
        metric_results = MetricResultsDict()
        metric_results[TotalOutputSequenceLengthMetric.tag] = total_osl
        metric_results[TotalDecodeIterationsMetric.tag] = total_decode_iters
        result = metric.derive_value(metric_results)
        assert result == approx(expected)

    def test_zero_iterations_raises(self) -> None:
        """Test that zero total decode iterations raises NoMetricValue."""
        metric = OverallAcceptanceLengthMetric()
        metric_results = MetricResultsDict()
        metric_results[TotalOutputSequenceLengthMetric.tag] = 100
        metric_results[TotalDecodeIterationsMetric.tag] = 0
        with pytest.raises(NoMetricValue):
            metric.derive_value(metric_results)


# ---- End-to-End Pipeline Test ------------------------------------------------


class TestSpecDecodeEndToEndPipeline:
    def test_full_pipeline_multiple_records(self) -> None:
        """Test the full metrics pipeline with speculative decoding records."""
        records = [
            _create_spec_decode_record(
                output_tokens=20, decode_iterations=4, max_draft_len=5
            ),
            _create_spec_decode_record(
                output_tokens=15, decode_iterations=2, max_draft_len=5
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            AcceptanceLengthMetric.tag,
            DraftAcceptanceRateMetric.tag,
            OverallDraftAcceptanceRateMetric.tag,
            OverallAcceptanceLengthMetric.tag,
        )

        # Per-record acceptance lengths: 20/5=4.0, 15/3=5.0
        assert metric_results[AcceptanceLengthMetric.tag] == approx([4.0, 5.0])

        # Per-record acceptance rates:
        # Record 1: accepted=20-4-1=15, drafted=5*5=25 -> 15/25=0.6
        # Record 2: accepted=15-2-1=12, drafted=5*3=15 -> 12/15=0.8
        assert metric_results[DraftAcceptanceRateMetric.tag] == approx([0.6, 0.8])

        # Overall acceptance rate: (15+12)/(25+15) = 27/40 = 0.675
        assert metric_results[OverallDraftAcceptanceRateMetric.tag] == approx(0.675)

        # Overall acceptance length: (20+15)/(5+3) = 35/8 = 4.375
        assert metric_results[OverallAcceptanceLengthMetric.tag] == approx(4.375)


# ---- _get_spec_decode_metadata Helper ----------------------------------------


class TestGetSpecDecodeMetadata:
    """Verify _get_spec_decode_metadata extracts metadata from reversed response list."""

    def test_extracts_from_last_response(self) -> None:
        record = _create_spec_decode_record(
            output_tokens=20, decode_iterations=7, max_draft_len=3
        )
        decode_iters, max_draft = _get_spec_decode_metadata(record)
        assert decode_iters == 7
        assert max_draft == 3

    def test_missing_metadata_raises_no_metric_value(self) -> None:
        record = _create_record_no_spec_metadata()
        with pytest.raises(NoMetricValue, match="decode_iterations"):
            _get_spec_decode_metadata(record)

    def test_max_draft_len_defaults_to_zero_when_absent(self) -> None:
        """When decode_iterations is present but max_draft_len is not, default to 0."""
        record = create_record(responses=[200], output_tokens_per_response=10)
        record.responses[-1].metadata["decode_iterations"] = 5
        # max_draft_len not set
        decode_iters, max_draft = _get_spec_decode_metadata(record)
        assert decode_iters == 5
        assert max_draft == 0

    def test_coerces_string_values_to_int(self) -> None:
        """Engine metadata may arrive as strings; _get_spec_decode_metadata casts to int."""
        record = create_record(responses=[200], output_tokens_per_response=10)
        record.responses[-1].metadata["decode_iterations"] = "4"
        record.responses[-1].metadata["max_draft_len"] = "5"
        decode_iters, max_draft = _get_spec_decode_metadata(record)
        assert decode_iters == 4
        assert max_draft == 5

    def test_multiple_responses_finds_metadata_on_last_with_key(self) -> None:
        """When multiple responses exist, metadata is found by scanning in reverse."""
        record = create_record(responses=[100, 200], output_tokens_per_response=10)
        # Only the second response has metadata
        record.responses[1].metadata["decode_iterations"] = 3
        record.responses[1].metadata["max_draft_len"] = 2
        decode_iters, max_draft = _get_spec_decode_metadata(record)
        assert decode_iters == 3
        assert max_draft == 2


# ---- AcceptanceLengthMetric: zero decode iterations -------------------------


class TestAcceptanceLengthMetricEdgeCases:
    """Verify AcceptanceLengthMetric edge cases."""

    def test_zero_decode_iteration_count_raises(self) -> None:
        """AcceptanceLengthMetric raises NoMetricValue when decode_iteration_count is 0."""
        metric = AcceptanceLengthMetric()
        record_metrics = MetricRecordDict()
        record_metrics[OutputSequenceLengthMetric.tag] = 10
        record_metrics[DecodeIterationCountMetric.tag] = 0
        record = _create_spec_decode_record(
            output_tokens=10, decode_iterations=0, max_draft_len=5
        )
        with pytest.raises(NoMetricValue, match="zero"):
            metric.parse_record(record, record_metrics)


# ---- Metric Flag Coverage ---------------------------------------------------


class TestMetricFlagsCoverage:
    """Verify all spec decode metrics have correct flags and tags."""

    @pytest.mark.parametrize(
        "metric_cls,expected_tag",
        [
            (DecodeIterationCountMetric, "decode_iteration_count"),
            (DraftTokenCountMetric, "draft_token_count"),
            (AcceptedDraftTokenCountMetric, "accepted_draft_token_count"),
            (DraftAcceptanceRateMetric, "draft_acceptance_rate"),
            (AcceptanceLengthMetric, "acceptance_length"),
            (TotalDecodeIterationsMetric, "total_decode_iterations"),
            (TotalDraftTokensMetric, "total_draft_tokens"),
            (TotalAcceptedDraftTokensMetric, "total_accepted_draft_tokens"),
            (OverallDraftAcceptanceRateMetric, "overall_draft_acceptance_rate"),
            (OverallAcceptanceLengthMetric, "overall_acceptance_length"),
        ],
    )  # fmt: skip
    def test_all_metrics_have_spec_decode_flag(
        self, metric_cls: type, expected_tag: str
    ) -> None:
        assert metric_cls.tag == expected_tag
        assert metric_cls.has_flags(MetricFlags.IN_ENGINE_SPEC_DECODE)
        assert metric_cls.has_flags(MetricFlags.PRODUCES_TOKENS_ONLY)

    @pytest.mark.parametrize(
        "metric_cls",
        [
            DraftAcceptanceRateMetric,
            AcceptanceLengthMetric,
            OverallDraftAcceptanceRateMetric,
            OverallAcceptanceLengthMetric,
        ],
    )  # fmt: skip
    def test_user_visible_metrics_not_hidden(self, metric_cls: type) -> None:
        """Console-visible metrics should NOT have NO_CONSOLE flag."""
        assert metric_cls.missing_flags(MetricFlags.NO_CONSOLE)

    @pytest.mark.parametrize(
        "metric_cls",
        [
            DecodeIterationCountMetric,
            DraftTokenCountMetric,
            AcceptedDraftTokenCountMetric,
            TotalDecodeIterationsMetric,
            TotalDraftTokensMetric,
            TotalAcceptedDraftTokensMetric,
        ],
    )  # fmt: skip
    def test_intermediate_metrics_hidden_from_console(self, metric_cls: type) -> None:
        """Intermediate/helper metrics should have NO_CONSOLE flag."""
        assert metric_cls.has_flags(MetricFlags.NO_CONSOLE)


# ---- Realistic Multi-Request Pipeline ----------------------------------------


class TestSpecDecodeRealisticPipeline:
    """End-to-end pipeline with realistic multi-request speculative decoding data."""

    def test_five_request_pipeline(self) -> None:
        """Simulate 5 requests with varying decode iterations and draft lengths."""
        # Realistic scenario: mix of easy/hard requests with draft length 5
        records = [
            _create_spec_decode_record(
                output_tokens=30, decode_iterations=5, max_draft_len=5
            ),
            _create_spec_decode_record(
                output_tokens=10, decode_iterations=3, max_draft_len=5
            ),
            _create_spec_decode_record(
                output_tokens=50, decode_iterations=8, max_draft_len=5
            ),
            _create_spec_decode_record(
                output_tokens=25, decode_iterations=6, max_draft_len=5
            ),
            _create_spec_decode_record(
                output_tokens=100, decode_iterations=15, max_draft_len=5
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            DecodeIterationCountMetric.tag,
            DraftTokenCountMetric.tag,
            AcceptedDraftTokenCountMetric.tag,
            DraftAcceptanceRateMetric.tag,
            AcceptanceLengthMetric.tag,
            OverallDraftAcceptanceRateMetric.tag,
            OverallAcceptanceLengthMetric.tag,
        )

        # Decode iteration counts: 5+1=6, 3+1=4, 8+1=9, 6+1=7, 15+1=16
        assert metric_results[DecodeIterationCountMetric.tag] == [6, 4, 9, 7, 16]

        # Draft token counts: 5*6=30, 5*4=20, 5*9=45, 5*7=35, 5*16=80
        assert metric_results[DraftTokenCountMetric.tag] == [30, 20, 45, 35, 80]

        # Accepted: osl-iters-1 = 30-5-1=24, 10-3-1=6, 50-8-1=41, 25-6-1=18, 100-15-1=84
        assert metric_results[AcceptedDraftTokenCountMetric.tag] == [24, 6, 41, 18, 84]

        # Acceptance rates
        assert metric_results[DraftAcceptanceRateMetric.tag] == approx(
            [24 / 30, 6 / 20, 41 / 45, 18 / 35, 84 / 80]
        )

        # Acceptance lengths: osl / decode_iter_count
        assert metric_results[AcceptanceLengthMetric.tag] == approx(
            [30 / 6, 10 / 4, 50 / 9, 25 / 7, 100 / 16]
        )

        # Overall acceptance rate: sum(accepted) / sum(drafted)
        total_accepted = 24 + 6 + 41 + 18 + 84  # 173
        total_drafted = 30 + 20 + 45 + 35 + 80  # 210
        assert metric_results[OverallDraftAcceptanceRateMetric.tag] == approx(
            total_accepted / total_drafted
        )

        # Overall acceptance length: sum(osl) / sum(decode_iter_count)
        total_osl = 30 + 10 + 50 + 25 + 100  # 215
        total_iters = 6 + 4 + 9 + 7 + 16  # 42
        assert metric_results[OverallAcceptanceLengthMetric.tag] == approx(
            total_osl / total_iters
        )

    def test_single_request_pipeline(self) -> None:
        """Single request pipeline should still compute all metrics."""
        records = [
            _create_spec_decode_record(
                output_tokens=20, decode_iterations=4, max_draft_len=3
            ),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            DraftAcceptanceRateMetric.tag,
            AcceptanceLengthMetric.tag,
            OverallDraftAcceptanceRateMetric.tag,
            OverallAcceptanceLengthMetric.tag,
        )

        # accepted = 20-4-1 = 15, drafted = 3*5 = 15
        assert metric_results[DraftAcceptanceRateMetric.tag] == approx([1.0])
        assert metric_results[AcceptanceLengthMetric.tag] == approx([4.0])
        # Overall should match per-request for single record
        assert metric_results[OverallDraftAcceptanceRateMetric.tag] == approx(1.0)
        assert metric_results[OverallAcceptanceLengthMetric.tag] == approx(4.0)

    def test_mixed_records_with_and_without_spec_decode(self) -> None:
        """Records without spec decode metadata are skipped gracefully."""
        records = [
            _create_spec_decode_record(
                output_tokens=20, decode_iterations=4, max_draft_len=5
            ),
            _create_record_no_spec_metadata(output_tokens=10),
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            DraftAcceptanceRateMetric.tag,
            AcceptanceLengthMetric.tag,
        )

        # Only the first record should produce spec decode metric values
        assert metric_results[DraftAcceptanceRateMetric.tag] == approx([0.6])
        assert metric_results[AcceptanceLengthMetric.tag] == approx([4.0])
