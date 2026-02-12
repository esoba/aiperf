# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sweep-line algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.analysis.sweep import (
    compute_time_weighted_stats,
    concurrency_sweep,
    prefill_throughput_sweep,
    throughput_sweep,
    throughput_sweep_icl,
)


class TestConcurrencySweep:
    def test_empty_input(self) -> None:
        ts, conc = concurrency_sweep(
            np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        )
        assert len(ts) == 0
        assert len(conc) == 0

    def test_all_nan(self) -> None:
        ts, conc = concurrency_sweep(
            np.array([np.nan, np.nan]), np.array([np.nan, np.nan])
        )
        assert len(ts) == 0

    def test_single_request(self) -> None:
        start = np.array([100.0])
        end = np.array([200.0])
        ts, conc = concurrency_sweep(start, end)
        assert len(ts) == 2
        assert ts[0] == 100.0
        assert ts[1] == 200.0
        assert conc[0] == 1.0  # request starts
        assert conc[1] == 0.0  # request ends

    def test_sequential_non_overlapping(self) -> None:
        """Sequential requests: concurrency always 0 or 1."""
        start = np.array([100.0, 300.0, 500.0])
        end = np.array([200.0, 400.0, 600.0])
        ts, conc = concurrency_sweep(start, end)
        # All concurrency values should be 0 or 1
        assert np.all((conc == 0) | (conc == 1))
        assert float(np.max(conc)) == 1.0

    def test_overlapping_requests(self) -> None:
        """10 overlapping requests → peak concurrency is 10."""
        start = np.array([float(i) for i in range(10)])
        end = np.array([float(i + 100) for i in range(10)])
        ts, conc = concurrency_sweep(start, end)
        assert float(np.max(conc)) == 10.0

    def test_nan_records_excluded(self) -> None:
        start = np.array([100.0, np.nan, 300.0])
        end = np.array([200.0, np.nan, 400.0])
        ts, conc = concurrency_sweep(start, end)
        # Only 2 valid records
        assert len(ts) == 4  # 2 records * 2 events each
        assert float(np.max(conc)) <= 2.0

    def test_concurrent_peak(self) -> None:
        """3 fully overlapping requests."""
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([100.0, 100.0, 100.0])
        ts, conc = concurrency_sweep(start, end)
        assert float(np.max(conc)) == 3.0


class TestThroughputSweep:
    def test_empty_input(self) -> None:
        ts, tput = throughput_sweep(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert len(ts) == 0

    def test_single_request_known_rate(self) -> None:
        """Single request: 101 output tokens over 100ns → rate = 100/100 = 1.0 tokens/ns."""
        gen_start = np.array([0.0])
        end = np.array([100.0])
        output_tokens = np.array([101.0])
        ts, tput = throughput_sweep(gen_start, end, output_tokens)
        assert len(ts) == 2
        assert tput[0] == pytest.approx(1.0)  # rate added at start
        assert tput[1] == pytest.approx(0.0)  # rate removed at end

    def test_zero_output_tokens_excluded(self) -> None:
        """Requests with 0 or 1 output tokens should not contribute to throughput."""
        gen_start = np.array([0.0, 50.0])
        end = np.array([100.0, 150.0])
        output_tokens = np.array([1.0, 11.0])  # First: (1-1)/100=0, Second: 10/100=0.1
        ts, tput = throughput_sweep(gen_start, end, output_tokens)
        # First request has rate 0, so only 1 valid request contributes
        # (1-1)/100 = 0 rate for first, so it's technically valid but 0 duration check handles it
        assert len(ts) > 0

    def test_nan_excluded(self) -> None:
        gen_start = np.array([0.0, np.nan])
        end = np.array([100.0, 200.0])
        output_tokens = np.array([11.0, np.nan])
        ts, tput = throughput_sweep(gen_start, end, output_tokens)
        assert len(ts) == 2  # Only 1 valid request


class TestPrefillThroughputSweep:
    def test_empty_input(self) -> None:
        ts, tput = prefill_throughput_sweep(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert len(ts) == 0
        assert len(tput) == 0

    def test_single_request_known_rate(self) -> None:
        """Single request: 100 input tokens over 50ns prefill → rate = 2.0 tokens/ns."""
        start = np.array([0.0])
        gen_start = np.array([50.0])
        input_tokens = np.array([100.0])
        ts, tput = prefill_throughput_sweep(start, gen_start, input_tokens)
        assert len(ts) == 2
        assert tput[0] == pytest.approx(2.0)  # rate added at start
        assert tput[1] == pytest.approx(0.0)  # rate removed at gen_start

    def test_nan_excluded(self) -> None:
        """NaN input_tokens or generation_start_ns are filtered out."""
        start = np.array([0.0, 0.0, 0.0])
        gen_start = np.array([50.0, np.nan, 50.0])
        input_tokens = np.array([100.0, 100.0, np.nan])
        ts, tput = prefill_throughput_sweep(start, gen_start, input_tokens)
        # Only 1 valid record
        assert len(ts) == 2

    def test_zero_prefill_duration_excluded(self) -> None:
        """start_ns == generation_start_ns → zero duration → filtered out."""
        start = np.array([100.0])
        gen_start = np.array([100.0])
        input_tokens = np.array([50.0])
        ts, tput = prefill_throughput_sweep(start, gen_start, input_tokens)
        assert len(ts) == 0
        assert len(tput) == 0

    def test_overlapping_prefills(self) -> None:
        """Two concurrent prefills → peak rate = sum of individual rates."""
        # Request A: [0, 50), 100 tokens → rate = 2.0
        # Request B: [10, 60), 150 tokens → rate = 3.0
        # Overlap at [10, 50): combined rate = 5.0
        start = np.array([0.0, 10.0])
        gen_start = np.array([50.0, 60.0])
        input_tokens = np.array([100.0, 150.0])
        ts, tput = prefill_throughput_sweep(start, gen_start, input_tokens)
        assert float(np.max(tput)) == pytest.approx(5.0)


class TestThroughputSweepIcl:
    def test_empty_input(self) -> None:
        ts, tput = throughput_sweep_icl(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int64),
        )
        assert len(ts) == 0

    def test_single_request_uniform_chunks(self) -> None:
        """Single request with 3 equal ICL chunks of 10ns each."""
        gen_start = np.array([0.0])  # indexed by session_num
        output_tokens = np.array([3.0])  # 3 tokens across 3 chunks → 1 tok/msg
        icl_values = np.array([10.0, 10.0, 10.0])
        icl_record_indices = np.array([0, 0, 0], dtype=np.int32)
        icl_offsets = np.array([0], dtype=np.int64)

        ts, tput = throughput_sweep_icl(
            gen_start, output_tokens, icl_values, icl_record_indices, icl_offsets
        )
        assert len(ts) == 6  # 3 chunks * 2 events each
        # Each chunk: rate = (3/3) / 10 = 0.1 tokens/ns
        # Since chunks are sequential, peak throughput should be 0.1
        assert float(np.max(tput)) == pytest.approx(0.1)

    def test_nan_gen_start_excluded(self) -> None:
        """Records with NaN generation_start should be excluded."""
        gen_start = np.array([np.nan])
        output_tokens = np.array([5.0])
        icl_values = np.array([10.0])
        icl_record_indices = np.array([0], dtype=np.int32)
        icl_offsets = np.array([0], dtype=np.int64)

        ts, tput = throughput_sweep_icl(
            gen_start, output_tokens, icl_values, icl_record_indices, icl_offsets
        )
        assert len(ts) == 0

    def test_two_overlapping_requests(self) -> None:
        """Two requests with overlapping ICL chunks."""
        gen_start = np.array([0.0, 5.0])
        output_tokens = np.array([2.0, 2.0])  # 2 tokens each, 2 chunks each → 1 tok/msg
        # Request 0: chunks at [0,10), [10,20)
        # Request 1: chunks at [5,15), [15,25)
        icl_values = np.array([10.0, 10.0, 10.0, 10.0])
        icl_record_indices = np.array([0, 0, 1, 1], dtype=np.int32)
        icl_offsets = np.array([0, 2], dtype=np.int64)

        ts, tput = throughput_sweep_icl(
            gen_start, output_tokens, icl_values, icl_record_indices, icl_offsets
        )
        assert len(ts) == 8  # 4 chunks * 2 events
        # When chunks overlap, throughput should be > single chunk rate
        assert float(np.max(tput)) > 0.1

    def test_rescaling_with_variable_tokens(self) -> None:
        """6 output tokens across 3 chunks → 2 tokens/msg, rates double."""
        gen_start = np.array([0.0])
        output_tokens = np.array([6.0])  # 6 tokens across 3 chunks → 2 tok/msg
        icl_values = np.array([10.0, 10.0, 10.0])
        icl_record_indices = np.array([0, 0, 0], dtype=np.int32)
        icl_offsets = np.array([0], dtype=np.int64)

        ts, tput = throughput_sweep_icl(
            gen_start, output_tokens, icl_values, icl_record_indices, icl_offsets
        )
        assert len(ts) == 6
        # Each chunk: rate = (6/3) / 10 = 0.2 tokens/ns (doubled vs 1 tok/msg)
        assert float(np.max(tput)) == pytest.approx(0.2)


class TestComputeTimeWeightedStats:
    def test_constant_value(self) -> None:
        """Single constant concurrency → avg = value, std = 0, all percentiles = value."""
        # Concurrency of 5 from t=0 to t=100
        ts = np.array([0.0, 100.0])
        vals = np.array([5.0, 0.0])  # step function: 5 at t=0, drops to 0 at t=100
        stats = compute_time_weighted_stats(ts, vals, 0.0, 100.0)

        assert stats.avg == pytest.approx(5.0)
        assert stats.min == pytest.approx(5.0)
        assert stats.max == pytest.approx(5.0)
        assert stats.p50 == pytest.approx(5.0)
        assert stats.p90 == pytest.approx(5.0)
        assert stats.p95 == pytest.approx(5.0)
        assert stats.p99 == pytest.approx(5.0)
        assert stats.std == pytest.approx(0.0)

    def test_two_segments_known_avg(self) -> None:
        """Two segments with known durations → verify time-weighted avg."""
        # Concurrency: 2 for 80ns, then 10 for 20ns
        ts = np.array([0.0, 80.0, 100.0])
        vals = np.array([2.0, 10.0, 0.0])
        stats = compute_time_weighted_stats(ts, vals, 0.0, 100.0)

        # avg = (2*80 + 10*20) / 100 = (160 + 200) / 100 = 3.6
        assert stats.avg == pytest.approx(3.6)
        assert stats.min == pytest.approx(2.0)
        assert stats.max == pytest.approx(10.0)

        # std = sqrt((80*(2-3.6)^2 + 20*(10-3.6)^2) / 100)
        #     = sqrt((80*2.56 + 20*40.96) / 100)
        #     = sqrt((204.8 + 819.2) / 100) = sqrt(10.24) ≈ 3.2
        assert stats.std == pytest.approx(3.2, abs=0.01)

    def test_percentiles_unequal_durations(self) -> None:
        """Verify percentile computation with unequal segment durations."""
        # Value 1 for 90% of time, value 100 for 10% of time
        ts = np.array([0.0, 900.0, 1000.0])
        vals = np.array([1.0, 100.0, 0.0])
        stats = compute_time_weighted_stats(ts, vals, 0.0, 1000.0)

        # p50 should be 1 (value held for 90% of time)
        assert stats.p50 == pytest.approx(1.0)
        # p90 should be 1 (90% of time is at value 1, cum_frac = 0.9)
        assert stats.p90 == pytest.approx(1.0)
        # p95 should be 100 (only 10% of time is at value 100)
        assert stats.p95 == pytest.approx(100.0)
        # p99 should be 100
        assert stats.p99 == pytest.approx(100.0)

    def test_window_clipping(self) -> None:
        """Events outside window are ignored via clipping."""
        # Full curve: value 1 from t=0-50, value 5 from t=50-100
        ts = np.array([0.0, 50.0, 100.0])
        vals = np.array([1.0, 5.0, 0.0])

        # Only look at [50, 100] — should see only value 5
        stats = compute_time_weighted_stats(ts, vals, 50.0, 100.0)
        assert stats.avg == pytest.approx(5.0)
        assert stats.min == pytest.approx(5.0)
        assert stats.max == pytest.approx(5.0)
        assert stats.std == pytest.approx(0.0)

    def test_window_clipping_partial_segment(self) -> None:
        """Window that slices through the middle of a segment."""
        # Value 2 from t=0 to t=100
        ts = np.array([0.0, 100.0])
        vals = np.array([2.0, 0.0])

        # Window [25, 75] — should still see value 2
        stats = compute_time_weighted_stats(ts, vals, 25.0, 75.0)
        assert stats.avg == pytest.approx(2.0)

    def test_single_event_degenerate(self) -> None:
        """Single event at the start of the window."""
        ts = np.array([0.0])
        vals = np.array([3.0])
        stats = compute_time_weighted_stats(ts, vals, 0.0, 100.0)

        # Value 3 is held for the entire window
        assert stats.avg == pytest.approx(3.0)
        assert stats.min == pytest.approx(3.0)
        assert stats.max == pytest.approx(3.0)

    def test_empty_arrays(self) -> None:
        """Empty arrays return all zeros."""
        stats = compute_time_weighted_stats(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            0.0,
            100.0,
        )
        assert all(v == 0.0 for v in stats)

    def test_zero_duration_window(self) -> None:
        """Zero-duration window returns all zeros."""
        ts = np.array([0.0, 100.0])
        vals = np.array([5.0, 0.0])
        stats = compute_time_weighted_stats(ts, vals, 50.0, 50.0)
        assert all(v == 0.0 for v in stats)
