# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vectorized sweep-line algorithms for concurrency and throughput curves.

All functions operate on numpy arrays — no record objects, no Python loops.
Input arrays are expected to be session_num-indexed (from ColumnStore).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import MetricResult


class SweepStats(NamedTuple):
    """Time-weighted statistics from a sweep-line step function."""

    avg: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float
    std: float


ZERO_SWEEP_STATS = SweepStats(
    avg=0.0, min=0.0, max=0.0, p50=0.0, p90=0.0, p95=0.0, p99=0.0, std=0.0
)


class SweepMetricSpec(NamedTuple):
    """Specification for a sweep-line metric (tag, header, unit, scale)."""

    tag: str
    header: str
    unit: str
    scale: float


SWEEP_METRIC_SPECS: tuple[SweepMetricSpec, ...] = (
    SweepMetricSpec("effective_concurrency", "Effective Concurrency", "requests", 1.0),
    SweepMetricSpec(
        "effective_throughput", "Effective Throughput", "tokens/sec", NANOS_PER_SECOND
    ),
    SweepMetricSpec(
        "effective_prefill_throughput",
        "Effective Prefill Throughput",
        "tokens/sec",
        NANOS_PER_SECOND,
    ),
    SweepMetricSpec(
        "effective_generation_concurrency",
        "Effective Generation Concurrency",
        "requests",
        1.0,
    ),
    SweepMetricSpec(
        "effective_prefill_concurrency",
        "Effective Prefill Concurrency",
        "requests",
        1.0,
    ),
    SweepMetricSpec(
        "effective_total_throughput",
        "Effective Total Throughput",
        "tokens/sec",
        NANOS_PER_SECOND,
    ),
    SweepMetricSpec(
        "effective_throughput_per_user",
        "Effective Throughput Per User",
        "tokens/sec/user",
        NANOS_PER_SECOND,
    ),
    SweepMetricSpec(
        "effective_prefill_throughput_per_user",
        "Effective Prefill Throughput Per User",
        "tokens/sec/user",
        NANOS_PER_SECOND,
    ),
    SweepMetricSpec(
        "tokens_in_flight",
        "Tokens In Flight",
        "tokens",
        1.0,
    ),
)


@dataclass(frozen=True, slots=True)
class SweepCurves:
    """Pre-computed sweep-line curves for concurrency, throughput, and prefill throughput."""

    concurrency_ts: NDArray[np.float64]
    concurrency: NDArray[np.float64]
    throughput_ts: NDArray[np.float64]
    throughput: NDArray[np.float64]
    prefill_throughput_ts: NDArray[np.float64]
    prefill_throughput: NDArray[np.float64]
    generation_concurrency_ts: NDArray[np.float64]
    generation_concurrency: NDArray[np.float64]
    prefill_concurrency_ts: NDArray[np.float64]
    prefill_concurrency: NDArray[np.float64]
    total_throughput_ts: NDArray[np.float64]
    total_throughput: NDArray[np.float64]
    throughput_per_user_ts: NDArray[np.float64]
    throughput_per_user: NDArray[np.float64]
    prefill_throughput_per_user_ts: NDArray[np.float64]
    prefill_throughput_per_user: NDArray[np.float64]
    tokens_in_flight_ts: NDArray[np.float64]
    tokens_in_flight: NDArray[np.float64]

    def curves(
        self,
    ) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...]:
        """Return (ts, values) pairs in SWEEP_METRIC_SPECS order."""
        return (
            (self.concurrency_ts, self.concurrency),
            (self.throughput_ts, self.throughput),
            (self.prefill_throughput_ts, self.prefill_throughput),
            (self.generation_concurrency_ts, self.generation_concurrency),
            (self.prefill_concurrency_ts, self.prefill_concurrency),
            (self.total_throughput_ts, self.total_throughput),
            (self.throughput_per_user_ts, self.throughput_per_user),
            (self.prefill_throughput_per_user_ts, self.prefill_throughput_per_user),
            (self.tokens_in_flight_ts, self.tokens_in_flight),
        )

    def compute_metrics(
        self, window_start: float, window_end: float
    ) -> dict[str, MetricResult]:
        """Compute all sweep MetricResults for a time window."""
        results: dict[str, MetricResult] = {}
        for spec, (ts, values) in zip(SWEEP_METRIC_SPECS, self.curves(), strict=True):
            stats = compute_time_weighted_stats(ts, values, window_start, window_end)
            results[spec.tag] = metric_result_from_sweep_stats(
                spec.tag, spec.header, spec.unit, stats, scale=spec.scale
            )
        return results


def _sweep_cumsum(
    timestamps: NDArray[np.float64],
    deltas: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sort events by timestamp (ends before starts at ties) and cumsum deltas."""
    # lexsort: primary key = timestamps, secondary key = event_type (0=end, 1=start).
    # Ends sort before starts at the same timestamp.
    event_type = (deltas > 0).astype(np.int8)
    order = np.lexsort((event_type, timestamps))
    return timestamps[order], np.cumsum(deltas[order])


def _step_lookup(
    event_ts: NDArray[np.float64],
    event_vals: NDArray[np.float64],
    query_ts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Look up step-function values at query timestamps (0 before first event)."""
    idx = np.searchsorted(event_ts, query_ts, side="right").astype(np.intp) - 1
    return np.where(idx >= 0, event_vals[np.clip(idx, 0, len(event_vals) - 1)], 0.0)


def add_step_functions(
    a_ts: NDArray[np.float64],
    a_vals: NDArray[np.float64],
    b_ts: NDArray[np.float64],
    b_vals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Add two step functions, returning a new step function on merged timestamps.

    Args:
        a_ts: Sorted timestamps of the first step function.
        a_vals: Values of the first step function.
        b_ts: Sorted timestamps of the second step function.
        b_vals: Values of the second step function.

    Returns:
        Tuple of (merged_timestamps, sum_values).
    """
    if len(a_ts) == 0:
        return b_ts.copy(), b_vals.copy()
    if len(b_ts) == 0:
        return a_ts.copy(), a_vals.copy()

    merged_ts = np.unique(np.concatenate([a_ts, b_ts]))
    return merged_ts, _step_lookup(a_ts, a_vals, merged_ts) + _step_lookup(
        b_ts, b_vals, merged_ts
    )


def divide_step_functions(
    num_ts: NDArray[np.float64],
    num_vals: NDArray[np.float64],
    den_ts: NDArray[np.float64],
    den_vals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Divide two step functions, returning a new step function on merged timestamps.

    Where denominator is zero the result is zero (safe division).

    Args:
        num_ts: Sorted timestamps of the numerator step function.
        num_vals: Values of the numerator step function.
        den_ts: Sorted timestamps of the denominator step function.
        den_vals: Values of the denominator step function.

    Returns:
        Tuple of (merged_timestamps, quotient_values).
    """
    if len(num_ts) == 0 or len(den_ts) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    merged_ts = np.unique(np.concatenate([num_ts, den_ts]))
    num_at = _step_lookup(num_ts, num_vals, merged_ts)
    den_at = _step_lookup(den_ts, den_vals, merged_ts)

    result = np.zeros_like(num_at)
    np.divide(num_at, den_at, out=result, where=den_at > 0)
    return merged_ts, result


def throughput_per_user_sweep(
    generation_start_ns: NDArray[np.float64],
    end_ns: NDArray[np.float64],
    tput_ts: NDArray[np.float64],
    tput_vals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-user throughput by dividing aggregate throughput by generation-phase concurrency.

    Args:
        generation_start_ns: First-token wall-clock timestamps. NaN for missing.
        end_ns: Request end timestamps. NaN for missing.
        tput_ts: Sorted timestamps from throughput_sweep (or ICL variant).
        tput_vals: Throughput values (tokens/ns) at each timestamp.

    Returns:
        Tuple of (timestamps, per_user_throughput) in tokens/ns/user.
    """
    conc_ts, conc_vals = concurrency_sweep(generation_start_ns, end_ns)
    return divide_step_functions(tput_ts, tput_vals, conc_ts, conc_vals)


def prefill_throughput_per_user_sweep(
    start_ns: NDArray[np.float64],
    generation_start_ns: NDArray[np.float64],
    ptput_ts: NDArray[np.float64],
    ptput_vals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-user prefill throughput by dividing aggregate prefill throughput by prefill-phase concurrency.

    Args:
        start_ns: Request start timestamps. NaN for missing.
        generation_start_ns: First-token wall-clock timestamps. NaN for missing.
        ptput_ts: Sorted timestamps from prefill_throughput_sweep.
        ptput_vals: Prefill throughput values (tokens/ns) at each timestamp.

    Returns:
        Tuple of (timestamps, per_user_prefill_throughput) in tokens/ns/user.
    """
    conc_ts, conc_vals = concurrency_sweep(start_ns, generation_start_ns)
    return divide_step_functions(ptput_ts, ptput_vals, conc_ts, conc_vals)


def concurrency_sweep(
    start_ns: NDArray[np.float64],
    end_ns: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute exact instantaneous concurrency at every event boundary.

    Args:
        start_ns: Request start timestamps (wall-clock). NaN for missing records.
        end_ns: Request end timestamps (wall-clock). NaN for missing records.

    Returns:
        Tuple of (sorted_timestamps, concurrency_values).
        sorted_timestamps has shape (2K,), concurrency_values has shape (2K,),
        where K is the number of valid (non-NaN) records.
    """
    valid = ~np.isnan(start_ns) & ~np.isnan(end_ns)
    k = int(valid.sum())
    if k == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    timestamps = np.concatenate([start_ns[valid], end_ns[valid]])
    deltas = np.concatenate(
        [np.ones(k, dtype=np.float64), -np.ones(k, dtype=np.float64)]
    )

    sorted_ts, concurrency = _sweep_cumsum(timestamps, deltas)
    return sorted_ts, concurrency


def throughput_sweep(
    generation_start_ns: NDArray[np.float64],
    end_ns: NDArray[np.float64],
    output_tokens: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute exact instantaneous throughput (tokens/ns) at every event boundary.

    Uses uniform per-request rate: (output_tokens - 1) / generation_duration.

    Args:
        generation_start_ns: First-token wall-clock timestamps. NaN for missing.
        end_ns: Request end timestamps. NaN for missing.
        output_tokens: Output token counts. NaN for missing.

    Returns:
        Tuple of (sorted_timestamps, throughput_values) in tokens/ns.
    """
    gen_dur = end_ns - generation_start_ns
    valid = ~np.isnan(generation_start_ns) & ~np.isnan(output_tokens) & (gen_dur > 0)
    k = int(valid.sum())
    if k == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    rates = (output_tokens[valid] - 1.0) / gen_dur[valid]

    timestamps = np.concatenate([generation_start_ns[valid], end_ns[valid]])
    deltas = np.concatenate([rates, -rates])

    sorted_ts, throughput = _sweep_cumsum(timestamps, deltas)
    return sorted_ts, throughput


def prefill_throughput_sweep(
    start_ns: NDArray[np.float64],
    generation_start_ns: NDArray[np.float64],
    input_tokens: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute exact instantaneous prefill throughput (tokens/ns) at every event boundary.

    During prefill [start_ns, generation_start_ns), the model processes
    input_tokens tokens. The per-request prefill rate is
    input_tokens / prefill_duration.

    Args:
        start_ns: Request start timestamps (wall-clock). NaN for missing.
        generation_start_ns: First-token wall-clock timestamps. NaN for missing.
        input_tokens: Input token counts. NaN for missing.

    Returns:
        Tuple of (sorted_timestamps, prefill_throughput_values) in tokens/ns.
    """
    prefill_dur = generation_start_ns - start_ns
    valid = (
        ~np.isnan(start_ns)
        & ~np.isnan(generation_start_ns)
        & ~np.isnan(input_tokens)
        & (prefill_dur > 0)
    )
    k = int(valid.sum())
    if k == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    rates = input_tokens[valid] / prefill_dur[valid]

    timestamps = np.concatenate([start_ns[valid], generation_start_ns[valid]])
    deltas = np.concatenate([rates, -rates])

    sorted_ts, prefill_tput = _sweep_cumsum(timestamps, deltas)
    return sorted_ts, prefill_tput


def total_throughput_sweep(
    start_ns: NDArray[np.float64],
    generation_start_ns: NDArray[np.float64],
    end_ns: NDArray[np.float64],
    input_tokens: NDArray[np.float64],
    output_tokens: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute total throughput (prefill + generation) in a single sweep pass.

    Combines prefill rate events [start_ns, generation_start_ns) and generation
    rate events [generation_start_ns, end_ns) into one sweep, avoiding the
    overhead of two separate sweeps + grid merge + searchsorted lookups.

    Args:
        start_ns: Request start timestamps. NaN for missing.
        generation_start_ns: First-token wall-clock timestamps. NaN for missing.
        end_ns: Request end timestamps. NaN for missing.
        input_tokens: Input token counts. NaN for missing.
        output_tokens: Output token counts. NaN for missing.

    Returns:
        Tuple of (sorted_timestamps, total_throughput_values) in tokens/ns.
    """
    # Prefill: input_tokens / prefill_duration during [start, gen_start)
    prefill_dur = generation_start_ns - start_ns
    pf_valid = (
        ~np.isnan(start_ns)
        & ~np.isnan(generation_start_ns)
        & ~np.isnan(input_tokens)
        & (prefill_dur > 0)
    )
    pf_k = int(pf_valid.sum())

    # Generation: (output_tokens - 1) / gen_duration during [gen_start, end)
    gen_dur = end_ns - generation_start_ns
    gn_valid = ~np.isnan(generation_start_ns) & ~np.isnan(output_tokens) & (gen_dur > 0)
    gn_k = int(gn_valid.sum())

    if pf_k == 0 and gn_k == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    parts_ts: list[NDArray[np.float64]] = []
    parts_delta: list[NDArray[np.float64]] = []

    if pf_k > 0:
        pf_rates = input_tokens[pf_valid] / prefill_dur[pf_valid]
        parts_ts.extend([start_ns[pf_valid], generation_start_ns[pf_valid]])
        parts_delta.extend([pf_rates, -pf_rates])

    if gn_k > 0:
        gn_rates = (output_tokens[gn_valid] - 1.0) / gen_dur[gn_valid]
        parts_ts.extend([generation_start_ns[gn_valid], end_ns[gn_valid]])
        parts_delta.extend([gn_rates, -gn_rates])

    return _sweep_cumsum(np.concatenate(parts_ts), np.concatenate(parts_delta))


def tokens_in_flight_sweep(
    start_ns: NDArray[np.float64],
    generation_start_ns: NDArray[np.float64],
    end_ns: NDArray[np.float64],
    input_tokens: NDArray[np.float64],
    output_tokens: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute instantaneous KV cache token load at every event boundary.

    Models the total tokens held in server memory (KV cache) per request:
    - During prefill [start_ns, generation_start_ns): input_tokens
    - During generation [generation_start_ns, end_ns): input_tokens + output_tokens

    Input tokens stay in the KV cache throughout the request lifetime, and
    output tokens accumulate on top during generation. This reveals GPU
    memory pressure — two concurrent 4K-token requests look identical to two
    128-token requests in concurrency but wildly different here.

    Events per request (up to 3):
      +input_tokens   at start_ns
      +output_tokens  at generation_start_ns
      -(input_tokens + output_tokens) at end_ns

    Args:
        start_ns: Request start timestamps. NaN for missing.
        generation_start_ns: First-token wall-clock timestamps. NaN for missing.
        end_ns: Request end timestamps. NaN for missing.
        input_tokens: Input token counts. NaN for missing.
        output_tokens: Output token counts. NaN for missing.

    Returns:
        Tuple of (sorted_timestamps, tokens_in_flight) in tokens.
    """
    # We need per-request validity for each event type:
    # 1. Prefill start (+input_tokens at start_ns): need valid start + input_tokens
    # 2. Gen start (+output_tokens at gen_start): need valid gen_start + output_tokens
    # 3. Request end (-(input+output) at end_ns): need valid end + knowledge of what was added

    # Requests with valid prefill: contribute +input_tokens at start_ns
    has_start = ~np.isnan(start_ns) & ~np.isnan(input_tokens)
    # Requests with valid generation: contribute +output_tokens at generation_start_ns
    gen_dur = end_ns - generation_start_ns
    has_gen = ~np.isnan(generation_start_ns) & ~np.isnan(output_tokens) & (gen_dur > 0)
    # Requests with valid end: release all accumulated tokens at end_ns
    has_end = ~np.isnan(end_ns)

    # A request contributes input_tokens if it has a valid start
    # and output_tokens if it has a valid generation phase.
    # At end_ns, we subtract whatever was added.
    # For simplicity and correctness, handle the three event types independently.

    parts_ts: list[NDArray[np.float64]] = []
    parts_delta: list[NDArray[np.float64]] = []

    # Event 1: +input_tokens at start_ns (prefill begins, tokens enter KV cache)
    pf_valid = (
        has_start & ~np.isnan(generation_start_ns) & (generation_start_ns > start_ns)
    )
    if pf_valid.any():
        parts_ts.append(start_ns[pf_valid])
        parts_delta.append(input_tokens[pf_valid])

    # Event 2: +output_tokens at generation_start_ns (generation begins, output tokens join KV cache)
    if has_gen.any():
        parts_ts.append(generation_start_ns[has_gen])
        parts_delta.append(output_tokens[has_gen])

    # Event 3: -(input_tokens + output_tokens) at end_ns (request completes, KV cache freed)
    # Only subtract tokens that were actually added
    end_with_input = pf_valid & has_end
    end_with_gen = has_gen & has_end

    # Combine: for requests that have both input and gen, subtract both at end
    both = end_with_input & end_with_gen
    input_only = end_with_input & ~end_with_gen
    gen_only = end_with_gen & ~end_with_input

    if both.any():
        parts_ts.append(end_ns[both])
        parts_delta.append(-(input_tokens[both] + output_tokens[both]))
    if input_only.any():
        parts_ts.append(end_ns[input_only])
        parts_delta.append(-input_tokens[input_only])
    if gen_only.any():
        parts_ts.append(end_ns[gen_only])
        parts_delta.append(-output_tokens[gen_only])

    if len(parts_ts) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    return _sweep_cumsum(np.concatenate(parts_ts), np.concatenate(parts_delta))


def tokens_in_flight_sweep_icl(
    start_ns: NDArray[np.float64],
    generation_start_ns: NDArray[np.float64],
    end_ns: NDArray[np.float64],
    input_tokens: NDArray[np.float64],
    output_tokens: NDArray[np.float64],
    icl_values: NDArray[np.float64],
    icl_record_indices: NDArray[np.int32],
    icl_offsets: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """ICL-aware tokens in flight: output tokens ramp up at chunk boundaries.

    Instead of adding all output_tokens at generation_start_ns, this function
    adds tokens_per_chunk at each SSE chunk boundary during generation,
    modeling the gradual KV cache growth as tokens are generated.

    Events:
      +input_tokens                     at start_ns        (prefill loads KV cache)
      +tokens_per_chunk                 at each chunk end  (output tokens accumulate)
      -(input_tokens + output_tokens)   at end_ns          (KV cache freed)

    Args:
        start_ns: Request start timestamps. NaN for missing.
        generation_start_ns: First-token wall-clock timestamps. NaN for missing.
        end_ns: Request end timestamps. NaN for missing.
        input_tokens: Input token counts. NaN for missing.
        output_tokens: Output token counts. NaN for missing.
        icl_values: Flat array of all ICL durations (M values).
        icl_record_indices: Session_num per ICL value (M values).
        icl_offsets: Per-session_num start offset into icl_values.

    Returns:
        Tuple of (sorted_timestamps, tokens_in_flight) in tokens.
    """
    if len(icl_values) == 0:
        return tokens_in_flight_sweep(
            start_ns, generation_start_ns, end_ns, input_tokens, output_tokens
        )

    rec_idx = icl_record_indices

    # --- ICL chunk events: +tokens_per_chunk at each chunk boundary ---
    # Wall-clock chunk boundaries (same computation as throughput_sweep_icl)
    global_cs = np.cumsum(icl_values)
    request_offsets = icl_offsets[rec_idx]
    start_cs = np.where(request_offsets > 0, global_cs[request_offsets - 1], 0.0)
    relative_cs = global_cs - start_cs

    gen_start = generation_start_ns[rec_idx]
    interval_end = gen_start + relative_cs

    # Per-request ICL count → tokens per chunk
    icl_counts = np.bincount(rec_idx, minlength=len(output_tokens)).astype(np.float64)
    per_req_tokens = output_tokens[rec_idx]
    per_req_icl_count = icl_counts[rec_idx]
    tokens_per_chunk = np.where(
        per_req_icl_count > 0, per_req_tokens / per_req_icl_count, 0.0
    )

    # Valid chunks: non-NaN gen_start, positive ICL, non-NaN output_tokens
    chunk_valid = ~np.isnan(gen_start) & (icl_values > 0) & ~np.isnan(per_req_tokens)

    parts_ts: list[NDArray[np.float64]] = []
    parts_delta: list[NDArray[np.float64]] = []

    # Chunk events: +tokens_per_chunk at each interval_end
    if chunk_valid.any():
        parts_ts.append(interval_end[chunk_valid])
        parts_delta.append(tokens_per_chunk[chunk_valid])

    # --- Prefill events: +input_tokens at start_ns ---
    has_start = ~np.isnan(start_ns) & ~np.isnan(input_tokens)
    pf_valid = (
        has_start & ~np.isnan(generation_start_ns) & (generation_start_ns > start_ns)
    )
    if pf_valid.any():
        parts_ts.append(start_ns[pf_valid])
        parts_delta.append(input_tokens[pf_valid])

    # --- End events: subtract all accumulated tokens at end_ns ---
    has_end = ~np.isnan(end_ns)
    # Requests that have ICL data — their output tokens were added chunk-by-chunk
    has_icl = icl_counts > 0
    end_with_input_and_icl = pf_valid & has_end & has_icl
    end_with_input_only = pf_valid & has_end & ~has_icl
    end_with_icl_only = ~pf_valid & has_end & has_icl

    if end_with_input_and_icl.any():
        parts_ts.append(end_ns[end_with_input_and_icl])
        parts_delta.append(
            -(
                input_tokens[end_with_input_and_icl]
                + output_tokens[end_with_input_and_icl]
            )
        )
    if end_with_input_only.any():
        parts_ts.append(end_ns[end_with_input_only])
        parts_delta.append(-input_tokens[end_with_input_only])
    if end_with_icl_only.any():
        parts_ts.append(end_ns[end_with_icl_only])
        parts_delta.append(-output_tokens[end_with_icl_only])

    if len(parts_ts) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    return _sweep_cumsum(np.concatenate(parts_ts), np.concatenate(parts_delta))


def throughput_sweep_icl(
    generation_start_ns: NDArray[np.float64],
    output_tokens: NDArray[np.float64],
    icl_values: NDArray[np.float64],
    icl_record_indices: NDArray[np.int32],
    icl_offsets: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute ICL-aware instantaneous throughput at every chunk boundary.

    Uses per-request rescaled rates: each ICL interval carries
    ``output_tokens / n_icl_intervals`` tokens instead of exactly 1.
    This preserves the accurate temporal shape from SSE message boundaries
    while matching the known total token count per request.

    Args:
        generation_start_ns: Per-record first-token wall-clock (indexed by session_num).
        output_tokens: Per-record output token count (indexed by session_num).
        icl_values: Flat array of all ICL durations (M values).
        icl_record_indices: Session_num per ICL value (M values).
        icl_offsets: Per-session_num start offset into icl_values.

    Returns:
        Tuple of (sorted_timestamps, throughput_values) in tokens/ns.
        Has 2M events (one +rate and one -rate per chunk interval).
    """
    if len(icl_values) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    rec_idx = icl_record_indices

    # Per-request cumulative ICL — vectorized grouped cumsum
    global_cs = np.cumsum(icl_values)
    request_offsets = icl_offsets[rec_idx]
    start_cs = np.where(request_offsets > 0, global_cs[request_offsets - 1], 0.0)
    relative_cs = global_cs - start_cs

    # Wall-clock chunk boundaries
    gen_start = generation_start_ns[rec_idx]
    interval_end = gen_start + relative_cs
    interval_start = interval_end - icl_values

    # Per-request ICL interval count, then per-interval tokens-per-message
    icl_counts = np.bincount(rec_idx, minlength=len(output_tokens)).astype(np.float64)
    per_req_tokens = output_tokens[rec_idx]
    per_req_icl_count = icl_counts[rec_idx]
    tokens_per_msg = np.where(
        per_req_icl_count > 0, per_req_tokens / per_req_icl_count, 0.0
    )
    rates = tokens_per_msg / icl_values

    # Filter out invalid (NaN gen_start, zero ICL, NaN output_tokens)
    valid = ~np.isnan(gen_start) & (icl_values > 0) & ~np.isnan(per_req_tokens)
    if not valid.any():
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    timestamps = np.concatenate([interval_start[valid], interval_end[valid]])
    deltas = np.concatenate([rates[valid], -rates[valid]])

    sorted_ts, throughput = _sweep_cumsum(timestamps, deltas)
    return sorted_ts, throughput


def compute_time_weighted_stats(
    sorted_ts: NDArray[np.float64],
    values: NDArray[np.float64],
    window_start: float,
    window_end: float,
) -> SweepStats:
    """Compute time-weighted statistics over a step-function within a window.

    The sweep-line output defines a step function: value[i] is held from
    sorted_ts[i] to sorted_ts[i+1]. This function clips the step function
    to [window_start, window_end] and computes time-weighted stats.

    Args:
        sorted_ts: Sorted event timestamps from a sweep function.
        values: Step-function values at each timestamp.
        window_start: Left boundary of the analysis window.
        window_end: Right boundary of the analysis window.

    Returns:
        SweepStats with avg, min, max, p50, p90, p95, p99, std.
    """
    total_dur = window_end - window_start
    if len(sorted_ts) == 0 or total_dur <= 0:
        return ZERO_SWEEP_STATS

    # Narrow to events relevant to [window_start, window_end] via searchsorted,
    # avoiding full-array operations on events entirely outside the window.
    lo = max(0, int(np.searchsorted(sorted_ts, window_start, side="right")) - 1)
    hi = min(
        len(sorted_ts), int(np.searchsorted(sorted_ts, window_end, side="left")) + 1
    )
    ts_slice = sorted_ts[lo:hi]
    val_slice = values[lo:hi]

    # Build segments clipped to [window_start, window_end].
    # Each segment i spans [sorted_ts[i], sorted_ts[i+1]) with value values[i].
    n_s = len(ts_slice)
    seg_starts = np.empty(n_s + 1, dtype=np.float64)
    seg_values = np.empty(n_s + 1, dtype=np.float64)

    # Segment before the first event in slice: value is from preceding event (or 0)
    seg_starts[0] = window_start
    seg_values[0] = float(values[lo - 1]) if lo > 0 else 0.0

    # Segments from events
    seg_starts[1:] = ts_slice
    seg_values[1:] = val_slice

    # Segment end boundaries (next start, capped at window_end)
    seg_ends = np.empty(n_s + 1, dtype=np.float64)
    seg_ends[:-1] = seg_starts[1:]
    seg_ends[-1] = window_end

    # Clip to window
    seg_starts = np.maximum(seg_starts, window_start)
    seg_ends = np.minimum(seg_ends, window_end)
    durations = np.maximum(seg_ends - seg_starts, 0.0)

    # Filter to segments with positive duration
    mask = durations > 0
    if not mask.any():
        return ZERO_SWEEP_STATS

    dur = durations[mask]
    val = seg_values[mask]

    # Time-weighted average
    avg = float(np.sum(val * dur) / total_dur)

    # Min / Max
    mn = float(np.min(val))
    mx = float(np.max(val))

    # Time-weighted standard deviation
    std = float(np.sqrt(np.sum(dur * (val - avg) ** 2) / total_dur))

    # Duration-weighted percentiles: sort by value, build CDF, single batched lookup
    order = np.argsort(val)
    sorted_val = val[order]
    sorted_dur = dur[order]
    cum_dur = np.cumsum(sorted_dur)
    cum_frac = cum_dur / cum_dur[-1]

    indices = np.searchsorted(cum_frac, [0.50, 0.90, 0.95, 0.99])
    np.minimum(indices, len(sorted_val) - 1, out=indices)
    p50, p90, p95, p99 = sorted_val[indices].tolist()

    return SweepStats(
        avg=avg, min=mn, max=mx, p50=p50, p90=p90, p95=p95, p99=p99, std=std
    )


def metric_result_from_sweep_stats(
    tag: str,
    header: str,
    unit: str,
    stats: SweepStats,
    scale: float = 1.0,
) -> MetricResult:
    """Build a MetricResult from compute_time_weighted_stats output."""
    return MetricResult(
        tag=tag,
        header=header,
        unit=unit,
        avg=stats.avg * scale,
        min=stats.min * scale,
        max=stats.max * scale,
        p50=stats.p50 * scale,
        p90=stats.p90 * scale,
        p95=stats.p95 * scale,
        p99=stats.p99 * scale,
        std=stats.std * scale,
    )
