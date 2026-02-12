# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vectorized sweep-line algorithms for concurrency and throughput curves.

All functions operate on numpy arrays — no record objects, no Python loops.
Input arrays are expected to be session_num-indexed (from ColumnStore).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
) -> dict[str, float]:
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
        Dict with keys: avg, min, max, p50, p90, p95, p99, std.
    """
    total_dur = window_end - window_start
    zero_stats = {
        k: 0.0 for k in ("avg", "min", "max", "p50", "p90", "p95", "p99", "std")
    }
    if len(sorted_ts) == 0 or total_dur <= 0:
        return zero_stats

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
        return zero_stats

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

    return {
        "avg": avg,
        "min": mn,
        "max": mx,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "std": std,
    }
