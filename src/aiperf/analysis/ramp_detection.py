# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ramp boundary detection from concurrency/throughput curves."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def detect_steady_state_window(
    sorted_ts: NDArray[np.float64],
    concurrency: NDArray[np.float64],
    stability_fraction: float = 0.90,
    sustained_window_pct: float = 5.0,
    min_window_pct: float = 10.0,
) -> tuple[float, float]:
    """Detect ramp-up end and ramp-down start from a concurrency curve.

    Args:
        sorted_ts: Sorted event timestamps from concurrency_sweep().
        concurrency: Concurrency values at each event boundary.
        stability_fraction: Fraction of target concurrency for "at steady state".
        sustained_window_pct: Min sustained duration as % of total for ramp detection.
        min_window_pct: Min window size as % of total; below this, fall back to full.

    Returns:
        (window_start, window_end) — wall-clock timestamps.
        Falls back to (min_ts, max_ts) if steady state cannot be detected.
    """
    if len(sorted_ts) == 0:
        return 0.0, 0.0

    min_ts = float(sorted_ts[0])
    max_ts = float(sorted_ts[-1])
    total_duration = max_ts - min_ts

    if total_duration <= 0:
        return min_ts, max_ts

    # Target concurrency from the 95th percentile of the curve
    target = float(np.percentile(concurrency, 95))
    threshold = target * stability_fraction

    sustained_window_ns = total_duration * sustained_window_pct / 100.0

    # Find contiguous runs where concurrency >= threshold
    above = (concurrency >= threshold).astype(np.int8)
    padded = np.empty(len(above) + 2, dtype=np.int8)
    padded[0] = 0
    padded[1:-1] = above
    padded[-1] = 0
    edges = np.diff(padded)

    run_starts = np.where(edges == 1)[0]  # original indices where runs begin
    run_ends = np.where(edges == -1)[0]  # one-past-end original indices

    if len(run_starts) == 0:
        # No concurrency above threshold — fall back to full range
        return min_ts, max_ts

    n = len(sorted_ts)
    run_start_ts = sorted_ts[run_starts]

    # Duration check: time from run start to the first below-threshold event.
    # Runs extending to the array end are always sustained (use inf).
    run_check_ts = np.where(run_ends < n, sorted_ts[run_ends], np.inf)
    qualifying = (run_check_ts - run_start_ts) >= sustained_window_ns

    if not qualifying.any():
        window_start = min_ts
        window_end = max_ts
    else:
        qual_idx = np.where(qualifying)[0]
        # Forward: first qualifying run's start
        window_start = float(run_start_ts[qual_idx[0]])
        # Backward: last above-threshold timestamp in the last qualifying run
        window_end = float(sorted_ts[run_ends[qual_idx[-1]] - 1])

    # Validate window size
    window_duration = window_end - window_start
    min_window_ns = total_duration * min_window_pct / 100.0

    if window_duration < min_window_ns:
        logger.warning(
            "Steady-state window (%.1f%% of total) below minimum (%.1f%%), "
            "falling back to full range",
            window_duration / total_duration * 100,
            min_window_pct,
        )
        return min_ts, max_ts

    return window_start, window_end


def manual_steady_state_window(
    min_ts: float,
    max_ts: float,
    start_pct: float,
    end_pct: float,
) -> tuple[float, float]:
    """Compute steady-state window from user-specified percentages.

    Args:
        min_ts: Minimum timestamp of the benchmark.
        max_ts: Maximum timestamp of the benchmark.
        start_pct: Start of window as % of total duration.
        end_pct: End of window as % of total duration.

    Returns:
        (window_start, window_end) — wall-clock timestamps.
    """
    duration = max_ts - min_ts
    return (min_ts + duration * start_pct / 100.0, min_ts + duration * end_pct / 100.0)
