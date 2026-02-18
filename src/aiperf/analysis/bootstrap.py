# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bootstrap confidence intervals for steady-state boundary estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aiperf.analysis.ramp_detection import detect_steady_state_window
from aiperf.analysis.sweepline import concurrency_sweep_line, throughput_sweep_line


@dataclass(frozen=True)
class BootstrapResult:
    """Confidence intervals from bootstrap resampling of detection."""

    ci_ramp_up_ns: tuple[float, float]
    ci_ramp_down_ns: tuple[float, float]
    ci_mean_latency: tuple[float, float]
    ci_p99_latency: tuple[float, float]
    n_iterations: int


def _circular_block_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Circular block bootstrap: sample blocks of consecutive indices.

    Preserves temporal adjacency within each block. Used with a
    time-sorted index to maintain local correlation structure.
    """
    n_blocks = -(-n // block_size)  # ceil division
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_size)
    return ((starts[:, np.newaxis] + offsets) % n).ravel()[:n]


def bootstrap_detection(
    start_ns: NDArray[np.float64],
    end_ns: NDArray[np.float64],
    latency: NDArray[np.float64],
    ttft: NDArray[np.float64],
    n_iterations: int = 500,
    confidence: float = 0.95,
    min_window_pct: float = 5.0,
    rng: np.random.Generator | None = None,
    generation_start_ns: NDArray[np.float64] | None = None,
    output_tokens: NDArray[np.float64] | None = None,
) -> BootstrapResult:
    """Circular block bootstrap: resample time-ordered blocks, rerun detection, report CIs.

    Uses block resampling (block_size = sqrt(n)) on records sorted by start_ns
    to preserve local temporal correlation. Individual-record resampling would
    duplicate intervals and create artificial concurrency spikes; block
    resampling keeps within-block temporal structure intact.

    Args:
        start_ns: Per-record start timestamps.
        end_ns: Per-record end timestamps.
        latency: Per-record request latency values.
        ttft: Per-record time-to-first-token values.
        n_iterations: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95 for 95% CI).
        min_window_pct: Minimum window size % passed to detection.
        rng: Random generator for reproducibility.
        generation_start_ns: Per-record generation start timestamps (optional).
        output_tokens: Per-record output token counts (optional).

    Returns:
        BootstrapResult with confidence intervals.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(start_ns)
    alpha = (1 - confidence) / 2

    # Sort records by start_ns so block resampling preserves temporal adjacency.
    # NaN entries (sparse session_num gaps) sort to the end.
    time_order = np.argsort(np.where(~np.isnan(start_ns), start_ns, np.inf))
    block_size = max(1, int(np.sqrt(n)))

    ramp_ups: list[float] = []
    ramp_downs: list[float] = []
    mean_lats: list[float] = []
    p99_lats: list[float] = []

    has_tput = generation_start_ns is not None and output_tokens is not None

    for _ in range(n_iterations):
        boot_positions = _circular_block_indices(n, block_size, rng)
        idx = time_order[boot_positions]
        s_ns = start_ns[idx]
        e_ns = end_ns[idx]
        lat = latency[idx]
        tt = ttft[idx]

        sorted_ts, conc = concurrency_sweep_line(s_ns, e_ns)
        if len(sorted_ts) == 0:
            continue

        s_tput_ts: NDArray[np.float64] | None = None
        tput: NDArray[np.float64] | None = None
        if has_tput:
            gen_ns = generation_start_ns[idx]  # type: ignore[index]
            out_tok = output_tokens[idx]  # type: ignore[index]
            s_tput_ts, tput = throughput_sweep_line(gen_ns, e_ns, out_tok)
            if len(s_tput_ts) == 0:
                s_tput_ts = None
                tput = None

        w_start, w_end, _method = detect_steady_state_window(
            sorted_ts,
            conc,
            s_ns,
            e_ns,
            lat,
            tt,
            min_window_pct=min_window_pct,
            sorted_tput_ts=s_tput_ts,
            throughput=tput,
        )

        ramp_ups.append(w_start)
        ramp_downs.append(w_end)

        mask = (s_ns >= w_start) & (e_ns <= w_end) & ~np.isnan(lat)
        windowed = lat[mask]
        if len(windowed) > 0:
            mean_lats.append(float(np.mean(windowed)))
            p99_lats.append(float(np.percentile(windowed, 99)))

    def _ci(values: list[float]) -> tuple[float, float]:
        if len(values) < 2:
            return (float("nan"), float("nan"))
        arr = np.array(values)
        return (float(np.quantile(arr, alpha)), float(np.quantile(arr, 1 - alpha)))

    return BootstrapResult(
        ci_ramp_up_ns=_ci(ramp_ups),
        ci_ramp_down_ns=_ci(ramp_downs),
        ci_mean_latency=_ci(mean_lats),
        ci_p99_latency=_ci(p99_lats),
        n_iterations=n_iterations,
    )
