#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a polished throughput-over-time area chart with in-flight overlay.

This script reads an AIPerf ``profile_export.jsonl`` snapshot, computes output-token
throughput using AIPerf's event-based token-dispersion algorithm, and overlays an
in-flight request curve derived from request start/end timestamps.

Example:
    uv run python scripts/plot_throughput_events_overlay.py \
      /tmp/profile_export.jsonl \
      --title "mock-250k snapshot" \
      --subtitle "242,007 successes / 693 errors"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import orjson
import pandas as pd

from aiperf.plot.core.data_preparation import calculate_throughput_events


def _metric_value(rec: dict, name: str) -> float | None:
    metric = rec.get("metrics", {}).get(name, {})
    if isinstance(metric, dict):
        return metric.get("value")
    return None


def load_snapshot(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    success_rows: list[dict] = []
    inflight_events: list[tuple[int, int]] = []
    min_ts: int | None = None
    min_generation_start_ns: int | None = None
    success_count = 0
    error_count = 0

    with path.open("rb") as f:
        for line in f:
            rec = orjson.loads(line)
            metadata = rec.get("metadata", {})
            request_start_ns = metadata.get("request_start_ns")
            request_end_ns = metadata.get("request_end_ns")

            if request_start_ns is not None and request_end_ns is not None:
                start_ns = int(request_start_ns)
                end_ns = int(request_end_ns)
                inflight_events.append((start_ns, 1))
                inflight_events.append((end_ns, -1))
                if min_ts is None or start_ns < min_ts:
                    min_ts = start_ns

            if rec.get("error") is not None:
                error_count += 1
                continue

            ttft_ms = _metric_value(rec, "time_to_first_token")
            success_rows.append(
                {
                    "request_start_ns": request_start_ns,
                    "request_end_ns": request_end_ns,
                    "time_to_first_token": ttft_ms,
                    "output_sequence_length": _metric_value(
                        rec, "output_sequence_length"
                    ),
                }
            )
            if request_start_ns is not None:
                generation_start_ns = int(request_start_ns)
                if ttft_ms is not None:
                    generation_start_ns += int(float(ttft_ms) * 1e6)
                if (
                    min_generation_start_ns is None
                    or generation_start_ns < min_generation_start_ns
                ):
                    min_generation_start_ns = generation_start_ns
            success_count += 1

    success_df = pd.DataFrame(success_rows)
    throughput_df = calculate_throughput_events(success_df)
    if (
        min_ts is not None
        and min_generation_start_ns is not None
        and not throughput_df.empty
    ):
        # Re-anchor throughput events to the same global origin as requests:
        # t=0 is the minimum request_start_ns seen in the snapshot.
        throughput_df = throughput_df.copy()
        throughput_df["timestamp_s"] += (min_generation_start_ns - min_ts) / 1e9

    inflight_events.sort(key=lambda item: (item[0], 0 if item[1] == 1 else 1))
    current = 0
    inflight_rows: list[dict] = []
    if min_ts is not None:
        for ts_ns, delta in inflight_events:
            current += delta
            inflight_rows.append(
                {
                    "timestamp_s": (ts_ns - min_ts) / 1e9,
                    "inflight_requests": current,
                }
            )
    inflight_df = pd.DataFrame(inflight_rows)

    return throughput_df, inflight_df, success_count, error_count


def format_throughput(value: float, _pos: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:,.0f}"


def format_count(value: float, _pos: int) -> str:
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:,.0f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Path to profile_export.jsonl snapshot"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path; defaults beside the input file",
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        default=None,
        help="Optional SVG output path",
    )
    parser.add_argument("--title", default=None, help="Primary chart title")
    parser.add_argument("--subtitle", default=None, help="Secondary title line")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    throughput_df, inflight_df, success_count, error_count = load_snapshot(args.input)
    if throughput_df.empty:
        raise SystemExit(
            "No successful records with throughput data found in input file."
        )

    peak_tp_idx = throughput_df["throughput_tokens_per_sec"].idxmax()
    peak_tp = float(throughput_df.loc[peak_tp_idx, "throughput_tokens_per_sec"])
    peak_tp_t = float(throughput_df.loc[peak_tp_idx, "timestamp_s"])

    peak_if_idx = inflight_df["inflight_requests"].idxmax()
    peak_if = int(inflight_df.loc[peak_if_idx, "inflight_requests"])
    peak_if_t = float(inflight_df.loc[peak_if_idx, "timestamp_s"])

    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f4ee",
            "axes.facecolor": "#fbfaf7",
            "axes.edgecolor": "#d8d2c4",
            "axes.labelcolor": "#2f3a3d",
            "xtick.color": "#48545a",
            "ytick.color": "#48545a",
            "font.size": 11,
        }
    )

    fig, ax1 = plt.subplots(figsize=(15.5, 8.5))
    fig.patch.set_facecolor("#f7f4ee")
    ax1.set_facecolor("#fcfbf8")

    x = throughput_df["timestamp_s"].to_numpy()
    y = throughput_df["throughput_tokens_per_sec"].to_numpy()
    ax1.fill_between(x, y, 0, color="#7fc8d8", alpha=0.22, linewidth=0)
    ax1.fill_between(x, y, 0, color="#2a9dbb", alpha=0.38, linewidth=0)
    ax1.plot(x, y, color="#0b6e88", linewidth=2.0, solid_capstyle="round", zorder=3)
    ax1.axvline(
        peak_tp_t,
        color="#c26a00",
        linestyle=(0, (4, 4)),
        linewidth=1.4,
        alpha=0.9,
        zorder=2,
    )
    ax1.scatter([peak_tp_t], [peak_tp], color="#c26a00", s=42, zorder=4)
    ax1.annotate(
        f"Peak throughput\n{peak_tp:,.0f} tok/s at {peak_tp_t:.1f}s",
        xy=(peak_tp_t, peak_tp),
        xytext=(14, -18),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#6e3d00",
        bbox={
            "boxstyle": "round,pad=0.35",
            "fc": "#fff9ef",
            "ec": "#d8a45c",
            "alpha": 0.96,
        },
    )

    ax2 = ax1.twinx()
    ix = inflight_df["timestamp_s"].to_numpy()
    iy = inflight_df["inflight_requests"].to_numpy()
    ax2.fill_between(ix, iy, 0, color="#94c973", alpha=0.12, linewidth=0, zorder=1)
    ax2.plot(ix, iy, color="#5a8d2f", linewidth=1.8, alpha=0.9, zorder=2)
    ax2.scatter([peak_if_t], [peak_if], color="#5a8d2f", s=28, zorder=4)
    ax2.annotate(
        f"Peak in-flight\n{peak_if:,.0f} reqs at {peak_if_t:.1f}s",
        xy=(peak_if_t, peak_if),
        xytext=(12, 16),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#36551a",
        bbox={
            "boxstyle": "round,pad=0.35",
            "fc": "#f5faef",
            "ec": "#9dc27c",
            "alpha": 0.96,
        },
    )

    title = args.title or "Output Token Throughput With In-Flight Request Overlay"
    subtitle = args.subtitle or f"{success_count:,} successes / {error_count:,} errors"
    ax1.set_title(f"{title}\n{subtitle}", fontsize=15, color="#243238", pad=16)
    ax1.set_xlabel("Time Since First Request Start (s)", labelpad=10)
    ax1.set_ylabel("Output Token Throughput (tokens/s)", color="#0b6e88", labelpad=10)
    ax2.set_ylabel("In-Flight Requests", color="#5a8d2f", labelpad=12)

    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(format_throughput))
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(format_count))
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _pos: f"{v:,.0f}s"))

    ax1.tick_params(axis="y", colors="#0b6e88")
    ax2.tick_params(axis="y", colors="#5a8d2f")
    ax1.tick_params(axis="x", pad=6)

    for spine in ax1.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#d8d2c4")
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax1.grid(True, axis="y", color="#d9dde2", linewidth=0.9, alpha=0.65)
    ax1.grid(True, axis="x", color="#ece7db", linewidth=0.7, alpha=0.55)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    fig.text(
        0.012,
        0.015,
        "Throughput uses AIPerf event-based token dispersion. In-flight uses request_start_ns/request_end_ns from the same frozen snapshot.",
        fontsize=9.5,
        color="#647077",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    output = args.output or args.input.with_name(
        args.input.stem + "-throughput-overlay.png"
    )
    fig.savefig(output, dpi=200)
    if args.svg_output:
        fig.savefig(args.svg_output)

    print(output)
    if args.svg_output:
        print(args.svg_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
