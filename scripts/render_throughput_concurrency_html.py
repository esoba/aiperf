#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render interactive HTML for AIPerf throughput and concurrency over time."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from analyze_profile_export import (
    build_events,
    build_expected_windows,
    build_wave_completion_windows,
    load_records,
    sweep,
)
from plotly.subplots import make_subplots

NVIDIA_GREEN = "#76B900"
BG = "#0B0F10"
PANEL = "#11181C"
GRID = "rgba(194, 209, 217, 0.14)"
TEXT = "#E6F0F2"
MUTED = "#9FB3B8"
BLUE = "#35B5FF"
ORANGE = "#FFB020"
RED = "#FF5D73"
TEAL = "#2DD4BF"
PURPLE = "#A78BFA"


def rolling_rate(arr: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def add_vertical_markers(
    fig: go.Figure, xs: list[float], color: str, dash: str, row: int
) -> None:
    for x in xs:
        fig.add_vline(
            x=x,
            line_color=color,
            line_width=1,
            line_dash=dash,
            opacity=0.55,
            row=row,
            col=1,
        )


def build_html(
    input_path: Path,
    output_path: Path,
    title: str,
    subtitle: str,
    expected_latency_s: float | None,
) -> None:
    records = load_records(input_path)
    events = build_events(records)
    state, ts, t0 = sweep(events)

    expected_windows = build_expected_windows(ts, expected_latency_s)
    wave_windows = build_wave_completion_windows(ts, records, t0)

    t_sec = np.arange(ts.n)
    active_end = ts.n
    for b in range(ts.n - 1, -1, -1):
        if ts.in_flight[b] > 0 or ts.ends[b] > 0:
            active_end = min(ts.n, b + 20)
            break
    z = slice(0, active_end)

    starts_s = rolling_rate(ts.starts[z], 5)
    ends_s = rolling_rate(ts.ends[z], 5)
    errors_s = rolling_rate(ts.errors[z], 5)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.42, 0.58],
        subplot_titles=(
            "Throughput & Error Rate",
            "Concurrency Over Time",
        ),
    )

    x = t_sec[z]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=starts_s,
            mode="lines",
            name="Requests Started/s",
            line=dict(color=BLUE, width=2.5),
            hovertemplate="t=%{x:.0f}s<br>started=%{y:,.1f}/s<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ends_s,
            mode="lines",
            name="Requests Completed/s",
            line=dict(color=TEAL, width=2.5),
            hovertemplate="t=%{x:.0f}s<br>completed=%{y:,.1f}/s<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=errors_s,
            mode="lines",
            name="Errors/s",
            line=dict(color=RED, width=1.6),
            fill="tozeroy",
            fillcolor="rgba(255, 93, 115, 0.26)",
            hovertemplate="t=%{x:.0f}s<br>errors=%{y:,.1f}/s<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=ts.in_flight[z],
            mode="lines",
            name="In-Flight Total",
            line=dict(color=BLUE, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(53, 181, 255, 0.16)",
            hovertemplate="t=%{x:.0f}s<br>in-flight=%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ts.pre_ttft[z],
            mode="lines",
            name="Pre-1st-Response",
            line=dict(color=ORANGE, width=2.0),
            fill="tozeroy",
            fillcolor="rgba(255, 176, 32, 0.18)",
            hovertemplate="t=%{x:.0f}s<br>pre-1st-response=%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ts.post_ttft[z],
            mode="lines",
            name="Post-1st-Response",
            line=dict(color=PURPLE, width=2.0),
            fill="tozeroy",
            fillcolor="rgba(167, 139, 250, 0.14)",
            hovertemplate="t=%{x:.0f}s<br>post-1st-response=%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    expected_xs: list[float] = []
    for win in expected_windows:
        expected_xs.extend([win.expected_start_s, win.expected_end_s])
    add_vertical_markers(fig, expected_xs, NVIDIA_GREEN, "dot", 1)
    add_vertical_markers(fig, expected_xs, NVIDIA_GREEN, "dot", 2)

    observed_xs = [
        float(win.first_end_s) for win in wave_windows if win.first_end_s is not None
    ]
    add_vertical_markers(fig, observed_xs, ORANGE, "dashdot", 1)
    add_vertical_markers(fig, observed_xs, ORANGE, "dashdot", 2)

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.17,
        showarrow=False,
        align="left",
        text=(
            f"<span style='color:{NVIDIA_GREEN}; font-size:30px; font-weight:800;'>NVIDIA</span>"
            f"<span style='color:{TEXT}; font-size:30px; font-weight:800;'> AIPerf</span>"
            f"<br><span style='color:{TEXT}; font-size:22px; font-weight:700;'>{title}</span>"
            f"<br><span style='color:{MUTED}; font-size:13px;'>{subtitle}</span>"
        ),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Arial, Helvetica, sans-serif"),
        hovermode="x unified",
        height=900,
        margin=dict(l=70, r=40, t=130, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )

    fig.update_xaxes(
        title_text="Time (seconds)",
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        ticksuffix="s",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Rate (req/s)",
        showgrid=True,
        gridcolor=GRID,
        separatethousands=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Concurrent Requests",
        showgrid=True,
        gridcolor=GRID,
        separatethousands=True,
        row=2,
        col=1,
    )

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Path to profile_export.jsonl snapshot"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path")
    parser.add_argument(
        "--title", default="Throughput & Concurrency", help="Main title"
    )
    parser.add_argument("--subtitle", default="", help="Subtitle text")
    parser.add_argument("--mock-ttft-ms", type=float, default=None)
    parser.add_argument("--mock-itl-ms", type=float, default=None)
    parser.add_argument("--mock-osl", type=float, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    expected_latency_s = None
    if None not in (args.mock_ttft_ms, args.mock_itl_ms, args.mock_osl):
        expected_latency_s = (
            args.mock_ttft_ms + args.mock_itl_ms * args.mock_osl
        ) / 1000.0
    output = args.output or args.input.with_name(
        args.input.stem + "-throughput-concurrency.html"
    )
    build_html(args.input, output, args.title, args.subtitle, expected_latency_s)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
