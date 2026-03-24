#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scatter plot of credit-pipeline latency over time from profile_export.jsonl."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import orjson
import pandas as pd


def get_metric_ms(rec: dict, key: str) -> float | None:
    metric = rec.get("metrics", {}).get(key)
    if not metric:
        return None
    value = metric.get("value")
    if value is None:
        return None
    unit = metric.get("unit", "")
    if unit == "s":
        return value * 1000.0
    return float(value)


def get_request_lifecycle_start_ns(rec: dict) -> int | None:
    metadata = rec["metadata"]
    req_end = metadata.get("request_end_ns")
    http_total_ms = get_metric_ms(rec, "http_req_total")
    if req_end and http_total_ms is not None:
        return req_end - int(http_total_ms * 1e6)
    return metadata.get("request_start_ns")


def load_credit_points(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    t0: int | None = None

    with path.open("rb") as f:
        for line in f:
            rec = orjson.loads(line)
            m = rec["metadata"]
            issued = m.get("credit_issued_ns")
            received = m.get("credit_received_ns")
            lifecycle_start = get_request_lifecycle_start_ns(rec)
            if not issued or not received or not lifecycle_start:
                continue
            issued = int(issued)
            received = int(received)
            lifecycle_start = int(lifecycle_start)
            if t0 is None or issued < t0:
                t0 = issued
            rows.append(
                {
                    "credit_issued_ns": issued,
                    "send_to_recv_ms": (received - issued) / 1e6,
                    "recv_to_start_ms": (lifecycle_start - received) / 1e6,
                    "is_error": rec.get("error") is not None,
                }
            )

    if not rows or t0 is None:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["time_s"] = (df["credit_issued_ns"] - t0) / 1e9
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--svg-output", type=Path, default=None)
    parser.add_argument(
        "--title",
        default="Credit Pipeline Scatter Over Time",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=120000,
        help="Max points to plot per series for readability",
    )
    return parser


def scatter_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    label: str,
    sample: int,
) -> None:
    if len(x) > sample:
        idx = np.linspace(0, len(x) - 1, sample, dtype=int)
        x = x[idx]
        y = y[idx]
    ax.scatter(x, y, s=4, alpha=0.18, c=color, edgecolors="none", label=label)


def make_plot(df: pd.DataFrame, output: Path, title: str, sample: int) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f4ee",
            "axes.facecolor": "#fcfbf8",
            "axes.edgecolor": "#d8d2c4",
            "font.size": 11,
        }
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    scatter_panel(
        ax1,
        df["time_s"].to_numpy(),
        df["send_to_recv_ms"].to_numpy(),
        color="#0b6e88",
        label="credit_sent -> credit_received",
        sample=sample,
    )
    ax1.set_ylabel("credit_sent -> credit_received (ms)")
    ax1.set_title(title, fontsize=15, color="#243238", pad=14)
    ax1.grid(True, color="#dde3e7", alpha=0.6)

    scatter_panel(
        ax2,
        df["time_s"].to_numpy(),
        df["recv_to_start_ms"].to_numpy(),
        color="#d97a00",
        label="credit_received -> request_start",
        sample=sample,
    )
    ax2.set_ylabel("credit_received -> request_start (ms)")
    ax2.set_xlabel("Time Since First Credit Issued (s)")
    ax2.grid(True, color="#dde3e7", alpha=0.6)

    for ax in (ax1, ax2):
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=min(-100.0, float(df["recv_to_start_ms"].min()) - 50))
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _p: f"{v:,.0f}s"))
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _p: f"{v:,.0f}"))

    fig.text(
        0.012,
        0.015,
        "Points are individual requests sampled from the frozen snapshot.",
        fontsize=9.5,
        color="#647077",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output, dpi=200)


def main() -> int:
    args = build_parser().parse_args()
    df = load_credit_points(args.input)
    if df.empty:
        raise SystemExit("No credit pipeline records found in input file.")
    output = args.output or args.input.with_name(
        args.input.stem + "-credit-scatter.png"
    )
    make_plot(df, output, args.title, args.sample)
    print(output)
    if args.svg_output:
        make_plot(df, args.svg_output, args.title, args.sample)
        print(args.svg_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
