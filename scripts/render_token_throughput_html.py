#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render an NVIDIA AIPerf themed HTML dashboard for streaming token throughput."""

from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path
from string import Template

import numpy as np
import orjson
import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.core.data_preparation import calculate_throughput_events

NVIDIA_GREEN = "#76B900"
NVIDIA_GREEN_BRIGHT = "#A4F72E"
BG = "#05070B"
BG_ALT = "#09111A"
PANEL = "rgba(10, 17, 24, 0.88)"
PANEL_STRONG = "rgba(13, 21, 30, 0.97)"
PLOT_BG = "rgba(7, 12, 18, 0.88)"
BORDER = "rgba(167, 190, 175, 0.16)"
GRID = "rgba(150, 181, 165, 0.12)"
TEXT = "#F5F8F2"
MUTED = "#9BAEA1"
SUBTLE = "#6E8177"
TOK_LINE = "#86E133"
TOK_FILL = "rgba(118, 185, 0, 0.20)"
SSE_LINE = "#4AB8FF"
SSE_FILL = "rgba(74, 184, 255, 0.17)"
ACT_LINE = "#FFC857"
ACT_FILL = "rgba(255, 200, 87, 0.16)"
MEAN_LINE = "rgba(226, 239, 233, 0.36)"
PEAK_LINE = "rgba(164, 247, 46, 0.70)"
FONT_DISPLAY = '"Space Grotesk", "Avenir Next", "Segoe UI", sans-serif'
FONT_BODY = '"IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif'


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def trim_zeros(text: str) -> str:
    return text.rstrip("0").rstrip(".")


def format_compact(value: float, suffix: str = "") -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{trim_zeros(f'{value / 1_000_000_000:.2f}')}B{suffix}"
    if abs_value >= 1_000_000:
        return f"{trim_zeros(f'{value / 1_000_000:.2f}')}M{suffix}"
    if abs_value >= 1_000:
        return f"{trim_zeros(f'{value / 1_000:.1f}')}K{suffix}"
    if abs_value >= 100:
        return f"{value:,.0f}{suffix}"
    if abs_value >= 10:
        return f"{trim_zeros(f'{value:,.1f}')}{suffix}"
    return f"{trim_zeros(f'{value:,.2f}')}{suffix}"


def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes:02d}m"
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = seconds % 60
        if seconds >= 600:
            return f"{minutes}m {secs:02.0f}s"
        return f"{minutes}m {secs:04.1f}s"
    return f"{seconds:.1f}s"


def format_millis(ms: float | None) -> str:
    if ms is None or np.isnan(ms):
        return "n/a"
    if ms >= 1000:
        return f"{trim_zeros(f'{ms / 1000:.2f}')}s"
    return f"{trim_zeros(f'{ms:.0f}')} ms"


def sample_durations(xs: np.ndarray) -> np.ndarray:
    if xs.size == 0:
        return np.array([], dtype=float)
    if xs.size == 1:
        return np.array([0.0], dtype=float)
    deltas = np.diff(xs)
    tail = float(np.median(deltas)) if deltas.size else 0.0
    return np.append(deltas, tail)


def time_weighted_mean(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    if df.empty:
        return 0.0

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    if x.size == 0:
        return 0.0
    if x.size == 1:
        return float(y[0])

    dt = np.diff(x)
    duration = float(dt.sum())
    if duration <= 0:
        return float(y[-1])
    return float(np.dot(y[:-1], dt) / duration)


def time_weighted_mean_segment(x: np.ndarray, y: np.ndarray, end_x: float) -> float:
    if x.size == 0:
        return 0.0
    if x.size == 1:
        return float(y[0]) if end_x > float(x[0]) else 0.0

    segment_edges = np.append(x, end_x)
    dt = np.diff(segment_edges)
    duration = float(dt.sum())
    if duration <= 0:
        return float(y[-1])
    return float(np.dot(y, dt) / duration)


def load_success_df(path: Path) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict[str, object]] = []
    success_count = 0
    error_count = 0

    with path.open("rb") as f:
        for line in f:
            rec = orjson.loads(line)
            if rec.get("error") is not None:
                error_count += 1
                continue

            m = rec.get("metadata", {})
            metrics = rec.get("metrics", {})

            def getv(name: str):
                v = metrics.get(name, {})
                return v.get("value") if isinstance(v, dict) else None

            rows.append(
                {
                    "request_start_ns": m.get("request_start_ns"),
                    "request_end_ns": m.get("request_end_ns"),
                    "time_to_first_token": getv("time_to_first_token"),
                    "output_sequence_length": getv("output_sequence_length"),
                    "http_req_chunks_received": getv("http_req_chunks_received"),
                }
            )
            success_count += 1

    return pd.DataFrame(rows), success_count, error_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Path to profile_export.jsonl snapshot"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path")
    parser.add_argument("--title", default="Streaming Token Throughput")
    parser.add_argument("--subtitle", default="")
    return parser


def build_sse_events(df: pd.DataFrame) -> pd.DataFrame:
    events: list[dict[str, float]] = []
    for _, row in df.iterrows():
        request_start_ns = row.get("request_start_ns")
        request_end_ns = row.get("request_end_ns")
        ttft_ms = row.get("time_to_first_token")
        chunk_count = row.get("http_req_chunks_received")
        if pd.isna(request_start_ns) or pd.isna(request_end_ns):
            continue
        if pd.isna(ttft_ms) or pd.isna(chunk_count):
            continue

        chunk_count = int(chunk_count)
        if chunk_count <= 0:
            continue

        request_start_ns = int(request_start_ns)
        request_end_ns = int(request_end_ns)
        generation_start_ns = request_start_ns + int(float(ttft_ms) * 1e6)
        generation_duration_ns = request_end_ns - generation_start_ns
        if generation_duration_ns <= 0:
            continue

        sse_rate = chunk_count / (generation_duration_ns / 1e9)
        events.append({"timestamp_ns": generation_start_ns, "delta_rate": sse_rate})
        events.append({"timestamp_ns": request_end_ns, "delta_rate": -sse_rate})

    if not events:
        return pd.DataFrame(columns=["timestamp_s", "sse_messages_per_sec"])

    events_df = pd.DataFrame(events).sort_values("timestamp_ns")
    events_df["sse_messages_per_sec"] = events_df["delta_rate"].cumsum()
    start_ns = events_df["timestamp_ns"].min()
    events_df["timestamp_s"] = (events_df["timestamp_ns"] - start_ns) / 1e9
    return events_df[["timestamp_s", "sse_messages_per_sec"]].reset_index(drop=True)


def compute_live_phases(
    throughput_df: pd.DataFrame, sse_df: pd.DataFrame, peak_tp: float
) -> list[dict[str, float]]:
    if throughput_df.empty or peak_tp <= 0:
        return []

    x = throughput_df["timestamp_s"].to_numpy(dtype=float)
    y = throughput_df["throughput_tokens_per_sec"].to_numpy(dtype=float)
    active = throughput_df["active_requests"].to_numpy(dtype=float)
    live_mask = y >= peak_tp * 0.08
    phases: list[dict[str, float]] = []
    phase_start: int | None = None

    for idx, is_live in enumerate(live_mask):
        if is_live and phase_start is None:
            phase_start = idx
        end_of_phase = phase_start is not None and (
            (not is_live) or idx == len(live_mask) - 1
        )
        if not end_of_phase:
            continue

        phase_end = idx if is_live and idx == len(live_mask) - 1 else idx - 1
        if phase_start is None or phase_end <= phase_start:
            phase_start = None
            continue

        phase_stop_s = float(x[idx]) if not is_live else float(x[phase_end])
        x0 = float(x[phase_start])
        duration_s = max(0.0, phase_stop_s - x0)
        y_phase = y[phase_start : phase_end + 1]
        x_phase = x[phase_start : phase_end + 1]
        phase_peak = float(y_phase.max())
        if duration_s < 1.0 and phase_peak < peak_tp * 0.18:
            phase_start = None
            continue

        phase_active = active[phase_start : phase_end + 1]
        phase_slice = sse_df[
            (sse_df["timestamp_s"] >= x0) & (sse_df["timestamp_s"] <= phase_stop_s)
        ]
        phase_dt = np.diff(np.append(x_phase, phase_stop_s))
        phase_tokens_est = float(np.dot(y_phase, phase_dt)) if phase_dt.size else 0.0
        phases.append(
            {
                "start_s": x0,
                "end_s": phase_stop_s,
                "duration_s": duration_s,
                "peak_tp": phase_peak,
                "mean_active": time_weighted_mean_segment(
                    x_phase, phase_active, phase_stop_s
                ),
                "peak_active": float(phase_active.max()),
                "peak_sse": float(phase_slice["sse_messages_per_sec"].max())
                if not phase_slice.empty
                else 0.0,
                "tokens_est": phase_tokens_est,
            }
        )
        phase_start = None

    return phases


def build_plot(
    *,
    x: pd.Series,
    y: pd.Series,
    y_title: str,
    line_color: str,
    fill_color: str,
    hover_label: str,
    peak_x: float,
    peak_y: float,
    peak_text: str,
    mean_y: float | None = None,
    mean_text: str | None = None,
    phase_windows: list[dict[str, float]] | None = None,
) -> go.Figure:
    fig = go.Figure()

    if phase_windows:
        for idx, phase in enumerate(phase_windows):
            tint = 0.085 if idx % 2 == 0 else 0.05
            fig.add_vrect(
                x0=phase["start_s"],
                x1=phase["end_s"],
                fillcolor=f"rgba(118, 185, 0, {tint})",
                opacity=1.0,
                layer="below",
                line_width=0,
            )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=line_color, width=3.3),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate=f"t=%{{x:.1f}}s<br>{hover_label}=%{{y:,.0f}}<extra></extra>",
            showlegend=False,
        )
    )

    fig.add_vline(
        x=peak_x,
        line_color=PEAK_LINE,
        line_width=1.3,
        line_dash="dot",
        opacity=0.95,
    )

    if mean_y is not None:
        fig.add_hline(
            y=mean_y,
            line_color=MEAN_LINE,
            line_width=1.0,
            line_dash="dash",
            annotation_text=mean_text or "",
            annotation_position="top left",
            annotation_font=dict(color=MUTED, size=11, family=FONT_BODY),
        )

    arrow_shift = 72 if peak_x <= float(x.max()) * 0.72 else -72
    fig.add_annotation(
        x=peak_x,
        y=peak_y,
        text=peak_text,
        showarrow=True,
        arrowhead=2,
        ax=arrow_shift,
        ay=-46,
        arrowcolor=NVIDIA_GREEN_BRIGHT,
        bordercolor=NVIDIA_GREEN,
        borderwidth=1,
        borderpad=8,
        bgcolor=PANEL_STRONG,
        font=dict(color=TEXT, size=11, family=FONT_BODY),
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT, family=FONT_BODY, size=13),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=PANEL_STRONG,
            bordercolor=BORDER,
            font=dict(color=TEXT, family=FONT_BODY, size=12),
        ),
        margin=dict(l=72, r=28, t=14, b=56),
        height=430,
        dragmode="pan",
    )

    fig.update_xaxes(
        title_text="Time Since First Generation Start",
        ticksuffix="s",
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=BORDER,
        linewidth=1,
        tickfont=dict(color=MUTED, size=12, family=FONT_BODY),
        title_font=dict(color=MUTED, size=13, family=FONT_BODY),
        showspikes=True,
        spikecolor=PEAK_LINE,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
    )
    fig.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=BORDER,
        linewidth=1,
        separatethousands=True,
        tickfont=dict(color=MUTED, size=12, family=FONT_BODY),
        title_font=dict(color=MUTED, size=13, family=FONT_BODY),
        tickformat="~s",
    )
    return fig


def render_metric_card(label: str, value: str, detail: str, tone: str) -> str:
    return f"""
        <article class="metric-card metric-card--{tone}">
          <span class="metric-label">{esc(label)}</span>
          <strong class="metric-value">{esc(value)}</strong>
          <span class="metric-detail">{esc(detail)}</span>
        </article>
    """


def render_insight_card(kicker: str, title: str, body: str) -> str:
    return f"""
        <article class="insight-card">
          <span class="insight-kicker">{esc(kicker)}</span>
          <h3 class="insight-title">{esc(title)}</h3>
          <p class="insight-copy">{esc(body)}</p>
        </article>
    """


def render_phase_card(index: int, phase: dict[str, float], global_peak: float) -> str:
    intensity = (
        max(18.0, min(100.0, (phase["peak_tp"] / global_peak) * 100.0))
        if global_peak > 0
        else 18.0
    )
    return f"""
        <article class="phase-card">
          <div class="phase-topline">
            <span class="phase-name">Burst {index:02d}</span>
            <span class="phase-window">{esc(format_duration(phase["duration_s"]))}</span>
          </div>
          <div class="phase-meter"><span style="width:{intensity:.1f}%"></span></div>
          <dl class="phase-stats">
            <div><dt>Peak</dt><dd>{esc(format_compact(phase["peak_tp"], " tok/s"))}</dd></div>
            <div><dt>Start</dt><dd>{esc(f"{phase['start_s']:.1f}s")}</dd></div>
            <div><dt>Peak SSE</dt><dd>{esc(format_compact(phase["peak_sse"], " msg/s"))}</dd></div>
            <div><dt>Peak Active</dt><dd>{esc(format_compact(phase["peak_active"]))}</dd></div>
            <div><dt>Tokens</dt><dd>{esc(format_compact(phase["tokens_est"], " tok"))}</dd></div>
            <div><dt>Mean Active</dt><dd>{esc(format_compact(phase["mean_active"]))}</dd></div>
          </dl>
        </article>
    """


def render_meta_row(label: str, value: str) -> str:
    return f"""
        <div class="meta-row">
          <span>{esc(label)}</span>
          <strong>{esc(value)}</strong>
        </div>
    """


PAGE_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>$page_title</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    @import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap");

    :root {
      --bg: #05070b;
      --bg-alt: #09111a;
      --panel: rgba(10, 17, 24, 0.88);
      --panel-strong: rgba(13, 21, 30, 0.97);
      --plot: rgba(7, 12, 18, 0.88);
      --border: rgba(167, 190, 175, 0.16);
      --text: #f5f8f2;
      --muted: #9baea1;
      --subtle: #6e8177;
      --nvidia: #76b900;
      --nvidia-bright: #a4f72e;
      --cyan: #4ab8ff;
      --amber: #ffc857;
      --shadow: 0 26px 90px rgba(0, 0, 0, 0.42);
      --hero-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    * {
      box-sizing: border-box;
    }

    html {
      color-scheme: dark;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: $font_body;
      color: var(--text);
      background:
        radial-gradient(circle at 14% 14%, rgba(118, 185, 0, 0.18), transparent 28%),
        radial-gradient(circle at 82% 10%, rgba(74, 184, 255, 0.12), transparent 24%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg-alt) 52%, var(--bg) 100%);
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px);
      background-size: 72px 72px;
      opacity: 0.18;
      mask-image: linear-gradient(180deg, rgba(255, 255, 255, 0.5), transparent 88%);
      pointer-events: none;
    }

    body::after {
      content: "";
      position: fixed;
      inset: 0;
      background:
        radial-gradient(circle at 50% 0%, rgba(164, 247, 46, 0.06), transparent 32%),
        radial-gradient(circle at 50% 100%, rgba(74, 184, 255, 0.05), transparent 28%);
      pointer-events: none;
    }

    .page {
      position: relative;
      max-width: 1480px;
      margin: 0 auto;
      padding: 38px 28px 56px;
    }

    .hero,
    .panel,
    .insight-card {
      animation: rise-in 0.72s cubic-bezier(0.2, 0.9, 0.22, 1) both;
    }

    .hero {
      position: relative;
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(360px, 0.95fr);
      gap: 24px;
      padding: 30px;
      overflow: hidden;
      border: 1px solid var(--border);
      border-radius: 30px;
      background:
        linear-gradient(155deg, rgba(16, 24, 32, 0.96), rgba(9, 15, 22, 0.9)),
        linear-gradient(135deg, rgba(118, 185, 0, 0.09), transparent 48%);
      box-shadow: var(--shadow), var(--hero-shadow);
      isolation: isolate;
    }

    .hero::before {
      content: "";
      position: absolute;
      inset: auto -4% -55% auto;
      width: 520px;
      height: 520px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(118, 185, 0, 0.18), transparent 64%);
      filter: blur(12px);
      opacity: 0.8;
      z-index: -1;
    }

    .hero::after {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(120deg, transparent 0 62%, rgba(164, 247, 46, 0.08) 62% 63%, transparent 63% 100%);
      pointer-events: none;
    }

    .eyebrow,
    .panel-kicker,
    .insight-kicker,
    .signal-kicker,
    .meta-eyebrow {
      display: inline-block;
      text-transform: uppercase;
      letter-spacing: 0.24em;
      font-size: 11px;
      font-weight: 700;
      color: var(--subtle);
    }

    .brandline {
      display: flex;
      align-items: baseline;
      gap: 12px;
      margin-top: 12px;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }

    .brandmark {
      font-family: $font_display;
      font-size: clamp(42px, 7vw, 74px);
      line-height: 0.9;
      letter-spacing: -0.06em;
      margin: 0;
      color: var(--text);
    }

    .brandmark .nvidia {
      color: var(--nvidia);
    }

    .hero-title {
      margin: 0;
      font-family: $font_display;
      font-size: clamp(28px, 4vw, 56px);
      line-height: 0.95;
      letter-spacing: -0.05em;
      max-width: 12ch;
    }

    .hero-copy {
      margin: 0;
      max-width: 68ch;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.68;
    }

    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 20px;
    }

    .chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      font-size: 12px;
      font-weight: 600;
      backdrop-filter: blur(12px);
    }

    .chip strong {
      color: var(--nvidia-bright);
      font-weight: 700;
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      align-content: start;
    }

    .metric-card {
      position: relative;
      min-height: 138px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 18px;
      border-radius: 22px;
      border: 1px solid rgba(255, 255, 255, 0.06);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.02));
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
      overflow: hidden;
    }

    .metric-card::before {
      content: "";
      position: absolute;
      inset: auto 18px 0 18px;
      height: 4px;
      border-radius: 999px 999px 0 0;
      opacity: 0.96;
    }

    .metric-card--green::before { background: linear-gradient(90deg, var(--nvidia), #a4f72e); }
    .metric-card--cyan::before { background: linear-gradient(90deg, #188dff, var(--cyan)); }
    .metric-card--amber::before { background: linear-gradient(90deg, #e18b16, var(--amber)); }
    .metric-card--neutral::before { background: linear-gradient(90deg, rgba(255,255,255,0.3), rgba(255,255,255,0.12)); }

    .metric-label {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--subtle);
    }

    .metric-value {
      font-family: $font_display;
      font-size: clamp(28px, 3.2vw, 38px);
      line-height: 0.95;
      letter-spacing: -0.05em;
      margin: 14px 0 10px;
    }

    .metric-detail {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    .insight-strip {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
      margin-top: 22px;
    }

    .insight-card {
      padding: 18px 18px 20px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: linear-gradient(180deg, rgba(12, 18, 24, 0.92), rgba(8, 13, 18, 0.88));
      box-shadow: var(--shadow);
    }

    .insight-title {
      margin: 10px 0 8px;
      font-family: $font_display;
      font-size: 24px;
      letter-spacing: -0.04em;
    }

    .insight-copy {
      margin: 0;
      color: var(--muted);
      line-height: 1.65;
      font-size: 14px;
    }

    .overview-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.38fr) minmax(320px, 0.86fr);
      gap: 22px;
      margin-top: 22px;
      align-items: start;
    }

    .chart-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 22px;
      margin-top: 22px;
    }

    .panel {
      position: relative;
      padding: 20px;
      border: 1px solid var(--border);
      border-radius: 26px;
      background: linear-gradient(180deg, rgba(12, 18, 24, 0.94), rgba(8, 13, 18, 0.9));
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      overflow: hidden;
    }

    .panel::after {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.02), transparent 22%);
      pointer-events: none;
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      margin-bottom: 16px;
    }

    .panel-title {
      margin: 8px 0 6px;
      font-family: $font_display;
      font-size: clamp(24px, 2.4vw, 36px);
      line-height: 1;
      letter-spacing: -0.05em;
    }

    .panel-copy {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 14px;
      max-width: 62ch;
    }

    .panel-pill {
      flex: 0 0 auto;
      display: inline-flex;
      align-items: center;
      padding: 10px 12px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .plot-shell .js-plotly-plot .plotly .modebar {
      opacity: 0 !important;
      transition: opacity 0.2s ease;
    }

    .plot-shell:hover .js-plotly-plot .plotly .modebar {
      opacity: 1 !important;
    }

    .signal-stack {
      display: grid;
      gap: 16px;
    }

    .signal-card {
      padding: 18px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.02));
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .signal-title {
      margin: 10px 0 8px;
      font-family: $font_display;
      font-size: 28px;
      line-height: 0.98;
      letter-spacing: -0.04em;
    }

    .signal-copy {
      margin: 0;
      color: var(--muted);
      line-height: 1.64;
      font-size: 14px;
    }

    .meta-grid {
      display: grid;
      gap: 10px;
      margin-top: 4px;
    }

    .meta-row {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      padding: 12px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      font-size: 13px;
    }

    .meta-row:last-child {
      border-bottom: 0;
      padding-bottom: 0;
    }

    .meta-row span {
      color: var(--muted);
    }

    .meta-row strong {
      color: var(--text);
      font-weight: 600;
      text-align: right;
    }

    .phase-stack {
      display: grid;
      gap: 14px;
    }

    .phase-card {
      padding: 16px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.035), rgba(255, 255, 255, 0.018));
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .phase-topline {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }

    .phase-name {
      font-family: $font_display;
      font-size: 20px;
      letter-spacing: -0.04em;
    }

    .phase-window {
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }

    .phase-meter {
      position: relative;
      height: 8px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.06);
      overflow: hidden;
      margin-bottom: 14px;
    }

    .phase-meter span {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--nvidia), var(--nvidia-bright));
      box-shadow: 0 0 18px rgba(118, 185, 0, 0.35);
      animation: fill-bar 1.2s ease-out both;
      transform-origin: left center;
    }

    .phase-stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 14px;
      margin: 0;
    }

    .phase-stats div {
      padding: 8px 0;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    .phase-stats dt {
      margin: 0 0 6px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--subtle);
      font-weight: 700;
    }

    .phase-stats dd {
      margin: 0;
      color: var(--text);
      font-size: 14px;
      font-weight: 600;
    }

    .footer {
      margin-top: 18px;
      color: var(--subtle);
      font-size: 13px;
      line-height: 1.7;
      text-align: center;
    }

    @keyframes rise-in {
      from {
        opacity: 0;
        transform: translateY(18px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @keyframes fill-bar {
      from {
        transform: scaleX(0.15);
        opacity: 0.4;
      }
      to {
        transform: scaleX(1);
        opacity: 1;
      }
    }

    @media (max-width: 1220px) {
      .hero,
      .overview-grid,
      .chart-grid {
        grid-template-columns: 1fr;
      }

      .hero-title {
        max-width: none;
      }
    }

    @media (max-width: 860px) {
      .page {
        padding: 22px 16px 36px;
      }

      .hero,
      .panel,
      .insight-card {
        border-radius: 22px;
      }

      .metrics-grid,
      .insight-strip,
      .phase-stats {
        grid-template-columns: 1fr;
      }

      .panel-head {
        align-items: start;
        flex-direction: column;
      }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div>
        <span class="eyebrow">Streaming Token Telemetry</span>
        <div class="brandline">
          <h1 class="brandmark"><span class="nvidia">NVIDIA</span> AIPerf</h1>
          <h2 class="hero-title">$hero_title</h2>
        </div>
        <p class="hero-copy">$hero_copy</p>
        <div class="chip-row">
          $chips_html
        </div>
      </div>
      <div class="metrics-grid">
        $metric_cards_html
      </div>
    </section>

    <section class="insight-strip">
      $insight_cards_html
    </section>

    <section class="overview-grid">
      <article class="panel">
        <div class="panel-head">
          <div>
            <span class="panel-kicker">Primary Signal</span>
            <h3 class="panel-title">Output Throughput Pulse</h3>
            <p class="panel-copy">The main delivery curve tracks how hard the cluster is pushing decoded output over time. Burst windows are lightly shaded so phase boundaries stay legible during zoom and hover.</p>
          </div>
          <span class="panel-pill">Interactive Plot</span>
        </div>
        <div class="plot-shell">
          $throughput_plot_html
        </div>
      </article>

      <aside class="panel">
        <div class="signal-stack">
          <section class="signal-card">
            <span class="signal-kicker">Peak Moment</span>
            <h3 class="signal-title">$peak_headline</h3>
            <p class="signal-copy">$peak_story</p>
          </section>

          <section class="signal-card">
            <span class="signal-kicker">Run Meta</span>
            <div class="meta-grid">
              $meta_rows_html
            </div>
          </section>

          <section class="signal-card">
            <span class="signal-kicker">Burst Anatomy</span>
            <div class="phase-stack">
              $phase_cards_html
            </div>
          </section>
        </div>
      </aside>
    </section>

    <section class="chart-grid">
      <article class="panel">
        <div class="panel-head">
          <div>
            <span class="panel-kicker">Transport</span>
            <h3 class="panel-title">Streaming SSE Cadence</h3>
            <p class="panel-copy">Estimated message emission rate derived from chunk cadence during active generation windows.</p>
          </div>
          <span class="panel-pill">Event Rate</span>
        </div>
        <div class="plot-shell">
          $sse_plot_html
        </div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <span class="panel-kicker">Concurrency</span>
            <h3 class="panel-title">Active Generator Pressure</h3>
            <p class="panel-copy">Concurrent generators alive in the token-emission sweep, which helps explain where output bandwidth is landing.</p>
          </div>
          <span class="panel-pill">Live Requests</span>
        </div>
        <div class="plot-shell">
          $active_plot_html
        </div>
      </article>
    </section>

    <footer class="footer">
      $footer_copy
    </footer>
  </main>
</body>
</html>
"""
)


def main() -> int:
    args = build_parser().parse_args()
    output = args.output or args.input.with_name(
        args.input.stem + "-token-throughput.html"
    )

    success_df, success_count, error_count = load_success_df(args.input)
    throughput_df = calculate_throughput_events(success_df)
    if throughput_df.empty:
        raise SystemExit("No successful token-throughput events found in input file.")

    sse_df = build_sse_events(success_df)

    throughput_series = throughput_df["throughput_tokens_per_sec"].to_numpy(dtype=float)
    active_series = throughput_df["active_requests"].to_numpy(dtype=float)
    time_series = throughput_df["timestamp_s"].to_numpy(dtype=float)
    durations = sample_durations(time_series)

    peak_idx = throughput_df["throughput_tokens_per_sec"].idxmax()
    peak_t = float(throughput_df.loc[peak_idx, "timestamp_s"])
    peak_tp = float(throughput_df.loc[peak_idx, "throughput_tokens_per_sec"])
    active_at_peak_tp = float(throughput_df.loc[peak_idx, "active_requests"])
    duration_s = float(throughput_df["timestamp_s"].max())
    mean_tp = time_weighted_mean(
        throughput_df, "timestamp_s", "throughput_tokens_per_sec"
    )

    peak_sse = float(sse_df["sse_messages_per_sec"].max()) if not sse_df.empty else 0.0
    mean_sse = time_weighted_mean(sse_df, "timestamp_s", "sse_messages_per_sec")
    peak_sse_idx = (
        int(sse_df["sse_messages_per_sec"].idxmax()) if not sse_df.empty else -1
    )
    peak_sse_t = (
        float(sse_df.loc[peak_sse_idx, "timestamp_s"]) if peak_sse_idx >= 0 else 0.0
    )

    active_peak_idx = throughput_df["active_requests"].idxmax()
    active_peak_t = float(throughput_df.loc[active_peak_idx, "timestamp_s"])
    peak_active = float(throughput_df.loc[active_peak_idx, "active_requests"])
    mean_active = time_weighted_mean(throughput_df, "timestamp_s", "active_requests")

    ttft_ms_series = pd.to_numeric(
        success_df.get("time_to_first_token"), errors="coerce"
    ).dropna()
    median_ttft_ms = (
        float(ttft_ms_series.median()) if not ttft_ms_series.empty else None
    )
    p95_ttft_ms = (
        float(ttft_ms_series.quantile(0.95)) if not ttft_ms_series.empty else None
    )

    output_tokens_series = pd.to_numeric(
        success_df.get("output_sequence_length"), errors="coerce"
    ).dropna()
    total_output_tokens = (
        float(output_tokens_series.sum()) if not output_tokens_series.empty else 0.0
    )
    mean_output_tokens = (
        float(output_tokens_series.mean()) if not output_tokens_series.empty else 0.0
    )

    total_requests = success_count + error_count
    success_rate = (success_count / total_requests) if total_requests else 1.0
    sustained_ratio = (mean_tp / peak_tp) if peak_tp > 0 else 0.0

    live_mask = throughput_series >= peak_tp * 0.08
    high_mask = throughput_series >= peak_tp * 0.90
    live_runtime_s = float(durations[live_mask].sum()) if live_mask.any() else 0.0
    near_peak_runtime_s = float(durations[high_mask].sum()) if high_mask.any() else 0.0
    near_peak_share = (
        (near_peak_runtime_s / live_runtime_s) if live_runtime_s > 0 else 0.0
    )

    phases = compute_live_phases(throughput_df, sse_df, peak_tp)
    largest_gap_s = 0.0
    if len(phases) > 1:
        largest_gap_s = max(
            phases[idx + 1]["start_s"] - phases[idx]["end_s"]
            for idx in range(len(phases) - 1)
        )

    throughput_fig = build_plot(
        x=throughput_df["timestamp_s"],
        y=throughput_df["throughput_tokens_per_sec"],
        y_title="Output Tokens / s",
        line_color=TOK_LINE,
        fill_color=TOK_FILL,
        hover_label="throughput",
        peak_x=peak_t,
        peak_y=peak_tp,
        peak_text=f"Peak {format_compact(peak_tp, ' tok/s')}",
        mean_y=mean_tp,
        mean_text=f"Mean {format_compact(mean_tp, ' tok/s')}",
        phase_windows=phases,
    )

    sse_x = sse_df["timestamp_s"] if not sse_df.empty else throughput_df["timestamp_s"]
    sse_y = (
        sse_df["sse_messages_per_sec"]
        if not sse_df.empty
        else pd.Series(np.zeros(len(throughput_df), dtype=float))
    )
    sse_annotation = (
        f"Peak {format_compact(peak_sse, ' msg/s')}"
        if peak_sse > 0
        else "No SSE data captured"
    )
    sse_peak_y = peak_sse if peak_sse > 0 else 0.0
    sse_fig = build_plot(
        x=sse_x,
        y=sse_y,
        y_title="SSE Messages / s",
        line_color=SSE_LINE,
        fill_color=SSE_FILL,
        hover_label="SSE rate",
        peak_x=peak_sse_t,
        peak_y=sse_peak_y,
        peak_text=sse_annotation,
        mean_y=mean_sse if peak_sse > 0 else None,
        mean_text=f"Mean {format_compact(mean_sse, ' msg/s')}"
        if peak_sse > 0
        else None,
        phase_windows=phases,
    )

    active_fig = build_plot(
        x=throughput_df["timestamp_s"],
        y=throughput_df["active_requests"],
        y_title="Active Generating Requests",
        line_color=ACT_LINE,
        fill_color=ACT_FILL,
        hover_label="active generators",
        peak_x=active_peak_t,
        peak_y=float(throughput_df.loc[active_peak_idx, "active_requests"]),
        peak_text=f"Peak {format_compact(float(throughput_df.loc[active_peak_idx, 'active_requests']))} active",
        mean_y=mean_active,
        mean_text=f"Mean {format_compact(mean_active)} active",
        phase_windows=phases,
    )

    plot_config = {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
            "autoScale2d",
            "toggleSpikelines",
        ],
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }

    throughput_plot_html = throughput_fig.to_html(
        include_plotlyjs=False, full_html=False, config=plot_config
    )
    sse_plot_html = sse_fig.to_html(
        include_plotlyjs=False, full_html=False, config=plot_config
    )
    active_plot_html = active_fig.to_html(
        include_plotlyjs=False, full_html=False, config=plot_config
    )

    source_label = args.input.parent.name or args.input.name
    subtitle = (
        args.subtitle
        or f"{source_label} • {success_count:,} successful streams • {error_count:,} errors"
    )
    hero_copy = (
        f"{subtitle}. This dashboard rebuilds the token throughput report as an NVIDIA AIPerf control-room view: "
        f"peak delivery, transport cadence, and generator pressure are all laid out so the load shape reads instantly."
    )

    chips_html = "".join(
        [
            f'<span class="chip"><strong>Source</strong> {esc(source_label)}</span>',
            f'<span class="chip"><strong>Success</strong> {success_rate:.1%}</span>',
            f'<span class="chip"><strong>Bursts</strong> {max(1, len(phases))}</span>',
            f'<span class="chip"><strong>Runtime</strong> {esc(format_duration(duration_s))}</span>',
        ]
    )

    metric_cards_html = "".join(
        [
            render_metric_card(
                "Peak Throughput",
                format_compact(peak_tp, " tok/s"),
                f"Reached at {peak_t:.1f}s",
                "green",
            ),
            render_metric_card(
                "Sustained Load",
                f"{sustained_ratio:.0%} of peak",
                f"Mean {format_compact(mean_tp, ' tok/s')}",
                "neutral",
            ),
            render_metric_card(
                "Peak SSE",
                format_compact(peak_sse, " msg/s"),
                f"Mean {format_compact(mean_sse, ' msg/s')}",
                "cyan",
            ),
            render_metric_card(
                "Peak Active",
                format_compact(peak_active),
                f"Reached at {active_peak_t:.1f}s",
                "amber",
            ),
            render_metric_card(
                "Median TTFT",
                format_millis(median_ttft_ms),
                f"P95 {format_millis(p95_ttft_ms)}",
                "neutral",
            ),
            render_metric_card(
                "Delivered Output",
                format_compact(total_output_tokens, " tok"),
                f"Avg {format_compact(mean_output_tokens, ' tok')} per success",
                "green",
            ),
        ]
    )

    burst_line = (
        f"{len(phases)} distinct load phases detected with a largest recovery gap of {largest_gap_s:.1f}s."
        if len(phases) > 1
        else "The run forms one dominant load phase without a major recovery gap."
    )
    insight_cards_html = "".join(
        [
            render_insight_card(
                "Peak Impact",
                f"{format_compact(peak_tp, ' tok/s')} at {peak_t:.1f}s",
                f"Peak output landed while {format_compact(active_at_peak_tp)} generators were simultaneously active and SSE cadence peaked at {format_compact(peak_sse, ' msg/s')}.",
            ),
            render_insight_card(
                "Stability",
                f"{near_peak_share:.0%} near-peak residency",
                f"Throughput stayed above 90% of peak for {format_duration(near_peak_runtime_s)} across {format_duration(live_runtime_s)} of active emission time.",
            ),
            render_insight_card(
                "Wave Shape",
                f"{max(1, len(phases))} burst phases",
                burst_line,
            ),
        ]
    )

    peak_headline = f"{format_compact(peak_tp, ' tok/s')} at {peak_t:.1f}s"
    peak_story = (
        f"At the apex of the run, output throughput hit {format_compact(peak_tp, ' tok/s')} while "
        f"{format_compact(active_at_peak_tp)} requests were actively generating. The transport layer simultaneously emitted "
        f"up to {format_compact(peak_sse, ' msg/s')} SSE messages per second, which makes the peak easy to correlate across all three views."
    )

    meta_rows_html = "".join(
        [
            render_meta_row("Input snapshot", args.input.name),
            render_meta_row("Artifact group", source_label),
            render_meta_row("Successful requests", f"{success_count:,}"),
            render_meta_row("Errors", f"{error_count:,}"),
            render_meta_row("Runtime", f"{duration_s:,.1f}s"),
            render_meta_row("Mean throughput", format_compact(mean_tp, " tok/s")),
            render_meta_row("Mean SSE rate", format_compact(mean_sse, " msg/s")),
            render_meta_row("Mean active generators", format_compact(mean_active)),
        ]
    )

    if phases:
        phase_cards_html = "".join(
            render_phase_card(index + 1, phase, peak_tp)
            for index, phase in enumerate(phases[:5])
        )
    else:
        phase_cards_html = """
            <article class="phase-card">
              <div class="phase-topline">
                <span class="phase-name">No burst phases</span>
                <span class="phase-window">n/a</span>
              </div>
              <div class="phase-meter"><span style="width:18%"></span></div>
              <p class="signal-copy">The run never crossed the live-throughput threshold required to extract burst windows.</p>
            </article>
        """

    rendered_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    footer_copy = (
        f"Rendered from {esc(args.input.name)} on {esc(rendered_at)}. "
        "Output throughput uses AIPerf's event-based token-dispersion sweep. "
        "SSE rate is derived from chunk cadence between TTFT and request end."
    )

    page_html = PAGE_TEMPLATE.substitute(
        page_title=esc(f"NVIDIA AIPerf | {args.title}"),
        font_body=FONT_BODY,
        font_display=FONT_DISPLAY,
        hero_title=esc(args.title),
        hero_copy=esc(hero_copy),
        chips_html=chips_html,
        metric_cards_html=metric_cards_html,
        insight_cards_html=insight_cards_html,
        throughput_plot_html=throughput_plot_html,
        peak_headline=esc(peak_headline),
        peak_story=esc(peak_story),
        meta_rows_html=meta_rows_html,
        phase_cards_html=phase_cards_html,
        sse_plot_html=sse_plot_html,
        active_plot_html=active_plot_html,
        footer_copy=footer_copy,
    )

    output.write_text(page_html, encoding="utf-8")
    print(output)
    print(f"mean_sse_messages_per_sec={mean_sse:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
