# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report generation for synthesized Claude Code datasets."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import orjson
import plotly.graph_objects as go
import plotly.io as pio
from pydantic import Field
from rich.console import Console
from rich.table import Table

from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.claude_code_gen.models import DatasetManifest
from aiperf.plot.constants import NVIDIA_GREEN


def _pct_error(target: float, observed: float) -> float:
    if target == 0:
        return 0.0
    return abs(observed - target) / target * 100.0


# ---------------------------------------------------------------------------
# Parsed row from JSONL
# ---------------------------------------------------------------------------
class ParsedTurn(AIPerfBaseModel):
    session_id: str = Field(description="Session identifier")
    input_length: int = Field(description="Total input token count for this turn")
    output_length: int = Field(description="Output token count for this turn")
    hash_ids: list[int] = Field(description="KV cache block hash IDs")
    delay_ms: float = Field(
        description="Inter-turn delay in milliseconds, 0.0 for first turn"
    )


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------
class PercentileStats(AIPerfBaseModel):
    """Descriptive statistics with percentiles."""

    count: int = Field(description="Number of observations")
    mean: float = Field(description="Arithmetic mean")
    std: float = Field(description="Standard deviation")
    median: float = Field(description="Median (50th percentile)")
    p05: float = Field(description="5th percentile")
    p25: float = Field(description="25th percentile")
    p75: float = Field(description="75th percentile")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")


class TargetComparison(AIPerfBaseModel):
    """Observed stats vs a target mean/median for one metric."""

    metric_name: str = Field(description="Name of the compared metric")
    target_mean: float | None = Field(
        default=None, description="Expected mean from config"
    )
    target_median: float | None = Field(
        default=None, description="Expected median from config"
    )
    observed: PercentileStats = Field(description="Observed descriptive statistics")
    pct_error_mean: float | None = Field(
        default=None, description="Percentage error between observed and target mean"
    )


class ReportData(AIPerfBaseModel):
    """Full report payload."""

    session_count: int = Field(description="Number of sessions in the dataset")
    total_turns: int = Field(description="Total number of turns across all sessions")
    comparisons: list[TargetComparison] = Field(
        description="Target-vs-observed metric comparisons"
    )
    hash_id_block_stats: PercentileStats = Field(
        description="Hash ID block count statistics"
    )
    request_latency_stats: PercentileStats = Field(
        description="Per-turn request latency statistics"
    )
    session_duration_min_stats: PercentileStats = Field(
        description="Session duration in minutes statistics"
    )
    prefix_length_stats: PercentileStats | None = Field(
        default=None, description="Prefix length statistics"
    )
    unique_prompt_length_stats: PercentileStats | None = Field(
        default=None, description="Unique prompt length statistics"
    )
    prefix_ratio_stats: PercentileStats | None = Field(
        default=None, description="Prefix ratio statistics"
    )
    sequential_cache_hit_rate_stats: PercentileStats | None = Field(
        default=None, description="Sequential cache hit rate statistics"
    )
    per_session_cache_hit_rate_stats: PercentileStats | None = Field(
        default=None, description="Per-session cache hit rate statistics"
    )


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[ParsedTurn]:
    turns: list[ParsedTurn] = []
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = orjson.loads(line)
            turns.append(
                ParsedTurn(
                    session_id=row["session_id"],
                    input_length=row["input_length"],
                    output_length=row["output_length"],
                    hash_ids=row.get("hash_ids", []),
                    delay_ms=row.get("delay", 0.0),
                )
            )
    return turns


def group_sessions(turns: list[ParsedTurn]) -> dict[str, list[ParsedTurn]]:
    sessions: dict[str, list[ParsedTurn]] = {}
    for t in turns:
        sessions.setdefault(t.session_id, []).append(t)
    return sessions


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------
def _percentile_stats(arr: np.ndarray) -> PercentileStats:
    return PercentileStats(
        count=len(arr),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        median=float(np.median(arr)),
        p05=float(np.percentile(arr, 5)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


def extract_metrics(
    sessions: dict[str, list[ParsedTurn]],
    prefill_tps: float = 20_000,
    decode_tps: float = 60,
) -> dict[str, np.ndarray]:
    initial_context: list[float] = []
    new_tokens_per_turn: list[float] = []
    generation_length: list[float] = []
    inter_turn_delay: list[float] = []
    turns_per_session: list[float] = []
    total_isl: list[float] = []
    total_osl: list[float] = []
    hash_id_block_count: list[float] = []
    request_latency_ms: list[float] = []
    session_duration_min: list[float] = []

    for turns in sessions.values():
        turns_per_session.append(float(len(turns)))
        session_lat = 0.0
        for i, t in enumerate(turns):
            total_isl.append(float(t.input_length))
            total_osl.append(float(t.output_length))
            generation_length.append(float(t.output_length))
            hash_id_block_count.append(float(len(t.hash_ids)))

            lat = (t.input_length / prefill_tps + t.output_length / decode_tps) * 1000
            request_latency_ms.append(lat)
            session_lat += t.delay_ms + lat

            if i == 0:
                initial_context.append(float(t.input_length))
            else:
                prev = turns[i - 1]
                new_tok = t.input_length - prev.input_length - prev.output_length
                new_tokens_per_turn.append(float(max(new_tok, 0)))
                inter_turn_delay.append(t.delay_ms / 1000.0)

        session_duration_min.append(session_lat / 1000.0 / 60.0)

    return {
        "initial_context": np.array(initial_context),
        "new_tokens_per_turn": np.array(new_tokens_per_turn),
        "generation_length": np.array(generation_length),
        "inter_turn_delay_s": np.array(inter_turn_delay),
        "turns_per_session": np.array(turns_per_session),
        "total_isl": np.array(total_isl),
        "total_osl": np.array(total_osl),
        "hash_id_block_count": np.array(hash_id_block_count),
        "request_latency_ms": np.array(request_latency_ms),
        "request_latency_s": np.array(request_latency_ms) / 1000.0,
        "session_duration_min": np.array(session_duration_min),
    }


def extract_cache_metrics(
    sessions: dict[str, list[ParsedTurn]],
    block_size: int = 512,
) -> dict[str, np.ndarray]:
    """Compute prefix/cache-reuse statistics from hash_ids.

    Args:
        sessions: Turns grouped by session id (insertion-ordered).
        block_size: KV cache page size in tokens.

    Returns:
        Dict with 5 metric arrays keyed by metric name.
    """
    # Flatten all turns in session order
    all_turns: list[ParsedTurn] = []
    session_boundaries: list[int] = []
    for turns in sessions.values():
        session_boundaries.append(len(all_turns))
        all_turns.extend(turns)
    session_boundary_set = set(session_boundaries)

    # Build Counter[(position, hash_id)] across entire dataset
    hash_counter: Counter[tuple[int, int]] = Counter()
    for t in all_turns:
        for pos, hid in enumerate(t.hash_ids):
            hash_counter[(pos, hid)] += 1
    repeated = {k for k, v in hash_counter.items() if v > 1}

    prefix_length: list[float] = []
    unique_prompt_length: list[float] = []
    prefix_ratio: list[float] = []
    sequential_cache_hit_rate: list[float] = []
    per_session_cache_hit_rate: list[float] = []

    # Sequential (global) cache hit rate
    global_seen: set[int] = set()
    # Per-session cache hit rate
    session_seen: set[int] = set()

    for idx, t in enumerate(all_turns):
        hash_ids = t.hash_ids
        il = t.input_length

        # Prefix length from repeated (position, hash_id) pairs
        if hash_ids and all((pos, hid) in repeated for pos, hid in enumerate(hash_ids)):
            pl = il
        else:
            repeated_count = sum(
                1 for pos, hid in enumerate(hash_ids) if (pos, hid) in repeated
            )
            pl = min(repeated_count * block_size, il)

        prefix_length.append(float(pl))
        upl = max(il - pl, 0)
        unique_prompt_length.append(float(upl))
        prefix_ratio.append(pl / il if il > 0 else 0.0)

        # Sequential cache hit rate (global)
        if hash_ids:
            first_unseen = len(hash_ids)
            for i, hid in enumerate(hash_ids):
                if hid not in global_seen:
                    first_unseen = i
                    break
            sequential_cache_hit_rate.append(first_unseen / len(hash_ids))
            global_seen.update(hash_ids)
        else:
            sequential_cache_hit_rate.append(0.0)

        # Per-session cache hit rate
        if idx in session_boundary_set:
            session_seen = set()

        if hash_ids:
            first_unseen = len(hash_ids)
            for i, hid in enumerate(hash_ids):
                if hid not in session_seen:
                    first_unseen = i
                    break
            per_session_cache_hit_rate.append(first_unseen / len(hash_ids))
            session_seen.update(hash_ids)
        else:
            per_session_cache_hit_rate.append(0.0)

    return {
        "prefix_length": np.array(prefix_length),
        "unique_prompt_length": np.array(unique_prompt_length),
        "prefix_ratio": np.array(prefix_ratio),
        "sequential_cache_hit_rate": np.array(sequential_cache_hit_rate),
        "per_session_cache_hit_rate": np.array(per_session_cache_hit_rate),
    }


# ---------------------------------------------------------------------------
# Target comparison table
# ---------------------------------------------------------------------------
# (metric_key, target_mean, target_median, display_name)
_TARGET_TABLE: list[tuple[str, float | None, float | None, str]] = [
    ("initial_context", 67_000, 54_000, "Initial Context (tokens)"),
    ("new_tokens_per_turn", 4_500, 2_100, "New Tokens/Turn"),
    ("generation_length", 600, 350, "Generation Length (tokens)"),
    ("inter_turn_delay_s", None, None, "Inter-Turn Delay (s)"),
    ("turns_per_session", None, None, "Turns/Session"),
]


def build_report_data(
    metrics: dict[str, np.ndarray],
    manifest: DatasetManifest | None = None,
) -> ReportData:
    comparisons: list[TargetComparison] = []
    for key, t_mean, t_median, display in _TARGET_TABLE:
        arr = metrics.get(key)
        if arr is None or len(arr) == 0:
            continue
        obs = _percentile_stats(arr)
        pct_err = _pct_error(t_mean, obs.mean) if t_mean is not None else None
        comparisons.append(
            TargetComparison(
                metric_name=display,
                target_mean=t_mean,
                target_median=t_median,
                observed=obs,
                pct_error_mean=round(pct_err, 2) if pct_err is not None else None,
            )
        )

    cache_fields: dict[str, PercentileStats | None] = {}
    for field_name, metric_key in [
        ("prefix_length_stats", "prefix_length"),
        ("unique_prompt_length_stats", "unique_prompt_length"),
        ("prefix_ratio_stats", "prefix_ratio"),
        ("sequential_cache_hit_rate_stats", "sequential_cache_hit_rate"),
        ("per_session_cache_hit_rate_stats", "per_session_cache_hit_rate"),
    ]:
        arr = metrics.get(metric_key)
        cache_fields[field_name] = (
            _percentile_stats(arr) if arr is not None and len(arr) > 0 else None
        )

    return ReportData(
        session_count=len(metrics.get("turns_per_session", np.array([]))),
        total_turns=int(metrics.get("total_isl", np.array([])).shape[0]),
        comparisons=comparisons,
        hash_id_block_stats=_percentile_stats(metrics["hash_id_block_count"]),
        request_latency_stats=_percentile_stats(metrics["request_latency_ms"]),
        session_duration_min_stats=_percentile_stats(metrics["session_duration_min"]),
        **cache_fields,
    )


# ---------------------------------------------------------------------------
# Text report (Rich)
# ---------------------------------------------------------------------------
def render_text_report(data: ReportData) -> str:
    console = Console(width=140, record=True)

    console.print()
    console.print("[bold]Dataset Report[/bold]")
    console.print(f"Sessions: {data.session_count:,}   Turns: {data.total_turns:,}")
    console.print()

    # Table 1: Target vs Observed
    t1 = Table(title="Target vs Observed")
    t1.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    for col in [
        "Target Mean",
        "Target Median",
        "Obs Mean",
        "Obs Median",
        "p05",
        "p25",
        "p75",
        "p95",
        "p99",
        "% Err",
    ]:
        t1.add_column(col, justify="right", style="green", no_wrap=True)

    for c in data.comparisons:
        t1.add_row(
            c.metric_name,
            f"{c.target_mean:,.0f}" if c.target_mean is not None else "-",
            f"{c.target_median:,.0f}" if c.target_median is not None else "-",
            f"{c.observed.mean:,.1f}",
            f"{c.observed.median:,.1f}",
            f"{c.observed.p05:,.1f}",
            f"{c.observed.p25:,.1f}",
            f"{c.observed.p75:,.1f}",
            f"{c.observed.p95:,.1f}",
            f"{c.observed.p99:,.1f}",
            f"{c.pct_error_mean:.1f}%" if c.pct_error_mean is not None else "-",
        )
    console.print(t1)
    console.print()

    # Table 2: Summary
    t2 = Table(title="Summary Statistics")
    t2.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    for col in ["Mean", "Median", "p05", "p25", "p75", "p95", "p99"]:
        t2.add_column(col, justify="right", style="green", no_wrap=True)

    def _add_stats_row(name: str, s: PercentileStats) -> None:
        t2.add_row(
            name,
            f"{s.mean:,.1f}",
            f"{s.median:,.1f}",
            f"{s.p05:,.1f}",
            f"{s.p25:,.1f}",
            f"{s.p75:,.1f}",
            f"{s.p95:,.1f}",
            f"{s.p99:,.1f}",
        )

    _add_stats_row("Hash ID Blocks/Turn", data.hash_id_block_stats)
    _add_stats_row("Request Latency (ms)", data.request_latency_stats)
    _add_stats_row("Session Duration (min)", data.session_duration_min_stats)
    console.print(t2)
    console.print()

    _render_cache_table(console, data)

    return console.export_text()


def _render_cache_table(console: Console, data: ReportData) -> None:
    """Print cache/prefix statistics table if data is available."""
    cache_rows = [
        ("Prefix Length", data.prefix_length_stats),
        ("Unique Prompt Length", data.unique_prompt_length_stats),
        ("Prefix Ratio", data.prefix_ratio_stats),
        ("Sequential Cache Hit Rate", data.sequential_cache_hit_rate_stats),
        ("Per-Session Cache Hit Rate", data.per_session_cache_hit_rate_stats),
    ]
    if not any(s for _, s in cache_rows):
        return

    t = Table(title="Cache / Prefix Statistics")
    t.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    for col in ["Mean", "Median", "p05", "p25", "p75", "p95", "p99"]:
        t.add_column(col, justify="right", style="green", no_wrap=True)

    for name, stats in cache_rows:
        if stats is None:
            continue
        fmt = ".4f" if stats.mean <= 1.0 else ",.1f"
        t.add_row(
            name,
            f"{stats.mean:{fmt}}",
            f"{stats.median:{fmt}}",
            f"{stats.p05:{fmt}}",
            f"{stats.p25:{fmt}}",
            f"{stats.p75:{fmt}}",
            f"{stats.p95:{fmt}}",
            f"{stats.p99:{fmt}}",
        )
    console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# Plot report (Plotly)
# ---------------------------------------------------------------------------
# (metric_key, title, row, col) -- grid positions for the dashboard
_HISTOGRAM_PLOTS: list[tuple[str, str, int, int]] = [
    ("total_isl", "ISL Distribution", 1, 1),
    ("total_osl", "OSL Distribution", 1, 2),
    ("initial_context", "Initial Context (Turn 0 Input Length)", 1, 3),
    ("new_tokens_per_turn", "New Tokens Per Turn", 1, 4),
    ("generation_length", "Generation Length", 2, 1),
    ("inter_turn_delay_s", "Inter-Turn Delay (s)", 2, 2),
    ("turns_per_session", "Turns Per Session", 2, 3),
    ("hash_id_block_count", "Hash ID Blocks Per Turn", 2, 4),
    ("request_latency_s", "Estimated Request Latency (s)", 3, 1),
    ("session_duration_min", "Estimated Session Duration (min)", 3, 2),
]

_CACHE_HISTOGRAM_PLOTS: list[tuple[str, str]] = [
    ("prefix_length", "Prefix Length (tokens)"),
    ("unique_prompt_length", "Unique Prompt Length (tokens)"),
    ("prefix_ratio", "Prefix Ratio"),
    ("sequential_cache_hit_rate", "Sequential Cache Hit Rate"),
]


_DASHBOARD_CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
     background:#f5f5f5;padding:20px}
h1{text-align:center;margin-bottom:20px;color:#333}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.card{background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.12);
      cursor:pointer;position:relative;overflow:hidden;transition:box-shadow .2s}
.card:hover{box-shadow:0 4px 12px rgba(0,0,0,.2)}
.card.span2{grid-column:span 2}
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:1000;
         align-items:center;justify-content:center;padding:24px}
.overlay.active{display:flex}
.overlay-inner{background:#fff;border-radius:8px;width:100%;height:100%;position:relative}
.close-btn{position:absolute;top:8px;right:12px;font-size:24px;cursor:pointer;
           z-index:1001;background:#fff;border:none;border-radius:50%;
           width:32px;height:32px;display:flex;align-items:center;justify-content:center;
           box-shadow:0 1px 4px rgba(0,0,0,.2)}
"""

_DASHBOARD_JS = """\
document.addEventListener('DOMContentLoaded',function(){
  var overlay=document.getElementById('overlay');
  var inner=document.getElementById('overlay-inner');
  var closeBtn=document.getElementById('close-btn');
  document.querySelectorAll('.card').forEach(function(card){
    card.addEventListener('click',function(){
      var plotDiv=card.querySelector('.js-plotly-plot');
      if(!plotDiv)return;
      var clone=plotDiv.cloneNode(true);
      clone.id='overlay-plot';
      clone.style.width='100%';
      clone.style.height='100%';
      inner.querySelectorAll('.js-plotly-plot').forEach(function(el){el.remove()});
      inner.appendChild(clone);
      overlay.classList.add('active');
      Plotly.newPlot(clone,plotDiv.data,
        Object.assign({},plotDiv.layout,{width:null,height:null,autosize:true}),
        {responsive:true});
    });
  });
  function closeOverlay(){
    overlay.classList.remove('active');
    inner.querySelectorAll('.js-plotly-plot').forEach(function(el){el.remove()});
  }
  closeBtn.addEventListener('click',function(e){e.stopPropagation();closeOverlay()});
  overlay.addEventListener('click',function(e){if(e.target===overlay)closeOverlay()});
  document.addEventListener('keydown',function(e){if(e.key==='Escape')closeOverlay()});
});
"""


def render_plot_report(
    metrics: dict[str, np.ndarray],
    sessions: dict[str, list[ParsedTurn]],
    output_dir: Path,
) -> Path:
    figures: list[tuple[go.Figure, str]] = []

    for key, title, _row, _col in _HISTOGRAM_PLOTS:
        arr = metrics.get(key)
        if arr is None or len(arr) == 0:
            continue
        fig = go.Figure(
            data=[go.Histogram(x=arr, marker_color=NVIDIA_GREEN, showlegend=False)],
            layout=go.Layout(
                title=title,
                xaxis_title=key.replace("_", " ").title(),
                yaxis_title="Count",
                template="plotly_white",
                height=320,
                margin=dict(l=50, r=20, t=40, b=40),
            ),
        )
        figures.append((fig, ""))

    # Cache histograms
    for key, title in _CACHE_HISTOGRAM_PLOTS:
        arr = metrics.get(key)
        if arr is None or len(arr) == 0:
            continue
        fig = go.Figure(
            data=[go.Histogram(x=arr, marker_color=NVIDIA_GREEN, showlegend=False)],
            layout=go.Layout(
                title=title,
                xaxis_title=key.replace("_", " ").title(),
                yaxis_title="Count",
                template="plotly_white",
                height=320,
                margin=dict(l=50, r=20, t=40, b=40),
            ),
        )
        figures.append((fig, ""))

    # Context growth overlay (~10 sessions)
    sample_ids = list(sessions.keys())[:10]
    if sample_ids:
        ctx_fig = go.Figure(
            layout=go.Layout(
                title="Context Growth (Sample Sessions)",
                xaxis_title="Turn Index",
                yaxis_title="Input Length (tokens)",
                template="plotly_white",
                height=320,
                margin=dict(l=50, r=20, t=40, b=40),
            )
        )
        for sid in sample_ids:
            turns = sessions[sid]
            xs = list(range(len(turns)))
            ys = [t.input_length for t in turns]
            ctx_fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=sid[:8]))
        figures.append((ctx_fig, "span2"))

    # Per-session cache evolution (~10 sessions)
    if sample_ids and "per_session_cache_hit_rate" in metrics:
        cache_evo_fig = go.Figure(
            layout=go.Layout(
                title="Per-Session Cache Hit Rate Evolution",
                xaxis_title="Turn Index",
                yaxis_title="Cache Hit Rate",
                template="plotly_white",
                height=320,
                margin=dict(l=50, r=20, t=40, b=40),
            )
        )
        for sid in sample_ids:
            turns = sessions[sid]
            session_seen: set[int] = set()
            rates: list[float] = []
            for t in turns:
                if t.hash_ids:
                    first_unseen = len(t.hash_ids)
                    for i, hid in enumerate(t.hash_ids):
                        if hid not in session_seen:
                            first_unseen = i
                            break
                    rates.append(first_unseen / len(t.hash_ids))
                    session_seen.update(t.hash_ids)
                else:
                    rates.append(0.0)
            cache_evo_fig.add_trace(
                go.Scatter(
                    x=list(range(len(turns))),
                    y=rates,
                    mode="lines",
                    name=sid[:8],
                )
            )
        figures.append((cache_evo_fig, "span2"))

    # Cache hit rate vs input length scatter
    seq_hr = metrics.get("sequential_cache_hit_rate")
    total_isl = metrics.get("total_isl")
    if seq_hr is not None and total_isl is not None and len(seq_hr) > 0:
        scatter_fig = go.Figure(
            data=[
                go.Scattergl(
                    x=total_isl,
                    y=seq_hr,
                    mode="markers",
                    marker=dict(color=NVIDIA_GREEN, size=3, opacity=0.5),
                    showlegend=False,
                )
            ],
            layout=go.Layout(
                title="Cache Hit Rate vs Input Length",
                xaxis_title="Input Length (tokens)",
                yaxis_title="Sequential Cache Hit Rate",
                template="plotly_white",
                height=320,
                margin=dict(l=50, r=20, t=40, b=40),
            ),
        )
        figures.append((scatter_fig, "span2"))

    # Build HTML
    plotly_js_cdn = "https://cdn.plot.ly/plotly-2.35.2.min.js"
    cards_html: list[str] = []
    for i, (fig, css_class) in enumerate(figures):
        div_id = f"plot-{i}"
        inner_html = pio.to_html(
            fig, full_html=False, include_plotlyjs=False, div_id=div_id
        )
        cls = f"card {css_class}".strip()
        cards_html.append(f'<div class="{cls}">{inner_html}</div>')

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Dataset Report</title>
<script src="{plotly_js_cdn}"></script>
<style>{_DASHBOARD_CSS}</style>
</head>
<body>
<h1>Dataset Report</h1>
<div class="grid">
{"".join(cards_html)}
</div>
<div class="overlay" id="overlay">
<div class="overlay-inner" id="overlay-inner">
<button class="close-btn" id="close-btn">&times;</button>
</div>
</div>
<script>{_DASHBOARD_JS}</script>
</body>
</html>"""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"
    report_path.write_text(html)
    return report_path


def _print_report_to_console(data: ReportData) -> None:
    """Print report tables directly to the terminal (no double-rendering)."""
    console = Console(width=140)
    console.print()
    console.print(
        f"[bold]Dataset Report[/bold]  Sessions: {data.session_count:,}  Turns: {data.total_turns:,}"
    )
    console.print()

    t1 = Table(title="Target vs Observed")
    t1.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    for col in [
        "Target Mean",
        "Target Median",
        "Obs Mean",
        "Obs Median",
        "p05",
        "p25",
        "p75",
        "p95",
        "p99",
        "% Err",
    ]:
        t1.add_column(col, justify="right", style="green", no_wrap=True)
    for c in data.comparisons:
        t1.add_row(
            c.metric_name,
            f"{c.target_mean:,.0f}" if c.target_mean is not None else "-",
            f"{c.target_median:,.0f}" if c.target_median is not None else "-",
            f"{c.observed.mean:,.1f}",
            f"{c.observed.median:,.1f}",
            f"{c.observed.p05:,.1f}",
            f"{c.observed.p25:,.1f}",
            f"{c.observed.p75:,.1f}",
            f"{c.observed.p95:,.1f}",
            f"{c.observed.p99:,.1f}",
            f"{c.pct_error_mean:.1f}%" if c.pct_error_mean is not None else "-",
        )
    console.print(t1)
    console.print()

    t2 = Table(title="Summary Statistics")
    t2.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    for col in ["Mean", "Median", "p05", "p25", "p75", "p95", "p99"]:
        t2.add_column(col, justify="right", style="green", no_wrap=True)

    for name, stats in [
        ("Hash ID Blocks/Turn", data.hash_id_block_stats),
        ("Request Latency (ms)", data.request_latency_stats),
        ("Session Duration (min)", data.session_duration_min_stats),
    ]:
        t2.add_row(
            name,
            f"{stats.mean:,.1f}",
            f"{stats.median:,.1f}",
            f"{stats.p05:,.1f}",
            f"{stats.p25:,.1f}",
            f"{stats.p75:,.1f}",
            f"{stats.p95:,.1f}",
            f"{stats.p99:,.1f}",
        )
    console.print(t2)
    console.print()

    _render_cache_table(console, data)


# ---------------------------------------------------------------------------
# Cache explorer visualization
# ---------------------------------------------------------------------------
def _classify_turn_blocks(
    hash_ids: list[int],
    prev_hash_id_set: set[int] | None,
    l1_blocks: int,
) -> list[dict]:
    """Classify each block in a turn by layer and cache status.

    Returns a list of dicts with keys: pos, hash_id, layer, status.
    """
    blocks: list[dict] = []
    for pos, hid in enumerate(hash_ids):
        if pos < l1_blocks:
            blocks.append(
                {"pos": pos, "hash_id": hid, "layer": "L1", "status": "cached"}
            )
        elif prev_hash_id_set is None:
            blocks.append(
                {"pos": pos, "hash_id": hid, "layer": "session", "status": "new"}
            )
        elif hid in prev_hash_id_set:
            blocks.append(
                {"pos": pos, "hash_id": hid, "layer": "session", "status": "cached"}
            )
        else:
            blocks.append({"pos": pos, "hash_id": hid, "layer": "L3", "status": "new"})
    return blocks


def write_cache_structure(
    sessions: dict[str, list[ParsedTurn]],
    manifest: DatasetManifest | None,
    output_dir: Path,
) -> dict:
    """Generate cache_structure.json with per-session, per-turn block classification."""
    l1_tokens = 32_000
    block_size = 512
    if manifest:
        block_size = manifest.block_size
        l1_tokens = manifest.generation_params.cache.layer1_tokens
    l1_blocks = math.ceil(l1_tokens / block_size) if block_size > 0 else 0

    session_data: list[dict] = []
    for i, (sid, turns) in enumerate(sessions.items()):
        if i >= 50:
            break
        turn_data: list[dict] = []
        prev_hash_id_set: set[int] | None = None
        for t in turns:
            classified = _classify_turn_blocks(t.hash_ids, prev_hash_id_set, l1_blocks)

            # Run-length encode consecutive blocks with same (layer, status)
            segments: list[dict] = []
            for block in classified:
                key = (block["layer"], block["status"])
                if segments and (segments[-1]["layer"], segments[-1]["status"]) == key:
                    segments[-1]["count"] += 1
                else:
                    segments.append(
                        {
                            "start": block["pos"],
                            "count": 1,
                            "layer": block["layer"],
                            "status": block["status"],
                        }
                    )

            turn_data.append(
                {
                    "turn_index": turns.index(t),
                    "input_length": t.input_length,
                    "output_length": t.output_length,
                    "num_blocks": len(t.hash_ids),
                    "segments": segments,
                }
            )
            prev_hash_id_set = set(t.hash_ids)

        session_data.append({"session_id": sid, "turns": turn_data})

    payload = {
        "block_size": block_size,
        "l1_blocks": l1_blocks,
        "sessions": session_data,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cache_structure.json"
    out_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return payload


_CACHE_EXPLORER_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Cache Explorer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
     background:#1a1a2e;color:#e0e0e0;padding:20px}
h1{text-align:center;margin-bottom:8px;color:#76b900;font-size:1.4em}
h2{font-size:1.1em;margin:12px 0 6px;color:#ccc}
.legend{display:flex;gap:16px;justify-content:center;margin:8px 0 16px;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:4px;font-size:12px}
.legend-swatch{width:14px;height:14px;border-radius:2px}
.stats{text-align:center;font-size:12px;color:#999;margin-bottom:12px}
#overview,#detail{background:#16213e;border-radius:8px;padding:12px;margin-bottom:16px}
.session-row{cursor:pointer;opacity:0.85}
.session-row:hover,.session-row.selected{opacity:1}
.tooltip{position:absolute;background:#0f3460;color:#e0e0e0;padding:6px 10px;
         border-radius:4px;font-size:11px;pointer-events:none;white-space:pre;
         box-shadow:0 2px 8px rgba(0,0,0,.4);z-index:100}
svg text{fill:#ccc;font-size:11px}
</style>
</head>
<body>
<h1>Cache Explorer</h1>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#76b900"></div>L1 cached</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#4a90d9"></div>Session cached</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#4a90d9;border:2px dashed #fff"></div>Session new</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f97316"></div>L3 new</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#9ca3af"></div>Output</div>
</div>
<div id="stats" class="stats"></div>
<h2>Session Overview</h2>
<div id="overview"></div>
<h2>Turn Detail <span id="selected-session" style="color:#76b900"></span></h2>
<div id="detail"></div>
<script>
const COLORS={
  'L1-cached':'#76b900','session-cached':'#4a90d9','session-new':'#4a90d9',
  'L3-new':'#f97316','output':'#9ca3af'
};
function segColor(s){return COLORS[s.layer+'-'+s.status]||'#555'}
function segDash(s){return s.layer==='session'&&s.status==='new'}

let DATA;
const tooltip=d3.select('body').append('div').attr('class','tooltip').style('display','none');

DATA=__INLINE_DATA__;
renderOverview(DATA);
if(DATA.sessions.length>0)selectSession(0);

function renderOverview(data){
  const margin={top:10,right:20,bottom:30,left:80};
  const W=Math.min(document.getElementById('overview').clientWidth-24,1200);
  const rowH=18,gap=2;
  const bs=data.block_size;
  const sessions=data.sessions;
  const H=sessions.length*(rowH+gap)+margin.top+margin.bottom;
  const maxTok=d3.max(sessions,s=>{
    const last=s.turns[s.turns.length-1];
    return last?last.num_blocks*bs:0;
  })||1;
  const x=d3.scaleLinear().domain([0,maxTok]).range([margin.left,W-margin.right]);
  const svg=d3.select('#overview').append('svg').attr('width',W).attr('height',H);
  sessions.forEach((sess,i)=>{
    const y=margin.top+i*(rowH+gap);
    const last=sess.turns[sess.turns.length-1];
    if(!last)return;
    const g=svg.append('g').attr('class','session-row').attr('data-idx',i)
      .on('click',()=>selectSession(i));
    last.segments.forEach(seg=>{
      const rect=g.append('rect')
        .attr('x',x(seg.start*bs)).attr('y',y)
        .attr('width',Math.max(1,x((seg.start+seg.count)*bs)-x(seg.start*bs)))
        .attr('height',rowH).attr('fill',segColor(seg)).attr('rx',1);
      if(segDash(seg))rect.attr('stroke','#fff').attr('stroke-dasharray','3,2').attr('fill-opacity',0.6);
    });
    svg.append('text').attr('x',margin.left-4).attr('y',y+rowH/2+4)
      .attr('text-anchor','end').attr('font-size','10px').text(sess.session_id.slice(0,8));
  });
  svg.append('g').attr('transform',`translate(0,${H-margin.bottom})`)
    .call(d3.axisBottom(x).ticks(8).tickFormat(d=>d>=1000?(d/1000).toFixed(0)+'K':d))
    .selectAll('text').style('fill','#ccc');
  svg.append('text').attr('x',W/2).attr('y',H-2).attr('text-anchor','middle')
    .attr('font-size','11px').text('Tokens');
}

function selectSession(idx){
  d3.selectAll('.session-row').classed('selected',false).attr('opacity',0.85);
  d3.selectAll(`.session-row[data-idx="${idx}"]`).classed('selected',true).attr('opacity',1);
  const sess=DATA.sessions[idx];
  d3.select('#selected-session').text(sess.session_id.slice(0,12));
  renderDetail(sess);
  renderStats(sess);
}

function renderStats(sess){
  const bs=DATA.block_size;
  const t0=sess.turns[0];
  if(!t0){d3.select('#stats').text('');return;}
  const l1Tok=DATA.l1_blocks*bs;
  const sessPrefixTok=(t0.num_blocks-DATA.l1_blocks)*bs;
  const lines=[`Turn 0: L1=${l1Tok.toLocaleString()} tok, session prefix=${Math.max(0,sessPrefixTok).toLocaleString()} tok`];
  sess.turns.forEach((t,i)=>{
    if(i===0)return;
    const cached=t.segments.filter(s=>s.status==='cached').reduce((a,s)=>a+s.count,0);
    const rate=t.num_blocks>0?(cached/t.num_blocks*100).toFixed(1):'0.0';
    lines.push(`Turn ${i}: cache hit ${rate}%`);
  });
  d3.select('#stats').html(lines.join(' &middot; '));
}

function renderDetail(sess){
  const container=d3.select('#detail');
  container.selectAll('*').remove();
  const margin={top:10,right:20,bottom:30,left:60};
  const W=Math.min(container.node().clientWidth-24,1200);
  const bs=DATA.block_size;
  const rowH=22,gap=4;
  const turns=sess.turns;
  const H=turns.length*(rowH+gap)+margin.top+margin.bottom;
  const maxTok=(d3.max(turns,t=>t.num_blocks)||1)*bs;
  const x=d3.scaleLinear().domain([0,maxTok+d3.max(turns,t=>t.output_length)])
    .range([margin.left,W-margin.right]);
  const svg=container.append('svg').attr('width',W).attr('height',H);

  turns.forEach((t,i)=>{
    const y=margin.top+i*(rowH+gap);
    const g=svg.append('g');
    t.segments.forEach(seg=>{
      const rect=g.append('rect')
        .attr('x',x(seg.start*bs)).attr('y',y)
        .attr('width',Math.max(1,x((seg.start+seg.count)*bs)-x(seg.start*bs)))
        .attr('height',rowH).attr('fill',segColor(seg)).attr('rx',1);
      if(segDash(seg))rect.attr('stroke','#fff').attr('stroke-dasharray','3,2').attr('fill-opacity',0.6);
      rect.on('mouseover',(ev)=>{
        const tokStart=seg.start*bs;
        const tokEnd=(seg.start+seg.count)*bs;
        tooltip.style('display','block')
          .html(`Turn ${t.turn_index} | ${tokStart.toLocaleString()}-${tokEnd.toLocaleString()} tokens\\n`+
                `Layer: ${seg.layer} | Status: ${seg.status}\\n`+
                `${seg.count} blocks (${(seg.count*bs).toLocaleString()} tokens)`);
      }).on('mousemove',(ev)=>{
        tooltip.style('left',(ev.pageX+12)+'px').style('top',(ev.pageY-28)+'px');
      }).on('mouseout',()=>tooltip.style('display','none'));
    });
    // Output indicator
    g.append('rect').attr('x',x(t.num_blocks*bs)).attr('y',y+4)
      .attr('width',Math.max(1,x(t.num_blocks*bs+t.output_length)-x(t.num_blocks*bs)))
      .attr('height',rowH-8).attr('fill','#9ca3af').attr('rx',1).attr('opacity',0.5);
    svg.append('text').attr('x',margin.left-4).attr('y',y+rowH/2+4)
      .attr('text-anchor','end').attr('font-size','10px').text(`T${t.turn_index}`);
  });

  svg.append('g').attr('transform',`translate(0,${H-margin.bottom})`)
    .call(d3.axisBottom(x).ticks(8).tickFormat(d=>d>=1000?(d/1000).toFixed(0)+'K':d))
    .selectAll('text').style('fill','#ccc');
  svg.append('text').attr('x',W/2).attr('y',H-2).attr('text-anchor','middle')
    .attr('font-size','11px').text('Token Position');
}
</script>
</body>
</html>
"""


def render_cache_explorer(output_dir: Path, cache_payload: dict) -> Path:
    """Write the standalone D3.js cache explorer HTML with inlined data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cache_explorer.html"
    json_str = orjson.dumps(cache_payload).decode()
    out_path.write_text(_CACHE_EXPLORER_HTML.replace("__INLINE_DATA__", json_str))
    return out_path


# ---------------------------------------------------------------------------
# Comparison text (shareable summary)
# ---------------------------------------------------------------------------
def render_comparison_text(
    quality: dict,
    session_duration_stats: PercentileStats | None = None,
    prefill_tps: float = 20_000,
    decode_tps: float = 60,
    target_p05: float | None = 30,
    target_p99: float | None = 3_750,
) -> str:
    """Render a shareable target-vs-dataset comparison from quality.json data."""
    cfg = quality["config_summary"]
    ovt = quality["observed_vs_target"]
    sess = quality["session_stats"]
    ends = quality["session_end_stats"]
    num_sessions = ends["total_sessions"]
    total_turns = ovt.get("generation_length", {}).get("observed", {}).get("count", 0)

    lines: list[str] = []
    w = lines.append

    w("Claude Code Session Profile: Target vs Synthesized Dataset")
    w("=" * 60)
    w("")
    w(
        f"{num_sessions:,} sessions | {total_turns:,} turns | block_size={cfg['cache_block_size']}"
    )
    w("")
    w(f"{'':40s} {'Target':>12s} {'Dataset':>12s} {'Error':>8s}")
    w("-" * 76)

    def _row(
        row_label: str,
        target_val: float | None,
        obs_val: float | None,
        pct_err: float | None = None,
    ) -> None:
        t_str = f"{target_val:>12,.0f}" if target_val is not None else f"{'-':>12s}"
        o_str = f"{obs_val:>12,.1f}" if obs_val is not None else f"{'-':>12s}"
        err_str = ""
        if pct_err is not None:
            sign = "+" if obs_val and target_val and obs_val > target_val else "-"
            err_str = f"  {sign}{abs(pct_err):.1f}%"
        w(f"  {row_label:38s}{t_str}{o_str}{err_str}")

    def _metric_block(label: str, key: str) -> None:
        w(label)
        m = ovt.get(key, {})
        obs = m.get("observed", {})
        _row("mean", m.get("target_mean"), obs.get("mean"), m.get("pct_error_mean"))
        _row(
            "median",
            m.get("target_median"),
            obs.get("median"),
            m.get("pct_error_median"),
        )

    _metric_block("Initial Context (tokens)", "initial_context")
    w("")

    _metric_block("New Tokens Per Turn", "new_tokens_per_turn")
    w("")

    gen = ovt.get("generation_length", {})
    gen_obs = gen.get("observed", {})
    w("Generation Length (tokens)")
    _row("mean", gen.get("target_mean"), gen_obs.get("mean"), gen.get("pct_error_mean"))
    _row(
        "median",
        gen.get("target_median"),
        gen_obs.get("median"),
        gen.get("pct_error_median"),
    )
    _row("p05", target_p05, gen_obs.get("p05"))
    _row("p99", target_p99, gen_obs.get("p99"))
    w("")

    w("Prompt")
    w(
        f"  {'max_prompt_tokens':38s}{cfg['max_prompt_tokens']:>12,d}{cfg['max_prompt_tokens']:>12,d}"
    )
    w(
        f"  {'system_prompt_tokens':38s}{cfg['system_prompt_tokens']:>12,d}{cfg['system_prompt_tokens']:>12,d}"
    )
    w("")

    w("Additional Dataset Statistics")
    w("-" * 76)

    w("Turns Per Session")
    for label, field in [
        ("mean", "mean"),
        ("median", "median"),
        ("p05", "p05"),
        ("p25", "p25"),
        ("p75", "p75"),
        ("p95", "p95"),
        ("p99", "p99"),
    ]:
        val = sess.get(field, 0)
        w(f"  {label:38s}{'-':>12s}{val:>12.1f}")
    w("")

    delay = ovt.get("inter_turn_delay_ms", {}).get("observed", {})
    if delay:
        w("Inter-Turn Delay (ms)")
        for label, field in [
            ("mean", "mean"),
            ("median", "median"),
            ("p05", "p05"),
            ("p95", "p95"),
        ]:
            val = delay.get(field, 0)
            w(f"  {label:38s}{'-':>12s}{val:>12,.0f}")
        af = cfg.get("inter_turn_delay_agentic_fraction", 0)
        am = cfg.get("inter_turn_delay_agentic_mean_ms", 0) / 1000
        hm = cfg.get("inter_turn_delay_human_mean_ms", 0) / 1000
        w(f"  ({af:.0%} agentic ~{am:.0f}s, {1 - af:.0%} human ~{hm:.0f}s)")
        w("")

    if session_duration_stats:
        sd = session_duration_stats
        w(
            f"Session Duration (estimated @ {prefill_tps:,.0f} prefill tok/s, {decode_tps:,.0f} decode tok/s)"
        )
        for label, val in [("mean", sd.mean), ("median", sd.median)]:
            w(f"  {label:38s}{'-':>12s}{val:>12.1f} min")
        w("")

    w("Session Endings")
    w(
        f"  {'forced retires (hit context limit)':38s}{'-':>12s}{ends['forced_retires']:>12d}"
    )
    w(f"  {'probabilistic resets':38s}{'-':>12s}{ends['probabilistic_resets']:>12d}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def generate_report(
    run_dir: Path,
    fmt: str = "text",
    prefill_tps: float = 20_000,
    decode_tps: float = 60,
) -> ReportData:
    """Load a run directory and print text report to console.

    Visualizations (report.html, cache_explorer.html) are generated during
    synthesis by write_dataset. This function is for text-only reporting.

    Args:
        run_dir: Path to the run directory containing dataset.jsonl and manifest.json.
        fmt: Output format - "text", "plot", or "both" (kept for backwards compat).
        prefill_tps: Prefill tokens per second for latency estimation.
        decode_tps: Decode tokens per second for latency estimation.

    Returns:
        The ReportData object.
    """
    jsonl_path = run_dir / "dataset.jsonl"
    manifest_path = run_dir / "manifest.json"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"dataset.jsonl not found in {run_dir}")

    manifest: DatasetManifest | None = None
    if manifest_path.exists():
        manifest = DatasetManifest(**orjson.loads(manifest_path.read_bytes()))

    turns = load_jsonl(jsonl_path)
    sessions = group_sessions(turns)
    metrics = extract_metrics(sessions, prefill_tps=prefill_tps, decode_tps=decode_tps)

    block_size = manifest.block_size if manifest else 512
    cache_metrics = extract_cache_metrics(sessions, block_size=block_size)
    metrics.update(cache_metrics)

    report_data = build_report_data(metrics, manifest)

    if fmt in ("text", "both"):
        _print_report_to_console(report_data)

    return report_data
