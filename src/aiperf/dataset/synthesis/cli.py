# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace analysis display logic."""

from __future__ import annotations

from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.synthesis.models import MetricStats
from aiperf.dataset.synthesis.prefix_analyzer import PrefixAnalyzer

_STAT_COLUMNS = ["Mean", "Std Dev", "Min", "P25", "Median", "P75", "Max"]


def _build_stats_table(
    metrics: dict[str, MetricStats | None], *, title: str = "Trace Statistics"
) -> Table:
    """Build a Rich table with metric statistics."""
    table = Table(title=title, show_lines=False, box=box.SIMPLE_HEAVY, pad_edge=False)
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    for col in _STAT_COLUMNS:
        table.add_column(col, justify="right", style="green", no_wrap=True)

    for name, stats in metrics.items():
        if stats is None:
            table.add_row(name, *["[dim]N/A[/dim]"] * len(_STAT_COLUMNS))
        else:
            table.add_row(
                name,
                f"{stats.mean:,.2f}",
                f"{stats.std_dev:,.2f}",
                f"{stats.min:,.2f}",
                f"{stats.p25:,.2f}",
                f"{stats.median:,.2f}",
                f"{stats.p75:,.2f}",
                f"{stats.max:,.2f}",
            )

    return table


def _kv_table(title: str, rows: list[tuple[str, str]]) -> Table:
    """Build a headerless key-value table."""
    table = Table(title=title, show_header=False, box=box.SIMPLE_HEAVY, pad_edge=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for key, value in rows:
        table.add_row(key, value)
    return table


def _save_report(
    console: Console, stats: AIPerfBaseModel, output_file: Path | None
) -> None:
    """Write JSON report if output_file is set."""
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(stats.model_dump_json(indent=2))
        console.print(f"Analysis report saved to {output_file}")


def _is_conflux_format(input_path: Path) -> bool:
    """Detect whether the input is a Conflux JSON file or directory."""
    from aiperf.dataset.loader.conflux import ConfluxLoader

    return ConfluxLoader.can_load(filename=str(input_path))


def analyze_trace(
    input_file: Path,
    block_size: int = 512,
    output_file: Path | None = None,
) -> None:
    """Analyze a trace file for distributions and statistics.

    Auto-detects Conflux JSON vs JSONL trace format.
    """
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    if _is_conflux_format(input_file):
        _analyze_conflux(input_file, output_file)
    elif input_file.is_dir():
        print(
            f"Error: Directory '{input_file}' does not contain Conflux JSON files. "
            "For JSONL trace analysis, provide a file path instead."
        )
    else:
        _analyze_prefix_trace(input_file, block_size, output_file)


def _analyze_prefix_trace(
    input_file: Path,
    block_size: int,
    output_file: Path | None,
) -> None:
    """Analyze a JSONL trace file for ISL/OSL distributions and cache hit rates."""
    analyzer = PrefixAnalyzer(block_size=block_size)
    stats = analyzer.analyze_file(input_file)

    console = Console(width=120)
    console.print()
    console.print("[bold]Trace Analysis Report[/bold]")
    console.print(
        _kv_table(
            "Overview",
            [
                ("Total requests", f"{stats.total_requests:,}"),
                ("Unique prefixes", f"{stats.unique_prefixes:,}"),
                ("Prefix groups", f"{stats.num_prefix_groups:,}"),
            ],
        )
    )
    console.print()
    console.print(
        _build_stats_table(
            {
                "Input Length": stats.isl_stats,
                "Context Length": stats.context_length_stats,
                "Unique Prompt Length": stats.unique_prompt_length_stats,
                "Output Length": stats.osl_stats,
                "Theoretical Hit Rates": stats.hit_rate_stats,
            }
        )
    )
    console.print()
    _save_report(console, stats, output_file)


def _fmt_tokens(n: int) -> str:
    """Format token count with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_duration(s: float) -> str:
    """Format seconds to human-readable duration."""
    if s >= 3600:
        return f"{s / 3600:.1f}h"
    if s >= 60:
        return f"{s / 60:.1f}m"
    return f"{s:.1f}s"


def _analyze_conflux(
    input_path: Path,
    output_file: Path | None,
) -> None:
    """Analyze a Conflux JSON file or directory."""
    from aiperf.dataset.loader.conflux_analyzer import (
        ConfluxAnalysisStats,
        analyze_conflux,
    )

    stats: ConfluxAnalysisStats = analyze_conflux(input_path)

    console = Console(width=120)
    console.print()
    console.print("[bold]Conflux Trace Analysis[/bold]")

    total_tok = stats.total_input_tokens + stats.total_output_tokens
    console.print(
        _kv_table(
            "Overview",
            [
                ("Files", f"{stats.total_files:,}"),
                ("Requests", f"{stats.total_records:,}"),
                (
                    "Agents",
                    f"{stats.total_agents:,} ({stats.parent_agents} parent, "
                    f"{stats.child_agents} child, {stats.orphan_records} orphan)",
                ),
                (
                    "Session span",
                    f"{_fmt_duration(stats.session_span_s)} "
                    f"({stats.active_pct:.0f}% active)",
                ),
                (
                    "Concurrency",
                    f"max {stats.max_concurrency}, avg {stats.avg_concurrency:.1f}",
                ),
                ("Streaming", f"{stats.streaming_pct:.1f}%"),
            ],
        )
    )
    console.print()

    console.print(
        _kv_table(
            "Token Economics",
            [
                ("Total tokens", f"{total_tok:,} ({stats.input_share_pct:.1f}% input)"),
                ("Input tokens", f"{stats.total_input_tokens:,}"),
                ("Output tokens", f"{stats.total_output_tokens:,}"),
                (
                    "Cached tokens",
                    f"{stats.total_cached_tokens:,} "
                    f"({stats.weighted_cache_hit_pct:.1f}% hit rate)",
                ),
                ("Uncached tokens", f"{stats.total_uncached_tokens:,}"),
                ("Cache writes", f"{stats.total_cache_write_tokens:,}"),
                ("Cache ROI", f"{stats.cache_roi:.1f}x (hits / writes)"),
                (
                    "Effective tokens",
                    f"{stats.effective_token_pct:.1f}% "
                    "(only uncached + output need compute)",
                ),
            ],
        )
    )
    console.print()

    # Models
    if stats.models_used:
        model_table = Table(title="Models", box=box.SIMPLE_HEAVY, pad_edge=False)
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Requests", justify="right", style="green")
        model_table.add_column("%", justify="right", style="green")
        for model, count in stats.models_used.items():
            pct = count / stats.total_records * 100
            model_table.add_row(model, f"{count:,}", f"{pct:.1f}%")
        console.print(model_table)
        console.print()

    # Per-agent breakdown
    if stats.agent_breakdown:
        agent_table = Table(
            title="Agent Breakdown (by input tokens)",
            box=box.SIMPLE_HEAVY,
            pad_edge=False,
        )
        agent_table.add_column("Agent", style="cyan", no_wrap=True)
        agent_table.add_column("Role", style="dim")
        agent_table.add_column("Model", style="dim")
        agent_table.add_column("Req", justify="right", style="green")
        agent_table.add_column("Input", justify="right", style="green")
        agent_table.add_column("Cached", justify="right", style="green")
        agent_table.add_column("Output", justify="right", style="green")
        agent_table.add_column("Hit%", justify="right", style="green")
        agent_table.add_column("ROI", justify="right", style="green")
        for a in stats.agent_breakdown:
            # Shorten model names for display
            model_short = a.model.replace("claude-", "").replace("-20251001", "")
            agent_table.add_row(
                a.agent_id[:20],
                "parent" if a.is_parent else "child",
                model_short,
                str(a.requests),
                _fmt_tokens(a.input_tokens),
                _fmt_tokens(a.cached_tokens),
                _fmt_tokens(a.output_tokens),
                f"{a.cache_hit_pct:.1f}%",
                f"{a.cache_roi:.1f}x" if a.cache_roi > 0 else "-",
            )
        console.print(agent_table)
        console.print()

    # Distribution tables
    console.print(
        _build_stats_table(
            {
                "Input Tokens": stats.input_tokens_stats,
                "Output Tokens": stats.output_tokens_stats,
                "Cached Tokens": stats.cached_tokens_stats,
                "Cache Hit %": stats.cache_hit_pct_stats,
                "OSL/ISL Ratio": stats.osl_isl_ratio_stats,
            },
            title="Token Distributions",
        )
    )
    console.print()

    console.print(
        _build_stats_table(
            {
                "Duration (ms)": stats.duration_ms_stats,
                "TTFT (ms)": stats.ttft_ms_stats,
            },
            title="Timing",
        )
    )
    console.print()

    console.print(
        _build_stats_table(
            {
                "Turns per Agent": stats.turns_per_agent_stats,
                "Tools per Request": stats.tool_count_stats,
                "Messages per Request": stats.message_count_stats,
            },
            title="Request Shape",
        )
    )
    console.print()

    _save_report(console, stats, output_file)
