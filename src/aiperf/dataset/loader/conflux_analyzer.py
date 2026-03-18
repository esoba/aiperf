# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyzer for Conflux trace files and directories.

Computes structural, token, cache, concurrency, and timing statistics for
Conflux proxy captures without loading the full dataset pipeline.
Used by ``aiperf analyze-trace``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson
from pydantic import Field

from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.synthesis.models import MetricStats

# ── Internal helpers ──


@dataclass(slots=True)
class _TokenAccum:
    """Accumulates token totals for a group of records."""

    input: int = 0
    output: int = 0
    cached: int = 0
    cache_write: int = 0
    models: Counter = field(default_factory=Counter)

    def add(self, record: dict[str, Any]) -> None:
        tokens = record.get("tokens")
        if isinstance(tokens, dict):
            self.input += tokens.get("input", 0) or 0
            self.output += tokens.get("output", 0) or 0
            self.cached += tokens.get("input_cached", 0) or 0
            self.cache_write += tokens.get("input_cache_write", 0) or 0
        model = record.get("model")
        if model:
            self.models[model] += 1

    @property
    def uncached(self) -> int:
        return self.input - self.cached

    @property
    def cache_hit_pct(self) -> float:
        return (self.cached / self.input * 100.0) if self.input > 0 else 0.0

    @property
    def cache_roi(self) -> float:
        return (self.cached / self.cache_write) if self.cache_write > 0 else 0.0

    @property
    def primary_model(self) -> str:
        return self.models.most_common(1)[0][0] if self.models else "unknown"


def _parse_ts(iso_str: str) -> float:
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp()


def _safe_pct(num: int | float, denom: int | float) -> float:
    return (num / denom * 100.0) if denom > 0 else 0.0


# ── Models ──


class AgentSummary(AIPerfBaseModel):
    """Per-agent breakdown row."""

    agent_id: str = Field(description="Agent identifier")
    is_parent: bool = Field(description="Whether this is the parent agent")
    model: str = Field(description="Primary model used")
    requests: int = Field(description="Number of API calls")
    input_tokens: int = Field(description="Total input tokens")
    cached_tokens: int = Field(description="Total cached input tokens")
    uncached_tokens: int = Field(description="Total uncached input tokens")
    output_tokens: int = Field(description="Total output tokens")
    cache_hit_pct: float = Field(description="Weighted cache hit rate (%)")
    cache_writes: int = Field(description="Total cache write tokens")
    cache_roi: float = Field(
        description="Cache ROI (cached hits / cache writes). 0 if no writes."
    )


class ConfluxAnalysisStats(AIPerfBaseModel):
    """Statistics extracted from Conflux trace analysis."""

    # Structure
    total_records: int = Field(description="Total API call records across all files")
    total_files: int = Field(description="Number of JSON files loaded")
    total_agents: int = Field(description="Distinct agent threads (with agent_id)")
    parent_agents: int = Field(description="Agent threads with is_subagent=False")
    child_agents: int = Field(description="Agent threads with is_subagent=True")
    orphan_records: int = Field(description="Records without agent_id (utility calls)")
    models_used: dict[str, int] = Field(description="Model name -> request count")

    # Session timeline
    session_span_s: float = Field(
        description="Wall clock span from first to last request (seconds)"
    )
    active_time_s: float = Field(description="Sum of all request durations (seconds)")
    active_pct: float = Field(description="Active time as % of span")

    # Concurrency
    max_concurrency: int = Field(description="Peak in-flight concurrent requests")
    avg_concurrency: float = Field(
        description="Average concurrency (active_time / span)"
    )

    # Token totals
    total_input_tokens: int = Field(description="Sum of all input tokens")
    total_output_tokens: int = Field(description="Sum of all output tokens")
    total_cached_tokens: int = Field(description="Sum of all cached input tokens")
    total_uncached_tokens: int = Field(
        description="Sum of uncached input tokens (input - cached)"
    )
    total_cache_write_tokens: int = Field(description="Sum of cache write tokens")
    input_share_pct: float = Field(description="Input tokens as % of total tokens")

    # Cache economics
    weighted_cache_hit_pct: float = Field(
        description="Weighted cache hit rate: cached / input (%)"
    )
    cache_roi: float = Field(description="Cache ROI: cached hits / cache writes (x)")
    effective_token_pct: float = Field(
        description="Effective tokens (uncached + output) as % of total"
    )

    # Per-request distributions
    input_tokens_stats: MetricStats | None = Field(
        default=None, description="Input tokens per request"
    )
    output_tokens_stats: MetricStats | None = Field(
        default=None, description="Output tokens per request"
    )
    cached_tokens_stats: MetricStats | None = Field(
        default=None, description="Cached input tokens per request"
    )
    cache_hit_pct_stats: MetricStats | None = Field(
        default=None, description="Per-request cache hit % distribution"
    )
    osl_isl_ratio_stats: MetricStats | None = Field(
        default=None, description="Output/input token ratio per request"
    )

    # Timing
    duration_ms_stats: MetricStats | None = Field(
        default=None, description="Request duration in ms"
    )
    ttft_ms_stats: MetricStats | None = Field(
        default=None, description="Time to first token in ms"
    )

    # Per-agent turn counts
    turns_per_agent_stats: MetricStats | None = Field(
        default=None, description="Turns per agent thread"
    )

    # Request shape
    tool_count_stats: MetricStats | None = Field(
        default=None, description="Tool definitions per request"
    )
    message_count_stats: MetricStats | None = Field(
        default=None, description="Messages per request"
    )

    # Streaming
    streaming_pct: float = Field(description="Percentage of requests using streaming")

    # Per-agent breakdown
    agent_breakdown: list[AgentSummary] = Field(
        default_factory=list,
        description="Per-agent token and cache summary (sorted by input tokens desc)",
    )


# ── Analysis ──


def analyze_conflux(input_path: Path) -> ConfluxAnalysisStats:
    """Analyze a Conflux JSON file or directory of files."""
    if input_path.is_dir():
        json_files = sorted(input_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json files found in {input_path}")
    else:
        json_files = [input_path]

    all_records: list[dict[str, Any]] = []
    for json_file in json_files:
        with open(json_file, "rb") as f:
            data = orjson.loads(f.read())
        if isinstance(data, list):
            all_records.extend(data)

    # Group by agent_id and classify
    agents: dict[str, list[dict]] = {}
    orphan_count = 0
    parent_ids: set[str] = set()
    child_ids: set[str] = set()

    for record in all_records:
        agent_id = record.get("agent_id")
        if agent_id is not None:
            agents.setdefault(agent_id, []).append(record)
        else:
            orphan_count += 1

    for agent_id, records in agents.items():
        if records[0].get("is_subagent") is True:
            child_ids.add(agent_id)
        elif records[0].get("is_subagent") is False:
            parent_ids.add(agent_id)

    # Session timeline and concurrency
    events: list[tuple[float, int]] = []
    first_ts = float("inf")
    last_ts = float("-inf")
    total_active_s = 0.0

    for r in all_records:
        ts_str = r.get("timestamp")
        if not ts_str:
            continue
        start_s = _parse_ts(ts_str)
        completed_str = r.get("completed_at")
        dur_s = (r.get("duration_ms", 0) or 0) / 1000.0

        if completed_str:
            end_s = _parse_ts(completed_str)
        elif dur_s > 0:
            end_s = start_s + dur_s
        else:
            end_s = start_s

        first_ts = min(first_ts, start_s)
        last_ts = max(last_ts, end_s)
        total_active_s += end_s - start_s
        events.append((start_s, 1))
        events.append((end_s, -1))

    session_span_s = max(last_ts - first_ts, 0.001)
    avg_concurrency = total_active_s / session_span_s if session_span_s > 0 else 0.0

    events.sort()
    max_conc = 0
    cur_conc = 0
    for _, delta in events:
        cur_conc += delta
        max_conc = max(max_conc, cur_conc)

    # Per-record distributions + global totals
    totals = _TokenAccum()
    input_tokens_list: list[int] = []
    output_tokens_list: list[int] = []
    cached_tokens_list: list[int] = []
    cache_hit_pcts: list[float] = []
    osl_isl_ratios: list[float] = []
    durations: list[float] = []
    ttfts: list[float] = []
    tool_counts: list[int] = []
    message_counts: list[int] = []
    streaming_count = 0
    streaming_known = 0

    for r in all_records:
        totals.add(r)

        tokens = r.get("tokens")
        if isinstance(tokens, dict):
            inp = tokens.get("input", 0) or 0
            out = tokens.get("output", 0) or 0
            cached = tokens.get("input_cached", 0) or 0
            if inp > 0:
                input_tokens_list.append(inp)
                cache_hit_pcts.append(cached / inp * 100.0)
            if out > 0:
                output_tokens_list.append(out)
            if cached > 0:
                cached_tokens_list.append(cached)
            if inp > 0 and out > 0:
                osl_isl_ratios.append(out / inp)

        dur = r.get("duration_ms", 0)
        if dur and dur > 0:
            durations.append(float(dur))
        ttft = r.get("ttft_ms")
        if ttft is not None and ttft > 0:
            ttfts.append(float(ttft))

        tools = r.get("tools")
        if isinstance(tools, list):
            tool_counts.append(len(tools))
        msgs = r.get("messages")
        if isinstance(msgs, list):
            message_counts.append(len(msgs))

        is_streaming = r.get("is_streaming")
        if is_streaming is not None:
            streaming_known += 1
            if is_streaming:
                streaming_count += 1

    total_tokens = totals.input + totals.output

    # Per-agent breakdown
    agent_breakdown: list[AgentSummary] = []
    for agent_id, records in agents.items():
        acc = _TokenAccum()
        for r in records:
            acc.add(r)
        agent_breakdown.append(
            AgentSummary(
                agent_id=agent_id,
                is_parent=agent_id in parent_ids,
                model=acc.primary_model,
                requests=len(records),
                input_tokens=acc.input,
                cached_tokens=acc.cached,
                uncached_tokens=acc.uncached,
                output_tokens=acc.output,
                cache_hit_pct=acc.cache_hit_pct,
                cache_writes=acc.cache_write,
                cache_roi=acc.cache_roi,
            )
        )
    agent_breakdown.sort(key=lambda a: a.input_tokens, reverse=True)

    return ConfluxAnalysisStats(
        total_records=len(all_records),
        total_files=len(json_files),
        total_agents=len(agents),
        parent_agents=len(parent_ids),
        child_agents=len(child_ids),
        orphan_records=orphan_count,
        models_used=dict(totals.models.most_common()),
        session_span_s=session_span_s,
        active_time_s=total_active_s,
        active_pct=_safe_pct(total_active_s, session_span_s),
        max_concurrency=max_conc,
        avg_concurrency=avg_concurrency,
        total_input_tokens=totals.input,
        total_output_tokens=totals.output,
        total_cached_tokens=totals.cached,
        total_uncached_tokens=totals.uncached,
        total_cache_write_tokens=totals.cache_write,
        input_share_pct=_safe_pct(totals.input, total_tokens),
        weighted_cache_hit_pct=totals.cache_hit_pct,
        cache_roi=totals.cache_roi,
        effective_token_pct=_safe_pct(totals.uncached + totals.output, total_tokens),
        input_tokens_stats=MetricStats.from_values(input_tokens_list),
        output_tokens_stats=MetricStats.from_values(output_tokens_list),
        cached_tokens_stats=MetricStats.from_values(cached_tokens_list),
        cache_hit_pct_stats=MetricStats.from_values(cache_hit_pcts),
        osl_isl_ratio_stats=MetricStats.from_values(osl_isl_ratios),
        duration_ms_stats=MetricStats.from_values(durations),
        ttft_ms_stats=MetricStats.from_values(ttfts),
        turns_per_agent_stats=MetricStats.from_values(
            [len(recs) for recs in agents.values()]
        ),
        tool_count_stats=MetricStats.from_values(tool_counts),
        message_count_stats=MetricStats.from_values(message_counts),
        streaming_pct=_safe_pct(streaming_count, streaming_known),
        agent_breakdown=agent_breakdown,
    )
