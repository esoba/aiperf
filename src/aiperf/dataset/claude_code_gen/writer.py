# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Writes synthesized sessions to Mooncake-compatible JSONL, manifest, and quality report."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import orjson

from aiperf.dataset.claude_code_gen.models import (
    DatasetManifest,
    QualityMetric,
    QualityReport,
    SessionDistributionConfig,
    SessionEndReason,
    SessionEndStats,
    SynthesizedSession,
    percentile_stats,
)
from aiperf.dataset.claude_code_gen.report import (
    ParsedTurn,
    _pct_error,
    _percentile_stats,
    extract_cache_metrics,
    extract_metrics,
    render_cache_explorer,
    render_comparison_text,
    render_plot_report,
    write_cache_structure,
)


def _config_summary(config: SessionDistributionConfig) -> dict[str, float | int]:
    """Flatten config into a readable dict of key parameters."""
    return {
        "system_prompt_tokens": config.system_prompt_tokens,
        "initial_context_mean": config.initial_context.mean,
        "initial_context_median": config.initial_context.median,
        "new_tokens_per_turn_mean": config.new_tokens_per_turn.mean,
        "new_tokens_per_turn_median": config.new_tokens_per_turn.median,
        "generation_length_mean": config.generation_length.mean,
        "generation_length_median": config.generation_length.median,
        "max_prompt_tokens": config.max_prompt_tokens,
        "inter_turn_delay_agentic_fraction": config.inter_turn_delay.agentic_fraction,
        "inter_turn_delay_agentic_mean_ms": config.inter_turn_delay.agentic_delay.mean,
        "inter_turn_delay_human_mean_ms": config.inter_turn_delay.human_delay.mean,
        "reset_base_probability": config.reset.base_probability,
        "reset_context_scaling": config.reset.context_scaling,
        "cache_block_size": config.cache.block_size,
    }


def _build_quality_metric(
    arr: np.ndarray,
    target_mean: float | None = None,
    target_median: float | None = None,
) -> QualityMetric:
    """Build a QualityMetric with percentile stats and error calculations."""
    observed = percentile_stats(arr)
    pct_error_mean = (
        round(_pct_error(target_mean, observed.mean), 2)
        if target_mean is not None
        else None
    )
    pct_error_median = (
        round(_pct_error(target_median, observed.median), 2)
        if target_median is not None
        else None
    )
    return QualityMetric(
        target_mean=target_mean,
        target_median=target_median,
        observed=observed,
        pct_error_mean=pct_error_mean,
        pct_error_median=pct_error_median,
    )


def write_dataset(
    sessions: list[SynthesizedSession],
    output_dir: Path,
    config: SessionDistributionConfig,
    seed: int,
    config_name: str | None = None,
) -> tuple[Path, Path, Path]:
    """Write JSONL dataset, manifest, quality report, and cache explorer into *output_dir*.

    Returns:
        Tuple of (jsonl_path, manifest_path, quality_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "dataset.jsonl"
    manifest_path = output_dir / "manifest.json"
    quality_path = output_dir / "quality.json"

    # Write JSONL
    _write_jsonl(sessions, jsonl_path, config.cache.block_size)

    # Write manifest
    manifest = DatasetManifest(
        seed=seed,
        block_size=config.cache.block_size,
        num_sessions=len(sessions),
        config_name=config_name,
        generation_params=config,
    )
    manifest_path.write_bytes(
        orjson.dumps(manifest.model_dump(), option=orjson.OPT_INDENT_2)
    )

    # Write quality report
    report = compute_quality_report(sessions, config)
    quality_dict = report.model_dump()
    quality_path.write_bytes(orjson.dumps(quality_dict, option=orjson.OPT_INDENT_2))

    # Write visualizations (cache explorer + plotly dashboard)
    _write_visualizations(sessions, manifest, output_dir)

    # Write shareable comparison text
    _write_comparison_text(sessions, quality_dict, output_dir)

    return jsonl_path, manifest_path, quality_path


def _sessions_to_parsed(
    sessions: list[SynthesizedSession],
) -> dict[str, list]:
    """Convert SynthesizedSessions to grouped ParsedTurns for report functions."""
    parsed: dict[str, list[ParsedTurn]] = {}
    for session in sessions:
        parsed[session.session_id] = [
            ParsedTurn(
                session_id=session.session_id,
                input_length=t.input_length,
                output_length=t.output_length,
                hash_ids=t.hash_ids,
                delay_ms=t.delay_ms,
            )
            for t in session.turns
        ]
    return parsed


def _write_visualizations(
    sessions: list[SynthesizedSession],
    manifest: DatasetManifest,
    output_dir: Path,
) -> None:
    """Generate cache explorer and plotly dashboard from synthesized sessions."""
    parsed_sessions = _sessions_to_parsed(sessions)

    cache_payload = write_cache_structure(parsed_sessions, manifest, output_dir)
    render_cache_explorer(output_dir, cache_payload)

    metrics = extract_metrics(parsed_sessions)
    cache_metrics = extract_cache_metrics(
        parsed_sessions, block_size=manifest.block_size
    )
    metrics.update(cache_metrics)
    render_plot_report(metrics, parsed_sessions, output_dir)


def _write_comparison_text(
    sessions: list[SynthesizedSession],
    quality_dict: dict,
    output_dir: Path,
) -> Path:
    """Generate comparison.txt with target-vs-dataset summary."""
    parsed = _sessions_to_parsed(sessions)
    metrics = extract_metrics(parsed)
    sd_stats = _percentile_stats(metrics["session_duration_min"])
    text = render_comparison_text(quality_dict, session_duration_stats=sd_stats)
    out_path = output_dir / "comparison.txt"
    out_path.write_text(text)
    return out_path


def _write_jsonl(
    sessions: list[SynthesizedSession], path: Path, block_size: int
) -> None:
    """Write Mooncake-compatible JSONL rows with incremental values.

    Each row's input_length = new_tokens for that turn, and hash_ids contains
    ceil(new_tokens / block_size) fresh block IDs (last block partial).
    The downstream worker accumulates turns to reconstruct the full ISL.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for session in sessions:
            next_hash_id: int | None = None
            for turn in session.turns:
                n_blocks = math.ceil(turn.new_tokens / block_size)
                if turn.turn_index == 0:
                    hash_ids = turn.hash_ids
                    next_hash_id = hash_ids[-1] + 1 if hash_ids else 0
                else:
                    hash_ids = list(range(next_hash_id, next_hash_id + n_blocks))
                    next_hash_id += n_blocks
                row: dict = {
                    "session_id": session.session_id,
                    "input_length": turn.new_tokens,
                    "output_length": turn.output_length,
                    "hash_ids": hash_ids,
                }
                if turn.turn_index == 0:
                    row["timestamp"] = round(turn.timestamp_ms, 1)
                else:
                    row["delay"] = round(turn.delay_ms, 1)
                f.write(orjson.dumps(row))
                f.write(b"\n")


def compute_quality_report(
    sessions: list[SynthesizedSession],
    config: SessionDistributionConfig,
) -> QualityReport:
    """Compute quality metrics comparing observed distributions to targets."""
    all_initial_ctx: list[float] = []
    all_new_tokens: list[float] = []
    all_output_lens: list[float] = []
    all_delays: list[float] = []
    turns_per_session: list[float] = []
    final_context_utils: list[float] = []
    forced_retires = 0
    probabilistic_resets = 0

    for session in sessions:
        turns_per_session.append(float(len(session.turns)))

        if session.end_reason == SessionEndReason.FORCED_RETIRE:
            forced_retires += 1
        else:
            probabilistic_resets += 1

        last_turn = session.turns[-1]
        final_context_utils.append(last_turn.input_length / config.max_prompt_tokens)

        for turn in session.turns:
            if turn.turn_index == 0:
                all_initial_ctx.append(float(turn.input_length))
            else:
                all_new_tokens.append(float(turn.new_tokens))
                all_delays.append(turn.delay_ms)
            all_output_lens.append(float(turn.output_length))

    observed_vs_target: dict[str, QualityMetric] = {}

    if all_initial_ctx:
        observed_vs_target["initial_context"] = _build_quality_metric(
            np.array(all_initial_ctx),
            target_mean=config.initial_context.mean,
            target_median=config.initial_context.median,
        )

    if all_output_lens:
        observed_vs_target["generation_length"] = _build_quality_metric(
            np.array(all_output_lens),
            target_mean=config.generation_length.mean,
            target_median=config.generation_length.median,
        )

    if all_new_tokens:
        observed_vs_target["new_tokens_per_turn"] = _build_quality_metric(
            np.array(all_new_tokens),
            target_mean=config.new_tokens_per_turn.mean,
            target_median=config.new_tokens_per_turn.median,
        )

    if all_delays:
        observed_vs_target["inter_turn_delay_ms"] = _build_quality_metric(
            np.array(all_delays),
            target_mean=None,
            target_median=None,
        )

    if turns_per_session:
        observed_vs_target["turns_per_session"] = _build_quality_metric(
            np.array(turns_per_session),
            target_mean=None,
            target_median=None,
        )

    tps_arr = np.array(turns_per_session) if turns_per_session else np.array([0.0])
    session_stats = percentile_stats(tps_arr)

    total = len(sessions)
    fcu_arr = np.array(final_context_utils) if final_context_utils else np.array([0.0])
    session_end_stats = SessionEndStats(
        total_sessions=total,
        forced_retires=forced_retires,
        probabilistic_resets=probabilistic_resets,
        retire_fraction=round(forced_retires / max(total, 1), 4),
        reset_fraction=round(probabilistic_resets / max(total, 1), 4),
        final_context_utilization=percentile_stats(fcu_arr),
    )

    return QualityReport(
        config_summary=_config_summary(config),
        observed_vs_target=observed_vs_target,
        session_stats=session_stats,
        session_end_stats=session_end_stats,
    )
