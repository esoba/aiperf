# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for Claude Code session dataset generation."""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from pydantic import Field, model_validator

from aiperf.common.models import AIPerfBaseModel


class PercentileStats(AIPerfBaseModel):
    """Descriptive statistics with percentile breakdown."""

    count: int = Field(description="Number of observations")
    mean: float = Field(description="Arithmetic mean")
    std: float = Field(description="Standard deviation")
    median: float = Field(description="50th percentile")
    p05: float = Field(description="5th percentile")
    p25: float = Field(description="25th percentile")
    p75: float = Field(description="75th percentile")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")


def percentile_stats(arr: np.ndarray) -> PercentileStats:
    """Compute PercentileStats from a numpy array."""
    return PercentileStats(
        count=len(arr),
        mean=round(float(np.mean(arr)), 2),
        std=round(float(np.std(arr)), 2),
        median=round(float(np.median(arr)), 2),
        p05=round(float(np.percentile(arr, 5)), 2),
        p25=round(float(np.percentile(arr, 25)), 2),
        p75=round(float(np.percentile(arr, 75)), 2),
        p95=round(float(np.percentile(arr, 95)), 2),
        p99=round(float(np.percentile(arr, 99)), 2),
    )


class SessionEndReason(str, Enum):
    """Why a session ended."""

    FORCED_RETIRE = "forced_retire"
    PROBABILISTIC_RESET = "probabilistic_reset"


class LognormalParams(AIPerfBaseModel):
    """Lognormal distribution parameters with real-space summary statistics.

    Can be constructed in two ways:
    1. Full: mu, sigma, mean, median all provided (e.g. from manifest.json or fit-stats)
    2. Simplified: just mean and median — mu/sigma auto-computed via model validator
    """

    mu: float | None = Field(default=None, description="Log-space mean")
    sigma: float | None = Field(
        default=None, ge=0.0, description="Log-space standard deviation"
    )
    mean: float = Field(gt=0.0, description="Real-space mean (derived)")
    median: float = Field(gt=0.0, description="Real-space median (derived)")

    @model_validator(mode="after")
    def compute_mu_sigma(self) -> LognormalParams:
        if self.mu is None or self.sigma is None:
            if self.mean < self.median:
                raise ValueError(
                    f"mean ({self.mean}) must be >= median ({self.median}) for lognormal"
                )
            self.mu = math.log(self.median)
            ratio = self.mean / self.median
            self.sigma = math.sqrt(2.0 * math.log(ratio)) if ratio > 1.0 else 0.0
        return self


def _default_agentic_delay() -> LognormalParams:
    return LognormalParams(mean=2_500, median=1_800)


def _default_human_delay() -> LognormalParams:
    return LognormalParams(mean=40_000, median=25_000)


class MixtureDelayConfig(AIPerfBaseModel):
    """Two-component mixture model for inter-turn delays.

    Agentic turns (tool-call follow-ups) are fast; human turns are slow.
    A Bernoulli draw selects which component to sample from.
    """

    agentic_fraction: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Probability of sampling the fast agentic delay",
    )
    agentic_delay: LognormalParams = Field(
        default_factory=_default_agentic_delay,
        description="Fast delay distribution (tool-call follow-ups)",
    )
    human_delay: LognormalParams = Field(
        default_factory=_default_human_delay,
        description="Slow delay distribution (human think time)",
    )


class ResetConfig(AIPerfBaseModel):
    """Context-dependent reset probability.

    Models --continue, CLAUDE.md edit, TTL expiry.
    P(reset) = base_probability * (1 + (context_scaling - 1) * input_length / max_prompt_tokens)
    """

    base_probability: float = Field(
        default=0.02, ge=0.0, le=1.0, description="Base reset chance per turn"
    )
    context_scaling: float = Field(
        default=2.0, ge=1.0, description="Multiplier at max_prompt_tokens"
    )


class CacheLayerConfig(AIPerfBaseModel):
    """Token sizes for the KV cache prefix model.

    L1: Global (tools + system prompt), shared by all sessions.
    L1.5: Group-shared (CLAUDE.md, repo context), shared within a group.
    L2: Session-specific prefix (initial files), sampled per session.
    L3: Conversation history, grows turn-by-turn (not configured here).
    """

    layer1_tokens: int = Field(
        default=32_000,
        ge=0,
        description="L1: tools + system prompt tokens (globally cached)",
    )
    layer1_5_tokens: int = Field(
        default=20_000,
        ge=0,
        description="L1.5: group-shared prefix tokens (CLAUDE.md, repo context)",
    )
    layer2: LognormalParams = Field(
        default_factory=lambda: LognormalParams(mean=10_000, median=5_000),
        description="L2: session-specific prefix token distribution",
    )
    block_size: int = Field(
        default=512, ge=1, description="KV cache page size in tokens"
    )


class GroupConfig(AIPerfBaseModel):
    """Group assignment for L1.5 cache sharing via Zipf distribution."""

    num_groups: int = Field(
        default=50, ge=1, description="Number of distinct groups (repos/projects)"
    )
    zipf_alpha: float = Field(
        default=1.2, ge=0.0, description="Zipf skew parameter (higher = more skewed)"
    )


def _default_new_tokens_per_turn() -> LognormalParams:
    return LognormalParams(mean=3_500, median=1_800)


def _default_generation_length() -> LognormalParams:
    return LognormalParams(mean=500, median=300)


class SessionDistributionConfig(AIPerfBaseModel):
    """Full configuration for synthesizing Claude Code sessions.

    initial_context is derived: L1 + L1.5 + sampled L2. Not directly configured.
    """

    system_prompt_tokens: int = Field(
        default=8_000, ge=0, description="System prompt token count"
    )
    new_tokens_per_turn: LognormalParams = Field(
        default_factory=_default_new_tokens_per_turn,
        description="New tokens added per turn",
    )
    generation_length: LognormalParams = Field(
        default_factory=_default_generation_length,
        description="Output token distribution",
    )
    inter_turn_delay: MixtureDelayConfig = Field(
        default_factory=MixtureDelayConfig, description="Inter-turn delay mixture model"
    )
    reset: ResetConfig = Field(
        default_factory=ResetConfig, description="Reset probability config"
    )
    max_prompt_tokens: int = Field(
        default=200_000, ge=1, description="Context window limit"
    )
    new_tokens_bias: float = Field(
        default=1.0,
        gt=0.0,
        description="Multiplier on new_tokens_per_turn mean to compensate for truncation bias",
    )
    cache: CacheLayerConfig = Field(
        default_factory=CacheLayerConfig, description="Cache layer config"
    )
    group: GroupConfig = Field(
        default_factory=GroupConfig,
        description="Group assignment config for L1.5 sharing",
    )
    restart_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of sessions that are restarts of earlier sessions (share L2 prefix)",
    )


class SynthesizedTurn(AIPerfBaseModel):
    """A single synthesized turn within a session."""

    turn_index: int = Field(ge=0, description="Turn number within session")
    input_length: int = Field(ge=1, description="Total input tokens for this turn")
    output_length: int = Field(ge=1, description="Output tokens generated")
    new_tokens: int = Field(ge=0, description="New tokens added since previous turn")
    delay_ms: float = Field(
        ge=0.0, description="Delay before this turn in milliseconds"
    )
    timestamp_ms: float = Field(
        ge=0.0, description="Absolute timestamp in milliseconds"
    )
    hash_ids: list[int] = Field(
        description="KV cache block hash IDs for prefix matching"
    )


class SynthesizedSession(AIPerfBaseModel):
    """A complete synthesized multi-turn session."""

    session_id: str = Field(description="Unique session identifier")
    group_id: int = Field(description="Group index for L1.5 cache sharing")
    turns: list[SynthesizedTurn] = Field(description="Ordered list of turns")
    end_reason: SessionEndReason = Field(description="Why the session ended")

    @model_validator(mode="after")
    def validate_turns_ordered(self) -> SynthesizedSession:
        for i, turn in enumerate(self.turns):
            if turn.turn_index != i:
                raise ValueError(
                    f"Turn {i} has turn_index={turn.turn_index}, expected {i}"
                )
        return self


class DatasetManifest(AIPerfBaseModel):
    """Metadata written alongside the JSONL dataset."""

    seed: int = Field(description="Random seed used for generation")
    block_size: int = Field(description="KV cache block size in tokens")
    num_sessions: int = Field(ge=1, description="Number of sessions generated")
    config_name: str | None = Field(
        default=None, description="Config name or path used for generation"
    )
    generation_params: SessionDistributionConfig = Field(
        description="Full generation config"
    )


class QualityMetric(AIPerfBaseModel):
    """Observed vs target comparison for one metric with full percentile breakdown."""

    target_mean: float | None = Field(
        default=None, description="Target mean from config"
    )
    target_median: float | None = Field(
        default=None, description="Target median from config"
    )
    observed: PercentileStats = Field(
        description="Full observed distribution statistics"
    )
    pct_error_mean: float | None = Field(
        default=None, description="Absolute percentage error on mean"
    )
    pct_error_median: float | None = Field(
        default=None, description="Absolute percentage error on median"
    )


class SessionEndStats(AIPerfBaseModel):
    """Statistics about how sessions ended."""

    total_sessions: int = Field(description="Total number of sessions")
    forced_retires: int = Field(description="Sessions ended by hitting context limit")
    probabilistic_resets: int = Field(
        description="Sessions ended by probabilistic reset"
    )
    retire_fraction: float = Field(description="Fraction of forced retires")
    reset_fraction: float = Field(description="Fraction of probabilistic resets")
    final_context_utilization: PercentileStats = Field(
        description="Distribution of last-turn input_length / max_prompt_tokens"
    )


class QualityReport(AIPerfBaseModel):
    """Quality report for a generated dataset."""

    config_summary: dict[str, float | int] = Field(
        description="Flat readable config parameters"
    )
    observed_vs_target: dict[str, QualityMetric] = Field(
        description="Per-metric quality checks with percentile breakdowns"
    )
    session_stats: PercentileStats = Field(description="Turns-per-session distribution")
    session_end_stats: SessionEndStats = Field(description="How sessions ended")
