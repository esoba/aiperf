# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Claude Code gen tests."""

from __future__ import annotations

import pytest

from aiperf.dataset.claude_code_gen.distributions import lognormal_from_mean_median
from aiperf.dataset.claude_code_gen.models import (
    CacheLayerConfig,
    MixtureDelayConfig,
    ResetConfig,
    SessionDistributionConfig,
)


@pytest.fixture
def coding_config() -> SessionDistributionConfig:
    """Default coding config."""
    return SessionDistributionConfig()


@pytest.fixture
def small_config() -> SessionDistributionConfig:
    """Small config for fast tests - low max_prompt_tokens to force resets."""
    return SessionDistributionConfig(
        system_prompt_tokens=100,
        initial_context=lognormal_from_mean_median(mean=500, median=400),
        new_tokens_per_turn=lognormal_from_mean_median(mean=200, median=100),
        generation_length=lognormal_from_mean_median(mean=50, median=30),
        inter_turn_delay=MixtureDelayConfig(
            agentic_fraction=0.7,
            agentic_delay=lognormal_from_mean_median(mean=3_000, median=2_000),
            human_delay=lognormal_from_mean_median(mean=45_000, median=30_000),
        ),
        reset=ResetConfig(base_probability=0.02, context_scaling=2.0),
        max_prompt_tokens=5_000,
        cache=CacheLayerConfig(layer1_tokens=200, block_size=64),
    )
