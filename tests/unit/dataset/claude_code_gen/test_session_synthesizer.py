# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the session synthesizer."""

from __future__ import annotations

import numpy as np

from aiperf.dataset.claude_code_gen.models import (
    SessionDistributionConfig,
    SessionEndReason,
)
from aiperf.dataset.claude_code_gen.session_synthesizer import SessionSynthesizer


class TestSessionSynthesizer:
    def test_reproducible_with_same_seed(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        s1 = SessionSynthesizer(coding_config, seed=42)
        s2 = SessionSynthesizer(coding_config, seed=42)
        session_1 = s1.synthesize_session()
        session_2 = s2.synthesize_session()
        assert session_1.session_id == session_2.session_id
        assert len(session_1.turns) == len(session_2.turns)
        for t1, t2 in zip(session_1.turns, session_2.turns, strict=True):
            assert t1.input_length == t2.input_length
            assert t1.output_length == t2.output_length

    def test_different_seeds_produce_different_sessions(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        s1 = SessionSynthesizer(coding_config, seed=42)
        s2 = SessionSynthesizer(coding_config, seed=99)
        session_1 = s1.synthesize_session()
        session_2 = s2.synthesize_session()
        assert session_1.session_id != session_2.session_id

    def test_turn_indices_sequential(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()
        for i, turn in enumerate(session.turns):
            assert turn.turn_index == i

    def test_input_length_grows(self, coding_config: SessionDistributionConfig) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()
        if len(session.turns) > 1:
            for i in range(1, len(session.turns)):
                assert session.turns[i].input_length > session.turns[i - 1].input_length

    def test_hash_ids_prefix_property(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()
        for i in range(1, len(session.turns)):
            prev_ids = session.turns[i - 1].hash_ids
            curr_ids = session.turns[i].hash_ids
            assert curr_ids[: len(prev_ids)] == prev_ids

    def test_l1_ids_consistent_across_sessions(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(3)
        l1_blocks = synth.allocator.l1_blocks
        canonical_l1 = list(range(l1_blocks))
        for session in sessions:
            ids = session.turns[0].hash_ids
            l1_used = min(l1_blocks, len(ids))
            assert ids[:l1_used] == canonical_l1[:l1_used]

    def test_context_stays_under_max(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()
        for turn in session.turns:
            assert turn.input_length < coding_config.max_prompt_tokens

    def test_output_length_clipped_at_minimum(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(10)
        for session in sessions:
            for turn in session.turns:
                assert turn.output_length >= 30

    def test_first_turn_has_zero_delay(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()
        assert session.turns[0].delay_ms == 0.0

    def test_subsequent_turns_have_positive_delay(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()
        for turn in session.turns[1:]:
            assert turn.delay_ms > 0

    def test_timestamps_monotonically_increase(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()
        for i in range(1, len(session.turns)):
            assert session.turns[i].timestamp_ms > session.turns[i - 1].timestamp_ms

    def test_multiple_sessions_have_unique_ids(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(20)
        ids = [s.session_id for s in sessions]
        assert len(set(ids)) == len(ids)

    def test_end_reason_is_set(self, coding_config: SessionDistributionConfig) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(20)
        for session in sessions:
            assert session.end_reason in (
                SessionEndReason.FORCED_RETIRE,
                SessionEndReason.PROBABILISTIC_RESET,
            )


class TestSessionSynthesizerSmallConfig:
    def test_forced_retire_at_context_limit(
        self, small_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(small_config, seed=42)
        sessions = synth.synthesize_sessions(50)
        for session in sessions:
            for turn in session.turns:
                assert turn.input_length < small_config.max_prompt_tokens

    def test_sessions_have_at_least_one_turn(
        self, small_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(small_config, seed=42)
        sessions = synth.synthesize_sessions(50)
        for session in sessions:
            assert len(session.turns) >= 1


class TestMaxIsl:
    def test_no_turn_exceeds_max_prompt_tokens(
        self, small_config: SessionDistributionConfig
    ) -> None:
        """Turn 0 initial_ctx is clipped to max_prompt_tokens."""
        synth = SessionSynthesizer(small_config, seed=42)
        sessions = synth.synthesize_sessions(200)
        for session in sessions:
            for turn in session.turns:
                assert turn.input_length <= small_config.max_prompt_tokens

    def test_max_isl_override_clips_sessions(self) -> None:
        """Simulates --max-isl by using model_copy to lower max_prompt_tokens."""
        from aiperf.dataset.claude_code_gen.distributions import (
            lognormal_from_mean_median,
        )
        from aiperf.dataset.claude_code_gen.models import (
            CacheLayerConfig,
            MixtureDelayConfig,
            ResetConfig,
        )

        base = SessionDistributionConfig(
            system_prompt_tokens=100,
            initial_context=lognormal_from_mean_median(mean=5_000, median=4_000),
            new_tokens_per_turn=lognormal_from_mean_median(mean=200, median=100),
            generation_length=lognormal_from_mean_median(mean=50, median=30),
            inter_turn_delay=MixtureDelayConfig(
                agentic_fraction=0.7,
                agentic_delay=lognormal_from_mean_median(mean=3_000, median=2_000),
                human_delay=lognormal_from_mean_median(mean=45_000, median=30_000),
            ),
            reset=ResetConfig(base_probability=0.02, context_scaling=2.0),
            max_prompt_tokens=50_000,
            cache=CacheLayerConfig(layer1_tokens=200, block_size=64),
        )
        max_isl = 2_000
        clipped = base.model_copy(update={"max_prompt_tokens": max_isl})
        synth = SessionSynthesizer(clipped, seed=42)
        sessions = synth.synthesize_sessions(200)
        for session in sessions:
            for turn in session.turns:
                assert turn.input_length <= max_isl


class TestInitialContextFloor:
    def test_initial_context_exceeds_layer1_tokens(
        self, small_config: SessionDistributionConfig
    ) -> None:
        """The synthesizer floors initial_ctx to layer1_tokens + 1."""
        synth = SessionSynthesizer(small_config, seed=42)
        sessions = synth.synthesize_sessions(100)
        l1_tokens = small_config.cache.layer1_tokens
        for session in sessions:
            assert session.turns[0].input_length > l1_tokens

    def test_turn0_block_count_ge_l1_blocks(
        self, small_config: SessionDistributionConfig
    ) -> None:
        """Turn 0 should have at least as many blocks as L1 requires."""
        synth = SessionSynthesizer(small_config, seed=42)
        sessions = synth.synthesize_sessions(50)
        alloc = synth.allocator
        for session in sessions:
            assert len(session.turns[0].hash_ids) >= alloc.l1_blocks


class TestDistributionFidelity:
    def test_initial_context_mean_within_tolerance(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(500)
        initial_contexts = [s.turns[0].input_length for s in sessions]
        observed_mean = np.mean(initial_contexts)
        target_mean = coding_config.initial_context.mean
        pct_error = abs(observed_mean - target_mean) / target_mean * 100
        assert pct_error < 10, (
            f"Initial context mean {observed_mean:.0f} vs target {target_mean:.0f} ({pct_error:.1f}%)"
        )

    def test_generation_length_mean_within_tolerance(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(500)
        output_lens = [t.output_length for s in sessions for t in s.turns]
        observed_mean = np.mean(output_lens)
        target_mean = coding_config.generation_length.mean
        pct_error = abs(observed_mean - target_mean) / target_mean * 100
        assert pct_error < 15, (
            f"Generation length mean {observed_mean:.0f} vs target {target_mean:.0f} ({pct_error:.1f}%)"
        )
