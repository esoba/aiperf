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
            GroupConfig,
            MixtureDelayConfig,
            ResetConfig,
        )

        base = SessionDistributionConfig(
            system_prompt_tokens=100,
            new_tokens_per_turn=lognormal_from_mean_median(mean=200, median=100),
            generation_length=lognormal_from_mean_median(mean=50, median=30),
            inter_turn_delay=MixtureDelayConfig(
                agentic_fraction=0.7,
                agentic_delay=lognormal_from_mean_median(mean=3_000, median=2_000),
                human_delay=lognormal_from_mean_median(mean=45_000, median=30_000),
            ),
            reset=ResetConfig(base_probability=0.02, context_scaling=2.0),
            max_prompt_tokens=50_000,
            cache=CacheLayerConfig(
                layer1_tokens=100,
                layer1_5_tokens=50,
                layer2=lognormal_from_mean_median(mean=4_000, median=3_000),
                block_size=64,
            ),
            group=GroupConfig(num_groups=5, zipf_alpha=1.2),
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


class TestGroupAssignment:
    def test_group_ids_within_range(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Every session's group_id must be in [0, num_groups)."""
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(100)
        num_groups = coding_config.group.num_groups
        for session in sessions:
            assert 0 <= session.group_id < num_groups

    def test_zipf_distribution_is_skewed(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Group 0 should appear more often than uniform (Zipf skew)."""
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(500)
        group_counts = np.bincount(
            [s.group_id for s in sessions],
            minlength=coding_config.group.num_groups,
        )
        uniform_expected = 500 / coding_config.group.num_groups
        assert group_counts[0] > uniform_expected * 2, (
            f"Group 0 count {group_counts[0]} not significantly above "
            f"uniform expectation {uniform_expected:.0f}"
        )

    def test_multiple_groups_used(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """With 500 sessions and 50 groups, at least 10 distinct groups should appear."""
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(500)
        distinct_groups = len({s.group_id for s in sessions})
        assert distinct_groups >= 10

    def test_same_group_shares_l15_blocks(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Sessions in the same group must share identical L1.5 hash IDs."""
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(50)
        alloc = synth.allocator
        l1 = alloc.l1_blocks
        l15 = alloc.l15_blocks

        by_group: dict[int, list] = {}
        for s in sessions:
            by_group.setdefault(s.group_id, []).append(s)

        for group_id, group_sessions in by_group.items():
            if len(group_sessions) < 2:
                continue
            ref = group_sessions[0].turns[0].hash_ids[l1 : l1 + l15]
            for s in group_sessions[1:]:
                actual = s.turns[0].hash_ids[l1 : l1 + l15]
                assert actual == ref, (
                    f"Group {group_id}: L1.5 mismatch between "
                    f"{group_sessions[0].session_id} and {s.session_id}"
                )

    def test_different_groups_have_different_l15_blocks(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Sessions in different groups must have different L1.5 hash IDs."""
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(50)
        alloc = synth.allocator
        l1 = alloc.l1_blocks
        l15 = alloc.l15_blocks

        by_group: dict[int, list] = {}
        for s in sessions:
            by_group.setdefault(s.group_id, []).append(s)

        group_ids = list(by_group.keys())
        if len(group_ids) >= 2:
            s_a = by_group[group_ids[0]][0]
            s_b = by_group[group_ids[1]][0]
            l15_a = s_a.turns[0].hash_ids[l1 : l1 + l15]
            l15_b = s_b.turns[0].hash_ids[l1 : l1 + l15]
            assert l15_a != l15_b


class TestDistributionFidelity:
    def test_initial_context_mean_within_tolerance(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(500)
        initial_contexts = [s.turns[0].input_length for s in sessions]
        observed_mean = np.mean(initial_contexts)
        cache = coding_config.cache
        target_mean = cache.layer1_tokens + cache.layer1_5_tokens + cache.layer2.mean
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
