# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AgenticLoadStrategy: closed-loop agentic load with deterministic trajectory assignment."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.strategies.agentic_load import (
    AgenticLoadStrategy,
    AgenticUser,
    _ActiveSession,
    _assign_conversations,
    _cache_bust_suffix,
)
from tests.unit.timing.conftest import make_sampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BENCHMARK_ID = "bench-test-001"


def _make_conversations(
    n: int,
    turns_per_conv: int = 5,
    include_children: bool = False,
) -> DatasetMetadata:
    convs = [
        ConversationMetadata(
            conversation_id=f"conv_{i}",
            turns=[TurnMetadata() for _ in range(turns_per_conv)],
        )
        for i in range(n)
    ]
    if include_children:
        convs.append(
            ConversationMetadata(
                conversation_id="child_0",
                turns=[TurnMetadata() for _ in range(3)],
                is_subagent_child=True,
            )
        )
        convs.append(
            ConversationMetadata(
                conversation_id="child_1",
                turns=[TurnMetadata() for _ in range(2)],
                is_subagent_child=True,
            )
        )
    return DatasetMetadata(
        conversations=convs,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _make_conversation_source(ds: DatasetMetadata) -> ConversationSource:
    conv_ids = [c.conversation_id for c in ds.conversations]
    sampler = make_sampler(conv_ids, DatasetSamplingStrategy.SEQUENTIAL)
    return ConversationSource(ds, sampler)


def _make_strategy(
    num_conversations: int = 10,
    turns_per_conv: int = 5,
    num_users: int = 3,
    user_spawn_rate: float = 2.0,
    settling_time_sec: float = 5.0,
    trajectories_per_user: int = 3,
    max_isl_offset: int = 0,
    seed: int | None = 42,
    benchmark_id: str = BENCHMARK_ID,
    include_children: bool = False,
) -> tuple[AgenticLoadStrategy, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Build strategy with mocked dependencies.

    Returns (strategy, scheduler, issuer, stop_checker, lifecycle).
    """
    ds = _make_conversations(num_conversations, turns_per_conv, include_children)
    src = _make_conversation_source(ds)

    scheduler = MagicMock()
    scheduler.schedule_at_perf_sec = MagicMock()
    scheduler.sleep = AsyncMock()

    stop_checker = MagicMock()
    stop_checker.can_send_any_turn = MagicMock(return_value=True)

    issuer = MagicMock()
    issuer.issue_credit = AsyncMock(return_value=True)

    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 1_000_000_000
    lifecycle.started_at_perf_sec = 1.0

    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_LOAD,
        expected_duration_sec=120.0,
        num_users=num_users,
        user_spawn_rate=user_spawn_rate,
        settling_time_sec=settling_time_sec,
        trajectories_per_user=trajectories_per_user,
        max_isl_offset=max_isl_offset,
        agentic_seed=seed,
        benchmark_id=benchmark_id,
    )

    strategy = AgenticLoadStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )
    return strategy, scheduler, issuer, stop_checker, lifecycle


def _make_credit(
    conv_id: str = "conv_0",
    corr_id: str = "xcorr-1",
    turn_index: int = 0,
    num_turns: int = 5,
    suffix: str | None = None,
) -> Credit:
    return Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id=conv_id,
        x_correlation_id=corr_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        system_prompt_suffix=suffix,
    )


# ===========================================================================
# Pure function tests
# ===========================================================================


class TestAssignConversations:
    def test_basic_assignment(self):
        ids = ["a", "b", "c", "d", "e", "f"]
        result = _assign_conversations(ids, num_users=2, per_user=3, seed=None)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3

    def test_deterministic_with_seed(self):
        ids = [f"conv_{i}" for i in range(10)]
        r1 = _assign_conversations(ids, num_users=3, per_user=3, seed=42)
        r2 = _assign_conversations(ids, num_users=3, per_user=3, seed=42)
        assert r1 == r2

    def test_different_seed_different_assignment(self):
        ids = [f"conv_{i}" for i in range(10)]
        r1 = _assign_conversations(ids, num_users=3, per_user=3, seed=42)
        r2 = _assign_conversations(ids, num_users=3, per_user=3, seed=99)
        assert r1 != r2

    def test_no_seed_preserves_order(self):
        ids = ["a", "b", "c", "d", "e", "f"]
        result = _assign_conversations(ids, num_users=2, per_user=3, seed=None)
        # Without seed, no shuffle: user 0 gets [a, b, c], user 1 gets [d, e, f]
        assert result[0] == ["a", "b", "c"]
        assert result[1] == ["d", "e", "f"]

    def test_non_overlapping_when_enough_conversations(self):
        ids = [f"conv_{i}" for i in range(20)]
        result = _assign_conversations(ids, num_users=4, per_user=5, seed=42)
        for u1 in range(4):
            for u2 in range(u1 + 1, 4):
                overlap = set(result[u1]) & set(result[u2])
                assert len(overlap) == 0, f"Users {u1} and {u2} overlap: {overlap}"

    def test_wraps_when_not_enough_conversations(self):
        ids = ["a", "b", "c"]
        result = _assign_conversations(ids, num_users=2, per_user=3, seed=None)
        # User 0 starts at 0: [a, b, c]
        # User 1 starts at 3 % 3 = 0: wraps around
        assert result[0] == ["a", "b", "c"]
        assert result[1] == ["a", "b", "c"]

    def test_single_user(self):
        ids = ["a", "b", "c"]
        result = _assign_conversations(ids, num_users=1, per_user=2, seed=None)
        assert result[0] == ["a", "b"]

    def test_single_conversation(self):
        ids = ["only"]
        result = _assign_conversations(ids, num_users=3, per_user=2, seed=None)
        for uid in range(3):
            assert result[uid] == ["only", "only"]

    def test_per_user_larger_than_pool(self):
        ids = ["a", "b"]
        result = _assign_conversations(ids, num_users=1, per_user=5, seed=None)
        assert result[0] == ["a", "b", "a", "b", "a"]

    def test_all_users_get_assignments(self):
        ids = [f"conv_{i}" for i in range(5)]
        result = _assign_conversations(ids, num_users=10, per_user=2, seed=42)
        assert len(result) == 10
        for uid in range(10):
            assert len(result[uid]) == 2


class TestCacheBustSuffix:
    def test_format(self):
        suffix = _cache_bust_suffix("bench-1", 0, 0, 0)
        assert suffix.startswith("\n\n[rid:")
        assert suffix.endswith("]")

    def test_hash_length(self):
        suffix = _cache_bust_suffix("bench-1", 0, 0, 0)
        # Extract hash between [rid: and ]
        hash_part = suffix.split("[rid:")[1].rstrip("]")
        assert len(hash_part) == 12

    def test_deterministic(self):
        s1 = _cache_bust_suffix("bench-1", 1, 2, 3)
        s2 = _cache_bust_suffix("bench-1", 1, 2, 3)
        assert s1 == s2

    def test_changes_with_pass_count(self):
        s1 = _cache_bust_suffix("bench-1", 0, 0, 0)
        s2 = _cache_bust_suffix("bench-1", 1, 0, 0)
        assert s1 != s2

    def test_changes_with_user_id(self):
        s1 = _cache_bust_suffix("bench-1", 0, 0, 0)
        s2 = _cache_bust_suffix("bench-1", 0, 1, 0)
        assert s1 != s2

    def test_changes_with_trajectory_index(self):
        s1 = _cache_bust_suffix("bench-1", 0, 0, 0)
        s2 = _cache_bust_suffix("bench-1", 0, 0, 1)
        assert s1 != s2

    def test_changes_with_benchmark_id(self):
        s1 = _cache_bust_suffix("bench-1", 0, 0, 0)
        s2 = _cache_bust_suffix("bench-2", 0, 0, 0)
        assert s1 != s2

    def test_matches_manual_sha256(self):
        benchmark_id = "test-bench"
        unique_str = f"{benchmark_id}:1:2:3"
        expected_hash = hashlib.sha256(unique_str.encode()).hexdigest()[:12]
        suffix = _cache_bust_suffix(benchmark_id, 1, 2, 3)
        assert suffix == f"\n\n[rid:{expected_hash}]"


# ===========================================================================
# Constructor tests
# ===========================================================================


class TestAgenticLoadInit:
    def test_valid_config(self):
        strategy, *_ = _make_strategy()
        assert strategy._num_users == 3
        assert strategy._user_spawn_rate == 2.0
        assert strategy._settling_time == 5.0
        assert strategy._trajectories_per_user == 3
        assert strategy._max_isl_offset == 0
        assert strategy._seed == 42
        assert strategy._benchmark_id == BENCHMARK_ID

    def test_num_users_none_raises(self):
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENTIC_LOAD,
            expected_duration_sec=120.0,
            num_users=None,
        )
        with pytest.raises(ValueError, match="num_users must be set and positive"):
            AgenticLoadStrategy(
                config=cfg,
                conversation_source=MagicMock(),
                scheduler=MagicMock(),
                stop_checker=MagicMock(),
                credit_issuer=MagicMock(),
                lifecycle=MagicMock(),
            )

    def test_num_users_zero_rejected_by_config(self):
        with pytest.raises(ValueError):
            CreditPhaseConfig(
                phase=CreditPhase.PROFILING,
                timing_mode=TimingMode.AGENTIC_LOAD,
                expected_duration_sec=120.0,
                num_users=0,
            )

    def test_num_users_negative_rejected_by_config(self):
        with pytest.raises(ValueError):
            CreditPhaseConfig(
                phase=CreditPhase.PROFILING,
                timing_mode=TimingMode.AGENTIC_LOAD,
                expected_duration_sec=120.0,
                num_users=-1,
            )

    def test_defaults_when_config_none(self):
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENTIC_LOAD,
            expected_duration_sec=120.0,
            num_users=5,
            user_spawn_rate=None,
            settling_time_sec=None,
            trajectories_per_user=None,
            max_isl_offset=None,
            agentic_seed=None,
            benchmark_id=None,
        )
        strategy = AgenticLoadStrategy(
            config=cfg,
            conversation_source=MagicMock(),
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=MagicMock(),
            lifecycle=MagicMock(),
        )
        assert strategy._user_spawn_rate == 1.0
        assert strategy._settling_time == 0.0
        assert strategy._trajectories_per_user == 20
        assert strategy._max_isl_offset == 0
        assert strategy._seed is None
        assert strategy._benchmark_id == "unknown"


# ===========================================================================
# setup_phase tests
# ===========================================================================


class TestSetupPhase:
    @pytest.mark.asyncio
    async def test_creates_users(self):
        strategy, *_ = _make_strategy(num_users=4, trajectories_per_user=2)
        await strategy.setup_phase()
        assert len(strategy._users) == 4
        for uid in range(4):
            user = strategy._users[uid]
            assert user.user_id == uid
            assert len(user.assigned_conversation_ids) == 2

    @pytest.mark.asyncio
    async def test_filters_subagent_children(self):
        strategy, *_ = _make_strategy(
            num_conversations=5,
            trajectories_per_user=2,
            include_children=True,
        )
        await strategy.setup_phase()
        for user in strategy._users.values():
            for conv_id in user.assigned_conversation_ids:
                assert not conv_id.startswith("child_")

    @pytest.mark.asyncio
    async def test_no_conversations_raises(self):
        ds = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="child_only",
                    turns=[TurnMetadata()],
                    is_subagent_child=True,
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        src = _make_conversation_source(ds)
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENTIC_LOAD,
            expected_duration_sec=120.0,
            num_users=1,
            trajectories_per_user=1,
            agentic_seed=42,
        )
        strategy = AgenticLoadStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=MagicMock(),
            lifecycle=MagicMock(),
        )
        with pytest.raises(ValueError, match="No conversations available"):
            await strategy.setup_phase()

    @pytest.mark.asyncio
    async def test_deterministic_assignment_with_seed(self):
        s1, *_ = _make_strategy(seed=42)
        s2, *_ = _make_strategy(seed=42)
        await s1.setup_phase()
        await s2.setup_phase()
        for uid in s1._users:
            assert (
                s1._users[uid].assigned_conversation_ids
                == s2._users[uid].assigned_conversation_ids
            )

    @pytest.mark.asyncio
    async def test_isl_offset_zero_when_disabled(self):
        strategy, *_ = _make_strategy(max_isl_offset=0)
        await strategy.setup_phase()
        for user in strategy._users.values():
            assert user.isl_offset == 0

    @pytest.mark.asyncio
    async def test_isl_offset_with_seed_deterministic(self):
        s1, *_ = _make_strategy(max_isl_offset=10, seed=42)
        s2, *_ = _make_strategy(max_isl_offset=10, seed=42)
        await s1.setup_phase()
        await s2.setup_phase()
        for uid in s1._users:
            assert s1._users[uid].isl_offset == s2._users[uid].isl_offset

    @pytest.mark.asyncio
    async def test_isl_offset_per_user_seed_independence(self):
        """Changing num_users should not change user 0's offset."""
        s1, *_ = _make_strategy(num_users=3, max_isl_offset=10, seed=42)
        s2, *_ = _make_strategy(num_users=5, max_isl_offset=10, seed=42)
        await s1.setup_phase()
        await s2.setup_phase()
        assert s1._users[0].isl_offset == s2._users[0].isl_offset
        assert s1._users[1].isl_offset == s2._users[1].isl_offset
        assert s1._users[2].isl_offset == s2._users[2].isl_offset

    @pytest.mark.asyncio
    async def test_isl_offset_in_range(self):
        strategy, *_ = _make_strategy(num_users=20, max_isl_offset=5, seed=42)
        await strategy.setup_phase()
        for user in strategy._users.values():
            assert 0 <= user.isl_offset <= 5

    @pytest.mark.asyncio
    async def test_isl_offset_without_seed(self):
        strategy, *_ = _make_strategy(max_isl_offset=10, seed=None)
        await strategy.setup_phase()
        for user in strategy._users.values():
            assert 0 <= user.isl_offset <= 10


# ===========================================================================
# execute_phase tests
# ===========================================================================


class TestExecutePhase:
    @pytest.mark.asyncio
    async def test_schedules_all_users(self):
        strategy, scheduler, _, stop_checker, _ = _make_strategy(num_users=4)
        stop_checker.can_send_any_turn = MagicMock(return_value=False)
        await strategy.setup_phase()
        await strategy.execute_phase()
        assert scheduler.schedule_at_perf_sec.call_count == 4

    @pytest.mark.asyncio
    async def test_staggered_spawn_times(self):
        strategy, scheduler, _, stop_checker, lifecycle = _make_strategy(
            num_users=3, user_spawn_rate=2.0
        )
        stop_checker.can_send_any_turn = MagicMock(return_value=False)
        await strategy.setup_phase()
        await strategy.execute_phase()

        calls = scheduler.schedule_at_perf_sec.call_args_list
        times = [call[0][0] for call in calls]
        # phase_start=1.0, spawn_rate=2.0
        # user 0: 1.0 + 0/2.0 = 1.0
        # user 1: 1.0 + 1/2.0 = 1.5
        # user 2: 1.0 + 2/2.0 = 2.0
        assert times[0] == pytest.approx(1.0)
        assert times[1] == pytest.approx(1.5)
        assert times[2] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_single_user_no_ramp(self):
        strategy, scheduler, _, stop_checker, lifecycle = _make_strategy(
            num_users=1, user_spawn_rate=1.0
        )
        stop_checker.can_send_any_turn = MagicMock(return_value=False)
        await strategy.setup_phase()
        await strategy.execute_phase()

        calls = scheduler.schedule_at_perf_sec.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_lifecycle_none_raises(self):
        strategy, _, _, _, lifecycle = _make_strategy()
        lifecycle.started_at_perf_ns = None
        await strategy.setup_phase()
        with pytest.raises(RuntimeError, match="started_at_perf_ns is not set"):
            await strategy.execute_phase()

    @pytest.mark.asyncio
    async def test_blocks_until_stop_condition(self):
        strategy, scheduler, _, stop_checker, _ = _make_strategy(num_users=1)
        stop_checker.can_send_any_turn = MagicMock(
            side_effect=[True, True, True, False]
        )
        await strategy.setup_phase()
        await strategy.execute_phase()
        # sleep called 3 times before can_send_any_turn returns False
        assert scheduler.sleep.call_count == 3
        scheduler.sleep.assert_called_with(1.0)


# ===========================================================================
# _spawn_user tests
# ===========================================================================


class TestSpawnUser:
    @pytest.mark.asyncio
    async def test_issues_credit_for_first_turn(self):
        strategy, _, issuer, _, _ = _make_strategy()
        await strategy.setup_phase()
        user = strategy._users[0]
        await strategy._spawn_user(user)
        assert issuer.issue_credit.call_count == 1
        turn: TurnToSend = issuer.issue_credit.call_args[0][0]
        assert turn.turn_index == 0
        assert turn.conversation_id in user.assigned_conversation_ids

    @pytest.mark.asyncio
    async def test_empty_assignments_no_credit(self):
        strategy, _, issuer, _, _ = _make_strategy()
        user = AgenticUser(user_id=99, assigned_conversation_ids=[])
        await strategy._spawn_user(user)
        assert issuer.issue_credit.call_count == 0


# ===========================================================================
# _build_first_turn_for_user tests
# ===========================================================================


class TestBuildFirstTurn:
    @pytest.mark.asyncio
    async def test_basic_turn(self):
        strategy, *_ = _make_strategy(turns_per_conv=5)
        await strategy.setup_phase()
        user = strategy._users[0]
        turn = strategy._build_first_turn_for_user(user)
        assert turn is not None
        assert turn.turn_index == 0
        assert turn.num_turns == 5
        assert turn.system_prompt_suffix is not None
        assert "[rid:" in turn.system_prompt_suffix

    @pytest.mark.asyncio
    async def test_registers_active_session(self):
        strategy, *_ = _make_strategy()
        await strategy.setup_phase()
        user = strategy._users[0]
        turn = strategy._build_first_turn_for_user(user)
        assert turn.x_correlation_id in strategy._active_sessions
        session = strategy._active_sessions[turn.x_correlation_id]
        assert session.user is user
        assert session.conversation_id == turn.conversation_id

    @pytest.mark.asyncio
    async def test_isl_offset_applied_on_first_trajectory(self):
        strategy, *_ = _make_strategy(max_isl_offset=3, seed=42, turns_per_conv=10)
        await strategy.setup_phase()
        user = strategy._users[0]
        # Force a known offset
        user.isl_offset = 3
        user.isl_offset_applied = False
        turn = strategy._build_first_turn_for_user(user)
        assert turn.turn_index == 3
        assert user.isl_offset_applied is True

    @pytest.mark.asyncio
    async def test_isl_offset_not_applied_on_second_trajectory(self):
        strategy, *_ = _make_strategy(max_isl_offset=3, seed=42, turns_per_conv=10)
        await strategy.setup_phase()
        user = strategy._users[0]
        user.isl_offset = 3
        user.isl_offset_applied = True  # Already applied
        turn = strategy._build_first_turn_for_user(user)
        assert turn.turn_index == 0

    @pytest.mark.asyncio
    async def test_isl_offset_clamped_to_num_turns(self):
        strategy, *_ = _make_strategy(max_isl_offset=100, seed=42, turns_per_conv=3)
        await strategy.setup_phase()
        user = strategy._users[0]
        user.isl_offset = 100
        user.isl_offset_applied = False
        turn = strategy._build_first_turn_for_user(user)
        # Clamped to num_turns - 1 = 2
        assert turn.turn_index == 2

    @pytest.mark.asyncio
    async def test_cache_bust_suffix_changes_per_trajectory(self):
        strategy, *_ = _make_strategy(trajectories_per_user=3)
        await strategy.setup_phase()
        user = strategy._users[0]

        user.current_trajectory_index = 0
        t1 = strategy._build_first_turn_for_user(user)
        user.current_trajectory_index = 1
        t2 = strategy._build_first_turn_for_user(user)

        assert t1.system_prompt_suffix != t2.system_prompt_suffix

    @pytest.mark.asyncio
    async def test_cache_bust_suffix_changes_per_pass(self):
        strategy, *_ = _make_strategy(trajectories_per_user=2)
        await strategy.setup_phase()
        user = strategy._users[0]

        user.pass_count = 0
        user.current_trajectory_index = 0
        t1 = strategy._build_first_turn_for_user(user)

        user.pass_count = 1
        user.current_trajectory_index = 0
        t2 = strategy._build_first_turn_for_user(user)

        assert t1.system_prompt_suffix != t2.system_prompt_suffix

    @pytest.mark.asyncio
    async def test_each_call_creates_unique_correlation_id(self):
        strategy, *_ = _make_strategy()
        await strategy.setup_phase()
        user = strategy._users[0]
        t1 = strategy._build_first_turn_for_user(user)
        t2 = strategy._build_first_turn_for_user(user)
        assert t1.x_correlation_id != t2.x_correlation_id

    def test_empty_assignments_returns_none(self):
        strategy, *_ = _make_strategy()
        user = AgenticUser(user_id=99, assigned_conversation_ids=[])
        assert strategy._build_first_turn_for_user(user) is None


# ===========================================================================
# handle_credit_return tests
# ===========================================================================


class TestHandleCreditReturn:
    @pytest.mark.asyncio
    async def test_non_final_turn_issues_next(self):
        strategy, _, issuer, _, _ = _make_strategy()
        await strategy.setup_phase()
        user = strategy._users[0]
        turn = strategy._build_first_turn_for_user(user)

        credit = _make_credit(
            conv_id=turn.conversation_id,
            corr_id=turn.x_correlation_id,
            turn_index=0,
            num_turns=5,
            suffix=turn.system_prompt_suffix,
        )
        await strategy.handle_credit_return(credit)

        assert issuer.issue_credit.call_count == 1
        next_turn: TurnToSend = issuer.issue_credit.call_args[0][0]
        assert next_turn.turn_index == 1
        assert next_turn.conversation_id == credit.conversation_id
        assert next_turn.x_correlation_id == credit.x_correlation_id
        assert next_turn.system_prompt_suffix == credit.system_prompt_suffix

    @pytest.mark.asyncio
    async def test_final_turn_advances_trajectory(self):
        strategy, _, issuer, stop_checker, _ = _make_strategy(trajectories_per_user=3)
        await strategy.setup_phase()
        user = strategy._users[0]

        turn = strategy._build_first_turn_for_user(user)
        assert user.current_trajectory_index == 0

        credit = _make_credit(
            conv_id=turn.conversation_id,
            corr_id=turn.x_correlation_id,
            turn_index=4,
            num_turns=5,
        )
        await strategy.handle_credit_return(credit)

        assert user.current_trajectory_index == 1
        assert issuer.issue_credit.call_count == 1
        next_turn: TurnToSend = issuer.issue_credit.call_args[0][0]
        # New trajectory means new correlation ID
        assert next_turn.x_correlation_id != credit.x_correlation_id
        assert next_turn.turn_index == 0

    @pytest.mark.asyncio
    async def test_final_turn_cleans_up_session(self):
        strategy, _, issuer, _, _ = _make_strategy()
        await strategy.setup_phase()
        user = strategy._users[0]

        turn = strategy._build_first_turn_for_user(user)
        corr_id = turn.x_correlation_id
        assert corr_id in strategy._active_sessions

        credit = _make_credit(
            conv_id=turn.conversation_id,
            corr_id=corr_id,
            turn_index=4,
            num_turns=5,
        )
        await strategy.handle_credit_return(credit)

        assert corr_id not in strategy._active_sessions

    @pytest.mark.asyncio
    async def test_wraps_around_increments_pass_count(self):
        strategy, _, issuer, _, _ = _make_strategy(trajectories_per_user=2)
        await strategy.setup_phase()
        user = strategy._users[0]
        user.current_trajectory_index = 1  # At last trajectory
        assert user.pass_count == 0

        turn = strategy._build_first_turn_for_user(user)
        credit = _make_credit(
            conv_id=turn.conversation_id,
            corr_id=turn.x_correlation_id,
            turn_index=4,
            num_turns=5,
        )
        await strategy.handle_credit_return(credit)

        assert user.current_trajectory_index == 0
        assert user.pass_count == 1

    @pytest.mark.asyncio
    async def test_stop_condition_prevents_new_trajectory(self):
        strategy, _, issuer, stop_checker, _ = _make_strategy()
        await strategy.setup_phase()
        user = strategy._users[0]

        turn = strategy._build_first_turn_for_user(user)
        stop_checker.can_send_any_turn = MagicMock(return_value=False)

        credit = _make_credit(
            conv_id=turn.conversation_id,
            corr_id=turn.x_correlation_id,
            turn_index=4,
            num_turns=5,
        )
        await strategy.handle_credit_return(credit)

        assert issuer.issue_credit.call_count == 0

    @pytest.mark.asyncio
    async def test_unknown_correlation_id_ignored(self):
        strategy, _, issuer, _, _ = _make_strategy()
        await strategy.setup_phase()

        credit = _make_credit(corr_id="unknown-corr-id")
        await strategy.handle_credit_return(credit)

        assert issuer.issue_credit.call_count == 0

    @pytest.mark.asyncio
    async def test_closed_loop_preserves_suffix(self):
        strategy, _, issuer, _, _ = _make_strategy()
        await strategy.setup_phase()
        user = strategy._users[0]
        turn = strategy._build_first_turn_for_user(user)
        suffix = turn.system_prompt_suffix

        # Simulate mid-conversation credit return
        credit = _make_credit(
            conv_id=turn.conversation_id,
            corr_id=turn.x_correlation_id,
            turn_index=2,
            num_turns=5,
            suffix=suffix,
        )
        await strategy.handle_credit_return(credit)

        next_turn: TurnToSend = issuer.issue_credit.call_args[0][0]
        assert next_turn.system_prompt_suffix == suffix

    @pytest.mark.asyncio
    async def test_new_trajectory_gets_new_suffix(self):
        strategy, _, issuer, _, _ = _make_strategy(trajectories_per_user=3)
        await strategy.setup_phase()
        user = strategy._users[0]

        turn = strategy._build_first_turn_for_user(user)
        old_suffix = turn.system_prompt_suffix

        credit = _make_credit(
            conv_id=turn.conversation_id,
            corr_id=turn.x_correlation_id,
            turn_index=4,
            num_turns=5,
            suffix=old_suffix,
        )
        await strategy.handle_credit_return(credit)

        next_turn: TurnToSend = issuer.issue_credit.call_args[0][0]
        assert next_turn.system_prompt_suffix != old_suffix


# ===========================================================================
# Full lifecycle / integration-style tests
# ===========================================================================


class TestAgenticLoadLifecycle:
    @pytest.mark.asyncio
    async def test_multi_turn_conversation_flow(self):
        """Simulate a user completing a full 3-turn conversation."""
        strategy, _, issuer, _, _ = _make_strategy(
            num_users=1, turns_per_conv=3, trajectories_per_user=2
        )
        await strategy.setup_phase()
        user = strategy._users[0]

        # Spawn user
        await strategy._spawn_user(user)
        assert issuer.issue_credit.call_count == 1
        turn: TurnToSend = issuer.issue_credit.call_args[0][0]
        conv_id = turn.conversation_id
        corr_id = turn.x_correlation_id
        suffix = turn.system_prompt_suffix

        # Turn 0 -> Turn 1
        credit = _make_credit(
            conv_id=conv_id, corr_id=corr_id, turn_index=0, num_turns=3, suffix=suffix
        )
        await strategy.handle_credit_return(credit)
        assert issuer.issue_credit.call_count == 2
        t1 = issuer.issue_credit.call_args[0][0]
        assert t1.turn_index == 1
        assert t1.system_prompt_suffix == suffix

        # Turn 1 -> Turn 2 (final)
        credit = _make_credit(
            conv_id=conv_id, corr_id=corr_id, turn_index=1, num_turns=3, suffix=suffix
        )
        await strategy.handle_credit_return(credit)
        assert issuer.issue_credit.call_count == 3
        t2 = issuer.issue_credit.call_args[0][0]
        assert t2.turn_index == 2
        assert t2.system_prompt_suffix == suffix

        # Turn 2 final -> advance trajectory
        credit = _make_credit(
            conv_id=conv_id, corr_id=corr_id, turn_index=2, num_turns=3, suffix=suffix
        )
        await strategy.handle_credit_return(credit)
        assert issuer.issue_credit.call_count == 4
        new_turn = issuer.issue_credit.call_args[0][0]
        assert new_turn.turn_index == 0
        assert new_turn.x_correlation_id != corr_id
        assert new_turn.system_prompt_suffix != suffix

    @pytest.mark.asyncio
    async def test_multiple_users_independent(self):
        strategy, _, issuer, _, _ = _make_strategy(
            num_users=2, turns_per_conv=2, trajectories_per_user=2
        )
        await strategy.setup_phase()

        # Spawn both users
        await strategy._spawn_user(strategy._users[0])
        await strategy._spawn_user(strategy._users[1])
        assert issuer.issue_credit.call_count == 2

        # Get their correlation IDs
        t0: TurnToSend = issuer.issue_credit.call_args_list[0][0][0]
        t1: TurnToSend = issuer.issue_credit.call_args_list[1][0][0]
        assert t0.x_correlation_id != t1.x_correlation_id

        # Complete user 0's first turn
        credit_0 = _make_credit(
            conv_id=t0.conversation_id,
            corr_id=t0.x_correlation_id,
            turn_index=0,
            num_turns=2,
            suffix=t0.system_prompt_suffix,
        )
        await strategy.handle_credit_return(credit_0)
        assert issuer.issue_credit.call_count == 3

        # Complete user 1's first turn
        credit_1 = _make_credit(
            conv_id=t1.conversation_id,
            corr_id=t1.x_correlation_id,
            turn_index=0,
            num_turns=2,
            suffix=t1.system_prompt_suffix,
        )
        await strategy.handle_credit_return(credit_1)
        assert issuer.issue_credit.call_count == 4

    @pytest.mark.asyncio
    async def test_pass_wrap_changes_cache_bust(self):
        """After wrapping all trajectories, pass_count increments and suffix changes."""
        strategy, _, issuer, _, _ = _make_strategy(
            num_users=1, turns_per_conv=1, trajectories_per_user=1
        )
        await strategy.setup_phase()
        user = strategy._users[0]

        # Spawn: trajectory 0, pass 0
        await strategy._spawn_user(user)
        t0: TurnToSend = issuer.issue_credit.call_args[0][0]
        suffix_pass0 = t0.system_prompt_suffix

        # Complete the single-turn conversation -> wraps to pass 1
        credit = _make_credit(
            conv_id=t0.conversation_id,
            corr_id=t0.x_correlation_id,
            turn_index=0,
            num_turns=1,
        )
        await strategy.handle_credit_return(credit)
        t1: TurnToSend = issuer.issue_credit.call_args[0][0]
        suffix_pass1 = t1.system_prompt_suffix

        assert user.pass_count == 1
        assert suffix_pass0 != suffix_pass1


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_single_turn_conversations(self):
        """Each conversation has only 1 turn, so every credit is final."""
        strategy, _, issuer, _, _ = _make_strategy(
            num_users=1, turns_per_conv=1, trajectories_per_user=3
        )
        await strategy.setup_phase()
        user = strategy._users[0]
        await strategy._spawn_user(user)

        for _i in range(3):
            turn: TurnToSend = issuer.issue_credit.call_args[0][0]
            credit = _make_credit(
                conv_id=turn.conversation_id,
                corr_id=turn.x_correlation_id,
                turn_index=0,
                num_turns=1,
            )
            await strategy.handle_credit_return(credit)

        # After 3 final turns + initial spawn = 4 issue_credit calls
        assert issuer.issue_credit.call_count == 4
        assert user.current_trajectory_index == 0
        assert user.pass_count == 1

    @pytest.mark.asyncio
    async def test_isl_offset_with_single_turn_conversation(self):
        """ISL offset on a 1-turn conversation should clamp to 0 (num_turns - 1 = 0)."""
        strategy, *_ = _make_strategy(turns_per_conv=1, max_isl_offset=5, seed=42)
        await strategy.setup_phase()
        user = strategy._users[0]
        user.isl_offset = 5
        user.isl_offset_applied = False
        turn = strategy._build_first_turn_for_user(user)
        assert turn.turn_index == 0  # min(5, 1-1) = 0

    @pytest.mark.asyncio
    async def test_large_number_of_users(self):
        strategy, scheduler, _, stop_checker, _ = _make_strategy(
            num_users=100, num_conversations=50, trajectories_per_user=10
        )
        stop_checker.can_send_any_turn = MagicMock(return_value=False)
        await strategy.setup_phase()
        await strategy.execute_phase()
        assert len(strategy._users) == 100
        assert scheduler.schedule_at_perf_sec.call_count == 100

    @pytest.mark.asyncio
    async def test_concurrent_sessions_tracked_independently(self):
        """Multiple active sessions from different users don't interfere."""
        strategy, _, issuer, _, _ = _make_strategy(num_users=3, turns_per_conv=3)
        await strategy.setup_phase()

        # Spawn all users
        for user in strategy._users.values():
            await strategy._spawn_user(user)

        assert len(strategy._active_sessions) == 3
        corr_ids = list(strategy._active_sessions.keys())
        assert len(set(corr_ids)) == 3  # All unique

    @pytest.mark.asyncio
    async def test_active_session_dataclass(self):
        user = AgenticUser(user_id=0, assigned_conversation_ids=["conv_0"])
        session = _ActiveSession(user=user, conversation_id="conv_0")
        assert session.user is user
        assert session.conversation_id == "conv_0"

    @pytest.mark.asyncio
    async def test_agentic_user_defaults(self):
        user = AgenticUser(user_id=5, assigned_conversation_ids=["a", "b"])
        assert user.current_trajectory_index == 0
        assert user.pass_count == 0
        assert user.isl_offset == 0
        assert user.isl_offset_applied is False
