# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SubagentSessionManager and per-strategy agent_depth guards."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    SubagentSpawnInfo,
    TurnMetadata,
)
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.subagent_manager import SubagentSessionManager
from tests.unit.timing.conftest import make_sampler

# =============================================================================
# Helpers
# =============================================================================


def _make_credit(
    *,
    conv_id: str = "conv_0",
    corr_id: str = "xcorr-1",
    turn_index: int = 0,
    num_turns: int = 5,
    agent_depth: int = 0,
) -> Credit:
    return Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id=conv_id,
        x_correlation_id=corr_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        agent_depth=agent_depth,
    )


def _make_dataset_and_source(
    *,
    spawn_at: int = 2,
    num_children: int = 2,
    child_turns: int = 3,
    is_background: bool = False,
) -> tuple[ConversationSource, list[str]]:
    """Create a dataset with one parent conversation that has a subagent spawn."""
    parent_turns = []
    for i in range(6):
        spawn_ids = ["s0"] if i == spawn_at + 1 else []
        parent_turns.append(
            TurnMetadata(
                delay_ms=200.0 if i > 0 else None,
                input_tokens=500 + i * 100,
                subagent_spawn_ids=spawn_ids,
            )
        )

    child_conv_ids = [f"conv_0_s0_c{ci}" for ci in range(num_children)]
    spawn = SubagentSpawnInfo(
        spawn_id="s0",
        child_conversation_ids=child_conv_ids,
        join_turn_index=spawn_at + 1,
        is_background=is_background,
    )

    convs = [
        ConversationMetadata(
            conversation_id="conv_0",
            turns=parent_turns,
            subagent_spawns=[spawn],
        )
    ]
    for child_id in child_conv_ids:
        convs.append(
            ConversationMetadata(
                conversation_id=child_id,
                turns=[
                    TurnMetadata(input_tokens=300 + j * 50) for j in range(child_turns)
                ],
                agent_depth=1,
            )
        )

    ds = DatasetMetadata(
        conversations=convs,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    sampler = make_sampler(["conv_0"], ds.sampling_strategy)
    src = ConversationSource(ds, sampler)
    return src, child_conv_ids


def _make_manager(
    *,
    spawn_at: int = 2,
    num_children: int = 2,
    is_background: bool = False,
    inner_has_child_hook: bool = False,
) -> tuple[SubagentSessionManager, MagicMock, MagicMock, MagicMock, list[str]]:
    """Create a SubagentSessionManager with a mocked inner strategy."""
    src, child_conv_ids = _make_dataset_and_source(
        spawn_at=spawn_at,
        num_children=num_children,
        is_background=is_background,
    )

    inner = MagicMock()
    inner.setup_phase = AsyncMock()
    inner.execute_phase = AsyncMock()
    inner.handle_credit_return = AsyncMock()
    inner.on_ttft_sample = MagicMock()
    inner.set_request_rate = MagicMock()
    if inner_has_child_hook:
        inner.on_child_session_started = MagicMock()
    else:
        del inner.on_child_session_started

    scheduler = MagicMock()
    scheduler.execute_async = MagicMock()

    issuer = MagicMock()
    issuer.issue_credit = AsyncMock(return_value=True)

    manager = SubagentSessionManager(
        inner=inner,
        conversation_source=src,
        credit_issuer=issuer,
        scheduler=scheduler,
    )
    return manager, inner, scheduler, issuer, child_conv_ids


# =============================================================================
# SubagentSessionManager unit tests
# =============================================================================


class TestSubagentSessionManagerDelegation:
    """Test that normal operations delegate to inner strategy."""

    @pytest.mark.asyncio
    async def test_setup_phase_delegates(self):
        manager, inner, _, _, _ = _make_manager()
        await manager.setup_phase()
        inner.setup_phase.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_phase_delegates(self):
        manager, inner, _, _, _ = _make_manager()
        await manager.execute_phase()
        inner.execute_phase.assert_awaited_once()

    def test_getattr_proxies_to_inner(self):
        manager, inner, _, _, _ = _make_manager()
        manager.on_ttft_sample(100)
        inner.on_ttft_sample.assert_called_once_with(100)

    def test_getattr_proxies_set_request_rate(self):
        manager, inner, _, _, _ = _make_manager()
        manager.set_request_rate(5.0)
        inner.set_request_rate.assert_called_once_with(5.0)

    @pytest.mark.asyncio
    async def test_normal_non_final_turn_delegates(self):
        """Path C: normal turn with no subagent spawn delegates to inner."""
        manager, inner, _, _, _ = _make_manager()

        # Turn 0 -> next turn (1) has no subagent_spawn_ids
        credit = _make_credit(turn_index=0, num_turns=6)
        await manager.handle_credit_return(credit)

        inner.handle_credit_return.assert_awaited_once_with(credit)

    @pytest.mark.asyncio
    async def test_normal_final_turn_delegates(self):
        """Path C: final turn with unknown corr_id delegates to inner."""
        manager, inner, _, _, _ = _make_manager()

        credit = _make_credit(turn_index=4, num_turns=5)
        await manager.handle_credit_return(credit)

        inner.handle_credit_return.assert_awaited_once_with(credit)


class TestSubagentSessionManagerSpawnInterception:
    """Test Path B: spawn interception."""

    @pytest.mark.asyncio
    async def test_spawn_intercepts_and_does_not_delegate(self):
        """When next turn has subagent_spawn_ids, inner is NOT called."""
        manager, inner, scheduler, _, child_ids = _make_manager()

        # Turn 2 complete -> next turn (3) has subagent_spawn_ids="s0"
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        await manager.handle_credit_return(credit)

        inner.handle_credit_return.assert_not_awaited()
        assert "parent-1" in manager._pending_subagent_joins
        # 2 children issued
        assert scheduler.execute_async.call_count == 2

    @pytest.mark.asyncio
    async def test_spawn_calls_child_hook_when_available(self):
        """on_child_session_started hook is called when inner has it."""
        manager, inner, _, _, _ = _make_manager(inner_has_child_hook=True)

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        await manager.handle_credit_return(credit)

        assert inner.on_child_session_started.call_count == 2
        for call in inner.on_child_session_started.call_args_list:
            assert call.args[1] == 1  # child_depth

    @pytest.mark.asyncio
    async def test_spawn_does_not_call_hook_when_missing(self):
        """No error when inner lacks on_child_session_started."""
        manager, inner, _, _, _ = _make_manager(inner_has_child_hook=False)

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        await manager.handle_credit_return(credit)

        # Should succeed without error
        assert "parent-1" in manager._pending_subagent_joins


class TestSubagentSessionManagerChildComplete:
    """Test Path A: child final turn with join accounting."""

    @pytest.mark.asyncio
    async def test_child_final_delegates_to_inner(self):
        """Path A: child final turn does join accounting AND delegates to inner."""
        manager, inner, scheduler, _, child_ids = _make_manager()

        # Set up spawn
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])
        scheduler.execute_async.reset_mock()

        child_corr_ids = list(manager._subagent_child_to_parent.keys())

        # Child's final turn
        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
        await manager.handle_credit_return(child_credit)

        # Join accounting happened
        pending = manager._pending_subagent_joins.get("parent-1")
        assert pending is not None
        assert pending.completed_count == 1

        # Inner was also called for cleanup
        inner.handle_credit_return.assert_awaited_once_with(child_credit)

    @pytest.mark.asyncio
    async def test_all_children_complete_dispatches_join(self):
        manager, inner, scheduler, _, child_ids = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])
        scheduler.execute_async.reset_mock()

        child_corr_ids = list(manager._subagent_child_to_parent.keys())

        for i, child_corr_id in enumerate(child_corr_ids):
            child_credit = _make_credit(
                conv_id=child_ids[i],
                corr_id=child_corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            manager._handle_subagent_child_complete(child_credit)

        assert "parent-1" not in manager._pending_subagent_joins
        assert scheduler.execute_async.call_count >= 1


class TestSubagentSessionManagerMultiSpawn:
    """Test multiple spawn_ids on a single join turn."""

    @pytest.mark.asyncio
    async def test_multi_spawn_blocking_aggregates_expected_count(self):
        """Two blocking spawns on one turn: expected_count = sum of all children."""
        # Build dataset with 2 spawns both joining at turn 3
        parent_turns = []
        for i in range(6):
            spawn_ids = ["s0", "s1"] if i == 3 else []
            parent_turns.append(
                TurnMetadata(
                    delay_ms=200.0 if i > 0 else None,
                    input_tokens=500 + i * 100,
                    subagent_spawn_ids=spawn_ids,
                )
            )

        # 2 children per spawn = 4 total
        s0_children = ["conv_0_s0_c0", "conv_0_s0_c1"]
        s1_children = ["conv_0_s1_c0", "conv_0_s1_c1"]
        all_child_ids = s0_children + s1_children

        convs = [
            ConversationMetadata(
                conversation_id="conv_0",
                turns=parent_turns,
                subagent_spawns=[
                    SubagentSpawnInfo(
                        spawn_id="s0",
                        child_conversation_ids=s0_children,
                        join_turn_index=3,
                    ),
                    SubagentSpawnInfo(
                        spawn_id="s1",
                        child_conversation_ids=s1_children,
                        join_turn_index=3,
                    ),
                ],
            )
        ]
        for cid in all_child_ids:
            convs.append(
                ConversationMetadata(
                    conversation_id=cid,
                    turns=[TurnMetadata(input_tokens=300 + j * 50) for j in range(3)],
                    agent_depth=1,
                )
            )

        ds = DatasetMetadata(
            conversations=convs,
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = make_sampler(["conv_0"], ds.sampling_strategy)
        src = ConversationSource(ds, sampler)

        inner = MagicMock()
        inner.setup_phase = AsyncMock()
        inner.execute_phase = AsyncMock()
        inner.handle_credit_return = AsyncMock()
        del inner.on_child_session_started

        scheduler = MagicMock()
        scheduler.execute_async = MagicMock()
        issuer = MagicMock()
        issuer.issue_credit = AsyncMock(return_value=True)

        manager = SubagentSessionManager(
            inner=inner,
            conversation_source=src,
            credit_issuer=issuer,
            scheduler=scheduler,
        )

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0", "s1"])

        # Should have one pending join with expected_count = 4
        assert "parent-1" in manager._pending_subagent_joins
        pending = manager._pending_subagent_joins["parent-1"]
        assert pending.expected_count == 4
        assert pending.completed_count == 0

        # 4 children issued
        assert scheduler.execute_async.call_count == 4
        assert len(manager._subagent_child_to_parent) == 4

        # Complete all 4 children -> join dispatches
        scheduler.execute_async.reset_mock()
        child_corr_ids = list(manager._subagent_child_to_parent.keys())
        for i, child_corr_id in enumerate(child_corr_ids):
            child_credit = _make_credit(
                conv_id=all_child_ids[i],
                corr_id=child_corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            manager._handle_subagent_child_complete(child_credit)

        assert "parent-1" not in manager._pending_subagent_joins
        assert scheduler.execute_async.call_count >= 1

    @pytest.mark.asyncio
    async def test_mixed_blocking_and_background_spawns(self):
        """Blocking + background spawns: only blocking children count toward join."""
        parent_turns = []
        for i in range(6):
            spawn_ids = ["s0", "s1"] if i == 3 else []
            parent_turns.append(
                TurnMetadata(
                    delay_ms=200.0 if i > 0 else None,
                    input_tokens=500 + i * 100,
                    subagent_spawn_ids=spawn_ids,
                )
            )

        # s0 = blocking (2 children), s1 = background (2 children)
        s0_children = ["conv_0_s0_c0", "conv_0_s0_c1"]
        s1_children = ["conv_0_s1_c0", "conv_0_s1_c1"]
        all_child_ids = s0_children + s1_children

        convs = [
            ConversationMetadata(
                conversation_id="conv_0",
                turns=parent_turns,
                subagent_spawns=[
                    SubagentSpawnInfo(
                        spawn_id="s0",
                        child_conversation_ids=s0_children,
                        join_turn_index=3,
                        is_background=False,
                    ),
                    SubagentSpawnInfo(
                        spawn_id="s1",
                        child_conversation_ids=s1_children,
                        join_turn_index=3,
                        is_background=True,
                    ),
                ],
            )
        ]
        for cid in all_child_ids:
            convs.append(
                ConversationMetadata(
                    conversation_id=cid,
                    turns=[TurnMetadata(input_tokens=300 + j * 50) for j in range(3)],
                    agent_depth=1,
                )
            )

        ds = DatasetMetadata(
            conversations=convs,
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = make_sampler(["conv_0"], ds.sampling_strategy)
        src = ConversationSource(ds, sampler)

        inner = MagicMock()
        inner.setup_phase = AsyncMock()
        inner.execute_phase = AsyncMock()
        inner.handle_credit_return = AsyncMock()
        del inner.on_child_session_started

        scheduler = MagicMock()
        scheduler.execute_async = MagicMock()
        issuer = MagicMock()
        issuer.issue_credit = AsyncMock(return_value=True)

        manager = SubagentSessionManager(
            inner=inner,
            conversation_source=src,
            credit_issuer=issuer,
            scheduler=scheduler,
        )

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0", "s1"])

        # 4 children fanned out
        assert scheduler.execute_async.call_count == 4
        # Only blocking children (s0's 2) tracked for join
        assert len(manager._subagent_child_to_parent) == 2
        pending = manager._pending_subagent_joins["parent-1"]
        assert pending.expected_count == 2

        # Complete both blocking children -> join dispatches
        scheduler.execute_async.reset_mock()
        child_corr_ids = list(manager._subagent_child_to_parent.keys())
        for i, child_corr_id in enumerate(child_corr_ids):
            child_credit = _make_credit(
                conv_id=s0_children[i],
                corr_id=child_corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            manager._handle_subagent_child_complete(child_credit)

        assert "parent-1" not in manager._pending_subagent_joins
        assert scheduler.execute_async.call_count >= 1


class TestSubagentSessionManagerBackground:
    """Test background spawn behavior."""

    @pytest.mark.asyncio
    async def test_background_spawn_returns_false(self):
        manager, inner, scheduler, _, child_ids = _make_manager(is_background=True)

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        result = manager._dispatch_subagent_spawns(credit, ["s0"])

        assert result is False
        assert "parent-1" not in manager._pending_subagent_joins
        # Only children dispatched (parent falls through to inner strategy)
        assert scheduler.execute_async.call_count == 2

    @pytest.mark.asyncio
    async def test_background_children_not_tracked(self):
        manager, inner, scheduler, _, child_ids = _make_manager(is_background=True)

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])

        # Background children are NOT registered for join accounting
        assert len(manager._subagent_child_to_parent) == 0
        assert "parent-1" not in manager._pending_subagent_joins


# =============================================================================
# Per-strategy guard tests
# =============================================================================


class TestUserCentricRateChildGuard:
    """UserCentricStrategy dispatches child non-final immediately, final is safe."""

    @pytest.mark.asyncio
    async def test_child_non_final_dispatches_immediately(self):
        from aiperf.timing.strategies.user_centric_rate import UserCentricStrategy

        scheduler = MagicMock()
        scheduler.execute_async = MagicMock()
        issuer = MagicMock()
        issuer.issue_credit = AsyncMock(return_value=True)

        ds = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="c1",
                    turns=[TurnMetadata() for _ in range(5)],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = make_sampler(["c1"], ds.sampling_strategy)
        src = ConversationSource(ds, sampler)

        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            num_users=5,
            request_rate=1.0,
        )
        strategy = UserCentricStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=scheduler,
            stop_checker=MagicMock(),
            credit_issuer=issuer,
            lifecycle=MagicMock(started_at_perf_ns=1_000_000_000),
        )

        child_credit = _make_credit(
            turn_index=1, num_turns=5, agent_depth=1, corr_id="child-1"
        )
        await strategy.handle_credit_return(child_credit)

        # Should dispatch immediately via scheduler
        scheduler.execute_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_child_final_is_safe_noop(self):
        from aiperf.timing.strategies.user_centric_rate import UserCentricStrategy

        scheduler = MagicMock()
        issuer = MagicMock()

        ds = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="c1",
                    turns=[TurnMetadata() for _ in range(5)],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = make_sampler(["c1"], ds.sampling_strategy)
        src = ConversationSource(ds, sampler)

        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            num_users=5,
            request_rate=1.0,
        )
        strategy = UserCentricStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=scheduler,
            stop_checker=MagicMock(),
            credit_issuer=issuer,
            lifecycle=MagicMock(started_at_perf_ns=1_000_000_000),
        )

        child_credit = _make_credit(
            turn_index=4, num_turns=5, agent_depth=1, corr_id="child-1"
        )
        # Should not raise (pop on unknown key is safe)
        await strategy.handle_credit_return(child_credit)


class TestAgenticLoadChildGuard:
    """AgenticLoadStrategy dispatches child non-final immediately, skips trajectory on final."""

    @pytest.mark.asyncio
    async def test_child_non_final_dispatches_immediately(self):
        from aiperf.timing.strategies.agentic_load import AgenticLoadStrategy

        issuer = MagicMock()
        issuer.issue_credit = AsyncMock(return_value=True)

        ds = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="c1",
                    turns=[TurnMetadata() for _ in range(5)],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = make_sampler(["c1"], ds.sampling_strategy)
        src = ConversationSource(ds, sampler)

        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENTIC_LOAD,
            num_users=2,
            expected_duration_sec=60.0,
        )
        strategy = AgenticLoadStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=issuer,
            lifecycle=MagicMock(started_at_perf_ns=1_000_000_000),
        )

        child_credit = _make_credit(
            turn_index=1, num_turns=5, agent_depth=1, corr_id="child-1"
        )
        await strategy.handle_credit_return(child_credit)

        issuer.issue_credit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_child_final_skips_trajectory_advance(self):
        from aiperf.timing.strategies.agentic_load import AgenticLoadStrategy

        issuer = MagicMock()
        issuer.issue_credit = AsyncMock(return_value=True)

        ds = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="c1",
                    turns=[TurnMetadata() for _ in range(5)],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = make_sampler(["c1"], ds.sampling_strategy)
        src = ConversationSource(ds, sampler)

        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENTIC_LOAD,
            num_users=2,
            expected_duration_sec=60.0,
        )
        strategy = AgenticLoadStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=issuer,
            lifecycle=MagicMock(started_at_perf_ns=1_000_000_000),
        )

        child_credit = _make_credit(
            turn_index=4, num_turns=5, agent_depth=1, corr_id="child-1"
        )
        await strategy.handle_credit_return(child_credit)

        # Should NOT issue any credit (no trajectory advance for child)
        issuer.issue_credit.assert_not_awaited()


class TestAdaptiveScaleChildGuard:
    """AdaptiveScaleStrategy cleans up child final without recycle/decrement."""

    @pytest.mark.asyncio
    async def test_child_final_does_cleanup_without_recycle(self):
        from aiperf.timing.strategies.adaptive_scale import AdaptiveScaleStrategy

        scheduler = MagicMock()
        scheduler.execute_async = MagicMock()
        issuer = MagicMock()
        issuer.issue_credit = AsyncMock(return_value=True)
        stop_checker = MagicMock()
        stop_checker.can_send_any_turn = MagicMock(return_value=True)

        ds = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="child_conv",
                    turns=[TurnMetadata() for _ in range(3)],
                    agent_depth=1,
                ),
                ConversationMetadata(
                    conversation_id="c1",
                    turns=[TurnMetadata() for _ in range(5)],
                ),
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = make_sampler(["c1"], ds.sampling_strategy)
        src = ConversationSource(ds, sampler)

        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.ADAPTIVE_SCALE,
            expected_duration_sec=120.0,
            start_users=2,
            max_users=10,
            recycle_sessions=True,
        )
        strategy = AdaptiveScaleStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=scheduler,
            stop_checker=stop_checker,
            credit_issuer=issuer,
            lifecycle=MagicMock(started_at_perf_ns=1_000_000_000),
        )

        initial_active = strategy._active_users

        child_credit = _make_credit(
            conv_id="child_conv",
            corr_id="child-1",
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
        await strategy.handle_credit_return(child_credit)

        # Should NOT recycle (child is not a user)
        assert scheduler.execute_async.call_count == 0
        # Should NOT decrement active users
        assert strategy._active_users == initial_active


# =============================================================================
# Gap coverage: child complete edge cases
# =============================================================================


class TestChildCompleteEdgeCases:
    """Edge cases in _handle_subagent_child_complete."""

    def test_join_suppressed_when_join_turn_index_gte_parent_num_turns(self) -> None:
        """Lines 191-192: join_turn_index >= parent_num_turns suppresses dispatch."""
        manager, inner, scheduler, _, child_ids = _make_manager()

        # Manually set up spawn so join_turn_index == parent_num_turns (out of range)
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])

        # Override the pending join to have join_turn_index >= parent_num_turns
        pending = manager._pending_subagent_joins["parent-1"]
        pending.join_turn_index = 6  # == parent_num_turns
        pending.parent_num_turns = 6

        scheduler.execute_async.reset_mock()
        child_corr_ids = list(manager._subagent_child_to_parent.keys())

        # Complete all children
        for i, child_corr_id in enumerate(child_corr_ids):
            child_credit = _make_credit(
                conv_id=child_ids[i],
                corr_id=child_corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            manager._handle_subagent_child_complete(child_credit)

        # Pending was removed but NO join dispatched
        assert "parent-1" not in manager._pending_subagent_joins
        scheduler.execute_async.assert_not_called()

    def test_double_completion_of_same_child_returns_early(self) -> None:
        """Lines 178-179: second completion of same child_corr_id finds parent_corr_id=None."""
        manager, inner, scheduler, _, child_ids = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])
        scheduler.execute_async.reset_mock()

        child_corr_ids = list(manager._subagent_child_to_parent.keys())
        first_child_corr = child_corr_ids[0]

        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=first_child_corr,
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )

        # First completion: pops from _subagent_child_to_parent, increments count
        manager._handle_subagent_child_complete(child_credit)
        pending = manager._pending_subagent_joins["parent-1"]
        assert pending.completed_count == 1

        # Second completion of same child: parent_corr_id is None, early return
        manager._handle_subagent_child_complete(child_credit)
        assert pending.completed_count == 1  # unchanged

    def test_pending_none_early_return(self) -> None:
        """Lines 181-182: child maps to parent but pending join was already removed."""
        manager, inner, scheduler, _, child_ids = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])
        scheduler.execute_async.reset_mock()

        child_corr_ids = list(manager._subagent_child_to_parent.keys())

        # Manually remove the pending join (simulates it was already consumed)
        manager._pending_subagent_joins.pop("parent-1")

        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
        # Should not raise, should early return
        manager._handle_subagent_child_complete(child_credit)
        scheduler.execute_async.assert_not_called()

    def test_partial_completion_does_not_dispatch_join(self) -> None:
        """Only some children complete -- join is NOT dispatched."""
        manager, inner, scheduler, _, child_ids = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])
        scheduler.execute_async.reset_mock()

        child_corr_ids = list(manager._subagent_child_to_parent.keys())
        assert len(child_corr_ids) == 2

        # Complete only the first child
        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
        manager._handle_subagent_child_complete(child_credit)

        # Pending still exists, join NOT dispatched
        assert "parent-1" in manager._pending_subagent_joins
        pending = manager._pending_subagent_joins["parent-1"]
        assert pending.completed_count == 1
        assert pending.expected_count == 2
        scheduler.execute_async.assert_not_called()


# =============================================================================
# Gap coverage: spawn dispatch edge cases
# =============================================================================


class TestSpawnDispatchEdgeCases:
    """Edge cases in _dispatch_subagent_spawns."""

    def test_all_spawn_ids_resolve_to_none_returns_false(self) -> None:
        """All spawn_ids not found -> returns False (parent not suspended)."""
        manager, inner, scheduler, issuer, _ = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )

        result = manager._dispatch_subagent_spawns(
            credit, ["nonexistent_s1", "nonexistent_s2"]
        )

        # No children found, parent not suspended
        assert result is False
        scheduler.execute_async.assert_not_called()
        assert "parent-1" not in manager._pending_subagent_joins

    def test_single_spawn_id_not_found_continues(self) -> None:
        """Lines 111-115: one spawn_id missing, others found -- missing skipped."""
        manager, inner, scheduler, _, child_ids = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )

        # "s0" exists in dataset, "bogus" does not
        manager._dispatch_subagent_spawns(credit, ["bogus", "s0"])

        # s0 has 2 children, bogus was skipped
        assert "parent-1" in manager._pending_subagent_joins
        pending = manager._pending_subagent_joins["parent-1"]
        assert pending.expected_count == 2
        # 2 children dispatched (from s0 only)
        assert scheduler.execute_async.call_count == 2


# =============================================================================
# Gap coverage: end-to-end through handle_credit_return
# =============================================================================


class TestBackgroundSpawnEndToEnd:
    """Background spawns tested end-to-end through handle_credit_return."""

    @pytest.mark.asyncio
    async def test_background_spawn_via_handle_credit_return(self) -> None:
        """Background spawn: children issued, parent falls through to inner strategy."""
        manager, inner, scheduler, _, child_ids = _make_manager(is_background=True)

        # Turn 2 -> next turn (3) has subagent_spawn_ids
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        await manager.handle_credit_return(credit)

        # Inner IS called (background spawns fall through to Path C)
        inner.handle_credit_return.assert_awaited_once()
        # 2 children dispatched
        assert scheduler.execute_async.call_count == 2
        # No pending join (background)
        assert "parent-1" not in manager._pending_subagent_joins
        # No children tracked for join accounting
        assert len(manager._subagent_child_to_parent) == 0


# =============================================================================
# Gap coverage: join turn content verification
# =============================================================================


class TestJoinTurnContentVerification:
    """Verify TurnToSend fields on dispatched join turns."""

    def test_join_turn_fields_after_all_children_complete(self) -> None:
        """Dispatched join TurnToSend has correct fields from PendingSubagentJoin."""
        manager, inner, scheduler, issuer, child_ids = _make_manager()

        credit = _make_credit(
            conv_id="conv_0",
            corr_id="parent-1",
            turn_index=2,
            num_turns=6,
            agent_depth=0,
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])
        scheduler.execute_async.reset_mock()
        issuer.issue_credit.reset_mock()

        child_corr_ids = list(manager._subagent_child_to_parent.keys())

        # Complete all children
        for i, child_corr_id in enumerate(child_corr_ids):
            child_credit = _make_credit(
                conv_id=child_ids[i],
                corr_id=child_corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            manager._handle_subagent_child_complete(child_credit)

        # Verify join was dispatched via scheduler
        assert scheduler.execute_async.call_count == 1

        # issue_credit was called with the join TurnToSend
        assert issuer.issue_credit.call_count == 1
        join_turn = issuer.issue_credit.call_args[0][0]

        assert isinstance(join_turn, TurnToSend)
        assert join_turn.conversation_id == "conv_0"
        assert join_turn.x_correlation_id == "parent-1"
        assert join_turn.turn_index == 3  # spawn_at(2) + 1
        assert join_turn.num_turns == 6
        assert join_turn.agent_depth == 0

    def test_background_spawn_only_dispatches_children(self) -> None:
        """Background spawn dispatches only children, no join turn."""
        manager, inner, scheduler, issuer, child_ids = _make_manager(is_background=True)

        credit = _make_credit(
            conv_id="conv_0",
            corr_id="parent-1",
            turn_index=2,
            num_turns=6,
            agent_depth=0,
        )
        result = manager._dispatch_subagent_spawns(credit, ["s0"])

        assert result is False
        # Only 2 child dispatches (no join — parent falls through to inner)
        assert scheduler.execute_async.call_count == 2
        assert issuer.issue_credit.call_count == 2
