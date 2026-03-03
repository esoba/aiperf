# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SubagentSessionManager: spawn/join, centralized child dispatch, metrics, cleanup."""

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
    inner_has_dispatch_child_turn: bool = False,
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
    inner.on_request_complete = MagicMock()
    if inner_has_child_hook:
        inner.on_child_session_started = MagicMock()
    else:
        del inner.on_child_session_started
    if inner_has_dispatch_child_turn:
        inner.dispatch_child_turn = MagicMock()
    else:
        del inner.dispatch_child_turn

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

    def test_explicit_on_ttft_sample_delegates(self):
        manager, inner, _, _, _ = _make_manager()
        manager.on_ttft_sample(100)
        inner.on_ttft_sample.assert_called_once_with(100)

    def test_explicit_set_request_rate_delegates(self):
        manager, inner, _, _, _ = _make_manager()
        manager.set_request_rate(5.0)
        inner.set_request_rate.assert_called_once_with(5.0)

    def test_getattr_fallback_proxies_unknown(self):
        """Unknown attributes fall through to inner via __getattr__."""
        manager, inner, _, _, _ = _make_manager()
        inner.some_future_hook = MagicMock()
        manager.some_future_hook(42)
        inner.some_future_hook.assert_called_once_with(42)

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

    @pytest.mark.asyncio
    async def test_pending_join_created_before_children_dispatched(self):
        """Race condition fix: PendingSubagentJoin exists before child scheduling."""
        manager, inner, scheduler, _, _ = _make_manager()

        # Track when PendingSubagentJoin is present during execute_async calls
        join_existed = []

        def capture_execute_async(coro):
            join_existed.append("parent-1" in manager._pending_subagent_joins)

        scheduler.execute_async.side_effect = capture_execute_async

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])

        # All children should have seen the pending join already created
        assert all(join_existed), "PendingSubagentJoin must exist before child dispatch"


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


class TestSubagentSessionManagerChildDispatch:
    """Test Path A2: child non-final turn dispatch."""

    @pytest.mark.asyncio
    async def test_child_non_final_dispatches_via_scheduler(self):
        """Path A2: child non-final dispatched directly when no dispatch_child_turn hook."""
        manager, inner, scheduler, _, _ = _make_manager()

        child_credit = _make_credit(
            turn_index=1, num_turns=5, agent_depth=1, corr_id="child-1"
        )
        await manager.handle_credit_return(child_credit)

        scheduler.execute_async.assert_called_once()
        inner.handle_credit_return.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_child_non_final_uses_dispatch_hook_when_available(self):
        """Path A2: uses inner.dispatch_child_turn when available."""
        manager, inner, scheduler, _, _ = _make_manager(
            inner_has_dispatch_child_turn=True
        )

        child_credit = _make_credit(
            turn_index=1, num_turns=5, agent_depth=1, corr_id="child-1"
        )
        await manager.handle_credit_return(child_credit)

        inner.dispatch_child_turn.assert_called_once()
        scheduler.execute_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_child_final_background_delegates_to_inner(self):
        """Path A3: child final turn with unknown parent delegates to inner."""
        manager, inner, _, _, _ = _make_manager()

        child_credit = _make_credit(
            turn_index=4, num_turns=5, agent_depth=1, corr_id="bg-child-1"
        )
        await manager.handle_credit_return(child_credit)

        inner.handle_credit_return.assert_awaited_once_with(child_credit)


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
        del inner.dispatch_child_turn

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
    async def test_multi_spawn_different_join_turn_index_uses_max(self):
        """Multiple blocking spawns with different join_turn_index: uses max()."""
        parent_turns = []
        for i in range(8):
            spawn_ids = ["s0", "s1"] if i == 3 else []
            parent_turns.append(
                TurnMetadata(
                    delay_ms=200.0 if i > 0 else None,
                    input_tokens=500 + i * 100,
                    subagent_spawn_ids=spawn_ids,
                )
            )

        s0_children = ["conv_0_s0_c0"]
        s1_children = ["conv_0_s1_c0"]

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
                        join_turn_index=5,
                    ),
                ],
            )
        ]
        for cid in s0_children + s1_children:
            convs.append(
                ConversationMetadata(
                    conversation_id=cid,
                    turns=[TurnMetadata(input_tokens=300)],
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
        del inner.on_child_session_started
        del inner.dispatch_child_turn
        scheduler = MagicMock()
        issuer = MagicMock()
        issuer.issue_credit = AsyncMock(return_value=True)

        manager = SubagentSessionManager(
            inner=inner,
            conversation_source=src,
            credit_issuer=issuer,
            scheduler=scheduler,
        )

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=8
        )
        manager._dispatch_subagent_spawns(credit, ["s0", "s1"])

        pending = manager._pending_subagent_joins["parent-1"]
        assert pending.join_turn_index == 5  # max(3, 5)

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
        del inner.dispatch_child_turn

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


class TestSubagentSessionManagerChildFirstDispatch:
    """Test child first turn dispatch delegation."""

    def test_dispatch_child_first_turn_delegates_when_hook_present(self):
        """Pass 2: uses inner.dispatch_child_first_turn when protocol is implemented."""
        src, child_conv_ids = _make_dataset_and_source()

        inner = MagicMock()
        inner.setup_phase = AsyncMock()
        inner.execute_phase = AsyncMock()
        inner.handle_credit_return = AsyncMock()
        inner.dispatch_child_first_turn = MagicMock()
        del inner.on_child_session_started
        del inner.dispatch_child_turn

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
        manager._dispatch_subagent_spawns(credit, ["s0"])

        assert inner.dispatch_child_first_turn.call_count == 2
        scheduler.execute_async.assert_not_called()

        # Verify arguments: (child_session, child_depth, parent_correlation_id)
        for call in inner.dispatch_child_first_turn.call_args_list:
            session, depth, parent_corr = call[0]
            assert session.conversation_id in child_conv_ids
            assert depth == 1
            assert parent_corr == "parent-1"

    def test_dispatch_child_first_turn_falls_back_when_absent(self):
        """Pass 2: falls back to execute_async when no dispatch_child_first_turn hook."""
        manager, inner, scheduler, _, _ = _make_manager()
        assert not manager._inner_has_child_first_dispatch

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])

        assert scheduler.execute_async.call_count == 2


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
# Error/cancellation propagation tests
# =============================================================================


class TestErrorAndCancellationPropagation:
    """Test on_request_complete and on_cancelled_return error handling."""

    def test_on_request_complete_error_child_triggers_join_accounting(self):
        manager, inner, scheduler, _, child_ids = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])
        child_corr_ids = list(manager._subagent_child_to_parent.keys())

        # Simulate errored non-final child turn
        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=1,
            num_turns=3,
            agent_depth=1,
        )
        credit_return = MagicMock()
        credit_return.credit = child_credit
        credit_return.error = "server error"
        credit_return.cancelled = False

        manager.on_request_complete(credit_return)

        # Should count as child error + completion
        assert manager._stats.children_errored == 1
        pending = manager._pending_subagent_joins.get("parent-1")
        assert pending is not None
        assert pending.completed_count == 1

    def test_on_cancelled_return_child_triggers_join_accounting(self):
        manager, inner, scheduler, _, child_ids = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])
        child_corr_ids = list(manager._subagent_child_to_parent.keys())

        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=1,
            num_turns=3,
            agent_depth=1,
        )
        manager.on_cancelled_return(child_credit)

        assert manager._stats.children_errored == 1
        pending = manager._pending_subagent_joins.get("parent-1")
        assert pending is not None
        assert pending.completed_count == 1

    def test_on_request_complete_delegates_to_inner(self):
        manager, inner, _, _, _ = _make_manager()

        credit = _make_credit(turn_index=1, num_turns=5)
        credit_return = MagicMock()
        credit_return.credit = credit
        credit_return.error = None
        credit_return.cancelled = False

        manager.on_request_complete(credit_return)

        inner.on_request_complete.assert_called_once_with(credit_return)


# =============================================================================
# Metrics tests
# =============================================================================


class TestSubagentStats:
    """Test subagent counter tracking."""

    def test_spawn_increments_counters(self):
        manager, inner, scheduler, _, _ = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])

        stats = manager.get_subagent_stats()
        assert stats["subagent_children_spawned"] == 2
        assert stats["subagent_parents_suspended"] == 1

    def test_child_complete_increments_counters(self):
        manager, inner, scheduler, _, child_ids = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])

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

        stats = manager.get_subagent_stats()
        assert stats["subagent_parents_resumed"] == 1


# =============================================================================
# Cleanup tests
# =============================================================================


class TestCleanup:
    """Test phase-end cleanup."""

    def test_cleanup_clears_leaked_state(self):
        manager, inner, scheduler, _, _ = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])

        assert len(manager._pending_subagent_joins) > 0
        assert len(manager._subagent_child_to_parent) > 0

        manager.cleanup()

        assert len(manager._pending_subagent_joins) == 0
        assert len(manager._subagent_child_to_parent) == 0


# =============================================================================
# Gap coverage: child complete edge cases
# =============================================================================


class TestChildCompleteEdgeCases:
    """Edge cases in _handle_subagent_child_complete."""

    def test_join_suppressed_when_join_turn_index_gte_parent_num_turns(self) -> None:
        """join_turn_index >= parent_num_turns suppresses dispatch."""
        manager, inner, scheduler, _, child_ids = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])

        pending = manager._pending_subagent_joins["parent-1"]
        pending.join_turn_index = 6
        pending.parent_num_turns = 6

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
        scheduler.execute_async.assert_not_called()

    def test_double_completion_of_same_child_returns_early(self) -> None:
        """Second completion of same child_corr_id finds parent_corr_id=None."""
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

        manager._handle_subagent_child_complete(child_credit)
        pending = manager._pending_subagent_joins["parent-1"]
        assert pending.completed_count == 1

        manager._handle_subagent_child_complete(child_credit)
        assert pending.completed_count == 1

    def test_pending_none_early_return(self) -> None:
        """Child maps to parent but pending join was already removed."""
        manager, inner, scheduler, _, child_ids = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        manager._dispatch_subagent_spawns(credit, ["s0"])
        scheduler.execute_async.reset_mock()

        child_corr_ids = list(manager._subagent_child_to_parent.keys())
        manager._pending_subagent_joins.pop("parent-1")

        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
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

        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
        manager._handle_subagent_child_complete(child_credit)

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

        assert result is False
        scheduler.execute_async.assert_not_called()
        assert "parent-1" not in manager._pending_subagent_joins

    def test_single_spawn_id_not_found_continues(self) -> None:
        """One spawn_id missing, others found -- missing skipped."""
        manager, inner, scheduler, _, child_ids = _make_manager()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )

        manager._dispatch_subagent_spawns(credit, ["bogus", "s0"])

        assert "parent-1" in manager._pending_subagent_joins
        pending = manager._pending_subagent_joins["parent-1"]
        assert pending.expected_count == 2
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

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        await manager.handle_credit_return(credit)

        inner.handle_credit_return.assert_awaited_once()
        assert scheduler.execute_async.call_count == 2
        assert "parent-1" not in manager._pending_subagent_joins
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

        for i, child_corr_id in enumerate(child_corr_ids):
            child_credit = _make_credit(
                conv_id=child_ids[i],
                corr_id=child_corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            manager._handle_subagent_child_complete(child_credit)

        assert scheduler.execute_async.call_count == 1

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
        assert scheduler.execute_async.call_count == 2
        assert issuer.issue_credit.call_count == 2


# =============================================================================
# Strategy guard removal tests (child dispatch is now centralized)
# =============================================================================


class TestAgenticLoadNoChildGuard:
    """AgenticLoadStrategy no longer handles child credits -- SubagentSessionManager does."""

    @pytest.mark.asyncio
    async def test_child_credit_not_handled_by_strategy(self):
        """Strategy receives only root credits from SubagentSessionManager."""
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

        # Root credit with unknown session just returns (no error)
        root_credit = _make_credit(turn_index=1, num_turns=5, agent_depth=0)
        await strategy.handle_credit_return(root_credit)


class TestUserCentricDispatchChildTurn:
    """UserCentricStrategy.dispatch_child_turn paces via parent."""

    def test_dispatch_child_turn_with_parent_user(self):
        from aiperf.timing.strategies.user_centric_rate import UserCentricStrategy

        scheduler = MagicMock()
        scheduler.schedule_at_perf_sec = MagicMock()
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

        # Set up parent user mapping
        strategy.on_child_session_started("child-1", 1, "1")
        strategy._session_to_user["1"] = MagicMock(next_send_time=0.0)

        child_credit = _make_credit(
            turn_index=1, num_turns=5, agent_depth=1, corr_id="child-1"
        )
        turn = TurnToSend.from_previous_credit(child_credit)
        strategy.dispatch_child_turn(child_credit, turn)

        scheduler.schedule_at_perf_sec.assert_called_once()

    def test_dispatch_child_turn_without_parent_falls_back(self):
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
        turn = TurnToSend.from_previous_credit(child_credit)
        strategy.dispatch_child_turn(child_credit, turn)

        scheduler.execute_async.assert_called_once()


class TestAdaptiveScaleDefensiveDepth:
    """AdaptiveScaleStrategy defensive depth tracking."""

    @pytest.mark.asyncio
    async def test_child_credit_sets_session_depth_defensively(self):
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

        # No prior on_child_session_started call
        child_credit = _make_credit(
            conv_id="child_conv",
            corr_id="child-1",
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
        await strategy.handle_credit_return(child_credit)

        # Depth should have been set defensively
        # (session cleaned up on final turn, but it was set before cleanup)
        # Verify no crash occurred
        assert scheduler.execute_async.call_count == 0
