# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SubagentOrchestrator: spawn/join, child dispatch, metrics, cleanup.

Tests the orchestrator component directly and via strategy integration.
Uses the composition API: intercept(), terminate_child(), cleanup(), get_stats().
"""

from unittest.mock import AsyncMock, MagicMock

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    SubagentSpawnInfo,
    TurnMetadata,
)
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.subagent_orchestrator import SubagentOrchestrator
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
    parent_turns = []
    for i in range(6):
        spawn_ids = ["s0"] if i == spawn_at else []
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


def _make_orchestrator(
    *,
    spawn_at: int = 2,
    num_children: int = 2,
    is_background: bool = False,
) -> tuple[SubagentOrchestrator, MagicMock, MagicMock, MagicMock, list[str]]:
    src, child_conv_ids = _make_dataset_and_source(
        spawn_at=spawn_at,
        num_children=num_children,
        is_background=is_background,
    )

    issuer = MagicMock()
    issuer.issue_credit = AsyncMock(return_value=True)

    stop_checker = MagicMock()
    stop_checker.can_send_any_turn = MagicMock(return_value=True)

    scheduler = MagicMock()
    scheduler.execute_async = MagicMock()
    dispatched: list[TurnToSend] = []

    orch = SubagentOrchestrator(
        conversation_source=src,
        credit_issuer=issuer,
        stop_checker=stop_checker,
        scheduler=scheduler,
        dispatch_fn=lambda turn: dispatched.append(turn),
    )
    orch._test_dispatched = dispatched  # type: ignore[attr-defined]
    return orch, issuer, stop_checker, scheduler, child_conv_ids


def _spawn_children(orch):
    """Spawn blocking children via intercept and return child corr_ids."""
    credit = _make_credit(
        conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
    )
    orch.intercept(credit)
    child_corr_ids = list(orch._child_to_parent.keys())
    return credit, child_corr_ids


# =============================================================================
# Spawn/join state machine
# =============================================================================


class TestSubagentSpawnAndJoin:
    """Core spawn -> child dispatch -> join lifecycle."""

    def test_intercept_non_spawn_turn_returns_false(self):
        orch, _, _, _, _ = _make_orchestrator()
        credit = _make_credit(turn_index=0, num_turns=6)
        assert orch.intercept(credit) is False

    def test_intercept_spawn_turn_returns_true(self):
        orch, _, _, _, _ = _make_orchestrator()
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        assert orch.intercept(credit) is True

    def test_spawn_creates_pending_join(self):
        orch, _, _, _, _ = _make_orchestrator()
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        orch.intercept(credit)

        assert "parent-1" in orch._pending_joins
        pending = orch._pending_joins["parent-1"]
        assert pending.expected_count == 2
        assert pending.join_turn_index == 3

    def test_spawn_dispatches_children_via_scheduler(self):
        orch, _, _, scheduler, _ = _make_orchestrator()
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        orch.intercept(credit)

        assert scheduler.execute_async.call_count == 2
        assert orch._stats.children_spawned == 2

    def test_child_non_final_dispatches_next_turn(self):
        orch, _, _, _, child_ids = _make_orchestrator()
        _, child_corr_ids = _spawn_children(orch)

        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=0,
            num_turns=3,
            agent_depth=1,
        )
        orch.intercept(child_credit)

        dispatched = orch._test_dispatched  # type: ignore[attr-defined]
        assert len(dispatched) == 1
        assert dispatched[0].conversation_id == child_ids[0]
        assert dispatched[0].turn_index == 1

    def test_child_final_completes_join(self):
        orch, _, _, _, child_ids = _make_orchestrator()
        _, child_corr_ids = _spawn_children(orch)

        # Complete all children
        for i, corr_id in enumerate(child_corr_ids):
            child_credit = _make_credit(
                conv_id=child_ids[i],
                corr_id=corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            orch.intercept(child_credit)

        assert orch._stats.children_completed == 2
        assert orch._stats.parents_resumed == 1
        assert "parent-1" not in orch._pending_joins

        dispatched = orch._test_dispatched  # type: ignore[attr-defined]
        join_turns = [t for t in dispatched if t.conversation_id == "conv_0"]
        assert len(join_turns) == 1
        assert join_turns[0].turn_index == 3

    def test_background_spawn_does_not_suspend_parent(self):
        orch, _, _, scheduler, child_ids = _make_orchestrator(is_background=True)
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        handled = orch.intercept(credit)

        assert handled is False
        assert "parent-1" not in orch._pending_joins
        assert len(orch._child_to_parent) == 0

        dispatched = orch._test_dispatched  # type: ignore[attr-defined]
        bg_dispatches = [d for d in dispatched if d.agent_depth == 1]
        assert len(bg_dispatches) == 2
        assert scheduler.execute_async.call_count == 0


# =============================================================================
# Metrics/stats collection
# =============================================================================


class TestSubagentStats:
    """Observability counters."""

    def test_get_stats_returns_all_counters(self):
        orch, _, _, _, _ = _make_orchestrator()
        stats = orch.get_stats()
        assert set(stats.keys()) == {
            "subagent_children_spawned",
            "subagent_children_completed",
            "subagent_children_errored",
            "subagent_parents_suspended",
            "subagent_parents_resumed",
            "subagent_joins_suppressed",
        }

    def test_spawn_increments_counters(self):
        orch, _, _, _, _ = _make_orchestrator()
        _spawn_children(orch)
        assert orch._stats.children_spawned == 2
        assert orch._stats.parents_suspended == 1

    def test_error_increments_errored(self):
        orch, _, _, _, child_ids = _make_orchestrator()
        _, child_corr_ids = _spawn_children(orch)

        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=0,
            num_turns=3,
            agent_depth=1,
        )
        orch.terminate_child(child_credit)
        assert orch._stats.children_errored == 1


# =============================================================================
# Cleanup
# =============================================================================


class TestSubagentCleanup:
    """Cleanup clears all tracking state."""

    def test_cleanup_clears_state(self):
        orch, _, _, _, _ = _make_orchestrator()
        _spawn_children(orch)

        orch.cleanup()

        assert len(orch._pending_joins) == 0
        assert len(orch._child_to_parent) == 0
        assert len(orch._terminated_children) == 0
        assert orch._cleaning_up is True

    def test_intercept_after_cleanup_returns_false(self):
        orch, _, _, _, _ = _make_orchestrator()
        orch.cleanup()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        assert orch.intercept(credit) is False


# =============================================================================
# Strategy integration: subagent hooks are one-liners
# =============================================================================


class TestStrategyIntegration:
    """Strategy uses intercept() in handle_credit_return."""

    def test_intercept_in_handle_credit_return_pattern(self):
        orch, _, _, _, child_ids = _make_orchestrator()
        _, child_corr_ids = _spawn_children(orch)

        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=0,
            num_turns=3,
            agent_depth=1,
        )

        # Strategy pattern: if intercept returns True, strategy returns early
        handled = orch.intercept(child_credit)
        assert handled is True

    def test_root_credit_passes_through(self):
        orch, _, _, _, _ = _make_orchestrator()
        credit = _make_credit(turn_index=1, num_turns=6)
        assert orch.intercept(credit) is False

    def test_terminate_child_dispatches_join_when_last(self):
        orch, _, _, _, child_ids = _make_orchestrator(num_children=1)
        _, child_corr_ids = _spawn_children(orch)

        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=0,
            num_turns=3,
            agent_depth=1,
        )
        orch.terminate_child(child_credit)

        assert orch._stats.children_errored == 1
        dispatched = orch._test_dispatched  # type: ignore[attr-defined]
        join_turns = [t for t in dispatched if t.conversation_id == "conv_0"]
        assert len(join_turns) == 1
        assert join_turns[0].turn_index == 3


# =============================================================================
# Turn-0 background pre-dispatch
# =============================================================================


class TestTurn0BackgroundSpawns:
    """dispatch_turn0_background_spawns pre-dispatches background children."""

    def test_dispatch_turn0_background_spawns(self):
        """Background children on turn 0 are pre-dispatched."""
        parent_turns = []
        for i in range(4):
            spawn_ids = ["s0"] if i == 0 else []
            parent_turns.append(
                TurnMetadata(
                    input_tokens=500,
                    subagent_spawn_ids=spawn_ids,
                )
            )

        child_conv_ids = ["conv_0_s0_c0", "conv_0_s0_c1"]
        spawn = SubagentSpawnInfo(
            spawn_id="s0",
            child_conversation_ids=child_conv_ids,
            join_turn_index=1,
            is_background=True,
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
                    turns=[TurnMetadata(input_tokens=300) for _ in range(2)],
                    agent_depth=1,
                )
            )

        ds = DatasetMetadata(
            conversations=convs,
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = make_sampler(["conv_0"], ds.sampling_strategy)
        src = ConversationSource(ds, sampler)

        dispatched: list[TurnToSend] = []
        orch = SubagentOrchestrator(
            conversation_source=src,
            credit_issuer=MagicMock(issue_credit=AsyncMock(return_value=True)),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
            scheduler=MagicMock(execute_async=MagicMock()),
            dispatch_fn=lambda turn: dispatched.append(turn),
        )

        orch.dispatch_turn0_background_spawns()

        assert len(dispatched) == 2
        assert all(d.agent_depth == 1 for d in dispatched)
        assert orch._stats.children_spawned == 2
