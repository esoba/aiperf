# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversary tests for SubagentOrchestrator flaw fixes.

Each test targets a specific attempt to break the terminated-children tracking,
cleanup, depth-aware gating, or state machine invariants.

Uses the composition API: intercept(), terminate_child(), cleanup(), get_stats().
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase, PrerequisiteKind
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    SubagentSpawnInfo,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.subagent_orchestrator import (
    PendingTurnGate,
    SubagentOrchestrator,
)
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
    child_conv_ids = [f"conv_0_s0_c{ci}" for ci in range(num_children)]
    spawn = SubagentSpawnInfo(
        spawn_id="s0",
        child_conversation_ids=child_conv_ids,
        is_background=is_background,
    )

    parent_turns = []
    for i in range(6):
        spawn_ids = ["s0"] if i == spawn_at else []
        prereqs = []
        if i == spawn_at + 1 and not is_background:
            prereqs = [
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, spawn_id="s0")
            ]
        parent_turns.append(
            TurnMetadata(
                delay_ms=200.0 if i > 0 else None,
                input_tokens=500 + i * 100,
                subagent_spawn_ids=spawn_ids,
                prerequisites=prereqs,
            )
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
    child_corr_ids = list(orch._child_to_gate.keys())
    return credit, child_corr_ids


# =============================================================================
# Double-event attacks: error + cancel on same child
# =============================================================================


class TestDoubleTerminationRace:
    """Try to corrupt state by erroring AND cancelling the same child."""

    def test_error_then_cancel_same_child_idempotent(self):
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
        assert child_credit.x_correlation_id in orch._terminated_children

        # Second terminate: child already popped from _child_to_gate
        orch.terminate_child(child_credit)

        assert orch._terminated_children == {child_credit.x_correlation_id}
        assert orch._stats.children_errored == 1

    def test_error_then_is_terminated_drains_once(self):
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
        assert orch._is_terminated(child_credit) is True
        assert orch._is_terminated(child_credit) is False


# =============================================================================
# Cleanup edge cases
# =============================================================================


class TestCleanupBoundary:
    """Attack cleanup with boundary conditions."""

    def test_cleanup_clears_all_state(self):
        orch, _, _, _, _ = _make_orchestrator()
        _spawn_children(orch)

        orch.cleanup()

        assert len(orch._gated_turns) == 0
        assert len(orch._child_to_gate) == 0
        assert len(orch._terminated_children) == 0

    def test_double_cleanup_is_safe(self):
        orch, _, _, _, _ = _make_orchestrator()
        _spawn_children(orch)

        orch.cleanup()
        orch.cleanup()

        assert len(orch._gated_turns) == 0
        assert len(orch._child_to_gate) == 0

    def test_cleanup_multiple_leaked_gates_all_abandoned(self):
        orch, _, _, _, _ = _make_orchestrator()

        for i in range(5):
            orch._gated_turns[f"p-{i}"] = PendingTurnGate(
                parent_conversation_id=f"conv_{i}",
                parent_correlation_id=f"p-{i}",
                gated_turn_index=2,
                parent_num_turns=6,
                parent_agent_depth=0,
                created_at_ns=0,
                outstanding={"spawn_join:s0": [3, i]},
            )

        orch.cleanup()
        assert len(orch._gated_turns) == 0

    def test_cleanup_with_created_at_ns_zero_does_not_crash(self):
        orch, _, _, _, _ = _make_orchestrator()

        orch._gated_turns["p-1"] = PendingTurnGate(
            parent_conversation_id="conv_0",
            parent_correlation_id="p-1",
            gated_turn_index=3,
            parent_num_turns=6,
            parent_agent_depth=0,
            created_at_ns=0,
            outstanding={"spawn_join:s0": [1, 0]},
        )

        orch.cleanup()


# =============================================================================
# Late arrivals after cleanup
# =============================================================================


class TestLateArrivalsAfterCleanup:
    """Children completing after cleanup has cleared all tracking state."""

    def test_intercept_child_after_cleanup_passes_through(self):
        orch, _, _, _, child_ids = _make_orchestrator()
        _, child_corr_ids = _spawn_children(orch)

        orch.cleanup()

        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
        # After cleanup, intercept returns False (cleaning_up flag set)
        handled = orch.intercept(child_credit)
        assert handled is False

    def test_terminate_child_after_cleanup_is_noop(self):
        orch, _, _, _, child_ids = _make_orchestrator()
        _, child_corr_ids = _spawn_children(orch)

        orch.cleanup()

        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=0,
            num_turns=3,
            agent_depth=1,
        )
        orch.terminate_child(child_credit)
        assert orch._stats.children_errored == 0

    def test_intercept_after_cleanup_returns_false(self):
        orch, _, _, _, _ = _make_orchestrator()
        orch.cleanup()

        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=3, num_turns=6
        )
        handled = orch.intercept(credit)
        assert handled is False


# =============================================================================
# Root credits should never trigger terminated tracking
# =============================================================================


class TestRootCreditTerminationGuard:
    """Root credits (depth=0) must never enter terminated tracking."""

    def test_terminate_root_credit_is_noop(self):
        orch, _, _, _, _ = _make_orchestrator()

        root_credit = _make_credit(
            conv_id="conv_0",
            corr_id="root-1",
            turn_index=1,
            num_turns=5,
            agent_depth=0,
        )
        orch.terminate_child(root_credit)
        assert "root-1" not in orch._terminated_children
        assert orch._stats.children_errored == 0


# =============================================================================
# Final turn guards
# =============================================================================


class TestFinalTurnTerminationGuard:
    """Error/cancel on a FINAL child turn must not terminate-track."""

    def test_terminate_final_child_is_noop(self):
        orch, _, _, _, child_ids = _make_orchestrator()
        _, child_corr_ids = _spawn_children(orch)

        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,  # final turn (num_turns=3)
            num_turns=3,
            agent_depth=1,
        )
        orch.terminate_child(child_credit)
        assert child_credit.x_correlation_id not in orch._terminated_children


# =============================================================================
# Untracked child: error on a child NOT in _child_to_gate
# =============================================================================


class TestUntrackedChildTermination:
    """Children not in tracking map (background or unknown) must not corrupt state."""

    def test_terminate_untracked_child_is_noop(self):
        orch, _, _, _, _ = _make_orchestrator()

        unknown_child = _make_credit(
            conv_id="unknown-conv",
            corr_id="unknown-corr",
            turn_index=0,
            num_turns=3,
            agent_depth=1,
        )
        orch.terminate_child(unknown_child)
        assert "unknown-corr" not in orch._terminated_children
        assert orch._stats.children_errored == 0


# =============================================================================
# Stop condition suppression
# =============================================================================


class TestStopConditionSuppression:
    """Gated turn dispatch suppressed when stop condition fires."""

    def test_gate_suppressed_when_stop_fired(self):
        orch, _, stop_checker, _, child_ids = _make_orchestrator()
        _, child_corr_ids = _spawn_children(orch)

        stop_checker.can_send_any_turn.return_value = False

        for i, child_corr_id in enumerate(child_corr_ids):
            child_credit = _make_credit(
                conv_id=child_ids[i],
                corr_id=child_corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            orch.intercept(child_credit)

        assert orch._stats.joins_suppressed == 1

    def test_terminate_all_children_with_stop_fired_suppresses_gate(self):
        orch, _, stop_checker, _, child_ids = _make_orchestrator()
        _, child_corr_ids = _spawn_children(orch)

        stop_checker.can_send_any_turn.return_value = False

        for i, child_corr_id in enumerate(child_corr_ids):
            child_credit = _make_credit(
                conv_id=child_ids[i],
                corr_id=child_corr_id,
                turn_index=0,
                num_turns=3,
                agent_depth=1,
            )
            orch.terminate_child(child_credit)

        assert orch._stats.joins_suppressed == 1


# =============================================================================
# Terminate-then-intercept races
# =============================================================================


class TestTerminateThenIntercept:
    """Race between terminate_child and subsequent intercept for the same child."""

    def test_terminate_then_intercept_child_final(self):
        """Error a child via terminate, then its final turn arrives. Gate accounting
        should count the child once (from terminate), not double-count."""
        orch, _, _, _, child_ids = _make_orchestrator()
        _, child_corr_ids = _spawn_children(orch)

        child_credit_nonfinal = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=0,
            num_turns=3,
            agent_depth=1,
        )

        # Terminate the child (non-final turn)
        orch.terminate_child(child_credit_nonfinal)
        assert orch._stats.children_errored == 1

        # Now the final turn arrives for same child via intercept
        child_credit_final = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
        orch.intercept(child_credit_final)

        # Child already popped from _child_to_gate by terminate,
        # so final intercept does not double-count
        parent_gates = list(orch._gated_turns.values())
        assert len(parent_gates) == 1
        assert (
            parent_gates[0].outstanding["spawn_join:s0"][1] == 1
        )  # only from terminate

    def test_terminate_then_non_final_intercept_suppressed(self):
        """After terminate, the next non-final intercept for that child
        should be suppressed (is_terminated consumed once)."""
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
        dispatched_before = len(orch._test_dispatched)  # type: ignore[attr-defined]

        # Non-final intercept: should be suppressed (is_terminated == True, consumed)
        child_credit_next = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=1,
            num_turns=3,
            agent_depth=1,
        )
        orch.intercept(child_credit_next)
        dispatched_after = len(orch._test_dispatched)  # type: ignore[attr-defined]

        # No new dispatch happened (is_terminated consumed the credit)
        assert dispatched_after == dispatched_before

        # Second non-final intercept: is_terminated already consumed, so dispatch fires
        child_credit_next2 = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=1,
            num_turns=3,
            agent_depth=1,
        )
        orch.intercept(child_credit_next2)
        dispatched_final = len(orch._test_dispatched)  # type: ignore[attr-defined]
        assert dispatched_final == dispatched_after + 1


# =============================================================================
# Stop fires mid-spawn
# =============================================================================


class TestStopMidSpawn:
    """Stop condition fires between spawn resolution and child completion."""

    def test_stop_fires_mid_spawn_resolution(self):
        """Stop fires after children dispatched. When children complete,
        gated turn suppressed because can_send_any_turn is False."""
        orch, _, stop_checker, _, child_ids = _make_orchestrator()
        _, child_corr_ids = _spawn_children(orch)

        dispatched_before = len(orch._test_dispatched)  # type: ignore[attr-defined]

        # Stop fires after children are dispatched
        stop_checker.can_send_any_turn.return_value = False

        # Children complete (final turns)
        for i, child_corr_id in enumerate(child_corr_ids):
            child_credit = _make_credit(
                conv_id=child_ids[i],
                corr_id=child_corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            orch.intercept(child_credit)

        # Gated turn should be suppressed (no new dispatch for parent)
        assert orch._stats.joins_suppressed == 1
        parent_dispatches = [
            t
            for t in orch._test_dispatched[dispatched_before:]  # type: ignore[attr-defined]
            if t.conversation_id == "conv_0"
        ]
        assert len(parent_dispatches) == 0


# =============================================================================
# Concurrent spawns on different parents
# =============================================================================


class TestConcurrentSpawnsOnDifferentParents:
    """Two parents spawn children simultaneously; each tracked independently."""

    def test_concurrent_spawns_on_different_parents(self):
        """Two parents each spawn children. Their gates tracked independently."""
        parent_ids = ["parent_A", "parent_B"]
        convs = []
        all_child_ids: dict[str, list[str]] = {}

        for pid in parent_ids:
            child_ids = [f"{pid}_s0_c0", f"{pid}_s0_c1"]
            all_child_ids[pid] = child_ids
            spawn = SubagentSpawnInfo(
                spawn_id="s0",
                child_conversation_ids=child_ids,
            )
            turns = []
            for i in range(5):
                spawn_ids = ["s0"] if i == 2 else []
                prereqs = []
                if i == 3:
                    prereqs = [
                        TurnPrerequisite(
                            kind=PrerequisiteKind.SPAWN_JOIN, spawn_id="s0"
                        )
                    ]
                turns.append(
                    TurnMetadata(
                        delay_ms=200.0 if i > 0 else None,
                        input_tokens=500,
                        subagent_spawn_ids=spawn_ids,
                        prerequisites=prereqs,
                    )
                )
            convs.append(
                ConversationMetadata(
                    conversation_id=pid,
                    turns=turns,
                    subagent_spawns=[spawn],
                )
            )
            for cid in child_ids:
                convs.append(
                    ConversationMetadata(
                        conversation_id=cid,
                        turns=[TurnMetadata(input_tokens=300) for _ in range(3)],
                        agent_depth=1,
                    )
                )

        ds = DatasetMetadata(
            conversations=convs,
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = make_sampler(parent_ids, ds.sampling_strategy)
        src = ConversationSource(ds, sampler)

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

        # Parent A spawns
        credit_a = _make_credit(
            conv_id="parent_A", corr_id="corr-A", turn_index=2, num_turns=5
        )
        assert orch.intercept(credit_a) is True

        # Parent B spawns
        credit_b = _make_credit(
            conv_id="parent_B", corr_id="corr-B", turn_index=2, num_turns=5
        )
        assert orch.intercept(credit_b) is True

        assert "corr-A" in orch._gated_turns
        assert "corr-B" in orch._gated_turns
        assert orch._gated_turns["corr-A"].outstanding == {"spawn_join:s0": [2, 0]}
        assert orch._gated_turns["corr-B"].outstanding == {"spawn_join:s0": [2, 0]}

        # Complete parent A's children
        a_child_corr_ids = [
            k for k, v in orch._child_to_gate.items() if v.parent_corr_id == "corr-A"
        ]
        for i, corr_id in enumerate(a_child_corr_ids):
            child_credit = _make_credit(
                conv_id=all_child_ids["parent_A"][i],
                corr_id=corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            orch.intercept(child_credit)

        # A's gate satisfied, B still pending
        assert "corr-A" not in orch._gated_turns
        assert "corr-B" in orch._gated_turns
        join_a = [t for t in dispatched if t.conversation_id == "parent_A"]
        assert len(join_a) == 1
        assert join_a[0].turn_index == 3

        # Complete parent B's children
        b_child_corr_ids = [
            k for k, v in orch._child_to_gate.items() if v.parent_corr_id == "corr-B"
        ]
        for i, corr_id in enumerate(b_child_corr_ids):
            child_credit = _make_credit(
                conv_id=all_child_ids["parent_B"][i],
                corr_id=corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            orch.intercept(child_credit)

        assert "corr-B" not in orch._gated_turns
        join_b = [t for t in dispatched if t.conversation_id == "parent_B"]
        assert len(join_b) == 1


# =============================================================================
# Background child error isolation
# =============================================================================


class TestBackgroundChildErrorIsolation:
    """Background child errors must not corrupt gate tracking."""

    def test_background_child_error_does_not_affect_gate(self):
        """A background child that errors should not touch gated turns
        or child_to_gate mappings."""
        orch, _, _, _, _ = _make_orchestrator()
        _spawn_children(orch)

        # Create a background child credit (not tracked)
        bg_credit = _make_credit(
            conv_id="bg-child-conv",
            corr_id="bg-corr-1",
            turn_index=0,
            num_turns=3,
            agent_depth=1,
        )

        gates_before = dict(orch._gated_turns)
        c2g_before = dict(orch._child_to_gate)

        # Terminate the background child -- it's not in _child_to_gate
        orch.terminate_child(bg_credit)

        # No state change
        assert orch._stats.children_errored == 0
        assert dict(orch._gated_turns) == gates_before
        assert dict(orch._child_to_gate) == c2g_before


# =============================================================================
# All blocking children fail to issue credit
# =============================================================================


class TestAllBlockingChildrenFailToIssue:
    """If ALL blocking children fail credit issuance, parent gate completes
    with all errored."""

    @pytest.mark.asyncio
    async def test_issue_credit_failure_releases_all_blocking_children(self):
        orch, issuer, _, _, child_ids = _make_orchestrator()
        _, child_corr_ids = _spawn_children(orch)

        issuer.issue_credit.return_value = False

        # Simulate failed issuance for each blocking child
        for i, child_corr_id in enumerate(child_corr_ids):
            turn = TurnToSend(
                conversation_id=child_ids[i],
                x_correlation_id=child_corr_id,
                turn_index=0,
                num_turns=3,
            )
            await orch._issue_child_credit_or_release(turn, child_corr_id)

        # All children errored
        assert orch._stats.children_errored == 2

        # Parent gate should have completed (all children released)
        assert len(orch._gated_turns) == 0

        # Parents resumed counter incremented
        assert orch._stats.parents_resumed == 1

        # Gated turn dispatched via dispatch_fn
        dispatched = orch._test_dispatched  # type: ignore[attr-defined]
        join_turns = [t for t in dispatched if t.conversation_id == "conv_0"]
        assert len(join_turns) == 1
        assert join_turns[0].turn_index == 3  # spawn_at(2) + 1

    @pytest.mark.asyncio
    async def test_issue_credit_exception_releases_child_from_gate(self):
        """If issue_credit raises, the child is released from gate tracking
        instead of leaking a pending gate forever."""
        orch, issuer, _, _, child_ids = _make_orchestrator(num_children=1)
        _, child_corr_ids = _spawn_children(orch)

        issuer.issue_credit = AsyncMock(side_effect=RuntimeError("connection lost"))

        turn = TurnToSend(
            conversation_id=child_ids[0],
            x_correlation_id=child_corr_ids[0],
            turn_index=0,
            num_turns=3,
        )
        await orch._issue_child_credit_or_release(turn, child_corr_ids[0])

        assert orch._stats.children_errored == 1
        assert len(orch._gated_turns) == 0
        assert orch._stats.parents_resumed == 1

        dispatched = orch._test_dispatched  # type: ignore[attr-defined]
        join_turns = [t for t in dispatched if t.conversation_id == "conv_0"]
        assert len(join_turns) == 1
        assert join_turns[0].turn_index == 3
