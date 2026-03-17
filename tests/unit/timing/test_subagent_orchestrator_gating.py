# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for prerequisite-based gating in SubagentOrchestrator.

Focuses on:
- PendingTurnGate.is_satisfied property semantics
- ChildGateEntry field tracking
- Multi-prerequisite gating (multiple spawns gate the same turn)
- _satisfy_prerequisite on future (not yet blocked) gates
- _find_gated_turn_index with multiple spawn IDs
- set_dispatch late-binding
- Gate pointing past conversation end
- _maybe_suspend_parent when future gate already satisfied
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

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
    ChildGateEntry,
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


def _make_multi_spawn_source(
    *,
    spawn_at_s0: int = 1,
    spawn_at_s1: int = 2,
    join_at: int = 3,
) -> tuple[ConversationSource, list[str], list[str]]:
    """Create a dataset with two blocking spawns that gate the same turn."""
    s0_children = ["conv_0_s0_c0"]
    s1_children = ["conv_0_s1_c0", "conv_0_s1_c1"]
    spawn_s0 = SubagentSpawnInfo(spawn_id="s0", child_conversation_ids=s0_children)
    spawn_s1 = SubagentSpawnInfo(spawn_id="s1", child_conversation_ids=s1_children)

    parent_turns = []
    for i in range(6):
        spawn_ids = []
        if i == spawn_at_s0:
            spawn_ids.append("s0")
        if i == spawn_at_s1:
            spawn_ids.append("s1")
        prereqs = []
        if i == join_at:
            prereqs = [
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, spawn_id="s0"),
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, spawn_id="s1"),
            ]
        parent_turns.append(
            TurnMetadata(
                input_tokens=500 + i * 100,
                subagent_spawn_ids=spawn_ids,
                prerequisites=prereqs,
            )
        )

    convs = [
        ConversationMetadata(
            conversation_id="conv_0",
            turns=parent_turns,
            subagent_spawns=[spawn_s0, spawn_s1],
        )
    ]
    for cid in s0_children + s1_children:
        convs.append(
            ConversationMetadata(
                conversation_id=cid,
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
    return src, s0_children, s1_children


def _make_simple_orchestrator(
    *,
    spawn_at: int = 2,
    join_at: int | None = None,
    num_children: int = 2,
    is_background: bool = False,
    num_parent_turns: int = 6,
) -> tuple[SubagentOrchestrator, MagicMock, MagicMock, MagicMock, list[str]]:
    join_at = spawn_at + 1 if join_at is None else join_at
    child_conv_ids = [f"conv_0_s0_c{ci}" for ci in range(num_children)]
    spawn = SubagentSpawnInfo(
        spawn_id="s0",
        child_conversation_ids=child_conv_ids,
        is_background=is_background,
    )

    parent_turns = []
    for i in range(num_parent_turns):
        spawn_ids = ["s0"] if i == spawn_at else []
        prereqs = []
        if i == join_at and not is_background:
            prereqs = [
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, spawn_id="s0")
            ]
        parent_turns.append(
            TurnMetadata(
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
                turns=[TurnMetadata(input_tokens=300) for _ in range(3)],
                agent_depth=1,
            )
        )

    ds = DatasetMetadata(
        conversations=convs,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    sampler = make_sampler(["conv_0"], ds.sampling_strategy)
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
    orch._test_dispatched = dispatched  # type: ignore[attr-defined]
    return orch, issuer, stop_checker, scheduler, child_conv_ids


# =============================================================================
# PendingTurnGate.is_satisfied
# =============================================================================


class TestPendingTurnGateIsSatisfied:
    """Verify is_satisfied semantics for various outstanding states."""

    @pytest.mark.parametrize(
        "outstanding,expected",
        [
            ({}, True),
            ({"spawn_join:s0": [2, 2]}, True),
            ({"spawn_join:s0": [2, 1]}, False),
            ({"spawn_join:s0": [2, 0]}, False),
            param(
                {"spawn_join:s0": [1, 1], "spawn_join:s1": [2, 2]},
                True,
                id="multi-prereq-all-satisfied",
            ),
            param(
                {"spawn_join:s0": [1, 1], "spawn_join:s1": [2, 1]},
                False,
                id="multi-prereq-one-unsatisfied",
            ),
            param(
                {"spawn_join:s0": [1, 5]},
                True,
                id="completed-exceeds-expected",
            ),
        ],
    )  # fmt: skip
    def test_is_satisfied_returns_expected(
        self, outstanding: dict[str, list[int]], expected: bool
    ) -> None:
        gate = PendingTurnGate(
            parent_conversation_id="conv_0",
            parent_correlation_id="p-1",
            gated_turn_index=3,
            outstanding=outstanding,
        )
        assert gate.is_satisfied is expected


# =============================================================================
# ChildGateEntry field tracking
# =============================================================================


class TestChildGateEntryTracking:
    """Verify ChildGateEntry fields are populated correctly after spawn."""

    def test_child_gate_entry_fields_after_spawn(self) -> None:
        orch, _, _, _, child_ids = _make_simple_orchestrator()
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        orch.intercept(credit)

        for _corr_id, entry in orch._child_to_gate.items():
            assert isinstance(entry, ChildGateEntry)
            assert entry.parent_corr_id == "parent-1"
            assert entry.gated_turn_index == 3
            assert entry.prereq_key == "spawn_join:s0"


# =============================================================================
# Multi-prerequisite gating
# =============================================================================


class TestMultiPrerequisiteGating:
    """Two spawns gate the same turn; both must complete."""

    def test_multi_spawn_both_must_complete(self) -> None:
        src, s0_children, s1_children = _make_multi_spawn_source(
            spawn_at_s0=1, spawn_at_s1=2, join_at=3
        )
        dispatched: list[TurnToSend] = []
        scheduler = MagicMock()
        scheduler.execute_async = MagicMock()

        orch = SubagentOrchestrator(
            conversation_source=src,
            credit_issuer=MagicMock(issue_credit=AsyncMock(return_value=True)),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
            scheduler=scheduler,
            dispatch_fn=lambda turn: dispatched.append(turn),
        )

        # Spawn s0 on turn 1
        credit_t1 = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=1, num_turns=6
        )
        orch.intercept(credit_t1)

        # Spawn s1 on turn 2
        credit_t2 = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        result = orch.intercept(credit_t2)
        assert result is True  # parent suspended at gated turn 3

        gate = orch._gated_turns["parent-1"]
        assert "spawn_join:s0" in gate.outstanding
        assert "spawn_join:s1" in gate.outstanding

        # Complete s0 child (1 child)
        s0_child_corr_ids = [
            k for k, v in orch._child_to_gate.items() if v.prereq_key == "spawn_join:s0"
        ]
        for corr_id in s0_child_corr_ids:
            child_credit = _make_credit(
                conv_id=s0_children[0],
                corr_id=corr_id,
                turn_index=1,
                num_turns=2,
                agent_depth=1,
            )
            orch.intercept(child_credit)

        # s0 done but s1 not done -- gate still active
        assert "parent-1" in orch._gated_turns
        join_turns = [t for t in dispatched if t.conversation_id == "conv_0"]
        assert len(join_turns) == 0

        # Complete s1 children (2 children)
        s1_child_corr_ids = [
            k for k, v in orch._child_to_gate.items() if v.prereq_key == "spawn_join:s1"
        ]
        for i, corr_id in enumerate(s1_child_corr_ids):
            child_credit = _make_credit(
                conv_id=s1_children[i],
                corr_id=corr_id,
                turn_index=1,
                num_turns=2,
                agent_depth=1,
            )
            orch.intercept(child_credit)

        # Now both prerequisites satisfied
        assert "parent-1" not in orch._gated_turns
        join_turns = [t for t in dispatched if t.conversation_id == "conv_0"]
        assert len(join_turns) == 1
        assert join_turns[0].turn_index == 3


# =============================================================================
# Satisfy prerequisite on future (unblocked) gate
# =============================================================================


class TestSatisfyPrerequisiteOnFutureGate:
    """When children complete before parent reaches gated turn, the
    future gate is cleaned up silently (no dispatch)."""

    def test_children_complete_before_parent_blocks_cleans_future_gate(self) -> None:
        orch, _, _, _, child_ids = _make_simple_orchestrator(spawn_at=1, join_at=5)
        # Spawn on turn 1 (join at turn 5, so future gate created)
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=1, num_turns=6
        )
        orch.intercept(credit)

        assert "parent-1" in orch._future_gates
        child_corr_ids = list(orch._child_to_gate.keys())

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

        # Future gate cleaned up
        assert "parent-1" not in orch._future_gates
        # No gated turn dispatched (parent not blocked yet)
        dispatched = orch._test_dispatched  # type: ignore[attr-defined]
        join_turns = [t for t in dispatched if t.conversation_id == "conv_0"]
        assert len(join_turns) == 0

    def test_parent_not_suspended_when_future_gate_already_satisfied(self) -> None:
        """If children complete before parent reaches gate, parent proceeds."""
        orch, _, _, _, child_ids = _make_simple_orchestrator(spawn_at=1, join_at=3)
        credit_t1 = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=1, num_turns=6
        )
        orch.intercept(credit_t1)

        child_corr_ids = list(orch._child_to_gate.keys())

        # Complete all children before parent reaches turn 2
        for i, corr_id in enumerate(child_corr_ids):
            child_credit = _make_credit(
                conv_id=child_ids[i],
                corr_id=corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            orch.intercept(child_credit)

        # Parent turn 2 completes (next turn is 3, the gated turn)
        credit_t2 = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        result = orch.intercept(credit_t2)

        # Parent should NOT be suspended -- prerequisites already met
        assert result is False
        assert orch._stats.parents_suspended == 0


# =============================================================================
# _find_gated_turn_index with multiple spawn IDs
# =============================================================================


class TestFindGatedTurnIndex:
    """_find_gated_turn_index returns first match from spawn_ids list."""

    def test_find_gated_turn_index_returns_first_match(self) -> None:
        src, _, _ = _make_multi_spawn_source(spawn_at_s0=1, spawn_at_s1=2, join_at=3)
        orch = SubagentOrchestrator(
            conversation_source=src,
            credit_issuer=MagicMock(issue_credit=AsyncMock(return_value=True)),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
            scheduler=MagicMock(execute_async=MagicMock()),
            dispatch_fn=lambda t: None,
        )

        # Both s0 and s1 point to turn 3
        assert orch._find_gated_turn_index("conv_0", ["s0"]) == 3
        assert orch._find_gated_turn_index("conv_0", ["s1"]) == 3
        assert orch._find_gated_turn_index("conv_0", ["s0", "s1"]) == 3

    def test_find_gated_turn_index_unknown_spawn_returns_none(self) -> None:
        src, _, _ = _make_multi_spawn_source()
        orch = SubagentOrchestrator(
            conversation_source=src,
            credit_issuer=MagicMock(issue_credit=AsyncMock(return_value=True)),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
            scheduler=MagicMock(execute_async=MagicMock()),
            dispatch_fn=lambda t: None,
        )
        assert orch._find_gated_turn_index("conv_0", ["nonexistent"]) is None
        assert orch._find_gated_turn_index("nonexistent_conv", ["s0"]) is None


# =============================================================================
# set_dispatch late-binding
# =============================================================================


class TestSetDispatch:
    """set_dispatch allows late-binding the dispatch callback."""

    def test_set_dispatch_replaces_callback(self) -> None:
        orch, _, _, _, child_ids = _make_simple_orchestrator(num_children=1)
        new_dispatched: list[TurnToSend] = []
        orch.set_dispatch(lambda turn: new_dispatched.append(turn))

        # Spawn and complete child to trigger dispatch of gated turn
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        orch.intercept(credit)

        child_corr_ids = list(orch._child_to_gate.keys())
        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
        orch.intercept(child_credit)

        # Gated turn dispatched via the NEW callback
        join_turns = [t for t in new_dispatched if t.conversation_id == "conv_0"]
        assert len(join_turns) == 1
        assert join_turns[0].turn_index == 3


# =============================================================================
# Gate pointing past conversation end
# =============================================================================


class TestGatePastConversationEnd:
    """Gate turn_index >= num_turns means gated turn is unreachable."""

    def test_release_blocked_gate_past_end_returns_none(self) -> None:
        """When gated_turn_index >= num_turns, _release_blocked_gate pops the gate but dispatches nothing."""
        orch, _, _, _, child_ids = _make_simple_orchestrator(
            num_children=1, num_parent_turns=4, spawn_at=2, join_at=3
        )

        # Spawn on turn 2 → creates gate at turn 3
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=4
        )
        orch.intercept(credit)

        # Mutate the gate to point past the end before any child completes
        gate = orch._gated_turns.get("parent-1")
        assert gate is not None
        gate.gated_turn_index = 10

        # Directly call _release_blocked_gate (the unit under test)
        result = orch._release_blocked_gate("parent-1")
        assert result is None
        assert "parent-1" not in orch._gated_turns


# =============================================================================
# Prerequisite index built correctly
# =============================================================================


class TestPrerequisiteIndexBuilding:
    """Verify the prerequisite index and spawn_join index are built at init."""

    def test_prerequisite_index_populated(self) -> None:
        orch, _, _, _, _ = _make_simple_orchestrator(spawn_at=2, join_at=3)

        # Turn 3 has a spawn_join prerequisite for s0
        assert ("conv_0", 3) in orch._prerequisite_index
        prereqs = orch._prerequisite_index[("conv_0", 3)]
        assert len(prereqs) == 1
        assert prereqs[0].kind == PrerequisiteKind.SPAWN_JOIN
        assert prereqs[0].spawn_id == "s0"

    def test_spawn_join_index_populated(self) -> None:
        orch, _, _, _, _ = _make_simple_orchestrator(spawn_at=2, join_at=3)

        assert ("conv_0", "s0") in orch._spawn_join_index
        assert orch._spawn_join_index[("conv_0", "s0")] == 3

    def test_multi_spawn_prerequisite_index(self) -> None:
        src, _, _ = _make_multi_spawn_source(spawn_at_s0=1, spawn_at_s1=2, join_at=3)
        orch = SubagentOrchestrator(
            conversation_source=src,
            credit_issuer=MagicMock(issue_credit=AsyncMock(return_value=True)),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
            scheduler=MagicMock(execute_async=MagicMock()),
            dispatch_fn=lambda t: None,
        )

        prereqs = orch._prerequisite_index[("conv_0", 3)]
        assert len(prereqs) == 2
        assert orch._spawn_join_index[("conv_0", "s0")] == 3
        assert orch._spawn_join_index[("conv_0", "s1")] == 3


# =============================================================================
# _get_gate: active vs future lookup
# =============================================================================


class TestGetGateLookup:
    """_get_gate finds active blocked gates and future gates."""

    def test_get_gate_returns_active_gate(self) -> None:
        orch, _, _, _, _ = _make_simple_orchestrator()
        gate = PendingTurnGate(
            parent_conversation_id="conv_0",
            parent_correlation_id="p-1",
            gated_turn_index=3,
            is_blocked=True,
            outstanding={"spawn_join:s0": [2, 0]},
        )
        orch._gated_turns["p-1"] = gate

        result = orch._get_gate("p-1", 3)
        assert result is gate

    def test_get_gate_returns_future_gate(self) -> None:
        orch, _, _, _, _ = _make_simple_orchestrator()
        gate = PendingTurnGate(
            parent_conversation_id="conv_0",
            parent_correlation_id="p-1",
            gated_turn_index=5,
            outstanding={"spawn_join:s0": [2, 0]},
        )
        orch._future_gates["p-1"] = {5: gate}

        result = orch._get_gate("p-1", 5)
        assert result is gate

    def test_get_gate_wrong_turn_index_returns_none(self) -> None:
        orch, _, _, _, _ = _make_simple_orchestrator()
        gate = PendingTurnGate(
            parent_conversation_id="conv_0",
            parent_correlation_id="p-1",
            gated_turn_index=3,
            is_blocked=True,
            outstanding={"spawn_join:s0": [2, 0]},
        )
        orch._gated_turns["p-1"] = gate

        assert orch._get_gate("p-1", 999) is None

    def test_get_gate_unknown_parent_returns_none(self) -> None:
        orch, _, _, _, _ = _make_simple_orchestrator()
        assert orch._get_gate("nonexistent", 3) is None


# =============================================================================
# _iter_future_gates
# =============================================================================


class TestIterFutureGates:
    """_iter_future_gates flattens nested dict for cleanup."""

    def test_iter_future_gates_flattens_correctly(self) -> None:
        orch, _, _, _, _ = _make_simple_orchestrator()
        gate_a = PendingTurnGate(
            parent_conversation_id="conv_0",
            parent_correlation_id="p-1",
            gated_turn_index=3,
        )
        gate_b = PendingTurnGate(
            parent_conversation_id="conv_0",
            parent_correlation_id="p-1",
            gated_turn_index=5,
        )
        gate_c = PendingTurnGate(
            parent_conversation_id="conv_1",
            parent_correlation_id="p-2",
            gated_turn_index=4,
        )
        orch._future_gates = {
            "p-1": {3: gate_a, 5: gate_b},
            "p-2": {4: gate_c},
        }

        result = orch._iter_future_gates()
        assert len(result) == 3
        # All gates present
        gates_found = {(corr_id, gate.gated_turn_index) for corr_id, gate in result}
        assert gates_found == {("p-1", 3), ("p-1", 5), ("p-2", 4)}

    def test_iter_future_gates_empty_returns_empty(self) -> None:
        orch, _, _, _, _ = _make_simple_orchestrator()
        assert orch._iter_future_gates() == []


# =============================================================================
# TurnPrerequisite model validation
# =============================================================================


class TestTurnPrerequisiteModel:
    """TurnPrerequisite model field behavior."""

    def test_spawn_join_prerequisite_has_spawn_id(self) -> None:
        prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, spawn_id="s0")
        assert prereq.kind == PrerequisiteKind.SPAWN_JOIN
        assert prereq.spawn_id == "s0"

    def test_spawn_join_prerequisite_spawn_id_defaults_to_none(self) -> None:
        prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN)
        assert prereq.spawn_id is None


# =============================================================================
# Stop condition narrowed to StopConditionChecker
# =============================================================================


class TestStopCheckerType:
    """Stop checker uses can_send_any_turn method."""

    def test_stop_checker_can_send_any_turn_gates_dispatch(self) -> None:
        orch, _, stop_checker, _, child_ids = _make_simple_orchestrator(num_children=1)
        credit = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        orch.intercept(credit)

        stop_checker.can_send_any_turn.return_value = False

        child_corr_ids = list(orch._child_to_gate.keys())
        child_credit = _make_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
        orch.intercept(child_credit)

        # Gated turn suppressed
        dispatched = orch._test_dispatched  # type: ignore[attr-defined]
        join_turns = [t for t in dispatched if t.conversation_id == "conv_0"]
        assert len(join_turns) == 0
        assert orch._stats.joins_suppressed == 1
        assert stop_checker.can_send_any_turn.call_count == 1
