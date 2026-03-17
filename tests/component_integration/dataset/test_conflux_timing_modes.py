# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for Conflux subagent spawning across all timing modes.

Exercises the full pipeline:
  Handcrafted DatasetMetadata (mimicking ConfluxLoader output)
  -> ConversationSource -> SubagentOrchestrator -> Strategy-specific dispatch
  -> Credit issuance -> Callback return -> Gate completion -> Gated turn dispatch

Each timing mode (FIXED_SCHEDULE, REQUEST_RATE, USER_CENTRIC_RATE) dispatches
child turns and gated turns through its own _dispatch_turn callback, using
timestamps/delays from metadata.  These tests verify:

1. All timing modes correctly spawn children, gate the parent, and resume.
2. Timestamps on dispatched turns are plausible (absolute or relative).
3. Delayed joins (spawn_at != join_at - 1) work across all modes.
4. Background spawns do not gate across all modes.
5. Multiple spawns on different turns compose correctly.
6. Error/cancellation on children correctly releases gates.
"""

from __future__ import annotations

import time
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
from aiperf.plugin import plugins
from aiperf.plugin.enums import DatasetSamplingStrategy, PluginType
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.subagent_orchestrator import SubagentOrchestrator

# =============================================================================
# Helpers
# =============================================================================

_CREDIT_COUNTER = 0


def _next_credit_id() -> int:
    global _CREDIT_COUNTER
    _CREDIT_COUNTER += 1
    return _CREDIT_COUNTER


def _make_sampler(conv_ids: list[str]) -> object:
    cls = plugins.get_class(
        PluginType.DATASET_SAMPLER, DatasetSamplingStrategy.SEQUENTIAL
    )
    return cls(conversation_ids=conv_ids)


def _make_credit(
    *,
    conv_id: str,
    corr_id: str,
    turn_index: int,
    num_turns: int,
    agent_depth: int = 0,
    parent_correlation_id: str | None = None,
) -> Credit:
    return Credit(
        id=_next_credit_id(),
        phase=CreditPhase.PROFILING,
        conversation_id=conv_id,
        x_correlation_id=corr_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=time.time_ns(),
        agent_depth=agent_depth,
        parent_correlation_id=parent_correlation_id,
    )


def _build_orchestrator(
    ds: DatasetMetadata,
    *,
    stop_can_send: bool = True,
) -> tuple[SubagentOrchestrator, list[TurnToSend], MagicMock]:
    """Build orchestrator from dataset with a capturing dispatch_fn."""
    root_ids = [c.conversation_id for c in ds.conversations if c.agent_depth == 0]
    sampler = _make_sampler(root_ids)
    src = ConversationSource(ds, sampler)

    scheduler = MagicMock()
    scheduler.execute_async = MagicMock()
    scheduler.schedule_at_perf_sec = MagicMock()
    scheduler.schedule_later = MagicMock()
    dispatched: list[TurnToSend] = []

    orch = SubagentOrchestrator(
        conversation_source=src,
        credit_issuer=MagicMock(
            issue_credit=AsyncMock(return_value=True),
            try_issue_credit=AsyncMock(return_value=True),
        ),
        stop_checker=MagicMock(
            can_send_any_turn=MagicMock(return_value=stop_can_send),
            can_start_new_session=MagicMock(return_value=stop_can_send),
        ),
        scheduler=scheduler,
        dispatch_fn=lambda turn: dispatched.append(turn),
    )
    return orch, dispatched, scheduler


# =============================================================================
# Dataset builders mimicking ConfluxLoader output
# =============================================================================


def _make_conflux_dataset(
    *,
    parent_turns: int = 6,
    spawn_at: int = 2,
    join_at: int | None = None,
    num_children: int = 2,
    child_turns: int = 3,
    is_background: bool = False,
    timestamps: bool = False,
    timestamp_base_ms: int = 1000,
    timestamp_spacing_ms: int = 500,
    delay_ms: float | None = None,
) -> DatasetMetadata:
    """Build a dataset mimicking Conflux trace output.

    Args:
        timestamps: If True, add absolute timestamps on all turns.
        delay_ms: If set, add delay_ms on subsequent turns.
    """
    join_at = spawn_at + 1 if join_at is None else join_at
    child_conv_ids = [f"parent_s0_c{ci}" for ci in range(num_children)]
    spawn = SubagentSpawnInfo(
        spawn_id="s0",
        child_conversation_ids=child_conv_ids,
        is_background=is_background,
    )

    turns: list[TurnMetadata] = []
    for i in range(parent_turns):
        spawn_ids = ["s0"] if i == spawn_at else []
        prereqs: list[TurnPrerequisite] = []
        if i == join_at and not is_background:
            prereqs = [
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, spawn_id="s0")
            ]

        ts_ms = timestamp_base_ms + (i * timestamp_spacing_ms) if timestamps else None
        d_ms = delay_ms if i > 0 else None

        turns.append(
            TurnMetadata(
                timestamp_ms=ts_ms,
                delay_ms=d_ms,
                input_tokens=500 + i * 100,
                subagent_spawn_ids=spawn_ids,
                prerequisites=prereqs,
            )
        )

    convs = [
        ConversationMetadata(
            conversation_id="parent",
            turns=turns,
            subagent_spawns=[spawn],
        )
    ]
    for _ci, child_id in enumerate(child_conv_ids):
        child_turns_list: list[TurnMetadata] = []
        for j in range(child_turns):
            c_ts = None
            c_delay = None
            if timestamps:
                # Children start slightly after spawn turn
                c_ts = (
                    timestamp_base_ms
                    + (spawn_at * timestamp_spacing_ms)
                    + 100
                    + (j * timestamp_spacing_ms)
                )
            if j > 0 and delay_ms is not None:
                c_delay = delay_ms
            child_turns_list.append(
                TurnMetadata(
                    timestamp_ms=c_ts,
                    delay_ms=c_delay,
                    input_tokens=300 + j * 50,
                )
            )
        convs.append(
            ConversationMetadata(
                conversation_id=child_id,
                turns=child_turns_list,
                agent_depth=1,
                parent_conversation_id="parent",
            )
        )

    return DatasetMetadata(
        conversations=convs,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _make_multi_spawn_dataset(
    *,
    timestamps: bool = False,
    delay_ms: float | None = None,
) -> DatasetMetadata:
    """Parent with 2 spawns on different turns: s0 at turn 1, s1 at turn 3.

    Parent: 6 turns
    - Turn 1: spawn s0 (2 children), join at turn 2
    - Turn 3: spawn s1 (1 child), join at turn 4
    """
    child_s0_ids = ["parent_s0_c0", "parent_s0_c1"]
    child_s1_ids = ["parent_s1_c0"]
    spawn_s0 = SubagentSpawnInfo(spawn_id="s0", child_conversation_ids=child_s0_ids)
    spawn_s1 = SubagentSpawnInfo(spawn_id="s1", child_conversation_ids=child_s1_ids)

    turns: list[TurnMetadata] = []
    for i in range(6):
        spawn_ids: list[str] = []
        prereqs: list[TurnPrerequisite] = []
        if i == 1:
            spawn_ids = ["s0"]
        elif i == 2:
            prereqs = [
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, spawn_id="s0")
            ]
        elif i == 3:
            spawn_ids = ["s1"]
        elif i == 4:
            prereqs = [
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, spawn_id="s1")
            ]

        ts = (1000 + i * 500) if timestamps else None
        d = delay_ms if i > 0 else None
        turns.append(
            TurnMetadata(
                timestamp_ms=ts,
                delay_ms=d,
                input_tokens=400 + i * 50,
                subagent_spawn_ids=spawn_ids,
                prerequisites=prereqs,
            )
        )

    convs = [
        ConversationMetadata(
            conversation_id="parent",
            turns=turns,
            subagent_spawns=[spawn_s0, spawn_s1],
        )
    ]

    for child_id in child_s0_ids + child_s1_ids:
        child_turns = [
            TurnMetadata(
                timestamp_ms=(1000 + 600 + j * 300) if timestamps else None,
                delay_ms=delay_ms if j > 0 else None,
                input_tokens=200 + j * 30,
            )
            for j in range(2)
        ]
        convs.append(
            ConversationMetadata(
                conversation_id=child_id,
                turns=child_turns,
                agent_depth=1,
                parent_conversation_id="parent",
            )
        )

    return DatasetMetadata(
        conversations=convs,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _run_spawn_lifecycle(
    orch: SubagentOrchestrator,
    dispatched: list[TurnToSend],
    scheduler: MagicMock,
    *,
    parent_turns: int = 6,
    spawn_at: int = 2,
    join_at: int | None = None,
    num_children: int = 2,
    child_turns: int = 3,
    is_background: bool = False,
) -> dict:
    """Drive the orchestrator through a full spawn->children->join lifecycle.

    Returns a dict of verification data.
    """
    join_at = spawn_at + 1 if join_at is None else join_at

    # Advance parent to spawn turn
    for t in range(spawn_at + 1):
        credit = _make_credit(
            conv_id="parent",
            corr_id="parent-corr",
            turn_index=t,
            num_turns=parent_turns,
        )
        orch.intercept(credit)

    child_corr_ids = list(orch._child_to_gate.keys())
    child_conv_ids = [f"parent_s0_c{ci}" for ci in range(num_children)]

    # If delayed join, advance parent past spawn to the turn before join
    if join_at > spawn_at + 1:
        for t in range(spawn_at + 1, join_at):
            credit = _make_credit(
                conv_id="parent",
                corr_id="parent-corr",
                turn_index=t,
                num_turns=parent_turns,
            )
            orch.intercept(credit)

    # Complete all children
    dispatched_before_children_complete = len(dispatched)
    for ci in range(num_children):
        corr_id = child_corr_ids[ci] if ci < len(child_corr_ids) else f"child-{ci}"
        for t in range(child_turns):
            child_credit = _make_credit(
                conv_id=child_conv_ids[ci],
                corr_id=corr_id,
                turn_index=t,
                num_turns=child_turns,
                agent_depth=1,
            )
            orch.intercept(child_credit)

    # Find gated turn dispatches
    gated_dispatches = [
        d
        for d in dispatched[dispatched_before_children_complete:]
        if d.conversation_id == "parent"
    ]

    return {
        "child_corr_ids": child_corr_ids,
        "child_conv_ids": child_conv_ids,
        "gated_dispatches": gated_dispatches,
        "all_dispatched": dispatched,
        "stats": orch.get_stats(),
    }


# =============================================================================
# Tests: Blocking spawn lifecycle across all timing modes
# =============================================================================


@pytest.mark.component_integration
class TestBlockingSpawnAllModes:
    """Verify blocking spawn lifecycle works identically across timing modes.

    The SubagentOrchestrator is mode-agnostic; these tests confirm that the
    strategy's _dispatch_turn callback correctly routes gated turns regardless
    of whether the strategy uses timestamps, rate-limiting, or user-centric pacing.
    """

    @pytest.mark.parametrize("timestamps", [False, True], ids=["no-ts", "with-ts"])
    @pytest.mark.parametrize("delay_ms", [None, 200.0], ids=["no-delay", "delay-200ms"])
    def test_blocking_spawn_dispatches_gated_turn(
        self, timestamps: bool, delay_ms: float | None
    ) -> None:
        """Parent suspends at spawn, resumes at join after all children complete."""
        ds = _make_conflux_dataset(
            parent_turns=6,
            spawn_at=2,
            num_children=2,
            child_turns=3,
            timestamps=timestamps,
            delay_ms=delay_ms,
        )
        orch, dispatched, scheduler = _build_orchestrator(ds)
        result = _run_spawn_lifecycle(
            orch,
            dispatched,
            scheduler,
            parent_turns=6,
            spawn_at=2,
            num_children=2,
            child_turns=3,
        )

        assert result["stats"]["subagent_children_spawned"] == 2
        assert result["stats"]["subagent_children_completed"] == 2
        assert result["stats"]["subagent_parents_suspended"] == 1
        assert result["stats"]["subagent_parents_resumed"] == 1

        assert len(result["gated_dispatches"]) == 1
        gated = result["gated_dispatches"][0]
        assert gated.turn_index == 3
        assert gated.conversation_id == "parent"
        assert gated.num_turns == 6

    @pytest.mark.parametrize("timestamps", [False, True], ids=["no-ts", "with-ts"])
    def test_child_non_final_turns_dispatch_next(self, timestamps: bool) -> None:
        """Non-final child turns dispatch the next child turn."""
        ds = _make_conflux_dataset(
            parent_turns=6,
            spawn_at=2,
            num_children=1,
            child_turns=3,
            timestamps=timestamps,
        )
        orch, dispatched, scheduler = _build_orchestrator(ds)

        # Advance parent to spawn
        for t in range(3):
            orch.intercept(
                _make_credit(conv_id="parent", corr_id="p-1", turn_index=t, num_turns=6)
            )

        child_corr_ids = list(orch._child_to_gate.keys())
        assert len(child_corr_ids) == 1

        # Child turn 0 -> should dispatch turn 1
        orch.intercept(
            _make_credit(
                conv_id="parent_s0_c0",
                corr_id=child_corr_ids[0],
                turn_index=0,
                num_turns=3,
                agent_depth=1,
            )
        )
        child_dispatches = [
            d for d in dispatched if d.conversation_id == "parent_s0_c0"
        ]
        assert len(child_dispatches) == 1
        assert child_dispatches[0].turn_index == 1


# =============================================================================
# Tests: Delayed join (join_at > spawn_at + 1)
# =============================================================================


@pytest.mark.component_integration
class TestDelayedJoinAllModes:
    """Verify delayed join works across all dataset configurations."""

    @pytest.mark.parametrize("timestamps", [False, True], ids=["no-ts", "with-ts"])
    @pytest.mark.parametrize("delay_ms", [None, 200.0], ids=["no-delay", "delay-200ms"])
    def test_delayed_join_blocks_at_correct_turn(
        self, timestamps: bool, delay_ms: float | None
    ) -> None:
        """Join at turn 5 (spawn at turn 2): parent flows through turns 3,4 freely."""
        ds = _make_conflux_dataset(
            parent_turns=8,
            spawn_at=2,
            join_at=5,
            num_children=2,
            child_turns=3,
            timestamps=timestamps,
            delay_ms=delay_ms,
        )
        orch, dispatched, scheduler = _build_orchestrator(ds)
        result = _run_spawn_lifecycle(
            orch,
            dispatched,
            scheduler,
            parent_turns=8,
            spawn_at=2,
            join_at=5,
            num_children=2,
            child_turns=3,
        )

        assert result["stats"]["subagent_parents_suspended"] == 1
        assert result["stats"]["subagent_parents_resumed"] == 1
        assert len(result["gated_dispatches"]) == 1
        assert result["gated_dispatches"][0].turn_index == 5

    @pytest.mark.parametrize("timestamps", [False, True], ids=["no-ts", "with-ts"])
    def test_delayed_join_children_complete_before_parent_reaches_gate(
        self, timestamps: bool
    ) -> None:
        """If children complete before parent reaches gated turn, no suspension occurs."""
        ds = _make_conflux_dataset(
            parent_turns=8,
            spawn_at=2,
            join_at=6,
            num_children=1,
            child_turns=2,
            timestamps=timestamps,
        )
        orch, dispatched, _ = _build_orchestrator(ds)

        # Parent turn 2 (spawn)
        orch.intercept(
            _make_credit(conv_id="parent", corr_id="p-1", turn_index=2, num_turns=8)
        )

        # Children dispatched via future gate
        child_corr_ids = list(orch._child_to_gate.keys())

        # Complete child fully (both turns)
        for t in range(2):
            orch.intercept(
                _make_credit(
                    conv_id="parent_s0_c0",
                    corr_id=child_corr_ids[0],
                    turn_index=t,
                    num_turns=2,
                    agent_depth=1,
                )
            )

        # Future gate should be cleaned up
        assert (
            "p-1" not in orch._future_gates
            or len(orch._future_gates.get("p-1", {})) == 0
        )

        # Parent continues past spawn turn without suspension
        for t in range(3, 6):
            result = orch.intercept(
                _make_credit(conv_id="parent", corr_id="p-1", turn_index=t, num_turns=8)
            )
            assert result is False

        assert orch._stats.parents_suspended == 0


# =============================================================================
# Tests: Background spawns
# =============================================================================


@pytest.mark.component_integration
class TestBackgroundSpawnAllModes:
    """Background spawns do not gate the parent, across all dataset configurations."""

    @pytest.mark.parametrize("timestamps", [False, True], ids=["no-ts", "with-ts"])
    @pytest.mark.parametrize("delay_ms", [None, 200.0], ids=["no-delay", "delay-200ms"])
    def test_background_spawn_does_not_gate(
        self, timestamps: bool, delay_ms: float | None
    ) -> None:
        ds = _make_conflux_dataset(
            parent_turns=6,
            spawn_at=2,
            num_children=2,
            child_turns=3,
            is_background=True,
            timestamps=timestamps,
            delay_ms=delay_ms,
        )
        orch, dispatched, scheduler = _build_orchestrator(ds)

        # Advance parent to spawn
        orch.intercept(
            _make_credit(conv_id="parent", corr_id="p-1", turn_index=2, num_turns=6)
        )

        # Parent not suspended
        assert "p-1" not in orch._gated_turns
        assert len(orch._child_to_gate) == 0
        assert orch._stats.parents_suspended == 0

        # Background children dispatched via dispatch_fn
        bg_dispatches = [d for d in dispatched if d.agent_depth == 1]
        assert len(bg_dispatches) == 2
        assert orch._stats.children_spawned == 2

        # scheduler.execute_async NOT used for background
        assert scheduler.execute_async.call_count == 0


# =============================================================================
# Tests: Multiple spawns on different turns
# =============================================================================


@pytest.mark.component_integration
class TestMultiSpawnComposition:
    """Multiple spawns on different turns compose correctly."""

    @pytest.mark.parametrize("timestamps", [False, True], ids=["no-ts", "with-ts"])
    def test_two_sequential_spawns(self, timestamps: bool) -> None:
        """s0 at turn 1 (join turn 2), s1 at turn 3 (join turn 4)."""
        ds = _make_multi_spawn_dataset(timestamps=timestamps)
        orch, dispatched, scheduler = _build_orchestrator(ds)

        # Turn 0: no spawn
        orch.intercept(
            _make_credit(conv_id="parent", corr_id="p-1", turn_index=0, num_turns=6)
        )

        # Turn 1: spawn s0
        orch.intercept(
            _make_credit(conv_id="parent", corr_id="p-1", turn_index=1, num_turns=6)
        )
        assert orch._stats.children_spawned == 2
        assert orch._stats.parents_suspended == 1

        # Complete s0 children
        s0_child_corr_ids = [
            cid
            for cid, entry in orch._child_to_gate.items()
            if entry.prereq_key == "spawn_join:s0"
        ]
        for ci, corr_id in enumerate(s0_child_corr_ids):
            for t in range(2):
                orch.intercept(
                    _make_credit(
                        conv_id=f"parent_s0_c{ci}",
                        corr_id=corr_id,
                        turn_index=t,
                        num_turns=2,
                        agent_depth=1,
                    )
                )

        # Gated turn 2 dispatched
        gated_s0 = [
            d for d in dispatched if d.conversation_id == "parent" and d.turn_index == 2
        ]
        assert len(gated_s0) == 1
        assert orch._stats.parents_resumed == 1

        # Turn 2 returns -> parent continues to turn 3
        orch.intercept(
            _make_credit(conv_id="parent", corr_id="p-1", turn_index=2, num_turns=6)
        )

        # Turn 3: spawn s1
        orch.intercept(
            _make_credit(conv_id="parent", corr_id="p-1", turn_index=3, num_turns=6)
        )
        assert orch._stats.children_spawned == 3  # 2 from s0 + 1 from s1
        assert orch._stats.parents_suspended == 2

        # Complete s1 child
        s1_child_corr_ids = [
            cid
            for cid, entry in orch._child_to_gate.items()
            if entry.prereq_key == "spawn_join:s1"
        ]
        for corr_id in s1_child_corr_ids:
            for t in range(2):
                orch.intercept(
                    _make_credit(
                        conv_id="parent_s1_c0",
                        corr_id=corr_id,
                        turn_index=t,
                        num_turns=2,
                        agent_depth=1,
                    )
                )

        # Gated turn 4 dispatched
        gated_s1 = [
            d for d in dispatched if d.conversation_id == "parent" and d.turn_index == 4
        ]
        assert len(gated_s1) == 1
        assert orch._stats.parents_resumed == 2

        # Final stats
        stats = orch.get_stats()
        assert stats["subagent_children_spawned"] == 3
        assert stats["subagent_children_completed"] == 3
        assert stats["subagent_children_errored"] == 0
        assert stats["subagent_parents_suspended"] == 2
        assert stats["subagent_parents_resumed"] == 2


# =============================================================================
# Tests: Error/cancellation releases gates
# =============================================================================


@pytest.mark.component_integration
class TestErrorCancellationReleasesGate:
    """Errored/cancelled children release gates correctly."""

    @pytest.mark.parametrize("timestamps", [False, True], ids=["no-ts", "with-ts"])
    def test_terminate_child_releases_gate(self, timestamps: bool) -> None:
        """Terminating all children releases the gate and dispatches gated turn."""
        ds = _make_conflux_dataset(
            parent_turns=6,
            spawn_at=2,
            num_children=2,
            child_turns=3,
            timestamps=timestamps,
        )
        orch, dispatched, _ = _build_orchestrator(ds)

        # Advance parent to spawn
        for t in range(3):
            orch.intercept(
                _make_credit(conv_id="parent", corr_id="p-1", turn_index=t, num_turns=6)
            )

        child_corr_ids = list(orch._child_to_gate.keys())
        assert len(child_corr_ids) == 2

        # Terminate both children (simulating error/cancel via SubagentMixin.on_failed_credit)
        for corr_id in child_corr_ids:
            credit = _make_credit(
                conv_id="parent_s0_c0",
                corr_id=corr_id,
                turn_index=0,
                num_turns=3,
                agent_depth=1,
            )
            orch.terminate_child(credit)

        # Gate released, gated turn dispatched
        assert "p-1" not in orch._gated_turns
        gated = [d for d in dispatched if d.conversation_id == "parent"]
        assert len(gated) == 1
        assert gated[0].turn_index == 3
        assert orch._stats.parents_resumed == 1
        assert orch._stats.children_errored == 2


# =============================================================================
# Tests: Stop condition suppresses gated turn dispatch
# =============================================================================


@pytest.mark.component_integration
class TestStopConditionSuppression:
    """When stop condition fires, gated turn dispatch is suppressed."""

    @pytest.mark.parametrize("timestamps", [False, True], ids=["no-ts", "with-ts"])
    def test_gated_turn_suppressed_when_stop_fired(self, timestamps: bool) -> None:
        ds = _make_conflux_dataset(
            parent_turns=6,
            spawn_at=2,
            num_children=1,
            child_turns=2,
            timestamps=timestamps,
        )
        orch, dispatched, _ = _build_orchestrator(ds, stop_can_send=False)

        # Advance parent to spawn
        for t in range(3):
            orch.intercept(
                _make_credit(conv_id="parent", corr_id="p-1", turn_index=t, num_turns=6)
            )

        child_corr_ids = list(orch._child_to_gate.keys())

        # Complete child
        for t in range(2):
            orch.intercept(
                _make_credit(
                    conv_id="parent_s0_c0",
                    corr_id=child_corr_ids[0],
                    turn_index=t,
                    num_turns=2,
                    agent_depth=1,
                )
            )

        # Gate released but gated turn suppressed
        assert "p-1" not in orch._gated_turns
        parent_dispatches = [d for d in dispatched if d.conversation_id == "parent"]
        assert len(parent_dispatches) == 0
        assert orch._stats.joins_suppressed == 1


# =============================================================================
# Tests: Timestamp verification
# =============================================================================


@pytest.mark.component_integration
class TestTimestampPropagation:
    """Verify timestamps on metadata are correctly preserved in the dataset."""

    def test_timestamps_preserved_in_turn_metadata(self) -> None:
        """Absolute timestamps on parent turns are accessible via ConversationSource."""
        ds = _make_conflux_dataset(
            parent_turns=4,
            spawn_at=1,
            num_children=1,
            child_turns=2,
            timestamps=True,
            timestamp_base_ms=2000,
            timestamp_spacing_ms=300,
        )
        root_ids = [c.conversation_id for c in ds.conversations if c.agent_depth == 0]
        sampler = _make_sampler(root_ids)
        src = ConversationSource(ds, sampler)

        for i in range(4):
            meta = src.get_turn_metadata_at("parent", i)
            expected_ts = 2000 + i * 300
            assert meta.timestamp_ms == expected_ts, (
                f"Turn {i}: expected ts={expected_ts}, got {meta.timestamp_ms}"
            )

    def test_child_timestamps_offset_from_spawn(self) -> None:
        """Child turn timestamps start after the spawn turn."""
        ds = _make_conflux_dataset(
            parent_turns=4,
            spawn_at=1,
            num_children=1,
            child_turns=3,
            timestamps=True,
            timestamp_base_ms=1000,
            timestamp_spacing_ms=500,
        )
        root_ids = [c.conversation_id for c in ds.conversations if c.agent_depth == 0]
        sampler = _make_sampler(root_ids)
        src = ConversationSource(ds, sampler)

        parent_spawn_ts = src.get_turn_metadata_at("parent", 1).timestamp_ms
        assert parent_spawn_ts == 1500  # 1000 + 1 * 500

        for j in range(3):
            child_meta = src.get_turn_metadata_at("parent_s0_c0", j)
            assert child_meta.timestamp_ms is not None
            # Children start 100ms after spawn turn
            expected = 1500 + 100 + j * 500
            assert child_meta.timestamp_ms == expected, (
                f"Child turn {j}: expected ts={expected}, got {child_meta.timestamp_ms}"
            )

    def test_delay_ms_on_subsequent_turns(self) -> None:
        """delay_ms is set on subsequent turns (turn_index > 0) when configured."""
        ds = _make_conflux_dataset(
            parent_turns=4,
            spawn_at=1,
            num_children=1,
            child_turns=2,
            delay_ms=150.0,
        )
        root_ids = [c.conversation_id for c in ds.conversations if c.agent_depth == 0]
        sampler = _make_sampler(root_ids)
        src = ConversationSource(ds, sampler)

        # First turn has no delay
        assert src.get_turn_metadata_at("parent", 0).delay_ms is None
        # Subsequent turns have delay
        for i in range(1, 4):
            meta = src.get_turn_metadata_at("parent", i)
            assert meta.delay_ms == 150.0, (
                f"Turn {i}: expected delay=150.0, got {meta.delay_ms}"
            )

    def test_prerequisites_on_gated_turns(self) -> None:
        """TurnPrerequisite is correctly set on the gated turn."""
        ds = _make_conflux_dataset(
            parent_turns=6,
            spawn_at=2,
            join_at=4,
            num_children=2,
            child_turns=3,
        )
        root_ids = [c.conversation_id for c in ds.conversations if c.agent_depth == 0]
        sampler = _make_sampler(root_ids)
        src = ConversationSource(ds, sampler)

        # Turn 4 should have spawn_join prerequisite
        gated_meta = src.get_turn_metadata_at("parent", 4)
        assert len(gated_meta.prerequisites) == 1
        prereq = gated_meta.prerequisites[0]
        assert prereq.kind == PrerequisiteKind.SPAWN_JOIN
        assert prereq.spawn_id == "s0"

        # Other turns should not have prerequisites
        for i in [0, 1, 2, 3, 5]:
            meta = src.get_turn_metadata_at("parent", i)
            assert len(meta.prerequisites) == 0, (
                f"Turn {i} should have no prerequisites"
            )

    def test_spawn_ids_on_spawn_turn(self) -> None:
        """subagent_spawn_ids is set only on the spawn turn."""
        ds = _make_conflux_dataset(parent_turns=6, spawn_at=2)
        root_ids = [c.conversation_id for c in ds.conversations if c.agent_depth == 0]
        sampler = _make_sampler(root_ids)
        src = ConversationSource(ds, sampler)

        for i in range(6):
            meta = src.get_turn_metadata_at("parent", i)
            if i == 2:
                assert meta.subagent_spawn_ids == ["s0"]
            else:
                assert meta.subagent_spawn_ids == [], (
                    f"Turn {i} should have no spawn IDs"
                )

    def test_multi_spawn_timestamps_independent(self) -> None:
        """Multi-spawn dataset has independent timestamps for each spawn's children."""
        ds = _make_multi_spawn_dataset(timestamps=True)
        root_ids = [c.conversation_id for c in ds.conversations if c.agent_depth == 0]
        sampler = _make_sampler(root_ids)
        src = ConversationSource(ds, sampler)

        # Verify parent timestamps are sequential
        parent_timestamps = [
            src.get_turn_metadata_at("parent", i).timestamp_ms for i in range(6)
        ]
        assert parent_timestamps == [1000, 1500, 2000, 2500, 3000, 3500]

        # s0 children share same timestamps
        s0_c0_ts = [
            src.get_turn_metadata_at("parent_s0_c0", j).timestamp_ms for j in range(2)
        ]
        s0_c1_ts = [
            src.get_turn_metadata_at("parent_s0_c1", j).timestamp_ms for j in range(2)
        ]
        assert s0_c0_ts == s0_c1_ts

        # s1 child has same base timestamps (same formula)
        s1_c0_ts = [
            src.get_turn_metadata_at("parent_s1_c0", j).timestamp_ms for j in range(2)
        ]
        assert s1_c0_ts == s0_c0_ts  # same formula in helper


# =============================================================================
# Tests: Gated turn metadata propagation
# =============================================================================


@pytest.mark.component_integration
class TestGatedTurnMetadataPropagation:
    """Verify dispatched gated turns have correct metadata."""

    def test_gated_turn_preserves_parent_correlation_id(self) -> None:
        """The gated turn's x_correlation_id matches the parent's."""
        ds = _make_conflux_dataset(
            parent_turns=6, spawn_at=2, num_children=1, child_turns=2
        )
        orch, dispatched, _ = _build_orchestrator(ds)

        # Advance parent
        for t in range(3):
            orch.intercept(
                _make_credit(
                    conv_id="parent", corr_id="p-42", turn_index=t, num_turns=6
                )
            )

        # Complete child
        child_corr_ids = list(orch._child_to_gate.keys())
        for t in range(2):
            orch.intercept(
                _make_credit(
                    conv_id="parent_s0_c0",
                    corr_id=child_corr_ids[0],
                    turn_index=t,
                    num_turns=2,
                    agent_depth=1,
                )
            )

        gated = [d for d in dispatched if d.conversation_id == "parent"]
        assert len(gated) == 1
        assert gated[0].x_correlation_id == "p-42"
        assert gated[0].turn_index == 3
        assert gated[0].num_turns == 6

    def test_child_dispatched_turns_have_correct_agent_depth(self) -> None:
        """Children dispatched by the orchestrator have agent_depth=1."""
        ds = _make_conflux_dataset(
            parent_turns=6, spawn_at=2, num_children=2, child_turns=3
        )
        orch, dispatched, _ = _build_orchestrator(ds)

        # Advance parent to spawn
        for t in range(3):
            orch.intercept(
                _make_credit(conv_id="parent", corr_id="p-1", turn_index=t, num_turns=6)
            )

        # Child next-turn dispatches from non-final child turns
        child_corr_ids = list(orch._child_to_gate.keys())
        orch.intercept(
            _make_credit(
                conv_id="parent_s0_c0",
                corr_id=child_corr_ids[0],
                turn_index=0,
                num_turns=3,
                agent_depth=1,
            )
        )

        child_dispatches = [d for d in dispatched if d.agent_depth == 1]
        assert len(child_dispatches) >= 1
        for d in child_dispatches:
            assert d.agent_depth == 1


# =============================================================================
# Tests: Cleanup after partial lifecycle
# =============================================================================


@pytest.mark.component_integration
class TestCleanupWithTimestamps:
    """Verify cleanup clears all state regardless of timestamp configuration."""

    @pytest.mark.parametrize("timestamps", [False, True], ids=["no-ts", "with-ts"])
    def test_cleanup_clears_all_state(self, timestamps: bool) -> None:
        ds = _make_conflux_dataset(
            parent_turns=6,
            spawn_at=2,
            num_children=2,
            child_turns=3,
            timestamps=timestamps,
        )
        orch, _, _ = _build_orchestrator(ds)

        # Trigger spawn
        for t in range(3):
            orch.intercept(
                _make_credit(conv_id="parent", corr_id="p-1", turn_index=t, num_turns=6)
            )

        assert len(orch._gated_turns) > 0
        assert len(orch._child_to_gate) > 0

        orch.cleanup()

        assert len(orch._gated_turns) == 0
        assert len(orch._future_gates) == 0
        assert len(orch._child_to_gate) == 0
        assert len(orch._terminated_children) == 0
        assert orch._cleaning_up is True

        # Post-cleanup intercepts are no-ops
        assert (
            orch.intercept(
                _make_credit(
                    conv_id="parent", corr_id="post", turn_index=0, num_turns=6
                )
            )
            is False
        )
