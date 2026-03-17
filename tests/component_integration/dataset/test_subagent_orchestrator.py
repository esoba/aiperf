# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for SubagentOrchestrator with real dataset pipeline.

Exercises the full lifecycle:
  DatasetMetadata -> ConversationSource -> SubagentOrchestrator
  -> spawn resolution -> child dispatch -> child completion -> gated turn dispatch

Uses real dataset generation (no mocks for data model) with a capturing dispatch_fn
to verify the orchestrator state machine end-to-end.
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
from aiperf.plugin import plugins
from aiperf.plugin.enums import DatasetSamplingStrategy, PluginType
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.subagent_orchestrator import SubagentOrchestrator

# =============================================================================
# Helpers
# =============================================================================


def _make_sampler(conv_ids, strategy=DatasetSamplingStrategy.SEQUENTIAL):
    SamplerClass = plugins.get_class(PluginType.DATASET_SAMPLER, strategy)
    return SamplerClass(conversation_ids=conv_ids)


def _make_credit(
    *,
    conv_id: str,
    corr_id: str,
    turn_index: int,
    num_turns: int,
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


def _build_orchestrator_from_metadata(
    ds: DatasetMetadata,
) -> tuple[SubagentOrchestrator, list[TurnToSend], MagicMock]:
    """Build orchestrator from dataset metadata with capturing dispatch_fn."""
    root_ids = [c.conversation_id for c in ds.conversations if c.agent_depth == 0]
    sampler = _make_sampler(root_ids, ds.sampling_strategy)
    src = ConversationSource(ds, sampler)

    scheduler = MagicMock()
    scheduler.execute_async = MagicMock()
    dispatched: list[TurnToSend] = []

    orch = SubagentOrchestrator(
        conversation_source=src,
        credit_issuer=MagicMock(issue_credit=AsyncMock(return_value=True)),
        stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
        scheduler=scheduler,
        dispatch_fn=lambda turn: dispatched.append(turn),
    )
    return orch, dispatched, scheduler


def _make_handcrafted_dataset(
    *,
    parent_turns: int = 6,
    spawn_at: int = 2,
    join_at: int | None = None,
    num_children: int = 2,
    child_turns: int = 3,
    is_background: bool = False,
) -> DatasetMetadata:
    """Create a handcrafted dataset with one parent and N children."""
    join_at = spawn_at + 1 if join_at is None else join_at
    child_conv_ids = [f"conv_0_s0_c{ci}" for ci in range(num_children)]
    spawn = SubagentSpawnInfo(
        spawn_id="s0",
        child_conversation_ids=child_conv_ids,
        is_background=is_background,
    )

    turns = []
    for i in range(parent_turns):
        spawn_ids = ["s0"] if i == spawn_at else []
        prereqs = []
        if i == join_at and not is_background:
            prereqs = [
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, spawn_id="s0")
            ]
        turns.append(
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
            turns=turns,
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

    return DatasetMetadata(
        conversations=convs,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


# =============================================================================
# Full lifecycle with handcrafted dataset
# =============================================================================


@pytest.mark.component_integration
class TestSubagentOrchestratorFullLifecycle:
    """Walk through the full orchestrator lifecycle with a handcrafted dataset."""

    def test_full_lifecycle_blocking_spawn(self):
        """Root -> spawn turn -> children dispatched -> children complete -> gated turn dispatched."""
        ds = _make_handcrafted_dataset(
            parent_turns=6, spawn_at=2, num_children=2, child_turns=3
        )
        orch, dispatched, scheduler = _build_orchestrator_from_metadata(ds)

        # Step 1: Root turn 0 credit returns -- no spawn
        credit_t0 = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=0, num_turns=6
        )
        handled = orch.intercept(credit_t0)
        assert handled is False
        assert len(dispatched) == 0

        # Step 2: Root turn 1 credit returns -- no spawn
        credit_t1 = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=1, num_turns=6
        )
        handled = orch.intercept(credit_t1)
        assert handled is False

        # Step 3: Root turn 2 (spawn turn) credit returns -- intercept resolves spawns
        credit_t2 = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        handled = orch.intercept(credit_t2)
        assert handled is True

        # Blocking children dispatched via scheduler.execute_async
        assert scheduler.execute_async.call_count == 2
        assert orch._stats.children_spawned == 2
        assert orch._stats.parents_suspended == 1

        # Pending gate created
        assert "parent-1" in orch._gated_turns
        gate = orch._gated_turns["parent-1"]
        assert gate.outstanding == {"spawn_join:s0": [2, 0]}
        assert gate.gated_turn_index == 3

        # Child-to-gate mapping
        child_corr_ids = list(orch._child_to_gate.keys())
        assert len(child_corr_ids) == 2
        for entry in orch._child_to_gate.values():
            assert entry.parent_corr_id == "parent-1"

        child_conv_ids = ["conv_0_s0_c0", "conv_0_s0_c1"]

        # Step 4: Child 0 non-final turns dispatch next turn via dispatch_fn
        child0_t0 = _make_credit(
            conv_id=child_conv_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=0,
            num_turns=3,
            agent_depth=1,
        )
        orch.intercept(child0_t0)
        assert len(dispatched) == 1
        assert dispatched[0].conversation_id == child_conv_ids[0]
        assert dispatched[0].turn_index == 1

        # Child 0 turn 1 (non-final)
        child0_t1 = _make_credit(
            conv_id=child_conv_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=1,
            num_turns=3,
            agent_depth=1,
        )
        orch.intercept(child0_t1)
        assert len(dispatched) == 2
        assert dispatched[1].turn_index == 2

        # Step 5: Child 0 final turn -> gate accounting
        child0_t2 = _make_credit(
            conv_id=child_conv_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
        orch.intercept(child0_t2)
        assert orch._stats.children_completed == 1
        assert gate.outstanding["spawn_join:s0"][1] == 1
        # No gated turn yet (1 of 2)
        join_dispatches = [d for d in dispatched if d.conversation_id == "conv_0"]
        assert len(join_dispatches) == 0

        # Step 6: Child 1 all turns (fast-forward to final)
        child1_final = _make_credit(
            conv_id=child_conv_ids[1],
            corr_id=child_corr_ids[1],
            turn_index=2,
            num_turns=3,
            agent_depth=1,
        )
        orch.intercept(child1_final)

        # Step 7: All children done -> gated turn dispatched
        assert orch._stats.children_completed == 2
        assert orch._stats.parents_resumed == 1
        assert "parent-1" not in orch._gated_turns

        join_dispatches = [d for d in dispatched if d.conversation_id == "conv_0"]
        assert len(join_dispatches) == 1
        join_turn = join_dispatches[0]
        assert join_turn.turn_index == 3  # spawn_at(2) + 1
        assert join_turn.x_correlation_id == "parent-1"
        assert join_turn.num_turns == 6

        # Step 8: Verify stats
        stats = orch.get_stats()
        assert stats["subagent_children_spawned"] == 2
        assert stats["subagent_children_completed"] == 2
        assert stats["subagent_children_errored"] == 0
        assert stats["subagent_parents_suspended"] == 1
        assert stats["subagent_parents_resumed"] == 1
        assert stats["subagent_joins_suppressed"] == 0

    def test_full_lifecycle_delayed_join_blocks_on_target_turn(self):
        """A later spawn_join blocks only when the parent reaches that turn."""
        ds = _make_handcrafted_dataset(
            parent_turns=7,
            spawn_at=2,
            join_at=5,
            num_children=2,
            child_turns=3,
        )
        orch, dispatched, scheduler = _build_orchestrator_from_metadata(ds)

        credit_t2 = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=7
        )
        handled = orch.intercept(credit_t2)
        assert handled is False
        assert scheduler.execute_async.call_count == 2
        assert orch._stats.parents_suspended == 0
        assert orch._future_gates["parent-1"][5].outstanding == {
            "spawn_join:s0": [2, 0]
        }

        credit_t3 = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=3, num_turns=7
        )
        assert orch.intercept(credit_t3) is False

        credit_t4 = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=4, num_turns=7
        )
        assert orch.intercept(credit_t4) is True
        assert orch._stats.parents_suspended == 1
        assert orch._gated_turns["parent-1"].gated_turn_index == 5

        child_corr_ids = list(orch._child_to_gate.keys())
        child_conv_ids = ["conv_0_s0_c0", "conv_0_s0_c1"]
        for i, corr_id in enumerate(child_corr_ids):
            child_credit = _make_credit(
                conv_id=child_conv_ids[i],
                corr_id=corr_id,
                turn_index=2,
                num_turns=3,
                agent_depth=1,
            )
            orch.intercept(child_credit)

        assert "parent-1" not in orch._gated_turns
        assert orch._stats.parents_resumed == 1

        join_dispatches = [d for d in dispatched if d.conversation_id == "conv_0"]
        assert len(join_dispatches) == 1
        assert join_dispatches[0].turn_index == 5
        stats = orch.get_stats()
        assert stats["subagent_children_spawned"] == 2
        assert stats["subagent_children_completed"] == 2
        assert stats["subagent_parents_suspended"] == 1
        assert stats["subagent_parents_resumed"] == 1

    def test_full_lifecycle_background_spawn(self):
        """Background spawn: parent not suspended, children dispatched via dispatch_fn."""
        ds = _make_handcrafted_dataset(
            parent_turns=6,
            spawn_at=2,
            num_children=2,
            child_turns=3,
            is_background=True,
        )
        orch, dispatched, scheduler = _build_orchestrator_from_metadata(ds)

        credit_t2 = _make_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        handled = orch.intercept(credit_t2)

        # Background: parent not suspended
        assert handled is False
        assert "parent-1" not in orch._gated_turns
        assert len(orch._child_to_gate) == 0

        # Background children dispatched via dispatch_fn
        bg_dispatches = [d for d in dispatched if d.agent_depth == 1]
        assert len(bg_dispatches) == 2
        assert orch._stats.children_spawned == 2

        # scheduler.execute_async NOT used for background
        assert scheduler.execute_async.call_count == 0

    def test_cleanup_after_partial_lifecycle(self):
        """Cleanup after partial lifecycle clears all state."""
        ds = _make_handcrafted_dataset(
            parent_turns=6, spawn_at=2, num_children=2, child_turns=3
        )
        orch, _, _ = _build_orchestrator_from_metadata(ds)

        # Trigger a spawn
        credit = _make_credit(
            conv_id="conv_0",
            corr_id="partial-corr",
            turn_index=2,
            num_turns=6,
        )
        orch.intercept(credit)

        # Cleanup without completing children
        orch.cleanup()

        assert len(orch._gated_turns) == 0
        assert len(orch._future_gates) == 0
        assert len(orch._child_to_gate) == 0
        assert len(orch._terminated_children) == 0
        assert orch._cleaning_up is True

        # Further intercepts are no-ops
        credit = _make_credit(
            conv_id="conv_0", corr_id="post-cleanup", turn_index=0, num_turns=5
        )
        assert orch.intercept(credit) is False
