# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SubagentOrchestrator: composable subagent spawn/join component.

Strategies own an instance and call ``intercept(credit)`` at the top of their
handle_credit_return. The orchestrator handles all child routing, spawn
resolution, child dispatch, and join dispatch internally using a strategy-
provided dispatch callback. Strategies forward observer methods
(terminate_child, cleanup) as one-liners.

Stop Condition Interaction
==========================

Three coordinated mechanisms achieve zero-overshoot, zero-deadlock:

1. **Callback handler gate** (CreditCallbackHandler):
   Final child turns always pass through for join accounting.
   Non-final blocked children trigger on_child_stopped -> terminate_child.

2. **Credit issuance failure** (_issue_child_credit_or_release):
   When issue_credit returns False, the child is released from join tracking.

3. **Join dispatch suppression** (_maybe_complete_join):
   Checks can_send_any_turn() before dispatching the join turn.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.credit.structs import TurnToSend

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.credit.structs import Credit
    from aiperf.timing.conversation_source import ConversationSource, SampledSession


@dataclass(slots=True)
class PendingSubagentJoin:
    """Tracks completion of subagent children before dispatching the parent's join turn."""

    parent_conversation_id: str
    parent_correlation_id: str
    expected_count: int
    completed_count: int = 0
    join_turn_index: int = 0
    parent_num_turns: int = 0
    parent_agent_depth: int = 0
    parent_subagent_type: str | None = None
    parent_parent_correlation_id: str | None = None
    created_at_ns: int = 0


@dataclass(slots=True)
class SubagentStats:
    """Counters for subagent observability."""

    children_spawned: int = 0
    children_completed: int = 0
    children_errored: int = 0
    parents_suspended: int = 0
    parents_resumed: int = 0
    joins_suppressed: int = 0


class SubagentOrchestrator(AIPerfLoggerMixin):
    """Composable subagent spawn/join component.

    Handles all subagent lifecycle internally. The strategy provides a
    dispatch callback and calls ``intercept(credit)`` from handle_credit_return.
    The orchestrator uses the callback to dispatch child turns, next turns for
    non-final children, and join turns when all children complete.

    Strategy integration (5 lines total)::

        # __init__:
        self._subagents = SubagentOrchestrator(..., dispatch_fn=self._my_dispatch)

        # handle_credit_return:
        if self._subagents and self._subagents.intercept(credit):
            return

        # on_cancelled_return / on_child_stopped:
        if self._subagents: self._subagents.terminate_child(credit)

        # on_request_complete (if strategy has one):
        if self._subagents and credit_return.error:
            self._subagents.terminate_child(credit_return.credit)

        # cleanup:
        if self._subagents: self._subagents.cleanup()
    """

    def __init__(
        self,
        *,
        conversation_source: ConversationSource,
        credit_issuer: CreditIssuer,
        stop_checker: object,
        scheduler: LoopScheduler,
        dispatch_fn: Callable[[TurnToSend], None] | None = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            conversation_source: Dataset metadata and session creation.
            credit_issuer: Issues credits to workers.
            stop_checker: Evaluates stop conditions (has can_send_any_turn()).
            scheduler: For scheduling async coroutines (abandon wrapper).
            dispatch_fn: Strategy-specific dispatch. Called with a TurnToSend
                for join turns, child next turns, and background child first turns.
        """
        super().__init__(logger_name="SubagentOrchestrator")
        self._conversation_source = conversation_source
        self._credit_issuer = credit_issuer
        self._stop_checker = stop_checker
        self._scheduler = scheduler
        self._dispatch = dispatch_fn

        self._pending_joins: dict[str, PendingSubagentJoin] = {}
        self._child_to_parent: dict[str, str] = {}
        self._terminated_children: set[str] = set()
        self._cleaning_up: bool = False
        self._stats = SubagentStats()

    def set_dispatch(self, dispatch_fn: Callable[[TurnToSend], None]) -> None:
        """Set the strategy-provided dispatch callback.

        Called by timing strategies during __init__ when the orchestrator
        is constructed by the runner before the strategy exists.
        """
        self._dispatch = dispatch_fn

    # =========================================================================
    # Primary entry point: intercept credit returns
    # =========================================================================

    def intercept(self, credit: Credit) -> bool:
        """Handle subagent-related credit returns.

        Call at the top of handle_credit_return. Returns True if the credit
        was fully handled (strategy should return early). Returns False if
        the strategy should process the credit normally.
        """
        if self._cleaning_up:
            return False

        # Child credit returns
        if credit.agent_depth > 0:
            self._handle_child_credit(credit)
            return True

        # Spawn detection on completed turn
        spawn_ids = self._get_spawn_ids(credit)
        if spawn_ids:
            if credit.turn_index == 0:
                spawn_ids = [
                    sid
                    for sid in spawn_ids
                    if not self._is_background(credit.conversation_id, sid)
                ]
            if spawn_ids:
                suspended = self._resolve_and_dispatch_spawns(credit, spawn_ids)
                if suspended:
                    return True

        return False

    # =========================================================================
    # Turn-0 background pre-dispatch (called from execute_phase)
    # =========================================================================

    def dispatch_turn0_background_spawns(self) -> None:
        """Pre-dispatch background children for turn 0 of all root conversations."""
        for conv in self._conversation_source.dataset_metadata.conversations:
            if conv.agent_depth > 0 or not conv.turns:
                continue
            bg_ids = self._get_turn0_background_ids(conv.conversation_id)
            if not bg_ids:
                continue
            parent_corr_id = f"bg-turn0-{conv.conversation_id}"
            for spawn_id in bg_ids:
                spawn = self._conversation_source.get_subagent_spawn(
                    conv.conversation_id, spawn_id
                )
                if spawn is None:
                    continue
                for child_cid in spawn.child_conversation_ids:
                    session = self._conversation_source.start_child_session(child_cid)
                    turn = session.build_first_turn(
                        agent_depth=1, parent_correlation_id=parent_corr_id
                    )
                    self._dispatch(turn)
                    self._stats.children_spawned += 1

    # =========================================================================
    # Observer forwarding targets (strategies call these as one-liners)
    # =========================================================================

    def terminate_child(self, credit: Credit) -> None:
        """Release an errored/cancelled/stopped child from join tracking.

        Called from on_request_complete (errors), on_cancelled_return,
        and on_child_stopped. Dispatches the join turn internally if this
        was the last child the parent was waiting for.
        """
        if (
            self._cleaning_up
            or credit.agent_depth == 0
            or credit.is_final_turn
            or credit.x_correlation_id not in self._child_to_parent
        ):
            return
        self._stats.children_errored += 1
        self._terminated_children.add(credit.x_correlation_id)
        join = self._release_child(credit.x_correlation_id)
        if join:
            self._dispatch(join)

    def cleanup(self) -> None:
        """Log leaked state and clear all tracking."""
        self._cleaning_up = True
        if self._pending_joins or self._child_to_parent:
            now_ns = time.time_ns()
            for parent_corr_id, pending in self._pending_joins.items():
                age_ms = (now_ns - pending.created_at_ns) / 1_000_000
                self.warning(
                    f"Abandoned pending join for parent {parent_corr_id} "
                    f"(age={age_ms:.0f}ms, completed={pending.completed_count}/"
                    f"{pending.expected_count})"
                )
        self._pending_joins.clear()
        self._child_to_parent.clear()
        self._terminated_children.clear()

    def get_stats(self) -> dict[str, int]:
        """Return subagent counters for phase stats."""
        s = self._stats
        return {
            "subagent_children_spawned": s.children_spawned,
            "subagent_children_completed": s.children_completed,
            "subagent_children_errored": s.children_errored,
            "subagent_parents_suspended": s.parents_suspended,
            "subagent_parents_resumed": s.parents_resumed,
            "subagent_joins_suppressed": s.joins_suppressed,
        }

    # =========================================================================
    # Internal: child credit handling
    # =========================================================================

    def _handle_child_credit(self, credit: Credit) -> None:
        """Route a child credit: join accounting for final, next turn for non-final."""
        if credit.is_final_turn:
            if credit.x_correlation_id in self._child_to_parent:
                self._stats.children_completed += 1
                join = self._release_child(credit.x_correlation_id)
                if join:
                    self._dispatch(join)
        else:
            if not self._is_terminated(credit):
                turn = TurnToSend.from_previous_credit(credit)
                self._dispatch(turn)

    def _is_terminated(self, credit: Credit) -> bool:
        """Check and consume terminated status."""
        if credit.x_correlation_id in self._terminated_children:
            self._terminated_children.discard(credit.x_correlation_id)
            return True
        return False

    # =========================================================================
    # Internal: spawn resolution and child dispatch
    # =========================================================================

    def _get_spawn_ids(self, credit: Credit) -> list[str]:
        meta = self._conversation_source.get_turn_metadata_at(
            credit.conversation_id, credit.turn_index
        )
        return meta.subagent_spawn_ids

    def _get_turn0_background_ids(self, conversation_id: str) -> list[str]:
        meta = self._conversation_source.get_metadata(conversation_id)
        if not meta.turns or not meta.turns[0].subagent_spawn_ids:
            return []
        return [
            sid
            for sid in meta.turns[0].subagent_spawn_ids
            if self._is_background(conversation_id, sid)
        ]

    def _resolve_and_dispatch_spawns(
        self, credit: Credit, spawn_ids: list[str]
    ) -> bool:
        """Resolve spawns, register join tracking, dispatch children.

        Returns True if parent is suspended (blocking children registered).
        """
        parent_corr_id = credit.x_correlation_id
        child_depth = credit.agent_depth + 1

        resolved: list[tuple[bool, list[SampledSession]]] = []
        total_blocking = 0
        join_turn_index: int | None = None

        for spawn_id in spawn_ids:
            spawn = self._conversation_source.get_subagent_spawn(
                credit.conversation_id, spawn_id
            )
            if spawn is None:
                continue

            is_blocking = not spawn.is_background
            child_sessions = [
                self._conversation_source.start_child_session(cid)
                for cid in spawn.child_conversation_ids
            ]
            resolved.append((is_blocking, child_sessions))

            if is_blocking:
                total_blocking += len(spawn.child_conversation_ids)
                if join_turn_index is None:
                    join_turn_index = spawn.join_turn_index
                elif spawn.join_turn_index != join_turn_index:
                    self.warning(
                        f"Multiple blocking spawns with different join_turn_index: "
                        f"{join_turn_index} vs {spawn.join_turn_index}; using max()"
                    )
                    join_turn_index = max(join_turn_index, spawn.join_turn_index)

        if not resolved:
            return False

        # Create PendingSubagentJoin BEFORE dispatching children (prevents race)
        if total_blocking > 0 and join_turn_index is not None:
            self._pending_joins[parent_corr_id] = PendingSubagentJoin(
                parent_conversation_id=credit.conversation_id,
                parent_correlation_id=parent_corr_id,
                expected_count=total_blocking,
                join_turn_index=join_turn_index,
                parent_num_turns=credit.num_turns,
                parent_agent_depth=credit.agent_depth,
                parent_subagent_type=credit.subagent_type,
                parent_parent_correlation_id=credit.parent_correlation_id,
                created_at_ns=time.time_ns(),
            )
            self._stats.parents_suspended += 1

        # Dispatch children
        for is_blocking, child_sessions in resolved:
            for session in child_sessions:
                turn = session.build_first_turn(
                    agent_depth=child_depth,
                    parent_correlation_id=parent_corr_id,
                )
                if is_blocking:
                    self._scheduler.execute_async(
                        self._issue_child_credit_or_release(
                            turn, session.x_correlation_id
                        ),
                    )
                else:
                    self._dispatch(turn)
                if is_blocking:
                    self._child_to_parent[session.x_correlation_id] = parent_corr_id
                self._stats.children_spawned += 1

        return total_blocking > 0

    async def _issue_child_credit_or_release(
        self, turn: TurnToSend, corr_id: str
    ) -> None:
        """Issue a blocking child credit; release from join if it fails."""
        try:
            issued = await self._credit_issuer.issue_credit(turn)
        except Exception:
            self.warning(
                f"Exception issuing credit for child {corr_id}, releasing from join"
            )
            issued = False
        if not issued and corr_id in self._child_to_parent:
            self._stats.children_errored += 1
            join = self._release_child(corr_id)
            if join:
                self._dispatch(join)

    # =========================================================================
    # Internal: join accounting
    # =========================================================================

    def _is_background(self, conversation_id: str, spawn_id: str) -> bool:
        spawn = self._conversation_source.get_subagent_spawn(conversation_id, spawn_id)
        return spawn is not None and spawn.is_background

    def _release_child(self, child_corr_id: str) -> TurnToSend | None:
        """Pop child from tracking, maybe complete the parent's join."""
        parent_corr_id = self._child_to_parent.pop(child_corr_id, None)
        if parent_corr_id is None:
            return None
        return self._maybe_complete_join(parent_corr_id)

    def _maybe_complete_join(self, parent_corr_id: str) -> TurnToSend | None:
        """Increment completion; return join TurnToSend when all children done."""
        pending = self._pending_joins.get(parent_corr_id)
        if pending is None:
            return None

        pending.completed_count += 1
        if pending.completed_count < pending.expected_count:
            return None

        self._pending_joins.pop(parent_corr_id, None)
        self._stats.parents_resumed += 1

        if pending.join_turn_index >= pending.parent_num_turns:
            return None

        if not self._stop_checker.can_send_any_turn():
            self._stats.joins_suppressed += 1
            self.debug(
                lambda: f"Suppressed join for parent {parent_corr_id} "
                f"(stop fired, turn {pending.join_turn_index}/"
                f"{pending.parent_num_turns})"
            )
            return None

        return TurnToSend(
            conversation_id=pending.parent_conversation_id,
            x_correlation_id=parent_corr_id,
            turn_index=pending.join_turn_index,
            num_turns=pending.parent_num_turns,
            agent_depth=pending.parent_agent_depth,
            subagent_type=pending.parent_subagent_type,
            parent_correlation_id=pending.parent_parent_correlation_id,
        )
