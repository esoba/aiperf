# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SubagentOrchestrator: composable subagent spawn/join component.

Strategies own an instance and call ``intercept(credit)`` at the top of their
handle_credit_return. The orchestrator handles all child routing, spawn
resolution, child dispatch, and gated turn dispatch internally using a strategy-
provided dispatch callback. Strategies forward observer methods
(terminate_child, cleanup) as one-liners.

Prerequisite-Based Turn Gating
==============================

Turns declare explicit prerequisites (e.g. ``spawn_join``) that must be
satisfied before the turn dispatches. The orchestrator builds a prerequisite
index at init time from TurnPrerequisite entries on each turn.

Credit Return Flow for Subagents
================================

CreditCallbackHandler.on_credit_return processing order::

    1. Atomic counting (increment_returned)
    2. Track prefill release
    3. Release concurrency slots
    4. on_failed_credit → terminate_child  (errored/cancelled, ALL credits)
    5. Signal all_credits_returned_event
    6. handle_credit_return → intercept    (can_send OR depth > 0)

Step 4 runs BEFORE step 6 so that terminate_child marks the child in
_terminated_children before _handle_child_credit checks _is_terminated.
Step 6 uses a child bypass (depth > 0) so gate accounting always completes,
even after stop fires.

Stop Condition Interaction
==========================

Four coordinated guards achieve zero-overshoot, zero-deadlock:

1. **Callback handler child bypass** (CreditCallbackHandler step 6):
   Child returns (depth > 0) always reach handle_credit_return for gate
   accounting. Without this, child final-turn returns would be silently
   dropped when stop fires, leaving parent gates permanently unsatisfied.

2. **Non-final child dispatch suppression** (_handle_child_credit):
   Checks can_send_any_turn() before dispatching a child's next turn.
   This prevents the child bypass from causing unbounded child work.

3. **Gate dispatch suppression** (_release_blocked_gate):
   Checks can_send_any_turn() before dispatching the gated parent turn.

4. **Credit issuance failure** (_issue_child_credit_or_release):
   When issue_credit returns False (stop fired, no slots), the child is
   released from gate tracking to prevent the parent from hanging.

Additionally, issue_credit itself checks stop conditions at slot
acquisition time, providing a belt-and-suspenders final guard.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiperf.common.enums import PrerequisiteKind
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.dataset_models import TurnPrerequisite
from aiperf.credit.structs import TurnToSend

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.credit.structs import Credit
    from aiperf.timing.conversation_source import ConversationSource, SampledSession
    from aiperf.timing.phase.stop_conditions import StopConditionChecker


@dataclass(slots=True)
class PendingTurnGate:
    """Tracks prerequisite completion before dispatching a gated turn."""

    parent_conversation_id: str
    """Template conversation ID of the parent."""

    parent_correlation_id: str
    """Runtime correlation ID of the parent session."""

    gated_turn_index: int
    """Turn index that is blocked until prerequisites are met."""

    parent_num_turns: int = 0
    """Total turns in the parent conversation."""

    parent_agent_depth: int = 0
    """Nesting depth of the parent (0 = root)."""

    parent_parent_correlation_id: str | None = None
    """Correlation ID of the parent's parent (for nested subagents)."""

    created_at_ns: int = 0
    """Monotonic timestamp when the gate was created, for leak diagnostics."""

    is_blocked: bool = False
    """True once the parent has reached the gated turn and is waiting."""

    outstanding: dict[str, list[int]] = field(default_factory=dict)
    """Maps prereq_key -> [expected_count, completed_count]."""

    @property
    def is_satisfied(self) -> bool:
        """True when all prerequisites have been met."""
        return all(c >= e for e, c in self.outstanding.values())


@dataclass(slots=True)
class ChildGateEntry:
    """Tracks which parent gate a blocking child belongs to."""

    parent_corr_id: str
    """Runtime correlation ID of the parent session."""

    gated_turn_index: int
    """Turn index on the parent that this child gates."""

    prereq_key: str
    """Prerequisite key in the parent's PendingTurnGate.outstanding dict."""


@dataclass(slots=True)
class SubagentStats:
    """Counters for subagent observability."""

    children_spawned: int = 0
    """Total child sessions started across all spawns."""

    children_completed: int = 0
    """Children that finished their final turn normally."""

    children_errored: int = 0
    """Children released from gate due to error/cancel or issuance failure."""

    parents_suspended: int = 0
    """Parent sessions that blocked on a gated turn."""

    parents_resumed: int = 0
    """Parent sessions resumed after all prerequisites were met."""

    joins_suppressed: int = 0
    """Gated turns suppressed because stop condition fired."""


class SubagentOrchestrator(AIPerfLoggerMixin):
    """Composable subagent spawn/join component.

    Strategies call ``intercept(credit)`` from handle_credit_return. The
    orchestrator routes child credits, manages gate tracking, and dispatches
    gated parent turns via a strategy-provided callback.

    Integration: SubagentMixin wires _init_subagents, on_failed_credit, cleanup.
    See module docstring for credit return flow ordering and stop-condition guards.
    """

    def __init__(
        self,
        *,
        conversation_source: ConversationSource,
        credit_issuer: CreditIssuer,
        stop_checker: StopConditionChecker,
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
                for gated turns, child next turns, and background child first turns.
        """
        super().__init__(logger_name="SubagentOrchestrator")
        self._conversation_source = conversation_source
        self._credit_issuer = credit_issuer
        self._stop_checker = stop_checker
        self._scheduler = scheduler
        self._dispatch = dispatch_fn

        self._gated_turns: dict[str, PendingTurnGate] = {}
        self._future_gates: dict[str, dict[int, PendingTurnGate]] = {}
        self._child_to_gate: dict[str, ChildGateEntry] = {}
        self._terminated_children: set[str] = set()
        self._cleaning_up: bool = False
        self._stats = SubagentStats()

        self._prerequisite_index: dict[tuple[str, int], list[TurnPrerequisite]] = {}
        self._spawn_join_index: dict[tuple[str, str], int] = {}
        self._build_prerequisite_index()

    def set_dispatch(self, dispatch_fn: Callable[[TurnToSend], None]) -> None:
        """Set the strategy-provided dispatch callback."""
        self._dispatch = dispatch_fn

    # =========================================================================
    # Primary entry point: intercept credit returns
    # =========================================================================

    def intercept(self, credit: Credit) -> bool:
        """Handle subagent-related credit returns.

        Called at the top of every strategy's handle_credit_return.
        Returns True if the credit was fully handled (strategy must NOT
        dispatch next turn). Returns False for normal strategy dispatch.

        True only for: (a) child credits (depth > 0), (b) parent suspended
        at a gate. Returning True incorrectly drops the parent's next turn.
        """
        if self._cleaning_up:
            return False

        # Child credits are fully handled here. Strategy must not look up
        # child conversation_ids -- they're not in the strategy's schedule.
        if credit.agent_depth > 0:
            self._handle_child_credit(credit)
            return True

        # Parent turn completed -- check if it spawned children.
        spawn_ids = self._get_spawn_ids(credit)
        if spawn_ids:
            self.debug(
                lambda: f"Parent {credit.x_correlation_id} turn[{credit.turn_index}] "
                f"completed with spawns: {spawn_ids}"
            )
            # Turn-0 background spawns are pre-dispatched in execute_phase.
            # Filter them out here to avoid double-dispatch.
            if credit.turn_index == 0:
                spawn_ids = [
                    sid
                    for sid in spawn_ids
                    if not self._is_background(credit.conversation_id, sid)
                ]
            if spawn_ids:
                self._resolve_and_dispatch_spawns(credit, spawn_ids)

        # Suspend parent if its next turn has unsatisfied prerequisites.
        suspended = self._maybe_suspend_parent(credit)
        if suspended:
            self.debug(
                lambda: f"Parent {credit.x_correlation_id} suspended at "
                f"turn[{credit.turn_index + 1}] waiting for prerequisites"
            )
        return suspended

    # =========================================================================
    # Turn-0 background pre-dispatch (called from execute_phase)
    # =========================================================================

    def dispatch_turn0_background_spawns(self) -> None:
        """Pre-dispatch background children for turn 0 of all root conversations.

        Called from execute_phase after first-turn credits are scheduled.
        Coordinates with intercept() which filters out turn-0 background
        spawns to avoid double-dispatch. Blocking turn-0 spawns go through
        intercept() instead.
        """
        if self._dispatch is None:
            self.warning("dispatch_turn0_background_spawns called before set_dispatch")
            return
        total_dispatched = 0
        for conv in self._conversation_source.dataset_metadata.conversations:
            if conv.agent_depth > 0 or not conv.turns:
                continue
            bg_ids = self._get_turn0_background_ids(conv.conversation_id)
            if not bg_ids:
                continue
            # Synthetic correlation ID -- these children are not gated so
            # the parent_corr_id is only used for logging/tracing.
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
                    total_dispatched += 1
        if total_dispatched > 0:
            self.info(f"Pre-dispatched {total_dispatched} turn-0 background children")

    # =========================================================================
    # Strategy-facing API (called via SubagentMixin)
    # =========================================================================

    def terminate_child(self, credit: Credit) -> None:
        """Release an errored/cancelled child from gate tracking.

        Called from SubagentMixin.on_failed_credit (callback handler step 4).
        Must run before _handle_child_credit (step 6) -- see module docstring.

        Skips final turns (gate accounting lives in _handle_child_credit) and
        background children (not tracked in _child_to_gate).
        """
        if (
            self._cleaning_up
            or credit.agent_depth == 0
            or credit.is_final_turn
            or credit.x_correlation_id not in self._child_to_gate
        ):
            return
        self._stats.children_errored += 1
        self._terminated_children.add(credit.x_correlation_id)
        entry = self._child_to_gate.get(credit.x_correlation_id)
        self.debug(
            lambda: f"Terminating child {credit.x_correlation_id} "
            f"(parent={entry.parent_corr_id if entry else '?'}, "
            f"turn[{credit.turn_index}]/{credit.num_turns})"
        )
        gated = self._release_child(credit.x_correlation_id)
        if gated:
            self._dispatch(gated)

    def cleanup(self) -> None:
        """Log leaked state and clear all tracking. Idempotent."""
        if self._cleaning_up:
            return
        self._cleaning_up = True
        s = self._stats
        self.info(
            f"Subagent stats: spawned={s.children_spawned}, "
            f"completed={s.children_completed}, errored={s.children_errored}, "
            f"suspended={s.parents_suspended}, resumed={s.parents_resumed}, "
            f"suppressed={s.joins_suppressed}"
        )
        if self._gated_turns or self._future_gates or self._child_to_gate:
            self.warning(
                f"Leaked state at cleanup: {len(self._gated_turns)} active gates, "
                f"{sum(len(g) for g in self._future_gates.values())} future gates, "
                f"{len(self._child_to_gate)} tracked children"
            )
            now_ns = time.time_ns()
            leaked_gates = list(self._gated_turns.items())
            leaked_gates.extend(self._iter_future_gates())
            for parent_corr_id, gate in leaked_gates:
                age_ms = (now_ns - gate.created_at_ns) / 1_000_000
                total_expected = sum(e for e, _ in gate.outstanding.values())
                total_completed = sum(c for _, c in gate.outstanding.values())
                self.warning(
                    f"Abandoned pending gate for parent {parent_corr_id} "
                    f"(age={age_ms:.0f}ms, completed={total_completed}/"
                    f"{total_expected})"
                )
        self._gated_turns.clear()
        self._future_gates.clear()
        self._child_to_gate.clear()
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
    # Internal: prerequisite index
    # =========================================================================

    def _build_prerequisite_index(self) -> None:
        """Build (conv_id, turn_index) -> prerequisites and (conv_id, spawn_id) -> turn_index lookups."""
        for conv in self._conversation_source.dataset_metadata.conversations:
            for idx, turn_meta in enumerate(conv.turns):
                if turn_meta.prerequisites:
                    self._prerequisite_index[(conv.conversation_id, idx)] = list(
                        turn_meta.prerequisites
                    )
                    for prereq in turn_meta.prerequisites:
                        if (
                            prereq.kind == PrerequisiteKind.SPAWN_JOIN
                            and prereq.spawn_id
                        ):
                            self._spawn_join_index[
                                (conv.conversation_id, prereq.spawn_id)
                            ] = idx
        if self._prerequisite_index:
            self.info(
                f"Prerequisite index: {len(self._prerequisite_index)} gated turns, "
                f"{len(self._spawn_join_index)} spawn-join entries"
            )

    def _find_gated_turn_index(
        self, conversation_id: str, spawn_ids: list[str]
    ) -> int | None:
        """Find the turn index gated by the given spawn_ids via the spawn_join index."""
        for spawn_id in spawn_ids:
            turn_idx = self._spawn_join_index.get((conversation_id, spawn_id))
            if turn_idx is not None:
                return turn_idx
        return None

    def _iter_future_gates(self) -> list[tuple[str, PendingTurnGate]]:
        """Flatten future gates for cleanup/logging."""
        leaked: list[tuple[str, PendingTurnGate]] = []
        for parent_corr_id, gates in self._future_gates.items():
            for gate in gates.values():
                leaked.append((parent_corr_id, gate))
        return leaked

    def _get_gate(
        self, parent_corr_id: str, gated_turn_index: int
    ) -> PendingTurnGate | None:
        """Look up either an active blocked gate or a future gate."""
        active_gate = self._gated_turns.get(parent_corr_id)
        if active_gate is not None and active_gate.gated_turn_index == gated_turn_index:
            return active_gate
        return self._future_gates.get(parent_corr_id, {}).get(gated_turn_index)

    def _add_future_gate(
        self,
        *,
        parent_corr_id: str,
        gated_turn_index: int,
        credit: Credit,
        prereq_key: str,
        expected_children: int,
    ) -> None:
        """Create or extend a future gate for a later parent turn."""
        gates_for_parent = self._future_gates.setdefault(parent_corr_id, {})
        gate = gates_for_parent.get(gated_turn_index)
        if gate is None:
            gate = PendingTurnGate(
                parent_conversation_id=credit.conversation_id,
                parent_correlation_id=parent_corr_id,
                gated_turn_index=gated_turn_index,
                parent_num_turns=credit.num_turns,
                parent_agent_depth=credit.agent_depth,
                parent_parent_correlation_id=credit.parent_correlation_id,
                created_at_ns=time.time_ns(),
            )
            gates_for_parent[gated_turn_index] = gate
        gate.outstanding[prereq_key] = [expected_children, 0]

    def _pop_future_gate(
        self, parent_corr_id: str, gated_turn_index: int
    ) -> PendingTurnGate | None:
        """Remove and return a future gate."""
        gates_for_parent = self._future_gates.get(parent_corr_id)
        if gates_for_parent is None:
            return None
        gate = gates_for_parent.pop(gated_turn_index, None)
        if not gates_for_parent:
            self._future_gates.pop(parent_corr_id, None)
        return gate

    def _maybe_suspend_parent(self, credit: Credit) -> bool:
        """Suspend parent if its next turn has unsatisfied prerequisites.

        Returns True to block the parent (strategy won't dispatch next turn).
        Promotes a "future gate" (created by _resolve_and_dispatch_spawns) to
        an active blocked gate in _gated_turns.

        RACE: Children can complete before the parent reaches the gate. If the
        future gate is already satisfied, we pop it and return False (no block).
        """
        next_turn_index = credit.turn_index + 1

        # Already blocked (multiple spawns targeting same gated turn).
        active_gate = self._gated_turns.get(credit.x_correlation_id)
        if (
            active_gate is not None
            and active_gate.gated_turn_index == next_turn_index
            and not active_gate.is_satisfied
        ):
            return True

        # Promote future gate to active.
        future_gate = self._pop_future_gate(credit.x_correlation_id, next_turn_index)
        if future_gate is None:
            return False

        # Children already finished -- no need to block.
        if future_gate.is_satisfied:
            return False

        future_gate.is_blocked = True
        self._gated_turns[credit.x_correlation_id] = future_gate
        self._stats.parents_suspended += 1
        return True

    def _satisfy_prerequisite(
        self, parent_corr_id: str, gated_turn_index: int, prereq_key: str
    ) -> TurnToSend | None:
        """Increment completion for a prerequisite; dispatch gated turn when all met.

        Returns TurnToSend only when the gate is fully satisfied AND the parent
        is already blocked. If satisfied before the parent arrives, pops the
        future gate so _maybe_suspend_parent sees no gate and skips suspension.
        """
        gate = self._get_gate(parent_corr_id, gated_turn_index)
        if gate is None:
            return None

        if prereq_key not in gate.outstanding:
            return None

        gate.outstanding[prereq_key][1] += 1
        if not gate.is_satisfied:
            return None

        # Satisfied before parent arrived -- clean up future gate.
        if not gate.is_blocked:
            self._pop_future_gate(parent_corr_id, gated_turn_index)
            return None

        return self._release_blocked_gate(parent_corr_id)

    def _release_blocked_gate(self, parent_corr_id: str) -> TurnToSend | None:
        """Release a blocked parent gate and build its gated turn for dispatch.

        Checks can_send_any_turn() to suppress dispatch after stop fires
        (one of the three stop-condition guards; see module docstring).
        """
        gate = self._gated_turns.pop(parent_corr_id, None)
        if gate is None:
            return None

        self._stats.parents_resumed += 1
        if gate.gated_turn_index >= gate.parent_num_turns:
            return None

        if not self._stop_checker.can_send_any_turn():
            self._stats.joins_suppressed += 1
            self.debug(
                lambda: f"Suppressed gated turn for parent {parent_corr_id} "
                f"(stop fired, turn {gate.gated_turn_index}/"
                f"{gate.parent_num_turns})"
            )
            return None

        return TurnToSend(
            conversation_id=gate.parent_conversation_id,
            x_correlation_id=parent_corr_id,
            turn_index=gate.gated_turn_index,
            num_turns=gate.parent_num_turns,
            agent_depth=gate.parent_agent_depth,
            parent_correlation_id=gate.parent_parent_correlation_id,
        )

    # =========================================================================
    # Internal: child credit handling
    # =========================================================================

    def _handle_child_credit(self, credit: Credit) -> None:
        """Route a child credit: gate accounting for final, next turn for non-final.

        Reached even after stop fires (callback handler child bypass), so both
        paths check stop conditions before dispatching new work.
        """
        if credit.is_final_turn:
            # Gate accounting for blocking children only. Background children
            # (not in _child_to_gate) are fire-and-forget.
            if credit.x_correlation_id in self._child_to_gate:
                self._stats.children_completed += 1
                entry = self._child_to_gate.get(credit.x_correlation_id)
                self.debug(
                    lambda: f"Child {credit.x_correlation_id} completed final turn "
                    f"(parent={entry.parent_corr_id if entry else '?'})"
                )
                gated = self._release_child(credit.x_correlation_id)
                if gated:
                    self.debug(
                        lambda: f"Gate satisfied, resuming parent "
                        f"{gated.x_correlation_id} at turn[{gated.turn_index}]"
                    )
                    self._dispatch(gated)
        else:
            # Consume terminated marker (set by terminate_child in step 4)
            # to prevent dispatching next turn for errored children.
            if self._is_terminated(credit):
                return
            if not self._stop_checker.can_send_any_turn():
                return
            turn = TurnToSend.from_previous_credit(credit)
            self._dispatch(turn)

    def _is_terminated(self, credit: Credit) -> bool:
        """Check and consume terminated marker for a non-final child credit.

        Consume-on-read: child turns are sequential (one in-flight at a time),
        so the marker only needs to block one return.
        """
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
    ) -> None:
        """Resolve spawns, register gate tracking, and dispatch children.

        Two-phase: register all gates THEN dispatch. No awaits between phases,
        so _child_to_gate entries exist before any child credit can return.
        """
        parent_corr_id = credit.x_correlation_id
        child_depth = credit.agent_depth + 1

        resolved: list[tuple[bool, str, list[SampledSession], int | None]] = []

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
            gated_turn_index = None
            if is_blocking:
                gated_turn_index = self._find_gated_turn_index(
                    credit.conversation_id, [spawn_id]
                )
                # Sanity check: gate must be on a future turn. A gate on
                # a past/current turn means the dataset is malformed.
                if (
                    gated_turn_index is not None
                    and gated_turn_index <= credit.turn_index
                ):
                    self.warning(
                        f"Ignoring spawn gate on past turn {gated_turn_index} for "
                        f"parent {parent_corr_id} spawn {spawn_id}"
                    )
                    gated_turn_index = None
            resolved.append((is_blocking, spawn_id, child_sessions, gated_turn_index))

        if not resolved:
            return

        if self.is_debug_enabled:
            n_blocking = sum(1 for b, _, _, _ in resolved if b)
            n_background = len(resolved) - n_blocking
            self.debug(
                f"Resolved spawns for parent {parent_corr_id}: "
                f"{n_blocking} blocking, {n_background} background"
            )

        # PHASE 1: Register all gate tracking before dispatching any children.
        for is_blocking, spawn_id, child_sessions, gated_turn_index in resolved:
            if not is_blocking:
                continue
            if gated_turn_index is None:
                # Blocking spawn with no matching prerequisite in the dataset.
                # The children will run but the parent won't wait for them.
                self.warning(
                    f"Blocking spawn {spawn_id} on parent {parent_corr_id} has no "
                    "matching prerequisite; parent will not be gated"
                )
                continue
            prereq_key = f"{PrerequisiteKind.SPAWN_JOIN}:{spawn_id}"
            self._add_future_gate(
                parent_corr_id=parent_corr_id,
                gated_turn_index=gated_turn_index,
                credit=credit,
                prereq_key=prereq_key,
                expected_children=len(child_sessions),
            )

        # PHASE 2: Dispatch children.
        for is_blocking, spawn_id, child_sessions, gated_turn_index in resolved:
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
                if is_blocking and gated_turn_index is not None:
                    self._child_to_gate[session.x_correlation_id] = ChildGateEntry(
                        parent_corr_id=parent_corr_id,
                        gated_turn_index=gated_turn_index,
                        prereq_key=f"spawn_join:{spawn_id}",
                    )
                self._stats.children_spawned += 1

    async def _issue_child_credit_or_release(
        self, turn: TurnToSend, corr_id: str
    ) -> None:
        """Issue a blocking child credit; release from gate if issuance fails.

        When issue_credit returns False (stop fired, no slots), releases the
        child from gate tracking so the parent doesn't hang indefinitely.
        """
        try:
            issued = await self._credit_issuer.issue_credit(turn)
        except Exception:
            self.warning(
                f"Exception issuing credit for child {corr_id}, releasing from gate"
            )
            issued = False
        if not issued and corr_id in self._child_to_gate:
            self._stats.children_errored += 1
            gated = self._release_child(corr_id)
            if gated:
                self._dispatch(gated)

    # =========================================================================
    # Internal: gate accounting
    # =========================================================================

    def _is_background(self, conversation_id: str, spawn_id: str) -> bool:
        spawn = self._conversation_source.get_subagent_spawn(conversation_id, spawn_id)
        return spawn is not None and spawn.is_background

    def _release_child(self, child_corr_id: str) -> TurnToSend | None:
        """Pop child from gate tracking and satisfy its parent's prerequisite.

        Returns TurnToSend if all prerequisites met and parent is blocked.
        Safe to call twice (pop returns None on second call).
        """
        entry = self._child_to_gate.pop(child_corr_id, None)
        if entry is None:
            return None
        return self._satisfy_prerequisite(
            entry.parent_corr_id, entry.gated_turn_index, entry.prereq_key
        )
