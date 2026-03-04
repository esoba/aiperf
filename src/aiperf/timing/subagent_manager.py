# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SubagentSessionManager: intercepts spawn/join events for any timing strategy.

Wraps a TimingStrategyProtocol, intercepting handle_credit_return for subagent
spawn and join events. Everything else delegates to the inner strategy.

Paths in handle_credit_return:
  A) Child final turn with known parent -> join accounting, then delegate to inner
  A2) Child non-final turn -> dispatch next child turn (via hook or direct)
  A3) Child final turn (background/unknown parent) -> delegate to inner
  B) Next turn has subagent_spawn_ids -> fan out children
  C) Everything else -> delegate to inner strategy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.timing.strategies.core import (
    CancelledReturnObserverProtocol,
    ChildFirstTurnDispatchProtocol,
    ChildSessionObserverProtocol,
    ChildTurnDispatchProtocol,
    CleanableProtocol,
    RateSettableProtocol,
    RequestCompleteObserverProtocol,
)

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.common.models.dataset_models import SubagentSpawnInfo
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.credit.messages import CreditReturn
    from aiperf.timing.conversation_source import ConversationSource, SampledSession
    from aiperf.timing.strategies.core import TimingStrategyProtocol


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


@dataclass(slots=True)
class SubagentStats:
    """Counters for subagent observability."""

    children_spawned: int = 0
    children_completed: int = 0
    children_errored: int = 0
    parents_suspended: int = 0
    parents_resumed: int = 0


class SubagentSessionManager(AIPerfLoggerMixin):
    """Intercepts subagent spawn/join events, delegating everything else to the inner strategy."""

    def __init__(
        self,
        *,
        inner: TimingStrategyProtocol,
        conversation_source: ConversationSource,
        credit_issuer: CreditIssuer,
        scheduler: LoopScheduler,
    ) -> None:
        super().__init__(logger_name="SubagentManager")
        self._inner = inner
        self._conversation_source = conversation_source
        self._credit_issuer = credit_issuer
        self._scheduler = scheduler

        # Cache protocol checks once
        self._inner_has_child_observer = isinstance(inner, ChildSessionObserverProtocol)
        self._inner_has_child_dispatch = isinstance(inner, ChildTurnDispatchProtocol)
        self._inner_has_child_first_dispatch = isinstance(
            inner, ChildFirstTurnDispatchProtocol
        )
        self._inner_has_request_complete = isinstance(
            inner, RequestCompleteObserverProtocol
        )
        self._inner_has_cancelled_return = isinstance(
            inner, CancelledReturnObserverProtocol
        )
        self._inner_has_rate = isinstance(inner, RateSettableProtocol)
        self._inner_has_cleanup = isinstance(inner, CleanableProtocol)

        self._pending_subagent_joins: dict[str, PendingSubagentJoin] = {}
        self._subagent_child_to_parent: dict[str, str] = {}
        self._stats = SubagentStats()

    # =========================================================================
    # Lifecycle delegation
    # =========================================================================

    async def setup_phase(self) -> None:
        """Delegate to inner strategy."""
        await self._inner.setup_phase()

    async def execute_phase(self) -> None:
        """Delegate to inner strategy."""
        await self._inner.execute_phase()

    # =========================================================================
    # Explicit delegation methods (replaces __getattr__ proxy)
    # =========================================================================

    def on_ttft_sample(self, ttft_ns: int, **kwargs) -> None:
        """Forward TTFT sample to inner strategy if supported."""
        if hasattr(self._inner, "on_ttft_sample"):
            self._inner.on_ttft_sample(ttft_ns, **kwargs)

    def set_request_rate(self, rate: float) -> None:
        """Forward rate adjustment to inner strategy if supported."""
        if self._inner_has_rate:
            self._inner.set_request_rate(rate)

    def on_child_session_started(
        self, corr_id: str, depth: int, parent_corr_id: str
    ) -> None:
        """Forward child session hook to inner strategy if supported."""
        if self._inner_has_child_observer:
            self._inner.on_child_session_started(corr_id, depth, parent_corr_id)

    def on_request_complete(self, credit_return: CreditReturn) -> None:
        """Handle completed request: propagate child errors, then delegate."""
        credit = credit_return.credit
        if (
            credit_return.error
            and credit.agent_depth > 0
            and credit.x_correlation_id in self._subagent_child_to_parent
            and not credit.is_final_turn
        ):
            self._stats.children_errored += 1
            self._handle_subagent_child_complete(credit)

        if self._inner_has_request_complete:
            self._inner.on_request_complete(credit_return)

    def on_cancelled_return(self, credit: Credit) -> None:
        """Handle cancelled credit: treat as implicit child completion if tracked."""
        if (
            credit.agent_depth > 0
            and not credit.is_final_turn
            and credit.x_correlation_id in self._subagent_child_to_parent
        ):
            self._stats.children_errored += 1
            self._handle_subagent_child_complete(credit)

        if self._inner_has_cancelled_return:
            self._inner.on_cancelled_return(credit)

    def __getattr__(self, name: str):
        """Fallback proxy for forward compatibility with new inner strategy hooks."""
        self.debug(lambda: f"Proxying unknown attribute '{name}' to inner strategy")
        return getattr(self._inner, name)

    # =========================================================================
    # Subagent metrics
    # =========================================================================

    def get_subagent_stats(self) -> dict[str, int]:
        """Return subagent counters for phase stats."""
        s = self._stats
        return {
            "subagent_children_spawned": s.children_spawned,
            "subagent_children_completed": s.children_completed,
            "subagent_children_errored": s.children_errored,
            "subagent_parents_suspended": s.parents_suspended,
            "subagent_parents_resumed": s.parents_resumed,
        }

    # =========================================================================
    # Phase-end cleanup
    # =========================================================================

    def cleanup(self) -> None:
        """Log and clear leaked tracking state at phase end, then delegate to inner."""
        leaked_joins = len(self._pending_subagent_joins)
        leaked_children = len(self._subagent_child_to_parent)
        if leaked_joins > 0 or leaked_children > 0:
            self.warning(
                f"Phase-end cleanup: {leaked_joins} pending joins, "
                f"{leaked_children} tracked children still in flight"
            )
        self._pending_subagent_joins.clear()
        self._subagent_child_to_parent.clear()
        if self._inner_has_cleanup:
            self._inner.cleanup()

    # =========================================================================
    # Core dispatch logic
    # =========================================================================

    async def handle_credit_return(self, credit: Credit) -> None:
        """Intercept subagent spawn/join events, delegate the rest.

        Path A: blocking child final turn with known parent -> join accounting
        Path A2: any child non-final -> dispatch next child turn
        Path A3: any child final (background/unknown) -> delegate to inner
        Path B: next turn has subagent_spawn_ids -> fan out children
        Path C: everything else -> delegate to inner
        """
        # Handle all child credits (agent_depth > 0) centrally
        if credit.agent_depth > 0:
            if credit.is_final_turn:
                if credit.x_correlation_id in self._subagent_child_to_parent:
                    # Path A: blocking child final -> join accounting
                    self._stats.children_completed += 1
                    self._handle_subagent_child_complete(credit)
                # Path A3: delegate to inner for cleanup (all final children)
                await self._inner.handle_credit_return(credit)
            else:
                # Path A2: child non-final -> dispatch next turn
                turn = TurnToSend.from_previous_credit(credit)
                if self._inner_has_child_dispatch:
                    self._inner.dispatch_child_turn(credit, turn)
                else:
                    self._scheduler.execute_async(
                        self._credit_issuer.issue_credit(turn),
                    )
            return

        # Path B0: Turn 0 just completed -- check turn 0's own spawn_ids.
        # execute_phase fires turn 0 directly without checking its spawns,
        # so we dispatch them here on first credit return.
        if credit.turn_index == 0:
            turn0_meta = self._conversation_source.get_turn_metadata_at(
                credit.conversation_id, 0
            )
            if turn0_meta.subagent_spawn_ids:
                self._dispatch_subagent_spawns(credit, turn0_meta.subagent_spawn_ids)

        # Path B: Next turn has subagent spawn(s) -> fan out children
        if not credit.is_final_turn:
            next_meta = self._conversation_source.get_next_turn_metadata(credit)
            if next_meta.subagent_spawn_ids:
                parent_suspended = self._dispatch_subagent_spawns(
                    credit, next_meta.subagent_spawn_ids
                )
                if parent_suspended:
                    return

        # Path C: Normal turn (or all-background spawn) -- delegate to inner
        await self._inner.handle_credit_return(credit)

    def _dispatch_subagent_spawns(self, credit: Credit, spawn_ids: list[str]) -> bool:
        """Fan out subagent child sessions and optionally suspend the parent.

        Two-pass approach:
        - Pass 1: Resolve spawns, count blocking children, find join_turn_index
        - Create PendingSubagentJoin BEFORE scheduling any children (prevents race)
        - Pass 2: Dispatch child credits

        Returns True if the parent is suspended (blocking children registered),
        False if all children are background and the parent should continue.
        """
        parent_corr_id = credit.x_correlation_id
        child_depth = credit.agent_depth + 1

        # === Pass 1: Resolve and count ===
        resolved_spawns: list[tuple[SubagentSpawnInfo, bool, list[SampledSession]]] = []
        total_blocking_children = 0
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

            resolved_spawns.append((spawn, is_blocking, child_sessions))

            if is_blocking:
                total_blocking_children += len(spawn.child_conversation_ids)
                if join_turn_index is None:
                    join_turn_index = spawn.join_turn_index
                elif spawn.join_turn_index != join_turn_index:
                    self.warning(
                        f"Multiple blocking spawns with different join_turn_index: "
                        f"{join_turn_index} vs {spawn.join_turn_index}, using max()"
                    )
                    join_turn_index = max(join_turn_index, spawn.join_turn_index)

        if not resolved_spawns:
            return False

        # Create PendingSubagentJoin BEFORE dispatching children (prevents race)
        if total_blocking_children > 0 and join_turn_index is not None:
            self._pending_subagent_joins[parent_corr_id] = PendingSubagentJoin(
                parent_conversation_id=credit.conversation_id,
                parent_correlation_id=parent_corr_id,
                expected_count=total_blocking_children,
                join_turn_index=join_turn_index,
                parent_num_turns=credit.num_turns,
                parent_agent_depth=credit.agent_depth,
                parent_subagent_type=credit.subagent_type,
                parent_parent_correlation_id=credit.parent_correlation_id,
            )
            self._stats.parents_suspended += 1

        # === Pass 2: Dispatch children ===
        for _spawn, is_blocking, child_sessions in resolved_spawns:
            for child_session in child_sessions:
                if is_blocking:
                    self._subagent_child_to_parent[child_session.x_correlation_id] = (
                        parent_corr_id
                    )
                self.on_child_session_started(
                    child_session.x_correlation_id,
                    child_depth,
                    parent_corr_id,
                )
                self._stats.children_spawned += 1
                if self._inner_has_child_first_dispatch:
                    self._inner.dispatch_child_first_turn(
                        child_session, child_depth, parent_corr_id
                    )
                else:
                    child_turn = child_session.build_first_turn(
                        agent_depth=child_depth,
                        parent_correlation_id=parent_corr_id,
                    )
                    self._scheduler.execute_async(
                        self._credit_issuer.issue_credit(child_turn),
                    )

        return total_blocking_children > 0 and join_turn_index is not None

    def _handle_subagent_child_complete(self, credit: Credit) -> None:
        """Handle a subagent child's final turn. Dispatch parent join when all children complete."""
        child_corr_id = credit.x_correlation_id
        parent_corr_id = self._subagent_child_to_parent.pop(child_corr_id, None)
        if parent_corr_id is None:
            return

        pending = self._pending_subagent_joins.get(parent_corr_id)
        if pending is None:
            return

        pending.completed_count += 1
        if pending.completed_count < pending.expected_count:
            return

        self._pending_subagent_joins.pop(parent_corr_id, None)
        self._stats.parents_resumed += 1

        if pending.join_turn_index >= pending.parent_num_turns:
            return

        join_turn = TurnToSend(
            conversation_id=pending.parent_conversation_id,
            x_correlation_id=parent_corr_id,
            turn_index=pending.join_turn_index,
            num_turns=pending.parent_num_turns,
            agent_depth=pending.parent_agent_depth,
            subagent_type=pending.parent_subagent_type,
            parent_correlation_id=pending.parent_parent_correlation_id,
        )
        self._scheduler.execute_async(
            self._credit_issuer.issue_credit(join_turn),
        )
