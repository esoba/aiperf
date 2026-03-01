# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SubagentSessionManager: intercepts spawn/join events for any timing strategy.

Wraps a TimingStrategyProtocol, intercepting handle_credit_return for subagent
spawn and join events. Everything else delegates to the inner strategy.

Three paths in handle_credit_return:
  A) Child final turn with known parent -> join accounting, then delegate to inner
  B) Next turn has subagent_spawn_ids -> fan out children, suspend/continue parent
  C) Everything else -> delegate to inner strategy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.credit.structs import Credit, TurnToSend

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.timing.conversation_source import ConversationSource
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

        self._pending_subagent_joins: dict[str, PendingSubagentJoin] = {}
        self._subagent_child_to_parent: dict[str, str] = {}

    async def setup_phase(self) -> None:
        """Delegate to inner strategy."""
        await self._inner.setup_phase()

    async def execute_phase(self) -> None:
        """Delegate to inner strategy."""
        await self._inner.execute_phase()

    def __getattr__(self, name: str):
        """Proxy attribute access to inner strategy for protocol compliance."""
        return getattr(self._inner, name)

    async def handle_credit_return(self, credit: Credit) -> None:
        """Intercept subagent spawn/join events, delegate the rest.

        Path A: child final turn with known parent -> join accounting, then delegate
        Path B: next turn has subagent_spawn_ids -> fan out children
        Path C: everything else -> delegate to inner
        """
        # Path A: Subagent child's final turn -- signal parent join
        if (
            credit.is_final_turn
            and credit.x_correlation_id in self._subagent_child_to_parent
        ):
            self._handle_subagent_child_complete(credit)
            await self._inner.handle_credit_return(credit)
            return

        # Path B: Next turn is blocked by subagent spawn(s)
        if not credit.is_final_turn:
            next_meta = self._conversation_source.get_next_turn_metadata(credit)
            if next_meta.subagent_spawn_ids:
                self._dispatch_subagent_spawns(credit, next_meta.subagent_spawn_ids)
                return

        # Path C: Normal turn -- delegate to inner
        await self._inner.handle_credit_return(credit)

    def _dispatch_subagent_spawns(self, credit: Credit, spawn_ids: list[str]) -> None:
        """Fan out subagent child sessions for all spawn_ids and register a pending join."""
        parent_corr_id = credit.x_correlation_id
        child_depth = credit.agent_depth + 1

        total_blocking_children = 0
        any_blocking = False
        all_background = True
        join_turn_index: int | None = None

        for spawn_id in spawn_ids:
            spawn = self._conversation_source.get_subagent_spawn(
                credit.conversation_id, spawn_id
            )
            if spawn is None:
                continue

            if join_turn_index is None:
                join_turn_index = spawn.join_turn_index

            is_blocking = not spawn.is_background

            for child_conv_id in spawn.child_conversation_ids:
                child_session = self._conversation_source.start_child_session(
                    child_conv_id
                )
                # Only track blocking children for join accounting.
                # Background children complete via Path C (delegate to inner).
                if is_blocking:
                    self._subagent_child_to_parent[child_session.x_correlation_id] = (
                        parent_corr_id
                    )
                if hasattr(self._inner, "on_child_session_started"):
                    self._inner.on_child_session_started(
                        child_session.x_correlation_id, child_depth
                    )
                child_turn = child_session.build_first_turn(agent_depth=child_depth)
                self._scheduler.execute_async(
                    self._credit_issuer.issue_credit(child_turn),
                )

            if is_blocking:
                any_blocking = True
                all_background = False
                total_blocking_children += len(spawn.child_conversation_ids)

        if join_turn_index is None:
            turn = TurnToSend.from_previous_credit(credit)
            self._scheduler.execute_async(
                self._credit_issuer.issue_credit(turn),
            )
            return

        if any_blocking:
            self._pending_subagent_joins[parent_corr_id] = PendingSubagentJoin(
                parent_conversation_id=credit.conversation_id,
                parent_correlation_id=parent_corr_id,
                expected_count=total_blocking_children,
                join_turn_index=join_turn_index,
                parent_num_turns=credit.num_turns,
                parent_agent_depth=credit.agent_depth,
            )
        elif all_background:
            join_turn = TurnToSend(
                conversation_id=credit.conversation_id,
                x_correlation_id=parent_corr_id,
                turn_index=join_turn_index,
                num_turns=credit.num_turns,
                agent_depth=credit.agent_depth,
            )
            self._scheduler.execute_async(
                self._credit_issuer.issue_credit(join_turn),
            )

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

        if pending.join_turn_index >= pending.parent_num_turns:
            return

        join_turn = TurnToSend(
            conversation_id=pending.parent_conversation_id,
            x_correlation_id=parent_corr_id,
            turn_index=pending.join_turn_index,
            num_turns=pending.parent_num_turns,
            agent_depth=pending.parent_agent_depth,
        )
        self._scheduler.execute_async(
            self._credit_issuer.issue_credit(join_turn),
        )
