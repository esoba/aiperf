# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic load generation strategy for trajectory benchmarking.

Simulates N concurrent users each working through a deterministic sequence of
multi-turn conversations (trajectories) with zero inter-turn delay. Unlike
rate-based strategies, concurrency is the only throttle.

Key characteristics:
- Pre-assigned trajectory sets: Each user gets all conversations from the dataset,
  starting at a deterministic random offset.
- Zero inter-turn delay: Next turn fires immediately on completion.
- Deterministic assignment: Same user->trajectory mapping regardless of concurrency.
- Random start indices: Prevents ISL bias by distributing users across positions.
- Wrap-around: Users loop back to the start when they finish all conversations.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.common import random_generator as rng
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.credit.structs import Credit, TurnToSend

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.stop_conditions import StopConditionChecker


@dataclass(slots=True)
class AgenticUser:
    """Per-user state for agentic load generation.

    Attributes:
        user_id: Unique identifier for this user.
        trajectory_index: Current position in the conversation_ids list.
        x_correlation_id: Current session's correlation ID (None = needs new session).
    """

    user_id: int
    trajectory_index: int
    x_correlation_id: str | None = None

    def advance_trajectory(self, num_conversations: int) -> None:
        """Move to next conversation in the trajectory, wrapping around."""
        self.trajectory_index = (self.trajectory_index + 1) % num_conversations
        self.x_correlation_id = None


class AgenticLoadStrategy(AIPerfLoggerMixin):
    """Agentic load generation strategy: N users, zero inter-turn delay, wrap-around."""

    def __init__(
        self,
        *,
        config: CreditPhaseConfig,
        conversation_source: ConversationSource,
        scheduler: LoopScheduler,
        stop_checker: StopConditionChecker,
        credit_issuer: CreditIssuer,
        lifecycle: PhaseLifecycle,
        **kwargs,
    ) -> None:
        super().__init__(logger_name="AgenticLoadTiming", **kwargs)
        self._config = config
        self._conversation_source = conversation_source
        self._stop_checker = stop_checker
        self._credit_issuer = credit_issuer
        self._lifecycle = lifecycle

        self._num_users = self._config.concurrency
        if self._num_users is None or self._num_users <= 0:
            raise ValueError(
                "concurrency must be set and positive for agentic load mode"
            )

        self._conversation_ids: list[str] = []
        self._users: dict[int, AgenticUser] = {}
        self._session_to_user: dict[str, AgenticUser] = {}

    async def setup_phase(self) -> None:
        """Pre-assign users with deterministic random trajectory offsets."""
        self._conversation_ids = [
            c.conversation_id
            for c in self._conversation_source.dataset_metadata.conversations
        ]
        num_conversations = len(self._conversation_ids)

        for i in range(self._num_users):
            # Per-user RNG derived from global seed ensures User i always gets
            # the same offset regardless of total user count.
            user_rng = rng.derive(f"timing.agentic_load.user.{i}")
            start_index = user_rng.randint(0, num_conversations - 1)
            self._users[i] = AgenticUser(
                user_id=i,
                trajectory_index=start_index,
            )

        self.info(
            f"Agentic load: {self._num_users} users, {num_conversations} conversations"
        )

    async def execute_phase(self) -> None:
        """Issue first turns for all users. Blocks on concurrency semaphore per user."""
        for user in self._users.values():
            first_turn = self._build_first_turn(user)
            should_continue = await self._credit_issuer.issue_credit(first_turn)
            if not should_continue:
                return

    async def handle_credit_return(self, credit: Credit) -> None:
        """Handle credit return: continue trajectory or start next one."""
        user = self._session_to_user.get(credit.x_correlation_id)
        if user is None:
            return

        if not credit.is_final_turn:
            next_turn = TurnToSend.from_previous_credit(credit)
            await self._credit_issuer.issue_credit(next_turn)
        else:
            self._start_next_trajectory(user, credit.x_correlation_id)
            first_turn = self._build_first_turn(user)
            await self._credit_issuer.issue_credit(first_turn)

    def _build_first_turn(self, user: AgenticUser) -> TurnToSend:
        """Build first turn for user's current trajectory and register session mapping."""
        conversation_id = self._conversation_ids[user.trajectory_index]
        metadata = self._conversation_source.get_metadata(conversation_id)
        x_correlation_id = str(uuid.uuid4())

        user.x_correlation_id = x_correlation_id
        self._session_to_user[x_correlation_id] = user

        return TurnToSend(
            conversation_id=conversation_id,
            x_correlation_id=x_correlation_id,
            turn_index=0,
            num_turns=len(metadata.turns),
        )

    def _start_next_trajectory(
        self, user: AgenticUser, old_correlation_id: str
    ) -> None:
        """Advance user to next conversation and clean up old session mapping."""
        self._session_to_user.pop(old_correlation_id, None)
        user.advance_trajectory(len(self._conversation_ids))
