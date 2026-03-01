# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop agentic load strategy with deterministic trajectory assignment.

- Pre-assigns non-overlapping conversations to users (deterministic via seed)
- Spawns users at a configurable rate (staggered ramp-up)
- Each user runs in a closed loop: wait for response, then immediately send next turn
- Users loop through their assigned conversations until the phase ends
- Optional ISL offset skips N turns in the first conversation to prevent bias

Phase Timeline:
    |--- ramp-up ---|--- settling ---|--- measurement window ---|
    ^               ^                ^                          ^
  phase start   all spawned    measure start              phase end
"""

from __future__ import annotations

import hashlib
import random
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    """Per-user state for agentic load mode.

    Each user has a fixed set of assigned conversation IDs and cycles
    through them in a closed loop until the phase ends.
    """

    user_id: int
    assigned_conversation_ids: list[str]
    current_trajectory_index: int = 0
    pass_count: int = 0
    isl_offset: int = 0
    isl_offset_applied: bool = False


@dataclass(slots=True)
class _ActiveSession:
    """Tracks an active session (x_correlation_id -> user mapping)."""

    user: AgenticUser
    conversation_id: str


def _assign_conversations(
    conversation_ids: list[str],
    num_users: int,
    per_user: int,
    seed: int | None,
) -> dict[int, list[str]]:
    """Assign conversations to users with non-overlapping indices where possible.

    Shuffles conversation IDs deterministically, then assigns per_user
    conversations to each user in round-robin fashion.
    """
    ids = list(conversation_ids)
    if seed is not None:
        random.Random(seed).shuffle(ids)

    assignments: dict[int, list[str]] = {}
    for user_id in range(num_users):
        start_idx = (user_id * per_user) % len(ids)
        user_convs: list[str] = []
        for i in range(per_user):
            idx = (start_idx + i) % len(ids)
            user_convs.append(ids[idx])
        assignments[user_id] = user_convs
    return assignments


def _cache_bust_suffix(
    benchmark_id: str, pass_count: int, user_id: int, trajectory_index: int
) -> str:
    """Generate a cache bust suffix for the system prompt.

    The suffix is the same for all turns within a trajectory (preserving
    intra-trajectory KV cache) but changes on trajectory/pass boundaries.
    benchmark_id ensures uniqueness across separate benchmark runs.
    """
    unique_str = f"{benchmark_id}:{pass_count}:{user_id}:{trajectory_index}"
    hash_digest = hashlib.sha256(unique_str.encode()).hexdigest()[:12]
    return f"\n\n[rid:{hash_digest}]"


class AgenticLoadStrategy(AIPerfLoggerMixin):
    """Closed-loop agentic load strategy with deterministic trajectory assignment."""

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
    ):
        super().__init__(logger_name="AgenticLoadTiming", **kwargs)
        self._config = config
        self._conversation_source = conversation_source
        self._scheduler = scheduler
        self._stop_checker = stop_checker
        self._credit_issuer = credit_issuer
        self._lifecycle = lifecycle

        self._num_users = config.num_users
        self._user_spawn_rate = config.user_spawn_rate or 1.0
        self._settling_time = config.settling_time_sec or 0.0
        self._trajectories_per_user = config.trajectories_per_user or 20
        self._max_isl_offset = config.max_isl_offset or 0
        self._seed = config.agentic_seed
        self._benchmark_id = config.benchmark_id or "unknown"

        if self._num_users is None or self._num_users <= 0:
            raise ValueError("num_users must be set and positive for agentic load mode")

        self._users: dict[int, AgenticUser] = {}
        self._active_sessions: dict[str, _ActiveSession] = {}

    async def setup_phase(self) -> None:
        """Pre-assign conversations to users deterministically."""
        conversations = self._conversation_source.dataset_metadata.conversations
        conversation_ids = [
            c.conversation_id for c in conversations if c.agent_depth == 0
        ]

        if not conversation_ids:
            raise ValueError("No conversations available for agentic load mode")

        assignments = _assign_conversations(
            conversation_ids,
            self._num_users,
            self._trajectories_per_user,
            self._seed,
        )

        for user_id, conv_ids in assignments.items():
            if self._max_isl_offset > 0 and self._seed is not None:
                # Per-user deterministic seed: user N always gets the same offset
                # regardless of num_users (matches reference implementation)
                isl_offset = random.Random(self._seed + user_id).randint(
                    0, self._max_isl_offset
                )
            elif self._max_isl_offset > 0:
                isl_offset = random.Random().randint(0, self._max_isl_offset)
            else:
                isl_offset = 0
            user = AgenticUser(
                user_id=user_id,
                assigned_conversation_ids=conv_ids,
                isl_offset=isl_offset,
            )
            self._users[user_id] = user

        self.info(
            f"Agentic load: {self._num_users} users, "
            f"{self._trajectories_per_user} trajectories/user, "
            f"spawn_rate={self._user_spawn_rate}/s, "
            f"settling={self._settling_time}s, "
            f"max_isl_offset={self._max_isl_offset}"
        )

    async def execute_phase(self) -> None:
        """Schedule first-turn credits for each user at staggered intervals.

        Timeline:
            |--- ramp-up ---|--- settling ---|--- measurement window ---|
            ^               ^                ^                          ^
          phase start   all spawned    settle end               phase end

        Ramp-up duration = (num_users - 1) / user_spawn_rate
        """
        if self._lifecycle.started_at_perf_ns is None:
            raise RuntimeError("started_at_perf_ns is not set in the lifecycle")

        phase_start = self._lifecycle.started_at_perf_sec
        ramp_duration = (
            (self._num_users - 1) / self._user_spawn_rate
            if self._num_users > 1
            else 0.0
        )

        self.info(
            f"Ramp-up: {ramp_duration:.1f}s, "
            f"settling: {self._settling_time:.1f}s, "
            f"measurement starts at: +{ramp_duration + self._settling_time:.1f}s"
        )

        for user_id, user in self._users.items():
            spawn_delay = user_id / self._user_spawn_rate
            spawn_time = phase_start + spawn_delay
            self._scheduler.schedule_at_perf_sec(
                spawn_time,
                self._spawn_user(user),
            )

        # Block until stop condition is met (duration-based)
        while self._stop_checker.can_send_any_turn():
            await self._scheduler.sleep(1.0)

    async def _spawn_user(self, user: AgenticUser) -> None:
        """Start a user's first conversation."""
        turn = self._build_first_turn_for_user(user)
        if turn is not None:
            await self._credit_issuer.issue_credit(turn)

    def _build_first_turn_for_user(self, user: AgenticUser) -> TurnToSend | None:
        """Build the first turn for a user's current trajectory."""
        if not user.assigned_conversation_ids:
            return None

        conv_id = user.assigned_conversation_ids[user.current_trajectory_index]
        metadata = self._conversation_source.get_metadata(conv_id)
        num_turns = len(metadata.turns)

        # Apply ISL offset on first trajectory of first pass only
        start_turn = 0
        if not user.isl_offset_applied:
            user.isl_offset_applied = True
            start_turn = min(user.isl_offset, num_turns - 1)

        x_correlation_id = str(uuid.uuid4())

        session = _ActiveSession(user=user, conversation_id=conv_id)
        self._active_sessions[x_correlation_id] = session

        # Cache bust suffix: same for all turns in a trajectory, changes
        # on trajectory/pass boundaries to prevent KV cache reuse across passes
        suffix = _cache_bust_suffix(
            self._benchmark_id,
            user.pass_count,
            user.user_id,
            user.current_trajectory_index,
        )

        return TurnToSend(
            conversation_id=conv_id,
            x_correlation_id=x_correlation_id,
            turn_index=start_turn,
            num_turns=num_turns,
            system_prompt_suffix=suffix,
        )

    async def handle_credit_return(self, credit: Credit) -> None:
        """Handle credit return: dispatch next turn or advance trajectory.

        - Not final turn: issue next turn immediately (closed-loop)
        - Final turn: advance to next conversation, issue its first turn
        - All conversations exhausted: loop back (increment pass_count)
        """
        session = self._active_sessions.get(credit.x_correlation_id)
        if session is None:
            return

        if not credit.is_final_turn:
            # Closed-loop: issue next turn immediately
            # system_prompt_suffix propagates via from_previous_credit
            turn = TurnToSend.from_previous_credit(credit)
            await self._credit_issuer.issue_credit(turn)
            return

        # Final turn: clean up current session and advance trajectory
        self._active_sessions.pop(credit.x_correlation_id, None)
        user = session.user

        if not self._stop_checker.can_send_any_turn():
            return

        # Advance to next trajectory
        user.current_trajectory_index += 1
        if user.current_trajectory_index >= len(user.assigned_conversation_ids):
            user.current_trajectory_index = 0
            user.pass_count += 1

        # Start new conversation (with new cache bust suffix)
        turn = self._build_first_turn_for_user(user)
        if turn is not None:
            await self._credit_issuer.issue_credit(turn)
