# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed schedule timing strategy for trace replay.

Replays conversation traces at precise timestamps from dataset metadata.
First turns sent by main loop at absolute timestamps, subsequent turns
dispatched using delay_ms or timestamp_ms from metadata.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, NamedTuple

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.credit.structs import Credit, TurnToSend

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.credit.messages import CreditReturn
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.stop_conditions import StopConditionChecker
    from aiperf.timing.subagent_orchestrator import SubagentOrchestrator


class ScheduleEntry(NamedTuple):
    """A single entry in the fixed schedule."""

    timestamp_ms: int | float
    turn: TurnToSend


class FixedScheduleStrategy(AIPerfLoggerMixin):
    """Timing strategy for replaying conversation traces with absolute timestamps.

    Sends first turns at precise timestamps from conversation metadata.
    Subsequent turns dispatched immediately or after calculated delay.
    """

    def __init__(
        self,
        *,
        config: CreditPhaseConfig,
        conversation_source: ConversationSource,
        scheduler: LoopScheduler,
        credit_issuer: CreditIssuer,
        lifecycle: PhaseLifecycle,
        stop_checker: StopConditionChecker,
        subagents: SubagentOrchestrator | None = None,
        **kwargs,
    ):
        super().__init__(logger_name="FixedScheduleTiming")
        self._config = config
        self._conversation_source = conversation_source
        self._scheduler = scheduler
        self._credit_issuer = credit_issuer
        self._lifecycle = lifecycle
        self._subagents = subagents
        self._time_scale = 1.0 / (config.fixed_schedule_speedup or 1.0)

        self._absolute_schedule: list[ScheduleEntry] = []
        self._schedule_zero_ms: float = 0.0

        if self._subagents is not None:
            self._subagents._dispatch = self._dispatch_turn

    def _timestamp_to_perf_sec(self, timestamp_ms: int | float) -> float:
        """Convert trace timestamp in milliseconds to perf counter seconds."""
        scaled_offset_ms = (timestamp_ms - self._schedule_zero_ms) * self._time_scale
        target_offset_sec = scaled_offset_ms / MILLIS_PER_SECOND
        return self._lifecycle.started_at_perf_sec + target_offset_sec

    async def setup_phase(self) -> None:
        """Build absolute schedule from dataset metadata."""
        self._absolute_schedule = []

        for conv in self._conversation_source.dataset_metadata.conversations:
            if not conv.turns or conv.agent_depth > 0:
                continue
            if conv.turns[0].timestamp_ms is None:
                raise ValueError(
                    f"First turn of {conv.conversation_id} missing timestamp_ms"
                )
            self._absolute_schedule.append(
                ScheduleEntry(
                    timestamp_ms=conv.turns[0].timestamp_ms,
                    turn=TurnToSend(
                        conversation_id=conv.conversation_id,
                        x_correlation_id=str(uuid.uuid4()),
                        turn_index=0,
                        num_turns=len(conv.turns),
                    ),
                )
            )

        if not self._absolute_schedule:
            raise ValueError("No conversations with valid first-turn timestamps found")

        self._absolute_schedule.sort(key=lambda x: x.timestamp_ms)
        if self._config.auto_offset_timestamps:
            self._schedule_zero_ms = self._absolute_schedule[0].timestamp_ms
        elif self._config.fixed_schedule_start_offset is not None:
            self._schedule_zero_ms = float(self._config.fixed_schedule_start_offset)
        else:
            self._schedule_zero_ms = 0.0

        speedup_msg = (
            f", speedup={self._config.fixed_schedule_speedup}x"
            if self._config.fixed_schedule_speedup
            else ""
        )
        self.info(
            f"Built schedule with {len(self._absolute_schedule)} timestamps, "
            f"zero_ms={self._schedule_zero_ms:.0f}, "
            f"auto_offset={self._config.auto_offset_timestamps}"
            f"{speedup_msg}"
        )

    async def execute_phase(self) -> None:
        """Execute absolute schedule: send first turns at precise timestamps."""
        if self._lifecycle.started_at_perf_ns is None:
            raise RuntimeError("started_at_perf_ns is not set in the lifecycle")

        for entry in self._absolute_schedule:
            self._scheduler.schedule_at_perf_sec(
                self._timestamp_to_perf_sec(entry.timestamp_ms),
                self._credit_issuer.issue_credit(entry.turn),
            )

        if self._subagents:
            self._subagents.dispatch_turn0_background_spawns()

    async def handle_credit_return(self, credit: Credit) -> None:
        """Handle credit return: dispatch next turn based on trace timing."""
        if self._subagents and self._subagents.intercept(credit):
            return

        if credit.is_final_turn:
            return

        next_meta = self._conversation_source.get_next_turn_metadata(credit)
        turn = TurnToSend.from_previous_credit(credit)
        self._dispatch_by_timing(turn, next_meta.timestamp_ms, next_meta.delay_ms)

    def on_request_complete(self, credit_return: CreditReturn) -> None:
        if self._subagents and credit_return.error:
            self._subagents.terminate_child(credit_return.credit)

    def on_cancelled_return(self, credit: Credit) -> None:
        if self._subagents:
            self._subagents.terminate_child(credit)

    def on_child_stopped(self, credit: Credit) -> None:
        if self._subagents:
            self._subagents.terminate_child(credit)

    def cleanup(self) -> None:
        if self._subagents:
            self._subagents.cleanup()

    def get_subagent_stats(self) -> dict[str, int]:
        return self._subagents.get_stats() if self._subagents else {}

    def _dispatch_turn(self, turn: TurnToSend) -> None:
        """Dispatch callback for SubagentOrchestrator: look up timing and schedule."""
        meta = self._conversation_source.get_turn_metadata_at(
            turn.conversation_id, turn.turn_index
        )
        self._dispatch_by_timing(turn, meta.timestamp_ms, meta.delay_ms)

    def _dispatch_by_timing(
        self,
        turn: TurnToSend,
        timestamp_ms: int | float | None,
        delay_ms: int | float | None,
    ) -> None:
        """Dispatch a turn using timestamp, delay, or immediate execution."""
        if timestamp_ms is not None:
            self._scheduler.schedule_at_perf_sec(
                self._timestamp_to_perf_sec(timestamp_ms),
                self._credit_issuer.issue_credit(turn),
            )
        elif delay_ms is not None:
            self._scheduler.schedule_later(
                delay_ms * self._time_scale / MILLIS_PER_SECOND,
                self._credit_issuer.issue_credit(turn),
            )
        else:
            self._scheduler.execute_async(
                self._credit_issuer.issue_credit(turn),
            )
