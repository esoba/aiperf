# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive user scaling strategy based on TTFT headroom.

Starts with a small number of concurrent users and scales up as long as
TTFT remains below a configured threshold. Designed for trace replay
workloads where the goal is to find the maximum sustainable concurrency.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.credit.structs import Credit, TurnToSend

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource, SampledSession
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.stop_conditions import StopConditionChecker

NS_PER_SEC = 1_000_000_000


class AdaptiveScaleStrategy(AIPerfLoggerMixin):
    """Adaptive user scaling based on TTFT headroom.

    Starts with start_users concurrent sessions and periodically assesses TTFT.
    If the measured TTFT metric is below max_ttft_sec, adds users proportional
    to the remaining headroom. Scaling stops when TTFT exceeds the threshold
    or max_users is reached.

    TTFT samples are received via on_ttft_sample() called from the
    CreditCallbackHandler.
    """

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
        super().__init__(logger_name="AdaptiveScaleTiming", **kwargs)
        self._config = config
        self._conversation_source = conversation_source
        self._scheduler = scheduler
        self._stop_checker = stop_checker
        self._credit_issuer = credit_issuer
        self._lifecycle = lifecycle

        self._start_users = config.start_users or 1
        self._max_users = config.max_users
        self._max_ttft_sec = config.max_ttft_sec or 2.0
        self._ttft_metric = config.ttft_metric or "p95"
        self._assessment_period_sec = config.assessment_period_sec or 30.0
        self._max_delay_sec = config.max_delay_sec
        self._time_scale = config.time_scale or 1.0
        self._recycle = config.recycle_sessions
        self._stagger_sec = (config.stagger_ms or 50.0) / MILLIS_PER_SECOND
        self._scaling_formula = config.scaling_formula or "conservative"
        self._max_new_tokens_per_period = config.max_new_tokens_per_period
        self._enable_rate_limiting = config.enable_rate_limiting

        # Active state
        self._active_users = 0
        self._period_ttft_samples: list[float] = []
        self._all_ttft_samples: list[float] = []
        self._period_new_tokens = 0

        # Per-session rate limiting state
        self._rate_limit_counts: dict[str, int] = {}
        self._session_backoffs: dict[str, float] = {}

        # Working set budget tracking
        self._max_working_set_tokens = config.max_working_set_tokens
        self._block_size = config.block_size or 64
        self._active_hash_ids: set[int] = set()
        self._session_hash_ids: dict[str, set[int]] = {}

    async def setup_phase(self) -> None:
        """Nothing to pre-compute; conversations are sampled on demand."""

    async def execute_phase(self) -> None:
        """Main loop: issue initial users, then assess and scale periodically."""
        if self._lifecycle.started_at_perf_ns is None:
            raise RuntimeError("started_at_perf_ns is not set in the lifecycle")

        # Issue first-turn credits for start_users with stagger
        for i in range(self._start_users):
            if not self._stop_checker.can_send_any_turn():
                return

            session = self._conversation_source.next()
            turn = session.build_first_turn()
            self._active_users += 1

            if i == 0:
                await self._credit_issuer.issue_credit(turn)
            else:
                self._scheduler.schedule_later(
                    self._stagger_sec * i,
                    self._credit_issuer.issue_credit(turn),
                )

        self.info(
            f"Adaptive scale: started {self._start_users} users, "
            f"max_ttft={self._max_ttft_sec}s, "
            f"metric={self._ttft_metric}, "
            f"assessment_period={self._assessment_period_sec}s"
        )

        # Assessment loop
        while self._stop_checker.can_send_any_turn():
            await asyncio.sleep(self._assessment_period_sec)

            if not self._stop_checker.can_send_any_turn():
                return

            self._assess_and_scale()

    def _assess_and_scale(self) -> None:
        """Assess TTFT samples and decide whether to add more users."""
        samples = self._period_ttft_samples
        if not samples:
            self.debug("No TTFT samples in assessment period, skipping")
            return

        metric_value = self._compute_ttft_metric(samples)
        self._all_ttft_samples.extend(samples)
        self._period_ttft_samples = []
        self._period_new_tokens = 0

        if metric_value >= self._max_ttft_sec:
            self.info(
                f"TTFT {self._ttft_metric}={metric_value:.3f}s >= "
                f"threshold {self._max_ttft_sec}s, not adding users "
                f"(active={self._active_users})"
            )
            return

        # Calculate headroom-based scaling
        headroom_ratio = 1.0 - (metric_value / self._max_ttft_sec)
        headroom_pct = headroom_ratio * 100.0
        users_to_add = self._compute_users_to_add(headroom_ratio, headroom_pct)

        if self._max_users is not None:
            users_to_add = min(users_to_add, self._max_users - self._active_users)

        if users_to_add <= 0:
            self.info(
                f"TTFT {self._ttft_metric}={metric_value:.3f}s, "
                f"at max_users={self._max_users}"
            )
            return

        self.info(
            f"TTFT {self._ttft_metric}={metric_value:.3f}s, "
            f"headroom={headroom_ratio:.1%}, "
            f"adding {users_to_add} users (total={self._active_users + users_to_add})"
        )

        actually_added = 0
        for i in range(users_to_add):
            if not self._stop_checker.can_send_any_turn():
                break

            session = self._conversation_source.next()

            if not self._check_token_budget(session):
                break

            turn = session.build_first_turn()
            self._active_users += 1
            actually_added += 1

            self._scheduler.schedule_later(
                self._stagger_sec * i,
                self._credit_issuer.issue_credit(turn),
            )

        if actually_added < users_to_add and actually_added > 0:
            self.debug(
                f"Token budget limited new users to {actually_added}/{users_to_add}"
            )

    def _check_token_budget(self, session: SampledSession) -> bool:
        """Check if a new session fits within token budget and working set budget.

        Checks all conditions before committing any side effects to avoid
        partial state updates when a later check fails.

        Returns True if the session can be issued, False if budget exhausted.
        """
        first_turn = session.metadata.turns[0] if session.metadata.turns else None

        # Check token budget
        first_turn_tokens = (first_turn.input_tokens or 0) if first_turn else 0
        if self._max_new_tokens_per_period is not None:
            remaining = self._max_new_tokens_per_period - self._period_new_tokens
            if first_turn_tokens > remaining:
                return False

        # Check working set budget
        new_ids = set(first_turn.hash_ids) if first_turn else set()
        if self._max_working_set_tokens is not None and new_ids:
            projected = len(self._active_hash_ids | new_ids) * self._block_size
            if projected > self._max_working_set_tokens:
                return False

        # All checks passed — commit side effects
        self._period_new_tokens += first_turn_tokens
        if new_ids and self._max_working_set_tokens is not None:
            self._active_hash_ids |= new_ids
            self._session_hash_ids[session.x_correlation_id] = new_ids

        return True

    def _compute_users_to_add(self, headroom_ratio: float, headroom_pct: float) -> int:
        """Compute the number of users to add based on the configured formula."""
        match self._scaling_formula:
            case "aggressive":
                return max(2, 2 + int(headroom_pct / 10))
            case "linear":
                return max(1, int(headroom_pct / 5))
            case _:  # conservative
                return max(1, int(self._active_users * headroom_ratio * 0.5))

    def _compute_ttft_metric(self, samples: list[float]) -> float:
        """Compute the configured TTFT metric from samples."""
        if not samples:
            return 0.0

        sorted_samples = sorted(samples)
        match self._ttft_metric:
            case "p95":
                idx = int(math.ceil(len(sorted_samples) * 0.95)) - 1
                return sorted_samples[max(0, idx)]
            case "avg":
                return sum(sorted_samples) / len(sorted_samples)
            case "max":
                return sorted_samples[-1]
            case _:
                return sorted_samples[int(math.ceil(len(sorted_samples) * 0.95)) - 1]

    async def handle_credit_return(self, credit: Credit) -> None:
        """Dispatch next turn with trace delay, or recycle on completion."""
        if credit.is_final_turn:
            self._cleanup_session(credit.x_correlation_id)
            if self._recycle and self._stop_checker.can_send_any_turn():
                session = self._conversation_source.next()
                turn = session.build_first_turn()
                self._scheduler.execute_async(
                    self._credit_issuer.issue_credit(turn),
                )
            else:
                self._active_users -= 1
            return

        # Non-final turn: dispatch next with delay from metadata
        next_meta = self._conversation_source.get_next_turn_metadata(credit)
        turn = TurnToSend.from_previous_credit(credit)

        delay_sec = 0.0
        if next_meta.delay_ms is not None:
            delay_sec = (next_meta.delay_ms / MILLIS_PER_SECOND) * self._time_scale
            if self._max_delay_sec is not None:
                delay_sec = min(delay_sec, self._max_delay_sec)

        session_backoff = self._session_backoffs.get(credit.x_correlation_id, 0.0)
        delay_sec += session_backoff

        if delay_sec > 0:
            self._scheduler.schedule_later(
                delay_sec,
                self._credit_issuer.issue_credit(turn),
            )
        else:
            self._scheduler.execute_async(
                self._credit_issuer.issue_credit(turn),
            )

    def _cleanup_session(self, corr_id: str) -> None:
        """Remove all per-session tracking state for a completed session."""
        self._rate_limit_counts.pop(corr_id, None)
        self._session_backoffs.pop(corr_id, None)
        removed_ids = self._session_hash_ids.pop(corr_id, None)
        if removed_ids is not None:
            self._active_hash_ids = (
                set().union(*self._session_hash_ids.values())
                if self._session_hash_ids
                else set()
            )

    def on_ttft_sample(self, ttft_ns: int, credit: Credit | None = None) -> None:
        """Receive a TTFT sample and optionally apply per-session rate limiting.

        When rate limiting is enabled and credit is provided, sessions with TTFT
        above the threshold receive exponential backoff on subsequent turn delays.
        """
        ttft_sec = ttft_ns / NS_PER_SEC
        self._period_ttft_samples.append(ttft_sec)

        if not self._enable_rate_limiting or credit is None:
            return

        corr_id = credit.x_correlation_id
        if ttft_sec > self._max_ttft_sec:
            count = self._rate_limit_counts.get(corr_id, 0)
            backoff = min(ttft_sec / self._max_ttft_sec - 1.0, 10.0)
            actual = backoff * (1.5**count)
            self._session_backoffs[corr_id] = actual
            self._rate_limit_counts[corr_id] = count + 1
        else:
            self._session_backoffs.pop(corr_id, None)
            self._rate_limit_counts.pop(corr_id, None)
