# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive user scaling strategy based on TTFT headroom or goodput ratio.

Starts with a small number of concurrent users and scales up as long as
TTFT remains below a configured threshold (headroom mode) or goodput ratio
stays above a minimum (SLO mode). Designed for trace replay workloads
where the goal is to find the maximum sustainable concurrency.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.credit.structs import Credit, TurnToSend

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.credit.messages import CreditReturn
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource, SampledSession
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.stop_conditions import StopConditionChecker

NS_PER_SEC = 1_000_000_000
NS_PER_MS = 1_000_000


@dataclass(slots=True)
class PendingSubagentJoin:
    """Tracks completion of subagent children before dispatching the parent's join turn."""

    parent_conversation_id: str
    parent_correlation_id: str
    expected_count: int
    completed_count: int = 0
    join_turn_index: int = 0
    parent_num_turns: int = 0


class AdaptiveScaleStrategy(AIPerfLoggerMixin):
    """Adaptive user scaling based on TTFT headroom or goodput ratio.

    Two scaling modes:
    - TTFT headroom (default): Scales up while TTFT metric < threshold.
    - Goodput ratio (when --adaptive-scale-slo configured): Scales up while
      the fraction of requests meeting all SLO thresholds >= min_goodput_ratio.

    TTFT samples are received via on_ttft_sample() called from the
    CreditCallbackHandler. Completed requests are received via
    on_request_complete() for SLO evaluation.
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
        self._period_new_tokens = 0

        # Per-session rate limiting state
        self._rate_limit_counts: dict[str, int] = {}
        self._session_backoffs: dict[str, float] = {}

        # Working set budget tracking
        self._max_working_set_tokens = config.max_working_set_tokens
        self._block_size = config.block_size or 64
        self._active_hash_ids: set[int] = set()
        self._session_hash_ids: dict[str, set[int]] = {}

        # SLO-based goodput scaling
        self._slo_thresholds: dict[str, int] | None = self._normalize_slo_thresholds(
            config.adaptive_scale_slo
        )
        self._min_goodput_ratio = config.min_goodput_ratio
        self._period_good_count = 0
        self._period_total_count = 0

        # TTL-aware working set eviction
        self._session_last_active_ns: dict[str, int] = {}
        self._session_is_subagent: dict[str, bool] = {}
        self._cache_ttl_ns = int(config.cache_ttl_sec * NS_PER_SEC)
        self._subagent_cache_ttl_ns = int(config.subagent_cache_ttl_sec * NS_PER_SEC)

        # Subagent spawn tracking
        self._pending_subagent_joins: dict[str, PendingSubagentJoin] = {}
        self._subagent_child_to_parent: dict[str, str] = {}

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

    def _evict_expired_sessions(self) -> None:
        """Remove sessions whose cache TTL has expired from the working set."""
        if not self._session_last_active_ns:
            return

        now_ns = time.perf_counter_ns()
        expired: list[str] = []
        for corr_id, last_ns in self._session_last_active_ns.items():
            ttl = (
                self._subagent_cache_ttl_ns
                if self._session_is_subagent.get(corr_id, False)
                else self._cache_ttl_ns
            )
            if now_ns - last_ns > ttl:
                expired.append(corr_id)

        needs_recompute = False
        for corr_id in expired:
            self._session_last_active_ns.pop(corr_id, None)
            self._session_is_subagent.pop(corr_id, None)
            if self._session_hash_ids.pop(corr_id, None) is not None:
                needs_recompute = True

        if needs_recompute:
            self._active_hash_ids = (
                set().union(*self._session_hash_ids.values())
                if self._session_hash_ids
                else set()
            )

        if expired:
            self.debug(f"TTL evicted {len(expired)} expired sessions from working set")

    def _assess_and_scale(self) -> None:
        """Assess metrics and decide whether to add more users."""
        self._evict_expired_sessions()
        if self._slo_thresholds:
            self._assess_and_scale_goodput()
        else:
            self._assess_and_scale_ttft()

    def _assess_and_scale_ttft(self) -> None:
        """Assess TTFT samples and decide whether to add more users (headroom mode)."""
        samples = self._period_ttft_samples
        self._period_ttft_samples = []
        self._period_new_tokens = 0

        if not samples:
            self.debug("No TTFT samples in assessment period, skipping")
            return

        metric_value = self._compute_ttft_metric(samples)

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
        self._add_users(headroom_ratio, headroom_pct)

    def _assess_and_scale_goodput(self) -> None:
        """Assess goodput ratio and decide whether to add more users (SLO mode)."""
        total = self._period_total_count
        good = self._period_good_count
        self._period_total_count = 0
        self._period_good_count = 0
        self._period_ttft_samples = []
        self._period_new_tokens = 0

        if total == 0:
            self.debug("No completed requests in assessment period, skipping")
            return

        goodput_ratio = good / total

        if goodput_ratio < self._min_goodput_ratio:
            self.info(
                f"Goodput ratio {goodput_ratio:.1%} ({good}/{total}) < "
                f"min {self._min_goodput_ratio:.1%}, not adding users "
                f"(active={self._active_users})"
            )
            return

        headroom_ratio = goodput_ratio - self._min_goodput_ratio
        headroom_pct = headroom_ratio * 100.0

        self.info(
            f"Goodput ratio {goodput_ratio:.1%} ({good}/{total}), "
            f"headroom={headroom_ratio:.1%}"
        )
        self._add_users(headroom_ratio, headroom_pct)

    def _add_users(self, headroom_ratio: float, headroom_pct: float) -> None:
        """Add users based on headroom. Shared by TTFT and goodput scaling."""
        users_to_add = self._compute_users_to_add(headroom_ratio, headroom_pct)

        if self._max_users is not None:
            users_to_add = min(users_to_add, self._max_users - self._active_users)

        if users_to_add <= 0:
            self.info(
                f"At max_users={self._max_users}, not adding users "
                f"(active={self._active_users})"
            )
            return

        self.info(
            f"Adding {users_to_add} users (total={self._active_users + users_to_add})"
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
        self._evict_expired_sessions()
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
        """Dispatch next turn with trace delay, or recycle on completion.

        Three paths:
        A) Subagent child's final turn: signal parent join
        B) Next turn is blocked by subagent spawn: fan-out child sessions
        C) Normal sequential turn: existing logic
        """
        # Update last-active timestamp for TTL tracking
        self._session_last_active_ns[credit.x_correlation_id] = time.perf_counter_ns()

        # Path A: Subagent child's final turn -- signal parent join
        if (
            credit.is_final_turn
            and credit.x_correlation_id in self._subagent_child_to_parent
        ):
            self._handle_subagent_child_complete(credit)
            return

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

        # Path B: Next turn is blocked by subagent spawn
        next_meta = self._conversation_source.get_next_turn_metadata(credit)
        if next_meta.subagent_spawn_id is not None:
            self._dispatch_subagent_spawn(credit, next_meta.subagent_spawn_id)
            return

        # Path C: Normal sequential turn
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

    def _dispatch_subagent_spawn(self, credit: Credit, spawn_id: str) -> None:
        """Fan out subagent child sessions and register a pending join.

        For background spawns, the parent continues immediately by dispatching
        the join turn without waiting for children to complete.
        """
        spawn = self._conversation_source.get_subagent_spawn(
            credit.conversation_id, spawn_id
        )
        if spawn is None:
            turn = TurnToSend.from_previous_credit(credit)
            self._scheduler.execute_async(
                self._credit_issuer.issue_credit(turn),
            )
            return

        parent_corr_id = credit.x_correlation_id

        # Fan out children (same for both background and blocking)
        for child_conv_id in spawn.child_conversation_ids:
            child_session = self._conversation_source.start_child_session(child_conv_id)
            self._subagent_child_to_parent[child_session.x_correlation_id] = (
                parent_corr_id
            )
            self._session_is_subagent[child_session.x_correlation_id] = True
            child_turn = child_session.build_first_turn()
            self._scheduler.execute_async(
                self._credit_issuer.issue_credit(child_turn),
            )

        if spawn.is_background:
            # Parent continues immediately -- dispatch join turn without waiting
            join_turn = TurnToSend(
                conversation_id=credit.conversation_id,
                x_correlation_id=parent_corr_id,
                turn_index=spawn.join_turn_index,
                num_turns=credit.num_turns,
            )
            self._scheduler.execute_async(
                self._credit_issuer.issue_credit(join_turn),
            )
        else:
            # Blocking: register pending join (wait for all children)
            self._pending_subagent_joins[parent_corr_id] = PendingSubagentJoin(
                parent_conversation_id=credit.conversation_id,
                parent_correlation_id=parent_corr_id,
                expected_count=len(spawn.child_conversation_ids),
                join_turn_index=spawn.join_turn_index,
                parent_num_turns=credit.num_turns,
            )

    def _handle_subagent_child_complete(self, credit: Credit) -> None:
        """Handle a subagent child's final turn. Dispatch parent join when all children complete."""
        child_corr_id = credit.x_correlation_id
        parent_corr_id = self._subagent_child_to_parent.pop(child_corr_id, None)
        if parent_corr_id is None:
            return

        self._cleanup_session(child_corr_id)

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
        )
        self._scheduler.execute_async(
            self._credit_issuer.issue_credit(join_turn),
        )

    def _cleanup_session(self, corr_id: str) -> None:
        """Remove all per-session tracking state for a completed session."""
        self._rate_limit_counts.pop(corr_id, None)
        self._session_backoffs.pop(corr_id, None)
        self._pending_subagent_joins.pop(corr_id, None)
        self._session_last_active_ns.pop(corr_id, None)
        self._session_is_subagent.pop(corr_id, None)
        orphaned = [
            k for k, v in self._subagent_child_to_parent.items() if v == corr_id
        ]
        for k in orphaned:
            self._subagent_child_to_parent.pop(k, None)
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
            actual = min(backoff * (1.5**count), 30.0)
            self._session_backoffs[corr_id] = actual
            self._rate_limit_counts[corr_id] = count + 1
        else:
            self._session_backoffs.pop(corr_id, None)
            self._rate_limit_counts.pop(corr_id, None)

    def on_request_complete(self, credit_return: CreditReturn) -> None:
        """Evaluate a completed request against SLO thresholds for goodput tracking.

        Called by CreditCallbackHandler for non-cancelled returns when SLO-based
        scaling is active. Updates period goodput counters and applies per-session
        rate limiting when SLOs are violated.
        """
        if not self._slo_thresholds:
            return

        self._period_total_count += 1
        if self._evaluate_slos(credit_return):
            self._period_good_count += 1
            if self._enable_rate_limiting:
                corr_id = credit_return.credit.x_correlation_id
                self._session_backoffs.pop(corr_id, None)
                self._rate_limit_counts.pop(corr_id, None)
        elif self._enable_rate_limiting:
            corr_id = credit_return.credit.x_correlation_id
            count = self._rate_limit_counts.get(corr_id, 0)
            backoff = min(2.0 * (1.5**count), 30.0)
            self._session_backoffs[corr_id] = backoff
            self._rate_limit_counts[corr_id] = count + 1

    def _evaluate_slos(self, credit_return: CreditReturn) -> bool:
        """Check if a completed request meets all configured SLO thresholds.

        Returns True if all SLOs are satisfied, False if any fail or data is missing.
        """
        if not self._slo_thresholds:
            return True

        for metric_tag, threshold_ns in self._slo_thresholds.items():
            match metric_tag:
                case "time_to_first_token":
                    if credit_return.ttft_ns is None:
                        return False
                    if credit_return.ttft_ns > threshold_ns:
                        return False
                case "request_latency":
                    if credit_return.request_latency_ns is None:
                        return False
                    if credit_return.request_latency_ns > threshold_ns:
                        return False
        return True

    @staticmethod
    def _normalize_slo_thresholds(
        slo_dict: dict[str, float] | None,
    ) -> dict[str, int] | None:
        """Convert SLO thresholds from display units (ms) to nanoseconds.

        Uses the metric registry to look up each metric's display_unit and base unit,
        then converts the user-provided value accordingly.
        """
        if not slo_dict:
            return None

        from aiperf.metrics.metric_registry import MetricRegistry

        normalized: dict[str, int] = {}
        for metric_tag, value in slo_dict.items():
            metric_cls = MetricRegistry.get_class(metric_tag)
            display_unit = metric_cls.display_unit
            base_unit = metric_cls.unit
            if (
                display_unit is not None
                and base_unit is not None
                and display_unit != base_unit
            ):
                value = display_unit.convert_to(base_unit, value)
            normalized[metric_tag] = int(value)
        return normalized
