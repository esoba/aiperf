# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import MetricResult
from aiperf.metrics.metric_dicts import MetricArray
from aiperf.plugin import plugins


class _SessionData:
    """Tracks per-session timing and turn count."""

    __slots__ = ("start_ns", "end_ns", "turn_count")

    def __init__(self) -> None:
        self.start_ns: int = 0
        self.end_ns: int = 0
        self.turn_count: int = 0

    def update(self, request_start_ns: int, request_end_ns: int) -> None:
        if self.turn_count == 0 or request_start_ns < self.start_ns:
            self.start_ns = request_start_ns
        if request_end_ns > self.end_ns:
            self.end_ns = request_end_ns
        self.turn_count += 1

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


class SessionMetricsResultsProcessor(AIPerfLifecycleMixin):
    """Results processor that computes session-level metrics for multi-turn conversations.

    Groups requests by session (x_correlation_id) and computes:
    - Session duration (time from first request start to last request end)
    - Turns per session (number of requests in each session)
    - Session count (total completed sessions)
    - Session throughput (sessions completed per second)

    Returns empty results for single-turn workloads where no session has
    more than one turn.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(user_config=user_config, **kwargs)

        # Multi-turn if config says so (turn mean > 1) or dataset is inherently multi-turn
        is_config_multi_turn = user_config.input.conversation.turn.mean > 1
        is_dataset_multi_turn = (
            user_config.input.custom_dataset_type is not None
            and plugins.supports_multi_turn_dataset(
                user_config.input.custom_dataset_type
            )
        )
        if not is_config_multi_turn and not is_dataset_multi_turn:
            raise PostProcessorDisabled(
                "Session metrics require multi-turn conversations"
            )

        self._sessions: dict[str, _SessionData] = defaultdict(_SessionData)
        self._benchmark_start_ns: int | None = None
        self._benchmark_end_ns: int | None = None

    async def process_result(self, record_data: MetricRecordsData) -> None:
        """Group incoming records by session."""
        if record_data.error is not None:
            return

        metadata = record_data.metadata
        session_key = metadata.x_correlation_id or str(metadata.session_num)

        self._sessions[session_key].update(
            metadata.request_start_ns, metadata.request_end_ns
        )

        # Track benchmark time bounds for throughput calculation
        if (
            self._benchmark_start_ns is None
            or metadata.request_start_ns < self._benchmark_start_ns
        ):
            self._benchmark_start_ns = metadata.request_start_ns
        if (
            self._benchmark_end_ns is None
            or metadata.request_end_ns > self._benchmark_end_ns
        ):
            self._benchmark_end_ns = metadata.request_end_ns

    async def summarize(self) -> list[MetricResult]:
        """Compute session-level metrics from accumulated session data."""
        if not self._sessions:
            return []

        duration_array = MetricArray()
        turns_array = MetricArray()

        for session in self._sessions.values():
            duration_array.append(session.duration_ns / 1e6)  # ns -> ms
            turns_array.append(session.turn_count)

        results: list[MetricResult] = [
            duration_array.to_result("session_duration", "Session Duration", "ms"),
            turns_array.to_result("session_turns", "Turns Per Session", "turns"),
        ]

        session_count = len(self._sessions)
        results.append(
            MetricResult(
                tag="session_count",
                header="Session Count",
                unit="sessions",
                avg=session_count,
                count=1,
            )
        )

        if self._benchmark_start_ns and self._benchmark_end_ns:
            duration_sec = (self._benchmark_end_ns - self._benchmark_start_ns) / 1e9
            if duration_sec > 0:
                results.append(
                    MetricResult(
                        tag="session_throughput",
                        header="Session Throughput",
                        unit="sessions/sec",
                        avg=session_count / duration_sec,
                        count=1,
                    )
                )

        return results
