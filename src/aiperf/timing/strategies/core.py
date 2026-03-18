# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.credit.messages import CreditReturn
    from aiperf.credit.structs import Credit
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.stop_conditions import StopConditionChecker

# =============================================================================
# TimingStrategyProtocol - Timing strategy interface
# =============================================================================


@runtime_checkable
class TimingStrategyProtocol(Protocol):
    """Protocol for pluggable timing strategies.

    Strategies define:
    1. __init__(): Receive all dependencies (sync)
    2. setup_phase(): Async initialization (no parameters)
    3. execute_phase(): Send first turns (main timing loop)
    4. handle_credit_return(): Handle credit return, dispatch next turn if needed

    Fresh strategy instances are created per-phase by PhaseRunner.
    All dependencies are injected via __init__ for clean, testable design.
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
    ) -> None: ...

    async def setup_phase(self) -> None:
        """Async initialization using dependencies from __init__.

        Called by PhaseRunner immediately before execute_phase.
        Use this for async setup work (pre-generating users, building schedules, etc.).
        """
        ...

    async def execute_phase(self) -> None:
        """Execute the main timing loop for first turns.

        Sends first turns according to the timing strategy (rate, schedule, etc.).
        Subsequent turns are dispatched by handle_credit_return via callbacks,
        or pulled from a continuation queue here for queue-based strategies.

        Must not return until all first-turn credits are sent (or queued).
        """
        ...

    async def handle_credit_return(self, credit: Credit) -> None:
        """Handle credit return: dispatch next turn if applicable.

        Called by CreditCallbackHandler when can_send_any_turn() is True OR
        credit.agent_depth > 0 (child bypass for gate accounting).
        Implementations with subagents must call intercept() first -- the
        orchestrator handles all child routing and stop-condition checks.
        """
        ...

    def on_failed_credit(self, credit_return: CreditReturn) -> None:
        """Handle errored/cancelled returns for subagent gate cleanup.

        Called BEFORE handle_credit_return, regardless of can_send_any_turn().
        No-op for strategies without subagents.
        """
        ...

    def cleanup(self) -> None:
        """Release resources at phase end. Called by PhaseRunner after completion."""
        ...


# =============================================================================
# RateSettableProtocol - Protocol for strategies that support dynamic rate adjustment
# =============================================================================


@runtime_checkable
class RateSettableProtocol(Protocol):
    """Protocol for timing strategies that support dynamic rate adjustment.

    Timing strategies implementing this protocol can have their request rate
    ramped during phase execution via the Ramper system.

    Note:
        This is separate from TimingStrategyProtocol because not all
        strategies have a rate concept (e.g., FixedScheduleStrategy).
    """

    def set_request_rate(self, rate: float) -> None:
        """Update the request rate dynamically.

        Args:
            rate: New request rate in requests per second (must be > 0).
        """
        ...
