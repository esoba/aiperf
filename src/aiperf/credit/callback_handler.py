# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credit callback handler for credit lifecycle events.

Handles ALL credit lifecycle callbacks (returns + TTFT) directly from CreditRouter.

Processing order for credit returns (see SubagentOrchestrator module docstring
for why this ordering is load-bearing for subagent correctness)::

    1. Atomic counting (increment_returned)
    2. Track prefill release if TTFT never arrived
    3. Release concurrency slots
    4. on_failed_credit for errored/cancelled child gate cleanup
    5. Signal all_credits_returned_event if final return
    6. handle_credit_return → strategy dispatch (with child bypass)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CreditPhase

if TYPE_CHECKING:
    from aiperf.credit.messages import CreditReturn, FirstToken
    from aiperf.credit.structs import Credit
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
    from aiperf.timing.phase.stop_conditions import StopConditionChecker
    from aiperf.timing.strategies.core import TimingStrategyProtocol

_logger = AIPerfLogger(__name__)


@dataclass(slots=True)
class PhaseCallbackContext:
    """Context for handling callbacks for a specific phase.

    Registered by PhaseRunner before phase execution starts.
    Contains all components needed to handle credit returns for this phase.
    """

    progress: PhaseProgressTracker
    lifecycle: PhaseLifecycle
    stop_checker: StopConditionChecker
    strategy: TimingStrategyProtocol
    concurrency_manager: ConcurrencyManager


# =============================================================================
# CreditCallbackHandler - Handle credit lifecycle callbacks
# =============================================================================


class CreditCallbackHandler:
    """Handles credit lifecycle callbacks from CreditRouter.

    Callback flow:
        Worker -> CreditRouter -> CreditCallbackHandler -> [count, release, dispatch]

    PhaseRunner calls register_phase() BEFORE any credits are sent.
    """

    def __init__(self, concurrency_manager: ConcurrencyManager) -> None:
        """Initialize callback handler.

        Args:
            concurrency_manager: Manages concurrency slots (shared across phases).
        """
        self._concurrency_manager = concurrency_manager
        self._phase_handlers: dict[CreditPhase, PhaseCallbackContext] = {}

    def register_phase(
        self,
        phase: CreditPhase,
        progress: PhaseProgressTracker,
        lifecycle: PhaseLifecycle,
        stop_checker: StopConditionChecker,
        strategy: TimingStrategyProtocol,
    ) -> None:
        """Register phase for callback handling.

        Called by PhaseRunner BEFORE phase execution starts.
        Must be called before any credits are sent for this phase.

        Args:
            phase: Phase enum (WARMUP or PROFILING).
            progress: Progress tracker for counting.
            lifecycle: Phase lifecycle for state checks.
            stop_checker: Evaluates stop conditions.
            strategy: Timing strategy for dispatching next turns.
        """
        self._phase_handlers[phase] = PhaseCallbackContext(
            progress=progress,
            lifecycle=lifecycle,
            stop_checker=stop_checker,
            strategy=strategy,
            concurrency_manager=self._concurrency_manager,
        )
        _logger.debug(lambda: f"Registered callback handler for phase {phase}")

    def unregister_phase(self, phase: CreditPhase) -> None:
        """Unregister phase when done.

        Called by PhaseRunner after phase completes.
        Late arrivals after unregister are logged but ignored.

        Args:
            phase: Phase to unregister.
        """
        if phase in self._phase_handlers:
            del self._phase_handlers[phase]
            _logger.debug(lambda: f"Unregistered callback handler for phase {phase}")

    async def on_credit_return(
        self, worker_id: str, credit_return: CreditReturn
    ) -> None:
        """Handle credit return from worker. See module docstring for step ordering."""
        credit = credit_return.credit
        phase = credit.phase

        # Get phase handler (returns None if phase already cleaned up)
        handler = self._phase_handlers.get(phase)
        if not handler:
            _logger.debug(
                lambda: f"Credit return for unregistered phase {phase}, "
                f"credit_id={credit.id}, worker={worker_id}"
            )
            return

        # Late arrivals after phase complete are logged but don't affect counts
        if handler.lifecycle.is_complete:
            _logger.warning(
                lambda: f"Credit return after phase {phase} complete, "
                f"credit_id={credit.id}, worker={worker_id}"
            )
            return

        # 1. ATOMIC COUNTING (no await before this!)
        is_final_returned = handler.progress.increment_returned(
            credit.is_final_turn,
            credit_return.cancelled,
            agent_depth=credit.agent_depth,
        )

        # 2. Track prefill release if TTFT never arrived
        if not credit_return.first_token_sent:
            handler.progress.increment_prefill_released()

        # 3. Release concurrency slots
        self._release_slots_for_return(
            phase, credit, credit_return, is_final_returned, handler
        )

        # 4. on_failed_credit for errored/cancelled child gate cleanup.
        # ORDER MATTERS: Must run BEFORE handle_credit_return (step 6) so
        # terminate_child marks the child before _handle_child_credit checks
        # _is_terminated. Reversing steps 4 and 6 causes zombie child dispatch.
        # Not gated by can_send_any_turn -- this is bookkeeping, not new work.
        if credit_return.error or credit_return.cancelled:
            handler.strategy.on_failed_credit(credit_return)

        # 5. Signal all_credits_returned_event if final return
        if is_final_returned:
            handler.progress.all_credits_returned_event.set()

        # 6. handle_credit_return with child bypass.
        # Child returns (depth > 0) MUST always reach the orchestrator for gate
        # accounting, even after stop fires. Without this bypass, child final
        # returns are silently dropped, leaving parent gates permanently stuck.
        # The orchestrator has its own guards against post-stop dispatch
        # (see SubagentOrchestrator module docstring "Stop Condition Interaction").
        if handler.stop_checker.can_send_any_turn() or credit.agent_depth > 0:
            await handler.strategy.handle_credit_return(credit)

    def _release_slots_for_return(
        self,
        phase: CreditPhase,
        credit: Credit,
        credit_return: CreditReturn,
        is_final_returned: bool,
        handler: PhaseCallbackContext,
    ) -> None:
        """Release slots based on credit state.

        Slot release rules:
        - Session slot: Released when conversation ends (final turn)
        - Prefill slot: Released if TTFT never arrived (error/cancellation path)
        - On final return: Cleanup in-flight sessions

        Args:
            phase: Credit phase.
            credit: The returned credit.
            credit_return: Return details.
            is_final_returned: True if this is the last credit of the phase.
            handler: Phase callback context.
        """
        concurrency = handler.concurrency_manager

        # Release session slot when root conversation ends (final turn, whether completed or cancelled).
        # Child sessions (depth > 0) never acquired a session slot, so skip release.
        if credit.is_final_turn and credit.agent_depth == 0:
            concurrency.release_session_slot(phase)

        # On phase end, release slots for sessions still in flight.
        # These are sessions that started but whose final turn was never sent/returned.
        if is_final_returned:
            in_flight = handler.progress.in_flight_sessions
            if in_flight > 0:
                _logger.debug(
                    lambda: f"Releasing {in_flight} in-flight session slots for phase {phase}"
                )
                for _ in range(in_flight):
                    concurrency.release_session_slot(phase)

        # Prefill slot is normally released on TTFT. If the request failed or was
        # cancelled before first token, we release here to prevent slot leaks.
        if not credit_return.first_token_sent:
            concurrency.release_prefill_slot(phase)

    async def on_first_token(self, first_token: FirstToken) -> None:
        """Handle first token event (TTFT) from worker.

        Releases prefill concurrency slot, allowing another request
        to start prefilling.

        Args:
            first_token: TTFT event details including credit_id and phase.
        """
        phase = first_token.phase
        handler = self._phase_handlers.get(phase)

        if not handler:
            _logger.debug(
                lambda: f"TTFT for unregistered phase {phase}, "
                f"credit_id={first_token.credit_id}"
            )
            return

        # Track the release
        handler.progress.increment_prefill_released()

        # Release the prefill slot
        handler.concurrency_manager.release_prefill_slot(phase)
