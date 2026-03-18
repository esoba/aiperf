# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixin providing subagent lifecycle delegation for timing strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.credit.messages import CreditReturn
    from aiperf.timing.subagent_orchestrator import SubagentOrchestrator


class SubagentMixin:
    """Mixin that delegates subagent lifecycle methods to a SubagentOrchestrator.

    Strategies inherit this to avoid duplicating the same guard-and-delegate
    pattern for error/cancel handling and cleanup.

    Subclasses must:
    - Call ``_init_subagents(subagents)`` in their ``__init__``
    - Implement ``_dispatch_turn(turn)`` (strategy-specific dispatch callback)

    The dispatch callback is how the orchestrator re-enters the strategy to
    schedule work. It receives a TurnToSend and must schedule it via the
    strategy's own timing mechanism (e.g., schedule_at_perf_sec for
    FixedSchedule, continuation queue for RequestRate, immediate for
    UserCentric). The orchestrator calls this for:
    - Gated parent turns (after all children complete)
    - Non-final child next turns (continuing a child conversation)
    - Background child first turns (fire-and-forget)
    """

    _subagents: SubagentOrchestrator | None

    def _init_subagents(self, subagents: SubagentOrchestrator | None) -> None:
        self._subagents = subagents
        if self._subagents is not None:
            # Wire the dispatch callback. The orchestrator is created by
            # PhaseRunner before the strategy, so it can't receive the
            # callback at construction time.
            self._subagents.set_dispatch(self._dispatch_turn)

    def on_failed_credit(self, credit_return: CreditReturn) -> None:
        """Release errored/cancelled non-final children from gate tracking.

        Only non-final children: final turns do gate accounting in
        _handle_child_credit. Calling terminate_child for final turns
        would double-release from _child_to_gate.
        """
        if not self._subagents:
            return
        credit = credit_return.credit
        if credit.agent_depth > 0 and not credit.is_final_turn:
            self._subagents.terminate_child(credit)

    def cleanup(self) -> None:
        if self._subagents:
            self._subagents.cleanup()
