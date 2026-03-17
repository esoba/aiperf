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
    """

    _subagents: SubagentOrchestrator | None

    def _init_subagents(self, subagents: SubagentOrchestrator | None) -> None:
        self._subagents = subagents
        if self._subagents is not None:
            self._subagents.set_dispatch(self._dispatch_turn)

    def on_failed_credit(self, credit_return: CreditReturn) -> None:
        """Handle errored or cancelled child credit returns.

        Called by CreditCallbackHandler for every errored/cancelled return,
        regardless of can_send_any_turn() — this is gate-tracking cleanup,
        not new work dispatch. The orchestrator's _satisfy_prerequisite checks
        can_send_any_turn() internally before dispatching gated turns.
        """
        if not self._subagents:
            return
        credit = credit_return.credit
        if credit.agent_depth > 0 and not credit.is_final_turn:
            self._subagents.terminate_child(credit)

    def cleanup(self) -> None:
        if self._subagents:
            self._subagents.cleanup()
