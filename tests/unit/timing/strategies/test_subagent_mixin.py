# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SubagentMixin lifecycle delegation."""

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit
from aiperf.timing.strategies.subagent_mixin import SubagentMixin


def make_credit(
    *,
    agent_depth: int = 0,
    turn_index: int = 0,
    num_turns: int = 2,
) -> Credit:
    return Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="conv-1",
        x_correlation_id="corr-1",
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=1000,
        agent_depth=agent_depth,
    )


def make_credit_return(
    credit: Credit,
    *,
    error: str | None = None,
    cancelled: bool = False,
) -> CreditReturn:
    return CreditReturn(credit=credit, error=error, cancelled=cancelled)


def make_orchestrator() -> MagicMock:
    o = MagicMock()
    o.set_dispatch = MagicMock()
    o.terminate_child = MagicMock()
    o.cleanup = MagicMock()
    return o


class ConcreteSubagentMixin(SubagentMixin):
    """Minimal concrete class for testing the mixin."""

    def _dispatch_turn(self, turn: object) -> None:
        pass


class TestSubagentMixinInit:
    def test_init_subagents_none_sets_none(self) -> None:
        obj = ConcreteSubagentMixin()
        obj._init_subagents(None)
        assert obj._subagents is None

    def test_init_subagents_calls_set_dispatch(self) -> None:
        obj = ConcreteSubagentMixin()
        orch = make_orchestrator()
        obj._init_subagents(orch)
        orch.set_dispatch.assert_called_once_with(obj._dispatch_turn)

    def test_init_subagents_stores_orchestrator(self) -> None:
        obj = ConcreteSubagentMixin()
        orch = make_orchestrator()
        obj._init_subagents(orch)
        assert obj._subagents is orch


class TestSubagentMixinOnFailedCredit:
    def test_no_subagents_does_nothing(self) -> None:
        obj = ConcreteSubagentMixin()
        obj._init_subagents(None)
        credit = make_credit(agent_depth=1, turn_index=0, num_turns=2)
        cr = make_credit_return(credit, error="boom")
        obj.on_failed_credit(cr)  # should not raise

    def test_depth_zero_does_not_terminate(self) -> None:
        obj = ConcreteSubagentMixin()
        orch = make_orchestrator()
        obj._init_subagents(orch)
        credit = make_credit(agent_depth=0, turn_index=0, num_turns=2)
        cr = make_credit_return(credit, error="boom")
        obj.on_failed_credit(cr)
        orch.terminate_child.assert_not_called()

    def test_final_turn_does_not_terminate(self) -> None:
        obj = ConcreteSubagentMixin()
        orch = make_orchestrator()
        obj._init_subagents(orch)
        credit = make_credit(agent_depth=1, turn_index=1, num_turns=2)
        cr = make_credit_return(credit, error="boom")
        obj.on_failed_credit(cr)
        orch.terminate_child.assert_not_called()

    def test_child_non_final_with_error_terminates(self) -> None:
        obj = ConcreteSubagentMixin()
        orch = make_orchestrator()
        obj._init_subagents(orch)
        credit = make_credit(agent_depth=1, turn_index=0, num_turns=2)
        cr = make_credit_return(credit, error="timeout")
        obj.on_failed_credit(cr)
        orch.terminate_child.assert_called_once_with(credit)

    def test_child_non_final_cancelled_terminates(self) -> None:
        obj = ConcreteSubagentMixin()
        orch = make_orchestrator()
        obj._init_subagents(orch)
        credit = make_credit(agent_depth=2, turn_index=0, num_turns=3)
        cr = make_credit_return(credit, cancelled=True)
        obj.on_failed_credit(cr)
        orch.terminate_child.assert_called_once_with(credit)

    def test_child_depth_greater_than_one_terminates(self) -> None:
        obj = ConcreteSubagentMixin()
        orch = make_orchestrator()
        obj._init_subagents(orch)
        credit = make_credit(agent_depth=5, turn_index=1, num_turns=4)
        cr = make_credit_return(credit, error="err")
        obj.on_failed_credit(cr)
        orch.terminate_child.assert_called_once_with(credit)

    @pytest.mark.parametrize("num_turns", [1, 3, 10])
    def test_only_final_turn_skipped(self, num_turns: int) -> None:
        obj = ConcreteSubagentMixin()
        orch = make_orchestrator()
        obj._init_subagents(orch)
        last = num_turns - 1
        credit = make_credit(agent_depth=1, turn_index=last, num_turns=num_turns)
        cr = make_credit_return(credit, error="err")
        obj.on_failed_credit(cr)
        orch.terminate_child.assert_not_called()


class TestSubagentMixinCleanup:
    def test_cleanup_no_subagents_does_nothing(self) -> None:
        obj = ConcreteSubagentMixin()
        obj._init_subagents(None)
        obj.cleanup()  # should not raise

    def test_cleanup_delegates_to_orchestrator(self) -> None:
        obj = ConcreteSubagentMixin()
        orch = make_orchestrator()
        obj._init_subagents(orch)
        obj.cleanup()
        orch.cleanup.assert_called_once()
