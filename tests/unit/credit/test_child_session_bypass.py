# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for child session (agent_depth > 0) bypass behavior.

These tests verify that child sessions spawned by SubagentOrchestrator
correctly skip session slot acquisition, session quota counting, and
session slot release. Without these fixes, children would:

1. Steal session slots from root conversations → deadlock at concurrency limit
2. Inflate sent_sessions count → premature quota exhaustion → gate hangs
3. Release session slots they never acquired → negative slot counts
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter

# =============================================================================
# Helpers
# =============================================================================


def _turn(
    conv: str = "conv1",
    idx: int = 0,
    num: int = 1,
    agent_depth: int = 0,
    parent_correlation_id: str | None = None,
) -> TurnToSend:
    return TurnToSend(
        conversation_id=conv,
        x_correlation_id=f"corr-{conv}",
        turn_index=idx,
        num_turns=num,
        agent_depth=agent_depth,
        parent_correlation_id=parent_correlation_id,
    )


def _credit(
    credit_id: int = 1,
    conv: str = "conv1",
    turn_index: int = 0,
    num_turns: int = 1,
    agent_depth: int = 0,
    phase: CreditPhase = CreditPhase.PROFILING,
) -> Credit:
    return Credit(
        id=credit_id,
        phase=phase,
        conversation_id=conv,
        x_correlation_id=f"corr-{conv}",
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=time.time_ns(),
        agent_depth=agent_depth,
    )


def _credit_return(
    credit: Credit,
    cancelled: bool = False,
    first_token_sent: bool = True,
    error: str | None = None,
) -> CreditReturn:
    return CreditReturn(
        credit=credit,
        cancelled=cancelled,
        first_token_sent=first_token_sent,
        error=error,
    )


def _cfg(
    reqs: int | None = None,
    sessions: int | None = None,
) -> CreditPhaseConfig:
    from aiperf.plugin.enums import TimingMode

    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=reqs,
        expected_num_sessions=sessions,
    )


def _make_issuer(**overrides) -> tuple:
    """Create a CreditIssuer with mocked deps, returning (issuer, mocks_dict)."""
    mocks = {
        "stop_checker": MagicMock(
            can_send_any_turn=MagicMock(return_value=True),
            can_start_new_session=MagicMock(return_value=True),
        ),
        "progress": MagicMock(
            increment_sent=MagicMock(return_value=(0, False)),
            freeze_sent_counts=MagicMock(),
            all_credits_sent_event=asyncio.Event(),
        ),
        "concurrency": MagicMock(
            acquire_session_slot=AsyncMock(return_value=True),
            acquire_prefill_slot=AsyncMock(return_value=True),
            try_acquire_session_slot=MagicMock(return_value=True),
            try_acquire_prefill_slot=MagicMock(return_value=True),
            release_session_slot=MagicMock(),
        ),
        "router": MagicMock(send_credit=AsyncMock()),
        "cancellation": MagicMock(
            next_cancellation_delay_ns=MagicMock(return_value=None)
        ),
        "lifecycle": MagicMock(
            started_at_ns=time.time_ns(),
            started_at_perf_ns=time.perf_counter_ns(),
        ),
    }
    mocks.update(overrides)
    issuer = CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=mocks["stop_checker"],
        progress=mocks["progress"],
        concurrency_manager=mocks["concurrency"],
        credit_router=mocks["router"],
        cancellation_policy=mocks["cancellation"],
        lifecycle=mocks["lifecycle"],
    )
    return issuer, mocks


# =============================================================================
# Bug 1: Children must NOT acquire session slots
# =============================================================================


class TestChildSkipsSessionSlotAcquisition:
    """Without this fix, children steal session slots from root conversations.

    Scenario: concurrency=2, 2 roots running, a child spawns → old code tries
    to acquire a 3rd session slot → blocks forever because semaphore is full
    → parent gate never fires → deadlock.
    """

    async def test_child_first_turn_skips_session_slot_acquire(self) -> None:
        """Child first turn (turn_index=0, agent_depth=1) must NOT acquire session slot."""
        issuer, mocks = _make_issuer()
        child_turn = _turn(conv="child-1", idx=0, num=2, agent_depth=1)

        await issuer.issue_credit(child_turn)

        mocks["concurrency"].acquire_session_slot.assert_not_called()
        mocks["concurrency"].acquire_prefill_slot.assert_called_once()
        mocks["router"].send_credit.assert_called_once()

    async def test_child_first_turn_skips_session_slot_try_acquire(self) -> None:
        """try_issue_credit: child first turn must NOT try-acquire session slot."""
        issuer, mocks = _make_issuer()
        child_turn = _turn(conv="child-1", idx=0, num=2, agent_depth=1)

        result = await issuer.try_issue_credit(child_turn)

        assert result is True
        mocks["concurrency"].try_acquire_session_slot.assert_not_called()
        mocks["concurrency"].try_acquire_prefill_slot.assert_called_once()

    async def test_root_first_turn_still_acquires_session_slot(self) -> None:
        """Sanity: root first turn (agent_depth=0) must still acquire session slot."""
        issuer, mocks = _make_issuer()
        root_turn = _turn(conv="root-1", idx=0, num=3, agent_depth=0)

        await issuer.issue_credit(root_turn)

        mocks["concurrency"].acquire_session_slot.assert_called_once()

    async def test_child_does_not_release_session_on_prefill_failure(self) -> None:
        """If child's prefill slot fails, no session slot release (none was acquired)."""
        issuer, mocks = _make_issuer()
        mocks["concurrency"].acquire_prefill_slot.return_value = False
        child_turn = _turn(conv="child-1", idx=0, num=2, agent_depth=1)

        result = await issuer.issue_credit(child_turn)

        assert result is False
        mocks["concurrency"].release_session_slot.assert_not_called()

    async def test_root_releases_session_on_prefill_failure(self) -> None:
        """Sanity: root releases session slot if prefill fails."""
        issuer, mocks = _make_issuer()
        mocks["concurrency"].acquire_prefill_slot.return_value = False
        root_turn = _turn(conv="root-1", idx=0, num=2, agent_depth=0)

        result = await issuer.issue_credit(root_turn)

        assert result is False
        mocks["concurrency"].release_session_slot.assert_called_once()

    @pytest.mark.parametrize("depth", [1, 2, 5])
    async def test_deeply_nested_children_skip_session_slot(self, depth: int) -> None:
        """Children at any nesting depth must skip session slot."""
        issuer, mocks = _make_issuer()
        child_turn = _turn(conv=f"child-d{depth}", idx=0, num=1, agent_depth=depth)

        await issuer.issue_credit(child_turn)

        mocks["concurrency"].acquire_session_slot.assert_not_called()

    async def test_mixed_root_and_child_interleaved(self) -> None:
        """Interleaved root/child issuance: only roots acquire session slots."""
        issuer, mocks = _make_issuer()
        credit_idx = [0]

        def inc_sent(turn):
            credit_idx[0] += 1
            return (credit_idx[0], False)

        mocks["progress"].increment_sent = inc_sent

        root_t0 = _turn(conv="root", idx=0, num=3, agent_depth=0)
        child_t0 = _turn(conv="child-a", idx=0, num=2, agent_depth=1)
        root_t1 = _turn(conv="root", idx=1, num=3, agent_depth=0)
        child_t1 = _turn(conv="child-a", idx=1, num=2, agent_depth=1)
        root_t2 = _turn(conv="root", idx=2, num=3, agent_depth=0)

        for turn in [root_t0, child_t0, root_t1, child_t1, root_t2]:
            await issuer.issue_credit(turn)

        # Only the root first turn (root_t0) should have acquired session slot
        assert mocks["concurrency"].acquire_session_slot.call_count == 1


# =============================================================================
# Bug 2: Children must NOT use can_start_new_session stop check
# =============================================================================


class TestChildUsesCanSendAnyTurn:
    """Without this fix, child first turns use can_start_new_session which
    checks session quota. When quota is exhausted by root sessions, children
    are blocked → parent gates hang waiting for children that can never start.
    """

    async def test_child_first_turn_uses_can_send_any_turn_not_can_start_new_session(
        self,
    ) -> None:
        """Child first turn must use can_send_any_turn for prefill slot check."""
        issuer, mocks = _make_issuer()
        # Make session quota exhausted: can_start_new_session=False but can_send_any_turn=True
        mocks["stop_checker"].can_start_new_session.return_value = False
        mocks["stop_checker"].can_send_any_turn.return_value = True

        child_turn = _turn(conv="child-1", idx=0, num=1, agent_depth=1)
        result = await issuer.issue_credit(child_turn)

        # Child must succeed despite session quota being exhausted
        assert result is True
        mocks["router"].send_credit.assert_called_once()

    async def test_child_first_turn_blocked_by_old_code_would_deadlock(self) -> None:
        """Demonstrates the deadlock: session quota full, child can't start.

        Old code used can_start_new_session for child first turns. With
        expected_num_sessions reached, can_start_new_session returns False.
        The child issuance fails, parent gate never fires → deadlock.
        """
        issuer, mocks = _make_issuer()
        mocks["stop_checker"].can_start_new_session.return_value = False
        mocks["stop_checker"].can_send_any_turn.return_value = True

        child_turn = _turn(conv="child-1", idx=0, num=1, agent_depth=1)

        # try_issue_credit: old code would return False (stop condition),
        # new code returns True (child uses can_send_any_turn)
        result = await issuer.try_issue_credit(child_turn)
        assert result is True

    async def test_root_first_turn_still_uses_can_start_new_session(self) -> None:
        """Root first turn must still use can_start_new_session."""
        issuer, mocks = _make_issuer()
        root_turn = _turn(conv="root-1", idx=0, num=1, agent_depth=0)

        await issuer.issue_credit(root_turn)

        # Verify can_start_new_session was passed to acquire_session_slot
        call_args = mocks["concurrency"].acquire_session_slot.call_args
        assert call_args[0][1] == mocks["stop_checker"].can_start_new_session

    async def test_try_issue_child_returns_false_only_when_can_send_any_turn_false(
        self,
    ) -> None:
        """Child try_issue: returns False only when can_send_any_turn is False."""
        issuer, mocks = _make_issuer()
        mocks["stop_checker"].can_send_any_turn.return_value = False
        mocks["stop_checker"].can_start_new_session.return_value = False

        child_turn = _turn(conv="child-1", idx=0, num=1, agent_depth=1)
        result = await issuer.try_issue_credit(child_turn)

        assert result is False  # Correctly stopped by can_send_any_turn


# =============================================================================
# Bug 3: Children must NOT inflate session counts in CreditCounter
# =============================================================================


class TestChildSessionCountExclusion:
    """Without this fix, children increment sent_sessions which is compared
    against expected_num_sessions. In non-FIXED_SCHEDULE modes, expected_num_sessions
    is user-specified and doesn't account for children → premature quota
    exhaustion → is_final_credit fires too early or child issuance fails.
    """

    def test_child_first_turn_does_not_increment_sent_sessions(self) -> None:
        """Child first turn must NOT increment sent_sessions."""
        c = CreditCounter(_cfg(sessions=5))

        c.increment_sent(_turn(conv="root-1", idx=0, num=2, agent_depth=0))
        assert c.sent_sessions == 1

        c.increment_sent(_turn(conv="child-1", idx=0, num=3, agent_depth=1))
        assert c.sent_sessions == 1  # Still 1, child didn't count

        c.increment_sent(_turn(conv="root-2", idx=0, num=1, agent_depth=0))
        assert c.sent_sessions == 2  # Root incremented

    def test_child_first_turn_does_not_increment_total_session_turns(self) -> None:
        """Child first turn must NOT add to total_session_turns."""
        c = CreditCounter(_cfg(sessions=5))

        c.increment_sent(_turn(conv="root-1", idx=0, num=3, agent_depth=0))
        assert c.total_session_turns == 3

        c.increment_sent(_turn(conv="child-1", idx=0, num=5, agent_depth=1))
        assert c.total_session_turns == 3  # Child's turns not counted

    def test_premature_quota_exhaustion_prevented(self) -> None:
        """With expected_num_sessions=2, 2 roots + 3 children must not hit quota early.

        Old code: sent_sessions would reach 2 after just root-1 + child-1,
        triggering is_final_credit prematurely before root-2 is ever sent.
        """
        c = CreditCounter(_cfg(sessions=2))

        # Root 1: 2-turn conversation
        c.increment_sent(_turn(conv="root-1", idx=0, num=2, agent_depth=0))
        assert c.sent_sessions == 1

        # Child spawned by root-1
        _, is_final = c.increment_sent(
            _turn(conv="child-1a", idx=0, num=1, agent_depth=1)
        )
        assert not is_final  # Must NOT be final
        assert c.sent_sessions == 1  # Children don't count

        # Another child
        _, is_final = c.increment_sent(
            _turn(conv="child-1b", idx=0, num=2, agent_depth=1)
        )
        assert not is_final
        assert c.sent_sessions == 1

        # Root 1 turn 1
        c.increment_sent(_turn(conv="root-1", idx=1, num=2, agent_depth=0))

        # Child turns
        c.increment_sent(_turn(conv="child-1b", idx=1, num=2, agent_depth=1))
        c.increment_sent(_turn(conv="child-1a", idx=0, num=1, agent_depth=1))

        # Root 2: 1-turn conversation (this is the 2nd and final session)
        _, is_final = c.increment_sent(
            _turn(conv="root-2", idx=0, num=1, agent_depth=0)
        )
        assert c.sent_sessions == 2
        # Now is_final should be True (2 sessions sent, all turns sent)
        assert is_final

    def test_child_completion_does_not_increment_completed_sessions(self) -> None:
        """Child final turn return must NOT increment completed_sessions."""
        c = CreditCounter(_cfg())

        # Send root + child
        c.increment_sent(_turn(conv="root-1", idx=0, num=2, agent_depth=0))
        c.increment_sent(_turn(conv="child-1", idx=0, num=1, agent_depth=1))

        # Child completes (final turn, depth=1)
        c.increment_returned(is_final_turn=True, cancelled=False, agent_depth=1)
        assert c.completed_sessions == 0  # Child doesn't count

        # Root non-final turn completes
        c.increment_returned(is_final_turn=False, cancelled=False, agent_depth=0)
        assert c.completed_sessions == 0

        # Root final turn completes
        c.increment_returned(is_final_turn=True, cancelled=False, agent_depth=0)
        assert c.completed_sessions == 1  # Only root counts

    def test_child_cancellation_does_not_increment_cancelled_sessions(self) -> None:
        """Child final turn cancellation must NOT increment cancelled_sessions."""
        c = CreditCounter(_cfg())

        c.increment_sent(_turn(conv="root-1", idx=0, num=1, agent_depth=0))
        c.increment_sent(_turn(conv="child-1", idx=0, num=1, agent_depth=1))

        # Child cancelled
        c.increment_returned(is_final_turn=True, cancelled=True, agent_depth=1)
        assert c.cancelled_sessions == 0

        # Root cancelled
        c.increment_returned(is_final_turn=True, cancelled=True, agent_depth=0)
        assert c.cancelled_sessions == 1

    def test_requests_sent_still_counts_all_turns(self) -> None:
        """requests_sent must count ALL turns (root + child) for progress tracking."""
        c = CreditCounter(_cfg())

        c.increment_sent(_turn(conv="root-1", idx=0, num=2, agent_depth=0))
        c.increment_sent(_turn(conv="child-1", idx=0, num=1, agent_depth=1))
        c.increment_sent(_turn(conv="root-1", idx=1, num=2, agent_depth=0))

        assert c.requests_sent == 3  # All turns counted

    def test_requests_completed_still_counts_all_returns(self) -> None:
        """requests_completed must count ALL returns (root + child)."""
        c = CreditCounter(_cfg())

        c.increment_sent(_turn(conv="root-1", idx=0, num=1, agent_depth=0))
        c.increment_sent(_turn(conv="child-1", idx=0, num=1, agent_depth=1))

        c.increment_returned(is_final_turn=True, cancelled=False, agent_depth=1)
        c.increment_returned(is_final_turn=True, cancelled=False, agent_depth=0)

        assert c.requests_completed == 2  # Both counted

    def test_in_flight_sessions_excludes_children(self) -> None:
        """in_flight_sessions = sent_sessions - completed_sessions - cancelled_sessions.

        Since children are excluded from all three, in_flight_sessions should
        reflect only root sessions.
        """
        c = CreditCounter(_cfg())

        # Start 2 roots + 1 child
        c.increment_sent(_turn(conv="root-1", idx=0, num=2, agent_depth=0))
        c.increment_sent(_turn(conv="root-2", idx=0, num=1, agent_depth=0))
        c.increment_sent(_turn(conv="child-1", idx=0, num=1, agent_depth=1))

        assert c.in_flight_sessions == 2  # Only roots

        # Child completes
        c.increment_returned(is_final_turn=True, cancelled=False, agent_depth=1)
        assert c.in_flight_sessions == 2  # Unchanged

        # Root-2 completes
        c.increment_returned(is_final_turn=True, cancelled=False, agent_depth=0)
        assert c.in_flight_sessions == 1

    def test_freeze_sent_counts_only_reflects_root_sessions(self) -> None:
        """Frozen sent_sessions should only count root sessions."""
        c = CreditCounter(_cfg())

        c.increment_sent(_turn(conv="root-1", idx=0, num=1, agent_depth=0))
        c.increment_sent(_turn(conv="child-1", idx=0, num=1, agent_depth=1))
        c.increment_sent(_turn(conv="child-2", idx=0, num=1, agent_depth=2))
        c.increment_sent(_turn(conv="root-2", idx=0, num=1, agent_depth=0))

        c.freeze_sent_counts()

        assert c.final_sent_sessions == 2  # Only roots
        assert c.final_requests_sent == 4  # All requests


# =============================================================================
# Bug 4: Children must NOT release session slots in callback handler
# =============================================================================


class TestChildSkipsSessionSlotRelease:
    """Without this fix, child final turns call release_session_slot even
    though they never acquired one. This causes negative semaphore counts
    or double-releases that corrupt the session slot pool.
    """

    async def test_child_final_turn_does_not_release_session_slot(self) -> None:
        """Child final turn return must NOT release session slot."""
        mock_concurrency = MagicMock(
            release_session_slot=MagicMock(),
            release_prefill_slot=MagicMock(),
        )
        handler = CreditCallbackHandler(mock_concurrency)
        handler.register_phase(
            phase=CreditPhase.PROFILING,
            progress=MagicMock(
                increment_returned=MagicMock(return_value=False),
                increment_prefill_released=MagicMock(),
                in_flight_sessions=0,
                all_credits_returned_event=asyncio.Event(),
            ),
            lifecycle=MagicMock(is_complete=False),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
            strategy=MagicMock(handle_credit_return=AsyncMock()),
        )

        # Child final turn (depth=1)
        child_credit = _credit(conv="child-1", turn_index=0, num_turns=1, agent_depth=1)
        child_return = _credit_return(child_credit)

        await handler.on_credit_return("worker-1", child_return)

        mock_concurrency.release_session_slot.assert_not_called()

    async def test_root_final_turn_still_releases_session_slot(self) -> None:
        """Sanity: root final turn must still release session slot."""
        mock_concurrency = MagicMock(
            release_session_slot=MagicMock(),
            release_prefill_slot=MagicMock(),
        )
        handler = CreditCallbackHandler(mock_concurrency)
        handler.register_phase(
            phase=CreditPhase.PROFILING,
            progress=MagicMock(
                increment_returned=MagicMock(return_value=False),
                increment_prefill_released=MagicMock(),
                in_flight_sessions=0,
                all_credits_returned_event=asyncio.Event(),
            ),
            lifecycle=MagicMock(is_complete=False),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
            strategy=MagicMock(handle_credit_return=AsyncMock()),
        )

        # Root final turn (depth=0)
        root_credit = _credit(conv="root-1", turn_index=2, num_turns=3, agent_depth=0)
        root_return = _credit_return(root_credit)

        await handler.on_credit_return("worker-1", root_return)

        mock_concurrency.release_session_slot.assert_called_once_with(
            CreditPhase.PROFILING
        )

    async def test_mixed_root_child_returns_only_root_releases_session(self) -> None:
        """Interleaved root/child returns: only root final turns release session slots."""
        mock_concurrency = MagicMock(
            release_session_slot=MagicMock(),
            release_prefill_slot=MagicMock(),
        )
        handler = CreditCallbackHandler(mock_concurrency)
        handler.register_phase(
            phase=CreditPhase.PROFILING,
            progress=MagicMock(
                increment_returned=MagicMock(return_value=False),
                increment_prefill_released=MagicMock(),
                in_flight_sessions=0,
                all_credits_returned_event=asyncio.Event(),
            ),
            lifecycle=MagicMock(is_complete=False),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
            strategy=MagicMock(handle_credit_return=AsyncMock()),
        )

        returns = [
            # (conv, turn_index, num_turns, agent_depth, should_release_session)
            ("root-1", 0, 3, 0, False),  # root non-final
            ("child-a", 0, 1, 1, False),  # child final (depth=1)
            ("child-b", 0, 2, 2, False),  # child non-final (depth=2)
            ("child-b", 1, 2, 2, False),  # child final (depth=2)
            ("root-1", 1, 3, 0, False),  # root non-final
            ("root-1", 2, 3, 0, True),  # root final → release
            ("root-2", 0, 1, 0, True),  # root final → release
        ]

        for i, (conv, tidx, nturns, depth, _) in enumerate(returns):
            mock_concurrency.reset_mock()
            credit = _credit(
                credit_id=i,
                conv=conv,
                turn_index=tidx,
                num_turns=nturns,
                agent_depth=depth,
            )
            await handler.on_credit_return("worker-1", _credit_return(credit))

        # Count total session releases across all calls
        # We reset mocks between calls, so check the last call only confirms pattern.
        # Instead, let's do it without reset:
        mock_concurrency2 = MagicMock(
            release_session_slot=MagicMock(),
            release_prefill_slot=MagicMock(),
        )
        handler2 = CreditCallbackHandler(mock_concurrency2)
        handler2.register_phase(
            phase=CreditPhase.PROFILING,
            progress=MagicMock(
                increment_returned=MagicMock(return_value=False),
                increment_prefill_released=MagicMock(),
                in_flight_sessions=0,
                all_credits_returned_event=asyncio.Event(),
            ),
            lifecycle=MagicMock(is_complete=False),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
            strategy=MagicMock(handle_credit_return=AsyncMock()),
        )

        for i, (conv, tidx, nturns, depth, _) in enumerate(returns):
            credit = _credit(
                credit_id=i,
                conv=conv,
                turn_index=tidx,
                num_turns=nturns,
                agent_depth=depth,
            )
            await handler2.on_credit_return("worker-1", _credit_return(credit))

        # Exactly 2 session slot releases (root-1 final + root-2 final)
        assert mock_concurrency2.release_session_slot.call_count == 2

    async def test_child_return_passes_agent_depth_to_increment_returned(self) -> None:
        """Callback handler must pass agent_depth to progress.increment_returned."""
        mock_progress = MagicMock(
            increment_returned=MagicMock(return_value=False),
            increment_prefill_released=MagicMock(),
            in_flight_sessions=0,
            all_credits_returned_event=asyncio.Event(),
        )
        handler = CreditCallbackHandler(MagicMock())
        handler.register_phase(
            phase=CreditPhase.PROFILING,
            progress=mock_progress,
            lifecycle=MagicMock(is_complete=False),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
            strategy=MagicMock(handle_credit_return=AsyncMock()),
        )

        for depth in [0, 1, 2, 5]:
            mock_progress.reset_mock()
            credit = _credit(conv=f"d{depth}", agent_depth=depth)
            await handler.on_credit_return("worker-1", _credit_return(credit))

            mock_progress.increment_returned.assert_called_once_with(
                credit.is_final_turn,
                False,
                agent_depth=depth,
            )


# =============================================================================
# Integration: Full scenario that would deadlock old code
# =============================================================================


class TestDeadlockScenario:
    """End-to-end scenario using real CreditCounter that demonstrates the
    deadlock the old code would produce.

    Setup: expected_num_sessions=2, concurrency=2
    Trace: root-1 (3 turns) spawns child-a (2 turns) at turn 1, root-2 (1 turn)

    Old behavior:
    - root-1 turn 0: sent_sessions=1, acquires session slot
    - child-a turn 0: sent_sessions=2 (BUG), acquires session slot (BUG)
    - root-2 turn 0: can_start_new_session=False (sessions=2 >= limit=2)
      AND no session slots available (2/2 used) → DEADLOCK
    """

    def test_counter_does_not_exhaust_session_quota_with_children(self) -> None:
        """CreditCounter: children don't count toward session quota."""
        c = CreditCounter(_cfg(sessions=2))

        # Root 1 starts
        _, is_final = c.increment_sent(
            _turn(conv="root-1", idx=0, num=3, agent_depth=0)
        )
        assert not is_final
        assert c.sent_sessions == 1

        # Child spawned by root-1
        _, is_final = c.increment_sent(
            _turn(conv="child-a", idx=0, num=2, agent_depth=1)
        )
        assert not is_final
        assert c.sent_sessions == 1  # CRITICAL: still 1

        # Root 1 turn 1
        c.increment_sent(_turn(conv="root-1", idx=1, num=3, agent_depth=0))

        # Child turn 1
        c.increment_sent(_turn(conv="child-a", idx=1, num=2, agent_depth=1))

        # Root 2 starts -- this is the 2nd and final session
        _, is_final = c.increment_sent(
            _turn(conv="root-2", idx=0, num=1, agent_depth=0)
        )
        assert c.sent_sessions == 2

        # Root 1 turn 2 (final turn of root-1)
        _, is_final = c.increment_sent(
            _turn(conv="root-1", idx=2, num=3, agent_depth=0)
        )
        # Now: 2 sessions started, total_session_turns=3+1=4, requests_sent=6
        # is_final should be True (sessions >= 2 AND requests >= total_turns)
        assert is_final

    async def test_issuer_allows_child_when_session_quota_full(self) -> None:
        """CreditIssuer: child issuance succeeds even when session quota is full."""
        issuer, mocks = _make_issuer()

        # Simulate: session quota is full
        mocks["stop_checker"].can_start_new_session.return_value = False
        mocks["stop_checker"].can_send_any_turn.return_value = True

        # Child must still be issuable
        child_turn = _turn(conv="child-a", idx=0, num=2, agent_depth=1)
        result = await issuer.issue_credit(child_turn)

        assert result is True
        mocks["concurrency"].acquire_session_slot.assert_not_called()
        mocks["router"].send_credit.assert_called_once()

        # Verify the credit has correct agent_depth
        sent_credit = mocks["router"].send_credit.call_args.kwargs["credit"]
        assert sent_credit.agent_depth == 1
