# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for UserSessionManager.

Tests ensure:
- Credit.num_turns is respected (ramp-up users who start mid-session)
- TurnThreadingMode controls history accumulation and response storage
"""

import pytest

from aiperf.common.enums import TurnThreadingMode
from aiperf.common.models import Conversation, Turn
from aiperf.workers.session_manager import UserSessionManager


@pytest.fixture
def session_manager():
    """Create a UserSessionManager instance."""
    return UserSessionManager()


@pytest.fixture
def sample_conversation():
    """Create a sample conversation with 5 turns."""
    return Conversation(
        conversation_id="test-conv",
        turns=[
            Turn(messages=[{"role": "user", "content": f"Question {i + 1}"}])
            for i in range(5)
        ],
    )


class TestUserSessionManager:
    """Tests for UserSessionManager Credit.num_turns handling."""

    def test_create_session_uses_credit_num_turns_not_conversation_length(
        self, session_manager, sample_conversation
    ):
        """Ensure UserSession.num_turns comes from Credit, not conversation.

        This is critical for ramp-up users who may only execute 1 turn even though
        the conversation template has 5 turns available.
        """
        # Conversation has 5 turns, but Credit says only do 1
        session = session_manager.create_and_store(
            x_correlation_id="test-corr-id",
            conversation=sample_conversation,
            num_turns=1,  # Artificial cap from Credit
        )

        # UserSession should use Credit.num_turns (1), not len(conversation.turns) (5)
        assert session.num_turns == 1
        assert len(session.conversation.turns) == 5  # Conversation still has all turns

    def test_advance_turn_validates_against_credit_num_turns(
        self, session_manager, sample_conversation
    ):
        """Ensure turn validation uses Credit.num_turns."""
        session = session_manager.create_and_store(
            x_correlation_id="test-corr-id",
            conversation=sample_conversation,
            num_turns=2,  # Only 2 turns allowed
        )

        # Should be able to advance to turn 0 and 1
        session.advance_turn(0)
        assert session.turn_index == 0

        session.advance_turn(1)
        assert session.turn_index == 1

        # Should reject turn 2 (out of range for num_turns=2)
        with pytest.raises(
            ValueError,
            match="Turn index 2 is out of range for conversation with 2 turns",
        ):
            session.advance_turn(2)

    def test_ramp_up_user_single_turn_scenario(
        self, session_manager, sample_conversation
    ):
        """Test ramp-up user who only executes 1 turn (e.g., User 1 starting at Turn 5).

        This simulates multi-round-qa's ramp-up behavior where some users are
        initialized mid-session and only complete their final turn.
        """
        # User 1 in ramp-up: starts at question_id=5, only does 1 turn
        session = session_manager.create_and_store(
            x_correlation_id="ramp-up-user-1",
            conversation=sample_conversation,
            num_turns=1,  # Only 1 turn to execute
        )

        # Advance to turn 0 (their only turn)
        turn = session.advance_turn(0)

        # Should access first turn of conversation (conversation has all 5 turns available)
        assert turn.messages[0]["content"] == "Question 1"

        # After turn 0, is_final_turn should be True (0 == 1-1)
        # This would be determined by Credit.is_final_turn, which we validate here
        assert session.turn_index == 0
        assert session.num_turns == 1
        # Credit.is_final_turn would be: turn_index (0) == num_turns (1) - 1 → True

    def test_full_session_uses_all_conversation_turns(
        self, session_manager, sample_conversation
    ):
        """Test normal user who executes all turns (e.g., steady-state users)."""
        session = session_manager.create_and_store(
            x_correlation_id="full-session-user",
            conversation=sample_conversation,
            num_turns=5,  # All turns
        )

        assert session.num_turns == 5

        # Should be able to advance through all 5 turns
        for turn_idx in range(5):
            turn = session.advance_turn(turn_idx)
            assert turn.messages[0]["content"] == f"Question {turn_idx + 1}"

    def test_partial_session_mid_conversation(
        self, session_manager, sample_conversation
    ):
        """Test user who starts mid-session and does partial turns (e.g., User 4 doing 3 turns)."""
        session = session_manager.create_and_store(
            x_correlation_id="partial-user",
            conversation=sample_conversation,
            num_turns=3,  # Only 3 turns (simulating User 4 at question_id=3)
        )

        assert session.num_turns == 3

        # Can advance turns 0, 1, 2
        for turn_idx in range(3):
            turn = session.advance_turn(turn_idx)
            assert turn is not None

        # Turn 3 should fail (out of range)
        with pytest.raises(ValueError, match="out of range"):
            session.advance_turn(3)

    def test_url_index_stored_for_multi_url_load_balancing(
        self, session_manager, sample_conversation
    ):
        """Test that url_index is stored in session for multi-URL load balancing.

        When using multiple --url endpoints with multi-turn conversations, the first
        turn gets a url_index from the round-robin sampler. All subsequent turns must
        use the same url_index to ensure the entire conversation hits the same backend.
        """
        # First turn: Credit provides url_index=2 from round-robin
        session = session_manager.create_and_store(
            x_correlation_id="multi-url-session",
            conversation=sample_conversation,
            num_turns=3,
            url_index=2,  # From Credit on first turn
        )

        # Session stores the url_index for subsequent turns
        assert session.url_index == 2

        # All turns should use this stored url_index (worker reads from session)
        for turn_idx in range(3):
            session.advance_turn(turn_idx)
            # Worker would use session.url_index (2) for every turn
            assert session.url_index == 2

    def test_url_index_none_for_single_url_mode(
        self, session_manager, sample_conversation
    ):
        """Test that url_index can be None when only one URL is configured."""
        session = session_manager.create_and_store(
            x_correlation_id="single-url-session",
            conversation=sample_conversation,
            num_turns=2,
            url_index=None,  # No multi-URL load balancing
        )

        assert session.url_index is None


# =========================================================================
# TurnThreadingMode resolution tests
# =========================================================================


class TestTurnThreadingModeResolution:
    """Tests for TurnThreadingMode precedence: conversation > dataset default > global."""

    @pytest.fixture
    def session_manager(self):
        return UserSessionManager()

    @pytest.fixture
    def conversation_3_turns(self):
        return Conversation(
            session_id="mode-test",
            turns=[
                Turn(texts=[], role="user"),
                Turn(texts=[], role="user"),
                Turn(texts=[], role="user"),
            ],
        )

    @pytest.mark.parametrize(
        "mode",
        [
            TurnThreadingMode.THREAD_ASSISTANT_RESPONSES,
            TurnThreadingMode.SKIP_ASSISTANT_RESPONSES,
            TurnThreadingMode.ISOLATED_TURNS,
        ],
    )
    def test_conversation_override_takes_precedence(
        self, session_manager, conversation_3_turns, mode
    ):
        conversation_3_turns.turn_threading_mode = mode
        session = session_manager.create_and_store(
            x_correlation_id="c1",
            conversation=conversation_3_turns,
            num_turns=3,
        )
        assert session.turn_threading_mode == mode

    def test_dataset_default_used_when_conversation_has_none(
        self, session_manager, conversation_3_turns
    ):
        session_manager.set_default_threading_mode(TurnThreadingMode.ISOLATED_TURNS)
        session = session_manager.create_and_store(
            x_correlation_id="c2",
            conversation=conversation_3_turns,
            num_turns=3,
        )
        assert session.turn_threading_mode == TurnThreadingMode.ISOLATED_TURNS

    def test_conversation_overrides_dataset_default(
        self, session_manager, conversation_3_turns
    ):
        session_manager.set_default_threading_mode(
            TurnThreadingMode.SKIP_ASSISTANT_RESPONSES
        )
        conversation_3_turns.turn_threading_mode = (
            TurnThreadingMode.THREAD_ASSISTANT_RESPONSES
        )
        session = session_manager.create_and_store(
            x_correlation_id="c3",
            conversation=conversation_3_turns,
            num_turns=3,
        )
        assert (
            session.turn_threading_mode == TurnThreadingMode.THREAD_ASSISTANT_RESPONSES
        )

    def test_global_default_when_both_none(self, session_manager, conversation_3_turns):
        session = session_manager.create_and_store(
            x_correlation_id="c4",
            conversation=conversation_3_turns,
            num_turns=3,
        )
        assert (
            session.turn_threading_mode == TurnThreadingMode.THREAD_ASSISTANT_RESPONSES
        )


# =========================================================================
# should_store_response tests
# =========================================================================


class TestShouldStoreResponse:
    """Tests for should_store_response() gating per threading mode."""

    @pytest.fixture
    def session_manager(self):
        return UserSessionManager()

    @pytest.fixture
    def conversation_3_turns(self):
        return Conversation(
            session_id="store-test",
            turns=[
                Turn(texts=[], role="user"),
                Turn(texts=[], role="user"),
                Turn(texts=[], role="user"),
            ],
        )

    @pytest.mark.parametrize(
        "mode,expected",
        [
            (TurnThreadingMode.THREAD_ASSISTANT_RESPONSES, True),
            (TurnThreadingMode.SKIP_ASSISTANT_RESPONSES, False),
            (TurnThreadingMode.ISOLATED_TURNS, False),
        ],
    )
    def test_store_response_per_mode(
        self, session_manager, conversation_3_turns, mode, expected
    ):
        conversation_3_turns.turn_threading_mode = mode
        session = session_manager.create_and_store(
            x_correlation_id="sr1",
            conversation=conversation_3_turns,
            num_turns=3,
        )
        assert session.should_store_response() is expected


# =========================================================================
# Turn list accumulation tests per threading mode
# =========================================================================


class TestTurnListAccumulation:
    """Tests for turn_list contents under each threading mode."""

    @pytest.fixture
    def session_manager(self):
        return UserSessionManager()

    @pytest.fixture
    def conversation_3_turns(self):
        return Conversation(
            session_id="acc-test",
            turns=[
                Turn(texts=[], role="user"),
                Turn(texts=[], role="user"),
                Turn(texts=[], role="user"),
            ],
        )

    def _make_response_turn(self) -> Turn:
        return Turn(texts=[], role="assistant")

    def test_thread_assistant_responses_full_history(
        self, session_manager, conversation_3_turns
    ):
        """THREAD_ASSISTANT_RESPONSES: turn_list accumulates all user + assistant turns."""
        conversation_3_turns.turn_threading_mode = (
            TurnThreadingMode.THREAD_ASSISTANT_RESPONSES
        )
        session = session_manager.create_and_store(
            x_correlation_id="ta1",
            conversation=conversation_3_turns,
            num_turns=3,
        )
        session.advance_turn(0)
        session.store_response(self._make_response_turn())
        session.advance_turn(1)
        session.store_response(self._make_response_turn())
        session.advance_turn(2)

        assert len(session.turn_list) == 5  # Q0, A0, Q1, A1, Q2

    def test_skip_assistant_responses_user_turns_only(
        self, session_manager, conversation_3_turns
    ):
        """SKIP_ASSISTANT_RESPONSES: turn_list has user turns only (responses not stored)."""
        conversation_3_turns.turn_threading_mode = (
            TurnThreadingMode.SKIP_ASSISTANT_RESPONSES
        )
        session = session_manager.create_and_store(
            x_correlation_id="sa1",
            conversation=conversation_3_turns,
            num_turns=3,
        )
        session.advance_turn(0)
        # should_store_response() is False, so worker would skip store_response
        assert not session.should_store_response()
        session.advance_turn(1)
        session.advance_turn(2)

        assert len(session.turn_list) == 3  # Q0, Q1, Q2

    def test_isolated_turns_only_current(self, session_manager, conversation_3_turns):
        """ISOLATED_TURNS: turn_list always has exactly the current turn."""
        conversation_3_turns.turn_threading_mode = TurnThreadingMode.ISOLATED_TURNS
        session = session_manager.create_and_store(
            x_correlation_id="it1",
            conversation=conversation_3_turns,
            num_turns=3,
        )
        session.advance_turn(0)
        assert len(session.turn_list) == 1
        first = session.turn_list[0]

        session.advance_turn(1)
        assert len(session.turn_list) == 1
        assert session.turn_list[0] is not first  # replaced, not accumulated

        session.advance_turn(2)
        assert len(session.turn_list) == 1


# =========================================================================
# Integration: threading mode + response storage workflow
# =========================================================================


class TestThreadingModeWorkflow:
    """Integration tests combining threading mode with should_store_response."""

    @pytest.fixture
    def session_manager(self):
        return UserSessionManager()

    @pytest.fixture
    def conversation_3_turns(self):
        return Conversation(
            session_id="wf-test",
            turns=[
                Turn(texts=[], role="user"),
                Turn(texts=[], role="user"),
                Turn(texts=[], role="user"),
            ],
        )

    def _make_response_turn(self) -> Turn:
        return Turn(texts=[], role="assistant")

    def test_accumulate_stores_responses_and_sends_full_history(
        self, session_manager, conversation_3_turns
    ):
        conversation_3_turns.turn_threading_mode = (
            TurnThreadingMode.THREAD_ASSISTANT_RESPONSES
        )
        session = session_manager.create_and_store(
            x_correlation_id="wf1",
            conversation=conversation_3_turns,
            num_turns=2,
        )

        session.advance_turn(0)
        assert session.should_store_response()
        session.store_response(self._make_response_turn())
        assert len(session.turn_list) == 2  # Q0, A0

        session.advance_turn(1)
        assert len(session.turn_list) == 3  # Q0, A0, Q1

    def test_skip_responses_sends_user_turns_only(
        self, session_manager, conversation_3_turns
    ):
        conversation_3_turns.turn_threading_mode = (
            TurnThreadingMode.SKIP_ASSISTANT_RESPONSES
        )
        session = session_manager.create_and_store(
            x_correlation_id="wf2",
            conversation=conversation_3_turns,
            num_turns=2,
        )

        session.advance_turn(0)
        assert not session.should_store_response()
        # Worker skips store_response
        assert len(session.turn_list) == 1  # Q0

        session.advance_turn(1)
        assert len(session.turn_list) == 2  # Q0, Q1

    def test_standalone_sends_only_current_turn(
        self, session_manager, conversation_3_turns
    ):
        conversation_3_turns.turn_threading_mode = TurnThreadingMode.ISOLATED_TURNS
        session = session_manager.create_and_store(
            x_correlation_id="wf3",
            conversation=conversation_3_turns,
            num_turns=2,
        )

        session.advance_turn(0)
        assert not session.should_store_response()
        assert len(session.turn_list) == 1

        session.advance_turn(1)
        assert len(session.turn_list) == 1
