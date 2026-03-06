# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for UserSessionManager to ensure Credit.num_turns is respected.

These tests ensure that the worker properly uses Credit.num_turns instead of
len(conversation.turns), which is critical for ramp-up users who start mid-session.
"""

import pytest
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Turn
from aiperf.workers.session_manager import UserSession, UserSessionManager


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


# ============================================================
# Fixtures for context mode tests
# ============================================================


def _make_session(
    context_mode: ConversationContextMode | None = None,
    num_turns: int = 3,
    default_context_mode: ConversationContextMode | None = None,
) -> UserSession:
    """Create a UserSession with the given context_mode on its conversation."""
    conversation = Conversation(
        conversation_id="ctx-conv",
        context_mode=context_mode,
        turns=[
            Turn(messages=[{"role": "user", "content": f"Q{i}"}])
            for i in range(num_turns)
        ],
    )
    mgr = UserSessionManager()
    mgr.set_default_context_mode(default_context_mode)
    return mgr.create_and_store(
        x_correlation_id="ctx-test",
        conversation=conversation,
        num_turns=num_turns,
    )


# ============================================================
# Context Mode Resolution
# ============================================================


class TestUserSessionContextModeResolution:
    """Verify context_mode resolves: conversation > dataset default > ACCUMULATE_ALL."""

    @pytest.mark.parametrize(
        "conversation_mode,expected",
        [
            (None, ConversationContextMode.ACCUMULATE_ALL),
            (ConversationContextMode.ACCUMULATE_ALL, ConversationContextMode.ACCUMULATE_ALL),
            (ConversationContextMode.DROP_RESPONSES, ConversationContextMode.DROP_RESPONSES),
            (ConversationContextMode.STANDALONE, ConversationContextMode.STANDALONE),
        ],
    )  # fmt: skip
    def test_context_mode_resolves_correctly(
        self,
        conversation_mode: ConversationContextMode | None,
        expected: ConversationContextMode,
    ) -> None:
        session = _make_session(context_mode=conversation_mode)
        assert session.context_mode == expected

    def test_dataset_default_used_when_conversation_has_none(self) -> None:
        session = _make_session(
            context_mode=None,
            default_context_mode=ConversationContextMode.STANDALONE,
        )
        assert session.context_mode == ConversationContextMode.STANDALONE

    def test_conversation_overrides_dataset_default(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.DROP_RESPONSES,
            default_context_mode=ConversationContextMode.STANDALONE,
        )
        assert session.context_mode == ConversationContextMode.DROP_RESPONSES

    def test_global_default_when_both_none(self) -> None:
        session = _make_session(context_mode=None, default_context_mode=None)
        assert session.context_mode == ConversationContextMode.ACCUMULATE_ALL


# ============================================================
# should_store_response
# ============================================================


class TestUserSessionShouldStoreResponse:
    """Verify should_store_response gates on context mode."""

    @pytest.mark.parametrize(
        "mode,expected",
        [
            (ConversationContextMode.ACCUMULATE_ALL, True),
            (ConversationContextMode.DROP_RESPONSES, False),
            (ConversationContextMode.STANDALONE, False),
            param(None, True, id="default-accumulate-all"),
        ],
    )  # fmt: skip
    def test_should_store_response_per_mode(
        self, mode: ConversationContextMode | None, expected: bool
    ) -> None:
        session = _make_session(context_mode=mode)
        assert session.should_store_response() is expected


# ============================================================
# turn_list with context mode
# ============================================================


class TestUserSessionTurnList:
    """Verify turn_list contains correct turns based on context mode."""

    def test_accumulate_all_returns_full_history(self) -> None:
        session = _make_session(context_mode=ConversationContextMode.ACCUMULATE_ALL)
        session.advance_turn(0)
        session.store_response(Turn(messages=[{"role": "assistant", "content": "A0"}]))
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 3  # Q0, A0, Q1
        assert turns[0].messages[0]["content"] == "Q0"
        assert turns[1].messages[0]["content"] == "A0"
        assert turns[2].messages[0]["content"] == "Q1"

    def test_drop_responses_returns_full_history(self) -> None:
        session = _make_session(context_mode=ConversationContextMode.DROP_RESPONSES)
        session.advance_turn(0)
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 2  # Q0, Q1 (no assistant responses stored)
        assert turns[0].messages[0]["content"] == "Q0"
        assert turns[1].messages[0]["content"] == "Q1"

    def test_standalone_returns_only_last(self) -> None:
        session = _make_session(context_mode=ConversationContextMode.STANDALONE)
        session.advance_turn(0)
        session.advance_turn(1)
        session.advance_turn(2)

        turns = session.turn_list
        assert len(turns) == 1
        assert turns[0].messages[0]["content"] == "Q2"

    def test_standalone_single_turn(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.STANDALONE, num_turns=1
        )
        session.advance_turn(0)

        turns = session.turn_list
        assert len(turns) == 1
        assert turns[0].messages[0]["content"] == "Q0"

    def test_default_mode_returns_full_history(self) -> None:
        session = _make_session(context_mode=None)
        session.advance_turn(0)
        session.store_response(Turn(messages=[{"role": "assistant", "content": "A0"}]))
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 3


# ============================================================
# Integration: context mode + should_store_response together
# ============================================================


class TestUserSessionContextModeWorkflow:
    """Verify the full workflow of context mode with store_response gating."""

    def test_accumulate_all_stores_responses_and_sends_full_history(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.ACCUMULATE_ALL, num_turns=2
        )
        session.advance_turn(0)
        assert session.should_store_response() is True
        session.store_response(Turn(messages=[{"role": "assistant", "content": "A0"}]))
        session.advance_turn(1)

        assert len(session.turn_list) == 3

    def test_drop_responses_skips_responses_sends_user_turns_only(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.DROP_RESPONSES, num_turns=2
        )
        session.advance_turn(0)
        assert session.should_store_response() is False
        # Worker would NOT call store_response based on should_store_response()
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 2
        assert all(t.messages[0]["role"] == "user" for t in turns)

    def test_standalone_skips_responses_sends_only_current_turn(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.STANDALONE, num_turns=2
        )
        session.advance_turn(0)
        assert session.should_store_response() is False
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 1
        assert turns[0].messages[0]["content"] == "Q1"
