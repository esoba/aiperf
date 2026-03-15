# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pre-generated response support in BaseDatasetComposer.

Focuses on:
- _apply_pregenerated_responses: multi-turn raw_messages construction
- _extract_turn_text: text extraction from turns
- _finalize_conversation integration with pre_generate_responses flag
"""

from unittest.mock import patch

import pytest

from aiperf.common.config import (
    ConversationConfig,
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    OutputTokensConfig,
    PrefixPromptConfig,
    PromptConfig,
    TurnConfig,
    UserConfig,
)
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.composer.base import BaseDatasetComposer


class ConcreteComposer(BaseDatasetComposer):
    """Concrete test implementation of BaseDatasetComposer."""

    def create_dataset(self):
        return []


def _make_config(
    *,
    pre_generate: bool = False,
    num_turns: int = 3,
    output_mean: int = 50,
    system_prompt_len: int | None = None,
    user_context_len: int | None = None,
) -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig(
            conversation=ConversationConfig(
                num_dataset_entries=2,
                turn=TurnConfig(mean=num_turns, stddev=0),
            ),
            prompt=PromptConfig(
                input_tokens=InputTokensConfig(mean=10, stddev=2),
                output_tokens=OutputTokensConfig(mean=output_mean, stddev=0),
                pre_generate_responses=pre_generate,
                prefix_prompt=PrefixPromptConfig(
                    pool_size=0,
                    length=0,
                    shared_system_prompt_length=system_prompt_len,
                    user_context_prompt_length=user_context_len,
                ),
            ),
        ),
    )


def _make_turn(text: str, *, max_tokens: int = 50) -> Turn:
    return Turn(
        texts=[Text(contents=[text])],
        role="user",
        model="test-model",
        max_tokens=max_tokens,
    )


def _single_turn_conversation() -> Conversation:
    return Conversation(
        session_id="single",
        turns=[_make_turn("Only turn")],
    )


def _multi_turn_conversation(n: int = 3) -> Conversation:
    return Conversation(
        session_id="multi",
        turns=[_make_turn(f"Turn {i} content", max_tokens=50) for i in range(n)],
    )


# ============================================================
# _apply_pregenerated_responses - Happy Path
# ============================================================


class TestApplyPregeneratedResponsesHappyPath:
    """Verify raw_messages construction for multi-turn conversations."""

    @pytest.fixture
    def composer(self, mock_tokenizer) -> ConcreteComposer:
        config = _make_config(pre_generate=True)
        return ConcreteComposer(config, mock_tokenizer)

    def test_single_turn_not_modified(self, composer: ConcreteComposer) -> None:
        conv = _single_turn_conversation()
        composer._apply_pregenerated_responses(conv)
        assert conv.turns[0].raw_messages is None

    def test_turn_zero_never_gets_raw_messages(
        self, composer: ConcreteComposer
    ) -> None:
        conv = _multi_turn_conversation(3)
        composer._apply_pregenerated_responses(conv)
        assert conv.turns[0].raw_messages is None

    def test_subsequent_turns_get_raw_messages(
        self, composer: ConcreteComposer
    ) -> None:
        conv = _multi_turn_conversation(3)
        composer._apply_pregenerated_responses(conv)
        for i in range(1, len(conv.turns)):
            assert conv.turns[i].raw_messages is not None
            assert isinstance(conv.turns[i].raw_messages, list)
            assert len(conv.turns[i].raw_messages) > 0

    def test_raw_messages_last_entry_is_current_user_turn(
        self, composer: ConcreteComposer
    ) -> None:
        conv = _multi_turn_conversation(3)
        composer._apply_pregenerated_responses(conv)
        for i in range(1, len(conv.turns)):
            last_msg = conv.turns[i].raw_messages[-1]
            assert last_msg["role"] == "user"
            assert f"Turn {i} content" in last_msg["content"]

    def test_raw_messages_contain_prior_user_turns(
        self, composer: ConcreteComposer
    ) -> None:
        conv = _multi_turn_conversation(4)
        composer._apply_pregenerated_responses(conv)
        # Turn 2 should have user messages for each prior turn j (0, 1) plus itself (2).
        # _extract_turn_text reads raw_messages once set, so prior turns
        # that already have raw_messages return their first user content.
        user_msgs = [m for m in conv.turns[2].raw_messages if m.get("role") == "user"]
        # Should have 3 user messages: one per prior turn (j=0, j=1) plus current (i=2)
        assert len(user_msgs) == 3
        # Current turn content is always last
        assert "Turn 2 content" in user_msgs[-1]["content"]

    def test_raw_messages_contain_assistant_responses(
        self, composer: ConcreteComposer
    ) -> None:
        conv = _multi_turn_conversation(3)
        composer._apply_pregenerated_responses(conv)
        # Turn 1 should have at least one assistant message (response to turn 0)
        assistant_msgs = [
            m for m in conv.turns[1].raw_messages if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) >= 1

    def test_turn_i_has_more_history_than_turn_i_minus_1(
        self, composer: ConcreteComposer
    ) -> None:
        conv = _multi_turn_conversation(4)
        composer._apply_pregenerated_responses(conv)
        for i in range(2, len(conv.turns)):
            assert len(conv.turns[i].raw_messages) > len(conv.turns[i - 1].raw_messages)

    def test_two_turn_conversation(self, composer: ConcreteComposer) -> None:
        conv = _multi_turn_conversation(2)
        composer._apply_pregenerated_responses(conv)
        assert conv.turns[0].raw_messages is None
        assert conv.turns[1].raw_messages is not None
        # Turn 1 messages: user turn 0 + assistant response + current user turn 1
        user_msgs = [m for m in conv.turns[1].raw_messages if m.get("role") == "user"]
        assert len(user_msgs) >= 2


# ============================================================
# _apply_pregenerated_responses - Context Messages
# ============================================================


class TestApplyPregeneratedResponsesContext:
    """Verify system_message and user_context_message inclusion."""

    def test_system_message_included(self, mock_tokenizer) -> None:
        config = _make_config(pre_generate=True)
        composer = ConcreteComposer(config, mock_tokenizer)
        conv = _multi_turn_conversation(2)
        conv.system_message = "You are a coding assistant."
        composer._apply_pregenerated_responses(conv)
        first_msg = conv.turns[1].raw_messages[0]
        assert first_msg["role"] == "system"
        assert first_msg["content"] == "You are a coding assistant."

    def test_user_context_message_included(self, mock_tokenizer) -> None:
        config = _make_config(pre_generate=True)
        composer = ConcreteComposer(config, mock_tokenizer)
        conv = _multi_turn_conversation(2)
        conv.user_context_message = "Project context here."
        composer._apply_pregenerated_responses(conv)
        msgs = conv.turns[1].raw_messages
        user_context = [
            m
            for m in msgs
            if m.get("role") == "user" and m["content"] == "Project context here."
        ]
        assert len(user_context) == 1

    def test_both_context_messages_included(self, mock_tokenizer) -> None:
        config = _make_config(pre_generate=True)
        composer = ConcreteComposer(config, mock_tokenizer)
        conv = _multi_turn_conversation(2)
        conv.system_message = "system msg"
        conv.user_context_message = "user context"
        composer._apply_pregenerated_responses(conv)
        msgs = conv.turns[1].raw_messages
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "system msg"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "user context"

    def test_no_context_messages_when_absent(self, mock_tokenizer) -> None:
        config = _make_config(pre_generate=True)
        composer = ConcreteComposer(config, mock_tokenizer)
        conv = _multi_turn_conversation(2)
        composer._apply_pregenerated_responses(conv)
        msgs = conv.turns[1].raw_messages
        # No system or user-context prefix; first message should be user turn 0
        assert msgs[0]["role"] == "user"


# ============================================================
# _apply_pregenerated_responses - Edge Cases
# ============================================================


class TestApplyPregeneratedResponsesEdgeCases:
    """Boundary conditions and skip-if-already-set behavior."""

    @pytest.fixture
    def composer(self, mock_tokenizer) -> ConcreteComposer:
        config = _make_config(pre_generate=True)
        return ConcreteComposer(config, mock_tokenizer)

    def test_empty_turns_no_crash(self, composer: ConcreteComposer) -> None:
        conv = Conversation(session_id="empty", turns=[])
        composer._apply_pregenerated_responses(conv)

    def test_existing_raw_messages_left_untouched(
        self, composer: ConcreteComposer
    ) -> None:
        conv = _multi_turn_conversation(3)
        existing = [{"role": "user", "content": "preexisting"}]
        conv.turns[1].raw_messages = existing
        composer._apply_pregenerated_responses(conv)
        # Turn 1 should keep preexisting raw_messages
        assert conv.turns[1].raw_messages is existing
        # Turn 2 should still get generated raw_messages
        assert conv.turns[2].raw_messages is not None
        assert conv.turns[2].raw_messages is not existing

    def test_turn_with_zero_max_tokens_no_response(
        self, composer: ConcreteComposer
    ) -> None:
        """When a turn has max_tokens=0, no response is generated for it."""
        conv = Conversation(
            session_id="zero_mt",
            turns=[
                Turn(texts=[Text(contents=["t0"])], role="user", max_tokens=0),
                Turn(texts=[Text(contents=["t1"])], role="user", max_tokens=50),
                Turn(texts=[Text(contents=["t2"])], role="user", max_tokens=50),
            ],
        )
        composer._apply_pregenerated_responses(conv)
        # Should not crash; turn 1 and 2 still get raw_messages
        assert conv.turns[1].raw_messages is not None
        assert conv.turns[2].raw_messages is not None

    def test_turn_with_none_max_tokens_no_response(
        self, composer: ConcreteComposer
    ) -> None:
        """When a turn has max_tokens=None, no response is generated for it."""
        conv = Conversation(
            session_id="none_mt",
            turns=[
                Turn(texts=[Text(contents=["t0"])], role="user", max_tokens=None),
                Turn(texts=[Text(contents=["t1"])], role="user", max_tokens=50),
            ],
        )
        composer._apply_pregenerated_responses(conv)
        assert conv.turns[1].raw_messages is not None


# ============================================================
# _extract_turn_text
# ============================================================


class TestExtractTurnText:
    """Verify text extraction from Turn objects."""

    def test_extracts_from_texts_field(self) -> None:
        turn = Turn(texts=[Text(contents=["hello ", "world"])])
        result = BaseDatasetComposer._extract_turn_text(turn)
        assert result == "hello world"

    def test_extracts_from_raw_messages_user_content(self) -> None:
        turn = Turn(
            raw_messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "the user said this"},
                {"role": "assistant", "content": "reply"},
            ]
        )
        result = BaseDatasetComposer._extract_turn_text(turn)
        assert result == "the user said this"

    def test_empty_turn_returns_empty_string(self) -> None:
        turn = Turn()
        result = BaseDatasetComposer._extract_turn_text(turn)
        assert result == ""

    def test_multiple_texts_concatenated(self) -> None:
        turn = Turn(
            texts=[
                Text(contents=["first "]),
                Text(contents=["second"]),
            ]
        )
        result = BaseDatasetComposer._extract_turn_text(turn)
        assert result == "first second"

    def test_raw_messages_takes_priority_over_texts(self) -> None:
        turn = Turn(
            texts=[Text(contents=["from texts"])],
            raw_messages=[
                {"role": "user", "content": "from raw_messages"},
            ],
        )
        result = BaseDatasetComposer._extract_turn_text(turn)
        assert result == "from raw_messages"

    def test_raw_messages_no_user_returns_empty(self) -> None:
        turn = Turn(
            raw_messages=[
                {"role": "assistant", "content": "only assistant"},
            ]
        )
        result = BaseDatasetComposer._extract_turn_text(turn)
        assert result == ""


# ============================================================
# _finalize_conversation integration
# ============================================================


class TestFinalizeConversationIntegration:
    """Verify _finalize_conversation calls _apply_pregenerated_responses correctly."""

    def test_disabled_does_not_set_raw_messages(self, mock_tokenizer) -> None:
        config = _make_config(pre_generate=False)
        composer = ConcreteComposer(config, mock_tokenizer)
        conv = _multi_turn_conversation(3)
        composer._finalize_conversation(conv, session_index=0)
        for turn in conv.turns:
            assert turn.raw_messages is None

    def test_enabled_sets_raw_messages_on_multi_turn(self, mock_tokenizer) -> None:
        config = _make_config(pre_generate=True)
        composer = ConcreteComposer(config, mock_tokenizer)
        conv = _multi_turn_conversation(3)
        composer._finalize_conversation(conv, session_index=0)
        assert conv.turns[0].raw_messages is None
        for i in range(1, len(conv.turns)):
            assert conv.turns[i].raw_messages is not None

    def test_enabled_single_turn_no_raw_messages(self, mock_tokenizer) -> None:
        config = _make_config(pre_generate=True)
        composer = ConcreteComposer(config, mock_tokenizer)
        conv = _single_turn_conversation()
        composer._finalize_conversation(conv, session_index=0)
        assert conv.turns[0].raw_messages is None

    def test_enabled_with_system_prompt_includes_in_raw_messages(
        self, mock_tokenizer
    ) -> None:
        config = _make_config(pre_generate=True, system_prompt_len=50)

        with patch(
            "aiperf.dataset.generator.prompt.PromptGenerator._generate_shared_system_prompt"
        ):
            composer = ConcreteComposer(config, mock_tokenizer)

        conv = _multi_turn_conversation(2)

        with patch.object(
            composer.prompt_generator,
            "get_shared_system_prompt",
            return_value="system prompt text",
        ):
            composer._finalize_conversation(conv, session_index=0)

        assert conv.system_message == "system prompt text"
        assert conv.turns[1].raw_messages is not None
        system_msgs = [
            m for m in conv.turns[1].raw_messages if m.get("role") == "system"
        ]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "system prompt text"

    def test_enabled_with_user_context_includes_in_raw_messages(
        self, mock_tokenizer
    ) -> None:
        config = _make_config(pre_generate=True, user_context_len=30)
        composer = ConcreteComposer(config, mock_tokenizer)
        conv = _multi_turn_conversation(2)

        with patch.object(
            composer.prompt_generator,
            "generate_user_context_prompt",
            return_value="user context text",
        ):
            composer._finalize_conversation(conv, session_index=0)

        assert conv.user_context_message == "user context text"
        assert conv.turns[1].raw_messages is not None
        user_msgs = [
            m
            for m in conv.turns[1].raw_messages
            if m.get("role") == "user" and m["content"] == "user context text"
        ]
        assert len(user_msgs) == 1
