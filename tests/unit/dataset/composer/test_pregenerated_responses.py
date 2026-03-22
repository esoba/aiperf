# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for coding tool-use ISL injection in BaseDatasetComposer.

Focuses on:
- _inject_coding_tool_history: multi-turn raw_messages construction with tool-use
- _extract_turn_text: text extraction from turns
- _finalize_conversation integration with coding corpus
"""

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
from aiperf.common.enums import PromptCorpus
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.composer.base import BaseDatasetComposer
from aiperf.dataset.generator.coding_content import CodingContentGenerator


class ConcreteComposer(BaseDatasetComposer):
    """Concrete test implementation of BaseDatasetComposer."""

    def create_dataset(self):
        return []


def _make_config(
    *,
    pre_generate: bool = False,
    prompt_corpus: PromptCorpus = PromptCorpus.CODING,
) -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig(
            conversation=ConversationConfig(
                num_dataset_entries=2,
                turn=TurnConfig(mean=3, stddev=0),
            ),
            prompt=PromptConfig(
                input_tokens=InputTokensConfig(mean=10, stddev=2),
                output_tokens=OutputTokensConfig(mean=50, stddev=0),
                pre_generate_responses=pre_generate,
                prompt_corpus=prompt_corpus,
                prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
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


# Use a longer text so token measurement works realistically
_LONG_TEXT = "This is a coding prompt about implementing a binary search tree " * 10


def _multi_turn_conversation(n: int = 3) -> Conversation:
    return Conversation(
        session_id="multi",
        turns=[_make_turn(_LONG_TEXT, max_tokens=50) for _ in range(n)],
    )


def _single_turn_conversation() -> Conversation:
    return Conversation(
        session_id="single",
        turns=[_make_turn(_LONG_TEXT)],
    )


# ============================================================
# _inject_coding_tool_history - Core Behavior
# ============================================================


class TestInjectCodingToolHistory:
    """Verify tool-use ISL injection for coding corpus multi-turn sessions."""

    @pytest.fixture
    def composer(self, mock_tokenizer) -> ConcreteComposer:
        config = _make_config(pre_generate=True)
        composer = ConcreteComposer(config, mock_tokenizer)
        coding_gen = CodingContentGenerator(
            config=config.input.prompt,
            tokenizer=mock_tokenizer,
        )
        composer.prompt_generator = coding_gen
        return composer

    def test_single_turn_not_modified(self, composer: ConcreteComposer) -> None:
        conv = _single_turn_conversation()
        composer._inject_coding_tool_history(conv)
        assert conv.turns[0].raw_messages is None

    def test_turn_zero_never_gets_raw_messages(
        self, composer: ConcreteComposer
    ) -> None:
        conv = _multi_turn_conversation(3)
        composer._inject_coding_tool_history(conv)
        assert conv.turns[0].raw_messages is None

    def test_subsequent_turns_get_raw_messages(
        self, composer: ConcreteComposer
    ) -> None:
        conv = _multi_turn_conversation(3)
        composer._inject_coding_tool_history(conv)
        for i in range(1, len(conv.turns)):
            assert conv.turns[i].raw_messages is not None
            assert isinstance(conv.turns[i].raw_messages, list)
            assert len(conv.turns[i].raw_messages) > 0

    def test_raw_messages_last_entry_is_user(self, composer: ConcreteComposer) -> None:
        conv = _multi_turn_conversation(3)
        composer._inject_coding_tool_history(conv)
        for i in range(1, len(conv.turns)):
            last_msg = conv.turns[i].raw_messages[-1]
            assert last_msg["role"] == "user"

    def test_raw_messages_contain_tool_calls(self, composer: ConcreteComposer) -> None:
        conv = _multi_turn_conversation(3)
        composer._inject_coding_tool_history(conv)
        for i in range(1, len(conv.turns)):
            has_tool_calls = any(
                m.get("tool_calls") for m in conv.turns[i].raw_messages
            )
            assert has_tool_calls, f"Turn {i} should have tool_calls"

    def test_raw_messages_contain_tool_results(
        self, composer: ConcreteComposer
    ) -> None:
        conv = _multi_turn_conversation(3)
        composer._inject_coding_tool_history(conv)
        for i in range(1, len(conv.turns)):
            has_tool_result = any(
                m.get("role") == "tool" for m in conv.turns[i].raw_messages
            )
            assert has_tool_result, f"Turn {i} should have tool results"

    def test_existing_raw_messages_left_untouched(
        self, composer: ConcreteComposer
    ) -> None:
        conv = _multi_turn_conversation(3)
        existing = [{"role": "user", "content": "preexisting"}]
        conv.turns[1].raw_messages = existing
        composer._inject_coding_tool_history(conv)
        assert conv.turns[1].raw_messages is existing
        assert conv.turns[2].raw_messages is not None
        assert conv.turns[2].raw_messages is not existing

    def test_empty_turns_no_crash(self, composer: ConcreteComposer) -> None:
        conv = Conversation(session_id="empty", turns=[])
        composer._inject_coding_tool_history(conv)

    def test_independent_per_turn(self, composer: ConcreteComposer) -> None:
        """Each turn's history is independently generated, not cumulative."""
        conv = _multi_turn_conversation(3)
        composer._inject_coding_tool_history(conv)
        # Both turns should have similar structure (independent generation)
        # rather than turn 2 being a superset of turn 1
        msgs_1 = conv.turns[1].raw_messages
        msgs_2 = conv.turns[2].raw_messages
        assert msgs_1 is not msgs_2


# ============================================================
# --pre-generate-responses controls assistant text
# ============================================================


class TestPreGenerateResponsesFlag:
    """The flag enables the entire pre-generation feature."""

    def test_without_flag_no_injection(self, mock_tokenizer) -> None:
        """Without --pre-generate-responses, normal multi-turn flow is used."""
        config = _make_config(pre_generate=False)
        composer = ConcreteComposer(config, mock_tokenizer)
        coding_gen = CodingContentGenerator(
            config=config.input.prompt,
            tokenizer=mock_tokenizer,
        )
        composer.prompt_generator = coding_gen

        conv = _multi_turn_conversation(3)
        composer._inject_coding_tool_history(conv)

        # No raw_messages set — normal DELTAS_WITHOUT_RESPONSES flow
        for turn in conv.turns:
            assert turn.raw_messages is None
        assert conv.context_mode is None

    def test_with_flag_injects_and_sets_context_mode(self, mock_tokenizer) -> None:
        """With --pre-generate-responses, raw_messages are set and context mode changed."""
        config = _make_config(pre_generate=True)
        composer = ConcreteComposer(config, mock_tokenizer)
        coding_gen = CodingContentGenerator(
            config=config.input.prompt,
            tokenizer=mock_tokenizer,
        )
        composer.prompt_generator = coding_gen

        conv = _multi_turn_conversation(3)
        composer._inject_coding_tool_history(conv)

        from aiperf.common.enums import ConversationContextMode

        assert conv.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        assert conv.turns[0].raw_messages is None
        for i in range(1, len(conv.turns)):
            assert conv.turns[i].raw_messages is not None
            has_tc = any(m.get("tool_calls") for m in conv.turns[i].raw_messages)
            assert has_tc


# ============================================================
# Non-coding corpus skips injection
# ============================================================


class TestNonCodingCorpusSkipsInjection:
    """Sonnet corpus should not get tool-use injection."""

    def test_sonnet_corpus_no_injection(self, mock_tokenizer) -> None:
        config = _make_config(prompt_corpus=PromptCorpus.SONNET)
        composer = ConcreteComposer(config, mock_tokenizer)
        conv = _multi_turn_conversation(3)
        composer._inject_coding_tool_history(conv)
        for turn in conv.turns:
            assert turn.raw_messages is None


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
