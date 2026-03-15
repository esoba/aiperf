# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generate_response() on CodingContentGenerator and PromptGenerator.

Focuses on:
- Anthropic content-block message structure from CodingContentGenerator.generate_response
- Tool use / tool result block structure, ordering, and uniqueness
- Style variety (text+tool_use, tool-only, parallel)
- PromptGenerator.generate_response single-message output
- _make_tool_call realistic tool name / input dict pairs
- Determinism and seed sensitivity
"""

import pytest
from pytest import param

from aiperf.common import random_generator as rng
from aiperf.common.config import PrefixPromptConfig, PromptConfig
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.generator.prompt import PromptGenerator

_EXPECTED_TOOL_NAMES = {"Read", "Edit", "Bash", "Grep", "Glob", "Write"}


def _default_config() -> PromptConfig:
    return PromptConfig(
        mean=100,
        stddev=20,
        block_size=512,
        prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
    )


def _get_blocks(msg: dict) -> list[dict]:
    content = msg.get("content")
    return content if isinstance(content, list) else []


def _tool_use_blocks(msgs: list[dict]) -> list[dict]:
    return [b for m in msgs for b in _get_blocks(m) if b.get("type") == "tool_use"]


def _tool_result_blocks(msgs: list[dict]) -> list[dict]:
    return [b for m in msgs for b in _get_blocks(m) if b.get("type") == "tool_result"]


# ============================================================
# CodingContentGenerator.generate_response - Happy Path
# ============================================================


class TestCodingGenerateResponseHappyPath:
    """Verify normal successful operation of generate_response."""

    @pytest.fixture
    def generator(self, mock_tokenizer_cls) -> CodingContentGenerator:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(_default_config(), tokenizer)

    def test_returns_list_of_dicts(self, generator: CodingContentGenerator) -> None:
        result = generator.generate_response(200)
        assert isinstance(result, list)
        assert all(isinstance(m, dict) for m in result)

    def test_all_messages_have_valid_role(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(200)
        for msg in result:
            assert "role" in msg
            assert msg["role"] in ("assistant", "user")

    def test_all_messages_have_content_list(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(200)
        for msg in result:
            assert isinstance(msg["content"], list)

    def test_last_message_is_text_only_assistant(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(200)
        last = result[-1]
        assert last["role"] == "assistant"
        blocks = _get_blocks(last)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert isinstance(blocks[0]["text"], str)
        assert len(blocks[0]["text"]) > 0

    def test_tool_use_block_structure(self, generator: CodingContentGenerator) -> None:
        result = generator.generate_response(500)
        for b in _tool_use_blocks(result):
            assert "id" in b
            assert b["id"].startswith("toolu_")
            assert "name" in b
            assert "input" in b
            assert isinstance(b["input"], dict)

    def test_tool_result_matches_prior_tool_use_id(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(500)
        emitted_ids: set[str] = set()
        for msg in result:
            for b in _get_blocks(msg):
                if b.get("type") == "tool_use":
                    emitted_ids.add(b["id"])
                elif b.get("type") == "tool_result":
                    assert b["tool_use_id"] in emitted_ids

    def test_tool_use_ids_unique(self, generator: CodingContentGenerator) -> None:
        result = generator.generate_response(500)
        ids = [b["id"] for b in _tool_use_blocks(result)]
        assert len(ids) == len(set(ids))

    def test_tool_results_in_user_messages(
        self, generator: CodingContentGenerator
    ) -> None:
        """tool_result blocks only appear in user messages."""
        result = generator.generate_response(500)
        for msg in result:
            if msg["role"] == "assistant":
                for b in _get_blocks(msg):
                    assert b["type"] != "tool_result"

    def test_tool_result_content_is_string(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(500)
        for b in _tool_result_blocks(result):
            assert isinstance(b["content"], str)
            assert len(b["content"]) > 0

    def test_user_message_follows_assistant_with_tool_use(
        self, generator: CodingContentGenerator
    ) -> None:
        """After an assistant message with tool_use, the next message is a user with tool_results."""
        result = generator.generate_response(500)
        for i, msg in enumerate(result):
            if msg["role"] == "assistant" and any(
                b.get("type") == "tool_use" for b in _get_blocks(msg)
            ):
                assert i + 1 < len(result)
                next_msg = result[i + 1]
                assert next_msg["role"] == "user"
                assert any(
                    b.get("type") == "tool_result" for b in _get_blocks(next_msg)
                )

    def test_tool_result_ids_match_preceding_tool_use_ids(
        self, generator: CodingContentGenerator
    ) -> None:
        """tool_result IDs in a user msg match tool_use IDs in the preceding assistant msg."""
        result = generator.generate_response(500)
        for i, msg in enumerate(result):
            if msg["role"] == "user":
                result_ids = {
                    b["tool_use_id"]
                    for b in _get_blocks(msg)
                    if b.get("type") == "tool_result"
                }
                if not result_ids:
                    continue
                assert i > 0
                prev = result[i - 1]
                use_ids = {
                    b["id"] for b in _get_blocks(prev) if b.get("type") == "tool_use"
                }
                assert result_ids == use_ids

    def test_tool_names_realistic(self, generator: CodingContentGenerator) -> None:
        result = generator.generate_response(500)
        for b in _tool_use_blocks(result):
            assert b["name"] in _EXPECTED_TOOL_NAMES


# ============================================================
# CodingContentGenerator.generate_response - Edge Cases
# ============================================================


class TestCodingGenerateResponseEdgeCases:
    """Verify boundary conditions and small/large token budgets."""

    @pytest.fixture
    def generator(self, mock_tokenizer_cls) -> CodingContentGenerator:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(_default_config(), tokenizer)

    @pytest.mark.parametrize(
        "num_tokens",
        [
            param(0, id="zero"),
            param(-1, id="negative"),
            param(-100, id="large-negative"),
        ],
    )  # fmt: skip
    def test_non_positive_returns_empty(
        self, generator: CodingContentGenerator, num_tokens: int
    ) -> None:
        assert generator.generate_response(num_tokens) == []

    def test_small_budget_single_text_message(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(20)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        blocks = _get_blocks(result[0])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"

    def test_large_budget_multiple_iterations(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(500)
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 2

    def test_budget_one_returns_single_message(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(1)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"


# ============================================================
# CodingContentGenerator.generate_response - Styles
# ============================================================


class TestCodingGenerateResponseStyles:
    """Verify that text+tool_use, tool-only, and parallel styles appear."""

    def _has_style(self, result: list[dict], style: str) -> bool:
        for msg in result:
            if msg["role"] != "assistant":
                continue
            blocks = _get_blocks(msg)
            text_blocks = [b for b in blocks if b.get("type") == "text"]
            tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
            if style == "tool_only" and tool_blocks and not text_blocks:
                return True
            if style == "text_and_tool" and tool_blocks and text_blocks:
                return True
            if style == "parallel" and len(tool_blocks) >= 2:
                return True
        return False

    def test_tool_only_style(self, mock_tokenizer_cls) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        found = False
        for seed in range(50, 70):
            rng.reset()
            rng.init(seed)
            gen = CodingContentGenerator(_default_config(), tokenizer)
            if self._has_style(gen.generate_response(500), "tool_only"):
                found = True
                break
        assert found, "No tool-only style seen across seed range"

    def test_text_and_tool_style(self, mock_tokenizer_cls) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        found = False
        for seed in range(50, 70):
            rng.reset()
            rng.init(seed)
            gen = CodingContentGenerator(_default_config(), tokenizer)
            if self._has_style(gen.generate_response(500), "text_and_tool"):
                found = True
                break
        assert found, "No text+tool style seen across seed range"

    def test_parallel_tool_calls(self, mock_tokenizer_cls) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        found = False
        for seed in range(50, 70):
            rng.reset()
            rng.init(seed)
            gen = CodingContentGenerator(_default_config(), tokenizer)
            if self._has_style(gen.generate_response(500), "parallel"):
                found = True
                break
        assert found, "No parallel style seen across seed range"


# ============================================================
# CodingContentGenerator.generate_response - Determinism
# ============================================================


class TestCodingGenerateResponseDeterminism:
    """Verify reproducibility guarantees."""

    def test_same_seed_identical_output(self, mock_tokenizer_cls) -> None:
        config = _default_config()
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")

        rng.reset()
        rng.init(42)
        gen1 = CodingContentGenerator(config, tokenizer)
        result1 = gen1.generate_response(300)

        rng.reset()
        rng.init(42)
        gen2 = CodingContentGenerator(config, tokenizer)
        result2 = gen2.generate_response(300)

        assert result1 == result2

    def test_different_seeds_different_output(self, mock_tokenizer_cls) -> None:
        config = _default_config()
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")

        rng.reset()
        rng.init(42)
        gen1 = CodingContentGenerator(config, tokenizer)
        result1 = gen1.generate_response(300)

        rng.reset()
        rng.init(99)
        gen2 = CodingContentGenerator(config, tokenizer)
        result2 = gen2.generate_response(300)

        assert result1 != result2


# ============================================================
# CodingContentGenerator._make_tool_call
# ============================================================


class TestMakeToolCall:
    """Verify _make_tool_call returns valid (name, input_dict) tuples."""

    @pytest.fixture
    def generator(self, mock_tokenizer_cls) -> CodingContentGenerator:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(_default_config(), tokenizer)

    def test_returns_name_and_dict(self, generator: CodingContentGenerator) -> None:
        name, inp = generator._make_tool_call()
        assert isinstance(name, str)
        assert isinstance(inp, dict)

    def test_name_in_expected_set(self, generator: CodingContentGenerator) -> None:
        for _ in range(30):
            name, _ = generator._make_tool_call()
            assert name in _EXPECTED_TOOL_NAMES

    @pytest.mark.parametrize(
        "tool_name,expected_key",
        [
            ("Read", "file_path"),
            ("Bash", "command"),
            ("Grep", "pattern"),
            ("Glob", "pattern"),
            ("Write", "file_path"),
        ],
    )  # fmt: skip
    def test_tool_input_keys(
        self, mock_tokenizer_cls, tool_name: str, expected_key: str
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        found = False
        for seed in range(42, 200):
            rng.reset()
            rng.init(seed)
            gen = CodingContentGenerator(_default_config(), tokenizer)
            for _ in range(50):
                name, inp = gen._make_tool_call()
                if name == tool_name:
                    assert expected_key in inp
                    found = True
                    break
            if found:
                break
        assert found, f"Never generated tool_name={tool_name}"

    def test_edit_has_old_and_new_string(self, mock_tokenizer_cls) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        found = False
        for seed in range(42, 200):
            rng.reset()
            rng.init(seed)
            gen = CodingContentGenerator(_default_config(), tokenizer)
            for _ in range(50):
                name, inp = gen._make_tool_call()
                if name == "Edit":
                    assert "old_string" in inp
                    assert "new_string" in inp
                    found = True
                    break
            if found:
                break
        assert found, "Never generated Edit tool call"


# ============================================================
# PromptGenerator.generate_response
# ============================================================


class TestPromptGeneratorGenerateResponse:
    """Verify PromptGenerator.generate_response returns single-message list."""

    @pytest.fixture
    def generator(self, mock_tokenizer_cls) -> PromptGenerator:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return PromptGenerator(_default_config(), tokenizer)

    def test_returns_single_element_list(self, generator: PromptGenerator) -> None:
        result = generator.generate_response(50)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_message_has_assistant_role(self, generator: PromptGenerator) -> None:
        result = generator.generate_response(50)
        assert result[0]["role"] == "assistant"

    def test_content_is_string(self, generator: PromptGenerator) -> None:
        result = generator.generate_response(50)
        assert isinstance(result[0]["content"], str)
        assert len(result[0]["content"]) > 0

    def test_no_tool_use_blocks(self, generator: PromptGenerator) -> None:
        result = generator.generate_response(50)
        assert not isinstance(result[0]["content"], list)


# ============================================================
# PromptConfig.pre_generate_responses
# ============================================================


class TestPromptConfigPreGenerateResponses:
    """Verify the config field default and settable."""

    def test_defaults_to_false(self) -> None:
        config = PromptConfig()
        assert config.pre_generate_responses is False

    def test_can_be_set_true(self) -> None:
        config = PromptConfig(pre_generate_responses=True)
        assert config.pre_generate_responses is True
