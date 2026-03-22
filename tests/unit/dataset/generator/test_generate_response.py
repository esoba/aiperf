# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generate_response() on CodingContentGenerator and PromptGenerator.

Covers both OpenAI chat-completions format (default) and Anthropic
content-block format (use_content_blocks=True).
"""

import orjson
import pytest
from pytest import param

from aiperf.common import random_generator as rng
from aiperf.common.config import PrefixPromptConfig, PromptConfig
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.generator.prompt import PromptGenerator

_EXPECTED_TOOL_NAMES = {
    "Read",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "Write",
    "TaskUpdate",
    "TaskCreate",
    "Agent",
}


class _RoundTripTokenizer:
    """Minimal reversible tokenizer for realism tests."""

    block_separation_token_id = 0

    def encode(self, text: str, **kwargs) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _default_config() -> PromptConfig:
    return PromptConfig(
        mean=100,
        stddev=20,
        block_size=512,
        prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
    )


def _response_budget_tokens(
    generator: CodingContentGenerator, messages: list[dict]
) -> int:
    total = 0
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        total += len(generator.tokenizer.encode(block["text"]))
            elif "tool_calls" in msg:
                content = msg.get("content")
                if isinstance(content, str) and content:
                    total += len(generator.tokenizer.encode(content))
            else:
                if isinstance(content, str):
                    total += len(generator.tokenizer.encode(content))
        elif msg["role"] == "tool":
            total += len(generator.tokenizer.encode(msg["content"]))
        elif msg["role"] == "user":
            for block in msg.get("content", []):
                if block.get("type") == "tool_result":
                    total += len(generator.tokenizer.encode(block["content"]))
    return total


# ============================================================
# OpenAI format (default)
# ============================================================


class TestOpenAIFormat:
    """Default format: tool_calls array + role='tool' messages."""

    @pytest.fixture
    def generator(self, mock_tokenizer_cls) -> CodingContentGenerator:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(_default_config(), tokenizer)

    def test_roles_are_assistant_or_tool(
        self, generator: CodingContentGenerator
    ) -> None:
        for msg in generator.generate_response(300):
            assert msg["role"] in ("assistant", "tool")

    def test_last_message_is_plain_text_assistant(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(300)
        last = result[-1]
        assert last["role"] == "assistant"
        assert isinstance(last["content"], str)
        assert "tool_calls" not in last

    def test_tool_calls_structure(self, generator: CodingContentGenerator) -> None:
        for msg in generator.generate_response(500):
            if msg["role"] == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    assert tc["type"] == "function"
                    assert tc["id"].startswith("toolu_")
                    assert tc["function"]["name"] in _EXPECTED_TOOL_NAMES
                    orjson.loads(tc["function"]["arguments"])  # valid JSON string

    def test_tool_result_follows_assistant(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(500)
        emitted_ids: set[str] = set()
        for msg in result:
            if msg["role"] == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    emitted_ids.add(tc["id"])
            elif msg["role"] == "tool":
                assert msg["tool_call_id"] in emitted_ids
                assert isinstance(msg["content"], str)

    def test_tool_call_ids_unique(self, generator: CodingContentGenerator) -> None:
        ids = []
        for msg in generator.generate_response(500):
            if msg["role"] == "assistant" and "tool_calls" in msg:
                ids.extend(tc["id"] for tc in msg["tool_calls"])
        assert len(ids) == len(set(ids))


# ============================================================
# Anthropic content-block format
# ============================================================


class TestAnthropicFormat:
    """use_content_blocks=True: content arrays with tool_use/tool_result."""

    @pytest.fixture
    def generator(self, mock_tokenizer_cls) -> CodingContentGenerator:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(_default_config(), tokenizer)

    def _gen(self, generator: CodingContentGenerator, n: int) -> list[dict]:
        return generator.generate_response(n, use_content_blocks=True)

    def test_roles_are_assistant_or_user(
        self, generator: CodingContentGenerator
    ) -> None:
        for msg in self._gen(generator, 300):
            assert msg["role"] in ("assistant", "user")

    def test_all_content_is_list(self, generator: CodingContentGenerator) -> None:
        for msg in self._gen(generator, 300):
            assert isinstance(msg["content"], list)

    def test_last_message_is_text_only(self, generator: CodingContentGenerator) -> None:
        result = self._gen(generator, 300)
        last = result[-1]
        assert last["role"] == "assistant"
        blocks = last["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"

    def test_tool_use_block_structure(self, generator: CodingContentGenerator) -> None:
        for msg in self._gen(generator, 500):
            for b in msg.get("content", []):
                if b.get("type") == "tool_use":
                    assert b["id"].startswith("toolu_")
                    assert b["name"] in _EXPECTED_TOOL_NAMES
                    assert isinstance(b["input"], dict)

    def test_tool_result_in_user_messages_only(
        self, generator: CodingContentGenerator
    ) -> None:
        for msg in self._gen(generator, 500):
            if msg["role"] == "assistant":
                for b in msg.get("content", []):
                    assert b["type"] != "tool_result"

    def test_tool_result_ids_match_preceding_tool_use(
        self, generator: CodingContentGenerator
    ) -> None:
        result = self._gen(generator, 500)
        seen_ids: set[str] = set()
        for msg in result:
            for b in msg.get("content", []):
                if b.get("type") == "tool_use":
                    seen_ids.add(b["id"])
                elif b.get("type") == "tool_result":
                    assert b["tool_use_id"] in seen_ids


# ============================================================
# Edge cases (format-independent)
# ============================================================


class TestGenerateResponseEdgeCases:
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
    )
    def test_non_positive_returns_empty(
        self, generator: CodingContentGenerator, num_tokens: int
    ) -> None:
        assert generator.generate_response(num_tokens) == []
        assert generator.generate_response(num_tokens, use_content_blocks=True) == []

    def test_small_budget_has_tool_calls(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(20)
        assert len(result) >= 3
        assert any(m.get("tool_calls") for m in result)
        assert any(m.get("role") == "tool" for m in result)
        assert result[-1]["role"] == "assistant"

    def test_large_budget_multiple_iterations(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(500)
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 2

    def test_budget_one(self, generator: CodingContentGenerator) -> None:
        result = generator.generate_response(1)
        assert len(result) >= 3
        assert any(m.get("tool_calls") for m in result)


class TestExactTokenBudgets:
    @pytest.fixture
    def generator(self, mock_tokenizer_cls) -> CodingContentGenerator:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(_default_config(), tokenizer)

    @pytest.mark.parametrize("budget", [1, 25, 100, 300])
    def test_response_budget_exact_openai(
        self, generator: CodingContentGenerator, budget: int
    ) -> None:
        result = generator.generate_response(budget)
        assert _response_budget_tokens(generator, result) == budget

    @pytest.mark.parametrize("budget", [1, 25, 100, 300])
    def test_response_budget_exact_content_blocks(
        self, generator: CodingContentGenerator, budget: int
    ) -> None:
        result = generator.generate_response(budget, use_content_blocks=True)
        assert _response_budget_tokens(generator, result) == budget

    def test_assistant_text_tokens_exact_when_explicit(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(120, assistant_text_tokens=80)
        final_message = result[-1]
        assert final_message["role"] == "assistant"
        assert "tool_calls" not in final_message
        assert len(generator.tokenizer.encode(final_message["content"])) == 80

        tool_round_only = result[:-1]
        assert _response_budget_tokens(generator, tool_round_only) == 120


# ============================================================
# Styles (both formats should produce all styles)
# ============================================================


class TestGenerateResponseStyles:
    def _has_openai_style(self, result: list[dict], style: str) -> bool:
        for msg in result:
            if msg["role"] != "assistant" or "tool_calls" not in msg:
                continue
            n = len(msg["tool_calls"])
            has_text = isinstance(msg.get("content"), str)
            if style == "tool_only" and not has_text:
                return True
            if style == "text_and_tool" and has_text and n == 1:
                return True
            if style == "parallel" and n >= 2:
                return True
        return False

    def _scan_seeds(self, mock_tokenizer_cls, style: str) -> bool:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        for seed in range(50, 150):
            rng.reset()
            rng.init(seed)
            gen = CodingContentGenerator(_default_config(), tokenizer)
            if self._has_openai_style(gen.generate_response(500), style):
                return True
        return False

    def test_tool_only(self, mock_tokenizer_cls) -> None:
        assert self._scan_seeds(mock_tokenizer_cls, "tool_only")

    def test_text_and_tool(self, mock_tokenizer_cls) -> None:
        assert self._scan_seeds(mock_tokenizer_cls, "text_and_tool")

    def test_parallel(self, mock_tokenizer_cls) -> None:
        assert self._scan_seeds(mock_tokenizer_cls, "parallel")


# ============================================================
# Determinism
# ============================================================


class TestDeterminism:
    def test_same_seed(self, mock_tokenizer_cls) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        config = _default_config()

        rng.reset()
        rng.init(42)
        r1 = CodingContentGenerator(config, tokenizer).generate_response(300)

        rng.reset()
        rng.init(42)
        r2 = CodingContentGenerator(config, tokenizer).generate_response(300)

        assert r1 == r2

    def test_different_seeds(self, mock_tokenizer_cls) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        config = _default_config()

        rng.reset()
        rng.init(42)
        r1 = CodingContentGenerator(config, tokenizer).generate_response(300)

        rng.reset()
        rng.init(99)
        r2 = CodingContentGenerator(config, tokenizer).generate_response(300)

        assert r1 != r2

    def test_same_seed_different_budgets_change_first_tool_round(
        self, mock_tokenizer_cls
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        config = _default_config()

        rng.reset()
        rng.init(42)
        r1 = CodingContentGenerator(config, tokenizer).generate_response(300)

        rng.reset()
        rng.init(42)
        r2 = CodingContentGenerator(config, tokenizer).generate_response(600)

        first_tool_round_300 = next(msg for msg in r1 if msg.get("tool_calls"))
        first_tool_round_600 = next(msg for msg in r2 if msg.get("tool_calls"))

        assert first_tool_round_300 != first_tool_round_600


class TestResponseRealism:
    _FORBIDDEN_TRANSCRIPT_MARKERS = (
        "<tool_name>",
        "</result>",
        "[Assistant]",
        "[User]",
    )
    _REPEATED_PREAMBLE = "I need to see the config first."

    @pytest.fixture
    def generator(self) -> CodingContentGenerator:
        rng.reset()
        rng.init(1)
        return CodingContentGenerator(_default_config(), _RoundTripTokenizer())

    def test_tool_results_do_not_include_transcript_markers(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(1000)

        tool_messages = [msg for msg in result if msg["role"] == "tool"]
        assert tool_messages
        for msg in tool_messages:
            content = msg["content"]
            assert content
            assert not any(
                marker in content for marker in self._FORBIDDEN_TRANSCRIPT_MARKERS
            )

    def test_final_summary_does_not_include_tool_transcript_markers(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(1000)

        final_message = result[-1]
        assert final_message["role"] == "assistant"
        assert "tool_calls" not in final_message
        content = final_message["content"]
        assert isinstance(content, str)
        assert content
        assert not any(
            marker in content for marker in self._FORBIDDEN_TRANSCRIPT_MARKERS
        )

    def test_padded_assistant_preamble_does_not_self_repeat(
        self, generator: CodingContentGenerator
    ) -> None:
        result = generator.generate_response(300)

        first_tool_round = next(msg for msg in result if msg.get("tool_calls"))
        content = first_tool_round["content"]
        assert isinstance(content, str)
        assert content
        assert self._REPEATED_PREAMBLE in content
        assert content.count(self._REPEATED_PREAMBLE) == 1


# ============================================================
# _make_tool_call
# ============================================================


class TestMakeToolCall:
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
            ("Bash", "command"),
            ("Read", "file_path"),
            ("Write", "file_path"),
            ("Grep", "pattern"),
            ("Glob", "pattern"),
            ("Edit", "old_string"),
            ("TaskUpdate", "taskId"),
            ("TaskCreate", "subject"),
            ("Agent", "prompt"),
        ],
    )
    def test_tool_input_keys(
        self, mock_tokenizer_cls, tool_name: str, expected_key: str
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        found = False
        for seed in range(42, 500):
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
        assert found

    def test_bash_always_has_command(self, generator: CodingContentGenerator) -> None:
        found = False
        for _ in range(50):
            name, inp = generator._make_tool_call()
            if name == "Bash":
                assert "command" in inp
                found = True
                break
        assert found

    def test_edit_has_old_and_new_string(self, mock_tokenizer_cls) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        found = False
        for seed in range(42, 500):
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
        assert found


# ============================================================
# PromptGenerator.generate_response
# ============================================================


class TestPromptGeneratorGenerateResponse:
    @pytest.fixture
    def generator(self, mock_tokenizer_cls) -> PromptGenerator:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return PromptGenerator(_default_config(), tokenizer)

    def test_returns_single_element_list(self, generator: PromptGenerator) -> None:
        assert len(generator.generate_response(50)) == 1

    def test_message_has_assistant_role(self, generator: PromptGenerator) -> None:
        assert generator.generate_response(50)[0]["role"] == "assistant"

    def test_content_is_string(self, generator: PromptGenerator) -> None:
        assert isinstance(generator.generate_response(50)[0]["content"], str)

    def test_no_tool_calls(self, generator: PromptGenerator) -> None:
        assert "tool_calls" not in generator.generate_response(50)[0]
