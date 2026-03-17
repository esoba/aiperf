# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dataset.message_normalizer.

Focuses on:
- Provider auto-detection from message/tool shapes
- Anthropic -> OpenAI canonical conversion (messages, tools)
- OpenAI canonical -> Anthropic wire conversion (emitters)
- Round-trip integrity (Anthropic -> canonical -> Anthropic)
- Content flattening, billing header stripping, tool_result extraction
- Image block bidirectional conversion
- Redacted thinking, server tools, citations, cache_control, is_error, caller
- Developer role, refusal, passthrough blocks
- OpenAI passthrough (messages unchanged)
"""

from typing import Any

import orjson
import pytest
from pytest import param

from aiperf.dataset.message_normalizer import (
    _anthropic_image_to_openai,
    _anthropic_tool_use_to_call,
    _detect_provider,
    _emit_anthropic_assistant,
    _emit_anthropic_tool_result,
    _emit_anthropic_user,
    _flatten_text_content,
    _merge_consecutive_roles,
    _normalize_anthropic_assistant,
    _normalize_anthropic_messages,
    _normalize_anthropic_tools,
    _normalize_anthropic_user,
    _openai_image_to_anthropic,
    _sanitize_tool_id,
    _tool_call_to_anthropic_block,
    normalize_messages,
    to_anthropic_messages,
    to_anthropic_tools,
)

# ============================================================
# Helpers -- reusable message builders
# ============================================================


def _tool_use_block(
    tool_id: str = "toolu_01",
    name: str = "Read",
    input_dict: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_use",
        "id": tool_id,
        "name": name,
        "input": input_dict or {"file_path": "/tmp/test.py"},
    }
    block.update(extra)
    return block


def _server_tool_use_block(
    tool_id: str = "srvtoolu_01",
    name: str = "web_search",
    input_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "server_tool_use",
        "id": tool_id,
        "name": name,
        "input": input_dict or {"query": "test"},
    }


def _tool_result_block(
    tool_use_id: str = "toolu_01",
    content: str | list[dict[str, Any]] = "file contents here",
    **extra: Any,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    block.update(extra)
    return block


def _text_block(text: str, **extra: Any) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "text", "text": text}
    block.update(extra)
    return block


def _thinking_block(text: str = "Let me think...") -> dict[str, Any]:
    return {"type": "thinking", "thinking": text}


def _redacted_thinking_block(data: str = "EqQBCgIYAh==") -> dict[str, Any]:
    return {"type": "redacted_thinking", "data": data}


def _anthropic_image_block(
    source_type: str = "base64",
    media_type: str = "image/jpeg",
    data: str = "abc123",
    url: str = "",
) -> dict[str, Any]:
    if source_type == "base64":
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _openai_image_url_part(url: str = "https://example.com/img.png") -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def _document_block(title: str = "doc") -> dict[str, Any]:
    return {
        "type": "document",
        "source": {"type": "text", "content": [_text_block("doc content")]},
        "title": title,
    }


def _search_result_block(title: str = "result") -> dict[str, Any]:
    return {
        "type": "search_result",
        "title": title,
        "source": "https://example.com",
        "content": [_text_block("search content")],
    }


def _web_search_tool_result_block() -> dict[str, Any]:
    return {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_01",
        "content": [_text_block("search results here")],
    }


def _anthropic_tool_def(
    name: str = "Read",
    description: str = "Read a file",
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties or {"file_path": {"type": "string"}},
        },
    }


def _openai_tool_def(
    name: str = "Read",
    description: str = "Read a file",
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _versioned_tool_def(tool_type: str = "web_search_20260209") -> dict[str, Any]:
    return {"type": tool_type, "name": "web_search"}


# ============================================================
# _detect_provider
# ============================================================


class TestDetectProvider:
    def test_detect_provider_anthropic_tool_use_in_messages(self) -> None:
        msgs = [{"role": "assistant", "content": [_tool_use_block()]}]
        assert _detect_provider(msgs) == "anthropic"

    def test_detect_provider_anthropic_tool_result_in_messages(self) -> None:
        msgs = [{"role": "user", "content": [_tool_result_block()]}]
        assert _detect_provider(msgs) == "anthropic"

    def test_detect_provider_anthropic_thinking_in_messages(self) -> None:
        msgs = [{"role": "assistant", "content": [_thinking_block()]}]
        assert _detect_provider(msgs) == "anthropic"

    def test_detect_provider_anthropic_redacted_thinking(self) -> None:
        msgs = [{"role": "assistant", "content": [_redacted_thinking_block()]}]
        assert _detect_provider(msgs) == "anthropic"

    def test_detect_provider_anthropic_server_tool_use(self) -> None:
        msgs = [{"role": "assistant", "content": [_server_tool_use_block()]}]
        assert _detect_provider(msgs) == "anthropic"

    def test_detect_provider_anthropic_image_with_source(self) -> None:
        msgs = [{"role": "user", "content": [_anthropic_image_block()]}]
        assert _detect_provider(msgs) == "anthropic"

    def test_detect_provider_anthropic_document_block(self) -> None:
        msgs = [{"role": "user", "content": [_document_block()]}]
        assert _detect_provider(msgs) == "anthropic"

    def test_detect_provider_anthropic_server_tool_result(self) -> None:
        msgs = [{"role": "assistant", "content": [_web_search_tool_result_block()]}]
        assert _detect_provider(msgs) == "anthropic"

    def test_detect_provider_anthropic_versioned_tool(self) -> None:
        tools = [_versioned_tool_def("computer_20251124")]
        assert _detect_provider([], tools) == "anthropic"

    def test_detect_provider_anthropic_tools(self) -> None:
        assert _detect_provider([], [_anthropic_tool_def()]) == "anthropic"

    def test_detect_provider_openai_tools(self) -> None:
        assert _detect_provider([], [_openai_tool_def()]) == "openai"

    def test_detect_provider_openai_tool_calls_on_assistant(self) -> None:
        msgs = [{"role": "assistant", "tool_calls": [{"id": "call_1"}]}]
        assert _detect_provider(msgs) == "openai"

    def test_detect_provider_openai_tool_role_message(self) -> None:
        msgs = [{"role": "tool", "tool_call_id": "call_1", "content": "ok"}]
        assert _detect_provider(msgs) == "openai"

    def test_detect_provider_plain_text_defaults_to_openai(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        assert _detect_provider(msgs) == "openai"

    def test_detect_provider_empty_messages_defaults_to_openai(self) -> None:
        assert _detect_provider([]) == "openai"

    def test_detect_provider_tools_checked_before_messages(self) -> None:
        """Tools are a more reliable signal and should be checked first."""
        msgs = [{"role": "assistant", "content": [_tool_use_block()]}]
        tools = [_openai_tool_def()]
        assert _detect_provider(msgs, tools) == "openai"

    def test_detect_provider_non_dict_content_blocks_skipped(self) -> None:
        msgs = [{"role": "user", "content": [42, "a string", None]}]
        assert _detect_provider(msgs) == "openai"

    def test_detect_provider_openai_tool_with_parameters_key(self) -> None:
        tools = [{"name": "foo", "parameters": {"type": "object"}}]
        assert _detect_provider([], tools) == "openai"


# ============================================================
# _flatten_text_content
# ============================================================


class TestFlattenTextContent:
    def test_flatten_plain_string(self) -> None:
        assert _flatten_text_content("hello") == "hello"

    def test_flatten_list_of_text_blocks(self) -> None:
        blocks = [_text_block("a"), _text_block("b")]
        assert _flatten_text_content(blocks) == "a\n\nb"

    def test_flatten_list_of_strings(self) -> None:
        assert _flatten_text_content(["x", "y"]) == "x\n\ny"

    def test_flatten_mixed_strings_and_text_blocks(self) -> None:
        content = ["raw", _text_block("block")]
        assert _flatten_text_content(content) == "raw\n\nblock"

    def test_flatten_none_returns_empty(self) -> None:
        assert _flatten_text_content(None) == ""

    def test_flatten_non_string_non_list_returns_str(self) -> None:
        assert _flatten_text_content(42) == "42"

    def test_flatten_empty_text_blocks_skipped(self) -> None:
        content = [_text_block(""), _text_block("ok")]
        assert _flatten_text_content(content) == "ok"

    def test_flatten_strip_billing_header_string(self) -> None:
        billing = "x-anthropic-billing-header: some data"
        assert _flatten_text_content(billing, strip_billing_headers=True) == ""

    def test_flatten_strip_billing_header_in_list(self) -> None:
        content = [
            _text_block("x-anthropic-billing-header: metadata"),
            _text_block("real system prompt"),
        ]
        result = _flatten_text_content(content, strip_billing_headers=True)
        assert result == "real system prompt"

    def test_flatten_strip_billing_header_raw_string_in_list(self) -> None:
        content = ["x-anthropic-billing-header: foo", "keep me"]
        result = _flatten_text_content(content, strip_billing_headers=True)
        assert result == "keep me"

    def test_flatten_billing_header_preserved_when_not_stripping(self) -> None:
        billing = "x-anthropic-billing-header: data"
        assert _flatten_text_content(billing) == billing


# ============================================================
# _flatten_text_content (also covers former _extract_tool_result_content)
# ============================================================


# ============================================================
# Image conversion helpers
# ============================================================


class TestAnthropicImageToOpenAI:
    def test_base64_image(self) -> None:
        block = _anthropic_image_block("base64", "image/jpeg", "abc123")
        result = _anthropic_image_to_openai(block)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "data:image/jpeg;base64,abc123"

    def test_url_image(self) -> None:
        block = _anthropic_image_block("url", url="https://example.com/img.png")
        result = _anthropic_image_to_openai(block)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "https://example.com/img.png"

    def test_unknown_source_type_passthrough(self) -> None:
        block = {"type": "image", "source": {"type": "file", "file_id": "f1"}}
        result = _anthropic_image_to_openai(block)
        assert result == block

    def test_cache_control_preserved(self) -> None:
        block = _anthropic_image_block("base64")
        block["cache_control"] = {"type": "ephemeral"}
        result = _anthropic_image_to_openai(block)
        assert result["_cache_control"] == {"type": "ephemeral"}

    def test_default_media_type(self) -> None:
        block = {"type": "image", "source": {"type": "base64", "data": "x"}}
        result = _anthropic_image_to_openai(block)
        assert "data:image/png;base64," in result["image_url"]["url"]


class TestOpenAIImageToAnthropic:
    def test_url_image(self) -> None:
        part = _openai_image_url_part("https://example.com/img.png")
        result = _openai_image_to_anthropic(part)
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://example.com/img.png"

    def test_data_uri_base64_image(self) -> None:
        url = "data:image/jpeg;base64,abc123"
        part = _openai_image_url_part(url)
        result = _openai_image_to_anthropic(part)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/jpeg"
        assert result["source"]["data"] == "abc123"

    def test_data_uri_png(self) -> None:
        url = "data:image/png;base64,xyz"
        result = _openai_image_to_anthropic(
            {"type": "image_url", "image_url": {"url": url}}
        )
        assert result["source"]["media_type"] == "image/png"

    def test_cache_control_restored(self) -> None:
        part = {
            "type": "image_url",
            "image_url": {"url": "https://x.com/i.png"},
            "_cache_control": {"type": "ephemeral"},
        }
        result = _openai_image_to_anthropic(part)
        assert result["cache_control"] == {"type": "ephemeral"}


# ============================================================
# _anthropic_tool_use_to_call
# ============================================================


class TestAnthropicToolUseToCall:
    def test_basic_conversion(self) -> None:
        block = _tool_use_block("t1", "Read", {"file_path": "f.py"})
        result = _anthropic_tool_use_to_call(block)
        assert result["id"] == "t1"
        assert result["type"] == "function"
        assert result["function"]["name"] == "Read"
        assert orjson.loads(result["function"]["arguments"]) == {"file_path": "f.py"}

    def test_caller_preserved(self) -> None:
        block = _tool_use_block()
        block["caller"] = {"type": "code_execution_20260120", "tool_id": "srv1"}
        result = _anthropic_tool_use_to_call(block)
        assert result["_caller"] == {
            "type": "code_execution_20260120",
            "tool_id": "srv1",
        }

    def test_cache_control_preserved(self) -> None:
        block = _tool_use_block()
        block["cache_control"] = {"type": "ephemeral"}
        result = _anthropic_tool_use_to_call(block)
        assert result["_cache_control"] == {"type": "ephemeral"}

    def test_non_dict_input(self) -> None:
        block = {"type": "tool_use", "id": "t1", "name": "X", "input": "raw string"}
        result = _anthropic_tool_use_to_call(block)
        assert result["function"]["arguments"] == "raw string"


# ============================================================
# _normalize_anthropic_assistant
# ============================================================


class TestNormalizeAnthropicAssistant:
    def test_text_only_flattened_to_string(self) -> None:
        msg = {"role": "assistant", "content": [_text_block("hello")]}
        result = _normalize_anthropic_assistant(msg)
        assert len(result) == 1
        assert result[0]["content"] == "hello"
        assert "tool_calls" not in result[0]

    def test_multiple_text_blocks_joined(self) -> None:
        msg = {"role": "assistant", "content": [_text_block("a"), _text_block("b")]}
        result = _normalize_anthropic_assistant(msg)
        assert result[0]["content"] == "a\n\nb"

    def test_tool_use_converted_to_tool_calls(self) -> None:
        tool = _tool_use_block("toolu_abc", "Bash", {"command": "ls"})
        msg = {"role": "assistant", "content": [tool]}
        result = _normalize_anthropic_assistant(msg)
        assert len(result) == 1
        out = result[0]
        assert out["content"] == ""
        assert len(out["tool_calls"]) == 1
        tc = out["tool_calls"][0]
        assert tc["id"] == "toolu_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "Bash"
        parsed = orjson.loads(tc["function"]["arguments"])
        assert parsed == {"command": "ls"}

    def test_text_plus_tool_use(self) -> None:
        msg = {
            "role": "assistant",
            "content": [
                _text_block("Let me check that file."),
                _tool_use_block("toolu_01", "Read", {"file_path": "/etc/hosts"}),
            ],
        }
        result = _normalize_anthropic_assistant(msg)
        out = result[0]
        assert out["content"] == "Let me check that file."
        assert len(out["tool_calls"]) == 1

    def test_thinking_blocks_preserved(self) -> None:
        msg = {
            "role": "assistant",
            "content": [_thinking_block("hmm"), _text_block("answer")],
        }
        result = _normalize_anthropic_assistant(msg)
        out = result[0]
        assert out["content"] == "answer"
        assert len(out["thinking_blocks"]) == 1
        assert out["thinking_blocks"][0]["type"] == "thinking"

    def test_redacted_thinking_preserved(self) -> None:
        msg = {
            "role": "assistant",
            "content": [_redacted_thinking_block("encrypted"), _text_block("answer")],
        }
        result = _normalize_anthropic_assistant(msg)
        out = result[0]
        assert len(out["thinking_blocks"]) == 1
        assert out["thinking_blocks"][0]["type"] == "redacted_thinking"
        assert out["thinking_blocks"][0]["data"] == "encrypted"

    def test_mixed_thinking_and_redacted_thinking(self) -> None:
        msg = {
            "role": "assistant",
            "content": [
                _thinking_block("visible"),
                _redacted_thinking_block("hidden"),
                _text_block("answer"),
            ],
        }
        result = _normalize_anthropic_assistant(msg)
        blocks = result[0]["thinking_blocks"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "thinking"
        assert blocks[1]["type"] == "redacted_thinking"

    def test_server_tool_use_converted_with_marker(self) -> None:
        msg = {
            "role": "assistant",
            "content": [_server_tool_use_block("srv1", "web_search")],
        }
        result = _normalize_anthropic_assistant(msg)
        tc = result[0]["tool_calls"][0]
        assert tc["function"]["name"] == "web_search"
        assert tc["_server_tool"] is True

    def test_server_tool_result_preserved_as_passthrough(self) -> None:
        msg = {
            "role": "assistant",
            "content": [
                _server_tool_use_block(),
                _web_search_tool_result_block(),
                _text_block("Based on my search..."),
            ],
        }
        result = _normalize_anthropic_assistant(msg)
        out = result[0]
        assert len(out["tool_calls"]) == 1
        assert len(out["_passthrough_blocks"]) == 1
        assert out["_passthrough_blocks"][0]["type"] == "web_search_tool_result"
        assert out["content"] == "Based on my search..."

    def test_caller_preserved_on_tool_call(self) -> None:
        block = _tool_use_block()
        block["caller"] = {"type": "code_execution_20260120", "tool_id": "srv1"}
        msg = {"role": "assistant", "content": [block]}
        result = _normalize_anthropic_assistant(msg)
        assert "_caller" in result[0]["tool_calls"][0]

    def test_citations_preserved(self) -> None:
        citations = [
            {"type": "char_location", "cited_text": "hello", "document_index": 0}
        ]
        text = _text_block("hello")
        text["citations"] = citations
        msg = {"role": "assistant", "content": [text]}
        result = _normalize_anthropic_assistant(msg)
        assert result[0]["_citations"] == citations

    def test_empty_content_list_returns_empty_string(self) -> None:
        msg = {"role": "assistant", "content": []}
        result = _normalize_anthropic_assistant(msg)
        assert result[0]["content"] == ""

    def test_non_list_content_passthrough(self) -> None:
        msg = {"role": "assistant", "content": "already a string"}
        result = _normalize_anthropic_assistant(msg)
        assert result == [msg]

    def test_tool_use_with_non_dict_input(self) -> None:
        block = {"type": "tool_use", "id": "t1", "name": "X", "input": "raw string"}
        msg = {"role": "assistant", "content": [block]}
        result = _normalize_anthropic_assistant(msg)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == "raw string"

    def test_non_dict_blocks_in_content_skipped(self) -> None:
        msg = {"role": "assistant", "content": [42, _text_block("ok")]}
        result = _normalize_anthropic_assistant(msg)
        assert result[0]["content"] == "ok"

    def test_multiple_tool_use_blocks(self) -> None:
        msg = {
            "role": "assistant",
            "content": [
                _tool_use_block("t1", "Read", {"path": "a"}),
                _tool_use_block("t2", "Bash", {"cmd": "b"}),
            ],
        }
        result = _normalize_anthropic_assistant(msg)
        assert len(result[0]["tool_calls"]) == 2
        names = [tc["function"]["name"] for tc in result[0]["tool_calls"]]
        assert names == ["Read", "Bash"]


# ============================================================
# _normalize_anthropic_user
# ============================================================


class TestNormalizeAnthropicUser:
    def test_plain_string_passthrough(self) -> None:
        msg = {"role": "user", "content": "hello"}
        result = _normalize_anthropic_user(msg)
        assert result == [msg]

    def test_tool_result_becomes_tool_role(self) -> None:
        msg = {
            "role": "user",
            "content": [_tool_result_block("toolu_01", "file data")],
        }
        result = _normalize_anthropic_user(msg)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "toolu_01"
        assert result[0]["content"] == "file data"

    def test_tool_result_is_error_preserved(self) -> None:
        block = _tool_result_block("t1", "Error!", is_error=True)
        msg = {"role": "user", "content": [block]}
        result = _normalize_anthropic_user(msg)
        assert result[0]["_is_error"] is True

    def test_tool_result_cache_control_preserved(self) -> None:
        block = _tool_result_block("t1", "data", cache_control={"type": "ephemeral"})
        msg = {"role": "user", "content": [block]}
        result = _normalize_anthropic_user(msg)
        assert result[0]["_cache_control"] == {"type": "ephemeral"}

    def test_text_plus_tool_result_split(self) -> None:
        """Text should appear as user message before tool result messages."""
        msg = {
            "role": "user",
            "content": [
                _text_block("Here is context"),
                _tool_result_block("toolu_01", "result data"),
            ],
        }
        result = _normalize_anthropic_user(msg)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Here is context"}
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "toolu_01"

    def test_multiple_tool_results(self) -> None:
        msg = {
            "role": "user",
            "content": [
                _tool_result_block("t1", "r1"),
                _tool_result_block("t2", "r2"),
            ],
        }
        result = _normalize_anthropic_user(msg)
        assert len(result) == 2
        assert result[0]["tool_call_id"] == "t1"
        assert result[1]["tool_call_id"] == "t2"

    def test_tool_result_with_nested_text_blocks(self) -> None:
        nested = [_text_block("line 1"), _text_block("line 2")]
        msg = {"role": "user", "content": [_tool_result_block("t1", nested)]}
        result = _normalize_anthropic_user(msg)
        assert result[0]["content"] == "line 1\n\nline 2"

    def test_raw_string_in_content_list(self) -> None:
        msg = {"role": "user", "content": ["a plain string"]}
        result = _normalize_anthropic_user(msg)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "a plain string"

    def test_empty_content_list_returns_original(self) -> None:
        msg = {"role": "user", "content": []}
        result = _normalize_anthropic_user(msg)
        assert result == [msg]

    def test_image_block_converted_to_image_url(self) -> None:
        msg = {
            "role": "user",
            "content": [
                _text_block("Look at this"),
                _anthropic_image_block("base64", "image/png", "imgdata"),
            ],
        }
        result = _normalize_anthropic_user(msg)
        assert len(result) == 1
        parts = result[0]["content"]
        assert parts[0] == {"type": "text", "text": "Look at this"}
        assert parts[1]["type"] == "image_url"
        assert "data:image/png;base64,imgdata" in parts[1]["image_url"]["url"]

    def test_image_url_block_converted(self) -> None:
        msg = {
            "role": "user",
            "content": [_anthropic_image_block("url", url="https://example.com/i.jpg")],
        }
        result = _normalize_anthropic_user(msg)
        assert result[0]["content"][0]["type"] == "image_url"
        assert (
            result[0]["content"][0]["image_url"]["url"] == "https://example.com/i.jpg"
        )

    def test_document_block_preserved(self) -> None:
        msg = {"role": "user", "content": [_text_block("context"), _document_block()]}
        result = _normalize_anthropic_user(msg)
        parts = result[0]["content"]
        assert any(p.get("type") == "document" for p in parts)

    def test_search_result_block_preserved(self) -> None:
        msg = {"role": "user", "content": [_search_result_block()]}
        result = _normalize_anthropic_user(msg)
        parts = result[0]["content"]
        assert parts[0]["type"] == "search_result"

    def test_other_block_types_kept_as_content_parts(self) -> None:
        """Non-text, non-tool_result blocks are preserved as content parts."""
        unknown = {"type": "custom_type", "data": "something"}
        msg = {"role": "user", "content": [_text_block("caption"), unknown]}
        result = _normalize_anthropic_user(msg)
        assert len(result) == 1
        parts = result[0]["content"]
        assert len(parts) == 2
        assert parts[1] == unknown


# ============================================================
# _normalize_anthropic_tools
# ============================================================


class TestNormalizeAnthropicTools:
    def test_anthropic_tool_converted(self) -> None:
        tools = [_anthropic_tool_def("Bash", "Run a command")]
        result = _normalize_anthropic_tools(tools)
        assert len(result) == 1
        t = result[0]
        assert t["type"] == "function"
        assert t["function"]["name"] == "Bash"
        assert t["function"]["description"] == "Run a command"
        assert t["function"]["parameters"]["type"] == "object"

    def test_openai_tool_passthrough(self) -> None:
        tools = [_openai_tool_def("Read")]
        result = _normalize_anthropic_tools(tools)
        assert result == tools

    def test_parameters_key_tool_wrapped(self) -> None:
        """Tools with top-level name + parameters get wrapped in function envelope."""
        tool = {"name": "foo", "description": "bar", "parameters": {"type": "object"}}
        result = _normalize_anthropic_tools([tool])
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "foo"

    def test_unknown_format_passthrough(self) -> None:
        tool = {"something": "else"}
        result = _normalize_anthropic_tools([tool])
        assert result == [tool]

    def test_missing_description_gets_default(self) -> None:
        tool = {"name": "X", "input_schema": {"type": "object"}}
        result = _normalize_anthropic_tools([tool])
        assert result[0]["function"]["description"] == "(no description)"

    def test_mixed_tool_formats(self) -> None:
        tools = [
            _anthropic_tool_def("A"),
            _openai_tool_def("B"),
            {"name": "C", "parameters": {"type": "object"}},
        ]
        result = _normalize_anthropic_tools(tools)
        names = [t["function"]["name"] for t in result]
        assert names == ["A", "B", "C"]

    def test_versioned_tool_passthrough(self) -> None:
        tools = [_versioned_tool_def("computer_20251124")]
        result = _normalize_anthropic_tools(tools)
        assert result == tools

    def test_versioned_tool_among_mixed(self) -> None:
        tools = [
            _anthropic_tool_def("A"),
            _versioned_tool_def("bash_20250124"),
            _versioned_tool_def("text_editor_20250429"),
        ]
        result = _normalize_anthropic_tools(tools)
        assert result[0]["type"] == "function"
        assert result[1]["type"] == "bash_20250124"
        assert result[2]["type"] == "text_editor_20250429"

    def test_cache_control_preserved(self) -> None:
        tool = _anthropic_tool_def("Read")
        tool["cache_control"] = {"type": "ephemeral"}
        result = _normalize_anthropic_tools([tool])
        assert result[0]["_cache_control"] == {"type": "ephemeral"}


# ============================================================
# _normalize_anthropic_messages (integration of sub-functions)
# ============================================================


class TestNormalizeAnthropicMessages:
    def test_system_message_flattened(self) -> None:
        msgs = [
            {
                "role": "system",
                "content": [
                    _text_block("x-anthropic-billing-header: proj-123"),
                    _text_block("You are a helpful assistant."),
                ],
            },
        ]
        result = _normalize_anthropic_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."

    def test_system_message_all_billing_headers_produces_no_output(self) -> None:
        msgs = [
            {
                "role": "system",
                "content": [_text_block("x-anthropic-billing-header: only")],
            }
        ]
        result = _normalize_anthropic_messages(msgs)
        assert len(result) == 0

    def test_unknown_role_passthrough(self) -> None:
        msg = {"role": "function", "content": "data"}
        result = _normalize_anthropic_messages([msg])
        assert result == [msg]

    def test_full_conversation_round_trip(self) -> None:
        """Realistic Claude Code multi-turn with system, assistant, user/tool_result."""
        msgs = [
            {
                "role": "system",
                "content": [
                    _text_block("x-anthropic-billing-header: abc"),
                    _text_block("System prompt here."),
                ],
            },
            {"role": "user", "content": "Fix the bug in main.py"},
            {
                "role": "assistant",
                "content": [
                    _text_block("I will read the file first."),
                    _tool_use_block("t1", "Read", {"file_path": "main.py"}),
                ],
            },
            {
                "role": "user",
                "content": [_tool_result_block("t1", "def main(): pass")],
            },
            {
                "role": "assistant",
                "content": [_text_block("The file is empty. Done.")],
            },
        ]
        result = _normalize_anthropic_messages(msgs)

        assert result[0] == {"role": "system", "content": "System prompt here."}
        assert result[1] == {"role": "user", "content": "Fix the bug in main.py"}

        # Assistant with tool call
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "I will read the file first."
        assert len(result[2]["tool_calls"]) == 1

        # Tool result
        assert result[3]["role"] == "tool"
        assert result[3]["tool_call_id"] == "t1"

        # Final assistant
        assert result[4] == {"role": "assistant", "content": "The file is empty. Done."}


# ============================================================
# normalize_messages (top-level entry point)
# ============================================================


class TestNormalizeMessages:
    def test_anthropic_auto_detected_and_normalized(self) -> None:
        msgs = [
            {"role": "assistant", "content": [_tool_use_block("t1", "Read", {"p": 1})]},
            {"role": "user", "content": [_tool_result_block("t1", "data")]},
        ]
        tools = [_anthropic_tool_def()]
        result_msgs, result_tools = normalize_messages(msgs, tools)

        assert result_msgs[0]["tool_calls"][0]["type"] == "function"
        assert result_msgs[1]["role"] == "tool"
        assert result_tools is not None
        assert result_tools[0]["type"] == "function"

    def test_openai_passthrough(self) -> None:
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result_msgs, result_tools = normalize_messages(msgs)
        assert result_msgs == msgs
        assert result_tools is None

    def test_explicit_provider_skips_detection(self) -> None:
        """When provider is given explicitly, detection is bypassed."""
        msgs = [
            {"role": "assistant", "content": [_tool_use_block()]},
        ]
        # Force openai -- should NOT normalize Anthropic content
        result_msgs, _ = normalize_messages(msgs, provider="openai")
        assert isinstance(result_msgs[0]["content"], list)

    def test_tokens_metadata_stripped(self) -> None:
        msgs = [{"role": "user", "content": "hi", "tokens": 42}]
        result_msgs, _ = normalize_messages(msgs)
        assert "tokens" not in result_msgs[0]

    def test_none_tools_stay_none(self) -> None:
        _, result_tools = normalize_messages([{"role": "user", "content": "x"}])
        assert result_tools is None

    def test_empty_tools_stay_empty(self) -> None:
        _, result_tools = normalize_messages(
            [{"role": "user", "content": "x"}], tools=[]
        )
        assert result_tools == []

    @pytest.mark.parametrize(
        "provider,msgs,tools",
        [
            param(
                None,
                [{"role": "user", "content": "hi"}],
                None,
                id="plain-text-no-tools",
            ),
            param(
                "openai",
                [{"role": "user", "content": "hi"}],
                [_openai_tool_def()],
                id="explicit-openai-with-tools",
            ),
        ],
    )  # fmt: skip
    def test_openai_messages_unchanged(
        self,
        provider: str | None,
        msgs: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> None:
        result_msgs, _ = normalize_messages(msgs, tools, provider=provider)
        # Content should remain identical (only tokens key stripped)
        for orig, res in zip(msgs, result_msgs, strict=False):
            assert res["content"] == orig["content"]


# ============================================================
# _emit_anthropic_assistant (canonical -> Anthropic)
# ============================================================


class TestEmitAnthropicAssistant:
    def test_text_only_becomes_text_block(self) -> None:
        msg = {"role": "assistant", "content": "hello world"}
        result = _emit_anthropic_assistant(msg)
        assert result["role"] == "assistant"
        assert result["content"] == [{"type": "text", "text": "hello world"}]

    def test_empty_string_content_becomes_empty_text_block(self) -> None:
        msg = {"role": "assistant", "content": ""}
        result = _emit_anthropic_assistant(msg)
        assert result["content"] == [{"type": "text", "text": ""}]

    def test_none_content_produces_empty_text_block(self) -> None:
        msg = {"role": "assistant", "content": None}
        result = _emit_anthropic_assistant(msg)
        assert result["content"] == [{"type": "text", "text": ""}]

    def test_tool_calls_become_tool_use_blocks(self) -> None:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": orjson.dumps({"command": "ls"}).decode(),
                    },
                }
            ],
        }
        result = _emit_anthropic_assistant(msg)
        blocks = result["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["id"] == "call_1"
        assert blocks[0]["name"] == "Bash"
        assert blocks[0]["input"] == {"command": "ls"}

    def test_text_plus_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"path": "f.py"}'},
                }
            ],
        }
        result = _emit_anthropic_assistant(msg)
        blocks = result["content"]
        assert len(blocks) == 2
        assert blocks[0] == {"type": "text", "text": "Let me check."}
        assert blocks[1]["type"] == "tool_use"

    def test_thinking_blocks_restored_first(self) -> None:
        msg = {
            "role": "assistant",
            "content": "answer",
            "thinking_blocks": [{"type": "thinking", "thinking": "hmm"}],
        }
        result = _emit_anthropic_assistant(msg)
        blocks = result["content"]
        assert blocks[0] == {"type": "thinking", "thinking": "hmm"}
        assert blocks[1] == {"type": "text", "text": "answer"}

    def test_redacted_thinking_restored(self) -> None:
        msg = {
            "role": "assistant",
            "content": "answer",
            "thinking_blocks": [{"type": "redacted_thinking", "data": "enc"}],
        }
        result = _emit_anthropic_assistant(msg)
        blocks = result["content"]
        assert blocks[0] == {"type": "redacted_thinking", "data": "enc"}

    def test_server_tool_use_restored(self) -> None:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "srv1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"q": "test"}'},
                    "_server_tool": True,
                }
            ],
        }
        result = _emit_anthropic_assistant(msg)
        block = result["content"][0]
        assert block["type"] == "server_tool_use"
        assert block["name"] == "web_search"

    def test_caller_restored_on_tool_use(self) -> None:
        caller = {"type": "code_execution_20260120", "tool_id": "srv1"}
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                    "_caller": caller,
                }
            ],
        }
        result = _emit_anthropic_assistant(msg)
        assert result["content"][0]["caller"] == caller

    def test_cache_control_restored_on_tool_use(self) -> None:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "X", "arguments": "{}"},
                    "_cache_control": {"type": "ephemeral"},
                }
            ],
        }
        result = _emit_anthropic_assistant(msg)
        assert result["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_passthrough_blocks_restored(self) -> None:
        sr = _web_search_tool_result_block()
        msg = {
            "role": "assistant",
            "content": "result",
            "_passthrough_blocks": [sr],
        }
        result = _emit_anthropic_assistant(msg)
        assert sr in result["content"]

    def test_refusal_becomes_text_block(self) -> None:
        msg = {"role": "assistant", "content": None, "refusal": "I cannot do that."}
        result = _emit_anthropic_assistant(msg)
        text_blocks = [b for b in result["content"] if b.get("type") == "text"]
        assert any(b["text"] == "I cannot do that." for b in text_blocks)

    def test_multiple_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "A", "arguments": "{}"},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "B", "arguments": "{}"},
                },
            ],
        }
        result = _emit_anthropic_assistant(msg)
        names = [b["name"] for b in result["content"] if b["type"] == "tool_use"]
        assert names == ["A", "B"]

    def test_invalid_json_arguments_passthrough(self) -> None:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "X", "arguments": "not valid json"},
                }
            ],
        }
        result = _emit_anthropic_assistant(msg)
        assert result["content"][0]["input"] == "not valid json"

    def test_list_content_passed_through(self) -> None:
        """Content that is already a list of blocks passes through."""
        blocks = [{"type": "text", "text": "already blocks"}]
        msg = {"role": "assistant", "content": blocks}
        result = _emit_anthropic_assistant(msg)
        assert {"type": "text", "text": "already blocks"} in result["content"]


# ============================================================
# _emit_anthropic_tool_result (canonical -> Anthropic)
# ============================================================


class TestEmitAnthropicToolResult:
    def test_basic_tool_result(self) -> None:
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "file data"}
        result = _emit_anthropic_tool_result(msg)
        assert result["role"] == "user"
        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call_1"
        assert block["content"] == "file data"

    def test_missing_tool_call_id_defaults_empty(self) -> None:
        msg = {"role": "tool", "content": "data"}
        result = _emit_anthropic_tool_result(msg)
        assert result["content"][0]["tool_use_id"] == ""

    def test_missing_content_defaults_empty(self) -> None:
        msg = {"role": "tool", "tool_call_id": "c1"}
        result = _emit_anthropic_tool_result(msg)
        assert result["content"][0]["content"] == ""

    def test_is_error_restored(self) -> None:
        msg = {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "Error!",
            "_is_error": True,
        }
        result = _emit_anthropic_tool_result(msg)
        assert result["content"][0]["is_error"] is True

    def test_is_error_not_set_when_false(self) -> None:
        msg = {"role": "tool", "tool_call_id": "c1", "content": "ok"}
        result = _emit_anthropic_tool_result(msg)
        assert "is_error" not in result["content"][0]

    def test_cache_control_restored(self) -> None:
        msg = {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "data",
            "_cache_control": {"type": "ephemeral"},
        }
        result = _emit_anthropic_tool_result(msg)
        assert result["content"][0]["cache_control"] == {"type": "ephemeral"}


# ============================================================
# _emit_anthropic_user (canonical -> Anthropic)
# ============================================================


class TestEmitAnthropicUser:
    def test_string_content_passthrough(self) -> None:
        msg = {"role": "user", "content": "hello"}
        result = _emit_anthropic_user(msg)
        assert result == {"role": "user", "content": "hello"}

    def test_image_url_converted_to_anthropic_image(self) -> None:
        part = _openai_image_url_part("https://example.com/img.png")
        msg = {"role": "user", "content": [{"type": "text", "text": "look"}, part]}
        result = _emit_anthropic_user(msg)
        blocks = result["content"]
        assert blocks[0] == {"type": "text", "text": "look"}
        assert blocks[1]["type"] == "image"
        assert blocks[1]["source"]["type"] == "url"
        assert blocks[1]["source"]["url"] == "https://example.com/img.png"

    def test_data_uri_image_url_converted(self) -> None:
        url = "data:image/jpeg;base64,abc123"
        msg = {"role": "user", "content": [_openai_image_url_part(url)]}
        result = _emit_anthropic_user(msg)
        block = result["content"][0]
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/jpeg"
        assert block["source"]["data"] == "abc123"

    def test_non_image_parts_passthrough(self) -> None:
        msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        result = _emit_anthropic_user(msg)
        assert result["content"] == [{"type": "text", "text": "hi"}]

    def test_non_dict_content_passthrough(self) -> None:
        msg = {"role": "user", "content": 42}
        result = _emit_anthropic_user(msg)
        assert result == msg

    def test_mixed_content_parts(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "caption"},
                _openai_image_url_part("https://x.com/i.png"),
                {"type": "document", "title": "doc"},
            ],
        }
        result = _emit_anthropic_user(msg)
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "image"
        assert result["content"][2]["type"] == "document"


# ============================================================
# _merge_consecutive_roles
# ============================================================


class TestMergeConsecutiveRoles:
    def test_empty_input(self) -> None:
        assert _merge_consecutive_roles([]) == []

    def test_single_message(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        assert _merge_consecutive_roles(msgs) == msgs

    def test_alternating_roles_unchanged(self) -> None:
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q2"},
        ]
        assert _merge_consecutive_roles(msgs) == msgs

    def test_consecutive_user_list_plus_list(self) -> None:
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "content": "r1"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "r2"}]},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2

    def test_consecutive_user_string_plus_string(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "world"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]

    def test_consecutive_user_list_plus_string(self) -> None:
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "user", "content": "b"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_consecutive_user_string_plus_list(self) -> None:
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": [{"type": "text", "text": "b"}]},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_three_consecutive_same_role(self) -> None:
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "content": "r1"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "r2"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "r3"}]},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert len(result[0]["content"]) == 3

    def test_different_roles_not_merged(self) -> None:
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 3


# ============================================================
# to_anthropic_messages (full emitter)
# ============================================================


class TestToAnthropicMessages:
    def test_system_extracted(self) -> None:
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"},
        ]
        result, system = to_anthropic_messages(msgs)
        assert system == "Be helpful."
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_developer_role_extracted_as_system(self) -> None:
        msgs = [
            {"role": "developer", "content": "Instructions"},
            {"role": "user", "content": "hi"},
        ]
        result, system = to_anthropic_messages(msgs)
        assert system == "Instructions"
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_multiple_system_messages_last_wins(self) -> None:
        msgs = [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "q"},
            {"role": "system", "content": "second"},
            {"role": "user", "content": "q2"},
        ]
        _, system = to_anthropic_messages(msgs)
        assert system == "second"

    def test_developer_overrides_system(self) -> None:
        msgs = [
            {"role": "system", "content": "old"},
            {"role": "developer", "content": "new"},
            {"role": "user", "content": "q"},
        ]
        _, system = to_anthropic_messages(msgs)
        assert system == "new"

    def test_no_system_returns_none(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        _, system = to_anthropic_messages(msgs)
        assert system is None

    def test_tool_role_becomes_user_tool_result(self) -> None:
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": "result data"}]
        result, _ = to_anthropic_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        block = result[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "c1"

    def test_assistant_with_tool_calls_emitted(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": '{"p": 1}'},
                    }
                ],
            }
        ]
        result, _ = to_anthropic_messages(msgs)
        blocks = result[0]["content"]
        types = [b["type"] for b in blocks]
        assert "text" in types
        assert "tool_use" in types

    def test_consecutive_tool_results_merged(self) -> None:
        """Multiple role:tool messages should merge into one user message."""
        msgs = [
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        ]
        result, _ = to_anthropic_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2

    def test_user_string_content_passthrough(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        result, _ = to_anthropic_messages(msgs)
        assert result[0] == {"role": "user", "content": "hello"}

    def test_user_list_content_passthrough(self) -> None:
        blocks = [{"type": "text", "text": "hi"}, {"type": "document", "title": "d"}]
        msgs = [{"role": "user", "content": blocks}]
        result, _ = to_anthropic_messages(msgs)
        assert result[0]["content"] == blocks

    def test_user_image_url_converted(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                _openai_image_url_part("https://x.com/i.png"),
            ],
        }
        result, _ = to_anthropic_messages([msg])
        blocks = result[0]["content"]
        assert blocks[1]["type"] == "image"

    def test_unknown_role_passthrough(self) -> None:
        msgs = [{"role": "function", "content": "legacy"}]
        result, _ = to_anthropic_messages(msgs)
        assert result[0] == msgs[0]

    def test_full_conversation(self) -> None:
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Fix main.py"},
            {
                "role": "assistant",
                "content": "Reading file.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": '{"file_path": "main.py"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "def main(): pass"},
            {"role": "assistant", "content": "Done."},
        ]
        result, system = to_anthropic_messages(msgs)
        assert system == "You are helpful."
        assert result[0] == {"role": "user", "content": "Fix main.py"}
        assert result[1]["role"] == "assistant"
        assert any(b["type"] == "tool_use" for b in result[1]["content"])
        assert result[2]["role"] == "user"
        assert result[2]["content"][0]["type"] == "tool_result"
        assert result[3]["role"] == "assistant"


# ============================================================
# to_anthropic_tools (canonical -> Anthropic)
# ============================================================


class TestToAnthropicTools:
    def test_openai_tool_converted(self) -> None:
        tools = [_openai_tool_def("Bash", "Run a command")]
        result = to_anthropic_tools(tools)
        assert len(result) == 1
        t = result[0]
        assert t["name"] == "Bash"
        assert t["description"] == "Run a command"
        assert "input_schema" in t

    def test_anthropic_tool_passthrough(self) -> None:
        tools = [_anthropic_tool_def("Read")]
        result = to_anthropic_tools(tools)
        assert result == tools

    def test_missing_description_omitted(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "X",
                "parameters": {"type": "object"},
            },
        }
        result = to_anthropic_tools([tool])
        assert result[0]["name"] == "X"
        assert "description" not in result[0]

    def test_missing_parameters_gets_empty_schema(self) -> None:
        tool = {
            "type": "function",
            "function": {"name": "Y", "description": "desc"},
        }
        result = to_anthropic_tools([tool])
        assert result[0]["input_schema"] == {}

    def test_unknown_format_passthrough(self) -> None:
        tool = {"something": "unknown"}
        assert to_anthropic_tools([tool]) == [tool]

    def test_mixed_formats(self) -> None:
        tools = [
            _openai_tool_def("A"),
            _anthropic_tool_def("B"),
        ]
        result = to_anthropic_tools(tools)
        assert result[0]["name"] == "A"
        assert "input_schema" in result[0]
        assert result[1] == tools[1]

    def test_versioned_tool_passthrough(self) -> None:
        tools = [_versioned_tool_def("computer_20251124")]
        result = to_anthropic_tools(tools)
        assert result == tools

    def test_cache_control_restored(self) -> None:
        tool = _openai_tool_def("X")
        tool["_cache_control"] = {"type": "ephemeral"}
        result = to_anthropic_tools([tool])
        assert result[0]["cache_control"] == {"type": "ephemeral"}


# ============================================================
# Round-trip: Anthropic -> canonical -> Anthropic
# ============================================================


class TestRoundTrip:
    def test_text_only_assistant_round_trip(self) -> None:
        original = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [_text_block("hello")]},
        ]
        canonical, _ = normalize_messages(original, provider="anthropic")
        restored, _ = to_anthropic_messages(canonical)

        assert restored[0] == {"role": "user", "content": "hi"}
        assert restored[1]["role"] == "assistant"
        assert any(
            b.get("type") == "text" and b.get("text") == "hello"
            for b in restored[1]["content"]
        )

    def test_tool_use_round_trip(self) -> None:
        original = [
            {
                "role": "assistant",
                "content": [
                    _text_block("Let me check."),
                    _tool_use_block("t1", "Read", {"file_path": "/tmp/f"}),
                ],
            },
            {
                "role": "user",
                "content": [_tool_result_block("t1", "file data")],
            },
        ]
        canonical, _ = normalize_messages(original, provider="anthropic")

        assert canonical[0]["role"] == "assistant"
        assert len(canonical[0]["tool_calls"]) == 1
        assert canonical[1]["role"] == "tool"

        restored, _ = to_anthropic_messages(canonical)

        assistant_blocks = restored[0]["content"]
        tool_use_blocks = [b for b in assistant_blocks if b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "Read"
        assert tool_use_blocks[0]["input"] == {"file_path": "/tmp/f"}

        user_msg = restored[1]
        assert user_msg["role"] == "user"
        tr_blocks = [b for b in user_msg["content"] if b.get("type") == "tool_result"]
        assert len(tr_blocks) == 1
        assert tr_blocks[0]["tool_use_id"] == "t1"

    def test_thinking_blocks_round_trip(self) -> None:
        original = [
            {
                "role": "assistant",
                "content": [_thinking_block("deep thought"), _text_block("42")],
            },
        ]
        canonical, _ = normalize_messages(original, provider="anthropic")
        assert canonical[0]["thinking_blocks"][0]["type"] == "thinking"

        restored, _ = to_anthropic_messages(canonical)
        blocks = restored[0]["content"]
        thinking = [b for b in blocks if b.get("type") == "thinking"]
        text = [b for b in blocks if b.get("type") == "text"]
        assert len(thinking) == 1
        assert thinking[0]["thinking"] == "deep thought"
        assert len(text) == 1
        assert text[0]["text"] == "42"

    def test_redacted_thinking_round_trip(self) -> None:
        original = [
            {
                "role": "assistant",
                "content": [
                    _redacted_thinking_block("encrypted_data"),
                    _text_block("answer"),
                ],
            },
        ]
        canonical, _ = normalize_messages(original, provider="anthropic")
        restored, _ = to_anthropic_messages(canonical)
        blocks = restored[0]["content"]
        redacted = [b for b in blocks if b.get("type") == "redacted_thinking"]
        assert len(redacted) == 1
        assert redacted[0]["data"] == "encrypted_data"

    def test_system_message_round_trip(self) -> None:
        original = [
            {"role": "system", "content": [_text_block("Be helpful.")]},
            {"role": "user", "content": "hi"},
        ]
        canonical, _ = normalize_messages(original, provider="anthropic")
        assert canonical[0] == {"role": "system", "content": "Be helpful."}

        restored, system = to_anthropic_messages(canonical)
        assert system == "Be helpful."
        assert restored[0]["role"] == "user"

    def test_tools_round_trip(self) -> None:
        anthropic_tools = [
            _anthropic_tool_def("Bash", "Run command", {"cmd": {"type": "string"}}),
        ]
        _, canonical_tools = normalize_messages(
            [{"role": "user", "content": "hi"}],
            tools=anthropic_tools,
            provider="anthropic",
        )
        assert canonical_tools is not None
        assert canonical_tools[0]["type"] == "function"

        restored_tools = to_anthropic_tools(canonical_tools)
        assert restored_tools[0]["name"] == "Bash"
        assert restored_tools[0]["description"] == "Run command"
        assert restored_tools[0]["input_schema"]["properties"] == {
            "cmd": {"type": "string"}
        }

    def test_multiple_tool_results_round_trip(self) -> None:
        original = [
            {
                "role": "assistant",
                "content": [
                    _tool_use_block("t1", "A", {"x": 1}),
                    _tool_use_block("t2", "B", {"y": 2}),
                ],
            },
            {
                "role": "user",
                "content": [
                    _tool_result_block("t1", "res1"),
                    _tool_result_block("t2", "res2"),
                ],
            },
        ]
        canonical, _ = normalize_messages(original, provider="anthropic")
        assert canonical[1]["role"] == "tool"
        assert canonical[2]["role"] == "tool"

        restored, _ = to_anthropic_messages(canonical)
        user_msg = restored[1]
        assert user_msg["role"] == "user"
        tr_ids = [
            b["tool_use_id"]
            for b in user_msg["content"]
            if b.get("type") == "tool_result"
        ]
        assert tr_ids == ["t1", "t2"]

    def test_image_round_trip_base64(self) -> None:
        original = [
            {
                "role": "user",
                "content": [
                    _text_block("Look at this"),
                    _anthropic_image_block("base64", "image/png", "imgdata"),
                ],
            },
        ]
        canonical, _ = normalize_messages(original, provider="anthropic")
        # Canonical should have image_url
        parts = canonical[0]["content"]
        assert parts[1]["type"] == "image_url"

        restored, _ = to_anthropic_messages(canonical)
        blocks = restored[0]["content"]
        img = [b for b in blocks if b.get("type") == "image"]
        assert len(img) == 1
        assert img[0]["source"]["type"] == "base64"
        assert img[0]["source"]["media_type"] == "image/png"
        assert img[0]["source"]["data"] == "imgdata"

    def test_image_round_trip_url(self) -> None:
        original = [
            {
                "role": "user",
                "content": [
                    _anthropic_image_block("url", url="https://example.com/i.jpg")
                ],
            },
        ]
        canonical, _ = normalize_messages(original, provider="anthropic")
        restored, _ = to_anthropic_messages(canonical)
        block = restored[0]["content"][0]
        assert block["type"] == "image"
        assert block["source"]["url"] == "https://example.com/i.jpg"

    def test_server_tool_use_round_trip(self) -> None:
        original = [
            {
                "role": "assistant",
                "content": [
                    _server_tool_use_block("srv1", "web_search", {"query": "test"}),
                    _web_search_tool_result_block(),
                    _text_block("Found it."),
                ],
            },
        ]
        canonical, _ = normalize_messages(original, provider="anthropic")
        assert canonical[0]["tool_calls"][0]["_server_tool"] is True
        assert len(canonical[0]["_passthrough_blocks"]) == 1

        restored, _ = to_anthropic_messages(canonical)
        blocks = restored[0]["content"]
        types = [b["type"] for b in blocks]
        assert "server_tool_use" in types
        assert "web_search_tool_result" in types
        assert "text" in types

    def test_is_error_round_trip(self) -> None:
        original = [
            {
                "role": "user",
                "content": [_tool_result_block("t1", "Error!", is_error=True)],
            },
        ]
        canonical, _ = normalize_messages(original, provider="anthropic")
        assert canonical[0]["_is_error"] is True

        restored, _ = to_anthropic_messages(canonical)
        block = restored[0]["content"][0]
        assert block["is_error"] is True

    def test_caller_round_trip(self) -> None:
        caller = {"type": "code_execution_20260120", "tool_id": "srv1"}
        block = _tool_use_block("t1", "Read", {"path": "f"})
        block["caller"] = caller
        original = [{"role": "assistant", "content": [block]}]
        canonical, _ = normalize_messages(original, provider="anthropic")
        assert canonical[0]["tool_calls"][0]["_caller"] == caller

        restored, _ = to_anthropic_messages(canonical)
        assert restored[0]["content"][0]["caller"] == caller

    def test_cache_control_on_tool_def_round_trip(self) -> None:
        tool = _anthropic_tool_def("X")
        tool["cache_control"] = {"type": "ephemeral"}
        _, canonical_tools = normalize_messages(
            [{"role": "user", "content": "hi"}], tools=[tool], provider="anthropic"
        )
        assert canonical_tools[0]["_cache_control"] == {"type": "ephemeral"}

        restored = to_anthropic_tools(canonical_tools)
        assert restored[0]["cache_control"] == {"type": "ephemeral"}

    def test_versioned_tool_round_trip(self) -> None:
        tools = [_versioned_tool_def("computer_20251124"), _anthropic_tool_def("Read")]
        _, canonical_tools = normalize_messages(
            [{"role": "user", "content": "hi"}], tools=tools, provider="anthropic"
        )
        # Versioned tool passes through, regular tool is converted
        assert canonical_tools[0]["type"] == "computer_20251124"
        assert canonical_tools[1]["type"] == "function"

        restored = to_anthropic_tools(canonical_tools)
        assert restored[0]["type"] == "computer_20251124"
        assert restored[1]["name"] == "Read"

    def test_document_block_round_trip(self) -> None:
        doc = _document_block("my doc")
        original = [{"role": "user", "content": [_text_block("context"), doc]}]
        canonical, _ = normalize_messages(original, provider="anthropic")
        # Document preserved as content part
        parts = canonical[0]["content"]
        assert any(p.get("type") == "document" for p in parts)

        restored, _ = to_anthropic_messages(canonical)
        blocks = restored[0]["content"]
        assert any(b.get("type") == "document" for b in blocks)

    def test_full_multi_turn_conversation_round_trip(self) -> None:
        """Realistic multi-turn Claude Code conversation with extended features."""
        original = [
            {
                "role": "system",
                "content": [
                    _text_block("x-anthropic-billing-header: proj-123"),
                    _text_block("You are a coding assistant."),
                ],
            },
            {"role": "user", "content": "Fix the bug"},
            {
                "role": "assistant",
                "content": [
                    _thinking_block("Need to read the file first"),
                    _redacted_thinking_block("classified"),
                    _text_block("I will read the file."),
                    _tool_use_block("t1", "Read", {"file_path": "bug.py"}),
                ],
            },
            {
                "role": "user",
                "content": [_tool_result_block("t1", "def buggy(): return None")],
            },
            {
                "role": "assistant",
                "content": [_text_block("Fixed!")],
            },
        ]
        canonical, _ = normalize_messages(original, provider="anthropic")
        restored, system = to_anthropic_messages(canonical)

        assert system == "You are a coding assistant."
        assert restored[0] == {"role": "user", "content": "Fix the bug"}

        asst = restored[1]
        assert asst["role"] == "assistant"
        types = [b["type"] for b in asst["content"]]
        assert "thinking" in types
        assert "redacted_thinking" in types
        assert "text" in types
        assert "tool_use" in types

        assert restored[2]["role"] == "user"
        assert restored[2]["content"][0]["type"] == "tool_result"

        assert restored[3]["role"] == "assistant"


# ============================================================
# _sanitize_tool_id
# ============================================================


class TestSanitizeToolId:
    def test_valid_id_unchanged(self) -> None:
        assert (
            _sanitize_tool_id("toolu_01A09q90qw90lq917835lq9")
            == "toolu_01A09q90qw90lq917835lq9"
        )

    def test_empty_string_unchanged(self) -> None:
        assert _sanitize_tool_id("") == ""

    def test_server_tool_id_unchanged(self) -> None:
        assert _sanitize_tool_id("srvtoolu_01abc") == "srvtoolu_01abc"

    def test_replaces_invalid_characters(self) -> None:
        assert _sanitize_tool_id("tool.call:123!@#") == "tool_call_123___"

    def test_hyphens_preserved(self) -> None:
        assert _sanitize_tool_id("tool-call-123") == "tool-call-123"

    def test_spaces_replaced(self) -> None:
        assert _sanitize_tool_id("tool call 123") == "tool_call_123"


# ============================================================
# Tool ID sanitization in emitters
# ============================================================


class TestToolIdSanitizationInEmitters:
    def test_emit_anthropic_assistant_sanitizes_tool_ids(self) -> None:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call.123!bad",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ],
        }
        result = _emit_anthropic_assistant(msg)
        assert result["content"][0]["id"] == "call_123_bad"

    def test_emit_anthropic_tool_result_sanitizes_tool_ids(self) -> None:
        msg: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": "call.123!bad",
            "content": "result",
        }
        result = _emit_anthropic_tool_result(msg)
        assert result["content"][0]["tool_use_id"] == "call_123_bad"


# ============================================================
# Thinking block interleaving with server tools
# ============================================================


class TestThinkingBlockInterleaving:
    def test_interleaved_thinking_server_tool_preserves_order(self) -> None:
        """Anthropic requires thinking blocks interleaved with server_tool_use
        to maintain signature verification. Verify round-trip preserves order."""
        original = [
            {"role": "user", "content": "Search for Python docs"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I'll search for this"},
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_01",
                        "name": "web_search",
                        "input": {"query": "Python docs"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_01",
                        "content": [{"type": "text", "text": "Results here"}],
                    },
                    {"type": "thinking", "thinking": "Now I'll summarize"},
                    {"type": "text", "text": "Here are the results"},
                ],
            },
        ]
        canonical, _ = normalize_messages(original, provider="anthropic")

        # Verify block_order is preserved
        asst = canonical[1]
        assert "_block_order" in asst
        order = asst["_block_order"]
        kinds = [k for k, _ in order]
        assert kinds == ["thinking", "tool_call", "passthrough", "thinking", "text"]

        # Round-trip back to Anthropic
        restored, _ = to_anthropic_messages(canonical)
        asst_restored = restored[1]
        types = [b["type"] for b in asst_restored["content"]]
        assert types == [
            "thinking",
            "server_tool_use",
            "web_search_tool_result",
            "thinking",
            "text",
        ]

    def test_no_block_order_when_no_server_tools(self) -> None:
        """Regular tool_use (not server) should not get _block_order."""
        original = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think"},
                    _tool_use_block(),
                ],
            }
        ]
        canonical = _normalize_anthropic_messages(original)
        assert "_block_order" not in canonical[0]

    def test_no_block_order_when_no_thinking(self) -> None:
        """Server tools without thinking should not get _block_order."""
        original = [
            {
                "role": "assistant",
                "content": [
                    _server_tool_use_block(),
                ],
            }
        ]
        canonical = _normalize_anthropic_messages(original)
        assert "_block_order" not in canonical[0]


# ============================================================
# MCP server tool support
# ============================================================


class TestMcpServerTools:
    def test_detect_provider_mcp_server_tool(self) -> None:
        tools = [
            {
                "type": "url",
                "url": "https://mcp.example.com",
                "name": "my_server",
                "tool_configuration": {"allowed_tools": ["get_data"]},
            }
        ]
        assert _detect_provider([], tools) == "anthropic"

    def test_normalize_anthropic_tools_mcp_passthrough(self) -> None:
        tools = [
            {
                "type": "url",
                "url": "https://mcp.example.com",
                "name": "my_server",
            }
        ]
        result = _normalize_anthropic_tools(tools)
        assert result == tools

    def test_to_anthropic_tools_mcp_passthrough(self) -> None:
        tools = [
            {
                "type": "url",
                "url": "https://mcp.example.com",
                "name": "my_server",
            }
        ]
        result = to_anthropic_tools(tools)
        assert result == tools


# ============================================================
# OpenAI passthrough content types
# ============================================================


class TestOpenAIPassthroughContentTypes:
    @pytest.mark.parametrize(
        "content_type",
        [
            param("input_audio", id="input_audio"),
            param("audio_url", id="audio_url"),
            param("guarded_text", id="guarded_text"),
            param("video_url", id="video_url"),
            param("file", id="file"),
        ],
    )
    def test_passthrough_content_types_in_user_message(self, content_type: str) -> None:
        """Content types valid in OpenAI but not Anthropic should pass through."""
        block = {"type": content_type, "data": "test_data"}
        msg: dict[str, Any] = {"role": "user", "content": [block]}
        result = _normalize_anthropic_user(msg)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert isinstance(content, list)
        assert block in content

    def test_file_content_alongside_text(self) -> None:
        msg: dict[str, Any] = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this file"},
                {"type": "file", "file": {"file_id": "abc123"}},
            ],
        }
        result = _normalize_anthropic_user(msg)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "file"


# ============================================================
# _tool_call_to_anthropic_block
# ============================================================


class TestToolCallToAnthropicBlock:
    def test_regular_tool_call(self) -> None:
        tc: dict[str, Any] = {
            "id": "toolu_01",
            "type": "function",
            "function": {"name": "Read", "arguments": '{"path": "/tmp"}'},
        }
        block = _tool_call_to_anthropic_block(tc)
        assert block["type"] == "tool_use"
        assert block["id"] == "toolu_01"
        assert block["name"] == "Read"
        assert block["input"] == {"path": "/tmp"}

    def test_server_tool_call(self) -> None:
        tc: dict[str, Any] = {
            "id": "srvtoolu_01",
            "type": "function",
            "function": {"name": "web_search", "arguments": '{"query": "test"}'},
            "_server_tool": True,
        }
        block = _tool_call_to_anthropic_block(tc)
        assert block["type"] == "server_tool_use"

    def test_caller_preserved(self) -> None:
        tc: dict[str, Any] = {
            "id": "toolu_01",
            "type": "function",
            "function": {"name": "Read", "arguments": "{}"},
            "_caller": {"type": "direct"},
        }
        block = _tool_call_to_anthropic_block(tc)
        assert block["caller"] == {"type": "direct"}

    def test_sanitizes_invalid_id(self) -> None:
        tc: dict[str, Any] = {
            "id": "call.123!bad",
            "type": "function",
            "function": {"name": "test", "arguments": "{}"},
        }
        block = _tool_call_to_anthropic_block(tc)
        assert block["id"] == "call_123_bad"
