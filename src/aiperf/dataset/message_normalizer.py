# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize provider-native messages and tools to OpenAI canonical format.

This module converts Anthropic, OpenAI, and other provider message formats
into OpenAI chat completions format -- the canonical internal representation
used by AIPerf. Each loader calls ``normalize_messages()`` once at load time,
and each endpoint emitter converts from canonical to its wire format.

Architecture: N normalizers (provider -> canonical) + M emitters (canonical -> wire)
instead of NxM direct converters.

Provider-specific metadata is stored in a single ``_meta`` dict on each message,
tool_call, or tool definition that needs it. This consolidates round-trip fidelity
data into one key rather than scattering underscore-prefixed fields.

Metadata keys stored in ``_meta``:
- ``is_error``: on role:tool messages, from Anthropic tool_result ``is_error``
- ``caller``: on tool_call dicts, from Anthropic tool_use ``caller``
- ``citations``: on canonical messages, from Anthropic text block ``citations``
- ``cache_control``: on canonical messages/tool dicts, from Anthropic ``cache_control``
- ``server_tool``: on tool_call dicts, marks Anthropic ``server_tool_use`` origin
- ``passthrough_blocks``: on canonical messages, Anthropic blocks with no OpenAI equivalent
- ``block_order``: on canonical messages, original block ordering for interleaved
  thinking + server tools (needed for Anthropic signature verification)
"""

from __future__ import annotations

import re
from typing import Any

import orjson

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anthropic block types preserved opaquely for round-trip fidelity
_ANTHROPIC_PASSTHROUGH_BLOCK_TYPES = frozenset(
    {"document", "search_result", "container_upload", "tool_reference"}
)

# Server tool result block types (returned inline in assistant messages)
_SERVER_TOOL_RESULT_TYPES = frozenset(
    {
        "web_search_tool_result",
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "tool_search_tool_result",
    }
)

# Union of both passthrough sets -- used in the normalizer where the distinction
# doesn't matter (both go to passthrough_blocks).
_ALL_PASSTHROUGH_BLOCK_TYPES = (
    _ANTHROPIC_PASSTHROUGH_BLOCK_TYPES | _SERVER_TOOL_RESULT_TYPES
)

# Versioned server/computer-use tool type pattern
_VERSIONED_TOOL_TYPE_RE = re.compile(
    r"^(computer|bash|text_editor|web_search|web_fetch|code_execution|memory|"
    r"tool_search_tool_bm25|tool_search_tool_regex)_\d{8}$"
)

# Tool ID must match ^[a-zA-Z0-9_-]+$
_VALID_TOOL_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_INVALID_TOOL_ID_CHAR_RE = re.compile(r"[^a-zA-Z0-9_-]")

# OpenAI content types that pass through without conversion
_OPENAI_PASSTHROUGH_CONTENT_TYPES = frozenset(
    {"input_audio", "audio_url", "guarded_text", "video_url", "file"}
)

_BILLING_PREFIX = "x-anthropic-billing-header:"

# Anthropic-specific message block types used for provider detection
_ANTHROPIC_MESSAGE_BLOCK_TYPES = frozenset(
    {"tool_use", "tool_result", "thinking", "redacted_thinking", "server_tool_use"}
)

_META = "_meta"


# ---------------------------------------------------------------------------
# _meta accessors -- single place for all metadata read/write
# ---------------------------------------------------------------------------


def _get_meta(d: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read a metadata value from the ``_meta`` dict."""
    meta = d.get(_META)
    if meta is None:
        return default
    return meta.get(key, default)


def _set_meta(d: dict[str, Any], key: str, value: Any) -> None:
    """Write a metadata value into the ``_meta`` dict."""
    meta = d.get(_META)
    if meta is None:
        meta = {}
        d[_META] = meta
    meta[key] = value


def _has_meta(d: dict[str, Any], key: str) -> bool:
    """Check whether a metadata key exists."""
    meta = d.get(_META)
    return meta is not None and key in meta


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _is_passthrough_tool(tool: dict[str, Any]) -> bool:
    """Return True if *tool* should pass through without conversion.

    Matches versioned server/computer-use tools and MCP server tools.
    """
    tool_type = tool.get("type", "")
    if _VERSIONED_TOOL_TYPE_RE.match(tool_type):
        return True
    return tool_type == "url" and "url" in tool


def _sanitize_tool_id(tool_id: str) -> str:
    """Sanitize a tool ID to match Anthropic's pattern: ``^[a-zA-Z0-9_-]+$``."""
    if not tool_id or _VALID_TOOL_ID_RE.match(tool_id):
        return tool_id
    return _INVALID_TOOL_ID_CHAR_RE.sub("_", tool_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    provider: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Normalize provider-native messages and tools to OpenAI canonical format.

    Args:
        messages: Provider-native message dicts (from Conflux UnifiedMessage,
            raw API capture, etc.)
        tools: Provider-native tool definitions, or None.
        provider: Provider hint -- ``"anthropic"``, ``"openai"``, or ``None``
            for auto-detection.

    Returns:
        Tuple of (normalized_messages, normalized_tools) in OpenAI format.
    """
    if provider is None:
        provider = _detect_provider(messages, tools)

    # Strip Conflux metadata
    messages = [
        {k: v for k, v in m.items() if k != "tokens"} if "tokens" in m else m
        for m in messages
    ]

    if provider == "anthropic":
        messages = _normalize_anthropic_messages(messages)
        tools = _normalize_anthropic_tools(tools) if tools else tools

    return messages, tools


def _detect_provider(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Auto-detect provider from message and tool content shapes."""
    if tools:
        for tool in tools:
            if "input_schema" in tool and "function" not in tool:
                return "anthropic"
            if _is_passthrough_tool(tool):
                return "anthropic"
            if "function" in tool or "parameters" in tool:
                return "openai"

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type in _ANTHROPIC_MESSAGE_BLOCK_TYPES:
                    return "anthropic"
                if block_type == "image" and isinstance(block.get("source"), dict):
                    return "anthropic"
                if block_type in _ALL_PASSTHROUGH_BLOCK_TYPES:
                    return "anthropic"
            # Assistant messages with list-of-blocks content are Anthropic-native;
            # OpenAI assistant messages use plain string content.
            if msg.get("role") == "assistant":
                return "anthropic"
        if msg.get("tool_calls") is not None:
            return "openai"
        if msg.get("role") == "tool" and "tool_call_id" in msg:
            return "openai"

    return "openai"


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI canonical
# ---------------------------------------------------------------------------


def _normalize_anthropic_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Anthropic message format to OpenAI canonical format."""
    result: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            flattened = _flatten_text_content(
                msg.get("content"), strip_billing_headers=True
            )
            if flattened:
                result.append({"role": "system", "content": flattened})
        elif role == "assistant":
            result.extend(_normalize_anthropic_assistant(msg))
        elif role == "user":
            result.extend(_normalize_anthropic_user(msg))
        else:
            result.append(msg)

    return result


def _normalize_anthropic_assistant(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an Anthropic assistant message to OpenAI format.

    Tracks original block ordering via ``block_order`` in _meta when thinking
    blocks are interleaved with server tools (needed for Anthropic signature
    verification).
    """
    content = msg.get("content")
    if not isinstance(content, list):
        return [msg]

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    thinking_blocks: list[dict[str, Any]] = []
    passthrough_blocks: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    block_order: list[tuple[str, int]] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
                # Must be inside `if text` -- moving outside produces stale index.
                block_order.append(("text", len(text_parts) - 1))
            if "citations" in block:
                citations.extend(block["citations"])

        elif block_type in ("tool_use", "server_tool_use"):
            tool_call = _anthropic_tool_use_to_call(block)
            if block_type == "server_tool_use":
                _set_meta(tool_call, "server_tool", True)
            tool_calls.append(tool_call)
            block_order.append(("tool_call", len(tool_calls) - 1))

        elif block_type in ("thinking", "redacted_thinking"):
            thinking_blocks.append(block)
            block_order.append(("thinking", len(thinking_blocks) - 1))

        elif block_type in _ALL_PASSTHROUGH_BLOCK_TYPES:
            passthrough_blocks.append(block)
            block_order.append(("passthrough", len(passthrough_blocks) - 1))

    out: dict[str, Any] = {"role": "assistant"}

    combined_text = "\n\n".join(text_parts) if text_parts else None
    if tool_calls:
        out["content"] = combined_text or ""
        out["tool_calls"] = tool_calls
    elif combined_text is not None:
        out["content"] = combined_text
    else:
        out["content"] = ""

    if thinking_blocks:
        out["thinking_blocks"] = thinking_blocks
    if passthrough_blocks:
        _set_meta(out, "passthrough_blocks", passthrough_blocks)
    if citations:
        _set_meta(out, "citations", citations)

    # Only store block order for interleaved thinking + server tool messages
    if thinking_blocks and any(_get_meta(tc, "server_tool") for tc in tool_calls):
        _set_meta(out, "block_order", block_order)

    return [out]


def _anthropic_tool_use_to_call(block: dict[str, Any]) -> dict[str, Any]:
    """Convert a single Anthropic tool_use/server_tool_use block to OpenAI tool_call."""
    tool_call: dict[str, Any] = {
        "id": block.get("id", ""),
        "type": "function",
        "function": {
            "name": block.get("name", ""),
            "arguments": (
                orjson.dumps(block["input"]).decode()
                if isinstance(block.get("input"), dict)
                else str(block.get("input", "{}"))
            ),
        },
    }
    if "caller" in block:
        _set_meta(tool_call, "caller", block["caller"])
    if "cache_control" in block:
        _set_meta(tool_call, "cache_control", block["cache_control"])
    return tool_call


def _normalize_anthropic_user(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an Anthropic user message to OpenAI format."""
    content = msg.get("content")
    if not isinstance(content, list):
        return [msg]

    text_parts: list[str] = []
    content_parts: list[dict[str, Any]] = []
    has_unconvertible_blocks = False
    result: list[dict[str, Any]] = []

    # Build result preserving original interleaved order of user content and
    # tool_results. Flush accumulated user text/content before each tool_result.
    def _flush_user() -> None:
        if text_parts and not content_parts and not has_unconvertible_blocks:
            result.append({"role": "user", "content": "\n\n".join(text_parts)})
        elif text_parts or content_parts:
            combined = [{"type": "text", "text": t} for t in text_parts]
            combined.extend(content_parts)
            result.append({"role": "user", "content": combined})
        text_parts.clear()
        content_parts.clear()

    for block in content:
        if not isinstance(block, dict):
            if isinstance(block, str):
                text_parts.append(block)
            continue
        block_type = block.get("type")

        if block_type == "tool_result":
            _flush_user()
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": _flatten_text_content(block.get("content")),
            }
            if block.get("is_error"):
                _set_meta(tool_msg, "is_error", True)
            if "cache_control" in block:
                _set_meta(tool_msg, "cache_control", block["cache_control"])
            result.append(tool_msg)
        elif block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
        elif block_type == "image":
            content_parts.append(_anthropic_image_to_openai(block))
        elif block_type in _OPENAI_PASSTHROUGH_CONTENT_TYPES:
            content_parts.append(block)
        else:
            content_parts.append(block)
            has_unconvertible_blocks = True

    _flush_user()
    return result if result else [msg]


def _flatten_text_content(content: Any, *, strip_billing_headers: bool = False) -> str:
    """Flatten Anthropic content (string, list-of-blocks, or nested) to a single string."""
    if isinstance(content, str):
        if strip_billing_headers:
            return _strip_billing_headers(content)
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text")
            else:
                continue
            if not isinstance(text, str) or not text:
                continue
            if strip_billing_headers and text.startswith(_BILLING_PREFIX):
                continue
            parts.append(text)
        return "\n\n".join(parts)
    if content is None:
        return ""
    if isinstance(content, dict):
        return orjson.dumps(content).decode()
    return str(content)


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------


def _anthropic_image_to_openai(block: dict[str, Any]) -> dict[str, Any]:
    """Convert Anthropic image block to OpenAI image_url content part."""
    source = block.get("source", {})
    source_type = source.get("type")

    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        url = f"data:{media_type};base64,{data}"
    elif source_type == "url":
        url = source.get("url", "")
    else:
        return block

    result: dict[str, Any] = {"type": "image_url", "image_url": {"url": url}}
    if "cache_control" in block:
        _set_meta(result, "cache_control", block["cache_control"])
    return result


def _openai_image_to_anthropic(part: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI image_url content part to Anthropic image block."""
    image_url = part.get("image_url", {})
    url = image_url.get("url", "")

    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = "image/png"
        if ":" in header and ";" in header:
            media_type = header.split(":", 1)[1].split(";", 1)[0]
        result: dict[str, Any] = {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    else:
        result = {"type": "image", "source": {"type": "url", "url": url}}

    if _has_meta(part, "cache_control"):
        result["cache_control"] = _get_meta(part, "cache_control")
    return result


# ---------------------------------------------------------------------------
# Anthropic tools -> OpenAI tools
# ---------------------------------------------------------------------------


def _normalize_anthropic_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    result: list[dict[str, Any]] = []

    for tool in tools:
        if "function" in tool or _is_passthrough_tool(tool):
            result.append(tool)
            continue

        schema_key = (
            "input_schema"
            if "input_schema" in tool
            else "parameters"
            if "parameters" in tool and "name" in tool
            else None
        )
        if schema_key:
            converted: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", "(no description)"),
                    "parameters": tool.get(schema_key, {}),
                },
            }
            if "cache_control" in tool:
                _set_meta(converted, "cache_control", tool["cache_control"])
            result.append(converted)
            continue

        result.append(tool)

    return result


def _strip_billing_headers(text: str) -> str:
    """Remove x-anthropic-billing-header lines from system text."""
    lines = text.split("\n")
    filtered = [line for line in lines if not line.startswith(_BILLING_PREFIX)]
    return "\n".join(filtered).strip()


# ---------------------------------------------------------------------------
# Canonical -> Anthropic (emitter)
# ---------------------------------------------------------------------------


def to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | list[dict[str, Any]] | None]:
    """Convert canonical (OpenAI) messages to Anthropic format.

    Returns:
        Tuple of (messages, system) where system is extracted from the first
        system message if present, or None.
    """
    raw: list[dict[str, Any]] = []
    system_parts: list[str] = []

    for msg in messages:
        role = msg.get("role")

        if role in ("system", "developer"):
            content = msg.get("content")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.append(_flatten_text_content(content))
            continue

        if role == "assistant":
            raw.append(_emit_anthropic_assistant(msg))
        elif role == "tool":
            raw.append(_emit_anthropic_tool_result(msg))
        elif role == "user":
            raw.append(_emit_anthropic_user(msg))
        else:
            raw.append(msg)

    system = "\n\n".join(system_parts) if system_parts else None
    return _merge_consecutive_roles(raw), system


def to_anthropic_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert canonical OpenAI tool definitions to Anthropic format."""
    result: list[dict[str, Any]] = []

    for tool in tools:
        if ("input_schema" in tool and "function" not in tool) or _is_passthrough_tool(
            tool
        ):
            result.append(tool)
            continue

        func = tool.get("function")
        if isinstance(func, dict):
            converted: dict[str, Any] = {
                "name": func.get("name", ""),
                "input_schema": func.get("parameters", {}),
            }
            desc = func.get("description")
            if desc:
                converted["description"] = desc
            if _has_meta(tool, "cache_control"):
                converted["cache_control"] = _get_meta(tool, "cache_control")
            result.append(converted)
            continue

        result.append(tool)

    return result


def _emit_anthropic_assistant(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a canonical assistant message to Anthropic content blocks.

    When ``block_order`` is present in _meta (interleaved thinking + server
    tools), restores the original block ordering required for Anthropic
    signature verification.
    """
    block_order = _get_meta(msg, "block_order")
    if block_order:
        return _emit_anthropic_assistant_ordered(msg, block_order)

    content_blocks: list[dict[str, Any]] = []
    content_blocks.extend(msg.get("thinking_blocks", []))

    text = msg.get("content")
    if isinstance(text, str) and text:
        content_blocks.append({"type": "text", "text": text})
    elif isinstance(text, list):
        content_blocks.extend(text)

    _append_refusal(msg, content_blocks)

    for tc in msg.get("tool_calls", []):
        content_blocks.append(_tool_call_to_anthropic_block(tc))

    content_blocks.extend(_get_meta(msg, "passthrough_blocks", []))

    result: dict[str, Any] = {
        "role": "assistant",
        "content": content_blocks or [{"type": "text", "text": ""}],
    }
    if _has_meta(msg, "cache_control"):
        result["cache_control"] = _get_meta(msg, "cache_control")
    return result


def _emit_anthropic_assistant_ordered(
    msg: dict[str, Any],
    block_order: list[tuple[str, int]],
) -> dict[str, Any]:
    """Emit assistant message with original block ordering restored."""
    thinking_blocks = msg.get("thinking_blocks", [])
    tool_calls = msg.get("tool_calls", [])
    passthrough_blocks = _get_meta(msg, "passthrough_blocks", [])

    text = msg.get("content")
    if isinstance(text, str):
        text_parts = [text] if text else []
    elif isinstance(text, list):
        text_parts = [b.get("text", "") for b in text if isinstance(b, dict)]
    else:
        text_parts = []

    content_blocks: list[dict[str, Any]] = []
    for kind, idx in block_order:
        if kind == "thinking" and idx < len(thinking_blocks):
            content_blocks.append(thinking_blocks[idx])
        elif kind == "text" and idx < len(text_parts):
            content_blocks.append({"type": "text", "text": text_parts[idx]})
        elif kind == "tool_call" and idx < len(tool_calls):
            content_blocks.append(_tool_call_to_anthropic_block(tool_calls[idx]))
        elif kind == "passthrough" and idx < len(passthrough_blocks):
            content_blocks.append(passthrough_blocks[idx])

    _append_refusal(msg, content_blocks)

    return {
        "role": "assistant",
        "content": content_blocks or [{"type": "text", "text": ""}],
    }


def _append_refusal(msg: dict[str, Any], blocks: list[dict[str, Any]]) -> None:
    """Append a refusal text block if present."""
    refusal = msg.get("refusal")
    if isinstance(refusal, str) and refusal:
        blocks.append({"type": "text", "text": refusal})


def _tool_call_to_anthropic_block(tc: dict[str, Any]) -> dict[str, Any]:
    """Convert a single OpenAI tool_call dict to an Anthropic tool_use block."""
    func = tc.get("function", {})
    arguments = func.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            input_val = orjson.loads(arguments)
        except orjson.JSONDecodeError:
            input_val = arguments
    else:
        input_val = arguments

    block_type = "server_tool_use" if _get_meta(tc, "server_tool") else "tool_use"
    tool_block: dict[str, Any] = {
        "type": block_type,
        "id": _sanitize_tool_id(tc.get("id", "")),
        "name": func.get("name", ""),
        "input": input_val,
    }
    if _has_meta(tc, "caller"):
        tool_block["caller"] = _get_meta(tc, "caller")
    if _has_meta(tc, "cache_control"):
        tool_block["cache_control"] = _get_meta(tc, "cache_control")
    return tool_block


def _emit_anthropic_tool_result(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a canonical role:tool message to Anthropic tool_result user message."""
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": _sanitize_tool_id(msg.get("tool_call_id", "")),
        "content": msg.get("content", ""),
    }
    if _get_meta(msg, "is_error"):
        block["is_error"] = True
    if _has_meta(msg, "cache_control"):
        block["cache_control"] = _get_meta(msg, "cache_control")
    return {"role": "user", "content": [block]}


def _emit_anthropic_user(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a canonical user message to Anthropic format."""
    content = msg.get("content")
    if isinstance(content, str):
        return {"role": "user", "content": content}
    if isinstance(content, list):
        converted = [
            _openai_image_to_anthropic(part)
            if isinstance(part, dict) and part.get("type") == "image_url"
            else part
            for part in content
        ]
        return {"role": "user", "content": converted}
    return msg


def _merge_consecutive_roles(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge consecutive messages with the same role.

    Anthropic requires strict user/assistant alternation.
    """
    if not messages:
        return []

    merged: list[dict[str, Any]] = [messages[0]]

    for msg in messages[1:]:
        prev = merged[-1]
        if msg.get("role") == prev.get("role"):
            prev["content"] = _merge_content(prev.get("content"), msg.get("content"))
        else:
            merged.append(msg)

    return merged


def _merge_content(
    a: str | list[dict[str, Any]] | None,
    b: str | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge two content values into a list of content blocks."""
    a_list = [{"type": "text", "text": a}] if isinstance(a, str) else (a or [])
    b_list = [{"type": "text", "text": b}] if isinstance(b, str) else (b or [])
    return a_list + b_list
