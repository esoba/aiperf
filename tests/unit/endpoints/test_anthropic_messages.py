# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AnthropicMessagesEndpoint."""

import pytest

from aiperf.common.enums import SSEFieldType
from aiperf.common.models import Text, Turn
from aiperf.common.models.record_models import (
    ReasoningResponseData,
    SSEField,
    SSEMessage,
    TextResponseData,
)
from aiperf.endpoints.anthropic_messages import AnthropicMessagesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestAnthropicMessagesFormatPayload:
    """Tests for AnthropicMessagesEndpoint format_payload."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(EndpointType.ANTHROPIC_MESSAGES)

    @pytest.fixture
    def streaming_model_endpoint(self):
        return create_model_endpoint(EndpointType.ANTHROPIC_MESSAGES, streaming=True)

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            AnthropicMessagesEndpoint, model_endpoint
        )

    def test_simple_text(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Hello!"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "claude-sonnet-4-20250514"
        assert payload["stream"] is False
        assert payload["max_tokens"] == 1024
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hello!"
        assert "system" not in payload

    def test_max_tokens_from_turn(self, endpoint, model_endpoint):
        turn = Turn(
            texts=[Text(contents=["Test"])],
            model="claude-sonnet-4-20250514",
            max_tokens=500,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["max_tokens"] == 500

    def test_max_tokens_default(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Test"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["max_tokens"] == 1024

    def test_system_message_as_top_level(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Hello"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            system_message="You are a helpful assistant.",
        )

        payload = endpoint.format_payload(request_info)

        assert payload["system"] == "You are a helpful assistant."
        # System message should NOT be in messages
        for msg in payload["messages"]:
            assert msg["role"] != "system"

    def test_multi_turn(self, endpoint, model_endpoint):
        turns = [
            Turn(
                texts=[Text(contents=["Hello"])],
                role="user",
                model="claude-sonnet-4-20250514",
            ),
            Turn(
                texts=[Text(contents=["Hi there!"])],
                role="assistant",
                model="claude-sonnet-4-20250514",
            ),
            Turn(
                texts=[Text(contents=["How are you?"])],
                role="user",
                model="claude-sonnet-4-20250514",
            ),
        ]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)

        payload = endpoint.format_payload(request_info)

        assert len(payload["messages"]) == 3
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][1]["role"] == "assistant"
        assert payload["messages"][2]["role"] == "user"

    def test_streaming_enabled(self, streaming_model_endpoint):
        endpoint = create_endpoint_with_mock_transport(
            AnthropicMessagesEndpoint, streaming_model_endpoint
        )
        turn = Turn(texts=[Text(contents=["Test"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(
            model_endpoint=streaming_model_endpoint, turns=[turn]
        )

        payload = endpoint.format_payload(request_info)

        assert payload["stream"] is True

    def test_extra_params(self):
        extra_params = [("temperature", 0.7), ("top_p", 0.9)]
        model_endpoint = create_model_endpoint(
            EndpointType.ANTHROPIC_MESSAGES, extra=extra_params
        )
        endpoint = create_endpoint_with_mock_transport(
            AnthropicMessagesEndpoint, model_endpoint
        )
        turn = Turn(texts=[Text(contents=["Test"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9

    def test_model_fallback(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Test"])], model=None)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == model_endpoint.primary_model_name

    def test_empty_turns_raises(self, endpoint, model_endpoint):
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_user_context_message_prepended(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Hello"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            user_context_message="Context info",
        )

        payload = endpoint.format_payload(request_info)

        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["content"] == "Context info"
        assert payload["messages"][0]["role"] == "user"

    def test_raw_messages_string_content(self, endpoint, model_endpoint):
        """raw_messages with string content is expanded into messages list."""
        turn = Turn(
            raw_messages=[{"role": "user", "content": "verbatim user text"}],
            model="claude-sonnet-4-20250514",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["messages"][0]["content"] == "verbatim user text"

    def test_raw_messages_content_blocks(self, endpoint, model_endpoint):
        """raw_messages with content blocks are expanded directly."""
        blocks = [
            {"type": "tool_result", "tool_use_id": "tu-1", "content": "file data"},
            {"type": "text", "text": "Here is the file"},
        ]
        turn = Turn(
            raw_messages=[{"role": "user", "content": blocks}],
            model="claude-sonnet-4-20250514",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["messages"][0]["content"] == blocks
        assert payload["messages"][0]["content"][0]["type"] == "tool_result"

    def test_raw_messages_multi_message_turn(self, endpoint, model_endpoint):
        """Multi-message raw_messages on a single turn expand into multiple messages."""
        turn = Turn(
            raw_messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu-1", "content": "OK"}
                    ],
                },
            ],
            model="claude-sonnet-4-20250514",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["messages"]) == 3
        assert payload["messages"][0]["content"] == "Hello"
        assert payload["messages"][1]["role"] == "assistant"
        assert payload["messages"][1]["content"] == [{"type": "text", "text": "Hi!"}]
        assert payload["messages"][2]["content"][0]["type"] == "tool_result"

    def test_raw_messages_takes_precedence_over_texts(self, endpoint, model_endpoint):
        """When raw_messages is set, texts are ignored."""
        turn = Turn(
            raw_messages=[{"role": "user", "content": "raw wins"}],
            texts=[Text(contents=["should be ignored"])],
            model="claude-sonnet-4-20250514",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["messages"][0]["content"] == "raw wins"


class TestAnthropicMessagesHeaders:
    """Tests for AnthropicMessagesEndpoint get_endpoint_headers."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(EndpointType.ANTHROPIC_MESSAGES)

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            AnthropicMessagesEndpoint, model_endpoint
        )

    def test_default_headers(self, endpoint, model_endpoint):
        request_info = create_request_info(model_endpoint=model_endpoint)

        headers = endpoint.get_endpoint_headers(request_info)

        assert headers["content-type"] == "application/json"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in headers

    def test_api_key_as_x_api_key(self):
        from aiperf.common.enums import ModelSelectionStrategy
        from aiperf.common.models.model_endpoint_info import (
            EndpointInfo,
            ModelEndpointInfo,
            ModelInfo,
            ModelListInfo,
        )

        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.ANTHROPIC_MESSAGES,
                base_url="http://localhost:8000",
                streaming=False,
                extra=[],
                api_key="sk-ant-test-key",
            ),
        )
        endpoint = create_endpoint_with_mock_transport(
            AnthropicMessagesEndpoint, model_endpoint
        )
        request_info = create_request_info(model_endpoint=model_endpoint)

        headers = endpoint.get_endpoint_headers(request_info)

        assert headers["x-api-key"] == "sk-ant-test-key"
        assert "Authorization" not in headers

    def test_custom_headers_merged(self):
        from aiperf.common.enums import ModelSelectionStrategy
        from aiperf.common.models.model_endpoint_info import (
            EndpointInfo,
            ModelEndpointInfo,
            ModelInfo,
            ModelListInfo,
        )

        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.ANTHROPIC_MESSAGES,
                base_url="http://localhost:8000",
                streaming=False,
                extra=[],
                headers=[("anthropic-beta", "extended-thinking-2025-04-11")],
            ),
        )
        endpoint = create_endpoint_with_mock_transport(
            AnthropicMessagesEndpoint, model_endpoint
        )
        request_info = create_request_info(model_endpoint=model_endpoint)

        headers = endpoint.get_endpoint_headers(request_info)

        assert headers["anthropic-beta"] == "extended-thinking-2025-04-11"
        assert headers["anthropic-version"] == "2023-06-01"


class TestAnthropicMessagesParseResponseNonStreaming:
    """Tests for AnthropicMessagesEndpoint parse_response (non-streaming)."""

    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.ANTHROPIC_MESSAGES)
        return create_endpoint_with_mock_transport(
            AnthropicMessagesEndpoint, model_endpoint
        )

    def test_text_block(self, endpoint):
        mock_response = create_mock_response(
            123456789,
            {
                "type": "message",
                "content": [{"type": "text", "text": "Hello, how can I help?"}],
                "usage": {"input_tokens": 10, "output_tokens": 8},
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hello, how can I help?"
        assert parsed.usage is not None
        assert parsed.usage.get("input_tokens") == 10
        assert parsed.usage.get("output_tokens") == 8

    def test_thinking_and_text_blocks(self, endpoint):
        mock_response = create_mock_response(
            123456789,
            {
                "type": "message",
                "content": [
                    {"type": "thinking", "thinking": "Let me analyze this..."},
                    {"type": "text", "text": "The answer is 42"},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.content == "The answer is 42"
        assert parsed.data.reasoning == "Let me analyze this..."

    def test_usage_mapping(self, endpoint):
        mock_response = create_mock_response(
            123456789,
            {
                "type": "message",
                "content": [{"type": "text", "text": "Hi"}],
                "usage": {
                    "input_tokens": 25,
                    "output_tokens": 10,
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 3,
                },
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed.usage.get("input_tokens") == 25
        assert parsed.usage.get("output_tokens") == 10

    def test_empty_content(self, endpoint):
        mock_response = create_mock_response(
            123456789,
            {
                "type": "message",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        )

        parsed = endpoint.parse_response(mock_response)

        # Should still return for usage
        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage is not None

    def test_null_json_returns_none(self, endpoint):
        mock_response = create_mock_response(123456789, None)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None


def _make_sse_response(event_type: str, json_data: dict, perf_ns: int = 123456789):
    """Helper to create an SSEMessage with event type and data."""
    return SSEMessage(
        perf_ns=perf_ns,
        packets=[
            SSEField(name=SSEFieldType.EVENT, value=event_type),
            SSEField(
                name=SSEFieldType.DATA,
                value=__import__("orjson").dumps(json_data).decode(),
            ),
        ],
    )


class TestAnthropicMessagesParseResponseStreaming:
    """Tests for AnthropicMessagesEndpoint parse_response (streaming SSE)."""

    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.ANTHROPIC_MESSAGES)
        return create_endpoint_with_mock_transport(
            AnthropicMessagesEndpoint, model_endpoint
        )

    def test_message_start_returns_usage(self, endpoint):
        response = _make_sse_response(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 25, "output_tokens": 0},
                },
            },
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage is not None
        assert parsed.usage.get("input_tokens") == 25

    def test_text_delta(self, endpoint):
        response = _make_sse_response(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hello"

    def test_thinking_delta(self, endpoint):
        response = _make_sse_response(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Analyzing..."},
            },
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.reasoning == "Analyzing..."

    def test_signature_delta_returns_none(self, endpoint):
        response = _make_sse_response(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "abc123"},
            },
        )

        parsed = endpoint.parse_response(response)

        assert parsed is None

    def test_message_delta_returns_usage(self, endpoint):
        response = _make_sse_response(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 42},
            },
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage is not None
        assert parsed.usage.get("output_tokens") == 42

    @pytest.mark.parametrize(
        "event_type",
        ["ping", "content_block_start", "content_block_stop", "message_stop"],
    )
    def test_non_content_events_return_none(self, endpoint, event_type):
        response = _make_sse_response(event_type, {"type": event_type})

        parsed = endpoint.parse_response(response)

        assert parsed is None

    def test_streaming_sequence(self, endpoint):
        """Test parsing a full streaming sequence returns correct data types."""
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                },
            ),
            ("ping", {"type": "ping"}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " world"},
                },
            ),
            (
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 5},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]

        results = []
        for event_type, data in events:
            response = _make_sse_response(event_type, data, perf_ns=123456789)
            parsed = endpoint.parse_response(response)
            if parsed:
                results.append(parsed)

        # message_start (usage) + 2 text_deltas + message_delta (usage)
        assert len(results) == 4
        assert results[0].usage is not None  # message_start
        assert isinstance(results[1].data, TextResponseData)
        assert results[1].data.text == "Hello"
        assert isinstance(results[2].data, TextResponseData)
        assert results[2].data.text == " world"
        assert results[3].usage is not None  # message_delta


class TestAnthropicMessagesRawMessages:
    """Tests for raw_messages verbatim replay in AnthropicMessagesEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(EndpointType.ANTHROPIC_MESSAGES)

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            AnthropicMessagesEndpoint, model_endpoint
        )

    def test_raw_messages_replaces_entire_message(self, endpoint, model_endpoint):
        """Turn with raw_messages produces those exact dicts in the messages list."""
        raw = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "tu-1",
                    "name": "read_file",
                    "input": {"path": "a.py"},
                },
            ],
        }
        turn = Turn(raw_messages=[raw], model="claude-sonnet-4-20250514")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["messages"] == [raw]

    def test_raw_messages_mixed_with_normal_turns(self, endpoint, model_endpoint):
        """raw_messages turns can be mixed with normal turns."""
        raw = {
            "role": "assistant",
            "content": [{"type": "text", "text": "I'll help."}],
        }
        turns = [
            Turn(
                texts=[Text(contents=["Hello"])],
                role="user",
                model="claude-sonnet-4-20250514",
            ),
            Turn(raw_messages=[raw], model="claude-sonnet-4-20250514"),
            Turn(
                texts=[Text(contents=["Thanks"])],
                role="user",
                model="claude-sonnet-4-20250514",
            ),
        ]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)

        payload = endpoint.format_payload(request_info)

        assert len(payload["messages"]) == 3
        assert payload["messages"][0]["content"] == "Hello"
        assert payload["messages"][1] == raw
        assert payload["messages"][2]["content"] == "Thanks"

    def test_tools_included_in_payload(self, endpoint, model_endpoint):
        """Tool definitions from request_info are included in the payload."""
        tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]
        turn = Turn(
            texts=[Text(contents=["Read a.py"])], model="claude-sonnet-4-20250514"
        )
        request_info = create_request_info(
            model_endpoint=model_endpoint, turns=[turn], tools=tools
        )

        payload = endpoint.format_payload(request_info)

        assert payload["tools"] == tools

    def test_tools_omitted_when_none(self, endpoint, model_endpoint):
        """No tools key in payload when tools is None."""
        turn = Turn(texts=[Text(contents=["Hello"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "tools" not in payload

    def test_raw_messages_empty_list_adds_nothing(self, endpoint, model_endpoint):
        """Turn with raw_messages=[] contributes zero messages."""
        turns = [
            Turn(
                texts=[Text(contents=["Before"])],
                role="user",
                model="claude-sonnet-4-20250514",
            ),
            Turn(raw_messages=[], model="claude-sonnet-4-20250514"),
            Turn(
                texts=[Text(contents=["After"])],
                role="user",
                model="claude-sonnet-4-20250514",
            ),
        ]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)

        payload = endpoint.format_payload(request_info)

        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["content"] == "Before"
        assert payload["messages"][1]["content"] == "After"

    def test_raw_messages_with_system_and_user_context(self, endpoint, model_endpoint):
        """raw_messages are appended after user_context; system goes to top-level."""
        turn = Turn(
            raw_messages=[
                {"role": "user", "content": "verbatim user"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "verbatim reply"}],
                },
            ],
            model="claude-sonnet-4-20250514",
        )
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            system_message="System prompt",
            user_context_message="User context",
        )

        payload = endpoint.format_payload(request_info)

        # System goes to top-level, not in messages
        assert payload["system"] == "System prompt"
        # user_context prepended, then raw_messages extended
        assert len(payload["messages"]) == 3
        assert payload["messages"][0] == {"role": "user", "content": "User context"}
        assert payload["messages"][1] == {
            "role": "user",
            "content": "verbatim user",
        }
        assert payload["messages"][2] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "verbatim reply"}],
        }

    def test_raw_messages_all_turns_raw_multi_turn(self, endpoint, model_endpoint):
        """Full conversation where every turn uses raw_messages."""
        turns = [
            Turn(
                raw_messages=[{"role": "user", "content": "First question"}],
                model="claude-sonnet-4-20250514",
            ),
            Turn(
                raw_messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Answer"},
                            {
                                "type": "tool_use",
                                "id": "tu-1",
                                "name": "read",
                                "input": {},
                            },
                        ],
                    }
                ],
                model="claude-sonnet-4-20250514",
            ),
            Turn(
                raw_messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tu-1",
                                "content": "file data",
                            }
                        ],
                    }
                ],
                model="claude-sonnet-4-20250514",
            ),
        ]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)

        payload = endpoint.format_payload(request_info)

        assert len(payload["messages"]) == 3
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][1]["role"] == "assistant"
        assert payload["messages"][2]["role"] == "user"
        assert payload["messages"][2]["content"][0]["type"] == "tool_result"
