# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ClaudeCodeTraceLoader."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.dataset.loader.claude_code_trace import (
    ClaudeCodeTraceLoader,
    _group_records_into_api_calls,
    _merge_assistant_content,
    _parse_timestamp_ms,
)
from aiperf.dataset.loader.models import (
    ClaudeCodeApiCall,
    ClaudeCodeTrace,
    ClaudeCodeTraceRecord,
)
from aiperf.plugin.enums import CustomDatasetType, DatasetSamplingStrategy

# =========================================================================
# Model tests
# =========================================================================


class TestClaudeCodeTraceRecord:
    def test_create_user_record(self):
        rec = ClaudeCodeTraceRecord.model_validate(
            {
                "type": "user",
                "message": {"content": "Hello"},
                "sessionId": "sess-123",
                "timestamp": "2025-01-01T00:00:00Z",
                "requestId": "req-1",
            }
        )
        assert rec.type == "user"
        assert rec.session_id == "sess-123"
        assert rec.request_id == "req-1"

    def test_create_assistant_record(self):
        rec = ClaudeCodeTraceRecord.model_validate(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Hello!"}],
                    "model": "claude-sonnet-4-20250514",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
                "requestId": "req-1",
            }
        )
        assert rec.type == "assistant"
        assert rec.message["model"] == "claude-sonnet-4-20250514"

    def test_optional_fields_default(self):
        rec = ClaudeCodeTraceRecord.model_validate({"type": "progress"})
        assert rec.message is None
        assert rec.session_id is None
        assert rec.timestamp is None
        assert rec.request_id is None

    def test_populate_by_name(self):
        rec = ClaudeCodeTraceRecord.model_validate(
            {"type": "user", "session_id": "sess-1", "request_id": "req-1"}
        )
        assert rec.session_id == "sess-1"
        assert rec.request_id == "req-1"


class TestClaudeCodeApiCall:
    def test_create_api_call(self):
        call = ClaudeCodeApiCall(
            user_content="Hello",
            assistant_content=[{"type": "text", "text": "Hi!"}],
            model="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
        )
        assert call.user_content == "Hello"
        assert len(call.assistant_content) == 1
        assert call.model == "claude-sonnet-4-20250514"

    def test_api_call_with_tool_use(self):
        call = ClaudeCodeApiCall(
            user_content=[
                {"type": "tool_result", "tool_use_id": "tu-1", "content": "OK"}
            ],
            assistant_content=[
                {"type": "tool_use", "id": "tu-2", "name": "read", "input": {}}
            ],
            input_tokens=200,
            output_tokens=100,
        )
        assert isinstance(call.user_content, list)
        assert call.assistant_content[0]["type"] == "tool_use"

    def test_defaults(self):
        call = ClaudeCodeApiCall(user_content="test", assistant_content=[])
        assert call.input_tokens == 0
        assert call.output_tokens == 0
        assert call.cache_creation_input_tokens == 0
        assert call.cache_read_input_tokens == 0
        assert call.timestamp_ms is None
        assert call.stop_reason is None


class TestClaudeCodeTrace:
    def test_create_trace(self):
        trace = ClaudeCodeTrace(
            id="trace-1",
            session_id="sess-1",
            api_calls=[
                ClaudeCodeApiCall(
                    user_content="Hello",
                    assistant_content=[{"type": "text", "text": "Hi"}],
                )
            ],
        )
        assert trace.type == CustomDatasetType.CLAUDE_CODE_TRACE
        assert trace.id == "trace-1"
        assert len(trace.api_calls) == 1

    def test_empty_api_calls_raises(self):
        with pytest.raises(ValidationError, match="at least one API call"):
            ClaudeCodeTrace(id="trace-1", session_id="sess-1", api_calls=[])


# =========================================================================
# Helper function tests
# =========================================================================


class TestParseTimestampMs:
    def test_iso_timestamp(self):
        ms = _parse_timestamp_ms("2025-01-01T00:00:00Z")
        assert ms is not None
        assert ms > 0

    def test_none_returns_none(self):
        assert _parse_timestamp_ms(None) is None

    def test_invalid_returns_none(self):
        assert _parse_timestamp_ms("not-a-date") is None


class TestMergeAssistantContent:
    def test_merge_single_record(self):
        rec = ClaudeCodeTraceRecord.model_validate(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Hello"}],
                    "model": "claude-sonnet-4-20250514",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                    "stop_reason": "end_turn",
                },
            }
        )
        blocks, usage, model, stop = _merge_assistant_content([rec])
        assert len(blocks) == 1
        assert blocks[0]["text"] == "Hello"
        assert usage["input_tokens"] == 100
        assert model == "claude-sonnet-4-20250514"
        assert stop == "end_turn"

    def test_merge_multiple_records(self):
        rec1 = ClaudeCodeTraceRecord.model_validate(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "thinking", "thinking": "Let me think..."}],
                    "model": "claude-sonnet-4-20250514",
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                },
            }
        )
        rec2 = ClaudeCodeTraceRecord.model_validate(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Answer"}],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                    "stop_reason": "end_turn",
                },
            }
        )
        blocks, usage, model, stop = _merge_assistant_content([rec1, rec2])
        assert len(blocks) == 2
        assert usage["output_tokens"] == 50
        assert stop == "end_turn"


class TestGroupRecordsIntoApiCalls:
    def test_simple_user_assistant_pair(self):
        records = [
            ClaudeCodeTraceRecord.model_validate(
                {"type": "user", "message": {"content": "Hello"}}
            ),
            ClaudeCodeTraceRecord.model_validate(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": "Hi!"}],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                    "requestId": "req-1",
                }
            ),
        ]
        calls = _group_records_into_api_calls(records)
        assert len(calls) == 1
        assert calls[0].user_content == "Hello"
        assert calls[0].assistant_content[0]["text"] == "Hi!"

    def test_multiple_turns(self):
        records = [
            ClaudeCodeTraceRecord.model_validate(
                {"type": "user", "message": {"content": "Turn 1"}}
            ),
            ClaudeCodeTraceRecord.model_validate(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": "Reply 1"}],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                    "requestId": "req-1",
                }
            ),
            ClaudeCodeTraceRecord.model_validate(
                {"type": "user", "message": {"content": "Turn 2"}}
            ),
            ClaudeCodeTraceRecord.model_validate(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": "Reply 2"}],
                        "usage": {"input_tokens": 20, "output_tokens": 10},
                    },
                    "requestId": "req-2",
                }
            ),
        ]
        calls = _group_records_into_api_calls(records)
        assert len(calls) == 2
        assert calls[0].user_content == "Turn 1"
        assert calls[1].user_content == "Turn 2"

    def test_multiple_assistant_records_same_request_id(self):
        records = [
            ClaudeCodeTraceRecord.model_validate(
                {"type": "user", "message": {"content": "Query"}}
            ),
            ClaudeCodeTraceRecord.model_validate(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "thinking", "thinking": "Hmm..."}],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                    "requestId": "req-1",
                }
            ),
            ClaudeCodeTraceRecord.model_validate(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": "Answer"}],
                        "usage": {"input_tokens": 10, "output_tokens": 20},
                        "stop_reason": "end_turn",
                    },
                    "requestId": "req-1",
                }
            ),
        ]
        calls = _group_records_into_api_calls(records)
        assert len(calls) == 1
        assert len(calls[0].assistant_content) == 2
        assert calls[0].assistant_content[0]["type"] == "thinking"
        assert calls[0].assistant_content[1]["type"] == "text"

    def test_tool_result_content(self):
        records = [
            ClaudeCodeTraceRecord.model_validate(
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tu-1",
                                "content": "file contents here",
                            }
                        ]
                    },
                }
            ),
            ClaudeCodeTraceRecord.model_validate(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tu-2",
                                "name": "write",
                                "input": {"path": "test.py"},
                            }
                        ],
                        "usage": {"input_tokens": 500, "output_tokens": 100},
                    },
                    "requestId": "req-1",
                }
            ),
        ]
        calls = _group_records_into_api_calls(records)
        assert len(calls) == 1
        assert isinstance(calls[0].user_content, list)
        assert calls[0].user_content[0]["type"] == "tool_result"

    def test_skips_orphan_assistant_without_user(self):
        records = [
            ClaudeCodeTraceRecord.model_validate(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": "orphan"}],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                    "requestId": "req-1",
                }
            ),
        ]
        calls = _group_records_into_api_calls(records)
        assert len(calls) == 0


# =========================================================================
# Loader tests
# =========================================================================


def _make_jsonl_file(tmp_path, filename, records):
    """Write a list of record dicts as JSONL."""
    filepath = tmp_path / filename
    with open(filepath, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return filepath


def _sample_session_records():
    """Create a minimal 2-turn Claude Code session."""
    return [
        {
            "type": "system",
            "message": {"content": "You are a helpful assistant."},
            "sessionId": "sess-abc",
            "timestamp": "2025-06-01T10:00:00Z",
        },
        {
            "type": "user",
            "message": {"content": "Hello, help me with Python"},
            "sessionId": "sess-abc",
            "timestamp": "2025-06-01T10:00:01Z",
            "requestId": "req-1",
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Sure! What do you need help with?"}
                ],
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 100, "output_tokens": 20},
                "stop_reason": "end_turn",
            },
            "sessionId": "sess-abc",
            "timestamp": "2025-06-01T10:00:02Z",
            "requestId": "req-1",
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-1",
                        "content": "def hello():\n    print('hello')",
                    }
                ]
            },
            "sessionId": "sess-abc",
            "timestamp": "2025-06-01T10:00:10Z",
            "requestId": "req-2",
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu-2",
                        "name": "edit",
                        "input": {"path": "hello.py"},
                    }
                ],
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 200, "output_tokens": 50},
                "stop_reason": "tool_use",
            },
            "sessionId": "sess-abc",
            "timestamp": "2025-06-01T10:00:12Z",
            "requestId": "req-2",
        },
    ]


class TestClaudeCodeTraceLoader:
    @pytest.fixture
    def default_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    @pytest.fixture
    def session_file(self, tmp_path):
        return _make_jsonl_file(tmp_path, "session.jsonl", _sample_session_records())

    @pytest.fixture
    def session_dir(self, tmp_path):
        _make_jsonl_file(tmp_path, "session1.jsonl", _sample_session_records())
        _make_jsonl_file(tmp_path, "session2.jsonl", _sample_session_records())
        return tmp_path

    # --- can_load tests ---

    def test_can_load_jsonl_file(self, session_file):
        assert ClaudeCodeTraceLoader.can_load(filename=str(session_file))

    def test_can_load_directory(self, session_dir):
        assert ClaudeCodeTraceLoader.can_load(filename=str(session_dir))

    def test_can_load_data_dict(self):
        data = {"type": "user", "message": {"content": "Hello"}}
        assert ClaudeCodeTraceLoader.can_load(data=data)

    def test_can_load_rejects_non_trace(self):
        assert not ClaudeCodeTraceLoader.can_load(data={"text": "hello"})
        assert not ClaudeCodeTraceLoader.can_load(data=None, filename=None)

    def test_can_load_rejects_json_file(self, tmp_path):
        json_file = tmp_path / "test.json"
        json_file.write_text("{}")
        assert not ClaudeCodeTraceLoader.can_load(filename=str(json_file))

    # --- load_dataset tests ---

    def test_load_single_file(self, session_file, default_user_config):
        loader = ClaudeCodeTraceLoader(
            filename=str(session_file), user_config=default_user_config
        )
        data = loader.load_dataset()
        assert len(data) == 1
        trace = list(data.values())[0][0]
        assert len(trace.api_calls) == 2
        assert trace.system_prompt == "You are a helpful assistant."
        assert trace.session_id == "sess-abc"

    def test_load_directory(self, session_dir, default_user_config):
        loader = ClaudeCodeTraceLoader(
            filename=str(session_dir), user_config=default_user_config
        )
        data = loader.load_dataset()
        assert len(data) == 2

    def test_load_empty_file(self, tmp_path, default_user_config):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        loader = ClaudeCodeTraceLoader(
            filename=str(empty), user_config=default_user_config
        )
        data = loader.load_dataset()
        assert len(data) == 0

    def test_load_skips_malformed_lines(self, tmp_path, default_user_config):
        filepath = tmp_path / "mixed.jsonl"
        with open(filepath, "w") as f:
            f.write("not valid json\n")
            f.write(json.dumps(_sample_session_records()[1]) + "\n")
            f.write(json.dumps(_sample_session_records()[2]) + "\n")
        loader = ClaudeCodeTraceLoader(
            filename=str(filepath), user_config=default_user_config
        )
        data = loader.load_dataset()
        assert len(data) == 1

    # --- convert_to_conversations tests ---

    def test_convert_verbatim_mode(self, session_file, default_user_config):
        loader = ClaudeCodeTraceLoader(
            filename=str(session_file), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conv = conversations[0]
        assert len(conv.turns) == 2
        assert conv.system_message == "You are a helpful assistant."

        # First turn: verbatim user content
        t0 = conv.turns[0]
        assert t0.role == "user"
        assert t0.raw_content == "Hello, help me with Python"
        assert t0.assistant_prefill is not None
        assert t0.assistant_prefill[0]["type"] == "text"
        assert t0.input_tokens == 100

        # Second turn: tool_result content
        t1 = conv.turns[1]
        assert t1.role == "user"
        assert isinstance(t1.raw_content, list)
        assert t1.raw_content[0]["type"] == "tool_result"
        assert t1.assistant_prefill[0]["type"] == "tool_use"
        assert t1.input_tokens == 200

    def test_convert_synthetic_mode(self, session_file, default_user_config):
        mock_gen = MagicMock()
        mock_gen.generate_prompt.return_value = "x" * 100

        loader = ClaudeCodeTraceLoader(
            filename=str(session_file),
            user_config=default_user_config,
            prompt_generator=mock_gen,
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conv = conversations[0]
        t0 = conv.turns[0]
        assert t0.raw_content is None
        assert t0.assistant_prefill is None
        assert len(t0.texts) == 1
        assert t0.texts[0].contents[0] == "x" * 100

    def test_convert_preserves_delays(self, session_file, default_user_config):
        loader = ClaudeCodeTraceLoader(
            filename=str(session_file), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        conv = conversations[0]
        assert conv.turns[0].delay is None
        assert conv.turns[1].delay is not None
        assert conv.turns[1].delay > 0

    def test_convert_preserves_max_tokens(self, session_file, default_user_config):
        loader = ClaudeCodeTraceLoader(
            filename=str(session_file), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        conv = conversations[0]
        assert conv.turns[0].max_tokens == 20
        assert conv.turns[1].max_tokens == 50

    def test_convert_preserves_model(self, session_file, default_user_config):
        loader = ClaudeCodeTraceLoader(
            filename=str(session_file), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        conv = conversations[0]
        assert conv.turns[0].model == "claude-sonnet-4-20250514"

    # --- Edge cases ---

    def test_single_turn_trace(self, tmp_path, default_user_config):
        records = [
            {
                "type": "user",
                "message": {"content": "One shot question"},
                "timestamp": "2025-06-01T10:00:00Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Answer"}],
                    "usage": {"input_tokens": 50, "output_tokens": 10},
                },
                "requestId": "req-1",
            },
        ]
        filepath = _make_jsonl_file(tmp_path, "single.jsonl", records)
        loader = ClaudeCodeTraceLoader(
            filename=str(filepath), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        assert len(conversations[0].turns) == 1

    def test_thinking_blocks_preserved(self, tmp_path, default_user_config):
        records = [
            {
                "type": "user",
                "message": {"content": "Think about this"},
                "timestamp": "2025-06-01T10:00:00Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "Deep thoughts..."},
                        {"type": "text", "text": "The answer"},
                    ],
                    "usage": {"input_tokens": 50, "output_tokens": 30},
                },
                "requestId": "req-1",
            },
        ]
        filepath = _make_jsonl_file(tmp_path, "thinking.jsonl", records)
        loader = ClaudeCodeTraceLoader(
            filename=str(filepath), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        prefill = conversations[0].turns[0].assistant_prefill
        assert len(prefill) == 2
        assert prefill[0]["type"] == "thinking"
        assert prefill[1]["type"] == "text"

    def test_progress_records_filtered(self, tmp_path, default_user_config):
        records = [
            {"type": "progress", "message": {"status": "loading"}},
            {"type": "file-history-snapshot", "message": {"files": []}},
            {
                "type": "user",
                "message": {"content": "Hello"},
                "timestamp": "2025-06-01T10:00:00Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Hi"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
                "requestId": "req-1",
            },
        ]
        filepath = _make_jsonl_file(tmp_path, "filtered.jsonl", records)
        loader = ClaudeCodeTraceLoader(
            filename=str(filepath), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        assert len(conversations[0].turns) == 1

    def test_preferred_sampling_strategy(self):
        assert (
            ClaudeCodeTraceLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    def test_system_prompt_extracted(self, session_file, default_user_config):
        loader = ClaudeCodeTraceLoader(
            filename=str(session_file), user_config=default_user_config
        )
        data = loader.load_dataset()
        trace = list(data.values())[0][0]
        assert trace.system_prompt == "You are a helpful assistant."

    def test_session_id_from_record(self, session_file, default_user_config):
        loader = ClaudeCodeTraceLoader(
            filename=str(session_file), user_config=default_user_config
        )
        data = loader.load_dataset()
        trace = list(data.values())[0][0]
        assert trace.session_id == "sess-abc"

    def test_session_id_fallback_to_filename(self, tmp_path, default_user_config):
        records = [
            {
                "type": "user",
                "message": {"content": "Hello"},
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Hi"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
                "requestId": "req-1",
            },
        ]
        filepath = _make_jsonl_file(tmp_path, "my_session.jsonl", records)
        loader = ClaudeCodeTraceLoader(
            filename=str(filepath), user_config=default_user_config
        )
        data = loader.load_dataset()
        trace = list(data.values())[0][0]
        assert trace.session_id == "my_session"
