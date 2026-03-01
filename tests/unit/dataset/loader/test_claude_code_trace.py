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

    def test_subagent_manifest_loading(self, tmp_path, default_user_config):
        """Test loading a directory with manifest linking parent and child sessions."""
        # Parent session: 3 API calls, spawns subagent after call 1
        parent_records = [
            {
                "type": "user",
                "message": {"content": "Build me a web app"},
                "sessionId": "parent-sess",
                "timestamp": "2025-06-01T10:00:00Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Starting..."}],
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "stop_reason": "end_turn",
                },
                "requestId": "req-p1",
            },
            {
                "type": "user",
                "message": {"content": "Use React for the frontend"},
                "sessionId": "parent-sess",
                "timestamp": "2025-06-01T10:00:05Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu-task",
                            "name": "Task",
                            "input": {"prompt": "Set up React"},
                        }
                    ],
                    "usage": {"input_tokens": 200, "output_tokens": 50},
                    "stop_reason": "tool_use",
                },
                "requestId": "req-p2",
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu-task",
                            "content": "React setup complete",
                        }
                    ]
                },
                "sessionId": "parent-sess",
                "timestamp": "2025-06-01T10:00:20Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "All done!"}],
                    "usage": {"input_tokens": 300, "output_tokens": 30},
                    "stop_reason": "end_turn",
                },
                "requestId": "req-p3",
            },
        ]

        # Child session
        child_records = [
            {
                "type": "user",
                "message": {"content": "Set up React project"},
                "sessionId": "child-sess",
                "timestamp": "2025-06-01T10:00:06Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Installing deps..."}],
                    "usage": {"input_tokens": 80, "output_tokens": 15},
                    "stop_reason": "end_turn",
                },
                "requestId": "req-c1",
            },
        ]

        # Write files
        _make_jsonl_file(tmp_path, "parent.jsonl", parent_records)
        _make_jsonl_file(tmp_path, "child_react.jsonl", child_records)

        # Write manifest
        manifest = {
            "parent": "parent.jsonl",
            "subagents": [{"file": "child_react.jsonl", "spawn_after_api_call": 1}],
        }
        (tmp_path / "_manifest.json").write_text(json.dumps(manifest))

        loader = ClaudeCodeTraceLoader(
            filename=str(tmp_path), user_config=default_user_config
        )
        data = loader.load_dataset()
        assert len(data) == 2

        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 2

        # Find parent and child
        parent = next(c for c in conversations if c.agent_depth == 0)
        child = next(c for c in conversations if c.agent_depth > 0)

        assert len(parent.turns) == 3
        assert len(child.turns) == 1
        assert len(parent.subagent_spawns) == 1
        assert parent.subagent_spawns[0].spawn_id == "s0"
        assert parent.subagent_spawns[0].child_conversation_ids == [child.session_id]
        assert parent.subagent_spawns[0].join_turn_index == 2

        # The join turn (not spawn turn) gets subagent_spawn_ids
        assert parent.turns[1].subagent_spawn_ids == []
        assert parent.turns[2].subagent_spawn_ids == ["s0"]

    def test_spawn_id_on_join_turn_not_spawn_turn(self, tmp_path, default_user_config):
        """Verify subagent_spawn_ids is placed on the join turn, not the spawning turn.

        adaptive_scale checks next_meta.subagent_spawn_ids to decide when to
        dispatch children. The spawn_id must be on the join turn so that:
        1. The spawn turn (turn N) is sent normally
        2. After turn N completes, adaptive_scale sees spawn_id on turn N+1
        3. Children are dispatched instead of sending turn N+1 immediately
        4. When children complete, turn N+1 (the join turn) is sent
        """
        # Parent: 4 API calls. Spawn after call 0, so join_turn_index = 1
        parent_records = [
            {
                "type": "user",
                "message": {"content": "Start"},
                "sessionId": "p",
                "timestamp": "2025-01-01T00:00:00Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "OK"}],
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "stop_reason": "tool_use",
                },
                "requestId": "r1",
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "done"}
                    ]
                },
                "sessionId": "p",
                "timestamp": "2025-01-01T00:00:05Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Next step"}],
                    "usage": {"input_tokens": 200, "output_tokens": 40},
                    "stop_reason": "tool_use",
                },
                "requestId": "r2",
            },
            {
                "type": "user",
                "message": {"content": "Continue"},
                "sessionId": "p",
                "timestamp": "2025-01-01T00:00:10Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "More work"}],
                    "usage": {"input_tokens": 300, "output_tokens": 60},
                    "stop_reason": "tool_use",
                },
                "requestId": "r3",
            },
            {
                "type": "user",
                "message": {"content": "Finish"},
                "sessionId": "p",
                "timestamp": "2025-01-01T00:00:15Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Done"}],
                    "usage": {"input_tokens": 400, "output_tokens": 30},
                    "stop_reason": "end_turn",
                },
                "requestId": "r4",
            },
        ]
        child_records = [
            {
                "type": "user",
                "message": {"content": "child task"},
                "sessionId": "c",
                "timestamp": "2025-01-01T00:00:01Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "child done"}],
                    "usage": {"input_tokens": 50, "output_tokens": 10},
                    "stop_reason": "end_turn",
                },
                "requestId": "rc1",
            },
        ]

        _make_jsonl_file(tmp_path, "parent.jsonl", parent_records)
        _make_jsonl_file(tmp_path, "child.jsonl", child_records)
        manifest = {
            "parent": "parent.jsonl",
            "subagents": [{"file": "child.jsonl", "spawn_after_api_call": 0}],
        }
        (tmp_path / "_manifest.json").write_text(json.dumps(manifest))

        loader = ClaudeCodeTraceLoader(
            filename=str(tmp_path), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = next(c for c in conversations if c.agent_depth == 0)

        assert len(parent.turns) == 4
        # Spawn after call 0 -> join at turn 1
        assert parent.subagent_spawns[0].join_turn_index == 1
        # Turn 0 (spawn turn) must NOT have spawn_ids
        assert parent.turns[0].subagent_spawn_ids == []
        # Turn 1 (join turn) MUST have spawn_ids
        assert parent.turns[1].subagent_spawn_ids == ["s0"]
        # Remaining turns are normal
        assert parent.turns[2].subagent_spawn_ids == []
        assert parent.turns[3].subagent_spawn_ids == []

    def test_spawn_id_placement_with_multiple_subagents(
        self, tmp_path, default_user_config
    ):
        """Verify spawn_id placement when multiple subagents spawn at different points."""
        parent_records = []
        # 5 API calls (10 records: 5 user + 5 assistant)
        for i in range(5):
            parent_records.append(
                {
                    "type": "user",
                    "message": {"content": f"Turn {i}"},
                    "sessionId": "p",
                    "timestamp": f"2025-01-01T00:00:{i * 5:02d}Z",
                }
            )
            parent_records.append(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": f"Response {i}"}],
                        "usage": {"input_tokens": 100 * (i + 1), "output_tokens": 20},
                        "stop_reason": "end_turn",
                    },
                    "requestId": f"r{i}",
                }
            )

        def _child(sid):
            return [
                {
                    "type": "user",
                    "message": {"content": "task"},
                    "sessionId": sid,
                    "timestamp": "2025-01-01T00:00:01Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": "done"}],
                        "usage": {"input_tokens": 50, "output_tokens": 10},
                        "stop_reason": "end_turn",
                    },
                    "requestId": "rc1",
                },
            ]

        _make_jsonl_file(tmp_path, "parent.jsonl", parent_records)
        _make_jsonl_file(tmp_path, "child_a.jsonl", _child("ca"))
        _make_jsonl_file(tmp_path, "child_b.jsonl", _child("cb"))

        manifest = {
            "parent": "parent.jsonl",
            "subagents": [
                {"file": "child_a.jsonl", "spawn_after_api_call": 1},
                {"file": "child_b.jsonl", "spawn_after_api_call": 3},
            ],
        }
        (tmp_path / "_manifest.json").write_text(json.dumps(manifest))

        loader = ClaudeCodeTraceLoader(
            filename=str(tmp_path), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = next(c for c in conversations if c.agent_depth == 0)

        assert len(parent.turns) == 5
        assert len(parent.subagent_spawns) == 2

        # First spawn: after call 1 -> join at turn 2
        assert parent.subagent_spawns[0].join_turn_index == 2
        # Second spawn: after call 3 -> join at turn 4
        assert parent.subagent_spawns[1].join_turn_index == 4

        # Only join turns have spawn_ids, all others are empty
        expected_spawn_ids = [[], [], ["s0"], [], ["s1"]]
        for i, expected in enumerate(expected_spawn_ids):
            assert parent.turns[i].subagent_spawn_ids == expected, (
                f"Turn {i}: expected spawn_ids={expected!r}, "
                f"got {parent.turns[i].subagent_spawn_ids!r}"
            )

    def test_spawn_at_last_turn_join_clamps_to_last(
        self, tmp_path, default_user_config
    ):
        """When spawn_after_api_call is the last turn, join_turn_index clamps."""
        parent_records = [
            {
                "type": "user",
                "message": {"content": "Only turn"},
                "sessionId": "p",
                "timestamp": "2025-01-01T00:00:00Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Done"}],
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "stop_reason": "end_turn",
                },
                "requestId": "r1",
            },
            {
                "type": "user",
                "message": {"content": "Last turn"},
                "sessionId": "p",
                "timestamp": "2025-01-01T00:00:05Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Final"}],
                    "usage": {"input_tokens": 200, "output_tokens": 30},
                    "stop_reason": "end_turn",
                },
                "requestId": "r2",
            },
        ]
        child_records = [
            {
                "type": "user",
                "message": {"content": "task"},
                "sessionId": "c",
                "timestamp": "2025-01-01T00:00:01Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "done"}],
                    "usage": {"input_tokens": 50, "output_tokens": 10},
                    "stop_reason": "end_turn",
                },
                "requestId": "rc1",
            },
        ]

        _make_jsonl_file(tmp_path, "parent.jsonl", parent_records)
        _make_jsonl_file(tmp_path, "child.jsonl", child_records)
        manifest = {
            "parent": "parent.jsonl",
            # Spawn after last call (index 1) -> join clamps to min(2, 2-1) = 1
            "subagents": [{"file": "child.jsonl", "spawn_after_api_call": 1}],
        }
        (tmp_path / "_manifest.json").write_text(json.dumps(manifest))

        loader = ClaudeCodeTraceLoader(
            filename=str(tmp_path), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = next(c for c in conversations if c.agent_depth == 0)

        assert len(parent.turns) == 2
        # join_turn_index clamps to last turn
        assert parent.subagent_spawns[0].join_turn_index == 1
        # spawn_ids goes on the clamped join turn (turn 1), not turn 0
        assert parent.turns[0].subagent_spawn_ids == []
        assert parent.turns[1].subagent_spawn_ids == ["s0"]

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
