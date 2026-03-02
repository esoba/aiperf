# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ApiCaptureTraceLoader."""

import json

import pytest
from pydantic import ValidationError

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.dataset.loader.api_capture_trace import (
    ApiCaptureTraceLoader,
    _extract_system_text,
    _is_prefetch,
    _thread_key,
)
from aiperf.dataset.loader.models import ApiCaptureApiCall, ApiCaptureTrace
from aiperf.plugin.enums import CustomDatasetType, DatasetSamplingStrategy

# =========================================================================
# Test data builders
# =========================================================================

PARENT_SYSTEM = [
    {"type": "text", "text": "You are an expert coding assistant."},
    {"type": "text", "text": "Follow these rules carefully."},
    {"type": "text", "text": "Always write tests."},
]

SUBAGENT_SYSTEM_A = [
    {"type": "text", "text": "You are a teammate subagent."},
    {"type": "text", "text": "Help the parent with research."},
]

SUBAGENT_SYSTEM_B = [
    {"type": "text", "text": "Quick classification task."},
]

PARENT_TOOLS = [{"name": f"tool_{i}", "description": f"Tool {i}"} for i in range(35)]
SUBAGENT_TOOLS_A = [
    {"name": f"tool_{i}", "description": f"Tool {i}"} for i in range(25)
]


def _make_req_file(directory, call_index, *, system, tools, messages, model, **extra):
    """Write a req_XXXX.json file."""
    body = {
        "model": model,
        "messages": messages,
        "system": system,
        "tools": tools,
        "max_tokens": extra.get("max_tokens", 32000),
        "stream": extra.get("stream", True),
    }
    if "thinking" in extra:
        body["thinking"] = extra["thinking"]
    path = directory / f"req_{call_index:04d}.json"
    path.write_text(json.dumps(body))
    return path


def _make_capture_entry(
    call_index, direction, timestamp, *, model="claude-opus-4-6", **extra
):
    """Build a capture.jsonl entry."""
    entry = {
        "call_index": call_index,
        "direction": direction,
        "timestamp": timestamp,
        "model": model,
    }
    if direction == "request":
        entry.update(
            {
                "max_tokens": extra.get("max_tokens"),
                "stream": extra.get("stream"),
                "tool_count": extra.get("tool_count", 0),
            }
        )
    elif direction == "response":
        entry.update(
            {
                "stop_reason": extra.get("stop_reason", "tool_use"),
                "usage": extra.get(
                    "usage",
                    {
                        "input_tokens": extra.get("input_tokens", 1000),
                        "output_tokens": extra.get("output_tokens", 200),
                        "cache_creation_input_tokens": extra.get(
                            "cache_creation_input_tokens", 50
                        ),
                        "cache_read_input_tokens": extra.get(
                            "cache_read_input_tokens", 500
                        ),
                    },
                ),
            }
        )
    return entry


def _build_messages(turn_count, role_prefix="user"):
    """Build a growing messages array like real captures: user, assistant, user, ..."""
    messages = []
    for i in range(turn_count):
        if i % 2 == 0:
            messages.append({"role": "user", "content": f"{role_prefix} turn {i // 2}"})
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"response {i // 2}"}],
                }
            )
    return messages


def _build_team_capture(tmp_path):
    """Build a realistic team capture directory with 3 threads.

    Parent: 17 real requests (Opus, 35 tools)
    Subagent A: 4 requests (Opus, 25 tools)
    Subagent B: 4 requests (Haiku, 0 tools, single-shot)

    Plus 16 prefetch requests at the start.
    """
    capture_dir = tmp_path / "team_capture"
    capture_dir.mkdir()

    capture_entries = []
    base_ts = 1771454900.0

    # --- Session 0 (old, will be ignored): 7 requests ---
    for i in range(7):
        capture_entries.append(
            _make_capture_entry(i, "request", base_ts + i, stream=None, max_tokens=None)
        )
        capture_entries.append(_make_capture_entry(i, "response", base_ts + i + 0.5))

    # --- Session 1 (also old): 3 requests ---
    for i in range(3):
        capture_entries.append(
            _make_capture_entry(
                i, "request", base_ts + 100 + i, stream=None, max_tokens=None
            )
        )
        capture_entries.append(
            _make_capture_entry(i, "response", base_ts + 100 + i + 0.5)
        )

    # --- Session 2 (last session): 41 requests = 16 prefetch + 25 real ---
    session_start_ts = base_ts + 200

    # 16 prefetch requests (req_0000 through req_0015)
    # All have max_tokens=1 so _is_prefetch filters them out
    for i in range(16):
        ts = session_start_ts + i
        capture_entries.append(
            _make_capture_entry(
                i,
                "request",
                ts,
                model="claude-haiku-4-5-20251001" if i == 0 else "claude-opus-4-6",
                stream=None,
                max_tokens=1,
            )
        )
        capture_entries.append(
            _make_capture_entry(
                i,
                "response",
                ts + 0.1,
                input_tokens=10,
                output_tokens=5,
            )
        )
        _make_req_file(
            capture_dir,
            i,
            system=[],
            tools=[],
            messages=[{"role": "user", "content": "quota"}],
            model="claude-haiku-4-5-20251001" if i == 0 else "claude-opus-4-6",
            max_tokens=1,
            stream=None,
        )

    # 25 real requests (req_0016 through req_0040)
    # Parent: 17 requests at indices 16-32
    # Subagent A: 4 requests at indices 33-36
    # Subagent B: 4 requests at indices 37-40
    real_idx = 16
    parent_indices = []
    for turn in range(17):
        ci = real_idx
        parent_indices.append(ci)
        ts = session_start_ts + 20 + turn * 5
        msg_count = 1 + turn * 2
        messages = _build_messages(msg_count)

        capture_entries.append(
            _make_capture_entry(
                ci, "request", ts, stream=True, max_tokens=32000, tool_count=35
            )
        )
        capture_entries.append(
            _make_capture_entry(
                ci,
                "response",
                ts + 2,
                input_tokens=1000 + turn * 500,
                output_tokens=200 + turn * 50,
                cache_creation_input_tokens=100,
                cache_read_input_tokens=800 + turn * 400,
            )
        )
        _make_req_file(
            capture_dir,
            ci,
            system=PARENT_SYSTEM,
            tools=PARENT_TOOLS,
            messages=messages,
            model="claude-opus-4-6",
            thinking={"type": "adaptive"},
        )
        real_idx += 1

    # Subagent A: 4 requests (Opus, 25 tools, growing messages)
    sub_a_first_ts = session_start_ts + 20 + 5 * 5  # spawned around parent turn 5
    for turn in range(4):
        ci = real_idx
        ts = sub_a_first_ts + turn * 3
        messages = _build_messages(1 + turn * 2, role_prefix="sub_a")

        capture_entries.append(
            _make_capture_entry(
                ci, "request", ts, stream=True, max_tokens=32000, tool_count=25
            )
        )
        capture_entries.append(
            _make_capture_entry(
                ci,
                "response",
                ts + 1.5,
                input_tokens=800 + turn * 300,
                output_tokens=150 + turn * 30,
            )
        )
        _make_req_file(
            capture_dir,
            ci,
            system=SUBAGENT_SYSTEM_A,
            tools=SUBAGENT_TOOLS_A,
            messages=messages,
            model="claude-opus-4-6",
        )
        real_idx += 1

    # Subagent B: 4 requests (Haiku, 0 tools, single-shot, each has 1 message)
    sub_b_first_ts = session_start_ts + 20 + 8 * 5  # spawned around parent turn 8
    for turn in range(4):
        ci = real_idx
        ts = sub_b_first_ts + turn * 2

        capture_entries.append(
            _make_capture_entry(
                ci,
                "request",
                ts,
                model="claude-haiku-4-5-20251001",
                stream=True,
                max_tokens=32000,
                tool_count=0,
            )
        )
        capture_entries.append(
            _make_capture_entry(
                ci,
                "response",
                ts + 0.5,
                model="claude-haiku-4-5-20251001",
                input_tokens=200,
                output_tokens=50,
            )
        )
        _make_req_file(
            capture_dir,
            ci,
            system=SUBAGENT_SYSTEM_B,
            tools=[],
            messages=[{"role": "user", "content": f"classify item {turn}"}],
            model="claude-haiku-4-5-20251001",
        )
        real_idx += 1

    # Write capture.jsonl
    capture_jsonl = capture_dir / "capture.jsonl"
    with open(capture_jsonl, "w") as f:
        for entry in capture_entries:
            f.write(json.dumps(entry) + "\n")

    return capture_dir


# =========================================================================
# Model tests
# =========================================================================


class TestApiCaptureApiCall:
    def test_create(self):
        call = ApiCaptureApiCall(
            messages=[{"role": "user", "content": "Hello"}],
            system=[{"type": "text", "text": "Be helpful"}],
            tools=[],
            model="claude-opus-4-6",
            max_tokens=32000,
            stream=True,
            input_tokens=1000,
            output_tokens=200,
        )
        assert call.model == "claude-opus-4-6"
        assert call.input_tokens == 1000
        assert len(call.messages) == 1

    def test_defaults(self):
        call = ApiCaptureApiCall(messages=[])
        assert call.system == []
        assert call.tools == []
        assert call.model is None
        assert call.input_tokens == 0
        assert call.output_tokens == 0
        assert call.cache_creation_input_tokens == 0
        assert call.cache_read_input_tokens == 0
        assert call.timestamp_ms is None
        assert call.stop_reason is None
        assert call.thinking is None


class TestApiCaptureTrace:
    def test_create(self):
        trace = ApiCaptureTrace(
            id="trace-1",
            api_calls=[ApiCaptureApiCall(messages=[{"role": "user", "content": "Hi"}])],
            thread_key="abc123",
        )
        assert trace.type == CustomDatasetType.API_CAPTURE_TRACE
        assert trace.id == "trace-1"
        assert len(trace.api_calls) == 1

    def test_empty_api_calls_raises(self):
        with pytest.raises(ValidationError, match="at least one API call"):
            ApiCaptureTrace(id="trace-1", api_calls=[], thread_key="abc")


# =========================================================================
# Helper function tests
# =========================================================================


class TestHelperFunctions:
    def test_extract_system_text(self):
        blocks = [
            {"type": "text", "text": "First block"},
            {"type": "text", "text": "Second block"},
        ]
        assert _extract_system_text(blocks) == "First block\nSecond block"

    def test_extract_system_text_empty(self):
        assert _extract_system_text([]) is None

    def test_thread_key_deterministic(self):
        blocks = [{"type": "text", "text": "Hello"}]
        key1 = _thread_key(blocks, 0)
        key2 = _thread_key(blocks, 0)
        assert key1 == key2
        assert len(key1) == 12

    def test_thread_key_different_for_different_system(self):
        key1 = _thread_key([{"type": "text", "text": "System A"}], 0)
        key2 = _thread_key([{"type": "text", "text": "System B"}], 0)
        assert key1 != key2

    def test_is_prefetch_max_tokens_1(self):
        assert _is_prefetch({"max_tokens": 1})

    def test_is_prefetch_max_tokens_0(self):
        assert _is_prefetch({"max_tokens": 0})

    def test_not_prefetch_no_max_tokens(self):
        assert not _is_prefetch({"max_tokens": None})

    def test_not_prefetch_normal_max_tokens(self):
        assert not _is_prefetch({"max_tokens": 32000})

    def test_not_prefetch_missing_max_tokens(self):
        assert not _is_prefetch({})


# =========================================================================
# can_load tests
# =========================================================================


class TestCanLoad:
    def test_valid_directory(self, tmp_path):
        (tmp_path / "capture.jsonl").write_text("{}\n")
        (tmp_path / "req_0000.json").write_text("{}")
        assert ApiCaptureTraceLoader.can_load(filename=str(tmp_path))

    def test_missing_capture_jsonl(self, tmp_path):
        (tmp_path / "req_0000.json").write_text("{}")
        assert not ApiCaptureTraceLoader.can_load(filename=str(tmp_path))

    def test_missing_req_files(self, tmp_path):
        (tmp_path / "capture.jsonl").write_text("{}\n")
        assert not ApiCaptureTraceLoader.can_load(filename=str(tmp_path))

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        assert not ApiCaptureTraceLoader.can_load(filename=str(f))

    def test_none_filename(self):
        assert not ApiCaptureTraceLoader.can_load(filename=None)

    def test_preferred_sampling_strategy(self):
        assert (
            ApiCaptureTraceLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )


# =========================================================================
# Loader integration tests
# =========================================================================


class TestApiCaptureTraceLoader:
    @pytest.fixture
    def default_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    @pytest.fixture
    def team_capture(self, tmp_path):
        return _build_team_capture(tmp_path)

    def test_load_dataset_thread_count(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        assert len(data) == 3

    def test_load_dataset_parent_has_17_calls(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        traces = [t[0] for t in data.values()]
        parent = max(traces, key=lambda t: len(t.api_calls))
        assert len(parent.api_calls) == 17

    def test_load_dataset_subagent_a_has_4_calls(
        self, team_capture, default_user_config
    ):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        traces = [t[0] for t in data.values()]
        call_counts = sorted([len(t.api_calls) for t in traces])
        assert call_counts == [4, 4, 17]

    def test_load_dataset_filters_prefetches(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        total_calls = sum(len(t[0].api_calls) for t in data.values())
        assert total_calls == 25

    def test_load_dataset_uses_last_session(self, team_capture, default_user_config):
        """Prefetch+real from session 2 only; sessions 0 and 1 are ignored."""
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        traces = [t[0] for t in data.values()]
        parent = max(traces, key=lambda t: len(t.api_calls))
        assert parent.system_prompt_text is not None
        assert "expert coding assistant" in parent.system_prompt_text

    def test_load_dataset_token_counts(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        traces = [t[0] for t in data.values()]
        parent = max(traces, key=lambda t: len(t.api_calls))
        first_call = parent.api_calls[0]
        assert first_call.input_tokens == 1000
        assert first_call.output_tokens == 200

    def test_load_dataset_thread_keys_unique(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        traces = [t[0] for t in data.values()]
        keys = {t.thread_key for t in traces}
        assert len(keys) == 3

    # --- convert_to_conversations tests ---

    def test_convert_conversation_count(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 3

    def test_convert_parent_turn_count(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.agent_depth == 0
        assert len(parent.turns) == 17

    def test_convert_children_marked(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        children = [c for c in conversations if c.agent_depth > 0]
        assert len(children) == 2

    def test_convert_parent_has_system_message(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.system_message is not None
        assert "expert coding assistant" in parent.system_message

    def test_convert_parent_has_tools(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.tools is not None
        assert len(parent.tools) == 35

    def test_convert_turn_has_raw_messages(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        for turn in parent.turns:
            assert turn.raw_messages is not None
            assert len(turn.raw_messages) == 1
            assert turn.raw_messages[0]["role"] == "user"

    def test_convert_turn_has_model(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.turns[0].model == "claude-opus-4-6"

    def test_convert_turn_has_input_tokens(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.turns[0].input_tokens == 1000

    def test_convert_turn_has_delay(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.turns[0].delay is None
        assert parent.turns[1].delay is not None
        assert parent.turns[1].delay > 0

    def test_convert_discard_responses(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        for conv in conversations:
            assert conv.discard_responses is True

    def test_convert_raw_payload_present(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        for turn in parent.turns:
            assert turn.raw_payload is not None
            assert "messages" in turn.raw_payload
            assert "model" in turn.raw_payload
            assert "max_tokens" in turn.raw_payload
            assert "stream" in turn.raw_payload
            assert turn.raw_messages is not None

    def test_convert_subagent_spawns(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert len(parent.subagent_spawns) == 2

    def test_convert_subagent_spawn_ids_on_join_turns(
        self, team_capture, default_user_config
    ):
        """Each child's spawn_id is appended to the parent join turn's subagent_spawn_ids."""
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]

        annotated_turns = [
            (i, t.subagent_spawn_ids)
            for i, t in enumerate(parent.turns)
            if t.subagent_spawn_ids
        ]
        assert len(annotated_turns) == 2
        for _, spawn_ids in annotated_turns:
            assert len(spawn_ids) == 1
            assert spawn_ids[0].startswith("s")

    def test_convert_non_spawn_turns_have_empty_spawn_ids(
        self, team_capture, default_user_config
    ):
        """Turns without subagent spawns have empty subagent_spawn_ids."""
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]

        non_spawn_turns = [t for t in parent.turns if not t.subagent_spawn_ids]
        assert len(non_spawn_turns) == 15

    def test_convert_growing_message_counts(self, team_capture, default_user_config):
        loader = ApiCaptureTraceLoader(
            filename=str(team_capture), user_config=default_user_config
        )
        data = loader.load_dataset()
        traces = [t[0] for t in data.values()]
        parent = max(traces, key=lambda t: len(t.api_calls))
        msg_counts = [len(call.messages) for call in parent.api_calls]
        # Each turn adds 2 messages (user + assistant), starting from 1
        assert msg_counts[0] == 1
        assert msg_counts[-1] == 33

    # --- Edge cases ---

    def test_empty_directory(self, tmp_path, default_user_config):
        d = tmp_path / "empty_capture"
        d.mkdir()
        assert not ApiCaptureTraceLoader.can_load(filename=str(d))

    def test_single_thread(self, tmp_path, default_user_config):
        """A capture with only parent thread (no subagents)."""
        d = tmp_path / "single_thread"
        d.mkdir()

        capture_entries = []
        for i in range(3):
            ts = 1000.0 + i * 5
            capture_entries.append(
                _make_capture_entry(i, "request", ts, stream=True, max_tokens=32000)
            )
            capture_entries.append(
                _make_capture_entry(i, "response", ts + 1, input_tokens=500)
            )
            _make_req_file(
                d,
                i,
                system=[{"type": "text", "text": "System prompt"}],
                tools=[],
                messages=[{"role": "user", "content": f"Turn {i}"}],
                model="claude-opus-4-6",
            )

        (d / "capture.jsonl").write_text(
            "\n".join(json.dumps(e) for e in capture_entries) + "\n"
        )

        loader = ApiCaptureTraceLoader(filename=str(d), user_config=default_user_config)
        data = loader.load_dataset()
        assert len(data) == 1

        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert conversations[0].agent_depth == 0
        assert len(conversations[0].turns) == 3
        assert len(conversations[0].subagent_spawns) == 0


# =========================================================================
# _extract_user_content and raw_messages wrapping tests
# =========================================================================


class TestExtractUserContent:
    """Tests for ApiCaptureTraceLoader._extract_user_content static method."""

    def test_extracts_last_user_message_string_content(self):
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
            {"role": "user", "content": "second question"},
        ]
        result = ApiCaptureTraceLoader._extract_user_content(messages)
        assert result == "second question"

    def test_extracts_last_user_message_list_content(self):
        messages = [
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
        ]
        result = ApiCaptureTraceLoader._extract_user_content(messages)
        assert isinstance(result, list)
        assert result[0]["type"] == "tool_result"

    def test_no_user_message_returns_none(self):
        messages = [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
        result = ApiCaptureTraceLoader._extract_user_content(messages)
        assert result is None

    def test_empty_messages_returns_none(self):
        result = ApiCaptureTraceLoader._extract_user_content([])
        assert result is None


class TestRawMessagesWrapping:
    """Tests for raw_messages format on turns produced by _build_conversation."""

    @pytest.fixture
    def default_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    def test_string_user_content_wrapped_as_raw_messages(
        self, tmp_path, default_user_config
    ):
        """When last user message has string content, raw_messages wraps it."""
        d = tmp_path / "string_content"
        d.mkdir()

        capture_entries = [
            _make_capture_entry(0, "request", 1000.0, stream=True, max_tokens=32000),
            _make_capture_entry(0, "response", 1001.0, input_tokens=100),
        ]
        _make_req_file(
            d,
            0,
            system=[{"type": "text", "text": "System"}],
            tools=[],
            messages=[{"role": "user", "content": "Hello world"}],
            model="test-model",
        )
        (d / "capture.jsonl").write_text(
            "\n".join(json.dumps(e) for e in capture_entries) + "\n"
        )

        loader = ApiCaptureTraceLoader(filename=str(d), user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        turn = conversations[0].turns[0]
        assert turn.raw_messages == [{"role": "user", "content": "Hello world"}]

    def test_list_user_content_wrapped_as_raw_messages(
        self, tmp_path, default_user_config
    ):
        """When last user message has content blocks, raw_messages preserves the list."""
        d = tmp_path / "list_content"
        d.mkdir()

        tool_result_content = [
            {"type": "tool_result", "tool_use_id": "tu-1", "content": "file data"}
        ]
        capture_entries = [
            _make_capture_entry(0, "request", 1000.0, stream=True, max_tokens=32000),
            _make_capture_entry(0, "response", 1001.0, input_tokens=100),
        ]
        _make_req_file(
            d,
            0,
            system=[{"type": "text", "text": "System"}],
            tools=[],
            messages=[{"role": "user", "content": tool_result_content}],
            model="test-model",
        )
        (d / "capture.jsonl").write_text(
            "\n".join(json.dumps(e) for e in capture_entries) + "\n"
        )

        loader = ApiCaptureTraceLoader(filename=str(d), user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        turn = conversations[0].turns[0]
        assert turn.raw_messages == [{"role": "user", "content": tool_result_content}]

    def test_no_user_message_produces_none_raw_messages(
        self, tmp_path, default_user_config
    ):
        """When messages have no user role, raw_messages is None."""
        d = tmp_path / "no_user"
        d.mkdir()

        capture_entries = [
            _make_capture_entry(0, "request", 1000.0, stream=True, max_tokens=32000),
            _make_capture_entry(0, "response", 1001.0, input_tokens=100),
        ]
        _make_req_file(
            d,
            0,
            system=[{"type": "text", "text": "System"}],
            tools=[],
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "only assistant"}],
                }
            ],
            model="test-model",
        )
        (d / "capture.jsonl").write_text(
            "\n".join(json.dumps(e) for e in capture_entries) + "\n"
        )

        loader = ApiCaptureTraceLoader(filename=str(d), user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        turn = conversations[0].turns[0]
        assert turn.raw_messages is None
