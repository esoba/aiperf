# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ConfluxLoader."""

import base64

import orjson
import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.dataset.loader.conflux import ConfluxLoader, _parse_timestamp_ms
from aiperf.dataset.loader.models import ConfluxRecord
from aiperf.plugin.enums import DatasetSamplingStrategy

# =========================================================================
# Test data builders
# =========================================================================

BASE_TS = "2026-02-25T02:02:00.000Z"
PARENT_TOOLS = [{"name": f"tool_{i}", "description": f"Tool {i}"} for i in range(10)]
CHILD_TOOLS = [
    {"name": f"sub_tool_{i}", "description": f"Sub Tool {i}"} for i in range(5)
]


def _ts(offset_s: float) -> str:
    """Generate ISO timestamp with offset from base."""
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 2, 25, 2, 2, 0, tzinfo=timezone.utc)
    dt = base + timedelta(seconds=offset_s)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _make_record(
    *,
    record_id: str = "req_001",
    agent_id: str | None = "claude",
    is_subagent: bool = False,
    timestamp: str = BASE_TS,
    model: str = "claude-sonnet-4-6",
    messages: list | None = None,
    tools: list | None = None,
    tokens: dict | None = None,
    hyperparameters: dict | None = None,
    is_streaming: bool | None = True,
    duration_ms: int = 1000,
) -> dict:
    """Build a raw Conflux record dict."""
    return {
        "id": record_id,
        "session_id": "sess-001",
        "agent_id": agent_id,
        "is_subagent": is_subagent,
        "timestamp": timestamp,
        "duration_ms": duration_ms,
        "model": model,
        "tokens": tokens
        or {
            "input": 1000,
            "input_cached": 800,
            "input_cache_write": 100,
            "output": 200,
        },
        "tools": tools or [],
        "messages": messages or [{"role": "user", "content": "Hello"}],
        "output": [{"type": "text", "text": "Hi there"}],
        "hyperparameters": hyperparameters or {"max_tokens": 4096},
        "is_streaming": is_streaming,
        "ttft_ms": 150,
    }


def _build_session_file(tmp_path, records: list[dict]) -> str:
    """Write records to a JSON file and return the path."""
    path = tmp_path / "session.json"
    path.write_bytes(orjson.dumps(records))
    return str(path)


def _build_team_session(tmp_path) -> str:
    """Build a session with parent (5 turns) + 2 subagents (3 turns each)."""
    records = []

    # Parent: 5 turns
    for i in range(5):
        records.append(
            _make_record(
                record_id=f"req_parent_{i:03d}",
                agent_id="claude",
                is_subagent=False,
                timestamp=_ts(i * 5),
                model="claude-opus-4-6",
                messages=[{"role": "user", "content": f"parent turn {i}"}],
                tools=PARENT_TOOLS,
                tokens={
                    "input": 1000 + i * 500,
                    "input_cached": 800,
                    "input_cache_write": 100,
                    "output": 200 + i * 50,
                },
                hyperparameters={"max_tokens": 32000},
            )
        )

    # Subagent A: 3 turns, spawned around parent turn 2
    for i in range(3):
        records.append(
            _make_record(
                record_id=f"req_sub_a_{i:03d}",
                agent_id="sub_a",
                is_subagent=True,
                timestamp=_ts(10 + i * 3),
                model="claude-opus-4-6",
                messages=[{"role": "user", "content": f"sub_a turn {i}"}],
                tools=CHILD_TOOLS,
                tokens={
                    "input": 500 + i * 200,
                    "input_cached": 400,
                    "input_cache_write": 50,
                    "output": 100,
                },
                hyperparameters={"max_tokens": 16000},
            )
        )

    # Subagent B: 3 turns, spawned around parent turn 3
    for i in range(3):
        records.append(
            _make_record(
                record_id=f"req_sub_b_{i:03d}",
                agent_id="sub_b",
                is_subagent=True,
                timestamp=_ts(15 + i * 2),
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": f"sub_b turn {i}"}],
                tools=[],
                tokens={
                    "input": 200,
                    "input_cached": 150,
                    "input_cache_write": 20,
                    "output": 50,
                },
            )
        )

    # Add a record with agent_id=None (should be filtered out)
    records.append(
        _make_record(
            record_id="req_haiku_tool",
            agent_id=None,
            timestamp=_ts(8),
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "tool check"}],
        )
    )

    return _build_session_file(tmp_path, records)


# =========================================================================
# Model tests
# =========================================================================


class TestConfluxRecord:
    def test_create(self):
        record = ConfluxRecord.model_validate(_make_record())
        assert record.id == "req_001"
        assert record.agent_id == "claude"
        assert not record.is_subagent
        assert record.tokens is not None
        assert record.tokens.input == 1000

    def test_defaults(self):
        raw = {
            "id": "req_min",
            "session_id": "sess",
            "timestamp": BASE_TS,
            "messages": [{"role": "user", "content": "hi"}],
        }
        record = ConfluxRecord.model_validate(raw)
        assert record.agent_id is None
        assert not record.is_subagent
        assert record.tokens is None
        assert record.tools == []
        assert record.output == []
        assert record.hyperparameters is None
        assert record.is_streaming is None


# =========================================================================
# can_load tests
# =========================================================================


class TestCanLoad:
    def test_valid_json_file(self, tmp_path):
        path = _build_session_file(tmp_path, [_make_record()])
        assert ConfluxLoader.can_load(filename=path)

    def test_not_json_extension(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_bytes(orjson.dumps([_make_record()]))
        assert not ConfluxLoader.can_load(filename=str(path))

    def test_empty_array(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_bytes(orjson.dumps([]))
        assert not ConfluxLoader.can_load(filename=str(path))

    def test_missing_agent_id_key(self, tmp_path):
        path = tmp_path / "no_agent.json"
        path.write_bytes(orjson.dumps([{"is_subagent": False, "messages": []}]))
        assert not ConfluxLoader.can_load(filename=str(path))

    def test_missing_is_subagent_key(self, tmp_path):
        path = tmp_path / "no_sub.json"
        path.write_bytes(orjson.dumps([{"agent_id": "x", "messages": []}]))
        assert not ConfluxLoader.can_load(filename=str(path))

    def test_missing_messages_key(self, tmp_path):
        path = tmp_path / "no_msg.json"
        path.write_bytes(orjson.dumps([{"agent_id": "x", "is_subagent": False}]))
        assert not ConfluxLoader.can_load(filename=str(path))

    def test_not_a_list(self, tmp_path):
        path = tmp_path / "obj.json"
        path.write_bytes(orjson.dumps({"agent_id": "x"}))
        assert not ConfluxLoader.can_load(filename=str(path))

    def test_none_filename(self):
        assert not ConfluxLoader.can_load(filename=None)

    def test_directory(self, tmp_path):
        assert not ConfluxLoader.can_load(filename=str(tmp_path))

    def test_nonexistent_file(self):
        assert not ConfluxLoader.can_load(filename="/nonexistent/file.json")

    def test_preferred_sampling_strategy(self):
        assert (
            ConfluxLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )


# =========================================================================
# load_dataset tests
# =========================================================================


class TestLoadDataset:
    @pytest.fixture
    def default_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    @pytest.fixture
    def team_session(self, tmp_path):
        return _build_team_session(tmp_path)

    def test_group_count(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        assert len(data) == 3  # parent + 2 subagents

    def test_filters_null_agent_id(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        total = sum(len(recs) for recs in data.values())
        assert total == 11  # 5 + 3 + 3, not 12

    def test_parent_has_5_records(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        assert len(data["claude"]) == 5

    def test_subagent_a_has_3_records(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        assert len(data["sub_a"]) == 3

    def test_sorted_by_timestamp(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        for records in data.values():
            timestamps = [_parse_timestamp_ms(r.timestamp) for r in records]
            assert timestamps == sorted(timestamps)

    def test_single_record(self, tmp_path, default_user_config):
        path = _build_session_file(tmp_path, [_make_record()])
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        assert len(data) == 1
        assert len(data["claude"]) == 1


# =========================================================================
# convert_to_conversations tests
# =========================================================================


class TestConvertToConversations:
    @pytest.fixture
    def default_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    @pytest.fixture
    def team_session(self, tmp_path):
        return _build_team_session(tmp_path)

    def test_conversation_count(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 3

    def test_parent_turn_count(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.agent_depth == 0
        assert len(parent.turns) == 5

    def test_children_marked(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        children = [c for c in conversations if c.agent_depth > 0]
        assert len(children) == 2

    def test_discard_responses(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        for conv in conversations:
            assert conv.discard_responses is True

    def test_parent_has_tools(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.tools is not None
        assert len(parent.tools) == 10

    def test_raw_payload_present(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        for turn in parent.turns:
            assert turn.raw_payload is not None
            assert "messages" in turn.raw_payload
            assert "model" in turn.raw_payload

    def test_raw_payload_has_max_tokens(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.turns[0].raw_payload["max_tokens"] == 32000

    def test_raw_payload_has_stream(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.turns[0].raw_payload["stream"] is True

    def test_raw_payload_has_tools(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert "tools" in parent.turns[0].raw_payload
        assert len(parent.turns[0].raw_payload["tools"]) == 10

    def test_turn_model(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.turns[0].model == "claude-opus-4-6"

    def test_turn_input_tokens(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.turns[0].input_tokens == 1000

    def test_turn_has_absolute_timestamps(self, team_session, default_user_config):
        """Turns use absolute timestamps, not relative delays."""
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.turns[0].timestamp is not None
        assert parent.turns[1].timestamp is not None
        assert parent.turns[1].timestamp > parent.turns[0].timestamp
        assert parent.turns[0].delay is None

    def test_timestamp_spacing(self, team_session, default_user_config):
        """Parent records are 5s apart, timestamps should reflect that."""
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        delta = parent.turns[1].timestamp - parent.turns[0].timestamp
        assert delta == pytest.approx(5000, abs=1)

    def test_raw_messages_present(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        for turn in parent.turns:
            assert turn.raw_messages is not None
            assert len(turn.raw_messages) == 1
            assert turn.raw_messages[0]["role"] == "user"

    def test_subagent_spawns(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert len(parent.subagent_spawns) == 2

    def test_subagent_spawns_are_background(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        for spawn in parent.subagent_spawns:
            assert spawn.is_background is True

    def test_subagent_spawn_ids_on_turns(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
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

    def test_empty_data(self, tmp_path, default_user_config):
        """Empty dict produces no conversations."""
        path = _build_session_file(tmp_path, [_make_record()])
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        conversations = loader.convert_to_conversations({})
        assert conversations == []

    def test_session_id_prefix(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        for conv in conversations:
            assert conv.session_id.startswith("conflux_")

    def test_no_hyperparameters_defaults_max_tokens(
        self, tmp_path, default_user_config
    ):
        """When hyperparameters is None, max_tokens defaults to 4096."""
        records = [_make_record(hyperparameters=None)]
        path = _build_session_file(tmp_path, records)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert conversations[0].turns[0].max_tokens == 4096

    def test_no_streaming_omits_stream_in_payload(self, tmp_path, default_user_config):
        """When is_streaming is None, stream key is omitted from raw_payload."""
        records = [_make_record(is_streaming=None)]
        path = _build_session_file(tmp_path, records)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert "stream" not in conversations[0].turns[0].raw_payload


# =========================================================================
# _parse_timestamp_ms tests
# =========================================================================


class TestParseTimestampMs:
    def test_utc_timestamp(self):
        ms = _parse_timestamp_ms("2026-02-25T02:02:00.000Z")
        assert ms > 0

    def test_millisecond_precision(self):
        ms1 = _parse_timestamp_ms("2026-02-25T02:02:00.000Z")
        ms2 = _parse_timestamp_ms("2026-02-25T02:02:01.000Z")
        assert ms2 - ms1 == pytest.approx(1000, abs=1)

    def test_fractional_seconds(self):
        ms1 = _parse_timestamp_ms("2026-02-25T02:02:00.000Z")
        ms2 = _parse_timestamp_ms("2026-02-25T02:02:00.500Z")
        assert ms2 - ms1 == pytest.approx(500, abs=1)


# =========================================================================
# _extract_user_content tests
# =========================================================================


class TestExtractUserContent:
    def test_string_content(self):
        messages = [{"role": "user", "content": "Hello"}]
        assert ConfluxLoader._extract_user_content(messages) == "Hello"

    def test_last_user_message(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
            {"role": "user", "content": "second"},
        ]
        assert ConfluxLoader._extract_user_content(messages) == "second"

    def test_list_content(self):
        content = [{"type": "tool_result", "tool_use_id": "tu-1", "content": "data"}]
        messages = [{"role": "user", "content": content}]
        result = ConfluxLoader._extract_user_content(messages)
        assert isinstance(result, list)

    def test_no_user_message(self):
        messages = [{"role": "assistant", "content": []}]
        assert ConfluxLoader._extract_user_content(messages) is None

    def test_empty_messages(self):
        assert ConfluxLoader._extract_user_content([]) is None


# =========================================================================
# _build_raw_payload tests (base64 vs fallback)
# =========================================================================


def _b64_encode(obj: dict) -> str:
    """Base64-encode a dict as JSON."""
    return base64.b64encode(orjson.dumps(obj)).decode()


class TestBuildRawPayload:
    """Tests for ConfluxLoader._build_raw_payload."""

    def test_base64_request_body_used_when_present(self):
        """When base64.request_body is present, it becomes the raw_payload."""
        ground_truth = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "system": [{"type": "text", "text": "Be helpful"}],
            "tools": [{"name": "Bash"}],
            "max_tokens": 32000,
            "stream": True,
            "thinking": {"type": "adaptive"},
        }
        raw = _make_record(
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "Be helpful"}]},
                {"role": "user", "content": "hello"},
            ],
        )
        raw["base64"] = {
            "request_body": _b64_encode(ground_truth),
            "response_body": "",
            "provider_usage": "",
        }
        record = ConfluxRecord.model_validate(raw)
        payload = ConfluxLoader._build_raw_payload(record)
        assert payload == ground_truth

    def test_base64_metadata_stripped(self):
        """metadata key is removed from base64 payload (contains PII)."""
        ground_truth = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32000,
            "stream": True,
            "metadata": {"user_id": "secret_user_id"},
        }
        raw = _make_record()
        raw["base64"] = {"request_body": _b64_encode(ground_truth)}
        record = ConfluxRecord.model_validate(raw)
        payload = ConfluxLoader._build_raw_payload(record)
        assert "metadata" not in payload
        assert payload["model"] == "claude-sonnet-4-6"

    def test_base64_includes_thinking(self):
        """thinking key is preserved from base64 payload."""
        ground_truth = {
            "model": "claude-sonnet-4-6",
            "messages": [],
            "thinking": {"type": "adaptive"},
            "max_tokens": 32000,
            "stream": True,
        }
        raw = _make_record()
        raw["base64"] = {"request_body": _b64_encode(ground_truth)}
        record = ConfluxRecord.model_validate(raw)
        payload = ConfluxLoader._build_raw_payload(record)
        assert payload["thinking"] == {"type": "adaptive"}

    def test_base64_includes_system(self):
        """system key is preserved from base64 payload."""
        system_blocks = [{"type": "text", "text": "System prompt"}]
        ground_truth = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "system": system_blocks,
            "max_tokens": 32000,
            "stream": True,
        }
        raw = _make_record()
        raw["base64"] = {"request_body": _b64_encode(ground_truth)}
        record = ConfluxRecord.model_validate(raw)
        payload = ConfluxLoader._build_raw_payload(record)
        assert payload["system"] == system_blocks

    def test_base64_includes_temperature(self):
        """temperature key is preserved from base64 payload."""
        ground_truth = {
            "model": "claude-sonnet-4-6",
            "messages": [],
            "max_tokens": 32000,
            "stream": True,
            "temperature": 1.0,
        }
        raw = _make_record()
        raw["base64"] = {"request_body": _b64_encode(ground_truth)}
        record = ConfluxRecord.model_validate(raw)
        payload = ConfluxLoader._build_raw_payload(record)
        assert payload["temperature"] == 1.0

    def test_fallback_strips_system_message(self):
        """Without base64, system-role message[0] is split into system key."""
        system_blocks = [{"type": "text", "text": "Be helpful"}]
        raw = _make_record(
            messages=[
                {"role": "system", "content": system_blocks},
                {"role": "user", "content": "hello"},
            ],
        )
        record = ConfluxRecord.model_validate(raw)
        payload = ConfluxLoader._build_raw_payload(record)
        assert payload["system"] == system_blocks
        assert payload["messages"] == [{"role": "user", "content": "hello"}]

    def test_fallback_no_system_message(self):
        """Without base64 and no system-role message, messages pass through."""
        raw = _make_record(
            messages=[{"role": "user", "content": "hello"}],
        )
        record = ConfluxRecord.model_validate(raw)
        payload = ConfluxLoader._build_raw_payload(record)
        assert "system" not in payload
        assert payload["messages"] == [{"role": "user", "content": "hello"}]

    def test_fallback_includes_temperature(self):
        """Without base64, temperature from hyperparameters is included."""
        raw = _make_record(
            hyperparameters={"max_tokens": 4096, "temperature": 0.7},
        )
        record = ConfluxRecord.model_validate(raw)
        payload = ConfluxLoader._build_raw_payload(record)
        assert payload["temperature"] == 0.7

    def test_fallback_no_temperature_when_absent(self):
        """Without base64, temperature is omitted when not in hyperparameters."""
        raw = _make_record(hyperparameters={"max_tokens": 4096})
        record = ConfluxRecord.model_validate(raw)
        payload = ConfluxLoader._build_raw_payload(record)
        assert "temperature" not in payload


class TestBase64Integration:
    """Integration tests: base64 payloads flow through to conversation turns."""

    @pytest.fixture
    def default_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    def test_base64_payload_on_turns(self, tmp_path, default_user_config):
        """Turns use the base64 request_body as raw_payload."""
        ground_truth = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "system": [{"type": "text", "text": "System"}],
            "tools": [{"name": "Bash"}],
            "max_tokens": 32000,
            "stream": True,
            "thinking": {"type": "adaptive"},
        }
        raw = _make_record(
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "System"}]},
                {"role": "user", "content": "hello"},
            ],
            tools=[{"name": "Bash"}],
        )
        raw["base64"] = {"request_body": _b64_encode(ground_truth)}
        path = _build_session_file(tmp_path, [raw])

        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        turn = conversations[0].turns[0]
        assert turn.raw_payload == ground_truth
        assert turn.raw_payload["thinking"] == {"type": "adaptive"}
        assert turn.raw_payload["system"] == [{"type": "text", "text": "System"}]
        assert len(turn.raw_payload["messages"]) == 1

    def test_base64_max_tokens_propagates(self, tmp_path, default_user_config):
        """max_tokens from base64 payload sets Turn.max_tokens."""
        ground_truth = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16000,
            "stream": True,
        }
        raw = _make_record()
        raw["base64"] = {"request_body": _b64_encode(ground_truth)}
        path = _build_session_file(tmp_path, [raw])

        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert conversations[0].turns[0].max_tokens == 16000
