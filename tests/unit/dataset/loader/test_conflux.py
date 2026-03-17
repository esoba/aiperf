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
        assert record.is_subagent is None
        assert record.source is None
        assert record.client is None
        assert record.provider is None
        assert record.completed_at is None
        assert record.client_version is None
        assert record.request_id is None
        assert record.tokens is None
        assert record.tools == []
        assert record.output == []
        assert record.hyperparameters is None
        assert record.is_streaming is None

    def test_unified_fields_roundtrip(self):
        """Fields from the Conflux unified schema are preserved."""
        raw = _make_record()
        raw["source"] = "proxy"
        raw["client"] = "claude"
        raw["provider"] = "anthropic"
        raw["completed_at"] = "2026-02-25T02:02:01.000Z"
        raw["client_version"] = "1.2.3"
        raw["request_id"] = "req_abc123"
        record = ConfluxRecord.model_validate(raw)
        assert record.source == "proxy"
        assert record.client == "claude"
        assert record.provider == "anthropic"
        assert record.completed_at == "2026-02-25T02:02:01.000Z"
        assert record.client_version == "1.2.3"
        assert record.request_id == "req_abc123"


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

    def test_large_file_byte_probe_fallback(self, tmp_path, monkeypatch):
        """When file exceeds probe limit, byte-level detection is used."""
        from aiperf.dataset.loader import conflux as conflux_mod

        # 300 bytes is enough to contain signature fields in the first record
        # but truncates the 2KB+ file, forcing the byte-level fallback path
        monkeypatch.setattr(conflux_mod, "_CAN_LOAD_PROBE_BYTES", 300)

        records = [_make_record() for _ in range(5)]
        path = _build_session_file(tmp_path, records)
        assert ConfluxLoader.can_load(filename=path)

    def test_large_file_byte_probe_rejects_non_array(self, tmp_path, monkeypatch):
        """Byte-level fallback rejects files that don't start with '['."""
        from aiperf.dataset.loader import conflux as conflux_mod

        monkeypatch.setattr(conflux_mod, "_CAN_LOAD_PROBE_BYTES", 8)

        # Content is longer than 8 bytes but not a JSON array
        path = tmp_path / "obj.json"
        path.write_bytes(
            b'{"agent_id": "x", "is_subagent": false, "messages": [],'
            b' "padding": "' + b"x" * 100 + b'"}'
        )
        assert not ConfluxLoader.can_load(filename=str(path))

    def test_large_file_byte_probe_rejects_missing_fields(self, tmp_path, monkeypatch):
        """Byte-level fallback rejects files without Conflux signature fields."""
        from aiperf.dataset.loader import conflux as conflux_mod

        monkeypatch.setattr(conflux_mod, "_CAN_LOAD_PROBE_BYTES", 8)

        # Array that's long enough to truncate but has no Conflux fields
        path = tmp_path / "other.json"
        path.write_bytes(b'[{"type": "single_turn", "text": "' + b"x" * 100 + b'"}]')
        assert not ConfluxLoader.can_load(filename=str(path))


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
                conflux_include_utility_calls=True,
            ),
        )

    @pytest.fixture
    def team_session(self, tmp_path):
        return _build_team_session(tmp_path)

    def test_group_count(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        assert len(data) == 4  # parent + 2 subagents + 1 orphan

    def test_orphan_as_separate_group(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        total = sum(len(recs) for recs in data.values())
        assert total == 12  # 5 + 3 + 3 + 1 orphan

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
                conflux_include_utility_calls=True,
            ),
        )

    @pytest.fixture
    def team_session(self, tmp_path):
        return _build_team_session(tmp_path)

    def test_conversation_count(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 4  # parent + 2 children + 1 orphan child

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
        assert len(children) == 3  # 2 explicit subagents + 1 orphan

    def test_every_turn_has_raw_tools(self, team_session, default_user_config):
        """Every turn should have raw_tools set from its record's tools."""
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        for turn in parent.turns:
            assert turn.raw_tools is not None

    def test_fallback_uses_raw_messages(self, team_session, default_user_config):
        """Without base64, turns use raw_messages for provider-agnostic replay."""
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        for turn in parent.turns:
            assert turn.raw_messages is not None

    def test_fallback_max_tokens_on_turn(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.turns[0].max_tokens == 32000

    def test_fallback_raw_tools_present(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert parent.turns[0].raw_tools is not None
        assert len(parent.turns[0].raw_tools) == 10

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
            assert len(turn.raw_messages) >= 1

    def test_subagent_spawns(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        assert len(parent.subagent_spawns) == 3  # 2 explicit + 1 orphan

    def test_subagent_spawn_blocking_detection(self, team_session, default_user_config):
        """Explicit subagent children are detected as blocking or background
        based on timestamp analysis; orphans are always background."""
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        # Explicit children have gap_ms <= 2000 -> blocking
        # Orphan is always background
        explicit_spawns = [
            s for s in parent.subagent_spawns if len(s.child_conversation_ids) >= 1
        ]
        orphan_spawns = [
            s
            for s in parent.subagent_spawns
            if any("orphan" in cid for cid in s.child_conversation_ids)
        ]
        non_orphan_spawns = [s for s in explicit_spawns if s not in orphan_spawns]
        for spawn in non_orphan_spawns:
            assert spawn.is_background is False
        for spawn in orphan_spawns:
            assert spawn.is_background is True

    def test_children_have_parent_conversation_id(
        self, team_session, default_user_config
    ):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        children = [c for c in conversations if c.agent_depth > 0]
        assert len(children) >= 1
        for child in children:
            assert child.parent_conversation_id == parent.session_id

    def test_subagent_spawn_ids_on_turns(self, team_session, default_user_config):
        loader = ConfluxLoader(filename=team_session, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]
        all_spawn_ids = [sid for t in parent.turns for sid in t.subagent_spawn_ids]
        assert len(all_spawn_ids) == 3  # 2 explicit + 1 orphan
        for sid in all_spawn_ids:
            assert sid.startswith("s")

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

    def test_fallback_uses_raw_messages_single_record(
        self, tmp_path, default_user_config
    ):
        """Without base64, fallback path uses raw_messages for a single record."""
        records = [_make_record(is_streaming=None)]
        path = _build_session_file(tmp_path, records)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert conversations[0].turns[0].raw_messages is not None


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
# _build_raw_payload tests (base64 vs fallback)
# =========================================================================


def _b64_encode(obj: dict) -> str:
    """Base64-encode a dict as JSON."""
    return base64.b64encode(orjson.dumps(obj)).decode()


class TestExtractRecordFields:
    """Tests for ConfluxLoader._extract_record_fields."""

    def test_base64_extracts_messages_with_system_inline(self):
        """Base64 path inlines system into messages array."""
        payload = {
            "messages": [{"role": "user", "content": "hello"}],
            "system": [{"type": "text", "text": "Be helpful"}],
            "tools": [{"name": "Bash"}],
            "max_tokens": 32000,
        }
        raw = _make_record()
        raw["base64"] = {"request_body": _b64_encode(payload)}
        record = ConfluxRecord.model_validate(raw)
        messages, tools, max_tokens = ConfluxLoader._extract_record_fields(record)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert tools == [{"name": "Bash"}]
        assert max_tokens == 32000

    def test_base64_strips_metadata(self):
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"user_id": "secret"},
        }
        raw = _make_record()
        raw["base64"] = {"request_body": _b64_encode(payload)}
        record = ConfluxRecord.model_validate(raw)
        messages, _, _ = ConfluxLoader._extract_record_fields(record)
        assert len(messages) == 1

    def test_fallback_uses_top_level_fields(self):
        raw = _make_record(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            tools=[{"name": "read_file", "input_schema": {}}],
            hyperparameters={"max_tokens": 8192},
        )
        record = ConfluxRecord.model_validate(raw)
        messages, tools, max_tokens = ConfluxLoader._extract_record_fields(record)
        assert len(messages) == 2
        assert tools == [{"name": "read_file", "input_schema": {}}]
        assert max_tokens == 8192

    def test_fallback_no_tools(self):
        raw = _make_record(tools=[])
        record = ConfluxRecord.model_validate(raw)
        _, tools, _ = ConfluxLoader._extract_record_fields(record)
        assert tools is None


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

    def test_base64_normalized_to_raw_messages(self, tmp_path, default_user_config):
        """Base64 records are normalized to raw_messages (not raw_payload)."""
        ground_truth = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "system": [{"type": "text", "text": "System"}],
            "tools": [{"name": "Bash", "description": "run", "input_schema": {}}],
            "max_tokens": 32000,
            "stream": True,
            "thinking": {"type": "adaptive"},
        }
        raw = _make_record(
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "System"}]},
                {"role": "user", "content": "hello"},
            ],
            tools=[{"name": "Bash", "description": "run", "input_schema": {}}],
        )
        raw["base64"] = {"request_body": _b64_encode(ground_truth)}
        path = _build_session_file(tmp_path, [raw])

        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        turn = conversations[0].turns[0]
        assert turn.raw_messages is not None
        # System message should be normalized to inline
        assert any(m["role"] == "system" for m in turn.raw_messages)
        # Tools should be normalized to OpenAI format
        assert turn.raw_tools is not None

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


# =========================================================================
# Provider detection tests
# =========================================================================


class TestDetectConfluxProvider:
    """Tests for ConfluxLoader._detect_conflux_provider."""

    def test_client_claude(self):
        raw = _make_record()
        raw["client"] = "claude"
        record = ConfluxRecord.model_validate(raw)
        assert ConfluxLoader._detect_conflux_provider(record) == "anthropic"

    def test_client_codex(self):
        raw = _make_record()
        raw["client"] = "codex"
        record = ConfluxRecord.model_validate(raw)
        assert ConfluxLoader._detect_conflux_provider(record) == "openai"

    def test_provider_field_takes_precedence(self):
        raw = _make_record()
        raw["client"] = "codex"
        raw["provider"] = "anthropic"
        record = ConfluxRecord.model_validate(raw)
        assert ConfluxLoader._detect_conflux_provider(record) == "anthropic"

    def test_provider_field_openai(self):
        raw = _make_record()
        raw["provider"] = "OpenAI"
        record = ConfluxRecord.model_validate(raw)
        assert ConfluxLoader._detect_conflux_provider(record) == "openai"

    def test_no_hints(self):
        raw = _make_record()
        record = ConfluxRecord.model_validate(raw)
        assert ConfluxLoader._detect_conflux_provider(record) is None


# =========================================================================
# Speedup ratio tests
# =========================================================================


# =========================================================================
# Spawn point overlap tests
# =========================================================================


class TestFindSpawnPoint:
    """Tests for ConfluxLoader._find_spawn_point."""

    def test_overlap_via_completed_at(self):
        """Child spawned during parent turn 1's in-flight period."""
        parent = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p0",
                    timestamp=_ts(0),
                    is_subagent=False,
                    duration_ms=3000,
                )
                | {"completed_at": _ts(3)}
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p1",
                    timestamp=_ts(5),
                    is_subagent=False,
                    duration_ms=8000,
                )
                | {"completed_at": _ts(13)}
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p2",
                    timestamp=_ts(15),
                    is_subagent=False,
                    duration_ms=2000,
                )
                | {"completed_at": _ts(17)}
            ),
        ]
        # Child spawned at t=7, which is during parent turn 1 (t=5 to t=13)
        child = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="c0",
                    timestamp=_ts(7),
                    is_subagent=True,
                )
            ),
        ]
        assert ConfluxLoader._find_spawn_point(parent, child) == 1

    def test_overlap_via_duration_ms(self):
        """Uses duration_ms when completed_at is absent."""
        parent = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p0",
                    timestamp=_ts(0),
                    is_subagent=False,
                    duration_ms=3000,
                )
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p1",
                    timestamp=_ts(5),
                    is_subagent=False,
                    duration_ms=10000,
                )
            ),
        ]
        child = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="c0",
                    timestamp=_ts(8),
                    is_subagent=True,
                )
            ),
        ]
        assert ConfluxLoader._find_spawn_point(parent, child) == 1

    def test_falls_back_to_closest_timestamp(self):
        """No overlap data, falls back to closest timestamp."""
        parent = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p0",
                    timestamp=_ts(0),
                    is_subagent=False,
                    duration_ms=0,
                )
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p1",
                    timestamp=_ts(10),
                    is_subagent=False,
                    duration_ms=0,
                )
            ),
        ]
        child = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="c0",
                    timestamp=_ts(8),
                    is_subagent=True,
                )
            ),
        ]
        assert ConfluxLoader._find_spawn_point(parent, child) == 1


# =========================================================================
# Un-enriched data (is_subagent=None) tests
# =========================================================================


class TestUnEnrichedData:
    """Tests for handling un-enriched proxy data where is_subagent is None."""

    @pytest.fixture
    def default_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    def test_unenriched_elects_largest_group_as_parent(
        self, tmp_path, default_user_config
    ):
        """When all groups have is_subagent=None, largest becomes parent."""
        records = []
        # Agent A: 5 turns (should become parent)
        for i in range(5):
            raw = _make_record(
                record_id=f"a_{i}",
                agent_id="agent_a",
                timestamp=_ts(i * 5),
            )
            del raw["is_subagent"]
            records.append(raw)
        # Agent B: 2 turns (should become child)
        for i in range(2):
            raw = _make_record(
                record_id=f"b_{i}",
                agent_id="agent_b",
                timestamp=_ts(3 + i * 3),
            )
            del raw["is_subagent"]
            records.append(raw)

        path = _build_session_file(tmp_path, records)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        parent = conversations[0]
        assert parent.agent_depth == 0
        assert len(parent.turns) == 5
        child = conversations[1]
        assert child.agent_depth == 1
        assert len(child.turns) == 2

    def test_proxy_source_can_load(self, tmp_path):
        """Records with source=proxy but no is_subagent key are loadable."""
        raw = _make_record()
        del raw["is_subagent"]
        del raw["agent_id"]
        raw["source"] = "proxy"
        path = _build_session_file(tmp_path, [raw])
        assert ConfluxLoader.can_load(filename=path)


# =========================================================================
# Propagated fields tests (extra_params, ground_truth, origin)
# =========================================================================


class TestPropagatedFields:
    """Tests for extra_params, ground_truth, and origin population."""

    @pytest.fixture
    def default_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    def _load_conversations(self, tmp_path, records, user_config):
        path = _build_session_file(tmp_path, records)
        loader = ConfluxLoader(filename=path, user_config=user_config)
        data = loader.load_dataset()
        return loader.convert_to_conversations(data)

    def test_extra_params_from_hyperparameters(self, tmp_path, default_user_config):
        """Hyperparameters beyond max_tokens populate extra_params."""
        records = [
            _make_record(
                hyperparameters={
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "seed": 42,
                }
            )
        ]
        convs = self._load_conversations(tmp_path, records, default_user_config)
        turn = convs[0].turns[0]
        assert turn.extra_params is not None
        assert turn.extra_params["temperature"] == 0.7
        assert turn.extra_params["top_p"] == 0.9
        assert turn.extra_params["seed"] == 42
        assert "max_tokens" not in turn.extra_params

    def test_extra_params_none_when_only_max_tokens(
        self, tmp_path, default_user_config
    ):
        """extra_params is None when hyperparameters only has max_tokens."""
        records = [_make_record(hyperparameters={"max_tokens": 4096})]
        convs = self._load_conversations(tmp_path, records, default_user_config)
        assert convs[0].turns[0].extra_params is None

    def test_extra_params_none_when_no_hyperparameters(
        self, tmp_path, default_user_config
    ):
        """extra_params is None when hyperparameters is absent."""
        records = [_make_record(hyperparameters=None)]
        convs = self._load_conversations(tmp_path, records, default_user_config)
        assert convs[0].turns[0].extra_params is None

    def test_ground_truth_from_tokens_and_timing(self, tmp_path, default_user_config):
        """Token breakdown, timing, and output populate ground_truth."""
        records = [
            _make_record(
                tokens={
                    "input": 1000,
                    "input_cached": 800,
                    "input_cache_write": 100,
                    "output": 200,
                    "output_reasoning": 50,
                },
                is_streaming=True,
                duration_ms=1500,
            )
        ]
        convs = self._load_conversations(tmp_path, records, default_user_config)
        gt = convs[0].turns[0].ground_truth
        assert gt is not None
        assert gt.input_cached_tokens == 800
        assert gt.input_cache_write_tokens == 100
        assert gt.output_tokens == 200
        assert gt.output_reasoning_tokens == 50
        assert gt.ttft_ms == 150  # from _make_record default
        assert gt.duration_ms == 1500
        assert gt.is_streaming is True

    def test_ground_truth_none_when_no_detail(self, tmp_path, default_user_config):
        """ground_truth is None when no token detail, timing, or output."""
        raw = _make_record(
            tokens={
                "input": 100,
                "input_cached": 0,
                "input_cache_write": 0,
                "output": 0,
            },
            is_streaming=None,
            duration_ms=0,
        )
        raw["ttft_ms"] = None
        raw["output"] = []
        records = [raw]
        convs = self._load_conversations(tmp_path, records, default_user_config)
        assert convs[0].turns[0].ground_truth is None

    def test_origin_from_record_metadata(self, tmp_path, default_user_config):
        """Provenance populated from first record's source/client/session fields."""
        raw = _make_record()
        raw["source"] = "proxy"
        raw["client"] = "claude"
        raw["client_version"] = "1.2.3"
        raw["request_id"] = "req_abc"
        records = [raw]
        convs = self._load_conversations(tmp_path, records, default_user_config)
        prov = convs[0].origin
        assert prov is not None
        assert prov.source == "proxy"
        assert prov.client == "claude"
        assert prov.client_version == "1.2.3"
        assert prov.original_session_id == "sess-001"
        assert prov.original_request_ids == ["req_abc"]

    def test_origin_collects_all_request_ids(self, tmp_path, default_user_config):
        """All request_ids across turns are collected in origin."""
        records = [
            _make_record(record_id="r0", timestamp=_ts(0)) | {"request_id": "req_0"},
            _make_record(record_id="r1", timestamp=_ts(5)) | {"request_id": "req_1"},
            _make_record(record_id="r2", timestamp=_ts(10)) | {"request_id": "req_2"},
        ]
        convs = self._load_conversations(tmp_path, records, default_user_config)
        assert convs[0].origin.original_request_ids == [
            "req_0",
            "req_1",
            "req_2",
        ]

    def test_origin_skips_null_request_ids(self, tmp_path, default_user_config):
        """request_ids that are None are excluded from origin."""
        records = [
            _make_record(record_id="r0", timestamp=_ts(0)),
            _make_record(record_id="r1", timestamp=_ts(5)) | {"request_id": "req_1"},
        ]
        convs = self._load_conversations(tmp_path, records, default_user_config)
        assert convs[0].origin.original_request_ids == ["req_1"]
