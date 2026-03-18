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
from aiperf.common.enums import ConversationContextMode, PrerequisiteKind
from aiperf.dataset.loader.conflux import (
    ConfluxLoader,
    _build_spawn_tuid_to_agent_id,
    _detect_join_turn_from_content,
    _extract_notification_joins,
    _find_join_turn_index,
    _iter_message_blocks,
    _new_messages,
    _parse_timestamp_ms,
    _record_end_ms,
    _stringify_block_content,
)
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


_UNSET = object()


def _make_record(
    *,
    record_id: str = "req_001",
    agent_id: str | None = "claude",
    is_subagent: bool = False,
    timestamp: str = BASE_TS,
    model: str = "claude-sonnet-4-6",
    messages: list | object = _UNSET,
    tools: list | object = _UNSET,
    tokens: dict | object = _UNSET,
    hyperparameters: dict | None | object = _UNSET,
    is_streaming: bool | None = True,
    duration_ms: int = 1000,
) -> dict:
    """Build a raw Conflux record dict."""
    record: dict = {
        "id": record_id,
        "session_id": "sess-001",
        "agent_id": agent_id,
        "is_subagent": is_subagent,
        "timestamp": timestamp,
        "duration_ms": duration_ms,
        "model": model,
        "tokens": (
            {
                "input": 1000,
                "input_cached": 800,
                "input_cache_write": 100,
                "output": 200,
            }
            if tokens is _UNSET
            else tokens
        ),
        "tools": [] if tools is _UNSET else tools,
        "messages": (
            [{"role": "user", "content": "Hello"}] if messages is _UNSET else messages
        ),
        "output": [{"type": "text", "text": "Hi there"}],
        "is_streaming": is_streaming,
        "ttft_ms": 150,
    }
    if hyperparameters is not _UNSET:
        record["hyperparameters"] = hyperparameters
    else:
        record["hyperparameters"] = {"max_tokens": 4096}
    return record


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


def _build_delayed_join_session(tmp_path, *, with_content_signals: bool) -> str:
    """Build a parent/child session where the join happens at parent turn 3."""
    if with_content_signals:
        turn0_messages = [{"role": "user", "content": "parent turn 0"}]
        agent_tool_use = {
            "type": "tool_use",
            "id": "toolu_spawn_1",
            "name": "Agent",
            "input": {"task": "inspect auth flow"},
        }
        queued_result = {
            "type": "tool_result",
            "tool_use_id": "toolu_spawn_1",
            "content": "queued for running",
        }
        full_result = {
            "type": "tool_result",
            "tool_use_id": "toolu_spawn_1",
            "content": "child finished and found the root cause",
        }
        turn1_messages = turn0_messages + [
            {"role": "assistant", "content": [agent_tool_use]},
            {"role": "user", "content": [queued_result]},
            {"role": "user", "content": "parent turn 1"},
        ]
        turn2_messages = turn1_messages + [
            {"role": "assistant", "content": "doing unrelated work"},
            {"role": "user", "content": "parent turn 2"},
        ]
        turn3_messages = turn2_messages + [
            {"role": "assistant", "content": "ready to use child output"},
            {"role": "user", "content": [full_result]},
            {"role": "user", "content": "parent turn 3"},
        ]
        parent_messages = [
            turn0_messages,
            turn1_messages,
            turn2_messages,
            turn3_messages,
        ]
    else:
        parent_messages = [
            [{"role": "user", "content": f"parent turn {i}"}] for i in range(4)
        ]

    records = []
    parent_offsets = [0, 2, 4, 8]
    for i, offset in enumerate(parent_offsets):
        records.append(
            _make_record(
                record_id=f"req_parent_delayed_{i:03d}",
                agent_id="claude",
                is_subagent=False,
                timestamp=_ts(offset),
                messages=parent_messages[i],
                tools=PARENT_TOOLS,
                duration_ms=1000,
            )
        )

    child_offsets = [0.5, 3, 6]
    for i, offset in enumerate(child_offsets):
        records.append(
            _make_record(
                record_id=f"req_child_delayed_{i:03d}",
                agent_id="sub_delayed",
                is_subagent=True,
                timestamp=_ts(offset),
                messages=[{"role": "user", "content": f"child turn {i}"}],
                tools=CHILD_TOOLS,
                duration_ms=1000,
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

    def test_delayed_join_prerequisite_inferred_from_content(
        self, tmp_path, default_user_config
    ):
        path = _build_delayed_join_session(tmp_path, with_content_signals=True)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]

        blocking_spawns = [
            spawn for spawn in parent.subagent_spawns if not spawn.is_background
        ]
        assert len(blocking_spawns) == 1
        spawn_id = blocking_spawns[0].spawn_id

        assert parent.turns[0].subagent_spawn_ids == [spawn_id]
        assert parent.turns[1].prerequisites == []
        assert parent.turns[2].prerequisites == []
        assert len(parent.turns[3].prerequisites) == 1
        assert parent.turns[3].prerequisites[0].kind == PrerequisiteKind.SPAWN_JOIN
        assert parent.turns[3].prerequisites[0].spawn_id == spawn_id

    def test_delayed_join_prerequisite_inferred_from_timing_fallback(
        self, tmp_path, default_user_config
    ):
        path = _build_delayed_join_session(tmp_path, with_content_signals=False)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        parent = conversations[0]

        blocking_spawns = [
            spawn for spawn in parent.subagent_spawns if not spawn.is_background
        ]
        assert len(blocking_spawns) == 1
        spawn_id = blocking_spawns[0].spawn_id

        assert parent.turns[0].subagent_spawn_ids == [spawn_id]
        assert parent.turns[1].prerequisites == []
        assert parent.turns[2].prerequisites == []
        assert len(parent.turns[3].prerequisites) == 1
        assert parent.turns[3].prerequisites[0].kind == PrerequisiteKind.SPAWN_JOIN
        assert parent.turns[3].prerequisites[0].spawn_id == spawn_id

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

    def test_no_hyperparameters_leaves_max_tokens_none(
        self, tmp_path, default_user_config
    ):
        """When hyperparameters is None, max_tokens is None (server default)."""
        records = [_make_record(hyperparameters=None)]
        path = _build_session_file(tmp_path, records)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert conversations[0].turns[0].max_tokens is None

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


# =========================================================================
# get_default_context_mode tests
# =========================================================================


class TestDefaultContextMode:
    def test_returns_message_array_with_responses(self) -> None:
        assert (
            ConfluxLoader.get_default_context_mode()
            == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )


# =========================================================================
# _new_messages helper tests
# =========================================================================


class TestNewMessages:
    def test_identical_prefix_returns_appended(self) -> None:
        prev = [{"role": "user", "content": "a"}]
        curr = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        result = _new_messages(prev, curr)
        assert result == [{"role": "assistant", "content": "b"}]

    def test_no_common_prefix(self) -> None:
        prev = [{"role": "user", "content": "x"}]
        curr = [{"role": "user", "content": "y"}]
        result = _new_messages(prev, curr)
        assert result == [{"role": "user", "content": "y"}]

    def test_empty_previous(self) -> None:
        curr = [{"role": "user", "content": "a"}]
        result = _new_messages([], curr)
        assert result == curr

    def test_empty_current(self) -> None:
        prev = [{"role": "user", "content": "a"}]
        result = _new_messages(prev, [])
        assert result == []

    def test_both_empty(self) -> None:
        assert _new_messages([], []) == []

    def test_full_prefix_match(self) -> None:
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        result = _new_messages(msgs, list(msgs))
        assert result == []


# =========================================================================
# _iter_message_blocks helper tests
# =========================================================================


class TestIterMessageBlocks:
    def test_extracts_dict_blocks_from_list_content(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Agent"},
                    {"type": "text", "text": "hello"},
                ],
            },
        ]
        blocks = _iter_message_blocks(messages)
        assert len(blocks) == 2
        assert blocks[0]["type"] == "tool_use"

    def test_skips_string_content(self) -> None:
        messages = [{"role": "user", "content": "just a string"}]
        blocks = _iter_message_blocks(messages)
        assert blocks == []

    def test_skips_non_dict_items_in_list(self) -> None:
        messages = [{"role": "user", "content": ["string_item", 42, {"type": "text"}]}]
        blocks = _iter_message_blocks(messages)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"

    def test_empty_messages(self) -> None:
        assert _iter_message_blocks([]) == []


# =========================================================================
# _stringify_block_content helper tests
# =========================================================================


class TestStringifyBlockContent:
    def test_string_passthrough(self) -> None:
        assert _stringify_block_content("hello") == "hello"

    def test_list_joins(self) -> None:
        result = _stringify_block_content(["a", "b"])
        assert result == "a b"

    def test_dict_with_text_key(self) -> None:
        assert _stringify_block_content({"text": "found it"}) == "found it"

    def test_dict_with_content_key_recurses(self) -> None:
        nested = {"content": [{"text": "inner"}]}
        result = _stringify_block_content(nested)
        assert "inner" in result

    def test_dict_fallback_to_json(self) -> None:
        result = _stringify_block_content({"key": "value"})
        assert "key" in result
        assert "value" in result

    def test_non_standard_type(self) -> None:
        assert _stringify_block_content(42) == "42"

    def test_nested_list_with_dicts(self) -> None:
        data = [{"text": "a"}, "b"]
        result = _stringify_block_content(data)
        assert "a" in result
        assert "b" in result


# =========================================================================
# _record_end_ms helper tests
# =========================================================================


class TestRecordEndMs:
    def test_uses_completed_at_when_available(self) -> None:
        record = ConfluxRecord.model_validate(
            _make_record(timestamp=_ts(0), duration_ms=5000) | {"completed_at": _ts(10)}
        )
        end = _record_end_ms(record)
        start = _parse_timestamp_ms(_ts(0))
        expected = _parse_timestamp_ms(_ts(10))
        assert end == expected
        assert end != start + 5000

    def test_uses_duration_ms_when_no_completed_at(self) -> None:
        record = ConfluxRecord.model_validate(
            _make_record(timestamp=_ts(0), duration_ms=3000)
        )
        end = _record_end_ms(record)
        start = _parse_timestamp_ms(_ts(0))
        assert end == pytest.approx(start + 3000, abs=1)

    def test_returns_start_when_no_timing_data(self) -> None:
        record = ConfluxRecord.model_validate(
            _make_record(timestamp=_ts(5), duration_ms=0)
        )
        end = _record_end_ms(record)
        start = _parse_timestamp_ms(_ts(5))
        assert end == start


# =========================================================================
# _detect_join_turn_from_content tests
# =========================================================================


def _make_content_records(
    *,
    spawn_tool_use_id: str = "toolu_spawn_x",
    queued_text: str = "queued for running",
    result_text: str = "child finished",
    result_at_turn: int = 3,
    num_turns: int = 4,
    background_only: bool = False,
) -> list[ConfluxRecord]:
    """Build parent records with Agent tool_use / tool_result signals."""
    turn0_msgs = [{"role": "user", "content": "turn 0"}]
    agent_block = {
        "type": "tool_use",
        "id": spawn_tool_use_id,
        "name": "Agent",
        "input": {"task": "do something"},
    }
    queued_block = {
        "type": "tool_result",
        "tool_use_id": spawn_tool_use_id,
        "content": queued_text,
    }
    result_block = {
        "type": "tool_result",
        "tool_use_id": spawn_tool_use_id,
        "content": result_text,
    }

    all_messages: list[list[dict]] = [turn0_msgs]
    for i in range(1, num_turns):
        prev = all_messages[i - 1]
        if i == 1:
            new = prev + [
                {"role": "assistant", "content": [agent_block]},
                {"role": "user", "content": [queued_block]},
                {"role": "user", "content": f"turn {i}"},
            ]
        elif not background_only and i == result_at_turn:
            new = prev + [
                {"role": "assistant", "content": f"work at turn {i}"},
                {"role": "user", "content": [result_block]},
                {"role": "user", "content": f"turn {i}"},
            ]
        else:
            new = prev + [
                {"role": "assistant", "content": f"work at turn {i}"},
                {"role": "user", "content": f"turn {i}"},
            ]
        all_messages.append(new)

    records = []
    for i, msgs in enumerate(all_messages):
        records.append(
            ConfluxRecord.model_validate(
                _make_record(
                    record_id=f"p{i}",
                    agent_id="claude",
                    is_subagent=False,
                    timestamp=_ts(i * 5),
                    messages=msgs,
                    duration_ms=2000,
                )
            )
        )
    return records


class TestDetectJoinTurnFromContent:
    def test_finds_join_at_result_turn(self) -> None:
        records = _make_content_records(result_at_turn=3)
        join_idx, saw_bg = _detect_join_turn_from_content(records, spawn_turn_index=0)
        assert join_idx == 3
        assert saw_bg is False

    def test_background_signal_when_only_queued(self) -> None:
        records = _make_content_records(background_only=True)
        join_idx, saw_bg = _detect_join_turn_from_content(records, spawn_turn_index=0)
        assert join_idx is None
        assert saw_bg is True

    def test_spawn_at_last_turn_returns_none(self) -> None:
        records = _make_content_records(num_turns=2)
        join_idx, saw_bg = _detect_join_turn_from_content(records, spawn_turn_index=1)
        assert join_idx is None
        assert saw_bg is False

    def test_ignores_existing_agent_ids(self) -> None:
        """Pre-existing Agent tool_use IDs in the spawn turn's history are not treated as new spawns."""
        old_agent_block = {
            "type": "tool_use",
            "id": "toolu_old",
            "name": "Agent",
            "input": {"task": "old task"},
        }
        turn0_msgs = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": [old_agent_block]},
        ]
        # Turn 1 re-surfaces the old tool_use (common-prefix diff edge case)
        turn1_msgs = turn0_msgs + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_old",
                        "content": "old result",
                    },
                ],
            },
            {"role": "user", "content": "turn 1"},
        ]
        records = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p0",
                    timestamp=_ts(0),
                    messages=turn0_msgs,
                    duration_ms=1000,
                )
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p1",
                    timestamp=_ts(5),
                    messages=turn1_msgs,
                    duration_ms=1000,
                )
            ),
        ]
        join_idx, saw_bg = _detect_join_turn_from_content(records, spawn_turn_index=0)
        assert join_idx is None
        assert saw_bg is False


# =========================================================================
# _find_join_turn_index tests
# =========================================================================


class TestFindJoinTurnIndex:
    def test_content_based_join_preferred(self) -> None:
        """Content-based detection takes priority over timing."""
        records = _make_content_records(result_at_turn=3, num_turns=5)
        child_records = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="c0",
                    agent_id="sub",
                    is_subagent=True,
                    timestamp=_ts(1),
                    duration_ms=1000,
                )
            )
        ]
        from aiperf.common.models import Conversation

        child_conv = Conversation(session_id="conflux_sub")
        children = [("sub", child_records, child_conv)]
        result = _find_join_turn_index(records, 0, children)
        assert result == 3

    def test_timing_fallback_when_no_content(self) -> None:
        """Falls back to timing when no content signals exist."""
        parent_records = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id=f"p{i}",
                    timestamp=_ts(i * 10),
                    duration_ms=2000,
                )
                | {"completed_at": _ts(i * 10 + 2)}
            )
            for i in range(4)
        ]
        # Child completes at t=15
        child_records = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="c0",
                    agent_id="sub",
                    is_subagent=True,
                    timestamp=_ts(1),
                    duration_ms=14000,
                )
                | {"completed_at": _ts(15)}
            )
        ]
        from aiperf.common.models import Conversation

        child_conv = Conversation(session_id="conflux_sub")
        children = [("sub", child_records, child_conv)]
        # Parent turn 2 starts at t=20, which is after child ends at t=15
        result = _find_join_turn_index(parent_records, 0, children)
        assert result == 2

    def test_returns_none_when_background_signal(self) -> None:
        """Returns None when content signals indicate background-only spawn."""
        records = _make_content_records(background_only=True, num_turns=4)
        child_records = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="c0",
                    agent_id="sub",
                    is_subagent=True,
                    timestamp=_ts(1),
                    duration_ms=500,
                )
            )
        ]
        from aiperf.common.models import Conversation

        child_conv = Conversation(session_id="conflux_sub")
        children = [("sub", child_records, child_conv)]
        result = _find_join_turn_index(records, 0, children)
        assert result is None


# =========================================================================
# _extract_notification_joins tests
# =========================================================================


class TestExtractNotificationJoins:
    def test_finds_task_notification(self) -> None:
        turn0_msgs = [{"role": "user", "content": "start"}]
        notification_text = (
            "Some preamble <task-notification>"
            "<tool-use-id>toolu_abc123</tool-use-id>"
            "</task-notification> more text"
        )
        turn1_msgs = turn0_msgs + [
            {"role": "assistant", "content": "working"},
            {"role": "user", "content": notification_text},
        ]
        records = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p0",
                    timestamp=_ts(0),
                    messages=turn0_msgs,
                    duration_ms=1000,
                )
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p1",
                    timestamp=_ts(5),
                    messages=turn1_msgs,
                    duration_ms=1000,
                )
            ),
        ]
        joins = _extract_notification_joins(records)
        assert joins == {"toolu_abc123": 1}

    def test_no_notifications(self) -> None:
        records = [
            ConfluxRecord.model_validate(
                _make_record(record_id="p0", timestamp=_ts(0), duration_ms=1000)
            ),
        ]
        joins = _extract_notification_joins(records)
        assert joins == {}

    def test_multiple_notifications_first_wins(self) -> None:
        """First occurrence of a tool_use_id is kept."""
        turn0_msgs = [{"role": "user", "content": "start"}]
        notif = (
            "<task-notification><tool-use-id>toolu_x</tool-use-id></task-notification>"
        )
        turn1_msgs = turn0_msgs + [{"role": "user", "content": notif}]
        turn2_msgs = turn1_msgs + [{"role": "user", "content": notif}]
        records = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p0",
                    timestamp=_ts(0),
                    messages=turn0_msgs,
                    duration_ms=1000,
                )
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p1",
                    timestamp=_ts(5),
                    messages=turn1_msgs,
                    duration_ms=1000,
                )
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p2",
                    timestamp=_ts(10),
                    messages=turn2_msgs,
                    duration_ms=1000,
                )
            ),
        ]
        joins = _extract_notification_joins(records)
        assert joins["toolu_x"] == 1

    def test_notification_in_list_content_block(self) -> None:
        """Handles notifications inside structured content blocks."""
        turn0_msgs = [{"role": "user", "content": "start"}]
        notif_block = {
            "type": "text",
            "text": "<task-notification><tool-use-id>toolu_y</tool-use-id></task-notification>",
        }
        turn1_msgs = turn0_msgs + [{"role": "user", "content": [notif_block]}]
        records = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p0",
                    timestamp=_ts(0),
                    messages=turn0_msgs,
                    duration_ms=1000,
                )
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p1",
                    timestamp=_ts(5),
                    messages=turn1_msgs,
                    duration_ms=1000,
                )
            ),
        ]
        joins = _extract_notification_joins(records)
        assert joins == {"toolu_y": 1}


# =========================================================================
# _build_spawn_tuid_to_agent_id tests
# =========================================================================


class TestBuildSpawnTuidToAgentId:
    def test_maps_async_agent_launched(self) -> None:
        turn0_msgs = [{"role": "user", "content": "start"}]
        agent_block = {
            "type": "tool_use",
            "id": "toolu_new_1",
            "name": "Agent",
            "input": {"task": "do work"},
        }
        result_block = {
            "type": "tool_result",
            "tool_use_id": "toolu_new_1",
            "content": "Async agent launched, agentId: agent_abc",
        }
        turn1_msgs = turn0_msgs + [
            {"role": "assistant", "content": [agent_block]},
            {"role": "user", "content": [result_block]},
        ]
        records = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p0",
                    timestamp=_ts(0),
                    messages=turn0_msgs,
                    duration_ms=1000,
                )
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p1",
                    timestamp=_ts(5),
                    messages=turn1_msgs,
                    duration_ms=1000,
                )
            ),
        ]
        mapping = _build_spawn_tuid_to_agent_id(records, spawn_turn_index=0)
        assert mapping == {"toolu_new_1": "agent_abc"}

    def test_ignores_pre_existing_agent_ids(self) -> None:
        old_agent = {
            "type": "tool_use",
            "id": "toolu_old",
            "name": "Agent",
            "input": {"task": "old"},
        }
        turn0_msgs = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": [old_agent]},
        ]
        # Turn 1: old agent completes (not a new spawn)
        result_block = {
            "type": "tool_result",
            "tool_use_id": "toolu_old",
            "content": "Async agent launched, agentId: old_agent",
        }
        turn1_msgs = turn0_msgs + [
            {"role": "user", "content": [result_block]},
        ]
        records = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p0",
                    timestamp=_ts(0),
                    messages=turn0_msgs,
                    duration_ms=1000,
                )
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p1",
                    timestamp=_ts(5),
                    messages=turn1_msgs,
                    duration_ms=1000,
                )
            ),
        ]
        mapping = _build_spawn_tuid_to_agent_id(records, spawn_turn_index=0)
        assert mapping == {}

    def test_spawn_at_last_turn_returns_empty(self) -> None:
        records = [
            ConfluxRecord.model_validate(
                _make_record(record_id="p0", timestamp=_ts(0), duration_ms=1000)
            ),
        ]
        mapping = _build_spawn_tuid_to_agent_id(records, spawn_turn_index=0)
        assert mapping == {}


# =========================================================================
# Orphan filtering tests
# =========================================================================


class TestOrphanFiltering:
    @pytest.fixture
    def no_orphan_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                conflux_include_utility_calls=False,
            ),
        )

    def test_orphans_excluded_when_disabled(self, tmp_path, no_orphan_config) -> None:
        """Orphan records are filtered out when conflux_include_utility_calls=False."""
        records = [
            _make_record(
                record_id="p0", agent_id="claude", is_subagent=False, timestamp=_ts(0)
            ),
            _make_record(record_id="orphan0", agent_id=None, timestamp=_ts(5)),
        ]
        path = _build_session_file(tmp_path, records)
        loader = ConfluxLoader(filename=path, user_config=no_orphan_config)
        data = loader.load_dataset()
        assert len(data) == 1
        assert "claude" in data

    def test_orphans_excluded_produce_no_child_conversations(
        self, tmp_path, no_orphan_config
    ) -> None:
        records = [
            _make_record(
                record_id="p0", agent_id="claude", is_subagent=False, timestamp=_ts(0)
            ),
            _make_record(record_id="orphan0", agent_id=None, timestamp=_ts(5)),
        ]
        path = _build_session_file(tmp_path, records)
        loader = ConfluxLoader(filename=path, user_config=no_orphan_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert conversations[0].agent_depth == 0
        assert conversations[0].subagent_spawns == []


# =========================================================================
# _find_spawn_point tier 2 (post-completion gap) tests
# =========================================================================


class TestFindSpawnPointPostCompletionGap:
    def test_gap_via_completed_at(self) -> None:
        """Child spawned in gap between turn 0 completing and turn 1 starting."""
        parent = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p0",
                    timestamp=_ts(0),
                    is_subagent=False,
                    duration_ms=2000,
                )
                | {"completed_at": _ts(2)}
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p1",
                    timestamp=_ts(10),
                    is_subagent=False,
                    duration_ms=2000,
                )
                | {"completed_at": _ts(12)}
            ),
        ]
        # Child at t=3 is in gap (t=2 to t=10)
        child = [
            ConfluxRecord.model_validate(
                _make_record(record_id="c0", timestamp=_ts(3), is_subagent=True)
            )
        ]
        assert ConfluxLoader._find_spawn_point(parent, child) == 0

    def test_gap_via_duration_ms(self) -> None:
        """Gap detection using duration_ms when completed_at is absent."""
        parent = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p0",
                    timestamp=_ts(0),
                    is_subagent=False,
                    duration_ms=2000,
                )
            ),
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p1",
                    timestamp=_ts(10),
                    is_subagent=False,
                    duration_ms=2000,
                )
            ),
        ]
        child = [
            ConfluxRecord.model_validate(
                _make_record(record_id="c0", timestamp=_ts(5), is_subagent=True)
            )
        ]
        assert ConfluxLoader._find_spawn_point(parent, child) == 0

    def test_gap_after_last_parent_turn(self) -> None:
        """Child spawned after the last parent turn completes (open-ended gap)."""
        parent = [
            ConfluxRecord.model_validate(
                _make_record(
                    record_id="p0",
                    timestamp=_ts(0),
                    is_subagent=False,
                    duration_ms=2000,
                )
                | {"completed_at": _ts(2)}
            ),
        ]
        child = [
            ConfluxRecord.model_validate(
                _make_record(record_id="c0", timestamp=_ts(5), is_subagent=True)
            )
        ]
        assert ConfluxLoader._find_spawn_point(parent, child) == 0


# =========================================================================
# Notification-based join splitting integration test
# =========================================================================


def _build_notification_join_session(tmp_path) -> str:
    """Build a session where async spawns complete via <task-notification>."""
    agent_tool_use_1 = {
        "type": "tool_use",
        "id": "toolu_async_1",
        "name": "Agent",
        "input": {"task": "research auth"},
    }
    agent_tool_use_2 = {
        "type": "tool_use",
        "id": "toolu_async_2",
        "name": "Agent",
        "input": {"task": "research db"},
    }
    async_result_1 = {
        "type": "tool_result",
        "tool_use_id": "toolu_async_1",
        "content": "Async agent launched, agentId: sub_a",
    }
    async_result_2 = {
        "type": "tool_result",
        "tool_use_id": "toolu_async_2",
        "content": "Async agent launched, agentId: sub_b",
    }

    turn0_msgs = [{"role": "user", "content": "turn 0"}]
    turn1_msgs = turn0_msgs + [
        {"role": "assistant", "content": [agent_tool_use_1, agent_tool_use_2]},
        {"role": "user", "content": [async_result_1, async_result_2]},
        {"role": "user", "content": "turn 1"},
    ]
    notif_a = (
        "<task-notification>"
        "<tool-use-id>toolu_async_1</tool-use-id>"
        "</task-notification>"
    )
    turn2_msgs = turn1_msgs + [
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": notif_a},
        {"role": "user", "content": "turn 2"},
    ]
    notif_b = (
        "<task-notification>"
        "<tool-use-id>toolu_async_2</tool-use-id>"
        "</task-notification>"
    )
    turn3_msgs = turn2_msgs + [
        {"role": "assistant", "content": "more work"},
        {"role": "user", "content": notif_b},
        {"role": "user", "content": "turn 3"},
    ]

    records = []
    parent_msgs = [turn0_msgs, turn1_msgs, turn2_msgs, turn3_msgs]
    for i, msgs in enumerate(parent_msgs):
        records.append(
            _make_record(
                record_id=f"p{i}",
                agent_id="claude",
                is_subagent=False,
                timestamp=_ts(i * 5),
                messages=msgs,
                duration_ms=2000,
            )
        )

    # Child A: spawned at t=1
    records.append(
        _make_record(
            record_id="ca0",
            agent_id="sub_a",
            is_subagent=True,
            timestamp=_ts(1),
            duration_ms=8000,
        )
    )

    # Child B: spawned at t=2
    records.append(
        _make_record(
            record_id="cb0",
            agent_id="sub_b",
            is_subagent=True,
            timestamp=_ts(2),
            duration_ms=12000,
        )
    )

    return _build_session_file(tmp_path, records)


class TestNotificationJoinIntegration:
    @pytest.fixture
    def default_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    def test_notification_splits_into_per_child_blocking_spawns(
        self, tmp_path, default_user_config
    ) -> None:
        """Async spawns with <task-notification> produce per-child blocking spawns."""
        path = _build_notification_join_session(tmp_path)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        parent = conversations[0]
        # Two children => two separate blocking spawns (not one grouped background)
        blocking = [s for s in parent.subagent_spawns if not s.is_background]
        assert len(blocking) == 2
        for spawn in blocking:
            assert len(spawn.child_conversation_ids) == 1

    def test_notification_join_prerequisites_on_correct_turns(
        self, tmp_path, default_user_config
    ) -> None:
        """Each notification-based join creates a prerequisite on the notification turn."""
        path = _build_notification_join_session(tmp_path)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        parent = conversations[0]
        # Turn 0: spawn_ids should be set
        assert len(parent.turns[0].subagent_spawn_ids) == 2
        # Turn 2: sub_a notification -> prerequisite
        prereqs_2 = parent.turns[2].prerequisites
        assert len(prereqs_2) == 1
        assert prereqs_2[0].kind == PrerequisiteKind.SPAWN_JOIN
        # Turn 3: sub_b notification -> prerequisite
        prereqs_3 = parent.turns[3].prerequisites
        assert len(prereqs_3) == 1
        assert prereqs_3[0].kind == PrerequisiteKind.SPAWN_JOIN
        # The two spawns should reference different spawn_ids
        assert prereqs_2[0].spawn_id != prereqs_3[0].spawn_id


class TestZeroAlignTimestamps:
    """Tests for zero-aligning timestamps in convert_to_conversations."""

    @pytest.fixture()
    def default_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    def test_single_conversation_timestamps_start_at_zero(
        self, tmp_path, default_user_config
    ) -> None:
        """All timestamps shift so the earliest becomes 0."""
        records = [
            _make_record(
                record_id=f"req_{i}",
                agent_id="claude",
                is_subagent=False,
                timestamp=_ts(i * 10),
            )
            for i in range(4)
        ]
        path = _build_session_file(tmp_path, records)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        timestamps = [t.timestamp for t in conversations[0].turns]
        assert timestamps[0] == 0.0
        assert timestamps[1] == 10_000.0
        assert timestamps[2] == 20_000.0
        assert timestamps[3] == 30_000.0

    def test_parent_and_children_all_zero_aligned(
        self, tmp_path, default_user_config
    ) -> None:
        """Parent and subagent timestamps are shifted by the same global minimum."""
        path = _build_team_session(tmp_path)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        all_timestamps = [
            t.timestamp
            for c in conversations
            for t in c.turns
            if t.timestamp is not None
        ]
        assert min(all_timestamps) == 0.0
        # Parent turn 0 is the global minimum (_ts(0)), so it should be 0
        parent = conversations[0]
        assert parent.turns[0].timestamp == 0.0

    def test_relative_spacing_preserved(self, tmp_path, default_user_config) -> None:
        """Inter-turn gaps are identical before and after alignment."""
        records = [
            _make_record(
                record_id=f"req_{i}",
                agent_id="claude",
                is_subagent=False,
                timestamp=_ts(i * 7),
            )
            for i in range(3)
        ]
        path = _build_session_file(tmp_path, records)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        ts = [t.timestamp for t in conversations[0].turns]
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        assert gaps == [7_000.0, 7_000.0]

    def test_child_earlier_than_parent_becomes_zero(
        self, tmp_path, default_user_config
    ) -> None:
        """When a child timestamp precedes the parent, the child becomes 0."""
        records = [
            _make_record(
                record_id="req_parent_0",
                agent_id="claude",
                is_subagent=False,
                timestamp=_ts(10),
                duration_ms=2000,
            ),
            _make_record(
                record_id="req_child_0",
                agent_id="sub_a",
                is_subagent=True,
                timestamp=_ts(5),
            ),
        ]
        path = _build_session_file(tmp_path, records)
        loader = ConfluxLoader(filename=path, user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        all_timestamps = [
            t.timestamp
            for c in conversations
            for t in c.turns
            if t.timestamp is not None
        ]
        assert min(all_timestamps) == 0.0
        # Child at _ts(5) is earliest -> becomes 0; parent at _ts(10) -> 5000
        child = next(c for c in conversations if c.agent_depth == 1)
        parent = next(c for c in conversations if c.agent_depth == 0)
        assert child.turns[0].timestamp == 0.0
        assert parent.turns[0].timestamp == 5_000.0

    def test_already_zero_aligned_is_noop(self) -> None:
        """If earliest timestamp is already 0, nothing changes."""
        from aiperf.common.models import Conversation, Turn

        conv = Conversation(session_id="test")
        conv.turns = [
            Turn(role="user", timestamp=0.0, max_tokens=100),
            Turn(role="user", timestamp=5_000.0, max_tokens=100),
        ]
        ConfluxLoader._zero_align_timestamps([conv])
        assert conv.turns[0].timestamp == 0.0
        assert conv.turns[1].timestamp == 5_000.0

    def test_empty_conversations_no_error(self) -> None:
        """Empty conversation list does not raise."""
        ConfluxLoader._zero_align_timestamps([])

    def test_single_turn_becomes_zero(self) -> None:
        """A single-turn session normalizes to timestamp 0."""
        from aiperf.common.models import Conversation, Turn

        conv = Conversation(session_id="test")
        conv.turns = [Turn(role="user", timestamp=999_999.0, max_tokens=100)]
        ConfluxLoader._zero_align_timestamps([conv])
        assert conv.turns[0].timestamp == 0.0


# =========================================================================
# Directory loading tests
# =========================================================================


class TestDirectoryCanLoad:
    """Test can_load with directories."""

    def test_directory_with_conflux_json(self, tmp_path):
        """Directory containing a valid Conflux JSON file is accepted."""
        f = tmp_path / "session1.json"
        f.write_bytes(orjson.dumps([_make_record()]))
        assert ConfluxLoader.can_load(filename=str(tmp_path))

    def test_empty_directory(self, tmp_path):
        """Directory with no JSON files is rejected."""
        assert not ConfluxLoader.can_load(filename=str(tmp_path))

    def test_directory_with_non_conflux_json(self, tmp_path):
        """Directory with non-Conflux JSON files is rejected."""
        f = tmp_path / "other.json"
        f.write_bytes(orjson.dumps({"key": "value"}))
        assert not ConfluxLoader.can_load(filename=str(tmp_path))

    def test_directory_with_mixed_files(self, tmp_path):
        """Directory with one valid and one non-JSON file is accepted."""
        (tmp_path / "session.json").write_bytes(orjson.dumps([_make_record()]))
        (tmp_path / "readme.txt").write_text("not json")
        assert ConfluxLoader.can_load(filename=str(tmp_path))


class TestDirectoryLoadDataset:
    """Test load_dataset and convert_to_conversations with directory input.

    Each file in a directory is an independent session (separate capture).
    Agent IDs are prefixed with ``f<idx>_`` to avoid cross-file collisions,
    and each file is zero-aligned independently.
    """

    @pytest.fixture
    def default_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                conflux_include_utility_calls=True,
            ),
        )

    def _write_session(self, tmp_path, filename, records):
        (tmp_path / filename).write_bytes(orjson.dumps(records))

    def test_loads_records_from_multiple_files(self, tmp_path, default_user_config):
        """Each file is an independent session with prefixed agent_ids."""
        # File 0: session with 3 turns
        self._write_session(
            tmp_path,
            "session_a.json",
            [
                _make_record(
                    record_id=f"req_a{i}",
                    agent_id="claude",
                    is_subagent=False,
                    timestamp=_ts(i * 5),
                )
                for i in range(3)
            ],
        )
        # File 1: separate session with 2 turns (same agent_id, different file)
        self._write_session(
            tmp_path,
            "session_b.json",
            [
                _make_record(
                    record_id=f"req_b{i}",
                    agent_id="claude",
                    is_subagent=False,
                    timestamp=_ts(100 + i * 5),
                )
                for i in range(2)
            ],
        )

        loader = ConfluxLoader(filename=str(tmp_path), user_config=default_user_config)
        data = loader.load_dataset()
        assert "f0_claude" in data
        assert "f1_claude" in data
        assert len(data["f0_claude"]) == 3
        assert len(data["f1_claude"]) == 2

    def test_converts_directory_to_independent_conversations(
        self, tmp_path, default_user_config
    ):
        """Each file produces its own parent+children conversations."""
        # File 0: parent + child session
        self._write_session(
            tmp_path,
            "session_a.json",
            [
                _make_record(
                    record_id=f"req_p{i}",
                    agent_id="claude",
                    is_subagent=False,
                    timestamp=_ts(i * 5),
                )
                for i in range(3)
            ]
            + [
                _make_record(
                    record_id=f"req_c{i}",
                    agent_id="sub_a",
                    is_subagent=True,
                    timestamp=_ts(5 + i * 3),
                )
                for i in range(2)
            ],
        )
        # File 1: standalone session
        self._write_session(
            tmp_path,
            "session_b.json",
            [
                _make_record(
                    record_id=f"req_x{i}",
                    agent_id="claude",
                    is_subagent=False,
                    timestamp=_ts(200 + i * 5),
                )
                for i in range(2)
            ],
        )

        loader = ConfluxLoader(filename=str(tmp_path), user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        # File 0: parent + child = 2 conversations
        # File 1: standalone parent = 1 conversation
        assert len(conversations) == 3

    def test_per_file_zero_alignment(self, tmp_path, default_user_config):
        """Each file's timestamps are zero-aligned independently."""
        # File 0: starts at t=1000s
        self._write_session(
            tmp_path,
            "early.json",
            [
                _make_record(
                    record_id=f"req_e{i}",
                    agent_id="claude",
                    is_subagent=False,
                    timestamp=_ts(1000 + i * 10),
                )
                for i in range(3)
            ],
        )
        # File 1: starts at t=5000s (completely different time origin)
        self._write_session(
            tmp_path,
            "late.json",
            [
                _make_record(
                    record_id=f"req_l{i}",
                    agent_id="claude",
                    is_subagent=False,
                    timestamp=_ts(5000 + i * 10),
                )
                for i in range(3)
            ],
        )

        loader = ConfluxLoader(filename=str(tmp_path), user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 2

        # Both files should start at timestamp 0 independently
        for conv in conversations:
            assert conv.turns[0].timestamp == 0.0

    def test_empty_directory_raises(self, tmp_path, default_user_config):
        """Loading from an empty directory raises FileNotFoundError."""
        loader = ConfluxLoader(filename=str(tmp_path), user_config=default_user_config)
        with pytest.raises(FileNotFoundError, match="No .json files found"):
            loader.load_dataset()

    def test_skips_non_json_files(self, tmp_path, default_user_config):
        """Non-JSON files in the directory are ignored."""
        self._write_session(
            tmp_path,
            "session.json",
            [
                _make_record(
                    record_id=f"req_{i}",
                    agent_id="claude",
                    is_subagent=False,
                    timestamp=_ts(i * 5),
                )
                for i in range(2)
            ],
        )
        (tmp_path / "notes.txt").write_text("not json")
        (tmp_path / "data.csv").write_text("a,b,c")

        loader = ConfluxLoader(filename=str(tmp_path), user_config=default_user_config)
        data = loader.load_dataset()
        assert len(data["f0_claude"]) == 2

    def test_single_file_directory(self, tmp_path, default_user_config):
        """Directory with one file behaves like a single session."""
        self._write_session(
            tmp_path,
            "all.json",
            [
                _make_record(
                    record_id=f"req_{i}",
                    agent_id="claude",
                    is_subagent=False,
                    timestamp=_ts(i * 5),
                )
                for i in range(4)
            ],
        )

        loader = ConfluxLoader(filename=str(tmp_path), user_config=default_user_config)
        data = loader.load_dataset()
        assert len(data["f0_claude"]) == 4

    def test_same_agent_ids_across_files_no_collision(
        self, tmp_path, default_user_config
    ):
        """Two files with identical agent_ids produce separate conversations."""
        for i, name in enumerate(["file_a.json", "file_b.json"]):
            self._write_session(
                tmp_path,
                name,
                [
                    _make_record(
                        record_id=f"req_{name}_{j}",
                        agent_id="claude",
                        is_subagent=False,
                        timestamp=_ts(i * 1000 + j * 5),
                    )
                    for j in range(3)
                ],
            )

        loader = ConfluxLoader(filename=str(tmp_path), user_config=default_user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 2
        session_ids = {c.session_id for c in conversations}
        assert len(session_ids) == 2
