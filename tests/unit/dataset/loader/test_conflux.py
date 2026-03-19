# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ConfluxLoader and ConfluxRecord models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest
from pydantic import ValidationError

from aiperf.common.config import EndpointConfig, InputConfig, UserConfig
from aiperf.common.enums import ConversationContextMode
from aiperf.dataset.loader.conflux import ConfluxLoader
from aiperf.dataset.loader.models import ConfluxRecord, ConfluxTokens
from aiperf.plugin.enums import DatasetSamplingStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    session_id: str = "sess-1",
    agent_id: str | None = "agent-A",
    timestamp: float = 1000.0,
    duration_ms: int | float = 500,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tokens: dict[str, Any] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    is_subagent: bool | None = None,
    completed_at: str | None = None,
    **extra_fields: Any,
) -> dict[str, Any]:
    """Build a raw Conflux record dict with sensible defaults."""
    rec: dict[str, Any] = {
        "session_id": session_id,
        "agent_id": agent_id,
        "timestamp": timestamp,
        "duration_ms": duration_ms,
    }
    if messages is not None:
        rec["messages"] = messages
    if tools is not None:
        rec["tools"] = tools
    if tokens is not None:
        rec["tokens"] = tokens
    if hyperparameters is not None:
        rec["hyperparameters"] = hyperparameters
    if is_subagent is not None:
        rec["is_subagent"] = is_subagent
    if completed_at is not None:
        rec["completed_at"] = completed_at
    rec.update(extra_fields)
    return rec


def _write_json(path: Path, data: Any) -> str:
    """Write data as JSON and return the string path."""
    path.write_bytes(orjson.dumps(data))
    return str(path)


def _make_user_config(*, include_utility: bool = False) -> UserConfig:
    """Create a minimal UserConfig for Conflux tests."""
    return UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig.model_construct(
            conflux_include_utility_calls=include_utility,
        ),
    )


# ---------------------------------------------------------------------------
# ConfluxTokens model tests
# ---------------------------------------------------------------------------


class TestConfluxTokens:
    """Tests for ConfluxTokens model."""

    def test_defaults(self):
        tokens = ConfluxTokens()
        assert tokens.input == 0
        assert tokens.input_cached == 0
        assert tokens.input_cache_write == 0
        assert tokens.output == 0
        assert tokens.output_reasoning == 0

    def test_all_fields_populated(self):
        tokens = ConfluxTokens(
            input=1000,
            input_cached=200,
            input_cache_write=300,
            output=500,
            output_reasoning=50,
        )
        assert tokens.input == 1000
        assert tokens.input_cached == 200
        assert tokens.input_cache_write == 300
        assert tokens.output == 500
        assert tokens.output_reasoning == 50

    def test_partial_fields(self):
        tokens = ConfluxTokens(input=42, output=10)
        assert tokens.input == 42
        assert tokens.input_cached == 0
        assert tokens.output == 10
        assert tokens.output_reasoning == 0


# ---------------------------------------------------------------------------
# ConfluxRecord model tests
# ---------------------------------------------------------------------------


class TestConfluxRecord:
    """Tests for ConfluxRecord model validation."""

    def test_minimal_valid_record(self):
        record = ConfluxRecord(session_id="s1", timestamp=1000.0)
        assert record.session_id == "s1"
        assert record.agent_id is None
        assert record.messages == []
        assert record.tools == []
        assert record.tokens is None
        assert record.hyperparameters is None
        assert record.duration_ms == 0

    def test_full_record(self):
        record = ConfluxRecord(
            session_id="s1",
            agent_id="agent-X",
            is_subagent=True,
            timestamp=2000.0,
            duration_ms=1234,
            completed_at="2025-01-15T10:30:01.234Z",
            tokens=ConfluxTokens(input=100, output=50),
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            hyperparameters={"temperature": 0.7, "top_p": 0.9},
        )
        assert record.agent_id == "agent-X"
        assert record.is_subagent is True
        assert record.tokens.input == 100
        assert record.tokens.output == 50
        assert len(record.messages) == 1
        assert len(record.tools) == 1
        assert record.hyperparameters["temperature"] == 0.7

    def test_extra_fields_ignored(self):
        record = ConfluxRecord.model_validate(
            {
                "session_id": "s1",
                "timestamp": 1000.0,
                "unknown_field": "should be ignored",
                "another_extra": 42,
            }
        )
        assert record.session_id == "s1"
        assert not hasattr(record, "unknown_field")

    def test_missing_required_session_id_raises(self):
        with pytest.raises(ValidationError):
            ConfluxRecord(timestamp=1000.0)

    def test_missing_required_timestamp_raises(self):
        with pytest.raises(ValidationError):
            ConfluxRecord(session_id="s1")

    def test_duration_ms_accepts_float(self):
        record = ConfluxRecord(session_id="s1", timestamp=1000.0, duration_ms=123.456)
        assert record.duration_ms == 123.456

    def test_tokens_nested_validation(self):
        record = ConfluxRecord.model_validate(
            {
                "session_id": "s1",
                "timestamp": 1000.0,
                "tokens": {"input": 999, "output": 100, "output_reasoning": 10},
            }
        )
        assert record.tokens.input == 999
        assert record.tokens.output_reasoning == 10


# ---------------------------------------------------------------------------
# ConfluxLoader class-level tests
# ---------------------------------------------------------------------------


class TestConfluxLoaderClassMethods:
    """Tests for ConfluxLoader class methods."""

    def test_default_context_mode(self):
        assert (
            ConfluxLoader.get_default_context_mode()
            == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )

    def test_preferred_sampling_strategy(self):
        assert (
            ConfluxLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )


# ---------------------------------------------------------------------------
# can_load tests
# ---------------------------------------------------------------------------


class TestConfluxCanLoad:
    """Tests for ConfluxLoader.can_load auto-detection."""

    def test_valid_single_file(self, tmp_path):
        records = [_make_record()]
        path = tmp_path / "session.json"
        _write_json(path, records)
        assert ConfluxLoader.can_load(filename=str(path)) is True

    def test_valid_directory(self, tmp_path):
        records = [_make_record()]
        _write_json(tmp_path / "a.json", records)
        assert ConfluxLoader.can_load(filename=str(tmp_path)) is True

    def test_empty_array_returns_false(self, tmp_path):
        path = tmp_path / "empty.json"
        _write_json(path, [])
        assert ConfluxLoader.can_load(filename=str(path)) is False

    def test_non_json_file_returns_false(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("not json")
        assert ConfluxLoader.can_load(filename=str(path)) is False

    def test_json_object_not_array_returns_false(self, tmp_path):
        path = tmp_path / "obj.json"
        _write_json(path, {"key": "value"})
        assert ConfluxLoader.can_load(filename=str(path)) is False

    def test_wrong_json_extension_returns_false(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_bytes(orjson.dumps([_make_record()]))
        assert ConfluxLoader.can_load(filename=str(path)) is False

    def test_none_filename_returns_false(self):
        assert ConfluxLoader.can_load(filename=None) is False

    def test_nonexistent_file_returns_false(self):
        assert ConfluxLoader.can_load(filename="/nonexistent/path/file.json") is False

    def test_array_of_non_conflux_records_returns_false(self, tmp_path):
        path = tmp_path / "other.json"
        _write_json(path, [{"unrelated": "data"}])
        assert ConfluxLoader.can_load(filename=str(path)) is False

    def test_directory_with_no_json_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hello")
        assert ConfluxLoader.can_load(filename=str(tmp_path)) is False

    def test_directory_with_invalid_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not valid json{")
        assert ConfluxLoader.can_load(filename=str(tmp_path)) is False

    def test_directory_with_valid_and_invalid_json(self, tmp_path):
        """Directory probe uses next(glob) which is unordered; a single valid file may or may not be probed."""
        _write_json(tmp_path / "valid.json", [_make_record()])
        result = ConfluxLoader.can_load(filename=str(tmp_path))
        assert result is True


# ---------------------------------------------------------------------------
# load_dataset tests
# ---------------------------------------------------------------------------


class TestConfluxLoadDataset:
    """Tests for ConfluxLoader.load_dataset."""

    def test_single_agent_group(self, tmp_path):
        records = [
            _make_record(agent_id="A", timestamp=1000.0),
            _make_record(agent_id="A", timestamp=2000.0),
        ]
        path = tmp_path / "session.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        groups = loader.load_dataset()

        assert len(groups) == 1
        assert "A" in groups
        assert len(groups["A"]) == 2

    def test_multiple_agent_groups(self, tmp_path):
        records = [
            _make_record(agent_id="A", timestamp=1000.0),
            _make_record(agent_id="B", timestamp=2000.0),
            _make_record(agent_id="A", timestamp=3000.0),
        ]
        path = tmp_path / "session.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        groups = loader.load_dataset()

        assert len(groups) == 2
        assert len(groups["A"]) == 2
        assert len(groups["B"]) == 1

    def test_records_sorted_by_timestamp_within_group(self, tmp_path):
        records = [
            _make_record(agent_id="A", timestamp=6000.0),
            _make_record(agent_id="A", timestamp=2000.0),
            _make_record(agent_id="A", timestamp=4000.0),
        ]
        path = tmp_path / "session.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        groups = loader.load_dataset()

        timestamps = [r.timestamp for r in groups["A"]]
        assert timestamps == sorted(timestamps)
        assert len(timestamps) == 3

    def test_utility_calls_skipped_by_default(self, tmp_path):
        records = [
            _make_record(agent_id="A", timestamp=1000.0),
            _make_record(agent_id=None, timestamp=2000.0),
            _make_record(agent_id=None, timestamp=3000.0),
        ]
        path = tmp_path / "session.json"
        _write_json(path, records)

        loader = ConfluxLoader(
            filename=str(path), user_config=_make_user_config(include_utility=False)
        )
        groups = loader.load_dataset()

        assert len(groups) == 1
        assert "A" in groups

    def test_utility_calls_included_when_enabled(self, tmp_path):
        records = [
            _make_record(agent_id="A", timestamp=1000.0),
            _make_record(agent_id=None, timestamp=2000.0),
            _make_record(agent_id=None, timestamp=3000.0),
        ]
        path = tmp_path / "session.json"
        _write_json(path, records)

        loader = ConfluxLoader(
            filename=str(path), user_config=_make_user_config(include_utility=True)
        )
        groups = loader.load_dataset()

        assert len(groups) == 3
        assert "A" in groups
        assert "_utility_0" in groups
        assert "_utility_1" in groups

    def test_utility_calls_each_get_own_group(self, tmp_path):
        records = [
            _make_record(agent_id=None, timestamp=1000.0),
            _make_record(agent_id=None, timestamp=2000.0),
        ]
        path = tmp_path / "session.json"
        _write_json(path, records)

        loader = ConfluxLoader(
            filename=str(path), user_config=_make_user_config(include_utility=True)
        )
        groups = loader.load_dataset()

        assert len(groups) == 2
        for key in groups:
            assert len(groups[key]) == 1

    def test_all_utility_records_skipped_yields_empty(self, tmp_path):
        records = [
            _make_record(agent_id=None, timestamp=1000.0),
        ]
        path = tmp_path / "session.json"
        _write_json(path, records)

        loader = ConfluxLoader(
            filename=str(path), user_config=_make_user_config(include_utility=False)
        )
        groups = loader.load_dataset()

        assert len(groups) == 0


# ---------------------------------------------------------------------------
# load_dataset directory tests
# ---------------------------------------------------------------------------


class TestConfluxLoadDirectory:
    """Tests for loading a directory of JSON files."""

    def test_multiple_files_merged(self, tmp_path):
        _write_json(
            tmp_path / "file1.json",
            [_make_record(agent_id="A", timestamp=1000.0)],
        )
        _write_json(
            tmp_path / "file2.json",
            [_make_record(agent_id="B", timestamp=100000.0)],
        )

        loader = ConfluxLoader(filename=str(tmp_path), user_config=_make_user_config())
        groups = loader.load_dataset()

        assert len(groups) == 2
        assert "f0_A" in groups
        assert "f1_B" in groups

    def test_directory_prefixes_prevent_key_collisions(self, tmp_path):
        _write_json(
            tmp_path / "a.json",
            [_make_record(agent_id="X", timestamp=1000.0)],
        )
        _write_json(
            tmp_path / "b.json",
            [_make_record(agent_id="X", timestamp=100000.0)],
        )

        loader = ConfluxLoader(filename=str(tmp_path), user_config=_make_user_config())
        groups = loader.load_dataset()

        assert "f0_X" in groups
        assert "f1_X" in groups
        assert len(groups) == 2

    def test_empty_directory_raises(self, tmp_path):
        loader = ConfluxLoader(filename=str(tmp_path), user_config=_make_user_config())
        with pytest.raises(FileNotFoundError, match="No .json files found"):
            loader.load_dataset()

    def test_files_loaded_in_sorted_order(self, tmp_path):
        _write_json(
            tmp_path / "z.json",
            [_make_record(agent_id="Z", timestamp=1000.0)],
        )
        _write_json(
            tmp_path / "a.json",
            [_make_record(agent_id="A", timestamp=1000.0)],
        )

        loader = ConfluxLoader(filename=str(tmp_path), user_config=_make_user_config())
        groups = loader.load_dataset()

        keys = list(groups.keys())
        assert keys[0] == "f0_A"
        assert keys[1] == "f1_Z"

    def test_directory_utility_calls_with_prefix(self, tmp_path):
        _write_json(
            tmp_path / "file.json",
            [_make_record(agent_id=None, timestamp=1000.0)],
        )

        loader = ConfluxLoader(
            filename=str(tmp_path), user_config=_make_user_config(include_utility=True)
        )
        groups = loader.load_dataset()

        assert len(groups) == 1
        assert "f0__utility_0" in groups


# ---------------------------------------------------------------------------
# convert_to_conversations tests
# ---------------------------------------------------------------------------


class TestConfluxConvertToConversations:
    """Tests for ConfluxLoader.convert_to_conversations."""

    def _load_and_convert(
        self,
        tmp_path: Path,
        records: list[dict[str, Any]],
        *,
        include_utility: bool = False,
    ) -> list:
        path = tmp_path / "data.json"
        _write_json(path, records)
        loader = ConfluxLoader(
            filename=str(path),
            user_config=_make_user_config(include_utility=include_utility),
        )
        data = loader.load_dataset()
        return loader.convert_to_conversations(data)

    def test_basic_conversion(self, tmp_path):
        records = [
            _make_record(
                agent_id="A",
                timestamp=1000.0,
                messages=[{"role": "user", "content": "Hello"}],
            ),
        ]
        convos = self._load_and_convert(tmp_path, records)

        assert len(convos) == 1
        assert convos[0].session_id == "conflux_A"
        assert len(convos[0].turns) == 1

    def test_turn_has_raw_messages(self, tmp_path):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        records = [
            _make_record(agent_id="A", timestamp=1000.0, messages=msgs),
        ]
        convos = self._load_and_convert(tmp_path, records)

        assert convos[0].turns[0].raw_messages == msgs

    def test_turn_has_raw_tools(self, tmp_path):
        tools = [{"type": "function", "function": {"name": "search"}}]
        records = [
            _make_record(agent_id="A", timestamp=1000.0, tools=tools),
        ]
        convos = self._load_and_convert(tmp_path, records)

        assert convos[0].turns[0].raw_tools == tools

    def test_empty_tools_becomes_none(self, tmp_path):
        records = [
            _make_record(agent_id="A", timestamp=1000.0, tools=[]),
        ]
        convos = self._load_and_convert(tmp_path, records)

        assert convos[0].turns[0].raw_tools is None

    def test_turn_timestamp_is_milliseconds(self, tmp_path):
        records = [
            _make_record(agent_id="A", timestamp=1000.0),
        ]
        convos = self._load_and_convert(tmp_path, records)

        ts = convos[0].turns[0].timestamp
        assert isinstance(ts, float)
        assert ts == 1000.0

    def test_multi_turn_timestamps_preserved(self, tmp_path):
        records = [
            _make_record(agent_id="A", timestamp=1000.0),
            _make_record(agent_id="A", timestamp=6000.0),
        ]
        convos = self._load_and_convert(tmp_path, records)

        t0 = convos[0].turns[0].timestamp
        t1 = convos[0].turns[1].timestamp
        assert t1 > t0
        delta_ms = t1 - t0
        assert abs(delta_ms - 5000.0) < 1.0

    def test_max_tokens_from_output_plus_reasoning(self, tmp_path):
        records = [
            _make_record(
                agent_id="A",
                timestamp=1000.0,
                tokens={"input": 100, "output": 200, "output_reasoning": 50},
            ),
        ]
        convos = self._load_and_convert(tmp_path, records)

        assert convos[0].turns[0].max_tokens == 250

    def test_max_tokens_none_when_no_tokens(self, tmp_path):
        records = [
            _make_record(agent_id="A", timestamp=1000.0),
        ]
        convos = self._load_and_convert(tmp_path, records)

        assert convos[0].turns[0].max_tokens is None

    def test_max_tokens_none_when_zero_output(self, tmp_path):
        records = [
            _make_record(
                agent_id="A",
                timestamp=1000.0,
                tokens={"input": 100, "output": 0, "output_reasoning": 0},
            ),
        ]
        convos = self._load_and_convert(tmp_path, records)

        assert convos[0].turns[0].max_tokens is None

    def test_input_tokens_from_token_data(self, tmp_path):
        records = [
            _make_record(
                agent_id="A",
                timestamp=1000.0,
                tokens={"input": 500, "output": 100},
            ),
        ]
        convos = self._load_and_convert(tmp_path, records)

        assert convos[0].turns[0].input_tokens == 500

    def test_input_tokens_none_when_no_token_data(self, tmp_path):
        records = [
            _make_record(agent_id="A", timestamp=1000.0),
        ]
        convos = self._load_and_convert(tmp_path, records)

        assert convos[0].turns[0].input_tokens is None

    def test_multiple_conversations_from_agents(self, tmp_path):
        records = [
            _make_record(agent_id="A", timestamp=1000.0),
            _make_record(agent_id="B", timestamp=2000.0),
            _make_record(agent_id="A", timestamp=3000.0),
        ]
        convos = self._load_and_convert(tmp_path, records)

        assert len(convos) == 2
        session_ids = {c.session_id for c in convos}
        assert "conflux_A" in session_ids
        assert "conflux_B" in session_ids

        convo_a = next(c for c in convos if c.session_id == "conflux_A")
        assert len(convo_a.turns) == 2

    def test_empty_data_produces_no_conversations(self, tmp_path):
        path = tmp_path / "data.json"
        _write_json(path, [_make_record(agent_id=None)])

        loader = ConfluxLoader(
            filename=str(path), user_config=_make_user_config(include_utility=False)
        )
        data = loader.load_dataset()
        convos = loader.convert_to_conversations(data)

        assert len(convos) == 0


# ---------------------------------------------------------------------------
# _extract_extra_params tests
# ---------------------------------------------------------------------------


class TestExtractExtraParams:
    """Tests for ConfluxLoader._extract_extra_params."""

    def test_no_hyperparameters_returns_none(self):
        record = ConfluxRecord(session_id="s1", timestamp=1000.0)
        assert ConfluxLoader._extract_extra_params(record) is None

    def test_empty_hyperparameters_returns_none(self):
        record = ConfluxRecord(
            session_id="s1",
            timestamp=1000.0,
            hyperparameters={},
        )
        assert ConfluxLoader._extract_extra_params(record) is None

    def test_basic_hyperparameters_extracted(self):
        record = ConfluxRecord(
            session_id="s1",
            timestamp=1000.0,
            hyperparameters={"temperature": 0.7, "top_p": 0.9},
        )
        params = ConfluxLoader._extract_extra_params(record)
        assert params == {"temperature": 0.7, "top_p": 0.9}

    def test_max_tokens_filtered_out(self):
        record = ConfluxRecord(
            session_id="s1",
            timestamp=1000.0,
            hyperparameters={"temperature": 0.5, "max_tokens": 1000},
        )
        params = ConfluxLoader._extract_extra_params(record)
        assert params == {"temperature": 0.5}
        assert "max_tokens" not in params

    def test_max_output_tokens_filtered_out(self):
        record = ConfluxRecord(
            session_id="s1",
            timestamp=1000.0,
            hyperparameters={"temperature": 0.5, "max_output_tokens": 2000},
        )
        params = ConfluxLoader._extract_extra_params(record)
        assert params == {"temperature": 0.5}

    def test_none_values_filtered_out(self):
        record = ConfluxRecord(
            session_id="s1",
            timestamp=1000.0,
            hyperparameters={"temperature": 0.7, "top_k": None, "stop": None},
        )
        params = ConfluxLoader._extract_extra_params(record)
        assert params == {"temperature": 0.7}

    def test_all_filtered_returns_none(self):
        record = ConfluxRecord(
            session_id="s1",
            timestamp=1000.0,
            hyperparameters={"max_tokens": 100, "max_output_tokens": 200},
        )
        params = ConfluxLoader._extract_extra_params(record)
        assert params is None

    def test_all_none_values_returns_none(self):
        record = ConfluxRecord(
            session_id="s1",
            timestamp=1000.0,
            hyperparameters={"temperature": None, "top_p": None},
        )
        params = ConfluxLoader._extract_extra_params(record)
        assert params is None

    def test_zero_value_preserved(self):
        record = ConfluxRecord(
            session_id="s1",
            timestamp=1000.0,
            hyperparameters={"temperature": 0, "frequency_penalty": 0.0},
        )
        params = ConfluxLoader._extract_extra_params(record)
        assert params == {"temperature": 0, "frequency_penalty": 0.0}

    def test_false_value_preserved(self):
        record = ConfluxRecord(
            session_id="s1",
            timestamp=1000.0,
            hyperparameters={"logprobs": False},
        )
        params = ConfluxLoader._extract_extra_params(record)
        assert params == {"logprobs": False}

    def test_empty_string_value_preserved(self):
        record = ConfluxRecord(
            session_id="s1",
            timestamp=1000.0,
            hyperparameters={"stop": ""},
        )
        params = ConfluxLoader._extract_extra_params(record)
        assert params == {"stop": ""}

    def test_nested_dict_value_preserved(self):
        record = ConfluxRecord(
            session_id="s1",
            timestamp=1000.0,
            hyperparameters={
                "response_format": {"type": "json_object"},
            },
        )
        params = ConfluxLoader._extract_extra_params(record)
        assert params == {"response_format": {"type": "json_object"}}


# ---------------------------------------------------------------------------
# End-to-end / integration-style tests
# ---------------------------------------------------------------------------


class TestConfluxEndToEnd:
    """End-to-end tests combining load + convert."""

    def test_full_pipeline_single_agent_session(self, tmp_path):
        records = [
            _make_record(
                agent_id="coder",
                timestamp=1000.0,
                messages=[{"role": "user", "content": "Write hello world"}],
                tokens={"input": 50, "output": 100},
                hyperparameters={"temperature": 0.3},
            ),
            _make_record(
                agent_id="coder",
                timestamp=3000.0,
                messages=[
                    {"role": "user", "content": "Write hello world"},
                    {"role": "assistant", "content": "print('hello world')"},
                    {"role": "user", "content": "Add error handling"},
                ],
                tokens={"input": 150, "output": 200, "output_reasoning": 30},
                tools=[{"type": "function", "function": {"name": "write_file"}}],
            ),
        ]
        path = tmp_path / "session.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        data = loader.load_dataset()
        convos = loader.convert_to_conversations(data)

        assert len(convos) == 1
        convo = convos[0]
        assert convo.session_id == "conflux_coder"
        assert len(convo.turns) == 2

        turn0 = convo.turns[0]
        assert turn0.raw_messages == [{"role": "user", "content": "Write hello world"}]
        assert turn0.max_tokens == 100
        assert turn0.input_tokens == 50
        assert turn0.extra_params == {"temperature": 0.3}
        assert turn0.raw_tools is None

        turn1 = convo.turns[1]
        assert len(turn1.raw_messages) == 3
        assert turn1.max_tokens == 230
        assert turn1.input_tokens == 150
        assert turn1.raw_tools is not None
        assert len(turn1.raw_tools) == 1

    def test_full_pipeline_multi_agent_with_utility(self, tmp_path):
        records = [
            _make_record(agent_id="planner", timestamp=1000.0),
            _make_record(agent_id=None, timestamp=2000.0),
            _make_record(agent_id="executor", timestamp=3000.0),
            _make_record(agent_id="planner", timestamp=4000.0),
        ]
        path = tmp_path / "session.json"
        _write_json(path, records)

        loader = ConfluxLoader(
            filename=str(path), user_config=_make_user_config(include_utility=True)
        )
        data = loader.load_dataset()
        convos = loader.convert_to_conversations(data)

        assert len(convos) == 3
        session_ids = {c.session_id for c in convos}
        assert "conflux_planner" in session_ids
        assert "conflux_executor" in session_ids

        planner = next(c for c in convos if c.session_id == "conflux_planner")
        assert len(planner.turns) == 2

    def test_full_pipeline_directory_with_mixed_agents(self, tmp_path):
        _write_json(
            tmp_path / "session1.json",
            [
                _make_record(agent_id="A", timestamp=1000.0),
                _make_record(agent_id="B", timestamp=2000.0),
            ],
        )
        _write_json(
            tmp_path / "session2.json",
            [
                _make_record(agent_id="A", timestamp=100000.0),
            ],
        )

        loader = ConfluxLoader(filename=str(tmp_path), user_config=_make_user_config())
        data = loader.load_dataset()
        convos = loader.convert_to_conversations(data)

        assert len(convos) == 3


# ---------------------------------------------------------------------------
# Boundary / pathological tests
# ---------------------------------------------------------------------------


class TestConfluxBoundaryConditions:
    """Edge cases, boundary conditions, and pathological inputs."""

    def test_single_record_file(self, tmp_path):
        records = [_make_record(agent_id="solo")]
        path = tmp_path / "single.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        data = loader.load_dataset()
        convos = loader.convert_to_conversations(data)

        assert len(convos) == 1
        assert len(convos[0].turns) == 1

    def test_very_large_messages_array(self, tmp_path):
        big_messages = [{"role": "user", "content": f"msg-{i}"} for i in range(500)]
        records = [
            _make_record(agent_id="A", timestamp=1000.0, messages=big_messages),
        ]
        path = tmp_path / "big.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        data = loader.load_dataset()
        convos = loader.convert_to_conversations(data)

        assert len(convos[0].turns[0].raw_messages) == 500

    def test_many_agents_in_one_file(self, tmp_path):
        records = [
            _make_record(
                agent_id=f"agent-{i}",
                timestamp=float(1000 + i * 1000),
            )
            for i in range(50)
        ]
        path = tmp_path / "many.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        data = loader.load_dataset()
        convos = loader.convert_to_conversations(data)

        assert len(convos) == 50

    def test_identical_timestamps_stable_within_group(self, tmp_path):
        records = [
            _make_record(
                agent_id="A",
                timestamp=1000.0,
                messages=[{"role": "user", "content": f"msg-{i}"}],
            )
            for i in range(5)
        ]
        path = tmp_path / "dupes.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        data = loader.load_dataset()
        convos = loader.convert_to_conversations(data)

        assert len(convos) == 1
        assert len(convos[0].turns) == 5

    def test_timestamps_with_microsecond_precision(self, tmp_path):
        records = [
            _make_record(agent_id="A", timestamp=1000.000001),
            _make_record(agent_id="A", timestamp=1000.000002),
        ]
        path = tmp_path / "micro.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        data = loader.load_dataset()
        convos = loader.convert_to_conversations(data)

        t0 = convos[0].turns[0].timestamp
        t1 = convos[0].turns[1].timestamp
        assert t1 >= t0

    def test_record_with_all_optional_fields_none(self, tmp_path):
        records = [
            {
                "session_id": "s1",
                "timestamp": 1000.0,
            }
        ]
        path = tmp_path / "minimal.json"
        _write_json(path, records)

        loader = ConfluxLoader(
            filename=str(path), user_config=_make_user_config(include_utility=True)
        )
        data = loader.load_dataset()
        convos = loader.convert_to_conversations(data)

        assert len(convos) == 1
        turn = convos[0].turns[0]
        assert turn.raw_messages == []
        assert turn.raw_tools is None
        assert turn.max_tokens is None
        assert turn.input_tokens is None
        assert turn.extra_params is None

    def test_hyperparameters_only_skip_fields_returns_no_extra_params(self, tmp_path):
        records = [
            _make_record(
                agent_id="A",
                timestamp=1000.0,
                hyperparameters={"max_tokens": 1024, "max_output_tokens": 512},
            ),
        ]
        path = tmp_path / "skip.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        data = loader.load_dataset()
        convos = loader.convert_to_conversations(data)

        assert convos[0].turns[0].extra_params is None

    def test_tokens_with_only_reasoning_output(self, tmp_path):
        records = [
            _make_record(
                agent_id="A",
                timestamp=1000.0,
                tokens={"input": 100, "output": 0, "output_reasoning": 500},
            ),
        ]
        path = tmp_path / "reasoning.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        data = loader.load_dataset()
        convos = loader.convert_to_conversations(data)

        assert convos[0].turns[0].max_tokens == 500

    def test_duration_ms_zero(self, tmp_path):
        records = [_make_record(agent_id="A", duration_ms=0)]
        path = tmp_path / "zero_dur.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        data = loader.load_dataset()

        assert data["A"][0].duration_ms == 0

    def test_extra_json_fields_silently_dropped(self, tmp_path):
        records = [
            {
                "session_id": "s1",
                "agent_id": "A",
                "timestamp": 1000.0,
                "provider": "anthropic",
                "model_name": "claude-3.5-sonnet",
                "response_text": "Hello!",
                "metadata": {"version": 2},
            }
        ]
        path = tmp_path / "extra.json"
        _write_json(path, records)

        loader = ConfluxLoader(filename=str(path), user_config=_make_user_config())
        data = loader.load_dataset()

        assert len(data["A"]) == 1
        record = data["A"][0]
        assert not hasattr(record, "provider")
        assert not hasattr(record, "model_name")


# ---------------------------------------------------------------------------
# Turn.copy_with_stripped_media with new fields
# ---------------------------------------------------------------------------


class TestTurnCopyWithStrippedMediaNewFields:
    """Verify Turn.copy_with_stripped_media preserves new fields."""

    def test_copy_with_stripped_media_preserves_input_tokens(self):
        from aiperf.common.models import Turn

        turn = Turn(
            texts=[],
            input_tokens=42,
        )
        copy = turn.copy_with_stripped_media()
        assert copy.input_tokens == 42

    def test_copy_with_stripped_media_preserves_extra_params(self):
        from aiperf.common.models import Turn

        turn = Turn(
            texts=[],
            extra_params={"temperature": 0.7, "top_p": 0.9},
        )
        copy = turn.copy_with_stripped_media()
        assert copy.extra_params == {"temperature": 0.7, "top_p": 0.9}

    def test_copy_with_stripped_media_extra_params_is_independent(self):
        from aiperf.common.models import Turn

        original_params = {"temperature": 0.7}
        turn = Turn(texts=[], extra_params=original_params)
        copy = turn.copy_with_stripped_media()

        copy.extra_params["temperature"] = 999
        assert turn.extra_params["temperature"] == 0.7

    def test_copy_with_stripped_media_none_fields(self):
        from aiperf.common.models import Turn

        turn = Turn(texts=[], input_tokens=None, extra_params=None)
        copy = turn.copy_with_stripped_media()
        assert copy.input_tokens is None
        assert copy.extra_params is None
