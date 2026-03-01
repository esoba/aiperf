# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RawPayloadDatasetLoader."""

import orjson
import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.dataset.loader.raw_payload import RawPayloadDatasetLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

CHAT_PAYLOAD = {
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "gpt-4",
    "max_tokens": 100,
}

CHAT_PAYLOAD_2 = {
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "How are you?"},
    ],
    "model": "gpt-4",
    "temperature": 0.7,
}


def _write_jsonl(path, records: list[dict]) -> None:
    with open(path, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record) + b"\n")


# =========================================================================
# TestCanLoad
# =========================================================================


class TestCanLoad:
    def test_valid_chat_payload(self):
        assert RawPayloadDatasetLoader.can_load(data=CHAT_PAYLOAD) is True

    def test_rejects_none(self):
        assert RawPayloadDatasetLoader.can_load(data=None) is False

    def test_rejects_no_messages(self):
        assert RawPayloadDatasetLoader.can_load(data={"model": "gpt-4"}) is False

    def test_rejects_messages_not_list(self):
        assert (
            RawPayloadDatasetLoader.can_load(data={"messages": "not a list"}) is False
        )

    def test_rejects_agentic_trajectory(self):
        data = {
            "conversation_id": "conv-1",
            "conversation_idx": 0,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        assert RawPayloadDatasetLoader.can_load(data=data) is False

    def test_rejects_inputs_json_format(self):
        data = {
            "data": [{"session_id": "s1", "payloads": [CHAT_PAYLOAD]}],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        assert RawPayloadDatasetLoader.can_load(data=data) is False

    def test_directory_with_valid_jsonl(self, tmp_path):
        jsonl = tmp_path / "conv1.jsonl"
        _write_jsonl(jsonl, [CHAT_PAYLOAD])
        assert RawPayloadDatasetLoader.can_load(filename=str(tmp_path)) is True

    def test_directory_empty(self, tmp_path):
        assert RawPayloadDatasetLoader.can_load(filename=str(tmp_path)) is False

    def test_directory_no_messages_key(self, tmp_path):
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(jsonl, [{"text": "not a payload"}])
        assert RawPayloadDatasetLoader.can_load(filename=str(tmp_path)) is False

    def test_no_data_no_filename(self):
        assert RawPayloadDatasetLoader.can_load() is False

    def test_preferred_sampling_strategy(self):
        assert (
            RawPayloadDatasetLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )


# =========================================================================
# TestLoadDataset
# =========================================================================


class TestLoadDataset:
    @pytest.fixture
    def user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    def test_single_line(self, tmp_path, user_config):
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(jsonl, [CHAT_PAYLOAD])
        loader = RawPayloadDatasetLoader(filename=str(jsonl), user_config=user_config)
        result = loader.load_dataset()

        assert len(result) == 1
        session_id = next(iter(result))
        assert len(result[session_id]) == 1
        assert result[session_id][0].payload == CHAT_PAYLOAD

    def test_multiple_lines(self, tmp_path, user_config):
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(jsonl, [CHAT_PAYLOAD, CHAT_PAYLOAD_2])
        loader = RawPayloadDatasetLoader(filename=str(jsonl), user_config=user_config)
        result = loader.load_dataset()

        assert len(result) == 2

    def test_empty_lines_skipped(self, tmp_path, user_config):
        jsonl = tmp_path / "data.jsonl"
        with open(jsonl, "wb") as f:
            f.write(orjson.dumps(CHAT_PAYLOAD))
            f.write(b"\n\n\n")
            f.write(orjson.dumps(CHAT_PAYLOAD_2))
            f.write(b"\n")

        loader = RawPayloadDatasetLoader(filename=str(jsonl), user_config=user_config)
        result = loader.load_dataset()
        assert len(result) == 2


# =========================================================================
# TestLoadDatasetDirectory
# =========================================================================


class TestLoadDatasetDirectory:
    @pytest.fixture
    def user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    def test_single_file_multi_turn(self, tmp_path, user_config):
        data_dir = tmp_path / "payloads"
        data_dir.mkdir()
        _write_jsonl(data_dir / "conv1.jsonl", [CHAT_PAYLOAD, CHAT_PAYLOAD_2])

        loader = RawPayloadDatasetLoader(
            filename=str(data_dir), user_config=user_config
        )
        result = loader.load_dataset()

        assert len(result) == 1
        session_id = next(iter(result))
        assert len(result[session_id]) == 2
        assert result[session_id][0].payload == CHAT_PAYLOAD
        assert result[session_id][1].payload == CHAT_PAYLOAD_2

    def test_multiple_files_multiple_conversations(self, tmp_path, user_config):
        data_dir = tmp_path / "payloads"
        data_dir.mkdir()
        _write_jsonl(data_dir / "conv1.jsonl", [CHAT_PAYLOAD])
        _write_jsonl(data_dir / "conv2.jsonl", [CHAT_PAYLOAD_2])

        loader = RawPayloadDatasetLoader(
            filename=str(data_dir), user_config=user_config
        )
        result = loader.load_dataset()

        assert len(result) == 2

    def test_empty_files_skipped(self, tmp_path, user_config):
        data_dir = tmp_path / "payloads"
        data_dir.mkdir()
        (data_dir / "empty.jsonl").write_text("")
        _write_jsonl(data_dir / "valid.jsonl", [CHAT_PAYLOAD])

        loader = RawPayloadDatasetLoader(
            filename=str(data_dir), user_config=user_config
        )
        result = loader.load_dataset()

        assert len(result) == 1

    def test_non_jsonl_files_ignored(self, tmp_path, user_config):
        data_dir = tmp_path / "payloads"
        data_dir.mkdir()
        (data_dir / "readme.txt").write_text("ignore me")
        _write_jsonl(data_dir / "conv.jsonl", [CHAT_PAYLOAD])

        loader = RawPayloadDatasetLoader(
            filename=str(data_dir), user_config=user_config
        )
        result = loader.load_dataset()

        assert len(result) == 1

    def test_files_processed_in_sorted_order(self, tmp_path, user_config):
        data_dir = tmp_path / "payloads"
        data_dir.mkdir()
        _write_jsonl(data_dir / "b_conv.jsonl", [CHAT_PAYLOAD_2])
        _write_jsonl(data_dir / "a_conv.jsonl", [CHAT_PAYLOAD])

        loader = RawPayloadDatasetLoader(
            filename=str(data_dir), user_config=user_config
        )
        result = loader.load_dataset()
        payloads_in_order = [v[0].payload for v in result.values()]

        assert payloads_in_order[0] == CHAT_PAYLOAD
        assert payloads_in_order[1] == CHAT_PAYLOAD_2


# =========================================================================
# TestConvertToConversations
# =========================================================================


class TestConvertToConversations:
    @pytest.fixture
    def user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            ),
        )

    def test_single_payload_produces_single_turn(self, tmp_path, user_config):
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(jsonl, [CHAT_PAYLOAD])
        loader = RawPayloadDatasetLoader(filename=str(jsonl), user_config=user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conv = conversations[0]
        assert len(conv.turns) == 1
        assert conv.turns[0].raw_payload == CHAT_PAYLOAD
        assert conv.turns[0].role == "user"

    def test_multiple_payloads_produce_separate_conversations(
        self, tmp_path, user_config
    ):
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(jsonl, [CHAT_PAYLOAD, CHAT_PAYLOAD_2])
        loader = RawPayloadDatasetLoader(filename=str(jsonl), user_config=user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        assert conversations[0].turns[0].raw_payload == CHAT_PAYLOAD
        assert conversations[1].turns[0].raw_payload == CHAT_PAYLOAD_2

    def test_payload_preserved_exactly(self, tmp_path, user_config):
        payload = {
            "messages": [{"role": "user", "content": "test"}],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "stream": True,
            "temperature": 0.5,
        }
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(jsonl, [payload])
        loader = RawPayloadDatasetLoader(filename=str(jsonl), user_config=user_config)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert conversations[0].turns[0].raw_payload == payload

    def test_directory_multi_turn_conversation(self, tmp_path, user_config):
        data_dir = tmp_path / "payloads"
        data_dir.mkdir()
        _write_jsonl(data_dir / "session.jsonl", [CHAT_PAYLOAD, CHAT_PAYLOAD_2])

        loader = RawPayloadDatasetLoader(
            filename=str(data_dir), user_config=user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conv = conversations[0]
        assert len(conv.turns) == 2
        assert conv.turns[0].raw_payload == CHAT_PAYLOAD
        assert conv.turns[1].raw_payload == CHAT_PAYLOAD_2
        assert all(t.role == "user" for t in conv.turns)
