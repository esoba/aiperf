# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for InputsJsonPayloadLoader."""

import orjson
import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

PAYLOAD_1 = {
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "gpt-4",
}

PAYLOAD_2 = {
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "How are you?"},
    ],
    "model": "gpt-4",
}

INPUTS_FILE = {
    "data": [
        {"session_id": "s1", "payloads": [PAYLOAD_1]},
        {"session_id": "s2", "payloads": [PAYLOAD_1, PAYLOAD_2]},
    ]
}


def _write_json(path, content: dict) -> None:
    path.write_bytes(orjson.dumps(content))


# =========================================================================
# TestCanLoad
# =========================================================================


class TestCanLoad:
    def test_valid_inputs_file_data(self):
        assert InputsJsonPayloadLoader.can_load(data=INPUTS_FILE) is True

    def test_rejects_none(self):
        assert InputsJsonPayloadLoader.can_load(data=None) is False

    def test_rejects_empty_data_list(self):
        assert InputsJsonPayloadLoader.can_load(data={"data": []}) is False

    def test_rejects_data_not_list(self):
        assert InputsJsonPayloadLoader.can_load(data={"data": "not a list"}) is False

    def test_rejects_missing_payloads_key(self):
        data = {"data": [{"session_id": "s1", "messages": []}]}
        assert InputsJsonPayloadLoader.can_load(data=data) is False

    def test_rejects_raw_payload_format(self):
        data = {"messages": [{"role": "user", "content": "Hello"}]}
        assert InputsJsonPayloadLoader.can_load(data=data) is False

    def test_file_probe_valid(self, tmp_path):
        json_file = tmp_path / "inputs.json"
        _write_json(json_file, INPUTS_FILE)
        assert InputsJsonPayloadLoader.can_load(filename=str(json_file)) is True

    def test_file_probe_wrong_extension(self, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_bytes(orjson.dumps(INPUTS_FILE))
        assert InputsJsonPayloadLoader.can_load(filename=str(jsonl_file)) is False

    def test_file_probe_invalid_content(self, tmp_path):
        json_file = tmp_path / "bad.json"
        json_file.write_text("not json")
        assert InputsJsonPayloadLoader.can_load(filename=str(json_file)) is False

    def test_file_probe_nonexistent(self, tmp_path):
        assert (
            InputsJsonPayloadLoader.can_load(filename=str(tmp_path / "nope.json"))
            is False
        )

    def test_preferred_sampling_strategy(self):
        assert (
            InputsJsonPayloadLoader.get_preferred_sampling_strategy()
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

    def test_single_session(self, tmp_path, user_config):
        inputs = {"data": [{"session_id": "s1", "payloads": [PAYLOAD_1]}]}
        json_file = tmp_path / "inputs.json"
        _write_json(json_file, inputs)

        loader = InputsJsonPayloadLoader(
            filename=str(json_file), user_config=user_config
        )
        result = loader.load_dataset()

        assert "s1" in result
        assert len(result["s1"]) == 1
        assert result["s1"][0].session_id == "s1"
        assert result["s1"][0].payloads == [PAYLOAD_1]

    def test_multi_session(self, tmp_path, user_config):
        json_file = tmp_path / "inputs.json"
        _write_json(json_file, INPUTS_FILE)

        loader = InputsJsonPayloadLoader(
            filename=str(json_file), user_config=user_config
        )
        result = loader.load_dataset()

        assert len(result) == 2
        assert "s1" in result
        assert "s2" in result
        assert len(result["s2"][0].payloads) == 2

    def test_multi_turn_payloads_preserved(self, tmp_path, user_config):
        inputs = {
            "data": [
                {"session_id": "s1", "payloads": [PAYLOAD_1, PAYLOAD_2]},
            ]
        }
        json_file = tmp_path / "inputs.json"
        _write_json(json_file, inputs)

        loader = InputsJsonPayloadLoader(
            filename=str(json_file), user_config=user_config
        )
        result = loader.load_dataset()

        assert result["s1"][0].payloads == [PAYLOAD_1, PAYLOAD_2]


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

    def test_single_turn_session(self, tmp_path, user_config):
        inputs = {"data": [{"session_id": "s1", "payloads": [PAYLOAD_1]}]}
        json_file = tmp_path / "inputs.json"
        _write_json(json_file, inputs)

        loader = InputsJsonPayloadLoader(
            filename=str(json_file), user_config=user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conv = conversations[0]
        assert conv.session_id == "s1"
        assert len(conv.turns) == 1
        assert conv.turns[0].raw_payload == PAYLOAD_1
        assert conv.turns[0].role == "user"

    def test_multi_turn_session(self, tmp_path, user_config):
        inputs = {
            "data": [
                {"session_id": "s1", "payloads": [PAYLOAD_1, PAYLOAD_2]},
            ]
        }
        json_file = tmp_path / "inputs.json"
        _write_json(json_file, inputs)

        loader = InputsJsonPayloadLoader(
            filename=str(json_file), user_config=user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conv = conversations[0]
        assert len(conv.turns) == 2
        assert conv.turns[0].raw_payload == PAYLOAD_1
        assert conv.turns[1].raw_payload == PAYLOAD_2

    def test_multiple_sessions_produce_separate_conversations(
        self, tmp_path, user_config
    ):
        json_file = tmp_path / "inputs.json"
        _write_json(json_file, INPUTS_FILE)

        loader = InputsJsonPayloadLoader(
            filename=str(json_file), user_config=user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        session_ids = {c.session_id for c in conversations}
        assert session_ids == {"s1", "s2"}

    def test_payload_preserved_exactly(self, tmp_path, user_config):
        payload = {
            "messages": [{"role": "user", "content": "test"}],
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "stream": True,
            "tools": [{"name": "read_file", "description": "Read"}],
        }
        inputs = {"data": [{"session_id": "s1", "payloads": [payload]}]}
        json_file = tmp_path / "inputs.json"
        _write_json(json_file, inputs)

        loader = InputsJsonPayloadLoader(
            filename=str(json_file), user_config=user_config
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert conversations[0].turns[0].raw_payload == payload
