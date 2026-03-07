# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.dataset.loader.session_step_replay import (
    SessionStepReplayDatasetLoader,
)
from aiperf.plugin.enums import DatasetSamplingStrategy

SAMPLE_DATA = {
    "session_a": [
        {
            "candidate_prompts": ["prompt A1", "prompt A2"],
            "expected_output_tokens": 512,
            "step": 1,
        },
        {
            "candidate_prompts": ["prompt B1"],
            "expected_output_tokens": 256,
            "step": 2,
        },
    ],
    "session_b": [
        {
            "candidate_prompts": ["prompt C1"],
            "expected_output_tokens": 100,
        },
    ],
}


@pytest.fixture
def session_json(tmp_path) -> Path:
    """Write sample session data to a temporary JSON file."""
    path = tmp_path / "sessions.json"
    path.write_bytes(orjson.dumps(SAMPLE_DATA))
    return path


@pytest.fixture
def loader(session_json, default_user_config) -> SessionStepReplayDatasetLoader:
    return SessionStepReplayDatasetLoader(
        filename=str(session_json), user_config=default_user_config
    )


class TestCanLoad:
    def test_valid_json_file(self, session_json):
        assert SessionStepReplayDatasetLoader.can_load(filename=session_json) is True

    @pytest.mark.parametrize(
        "suffix,content",
        [
            param(".jsonl", b'{"text": "hello"}\n', id="jsonl_extension"),
            param(".txt", b"hello", id="txt_extension"),
        ],
    )
    def test_wrong_extension(self, tmp_path, suffix, content):
        path = tmp_path / f"data{suffix}"
        path.write_bytes(content)
        assert SessionStepReplayDatasetLoader.can_load(filename=path) is False

    def test_no_filename(self):
        assert SessionStepReplayDatasetLoader.can_load(data={"text": "hi"}) is False

    def test_empty_dict(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_bytes(orjson.dumps({}))
        assert SessionStepReplayDatasetLoader.can_load(filename=path) is False

    def test_missing_candidate_prompts(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_bytes(orjson.dumps({"s1": [{"other_field": 1}]}))
        assert SessionStepReplayDatasetLoader.can_load(filename=path) is False

    def test_non_list_value(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_bytes(orjson.dumps({"s1": "not a list"}))
        assert SessionStepReplayDatasetLoader.can_load(filename=path) is False


class TestDefaultContextMode:
    def test_returns_standalone(self):
        assert (
            SessionStepReplayDatasetLoader.get_default_context_mode()
            == ConversationContextMode.STANDALONE
        )

    def test_preferred_sampling_strategy(self):
        assert (
            SessionStepReplayDatasetLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SHUFFLE
        )


class TestLoadAndConvert:
    def test_load_dataset(self, loader):
        data = loader.load_dataset()
        assert set(data.keys()) == {"session_a", "session_b"}
        assert len(data["session_a"]) == 2
        assert len(data["session_b"]) == 1

    def test_convert_to_conversations(self, loader):
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2

        by_session = {c.session_id: c for c in conversations}

        conv_a = by_session["session_a"]
        assert len(conv_a.turns) == 2
        assert conv_a.turns[0].prompt_candidates == ["prompt A1", "prompt A2"]
        assert conv_a.turns[0].max_tokens == 512
        assert conv_a.turns[1].prompt_candidates == ["prompt B1"]
        assert conv_a.turns[1].max_tokens == 256

        conv_b = by_session["session_b"]
        assert len(conv_b.turns) == 1
        assert conv_b.turns[0].max_tokens == 100

    def test_skips_non_list_sessions(self, tmp_path, default_user_config):
        bad_data = {
            "good_session": [
                {"candidate_prompts": ["p1"], "expected_output_tokens": 100}
            ],
            "bad_session": "not a list",
        }
        path = tmp_path / "mixed.json"
        path.write_bytes(orjson.dumps(bad_data))
        loader = SessionStepReplayDatasetLoader(
            filename=str(path), user_config=default_user_config
        )
        data = loader.load_dataset()
        assert "good_session" in data
        assert "bad_session" not in data

    def test_empty_candidate_prompts_raises(self, tmp_path, default_user_config):
        bad_data = {"s1": [{"candidate_prompts": [], "expected_output_tokens": 100}]}
        path = tmp_path / "empty_candidates.json"
        path.write_bytes(orjson.dumps(bad_data))
        loader = SessionStepReplayDatasetLoader(
            filename=str(path), user_config=default_user_config
        )
        with pytest.raises(Exception, match="candidate_prompts"):
            loader.load_dataset()
