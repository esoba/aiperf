# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AgenticTrajectoryLoader."""

import orjson
import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.dataset.loader.agentic_trajectory import (
    AgenticTrajectoryLoader,
    _extract_system_message,
    _is_prefix,
    _strip_system,
)
from aiperf.dataset.loader.models import AgenticTrajectoryRecord
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.workers.session_manager import UserSession

# =========================================================================
# Test data builders
# =========================================================================

SYSTEM_MSG = {"role": "system", "content": "You are a helpful assistant."}
SYSTEM_MSG_2 = {"role": "system", "content": "Follow these rules carefully."}
USER_MSG = {"role": "user", "content": "Hello"}
ASSISTANT_MSG = {
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi"}],
    "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "read_file"}}],
}
TOOL_RESULT_MSG = {"role": "tool", "tool_call_id": "t1", "content": "file contents"}

TOOLS = [
    {"name": "read_file", "description": "Read a file"},
    {"name": "write_file", "description": "Write a file"},
]


def _make_trajectory_record(
    conv_id: str, idx: int, messages: list[dict], tools: list[dict] | None = None
) -> dict:
    record = {
        "conversation_id": conv_id,
        "conversation_idx": idx,
        "messages": messages,
    }
    if tools:
        record["tools"] = tools
    return record


def _write_jsonl(path, records: list[dict]) -> None:
    with open(path, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record) + b"\n")


# =========================================================================
# TestCanLoad
# =========================================================================


class TestCanLoad:
    def test_valid_data(self):
        data = {
            "conversation_id": "conv-1",
            "conversation_idx": 0,
            "messages": [SYSTEM_MSG, USER_MSG],
        }
        assert AgenticTrajectoryLoader.can_load(data=data) is True

    def test_missing_conversation_id(self):
        data = {"conversation_idx": 0, "messages": [SYSTEM_MSG]}
        assert AgenticTrajectoryLoader.can_load(data=data) is False

    def test_missing_conversation_idx(self):
        data = {"conversation_id": "c1", "messages": [SYSTEM_MSG]}
        assert AgenticTrajectoryLoader.can_load(data=data) is False

    def test_missing_messages(self):
        data = {"conversation_id": "c1", "conversation_idx": 0}
        assert AgenticTrajectoryLoader.can_load(data=data) is False

    def test_wrong_types(self):
        data = {"conversation_id": 123, "conversation_idx": "zero", "messages": "bad"}
        assert AgenticTrajectoryLoader.can_load(data=data) is False

    def test_file_probe_valid(self, tmp_path):
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(
            jsonl,
            [_make_trajectory_record("c1", 0, [SYSTEM_MSG, USER_MSG])],
        )
        assert AgenticTrajectoryLoader.can_load(filename=str(jsonl)) is True

    def test_file_probe_corrupt_non_json_returns_false(self, tmp_path):
        """Corrupt non-JSON content in a .jsonl triggers the except-Exception catch-all."""
        jsonl = tmp_path / "corrupt.jsonl"
        jsonl.write_bytes(b"\x80\x81\x82 not valid json\n")
        assert AgenticTrajectoryLoader.can_load(filename=str(jsonl)) is False

    def test_file_probe_only_blank_lines_returns_false(self, tmp_path):
        """A .jsonl file with only blank lines has no parseable record."""
        jsonl = tmp_path / "blanks.jsonl"
        jsonl.write_text("\n\n   \n\n")
        assert AgenticTrajectoryLoader.can_load(filename=str(jsonl)) is False

    def test_file_probe_non_jsonl(self, tmp_path):
        txt = tmp_path / "data.txt"
        txt.write_text("hello")
        assert AgenticTrajectoryLoader.can_load(filename=str(txt)) is False

    def test_file_probe_directory(self, tmp_path):
        assert AgenticTrajectoryLoader.can_load(filename=str(tmp_path)) is False

    def test_no_data_no_filename(self):
        assert AgenticTrajectoryLoader.can_load() is False

    def test_preferred_sampling_strategy(self):
        assert (
            AgenticTrajectoryLoader.get_preferred_sampling_strategy()
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

    def test_single_conversation(self, tmp_path, user_config):
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(
            jsonl,
            [
                _make_trajectory_record("c1", 0, [SYSTEM_MSG, USER_MSG]),
                _make_trajectory_record(
                    "c1", 1, [SYSTEM_MSG, USER_MSG, ASSISTANT_MSG, TOOL_RESULT_MSG]
                ),
            ],
        )
        loader = AgenticTrajectoryLoader(filename=str(jsonl), user_config=user_config)
        result = loader.load_dataset()

        assert "c1" in result
        assert len(result["c1"]) == 2
        assert result["c1"][0].conversation_idx == 0
        assert result["c1"][1].conversation_idx == 1

    def test_multi_conversation_grouping(self, tmp_path, user_config):
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(
            jsonl,
            [
                _make_trajectory_record("c1", 0, [SYSTEM_MSG, USER_MSG]),
                _make_trajectory_record("c2", 0, [SYSTEM_MSG, USER_MSG]),
                _make_trajectory_record("c1", 1, [SYSTEM_MSG, USER_MSG, ASSISTANT_MSG]),
            ],
        )
        loader = AgenticTrajectoryLoader(filename=str(jsonl), user_config=user_config)
        result = loader.load_dataset()

        assert len(result) == 2
        assert len(result["c1"]) == 2
        assert len(result["c2"]) == 1

    def test_sorts_by_conversation_idx(self, tmp_path, user_config):
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(
            jsonl,
            [
                _make_trajectory_record("c1", 2, [SYSTEM_MSG]),
                _make_trajectory_record("c1", 0, [SYSTEM_MSG]),
                _make_trajectory_record("c1", 1, [SYSTEM_MSG]),
            ],
        )
        loader = AgenticTrajectoryLoader(filename=str(jsonl), user_config=user_config)
        result = loader.load_dataset()

        indices = [r.conversation_idx for r in result["c1"]]
        assert indices == [0, 1, 2]

    def test_empty_lines_skipped(self, tmp_path, user_config):
        jsonl = tmp_path / "data.jsonl"
        with open(jsonl, "wb") as f:
            f.write(
                orjson.dumps(_make_trajectory_record("c1", 0, [SYSTEM_MSG, USER_MSG]))
            )
            f.write(b"\n\n\n")
            f.write(
                orjson.dumps(
                    _make_trajectory_record(
                        "c1", 1, [SYSTEM_MSG, USER_MSG, ASSISTANT_MSG]
                    )
                )
            )
            f.write(b"\n")

        loader = AgenticTrajectoryLoader(filename=str(jsonl), user_config=user_config)
        result = loader.load_dataset()
        assert len(result["c1"]) == 2

    def test_tools_preserved(self, tmp_path, user_config):
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(
            jsonl,
            [_make_trajectory_record("c1", 0, [SYSTEM_MSG, USER_MSG], tools=TOOLS)],
        )
        loader = AgenticTrajectoryLoader(filename=str(jsonl), user_config=user_config)
        result = loader.load_dataset()
        assert result["c1"][0].tools == TOOLS

    def test_missing_tools_key_defaults_to_empty_list(self, tmp_path, user_config):
        """When a JSONL record omits the 'tools' key, tools defaults to []."""
        jsonl = tmp_path / "data.jsonl"
        # Write raw dict without tools key (not using _make_trajectory_record
        # which conditionally includes it)
        raw = {
            "conversation_id": "c1",
            "conversation_idx": 0,
            "messages": [SYSTEM_MSG, USER_MSG],
        }
        with open(jsonl, "wb") as f:
            f.write(orjson.dumps(raw) + b"\n")

        loader = AgenticTrajectoryLoader(filename=str(jsonl), user_config=user_config)
        result = loader.load_dataset()
        assert result["c1"][0].tools == []


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

    def test_system_message_extraction(self, user_config):
        data = {
            "c1": [
                AgenticTrajectoryRecord(
                    conversation_id="c1",
                    conversation_idx=0,
                    messages=[SYSTEM_MSG, USER_MSG],
                ),
            ]
        }
        loader = AgenticTrajectoryLoader(
            filename="dummy.jsonl", user_config=user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        assert conversations[0].system_message == "You are a helpful assistant."

    def test_tools_extraction(self, user_config):
        data = {
            "c1": [
                AgenticTrajectoryRecord(
                    conversation_id="c1",
                    conversation_idx=0,
                    messages=[SYSTEM_MSG, USER_MSG],
                    tools=[],
                ),
                AgenticTrajectoryRecord(
                    conversation_id="c1",
                    conversation_idx=1,
                    messages=[SYSTEM_MSG, USER_MSG, ASSISTANT_MSG],
                    tools=TOOLS,
                ),
            ]
        }
        loader = AgenticTrajectoryLoader(
            filename="dummy.jsonl", user_config=user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert conversations[0].tools == TOOLS

    def test_cumulative_turns_produce_deltas(self, user_config):
        """Consecutive cumulative turns produce small deltas without replaces_history."""
        messages_t0 = [SYSTEM_MSG, USER_MSG]
        messages_t1 = [SYSTEM_MSG, USER_MSG, ASSISTANT_MSG, TOOL_RESULT_MSG]

        data = {
            "c1": [
                AgenticTrajectoryRecord(
                    conversation_id="c1",
                    conversation_idx=0,
                    messages=messages_t0,
                ),
                AgenticTrajectoryRecord(
                    conversation_id="c1",
                    conversation_idx=1,
                    messages=messages_t1,
                ),
            ]
        }
        loader = AgenticTrajectoryLoader(
            filename="dummy.jsonl", user_config=user_config
        )
        conversations = loader.convert_to_conversations(data)

        conv = conversations[0]
        assert len(conv.turns) == 2

        # Turn 0: first turn always gets replaces_history, full non-system messages
        assert conv.turns[0].replaces_history is True
        assert conv.turns[0].raw_messages == [USER_MSG]

        # Turn 1: cumulative, so delta only (new messages since turn 0)
        assert conv.turns[1].replaces_history is False
        assert conv.turns[1].raw_messages == [ASSISTANT_MSG, TOOL_RESULT_MSG]

    def test_context_break_triggers_replaces_history(self, user_config):
        """Non-cumulative turn (context break) sets replaces_history=True."""
        compacted_user = {"role": "user", "content": "Compacted context"}
        messages_t0 = [SYSTEM_MSG, USER_MSG]
        messages_t1 = [SYSTEM_MSG, USER_MSG, ASSISTANT_MSG, TOOL_RESULT_MSG]
        # Turn 2 breaks cumulative invariant (compacted history)
        messages_t2 = [SYSTEM_MSG, compacted_user]

        data = {
            "c1": [
                AgenticTrajectoryRecord(
                    conversation_id="c1", conversation_idx=0, messages=messages_t0
                ),
                AgenticTrajectoryRecord(
                    conversation_id="c1", conversation_idx=1, messages=messages_t1
                ),
                AgenticTrajectoryRecord(
                    conversation_id="c1", conversation_idx=2, messages=messages_t2
                ),
            ]
        }
        loader = AgenticTrajectoryLoader(
            filename="dummy.jsonl", user_config=user_config
        )
        conversations = loader.convert_to_conversations(data)

        conv = conversations[0]
        assert len(conv.turns) == 3

        assert conv.turns[0].replaces_history is True
        assert conv.turns[1].replaces_history is False
        # Context break: full history replacement
        assert conv.turns[2].replaces_history is True
        assert conv.turns[2].raw_messages == [compacted_user]

    def test_delta_resumes_after_context_break(self, user_config):
        """After a context break, subsequent cumulative turns resume delta mode."""
        compacted_user = {"role": "user", "content": "Compacted"}
        new_assistant = {"role": "assistant", "content": "Response"}
        messages_t0 = [SYSTEM_MSG, USER_MSG]
        # Context break
        messages_t1 = [SYSTEM_MSG, compacted_user]
        # Cumulative from t1
        messages_t2 = [SYSTEM_MSG, compacted_user, new_assistant]

        data = {
            "c1": [
                AgenticTrajectoryRecord(
                    conversation_id="c1", conversation_idx=0, messages=messages_t0
                ),
                AgenticTrajectoryRecord(
                    conversation_id="c1", conversation_idx=1, messages=messages_t1
                ),
                AgenticTrajectoryRecord(
                    conversation_id="c1", conversation_idx=2, messages=messages_t2
                ),
            ]
        }
        loader = AgenticTrajectoryLoader(
            filename="dummy.jsonl", user_config=user_config
        )
        conversations = loader.convert_to_conversations(data)

        conv = conversations[0]
        assert conv.turns[0].replaces_history is True  # first turn
        assert conv.turns[1].replaces_history is True  # context break
        assert conv.turns[2].replaces_history is False  # delta from new baseline
        assert conv.turns[2].raw_messages == [new_assistant]

    def test_single_turn_conversation(self, user_config):
        data = {
            "c1": [
                AgenticTrajectoryRecord(
                    conversation_id="c1",
                    conversation_idx=0,
                    messages=[SYSTEM_MSG, USER_MSG],
                ),
            ]
        }
        loader = AgenticTrajectoryLoader(
            filename="dummy.jsonl", user_config=user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        assert len(conversations[0].turns) == 1
        assert conversations[0].turns[0].raw_messages == [USER_MSG]

    def test_missing_tools(self, user_config):
        data = {
            "c1": [
                AgenticTrajectoryRecord(
                    conversation_id="c1",
                    conversation_idx=0,
                    messages=[SYSTEM_MSG, USER_MSG],
                ),
            ]
        }
        loader = AgenticTrajectoryLoader(
            filename="dummy.jsonl", user_config=user_config
        )
        conversations = loader.convert_to_conversations(data)
        assert conversations[0].tools is None

    def test_multiple_leading_system_messages(self, user_config):
        """Multiple leading system messages are joined into Conversation.system_message."""
        data = {
            "c1": [
                AgenticTrajectoryRecord(
                    conversation_id="c1",
                    conversation_idx=0,
                    messages=[SYSTEM_MSG, SYSTEM_MSG_2, USER_MSG],
                ),
            ]
        }
        loader = AgenticTrajectoryLoader(
            filename="dummy.jsonl", user_config=user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert conversations[0].system_message == (
            "You are a helpful assistant.\nFollow these rules carefully."
        )
        assert conversations[0].turns[0].raw_messages == [USER_MSG]

    def test_no_system_message(self, user_config):
        data = {
            "c1": [
                AgenticTrajectoryRecord(
                    conversation_id="c1",
                    conversation_idx=0,
                    messages=[USER_MSG],
                ),
            ]
        }
        loader = AgenticTrajectoryLoader(
            filename="dummy.jsonl", user_config=user_config
        )
        conversations = loader.convert_to_conversations(data)
        assert conversations[0].system_message is None
        assert conversations[0].turns[0].raw_messages == [USER_MSG]

    def test_empty_records_skipped(self, user_config):
        data = {"c1": []}
        loader = AgenticTrajectoryLoader(
            filename="dummy.jsonl", user_config=user_config
        )
        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 0

    def test_multi_conversation_data_dict(self, user_config):
        """Multiple conversation groups produce independent Conversation objects."""
        user2 = {"role": "user", "content": "Goodbye"}
        data = {
            "c1": [
                AgenticTrajectoryRecord(
                    conversation_id="c1",
                    conversation_idx=0,
                    messages=[SYSTEM_MSG, USER_MSG],
                ),
            ],
            "c2": [
                AgenticTrajectoryRecord(
                    conversation_id="c2",
                    conversation_idx=0,
                    messages=[SYSTEM_MSG_2, user2],
                ),
            ],
        }
        loader = AgenticTrajectoryLoader(
            filename="dummy.jsonl", user_config=user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        sys_messages = {c.system_message for c in conversations}
        assert "You are a helpful assistant." in sys_messages
        assert "Follow these rules carefully." in sys_messages

        # Each conversation has its own turn content
        all_raw = [c.turns[0].raw_messages for c in conversations]
        assert [USER_MSG] in all_raw
        assert [user2] in all_raw


# =========================================================================
# TestAdvanceTurnRawMessages
# =========================================================================


class TestAdvanceTurnRawMessages:
    def _make_session(self, conversation: Conversation) -> UserSession:
        return UserSession(
            x_correlation_id="test",
            num_turns=len(conversation.turns),
            conversation=conversation,
            turn_list=[],
        )

    def test_raw_messages_appended_as_single_turn(self):
        """raw_messages stay as a single turn, not expanded into individual turns."""
        conversation = Conversation(
            session_id="test",
            turns=[
                Turn(
                    role="user",
                    raw_messages=[USER_MSG, ASSISTANT_MSG, TOOL_RESULT_MSG],
                    replaces_history=True,
                ),
            ],
        )
        session = self._make_session(conversation)
        session.advance_turn(0)

        assert len(session.turn_list) == 1
        assert session.turn_list[0].raw_messages == [
            USER_MSG,
            ASSISTANT_MSG,
            TOOL_RESULT_MSG,
        ]

    def test_replaces_history_clears_prior(self):
        """replaces_history=True clears turn_list before appending."""
        conversation = Conversation(
            session_id="test",
            turns=[
                Turn(role="user", raw_messages=[USER_MSG], replaces_history=True),
                Turn(
                    role="user",
                    raw_messages=[USER_MSG, ASSISTANT_MSG, TOOL_RESULT_MSG],
                    replaces_history=True,
                ),
            ],
        )
        session = self._make_session(conversation)

        session.advance_turn(0)
        assert len(session.turn_list) == 1

        session.store_response(Turn(role="assistant"))

        session.advance_turn(1)
        assert len(session.turn_list) == 1
        assert session.turn_list[0].raw_messages == [
            USER_MSG,
            ASSISTANT_MSG,
            TOOL_RESULT_MSG,
        ]

    def test_delta_raw_messages_append_without_clearing(self):
        """Delta turns (no replaces_history) append to existing turn_list."""
        conversation = Conversation(
            session_id="test",
            turns=[
                Turn(role="user", raw_messages=[USER_MSG], replaces_history=True),
                Turn(role="user", raw_messages=[ASSISTANT_MSG, TOOL_RESULT_MSG]),
            ],
        )
        session = self._make_session(conversation)

        session.advance_turn(0)
        assert len(session.turn_list) == 1

        session.advance_turn(1)
        assert len(session.turn_list) == 2
        assert session.turn_list[0].raw_messages == [USER_MSG]
        assert session.turn_list[1].raw_messages == [ASSISTANT_MSG, TOOL_RESULT_MSG]

    def test_without_raw_messages_appends_normally(self):
        """Turns without raw_messages should append as before."""
        conversation = Conversation(
            session_id="test",
            turns=[
                Turn(role="user"),
                Turn(role="user"),
            ],
        )
        session = self._make_session(conversation)

        session.advance_turn(0)
        session.advance_turn(1)

        assert len(session.turn_list) == 2
        assert all(t.raw_messages is None for t in session.turn_list)


# =========================================================================
# TestHelpers
# =========================================================================


class TestHelpers:
    def test_strip_system_removes_leading_system(self):
        assert _strip_system([SYSTEM_MSG, USER_MSG]) == [USER_MSG]

    def test_strip_system_removes_multiple_leading(self):
        assert _strip_system([SYSTEM_MSG, SYSTEM_MSG_2, USER_MSG]) == [USER_MSG]

    def test_strip_system_no_system(self):
        assert _strip_system([USER_MSG]) == [USER_MSG]

    def test_strip_system_empty(self):
        assert _strip_system([]) == []

    def test_extract_system_message_single(self):
        assert _extract_system_message([SYSTEM_MSG, USER_MSG]) == (
            "You are a helpful assistant."
        )

    def test_extract_system_message_multiple(self):
        assert _extract_system_message([SYSTEM_MSG, SYSTEM_MSG_2, USER_MSG]) == (
            "You are a helpful assistant.\nFollow these rules carefully."
        )

    def test_extract_system_message_content_blocks(self):
        msg = {
            "role": "system",
            "content": [
                {"type": "text", "text": "Block one."},
                {"type": "text", "text": "Block two."},
            ],
        }
        assert _extract_system_message([msg]) == "Block one.\nBlock two."

    def test_extract_system_message_none_when_absent(self):
        assert _extract_system_message([USER_MSG]) is None

    def test_is_prefix_true(self):
        assert _is_prefix([USER_MSG], [USER_MSG, ASSISTANT_MSG]) is True

    def test_is_prefix_exact_match(self):
        assert _is_prefix([USER_MSG], [USER_MSG]) is True

    def test_is_prefix_empty_baseline(self):
        assert _is_prefix([], [USER_MSG]) is True

    def test_is_prefix_false_different_content(self):
        other = {"role": "user", "content": "Different"}
        assert _is_prefix([USER_MSG], [other, ASSISTANT_MSG]) is False

    def test_extract_system_message_mixed_content_types(self):
        """Non-text blocks (e.g. image_url) are ignored; only text blocks contribute."""
        msg = {
            "role": "system",
            "content": [
                {"type": "text", "text": "Rule one."},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/img.png"},
                },
                {"type": "text", "text": "Rule two."},
                {"type": "tool_use", "id": "t1"},
            ],
        }
        result = _extract_system_message([msg, USER_MSG])
        assert result == "Rule one.\nRule two."

    def test_strip_system_all_system_messages_returns_empty(self):
        """When every message is a system message, result is empty."""
        messages = [SYSTEM_MSG, SYSTEM_MSG_2]
        assert _strip_system(messages) == []

    def test_is_prefix_false_baseline_longer(self):
        assert _is_prefix([USER_MSG, ASSISTANT_MSG], [USER_MSG]) is False
