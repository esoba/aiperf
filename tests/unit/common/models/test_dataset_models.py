# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dataset_models Turn and Conversation.

Focuses on:
- raw_messages field behavior (None, delta, complete)
- copy_with_stripped_media preserving raw_messages
- replaces_history field
- Removed fields (raw_content, raw_message) no longer exist as declared fields
"""

from aiperf.common.models.dataset_models import (
    Audio,
    Image,
    Text,
    Turn,
    Video,
)

# ============================================================
# Turn.raw_messages field
# ============================================================


class TestTurnRawMessages:
    """Verify raw_messages field behavior on Turn."""

    def test_raw_messages_defaults_to_none(self) -> None:
        turn = Turn()
        assert turn.raw_messages is None

    def test_raw_messages_single_user_message(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        turn = Turn(raw_messages=msgs)
        assert turn.raw_messages == msgs

    def test_raw_messages_multi_message_list(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu-1", "content": "OK"}
                ],
            },
        ]
        turn = Turn(raw_messages=msgs)
        assert len(turn.raw_messages) == 3
        assert turn.raw_messages[1]["role"] == "assistant"

    def test_raw_messages_empty_list(self) -> None:
        turn = Turn(raw_messages=[])
        assert turn.raw_messages == []

    def test_raw_messages_with_complex_content_blocks(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me analyze..."},
                    {
                        "type": "tool_use",
                        "id": "tu-1",
                        "name": "read_file",
                        "input": {"path": "a.py"},
                    },
                ],
            }
        ]
        turn = Turn(raw_messages=msgs)
        assert turn.raw_messages[0]["content"][0]["type"] == "thinking"
        assert turn.raw_messages[0]["content"][1]["type"] == "tool_use"


# ============================================================
# Turn.replaces_history field
# ============================================================


class TestTurnReplacesHistory:
    """Verify replaces_history field defaults and interaction with raw_messages."""

    def test_replaces_history_defaults_to_false(self) -> None:
        turn = Turn()
        assert turn.replaces_history is False

    def test_replaces_history_set_true(self) -> None:
        turn = Turn(
            raw_messages=[{"role": "user", "content": "fresh context"}],
            replaces_history=True,
        )
        assert turn.replaces_history is True
        assert turn.raw_messages is not None

    def test_replaces_history_independent_of_raw_messages(self) -> None:
        turn = Turn(replaces_history=True)
        assert turn.raw_messages is None


# ============================================================
# Turn.copy_with_stripped_media
# ============================================================


class TestTurnCopyWithStrippedMedia:
    """Verify copy_with_stripped_media preserves raw_messages and replaces_history."""

    def test_preserves_raw_messages(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        turn = Turn(raw_messages=msgs, role="user")
        stripped = turn.copy_with_stripped_media()

        assert stripped.raw_messages == msgs

    def test_preserves_none_raw_messages(self) -> None:
        turn = Turn(texts=[Text(contents=["hello"])], role="user")
        stripped = turn.copy_with_stripped_media()

        assert stripped.raw_messages is None

    def test_preserves_replaces_history(self) -> None:
        turn = Turn(
            raw_messages=[{"role": "user", "content": "new context"}],
            replaces_history=True,
        )
        stripped = turn.copy_with_stripped_media()

        assert stripped.replaces_history is True

    def test_preserves_replaces_history_false(self) -> None:
        turn = Turn(
            raw_messages=[{"role": "user", "content": "delta"}],
            replaces_history=False,
        )
        stripped = turn.copy_with_stripped_media()

        assert stripped.replaces_history is False

    def test_strips_images_preserves_raw_messages(self) -> None:
        msgs = [{"role": "user", "content": "check this image"}]
        turn = Turn(
            raw_messages=msgs,
            images=[Image(name="photo", contents=["base64data_very_large"])],
        )
        stripped = turn.copy_with_stripped_media()

        assert stripped.raw_messages == msgs
        assert stripped.images[0].contents == ["image_0"]

    def test_strips_audio_preserves_raw_messages(self) -> None:
        msgs = [{"role": "user", "content": "listen"}]
        turn = Turn(
            raw_messages=msgs,
            audios=[Audio(name="clip", contents=["mp3,base64data"])],
        )
        stripped = turn.copy_with_stripped_media()

        assert stripped.raw_messages == msgs
        assert stripped.audios[0].contents == ["audio_0"]

    def test_strips_video_preserves_raw_messages(self) -> None:
        msgs = [{"role": "user", "content": "watch"}]
        turn = Turn(
            raw_messages=msgs,
            videos=[Video(name="vid", contents=["mp4,base64data"])],
        )
        stripped = turn.copy_with_stripped_media()

        assert stripped.raw_messages == msgs
        assert stripped.videos[0].contents == ["video_0"]

    def test_raw_payload_cleared_on_strip(self) -> None:
        turn = Turn(
            raw_messages=[{"role": "user", "content": "test"}],
            raw_payload={"model": "test", "messages": []},
        )
        stripped = turn.copy_with_stripped_media()

        assert stripped.raw_payload is None
        assert stripped.raw_messages is not None


# ============================================================
# Turn removed fields
# ============================================================


class TestTurnRemovedFields:
    """Verify raw_content and raw_message are not declared fields on Turn."""

    def test_raw_content_not_a_declared_field(self) -> None:
        assert "raw_content" not in Turn.model_fields

    def test_raw_message_not_a_declared_field(self) -> None:
        assert "raw_message" not in Turn.model_fields

    def test_raw_messages_is_a_declared_field(self) -> None:
        assert "raw_messages" in Turn.model_fields


# ============================================================
# Turn.metadata()
# ============================================================


class TestTurnMetadata:
    """Verify metadata() excludes raw_messages (it is not part of TurnMetadata)."""

    def test_metadata_does_not_include_raw_messages(self) -> None:
        turn = Turn(
            raw_messages=[{"role": "user", "content": "test"}],
            input_tokens=42,
        )
        meta = turn.metadata()
        assert meta.input_tokens == 42
        assert not hasattr(meta, "raw_messages")
