# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared Riva audio encoding map."""

from __future__ import annotations

import pytest

from aiperf.transports.grpc.proto.riva import riva_audio_pb2
from aiperf.transports.grpc.riva_encoding import ENCODING_MAP


class TestEncodingMap:
    """Tests for the shared ENCODING_MAP."""

    @pytest.mark.parametrize(
        ("name", "expected_value"),
        [
            ("LINEAR_PCM", riva_audio_pb2.LINEAR_PCM),
            ("FLAC", riva_audio_pb2.FLAC),
            ("MULAW", riva_audio_pb2.MULAW),
            ("OGGOPUS", riva_audio_pb2.OGGOPUS),
            ("ALAW", riva_audio_pb2.ALAW),
        ],
    )
    def test_contains_all_encodings(self, name: str, expected_value: int) -> None:
        """All Riva audio encodings should be in the map."""
        assert ENCODING_MAP[name] == expected_value

    def test_exactly_five_entries(self) -> None:
        """ENCODING_MAP should contain exactly 5 entries."""
        assert len(ENCODING_MAP) == 5

    def test_unknown_encoding_not_in_map(self) -> None:
        """Unknown encoding names should not be in the map."""
        assert "UNKNOWN" not in ENCODING_MAP
        assert "MP3" not in ENCODING_MAP

    def test_linear_pcm_is_nonzero(self) -> None:
        """LINEAR_PCM should be nonzero (protobuf enum value 1, not default 0)."""
        assert ENCODING_MAP["LINEAR_PCM"] != 0

    def test_asr_and_tts_use_same_map(self) -> None:
        """Both ASR and TTS serializers should import from the same module."""
        from aiperf.transports.grpc import riva_asr_serializers, riva_tts_serializers

        # Verify they both reference the shared module (not local copies)
        assert riva_asr_serializers.ENCODING_MAP is ENCODING_MAP
        assert riva_tts_serializers.ENCODING_MAP is ENCODING_MAP
