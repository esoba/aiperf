# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RivaTtsSerializer."""

from __future__ import annotations

import pytest

from aiperf.transports.grpc.grpc_transport import GrpcSerializerProtocol
from aiperf.transports.grpc.proto.riva import riva_audio_pb2, riva_tts_pb2
from aiperf.transports.grpc.riva_tts_serializers import RivaTtsSerializer
from aiperf.transports.grpc.stream_chunk import StreamChunk


class TestRivaTtsSerializerProtocol:
    """Tests for protocol conformance."""

    def test_implements_protocol(self) -> None:
        assert isinstance(RivaTtsSerializer(), GrpcSerializerProtocol)


class TestSerializeRequest:
    """Tests for RivaTtsSerializer.serialize_request."""

    def test_roundtrip_basic(self) -> None:
        """Serialized bytes should parse back to a valid SynthesizeSpeechRequest."""
        payload = {
            "text": "Hello world",
            "voice_name": "English-US-Female-1",
            "language_code": "en-US",
            "encoding": "LINEAR_PCM",
            "sample_rate_hz": 22050,
        }
        serializer = RivaTtsSerializer()
        data = serializer.serialize_request(
            payload, model_name="tts_model", request_id="r1"
        )
        assert isinstance(data, bytes)
        assert len(data) > 0

        parsed = riva_tts_pb2.SynthesizeSpeechRequest()
        parsed.ParseFromString(data)
        assert parsed.text == "Hello world"
        assert parsed.voice_name == "English-US-Female-1"
        assert parsed.language_code == "en-US"
        assert parsed.encoding == riva_audio_pb2.LINEAR_PCM
        assert parsed.sample_rate_hz == 22050
        assert parsed.id.value == "r1"

    def test_default_encoding(self) -> None:
        """Default encoding should be LINEAR_PCM."""
        payload = {"text": "test"}
        serializer = RivaTtsSerializer()
        data = serializer.serialize_request(payload, model_name="m")
        parsed = riva_tts_pb2.SynthesizeSpeechRequest()
        parsed.ParseFromString(data)
        assert parsed.encoding == riva_audio_pb2.LINEAR_PCM

    @pytest.mark.parametrize(
        ("encoding_str", "expected_enum"),
        [
            ("FLAC", riva_audio_pb2.FLAC),
            ("MULAW", riva_audio_pb2.MULAW),
            ("OGGOPUS", riva_audio_pb2.OGGOPUS),
            ("ALAW", riva_audio_pb2.ALAW),
            ("LINEAR_PCM", riva_audio_pb2.LINEAR_PCM),
        ],
    )
    def test_all_encoding_types(self, encoding_str: str, expected_enum: int) -> None:
        """All supported encoding types should be correctly mapped."""
        payload = {"text": "test", "encoding": encoding_str}
        data = RivaTtsSerializer.serialize_request(payload, model_name="m")
        parsed = riva_tts_pb2.SynthesizeSpeechRequest()
        parsed.ParseFromString(data)
        assert parsed.encoding == expected_enum

    def test_unknown_encoding_defaults_to_linear_pcm(self) -> None:
        """Unknown encoding should fall back to LINEAR_PCM."""
        payload = {"text": "test", "encoding": "UNKNOWN_CODEC"}
        data = RivaTtsSerializer.serialize_request(payload, model_name="m")
        parsed = riva_tts_pb2.SynthesizeSpeechRequest()
        parsed.ParseFromString(data)
        assert parsed.encoding == riva_audio_pb2.LINEAR_PCM

    def test_empty_text(self) -> None:
        """Empty text should produce a valid request."""
        data = RivaTtsSerializer.serialize_request({}, model_name="m")
        parsed = riva_tts_pb2.SynthesizeSpeechRequest()
        parsed.ParseFromString(data)
        assert parsed.text == ""

    def test_no_request_id(self) -> None:
        """Request without request_id should not set the id field."""
        data = RivaTtsSerializer.serialize_request({"text": "hi"}, model_name="m")
        parsed = riva_tts_pb2.SynthesizeSpeechRequest()
        parsed.ParseFromString(data)
        assert not parsed.HasField("id")

    def test_default_sample_rate(self) -> None:
        """Default sample rate should be 22050."""
        data = RivaTtsSerializer.serialize_request({"text": "hi"}, model_name="m")
        parsed = riva_tts_pb2.SynthesizeSpeechRequest()
        parsed.ParseFromString(data)
        assert parsed.sample_rate_hz == 22050

    def test_default_language_code(self) -> None:
        """Default language code should be en-US."""
        data = RivaTtsSerializer.serialize_request({"text": "hi"}, model_name="m")
        parsed = riva_tts_pb2.SynthesizeSpeechRequest()
        parsed.ParseFromString(data)
        assert parsed.language_code == "en-US"

    def test_custom_sample_rate(self) -> None:
        """Custom sample rate should be set."""
        payload = {"text": "hi", "sample_rate_hz": 44100}
        data = RivaTtsSerializer.serialize_request(payload, model_name="m")
        parsed = riva_tts_pb2.SynthesizeSpeechRequest()
        parsed.ParseFromString(data)
        assert parsed.sample_rate_hz == 44100

    def test_empty_voice_name(self) -> None:
        """Default voice_name should be empty string."""
        data = RivaTtsSerializer.serialize_request({"text": "hi"}, model_name="m")
        parsed = riva_tts_pb2.SynthesizeSpeechRequest()
        parsed.ParseFromString(data)
        assert parsed.voice_name == ""


class TestDeserializeResponse:
    """Tests for RivaTtsSerializer.deserialize_response."""

    def test_audio_bytes_preserved(self) -> None:
        """Audio bytes should survive serialization round-trip."""
        audio_data = b"\x00\x01\x02\x03" * 100
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        response.audio = audio_data
        response.meta.text = "Hello world"
        data = response.SerializeToString()

        result_dict, size = RivaTtsSerializer.deserialize_response(data)

        assert size == len(data)
        assert result_dict["audio"] == audio_data
        assert result_dict["meta"]["text"] == "Hello world"

    def test_empty_audio(self) -> None:
        """Empty audio should be deserialized correctly."""
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        data = response.SerializeToString()

        result_dict, size = RivaTtsSerializer.deserialize_response(data)

        assert result_dict["audio"] == b""

    def test_no_meta_text(self) -> None:
        """Response without meta.text should not include meta key."""
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        response.audio = b"\x01\x02"
        data = response.SerializeToString()

        result_dict, _ = RivaTtsSerializer.deserialize_response(data)

        assert "meta" not in result_dict
        assert result_dict["audio"] == b"\x01\x02"

    def test_meta_processed_text(self) -> None:
        """Meta with processed_text should be included."""
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        response.audio = b"\x01"
        response.meta.text = "hello"
        response.meta.processed_text = "Hello."
        data = response.SerializeToString()

        result_dict, _ = RivaTtsSerializer.deserialize_response(data)

        assert result_dict["meta"]["text"] == "hello"
        assert result_dict["meta"]["processed_text"] == "Hello."

    def test_large_audio_payload(self) -> None:
        """Large audio payloads should be preserved."""
        audio_data = b"\xff" * 1_000_000
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        response.audio = audio_data
        data = response.SerializeToString()

        result_dict, size = RivaTtsSerializer.deserialize_response(data)

        assert len(result_dict["audio"]) == 1_000_000
        assert size == len(data)

    def test_wire_size_matches(self) -> None:
        """Wire size should equal the serialized bytes length."""
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        response.audio = b"\x00" * 500
        data = response.SerializeToString()

        _, size = RivaTtsSerializer.deserialize_response(data)

        assert size == len(data)


class TestDeserializeStreamResponse:
    """Tests for RivaTtsSerializer.deserialize_stream_response."""

    def test_stream_chunk_with_audio(self) -> None:
        """Streaming chunk should produce StreamChunk with audio data."""
        audio_data = b"\x00\x01\x02\x03" * 50
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        response.audio = audio_data
        data = response.SerializeToString()

        chunk = RivaTtsSerializer.deserialize_stream_response(data)

        assert isinstance(chunk, StreamChunk)
        assert chunk.error_message is None
        assert chunk.response_dict is not None
        assert chunk.response_dict["audio"] == audio_data
        assert chunk.response_size == len(data)

    def test_stream_chunk_empty_audio(self) -> None:
        """Empty streaming chunk should still be valid."""
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        data = response.SerializeToString()

        chunk = RivaTtsSerializer.deserialize_stream_response(data)

        assert chunk.error_message is None
        assert chunk.response_dict is not None
        assert chunk.response_dict["audio"] == b""

    def test_stream_chunk_with_meta(self) -> None:
        """Streaming chunk with meta should include it."""
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        response.audio = b"\x01"
        response.meta.text = "test"
        data = response.SerializeToString()

        chunk = RivaTtsSerializer.deserialize_stream_response(data)

        assert chunk.response_dict["meta"]["text"] == "test"

    def test_stream_chunk_no_meta(self) -> None:
        """Streaming chunk without meta should not include meta key."""
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        response.audio = b"\x01"
        data = response.SerializeToString()

        chunk = RivaTtsSerializer.deserialize_stream_response(data)

        assert "meta" not in chunk.response_dict

    def test_stream_response_size(self) -> None:
        """Stream response_size should match wire bytes length."""
        audio = b"\xff" * 10000
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        response.audio = audio
        data = response.SerializeToString()

        chunk = RivaTtsSerializer.deserialize_stream_response(data)

        assert chunk.response_size == len(data)
