# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Riva ASR serializer (unified for offline and streaming)."""

from __future__ import annotations

import pytest

from aiperf.transports.grpc.grpc_transport import GrpcSerializerProtocol
from aiperf.transports.grpc.proto.riva import riva_asr_pb2, riva_audio_pb2
from aiperf.transports.grpc.riva_asr_serializers import RivaAsrSerializer
from aiperf.transports.grpc.stream_chunk import StreamChunk


class TestRivaAsrSerializerProtocol:
    def test_implements_protocol(self) -> None:
        assert isinstance(RivaAsrSerializer(), GrpcSerializerProtocol)


class TestAsrSerializeRequest:
    def test_roundtrip_basic(self) -> None:
        """Serialized bytes should parse back to a valid RecognizeRequest."""
        audio = b"\x00\x01\x02\x03" * 100
        payload = {
            "audio": audio,
            "language_code": "en-US",
            "sample_rate_hertz": 16000,
            "encoding": "LINEAR_PCM",
        }
        data = RivaAsrSerializer.serialize_request(
            payload, model_name="asr_model", request_id="r1"
        )

        parsed = riva_asr_pb2.RecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.audio == audio
        assert parsed.config.language_code == "en-US"
        assert parsed.config.sample_rate_hertz == 16000
        assert parsed.config.encoding == riva_audio_pb2.LINEAR_PCM
        assert parsed.config.model == "asr_model"
        assert parsed.id.value == "r1"

    def test_model_from_payload_takes_precedence(self) -> None:
        """Model from payload should override model_name argument."""
        payload = {"audio": b"audio", "model": "custom_model"}
        data = RivaAsrSerializer.serialize_request(payload, model_name="default_model")

        parsed = riva_asr_pb2.RecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.config.model == "custom_model"

    def test_no_request_id(self) -> None:
        """Request without request_id should not set the id field."""
        data = RivaAsrSerializer.serialize_request({"audio": b"data"}, model_name="m")
        parsed = riva_asr_pb2.RecognizeRequest()
        parsed.ParseFromString(data)
        assert not parsed.HasField("id")

    def test_empty_audio(self) -> None:
        """Empty audio bytes should produce a valid request."""
        data = RivaAsrSerializer.serialize_request({"audio": b""}, model_name="m")
        parsed = riva_asr_pb2.RecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.audio == b""

    def test_default_values(self) -> None:
        """Default config values should be set correctly."""
        data = RivaAsrSerializer.serialize_request({}, model_name="m")
        parsed = riva_asr_pb2.RecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.config.language_code == "en-US"
        assert parsed.config.sample_rate_hertz == 16000
        assert parsed.config.encoding == riva_audio_pb2.LINEAR_PCM
        assert parsed.config.max_alternatives == 1
        assert parsed.config.enable_automatic_punctuation is True

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
        payload = {"audio": b"data", "encoding": encoding_str}
        data = RivaAsrSerializer.serialize_request(payload, model_name="m")
        parsed = riva_asr_pb2.RecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.config.encoding == expected_enum

    def test_unknown_encoding_defaults_to_linear_pcm(self) -> None:
        """Unknown encoding should fall back to LINEAR_PCM."""
        payload = {"audio": b"data", "encoding": "UNKNOWN"}
        data = RivaAsrSerializer.serialize_request(payload, model_name="m")
        parsed = riva_asr_pb2.RecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.config.encoding == riva_audio_pb2.LINEAR_PCM

    def test_audio_string_encoded_as_bytes(self) -> None:
        """String audio should be encoded to bytes."""
        payload = {"audio": "string_audio_data"}
        data = RivaAsrSerializer.serialize_request(payload, model_name="m")
        parsed = riva_asr_pb2.RecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.audio == b"string_audio_data"

    def test_custom_max_alternatives(self) -> None:
        """Custom max_alternatives should be set."""
        payload = {"audio": b"data", "max_alternatives": 5}
        data = RivaAsrSerializer.serialize_request(payload, model_name="m")
        parsed = riva_asr_pb2.RecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.config.max_alternatives == 5

    def test_disable_automatic_punctuation(self) -> None:
        """Automatic punctuation can be disabled."""
        payload = {"audio": b"data", "enable_automatic_punctuation": False}
        data = RivaAsrSerializer.serialize_request(payload, model_name="m")
        parsed = riva_asr_pb2.RecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.config.enable_automatic_punctuation is False

    def test_model_name_fallback_when_no_payload_model(self) -> None:
        """model_name argument should be used when payload has no model."""
        payload = {"audio": b"data"}
        data = RivaAsrSerializer.serialize_request(payload, model_name="fallback_model")
        parsed = riva_asr_pb2.RecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.config.model == "fallback_model"


class TestAsrDeserializeResponse:
    def test_transcript_extraction(self) -> None:
        """Should extract transcript from RecognizeResponse."""
        response = riva_asr_pb2.RecognizeResponse()
        result = response.results.add()
        alt = result.alternatives.add()
        alt.transcript = "hello world"
        alt.confidence = 0.95
        data = response.SerializeToString()

        result_dict, size = RivaAsrSerializer.deserialize_response(data)

        assert size == len(data)
        assert result_dict["transcript"] == "hello world"
        assert (
            result_dict["results"][0]["alternatives"][0]["transcript"] == "hello world"
        )
        assert result_dict["results"][0]["alternatives"][0][
            "confidence"
        ] == pytest.approx(0.95)

    def test_multiple_results(self) -> None:
        """Multiple results should be concatenated with space."""
        response = riva_asr_pb2.RecognizeResponse()
        r1 = response.results.add()
        r1.alternatives.add().transcript = "hello"
        r2 = response.results.add()
        r2.alternatives.add().transcript = "world"
        data = response.SerializeToString()

        result_dict, _ = RivaAsrSerializer.deserialize_response(data)

        assert result_dict["transcript"] == "hello world"
        assert len(result_dict["results"]) == 2

    def test_empty_results(self) -> None:
        """Empty response should produce empty transcript."""
        response = riva_asr_pb2.RecognizeResponse()
        data = response.SerializeToString()

        result_dict, _ = RivaAsrSerializer.deserialize_response(data)

        assert result_dict["transcript"] == ""
        assert result_dict["results"] == []

    def test_multiple_alternatives(self) -> None:
        """Multiple alternatives should all be included."""
        response = riva_asr_pb2.RecognizeResponse()
        result = response.results.add()
        alt1 = result.alternatives.add()
        alt1.transcript = "hello world"
        alt1.confidence = 0.95
        alt2 = result.alternatives.add()
        alt2.transcript = "hello word"
        alt2.confidence = 0.7
        data = response.SerializeToString()

        result_dict, _ = RivaAsrSerializer.deserialize_response(data)

        # Top transcript comes from first alternative
        assert result_dict["transcript"] == "hello world"
        assert len(result_dict["results"][0]["alternatives"]) == 2
        assert (
            result_dict["results"][0]["alternatives"][1]["transcript"] == "hello word"
        )

    def test_result_with_no_alternatives(self) -> None:
        """Result with empty alternatives list should be handled."""
        response = riva_asr_pb2.RecognizeResponse()
        response.results.add()  # no alternatives
        data = response.SerializeToString()

        result_dict, _ = RivaAsrSerializer.deserialize_response(data)

        assert result_dict["transcript"] == ""
        assert result_dict["results"][0]["alternatives"] == []


class TestAsrSerializeStreamConfig:
    def test_stream_config_roundtrip(self) -> None:
        """First streaming message should have config."""
        payload = {
            "language_code": "en-US",
            "sample_rate_hertz": 16000,
            "encoding": "LINEAR_PCM",
            "interim_results": True,
        }
        data = RivaAsrSerializer.serialize_stream_config(
            payload, model_name="asr_model", request_id="r1"
        )

        parsed = riva_asr_pb2.StreamingRecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.HasField("streaming_config")
        assert parsed.streaming_config.config.language_code == "en-US"
        assert parsed.streaming_config.config.sample_rate_hertz == 16000
        assert parsed.streaming_config.interim_results is True
        assert parsed.streaming_config.config.model == "asr_model"

    def test_stream_config_no_interim(self) -> None:
        """Interim results can be disabled."""
        payload = {"interim_results": False}
        data = RivaAsrSerializer.serialize_stream_config(payload, model_name="m")

        parsed = riva_asr_pb2.StreamingRecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.streaming_config.interim_results is False

    def test_stream_config_default_interim_results(self) -> None:
        """Default interim_results should be True."""
        data = RivaAsrSerializer.serialize_stream_config({}, model_name="m")

        parsed = riva_asr_pb2.StreamingRecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.streaming_config.interim_results is True

    def test_stream_config_with_request_id(self) -> None:
        """Request ID should be set in config message."""
        data = RivaAsrSerializer.serialize_stream_config(
            {}, model_name="m", request_id="req-42"
        )

        parsed = riva_asr_pb2.StreamingRecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.id.value == "req-42"

    def test_stream_config_no_request_id(self) -> None:
        """Config without request_id should not set id field."""
        data = RivaAsrSerializer.serialize_stream_config({}, model_name="m")

        parsed = riva_asr_pb2.StreamingRecognizeRequest()
        parsed.ParseFromString(data)
        assert not parsed.HasField("id")

    def test_stream_config_model_from_payload(self) -> None:
        """Payload model should override model_name."""
        payload = {"model": "payload_model"}
        data = RivaAsrSerializer.serialize_stream_config(payload, model_name="default")

        parsed = riva_asr_pb2.StreamingRecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.streaming_config.config.model == "payload_model"

    @pytest.mark.parametrize(
        ("encoding_str", "expected_enum"),
        [
            ("FLAC", riva_audio_pb2.FLAC),
            ("MULAW", riva_audio_pb2.MULAW),
        ],
    )
    def test_stream_config_encodings(
        self, encoding_str: str, expected_enum: int
    ) -> None:
        """Stream config should support all encoding types."""
        payload = {"encoding": encoding_str}
        data = RivaAsrSerializer.serialize_stream_config(payload, model_name="m")

        parsed = riva_asr_pb2.StreamingRecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.streaming_config.config.encoding == expected_enum


class TestAsrSerializeStreamChunk:
    def test_audio_chunk_roundtrip(self) -> None:
        """Audio chunk should be preserved."""
        audio_chunk = b"\x00\x01\x02\x03" * 50
        data = RivaAsrSerializer.serialize_stream_chunk(audio_chunk)

        parsed = riva_asr_pb2.StreamingRecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.audio_content == audio_chunk

    def test_empty_chunk(self) -> None:
        """Empty audio chunk should produce valid message."""
        data = RivaAsrSerializer.serialize_stream_chunk(b"")

        parsed = riva_asr_pb2.StreamingRecognizeRequest()
        parsed.ParseFromString(data)
        assert parsed.audio_content == b""

    def test_large_chunk(self) -> None:
        """Large audio chunk should be preserved."""
        audio = b"\xff" * 100_000
        data = RivaAsrSerializer.serialize_stream_chunk(audio)

        parsed = riva_asr_pb2.StreamingRecognizeRequest()
        parsed.ParseFromString(data)
        assert len(parsed.audio_content) == 100_000


class TestAsrDeserializeBidiResponse:
    def test_final_response(self) -> None:
        """Final streaming response should have transcript and is_final=True."""
        response = riva_asr_pb2.StreamingRecognizeResponse()
        result = response.results.add()
        result.is_final = True
        alt = result.alternatives.add()
        alt.transcript = "hello world"
        alt.confidence = 0.95
        data = response.SerializeToString()

        chunk = RivaAsrSerializer.deserialize_bidi_response(data)

        assert isinstance(chunk, StreamChunk)
        assert chunk.error_message is None
        assert chunk.response_dict["transcript"] == "hello world"
        assert chunk.response_dict["is_final"] is True

    def test_interim_response(self) -> None:
        """Interim streaming response should have is_final=False."""
        response = riva_asr_pb2.StreamingRecognizeResponse()
        result = response.results.add()
        result.is_final = False
        result.stability = 0.5
        alt = result.alternatives.add()
        alt.transcript = "hell"
        data = response.SerializeToString()

        chunk = RivaAsrSerializer.deserialize_bidi_response(data)

        assert chunk.response_dict["is_final"] is False
        assert chunk.response_dict["transcript"] == "hell"

    def test_stability_preserved(self) -> None:
        """Stability value should be in the results."""
        response = riva_asr_pb2.StreamingRecognizeResponse()
        result = response.results.add()
        result.stability = 0.8
        result.alternatives.add().transcript = "hello"
        data = response.SerializeToString()

        chunk = RivaAsrSerializer.deserialize_bidi_response(data)

        assert chunk.response_dict["results"][0]["stability"] == pytest.approx(0.8)

    def test_multiple_results_is_final_or(self) -> None:
        """is_final should be True if ANY result is final."""
        response = riva_asr_pb2.StreamingRecognizeResponse()
        r1 = response.results.add()
        r1.is_final = False
        r1.alternatives.add().transcript = "hell"
        r2 = response.results.add()
        r2.is_final = True
        r2.alternatives.add().transcript = "hello"
        data = response.SerializeToString()

        chunk = RivaAsrSerializer.deserialize_bidi_response(data)

        assert chunk.response_dict["is_final"] is True
        assert chunk.response_dict["transcript"] == "hell hello"

    def test_empty_response(self) -> None:
        """Empty streaming response should produce valid chunk."""
        response = riva_asr_pb2.StreamingRecognizeResponse()
        data = response.SerializeToString()

        chunk = RivaAsrSerializer.deserialize_bidi_response(data)

        assert chunk.error_message is None
        assert chunk.response_dict["transcript"] == ""
        assert chunk.response_dict["is_final"] is False
        assert chunk.response_dict["results"] == []

    def test_confidence_preserved(self) -> None:
        """Confidence should be in the alternatives."""
        response = riva_asr_pb2.StreamingRecognizeResponse()
        result = response.results.add()
        result.is_final = True
        alt = result.alternatives.add()
        alt.transcript = "test"
        alt.confidence = 0.99
        data = response.SerializeToString()

        chunk = RivaAsrSerializer.deserialize_bidi_response(data)

        assert chunk.response_dict["results"][0]["alternatives"][0][
            "confidence"
        ] == pytest.approx(0.99)

    def test_response_size(self) -> None:
        """response_size should match wire bytes."""
        response = riva_asr_pb2.StreamingRecognizeResponse()
        result = response.results.add()
        result.alternatives.add().transcript = "test"
        data = response.SerializeToString()

        chunk = RivaAsrSerializer.deserialize_bidi_response(data)

        assert chunk.response_size == len(data)


class TestAsrDeserializeStreamResponse:
    """Tests for deserialize_stream_response (delegates to bidi)."""

    def test_delegates_to_bidi(self) -> None:
        """deserialize_stream_response should delegate to deserialize_bidi_response."""
        response = riva_asr_pb2.StreamingRecognizeResponse()
        result = response.results.add()
        result.is_final = True
        result.alternatives.add().transcript = "hello"
        data = response.SerializeToString()

        chunk = RivaAsrSerializer.deserialize_stream_response(data)

        assert chunk.response_dict["transcript"] == "hello"
        assert chunk.response_dict["is_final"] is True
