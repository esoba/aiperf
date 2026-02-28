# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Riva TTS endpoints."""

from __future__ import annotations

import base64

import pytest

from aiperf.common.models import Text, Turn
from aiperf.common.models.record_models import AudioResponseData
from aiperf.endpoints.riva_tts import (
    RivaTtsEndpoint,
    RivaTtsStreamingEndpoint,
    _calc_duration_ms,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


# ---------------------------------------------------------------------------
# _calc_duration_ms
# ---------------------------------------------------------------------------
class TestCalcDurationMs:
    def test_linear_pcm_duration(self) -> None:
        """LINEAR_PCM: 2 bytes per sample, 16kHz -> 1000ms for 32000 bytes."""
        audio = b"\x00" * 32000
        result = _calc_duration_ms(audio, sample_rate_hz=16000, encoding="LINEAR_PCM")
        assert result == pytest.approx(1000.0)

    def test_linear_pcm_22050(self) -> None:
        """LINEAR_PCM at 22050Hz: 44100 bytes -> 1000ms."""
        audio = b"\x00" * 44100
        result = _calc_duration_ms(audio, sample_rate_hz=22050, encoding="LINEAR_PCM")
        assert result == pytest.approx(1000.0)

    def test_non_pcm_encoding_returns_none(self) -> None:
        """Non-PCM encodings should return None (can't compute duration)."""
        assert _calc_duration_ms(b"\x00" * 100, 16000, "FLAC") is None
        assert _calc_duration_ms(b"\x00" * 100, 16000, "MULAW") is None
        assert _calc_duration_ms(b"\x00" * 100, 16000, "OGGOPUS") is None

    def test_zero_sample_rate_returns_none(self) -> None:
        """Zero sample rate should return None."""
        assert _calc_duration_ms(b"\x00" * 100, 0, "LINEAR_PCM") is None

    def test_negative_sample_rate_returns_none(self) -> None:
        """Negative sample rate should return None."""
        assert _calc_duration_ms(b"\x00" * 100, -1, "LINEAR_PCM") is None

    def test_empty_audio_returns_none(self) -> None:
        """Empty audio should return None."""
        assert _calc_duration_ms(b"", 16000, "LINEAR_PCM") is None

    def test_small_audio(self) -> None:
        """2 bytes at 16kHz -> 1 sample -> ~0.0625ms."""
        result = _calc_duration_ms(b"\x00\x00", 16000, "LINEAR_PCM")
        assert result == pytest.approx(0.0625)


# ---------------------------------------------------------------------------
# RivaTtsEndpoint.format_payload
# ---------------------------------------------------------------------------
class TestRivaTtsEndpointFormatPayload:
    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.RIVA_TTS,
            model_name="tts_model",
            extra=[
                ("voice_name", "English-US-Female-1"),
                ("sample_rate_hz", 22050),
            ],
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(RivaTtsEndpoint, model_endpoint)

    def test_format_payload_basic(self, endpoint, model_endpoint) -> None:
        turn = Turn(texts=[Text(contents=["Hello world"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["text"] == "Hello world"
        assert payload["voice_name"] == "English-US-Female-1"
        assert payload["language_code"] == "en-US"
        assert payload["encoding"] == "LINEAR_PCM"
        assert payload["sample_rate_hz"] == 22050

    def test_format_payload_empty_turns_raises(self, endpoint, model_endpoint) -> None:
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])
        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_multiple_texts_joined(self, model_endpoint) -> None:
        """Multiple text contents should be joined with space."""
        endpoint = create_endpoint_with_mock_transport(RivaTtsEndpoint, model_endpoint)
        turn = Turn(texts=[Text(contents=["Hello", "world"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["text"] == "Hello world"

    def test_format_payload_multiple_text_objects(self, model_endpoint) -> None:
        """Multiple Text objects should all contribute contents."""
        endpoint = create_endpoint_with_mock_transport(RivaTtsEndpoint, model_endpoint)
        turn = Turn(texts=[Text(contents=["Hello"]), Text(contents=["world"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["text"] == "Hello world"

    def test_format_payload_empty_content_filtered(self, model_endpoint) -> None:
        """Empty content strings should be filtered out."""
        endpoint = create_endpoint_with_mock_transport(RivaTtsEndpoint, model_endpoint)
        turn = Turn(texts=[Text(contents=["Hello", "", "world"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["text"] == "Hello world"

    def test_format_payload_default_extra(self) -> None:
        """Endpoint with no extra config should use defaults."""
        model_endpoint = create_model_endpoint(EndpointType.RIVA_TTS)
        endpoint = create_endpoint_with_mock_transport(RivaTtsEndpoint, model_endpoint)
        turn = Turn(texts=[Text(contents=["test"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["voice_name"] == ""
        assert payload["language_code"] == "en-US"
        assert payload["encoding"] == "LINEAR_PCM"
        assert payload["sample_rate_hz"] == 22050

    def test_format_payload_custom_encoding(self) -> None:
        """Custom encoding from extra config should be used."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_TTS,
            extra=[("encoding", "FLAC")],
        )
        endpoint = create_endpoint_with_mock_transport(RivaTtsEndpoint, model_endpoint)
        turn = Turn(texts=[Text(contents=["test"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["encoding"] == "FLAC"


# ---------------------------------------------------------------------------
# RivaTtsEndpoint.parse_response
# ---------------------------------------------------------------------------
class TestRivaTtsEndpointParseResponse:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_TTS, model_name="tts_model"
        )
        return create_endpoint_with_mock_transport(RivaTtsEndpoint, model_endpoint)

    def test_parse_response_with_base64_audio(self, endpoint) -> None:
        """Base64-encoded audio should be decoded."""
        audio = b"\x00\x01\x02\x03" * 100
        audio_b64 = base64.b64encode(audio).decode("utf-8")
        response = create_mock_response(json_data={"audio": audio_b64})

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert isinstance(parsed.data, AudioResponseData)
        assert parsed.data.audio_bytes == audio
        assert parsed.data.sample_rate_hz == 22050

    def test_parse_response_no_json(self, endpoint) -> None:
        response = create_mock_response(json_data=None)
        assert endpoint.parse_response(response) is None

    def test_parse_response_no_audio(self, endpoint) -> None:
        response = create_mock_response(json_data={"meta": {}})
        assert endpoint.parse_response(response) is None

    def test_parse_response_empty_audio(self, endpoint) -> None:
        """Empty string audio should return None."""
        response = create_mock_response(json_data={"audio": ""})
        assert endpoint.parse_response(response) is None

    def test_parse_response_preserves_perf_ns(self, endpoint) -> None:
        """perf_ns from response should be preserved."""
        audio_b64 = base64.b64encode(b"\x01\x02").decode()
        response = create_mock_response(perf_ns=999, json_data={"audio": audio_b64})

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.perf_ns == 999

    def test_parse_response_with_duration(self) -> None:
        """LINEAR_PCM audio should compute duration_ms."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_TTS,
            extra=[("sample_rate_hz", 16000)],
        )
        endpoint = create_endpoint_with_mock_transport(RivaTtsEndpoint, model_endpoint)
        # 32000 bytes = 16000 samples at 16kHz = 1000ms
        audio = b"\x00" * 32000
        audio_b64 = base64.b64encode(audio).decode()
        response = create_mock_response(json_data={"audio": audio_b64})

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.duration_ms == pytest.approx(1000.0)

    def test_parse_response_flac_no_duration(self) -> None:
        """FLAC audio should not compute duration_ms."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_TTS,
            extra=[("encoding", "FLAC")],
        )
        endpoint = create_endpoint_with_mock_transport(RivaTtsEndpoint, model_endpoint)
        audio_b64 = base64.b64encode(b"\x00" * 100).decode()
        response = create_mock_response(json_data={"audio": audio_b64})

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.duration_ms is None
        assert parsed.data.encoding == "FLAC"

    def test_parse_response_raw_bytes_audio(self) -> None:
        """Raw bytes audio (not base64) should be handled."""
        model_endpoint = create_model_endpoint(EndpointType.RIVA_TTS)
        endpoint = create_endpoint_with_mock_transport(RivaTtsEndpoint, model_endpoint)
        audio = b"\x00\x01\x02\x03"
        response = create_mock_response(json_data={"audio": audio})

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.audio_bytes == audio


# ---------------------------------------------------------------------------
# RivaTtsStreamingEndpoint
# ---------------------------------------------------------------------------
class TestRivaTtsStreamingEndpointFormatPayload:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_TTS_STREAMING,
            model_name="tts_model",
            streaming=True,
        )
        return create_endpoint_with_mock_transport(
            RivaTtsStreamingEndpoint, model_endpoint
        )

    def test_format_payload_basic(self, endpoint) -> None:
        model_endpoint = endpoint.model_endpoint
        turn = Turn(texts=[Text(contents=["Hello world"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["text"] == "Hello world"

    def test_format_payload_empty_turns_raises(self, endpoint) -> None:
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[]
        )
        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_default_config(self) -> None:
        """Streaming endpoint with no extra should use defaults."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_TTS_STREAMING, streaming=True
        )
        endpoint = create_endpoint_with_mock_transport(
            RivaTtsStreamingEndpoint, model_endpoint
        )
        turn = Turn(texts=[Text(contents=["test"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["voice_name"] == ""
        assert payload["language_code"] == "en-US"
        assert payload["encoding"] == "LINEAR_PCM"
        assert payload["sample_rate_hz"] == 22050

    def test_format_payload_custom_config(self) -> None:
        """Custom config from extra should be used."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_TTS_STREAMING,
            streaming=True,
            extra=[
                ("voice_name", "German-Female-1"),
                ("language_code", "de-DE"),
                ("encoding", "FLAC"),
                ("sample_rate_hz", 44100),
            ],
        )
        endpoint = create_endpoint_with_mock_transport(
            RivaTtsStreamingEndpoint, model_endpoint
        )
        turn = Turn(texts=[Text(contents=["Hallo Welt"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["voice_name"] == "German-Female-1"
        assert payload["language_code"] == "de-DE"
        assert payload["encoding"] == "FLAC"
        assert payload["sample_rate_hz"] == 44100


class TestRivaTtsStreamingEndpointParseResponse:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_TTS_STREAMING, streaming=True
        )
        return create_endpoint_with_mock_transport(
            RivaTtsStreamingEndpoint, model_endpoint
        )

    def test_parse_response_with_audio(self, endpoint) -> None:
        audio = b"\x00\x01" * 50
        audio_b64 = base64.b64encode(audio).decode()
        response = create_mock_response(json_data={"audio": audio_b64})

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert isinstance(parsed.data, AudioResponseData)
        assert parsed.data.audio_bytes == audio

    def test_parse_response_no_json(self, endpoint) -> None:
        response = create_mock_response(json_data=None)
        assert endpoint.parse_response(response) is None

    def test_parse_response_no_audio(self, endpoint) -> None:
        response = create_mock_response(json_data={})
        assert endpoint.parse_response(response) is None
