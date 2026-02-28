# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Riva ASR endpoints."""

from __future__ import annotations

import base64

import pytest

from aiperf.common.models import Turn
from aiperf.common.models.dataset_models import Audio
from aiperf.endpoints.riva_asr import RivaAsrOfflineEndpoint, RivaAsrStreamingEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


def _make_audio_turn(audio_bytes: bytes) -> Turn:
    """Create a turn with base64-encoded audio data."""
    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    return Turn(audios=[Audio(contents=[encoded])])


# ---------------------------------------------------------------------------
# RivaAsrOfflineEndpoint.format_payload
# ---------------------------------------------------------------------------
class TestRivaAsrOfflineFormatPayload:
    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.RIVA_ASR_OFFLINE,
            model_name="asr_model",
            extra=[("language_code", "en-US"), ("sample_rate_hertz", 16000)],
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            RivaAsrOfflineEndpoint, model_endpoint
        )

    def test_format_payload_with_audio(self, endpoint, model_endpoint) -> None:
        audio = b"\x00\x01\x02\x03" * 100
        turn = _make_audio_turn(audio)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["audio"] == audio
        assert payload["language_code"] == "en-US"
        assert payload["sample_rate_hertz"] == 16000
        assert payload["encoding"] == "LINEAR_PCM"

    def test_format_payload_empty_turns_raises(self, endpoint, model_endpoint) -> None:
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])
        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_no_audio_raises(self, endpoint, model_endpoint) -> None:
        turn = Turn()
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        with pytest.raises(ValueError, match="requires audio data"):
            endpoint.format_payload(request_info)

    def test_format_payload_default_config(self) -> None:
        """Endpoint with no extra config should use defaults."""
        model_endpoint = create_model_endpoint(EndpointType.RIVA_ASR_OFFLINE)
        endpoint = create_endpoint_with_mock_transport(
            RivaAsrOfflineEndpoint, model_endpoint
        )
        audio = b"\x01\x02"
        turn = _make_audio_turn(audio)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["language_code"] == "en-US"
        assert payload["sample_rate_hertz"] == 16000
        assert payload["encoding"] == "LINEAR_PCM"

    def test_format_payload_custom_encoding(self) -> None:
        """Custom encoding from extra config should be used."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_ASR_OFFLINE,
            extra=[("encoding", "FLAC")],
        )
        endpoint = create_endpoint_with_mock_transport(
            RivaAsrOfflineEndpoint, model_endpoint
        )
        turn = _make_audio_turn(b"\x01\x02")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["encoding"] == "FLAC"

    def test_format_payload_large_audio(self, endpoint, model_endpoint) -> None:
        """Large audio payloads should be handled correctly."""
        audio = b"\xff" * 1_000_000
        turn = _make_audio_turn(audio)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["audio"]) == 1_000_000


# ---------------------------------------------------------------------------
# RivaAsrOfflineEndpoint.parse_response
# ---------------------------------------------------------------------------
class TestRivaAsrOfflineParseResponse:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.RIVA_ASR_OFFLINE)
        return create_endpoint_with_mock_transport(
            RivaAsrOfflineEndpoint, model_endpoint
        )

    def test_parse_response_with_transcript(self, endpoint) -> None:
        response = create_mock_response(json_data={"transcript": "hello world"})
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "hello world"

    def test_parse_response_no_json(self, endpoint) -> None:
        response = create_mock_response(json_data=None)
        assert endpoint.parse_response(response) is None

    def test_parse_response_empty_transcript(self, endpoint) -> None:
        response = create_mock_response(json_data={"transcript": ""})
        assert endpoint.parse_response(response) is None

    def test_parse_response_no_transcript_key(self, endpoint) -> None:
        """Missing transcript key should return None."""
        response = create_mock_response(json_data={"results": []})
        assert endpoint.parse_response(response) is None

    def test_parse_response_preserves_perf_ns(self, endpoint) -> None:
        """perf_ns from response should be preserved."""
        response = create_mock_response(perf_ns=42000, json_data={"transcript": "test"})
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.perf_ns == 42000

    def test_parse_response_long_transcript(self, endpoint) -> None:
        """Long transcripts should be preserved."""
        long_text = "word " * 1000
        response = create_mock_response(json_data={"transcript": long_text.strip()})
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert len(parsed.data.get_text()) > 4000


# ---------------------------------------------------------------------------
# RivaAsrStreamingEndpoint.format_payload
# ---------------------------------------------------------------------------
class TestRivaAsrStreamingFormatPayload:
    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.RIVA_ASR_STREAMING,
            model_name="asr_model",
            streaming=True,
            extra=[("chunk_size", 4000)],
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            RivaAsrStreamingEndpoint, model_endpoint
        )

    def test_format_payload_chunks_audio(self, endpoint, model_endpoint) -> None:
        audio = b"\x00" * 10000
        turn = _make_audio_turn(audio)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "audio_chunks" in payload
        # 10000 bytes / 4000 chunk_size = 3 chunks (2 full + 1 partial)
        assert len(payload["audio_chunks"]) == 3
        assert len(payload["audio_chunks"][0]) == 4000
        assert len(payload["audio_chunks"][1]) == 4000
        assert len(payload["audio_chunks"][2]) == 2000
        assert payload["interim_results"] is True

    def test_format_payload_small_audio_single_chunk(
        self, endpoint, model_endpoint
    ) -> None:
        audio = b"\x00" * 100
        turn = _make_audio_turn(audio)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["audio_chunks"]) == 1
        assert len(payload["audio_chunks"][0]) == 100

    def test_format_payload_exact_chunk_size(self, endpoint, model_endpoint) -> None:
        """Audio exactly divisible by chunk_size should produce no partial chunk."""
        audio = b"\x00" * 8000  # 8000 / 4000 = 2 exact chunks
        turn = _make_audio_turn(audio)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["audio_chunks"]) == 2
        assert all(len(c) == 4000 for c in payload["audio_chunks"])

    def test_format_payload_default_chunk_size(self) -> None:
        """Default chunk size should be DEFAULT_CHUNK_SIZE (8000)."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_ASR_STREAMING, streaming=True
        )
        endpoint = create_endpoint_with_mock_transport(
            RivaAsrStreamingEndpoint, model_endpoint
        )
        audio = b"\x00" * 20000  # 20000 / 8000 = 3 chunks (2 full + 1 partial)
        turn = _make_audio_turn(audio)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["audio_chunks"]) == 3
        assert len(payload["audio_chunks"][0]) == 8000
        assert len(payload["audio_chunks"][1]) == 8000
        assert len(payload["audio_chunks"][2]) == 4000

    def test_format_payload_includes_config(self, endpoint, model_endpoint) -> None:
        """Streaming payload should include recognition config fields."""
        audio = b"\x01\x02"
        turn = _make_audio_turn(audio)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["language_code"] == "en-US"
        assert payload["sample_rate_hertz"] == 16000
        assert payload["encoding"] == "LINEAR_PCM"

    def test_format_payload_custom_language(self) -> None:
        """Custom language_code should be used."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_ASR_STREAMING,
            streaming=True,
            extra=[("language_code", "es-ES")],
        )
        endpoint = create_endpoint_with_mock_transport(
            RivaAsrStreamingEndpoint, model_endpoint
        )
        turn = _make_audio_turn(b"\x01")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["language_code"] == "es-ES"

    def test_format_payload_empty_turns_raises(self, endpoint, model_endpoint) -> None:
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])
        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_no_audio_raises(self, endpoint, model_endpoint) -> None:
        turn = Turn()
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        with pytest.raises(ValueError, match="requires audio data"):
            endpoint.format_payload(request_info)

    def test_chunk_contents_reconstruct_original(
        self, endpoint, model_endpoint
    ) -> None:
        """Concatenating all chunks should reproduce the original audio."""
        audio = b"\x00\x01\x02\x03\x04\x05\x06\x07" * 1250  # 10000 bytes
        turn = _make_audio_turn(audio)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        reconstructed = b"".join(payload["audio_chunks"])
        assert reconstructed == audio


# ---------------------------------------------------------------------------
# RivaAsrStreamingEndpoint.parse_response
# ---------------------------------------------------------------------------
class TestRivaAsrStreamingParseResponse:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_ASR_STREAMING, streaming=True
        )
        return create_endpoint_with_mock_transport(
            RivaAsrStreamingEndpoint, model_endpoint
        )

    def test_parse_response_with_transcript(self, endpoint) -> None:
        response = create_mock_response(
            json_data={"transcript": "hello world", "is_final": True}
        )
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "hello world"

    def test_parse_response_no_json(self, endpoint) -> None:
        response = create_mock_response(json_data=None)
        assert endpoint.parse_response(response) is None

    def test_parse_response_empty_transcript(self, endpoint) -> None:
        """Empty transcript should return None."""
        response = create_mock_response(json_data={"transcript": ""})
        assert endpoint.parse_response(response) is None

    def test_parse_response_no_transcript_key(self, endpoint) -> None:
        """Missing transcript key should return None."""
        response = create_mock_response(json_data={"is_final": True})
        assert endpoint.parse_response(response) is None

    def test_parse_response_preserves_perf_ns(self, endpoint) -> None:
        response = create_mock_response(perf_ns=12345, json_data={"transcript": "test"})
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.perf_ns == 12345
