# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Riva TTS (Text-to-Speech) endpoints for batch and streaming synthesis."""

from __future__ import annotations

import base64
from typing import Any

from aiperf.common.models import InferenceServerResponse, ParsedResponse, RequestInfo
from aiperf.common.models.record_models import AudioResponseData
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.riva_helpers import get_extra


def _calc_duration_ms(
    audio_bytes: bytes, sample_rate_hz: int, encoding: str
) -> float | None:
    """Calculate audio duration from PCM audio bytes."""
    if encoding != "LINEAR_PCM" or sample_rate_hz <= 0 or not audio_bytes:
        return None
    # LINEAR_PCM is 16-bit (2 bytes per sample), mono
    num_samples = len(audio_bytes) / 2
    return (num_samples / sample_rate_hz) * 1000.0


def _parse_tts_response(
    response: InferenceServerResponse,
    sample_rate_hz: int,
    encoding: str,
) -> ParsedResponse | None:
    """Parse a Riva TTS response into a ParsedResponse with AudioResponseData."""
    json_obj = response.get_json()
    if not json_obj:
        return None

    audio = json_obj.get("audio")
    if not audio:
        return None

    # Audio may be bytes or base64-encoded string from JSON serialization
    if isinstance(audio, str):
        audio_bytes = base64.b64decode(audio)
    else:
        audio_bytes = bytes(audio) if not isinstance(audio, bytes) else audio

    duration_ms = _calc_duration_ms(audio_bytes, sample_rate_hz, encoding)

    return ParsedResponse(
        perf_ns=response.perf_ns,
        data=AudioResponseData(
            audio_bytes=audio_bytes,
            sample_rate_hz=sample_rate_hz,
            encoding=encoding,
            duration_ms=duration_ms,
        ),
    )


class _RivaTtsBaseEndpoint(BaseEndpoint):
    """Shared base for Riva TTS batch and streaming endpoints."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = get_extra(self)
        self._voice_name: str = extra.get("voice_name", "")
        self._language_code: str = extra.get("language_code", "en-US")
        self._encoding: str = extra.get("encoding", "LINEAR_PCM")
        self._sample_rate_hz: int = int(extra.get("sample_rate_hz", 22050))

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format Riva TTS request payload from text input.

        Args:
            request_info: Request context with text in turns[0].texts.

        Returns:
            Dict payload for TTS synthesis.
        """
        if not request_info.turns:
            raise ValueError("Riva TTS endpoint requires at least one turn.")

        turn = request_info.turns[0]
        prompts = [
            content for text in turn.texts for content in text.contents if content
        ]
        text = " ".join(prompts) if prompts else ""

        return {
            "text": text,
            "voice_name": self._voice_name,
            "language_code": self._language_code,
            "encoding": self._encoding,
            "sample_rate_hz": self._sample_rate_hz,
        }

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse Riva TTS response into AudioResponseData.

        Args:
            response: Raw response from inference server.

        Returns:
            ParsedResponse with AudioResponseData, or None.
        """
        return _parse_tts_response(response, self._sample_rate_hz, self._encoding)


class RivaTtsEndpoint(_RivaTtsBaseEndpoint):
    """Riva TTS batch endpoint for text-to-speech synthesis.

    Sends text to Riva TTS and returns synthesized audio.
    Configure via --extra: voice_name, language_code, encoding, sample_rate_hz.
    """


class RivaTtsStreamingEndpoint(_RivaTtsBaseEndpoint):
    """Riva TTS streaming endpoint for text-to-speech synthesis.

    Same as batch but uses server-streaming RPC (SynthesizeOnline)
    to receive audio chunks as they are generated.
    """
