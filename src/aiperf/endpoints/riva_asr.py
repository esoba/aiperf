# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Riva ASR (Automatic Speech Recognition) endpoints for offline and streaming."""

from __future__ import annotations

import base64
import binascii
from typing import Any

from aiperf.common.models import InferenceServerResponse, ParsedResponse, RequestInfo
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.riva_helpers import get_extra


def _extract_audio_bytes(request_info: RequestInfo) -> bytes:
    """Extract audio bytes from the first audio content in the turn.

    Args:
        request_info: Request context with audio data in turns[0].audios.

    Returns:
        Audio bytes.

    Raises:
        ValueError: If no audio data found.
    """
    if not request_info.turns:
        raise ValueError("Riva ASR endpoint requires at least one turn.")

    turn = request_info.turns[0]
    if not turn.audios:
        raise ValueError("Riva ASR endpoint requires audio data in the turn.")

    audio_content = turn.audios[0].contents[0] if turn.audios[0].contents else ""
    if not audio_content:
        raise ValueError("Riva ASR endpoint requires non-empty audio content.")

    # Audio content may be base64-encoded or raw bytes stored as string
    try:
        return base64.b64decode(audio_content)
    except (ValueError, binascii.Error):
        return (
            audio_content.encode("utf-8")
            if isinstance(audio_content, str)
            else audio_content
        )


def _parse_asr_response(
    response: InferenceServerResponse, endpoint: BaseEndpoint
) -> ParsedResponse | None:
    """Parse a Riva ASR response into TextResponseData with transcript."""
    json_obj = response.get_json()
    if not json_obj:
        return None

    transcript = json_obj.get("transcript", "")
    if not transcript:
        return None

    return ParsedResponse(
        perf_ns=response.perf_ns,
        data=endpoint.make_text_response_data(transcript),
    )


class RivaAsrOfflineEndpoint(BaseEndpoint):
    """Riva ASR offline endpoint for batch speech recognition.

    Sends complete audio to Riva ASR and returns the full transcript.
    Configure via --extra: language_code, sample_rate_hertz, encoding.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = get_extra(self)
        self._language_code: str = extra.get("language_code", "en-US")
        self._sample_rate_hertz: int = int(extra.get("sample_rate_hertz", 16000))
        self._encoding: str = extra.get("encoding", "LINEAR_PCM")

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format Riva ASR offline request payload with audio data.

        Args:
            request_info: Request context with audio in turns[0].audios.

        Returns:
            Dict payload for ASR recognition.
        """
        audio_bytes = _extract_audio_bytes(request_info)

        return {
            "audio": audio_bytes,
            "language_code": self._language_code,
            "sample_rate_hertz": self._sample_rate_hertz,
            "encoding": self._encoding,
        }

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse Riva ASR offline response into TextResponseData.

        Args:
            response: Raw response from inference server.

        Returns:
            ParsedResponse with transcript text, or None.
        """
        return _parse_asr_response(response, self)


class RivaAsrStreamingEndpoint(BaseEndpoint):
    """Riva ASR streaming endpoint for bidirectional speech recognition.

    Streams audio chunks to Riva ASR and receives transcript results
    as they become available. Uses bidi streaming gRPC.
    Configure via --extra: language_code, sample_rate_hertz, encoding, chunk_size.
    """

    DEFAULT_CHUNK_SIZE = 8000  # ~0.5s of 16kHz 16-bit audio

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = get_extra(self)
        self._language_code: str = extra.get("language_code", "en-US")
        self._sample_rate_hertz: int = int(extra.get("sample_rate_hertz", 16000))
        self._encoding: str = extra.get("encoding", "LINEAR_PCM")
        self._chunk_size: int = int(extra.get("chunk_size", self.DEFAULT_CHUNK_SIZE))

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format Riva ASR streaming request payload with audio chunks.

        The payload includes config for the first message and chunked audio
        for subsequent messages. The transport's bidi streaming handler
        reads ``audio_chunks`` to send individual stream messages.

        Args:
            request_info: Request context with audio in turns[0].audios.

        Returns:
            Dict payload with config and audio_chunks for bidi streaming.
        """
        audio_bytes = _extract_audio_bytes(request_info)

        chunks = [
            audio_bytes[i : i + self._chunk_size]
            for i in range(0, len(audio_bytes), self._chunk_size)
        ]

        return {
            "language_code": self._language_code,
            "sample_rate_hertz": self._sample_rate_hertz,
            "encoding": self._encoding,
            "interim_results": True,
            "audio_chunks": chunks,
        }

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse Riva ASR streaming response into TextResponseData.

        Args:
            response: Raw response chunk from inference server.

        Returns:
            ParsedResponse with transcript text, or None.
        """
        return _parse_asr_response(response, self)
