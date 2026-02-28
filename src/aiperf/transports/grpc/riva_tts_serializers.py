# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Riva TTS gRPC serializer for text-to-speech synthesis.

Converts between endpoint dict payloads and Riva TTS protobuf wire bytes.
Discovered via plugins.yaml endpoint metadata (``grpc.serializer``).
"""

from __future__ import annotations

from typing import Any

from aiperf.transports.grpc.proto.riva import riva_audio_pb2, riva_tts_pb2
from aiperf.transports.grpc.riva_encoding import ENCODING_MAP
from aiperf.transports.grpc.stream_chunk import StreamChunk


def _response_to_dict(response: Any) -> dict[str, Any]:
    """Convert a SynthesizeSpeechResponse to a dict."""
    result: dict[str, Any] = {"audio": response.audio}
    if response.meta.text:
        result["meta"] = {
            "text": response.meta.text,
            "processed_text": response.meta.processed_text,
        }
    return result


class RivaTtsSerializer:
    """Riva TTS gRPC serializer for the generic gRPC transport.

    Converts between endpoint dict payloads and Riva SynthesizeSpeech
    protobuf wire bytes. Used for both batch and streaming TTS.
    """

    @staticmethod
    def serialize_request(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        """Convert a dict payload to serialized SynthesizeSpeechRequest bytes.

        Args:
            payload: Dict with text, voice_name, language_code, encoding, sample_rate_hz.
            model_name: Not used by Riva TTS (voice_name is used instead).
            request_id: Optional request ID.

        Returns:
            Serialized protobuf bytes.
        """
        request = riva_tts_pb2.SynthesizeSpeechRequest()
        request.text = payload.get("text", "")
        request.language_code = payload.get("language_code", "en-US")
        request.voice_name = payload.get("voice_name", "")
        request.sample_rate_hz = payload.get("sample_rate_hz", 22050)

        encoding_str = payload.get("encoding", "LINEAR_PCM")
        request.encoding = ENCODING_MAP.get(encoding_str, riva_audio_pb2.LINEAR_PCM)

        if request_id:
            request.id.value = request_id

        return request.SerializeToString()

    @staticmethod
    def deserialize_response(data: bytes) -> tuple[dict[str, Any], int]:
        """Deserialize SynthesizeSpeechResponse bytes to a dict and wire size.

        Args:
            data: Raw bytes from the gRPC wire.

        Returns:
            Tuple of (response dict with audio bytes, wire size).
        """
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        response.ParseFromString(data)
        return _response_to_dict(response), len(data)

    @staticmethod
    def deserialize_stream_response(data: bytes) -> StreamChunk:
        """Deserialize streaming SynthesizeSpeechResponse bytes to a StreamChunk.

        Args:
            data: Raw bytes from the gRPC wire.

        Returns:
            StreamChunk with audio response data.
        """
        response = riva_tts_pb2.SynthesizeSpeechResponse()
        response.ParseFromString(data)
        return StreamChunk(
            error_message=None,
            response_dict=_response_to_dict(response),
            response_size=len(data),
        )
