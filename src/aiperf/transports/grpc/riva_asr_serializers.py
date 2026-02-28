# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Riva ASR gRPC serializers for offline and streaming speech recognition.

Converts between endpoint dict payloads and Riva ASR protobuf wire bytes.
Discovered via plugins.yaml endpoint metadata (``grpc.serializer``).
"""

from __future__ import annotations

from typing import Any

from aiperf.transports.grpc.proto.riva import riva_asr_pb2, riva_audio_pb2
from aiperf.transports.grpc.stream_chunk import StreamChunk

_ENCODING_MAP: dict[str, int] = {
    "LINEAR_PCM": riva_audio_pb2.LINEAR_PCM,
    "FLAC": riva_audio_pb2.FLAC,
    "MULAW": riva_audio_pb2.MULAW,
    "OGGOPUS": riva_audio_pb2.OGGOPUS,
    "ALAW": riva_audio_pb2.ALAW,
}


def _build_recognition_config(
    payload: dict[str, Any],
) -> riva_asr_pb2.RecognitionConfig:
    """Build a RecognitionConfig from the payload dict.

    Args:
        payload: Dict with encoding, sample_rate_hertz, language_code, model, etc.

    Returns:
        RecognitionConfig protobuf.
    """
    config = riva_asr_pb2.RecognitionConfig()
    encoding_str = payload.get("encoding", "LINEAR_PCM")
    config.encoding = _ENCODING_MAP.get(encoding_str, riva_audio_pb2.LINEAR_PCM)
    config.sample_rate_hertz = payload.get("sample_rate_hertz", 16000)
    config.language_code = payload.get("language_code", "en-US")
    config.max_alternatives = payload.get("max_alternatives", 1)
    config.enable_automatic_punctuation = payload.get(
        "enable_automatic_punctuation", True
    )
    if payload.get("model"):
        config.model = payload["model"]
    return config


def _extract_transcript(results: Any) -> str:
    """Extract the top transcript from ASR results.

    Args:
        results: Repeated SpeechRecognitionResult or StreamingRecognitionResult.

    Returns:
        Concatenated transcript text.
    """
    transcripts = []
    for result in results:
        if result.alternatives:
            transcripts.append(result.alternatives[0].transcript)
    return " ".join(transcripts)


class RivaAsrOfflineSerializer:
    """Riva ASR offline (batch) gRPC serializer.

    Serializes RecognizeRequest and deserializes RecognizeResponse.
    """

    @staticmethod
    def serialize_request(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        """Convert dict payload to serialized RecognizeRequest bytes.

        Args:
            payload: Dict with audio (bytes), encoding, sample_rate_hertz, language_code.
            model_name: Used as the model field in RecognitionConfig if no explicit model.
            request_id: Optional request ID.

        Returns:
            Serialized protobuf bytes.
        """
        request = riva_asr_pb2.RecognizeRequest()
        request.config.CopyFrom(_build_recognition_config(payload))
        if not request.config.model and model_name:
            request.config.model = model_name

        audio = payload.get("audio", b"")
        request.audio = audio if isinstance(audio, bytes) else audio.encode("utf-8")

        if request_id:
            request.id.value = request_id

        return request.SerializeToString()

    @staticmethod
    def deserialize_response(data: bytes) -> tuple[dict[str, Any], int]:
        """Deserialize RecognizeResponse bytes to a dict and wire size.

        Args:
            data: Raw bytes from the gRPC wire.

        Returns:
            Tuple of (response dict with transcript, wire size).
        """
        response = riva_asr_pb2.RecognizeResponse()
        response.ParseFromString(data)

        transcript = _extract_transcript(response.results)

        result: dict[str, Any] = {
            "transcript": transcript,
            "results": [
                {
                    "alternatives": [
                        {
                            "transcript": alt.transcript,
                            "confidence": alt.confidence,
                        }
                        for alt in r.alternatives
                    ],
                }
                for r in response.results
            ],
        }
        return result, len(data)

    @staticmethod
    def deserialize_stream_response(data: bytes) -> StreamChunk:
        """Not used for offline ASR but required by protocol."""
        return StreamChunk(
            error_message="Offline ASR does not support streaming responses",
            response_dict=None,
            response_size=len(data),
        )


class RivaAsrStreamingSerializer:
    """Riva ASR streaming gRPC serializer for bidirectional streaming.

    Implements the bidi streaming protocol: config message first, then audio chunks.
    """

    @staticmethod
    def serialize_request(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        """Serialize the full audio as a single RecognizeRequest (fallback for unary).

        Args:
            payload: Dict with audio, encoding, sample_rate_hertz, language_code.
            model_name: Model name.
            request_id: Optional request ID.

        Returns:
            Serialized RecognizeRequest bytes.
        """
        request = riva_asr_pb2.RecognizeRequest()
        request.config.CopyFrom(_build_recognition_config(payload))
        if not request.config.model and model_name:
            request.config.model = model_name

        audio = payload.get("audio", b"")
        request.audio = audio if isinstance(audio, bytes) else audio.encode("utf-8")

        if request_id:
            request.id.value = request_id

        return request.SerializeToString()

    @staticmethod
    def serialize_stream_config(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        """Serialize the first StreamingRecognizeRequest with config.

        Args:
            payload: Dict with encoding, sample_rate_hertz, language_code.
            model_name: Model name for recognition config.
            request_id: Optional request ID.

        Returns:
            Serialized StreamingRecognizeRequest with streaming_config.
        """
        config = _build_recognition_config(payload)
        if not config.model and model_name:
            config.model = model_name

        streaming_config = riva_asr_pb2.StreamingRecognitionConfig()
        streaming_config.config.CopyFrom(config)
        streaming_config.interim_results = payload.get("interim_results", True)

        request = riva_asr_pb2.StreamingRecognizeRequest()
        request.streaming_config.CopyFrom(streaming_config)

        if request_id:
            request.id.value = request_id

        return request.SerializeToString()

    @staticmethod
    def serialize_stream_chunk(chunk_data: bytes) -> bytes:
        """Serialize an audio chunk as a StreamingRecognizeRequest.

        Args:
            chunk_data: Raw audio bytes for this chunk.

        Returns:
            Serialized StreamingRecognizeRequest with audio_content.
        """
        request = riva_asr_pb2.StreamingRecognizeRequest()
        request.audio_content = chunk_data
        return request.SerializeToString()

    @staticmethod
    def deserialize_response(data: bytes) -> tuple[dict[str, Any], int]:
        """Deserialize RecognizeResponse bytes (fallback for unary)."""
        response = riva_asr_pb2.RecognizeResponse()
        response.ParseFromString(data)
        transcript = _extract_transcript(response.results)
        return {"transcript": transcript}, len(data)

    @staticmethod
    def deserialize_stream_response(data: bytes) -> StreamChunk:
        """Deserialize a server-stream response (not used for bidi, but required)."""
        return RivaAsrStreamingSerializer.deserialize_bidi_response(data)

    @staticmethod
    def deserialize_bidi_response(data: bytes) -> StreamChunk:
        """Deserialize a StreamingRecognizeResponse from the bidi stream.

        Args:
            data: Raw bytes from the gRPC wire.

        Returns:
            StreamChunk with transcript and is_final status.
        """
        response = riva_asr_pb2.StreamingRecognizeResponse()
        response.ParseFromString(data)

        results = []
        is_final = False
        for result in response.results:
            is_final = is_final or result.is_final
            alts = [
                {
                    "transcript": alt.transcript,
                    "confidence": alt.confidence,
                }
                for alt in result.alternatives
            ]
            results.append(
                {
                    "alternatives": alts,
                    "is_final": result.is_final,
                    "stability": result.stability,
                }
            )

        transcript = _extract_transcript(response.results)

        return StreamChunk(
            error_message=None,
            response_dict={
                "transcript": transcript,
                "is_final": is_final,
                "results": results,
            },
            response_size=len(data),
        )
