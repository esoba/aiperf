# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Riva NLP gRPC serializers for language understanding services.

One serializer class per NLP API. Each implements GrpcSerializerProtocol.
Discovered via plugins.yaml endpoint metadata (``grpc.serializer``).
"""

from __future__ import annotations

from typing import Any

from aiperf.transports.grpc.proto.riva import riva_nlp_pb2
from aiperf.transports.grpc.stream_chunk import StreamChunk


def _make_nlp_model_params(
    payload: dict[str, Any], model_name: str
) -> riva_nlp_pb2.NLPModelParams:
    """Build NLPModelParams from payload and model_name.

    Args:
        payload: Dict with optional model_name and language_code.
        model_name: Fallback model name.

    Returns:
        NLPModelParams protobuf.
    """
    params = riva_nlp_pb2.NLPModelParams()
    params.model_name = payload.get("model_name", model_name)
    params.language_code = payload.get("language_code", "en-US")
    return params


def _not_streaming(data: bytes) -> StreamChunk:
    """Return error StreamChunk for NLP endpoints (no streaming support)."""
    return StreamChunk(
        error_message="NLP endpoints do not support streaming responses",
        response_dict=None,
        response_size=len(data),
    )


def _serialize_text_list_request(
    request: Any,
    payload: dict[str, Any],
    model_name: str,
    request_id: str,
) -> bytes:
    """Serialize a text-list NLP request (TextClass, TokenClass, TextTransform).

    Args:
        request: Protobuf request object with text, top_n, model, and id fields.
        payload: Dict with texts, top_n, model_name, language_code.
        model_name: Fallback model name.
        request_id: Optional request ID.

    Returns:
        Serialized protobuf bytes.
    """
    for text in payload.get("texts", []):
        request.text.append(text)
    request.top_n = payload.get("top_n", 0)
    request.model.CopyFrom(_make_nlp_model_params(payload, model_name))
    if request_id:
        request.id.value = request_id
    return request.SerializeToString()


class RivaTextClassifySerializer:
    """Serializer for Riva ClassifyText API."""

    @staticmethod
    def serialize_request(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        return _serialize_text_list_request(
            riva_nlp_pb2.TextClassRequest(), payload, model_name, request_id
        )

    @staticmethod
    def deserialize_response(data: bytes) -> tuple[dict[str, Any], int]:
        response = riva_nlp_pb2.TextClassResponse()
        response.ParseFromString(data)
        results = []
        for result in response.results:
            labels = [
                {"class_name": label.class_name, "score": label.score}
                for label in result.labels
            ]
            results.append({"labels": labels})
        return {"results": results}, len(data)

    @staticmethod
    def deserialize_stream_response(data: bytes) -> StreamChunk:
        return _not_streaming(data)


class RivaTokenClassifySerializer:
    """Serializer for Riva ClassifyTokens API."""

    @staticmethod
    def serialize_request(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        return _serialize_text_list_request(
            riva_nlp_pb2.TokenClassRequest(), payload, model_name, request_id
        )

    @staticmethod
    def deserialize_response(data: bytes) -> tuple[dict[str, Any], int]:
        response = riva_nlp_pb2.TokenClassResponse()
        response.ParseFromString(data)
        results = []
        for seq in response.results:
            tokens = []
            for tv in seq.results:
                labels = [
                    {"class_name": lbl.class_name, "score": lbl.score}
                    for lbl in tv.label
                ]
                tokens.append({"token": tv.token, "labels": labels})
            results.append({"tokens": tokens})
        return {"results": results}, len(data)

    @staticmethod
    def deserialize_stream_response(data: bytes) -> StreamChunk:
        return _not_streaming(data)


class RivaTransformTextSerializer:
    """Serializer for Riva TransformText API."""

    @staticmethod
    def serialize_request(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        return _serialize_text_list_request(
            riva_nlp_pb2.TextTransformRequest(), payload, model_name, request_id
        )

    @staticmethod
    def deserialize_response(data: bytes) -> tuple[dict[str, Any], int]:
        response = riva_nlp_pb2.TextTransformResponse()
        response.ParseFromString(data)
        return {"texts": list(response.text)}, len(data)

    @staticmethod
    def deserialize_stream_response(data: bytes) -> StreamChunk:
        return _not_streaming(data)


class RivaPunctuateTextSerializer:
    """Serializer for Riva PunctuateText API.

    Uses same request/response types as TransformText but different method path.
    """

    @staticmethod
    def serialize_request(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        return RivaTransformTextSerializer.serialize_request(
            payload, model_name, request_id
        )

    @staticmethod
    def deserialize_response(data: bytes) -> tuple[dict[str, Any], int]:
        return RivaTransformTextSerializer.deserialize_response(data)

    @staticmethod
    def deserialize_stream_response(data: bytes) -> StreamChunk:
        return _not_streaming(data)


class RivaNaturalQuerySerializer:
    """Serializer for Riva NaturalQuery API."""

    @staticmethod
    def serialize_request(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        request = riva_nlp_pb2.NaturalQueryRequest()
        request.query = payload.get("query", "")
        request.context = payload.get("context", "")
        request.top_n = payload.get("top_n", 1)
        if request_id:
            request.id.value = request_id
        return request.SerializeToString()

    @staticmethod
    def deserialize_response(data: bytes) -> tuple[dict[str, Any], int]:
        response = riva_nlp_pb2.NaturalQueryResponse()
        response.ParseFromString(data)
        results = [{"answer": r.answer, "score": r.score} for r in response.results]
        return {"results": results}, len(data)

    @staticmethod
    def deserialize_stream_response(data: bytes) -> StreamChunk:
        return _not_streaming(data)


class RivaAnalyzeIntentSerializer:
    """Serializer for Riva AnalyzeIntent API."""

    @staticmethod
    def serialize_request(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        request = riva_nlp_pb2.AnalyzeIntentRequest()
        request.query = payload.get("query", "")
        if payload.get("domain"):
            request.options.domain = payload["domain"]
        if request_id:
            request.id.value = request_id
        return request.SerializeToString()

    @staticmethod
    def deserialize_response(data: bytes) -> tuple[dict[str, Any], int]:
        response = riva_nlp_pb2.AnalyzeIntentResponse()
        response.ParseFromString(data)
        slots = [
            {
                "token": s.token,
                "labels": [
                    {"class_name": lbl.class_name, "score": lbl.score}
                    for lbl in s.label
                ],
            }
            for s in response.slots
        ]
        result: dict[str, Any] = {
            "intent": {
                "class_name": response.intent.class_name,
                "score": response.intent.score,
            },
            "slots": slots,
        }
        if response.domain.class_name:
            result["domain"] = {
                "class_name": response.domain.class_name,
                "score": response.domain.score,
            }
        return result, len(data)

    @staticmethod
    def deserialize_stream_response(data: bytes) -> StreamChunk:
        return _not_streaming(data)


class RivaAnalyzeEntitiesSerializer:
    """Serializer for Riva AnalyzeEntities API (returns TokenClassResponse)."""

    @staticmethod
    def serialize_request(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        request = riva_nlp_pb2.AnalyzeEntitiesRequest()
        request.query = payload.get("query", "")
        if request_id:
            request.id.value = request_id
        return request.SerializeToString()

    @staticmethod
    def deserialize_response(data: bytes) -> tuple[dict[str, Any], int]:
        # AnalyzeEntities returns TokenClassResponse
        return RivaTokenClassifySerializer.deserialize_response(data)

    @staticmethod
    def deserialize_stream_response(data: bytes) -> StreamChunk:
        return _not_streaming(data)
