# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Riva NLP serializers."""

from __future__ import annotations

import pytest

from aiperf.transports.grpc.grpc_transport import GrpcSerializerProtocol
from aiperf.transports.grpc.proto.riva import riva_nlp_pb2
from aiperf.transports.grpc.riva_nlp_serializers import (
    RivaAnalyzeEntitiesSerializer,
    RivaAnalyzeIntentSerializer,
    RivaNaturalQuerySerializer,
    RivaPunctuateTextSerializer,
    RivaTextClassifySerializer,
    RivaTokenClassifySerializer,
    RivaTransformTextSerializer,
    _serialize_text_list_request,
)
from aiperf.transports.grpc.stream_chunk import StreamChunk


class TestProtocolConformance:
    """All NLP serializers should implement GrpcSerializerProtocol."""

    @pytest.mark.parametrize(
        "serializer_cls",
        [
            RivaTextClassifySerializer,
            RivaTokenClassifySerializer,
            RivaTransformTextSerializer,
            RivaPunctuateTextSerializer,
            RivaNaturalQuerySerializer,
            RivaAnalyzeIntentSerializer,
            RivaAnalyzeEntitiesSerializer,
        ],
    )
    def test_implements_protocol(self, serializer_cls: type) -> None:
        assert isinstance(serializer_cls(), GrpcSerializerProtocol)


# ---------------------------------------------------------------------------
# TextClassifySerializer
# ---------------------------------------------------------------------------
class TestTextClassifySerializeRequest:
    def test_roundtrip(self) -> None:
        payload = {
            "texts": ["This is great!", "Terrible product."],
            "language_code": "en-US",
        }
        data = RivaTextClassifySerializer.serialize_request(
            payload, model_name="classify_model", request_id="r1"
        )

        parsed = riva_nlp_pb2.TextClassRequest()
        parsed.ParseFromString(data)
        assert list(parsed.text) == ["This is great!", "Terrible product."]
        assert parsed.model.model_name == "classify_model"
        assert parsed.model.language_code == "en-US"
        assert parsed.id.value == "r1"

    def test_empty_texts(self) -> None:
        """Empty texts list should produce valid request."""
        data = RivaTextClassifySerializer.serialize_request(
            {"texts": []}, model_name="m"
        )
        parsed = riva_nlp_pb2.TextClassRequest()
        parsed.ParseFromString(data)
        assert list(parsed.text) == []

    def test_no_texts_key(self) -> None:
        """Missing texts key should default to empty."""
        data = RivaTextClassifySerializer.serialize_request({}, model_name="m")
        parsed = riva_nlp_pb2.TextClassRequest()
        parsed.ParseFromString(data)
        assert list(parsed.text) == []

    def test_no_request_id(self) -> None:
        """No request_id should not set id field."""
        data = RivaTextClassifySerializer.serialize_request(
            {"texts": ["hi"]}, model_name="m"
        )
        parsed = riva_nlp_pb2.TextClassRequest()
        parsed.ParseFromString(data)
        assert not parsed.HasField("id")

    def test_top_n(self) -> None:
        """top_n should be passed through."""
        payload = {"texts": ["test"], "top_n": 5}
        data = RivaTextClassifySerializer.serialize_request(payload, model_name="m")
        parsed = riva_nlp_pb2.TextClassRequest()
        parsed.ParseFromString(data)
        assert parsed.top_n == 5

    def test_model_name_from_payload(self) -> None:
        """model_name from payload should override argument."""
        payload = {"texts": ["test"], "model_name": "custom"}
        data = RivaTextClassifySerializer.serialize_request(
            payload, model_name="default"
        )
        parsed = riva_nlp_pb2.TextClassRequest()
        parsed.ParseFromString(data)
        assert parsed.model.model_name == "custom"

    def test_default_language_code(self) -> None:
        """Default language_code should be en-US."""
        data = RivaTextClassifySerializer.serialize_request(
            {"texts": ["test"]}, model_name="m"
        )
        parsed = riva_nlp_pb2.TextClassRequest()
        parsed.ParseFromString(data)
        assert parsed.model.language_code == "en-US"


class TestTextClassifyDeserializeResponse:
    def test_single_result(self) -> None:
        response = riva_nlp_pb2.TextClassResponse()
        result = response.results.add()
        label = result.labels.add()
        label.class_name = "positive"
        label.score = 0.9
        data = response.SerializeToString()

        result_dict, size = RivaTextClassifySerializer.deserialize_response(data)

        assert size == len(data)
        assert len(result_dict["results"]) == 1
        assert result_dict["results"][0]["labels"][0]["class_name"] == "positive"
        assert result_dict["results"][0]["labels"][0]["score"] == pytest.approx(0.9)

    def test_multiple_results(self) -> None:
        """Multiple classification results should all be returned."""
        response = riva_nlp_pb2.TextClassResponse()
        for cls_name, score in [("positive", 0.9), ("negative", 0.7)]:
            result = response.results.add()
            label = result.labels.add()
            label.class_name = cls_name
            label.score = score
        data = response.SerializeToString()

        result_dict, _ = RivaTextClassifySerializer.deserialize_response(data)

        assert len(result_dict["results"]) == 2
        assert result_dict["results"][0]["labels"][0]["class_name"] == "positive"
        assert result_dict["results"][1]["labels"][0]["class_name"] == "negative"

    def test_multiple_labels_per_result(self) -> None:
        """A single result can have multiple labels."""
        response = riva_nlp_pb2.TextClassResponse()
        result = response.results.add()
        for cls_name, score in [
            ("positive", 0.9),
            ("neutral", 0.05),
            ("negative", 0.05),
        ]:
            label = result.labels.add()
            label.class_name = cls_name
            label.score = score
        data = response.SerializeToString()

        result_dict, _ = RivaTextClassifySerializer.deserialize_response(data)

        assert len(result_dict["results"][0]["labels"]) == 3

    def test_empty_results(self) -> None:
        """Empty response should return empty results list."""
        response = riva_nlp_pb2.TextClassResponse()
        data = response.SerializeToString()

        result_dict, _ = RivaTextClassifySerializer.deserialize_response(data)

        assert result_dict["results"] == []


class TestTextClassifyStreamResponse:
    def test_returns_error_chunk(self) -> None:
        """NLP endpoints should return error for streaming."""
        chunk = RivaTextClassifySerializer.deserialize_stream_response(b"\x01")
        assert isinstance(chunk, StreamChunk)
        assert chunk.error_message is not None
        assert "NLP" in chunk.error_message
        assert chunk.response_dict is None
        assert chunk.response_size == 1


# ---------------------------------------------------------------------------
# TokenClassifySerializer
# ---------------------------------------------------------------------------
class TestTokenClassifySerializeRequest:
    def test_roundtrip(self) -> None:
        payload = {"texts": ["John lives in New York."], "language_code": "en-US"}
        data = RivaTokenClassifySerializer.serialize_request(
            payload, model_name="ner_model"
        )

        parsed = riva_nlp_pb2.TokenClassRequest()
        parsed.ParseFromString(data)
        assert list(parsed.text) == ["John lives in New York."]

    def test_multiple_texts(self) -> None:
        """Multiple text inputs should all be included."""
        payload = {"texts": ["text1", "text2", "text3"]}
        data = RivaTokenClassifySerializer.serialize_request(payload, model_name="m")
        parsed = riva_nlp_pb2.TokenClassRequest()
        parsed.ParseFromString(data)
        assert list(parsed.text) == ["text1", "text2", "text3"]


class TestTokenClassifyDeserializeResponse:
    def test_single_token(self) -> None:
        response = riva_nlp_pb2.TokenClassResponse()
        seq = response.results.add()
        tv = seq.results.add()
        tv.token = "John"
        label = tv.label.add()
        label.class_name = "PER"
        label.score = 0.98
        data = response.SerializeToString()

        result_dict, size = RivaTokenClassifySerializer.deserialize_response(data)

        assert size == len(data)
        assert result_dict["results"][0]["tokens"][0]["token"] == "John"
        assert (
            result_dict["results"][0]["tokens"][0]["labels"][0]["class_name"] == "PER"
        )

    def test_multiple_tokens(self) -> None:
        """Multiple tokens in a sequence should all be returned."""
        response = riva_nlp_pb2.TokenClassResponse()
        seq = response.results.add()
        for token_text, cls_name in [("John", "PER"), ("NYC", "LOC")]:
            tv = seq.results.add()
            tv.token = token_text
            label = tv.label.add()
            label.class_name = cls_name
            label.score = 0.95
        data = response.SerializeToString()

        result_dict, _ = RivaTokenClassifySerializer.deserialize_response(data)

        tokens = result_dict["results"][0]["tokens"]
        assert len(tokens) == 2
        assert tokens[0]["token"] == "John"
        assert tokens[1]["token"] == "NYC"

    def test_multiple_labels_per_token(self) -> None:
        """A token can have multiple labels."""
        response = riva_nlp_pb2.TokenClassResponse()
        seq = response.results.add()
        tv = seq.results.add()
        tv.token = "Bank"
        for cls_name, score in [("ORG", 0.6), ("LOC", 0.3)]:
            label = tv.label.add()
            label.class_name = cls_name
            label.score = score
        data = response.SerializeToString()

        result_dict, _ = RivaTokenClassifySerializer.deserialize_response(data)

        labels = result_dict["results"][0]["tokens"][0]["labels"]
        assert len(labels) == 2

    def test_empty_results(self) -> None:
        """Empty response should return empty results."""
        response = riva_nlp_pb2.TokenClassResponse()
        data = response.SerializeToString()

        result_dict, _ = RivaTokenClassifySerializer.deserialize_response(data)

        assert result_dict["results"] == []

    def test_multiple_sequences(self) -> None:
        """Multiple text sequences should each produce separate results."""
        response = riva_nlp_pb2.TokenClassResponse()
        for _ in range(3):
            seq = response.results.add()
            tv = seq.results.add()
            tv.token = "token"
            tv.label.add().class_name = "O"
        data = response.SerializeToString()

        result_dict, _ = RivaTokenClassifySerializer.deserialize_response(data)

        assert len(result_dict["results"]) == 3


class TestTokenClassifyStreamResponse:
    def test_returns_error_chunk(self) -> None:
        chunk = RivaTokenClassifySerializer.deserialize_stream_response(b"\x01\x02")
        assert chunk.error_message is not None
        assert chunk.response_size == 2


# ---------------------------------------------------------------------------
# TransformTextSerializer
# ---------------------------------------------------------------------------
class TestTransformTextSerializeRequest:
    def test_roundtrip(self) -> None:
        payload = {"texts": ["hello world"], "language_code": "en-US"}
        data = RivaTransformTextSerializer.serialize_request(
            payload, model_name="transform_model"
        )

        parsed = riva_nlp_pb2.TextTransformRequest()
        parsed.ParseFromString(data)
        assert list(parsed.text) == ["hello world"]

    def test_empty_texts(self) -> None:
        data = RivaTransformTextSerializer.serialize_request(
            {"texts": []}, model_name="m"
        )
        parsed = riva_nlp_pb2.TextTransformRequest()
        parsed.ParseFromString(data)
        assert list(parsed.text) == []


class TestTransformTextDeserializeResponse:
    def test_single_text(self) -> None:
        response = riva_nlp_pb2.TextTransformResponse()
        response.text.append("Hello World!")
        data = response.SerializeToString()

        result_dict, size = RivaTransformTextSerializer.deserialize_response(data)

        assert result_dict["texts"] == ["Hello World!"]
        assert size == len(data)

    def test_multiple_texts(self) -> None:
        """Multiple transformed texts should all be returned."""
        response = riva_nlp_pb2.TextTransformResponse()
        response.text.append("Text 1")
        response.text.append("Text 2")
        data = response.SerializeToString()

        result_dict, _ = RivaTransformTextSerializer.deserialize_response(data)

        assert result_dict["texts"] == ["Text 1", "Text 2"]

    def test_empty_response(self) -> None:
        """Empty response should return empty texts list."""
        response = riva_nlp_pb2.TextTransformResponse()
        data = response.SerializeToString()

        result_dict, _ = RivaTransformTextSerializer.deserialize_response(data)

        assert result_dict["texts"] == []


class TestTransformTextStreamResponse:
    def test_returns_error_chunk(self) -> None:
        chunk = RivaTransformTextSerializer.deserialize_stream_response(b"")
        assert chunk.error_message is not None
        assert chunk.response_size == 0


# ---------------------------------------------------------------------------
# PunctuateTextSerializer
# ---------------------------------------------------------------------------
class TestPunctuateTextSerializeRequest:
    def test_delegates_to_transform(self) -> None:
        """PunctuateText uses same types as TransformText."""
        payload = {"texts": ["hello world"], "language_code": "en-US"}
        data = RivaPunctuateTextSerializer.serialize_request(
            payload, model_name="punct_model"
        )

        parsed = riva_nlp_pb2.TextTransformRequest()
        parsed.ParseFromString(data)
        assert list(parsed.text) == ["hello world"]
        assert parsed.model.model_name == "punct_model"

    def test_with_request_id(self) -> None:
        """PunctuateText should pass request_id through."""
        data = RivaPunctuateTextSerializer.serialize_request(
            {"texts": ["test"]}, model_name="m", request_id="r42"
        )
        parsed = riva_nlp_pb2.TextTransformRequest()
        parsed.ParseFromString(data)
        assert parsed.id.value == "r42"


class TestPunctuateTextDeserializeResponse:
    def test_delegates_to_transform(self) -> None:
        """PunctuateText deserialization should use TransformText format."""
        response = riva_nlp_pb2.TextTransformResponse()
        response.text.append("Hello, world!")
        data = response.SerializeToString()

        result_dict, size = RivaPunctuateTextSerializer.deserialize_response(data)

        assert result_dict["texts"] == ["Hello, world!"]
        assert size == len(data)


class TestPunctuateTextStreamResponse:
    def test_returns_error_chunk(self) -> None:
        chunk = RivaPunctuateTextSerializer.deserialize_stream_response(b"\x01")
        assert chunk.error_message is not None


# ---------------------------------------------------------------------------
# NaturalQuerySerializer
# ---------------------------------------------------------------------------
class TestNaturalQuerySerializeRequest:
    def test_roundtrip(self) -> None:
        payload = {
            "query": "What is the capital?",
            "context": "France is a country.",
            "top_n": 3,
        }
        data = RivaNaturalQuerySerializer.serialize_request(
            payload, model_name="qa_model", request_id="r1"
        )

        parsed = riva_nlp_pb2.NaturalQueryRequest()
        parsed.ParseFromString(data)
        assert parsed.query == "What is the capital?"
        assert parsed.context == "France is a country."
        assert parsed.top_n == 3

    def test_defaults(self) -> None:
        """Default values should be set correctly."""
        data = RivaNaturalQuerySerializer.serialize_request({}, model_name="m")
        parsed = riva_nlp_pb2.NaturalQueryRequest()
        parsed.ParseFromString(data)
        assert parsed.query == ""
        assert parsed.context == ""
        assert parsed.top_n == 1

    def test_no_request_id(self) -> None:
        data = RivaNaturalQuerySerializer.serialize_request(
            {"query": "test"}, model_name="m"
        )
        parsed = riva_nlp_pb2.NaturalQueryRequest()
        parsed.ParseFromString(data)
        assert not parsed.HasField("id")


class TestNaturalQueryDeserializeResponse:
    def test_single_result(self) -> None:
        response = riva_nlp_pb2.NaturalQueryResponse()
        result = response.results.add()
        result.answer = "Paris"
        result.score = 0.85
        data = response.SerializeToString()

        result_dict, size = RivaNaturalQuerySerializer.deserialize_response(data)

        assert result_dict["results"][0]["answer"] == "Paris"
        assert result_dict["results"][0]["score"] == pytest.approx(0.85)
        assert size == len(data)

    def test_multiple_results(self) -> None:
        """Multiple answers should all be returned."""
        response = riva_nlp_pb2.NaturalQueryResponse()
        for answer, score in [("Paris", 0.9), ("Lyon", 0.3)]:
            result = response.results.add()
            result.answer = answer
            result.score = score
        data = response.SerializeToString()

        result_dict, _ = RivaNaturalQuerySerializer.deserialize_response(data)

        assert len(result_dict["results"]) == 2
        assert result_dict["results"][0]["answer"] == "Paris"
        assert result_dict["results"][1]["answer"] == "Lyon"

    def test_empty_results(self) -> None:
        """Empty response should return empty results list."""
        response = riva_nlp_pb2.NaturalQueryResponse()
        data = response.SerializeToString()

        result_dict, _ = RivaNaturalQuerySerializer.deserialize_response(data)

        assert result_dict["results"] == []


class TestNaturalQueryStreamResponse:
    def test_returns_error_chunk(self) -> None:
        chunk = RivaNaturalQuerySerializer.deserialize_stream_response(b"\x01")
        assert chunk.error_message is not None


# ---------------------------------------------------------------------------
# AnalyzeIntentSerializer
# ---------------------------------------------------------------------------
class TestAnalyzeIntentSerializeRequest:
    def test_roundtrip(self) -> None:
        payload = {"query": "Turn on the lights", "domain": "smart_home"}
        data = RivaAnalyzeIntentSerializer.serialize_request(
            payload, model_name="intent_model"
        )

        parsed = riva_nlp_pb2.AnalyzeIntentRequest()
        parsed.ParseFromString(data)
        assert parsed.query == "Turn on the lights"
        assert parsed.options.domain == "smart_home"

    def test_no_domain(self) -> None:
        """Request without domain should not set domain option."""
        payload = {"query": "Turn on the lights"}
        data = RivaAnalyzeIntentSerializer.serialize_request(payload, model_name="m")

        parsed = riva_nlp_pb2.AnalyzeIntentRequest()
        parsed.ParseFromString(data)
        assert parsed.query == "Turn on the lights"
        assert parsed.options.domain == ""

    def test_with_request_id(self) -> None:
        data = RivaAnalyzeIntentSerializer.serialize_request(
            {"query": "test"}, model_name="m", request_id="r42"
        )
        parsed = riva_nlp_pb2.AnalyzeIntentRequest()
        parsed.ParseFromString(data)
        assert parsed.id.value == "r42"

    def test_empty_query(self) -> None:
        data = RivaAnalyzeIntentSerializer.serialize_request({}, model_name="m")
        parsed = riva_nlp_pb2.AnalyzeIntentRequest()
        parsed.ParseFromString(data)
        assert parsed.query == ""


class TestAnalyzeIntentDeserializeResponse:
    def test_full_response(self) -> None:
        response = riva_nlp_pb2.AnalyzeIntentResponse()
        response.intent.class_name = "lights_on"
        response.intent.score = 0.92
        slot = response.slots.add()
        slot.token = "lights"
        label = slot.label.add()
        label.class_name = "device"
        label.score = 0.88
        data = response.SerializeToString()

        result_dict, size = RivaAnalyzeIntentSerializer.deserialize_response(data)

        assert result_dict["intent"]["class_name"] == "lights_on"
        assert result_dict["intent"]["score"] == pytest.approx(0.92)
        assert result_dict["slots"][0]["token"] == "lights"
        assert result_dict["slots"][0]["labels"][0]["class_name"] == "device"
        assert size == len(data)

    def test_with_domain(self) -> None:
        """Response with domain should include it."""
        response = riva_nlp_pb2.AnalyzeIntentResponse()
        response.intent.class_name = "lights_on"
        response.intent.score = 0.9
        response.domain.class_name = "smart_home"
        response.domain.score = 0.95
        data = response.SerializeToString()

        result_dict, _ = RivaAnalyzeIntentSerializer.deserialize_response(data)

        assert result_dict["domain"]["class_name"] == "smart_home"
        assert result_dict["domain"]["score"] == pytest.approx(0.95)

    def test_no_domain(self) -> None:
        """Response without domain should not include domain key."""
        response = riva_nlp_pb2.AnalyzeIntentResponse()
        response.intent.class_name = "lights_on"
        response.intent.score = 0.9
        data = response.SerializeToString()

        result_dict, _ = RivaAnalyzeIntentSerializer.deserialize_response(data)

        assert "domain" not in result_dict

    def test_multiple_slots(self) -> None:
        """Multiple slots should all be returned."""
        response = riva_nlp_pb2.AnalyzeIntentResponse()
        response.intent.class_name = "set_temp"
        response.intent.score = 0.9
        for token, cls in [("temperature", "setting"), ("72", "value")]:
            slot = response.slots.add()
            slot.token = token
            label = slot.label.add()
            label.class_name = cls
            label.score = 0.8
        data = response.SerializeToString()

        result_dict, _ = RivaAnalyzeIntentSerializer.deserialize_response(data)

        assert len(result_dict["slots"]) == 2
        assert result_dict["slots"][0]["token"] == "temperature"
        assert result_dict["slots"][1]["token"] == "72"

    def test_empty_slots(self) -> None:
        """Response with no slots should have empty slots list."""
        response = riva_nlp_pb2.AnalyzeIntentResponse()
        response.intent.class_name = "greet"
        response.intent.score = 0.99
        data = response.SerializeToString()

        result_dict, _ = RivaAnalyzeIntentSerializer.deserialize_response(data)

        assert result_dict["slots"] == []

    def test_slot_with_multiple_labels(self) -> None:
        """A slot can have multiple labels."""
        response = riva_nlp_pb2.AnalyzeIntentResponse()
        response.intent.class_name = "test"
        response.intent.score = 0.9
        slot = response.slots.add()
        slot.token = "bank"
        for cls_name in ["B-ORG", "B-LOC"]:
            label = slot.label.add()
            label.class_name = cls_name
            label.score = 0.5
        data = response.SerializeToString()

        result_dict, _ = RivaAnalyzeIntentSerializer.deserialize_response(data)

        assert len(result_dict["slots"][0]["labels"]) == 2


class TestAnalyzeIntentStreamResponse:
    def test_returns_error_chunk(self) -> None:
        chunk = RivaAnalyzeIntentSerializer.deserialize_stream_response(b"\x01")
        assert chunk.error_message is not None


# ---------------------------------------------------------------------------
# AnalyzeEntitiesSerializer
# ---------------------------------------------------------------------------
class TestAnalyzeEntitiesSerializeRequest:
    def test_roundtrip(self) -> None:
        payload = {"query": "John lives in New York."}
        data = RivaAnalyzeEntitiesSerializer.serialize_request(
            payload, model_name="ner_model", request_id="r1"
        )

        parsed = riva_nlp_pb2.AnalyzeEntitiesRequest()
        parsed.ParseFromString(data)
        assert parsed.query == "John lives in New York."
        assert parsed.id.value == "r1"

    def test_empty_query(self) -> None:
        data = RivaAnalyzeEntitiesSerializer.serialize_request({}, model_name="m")
        parsed = riva_nlp_pb2.AnalyzeEntitiesRequest()
        parsed.ParseFromString(data)
        assert parsed.query == ""

    def test_no_request_id(self) -> None:
        data = RivaAnalyzeEntitiesSerializer.serialize_request(
            {"query": "test"}, model_name="m"
        )
        parsed = riva_nlp_pb2.AnalyzeEntitiesRequest()
        parsed.ParseFromString(data)
        assert not parsed.HasField("id")


class TestAnalyzeEntitiesDeserializeResponse:
    def test_delegates_to_token_classify(self) -> None:
        """AnalyzeEntities returns TokenClassResponse."""
        response = riva_nlp_pb2.TokenClassResponse()
        seq = response.results.add()
        tv = seq.results.add()
        tv.token = "John"
        label = tv.label.add()
        label.class_name = "PER"
        label.score = 0.98
        data = response.SerializeToString()

        result_dict, size = RivaAnalyzeEntitiesSerializer.deserialize_response(data)

        assert result_dict["results"][0]["tokens"][0]["token"] == "John"
        assert size == len(data)

    def test_empty_response(self) -> None:
        """Empty TokenClassResponse should return empty results."""
        response = riva_nlp_pb2.TokenClassResponse()
        data = response.SerializeToString()

        result_dict, _ = RivaAnalyzeEntitiesSerializer.deserialize_response(data)

        assert result_dict["results"] == []


class TestAnalyzeEntitiesStreamResponse:
    def test_returns_error_chunk(self) -> None:
        chunk = RivaAnalyzeEntitiesSerializer.deserialize_stream_response(b"\x01")
        assert chunk.error_message is not None


# ---------------------------------------------------------------------------
# Cross-cutting: All NLP serializers stream error path
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# _serialize_text_list_request shared helper
# ---------------------------------------------------------------------------
class TestSerializeTextListRequest:
    """Tests for the shared _serialize_text_list_request helper."""

    @pytest.mark.parametrize(
        "request_cls",
        [
            riva_nlp_pb2.TextClassRequest,
            riva_nlp_pb2.TokenClassRequest,
            riva_nlp_pb2.TextTransformRequest,
        ],
    )
    def test_fills_text_field(self, request_cls: type) -> None:
        """All text-list request types should have texts populated."""
        payload = {"texts": ["one", "two"]}
        data = _serialize_text_list_request(request_cls(), payload, "model", "")
        parsed = request_cls()
        parsed.ParseFromString(data)
        assert list(parsed.text) == ["one", "two"]

    @pytest.mark.parametrize(
        "request_cls",
        [
            riva_nlp_pb2.TextClassRequest,
            riva_nlp_pb2.TokenClassRequest,
            riva_nlp_pb2.TextTransformRequest,
        ],
    )
    def test_fills_model_name(self, request_cls: type) -> None:
        data = _serialize_text_list_request(
            request_cls(), {"texts": []}, "my_model", ""
        )
        parsed = request_cls()
        parsed.ParseFromString(data)
        assert parsed.model.model_name == "my_model"

    def test_fills_request_id(self) -> None:
        data = _serialize_text_list_request(
            riva_nlp_pb2.TextClassRequest(), {"texts": ["t"]}, "m", "req-42"
        )
        parsed = riva_nlp_pb2.TextClassRequest()
        parsed.ParseFromString(data)
        assert parsed.id.value == "req-42"

    def test_no_request_id(self) -> None:
        data = _serialize_text_list_request(
            riva_nlp_pb2.TextClassRequest(), {"texts": ["t"]}, "m", ""
        )
        parsed = riva_nlp_pb2.TextClassRequest()
        parsed.ParseFromString(data)
        assert not parsed.HasField("id")

    def test_fills_top_n(self) -> None:
        data = _serialize_text_list_request(
            riva_nlp_pb2.TextClassRequest(), {"texts": [], "top_n": 10}, "m", ""
        )
        parsed = riva_nlp_pb2.TextClassRequest()
        parsed.ParseFromString(data)
        assert parsed.top_n == 10

    def test_all_text_list_serializers_delegate(self) -> None:
        """TextClassify, TokenClassify, TransformText should all use _serialize_text_list_request."""
        payload = {"texts": ["hello"], "language_code": "de-DE"}
        for serializer_cls, proto_cls in [
            (RivaTextClassifySerializer, riva_nlp_pb2.TextClassRequest),
            (RivaTokenClassifySerializer, riva_nlp_pb2.TokenClassRequest),
            (RivaTransformTextSerializer, riva_nlp_pb2.TextTransformRequest),
        ]:
            data = serializer_cls.serialize_request(
                payload, model_name="m", request_id="r1"
            )
            parsed = proto_cls()
            parsed.ParseFromString(data)
            assert list(parsed.text) == ["hello"]
            assert parsed.model.language_code == "de-DE"
            assert parsed.id.value == "r1"


class TestAllNlpStreamErrors:
    """Verify all NLP serializers return error for streaming deserialization."""

    @pytest.mark.parametrize(
        "serializer_cls",
        [
            RivaTextClassifySerializer,
            RivaTokenClassifySerializer,
            RivaTransformTextSerializer,
            RivaPunctuateTextSerializer,
            RivaNaturalQuerySerializer,
            RivaAnalyzeIntentSerializer,
            RivaAnalyzeEntitiesSerializer,
        ],
    )
    def test_stream_response_error(self, serializer_cls: type) -> None:
        """All NLP serializers should return error StreamChunk for streaming."""
        chunk = serializer_cls.deserialize_stream_response(b"\x01\x02\x03")
        assert isinstance(chunk, StreamChunk)
        assert chunk.error_message is not None
        assert chunk.response_dict is None
        assert chunk.response_size == 3
