# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Riva NLP endpoints."""

from __future__ import annotations

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.riva_nlp import (
    RivaAnalyzeEntitiesEndpoint,
    RivaAnalyzeIntentEndpoint,
    RivaNaturalQueryEndpoint,
    RivaPunctuateTextEndpoint,
    RivaTextClassifyEndpoint,
    RivaTokenClassifyEndpoint,
    RivaTransformTextEndpoint,
    _parse_texts_response,
    _RivaTextListEndpoint,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


# ---------------------------------------------------------------------------
# RivaTextClassifyEndpoint
# ---------------------------------------------------------------------------
class TestRivaTextClassifyEndpoint:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.RIVA_TEXT_CLASSIFY)
        return create_endpoint_with_mock_transport(
            RivaTextClassifyEndpoint, model_endpoint
        )

    def test_format_payload(self, endpoint) -> None:
        turn = Turn(texts=[Text(contents=["This is great!"])])
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[turn]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["texts"] == ["This is great!"]
        assert payload["language_code"] == "en-US"

    def test_format_payload_multiple_texts(self, endpoint) -> None:
        """Multiple text contents should all be included."""
        turn = Turn(texts=[Text(contents=["text1", "text2"])])
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[turn]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["texts"] == ["text1", "text2"]

    def test_format_payload_empty_turns_raises(self, endpoint) -> None:
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[]
        )
        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_custom_language(self) -> None:
        """Custom language_code from extra config should be used."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_TEXT_CLASSIFY,
            extra=[("language_code", "de-DE")],
        )
        endpoint = create_endpoint_with_mock_transport(
            RivaTextClassifyEndpoint, model_endpoint
        )
        turn = Turn(texts=[Text(contents=["Hallo"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        payload = endpoint.format_payload(request_info)

        assert payload["language_code"] == "de-DE"

    def test_parse_response(self, endpoint) -> None:
        response = create_mock_response(
            json_data={
                "results": [{"labels": [{"class_name": "positive", "score": 0.9}]}]
            }
        )
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert "positive" in parsed.data.get_text()

    def test_parse_response_no_json(self, endpoint) -> None:
        response = create_mock_response(json_data=None)
        assert endpoint.parse_response(response) is None


# ---------------------------------------------------------------------------
# RivaTokenClassifyEndpoint
# ---------------------------------------------------------------------------
class TestRivaTokenClassifyEndpoint:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.RIVA_TOKEN_CLASSIFY)
        return create_endpoint_with_mock_transport(
            RivaTokenClassifyEndpoint, model_endpoint
        )

    def test_format_payload(self, endpoint) -> None:
        turn = Turn(texts=[Text(contents=["John lives in NYC"])])
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[turn]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["texts"] == ["John lives in NYC"]
        assert payload["language_code"] == "en-US"

    def test_format_payload_empty_filtered(self, endpoint) -> None:
        """Empty content strings should be filtered out."""
        turn = Turn(texts=[Text(contents=["text1", "", "text2"])])
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[turn]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["texts"] == ["text1", "text2"]

    def test_parse_response(self, endpoint) -> None:
        response = create_mock_response(
            json_data={
                "results": [
                    {
                        "tokens": [
                            {
                                "token": "John",
                                "labels": [{"class_name": "PER", "score": 0.98}],
                            }
                        ]
                    }
                ]
            }
        )
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert "John" in parsed.data.get_text()

    def test_parse_response_no_json(self, endpoint) -> None:
        response = create_mock_response(json_data=None)
        assert endpoint.parse_response(response) is None


# ---------------------------------------------------------------------------
# RivaTransformTextEndpoint
# ---------------------------------------------------------------------------
class TestRivaTransformTextEndpoint:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.RIVA_TRANSFORM_TEXT)
        return create_endpoint_with_mock_transport(
            RivaTransformTextEndpoint, model_endpoint
        )

    def test_format_payload(self, endpoint) -> None:
        turn = Turn(texts=[Text(contents=["hello world"])])
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[turn]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["texts"] == ["hello world"]

    def test_parse_response_with_text(self, endpoint) -> None:
        response = create_mock_response(json_data={"texts": ["Hello World!"]})
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Hello World!"

    def test_parse_response_multiple_texts(self, endpoint) -> None:
        """Multiple texts should be joined with space."""
        response = create_mock_response(json_data={"texts": ["Hello", "World!"]})
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Hello World!"

    def test_parse_response_no_texts(self, endpoint) -> None:
        response = create_mock_response(json_data={"texts": []})
        assert endpoint.parse_response(response) is None

    def test_parse_response_no_json(self, endpoint) -> None:
        response = create_mock_response(json_data=None)
        assert endpoint.parse_response(response) is None


# ---------------------------------------------------------------------------
# RivaPunctuateTextEndpoint
# ---------------------------------------------------------------------------
class TestRivaPunctuateTextEndpoint:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.RIVA_PUNCTUATE_TEXT)
        return create_endpoint_with_mock_transport(
            RivaPunctuateTextEndpoint, model_endpoint
        )

    def test_format_payload(self, endpoint) -> None:
        turn = Turn(texts=[Text(contents=["hello world"])])
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[turn]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["texts"] == ["hello world"]
        assert payload["language_code"] == "en-US"

    def test_parse_response_with_text(self, endpoint) -> None:
        response = create_mock_response(json_data={"texts": ["Hello, world!"]})
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Hello, world!"

    def test_parse_response_no_texts(self, endpoint) -> None:
        response = create_mock_response(json_data={"texts": []})
        assert endpoint.parse_response(response) is None

    def test_parse_response_no_json(self, endpoint) -> None:
        response = create_mock_response(json_data=None)
        assert endpoint.parse_response(response) is None

    def test_format_payload_empty_turns_raises(self, endpoint) -> None:
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[]
        )
        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)


# ---------------------------------------------------------------------------
# RivaNaturalQueryEndpoint
# ---------------------------------------------------------------------------
class TestRivaNaturalQueryEndpoint:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_NATURAL_QUERY,
            extra=[("context", "France is a country in Europe.")],
        )
        return create_endpoint_with_mock_transport(
            RivaNaturalQueryEndpoint, model_endpoint
        )

    def test_format_payload(self, endpoint) -> None:
        turn = Turn(texts=[Text(contents=["What is France?"])])
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[turn]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["query"] == "What is France?"
        assert payload["context"] == "France is a country in Europe."

    def test_format_payload_default_top_n(self, endpoint) -> None:
        """Default top_n should be 1."""
        turn = Turn(texts=[Text(contents=["test"])])
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[turn]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["top_n"] == 1

    def test_format_payload_custom_top_n(self) -> None:
        """Custom top_n should be used."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_NATURAL_QUERY,
            extra=[("context", "ctx"), ("top_n", 5)],
        )
        endpoint = create_endpoint_with_mock_transport(
            RivaNaturalQueryEndpoint, model_endpoint
        )
        turn = Turn(texts=[Text(contents=["test"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        payload = endpoint.format_payload(request_info)

        assert payload["top_n"] == 5

    def test_format_payload_multiple_texts_joined(self, endpoint) -> None:
        """Multiple texts should be joined into a single query."""
        turn = Turn(texts=[Text(contents=["What is", "France?"])])
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[turn]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["query"] == "What is France?"

    def test_parse_response(self, endpoint) -> None:
        response = create_mock_response(
            json_data={"results": [{"answer": "A country in Europe", "score": 0.85}]}
        )
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "A country in Europe"

    def test_parse_response_no_results(self, endpoint) -> None:
        response = create_mock_response(json_data={"results": []})
        assert endpoint.parse_response(response) is None

    def test_parse_response_no_json(self, endpoint) -> None:
        response = create_mock_response(json_data=None)
        assert endpoint.parse_response(response) is None


# ---------------------------------------------------------------------------
# RivaAnalyzeIntentEndpoint
# ---------------------------------------------------------------------------
class TestRivaAnalyzeIntentEndpoint:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_ANALYZE_INTENT,
            extra=[("domain", "smart_home")],
        )
        return create_endpoint_with_mock_transport(
            RivaAnalyzeIntentEndpoint, model_endpoint
        )

    def test_format_payload(self, endpoint) -> None:
        turn = Turn(texts=[Text(contents=["Turn on the lights"])])
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[turn]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["query"] == "Turn on the lights"
        assert payload["domain"] == "smart_home"

    def test_format_payload_no_domain(self) -> None:
        """Endpoint without domain should not include it."""
        model_endpoint = create_model_endpoint(EndpointType.RIVA_ANALYZE_INTENT)
        endpoint = create_endpoint_with_mock_transport(
            RivaAnalyzeIntentEndpoint, model_endpoint
        )
        turn = Turn(texts=[Text(contents=["hello"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        payload = endpoint.format_payload(request_info)

        assert "domain" not in payload

    def test_format_payload_empty_turns_raises(self, endpoint) -> None:
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[]
        )
        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_parse_response(self, endpoint) -> None:
        response = create_mock_response(
            json_data={
                "intent": {"class_name": "lights_on", "score": 0.92},
                "slots": [
                    {
                        "token": "lights",
                        "labels": [{"class_name": "device", "score": 0.88}],
                    }
                ],
            }
        )
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert "lights_on" in parsed.data.get_text()

    def test_parse_response_no_json(self, endpoint) -> None:
        response = create_mock_response(json_data=None)
        assert endpoint.parse_response(response) is None


# ---------------------------------------------------------------------------
# RivaAnalyzeEntitiesEndpoint
# ---------------------------------------------------------------------------
class TestRivaAnalyzeEntitiesEndpoint:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.RIVA_ANALYZE_ENTITIES)
        return create_endpoint_with_mock_transport(
            RivaAnalyzeEntitiesEndpoint, model_endpoint
        )

    def test_format_payload(self, endpoint) -> None:
        turn = Turn(texts=[Text(contents=["John lives in NYC"])])
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[turn]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["query"] == "John lives in NYC"

    def test_format_payload_empty_turns_raises(self, endpoint) -> None:
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[]
        )
        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_multiple_texts_joined(self, endpoint) -> None:
        """Multiple text contents should be joined."""
        turn = Turn(texts=[Text(contents=["John", "lives in NYC"])])
        request_info = create_request_info(
            model_endpoint=endpoint.model_endpoint, turns=[turn]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["query"] == "John lives in NYC"

    def test_parse_response(self, endpoint) -> None:
        response = create_mock_response(
            json_data={
                "results": [
                    {
                        "tokens": [
                            {
                                "token": "John",
                                "labels": [{"class_name": "PER", "score": 0.98}],
                            }
                        ]
                    }
                ]
            }
        )
        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert "John" in parsed.data.get_text()

    def test_parse_response_no_json(self, endpoint) -> None:
        response = create_mock_response(json_data=None)
        assert endpoint.parse_response(response) is None


# ---------------------------------------------------------------------------
# _RivaTextListEndpoint base class
# ---------------------------------------------------------------------------
class TestRivaTextListEndpointInheritance:
    """Verify shared base class is used by all text-list NLP endpoints."""

    @pytest.mark.parametrize(
        "cls",
        [
            RivaTextClassifyEndpoint,
            RivaTokenClassifyEndpoint,
            RivaTransformTextEndpoint,
            RivaPunctuateTextEndpoint,
        ],
    )
    def test_inherits_from_base(self, cls: type) -> None:
        assert issubclass(cls, _RivaTextListEndpoint)

    @pytest.mark.parametrize(
        "cls,endpoint_type",
        [
            (RivaTextClassifyEndpoint, EndpointType.RIVA_TEXT_CLASSIFY),
            (RivaTokenClassifyEndpoint, EndpointType.RIVA_TOKEN_CLASSIFY),
            (RivaTransformTextEndpoint, EndpointType.RIVA_TRANSFORM_TEXT),
            (RivaPunctuateTextEndpoint, EndpointType.RIVA_PUNCTUATE_TEXT),
        ],
    )
    def test_shared_format_payload(
        self, cls: type, endpoint_type: EndpointType
    ) -> None:
        """All text-list endpoints should produce the same texts list from format_payload."""
        model_endpoint = create_model_endpoint(endpoint_type)
        endpoint = create_endpoint_with_mock_transport(cls, model_endpoint)
        turn = Turn(texts=[Text(contents=["a", "b"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        payload = endpoint.format_payload(request_info)

        assert payload["texts"] == ["a", "b"]

    def test_non_text_list_endpoints_do_not_inherit(self) -> None:
        """NaturalQuery, AnalyzeIntent, AnalyzeEntities should NOT inherit."""
        assert not issubclass(RivaNaturalQueryEndpoint, _RivaTextListEndpoint)
        assert not issubclass(RivaAnalyzeIntentEndpoint, _RivaTextListEndpoint)
        assert not issubclass(RivaAnalyzeEntitiesEndpoint, _RivaTextListEndpoint)


# ---------------------------------------------------------------------------
# _parse_texts_response helper
# ---------------------------------------------------------------------------
class TestParseTextsResponse:
    """Tests for the shared _parse_texts_response helper."""

    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.RIVA_TRANSFORM_TEXT)
        return create_endpoint_with_mock_transport(
            RivaTransformTextEndpoint, model_endpoint
        )

    def test_single_text(self, endpoint) -> None:
        response = create_mock_response(json_data={"texts": ["Hello!"]})
        parsed = _parse_texts_response(response, endpoint)

        assert parsed is not None
        assert parsed.data.get_text() == "Hello!"

    def test_multiple_texts_joined(self, endpoint) -> None:
        response = create_mock_response(json_data={"texts": ["Hello", "World!"]})
        parsed = _parse_texts_response(response, endpoint)

        assert parsed is not None
        assert parsed.data.get_text() == "Hello World!"

    def test_empty_texts_returns_none(self, endpoint) -> None:
        response = create_mock_response(json_data={"texts": []})
        assert _parse_texts_response(response, endpoint) is None

    def test_no_json_returns_none(self, endpoint) -> None:
        response = create_mock_response(json_data=None)
        assert _parse_texts_response(response, endpoint) is None

    def test_missing_texts_key_returns_none(self, endpoint) -> None:
        response = create_mock_response(json_data={"other": "data"})
        assert _parse_texts_response(response, endpoint) is None
