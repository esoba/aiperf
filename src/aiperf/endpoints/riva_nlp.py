# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Riva NLP endpoints for text classification, token classification,
text transformation, punctuation, natural query, intent analysis, and entity analysis.
"""

from __future__ import annotations

from typing import Any

import orjson

from aiperf.common.models import InferenceServerResponse, ParsedResponse, RequestInfo
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.riva_helpers import get_extra


def _extract_texts(request_info: RequestInfo) -> list[str]:
    """Extract text contents from the first turn."""
    if not request_info.turns:
        raise ValueError("Riva NLP endpoint requires at least one turn.")
    turn = request_info.turns[0]
    return [content for text in turn.texts for content in text.contents if content]


def _parse_json_response(
    response: InferenceServerResponse, endpoint: BaseEndpoint
) -> ParsedResponse | None:
    """Parse a JSON response into a TextResponseData with the JSON as text."""
    json_obj = response.get_json()
    if not json_obj:
        return None
    text = orjson.dumps(json_obj).decode("utf-8")
    return ParsedResponse(
        perf_ns=response.perf_ns,
        data=endpoint.make_text_response_data(text),
    )


def _parse_texts_response(
    response: InferenceServerResponse, endpoint: BaseEndpoint
) -> ParsedResponse | None:
    """Parse a response containing a 'texts' list into joined text."""
    json_obj = response.get_json()
    if not json_obj:
        return None
    texts = json_obj.get("texts", [])
    if not texts:
        return None
    return ParsedResponse(
        perf_ns=response.perf_ns,
        data=endpoint.make_text_response_data(" ".join(texts)),
    )


class _RivaTextListEndpoint(BaseEndpoint):
    """Base for Riva NLP endpoints that accept a list of texts + language_code."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = get_extra(self)
        self._language_code: str = extra.get("language_code", "en-US")

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        texts = _extract_texts(request_info)
        return {
            "texts": texts,
            "language_code": self._language_code,
        }


class RivaTextClassifyEndpoint(_RivaTextListEndpoint):
    """Riva text classification endpoint.

    Sends text to Riva ClassifyText and returns classification labels with scores.
    """

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        return _parse_json_response(response, self)


class RivaTokenClassifyEndpoint(_RivaTextListEndpoint):
    """Riva token classification endpoint.

    Sends text to Riva ClassifyTokens and returns per-token classification labels.
    """

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        return _parse_json_response(response, self)


class RivaTransformTextEndpoint(_RivaTextListEndpoint):
    """Riva text transformation endpoint.

    Sends text to Riva TransformText and returns transformed text.
    """

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        return _parse_texts_response(response, self)


class RivaPunctuateTextEndpoint(_RivaTextListEndpoint):
    """Riva text punctuation endpoint.

    Sends text to Riva PunctuateText and returns punctuated text.
    """

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        return _parse_texts_response(response, self)


class RivaNaturalQueryEndpoint(BaseEndpoint):
    """Riva natural query endpoint.

    Sends a query with context to Riva NaturalQuery and returns answers.
    Configure via --extra: context (the document to search).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = get_extra(self)
        self._context: str = extra.get("context", "")
        self._top_n: int = int(extra.get("top_n", 1))

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        texts = _extract_texts(request_info)
        query = " ".join(texts) if texts else ""
        return {
            "query": query,
            "context": self._context,
            "top_n": self._top_n,
        }

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        json_obj = response.get_json()
        if not json_obj:
            return None
        results = json_obj.get("results", [])
        if not results:
            return None
        # Return the top answer as text
        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=self.make_text_response_data(results[0].get("answer", "")),
        )


class RivaAnalyzeIntentEndpoint(BaseEndpoint):
    """Riva intent analysis endpoint.

    Sends text to Riva AnalyzeIntent and returns intent + slots.
    Configure via --extra: domain.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = get_extra(self)
        self._domain: str = extra.get("domain", "")

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        texts = _extract_texts(request_info)
        query = " ".join(texts) if texts else ""
        payload: dict[str, Any] = {"query": query}
        if self._domain:
            payload["domain"] = self._domain
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        return _parse_json_response(response, self)


class RivaAnalyzeEntitiesEndpoint(BaseEndpoint):
    """Riva entity analysis endpoint.

    Sends text to Riva AnalyzeEntities and returns named entities.
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        texts = _extract_texts(request_info)
        query = " ".join(texts) if texts else ""
        return {"query": query}

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        return _parse_json_response(response, self)
