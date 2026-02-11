# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EngineGenerateEndpoint.

Verifies:
- format_payload produces messages + sampling_params (not OpenAI format)
- max_tokens and extra inputs land in sampling_params
- parse_response handles InEngineResponse directly
- parse_response falls back to parent for non-InEngineResponse types
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from aiperf.common.models import InEngineResponse, Text, Turn
from aiperf.common.models.record_models import InferenceServerResponse
from aiperf.endpoints.engine_generate import EngineGenerateEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def model_endpoint():
    """Create a test ModelEndpointInfo for engine generate."""
    return create_model_endpoint(
        EndpointType.CHAT, base_url="vllm://meta-llama/Llama-3.1-8B"
    )


@pytest.fixture
def endpoint(model_endpoint):
    """Create an EngineGenerateEndpoint instance."""
    return create_endpoint_with_mock_transport(EngineGenerateEndpoint, model_endpoint)


# ============================================================
# format_payload
# ============================================================


class TestFormatPayload:
    """Verify format_payload produces the engine generate payload structure."""

    def test_returns_messages_and_sampling_params(
        self, endpoint, model_endpoint
    ) -> None:
        turn = Turn(texts=[Text(contents=["Hello!"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "messages" in payload
        assert "sampling_params" in payload
        assert "model" in payload
        assert "stream" in payload
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["content"] == "Hello!"

    def test_no_openai_keys(self, endpoint, model_endpoint) -> None:
        """Payload should NOT contain OpenAI-specific keys."""
        turn = Turn(texts=[Text(contents=["Hi"])], model="test-model", max_tokens=256)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "max_completion_tokens" not in payload
        assert "stream_options" not in payload

    def test_includes_max_tokens_from_turn(self, endpoint, model_endpoint) -> None:
        turn = Turn(
            texts=[Text(contents=["Generate"])], model="test-model", max_tokens=512
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["sampling_params"]["max_tokens"] == 512

    def test_no_max_tokens_when_none(self, endpoint, model_endpoint) -> None:
        turn = Turn(
            texts=[Text(contents=["Generate"])], model="test-model", max_tokens=None
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "max_tokens" not in payload["sampling_params"]

    def test_includes_extra_inputs(self) -> None:
        extra = [("temperature", 0.7), ("top_k", 50)]
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT,
            base_url="vllm://org/model",
            extra=extra,
        )
        endpoint = create_endpoint_with_mock_transport(
            EngineGenerateEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["Test"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["sampling_params"]["temperature"] == 0.7
        assert payload["sampling_params"]["top_k"] == 50

    def test_system_message_prepended(self, endpoint, model_endpoint) -> None:
        turn = Turn(texts=[Text(contents=["Hello"])], model="test-model")
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            system_message="You are a helpful assistant.",
        )

        payload = endpoint.format_payload(request_info)

        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are a helpful assistant."

    def test_empty_turns_raises(self, endpoint, model_endpoint) -> None:
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])
        request_info.turns = []

        with pytest.raises(ValueError, match="at least one turn"):
            endpoint.format_payload(request_info)

    def test_model_from_turn(self, endpoint, model_endpoint) -> None:
        turn = Turn(texts=[Text(contents=["Hi"])], model="custom-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "custom-model"

    def test_model_fallback_to_endpoint(self, endpoint, model_endpoint) -> None:
        turn = Turn(texts=[Text(contents=["Hi"])], model=None)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == model_endpoint.primary_model_name


# ============================================================
# parse_response
# ============================================================


class TestParseResponse:
    """Verify parse_response handles InEngineResponse directly."""

    def test_extracts_text_and_usage(self, endpoint) -> None:
        response = InEngineResponse(
            perf_ns=123456789,
            text="Generated output",
            input_tokens=10,
            output_tokens=20,
            finish_reason="stop",
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert parsed.data.text == "Generated output"
        assert parsed.usage.prompt_tokens == 10
        assert parsed.usage.completion_tokens == 20
        assert parsed.usage.total_tokens == 30

    def test_returns_none_for_non_in_engine_response(self, endpoint) -> None:
        """Non-InEngineResponse without JSON should return None (parent behavior)."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 100
        mock_response.get_json.return_value = None
        mock_response.get_text.return_value = None

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None
