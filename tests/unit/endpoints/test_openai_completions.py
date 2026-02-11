# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_mock_response, create_request_info


class TestCompletionsEndpoint:
    """Test CompletionsEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo."""
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.RANDOM,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000",
                custom_endpoint="/v1/chat/completions",
                api_key="test-api-key",
            ),
        )

    def test_format_payload_basic(self, model_endpoint, sample_conversations):
        endpoint = CompletionsEndpoint(model_endpoint)
        # Use the first turn from the sample_conversations fixture
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        expected_payload = {
            "prompt": ["Hello, world!"],
            "model": "test-model",
            "stream": False,
        }
        assert payload == expected_payload

    def test_format_payload_with_extra_options(
        self, model_endpoint, sample_conversations
    ):
        endpoint = CompletionsEndpoint(model_endpoint)
        # Use the first turn from the sample_conversations fixture
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        turns[0].max_tokens = 50
        model_endpoint.endpoint.streaming = True
        model_endpoint.endpoint.extra = [("ignore_eos", True)]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        expected_payload = {
            "prompt": ["Hello, world!"],
            "model": "test-model",
            "stream": True,
            "max_tokens": 50,
            "ignore_eos": True,
        }
        assert payload == expected_payload

    @pytest.mark.parametrize(
        "streaming,use_server_token_count,user_extra,expected_stream_options",
        [
            # Auto-add when both flags enabled
            (True, True, None, {"include_usage": True}),
            # Don't add when not streaming
            (False, True, None, None),
            # Don't add when flag disabled
            (True, False, None, None),
            # Don't add when neither enabled
            (False, False, None, None),
            # Preserve user's include_usage=False
            (True, True, [("stream_options", {"include_usage": False})], {"include_usage": False}),
            # Merge with user's other options
            (True, True, [("stream_options", {"continuous_updates": True})], {"continuous_updates": True, "include_usage": True}),
        ],
    )  # fmt: skip
    def test_stream_options_auto_configuration(
        self,
        model_endpoint,
        sample_conversations,
        streaming,
        use_server_token_count,
        user_extra,
        expected_stream_options,
    ):
        """Verify stream_options.include_usage is automatically configured based on flags and user settings."""
        endpoint = CompletionsEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        model_endpoint.endpoint.streaming = streaming
        model_endpoint.endpoint.use_server_token_count = use_server_token_count
        if user_extra:
            model_endpoint.endpoint.extra = user_extra
        request_info = create_request_info(turns=turns, model_endpoint=model_endpoint)
        payload = endpoint.format_payload(request_info)

        if expected_stream_options is None:
            assert "stream_options" not in payload
        else:
            assert "stream_options" in payload
            assert payload["stream_options"] == expected_stream_options

    def test_format_payload_with_service_tier(
        self, model_endpoint, sample_conversations
    ):
        """Verify service_tier from Turn is included in payload."""
        endpoint = CompletionsEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turn.service_tier = "flex"
        turns = [turn]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        assert payload["service_tier"] == "flex"

    def test_format_payload_without_service_tier(
        self, model_endpoint, sample_conversations
    ):
        """Verify service_tier is not in payload when not set."""
        endpoint = CompletionsEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        assert "service_tier" not in payload

    def test_parse_response_extracts_service_tier(self, model_endpoint):
        """Verify service_tier is extracted from response JSON into metadata."""
        endpoint = CompletionsEndpoint(model_endpoint)
        response = create_mock_response(
            json_data={
                "object": "text_completion",
                "service_tier": "priority",
                "choices": [{"text": "Hello"}],
            }
        )
        parsed = endpoint.parse_response(response)
        assert parsed is not None
        assert parsed.metadata.get("service_tier") == "priority"

    def test_parse_response_no_service_tier(self, model_endpoint):
        """Verify metadata is empty when response has no service_tier."""
        endpoint = CompletionsEndpoint(model_endpoint)
        response = create_mock_response(
            json_data={
                "object": "text_completion",
                "choices": [{"text": "Hello"}],
            }
        )
        parsed = endpoint.parse_response(response)
        assert parsed is not None
        assert "service_tier" not in parsed.metadata
