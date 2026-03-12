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

from typing import Any
from unittest.mock import Mock

import pytest

from aiperf.common.models import InEngineResponse, Text, Turn
from aiperf.common.models.record_models import InferenceServerResponse
from aiperf.endpoints.engine_generate import (
    EngineGenerateEndpoint,
    SGLangGenerateEndpoint,
    TRTLLMGenerateEndpoint,
    VLLMGenerateEndpoint,
)
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


# ============================================================
# VLLMGenerateEndpoint — detokenize bool coercion
# ============================================================


class TestVLLMDetokenizeCoercion:
    """Verify detokenize string values are coerced to bool."""

    @pytest.fixture
    def vllm_endpoint(self):
        """Create a VLLMGenerateEndpoint instance."""
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT, base_url="vllm://org/model"
        )
        return create_endpoint_with_mock_transport(VLLMGenerateEndpoint, model_endpoint)

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("false", False),
            ("False", False),
            ("true", True),
            ("True", True),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
            (True, True),
            (False, False),
        ],
    )
    def test_detokenize_coerced_to_bool(self, input_val: Any, expected: bool) -> None:
        extra = [("detokenize", input_val)]
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT, base_url="vllm://org/model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            VLLMGenerateEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["Hi"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["sampling_params"]["detokenize"] is expected

    def test_no_detokenize_key_when_not_provided(self, vllm_endpoint) -> None:
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT, base_url="vllm://org/model"
        )
        turn = Turn(texts=[Text(contents=["Hi"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = vllm_endpoint.format_payload(request_info)
        assert "detokenize" not in payload["sampling_params"]


# ============================================================
# parse_response — Speculative Decoding Metadata Propagation
# ============================================================


class TestParseResponseSpecDecodeMetadata:
    """Verify parse_response propagates decode_iterations/max_draft_len to metadata."""

    def test_decode_iterations_in_metadata(self, endpoint) -> None:
        response = InEngineResponse(
            perf_ns=100,
            text="output",
            input_tokens=10,
            output_tokens=20,
            decode_iterations=7,
            max_draft_len=5,
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.metadata["decode_iterations"] == 7
        assert parsed.metadata["max_draft_len"] == 5

    def test_no_spec_decode_metadata_when_none(self, endpoint) -> None:
        """When decode_iterations and max_draft_len are None, metadata is empty."""
        response = InEngineResponse(
            perf_ns=100,
            text="output",
            input_tokens=10,
            output_tokens=20,
            decode_iterations=None,
            max_draft_len=None,
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert "decode_iterations" not in parsed.metadata
        assert "max_draft_len" not in parsed.metadata

    def test_decode_iterations_without_max_draft_len(self, endpoint) -> None:
        """decode_iterations present but max_draft_len is None."""
        response = InEngineResponse(
            perf_ns=100,
            text="output",
            input_tokens=10,
            output_tokens=20,
            decode_iterations=3,
            max_draft_len=None,
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.metadata["decode_iterations"] == 3
        assert "max_draft_len" not in parsed.metadata

    def test_max_draft_len_without_decode_iterations(self, endpoint) -> None:
        """max_draft_len present but decode_iterations is None (unusual but valid)."""
        response = InEngineResponse(
            perf_ns=100,
            text="output",
            input_tokens=10,
            output_tokens=20,
            decode_iterations=None,
            max_draft_len=5,
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert "decode_iterations" not in parsed.metadata
        assert parsed.metadata["max_draft_len"] == 5

    def test_decode_iterations_zero_is_propagated(self, endpoint) -> None:
        """decode_iterations=0 is a valid value and should be propagated."""
        response = InEngineResponse(
            perf_ns=100,
            text="output",
            input_tokens=10,
            output_tokens=1,
            decode_iterations=0,
            max_draft_len=5,
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.metadata["decode_iterations"] == 0
        assert parsed.metadata["max_draft_len"] == 5


# ============================================================
# SGLangGenerateEndpoint — max_new_tokens remapping
# ============================================================


class TestSGLangGenerateEndpoint:
    """Verify SGLang-specific sampling param remapping."""

    @pytest.fixture
    def sglang_endpoint(self):
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT, base_url="sglang://org/model"
        )
        return create_endpoint_with_mock_transport(
            SGLangGenerateEndpoint, model_endpoint
        )

    def test_max_tokens_remapped_to_max_new_tokens(self) -> None:
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT, base_url="sglang://org/model"
        )
        endpoint = create_endpoint_with_mock_transport(
            SGLangGenerateEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["Hi"])], model="test-model", max_tokens=128)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "max_new_tokens" in payload["sampling_params"]
        assert "max_tokens" not in payload["sampling_params"]
        assert payload["sampling_params"]["max_new_tokens"] == 128

    def test_no_max_tokens_no_max_new_tokens(self, sglang_endpoint) -> None:
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT, base_url="sglang://org/model"
        )
        turn = Turn(texts=[Text(contents=["Hi"])], model="test-model", max_tokens=None)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = sglang_endpoint.format_payload(request_info)

        assert "max_tokens" not in payload["sampling_params"]
        assert "max_new_tokens" not in payload["sampling_params"]


# ============================================================
# TRTLLMGenerateEndpoint — key remapping
# ============================================================


class TestTRTLLMGenerateEndpoint:
    """Verify TRT-LLM-specific sampling param remapping."""

    def _make_endpoint(
        self, extra: list[tuple[str, Any]] | None = None
    ) -> TRTLLMGenerateEndpoint:
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT, base_url="trtllm://org/model", extra=extra
        )
        return create_endpoint_with_mock_transport(
            TRTLLMGenerateEndpoint, model_endpoint
        )

    def test_seed_remapped_to_random_seed(self) -> None:
        endpoint = self._make_endpoint(extra=[("seed", 42)])
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT, base_url="trtllm://org/model", extra=[("seed", 42)]
        )
        turn = Turn(texts=[Text(contents=["Hi"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "random_seed" in payload["sampling_params"]
        assert "seed" not in payload["sampling_params"]
        assert payload["sampling_params"]["random_seed"] == 42

    def test_stop_string_remapped_to_stop_words_list(self) -> None:
        endpoint = self._make_endpoint(extra=[("stop", "</s>")])
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT, base_url="trtllm://org/model", extra=[("stop", "</s>")]
        )
        turn = Turn(texts=[Text(contents=["Hi"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "stop_words" in payload["sampling_params"]
        assert "stop" not in payload["sampling_params"]
        assert payload["sampling_params"]["stop_words"] == ["</s>"]

    def test_stop_list_stays_as_list(self) -> None:
        endpoint = self._make_endpoint(extra=[("stop", ["</s>", "<|end|>"])])
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT,
            base_url="trtllm://org/model",
            extra=[("stop", ["</s>", "<|end|>"])],
        )
        turn = Turn(texts=[Text(contents=["Hi"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["sampling_params"]["stop_words"] == ["</s>", "<|end|>"]

    def test_parse_response_propagates_spec_decode_metadata(self) -> None:
        """TRTLLMGenerateEndpoint inherits parse_response spec decode propagation."""
        endpoint = self._make_endpoint()

        response = InEngineResponse(
            perf_ns=100,
            text="output",
            input_tokens=10,
            output_tokens=20,
            decode_iterations=5,
            max_draft_len=3,
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.metadata["decode_iterations"] == 5
        assert parsed.metadata["max_draft_len"] == 3
