# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseInEngineTransport shared functionality.

Focuses on:
- Model path extraction from URL schemes
- Error record construction
- Full send_request flow with mocked _generate
- InEngineResponse construction
- Message-to-prompt conversion with fallback
- Warmup prompt generation with various target_tokens values
- Warmup config extraction from CLI string params
- Warmup exception handling (graceful degradation)
- No stale _pop_warmup_iterations references
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import (
    InEngineResponse,
    RequestInfo,
    RequestRecord,
)
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.plugin.enums import EndpointType
from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import FirstTokenCallback
from aiperf.transports.in_engine.base_in_engine_transport import BaseInEngineTransport

# ============================================================
# Concrete Test Subclass (implements abstract methods)
# ============================================================


class ConcreteTestTransport(BaseInEngineTransport):
    """Concrete subclass of BaseInEngineTransport for testing shared utilities.

    Provides simple implementations of all abstract methods so that
    the shared base class behaviour can be tested in isolation.
    """

    def __init__(
        self,
        *,
        generate_result: tuple[str, int, int, str] | None = None,
        generate_error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._generate_result = generate_result or ("Hello world", 10, 5, "stop")
        self._generate_error = generate_error

    @classmethod
    def metadata(cls) -> TransportMetadata:
        return TransportMetadata(
            transport_type="test-engine",
            url_schemes=["test-engine"],
        )

    async def _init_engine(self) -> None:
        pass

    async def _start_engine(self) -> None:
        pass

    async def _stop_engine(self) -> None:
        pass

    async def _warmup_single(
        self, prompt: str, max_tokens: int, *, streaming: bool
    ) -> None:
        pass

    async def _generate(
        self,
        *,
        messages: list[dict[str, Any]],
        sampling_params: Any,
        request_id: str,
        first_token_callback: FirstTokenCallback | None = None,
        input_ids: list[int] | None = None,
    ) -> tuple[str, int, int, str]:
        if self._generate_error:
            raise self._generate_error
        return self._generate_result


# ============================================================
# Fixtures
# ============================================================


def _make_model_endpoint(
    base_url: str = "test-engine://meta-llama/Llama-3.1-8B",
    model_name: str = "meta-llama/Llama-3.1-8B",
) -> ModelEndpointInfo:
    """Create a ModelEndpointInfo for in-engine transport testing."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name=model_name)],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_urls=[base_url],
        ),
    )


def _make_request_info(model_endpoint: ModelEndpointInfo) -> RequestInfo:
    """Create a RequestInfo with sensible defaults for transport tests."""
    return RequestInfo(
        model_endpoint=model_endpoint,
        turns=[],
        endpoint_headers={},
        endpoint_params={},
        turn_index=0,
        credit_num=1,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="req-001",
        x_correlation_id="corr-001",
        conversation_id="conv-001",
    )


@pytest.fixture
def model_endpoint() -> ModelEndpointInfo:
    """Default model endpoint with test-engine:// scheme."""
    return _make_model_endpoint()


@pytest.fixture
def transport(model_endpoint: ModelEndpointInfo) -> ConcreteTestTransport:
    """Create a ConcreteTestTransport with default config."""
    return ConcreteTestTransport(model_endpoint=model_endpoint)


@pytest.fixture
def request_info(model_endpoint: ModelEndpointInfo) -> RequestInfo:
    """Create a basic RequestInfo."""
    return _make_request_info(model_endpoint)


# ============================================================
# Model Path Extraction
# ============================================================


class TestExtractModelPath:
    """Verify model path extraction prefers --model name, falls back to URL."""

    def test_prefers_primary_model_name(self) -> None:
        """Model path uses --model name even when URL has an engine scheme."""
        endpoint = _make_model_endpoint(
            base_url="vllm://url-org/url-model",
            model_name="cli-org/cli-model",
        )
        transport = ConcreteTestTransport(model_endpoint=endpoint)
        assert transport._extract_model_path() == "cli-org/cli-model"

    def test_prefers_model_name_over_plain_http_url(self) -> None:
        """Model path uses --model name when URL has no engine scheme."""
        endpoint = _make_model_endpoint(base_url="http://localhost:8000")
        transport = ConcreteTestTransport(model_endpoint=endpoint)
        assert transport._extract_model_path() == "meta-llama/Llama-3.1-8B"

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("vllm://meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B"),
            ("sglang://org/model-name", "org/model-name"),
            ("trtllm://nvidia/model", "nvidia/model"),
            param("vllm:///path/to/model", "/path/to/model", id="absolute-path"),
            param("vllm:///opt/models/llama", "/opt/models/llama", id="absolute-path-deep"),
            param("vllm://meta-llama/Llama-3.1-8B/", "meta-llama/Llama-3.1-8B", id="trailing-slash-stripped"),
            param("sglang://org/model///", "org/model", id="multiple-trailing-slashes"),
        ],
    )  # fmt: skip
    def test_url_extraction_when_model_name_matches(
        self, url: str, expected: str
    ) -> None:
        """Model path returned correctly when --model matches URL path."""
        endpoint = _make_model_endpoint(base_url=url, model_name=expected)
        transport = ConcreteTestTransport(model_endpoint=endpoint)
        assert transport._extract_model_path() == expected


# ============================================================
# Error Record Building
# ============================================================


class TestBuildErrorRecord:
    """Verify error record construction from exceptions."""

    def test_build_error_record(
        self, transport: ConcreteTestTransport, request_info: RequestInfo
    ) -> None:
        error = RuntimeError("Engine OOM")
        start_ns = 1000

        record = transport._build_error_record(
            request_info=request_info,
            start_perf_ns=start_ns,
            error=error,
        )

        assert isinstance(record, RequestRecord)
        assert record.error is not None
        assert record.error.type == "RuntimeError"
        assert "Engine OOM" in record.error.message
        assert record.start_perf_ns == start_ns
        assert record.end_perf_ns is not None
        assert record.end_perf_ns >= start_ns
        assert record.request_info is request_info


# ============================================================
# send_request Flow
# ============================================================


class TestSendRequest:
    """Verify the full send_request flow with InEngineResponse."""

    @pytest.mark.asyncio
    async def test_send_request_returns_in_engine_response(self) -> None:
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_result=("Generated text", 15, 20, "stop"),
        )
        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {"temperature": 0.7},
        }

        record = await transport.send_request(request_info, payload)

        assert isinstance(record, RequestRecord)
        assert record.status == 200
        assert record.error is None
        assert record.request_info is request_info
        assert len(record.responses) == 1
        assert record.start_perf_ns > 0
        assert record.end_perf_ns >= record.start_perf_ns

        # Verify InEngineResponse content
        response = record.responses[0]
        assert isinstance(response, InEngineResponse)
        assert response.text == "Generated text"
        assert response.input_tokens == 15
        assert response.output_tokens == 20
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_send_request_error_returns_error_record(self) -> None:
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_error=RuntimeError("CUDA out of memory"),
        )
        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        record = await transport.send_request(request_info, payload)

        assert isinstance(record, RequestRecord)
        assert record.error is not None
        assert record.error.type == "RuntimeError"
        assert "CUDA out of memory" in record.error.message
        assert record.start_perf_ns > 0
        assert record.end_perf_ns >= record.start_perf_ns

    @pytest.mark.asyncio
    async def test_send_request_cancelled_error_reraises(self) -> None:
        """CancelledError must propagate - it should not be caught."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_error=None,
        )
        # Patch _generate to raise CancelledError
        transport._generate = AsyncMock(side_effect=asyncio.CancelledError)  # type: ignore[method-assign]

        request_info = _make_request_info(model_endpoint)
        payload = {"messages": [{"role": "user", "content": "Hi"}]}

        with pytest.raises(asyncio.CancelledError):
            await transport.send_request(request_info, payload)


# ============================================================
# get_url / get_transport_headers
# ============================================================


class TestTransportInterface:
    """Verify BaseTransport interface methods for in-engine transports."""

    def test_get_url_returns_model_path(
        self, transport: ConcreteTestTransport, request_info: RequestInfo
    ) -> None:
        transport._model_path = "meta-llama/Llama-3.1-8B"
        assert transport.get_url(request_info) == "meta-llama/Llama-3.1-8B"

    def test_get_transport_headers_returns_empty(
        self, transport: ConcreteTestTransport, request_info: RequestInfo
    ) -> None:
        assert transport.get_transport_headers(request_info) == {}


# ============================================================
# configure() – deferred engine loading
# ============================================================


class TestConfigure:
    """Verify configure() delegates to _start_engine with daemon flag handling."""

    @pytest.mark.asyncio
    async def test_configure_calls_start_engine(
        self, transport: ConcreteTestTransport
    ) -> None:
        transport._start_engine = AsyncMock()  # type: ignore[method-assign]
        await transport.configure()
        transport._start_engine.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_configure_restores_daemon_flag_on_success(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        transport = ConcreteTestTransport(model_endpoint=model_endpoint)
        transport._start_engine = AsyncMock()  # type: ignore[method-assign]

        import multiprocessing

        proc = multiprocessing.current_process()
        original_daemon = proc.daemon
        await transport.configure()
        assert proc.daemon == original_daemon

    @pytest.mark.asyncio
    async def test_configure_restores_daemon_flag_on_error(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        transport = ConcreteTestTransport(model_endpoint=model_endpoint)
        transport._start_engine = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

        import multiprocessing

        proc = multiprocessing.current_process()
        original_daemon = proc.daemon
        with pytest.raises(RuntimeError, match="boom"):
            await transport.configure()
        assert proc.daemon == original_daemon


# ============================================================
# _messages_to_prompt (Fallback)
# ============================================================


class TestMessagesToPrompt:
    """Verify message-to-prompt conversion in the base class."""

    def test_fallback_format_includes_roles(
        self, transport: ConcreteTestTransport
    ) -> None:
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        prompt = transport._messages_to_prompt(messages)

        assert "<|system|>" in prompt
        assert "You are helpful" in prompt
        assert "<|user|>" in prompt
        assert "Hello" in prompt
        assert "<|assistant|>" in prompt

    def test_fallback_with_multimodal_content(
        self, transport: ConcreteTestTransport
    ) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com"}},
                ],
            },
        ]
        prompt = transport._messages_to_prompt(messages)

        assert "What is this?" in prompt
        assert "image_url" not in prompt

    def test_get_tokenizer_default_returns_none(
        self, transport: ConcreteTestTransport
    ) -> None:
        assert transport._get_tokenizer() is None


# ============================================================
# Streaming Response Construction (first_token_perf_ns)
# ============================================================


class TestStreamingResponseConstruction:
    """Verify send_request produces two responses when _first_token_perf_ns is set."""

    @pytest.mark.asyncio
    async def test_streaming_produces_two_responses(self) -> None:
        """When _generate sets _first_token_perf_ns, send_request returns two responses."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_result=("Full text", 10, 20, "stop"),
        )

        # Simulate what vLLM streaming does: set _first_token_perf_ns during _generate
        original_generate = transport._generate

        async def generate_with_ttft(**kwargs: Any) -> tuple[str, int, int, str]:
            result = await original_generate(**kwargs)
            transport._first_token_perf_ns = 500_000
            return result

        transport._generate = generate_with_ttft  # type: ignore[method-assign]

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        record = await transport.send_request(request_info, payload)

        assert len(record.responses) == 2

        # First response is the TTFT marker
        first = record.responses[0]
        assert isinstance(first, InEngineResponse)
        assert first.perf_ns == 500_000
        assert first.text == ""
        assert first.input_tokens == 0
        assert first.output_tokens == 0
        assert first.finish_reason == ""

        # Second response has the full content
        final = record.responses[1]
        assert isinstance(final, InEngineResponse)
        assert final.text == "Full text"
        assert final.input_tokens == 10
        assert final.output_tokens == 20
        assert final.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_non_streaming_produces_single_response(self) -> None:
        """Without _first_token_perf_ns, send_request returns a single response."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_result=("Output", 5, 10, "length"),
        )
        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        record = await transport.send_request(request_info, payload)

        assert len(record.responses) == 1
        assert record.responses[0].text == "Output"

    @pytest.mark.asyncio
    async def test_streaming_fires_first_token_callback(self) -> None:
        """first_token_callback is called with TTFT and first response when streaming."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_result=("Text", 5, 10, "stop"),
        )

        original_generate = transport._generate

        async def generate_with_ttft(**kwargs: Any) -> tuple[str, int, int, str]:
            result = await original_generate(**kwargs)
            transport._first_token_perf_ns = 500_000
            return result

        transport._generate = generate_with_ttft  # type: ignore[method-assign]

        callback = AsyncMock(return_value=True)
        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        await transport.send_request(
            request_info, payload, first_token_callback=callback
        )

        callback.assert_awaited_once()
        ttft_ns, first_response = callback.call_args.args
        assert isinstance(ttft_ns, int)
        assert isinstance(first_response, InEngineResponse)
        assert first_response.text == ""

    @pytest.mark.asyncio
    async def test_first_token_perf_ns_reset_after_request(self) -> None:
        """_first_token_perf_ns is reset to None after send_request completes."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(model_endpoint=model_endpoint)

        original_generate = transport._generate

        async def generate_with_ttft(**kwargs: Any) -> tuple[str, int, int, str]:
            result = await original_generate(**kwargs)
            transport._first_token_perf_ns = 123
            return result

        transport._generate = generate_with_ttft  # type: ignore[method-assign]

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        await transport.send_request(request_info, payload)
        assert transport._first_token_perf_ns is None


# ============================================================
# _pop_warmup_config
# ============================================================


class TestPopWarmupConfig:
    """Verify _pop_warmup_config extracts all warmup params and removes from dict."""

    def test_pops_all_warmup_params(self, transport: ConcreteTestTransport) -> None:
        params = {
            "warmup_iterations": "5",
            "warmup_input_tokens": "256",
            "warmup_output_tokens": "8",
            "other_key": "val",
        }
        transport._pop_warmup_config(params)

        assert transport._warmup_iterations == 5
        assert transport._warmup_input_tokens == 256
        assert transport._warmup_output_tokens == 8
        assert "warmup_iterations" not in params
        assert "warmup_input_tokens" not in params
        assert "warmup_output_tokens" not in params
        assert params == {"other_key": "val"}

    def test_no_warmup_params_leaves_defaults(
        self, transport: ConcreteTestTransport
    ) -> None:
        params = {"other_key": "val"}
        transport._pop_warmup_config(params)

        assert transport._warmup_iterations == 0
        assert transport._warmup_input_tokens == 128
        assert transport._warmup_output_tokens == 4
        assert params == {"other_key": "val"}

    @pytest.mark.parametrize(
        "key,value,attr,expected",
        [
            ("warmup_iterations", "10", "_warmup_iterations", 10),
            ("warmup_input_tokens", "512", "_warmup_input_tokens", 512),
            ("warmup_output_tokens", "16", "_warmup_output_tokens", 16),
        ],
    )
    def test_individual_params_coerce_to_int(
        self,
        transport: ConcreteTestTransport,
        key: str,
        value: str,
        attr: str,
        expected: int,
    ) -> None:
        params = {key: value}
        transport._pop_warmup_config(params)
        assert getattr(transport, attr) == expected
        assert key not in params


# ============================================================
# _generate_warmup_prompt
# ============================================================


class TestGenerateWarmupPrompt:
    """Verify warmup prompt generation with and without a tokenizer."""

    def test_with_mock_tokenizer_produces_correct_length(
        self, transport: ConcreteTestTransport
    ) -> None:
        """When a tokenizer is available, prompt has exactly target_tokens token IDs."""
        mock_tokenizer = MagicMock()
        # Seed text encodes to 10 token IDs
        mock_tokenizer.encode.return_value = list(range(10))
        mock_tokenizer.decode.return_value = "decoded prompt text"

        with patch.object(transport, "_get_tokenizer", return_value=mock_tokenizer):
            transport._generate_warmup_prompt(128)

        # Verify encode was called with the seed text
        mock_tokenizer.encode.assert_called_once_with(transport._WARMUP_SEED_TEXT)
        # Verify decode was called with exactly 128 token IDs
        decode_args = mock_tokenizer.decode.call_args[0][0]
        assert len(decode_args) == 128

    def test_with_tokenizer_truncates_to_target(
        self, transport: ConcreteTestTransport
    ) -> None:
        """Token IDs are repeated and truncated to exactly target_tokens."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "result"

        with patch.object(transport, "_get_tokenizer", return_value=mock_tokenizer):
            transport._generate_warmup_prompt(7)

        decode_args = mock_tokenizer.decode.call_args[0][0]
        assert decode_args == [1, 2, 3, 1, 2, 3, 1]

    def test_fallback_without_tokenizer(self, transport: ConcreteTestTransport) -> None:
        """Without a tokenizer, falls back to repeating seed text."""
        prompt = transport._generate_warmup_prompt(128)

        # Should contain the seed text repeated
        seed = transport._WARMUP_SEED_TEXT
        assert seed.strip() in prompt
        # Rough word count should be >= target (each repeat has ~10 words)
        assert len(prompt) > 0

    def test_fallback_with_empty_encode(self, transport: ConcreteTestTransport) -> None:
        """If tokenizer.encode returns empty, falls back to text repetition."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = []

        with patch.object(transport, "_get_tokenizer", return_value=mock_tokenizer):
            prompt = transport._generate_warmup_prompt(128)

        # Should still produce a non-empty prompt via fallback
        assert len(prompt) > 0
        mock_tokenizer.decode.assert_not_called()


# ============================================================
# _run_warmup
# ============================================================


class TestRunWarmup:
    """Verify _run_warmup orchestration logic."""

    @pytest.mark.asyncio
    async def test_zero_iterations_returns_immediately(
        self, transport: ConcreteTestTransport
    ) -> None:
        """When _warmup_iterations is 0, _warmup_single is never called."""
        transport._warmup_iterations = 0
        transport._warmup_single = AsyncMock()  # type: ignore[method-assign]

        await transport._run_warmup()

        transport._warmup_single.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("iterations", [1, 3, 5])
    async def test_calls_warmup_single_n_times(
        self, transport: ConcreteTestTransport, iterations: int
    ) -> None:
        """_warmup_single is called exactly N times with correct args."""
        transport._warmup_iterations = iterations
        transport._warmup_input_tokens = 64
        transport._warmup_output_tokens = 8
        transport._warmup_single = AsyncMock()  # type: ignore[method-assign]

        await transport._run_warmup()

        assert transport._warmup_single.await_count == iterations
        for call in transport._warmup_single.call_args_list:
            prompt, max_tokens = call.args
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            assert max_tokens == 8
            assert call.kwargs["streaming"] is False

    @pytest.mark.asyncio
    async def test_matches_streaming_config_from_endpoint(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        """_warmup_single receives streaming=True when endpoint.streaming is True."""
        model_endpoint.endpoint.streaming = True
        transport = ConcreteTestTransport(model_endpoint=model_endpoint)
        transport._warmup_iterations = 2
        transport._warmup_single = AsyncMock()  # type: ignore[method-assign]

        await transport._run_warmup()

        for call in transport._warmup_single.call_args_list:
            assert call.kwargs["streaming"] is True

    @pytest.mark.asyncio
    async def test_matches_non_streaming_config_from_endpoint(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        """_warmup_single receives streaming=False when endpoint.streaming is False."""
        model_endpoint.endpoint.streaming = False
        transport = ConcreteTestTransport(model_endpoint=model_endpoint)
        transport._warmup_iterations = 1
        transport._warmup_single = AsyncMock()  # type: ignore[method-assign]

        await transport._run_warmup()

        call = transport._warmup_single.call_args
        assert call.kwargs["streaming"] is False

    @pytest.mark.asyncio
    async def test_negative_iterations_returns_immediately(
        self, transport: ConcreteTestTransport
    ) -> None:
        """Negative _warmup_iterations acts like zero (no warmup)."""
        transport._warmup_iterations = -1
        transport._warmup_single = AsyncMock()  # type: ignore[method-assign]

        await transport._run_warmup()

        transport._warmup_single.assert_not_awaited()


# ============================================================
# Warmup Prompt Target Tokens Variations
# ============================================================


class TestWarmupPromptTargetTokens:
    """Verify warmup prompt generation across various target_tokens values."""

    @pytest.mark.parametrize(
        "target_tokens",
        [1, 10, 128, 1024],
    )  # fmt: skip
    def test_tokenizer_path_produces_exact_token_count(
        self, transport: ConcreteTestTransport, target_tokens: int
    ) -> None:
        """Tokenizer-based prompt produces exactly target_tokens token IDs."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [100, 200, 300, 400, 500]
        mock_tokenizer.decode.return_value = "decoded"

        with patch.object(transport, "_get_tokenizer", return_value=mock_tokenizer):
            transport._generate_warmup_prompt(target_tokens)

        decode_args = mock_tokenizer.decode.call_args[0][0]
        assert len(decode_args) == target_tokens

    @pytest.mark.parametrize(
        "target_tokens",
        [1, 10, 128, 1024],
    )  # fmt: skip
    def test_fallback_path_produces_nonempty_prompt(
        self, transport: ConcreteTestTransport, target_tokens: int
    ) -> None:
        """Without tokenizer, prompt is always non-empty for any target_tokens."""
        prompt = transport._generate_warmup_prompt(target_tokens)
        assert len(prompt) > 0

    def test_small_vocab_tokenizer_single_token(
        self, transport: ConcreteTestTransport
    ) -> None:
        """Tokenizer with very small vocab (single token) repeats correctly."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [42]
        mock_tokenizer.decode.return_value = "x " * 10

        with patch.object(transport, "_get_tokenizer", return_value=mock_tokenizer):
            transport._generate_warmup_prompt(10)

        decode_args = mock_tokenizer.decode.call_args[0][0]
        assert decode_args == [42] * 10

    def test_small_vocab_tokenizer_two_tokens(
        self, transport: ConcreteTestTransport
    ) -> None:
        """Tokenizer with two-token vocab repeats and truncates correctly."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2]
        mock_tokenizer.decode.return_value = "ab " * 3

        with patch.object(transport, "_get_tokenizer", return_value=mock_tokenizer):
            transport._generate_warmup_prompt(5)

        decode_args = mock_tokenizer.decode.call_args[0][0]
        assert decode_args == [1, 2, 1, 2, 1]


# ============================================================
# _run_warmup Exception Handling
# ============================================================


class TestRunWarmupExceptionHandling:
    """Verify warmup failure is gracefully handled and does not crash."""

    @pytest.mark.asyncio
    async def test_warmup_single_exception_propagates(
        self, transport: ConcreteTestTransport
    ) -> None:
        """_run_warmup propagates exceptions from _warmup_single (no silent swallow).

        The base _run_warmup does not catch exceptions -- callers (_start_engine)
        are responsible for handling warmup failures. This test documents that.
        """
        transport._warmup_iterations = 2
        transport._warmup_single = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("CUDA OOM during warmup")
        )

        with pytest.raises(RuntimeError, match="CUDA OOM during warmup"):
            await transport._run_warmup()

    @pytest.mark.asyncio
    async def test_warmup_single_failure_on_second_iteration(
        self, transport: ConcreteTestTransport
    ) -> None:
        """Exception on iteration 2 still propagates after iteration 1 succeeded."""
        call_count = 0

        async def fail_on_second(
            prompt: str, max_tokens: int, *, streaming: bool
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Engine timeout")

        transport._warmup_iterations = 3
        transport._warmup_single = fail_on_second  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="Engine timeout"):
            await transport._run_warmup()

        assert call_count == 2


# ============================================================
# _pop_warmup_config (String Value Coercion from CLI)
# ============================================================


class TestPopWarmupConfigStringValues:
    """Verify _pop_warmup_config handles CLI string values correctly."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
        ],
    )  # fmt: skip
    def test_preserve_token_ids_bool_coercion(
        self, transport: ConcreteTestTransport, value: Any, expected: bool
    ) -> None:
        """preserve_token_ids handles both bool and string values from CLI."""
        params: dict[str, Any] = {"preserve_token_ids": value}
        transport._pop_warmup_config(params)
        assert transport._preserve_token_ids is expected
        assert "preserve_token_ids" not in params

    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            ("true", True),
            ("1", True),
            ("false", False),
            ("0", False),
        ],
    )  # fmt: skip
    def test_telemetry_bool_coercion(
        self, transport: ConcreteTestTransport, value: Any, expected: bool
    ) -> None:
        """telemetry handles both bool and string values from CLI."""
        params: dict[str, Any] = {"telemetry": value}
        transport._pop_warmup_config(params)
        assert transport._telemetry_enabled is expected
        assert "telemetry" not in params

    def test_telemetry_interval_ms_coerced_to_int(
        self, transport: ConcreteTestTransport
    ) -> None:
        """telemetry_interval_ms string is coerced to int."""
        params: dict[str, Any] = {"telemetry_interval_ms": "250"}
        transport._pop_warmup_config(params)
        assert transport._telemetry_interval_ms == 250
        assert "telemetry_interval_ms" not in params

    def test_all_warmup_and_telemetry_params_popped(
        self, transport: ConcreteTestTransport
    ) -> None:
        """All recognized keys are consumed; unrecognized survive."""
        params: dict[str, Any] = {
            "warmup_iterations": "3",
            "warmup_input_tokens": "256",
            "warmup_output_tokens": "16",
            "preserve_token_ids": "true",
            "telemetry": "true",
            "telemetry_interval_ms": "1000",
            "tensor_parallel_size": "4",
        }
        transport._pop_warmup_config(params)

        assert transport._warmup_iterations == 3
        assert transport._warmup_input_tokens == 256
        assert transport._warmup_output_tokens == 16
        assert transport._preserve_token_ids is True
        assert transport._telemetry_enabled is True
        assert transport._telemetry_interval_ms == 1000
        # Only unrecognized key remains
        assert params == {"tensor_parallel_size": "4"}


# ============================================================
# No Stale _pop_warmup_iterations References
# ============================================================


class TestNoStaleWarmupIterationsMethod:
    """Verify the old _pop_warmup_iterations method is fully removed."""

    def test_base_transport_has_no_pop_warmup_iterations(self) -> None:
        """BaseInEngineTransport should not have a _pop_warmup_iterations method."""
        assert not hasattr(BaseInEngineTransport, "_pop_warmup_iterations")

    def test_concrete_transport_has_no_pop_warmup_iterations(
        self, transport: ConcreteTestTransport
    ) -> None:
        """Concrete instances should not have _pop_warmup_iterations."""
        assert not hasattr(transport, "_pop_warmup_iterations")


# ============================================================
# Token ID Preservation (send_request wiring)
# ============================================================


class TestTokenIdPreservation:
    """Verify output_token_ids flows through send_request when _preserve_token_ids is set."""

    @pytest.mark.asyncio
    async def test_output_token_ids_passed_to_response(self) -> None:
        """When _output_token_ids is set by _generate, the final response carries them."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_result=("text", 5, 3, "stop"),
        )

        original_generate = transport._generate

        async def generate_with_token_ids(**kwargs: Any) -> tuple[str, int, int, str]:
            result = await original_generate(**kwargs)
            transport._output_token_ids = [10, 20, 30]
            return result

        transport._generate = generate_with_token_ids  # type: ignore[method-assign]

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        record = await transport.send_request(request_info, payload)

        final = record.responses[-1]
        assert isinstance(final, InEngineResponse)
        assert final.output_token_ids == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_output_token_ids_none_when_not_set(self) -> None:
        """Without _output_token_ids, the response has None."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_result=("text", 5, 3, "stop"),
        )

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        record = await transport.send_request(request_info, payload)

        final = record.responses[-1]
        assert isinstance(final, InEngineResponse)
        assert final.output_token_ids is None

    @pytest.mark.asyncio
    async def test_output_token_ids_reset_after_request(self) -> None:
        """_output_token_ids is reset to None after send_request completes."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(model_endpoint=model_endpoint)

        original_generate = transport._generate

        async def generate_with_token_ids(**kwargs: Any) -> tuple[str, int, int, str]:
            result = await original_generate(**kwargs)
            transport._output_token_ids = [1, 2, 3]
            return result

        transport._generate = generate_with_token_ids  # type: ignore[method-assign]

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        await transport.send_request(request_info, payload)
        assert transport._output_token_ids is None


# ============================================================
# Speculative Decoding Metadata Plumbing (send_request)
# ============================================================


class TestSpecDecodeMetadataPlumbing:
    """Verify send_request passes decode_iterations/max_draft_len to InEngineResponse."""

    @pytest.mark.asyncio
    async def test_decode_iterations_set_by_generate(self) -> None:
        """When _generate sets _decode_iterations, final InEngineResponse carries it."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_result=("text", 10, 20, "stop"),
        )
        transport._max_draft_len = 5

        original_generate = transport._generate

        async def generate_with_spec_decode(**kwargs: Any) -> tuple[str, int, int, str]:
            result = await original_generate(**kwargs)
            transport._decode_iterations = 7
            return result

        transport._generate = generate_with_spec_decode  # type: ignore[method-assign]

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        record = await transport.send_request(request_info, payload)

        final_response = record.responses[-1]
        assert isinstance(final_response, InEngineResponse)
        assert final_response.decode_iterations == 7
        assert final_response.max_draft_len == 5

    @pytest.mark.asyncio
    async def test_decode_iterations_none_when_not_set(self) -> None:
        """When _decode_iterations is never set, InEngineResponse fields are None."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_result=("text", 10, 20, "stop"),
        )

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        record = await transport.send_request(request_info, payload)

        final_response = record.responses[-1]
        assert isinstance(final_response, InEngineResponse)
        assert final_response.decode_iterations is None
        assert final_response.max_draft_len is None

    @pytest.mark.asyncio
    async def test_decode_iterations_reset_after_request(self) -> None:
        """_decode_iterations is reset to None after send_request completes."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(model_endpoint=model_endpoint)

        original_generate = transport._generate

        async def generate_with_spec_decode(**kwargs: Any) -> tuple[str, int, int, str]:
            result = await original_generate(**kwargs)
            transport._decode_iterations = 3
            return result

        transport._generate = generate_with_spec_decode  # type: ignore[method-assign]

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        await transport.send_request(request_info, payload)
        assert transport._decode_iterations is None

    @pytest.mark.asyncio
    async def test_max_draft_len_zero_becomes_none(self) -> None:
        """When _max_draft_len is 0 (not configured), InEngineResponse.max_draft_len is None."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(model_endpoint=model_endpoint)
        transport._max_draft_len = 0

        original_generate = transport._generate

        async def generate_with_spec_decode(**kwargs: Any) -> tuple[str, int, int, str]:
            result = await original_generate(**kwargs)
            transport._decode_iterations = 5
            return result

        transport._generate = generate_with_spec_decode  # type: ignore[method-assign]

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        record = await transport.send_request(request_info, payload)
        final_response = record.responses[-1]
        assert final_response.decode_iterations == 5
        assert final_response.max_draft_len is None

    @pytest.mark.asyncio
    async def test_streaming_spec_decode_only_on_final_response(self) -> None:
        """Streaming path: spec decode metadata only on final response, not TTFT marker."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_result=("Full text", 10, 20, "stop"),
        )
        transport._max_draft_len = 3

        original_generate = transport._generate

        async def generate_with_ttft_and_spec(
            **kwargs: Any,
        ) -> tuple[str, int, int, str]:
            result = await original_generate(**kwargs)
            transport._first_token_perf_ns = 500_000
            transport._decode_iterations = 4
            return result

        transport._generate = generate_with_ttft_and_spec  # type: ignore[method-assign]

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        record = await transport.send_request(request_info, payload)

        assert len(record.responses) == 2
        # TTFT marker has no spec decode metadata
        ttft_response = record.responses[0]
        assert ttft_response.decode_iterations is None
        assert ttft_response.max_draft_len is None
        # Final response has spec decode metadata
        final_response = record.responses[1]
        assert final_response.decode_iterations == 4
        assert final_response.max_draft_len == 3


# ============================================================
# Concurrency Semaphore
# ============================================================


class TestConcurrencySemaphore:
    """Verify concurrency semaphore limits concurrent _generate() calls."""

    def test_default_semaphore_is_none(self, transport: ConcreteTestTransport) -> None:
        """No concurrency limit by default."""
        assert transport._concurrency_semaphore is None

    @pytest.mark.parametrize(
        "value,expected_value",
        [
            ("1", 1),
            ("4", 4),
            ("64", 64),
            (10, 10),
        ],
    )  # fmt: skip
    def test_pop_warmup_config_sets_semaphore(
        self,
        transport: ConcreteTestTransport,
        value: Any,
        expected_value: int,
    ) -> None:
        """concurrency engine param creates a Semaphore with correct value."""
        params: dict[str, Any] = {"concurrency": value}
        transport._pop_warmup_config(params)
        assert isinstance(transport._concurrency_semaphore, asyncio.Semaphore)
        assert transport._concurrency_semaphore._value == expected_value
        assert "concurrency" not in params

    @pytest.mark.parametrize("value", ["0", 0, "-1"])
    def test_pop_warmup_config_zero_or_negative_no_semaphore(
        self,
        transport: ConcreteTestTransport,
        value: Any,
    ) -> None:
        """concurrency <= 0 leaves semaphore as None (no limit)."""
        params: dict[str, Any] = {"concurrency": value}
        transport._pop_warmup_config(params)
        assert transport._concurrency_semaphore is None
        assert "concurrency" not in params

    def test_pop_warmup_config_no_concurrency_key(
        self, transport: ConcreteTestTransport
    ) -> None:
        """Missing concurrency key leaves semaphore as None."""
        params: dict[str, Any] = {"warmup_iterations": "1"}
        transport._pop_warmup_config(params)
        assert transport._concurrency_semaphore is None

    @pytest.mark.asyncio
    async def test_send_request_works_without_semaphore(
        self,
    ) -> None:
        """send_request succeeds when no semaphore is configured."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_result=("ok", 5, 3, "stop"),
        )
        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        record = await transport.send_request(request_info, payload)
        assert record.error is None
        assert record.responses[-1].text == "ok"

    @pytest.mark.asyncio
    async def test_send_request_works_with_semaphore(
        self,
    ) -> None:
        """send_request succeeds when semaphore is configured."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_result=("ok", 5, 3, "stop"),
        )
        transport._concurrency_semaphore = asyncio.Semaphore(2)

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        record = await transport.send_request(request_info, payload)
        assert record.error is None
        assert record.responses[-1].text == "ok"

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_generate_calls(self) -> None:
        """Semaphore(1) serializes _generate calls so at most 1 runs at a time."""
        model_endpoint = _make_model_endpoint()
        max_concurrent = 0
        current_concurrent = 0

        class TrackingTransport(ConcreteTestTransport):
            async def _generate(
                self,
                *,
                messages: list[dict[str, Any]],
                sampling_params: Any,
                request_id: str,
                first_token_callback: FirstTokenCallback | None = None,
                input_ids: list[int] | None = None,
            ) -> tuple[str, int, int, str]:
                nonlocal max_concurrent, current_concurrent
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.01)
                current_concurrent -= 1
                return ("result", 5, 3, "stop")

        transport = TrackingTransport(model_endpoint=model_endpoint)
        transport._concurrency_semaphore = asyncio.Semaphore(1)

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        # Launch 5 concurrent requests
        tasks = [
            asyncio.create_task(transport.send_request(request_info, payload))
            for _ in range(5)
        ]
        records = await asyncio.gather(*tasks)

        assert max_concurrent == 1
        assert all(r.error is None for r in records)

    @pytest.mark.asyncio
    async def test_semaphore_allows_configured_concurrency(self) -> None:
        """Semaphore(3) allows up to 3 concurrent _generate calls."""
        model_endpoint = _make_model_endpoint()
        max_concurrent = 0
        current_concurrent = 0

        class TrackingTransport(ConcreteTestTransport):
            async def _generate(
                self,
                *,
                messages: list[dict[str, Any]],
                sampling_params: Any,
                request_id: str,
                first_token_callback: FirstTokenCallback | None = None,
                input_ids: list[int] | None = None,
            ) -> tuple[str, int, int, str]:
                nonlocal max_concurrent, current_concurrent
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.01)
                current_concurrent -= 1
                return ("result", 5, 3, "stop")

        transport = TrackingTransport(model_endpoint=model_endpoint)
        transport._concurrency_semaphore = asyncio.Semaphore(3)

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        # Launch 6 concurrent requests
        tasks = [
            asyncio.create_task(transport.send_request(request_info, payload))
            for _ in range(6)
        ]
        records = await asyncio.gather(*tasks)

        assert max_concurrent <= 3
        assert all(r.error is None for r in records)

    @pytest.mark.asyncio
    async def test_semaphore_released_on_generate_error(self) -> None:
        """Semaphore is released even when _generate raises an exception."""
        model_endpoint = _make_model_endpoint()
        transport = ConcreteTestTransport(
            model_endpoint=model_endpoint,
            generate_error=RuntimeError("Engine OOM"),
        )
        transport._concurrency_semaphore = asyncio.Semaphore(1)

        request_info = _make_request_info(model_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        # First request fails
        record = await transport.send_request(request_info, payload)
        assert record.error is not None

        # Semaphore should be released, so a second request can proceed
        transport._generate_error = None
        transport._generate_result = ("ok", 5, 3, "stop")
        record2 = await transport.send_request(request_info, payload)
        assert record2.error is None
