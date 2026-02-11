# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseInEngineTransport shared functionality.

Focuses on:
- Model path extraction from URL schemes
- Error record construction
- Full send_request flow with mocked _generate
- InEngineResponse construction
- Message-to-prompt conversion with fallback
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

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

    async def _generate(
        self,
        *,
        messages: list[dict[str, Any]],
        sampling_params: Any,
        request_id: str,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> tuple[str, int, int, str]:
        if self._generate_error:
            raise self._generate_error
        return self._generate_result


# ============================================================
# Fixtures
# ============================================================


def _make_model_endpoint(
    base_url: str = "test-engine://meta-llama/Llama-3.1-8B",
) -> ModelEndpointInfo:
    """Create a ModelEndpointInfo for in-engine transport testing."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="meta-llama/Llama-3.1-8B")],
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
    """Verify model path extraction from various URL schemes."""

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
    def test_extract_model_path(self, url: str, expected: str) -> None:
        endpoint = _make_model_endpoint(base_url=url)
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
