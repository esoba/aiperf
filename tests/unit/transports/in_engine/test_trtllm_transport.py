# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TRTLLMTransport with mocked TensorRT-LLM engine.

Focuses on:
- Transport metadata (type and URL schemes)
- Import validation when tensorrt_llm is not installed
- Engine kwargs mapping (tensor_parallel_size, backend, max_seq_len, etc.)
- Generation output extraction from TRT-LLM RequestOutput
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.plugin.enums import EndpointType

# ============================================================
# Mock TRT-LLM Module & Types
# ============================================================


class MockTRTCompletionOutput:
    """Mock TRT-LLM CompletionOutput."""

    def __init__(
        self,
        text: str = "TRT response",
        token_ids: list[int] | None = None,
        finish_reason: str | None = "stop",
    ) -> None:
        self.text = text
        self.token_ids = token_ids or list(range(10))
        self.finish_reason = finish_reason


class MockTRTRequestOutput:
    """Mock TRT-LLM RequestOutput."""

    def __init__(
        self,
        prompt_token_ids: list[int] | None = None,
        outputs: list[MockTRTCompletionOutput] | None = None,
    ) -> None:
        self.prompt_token_ids = prompt_token_ids or list(range(8))
        self.outputs = outputs or [MockTRTCompletionOutput()]


class MockTRTSamplingParams:
    """Mock TRT-LLM SamplingParams that captures constructor kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


def _build_mock_trtllm_module() -> types.ModuleType:
    """Build a mock tensorrt_llm module with needed classes."""
    mock_trtllm = types.ModuleType("tensorrt_llm")
    mock_trtllm.SamplingParams = MockTRTSamplingParams  # type: ignore[attr-defined]
    mock_trtllm.LLM = MagicMock  # type: ignore[attr-defined]
    return mock_trtllm


# ============================================================
# Fixtures
# ============================================================


def _make_trtllm_endpoint(
    base_url: str = "trtllm://meta-llama/Llama-3.1-8B",
    engine_params: list[tuple[str, Any]] | None = None,
    extra: list[tuple[str, Any]] | None = None,
) -> ModelEndpointInfo:
    """Create a ModelEndpointInfo configured for TRT-LLM transport."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="meta-llama/Llama-3.1-8B")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_urls=[base_url],
            engine_params=engine_params or [],
            extra=extra or [],
        ),
    )


@pytest.fixture
def mock_trtllm_module() -> types.ModuleType:
    """Provide a mock tensorrt_llm module for test isolation."""
    return _build_mock_trtllm_module()


@pytest.fixture
def trtllm_transport(mock_trtllm_module: types.ModuleType) -> Any:
    """Create a TRTLLMTransport with mocked tensorrt_llm import."""
    with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm_module}):
        from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

        transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
        yield transport


# ============================================================
# Metadata
# ============================================================


class TestTRTLLMTransportMetadata:
    """Verify transport metadata for plugin discovery."""

    def test_metadata_transport_type(self) -> None:
        with patch.dict(sys.modules, {"tensorrt_llm": _build_mock_trtllm_module()}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            assert TRTLLMTransport.metadata().transport_type == "trtllm"

    def test_metadata_url_schemes(self) -> None:
        with patch.dict(sys.modules, {"tensorrt_llm": _build_mock_trtllm_module()}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            assert TRTLLMTransport.metadata().url_schemes == ["trtllm"]


# ============================================================
# Init Engine (Import Validation)
# ============================================================


class TestTRTLLMInitEngine:
    """Verify engine initialization and import validation."""

    @pytest.mark.asyncio
    async def test_init_engine_missing_trtllm_raises(self) -> None:
        with patch.dict(sys.modules, {"tensorrt_llm": None}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            with pytest.raises(ImportError, match="TensorRT-LLM is required"):
                await transport._init_engine()


# ============================================================
# Engine Kwargs Building
# ============================================================


class TestTRTLLMBuildEngineKwargs:
    """Verify engine_params to LLM constructor kwargs mapping."""

    def test_build_engine_kwargs_backend(self) -> None:
        with patch.dict(sys.modules, {"tensorrt_llm": _build_mock_trtllm_module()}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(engine_params=[("backend", "pytorch")])
            transport = TRTLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["backend"] == "pytorch"

    def test_build_engine_kwargs_max_seq_len(self) -> None:
        with patch.dict(sys.modules, {"tensorrt_llm": _build_mock_trtllm_module()}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(engine_params=[("max_seq_len", "4096")])
            transport = TRTLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["max_seq_len"] == 4096

    def test_build_engine_kwargs_tensor_parallel(self) -> None:
        with patch.dict(sys.modules, {"tensorrt_llm": _build_mock_trtllm_module()}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(
                engine_params=[("tensor_parallel_size", "2")]
            )
            transport = TRTLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["tensor_parallel_size"] == 2

    def test_build_engine_kwargs_max_batch_size(self) -> None:
        with patch.dict(sys.modules, {"tensorrt_llm": _build_mock_trtllm_module()}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(engine_params=[("max_batch_size", "64")])
            transport = TRTLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["max_batch_size"] == 64

    def test_build_engine_kwargs_passthrough(self) -> None:
        with patch.dict(sys.modules, {"tensorrt_llm": _build_mock_trtllm_module()}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(
                engine_params=[("custom_trt_option", "value")]
            )
            transport = TRTLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["custom_trt_option"] == "value"

    def test_build_engine_kwargs_empty(self) -> None:
        with patch.dict(sys.modules, {"tensorrt_llm": _build_mock_trtllm_module()}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(engine_params=[])
            transport = TRTLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs == {}

    def test_build_engine_kwargs_multiple_params(self) -> None:
        with patch.dict(sys.modules, {"tensorrt_llm": _build_mock_trtllm_module()}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(
                engine_params=[
                    ("tensor_parallel_size", "4"),
                    ("dtype", "float16"),
                    ("max_seq_len", "8192"),
                    ("backend", "pytorch"),
                ]
            )
            transport = TRTLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["tensor_parallel_size"] == 4
            assert kwargs["dtype"] == "float16"
            assert kwargs["max_seq_len"] == 8192
            assert kwargs["backend"] == "pytorch"


# ============================================================
# Generate (Output Extraction)
# ============================================================


class TestTRTLLMGenerate:
    """Verify generation output extraction from mock TRT-LLM engine."""

    @pytest.mark.asyncio
    async def test_generate_returns_text_and_counts(self) -> None:
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            transport = TRTLLMTransport(model_endpoint=endpoint)

            mock_output = MockTRTRequestOutput(
                prompt_token_ids=list(range(12)),
                outputs=[
                    MockTRTCompletionOutput(
                        text="TRT-LLM generated text",
                        token_ids=list(range(15)),
                        finish_reason="stop",
                    )
                ],
            )
            mock_engine = MagicMock()
            mock_engine.generate.return_value = [mock_output]
            # No tokenizer attr -> fallback prompt format
            del mock_engine.tokenizer
            transport._engine = mock_engine

            (
                text,
                input_tokens,
                output_tokens,
                finish_reason,
            ) = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={"temperature": 0.5},
                request_id="req-trt-001",
            )

            assert text == "TRT-LLM generated text"
            assert input_tokens == 12
            assert output_tokens == 15
            assert finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_handles_missing_prompt_token_ids(self) -> None:
        """Robustness when prompt_token_ids is missing (older TRT-LLM versions)."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            transport = TRTLLMTransport(model_endpoint=endpoint)

            # Create output WITHOUT prompt_token_ids attribute
            mock_completion = MockTRTCompletionOutput(
                text="Result", token_ids=[1, 2, 3]
            )
            mock_output = MagicMock()
            mock_output.outputs = [mock_completion]
            # Simulate missing prompt_token_ids
            del mock_output.prompt_token_ids

            mock_engine = MagicMock()
            mock_engine.generate.return_value = [mock_output]
            del mock_engine.tokenizer
            transport._engine = mock_engine

            (
                text,
                input_tokens,
                output_tokens,
                finish_reason,
            ) = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-trt-002",
            )

            assert text == "Result"
            assert input_tokens == 0  # Defaults to 0 when missing
            assert output_tokens == 3

    @pytest.mark.asyncio
    async def test_generate_none_finish_reason_defaults_to_stop(self) -> None:
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            transport = TRTLLMTransport(model_endpoint=endpoint)

            mock_output = MockTRTRequestOutput(
                outputs=[MockTRTCompletionOutput(finish_reason=None)],
            )
            mock_engine = MagicMock()
            mock_engine.generate.return_value = [mock_output]
            del mock_engine.tokenizer
            transport._engine = mock_engine

            _, _, _, finish_reason = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-trt-003",
            )
            assert finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_enum_finish_reason_normalized_to_string(self) -> None:
        """TRT-LLM may return enum objects for finish_reason."""

        class FinishReasonEnum:
            def __str__(self) -> str:
                return "length"

        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            transport = TRTLLMTransport(model_endpoint=endpoint)

            mock_completion = MockTRTCompletionOutput(text="ok", token_ids=[1])
            mock_completion.finish_reason = FinishReasonEnum()  # type: ignore[assignment]
            mock_output = MockTRTRequestOutput(outputs=[mock_completion])
            mock_engine = MagicMock()
            mock_engine.generate.return_value = [mock_output]
            del mock_engine.tokenizer
            transport._engine = mock_engine

            _, _, _, finish_reason = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-trt-enum",
            )
            assert finish_reason == "length"
            assert isinstance(finish_reason, str)


# ============================================================
# Warmup Iterations
# ============================================================


class TestTRTLLMWarmupIterations:
    """Verify warmup_iterations is popped from engine params."""

    def test_warmup_iterations_popped_from_kwargs(self) -> None:
        with patch.dict(sys.modules, {"tensorrt_llm": _build_mock_trtllm_module()}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(
                engine_params=[
                    ("warmup_iterations", "3"),
                    ("tensor_parallel_size", "2"),
                ]
            )
            transport = TRTLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()

            assert "warmup_iterations" not in kwargs
            assert transport._warmup_iterations == 3
            assert kwargs["tensor_parallel_size"] == 2
