# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SGLangTransport with mocked SGLang engine.

Focuses on:
- Transport metadata (type and URL schemes)
- Import validation when SGLang is not installed
- Engine kwargs mapping (tensor_parallel_size -> tp, mem_fraction_static, etc.)
- Generation output extraction from SGLang dict response
- finish_reason dict handling ({"type": "stop"} -> "stop")
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
# Mock SGLang Module
# ============================================================


def _build_mock_sglang_module() -> types.ModuleType:
    """Build a mock sglang module with Engine class."""
    mock_sglang = types.ModuleType("sglang")
    mock_sglang.Engine = MagicMock  # type: ignore[attr-defined]
    return mock_sglang


# ============================================================
# Fixtures
# ============================================================


def _make_sglang_endpoint(
    base_url: str = "sglang://meta-llama/Llama-3.1-8B",
    engine_params: list[tuple[str, Any]] | None = None,
    extra: list[tuple[str, Any]] | None = None,
) -> ModelEndpointInfo:
    """Create a ModelEndpointInfo configured for SGLang transport."""
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
def mock_sglang_module() -> types.ModuleType:
    """Provide a mock sglang module for test isolation."""
    return _build_mock_sglang_module()


@pytest.fixture
def sglang_transport(mock_sglang_module: types.ModuleType) -> Any:
    """Create an SGLangTransport with mocked sglang import."""
    with patch.dict(sys.modules, {"sglang": mock_sglang_module}):
        from aiperf.transports.in_engine.sglang_transport import SGLangTransport

        transport = SGLangTransport(model_endpoint=_make_sglang_endpoint())
        yield transport


# ============================================================
# Metadata
# ============================================================


class TestSGLangTransportMetadata:
    """Verify transport metadata for plugin discovery."""

    def test_metadata_transport_type(self) -> None:
        with patch.dict(sys.modules, {"sglang": _build_mock_sglang_module()}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            assert SGLangTransport.metadata().transport_type == "sglang"

    def test_metadata_url_schemes(self) -> None:
        with patch.dict(sys.modules, {"sglang": _build_mock_sglang_module()}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            assert SGLangTransport.metadata().url_schemes == ["sglang"]


# ============================================================
# Init Engine (Import Validation)
# ============================================================


class TestSGLangInitEngine:
    """Verify engine initialization and import validation."""

    @pytest.mark.asyncio
    async def test_init_engine_missing_sglang_raises(self) -> None:
        with patch.dict(sys.modules, {"sglang": None}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            transport = SGLangTransport(model_endpoint=_make_sglang_endpoint())
            with pytest.raises(ImportError, match="SGLang is required"):
                await transport._init_engine()


# ============================================================
# Engine Kwargs Building
# ============================================================


class TestSGLangBuildEngineKwargs:
    """Verify engine_params to Engine constructor kwargs mapping."""

    def test_build_engine_kwargs_tp(self) -> None:
        with patch.dict(sys.modules, {"sglang": _build_mock_sglang_module()}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint(
                engine_params=[("tensor_parallel_size", "4")]
            )
            transport = SGLangTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["tp"] == 4

    def test_build_engine_kwargs_tp_native_name(self) -> None:
        """SGLang accepts 'tp' directly as well."""
        with patch.dict(sys.modules, {"sglang": _build_mock_sglang_module()}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint(engine_params=[("tp", "2")])
            transport = SGLangTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["tp"] == 2

    def test_build_engine_kwargs_mem_fraction(self) -> None:
        with patch.dict(sys.modules, {"sglang": _build_mock_sglang_module()}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint(
                engine_params=[("mem_fraction_static", "0.85")]
            )
            transport = SGLangTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["mem_fraction_static"] == 0.85

    def test_build_engine_kwargs_unknown_params_passed_through(self) -> None:
        with patch.dict(sys.modules, {"sglang": _build_mock_sglang_module()}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint(engine_params=[("custom_setting", "abc")])
            transport = SGLangTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["custom_setting"] == "abc"

    def test_build_engine_kwargs_empty(self) -> None:
        with patch.dict(sys.modules, {"sglang": _build_mock_sglang_module()}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint(engine_params=[])
            transport = SGLangTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs == {}


# ============================================================
# Generate (Output Extraction)
# ============================================================


class TestSGLangGenerate:
    """Verify generation output extraction from SGLang dict response."""

    @pytest.mark.asyncio
    async def test_generate_extracts_text_from_dict(self) -> None:
        mock_sglang = _build_mock_sglang_module()
        with patch.dict(sys.modules, {"sglang": mock_sglang}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint()
            transport = SGLangTransport(model_endpoint=endpoint)

            # Mock engine with async_generate returning a dict
            mock_engine = MagicMock()
            mock_engine.async_generate = AsyncMock(
                return_value={
                    "text": "Generated response",
                    "meta_info": {
                        "prompt_tokens": 12,
                        "completion_tokens": 7,
                        "finish_reason": {"type": "stop"},
                    },
                }
            )
            # No tokenizer_manager so fallback prompt formatting is used
            del mock_engine.tokenizer_manager
            transport._engine = mock_engine

            (
                text,
                input_tokens,
                output_tokens,
                finish_reason,
            ) = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={"max_new_tokens": 256, "temperature": 0.5},
                request_id="req-sgl-001",
            )

            assert text == "Generated response"
            assert input_tokens == 12
            assert output_tokens == 7
            assert finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_handles_dict_finish_reason(self) -> None:
        """SGLang finish_reason can be a dict like {"type": "length"}."""
        mock_sglang = _build_mock_sglang_module()
        with patch.dict(sys.modules, {"sglang": mock_sglang}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint()
            transport = SGLangTransport(model_endpoint=endpoint)

            mock_engine = MagicMock()
            mock_engine.async_generate = AsyncMock(
                return_value={
                    "text": "Truncated response",
                    "meta_info": {
                        "prompt_tokens": 10,
                        "completion_tokens": 50,
                        "finish_reason": {"type": "length"},
                    },
                }
            )
            del mock_engine.tokenizer_manager
            transport._engine = mock_engine

            _, _, _, finish_reason = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-sgl-002",
            )
            assert finish_reason == "length"

    @pytest.mark.asyncio
    async def test_generate_handles_string_finish_reason(self) -> None:
        """SGLang finish_reason can also be a plain string."""
        mock_sglang = _build_mock_sglang_module()
        with patch.dict(sys.modules, {"sglang": mock_sglang}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint()
            transport = SGLangTransport(model_endpoint=endpoint)

            mock_engine = MagicMock()
            mock_engine.async_generate = AsyncMock(
                return_value={
                    "text": "Done",
                    "meta_info": {
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "finish_reason": "stop",
                    },
                }
            )
            del mock_engine.tokenizer_manager
            transport._engine = mock_engine

            _, _, _, finish_reason = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-sgl-003",
            )
            assert finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_missing_meta_info_defaults(self) -> None:
        """Missing meta_info defaults to zero tokens and 'stop' finish reason."""
        mock_sglang = _build_mock_sglang_module()
        with patch.dict(sys.modules, {"sglang": mock_sglang}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint()
            transport = SGLangTransport(model_endpoint=endpoint)

            mock_engine = MagicMock()
            mock_engine.async_generate = AsyncMock(
                return_value={
                    "text": "Output",
                }
            )
            del mock_engine.tokenizer_manager
            transport._engine = mock_engine

            (
                text,
                input_tokens,
                output_tokens,
                finish_reason,
            ) = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-sgl-004",
            )
            assert text == "Output"
            assert input_tokens == 0
            assert output_tokens == 0
            assert finish_reason == "stop"


# ============================================================
# Messages to Prompt (Fallback)
# ============================================================


# ============================================================
# Warmup Iterations
# ============================================================


class TestSGLangWarmupIterations:
    """Verify warmup_iterations is popped from engine params."""

    def test_warmup_iterations_popped_from_kwargs(self) -> None:
        with patch.dict(sys.modules, {"sglang": _build_mock_sglang_module()}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint(
                engine_params=[("warmup_iterations", "3"), ("tp", "2")]
            )
            transport = SGLangTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()

            assert "warmup_iterations" not in kwargs
            assert transport._warmup_iterations == 3
            assert kwargs["tp"] == 2


# ============================================================
# _warmup_single (SGLang Engine)
# ============================================================


class TestSGLangWarmupSingle:
    """Verify _warmup_single calls SGLang engine in correct streaming mode."""

    @pytest.mark.asyncio
    async def test_warmup_non_streaming_calls_async_generate(self) -> None:
        """streaming=False calls async_generate without stream=True."""
        mock_sglang = _build_mock_sglang_module()
        with patch.dict(sys.modules, {"sglang": mock_sglang}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint()
            transport = SGLangTransport(model_endpoint=endpoint)

            mock_engine = MagicMock()
            mock_engine.async_generate = AsyncMock(return_value={"text": "warm"})
            transport._engine = mock_engine

            await transport._warmup_single("test prompt", 4, streaming=False)

            mock_engine.async_generate.assert_awaited_once()
            call_kwargs = mock_engine.async_generate.call_args[1]
            assert "stream" not in call_kwargs
            assert call_kwargs["sampling_params"]["max_new_tokens"] == 4

    @pytest.mark.asyncio
    async def test_warmup_streaming_calls_async_generate_with_stream(self) -> None:
        """streaming=True calls async_generate with stream=True and iterates."""
        mock_sglang = _build_mock_sglang_module()
        with patch.dict(sys.modules, {"sglang": mock_sglang}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint()
            transport = SGLangTransport(model_endpoint=endpoint)

            # async_generate with stream=True returns a coroutine that yields an async gen
            async def mock_async_gen() -> Any:
                yield {"text": "warm", "meta_info": {}}

            mock_engine = MagicMock()
            mock_engine.async_generate = AsyncMock(return_value=mock_async_gen())
            transport._engine = mock_engine

            await transport._warmup_single("test prompt", 4, streaming=True)

            mock_engine.async_generate.assert_awaited_once()
            call_kwargs = mock_engine.async_generate.call_args[1]
            assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_warmup_single_passes_max_new_tokens(self) -> None:
        """_warmup_single passes max_new_tokens in sampling_params."""
        mock_sglang = _build_mock_sglang_module()
        with patch.dict(sys.modules, {"sglang": mock_sglang}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint()
            transport = SGLangTransport(model_endpoint=endpoint)

            mock_engine = MagicMock()
            mock_engine.async_generate = AsyncMock(return_value={"text": "warm"})
            transport._engine = mock_engine

            await transport._warmup_single("test prompt", 32, streaming=False)

            call_kwargs = mock_engine.async_generate.call_args[1]
            assert call_kwargs["sampling_params"] == {"max_new_tokens": 32}

    @pytest.mark.asyncio
    async def test_warmup_single_passes_prompt_text(self) -> None:
        """_warmup_single passes the prompt string to async_generate."""
        mock_sglang = _build_mock_sglang_module()
        with patch.dict(sys.modules, {"sglang": mock_sglang}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint()
            transport = SGLangTransport(model_endpoint=endpoint)

            mock_engine = MagicMock()
            mock_engine.async_generate = AsyncMock(return_value={"text": "warm"})
            transport._engine = mock_engine

            await transport._warmup_single("my warmup prompt", 4, streaming=False)

            call_kwargs = mock_engine.async_generate.call_args[1]
            assert call_kwargs["prompt"] == "my warmup prompt"


# ============================================================
# finish_reason Normalization
# ============================================================


class TestSGLangFinishReasonNormalization:
    """Verify finish_reason is normalized to string."""

    @pytest.mark.asyncio
    async def test_empty_string_finish_reason_defaults_to_stop(self) -> None:
        mock_sglang = _build_mock_sglang_module()
        with patch.dict(sys.modules, {"sglang": mock_sglang}):
            from aiperf.transports.in_engine.sglang_transport import SGLangTransport

            endpoint = _make_sglang_endpoint()
            transport = SGLangTransport(model_endpoint=endpoint)

            mock_engine = MagicMock()
            mock_engine.async_generate = AsyncMock(
                return_value={
                    "text": "Done",
                    "meta_info": {
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "finish_reason": "",
                    },
                }
            )
            del mock_engine.tokenizer_manager
            transport._engine = mock_engine

            _, _, _, finish_reason = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-sgl-norm",
            )
            assert finish_reason == "stop"


# ============================================================
# Messages to Prompt (Fallback)
# ============================================================


class TestSGLangMessagesToPrompt:
    """Verify message-to-prompt conversion fallback when no tokenizer."""

    def test_messages_to_prompt_fallback_format(self, sglang_transport: Any) -> None:
        # Ensure no tokenizer_manager attribute triggers the fallback path
        sglang_transport._engine = MagicMock(spec=[])  # empty spec -> no attrs

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        prompt = sglang_transport._messages_to_prompt(messages)

        assert "<|system|>" in prompt
        assert "You are helpful" in prompt
        assert "<|user|>" in prompt
        assert "Hello" in prompt
        assert "<|assistant|>" in prompt
