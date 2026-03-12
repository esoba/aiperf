# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for VLLMTransport with mocked vLLM async engine.

Focuses on:
- Transport metadata (type and URL schemes)
- Import validation when vLLM is not installed
- Engine kwargs mapping from endpoint extra config
- Generation output extraction from async vLLM RequestOutput
"""

from __future__ import annotations

import enum
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
# Mock vLLM Module & Types
# ============================================================


class MockCompletionOutput:
    """Mock vLLM CompletionOutput."""

    def __init__(
        self,
        text: str = "Mock response",
        token_ids: list[int] | None = None,
        finish_reason: str | None = "stop",
    ) -> None:
        self.text = text
        self.token_ids = token_ids or list(range(20))
        self.finish_reason = finish_reason


class MockRequestOutput:
    """Mock vLLM RequestOutput."""

    def __init__(
        self,
        prompt_token_ids: list[int] | None = None,
        outputs: list[MockCompletionOutput] | None = None,
        finished: bool = True,
    ) -> None:
        self.prompt_token_ids = prompt_token_ids or list(range(10))
        self.outputs = outputs or [MockCompletionOutput()]
        self.finished = finished


class MockVLLMSamplingParams:
    """Mock vLLM SamplingParams that captures constructor kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockRequestOutputKind(enum.IntEnum):
    """Mock vLLM RequestOutputKind enum."""

    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


class MockAsyncEngine:
    """Mock AsyncLLMEngine with async generate() support."""

    def __init__(self, outputs: list[MockRequestOutput] | None = None) -> None:
        self._outputs = outputs or []
        self._tokenizer = MagicMock()

    async def generate(
        self,
        prompt: str,
        sampling_params: Any,
        request_id: str,
    ) -> Any:
        for output in self._outputs:
            yield output

    def get_tokenizer(self) -> Any:
        return self._tokenizer

    def shutdown(self) -> None:
        pass


def _build_mock_vllm_module() -> types.ModuleType:
    """Build a mock vllm module with the needed classes."""
    mock_vllm = types.ModuleType("vllm")
    mock_vllm.SamplingParams = MockVLLMSamplingParams  # type: ignore[attr-defined]
    mock_vllm.LLM = MagicMock  # type: ignore[attr-defined]
    mock_vllm.AsyncLLMEngine = MagicMock  # type: ignore[attr-defined]
    mock_vllm.AsyncEngineArgs = MagicMock  # type: ignore[attr-defined]

    # Build vllm.sampling_params submodule with RequestOutputKind
    sampling_params_mod = types.ModuleType("vllm.sampling_params")
    sampling_params_mod.RequestOutputKind = MockRequestOutputKind  # type: ignore[attr-defined]
    mock_vllm.sampling_params = sampling_params_mod  # type: ignore[attr-defined]

    return mock_vllm


# ============================================================
# Fixtures
# ============================================================


def _make_vllm_endpoint(
    base_url: str = "vllm://meta-llama/Llama-3.1-8B",
    engine_params: list[tuple[str, Any]] | None = None,
    extra: list[tuple[str, Any]] | None = None,
) -> ModelEndpointInfo:
    """Create a ModelEndpointInfo configured for vLLM transport."""
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
def mock_vllm_module() -> types.ModuleType:
    """Provide a mock vllm module for test isolation."""
    return _build_mock_vllm_module()


@pytest.fixture
def vllm_transport(mock_vllm_module: types.ModuleType) -> Any:
    """Create a VLLMTransport with mocked vllm import.

    Uses sys.modules patching so that 'import vllm' resolves to the mock.
    """
    sampling_params_mod = mock_vllm_module.sampling_params  # type: ignore[attr-defined]
    with patch.dict(
        sys.modules,
        {"vllm": mock_vllm_module, "vllm.sampling_params": sampling_params_mod},
    ):
        from aiperf.transports.in_engine.vllm_transport import VLLMTransport

        transport = VLLMTransport(model_endpoint=_make_vllm_endpoint())
        yield transport


def _patch_vllm_modules():
    """Context manager that patches both vllm and vllm.sampling_params."""
    mock = _build_mock_vllm_module()
    return patch.dict(
        sys.modules,
        {
            "vllm": mock,
            "vllm.sampling_params": mock.sampling_params,  # type: ignore[attr-defined]
        },
    )


# ============================================================
# Metadata
# ============================================================


class TestVLLMTransportMetadata:
    """Verify transport metadata for plugin discovery."""

    def test_metadata_transport_type(self) -> None:
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            metadata = VLLMTransport.metadata()
            assert metadata.transport_type == "vllm"

    def test_metadata_url_schemes(self) -> None:
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            metadata = VLLMTransport.metadata()
            assert metadata.url_schemes == ["vllm"]


# ============================================================
# Init Engine (Import Validation)
# ============================================================


class TestVLLMInitEngine:
    """Verify engine initialization and import validation."""

    @pytest.mark.asyncio
    async def test_init_engine_missing_vllm_raises(self) -> None:
        """Clear ImportError when vllm is not installed."""
        with patch.dict(sys.modules, {"vllm": None}):
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            transport = VLLMTransport(model_endpoint=_make_vllm_endpoint())
            with pytest.raises(ImportError, match="vLLM is required"):
                await transport._init_engine()


# ============================================================
# Engine Kwargs Building
# ============================================================


class TestVLLMBuildEngineKwargs:
    """Verify engine_params to LLM constructor kwargs mapping."""

    def test_build_engine_kwargs_tensor_parallel(self) -> None:
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(
                engine_params=[("tensor_parallel_size", "4")]
            )
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["tensor_parallel_size"] == 4

    def test_build_engine_kwargs_gpu_memory(self) -> None:
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(
                engine_params=[("gpu_memory_utilization", "0.9")]
            )
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["gpu_memory_utilization"] == 0.9

    def test_build_engine_kwargs_passthrough(self) -> None:
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(
                engine_params=[
                    ("enforce_eager", True),
                    ("trust_remote_code", True),
                ]
            )
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["enforce_eager"] is True
            assert kwargs["trust_remote_code"] is True

    def test_build_engine_kwargs_unknown_params_passed_through(self) -> None:
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(engine_params=[("custom_flag", "value")])
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["custom_flag"] == "value"

    def test_build_engine_kwargs_empty(self) -> None:
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(engine_params=[])
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs == {}

    def test_build_engine_kwargs_string_params(self) -> None:
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(
                engine_params=[
                    ("dtype", "float16"),
                    ("quantization", "awq"),
                ]
            )
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["dtype"] == "float16"
            assert kwargs["quantization"] == "awq"


# ============================================================
# Speculative Config Building
# ============================================================


class TestVLLMBuildSpeculativeConfig:
    """Verify speculative decoding params are extracted into speculative_config."""

    def test_eagle3_config(self) -> None:
        """EAGLE3: method + model + num_speculative_tokens."""
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(
                engine_params=[
                    ("speculative_method", "eagle3"),
                    ("speculative_model", "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"),
                    ("num_speculative_tokens", "3"),
                ]
            )
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["speculative_config"] == {
                "method": "eagle3",
                "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
                "num_speculative_tokens": 3,
            }
            assert "speculative_method" not in kwargs
            assert "speculative_model" not in kwargs
            assert "num_speculative_tokens" not in kwargs

    def test_ngram_config(self) -> None:
        """NGRAM: method + num_speculative_tokens + prompt_lookup_max."""
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(
                engine_params=[
                    ("speculative_method", "ngram"),
                    ("num_speculative_tokens", "5"),
                    ("prompt_lookup_max", "4"),
                ]
            )
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["speculative_config"] == {
                "method": "ngram",
                "num_speculative_tokens": 5,
                "prompt_lookup_max": 4,
            }

    def test_draft_model_with_tp(self) -> None:
        """Draft model with tensor parallel size for the drafter."""
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(
                engine_params=[
                    ("speculative_method", "eagle"),
                    ("speculative_model", "some/draft-model"),
                    ("speculative_draft_tensor_parallel_size", "2"),
                ]
            )
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["speculative_config"] == {
                "method": "eagle",
                "model": "some/draft-model",
                "draft_tensor_parallel_size": 2,
            }

    def test_no_speculative_params_returns_none(self) -> None:
        """No speculative params → no speculative_config key."""
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(
                engine_params=[("tensor_parallel_size", "4")]
            )
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert "speculative_config" not in kwargs
            assert kwargs["tensor_parallel_size"] == 4

    def test_mixed_speculative_and_regular_params(self) -> None:
        """Speculative params separated from regular engine params."""
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(
                engine_params=[
                    ("tensor_parallel_size", "4"),
                    ("speculative_method", "eagle3"),
                    ("speculative_model", "draft/model"),
                    ("num_speculative_tokens", "3"),
                    ("gpu_memory_utilization", "0.9"),
                    ("enforce_eager", True),
                ]
            )
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["speculative_config"] == {
                "method": "eagle3",
                "model": "draft/model",
                "num_speculative_tokens": 3,
            }
            assert kwargs["tensor_parallel_size"] == 4
            assert kwargs["gpu_memory_utilization"] == 0.9
            assert kwargs["enforce_eager"] is True

    def test_mtp_no_draft_model(self) -> None:
        """MTP: method + num_speculative_tokens only (no draft model)."""
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(
                engine_params=[
                    ("speculative_method", "mtp"),
                    ("num_speculative_tokens", "2"),
                ]
            )
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert kwargs["speculative_config"] == {
                "method": "mtp",
                "num_speculative_tokens": 2,
            }


# ============================================================
# Generate (Output Extraction via Async Engine)
# ============================================================


class TestVLLMGenerate:
    """Verify generation output extraction from mock async vLLM engine."""

    @pytest.mark.asyncio
    async def test_generate_returns_text_and_counts(self) -> None:
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint()
            transport = VLLMTransport(model_endpoint=endpoint)

            mock_output = MockRequestOutput(
                prompt_token_ids=list(range(15)),
                outputs=[
                    MockCompletionOutput(
                        text="Hello, how can I help?",
                        token_ids=list(range(8)),
                        finish_reason="stop",
                    )
                ],
            )
            transport._engine = MockAsyncEngine(outputs=[mock_output])

            (
                text,
                input_tokens,
                output_tokens,
                finish_reason,
            ) = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={"temperature": 0.7},
                request_id="req-001",
            )

            assert text == "Hello, how can I help?"
            assert input_tokens == 15
            assert output_tokens == 8
            assert finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_none_finish_reason_defaults_to_stop(self) -> None:
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint()
            transport = VLLMTransport(model_endpoint=endpoint)

            mock_output = MockRequestOutput(
                outputs=[MockCompletionOutput(finish_reason=None)],
            )
            transport._engine = MockAsyncEngine(outputs=[mock_output])

            _, _, _, finish_reason = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-001",
            )
            assert finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_empty_output_raises(self) -> None:
        """RuntimeError when the engine yields no output at all."""
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint()
            transport = VLLMTransport(model_endpoint=endpoint)
            transport._engine = MockAsyncEngine(outputs=[])

            with pytest.raises(RuntimeError, match="no output"):
                await transport._generate(
                    messages=[{"role": "user", "content": "Hi"}],
                    sampling_params={},
                    request_id="req-001",
                )

    @pytest.mark.asyncio
    async def test_generate_none_prompt_token_ids_returns_zero(self) -> None:
        """Handle None prompt_token_ids gracefully."""
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint()
            transport = VLLMTransport(model_endpoint=endpoint)

            mock_output = MockRequestOutput(
                outputs=[MockCompletionOutput(text="ok", token_ids=[1, 2])],
            )
            mock_output.prompt_token_ids = None  # type: ignore[assignment]
            transport._engine = MockAsyncEngine(outputs=[mock_output])

            _, input_tokens, output_tokens, _ = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-001",
            )

            assert input_tokens == 0
            assert output_tokens == 2

    @pytest.mark.asyncio
    async def test_generate_finish_reason_normalized_to_string(self) -> None:
        """Enum-like finish_reason is converted to string."""

        class FinishReasonEnum:
            def __str__(self) -> str:
                return "length"

        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint()
            transport = VLLMTransport(model_endpoint=endpoint)

            mock_output = MockRequestOutput(
                outputs=[
                    MockCompletionOutput(
                        text="ok",
                        token_ids=[1],
                        finish_reason=FinishReasonEnum(),  # type: ignore[arg-type]
                    )
                ],
            )
            transport._engine = MockAsyncEngine(outputs=[mock_output])

            _, _, _, finish_reason = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-001",
            )
            assert finish_reason == "length"
            assert isinstance(finish_reason, str)


# ============================================================
# Streaming DELTA Mode
# ============================================================


class MockDeltaCompletionOutput:
    """Mock vLLM CompletionOutput for DELTA mode (incremental text/tokens)."""

    def __init__(
        self,
        text: str = "",
        token_ids: list[int] | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.text = text
        self.token_ids = token_ids or []
        self.finish_reason = finish_reason


class MockDeltaRequestOutput:
    """Mock vLLM RequestOutput for DELTA mode."""

    def __init__(
        self,
        prompt_token_ids: list[int] | None = None,
        outputs: list[MockDeltaCompletionOutput] | None = None,
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs or []


class MockStreamingAsyncEngine:
    """Mock AsyncLLMEngine that yields multiple DELTA outputs."""

    def __init__(self, outputs: list[MockDeltaRequestOutput] | None = None) -> None:
        self._outputs = outputs or []
        self._tokenizer = MagicMock()

    async def generate(self, prompt: str, sampling_params: Any, request_id: str) -> Any:
        for output in self._outputs:
            yield output

    def get_tokenizer(self) -> Any:
        return self._tokenizer

    def shutdown(self) -> None:
        pass


class TestVLLMStreamingGenerate:
    """Verify DELTA mode streaming generation."""

    @pytest.mark.asyncio
    async def test_streaming_accumulates_text_and_tokens(self) -> None:
        """DELTA mode accumulates text and token_ids across deltas."""
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint()
            endpoint.endpoint.streaming = True
            transport = VLLMTransport(model_endpoint=endpoint)

            deltas = [
                MockDeltaRequestOutput(
                    prompt_token_ids=list(range(10)),
                    outputs=[MockDeltaCompletionOutput(text="Hello", token_ids=[1, 2])],
                ),
                MockDeltaRequestOutput(
                    prompt_token_ids=list(range(10)),
                    outputs=[
                        MockDeltaCompletionOutput(text=" world", token_ids=[3, 4])
                    ],
                ),
                MockDeltaRequestOutput(
                    prompt_token_ids=list(range(10)),
                    outputs=[
                        MockDeltaCompletionOutput(
                            text="!", token_ids=[5], finish_reason="stop"
                        )
                    ],
                ),
            ]
            transport._engine = MockStreamingAsyncEngine(outputs=deltas)

            (
                text,
                input_tokens,
                output_tokens,
                finish_reason,
            ) = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-stream-001",
            )

            assert text == "Hello world!"
            assert input_tokens == 10
            assert output_tokens == 5
            assert finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_streaming_sets_first_token_perf_ns(self) -> None:
        """DELTA mode sets _first_token_perf_ns on first delta."""
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint()
            endpoint.endpoint.streaming = True
            transport = VLLMTransport(model_endpoint=endpoint)

            deltas = [
                MockDeltaRequestOutput(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[
                        MockDeltaCompletionOutput(
                            text="Hi", token_ids=[10], finish_reason="stop"
                        )
                    ],
                ),
            ]
            transport._engine = MockStreamingAsyncEngine(outputs=deltas)

            assert transport._first_token_perf_ns is None

            await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-stream-002",
            )

            assert transport._first_token_perf_ns is not None
            assert transport._first_token_perf_ns > 0

    @pytest.mark.asyncio
    async def test_streaming_empty_output_raises(self) -> None:
        """RuntimeError when DELTA mode yields no deltas."""
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint()
            endpoint.endpoint.streaming = True
            transport = VLLMTransport(model_endpoint=endpoint)
            transport._engine = MockStreamingAsyncEngine(outputs=[])

            with pytest.raises(RuntimeError, match="no output"):
                await transport._generate(
                    messages=[{"role": "user", "content": "Hi"}],
                    sampling_params={},
                    request_id="req-stream-003",
                )

    @pytest.mark.asyncio
    async def test_non_streaming_uses_final_only(self) -> None:
        """When streaming is False, generate uses FINAL_ONLY (existing path)."""
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint()
            endpoint.endpoint.streaming = False
            transport = VLLMTransport(model_endpoint=endpoint)

            mock_output = MockRequestOutput(
                prompt_token_ids=list(range(5)),
                outputs=[
                    MockCompletionOutput(
                        text="Final", token_ids=[1, 2], finish_reason="stop"
                    )
                ],
            )
            transport._engine = MockAsyncEngine(outputs=[mock_output])

            text, _, _, _ = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-final-001",
            )
            assert text == "Final"
            assert transport._first_token_perf_ns is None


# ============================================================
# Warmup Iterations
# ============================================================


class TestVLLMWarmupIterations:
    """Verify warmup_iterations is popped from engine params."""

    def test_warmup_iterations_popped_from_kwargs(self) -> None:
        with _patch_vllm_modules():
            from aiperf.transports.in_engine.vllm_transport import VLLMTransport

            endpoint = _make_vllm_endpoint(
                engine_params=[
                    ("warmup_iterations", "5"),
                    ("tensor_parallel_size", "2"),
                ]
            )
            transport = VLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()

            assert "warmup_iterations" not in kwargs
            assert transport._warmup_iterations == 5
            assert kwargs["tensor_parallel_size"] == 2
