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

import asyncio
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

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
        self.token_ids = list(range(10)) if token_ids is None else token_ids
        self.finish_reason = finish_reason


class MockTRTRequestOutput:
    """Mock TRT-LLM RequestOutput."""

    def __init__(
        self,
        prompt_token_ids: list[int] | None = None,
        outputs: list[MockTRTCompletionOutput] | None = None,
    ) -> None:
        self.prompt_token_ids = (
            list(range(8)) if prompt_token_ids is None else prompt_token_ids
        )
        self.outputs = [MockTRTCompletionOutput()] if outputs is None else outputs


class MockTRTGenerateHandle:
    """Mock generate_async handle with prompt_token_ids and async aresult().

    generate_async() returns this synchronously. aresult() is awaited to get
    the final RequestOutput.
    """

    def __init__(
        self,
        prompt_token_ids: list[int] | None = None,
        response: MockTRTRequestOutput | None = None,
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self._response = response or MockTRTRequestOutput()

    async def aresult(self) -> MockTRTRequestOutput:
        """Async method that returns the final response."""
        return self._response


class MockTRTSamplingParams:
    """Mock TRT-LLM SamplingParams that captures constructor kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockCapacitySchedulerPolicy:
    """Mock CapacitySchedulerPolicy enum."""

    GUARANTEED_NO_EVICT = "GUARANTEED_NO_EVICT"
    MAX_UTILIZATION = "MAX_UTILIZATION"


class MockSchedulerConfig:
    """Mock SchedulerConfig that captures constructor kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockKvCacheConfig:
    """Mock KvCacheConfig that captures constructor kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockExtendedRuntimePerfKnobConfig:
    """Mock ExtendedRuntimePerfKnobConfig with mutable attributes."""

    def __init__(self) -> None:
        self.cuda_graph_mode: bool = False
        self.multi_block_mode: bool = False
        self.cuda_graph_cache_size: int = 1000


def _build_mock_trtllm_module() -> types.ModuleType:
    """Build a mock tensorrt_llm module with needed classes and llmapi submodule."""
    mock_trtllm = types.ModuleType("tensorrt_llm")
    mock_trtllm.SamplingParams = MockTRTSamplingParams  # type: ignore[attr-defined]
    mock_trtllm.LLM = MagicMock  # type: ignore[attr-defined]

    # Build llmapi submodule with config classes
    mock_llmapi = types.ModuleType("tensorrt_llm.llmapi")
    mock_llmapi.CapacitySchedulerPolicy = MockCapacitySchedulerPolicy  # type: ignore[attr-defined]
    mock_llmapi.SchedulerConfig = MockSchedulerConfig  # type: ignore[attr-defined]
    mock_llmapi.KvCacheConfig = MockKvCacheConfig  # type: ignore[attr-defined]
    mock_llmapi.ExtendedRuntimePerfKnobConfig = MockExtendedRuntimePerfKnobConfig  # type: ignore[attr-defined]
    mock_trtllm.llmapi = mock_llmapi  # type: ignore[attr-defined]

    return mock_trtllm


def _build_mock_trtllm_modules() -> dict[str, types.ModuleType]:
    """Build dict of all mock tensorrt_llm modules for sys.modules patching."""
    mod = _build_mock_trtllm_module()
    return {
        "tensorrt_llm": mod,
        "tensorrt_llm.llmapi": mod.llmapi,  # type: ignore[attr-defined]
    }


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

            mock_response = MockTRTRequestOutput(
                prompt_token_ids=list(range(12)),
                outputs=[
                    MockTRTCompletionOutput(
                        text="TRT-LLM generated text",
                        token_ids=list(range(15)),
                        finish_reason="stop",
                    )
                ],
            )
            handle = MockTRTGenerateHandle(
                prompt_token_ids=list(range(12)),
                response=mock_response,
            )
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
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

            # Create response WITHOUT prompt_token_ids attribute
            mock_completion = MockTRTCompletionOutput(
                text="Result", token_ids=[1, 2, 3]
            )
            mock_response = MagicMock()
            mock_response.outputs = [mock_completion]
            del mock_response.prompt_token_ids

            # Handle also missing prompt_token_ids
            handle = MagicMock()
            del handle.prompt_token_ids
            handle.aresult = AsyncMock(return_value=mock_response)

            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
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

            mock_response = MockTRTRequestOutput(
                outputs=[MockTRTCompletionOutput(finish_reason=None)],
            )
            handle = MockTRTGenerateHandle(response=mock_response)
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
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
            mock_response = MockTRTRequestOutput(outputs=[mock_completion])
            handle = MockTRTGenerateHandle(response=mock_response)
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
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


# ============================================================
# _warmup_single (TRT-LLM Engine)
# ============================================================


class TestTRTLLMWarmupSingle:
    """Verify _warmup_single calls TRT-LLM engine in correct streaming mode."""

    @pytest.mark.asyncio
    async def test_warmup_non_streaming_calls_aresult(self) -> None:
        """streaming=False calls generate_async(streaming=False) then aresult()."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            transport = TRTLLMTransport(model_endpoint=endpoint)

            mock_handle = MagicMock()
            mock_handle.aresult = AsyncMock(return_value=MockTRTRequestOutput())
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = mock_handle
            transport._engine = mock_engine

            await transport._warmup_single("test prompt", 4, streaming=False)

            mock_engine.generate_async.assert_called_once()
            call_kwargs = mock_engine.generate_async.call_args
            assert call_kwargs[1]["streaming"] is False
            mock_handle.aresult.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_warmup_streaming_iterates_async_generator(self) -> None:
        """streaming=True calls generate_async(streaming=True) and iterates."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            transport = TRTLLMTransport(model_endpoint=endpoint)

            async def mock_stream(*args: Any, **kwargs: Any) -> Any:
                yield MockTRTRequestOutput()
                yield MockTRTRequestOutput()

            mock_engine = MagicMock()
            mock_engine.generate_async = mock_stream
            transport._engine = mock_engine

            await transport._warmup_single("test prompt", 4, streaming=True)

    @pytest.mark.asyncio
    async def test_warmup_single_passes_max_tokens_in_sampling_params(self) -> None:
        """_warmup_single creates SamplingParams with the given max_tokens."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            transport = TRTLLMTransport(model_endpoint=endpoint)

            mock_handle = MagicMock()
            mock_handle.aresult = AsyncMock(return_value=MockTRTRequestOutput())
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = mock_handle
            transport._engine = mock_engine

            await transport._warmup_single("test prompt", 16, streaming=False)

            # Verify the sampling params passed to generate_async
            call_args = mock_engine.generate_async.call_args
            sampling_params = call_args[0][1]  # Second positional arg
            assert sampling_params._kwargs["max_tokens"] == 16


# ============================================================
# Latency-Optimized Preset
# ============================================================


def _build_latency_kwargs(
    engine_params: list[tuple[str, Any]],
) -> tuple[dict[str, Any], Any]:
    """Helper: build engine kwargs with mocked llmapi and return (kwargs, transport)."""
    mods = _build_mock_trtllm_modules()
    with patch.dict(sys.modules, mods):
        from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

        endpoint = _make_trtllm_endpoint(engine_params=engine_params)
        transport = TRTLLMTransport(model_endpoint=endpoint)
        kwargs = transport._build_engine_kwargs()
        return kwargs, transport


class TestTRTLLMLatencyOptimized:
    """Verify the latency_optimized preset applies trtllm-bench low-latency defaults."""

    def test_preset_sets_all_defaults(self) -> None:
        """latency_optimized=true sets batch=1, scheduler, kv_cache, perf, env, warmup."""
        kwargs, transport = _build_latency_kwargs([("latency_optimized", "true")])

        # Scalar defaults
        assert kwargs["max_batch_size"] == 1
        assert kwargs["enable_chunked_prefill"] is False
        assert transport._warmup_iterations == 2

        # SchedulerConfig with GUARANTEED_NO_EVICT
        sc = kwargs["scheduler_config"]
        assert isinstance(sc, MockSchedulerConfig)
        assert sc.capacity_scheduler_policy == "GUARANTEED_NO_EVICT"

        # KvCacheConfig at 90%
        kv = kwargs["kv_cache_config"]
        assert isinstance(kv, MockKvCacheConfig)
        assert kv.free_gpu_memory_fraction == 0.90

        # PerfKnobConfig with cuda_graphs and multi_block
        perf = kwargs["extended_runtime_perf_knob_config"]
        assert isinstance(perf, MockExtendedRuntimePerfKnobConfig)
        assert perf.cuda_graph_mode is True
        assert perf.multi_block_mode is True

        # Environment overrides
        env = kwargs["env_overrides"]
        assert env["TRTLLM_ENABLE_PDL"] == "1"
        assert env["FORCE_MULTI_BLOCK_MODE"] == "1"
        assert env["TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG"] == "1"
        assert env["TRTLLM_MMHA_KERNEL_BLOCK_SIZE"] == "256"

    def test_user_override_takes_priority(self) -> None:
        """User-provided values override latency preset defaults."""
        kwargs, transport = _build_latency_kwargs(
            [
                ("latency_optimized", "true"),
                ("max_batch_size", "8"),
                ("enable_chunked_prefill", "true"),
                ("kv_cache_free_gpu_mem_fraction", "0.75"),
                ("cuda_graphs", "false"),
                ("warmup_iterations", "5"),
            ]
        )

        assert kwargs["max_batch_size"] == 8
        assert kwargs["enable_chunked_prefill"] is True
        assert kwargs["kv_cache_config"].free_gpu_memory_fraction == 0.75
        assert kwargs["extended_runtime_perf_knob_config"].cuda_graph_mode is False
        assert transport._warmup_iterations == 5

    def test_latency_optimized_false_does_nothing(self) -> None:
        """latency_optimized=false produces same output as omitting it entirely."""
        kwargs_false, _ = _build_latency_kwargs([("latency_optimized", "false")])
        kwargs_omit, _ = _build_latency_kwargs([])

        assert kwargs_false == kwargs_omit

    def test_latency_optimized_is_popped(self) -> None:
        """latency_optimized must not appear in final kwargs (not an LLM param)."""
        kwargs, _ = _build_latency_kwargs([("latency_optimized", "true")])
        assert "latency_optimized" not in kwargs


# ============================================================
# Structured Configs (individual knobs, no preset)
# ============================================================


class TestTRTLLMStructuredConfigs:
    """Verify individual structured config knobs without the latency preset."""

    def test_scheduler_policy_alone(self) -> None:
        """scheduler_policy creates SchedulerConfig when set without preset."""
        kwargs, _ = _build_latency_kwargs([("scheduler_policy", "MAX_UTILIZATION")])
        sc = kwargs["scheduler_config"]
        assert isinstance(sc, MockSchedulerConfig)
        assert sc.capacity_scheduler_policy == "MAX_UTILIZATION"
        assert "kv_cache_config" not in kwargs
        assert "extended_runtime_perf_knob_config" not in kwargs

    def test_kv_cache_percent_alone(self) -> None:
        """kv_cache_free_gpu_mem_fraction creates KvCacheConfig when set alone."""
        kwargs, _ = _build_latency_kwargs([("kv_cache_free_gpu_mem_fraction", "0.80")])
        kv = kwargs["kv_cache_config"]
        assert isinstance(kv, MockKvCacheConfig)
        assert kv.free_gpu_memory_fraction == 0.80
        assert "scheduler_config" not in kwargs

    def test_cuda_graphs_alone(self) -> None:
        """cuda_graphs creates ExtendedRuntimePerfKnobConfig when set alone."""
        kwargs, _ = _build_latency_kwargs([("cuda_graphs", "true")])
        perf = kwargs["extended_runtime_perf_knob_config"]
        assert isinstance(perf, MockExtendedRuntimePerfKnobConfig)
        assert perf.cuda_graph_mode is True
        # multi_block_mode stays at default (False) when not set
        assert perf.multi_block_mode is False

    def test_multi_block_mode_alone(self) -> None:
        """multi_block_mode creates ExtendedRuntimePerfKnobConfig when set alone."""
        kwargs, _ = _build_latency_kwargs([("multi_block_mode", "true")])
        perf = kwargs["extended_runtime_perf_knob_config"]
        assert isinstance(perf, MockExtendedRuntimePerfKnobConfig)
        assert perf.multi_block_mode is True
        assert perf.cuda_graph_mode is False

    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            param(True, True, id="bool-True"),
            param(False, False, id="bool-False"),
        ],
    )  # fmt: skip
    def test_parse_bool(self, input_val: Any, expected: bool) -> None:
        """_parse_bool coerces various string/bool inputs correctly."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            assert TRTLLMTransport._parse_bool(input_val) is expected


# ============================================================
# generate_async Call Contract
# ============================================================


class TestTRTLLMGenerateAsyncCallContract:
    """Verify generate_async is called with correct arguments."""

    @pytest.mark.asyncio
    async def test_generate_final_only_calls_generate_async_with_string_prompt(
        self,
    ) -> None:
        """generate_async receives a single string prompt, not a list."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            handle = MockTRTGenerateHandle(prompt_token_ids=list(range(5)))
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            await transport._generate(
                messages=[{"role": "user", "content": "Hello"}],
                sampling_params={},
                request_id="req-contract-01",
            )

            call_args = mock_engine.generate_async.call_args
            prompt_arg = call_args[0][0]
            assert isinstance(prompt_arg, str)
            assert "Hello" in prompt_arg

    @pytest.mark.asyncio
    async def test_generate_final_only_passes_streaming_false(self) -> None:
        """Non-streaming path passes streaming=False to generate_async."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            handle = MockTRTGenerateHandle(prompt_token_ids=list(range(5)))
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-contract-02",
            )

            call_kwargs = mock_engine.generate_async.call_args
            assert call_kwargs[1]["streaming"] is False

    @pytest.mark.asyncio
    async def test_generate_with_input_ids_passes_list_directly(self) -> None:
        """When input_ids provided, generate_async receives list[int], not string."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            handle = MockTRTGenerateHandle(prompt_token_ids=[100, 200, 300])
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            token_ids = [100, 200, 300]
            await transport._generate(
                messages=[{"role": "user", "content": "ignored"}],
                sampling_params={},
                request_id="req-input-ids",
                input_ids=token_ids,
            )

            prompt_arg = mock_engine.generate_async.call_args[0][0]
            assert prompt_arg == [100, 200, 300]


# ============================================================
# aresult() Error Propagation
# ============================================================


class TestTRTLLMAresultErrors:
    """Verify errors from aresult() propagate correctly."""

    @pytest.mark.asyncio
    async def test_generate_final_only_aresult_exception_propagates(self) -> None:
        """RuntimeError from aresult() surfaces to the caller."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())

            handle = MagicMock()
            handle.prompt_token_ids = [1, 2, 3]
            handle.aresult = AsyncMock(side_effect=RuntimeError("engine OOM"))

            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            with pytest.raises(RuntimeError, match="engine OOM"):
                await transport._generate(
                    messages=[{"role": "user", "content": "Hi"}],
                    sampling_params={},
                    request_id="req-oom",
                )

    @pytest.mark.asyncio
    async def test_generate_final_only_cancelled_error_propagates(self) -> None:
        """CancelledError from aresult() is not swallowed."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())

            handle = MagicMock()
            handle.prompt_token_ids = [1, 2]
            handle.aresult = AsyncMock(side_effect=asyncio.CancelledError)

            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            with pytest.raises(asyncio.CancelledError):
                await transport._generate(
                    messages=[{"role": "user", "content": "Hi"}],
                    sampling_params={},
                    request_id="req-cancel",
                )


# ============================================================
# Warmup Uses generate_async + aresult
# ============================================================


class TestTRTLLMWarmupMechanism:
    """Verify warmup uses generate_async + aresult (not run_in_executor)."""

    @pytest.mark.asyncio
    async def test_warmup_single_non_streaming_calls_generate_async_and_aresult(
        self,
    ) -> None:
        """Non-streaming warmup calls generate_async(streaming=False) then aresult()."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())

            mock_handle = MagicMock()
            mock_handle.aresult = AsyncMock(return_value=MockTRTRequestOutput())
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = mock_handle
            transport._engine = mock_engine

            await transport._warmup_single("warmup prompt", 4, streaming=False)

            mock_engine.generate_async.assert_called_once()
            call_kwargs = mock_engine.generate_async.call_args[1]
            assert call_kwargs["streaming"] is False
            mock_handle.aresult.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_warmup_single_streaming_iterates_generate_async(self) -> None:
        """Streaming warmup iterates generate_async(streaming=True)."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())

            # generate_async returns an async iterator when streaming=True
            async def mock_stream(*args: Any, **kwargs: Any) -> Any:
                for output in [MockTRTRequestOutput(), MockTRTRequestOutput()]:
                    yield output

            mock_engine = MagicMock()
            mock_engine.generate_async = mock_stream
            transport._engine = mock_engine

            # Should complete without error
            await transport._warmup_single("warmup prompt", 4, streaming=True)

    @pytest.mark.asyncio
    async def test_warmup_does_not_use_run_in_executor(self) -> None:
        """Confirm warmup path does not call loop.run_in_executor."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            transport._warmup_iterations = 1
            transport._warmup_input_tokens = 10
            transport._warmup_output_tokens = 2

            mock_handle = MagicMock()
            mock_handle.aresult = AsyncMock(return_value=MockTRTRequestOutput())
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = mock_handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            with patch.object(
                asyncio.get_event_loop(), "run_in_executor"
            ) as mock_executor:
                await transport._run_warmup()
                mock_executor.assert_not_called()


# ============================================================
# prompt_token_ids Fallback Logic
# ============================================================


class TestTRTLLMPromptTokenIdsFallback:
    """Verify prompt_token_ids is read from handle first, then response."""

    @pytest.mark.asyncio
    async def test_prompt_token_ids_from_handle_takes_priority(self) -> None:
        """When handle has prompt_token_ids, response's value is ignored."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())

            response = MockTRTRequestOutput(
                prompt_token_ids=list(range(20)),
                outputs=[MockTRTCompletionOutput(text="out", token_ids=[1])],
            )
            handle = MockTRTGenerateHandle(
                prompt_token_ids=list(range(5)),
                response=response,
            )
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            _, input_tokens, _, _ = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-fallback-handle",
            )
            assert input_tokens == 5

    @pytest.mark.asyncio
    async def test_prompt_token_ids_falls_back_to_response(self) -> None:
        """When handle has no prompt_token_ids, reads from response."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())

            response = MockTRTRequestOutput(
                prompt_token_ids=list(range(7)),
                outputs=[MockTRTCompletionOutput(text="out", token_ids=[1])],
            )
            # Handle has None for prompt_token_ids -> fallback to response
            handle = MockTRTGenerateHandle(
                prompt_token_ids=None,
                response=response,
            )
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            _, input_tokens, _, _ = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-fallback-response",
            )
            assert input_tokens == 7

    @pytest.mark.asyncio
    async def test_prompt_token_ids_empty_on_handle_falls_back_to_response(
        self,
    ) -> None:
        """When handle has empty prompt_token_ids, reads from response."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())

            response = MockTRTRequestOutput(
                prompt_token_ids=list(range(9)),
                outputs=[MockTRTCompletionOutput(text="out", token_ids=[1])],
            )
            # Handle has empty list -> falsy, falls back to response
            handle = MockTRTGenerateHandle(
                prompt_token_ids=[],
                response=response,
            )
            # Override aresult to return our response since MockTRTGenerateHandle
            # stores prompt_token_ids=[] which is falsy
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            _, input_tokens, _, _ = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-fallback-empty",
            )
            assert input_tokens == 9


# ============================================================
# Edge Cases
# ============================================================


class TestTRTLLMGenerateEdgeCases:
    """Verify edge cases: empty text, zero tokens, speculative decoding metadata."""

    @pytest.mark.asyncio
    async def test_generate_empty_text_returns_empty_string(self) -> None:
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())

            response = MockTRTRequestOutput(
                prompt_token_ids=list(range(3)),
                outputs=[MockTRTCompletionOutput(text="", token_ids=[])],
            )
            handle = MockTRTGenerateHandle(
                prompt_token_ids=list(range(3)),
                response=response,
            )
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            text, _, output_tokens, _ = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-empty",
            )
            assert text == ""
            assert output_tokens == 0

    @pytest.mark.asyncio
    async def test_generate_missing_token_ids_returns_zero(self) -> None:
        """When completion has no token_ids attr, output_tokens defaults to 0."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())

            mock_completion = MagicMock()
            mock_completion.text = "some text"
            del mock_completion.token_ids
            mock_completion.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.outputs = [mock_completion]
            del mock_response.prompt_token_ids

            handle = MagicMock()
            handle.prompt_token_ids = [1, 2]
            handle.aresult = AsyncMock(return_value=mock_response)

            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            _, _, output_tokens, _ = await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-no-tokenids",
            )
            assert output_tokens == 0

    @pytest.mark.asyncio
    async def test_generate_captures_decoding_iter_metadata(self) -> None:
        """Speculative decoding metadata (decoding_iter) is captured from response."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())

            response = MockTRTRequestOutput(
                prompt_token_ids=list(range(5)),
                outputs=[MockTRTCompletionOutput(text="ok", token_ids=[1, 2])],
            )
            response.decoding_iter = 42  # type: ignore[attr-defined]

            handle = MockTRTGenerateHandle(
                prompt_token_ids=list(range(5)),
                response=response,
            )
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-speculative",
            )
            assert transport._decode_iterations == 42

    @pytest.mark.asyncio
    async def test_generate_no_decoding_iter_defaults_to_none(self) -> None:
        """Without decoding_iter on response, _decode_iterations is None."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())

            response = MockTRTRequestOutput(
                prompt_token_ids=list(range(5)),
                outputs=[MockTRTCompletionOutput(text="ok", token_ids=[1])],
            )
            # MockTRTRequestOutput does NOT have decoding_iter attribute
            handle = MockTRTGenerateHandle(
                prompt_token_ids=list(range(5)),
                response=response,
            )
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-no-spec",
            )
            assert transport._decode_iterations is None

    @pytest.mark.asyncio
    async def test_generate_sampling_params_forwarded_to_trt(self) -> None:
        """Sampling params dict is unpacked into TRTSamplingParams."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            handle = MockTRTGenerateHandle(prompt_token_ids=list(range(5)))
            mock_engine = MagicMock()
            mock_engine.generate_async.return_value = handle
            del mock_engine.tokenizer
            transport._engine = mock_engine

            await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={"temperature": 0.7, "max_tokens": 256},
                request_id="req-params",
            )

            # SamplingParams constructor receives the dict kwargs
            params_arg = mock_engine.generate_async.call_args[0][1]
            assert isinstance(params_arg, MockTRTSamplingParams)
            assert params_arg.temperature == 0.7
            assert params_arg.max_tokens == 256


# ============================================================
# Streaming Generation
# ============================================================


class TestTRTLLMGenerateStreaming:
    """Verify streaming generation path (_generate_streaming)."""

    @pytest.mark.asyncio
    async def test_streaming_calls_generate_async_with_streaming_true(self) -> None:
        """Streaming path passes streaming=True to generate_async."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            endpoint.endpoint.streaming = True
            transport = TRTLLMTransport(model_endpoint=endpoint)

            chunks = [
                MockTRTRequestOutput(
                    prompt_token_ids=list(range(5)),
                    outputs=[
                        MockTRTCompletionOutput(
                            text="He", token_ids=[1], finish_reason=None
                        )
                    ],
                ),
                MockTRTRequestOutput(
                    prompt_token_ids=list(range(5)),
                    outputs=[
                        MockTRTCompletionOutput(
                            text="Hello", token_ids=[1, 2], finish_reason="stop"
                        )
                    ],
                ),
            ]

            call_record: dict[str, Any] = {}

            async def mock_generate_async(
                prompt: Any, params: Any, *, streaming: bool
            ) -> Any:
                call_record["prompt"] = prompt
                call_record["streaming"] = streaming
                for chunk in chunks:
                    yield chunk

            mock_engine = MagicMock()
            mock_engine.generate_async = mock_generate_async
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
                request_id="req-stream-01",
            )

            assert call_record["streaming"] is True
            assert isinstance(call_record["prompt"], str)
            assert text == "Hello"
            assert input_tokens == 5
            assert output_tokens == 2
            assert finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_streaming_sets_first_token_perf_ns(self) -> None:
        """Streaming path records _first_token_perf_ns on first content."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            endpoint.endpoint.streaming = True
            transport = TRTLLMTransport(model_endpoint=endpoint)

            chunks = [
                MockTRTRequestOutput(
                    prompt_token_ids=list(range(3)),
                    outputs=[
                        MockTRTCompletionOutput(
                            text="A", token_ids=[1], finish_reason=None
                        )
                    ],
                ),
                MockTRTRequestOutput(
                    prompt_token_ids=list(range(3)),
                    outputs=[
                        MockTRTCompletionOutput(
                            text="AB", token_ids=[1, 2], finish_reason="stop"
                        )
                    ],
                ),
            ]

            async def mock_generate_async(
                prompt: Any, params: Any, *, streaming: bool
            ) -> Any:
                for chunk in chunks:
                    yield chunk

            mock_engine = MagicMock()
            mock_engine.generate_async = mock_generate_async
            del mock_engine.tokenizer
            transport._engine = mock_engine

            assert transport._first_token_perf_ns is None
            await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-stream-ttft",
            )
            assert transport._first_token_perf_ns is not None
            assert isinstance(transport._first_token_perf_ns, int)

    @pytest.mark.asyncio
    async def test_streaming_no_output_raises_runtime_error(self) -> None:
        """When stream yields chunks with no content, raises RuntimeError."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            endpoint.endpoint.streaming = True
            transport = TRTLLMTransport(model_endpoint=endpoint)

            # All chunks have empty outputs list
            chunks = [
                MockTRTRequestOutput(prompt_token_ids=[], outputs=[]),
            ]
            # Override outputs to empty
            chunks[0].outputs = []

            async def mock_generate_async(
                prompt: Any, params: Any, *, streaming: bool
            ) -> Any:
                for chunk in chunks:
                    yield chunk

            mock_engine = MagicMock()
            mock_engine.generate_async = mock_generate_async
            del mock_engine.tokenizer
            transport._engine = mock_engine

            with pytest.raises(RuntimeError, match="no output"):
                await transport._generate(
                    messages=[{"role": "user", "content": "Hi"}],
                    sampling_params={},
                    request_id="req-stream-empty",
                )

    @pytest.mark.asyncio
    async def test_streaming_empty_generator_raises_runtime_error(self) -> None:
        """When generate_async yields nothing at all, raises RuntimeError."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            endpoint.endpoint.streaming = True
            transport = TRTLLMTransport(model_endpoint=endpoint)

            async def mock_generate_async(
                prompt: Any, params: Any, *, streaming: bool
            ) -> Any:
                return
                yield  # noqa: unreachable - makes this an async generator

            mock_engine = MagicMock()
            mock_engine.generate_async = mock_generate_async
            del mock_engine.tokenizer
            transport._engine = mock_engine

            with pytest.raises(RuntimeError, match="no output"):
                await transport._generate(
                    messages=[{"role": "user", "content": "Hi"}],
                    sampling_params={},
                    request_id="req-stream-none",
                )

    @pytest.mark.asyncio
    async def test_streaming_captures_decoding_iter_from_last_chunk(self) -> None:
        """Streaming path reads decoding_iter from the last yielded output."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            endpoint.endpoint.streaming = True
            transport = TRTLLMTransport(model_endpoint=endpoint)

            chunk1 = MockTRTRequestOutput(
                prompt_token_ids=list(range(3)),
                outputs=[
                    MockTRTCompletionOutput(text="A", token_ids=[1], finish_reason=None)
                ],
            )
            chunk2 = MockTRTRequestOutput(
                prompt_token_ids=list(range(3)),
                outputs=[
                    MockTRTCompletionOutput(
                        text="AB", token_ids=[1, 2], finish_reason="stop"
                    )
                ],
            )
            chunk2.decoding_iter = 7  # type: ignore[attr-defined]

            async def mock_generate_async(
                prompt: Any, params: Any, *, streaming: bool
            ) -> Any:
                yield chunk1
                yield chunk2

            mock_engine = MagicMock()
            mock_engine.generate_async = mock_generate_async
            del mock_engine.tokenizer
            transport._engine = mock_engine

            await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-stream-spec",
            )
            assert transport._decode_iterations == 7


# ============================================================
# Stop Engine
# ============================================================


class TestTRTLLMStopEngine:
    """Verify engine shutdown behavior."""

    @pytest.mark.asyncio
    async def test_stop_engine_calls_shutdown(self) -> None:
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            mock_engine = MagicMock()
            transport._engine = mock_engine

            await transport._stop_engine()

            mock_engine.shutdown.assert_called_once()
            assert transport._engine is None

    @pytest.mark.asyncio
    async def test_stop_engine_no_shutdown_method(self) -> None:
        """Engine without shutdown() method doesn't raise."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            mock_engine = MagicMock(spec=[])  # No methods at all
            transport._engine = mock_engine

            await transport._stop_engine()
            assert transport._engine is None

    @pytest.mark.asyncio
    async def test_stop_engine_already_none(self) -> None:
        """Stopping when engine is None is a no-op."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            transport._engine = None

            await transport._stop_engine()  # Should not raise
            assert transport._engine is None


# ============================================================
# Speculative Decoding: max_draft_len Extraction
# ============================================================


class TestTRTLLMMaxDraftLenExtraction:
    """Verify max_draft_len is extracted from engine_params during _build_engine_kwargs."""

    def test_max_draft_len_extracted_and_stored(self) -> None:
        """max_draft_len from engine_params is stored on transport, not passed to LLM."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(engine_params=[("max_draft_len", "5")])
            transport = TRTLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()

            assert transport._max_draft_len == 5
            assert "max_draft_len" not in kwargs

    def test_max_draft_len_not_present_defaults_to_zero(self) -> None:
        """Without max_draft_len in engine_params, _max_draft_len stays at 0."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(engine_params=[])
            transport = TRTLLMTransport(model_endpoint=endpoint)
            transport._build_engine_kwargs()

            assert transport._max_draft_len == 0

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", 1),
            ("10", 10),
            ("0", 0),
        ],
    )  # fmt: skip
    def test_max_draft_len_string_coerced_to_int(
        self, value: str, expected: int
    ) -> None:
        """max_draft_len string values from CLI are coerced to int."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(engine_params=[("max_draft_len", value)])
            transport = TRTLLMTransport(model_endpoint=endpoint)
            transport._build_engine_kwargs()

            assert transport._max_draft_len == expected


# ============================================================
# Speculative Decoding: Streaming decoding_iter Absent
# ============================================================


class TestTRTLLMStreamingSpecDecodeEdgeCases:
    """Verify streaming getattr fallback when decoding_iter is absent."""

    @pytest.mark.asyncio
    async def test_streaming_no_decoding_iter_defaults_to_none(self) -> None:
        """Streaming path: when last chunk has no decoding_iter attr, _decode_iterations is None."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            endpoint.endpoint.streaming = True
            transport = TRTLLMTransport(model_endpoint=endpoint)

            chunk = MockTRTRequestOutput(
                prompt_token_ids=list(range(3)),
                outputs=[
                    MockTRTCompletionOutput(
                        text="Hello", token_ids=[1, 2], finish_reason="stop"
                    )
                ],
            )
            # MockTRTRequestOutput does NOT have decoding_iter attribute

            async def mock_generate_async(
                prompt: Any, params: Any, *, streaming: bool
            ) -> Any:
                yield chunk

            mock_engine = MagicMock()
            mock_engine.generate_async = mock_generate_async
            del mock_engine.tokenizer
            transport._engine = mock_engine

            await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-stream-no-spec",
            )
            assert transport._decode_iterations is None

    @pytest.mark.asyncio
    async def test_streaming_decoding_iter_zero_is_valid(self) -> None:
        """decoding_iter=0 is a valid value (all tokens accepted in single step)."""
        mock_trtllm = _build_mock_trtllm_module()
        with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm}):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint()
            endpoint.endpoint.streaming = True
            transport = TRTLLMTransport(model_endpoint=endpoint)

            chunk = MockTRTRequestOutput(
                prompt_token_ids=list(range(3)),
                outputs=[
                    MockTRTCompletionOutput(
                        text="Hi", token_ids=[1], finish_reason="stop"
                    )
                ],
            )
            chunk.decoding_iter = 0  # type: ignore[attr-defined]

            async def mock_generate_async(
                prompt: Any, params: Any, *, streaming: bool
            ) -> Any:
                yield chunk

            mock_engine = MagicMock()
            mock_engine.generate_async = mock_generate_async
            del mock_engine.tokenizer
            transport._engine = mock_engine

            await transport._generate(
                messages=[{"role": "user", "content": "Hi"}],
                sampling_params={},
                request_id="req-stream-zero-spec",
            )
            assert transport._decode_iterations == 0


# ============================================================
# _parse_bool Edge Cases
# ============================================================


class TestParseBoolEdgeCases:
    """Verify _parse_bool handles pathological and uncommon inputs."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (None, False),
            ("", False),
            (0, False),
            (1, True),
            (-1, True),
            (42, True),
            param("YES", True, id="uppercase-YES"),
            param("Yes", True, id="mixed-case-Yes"),
            param("tRuE", True, id="mixed-case-tRuE"),
            param("  true  ", False, id="whitespace-not-stripped"),
            param("on", False, id="on-not-recognized"),
        ],
    )  # fmt: skip
    def test_parse_bool_edge_input(self, input_val: Any, expected: bool) -> None:
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            assert TRTLLMTransport._parse_bool(input_val) is expected


# ============================================================
# _build_scheduler_config (Isolated)
# ============================================================


class TestBuildSchedulerConfigIsolated:
    """Verify _build_scheduler_config individually."""

    def test_no_params_no_preset_omits_scheduler(self) -> None:
        kwargs, _ = _build_latency_kwargs([])
        assert "scheduler_config" not in kwargs

    def test_user_policy_overrides_preset_default(self) -> None:
        """User scheduler_policy takes priority over preset's GUARANTEED_NO_EVICT."""
        kwargs, _ = _build_latency_kwargs(
            [
                ("latency_optimized", "true"),
                ("scheduler_policy", "MAX_UTILIZATION"),
            ]
        )
        sc = kwargs["scheduler_config"]
        assert sc.capacity_scheduler_policy == "MAX_UTILIZATION"

    def test_unknown_policy_string_passed_through(self) -> None:
        """Unrecognized policy string is used as-is via getattr fallback."""
        kwargs, _ = _build_latency_kwargs([("scheduler_policy", "CUSTOM_POLICY_V2")])
        sc = kwargs["scheduler_config"]
        assert sc.capacity_scheduler_policy == "CUSTOM_POLICY_V2"

    def test_import_error_returns_none_for_all_structured_configs(self) -> None:
        """When tensorrt_llm.llmapi is not importable, structured configs are omitted."""
        mock_trtllm = types.ModuleType("tensorrt_llm")
        mock_trtllm.SamplingParams = MockTRTSamplingParams  # type: ignore[attr-defined]
        mock_trtllm.LLM = MagicMock  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {"tensorrt_llm": mock_trtllm, "tensorrt_llm.llmapi": None},
        ):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(
                engine_params=[("latency_optimized", "true")]
            )
            transport = TRTLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()

            assert "scheduler_config" not in kwargs
            assert "kv_cache_config" not in kwargs
            assert "extended_runtime_perf_knob_config" not in kwargs
            # Scalar defaults and env_overrides are not gated by llmapi
            assert kwargs["max_batch_size"] == 1
            assert kwargs["enable_chunked_prefill"] is False
            assert kwargs["env_overrides"]["TRTLLM_ENABLE_PDL"] == "1"


# ============================================================
# _build_kv_cache_config (Isolated)
# ============================================================


class TestBuildKvCacheConfigIsolated:
    """Verify _build_kv_cache_config individually."""

    def test_no_params_no_preset_omits_kv_cache(self) -> None:
        kwargs, _ = _build_latency_kwargs([])
        assert "kv_cache_config" not in kwargs

    def test_user_fraction_overrides_preset_default(self) -> None:
        """User kv_cache_free_gpu_mem_fraction overrides preset's 0.90."""
        kwargs, _ = _build_latency_kwargs(
            [
                ("latency_optimized", "true"),
                ("kv_cache_free_gpu_mem_fraction", "0.50"),
            ]
        )
        assert kwargs["kv_cache_config"].free_gpu_memory_fraction == 0.50

    def test_user_fraction_alone_without_preset(self) -> None:
        kwargs, _ = _build_latency_kwargs([("kv_cache_free_gpu_mem_fraction", "0.65")])
        kv = kwargs["kv_cache_config"]
        assert isinstance(kv, MockKvCacheConfig)
        assert kv.free_gpu_memory_fraction == 0.65

    def test_import_error_kv_cache_returns_none(self) -> None:
        """When llmapi is not importable, kv_cache_config is omitted."""
        mock_trtllm = types.ModuleType("tensorrt_llm")
        mock_trtllm.SamplingParams = MockTRTSamplingParams  # type: ignore[attr-defined]
        mock_trtllm.LLM = MagicMock  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {"tensorrt_llm": mock_trtllm, "tensorrt_llm.llmapi": None},
        ):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(
                engine_params=[("kv_cache_free_gpu_mem_fraction", "0.80")]
            )
            transport = TRTLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert "kv_cache_config" not in kwargs


# ============================================================
# _build_perf_knob_config (Partial Overrides)
# ============================================================


class TestBuildPerfKnobConfigPartialOverrides:
    """Verify partial overrides: one knob user-set, preset fills other."""

    def test_user_cuda_graphs_false_preset_fills_multi_block(self) -> None:
        """User disables cuda_graphs, preset still enables multi_block_mode."""
        kwargs, _ = _build_latency_kwargs(
            [
                ("latency_optimized", "true"),
                ("cuda_graphs", "false"),
            ]
        )
        perf = kwargs["extended_runtime_perf_knob_config"]
        assert perf.cuda_graph_mode is False
        assert perf.multi_block_mode is True

    def test_user_multi_block_false_preset_fills_cuda_graphs(self) -> None:
        """User disables multi_block, preset still enables cuda_graph_mode."""
        kwargs, _ = _build_latency_kwargs(
            [
                ("latency_optimized", "true"),
                ("multi_block_mode", "false"),
            ]
        )
        perf = kwargs["extended_runtime_perf_knob_config"]
        assert perf.cuda_graph_mode is True
        assert perf.multi_block_mode is False

    def test_user_overrides_both_knobs_with_preset(self) -> None:
        """User overrides both knobs; preset does not force anything."""
        kwargs, _ = _build_latency_kwargs(
            [
                ("latency_optimized", "true"),
                ("cuda_graphs", "false"),
                ("multi_block_mode", "false"),
            ]
        )
        perf = kwargs["extended_runtime_perf_knob_config"]
        assert perf.cuda_graph_mode is False
        assert perf.multi_block_mode is False

    def test_import_error_perf_knob_returns_none(self) -> None:
        """When llmapi is not importable, perf config is omitted."""
        mock_trtllm = types.ModuleType("tensorrt_llm")
        mock_trtllm.SamplingParams = MockTRTSamplingParams  # type: ignore[attr-defined]
        mock_trtllm.LLM = MagicMock  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {"tensorrt_llm": mock_trtllm, "tensorrt_llm.llmapi": None},
        ):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(engine_params=[("cuda_graphs", "true")])
            transport = TRTLLMTransport(model_endpoint=endpoint)
            kwargs = transport._build_engine_kwargs()
            assert "extended_runtime_perf_knob_config" not in kwargs


# ============================================================
# _build_env_overrides (Merge Behavior)
# ============================================================


class TestBuildEnvOverridesMerge:
    """Verify env_overrides merge behavior between preset and user values."""

    def test_no_preset_no_user_omits_env_overrides(self) -> None:
        kwargs, _ = _build_latency_kwargs([])
        assert "env_overrides" not in kwargs

    def test_preset_applies_all_latency_env_vars(self) -> None:
        kwargs, _ = _build_latency_kwargs([("latency_optimized", "true")])
        env = kwargs["env_overrides"]
        assert env == {
            "TRTLLM_ENABLE_PDL": "1",
            "FORCE_MULTI_BLOCK_MODE": "1",
            "TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG": "1",
            "TRTLLM_MMHA_KERNEL_BLOCK_SIZE": "256",
        }

    def test_user_overrides_merge_on_top_of_preset(self) -> None:
        """User env_overrides dict is merged on top of preset defaults."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            params: dict[str, Any] = {
                "env_overrides": {"MY_CUSTOM_VAR": "custom", "TRTLLM_ENABLE_PDL": "0"},
            }
            result = transport._build_env_overrides(params, latency_optimized=True)

            assert result is not None
            assert result["TRTLLM_ENABLE_PDL"] == "0"
            assert result["MY_CUSTOM_VAR"] == "custom"
            assert result["FORCE_MULTI_BLOCK_MODE"] == "1"

    def test_user_overrides_alone_no_preset(self) -> None:
        """env_overrides without preset returns only user values."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            params: dict[str, Any] = {"env_overrides": {"MY_VAR": "value"}}
            result = transport._build_env_overrides(params, latency_optimized=False)

            assert result == {"MY_VAR": "value"}

    def test_non_dict_env_overrides_ignored(self) -> None:
        """Non-dict env_overrides value is popped but not merged."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            params: dict[str, Any] = {"env_overrides": "not-a-dict"}
            result = transport._build_env_overrides(params, latency_optimized=False)

            assert result is None

    def test_empty_dict_env_overrides_with_preset(self) -> None:
        """Empty user dict with preset still returns preset overrides."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            params: dict[str, Any] = {"env_overrides": {}}
            result = transport._build_env_overrides(params, latency_optimized=True)

            assert result is not None
            assert len(result) == 4
            assert result["TRTLLM_ENABLE_PDL"] == "1"


# ============================================================
# Latency Preset Partial Override Integration
# ============================================================


class TestLatencyPresetPartialOverrideIntegration:
    """Verify partial overrides: some knobs user-set, others from preset."""

    def test_user_batch_size_preset_fills_rest(self) -> None:
        """User sets max_batch_size=4, preset still fills chunked_prefill, scheduler, etc."""
        kwargs, transport = _build_latency_kwargs(
            [
                ("latency_optimized", "true"),
                ("max_batch_size", "4"),
            ]
        )
        assert kwargs["max_batch_size"] == 4
        assert kwargs["enable_chunked_prefill"] is False
        assert "scheduler_config" in kwargs
        assert "kv_cache_config" in kwargs
        assert "extended_runtime_perf_knob_config" in kwargs
        assert "env_overrides" in kwargs
        assert transport._warmup_iterations == 2

    def test_user_warmup_overrides_preset_default(self) -> None:
        """User warmup_iterations overrides the preset default of 2."""
        _, transport = _build_latency_kwargs(
            [
                ("latency_optimized", "true"),
                ("warmup_iterations", "10"),
            ]
        )
        assert transport._warmup_iterations == 10

    def test_preset_warmup_default_when_user_sets_zero(self) -> None:
        """warmup_iterations=0 is treated as not-set; preset overrides to 2."""
        _, transport = _build_latency_kwargs(
            [
                ("latency_optimized", "true"),
                ("warmup_iterations", "0"),
            ]
        )
        assert transport._warmup_iterations == 2

    def test_preset_warmup_default_when_omitted(self) -> None:
        """Without warmup_iterations, preset sets _warmup_iterations=2."""
        _, transport = _build_latency_kwargs([("latency_optimized", "true")])
        assert transport._warmup_iterations == 2


# ============================================================
# Unrecognized Params Passthrough
# ============================================================


class TestUnrecognizedParamsPassthrough:
    """Verify unknown engine params survive even with latency preset active."""

    def test_unknown_params_preserved_with_preset(self) -> None:
        kwargs, _ = _build_latency_kwargs(
            [
                ("latency_optimized", "true"),
                ("custom_knob", "42"),
                ("experimental_flag", "on"),
            ]
        )
        assert kwargs["custom_knob"] == "42"
        assert kwargs["experimental_flag"] == "on"
        assert kwargs["max_batch_size"] == 1

    def test_unknown_params_preserved_without_preset(self) -> None:
        kwargs, _ = _build_latency_kwargs(
            [
                ("my_special_param", "hello"),
                ("another_param", "world"),
            ]
        )
        assert kwargs["my_special_param"] == "hello"
        assert kwargs["another_param"] == "world"
        assert "scheduler_config" not in kwargs


# ============================================================
# Log Message Verification
# ============================================================


class TestLatencyOptimizedLogging:
    """Verify log messages when latency_optimized preset is active."""

    def test_latency_preset_logs_info_with_key_settings(self) -> None:
        """Activating latency_optimized emits an info log with batch size and warmup."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(
                engine_params=[("latency_optimized", "true")]
            )
            transport = TRTLLMTransport(model_endpoint=endpoint)

            logged: list[str] = []

            def capture_info(msg: Any) -> None:
                logged.append(msg() if callable(msg) else str(msg))

            transport.info = capture_info  # type: ignore[assignment]
            transport._build_engine_kwargs()

            latency_logs = [m for m in logged if "Latency-optimized" in m]
            assert len(latency_logs) == 1
            assert "max_batch_size=1" in latency_logs[0]
            assert "warmup=2" in latency_logs[0]

    def test_no_latency_log_when_preset_disabled(self) -> None:
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(engine_params=[])
            transport = TRTLLMTransport(model_endpoint=endpoint)

            logged: list[str] = []

            def capture_info(msg: Any) -> None:
                logged.append(msg() if callable(msg) else str(msg))

            transport.info = capture_info  # type: ignore[assignment]
            transport._build_engine_kwargs()

            assert not any("Latency-optimized" in m for m in logged)

    def test_latency_log_contains_chunked_prefill_status(self) -> None:
        """Log message includes enable_chunked_prefill value."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = _make_trtllm_endpoint(
                engine_params=[("latency_optimized", "true")]
            )
            transport = TRTLLMTransport(model_endpoint=endpoint)

            logged: list[str] = []

            def capture_info(msg: Any) -> None:
                logged.append(msg() if callable(msg) else str(msg))

            transport.info = capture_info  # type: ignore[assignment]
            transport._build_engine_kwargs()

            latency_logs = [m for m in logged if "Latency-optimized" in m]
            assert "chunked_prefill=False" in latency_logs[0]


# ============================================================
# enable_chunked_prefill Bool Coercion
# ============================================================


class TestEnableChunkedPrefillCoercion:
    """Verify enable_chunked_prefill is parsed as bool from string."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
        ],
    )  # fmt: skip
    def test_enable_chunked_prefill_values(
        self, input_val: str, expected: bool
    ) -> None:
        kwargs, _ = _build_latency_kwargs([("enable_chunked_prefill", input_val)])
        assert kwargs["enable_chunked_prefill"] is expected


# ============================================================
# pipeline_parallel_size Coercion
# ============================================================


class TestPipelineParallelSizeCoercion:
    """Verify pipeline_parallel_size is coerced to int."""

    def test_pipeline_parallel_size_string_to_int(self) -> None:
        kwargs, _ = _build_latency_kwargs([("pipeline_parallel_size", "2")])
        assert kwargs["pipeline_parallel_size"] == 2
        assert isinstance(kwargs["pipeline_parallel_size"], int)


# ============================================================
# _get_tokenizer
# ============================================================


class TestGetTokenizerBehavior:
    """Verify tokenizer extraction from engine."""

    def test_returns_engine_tokenizer_when_present(self) -> None:
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            mock_tokenizer = MagicMock()
            mock_engine = MagicMock()
            mock_engine.tokenizer = mock_tokenizer
            transport._engine = mock_engine

            assert transport._get_tokenizer() is mock_tokenizer

    def test_returns_none_when_no_engine(self) -> None:
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            assert transport._get_tokenizer() is None

    def test_returns_none_when_engine_has_no_tokenizer_attr(self) -> None:
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            mock_engine = MagicMock(spec=[])
            transport._engine = mock_engine

            assert transport._get_tokenizer() is None


# ============================================================
# _reset_engine_stats (WAR for warmup telemetry leakage)
# ============================================================


class TestResetEngineStats:
    """Verify _reset_engine_stats clears _iter_stats_result after warmup."""

    def test_resets_iter_stats_result_when_present(self) -> None:
        """_iter_stats_result is set to None when executor exposes it."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            mock_executor = MagicMock()
            mock_executor._iter_stats_result = {"some": "warmup_data"}
            mock_engine = MagicMock()
            mock_engine._executor = mock_executor
            transport._engine = mock_engine

            transport._reset_engine_stats()

            assert mock_executor._iter_stats_result is None

    def test_noop_when_engine_is_none(self) -> None:
        """No error when engine has not been initialized."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            transport._engine = None

            transport._reset_engine_stats()  # Should not raise

    def test_noop_when_engine_has_no_executor(self) -> None:
        """No error when engine lacks _executor attribute."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            transport._engine = MagicMock(spec=[])  # no _executor

            transport._reset_engine_stats()  # Should not raise

    def test_noop_when_executor_has_no_iter_stats(self) -> None:
        """No error when executor lacks _iter_stats_result attribute."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            mock_executor = MagicMock(spec=[])  # no _iter_stats_result
            mock_engine = MagicMock()
            mock_engine._executor = mock_executor
            transport._engine = mock_engine

            transport._reset_engine_stats()  # Should not raise

    @pytest.mark.asyncio
    async def test_start_engine_calls_reset_after_warmup(self) -> None:
        """_start_engine calls _reset_engine_stats after _run_warmup completes."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            transport._engine_kwargs = {}
            transport._model_path = "test-model"

            call_order: list[str] = []

            mock_engine = MagicMock()
            original_reset = transport._reset_engine_stats

            async def mock_run_warmup() -> None:
                call_order.append("warmup")

            def mock_reset() -> None:
                call_order.append("reset")
                original_reset()

            transport._run_warmup = mock_run_warmup  # type: ignore[assignment]
            transport._reset_engine_stats = mock_reset  # type: ignore[assignment]

            with patch(
                "aiperf.transports.in_engine.trtllm_transport.asyncio"
            ) as mock_asyncio:
                mock_loop = MagicMock()
                mock_loop.run_in_executor = AsyncMock(return_value=mock_engine)
                mock_asyncio.get_event_loop.return_value = mock_loop

                await transport._start_engine()

            assert call_order == ["warmup", "reset"]

    def test_reset_emits_debug_log(self) -> None:
        """_reset_engine_stats logs a debug message when it resets."""
        mods = _build_mock_trtllm_modules()
        with patch.dict(sys.modules, mods):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            transport = TRTLLMTransport(model_endpoint=_make_trtllm_endpoint())
            mock_executor = MagicMock()
            mock_executor._iter_stats_result = {"data": "warmup"}
            mock_engine = MagicMock()
            mock_engine._executor = mock_executor
            transport._engine = mock_engine

            logged: list[str] = []

            def capture_debug(msg: Any) -> None:
                logged.append(msg() if callable(msg) else str(msg))

            transport.debug = capture_debug  # type: ignore[assignment]
            transport._reset_engine_stats()

            assert any("_iter_stats_result" in m for m in logged)
