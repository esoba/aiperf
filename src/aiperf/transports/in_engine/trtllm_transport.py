# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT-LLM in-engine transport for offline inference benchmarking.

Uses TRT-LLM's Python LLM API with the PyTorch backend (default since v1.0)
for direct inference without an HTTP server. Supports inflight batching and
MAX_UTILIZATION scheduling for maximum throughput.

IMPORTANT: TRT-LLM uses mpi4py for multi-GPU workflows. The main entry point
of the application must be protected under `if __name__ == '__main__':` to
avoid issues with MPI process spawning.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import FirstTokenCallback
from aiperf.transports.in_engine.base_in_engine_transport import (
    BaseInEngineTransport,
)

# Environment variable overrides applied when latency_optimized=True.
# Matches trtllm-bench's low_latency preset for optimal single-request perf.
_LATENCY_ENV_OVERRIDES: dict[str, str] = {
    "TRTLLM_ENABLE_PDL": "1",
    "FORCE_MULTI_BLOCK_MODE": "1",
    "TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG": "1",
    "TRTLLM_MMHA_KERNEL_BLOCK_SIZE": "256",
}


class TRTLLMTransport(BaseInEngineTransport):
    """TensorRT-LLM in-engine transport for offline inference benchmarking.

    Uses TRT-LLM's Python LLM API with the PyTorch backend for direct inference.
    Supports inflight batching and MAX_UTILIZATION scheduler for maximum throughput.

    IMPORTANT: TRT-LLM requires the main entry point to be protected under
    `if __name__ == '__main__':` due to mpi4py dependency for multi-GPU workflows.
    """

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return TRT-LLM transport metadata for discovery and registration."""
        return TransportMetadata(
            transport_type="trtllm",
            url_schemes=["trtllm"],
        )

    async def _init_engine(self) -> None:
        """Validate TensorRT-LLM is installed and parse engine configuration.

        Checks for the `tensorrt_llm` package and extracts engine-specific
        kwargs from the endpoint extra configuration.
        """
        try:
            import tensorrt_llm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TensorRT-LLM is required for the trtllm:// transport. "
                "Install with: pip install tensorrt-llm"
            ) from e

        self._engine_kwargs = self._build_engine_kwargs()
        self.debug(lambda: f"TRT-LLM engine kwargs: {self._engine_kwargs}")

    async def _start_engine(self) -> None:
        """Load model via TRT-LLM LLM class.

        With the PyTorch backend (default v1.0+), this loads the model
        directly without a separate engine-build step. Engine creation is
        blocking so it runs in an executor to keep the event loop responsive
        for heartbeats and ZMQ message processing during model load.
        """
        from tensorrt_llm import LLM

        model_path = self._model_path
        engine_kwargs = self._engine_kwargs

        loop = asyncio.get_event_loop()
        self._engine = await loop.run_in_executor(
            None,
            lambda: LLM(model=model_path, **engine_kwargs),
        )

        await self._run_warmup()
        self._reset_engine_stats()

    async def _stop_engine(self) -> None:
        """Shutdown TRT-LLM engine and free GPU resources.

        Calls `engine.shutdown()` if available, then removes the reference
        to allow garbage collection. Safe to call even if the engine was never
        initialized.
        """
        if self._engine is not None:
            if hasattr(self._engine, "shutdown"):
                self._engine.shutdown()
            del self._engine
            self._engine = None

    def _reset_engine_stats(self) -> None:
        """Reset iteration stats after warmup to prevent data leakage.

        WAR: TRT-LLM's executor binds ``_iter_stats_result`` to the event loop
        during generation. Warmup populates this with warmup-only data that
        would otherwise leak into production telemetry. Resetting it mirrors
        trtllm-bench's approach (throughput.py:450-453).

        Safe to call even when the engine or executor lacks the attribute.
        """
        if (
            self._engine is not None
            and hasattr(self._engine, "_executor")
            and hasattr(self._engine._executor, "_iter_stats_result")
        ):
            self._engine._executor._iter_stats_result = None
            self.debug(lambda: "Reset _iter_stats_result after warmup")

    async def _warmup_single(
        self, prompt: str, max_tokens: int, *, streaming: bool
    ) -> None:
        """Run a single TRT-LLM warmup call matching the endpoint's streaming mode.

        Args:
            prompt: Warmup prompt text.
            max_tokens: Maximum tokens to generate.
            streaming: Use streaming iteration when True, aresult() when False.
        """
        from tensorrt_llm import SamplingParams as TRTSamplingParams

        params = TRTSamplingParams(max_tokens=max_tokens)

        if streaming:
            async for _ in self._engine.generate_async(prompt, params, streaming=True):
                pass
        else:
            output = self._engine.generate_async(prompt, params, streaming=False)
            await output.aresult()

    async def _generate(
        self,
        *,
        messages: list[dict[str, Any]],
        sampling_params: Any,
        request_id: str,
        first_token_callback: FirstTokenCallback | None = None,
        input_ids: list[int] | None = None,
    ) -> tuple[str, int, int, str]:
        """Generate text using TRT-LLM.

        Dispatches to streaming or non-streaming path based on endpoint config.

        Args:
            messages: Chat messages in OpenAI format.
            sampling_params: Dict of sampling params (built by endpoint).
            request_id: Unique request identifier.
            first_token_callback: Optional TTFT callback for streaming.
            input_ids: Pre-tokenized input IDs (bypasses chat template when provided).

        Returns:
            Tuple of (generated_text, input_token_count, output_token_count, finish_reason).
        """
        # Pre-tokenized path: pass token IDs directly (TRT-LLM accepts list[int])
        prompt: str | list[int]
        if input_ids is not None:
            prompt = input_ids
        else:
            prompt = self._messages_to_prompt(messages)
        streaming = self.model_endpoint.endpoint.streaming

        if streaming:
            return await self._generate_streaming(prompt, sampling_params, request_id)

        return await self._generate_final_only(prompt, sampling_params, request_id)

    async def _generate_final_only(
        self,
        prompt: str | list[int],
        sampling_params: Any,
        request_id: str,
    ) -> tuple[str, int, int, str]:
        """Non-streaming generation using generate_async + aresult().

        generate_async() returns a RequestOutput handle synchronously.
        aresult() is a coroutine that awaits the final result without
        blocking the event loop (no executor needed).
        """
        from tensorrt_llm import SamplingParams as TRTSamplingParams

        params = TRTSamplingParams(**sampling_params)

        output = self._engine.generate_async(prompt, params, streaming=False)

        # prompt_token_ids lives on the handle, not the awaited response
        input_tokens = 0
        if hasattr(output, "prompt_token_ids") and output.prompt_token_ids:
            input_tokens = len(output.prompt_token_ids)

        response = await output.aresult()
        completion = response.outputs[0]
        text = completion.text

        # Re-check prompt_token_ids on response if not found on handle
        if (
            input_tokens == 0
            and hasattr(response, "prompt_token_ids")
            and response.prompt_token_ids
        ):
            input_tokens = len(response.prompt_token_ids)

        output_tokens = 0
        output_token_ids: list[int] = []
        if hasattr(completion, "token_ids") and completion.token_ids:
            output_token_ids = list(completion.token_ids)
            output_tokens = len(output_token_ids)

        raw_finish = getattr(completion, "finish_reason", None)
        finish_reason = str(raw_finish) if raw_finish is not None else "stop"

        # Capture speculative decoding metadata
        self._decode_iterations = getattr(response, "decoding_iter", None)

        if self._preserve_token_ids:
            self._output_token_ids = output_token_ids

        return text, input_tokens, output_tokens, finish_reason

    async def _generate_streaming(
        self,
        prompt: str | list[int],
        sampling_params: Any,
        request_id: str,
    ) -> tuple[str, int, int, str]:
        """Streaming generation using generate_async for TTFT capture.

        TRT-LLM yields cumulative text (not deltas), so we track prev_text_len
        to detect new content for first-token timing.
        """
        from tensorrt_llm import SamplingParams as TRTSamplingParams

        params = TRTSamplingParams(**sampling_params)

        prev_text_len = 0
        input_tokens = 0
        output_tokens = 0
        finish_reason = "stop"
        final_text = ""
        is_first_token = True
        latest_token_ids: list[int] = []

        async for output in self._engine.generate_async(prompt, params, streaming=True):
            if not output.outputs:
                continue

            completion = output.outputs[0]

            # Detect new content by comparing cumulative text length
            current_text = completion.text
            if is_first_token and len(current_text) > prev_text_len:
                self._first_token_perf_ns = time.perf_counter_ns()
                is_first_token = False

            prev_text_len = len(current_text)
            final_text = current_text

            # Prompt tokens (available from first chunk)
            if hasattr(output, "prompt_token_ids") and output.prompt_token_ids:
                input_tokens = len(output.prompt_token_ids)

            # Output tokens (cumulative, take latest)
            if hasattr(completion, "token_ids") and completion.token_ids:
                output_tokens = len(completion.token_ids)
                latest_token_ids = list(completion.token_ids)

            # Finish reason: None until last chunk
            raw_finish = getattr(completion, "finish_reason", None)
            if raw_finish is not None:
                finish_reason = str(raw_finish)

        if is_first_token:
            raise RuntimeError(f"TRT-LLM returned no output for request {request_id}")

        # Capture speculative decoding metadata from the last output chunk
        self._decode_iterations = getattr(output, "decoding_iter", None)

        if self._preserve_token_ids:
            self._output_token_ids = latest_token_ids or None

        return final_text, input_tokens, output_tokens, finish_reason

    async def _get_engine_stats(self) -> dict[str, Any] | None:
        """Poll TRT-LLM iteration stats via ``get_stats_async``.

        Returns the first stats dict yielded by the async generator, or None
        if the engine does not expose the stats API.
        """
        if self._engine is not None and hasattr(self._engine, "get_stats_async"):
            async for stats in self._engine.get_stats_async(timeout=1):
                return stats
        return None

    def _get_tokenizer(self) -> Any | None:
        """Return TRT-LLM's tokenizer from the engine."""
        if self._engine is not None and hasattr(self._engine, "tokenizer"):
            return self._engine.tokenizer
        return None

    def _build_engine_kwargs(self) -> dict[str, Any]:
        """Build TRT-LLM LLM constructor kwargs from ``--engine-params``.

        Extracts known engine parameters, builds structured TRT-LLM config
        objects (SchedulerConfig, KvCacheConfig, etc.), and applies the
        ``latency_optimized`` preset when requested. User-provided values
        always override preset defaults.

        Returns:
            Dictionary of kwargs for the ``tensorrt_llm.LLM`` constructor.
        """
        params = self._get_raw_engine_params()
        self._pop_warmup_config(params)

        # Extract max_draft_len for speculative decoding metrics
        if "max_draft_len" in params:
            self._max_draft_len = int(params.pop("max_draft_len"))

        latency_optimized = self._parse_bool(params.pop("latency_optimized", "false"))

        kwargs: dict[str, Any] = {}

        # Scalar params with type coercion
        if "tensor_parallel_size" in params:
            kwargs["tensor_parallel_size"] = int(params.pop("tensor_parallel_size"))
        if "pipeline_parallel_size" in params:
            kwargs["pipeline_parallel_size"] = int(params.pop("pipeline_parallel_size"))
        if "dtype" in params:
            kwargs["dtype"] = params.pop("dtype")
        if "max_seq_len" in params:
            kwargs["max_seq_len"] = int(params.pop("max_seq_len"))
        if "backend" in params:
            kwargs["backend"] = params.pop("backend")

        # Latency preset defaults (user overrides take priority)
        if latency_optimized:
            kwargs.setdefault("max_batch_size", 1)
            kwargs.setdefault("enable_chunked_prefill", False)
        if "max_batch_size" in params:
            kwargs["max_batch_size"] = int(params.pop("max_batch_size"))
        if "enable_chunked_prefill" in params:
            kwargs["enable_chunked_prefill"] = self._parse_bool(
                params.pop("enable_chunked_prefill")
            )

        # Structured configs
        scheduler_config = self._build_scheduler_config(params, latency_optimized)
        if scheduler_config is not None:
            kwargs["scheduler_config"] = scheduler_config

        kv_cache_config = self._build_kv_cache_config(params, latency_optimized)
        if kv_cache_config is not None:
            kwargs["kv_cache_config"] = kv_cache_config

        perf_config = self._build_perf_knob_config(params, latency_optimized)
        if perf_config is not None:
            kwargs["extended_runtime_perf_knob_config"] = perf_config

        env_overrides = self._build_env_overrides(params, latency_optimized)
        if env_overrides is not None:
            kwargs["env_overrides"] = env_overrides

        # Warmup default for latency preset
        if latency_optimized and self._warmup_iterations == 0:
            self._warmup_iterations = 2

        # Pass through remaining unknown params
        kwargs.update(params)

        if latency_optimized:
            self.info(
                lambda: (
                    f"Latency-optimized preset active: max_batch_size={kwargs.get('max_batch_size')}, "
                    f"chunked_prefill={kwargs.get('enable_chunked_prefill')}, "
                    f"warmup={self._warmup_iterations}"
                )
            )

        return kwargs

    def _build_scheduler_config(
        self, params: dict[str, Any], latency_optimized: bool
    ) -> Any | None:
        """Build a SchedulerConfig from engine params.

        Args:
            params: Mutable engine params dict (consumed keys are popped).
            latency_optimized: Whether the latency preset is active.

        Returns:
            SchedulerConfig instance, or None if no scheduler params present.
        """
        policy = params.pop("scheduler_policy", None)
        if policy is None and not latency_optimized:
            return None

        try:
            from tensorrt_llm.llmapi import CapacitySchedulerPolicy, SchedulerConfig
        except ImportError:
            return None

        if policy is not None:
            resolved = getattr(CapacitySchedulerPolicy, policy, policy)
        else:
            resolved = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT

        return SchedulerConfig(capacity_scheduler_policy=resolved)

    def _build_kv_cache_config(
        self, params: dict[str, Any], latency_optimized: bool
    ) -> Any | None:
        """Build a KvCacheConfig from engine params.

        Args:
            params: Mutable engine params dict (consumed keys are popped).
            latency_optimized: Whether the latency preset is active.

        Returns:
            KvCacheConfig instance, or None if no kv-cache params present.
        """
        kv_pct = params.pop("kv_cache_free_gpu_mem_fraction", None)
        if kv_pct is None and not latency_optimized:
            return None

        try:
            from tensorrt_llm.llmapi import KvCacheConfig
        except ImportError:
            return None

        fraction = float(kv_pct) if kv_pct is not None else 0.90
        return KvCacheConfig(free_gpu_memory_fraction=fraction)

    def _build_perf_knob_config(
        self, params: dict[str, Any], latency_optimized: bool
    ) -> Any | None:
        """Build an ExtendedRuntimePerfKnobConfig from engine params.

        Args:
            params: Mutable engine params dict (consumed keys are popped).
            latency_optimized: Whether the latency preset is active.

        Returns:
            ExtendedRuntimePerfKnobConfig instance, or None if no perf params present.
        """
        cuda_graphs = params.pop("cuda_graphs", None)
        multi_block = params.pop("multi_block_mode", None)

        has_user_values = cuda_graphs is not None or multi_block is not None
        if not has_user_values and not latency_optimized:
            return None

        try:
            from tensorrt_llm.llmapi import ExtendedRuntimePerfKnobConfig
        except ImportError:
            return None

        config = ExtendedRuntimePerfKnobConfig()

        if cuda_graphs is not None:
            config.cuda_graph_mode = self._parse_bool(cuda_graphs)
        elif latency_optimized:
            config.cuda_graph_mode = True

        if multi_block is not None:
            config.multi_block_mode = self._parse_bool(multi_block)
        elif latency_optimized:
            config.multi_block_mode = True

        return config

    def _build_env_overrides(
        self, params: dict[str, Any], latency_optimized: bool
    ) -> dict[str, str] | None:
        """Build environment variable overrides for the LLM constructor.

        When ``latency_optimized`` is active, applies ``_LATENCY_ENV_OVERRIDES``
        as defaults. Any user-provided ``env_overrides`` take priority.

        Args:
            params: Mutable engine params dict (consumed keys are popped).
            latency_optimized: Whether the latency preset is active.

        Returns:
            Dict of env overrides, or None if none needed.
        """
        user_overrides = params.pop("env_overrides", None)
        if not latency_optimized and user_overrides is None:
            return None

        result: dict[str, str] = {}
        if latency_optimized:
            result.update(_LATENCY_ENV_OVERRIDES)
        if user_overrides is not None and isinstance(user_overrides, dict):
            result.update(user_overrides)
        return result or None

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        """Coerce a string or bool value to bool.

        Engine params arrive as strings from CLI (e.g. ``"true"``, ``"1"``).

        Args:
            value: String, bool, or int to coerce.

        Returns:
            Boolean interpretation of *value*.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
