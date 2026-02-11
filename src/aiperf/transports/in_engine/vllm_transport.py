# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import FirstTokenCallback
from aiperf.transports.in_engine.base_in_engine_transport import (
    BaseInEngineTransport,
)

_INT_ENGINE_PARAMS: dict[str, str] = {
    "tensor_parallel_size": "tensor_parallel_size",
    "pipeline_parallel_size": "pipeline_parallel_size",
    "seed": "seed",
    "max_model_len": "max_model_len",
}
_FLOAT_ENGINE_PARAMS: dict[str, str] = {
    "gpu_memory_utilization": "gpu_memory_utilization",
}
_STR_ENGINE_PARAMS: dict[str, str] = {
    "dtype": "dtype",
    "quantization": "quantization",
}
_PASSTHROUGH_ENGINE_PARAMS: set[str] = {
    "enforce_eager",
    "trust_remote_code",
}

# Speculative decoding params → speculative_config dict
_SPECULATIVE_INT_PARAMS: set[str] = {
    "num_speculative_tokens",
    "prompt_lookup_max",
    "prompt_lookup_min",
    "speculative_draft_tensor_parallel_size",
    "speculative_disable_by_batch_size",
}
_SPECULATIVE_STR_PARAMS: set[str] = {
    "speculative_method",
    "speculative_model",
    "speculative_quantization",
}
_SPECULATIVE_BOOL_PARAMS: set[str] = {
    "speculative_disable_padded_drafter_batch",
    "speculative_parallel_drafting",
}


class VLLMTransport(BaseInEngineTransport):
    """vLLM in-engine transport using AsyncLLMEngine for concurrent inference.

    Uses vLLM's async engine to run inference directly without an HTTP server.
    The engine handles continuous batching, PagedAttention, and CUDA graph
    capture internally. AsyncLLMEngine supports concurrent `generate()` calls
    natively, enabling concurrency > 1 without threading issues.

    Engine lifecycle:
        - `_init_engine`: Validates vLLM is installed, parses engine kwargs
        - `_start_engine`: Creates `AsyncLLMEngine` + warmup probe
        - `_stop_engine`: Calls `engine.shutdown()` to free GPU memory
    """

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return vLLM transport metadata for plugin discovery."""
        return TransportMetadata(
            transport_type="vllm",
            url_schemes=["vllm"],
        )

    async def _init_engine(self) -> None:
        """Validate vLLM is installed and parse engine configuration."""
        try:
            import vllm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "vLLM is required for the vllm:// transport. Install with: uv add vllm"
            ) from e

        self._engine_kwargs = self._build_engine_kwargs()
        self.debug(lambda: f"vLLM engine kwargs: {self._engine_kwargs}")

    async def _start_engine(self) -> None:
        """Create AsyncLLMEngine and run a warmup probe.

        `AsyncLLMEngine.from_engine_args()` spawns an engine core subprocess
        that loads model weights and allocates KV cache. A single-token warmup
        probe ensures the engine is fully ready before we return.
        """
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
        from vllm.sampling_params import RequestOutputKind

        engine_args = AsyncEngineArgs(
            model=self._model_path,
            **self._engine_kwargs,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Warmup probe — blocks until the engine core has loaded the model,
        # allocated KV cache, and captured CUDA graphs.
        self.info("Running engine warmup probe...")
        warmup_params = SamplingParams(
            max_tokens=1,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        async for _ in self._engine.generate(
            prompt="warmup",
            sampling_params=warmup_params,
            request_id="warmup-probe",
        ):
            pass

    async def _stop_engine(self) -> None:
        """Shutdown vLLM async engine and free GPU memory."""
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None

    async def _generate(
        self,
        *,
        messages: list[dict[str, Any]],
        sampling_params: Any,
        request_id: str,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> tuple[str, int, int, str]:
        """Generate a response using vLLM's async engine.

        Converts messages to a prompt string via the engine's chat template,
        then iterates over the async generator from `engine.generate()`.
        Uses `FINAL_ONLY` output mode for efficiency (single yield).

        Args:
            messages: Chat messages in OpenAI format
            sampling_params: `vllm.SamplingParams` instance (built by endpoint)
            request_id: Unique request identifier
            first_token_callback: Optional TTFT callback (not yet supported)

        Returns:
            Tuple of (generated_text, input_token_count, output_token_count, finish_reason)
        """
        from vllm.sampling_params import RequestOutputKind

        prompt = self._messages_to_prompt(messages)
        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY

        final_output = None
        async for output in self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            final_output = output

        if final_output is None:
            raise RuntimeError(f"vLLM returned no output for request {request_id}")

        completion = final_output.outputs[0]
        prompt_token_ids = final_output.prompt_token_ids or []
        return (
            completion.text,
            len(prompt_token_ids),
            len(completion.token_ids),
            completion.finish_reason or "stop",
        )

    def _get_tokenizer(self) -> Any | None:
        """Return vLLM's tokenizer from the async engine."""
        if self._engine is not None:
            return self._engine.get_tokenizer()
        return None

    # ─── Private Helpers ─────────────────────────────────────

    def _build_engine_kwargs(self) -> dict[str, Any]:
        """Build vLLM LLM constructor kwargs from `--engine-params`.

        Engine-specific parameters are passed via `endpoint.engine_params`
        (a `list[tuple[str, Any]]`). Known parameters are coerced to the
        correct types; unknown parameters are passed through unchanged.

        Returns:
            Dict of kwargs suitable for `vllm.LLM(model=..., **kwargs)`
        """
        params = self._get_raw_engine_params()
        kwargs: dict[str, Any] = {}

        for src_key, dst_key in _INT_ENGINE_PARAMS.items():
            if src_key in params:
                kwargs[dst_key] = int(params.pop(src_key))

        for src_key, dst_key in _FLOAT_ENGINE_PARAMS.items():
            if src_key in params:
                kwargs[dst_key] = float(params.pop(src_key))

        for src_key, dst_key in _STR_ENGINE_PARAMS.items():
            if src_key in params:
                kwargs[dst_key] = params.pop(src_key)

        for key in _PASSTHROUGH_ENGINE_PARAMS:
            if key in params:
                kwargs[key] = params.pop(key)

        spec_config = self._build_speculative_config(params)
        if spec_config is not None:
            kwargs["speculative_config"] = spec_config

        kwargs.update(params)
        return kwargs

    def _build_speculative_config(
        self, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract speculative decoding params into a `speculative_config` dict.

        Pops all speculative-related keys from *params* (mutating it), strips
        the `speculative_` prefix where present, coerces types, and returns
        the config dict.  Returns `None` when no speculative params are found.

        Args:
            params: Mutable dict of remaining engine params (keys are consumed).

        Returns:
            Dict suitable for `AsyncEngineArgs(speculative_config=...)` or None.
        """
        _KNOWN = (
            _SPECULATIVE_INT_PARAMS | _SPECULATIVE_STR_PARAMS | _SPECULATIVE_BOOL_PARAMS
        )
        config: dict[str, Any] = {}

        for key in list(params):
            if key not in _KNOWN and not key.startswith("speculative_"):
                continue

            cfg_key = key.removeprefix("speculative_")
            value = params.pop(key)

            if key in _SPECULATIVE_INT_PARAMS:
                config[cfg_key] = int(value)
            elif key in _SPECULATIVE_BOOL_PARAMS:
                config[cfg_key] = (
                    value
                    if isinstance(value, bool)
                    else str(value).lower() in ("true", "1", "yes")
                )
            else:
                config[cfg_key] = value

        return config if config else None
