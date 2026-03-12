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
from typing import Any

from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import FirstTokenCallback
from aiperf.transports.in_engine.base_in_engine_transport import (
    BaseInEngineTransport,
)


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
        blocking so it is run in an executor to avoid blocking the event loop.

        IMPORTANT: TRT-LLM requires the main entry point to be protected
        under `if __name__ == '__main__':` due to mpi4py dependency for
        multi-GPU workflows.
        """
        from tensorrt_llm import LLM

        model_path = self._model_path
        engine_kwargs = self._engine_kwargs

        loop = asyncio.get_event_loop()
        self._engine = await loop.run_in_executor(
            None,
            lambda: LLM(model=model_path, **engine_kwargs),
        )

        if self._warmup_iterations > 0:
            from tensorrt_llm import SamplingParams as TRTSamplingParams

            self.info(f"Running {self._warmup_iterations} warmup iterations...")
            warmup_params = TRTSamplingParams(max_tokens=1)
            engine = self._engine
            for _ in range(self._warmup_iterations):
                await loop.run_in_executor(
                    None,
                    lambda: engine.generate(["warmup"], warmup_params),
                )

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

    async def _generate(
        self,
        *,
        messages: list[dict[str, Any]],
        sampling_params: Any,
        request_id: str,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> tuple[str, int, int, str]:
        """Generate text using TRT-LLM's synchronous generate() method.

        Converts messages to a prompt string via the engine's chat template,
        then calls the blocking generate() in run_in_executor.

        Args:
            messages: Chat messages in OpenAI format.
            sampling_params: `tensorrt_llm.SamplingParams` instance (built by endpoint).
            request_id: Unique request identifier.
            first_token_callback: Optional TTFT callback (unused in non-streaming).

        Returns:
            Tuple of (generated_text, input_token_count, output_token_count, finish_reason).
        """
        from tensorrt_llm import SamplingParams as TRTSamplingParams

        prompt = self._messages_to_prompt(messages)
        sampling_params = TRTSamplingParams(**sampling_params)

        loop = asyncio.get_event_loop()
        engine = self._engine
        outputs = await loop.run_in_executor(
            None,
            lambda: engine.generate([prompt], sampling_params),
        )

        output = outputs[0]
        completion = output.outputs[0]
        text = completion.text

        # Token counting: guard with hasattr for robustness across TRT-LLM versions
        input_tokens = 0
        if hasattr(output, "prompt_token_ids") and output.prompt_token_ids:
            input_tokens = len(output.prompt_token_ids)

        output_tokens = 0
        if hasattr(completion, "token_ids") and completion.token_ids:
            output_tokens = len(completion.token_ids)
        raw_finish = getattr(completion, "finish_reason", None)
        finish_reason = str(raw_finish) if raw_finish is not None else "stop"

        return text, input_tokens, output_tokens, finish_reason

    def _get_tokenizer(self) -> Any | None:
        """Return TRT-LLM's tokenizer from the engine."""
        if self._engine is not None and hasattr(self._engine, "tokenizer"):
            return self._engine.tokenizer
        return None

    def _build_engine_kwargs(self) -> dict[str, Any]:
        """Build TRT-LLM LLM constructor kwargs from `--engine-params`.

        Extracts known engine parameters from `endpoint.engine_params`
        and converts them to appropriate types. Unknown params are passed
        through directly to the LLM constructor.

        Returns:
            Dictionary of kwargs for the `tensorrt_llm.LLM` constructor.
        """
        params = self._get_raw_engine_params()
        self._pop_warmup_iterations(params)
        kwargs: dict[str, Any] = {}

        if "tensor_parallel_size" in params:
            kwargs["tensor_parallel_size"] = int(params.pop("tensor_parallel_size"))
        if "pipeline_parallel_size" in params:
            kwargs["pipeline_parallel_size"] = int(params.pop("pipeline_parallel_size"))
        if "dtype" in params:
            kwargs["dtype"] = params.pop("dtype")
        if "max_batch_size" in params:
            kwargs["max_batch_size"] = int(params.pop("max_batch_size"))
        if "max_seq_len" in params:
            kwargs["max_seq_len"] = int(params.pop("max_seq_len"))
        if "backend" in params:
            kwargs["backend"] = params.pop("backend")

        kwargs.update(params)
        return kwargs
