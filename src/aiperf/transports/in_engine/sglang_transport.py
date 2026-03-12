# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import FirstTokenCallback
from aiperf.transports.in_engine.base_in_engine_transport import (
    BaseInEngineTransport,
)


class SGLangTransport(BaseInEngineTransport):
    """SGLang in-engine transport for offline inference benchmarking.

    Uses SGLang's Engine class for direct inference without an HTTP server.
    The Engine spawns 3 subprocesses internally (tokenizer, scheduler,
    detokenizer) coordinated via ZMQ IPC.
    """

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return SGLang transport metadata for discovery and registration."""
        return TransportMetadata(
            transport_type="sglang",
            url_schemes=["sglang"],
        )

    async def _init_engine(self) -> None:
        """Validate SGLang is installed and parse engine configuration."""
        try:
            import sglang  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "SGLang is required for the sglang:// transport. "
                "Install with: uv add 'sglang[srt]'"
            ) from e

        self._engine_kwargs = self._build_engine_kwargs()
        self.debug(lambda: f"SGLang engine kwargs: {self._engine_kwargs}")

    async def _start_engine(self) -> None:
        """Load model via SGLang Engine.

        SGLang Engine launches 3 subprocesses:
        1. Tokenizer Manager (preprocessing)
        2. Scheduler (memory + model execution)
        3. Detokenizer (incremental decoding)

        Communication between processes uses ZMQ IPC.
        Engine creation is blocking so it runs in an executor.
        """
        import sglang as sgl

        loop = asyncio.get_event_loop()
        self._engine = await loop.run_in_executor(
            None,
            lambda: sgl.Engine(
                model_path=self._model_path,
                **self._engine_kwargs,
            ),
        )

        if self._warmup_iterations > 0:
            self.info(f"Running {self._warmup_iterations} warmup iterations...")
            for _ in range(self._warmup_iterations):
                await self._engine.async_generate(
                    prompt="warmup",
                    sampling_params={"max_new_tokens": 1},
                )

    async def _stop_engine(self) -> None:
        """Shutdown SGLang Engine and all its subprocesses."""
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
        """Generate using SGLang's async_generate().

        Converts chat messages to a prompt string via the engine's tokenizer
        chat template, then calls the async generation API.

        Args:
            messages: Chat messages in OpenAI format
            sampling_params: SGLang-format sampling dict (built by endpoint)
            request_id: Unique request identifier
            first_token_callback: Optional TTFT callback (unused in non-streaming)

        Returns:
            Tuple of (generated_text, input_token_count, output_token_count, finish_reason)
        """
        prompt = self._messages_to_prompt(messages)

        self.debug(lambda: f"SGLang generate request_id={request_id}")

        output = await self._engine.async_generate(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        text = output["text"]
        meta_info = output.get("meta_info", {})

        input_tokens = meta_info.get("prompt_tokens", 0)
        output_tokens = meta_info.get("completion_tokens", 0)

        # SGLang finish_reason is a dict like {"type": "stop"} or {"type": "length"}
        finish_reason = meta_info.get("finish_reason", {})
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason.get("type", "stop")
        finish_reason = str(finish_reason) if finish_reason else "stop"

        self.debug(
            lambda: f"SGLang output: input_tokens={input_tokens}, "
            f"output_tokens={output_tokens}, finish_reason={finish_reason}"
        )

        return text, input_tokens, output_tokens, finish_reason

    def _get_tokenizer(self) -> Any | None:
        """Return SGLang's tokenizer from the tokenizer_manager."""
        if self._engine is not None and hasattr(self._engine, "tokenizer_manager"):
            return self._engine.tokenizer_manager.tokenizer
        return None

    def _build_engine_kwargs(self) -> dict[str, Any]:
        """Build SGLang Engine constructor kwargs from `--engine-params`.

        Extracts engine-specific parameters from `endpoint.engine_params`
        and maps them to SGLang Engine constructor arguments. Unrecognized
        params are passed through directly.

        Returns:
            Keyword arguments for `sgl.Engine()`
        """
        params = self._get_raw_engine_params()
        self._pop_warmup_iterations(params)
        kwargs: dict[str, Any] = {}

        # Accept both canonical and SGLang-native names for parallelism
        if "tensor_parallel_size" in params or "tp" in params:
            kwargs["tp"] = int(params.pop("tensor_parallel_size", params.pop("tp", 1)))
        if "pipeline_parallel_size" in params or "pp" in params:
            kwargs["pp"] = int(
                params.pop("pipeline_parallel_size", params.pop("pp", 1))
            )
        if "mem_fraction_static" in params:
            kwargs["mem_fraction_static"] = float(params.pop("mem_fraction_static"))
        if "dtype" in params:
            kwargs["dtype"] = params.pop("dtype")
        if "quantization" in params:
            kwargs["quantization"] = params.pop("quantization")
        if "trust_remote_code" in params:
            kwargs["trust_remote_code"] = params.pop("trust_remote_code")
        if "load_format" in params:
            kwargs["load_format"] = params.pop("load_format")

        kwargs.update(params)
        return kwargs
