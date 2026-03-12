# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import Any

from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import FirstTokenCallback
from aiperf.transports.in_engine.base_in_engine_transport import (
    BaseInEngineTransport,
)


def _to_bool(value: Any) -> bool:
    """Convert a string or bool value to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


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
        Engine creation must run on the main thread because SGLang
        internally registers signal handlers (which Python restricts
        to the main thread). The constructor itself is fast — it forks
        off subprocesses for the heavy GPU work.
        """
        import sglang as sgl

        self._engine = sgl.Engine(
            model_path=self._model_path,
            **self._engine_kwargs,
        )

        await self._run_warmup()

    async def _stop_engine(self) -> None:
        """Shutdown SGLang Engine and all its subprocesses."""
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None

    async def _warmup_single(
        self, prompt: str, max_tokens: int, *, streaming: bool
    ) -> None:
        """Run a single SGLang warmup call matching the endpoint's streaming mode.

        Args:
            prompt: Warmup prompt text.
            max_tokens: Maximum tokens to generate.
            streaming: Use stream=True when True, single call when False.
        """
        params: dict[str, Any] = {"max_new_tokens": max_tokens}

        if streaming:
            async for _ in await self._engine.async_generate(
                prompt=prompt,
                sampling_params=params,
                stream=True,
            ):
                pass
        else:
            await self._engine.async_generate(
                prompt=prompt,
                sampling_params=params,
            )

    async def _generate(
        self,
        *,
        messages: list[dict[str, Any]],
        sampling_params: Any,
        request_id: str,
        first_token_callback: FirstTokenCallback | None = None,
        input_ids: list[int] | None = None,
    ) -> tuple[str, int, int, str]:
        """Generate using SGLang's async_generate().

        When streaming is enabled, iterates over the async generator to capture
        first-token timing. Otherwise uses a single non-streaming call.

        Args:
            messages: Chat messages in OpenAI format
            sampling_params: SGLang-format sampling dict (built by endpoint)
            request_id: Unique request identifier
            first_token_callback: Optional TTFT callback for streaming
            input_ids: Pre-tokenized input IDs (bypasses chat template when provided)

        Returns:
            Tuple of (generated_text, input_token_count, output_token_count, finish_reason)
        """
        # Pre-tokenized path: pass input_ids directly to SGLang
        prompt = None if input_ids is not None else self._messages_to_prompt(messages)
        streaming = self.model_endpoint.endpoint.streaming

        self.debug(
            lambda: f"SGLang generate request_id={request_id}, streaming={streaming}"
        )

        if streaming:
            return await self._generate_streaming(
                prompt, sampling_params, request_id, input_ids=input_ids
            )

        return await self._generate_final_only(
            prompt, sampling_params, request_id, input_ids=input_ids
        )

    async def _generate_final_only(
        self,
        prompt: str | None,
        sampling_params: Any,
        request_id: str,
        *,
        input_ids: list[int] | None = None,
    ) -> tuple[str, int, int, str]:
        """Non-streaming generation using a single async_generate() call."""
        generate_kwargs: dict[str, Any] = {"sampling_params": sampling_params}
        if input_ids is not None:
            generate_kwargs["input_ids"] = input_ids
        else:
            generate_kwargs["prompt"] = prompt
        output = await self._engine.async_generate(**generate_kwargs)

        text = output["text"]
        meta_info = output.get("meta_info", {})

        input_tokens = meta_info.get("prompt_tokens", 0)
        output_tokens = meta_info.get("completion_tokens", 0)

        finish_reason = self._parse_finish_reason(meta_info)

        if self._preserve_token_ids:
            output_ids = meta_info.get("output_ids")
            self._output_token_ids = list(output_ids) if output_ids else None

        self.debug(
            lambda: f"SGLang output: input_tokens={input_tokens}, "
            f"output_tokens={output_tokens}, finish_reason={finish_reason}"
        )

        return text, input_tokens, output_tokens, finish_reason

    async def _generate_streaming(
        self,
        prompt: str | None,
        sampling_params: Any,
        request_id: str,
        *,
        input_ids: list[int] | None = None,
    ) -> tuple[str, int, int, str]:
        """Streaming generation for TTFT capture.

        SGLang's async_generate with stream=True yields dicts with cumulative
        text. We track the previous text length to extract deltas.
        """
        prev_text_len = 0
        text_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        finish_reason = "stop"
        is_first_chunk = True

        generate_kwargs: dict[str, Any] = {
            "sampling_params": sampling_params,
            "stream": True,
        }
        if input_ids is not None:
            generate_kwargs["input_ids"] = input_ids
        else:
            generate_kwargs["prompt"] = prompt

        async for chunk in await self._engine.async_generate(**generate_kwargs):
            if is_first_chunk:
                self._first_token_perf_ns = time.perf_counter_ns()
                is_first_chunk = False

            cumulative_text = chunk.get("text", "")
            if len(cumulative_text) > prev_text_len:
                text_parts.append(cumulative_text[prev_text_len:])
                prev_text_len = len(cumulative_text)

            meta_info = chunk.get("meta_info", {})
            input_tokens = meta_info.get("prompt_tokens", input_tokens)
            output_tokens = meta_info.get("completion_tokens", output_tokens)
            finish_reason = self._parse_finish_reason(meta_info) or finish_reason

        if is_first_chunk:
            raise RuntimeError(f"SGLang returned no output for request {request_id}")

        if self._preserve_token_ids:
            output_ids = meta_info.get("output_ids")
            self._output_token_ids = list(output_ids) if output_ids else None

        generated_text = "".join(text_parts)

        self.debug(
            lambda: f"SGLang streaming output: input_tokens={input_tokens}, "
            f"output_tokens={output_tokens}, finish_reason={finish_reason}"
        )

        return generated_text, input_tokens, output_tokens, finish_reason

    @staticmethod
    def _parse_finish_reason(meta_info: dict[str, Any]) -> str:
        """Extract finish_reason from SGLang meta_info.

        SGLang finish_reason is a dict like {"type": "stop"} or {"type": "length"}.
        """
        finish_reason = meta_info.get("finish_reason", {})
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason.get("type", "stop")
        return str(finish_reason) if finish_reason else "stop"

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
        self._pop_warmup_config(params)
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
            kwargs["trust_remote_code"] = _to_bool(params.pop("trust_remote_code"))
        if "load_format" in params:
            kwargs["load_format"] = params.pop("load_format")
        if "disable_cuda_graph" in params:
            kwargs["disable_cuda_graph"] = _to_bool(params.pop("disable_cuda_graph"))
        if "context_length" in params:
            kwargs["context_length"] = int(params.pop("context_length"))
        if "max_model_len" in params:
            kwargs["context_length"] = int(params.pop("max_model_len"))

        # Pass remaining params through with basic type coercion
        for key, value in params.items():
            if isinstance(value, str):
                if value.lower() in ("true", "false"):
                    kwargs[key] = _to_bool(value)
                    continue
                try:
                    kwargs[key] = int(value)
                    continue
                except ValueError:
                    pass
                try:
                    kwargs[key] = float(value)
                    continue
                except ValueError:
                    pass
            kwargs[key] = value
        return kwargs
