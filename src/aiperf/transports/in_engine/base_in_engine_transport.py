# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import time
from abc import abstractmethod
from typing import Any
from urllib.parse import urlparse

from aiperf.common.hooks import on_init, on_stop
from aiperf.common.models import (
    EngineIterationStats,
    ErrorDetails,
    InEngineResponse,
    RequestInfo,
    RequestRecord,
)
from aiperf.transports.base_transports import BaseTransport, FirstTokenCallback


class BaseInEngineTransport(BaseTransport):
    """Base class for in-engine (offline) transport implementations.

    Provides shared functionality for all in-engine transports:
    - Model path extraction from URL scheme (e.g., vllm://org/model)
    - InEngineResponse construction (no JSON round-trip)
    - Error handling with ErrorDetails
    - Lifecycle hooks delegating to abstract engine methods

    Subclasses implement engine-specific logic via four abstract methods:
    ``_init_engine``, ``_start_engine``, ``_stop_engine``, and ``_generate``.
    """

    _WARMUP_SEED_TEXT: str = "The quick brown fox jumps over the lazy dog. "

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model_path: str = ""
        self._engine: Any = None  # Set by subclass in _start_engine
        self._first_token_perf_ns: int | None = None
        self._warmup_iterations: int = 0
        self._warmup_input_tokens: int = 128
        self._warmup_output_tokens: int = 4
        self._preserve_token_ids: bool = False
        self._output_token_ids: list[int] | None = None
        self._decode_iterations: int | None = None
        self._max_draft_len: int = 0
        # Concurrency control
        self._concurrency_semaphore: asyncio.Semaphore | None = None
        # Telemetry
        self._telemetry_enabled: bool = False
        self._telemetry_interval_ms: int = 500
        self._telemetry_log: list[EngineIterationStats] = []
        self._telemetry_task: asyncio.Task[None] | None = None

    # ---- BaseTransport interface -----------------------------------------------

    def get_url(self, request_info: RequestInfo) -> str:
        """Return the model path (no HTTP URL needed for in-engine transports)."""
        return self._model_path

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Return empty headers (no HTTP layer for in-engine transports)."""
        return {}

    # ---- Lifecycle hooks -------------------------------------------------------

    @on_init
    async def _on_init_engine(self) -> None:
        """Initialize engine during INITIALIZING phase."""
        self._model_path = self._extract_model_path()
        self.info(f"Initializing in-engine transport for model: {self._model_path}")
        await self._init_engine()

    async def configure(self) -> None:
        """Load the engine model (heavy initialization).

        Called explicitly during PROFILE_CONFIGURE, not during the service
        start phase, so that Worker can finish registration first.

        AIPerf workers run as daemon processes, but engine libraries (vLLM,
        SGLang, TRT-LLM) may spawn child processes for GPU management. Python's
        stdlib forbids daemon processes from creating children. Temporarily
        lifting the daemon flag allows engine initialization to proceed.
        Same technique used by billiard (see dataset/generator/parallel_decode.py).
        """
        self.info(f"Loading model: {self._model_path} (this may take several minutes)")
        current_process = multiprocessing.current_process()
        was_daemon = current_process.daemon
        current_process.daemon = False
        try:
            await self._start_engine()
        finally:
            current_process.daemon = was_daemon
        self._start_telemetry_loop()
        self.info("Engine loaded and ready for inference")

    @on_stop
    async def _on_stop_engine(self) -> None:
        """Stop engine during STOPPING phase."""
        self.info("Shutting down engine")
        await self._stop_telemetry_loop()
        await self._stop_engine()
        self._engine = None
        self.info("Engine shutdown complete")

    # ---- Transport interface ---------------------------------------------------

    async def send_request(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any],
        *,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send request to the in-engine LLM.

        Reads messages and sampling params directly from the payload
        (provided by EngineGenerateEndpoint), calls the engine, and wraps
        the result in a RequestRecord with an InEngineResponse.

        Args:
            request_info: Request context and metadata
            payload: Payload from EngineGenerateEndpoint with messages and sampling_params
            first_token_callback: Optional callback fired on first token

        Returns:
            Record containing responses, timing, and any errors
        """
        start_perf_ns = time.perf_counter_ns()

        try:
            messages = payload["messages"]
            sampling_params = payload.get("sampling_params", {})
            request_id = request_info.x_request_id
            input_ids = payload.get("input_ids")

            generate_coro = self._generate(
                messages=messages,
                sampling_params=sampling_params,
                request_id=request_id,
                first_token_callback=first_token_callback,
                input_ids=input_ids,
            )
            if self._concurrency_semaphore is not None:
                async with self._concurrency_semaphore:
                    (
                        text,
                        input_tokens,
                        output_tokens,
                        finish_reason,
                    ) = await generate_coro
            else:
                text, input_tokens, output_tokens, finish_reason = await generate_coro

            first_token_perf_ns = self._first_token_perf_ns
            self._first_token_perf_ns = None

            # Capture token IDs and speculative decoding metadata (set by subclass in _generate)
            output_token_ids = self._output_token_ids
            self._output_token_ids = None
            decode_iterations = self._decode_iterations
            self._decode_iterations = None
            max_draft_len = self._max_draft_len if self._max_draft_len > 0 else None

            now = time.perf_counter_ns()
            responses: list[InEngineResponse] = []

            if first_token_perf_ns is not None:
                # Streaming path: two responses mimic SSE chunked output
                first_response = InEngineResponse(
                    perf_ns=first_token_perf_ns,
                    text="",
                    input_tokens=0,
                    output_tokens=0,
                    finish_reason="",
                )
                responses.append(first_response)
                if first_token_callback is not None:
                    ttft_ns = first_token_perf_ns - start_perf_ns
                    await first_token_callback(ttft_ns, first_response)

            responses.append(
                InEngineResponse(
                    perf_ns=now,
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reason=finish_reason,
                    output_token_ids=output_token_ids,
                    decode_iterations=decode_iterations,
                    max_draft_len=max_draft_len,
                )
            )

            return RequestRecord(
                request_info=request_info,
                start_perf_ns=start_perf_ns,
                end_perf_ns=time.perf_counter_ns(),
                responses=responses,
                status=200,
            )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.error(f"In-engine request failed: {e!r}")
            return self._build_error_record(
                request_info=request_info,
                start_perf_ns=start_perf_ns,
                error=e,
            )

    # ---- Shared utilities ------------------------------------------------------

    def _extract_model_path(self) -> str:
        """Extract model path from ``--model`` name or the configured URL scheme.

        Prefers the primary model name (from ``--model``) so that users don't
        need to duplicate model info in the URL. Falls back to URL-scheme
        parsing for explicit engine URLs (e.g., ``vllm://org/model``).

        Handles:
          ``--model meta-llama/Llama-3.1-8B``    -> ``meta-llama/Llama-3.1-8B``
          ``vllm://meta-llama/Llama-3.1-8B``     -> ``meta-llama/Llama-3.1-8B``
          ``vllm:///absolute/path/to/model``      -> ``/absolute/path/to/model``
          ``sglang://org/model``                  -> ``org/model``
          ``trtllm:///path/to/engine``            -> ``/path/to/engine``

        Returns:
            Model path derived from model name or base URL
        """
        model_name = self.model_endpoint.primary_model_name
        if model_name:
            return model_name

        # Fallback: extract from URL scheme (e.g., vllm://org/model)
        base_url = self.model_endpoint.endpoint.base_url
        parsed = urlparse(base_url)
        # For "vllm://meta-llama/Model" -> netloc="meta-llama", path="/Model"
        # For "vllm:///absolute/path"   -> netloc="", path="/absolute/path"
        if parsed.netloc:
            return f"{parsed.netloc}{parsed.path}".rstrip("/")
        return parsed.path.rstrip("/")

    def _build_error_record(
        self,
        *,
        request_info: RequestInfo,
        start_perf_ns: int,
        error: Exception,
    ) -> RequestRecord:
        """Build a RequestRecord for a failed engine call.

        Args:
            request_info: Original request context
            start_perf_ns: Request start timestamp from perf_counter_ns
            error: The exception that caused the failure

        Returns:
            RequestRecord with error details and timing
        """
        return RequestRecord(
            request_info=request_info,
            start_perf_ns=start_perf_ns,
            end_perf_ns=time.perf_counter_ns(),
            error=ErrorDetails.from_exception(error),
        )

    def _get_raw_engine_params(self) -> dict[str, Any]:
        """Get engine params as a mutable dict for ``_build_engine_kwargs``.

        Returns:
            Copy of ``endpoint.engine_params`` as a dict, or empty dict
        """
        if self.model_endpoint.endpoint.engine_params:
            return dict(self.model_endpoint.endpoint.engine_params)
        return {}

    def _pop_warmup_config(self, params: dict[str, Any]) -> None:
        """Extract warmup, telemetry, and concurrency parameters from engine params.

        Pops ``concurrency``, ``warmup_iterations``, ``warmup_input_tokens``,
        ``warmup_output_tokens``, ``telemetry``, and ``telemetry_interval_ms``
        from *params* so they are not passed to the engine constructor.

        When ``concurrency`` is a positive integer, creates an
        ``asyncio.Semaphore`` that limits how many concurrent ``_generate()``
        calls can run simultaneously (similar to trtllm-bench's LlmManager).

        Args:
            params: Mutable dict of engine params (keys are consumed if present)
        """
        if "concurrency" in params:
            concurrency = int(params.pop("concurrency"))
            if concurrency > 0:
                self._concurrency_semaphore = asyncio.Semaphore(concurrency)
        if "warmup_iterations" in params:
            self._warmup_iterations = int(params.pop("warmup_iterations"))
        if "warmup_input_tokens" in params:
            self._warmup_input_tokens = int(params.pop("warmup_input_tokens"))
        if "warmup_output_tokens" in params:
            self._warmup_output_tokens = int(params.pop("warmup_output_tokens"))
        if "preserve_token_ids" in params:
            val = params.pop("preserve_token_ids")
            self._preserve_token_ids = (
                val
                if isinstance(val, bool)
                else str(val).lower() in ("true", "1", "yes")
            )
        if "telemetry" in params:
            val = params.pop("telemetry")
            self._telemetry_enabled = (
                val
                if isinstance(val, bool)
                else str(val).lower() in ("true", "1", "yes")
            )
        if "telemetry_interval_ms" in params:
            self._telemetry_interval_ms = int(params.pop("telemetry_interval_ms"))

    # ---- Telemetry infrastructure ----------------------------------------------

    async def _get_engine_stats(self) -> dict[str, Any] | None:
        """Return engine-specific stats for the current iteration.

        Override in subclasses to poll the engine's stats API. The returned
        dict is stored in ``EngineIterationStats.raw`` and its recognised
        keys (``batch_size``, ``num_tokens``, ``queue_depth``) are promoted
        to top-level fields.

        Returns:
            Dict of engine stats, or None when stats are unavailable.
        """
        return None

    def _start_telemetry_loop(self) -> None:
        """Start the background telemetry polling task if telemetry is enabled."""
        if not self._telemetry_enabled:
            return
        self.info(
            f"Starting engine telemetry (interval={self._telemetry_interval_ms}ms)"
        )
        self._telemetry_task = asyncio.create_task(self._telemetry_loop())

    async def _stop_telemetry_loop(self) -> None:
        """Cancel and await the telemetry task if it is running."""
        if self._telemetry_task is None:
            return
        self._telemetry_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._telemetry_task
        self._telemetry_task = None
        self.debug(
            lambda: f"Telemetry stopped, {len(self._telemetry_log)} entries collected"
        )

    async def _telemetry_loop(self) -> None:
        """Poll ``_get_engine_stats`` at the configured interval.

        Appends an ``EngineIterationStats`` entry for each non-None response.
        Runs until cancelled.
        """
        interval_s = self._telemetry_interval_ms / 1000.0
        while True:
            try:
                raw = await self._get_engine_stats()
                if raw is not None:
                    entry = EngineIterationStats(
                        timestamp_ns=time.time_ns(),
                        batch_size=raw.get("batch_size"),
                        num_tokens=raw.get("num_tokens"),
                        queue_depth=raw.get("queue_depth"),
                        raw=raw,
                    )
                    self._telemetry_log.append(entry)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.debug(lambda e=e: f"Telemetry poll error: {e!r}")
            await asyncio.sleep(interval_s)

    def get_telemetry_log(self) -> list[EngineIterationStats]:
        """Return the accumulated telemetry entries for export.

        Returns:
            List of ``EngineIterationStats`` collected during the run.
        """
        return list(self._telemetry_log)

    # ---- Prompt formatting -----------------------------------------------------

    def _get_tokenizer(self) -> Any | None:
        """Return the engine's tokenizer, if available.

        Override in subclasses to provide the engine-specific tokenizer path.
        Used by ``_messages_to_prompt`` for chat template application.

        Returns:
            Tokenizer object with ``apply_chat_template``, or None
        """
        return None

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert OpenAI-format messages to a prompt string.

        Uses the engine's tokenizer to apply the model's chat template
        when available, falling back to simple tag-based concatenation.

        Args:
            messages: Chat messages with role/content keys

        Returns:
            Formatted prompt string ready for generation
        """
        tokenizer = self._get_tokenizer()
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Fallback: tag-based concatenation for models without a chat template
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                )
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    # ---- Warmup infrastructure --------------------------------------------------

    def _generate_warmup_prompt(self, target_tokens: int) -> str:
        """Generate a warmup prompt string targeting *target_tokens* input tokens.

        Uses the engine tokenizer (via ``_get_tokenizer()``) to encode a seed
        text, repeats the token IDs to reach the target length, then decodes
        back to a string. Falls back to repeating the seed text when no
        tokenizer is available.

        Args:
            target_tokens: Desired number of input tokens for the warmup prompt.

        Returns:
            A prompt string approximately *target_tokens* long.
        """
        seed = self._WARMUP_SEED_TEXT
        tokenizer = self._get_tokenizer()

        if (
            tokenizer is not None
            and hasattr(tokenizer, "encode")
            and hasattr(tokenizer, "decode")
        ):
            token_ids = tokenizer.encode(seed)
            if not token_ids:
                return seed * max(1, target_tokens // len(seed.split()))

            # Repeat token IDs to reach the target length
            repeated: list[int] = []
            while len(repeated) < target_tokens:
                repeated.extend(token_ids)
            repeated = repeated[:target_tokens]
            return tokenizer.decode(repeated)

        # Fallback: repeat seed text (rough approximation: ~1 token per word)
        word_count = len(seed.split())
        repeats = max(1, target_tokens // word_count)
        return seed * repeats

    async def _run_warmup(self) -> None:
        """Run warmup iterations using realistic prompts matching endpoint config.

        Generates a prompt with ``_warmup_input_tokens`` tokens and calls
        ``_warmup_single`` for ``_warmup_iterations`` iterations, using the
        endpoint's streaming setting to exercise the same code path as
        real inference.
        """
        if self._warmup_iterations <= 0:
            return

        streaming = self.model_endpoint.endpoint.streaming
        prompt = self._generate_warmup_prompt(self._warmup_input_tokens)
        max_tokens = self._warmup_output_tokens

        self.info(
            f"Running {self._warmup_iterations} warmup iterations "
            f"(input≈{self._warmup_input_tokens}tok, output={max_tokens}tok, "
            f"streaming={streaming})"
        )

        for i in range(self._warmup_iterations):
            await self._warmup_single(prompt, max_tokens, streaming=streaming)
            self.debug(
                lambda i=i: f"Warmup iteration {i + 1}/{self._warmup_iterations} complete"
            )

    @abstractmethod
    async def _warmup_single(
        self, prompt: str, max_tokens: int, *, streaming: bool
    ) -> None:
        """Run a single warmup inference call.

        Subclasses implement this with engine-specific generation logic,
        using the streaming path when *streaming* is True to exercise
        the same code path as real requests.

        Args:
            prompt: Warmup prompt text.
            max_tokens: Maximum tokens to generate.
            streaming: Whether to use the streaming code path.
        """
        ...

    # ---- Abstract methods for subclasses ---------------------------------------

    @abstractmethod
    async def _init_engine(self) -> None:
        """Import engine library and prepare configuration.

        Called during ``@on_init``. Should NOT load the model yet.
        Validate that engine dependencies are installed.
        """
        ...

    @abstractmethod
    async def _start_engine(self) -> None:
        """Load model, allocate memory, and run warmup.

        Called during ``configure()``. This is where heavy initialization
        happens (model loading, CUDA graph capture, JIT compilation).
        """
        ...

    @abstractmethod
    async def _stop_engine(self) -> None:
        """Shutdown engine and free all resources.

        Called during ``@on_stop``. Must be safe to call even if the
        engine failed to initialize.
        """
        ...

    @abstractmethod
    async def _generate(
        self,
        *,
        messages: list[dict[str, Any]],
        sampling_params: Any,
        request_id: str,
        first_token_callback: FirstTokenCallback | None = None,
        input_ids: list[int] | None = None,
    ) -> tuple[str, int, int, str]:
        """Run generation on the engine.

        Args:
            messages: Chat messages
            sampling_params: Engine-native sampling params (built by the endpoint)
            request_id: Unique request identifier
            first_token_callback: Optional TTFT callback for streaming
            input_ids: Pre-tokenized input IDs (bypasses chat template when provided)

        Returns:
            Tuple of (generated_text, input_token_count, output_token_count, finish_reason)
        """
        ...
