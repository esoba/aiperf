# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import multiprocessing
import time
from abc import abstractmethod
from typing import Any
from urllib.parse import urlparse

from aiperf.common.hooks import on_init, on_stop
from aiperf.common.models import (
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model_path: str = ""
        self._engine: Any = None  # Set by subclass in _start_engine

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
        self.info("Engine loaded and ready for inference")

    @on_stop
    async def _on_stop_engine(self) -> None:
        """Stop engine during STOPPING phase."""
        self.info("Shutting down engine")
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

            text, input_tokens, output_tokens, finish_reason = await self._generate(
                messages=messages,
                sampling_params=sampling_params,
                request_id=request_id,
                first_token_callback=first_token_callback,
            )

            response = InEngineResponse(
                perf_ns=time.perf_counter_ns(),
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
            )

            return RequestRecord(
                request_info=request_info,
                start_perf_ns=start_perf_ns,
                end_perf_ns=time.perf_counter_ns(),
                responses=[response],
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
        """Extract model path from the configured URL scheme.

        Handles:
          ``vllm://meta-llama/Llama-3.1-8B``    -> ``meta-llama/Llama-3.1-8B``
          ``vllm:///absolute/path/to/model``     -> ``/absolute/path/to/model``
          ``sglang://org/model``                 -> ``org/model``
          ``trtllm:///path/to/engine``           -> ``/path/to/engine``

        Returns:
            Model path extracted from the base URL
        """
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
    ) -> tuple[str, int, int, str]:
        """Run generation on the engine.

        Args:
            messages: Chat messages
            sampling_params: Engine-native sampling params (built by the endpoint)
            request_id: Unique request identifier
            first_token_callback: Optional TTFT callback for streaming

        Returns:
            Tuple of (generated_text, input_token_count, output_token_count, finish_reason)
        """
        ...
