# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    InEngineResponse,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
    TextResponseData,
)
from aiperf.common.models.usage_models import Usage
from aiperf.endpoints.openai_chat import ChatEndpoint


class EngineGenerateEndpoint(ChatEndpoint):
    """Base direct-generate endpoint for in-engine transports.

    Formats payloads with messages and engine-native sampling params,
    bypassing OpenAI format serialization. Subclasses override
    ``_build_sampling_params`` to construct engine-specific objects.
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format payload for in-engine transports.

        Builds a flat dict with messages and engine-native sampling_params
        that the transport passes directly to the engine.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            Dict with messages, sampling_params, model, and stream keys
        """
        if not request_info.turns:
            raise ValueError("Chat endpoint requires at least one turn.")

        turns = request_info.turns
        model_endpoint = request_info.model_endpoint
        messages = self._create_messages(
            turns, request_info.system_message, request_info.user_context_message
        )

        # Strip 'name' field — it's an OpenAI-specific extension that can
        # break engine chat templates (e.g., vLLM's Qwen3 template).
        for msg in messages:
            msg.pop("name", None)

        sampling_params = self._build_sampling_params(request_info)

        payload = {
            "messages": messages,
            "sampling_params": sampling_params,
            "model": turns[-1].model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        self.trace(lambda: f"Formatted engine generate payload: {payload}")
        return payload

    def _build_sampling_params(self, request_info: RequestInfo) -> Any:
        """Build sampling params dict from turn config and extra inputs.

        Override in subclasses to return engine-native SamplingParams objects.

        Args:
            request_info: Request context with turn and endpoint config

        Returns:
            Dict of sampling parameters (base implementation)
        """
        params: dict[str, Any] = {}
        turn = request_info.turns[-1]

        if turn.max_tokens is not None:
            params["max_tokens"] = turn.max_tokens

        # Extra inputs go into sampling_params (not top-level payload)
        if request_info.model_endpoint.endpoint.extra:
            params.update(request_info.model_endpoint.endpoint.extra)

        return params

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse InEngineResponse directly — no JSON parsing needed.

        Falls back to parent ChatEndpoint.parse_response for non-InEngineResponse
        types (e.g., TextResponse from legacy transports).

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with text and usage data, or None
        """
        if not isinstance(response, InEngineResponse):
            return super().parse_response(response)

        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=TextResponseData(text=response.text),
            usage=Usage(
                {
                    "prompt_tokens": response.input_tokens,
                    "completion_tokens": response.output_tokens,
                    "total_tokens": response.input_tokens + response.output_tokens,
                }
            ),
        )


class VLLMGenerateEndpoint(EngineGenerateEndpoint):
    """vLLM direct-generate endpoint. Builds ``vllm.SamplingParams``."""

    def _build_sampling_params(self, request_info: RequestInfo) -> Any:
        """Build a ``vllm.SamplingParams`` from turn config and extra inputs."""
        from vllm import SamplingParams

        params = super()._build_sampling_params(request_info)
        return SamplingParams(**params)


class SGLangGenerateEndpoint(EngineGenerateEndpoint):
    """SGLang direct-generate endpoint. Builds SGLang-format sampling dict."""

    def _build_sampling_params(self, request_info: RequestInfo) -> dict[str, Any]:
        """Build SGLang sampling params dict (max_tokens -> max_new_tokens)."""
        params = super()._build_sampling_params(request_info)

        # SGLang uses max_new_tokens instead of max_tokens
        if "max_tokens" in params:
            params["max_new_tokens"] = params.pop("max_tokens")

        return params


class TRTLLMGenerateEndpoint(EngineGenerateEndpoint):
    """TensorRT-LLM direct-generate endpoint. Builds ``tensorrt_llm.SamplingParams``."""

    def _build_sampling_params(self, request_info: RequestInfo) -> Any:
        """Build a ``tensorrt_llm.SamplingParams`` from turn config and extra inputs."""
        from tensorrt_llm import SamplingParams

        params = super()._build_sampling_params(request_info)

        # TRT-LLM uses random_seed instead of seed
        if "seed" in params:
            params["random_seed"] = params.pop("seed")

        # TRT-LLM uses stop_words (always a list) instead of stop
        if "stop" in params:
            stop = params.pop("stop")
            params["stop_words"] = stop if isinstance(stop, list) else [stop]

        return SamplingParams(**params)
