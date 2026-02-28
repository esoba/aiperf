# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.enums import SSEFieldType
from aiperf.common.models import (
    BaseResponseData,
    InferenceServerResponse,
    ParsedResponse,
    ReasoningResponseData,
    RequestInfo,
    TextResponseData,
    Turn,
)
from aiperf.common.types import JsonObject
from aiperf.endpoints.base_endpoint import BaseEndpoint

_DEFAULT_ROLE: str = "user"
_ANTHROPIC_VERSION: str = "2023-06-01"


class AnthropicMessagesEndpoint(BaseEndpoint):
    """Anthropic Messages endpoint.

    Supports text content, tool use, extended thinking, and both
    streaming and non-streaming responses via /v1/messages.
    """

    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Get Anthropic-specific headers using x-api-key auth."""
        cfg = self.model_endpoint.endpoint
        headers: dict[str, str] = {"content-type": "application/json"}
        if cfg.headers:
            headers.update(cfg.headers)
        if cfg.api_key:
            headers["x-api-key"] = cfg.api_key
        headers.setdefault("anthropic-version", _ANTHROPIC_VERSION)
        return headers

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format Anthropic Messages API request payload.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            Anthropic Messages API payload
        """
        if not request_info.turns:
            raise ValueError("Anthropic Messages endpoint requires at least one turn.")

        turns = request_info.turns
        model_endpoint = request_info.model_endpoint
        messages, system = self._create_messages(
            turns, request_info.system_message, request_info.user_context_message
        )

        payload: dict[str, Any] = {
            "model": turns[-1].model or model_endpoint.primary_model_name,
            "messages": messages,
            "max_tokens": turns[-1].max_tokens
            if turns[-1].max_tokens is not None
            else 1024,
            "stream": model_endpoint.endpoint.streaming,
        }

        if system:
            payload["system"] = system

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def _create_messages(
        self,
        turns: list[Turn],
        system_message: str | None,
        user_context_message: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Create messages and extract system prompt for Anthropic Messages API.

        Args:
            turns: List of turns in the request
            system_message: Optional shared system message (becomes top-level system param)
            user_context_message: Optional per-conversation user context to prepend

        Returns:
            Tuple of (messages list, system string or None)
        """
        messages: list[dict[str, Any]] = []

        if user_context_message:
            messages.append({"role": "user", "content": user_context_message})

        for turn in turns:
            role = turn.role or _DEFAULT_ROLE
            content = self._build_turn_content(turn)
            messages.append({"role": role, "content": content})

        return messages, system_message

    def _build_turn_content(self, turn: Turn) -> str | list[dict[str, Any]]:
        """Build content for a single turn.

        Returns a plain string for simple single-text turns,
        or a list of content blocks for complex turns.
        When raw_content is set (verbatim replay), it is used directly.
        """
        if turn.raw_content is not None:
            return turn.raw_content

        if (
            len(turn.texts) == 1
            and len(turn.texts[0].contents) == 1
            and len(turn.images) == 0
            and len(turn.audios) == 0
            and len(turn.videos) == 0
        ):
            return turn.texts[0].contents[0] if turn.texts[0].contents else ""

        content_blocks: list[dict[str, Any]] = []
        for text in turn.texts:
            for item in text.contents:
                if not item:
                    continue
                content_blocks.append({"type": "text", "text": item})

        for image in turn.images:
            for item in image.contents:
                if not item:
                    continue
                content_blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": item},
                    }
                )

        return content_blocks

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse Anthropic Messages response.

        Handles both streaming SSE events and non-streaming JSON responses.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text/reasoning content and usage data
        """
        event_type = self._extract_event_type(response)
        if event_type is not None:
            return self._parse_streaming_event(response, event_type)
        return self._parse_non_streaming(response)

    def _extract_event_type(self, response: InferenceServerResponse) -> str | None:
        """Extract SSE event type from response packets if present."""
        raw = response.get_raw()
        if not isinstance(raw, list):
            return None
        for packet in raw:
            if hasattr(packet, "name") and packet.name == SSEFieldType.EVENT:
                return packet.value
        return None

    def _parse_non_streaming(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse non-streaming Anthropic Messages response."""
        json_obj = response.get_json()
        if not json_obj:
            return None

        data = self._extract_content_data(json_obj)
        usage = self._map_usage(json_obj.get("usage"))

        if data or usage:
            return ParsedResponse(perf_ns=response.perf_ns, data=data, usage=usage)
        return None

    def _extract_content_data(self, json_obj: JsonObject) -> BaseResponseData | None:
        """Extract content from Anthropic non-streaming response content array."""
        content_blocks = json_obj.get("content")
        if not content_blocks or not isinstance(content_blocks, list):
            return None

        text_parts: list[str] = []
        thinking_parts: list[str] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_val = block.get("text")
                if text_val:
                    text_parts.append(text_val)
            elif block_type == "thinking":
                thinking_val = block.get("thinking")
                if thinking_val:
                    thinking_parts.append(thinking_val)

        text = "".join(text_parts) or None
        thinking = "".join(thinking_parts) or None

        if thinking:
            return ReasoningResponseData(content=text, reasoning=thinking)
        return self.make_text_response_data(text)

    def _parse_streaming_event(
        self, response: InferenceServerResponse, event_type: str
    ) -> ParsedResponse | None:
        """Parse a streaming SSE event from the Anthropic Messages API."""
        json_obj = response.get_json()

        match event_type:
            case "message_start":
                if not json_obj:
                    return None
                message = json_obj.get("message", {})
                usage = self._map_usage(message.get("usage"))
                if usage:
                    return ParsedResponse(perf_ns=response.perf_ns, usage=usage)
                return None

            case "content_block_delta":
                if not json_obj:
                    return None
                delta = json_obj.get("delta", {})
                delta_type = delta.get("type")

                if delta_type == "text_delta":
                    text = delta.get("text")
                    if text:
                        return ParsedResponse(
                            perf_ns=response.perf_ns,
                            data=TextResponseData(text=text),
                        )

                elif delta_type == "thinking_delta":
                    thinking = delta.get("thinking")
                    if thinking:
                        return ParsedResponse(
                            perf_ns=response.perf_ns,
                            data=ReasoningResponseData(reasoning=thinking),
                        )

                elif delta_type == "signature_delta":
                    return None

                return None

            case "message_delta":
                if not json_obj:
                    return None
                usage = self._map_usage(json_obj.get("usage"))
                if usage:
                    return ParsedResponse(perf_ns=response.perf_ns, usage=usage)
                return None

            case "ping" | "content_block_start" | "content_block_stop" | "message_stop":
                return None

            case _:
                self.debug(lambda: f"Unknown Anthropic SSE event type: {event_type!r}")
                return None

    @staticmethod
    def _map_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
        """Map Anthropic usage fields to normalized format.

        Anthropic uses input_tokens/output_tokens; we pass through as-is
        since Usage model handles both naming conventions via properties.
        """
        if not usage:
            return None
        return usage
