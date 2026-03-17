# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, TypeVar

from pydantic import ConfigDict, Field, model_validator

from aiperf.common.models import AIPerfBaseModel, Audio, Image, Text, Video
from aiperf.plugin.enums import CustomDatasetType


class SingleTurn(AIPerfBaseModel):
    """Defines the schema for single-turn data.

    User can use this format to quickly provide a custom single turn dataset.
    Each line in the file will be treated as a single turn conversation.

    The single turn type
      - supports multi-modal (e.g. text, image, audio, video)
      - supports client-side batching for each data (e.g. batch_size > 1)
      - DOES NOT support multi-turn features (e.g. session_id)
    """

    type: Literal[CustomDatasetType.SINGLE_TURN] = CustomDatasetType.SINGLE_TURN

    # TODO (TL-89): investigate if we only want to support single field for each modality
    text: str | None = Field(None, description="Simple text string content")
    texts: list[str] | list[Text] | None = Field(
        None,
        description="List of text strings or Text objects format",
    )
    image: str | None = Field(None, description="Simple image string content")
    images: list[str] | list[Image] | None = Field(
        None,
        description="List of image strings or Image objects format",
    )
    audio: str | None = Field(None, description="Simple audio string content")
    audios: list[str] | list[Audio] | None = Field(
        None,
        description="List of audio strings or Audio objects format",
    )
    video: str | None = Field(
        None,
        description="Simple video string content. Can be a URL, local file path, or base64 encoded data URL.",
    )
    videos: list[str] | list[Video] | None = Field(
        None,
        description="List of video strings or Video objects format",
    )
    timestamp: int | float | None = Field(
        default=None,
        description="Timestamp of the turn in milliseconds. Supports floating point, but scheduling accuracy is at the millisecond level.",
    )
    delay: int | float | None = Field(
        default=None,
        description="Amount of milliseconds to wait before sending the turn. Supports floating point, but scheduling accuracy is at the millisecond level.",
    )
    role: str | None = Field(default=None, description="Role of the turn.")

    @model_validator(mode="after")
    def validate_mutually_exclusive_fields(self) -> "SingleTurn":
        """Ensure mutually exclusive fields are not set together"""
        if self.text and self.texts:
            raise ValueError("text and texts cannot be set together")
        if self.image and self.images:
            raise ValueError("image and images cannot be set together")
        if self.audio and self.audios:
            raise ValueError("audio and audios cannot be set together")
        if self.video and self.videos:
            raise ValueError("video and videos cannot be set together")
        if self.timestamp and self.delay:
            raise ValueError("timestamp and delay cannot be set together")
        return self

    @model_validator(mode="after")
    def validate_at_least_one_modality(self) -> "SingleTurn":
        """Ensure at least one modality is provided"""
        if not any(
            [
                self.text,
                self.texts,
                self.image,
                self.images,
                self.audio,
                self.audios,
                self.video,
                self.videos,
            ]
        ):
            raise ValueError("At least one modality must be provided")
        return self


class MultiTurn(AIPerfBaseModel):
    """Defines the schema for multi-turn conversations.

    The multi-turn custom dataset
      - supports multi-modal data (e.g. text, image, audio, video)
      - supports multi-turn features (e.g. delay, sessions, etc.)
      - supports client-side batching for each data (e.g. batch size > 1)
    """

    type: Literal[CustomDatasetType.MULTI_TURN] = CustomDatasetType.MULTI_TURN

    session_id: str | None = Field(
        None, description="Unique identifier for the conversation session"
    )
    turns: list[SingleTurn] = Field(
        ..., description="List of turns in the conversation"
    )

    @model_validator(mode="after")
    def validate_turns_not_empty(self) -> "MultiTurn":
        """Ensure at least one turn is provided"""
        if not self.turns:
            raise ValueError("At least one turn must be provided")
        return self


class RandomPool(AIPerfBaseModel):
    """Defines the schema for random pool data entry.

    The random pool custom dataset
      - supports multi-modal data (e.g. text, image, audio, video)
      - supports client-side batching for each data (e.g. batch size > 1)
      - supports named fields for each modality (e.g. text_field_a, text_field_b, etc.)
      - DOES NOT support multi-turn or its features (e.g. delay, sessions, etc.)
    """

    type: Literal[CustomDatasetType.RANDOM_POOL] = CustomDatasetType.RANDOM_POOL

    text: str | None = Field(None, description="Simple text string content")
    texts: list[str] | list[Text] | None = Field(
        None,
        description="List of text strings or Text objects format",
    )
    image: str | None = Field(None, description="Simple image string content")
    images: list[str] | list[Image] | None = Field(
        None,
        description="List of image strings or Image objects format",
    )
    audio: str | None = Field(None, description="Simple audio string content")
    audios: list[str] | list[Audio] | None = Field(
        None,
        description="List of audio strings or Audio objects format",
    )
    video: str | None = Field(
        None,
        description="Simple video string content. Can be a URL, local file path, or base64 encoded data URL.",
    )
    videos: list[str] | list[Video] | None = Field(
        None,
        description="List of video strings or Video objects format",
    )

    @model_validator(mode="after")
    def validate_mutually_exclusive_fields(self) -> "RandomPool":
        """Ensure mutually exclusive fields are not set together"""
        if self.text and self.texts:
            raise ValueError("text and texts cannot be set together")
        if self.image and self.images:
            raise ValueError("image and images cannot be set together")
        if self.audio and self.audios:
            raise ValueError("audio and audios cannot be set together")
        if self.video and self.videos:
            raise ValueError("video and videos cannot be set together")
        return self

    @model_validator(mode="after")
    def validate_at_least_one_modality(self) -> "RandomPool":
        """Ensure at least one modality is provided"""
        if not any(
            [
                self.text,
                self.texts,
                self.image,
                self.images,
                self.audio,
                self.audios,
                self.video,
                self.videos,
            ]
        ):
            raise ValueError("At least one modality must be provided")
        return self


class MooncakeTrace(AIPerfBaseModel):
    """Defines the schema for Mooncake trace data.

    See https://github.com/kvcache-ai/Mooncake for more details.

    Supports three input modes (exactly one required):
    - input_length: Synthetic text generated from token count (optionally with hash_ids)
    - text_input: Literal text string sent as the prompt
    - messages: List of OpenAI-compatible message dicts sent directly to the API

    Examples:
    - Minimal: {"input_length": 10, "hash_ids": [123]}
    - With input_length: {"input_length": 10, "output_length": 4}
    - With text_input: {"text_input": "Hello world", "output_length": 4}
    - With messages: {"messages": [{"role": "user", "content": "Hello"}], "output_length": 4}
    - With timestamp and hash ID: {"timestamp": 1000, "input_length": 10, "hash_ids": [123]}
    """

    type: Literal[CustomDatasetType.MOONCAKE_TRACE] = CustomDatasetType.MOONCAKE_TRACE
    input_length: int | None = Field(
        None,
        description="The input sequence length of a request. Required if text_input and messages are not provided.",
    )
    text_input: str | None = Field(
        None,
        description="The actual text input for the request.",
    )
    messages: list[dict[str, Any]] | None = Field(
        None,
        description="List of OpenAI-compatible message dicts (each must have a 'role' key) sent directly to the API.",
    )
    tools: list[dict[str, Any]] | None = Field(
        None,
        description="List of OpenAI-compatible tool definitions. Only allowed when 'messages' is provided.",
    )

    # Optional fields
    output_length: int | None = Field(
        None, description="The output sequence length of a request"
    )
    hash_ids: list[int] | None = Field(None, description="The hash ids of a request")
    timestamp: int | float | None = Field(
        None,
        description="The timestamp of a request in milliseconds. Supports floating point, but scheduling accuracy is at the millisecond level.",
    )
    delay: int | float | None = Field(
        None,
        description="Amount of milliseconds to wait before sending the turn. Supports floating point, but scheduling accuracy is at the millisecond level.",
    )
    session_id: str | None = Field(
        None, description="Unique identifier for the conversation session"
    )

    @model_validator(mode="after")
    def validate_input(self) -> "MooncakeTrace":
        """Validate that exactly one input mode is provided."""
        input_modes = [
            self.input_length is not None,
            self.text_input is not None,
            self.messages is not None,
        ]
        input_mode_count = sum(input_modes)
        if input_mode_count == 0:
            raise ValueError(
                "Exactly one of 'input_length', 'text_input', or 'messages' must be provided"
            )
        if input_mode_count > 1:
            raise ValueError(
                "'input_length', 'text_input', and 'messages' are mutually exclusive. Use only one of them."
            )

        if self.hash_ids is not None and self.input_length is None:
            raise ValueError(
                "'hash_ids' is only allowed when 'input_length' is provided, not when 'text_input' or 'messages' are provided"
            )

        return self

    @model_validator(mode="after")
    def validate_messages(self) -> "MooncakeTrace":
        """Validate the messages and tools field structure."""
        if self.tools is not None:
            if self.messages is None:
                raise ValueError("'tools' is only allowed when 'messages' is provided")
            if not self.tools:
                raise ValueError("'tools' must be a non-empty list")

        if self.messages is None:
            return self

        if not self.messages:
            raise ValueError("'messages' must be a non-empty list")

        for i, msg in enumerate(self.messages):
            if not isinstance(msg, dict) or "role" not in msg:
                raise ValueError(
                    f"Each message must have a 'role' key, but message at index {i} does not"
                )

        return self


class BailianTrace(AIPerfBaseModel):
    """Defines the schema for Alibaba Bailian trace data.

    See https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon for the
    upstream dataset and full documentation.

    Each entry represents a single request in a conversation chain. Multi-turn
    conversations are linked via ``chat_id`` and ``parent_chat_id``: entries
    sharing the same root ``chat_id`` (reachable through ``parent_chat_id``)
    belong to the same session and are ordered by ``turn``.

    Important: Bailian traces use a block size of 16 tokens per salted SipHash
    block.  Use ``--isl-block-size 16`` when using this format (this is set
    automatically in CLI flows).

    Examples:
    - Root request:  ``{"chat_id": 159, "parent_chat_id": -1, "timestamp": 61.114, "input_length": 521, "output_length": 132, "type": "text", "turn": 1, "hash_ids": [1089, 1090, 1091]}``
    - Follow-up:     ``{"chat_id": 160, "parent_chat_id": 159, "timestamp": 62.5, "input_length": 400, "output_length": 80, "type": "text", "turn": 2, "hash_ids": [1089, 1090]}``

    Note:
    The ``type`` field in Bailian JSONL is the request type (text/search/image/file),
    not the dataset type. Use ``--custom-dataset-type bailian_trace`` when loading
    this format.
    """

    model_config = ConfigDict(populate_by_name=True)

    chat_id: int = Field(description="Randomized chat identifier")
    parent_chat_id: int = Field(
        default=-1,
        description="Parent chat ID for multi-turn conversation chains. -1 indicates a root request.",
    )
    timestamp: float = Field(
        description="Seconds since request arrival. Converted to milliseconds internally.",
    )
    input_length: int = Field(description="Input token count")
    output_length: int = Field(description="Output token count")
    request_type: str = Field(
        default="",
        alias="type",
        description="Request type from the trace (text/search/image/file). Aliased from 'type' in JSONL.",
    )
    turn: int = Field(default=1, description="Conversation turn number")
    hash_ids: list[int] = Field(
        default_factory=list,
        description="Salted SipHash block IDs (16 tokens per block)",
    )


class ConfluxTokens(AIPerfBaseModel):
    """Normalized token counts across providers."""

    input: int = Field(
        default=0,
        description="Total input tokens processed (all input the model saw). "
        "For Anthropic: input_tokens + cache_creation_input_tokens + cache_read_input_tokens. "
        "For OpenAI: prompt_tokens (already includes cached).",
    )
    input_cached: int = Field(
        default=0,
        description="Tokens read from cache. "
        "For Anthropic: cache_read_input_tokens. For OpenAI: equivalent cached tokens.",
    )
    input_cache_write: int = Field(
        default=0,
        description="Tokens written to cache. For Anthropic: cache_creation_input_tokens.",
    )
    output: int = Field(default=0, description="Total output tokens generated.")
    output_reasoning: int = Field(
        default=0,
        description="Output tokens used for reasoning/thinking, when available from the provider.",
    )


class ConfluxHyperparameters(AIPerfBaseModel):
    """Normalized generation hyperparameters extracted from the request body.

    Matches the canonical keys defined in Conflux's hyperparameters.rs
    normalization layer. Unknown keys are dropped during normalization;
    null values are omitted.
    """

    max_tokens: int | None = Field(
        default=None, description="Maximum tokens to generate (Anthropic max_tokens)."
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="Maximum output tokens (OpenAI max_completion_tokens).",
    )
    temperature: float | None = Field(default=None, description="Sampling temperature.")
    top_p: float | None = Field(default=None, description="Nucleus sampling cutoff.")
    top_k: int | None = Field(default=None, description="Top-k sampling cutoff.")
    presence_penalty: float | None = Field(
        default=None, description="Presence penalty for repeated tokens."
    )
    frequency_penalty: float | None = Field(
        default=None, description="Frequency penalty for repeated tokens."
    )
    seed: int | None = Field(
        default=None, description="Random seed for deterministic generation."
    )
    stop: Any = Field(default=None, description="Stop sequences or tokens.")
    reasoning_effort: str | None = Field(
        default=None, description="Reasoning effort level (e.g. low, medium, high)."
    )
    reasoning_summary: str | None = Field(
        default=None, description="Reasoning summary mode (OpenAI-specific)."
    )
    text_verbosity: str | None = Field(
        default=None, description="Text verbosity mode (OpenAI-specific)."
    )


class ConfluxRecord(AIPerfBaseModel):
    """A single unified API call from a Conflux proxy capture.

    Conforms to the Conflux unified canonical schema. Each record represents
    one API request/response cycle captured via MITM proxy intercept, with
    agent threading metadata and full request payload for verbatim replay.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal[CustomDatasetType.CONFLUX] = CustomDatasetType.CONFLUX

    id: str = Field(
        description="Unique identifier for this API call. "
        "Typically the provider request ID (e.g. req_...) or a synthesized key.",
    )
    source: str | None = Field(
        default=None,
        description="How this record was captured (e.g. 'proxy' for MITM proxy intercept).",
    )
    client: str | None = Field(
        default=None,
        description="Which AI coding tool made this API call (claude, codex, unknown).",
    )
    request_id: str | None = Field(
        default=None,
        description="Provider-assigned request identifier "
        "(e.g. Anthropic req_... or OpenAI chatcmpl-...).",
    )
    session_id: str = Field(
        description="Session identifier grouping related API calls in a single coding session.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Identifier of the agent/persona that made this call.",
    )
    is_subagent: bool | None = Field(
        default=None,
        description="Whether this call was made by a sub-agent (e.g. a tool-spawned background task). "
        "None means un-enriched (not yet classified by the adapter pipeline).",
    )
    timestamp: str = Field(
        description="ISO 8601 timestamp when the API request was sent.",
    )
    duration_ms: int | float = Field(
        default=0,
        description="Time in milliseconds from request start to response completion.",
    )
    completed_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when the API response completed. "
        "Derived from timestamp + duration_ms.",
    )
    provider: str | None = Field(
        default=None,
        description="The LLM provider that served this request (anthropic, openai, unknown).",
    )
    model: str | None = Field(
        default=None,
        description="Model identifier (e.g. claude-opus-4-6, gpt-4o).",
    )
    client_version: str | None = Field(
        default=None,
        description="Client CLI/runtime version for the calling tool (e.g. Claude Code 2.1.39).",
    )
    tokens: ConfluxTokens | None = Field(
        default=None,
        description="Normalized token counts across providers.",
    )
    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full tool definitions available to the model for this API call. "
        "Each element is the complete tool object from the provider request body.",
    )
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Input messages from the request "
        "(system, user, tool results, prior assistant turns).",
    )
    output: list[dict[str, Any]] = Field(
        default_factory=list,
        description="The assistant's output messages extracted from the API response "
        "(text, tool calls, etc.).",
    )
    hyperparameters: ConfluxHyperparameters | None = Field(
        default=None,
        description="Normalized generation hyperparameters extracted from the request body.",
    )
    is_streaming: bool | None = Field(
        default=None,
        description="Whether this API call used SSE streaming. "
        "Inferred from response Content-Type or response body format.",
    )
    ttft_ms: int | float | None = Field(
        default=None,
        description="Time to first token in milliseconds. Only present for streaming API calls. "
        "Measured from request sent to first SSE data chunk received.",
    )
    base64: dict[str, str] | None = Field(
        default=None,
        description="Raw base64-encoded artifacts captured by the proxy. "
        "Keys: request_body, response_body, provider_usage. "
        "May be gzip or zstd compressed before encoding.",
    )


CustomDatasetT = TypeVar(
    "CustomDatasetT",
    bound=SingleTurn
    | MultiTurn
    | RandomPool
    | MooncakeTrace
    | BailianTrace
    | ConfluxRecord,
)
"""A union type of all custom data types."""
