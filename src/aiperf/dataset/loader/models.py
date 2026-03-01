# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, TypeVar

from pydantic import Field, model_validator

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
        if self.text is not None and self.texts is not None:
            raise ValueError("text and texts cannot be set together")
        if self.image is not None and self.images is not None:
            raise ValueError("image and images cannot be set together")
        if self.audio is not None and self.audios is not None:
            raise ValueError("audio and audios cannot be set together")
        if self.video is not None and self.videos is not None:
            raise ValueError("video and videos cannot be set together")
        if self.timestamp is not None and self.delay is not None:
            raise ValueError("timestamp and delay cannot be set together")
        return self

    @model_validator(mode="after")
    def validate_at_least_one_modality(self) -> "SingleTurn":
        """Ensure at least one modality is provided"""
        if not any(
            field is not None
            for field in [
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
        if self.text is not None and self.texts is not None:
            raise ValueError("text and texts cannot be set together")
        if self.image is not None and self.images is not None:
            raise ValueError("image and images cannot be set together")
        if self.audio is not None and self.audios is not None:
            raise ValueError("audio and audios cannot be set together")
        if self.video is not None and self.videos is not None:
            raise ValueError("video and videos cannot be set together")
        return self

    @model_validator(mode="after")
    def validate_at_least_one_modality(self) -> "RandomPool":
        """Ensure at least one modality is provided"""
        if not any(
            field is not None
            for field in [
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


class CodingTraceRequest(AIPerfBaseModel):
    """A single request within a coding trace.

    Represents one LLM API call in an agentic coding session, with token counts,
    KV cache block hashes, and optional nested subagent requests.
    """

    t: int | float = Field(
        description="Relative timestamp within the trace in seconds."
    )
    type: str = Field(
        description="Request type (e.g. 's' for streaming, 'n' for non-streaming, 'tool_result')."
    )
    model: str | None = Field(
        default=None, description="Model name used for this request."
    )
    input_tokens: int = Field(
        default=0, alias="in", description="Number of input tokens for this request."
    )
    output_tokens: int = Field(
        default=0, alias="out", description="Number of output tokens for this request."
    )
    hash_ids: list[int] = Field(
        default_factory=list, description="KV cache block hash IDs for prefix sharing."
    )
    input_types: list[str] = Field(
        default_factory=list, description="Types of input content blocks."
    )
    output_types: list[str] = Field(
        default_factory=list, description="Types of output content blocks."
    )
    stop: str | None = Field(
        default=None,
        description="Stop reason for this request (e.g. 'end_turn', 'tool_use').",
    )
    requests: list["CodingTraceRequest"] = Field(
        default_factory=list, description="Nested subagent requests."
    )
    is_pair_repeat: bool = Field(
        default=False,
        description="True if this request is the second of a streaming/non-streaming pair "
        "with identical hash_ids. These requests re-send the same conversation.",
    )

    model_config = {"populate_by_name": True}


class CodingTrace(AIPerfBaseModel):
    """Defines the schema for a single agentic coding trace (kv-cache-tester format).

    Each trace represents one coding session with a sequence of LLM requests,
    including nested subagent calls. Loaded from JSON files in a directory.

    Example:
    ```json
    {
      "id": "trace-001",
      "models": ["claude-sonnet-4-20250514"],
      "block_size": 64,
      "tool_tokens": 5000,
      "system_tokens": 3000,
      "requests": [
        {"t": 0.0, "type": "init", "in": 1000, "out": 500, "hash_ids": [1, 2, 3], ...}
      ]
    }
    ```
    """

    type: Literal[CustomDatasetType.CODING_TRACE] = CustomDatasetType.CODING_TRACE

    id: str = Field(description="Unique identifier for the coding trace.")
    models: list[str] = Field(
        default_factory=list, description="Model names used in this trace."
    )
    block_size: int = Field(default=64, description="KV cache block size in tokens.")
    tool_tokens: int = Field(
        default=0, description="Estimated token count for tool definitions."
    )
    system_tokens: int = Field(
        default=0, description="Estimated token count for system prompt."
    )
    requests: list[CodingTraceRequest] = Field(
        default_factory=list, description="Sequence of LLM requests in this trace."
    )

    @model_validator(mode="after")
    def validate_has_requests(self) -> "CodingTrace":
        """Ensure the trace has at least one request."""
        if not self.requests:
            raise ValueError("A coding trace must have at least one request")
        return self


class TraceStatistics(AIPerfBaseModel):
    """Derived statistics for a coding trace, computed at load time."""

    total_input_tokens: int = Field(
        description="Sum of input_tokens across all requests."
    )
    total_output_tokens: int = Field(
        description="Sum of output_tokens across all requests."
    )
    num_requests: int = Field(description="Number of requests in the trace.")
    max_input_tokens: int = Field(
        description="Maximum input_tokens in any single request."
    )
    estimated_cache_hit_ratio: float = Field(
        description="Estimated cache hit ratio based on hash_id overlap between "
        "consecutive requests (0.0 = no hits, 1.0 = all hits)."
    )


class MooncakeTrace(AIPerfBaseModel):
    """Defines the schema for Mooncake trace data.

    See https://github.com/kvcache-ai/Mooncake for more details.

    Examples:
    - Minimal: {"input_length": 10, "hash_ids": [123]}
    - With input_length: {"input_length": 10, "output_length": 4}
    - With text_input: {"text_input": "Hello world", "output_length": 4}
    - With timestamp and hash ID: {"timestamp": 1000, "input_length": 10, "hash_ids": [123]}

    Note:
    Only one of the following input combinations is allowed:
    - text_input only (uses text input directly)
    - input_length only (uses input length to generate synthetic text input)
    - input_length and hash_ids (uses input length and hash ids to generate reproducible synthetic text input)
    """

    type: Literal[CustomDatasetType.MOONCAKE_TRACE] = CustomDatasetType.MOONCAKE_TRACE

    # Exactly one of input_length or text_input must be provided
    input_length: int | None = Field(
        None,
        description="The input sequence length of a request. Required if text_input is not provided.",
    )
    text_input: str | None = Field(
        None,
        description="The actual text input for the request. Required if input_length is not provided.",
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
        """Validate that either input_length or text_input is provided."""
        if self.input_length is None and self.text_input is None:
            raise ValueError("Either 'input_length' or 'text_input' must be provided")

        if self.input_length is not None and self.text_input is not None:
            raise ValueError(
                "'input_length' and 'text_input' cannot be provided together. Use only one of them."
            )

        if self.hash_ids is not None and self.input_length is None:
            raise ValueError(
                "'hash_ids' is only allowed when 'input_length' is provided, not when 'text_input' is provided"
            )

        return self


class ClaudeCodeTraceRecord(AIPerfBaseModel):
    """A single record from a Claude Code JSONL session transcript."""

    type: str = Field(
        description="Record type: user, assistant, system, progress, etc."
    )
    message: dict[str, Any] | None = Field(
        default=None, description="Message payload with content, usage, model, etc."
    )
    session_id: str | None = Field(
        default=None, alias="sessionId", description="Claude Code session identifier."
    )
    timestamp: str | None = Field(
        default=None, description="ISO timestamp of the record."
    )
    request_id: str | None = Field(
        default=None,
        alias="requestId",
        description="Groups assistant records from the same API response.",
    )
    uuid: str | None = Field(
        default=None, description="Unique identifier for this record."
    )
    parent_uuid: str | None = Field(
        default=None,
        alias="parentUuid",
        description="Parent record UUID for threading.",
    )

    model_config = {"populate_by_name": True}


class ClaudeCodeApiCall(AIPerfBaseModel):
    """A reconstructed API call from grouped trace records."""

    user_content: str | list[dict[str, Any]] = Field(
        description="User message content (string or content blocks)."
    )
    assistant_content: list[dict[str, Any]] = Field(
        description="Assistant response content blocks (text, tool_use, thinking)."
    )
    model: str | None = Field(default=None, description="Model used for this API call.")
    input_tokens: int = Field(
        default=0, description="Input token count from usage data."
    )
    output_tokens: int = Field(
        default=0, description="Output token count from usage data."
    )
    cache_creation_input_tokens: int = Field(
        default=0, description="Cache creation input tokens from usage data."
    )
    cache_read_input_tokens: int = Field(
        default=0, description="Cache read input tokens from usage data."
    )
    timestamp_ms: float | None = Field(
        default=None, description="Timestamp in milliseconds from the trace."
    )
    stop_reason: str | None = Field(
        default=None, description="Stop reason (end_turn, tool_use, etc.)."
    )


class ClaudeCodeTrace(AIPerfBaseModel):
    """A full Claude Code session parsed into API calls."""

    type: Literal[CustomDatasetType.CLAUDE_CODE_TRACE] = (
        CustomDatasetType.CLAUDE_CODE_TRACE
    )
    id: str = Field(description="Unique identifier for this trace.")
    session_id: str = Field(description="Claude Code session ID.")
    api_calls: list[ClaudeCodeApiCall] = Field(
        description="Reconstructed API calls from the session."
    )
    system_prompt: str | None = Field(
        default=None, description="System prompt extracted from the trace."
    )

    @model_validator(mode="after")
    def validate_has_api_calls(self) -> "ClaudeCodeTrace":
        """Ensure the trace has at least one API call."""
        if not self.api_calls:
            raise ValueError("A Claude Code trace must have at least one API call")
        return self


class ApiCaptureApiCall(AIPerfBaseModel):
    """A single real API call from an mitmproxy-style capture directory.

    Correlates a req_XXXX.json request body with its corresponding
    capture.jsonl response metadata (token counts, stop reason).
    """

    messages: list[dict] = Field(description="Full messages array from request.")
    system: list[dict] = Field(
        default_factory=list, description="System prompt blocks."
    )
    tools: list[dict] = Field(default_factory=list, description="Tool definitions.")
    model: str | None = Field(default=None, description="Model name from request.")
    max_tokens: int | None = Field(default=None, description="Max tokens from request.")
    stream: bool | None = Field(default=None, description="Stream flag from request.")
    thinking: dict | None = Field(
        default=None, description="Thinking config (e.g. {'type': 'adaptive'})."
    )
    input_tokens: int = Field(
        default=0, description="Input token count from response metadata."
    )
    output_tokens: int = Field(
        default=0, description="Output token count from response metadata."
    )
    cache_creation_input_tokens: int = Field(
        default=0, description="Cache creation input tokens from response metadata."
    )
    cache_read_input_tokens: int = Field(
        default=0, description="Cache read input tokens from response metadata."
    )
    timestamp_ms: float | None = Field(
        default=None, description="Timestamp in milliseconds from capture.jsonl."
    )
    stop_reason: str | None = Field(
        default=None, description="Stop reason from response metadata."
    )


class ApiCaptureTrace(AIPerfBaseModel):
    """One conversation thread extracted from an API capture directory.

    Groups related API calls by system prompt hash into threads.
    The parent thread has the most calls; others are subagent children.
    """

    type: Literal[CustomDatasetType.API_CAPTURE_TRACE] = (
        CustomDatasetType.API_CAPTURE_TRACE
    )
    id: str = Field(description="Unique identifier for this trace.")
    api_calls: list[ApiCaptureApiCall] = Field(
        description="Ordered API calls in this thread."
    )
    system_prompt_text: str | None = Field(
        default=None,
        description="Extracted text from system blocks for convenience.",
    )
    thread_key: str = Field(
        description="Hash-based grouping key identifying this thread."
    )

    @model_validator(mode="after")
    def validate_has_api_calls(self) -> "ApiCaptureTrace":
        """Ensure the trace has at least one API call."""
        if not self.api_calls:
            raise ValueError("An API capture trace must have at least one API call")
        return self


class ClaudeCodeSubagentLink(AIPerfBaseModel):
    """Links a subagent JSONL file to a parent session spawn point."""

    file: str = Field(description="JSONL filename for the subagent session.")
    spawn_after_api_call: int = Field(
        description="0-based index of the parent API call after which this subagent spawns."
    )
    is_background: bool = Field(
        default=False,
        description="If true, parent continued without waiting for this subagent.",
    )


class ClaudeCodeManifest(AIPerfBaseModel):
    """Manifest linking parent and subagent sessions in a directory.

    When present as _manifest.json in a trace directory, declares which
    JSONL file is the parent session and which are subagent children.
    """

    parent: str = Field(description="JSONL filename for the parent session.")
    subagents: list[ClaudeCodeSubagentLink] = Field(
        default_factory=list,
        description="Subagent sessions spawned from the parent.",
    )


CustomDatasetT = TypeVar(
    "CustomDatasetT",
    bound=SingleTurn
    | MultiTurn
    | RandomPool
    | MooncakeTrace
    | CodingTrace
    | ClaudeCodeTrace
    | ApiCaptureTrace,
)
"""A union type of all custom data types."""
