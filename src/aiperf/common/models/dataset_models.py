# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator

from aiperf.common.enums import ConversationContextMode, MediaType, PrerequisiteKind
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.types import MediaTypeT
from aiperf.plugin.enums import DatasetClientStoreType, DatasetSamplingStrategy


class DatasetClientMetadata(AIPerfBaseModel):
    """Base class for dataset client access metadata.

    Uses discriminated union pattern based on client_type for extensibility.
    Workers receive this metadata to know how to access the dataset backing store.
    """

    discriminator_field: ClassVar[str] = "client_type"

    client_type: DatasetClientStoreType = Field(
        ...,
        description="The type of client store to use for dataset access.",
    )


class MemoryMapClientMetadata(DatasetClientMetadata):
    """Client metadata for memory-mapped dataset access.

    Contains paths to mmap files that workers use for zero-copy,
    O(1) conversation lookups.
    """

    client_type: DatasetClientStoreType = DatasetClientStoreType.MEMORY_MAP

    data_file_path: Path = Field(
        ...,
        description="Path to the memory-mapped data file containing serialized conversations.",
    )
    index_file_path: Path = Field(
        ...,
        description="Path to the memory-mapped index file for O(1) conversation lookups.",
    )
    conversation_count: int = Field(
        default=0,
        description="Number of conversations stored in the mmap files.",
    )
    total_size_bytes: int = Field(
        default=0,
        description="Total size of the data file in bytes.",
    )
    # Pre-compressed files for Kubernetes HTTP transfer (optional)
    compressed_data_file_path: Path | None = Field(
        default=None,
        description="Path to zstd-compressed data file for HTTP transfer (K8s only).",
    )
    compressed_index_file_path: Path | None = Field(
        default=None,
        description="Path to zstd-compressed index file for HTTP transfer (K8s only).",
    )
    compressed_size_bytes: int = Field(
        default=0,
        description="Total size of the compressed data file in bytes.",
    )


class Media(AIPerfBaseModel):
    """Base class for all media fields. Contains name and contents of the media data."""

    name: str = Field(default="", description="Name of the media field.")

    contents: list[str] = Field(
        default=[],
        description="List of media contents. Supports batched media payload in a single turn.",
    )


class Text(Media):
    """Media that contains text/prompt data."""

    media_type: ClassVar[MediaTypeT] = MediaType.TEXT


class Image(Media):
    """Media that contains image data."""

    media_type: ClassVar[MediaTypeT] = MediaType.IMAGE


class Audio(Media):
    """Media that contains audio data."""

    media_type: ClassVar[MediaTypeT] = MediaType.AUDIO


class Video(Media):
    """Media that contains video data."""

    media_type: ClassVar[MediaTypeT] = MediaType.VIDEO


class TurnPrerequisite(AIPerfBaseModel):
    """A condition that must be satisfied before a turn dispatches.

    Used by the SubagentOrchestrator to gate turn dispatch on prerequisite
    completion. Currently supports 'spawn_join' (all blocking children from
    a spawn must complete). Extensible to other gate types.
    """

    kind: PrerequisiteKind = Field(
        description="Prerequisite type.",
    )
    spawn_id: str | None = Field(
        default=None,
        description="For spawn_join: which spawn's children must complete.",
    )


class TurnMetadata(AIPerfBaseModel):
    """Metadata of a turn."""

    timestamp_ms: int | float | None = Field(
        default=None,
        description="The absolute timestamp of the turn in milliseconds.",
    )
    delay_ms: int | float | None = Field(
        default=None,
        description="The delay of the turn in the conversation (in milliseconds).",
    )
    input_tokens: int | None = Field(
        default=None,
        description="Expected input token count for this turn (from trace data). "
        "Can be used for per-period token budget enforcement.",
    )
    subagent_spawn_ids: list[str] = Field(
        default_factory=list,
        description="Spawn IDs if this turn is blocked by subagent spawns.",
    )
    prerequisites: list[TurnPrerequisite] = Field(
        default_factory=list,
        description="Conditions that must be met before this turn dispatches.",
    )


class TurnGroundTruth(AIPerfBaseModel):
    """Original capture metadata for observability and comparison.

    Stores token breakdowns, timing, and output content from the original
    API response. Never sent to the inference API — informational only.
    """

    input_cached_tokens: int | None = Field(
        default=None, description="Input tokens served from provider cache."
    )
    input_cache_write_tokens: int | None = Field(
        default=None, description="Input tokens written to provider cache."
    )
    output_tokens: int | None = Field(
        default=None, description="Output tokens generated in the original response."
    )
    output_reasoning_tokens: int | None = Field(
        default=None, description="Output tokens used for reasoning/thinking."
    )
    ttft_ms: float | None = Field(
        default=None, description="Time to first token in milliseconds."
    )
    duration_ms: float | None = Field(
        default=None, description="Total request duration in milliseconds."
    )
    is_streaming: bool | None = Field(
        default=None, description="Whether the original request used streaming."
    )


class Turn(AIPerfBaseModel):
    """A dataset representation of a single turn within a conversation.

    A turn is a single interaction between a user and an AI assistant,
    and it contains timestamp, delay, and raw data that user sends in each turn.
    """

    model: str | None = Field(default=None, description="Model name used for the turn.")
    role: str | None = Field(default=None, description="Role of the turn.")
    timestamp: int | float | None = Field(
        default=None,
        description="The absolute timestamp of the turn in milliseconds.",
    )
    delay: int | float | None = Field(
        default=None,
        description="The delay of the turn in the conversation (in milliseconds).",
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum number of tokens to generate for this turn."
    )
    raw_messages: list[dict[str, Any]] | None = Field(
        default=None,
        description="Pre-formatted OpenAI-compatible messages array. "
        "When set, bypasses normal turn-based message construction in endpoints.",
    )
    raw_tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="Pre-formatted OpenAI-compatible tool definitions. "
        "When set alongside raw_messages, injected into the API payload.",
    )
    texts: list[Text] = Field(
        default=[], description="Collection of text data in each turn."
    )
    images: list[Image] = Field(
        default=[], description="Collection of image data in each turn."
    )
    audios: list[Audio] = Field(
        default=[], description="Collection of audio data in each turn."
    )
    videos: list[Video] = Field(
        default=[], description="Collection of video data in each turn."
    )
    input_tokens: int | None = Field(
        default=None,
        description="Expected input token count for this turn (from trace data).",
    )
    subagent_spawn_ids: list[str] = Field(
        default_factory=list,
        description="Spawn IDs if this turn is blocked by subagent spawns.",
    )
    prerequisites: list[TurnPrerequisite] = Field(
        default_factory=list,
        description="Conditions that must be met before this turn dispatches.",
    )
    extra_params: dict[str, Any] | None = Field(
        default=None,
        description="Per-turn hyperparameter overrides merged into the API payload "
        "after format_payload(). Populated from dataset capture metadata.",
    )
    ground_truth: TurnGroundTruth | None = Field(
        default=None,
        description="Original capture metadata (token breakdown, timing, output) "
        "for observability. Never sent to the inference API.",
    )

    def metadata(self) -> TurnMetadata:
        """Get the metadata of the turn."""
        return TurnMetadata(
            timestamp_ms=self.timestamp,
            delay_ms=self.delay,
            input_tokens=self.input_tokens,
            subagent_spawn_ids=list(self.subagent_spawn_ids),
            prerequisites=list(self.prerequisites),
        )

    def copy_with_stripped_media(self) -> "Turn":
        """Create a copy of this turn with multimodal data replaced by placeholders.

        This preserves text data (needed for tokenization) and raw messages/tools
        (needed for API payload reconstruction) but replaces potentially large
        image/audio/video contents with small placeholder strings. This is
        more efficient than a full deep copy followed by stripping.

        Returns:
            A new Turn with stripped multimodal contents and messages.
        """
        return Turn(
            model=self.model,
            role=self.role,
            timestamp=self.timestamp,
            delay=self.delay,
            max_tokens=self.max_tokens,
            raw_messages=list(self.raw_messages)
            if self.raw_messages is not None
            else None,
            raw_tools=list(self.raw_tools) if self.raw_tools is not None else None,
            texts=[Text(name=t.name, contents=list(t.contents)) for t in self.texts],
            images=[
                Image(
                    name=img.name,
                    contents=[f"image_{i}" for i in range(len(img.contents))],
                )
                for img in self.images
            ],
            audios=[
                Audio(
                    name=aud.name,
                    contents=[f"audio_{i}" for i in range(len(aud.contents))],
                )
                for aud in self.audios
            ],
            videos=[
                Video(
                    name=vid.name,
                    contents=[f"video_{i}" for i in range(len(vid.contents))],
                )
                for vid in self.videos
            ],
            input_tokens=self.input_tokens,
            subagent_spawn_ids=list(self.subagent_spawn_ids),
            prerequisites=list(self.prerequisites),
            extra_params=dict(self.extra_params) if self.extra_params else None,
            ground_truth=self.ground_truth,
        )


class SubagentSpawnInfo(AIPerfBaseModel):
    """Describes a subagent spawn point linking parent to child conversations.

    When a parent conversation spawns subagents, blocking children must
    complete before the gated turn (declared via TurnPrerequisite) dispatches.
    Children are separate Conversations with independent hash_ids and sessions.
    """

    spawn_id: str = Field(
        description="Subagent spawn identifier, e.g. 's0'.",
    )
    child_conversation_ids: list[str] = Field(
        description="Conversation IDs of child subagent sessions to start.",
    )
    is_background: bool = Field(
        default=False,
        description="If true, parent continues without waiting for children.",
    )


class ConversationMetadata(AIPerfBaseModel):
    """Metadata of a conversation."""

    conversation_id: str = Field(
        ...,
        description="The ID of the conversation.",
    )
    turns: list[TurnMetadata] = Field(
        default_factory=list,
        description="The metadata of the turns in the conversation.",
    )
    subagent_spawns: list[SubagentSpawnInfo] = Field(
        default_factory=list,
        description="Subagent spawn points linking to child conversations.",
    )
    agent_depth: int = Field(
        default=0,
        description="Nesting depth of this conversation. 0=root, 1=child, 2=grandchild, etc.",
    )
    parent_conversation_id: str | None = Field(
        default=None,
        description="Template conversation_id of the parent conversation. None for root conversations.",
    )


class DatasetMetadata(AIPerfBaseModel):
    """Metadata of a dataset's structure.

    Contains dataset structure information (conversations, timing) used by
    timing strategies to schedule requests. Does NOT contain data access
    metadata - that's in DatasetClientMetadata (sent separately in
    DatasetConfiguredNotification).
    """

    conversations: list[ConversationMetadata] = Field(
        default_factory=list,
        description="The conversation metadata of the dataset.",
    )
    sampling_strategy: DatasetSamplingStrategy = Field(
        ...,
        description="The sampling strategy to use when choosing conversations from the dataset.",
    )
    has_timing_data: bool = Field(
        default=False,
        description="Whether the dataset has timing data (timestamps/delays in turns).",
    )
    default_context_mode: ConversationContextMode | None = Field(
        default=None,
        description="Dataset-level default for how prior turns are accumulated. "
        "Set by the loader based on dataset format semantics. "
        "Individual conversations can override this via their own context_mode field.",
    )

    @field_validator("default_context_mode")
    @classmethod
    def _reject_unimplemented_context_mode(
        cls,
        v: ConversationContextMode | None,
    ) -> ConversationContextMode | None:
        if v == ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES:
            raise ValueError(
                f"{ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES} is not yet supported"
            )
        return v

    @cached_property
    def total_turn_count(self) -> int:
        """Get the total number of turns in the dataset."""
        return sum(len(conversation.turns) for conversation in self.conversations)

    @cached_property
    def average_turn_count(self) -> float:
        """Get the average number of turns across all conversations in the dataset."""
        if len(self.conversations) == 0:
            return 0
        return self.total_turn_count / len(self.conversations)


class ConversationOrigin(AIPerfBaseModel):
    """Source traceability back to the original capture.

    Stores origin metadata so benchmark results can be linked to
    the specific capture session and client that produced the data.
    """

    source: str | None = Field(
        default=None, description="Record source: proxy, claude, codex."
    )
    client: str | None = Field(
        default=None, description="Client that produced this record: claude, codex."
    )
    client_version: str | None = Field(
        default=None, description="Client version string."
    )
    original_session_id: str | None = Field(
        default=None, description="Original session identifier from the capture."
    )
    original_request_ids: list[str] = Field(
        default_factory=list,
        description="Provider request identifiers, one per turn.",
    )


class Conversation(AIPerfBaseModel):
    """A dataset representation of a full conversation.

    A conversation is a sequence of turns between a user and an endpoint,
    and it contains the session ID and all the turns that consists the conversation.
    """

    session_id: str = Field(
        default="", description="Unique identifier for the conversation."
    )
    context_mode: ConversationContextMode | None = Field(
        default=None,
        description="How prior turns are accumulated for this conversation. "
        "When None, inherits the dataset-level default.",
    )

    @field_validator("context_mode")
    @classmethod
    def _reject_unimplemented_context_mode(
        cls,
        v: ConversationContextMode | None,
    ) -> ConversationContextMode | None:
        if v == ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES:
            raise ValueError(
                f"{ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES} is not yet supported"
            )
        return v

    turns: list[Turn] = Field(
        default=[], description="List of turns in the conversation."
    )
    system_message: str | None = Field(
        default=None,
        description="Optional shared system message prepended to the first turn. "
        "Identical across all conversations when using --shared-system-prompt-length.",
    )
    user_context_message: str | None = Field(
        default=None,
        description="Optional per-conversation user context prepended to the first turn. "
        "Unique for each conversation when using --user-context-prompt-length.",
    )
    agent_depth: int = Field(
        default=0,
        description="Nesting depth of this conversation. 0=root, 1=child, 2=grandchild, etc.",
    )
    parent_conversation_id: str | None = Field(
        default=None,
        description="Template session_id of the parent conversation. None for root conversations.",
    )
    subagent_spawns: list[SubagentSpawnInfo] = Field(
        default_factory=list,
        description="Subagent spawn points linking to child conversations.",
    )
    origin: ConversationOrigin | None = Field(
        default=None,
        description="Source traceability back to the original capture. "
        "Populated by loaders that have origin metadata (e.g. Conflux).",
    )

    def metadata(self) -> ConversationMetadata:
        """Get the metadata of the conversation."""
        turn_metas = [turn.metadata() for turn in self.turns]
        return ConversationMetadata(
            conversation_id=self.session_id,
            turns=turn_metas,
            subagent_spawns=self.subagent_spawns,
            agent_depth=self.agent_depth,
            parent_conversation_id=self.parent_conversation_id,
        )


class SessionPayloads(AIPerfBaseModel):
    """A single session, with its session ID and a list of formatted payloads (one per turn)."""

    session_id: str | None = Field(
        default=None, description="Session ID of the conversation."
    )
    payloads: list[dict[str, Any]] = Field(
        default=[],
        description="List of formatted payloads in the session (one per turn). These have been formatted for the model and endpoint.",
    )


class InputsFile(AIPerfBaseModel):
    """A list of all dataset sessions. Each session contains a list of formatted payloads (one per turn).
    This is similar to the format used by GenAI-Perf for the inputs.json file.
    """

    data: list[SessionPayloads] = Field(
        default=[], description="List of all dataset sessions."
    )
