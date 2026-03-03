# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from aiperf.common.enums import MediaType
from aiperf.common.enums.enums import SubagentType
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
    # Payload mmap paths (optional, only present when raw_payload data exists)
    payload_data_file_path: Path | None = Field(
        default=None,
        description="Path to payload data file containing pre-encoded JSON bytes.",
    )
    payload_index_file_path: Path | None = Field(
        default=None,
        description="Path to payload index file for O(1) payload lookups.",
    )
    compressed_payload_data_file_path: Path | None = Field(
        default=None,
        description="Path to zstd-compressed payload data file (K8s only).",
    )
    compressed_payload_index_file_path: Path | None = Field(
        default=None,
        description="Path to zstd-compressed payload index file (K8s only).",
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


class CacheLayerSizes(AIPerfBaseModel):
    """Per-turn annotation of cache layer block counts.

    Describes how hash_ids decompose into L1 (global shared), L2 (session-stable),
    and L3 (conversation history) layers. Used by working set tracking to avoid
    overcounting shared L1 blocks across sessions.
    """

    l1: int = Field(default=0, description="Number of L1 (global shared) cache blocks.")
    l2: int = Field(
        default=0, description="Number of L2 (session-stable) cache blocks."
    )
    l3: int = Field(
        default=0, description="Number of L3 (conversation history) cache blocks."
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
        "Used by adaptive scale for per-period token budget enforcement.",
    )
    hash_ids: list[int] = Field(
        default_factory=list,
        description="KV cache block hash IDs for working set tracking.",
    )
    cache_layer_sizes: CacheLayerSizes | None = Field(
        default=None,
        description="Cache layer decomposition of hash_ids into L1/L2/L3 blocks.",
    )
    subagent_spawn_ids: list[str] = Field(
        default_factory=list,
        description="Spawn IDs if this turn is blocked by subagent spawns.",
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
    hash_ids: list[int] = Field(
        default_factory=list,
        description="KV cache block hash IDs for working set tracking.",
    )
    cache_layer_sizes: CacheLayerSizes | None = Field(
        default=None,
        description="Cache layer decomposition of hash_ids into L1/L2/L3 blocks.",
    )
    subagent_spawn_ids: list[str] = Field(
        default_factory=list,
        description="Spawn IDs if this turn is blocked by subagent spawns.",
    )
    raw_messages: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of complete message dicts for verbatim replay. "
        "Can be a full history (with replaces_history=True) or a delta of new "
        "messages to append. Endpoints expand these directly into the messages list.",
    )
    replaces_history: bool = Field(
        default=False,
        description="When true, this turn replaces all prior conversation history. "
        "Used after context-loss events (restart, compression, thinking strip) "
        "so the worker clears its accumulated turn_list before appending this turn.",
    )
    raw_payload: dict[str, Any] | None = Field(
        default=None,
        description="Complete pre-built API request payload for verbatim replay. "
        "When set, bypasses all endpoint payload construction (format_payload) "
        "and sends this dict directly to the transport.",
    )

    def metadata(self) -> TurnMetadata:
        """Get the metadata of the turn."""
        return TurnMetadata(
            timestamp_ms=self.timestamp,
            delay_ms=self.delay,
            input_tokens=self.input_tokens,
            hash_ids=self.hash_ids,
            cache_layer_sizes=self.cache_layer_sizes,
            subagent_spawn_ids=list(self.subagent_spawn_ids),
        )

    def copy_with_stripped_media(self) -> "Turn":
        """Create a copy of this turn with multimodal data replaced by placeholders.

        This preserves text data (needed for tokenization) but replaces potentially
        large image/audio/video contents with small placeholder strings. This is
        more efficient than a full deep copy followed by stripping.

        Returns:
            A new Turn with stripped multimodal contents.
        """
        return Turn(
            model=self.model,
            role=self.role,
            timestamp=self.timestamp,
            delay=self.delay,
            max_tokens=self.max_tokens,
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
            hash_ids=list(self.hash_ids),
            cache_layer_sizes=self.cache_layer_sizes,
            subagent_spawn_ids=list(self.subagent_spawn_ids),
            raw_messages=self.raw_messages,
            replaces_history=self.replaces_history,
            raw_payload=None,
        )


class SubagentSpawnInfo(AIPerfBaseModel):
    """Describes a subagent spawn point linking parent to child conversations.

    When a parent conversation spawns subagents, the parent pauses at
    the spawn turn and resumes at join_turn_index after all children complete.
    Children are separate Conversations with independent hash_ids and sessions.
    """

    spawn_id: str = Field(
        description="Subagent spawn identifier, e.g. 's0'.",
    )
    child_conversation_ids: list[str] = Field(
        description="Conversation IDs of child subagent sessions to start.",
    )
    join_turn_index: int = Field(
        description="Parent turn index to resume after all children complete.",
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
    subagent_type: SubagentType | None = Field(
        default=None,
        description="Type of subagent (EXPLORE, GENERAL, PLAN). None for root conversations.",
    )
    parent_conversation_id: str | None = Field(
        default=None,
        description="Template session_id of the parent conversation. None for root conversations.",
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


class Conversation(AIPerfBaseModel):
    """A dataset representation of a full conversation.

    A conversation is a sequence of turns between a user and an endpoint,
    and it contains the session ID and all the turns that consists the conversation.
    """

    session_id: str = Field(
        default="", description="Unique identifier for the conversation."
    )
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
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="Tool definitions to include in API requests for this conversation.",
    )
    agent_depth: int = Field(
        default=0,
        description="Nesting depth of this conversation. 0=root, 1=child, 2=grandchild, etc.",
    )
    subagent_type: SubagentType | None = Field(
        default=None,
        description="Type of subagent (EXPLORE, GENERAL, PLAN). None for root conversations.",
    )
    parent_conversation_id: str | None = Field(
        default=None,
        description="Template session_id of the parent conversation. None for root conversations.",
    )
    subagent_spawns: list[SubagentSpawnInfo] = Field(
        default_factory=list,
        description="Subagent spawn points linking to child conversations.",
    )
    discard_responses: bool = Field(
        default=False,
        description="When true, worker discards server responses instead of storing "
        "them in turn history (except for the final turn). Used in verbatim trace "
        "replay where each turn already carries its own context.",
    )

    def metadata(self) -> ConversationMetadata:
        """Get the metadata of the conversation."""
        turn_metas = [turn.metadata() for turn in self.turns]
        return ConversationMetadata(
            conversation_id=self.session_id,
            turns=turn_metas,
            subagent_spawns=self.subagent_spawns,
            agent_depth=self.agent_depth,
            subagent_type=self.subagent_type,
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
