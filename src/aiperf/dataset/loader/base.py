# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.common.config import UserConfig
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation, Turn
from aiperf.common.models.dataset_models import ConversationMetadata
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.model_selection_strategies import (
    ModelSelectionStrategyProtocol,
    RandomModelSelectionStrategy,
    RoundRobinModelSelectionStrategy,
    ShuffleModelSelectionStrategy,
)
from aiperf.dataset.output_tokens_sampler import OutputTokensSampler
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.dataset.protocols import DatasetBackingStoreProtocol

# Map config enum values to strategy classes
_MODEL_SELECTION_MAP: dict[str, type] = {
    "round_robin": RoundRobinModelSelectionStrategy,
    "random": RandomModelSelectionStrategy,
    "shuffle": ShuffleModelSelectionStrategy,
}


class BaseDatasetLoader(AIPerfLoggerMixin, ABC):
    """Root base class for all dataset loaders.

    Absorbs turn finalization, model selection, context prompt injection,
    and output token sampling that previously lived in BaseDatasetComposer.

    Args:
        config: User configuration.
        tokenizer: Tokenizer for token counting/generation.
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer, **kwargs: Any) -> None:
        self.config = config
        self.tokenizer = tokenizer
        super().__init__(config=config, tokenizer=tokenizer, **kwargs)

        self.session_id_generator = SessionIDGenerator(seed=config.input.random_seed)
        self.prompt_generator = PromptGenerator(config.input.prompt, tokenizer)
        self.output_tokens_sampler = OutputTokensSampler(config)
        self.model_selector: ModelSelectionStrategyProtocol = (
            self._create_model_selector()
        )

        # Sequence distribution for ISL/OSL pairing
        self._seq_distribution = config.input.prompt.get_sequence_distribution()
        self._turn_sequence_cache: dict[int, tuple[int, int]] = {}

        # Metadata accumulated during load()
        self._loaded_metadata: list[ConversationMetadata] = []
        self._conversation_index: int = 0

    @property
    def loaded_metadata(self) -> list[ConversationMetadata]:
        """Metadata accumulated during load() for each conversation, in insertion order."""
        return self._loaded_metadata

    def _create_model_selector(self) -> ModelSelectionStrategyProtocol:
        """Create a model selection strategy from the endpoint config."""
        strategy_name = str(self.config.endpoint.model_selection_strategy)
        model_names = self.config.endpoint.model_names
        cls = _MODEL_SELECTION_MAP.get(strategy_name)
        if cls is None:
            raise ValueError(
                f"Unknown model selection strategy: {strategy_name}. "
                f"Available: {list(_MODEL_SELECTION_MAP.keys())}"
            )
        return cls(model_names=model_names)

    @abstractmethod
    async def load(self, store: DatasetBackingStoreProtocol) -> None:
        """Load conversations and stream them directly into the backing store.

        Args:
            store: Backing store to write conversations into.
        """
        ...

    @classmethod
    @abstractmethod
    def can_load(
        cls,
        data: dict[str, Any] | None = None,
        filename: str | Path | None = None,
    ) -> bool:
        """Check if this loader can handle the given data format.

        Args:
            data: Optional dictionary from a single JSONL line.
            filename: Optional file/directory path.

        Returns:
            True if this loader can handle the input.
        """
        ...

    @classmethod
    @abstractmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred sampling strategy for this loader.

        Returns:
            The preferred DatasetSamplingStrategy.
        """
        ...

    # ---- Turn finalization (from BaseDatasetComposer) ----

    def _finalize_turn(self, turn: Turn) -> None:
        """Finalize a turn by setting model name and max_tokens.

        Args:
            turn: The turn to finalize.
        """
        turn.model = self.model_selector.select(turn)
        self._set_max_tokens(turn)
        self._clear_turn_cache(id(turn))

    def _set_max_tokens(self, turn: Turn) -> None:
        """Set max_tokens using sequence distribution or output config.

        Args:
            turn: The turn to set max_tokens on.
        """
        if self._seq_distribution is not None:
            _, osl = self._get_turn_sequence_lengths(id(turn))
            turn.max_tokens = osl
        else:
            sampled = self.output_tokens_sampler.sample()
            if sampled is not None:
                turn.max_tokens = sampled

    def _get_turn_sequence_lengths(self, turn_id: int) -> tuple[int, int]:
        """Get or sample ISL/OSL pair for a turn, with caching for consistency.

        Args:
            turn_id: Unique identifier for the turn (typically id(turn)).

        Returns:
            Tuple of (input_seq_len, output_seq_len).
        """
        if turn_id in self._turn_sequence_cache:
            return self._turn_sequence_cache[turn_id]

        if self._seq_distribution is None:
            seq_lengths = (
                self.config.input.prompt.input_tokens.mean,
                self.config.input.prompt.output_tokens.mean
                or max(128, self.config.input.prompt.input_tokens.mean // 2),
            )
        else:
            seq_lengths = self._seq_distribution.sample()

        self._turn_sequence_cache[turn_id] = seq_lengths
        return seq_lengths

    def _clear_turn_cache(self, turn_id: int) -> None:
        """Clear cached sequence lengths for a turn.

        Args:
            turn_id: Turn identifier to remove from cache.
        """
        self._turn_sequence_cache.pop(turn_id, None)

    # ---- Conversation finalization (streaming) ----

    def _finalize_conversation(self, conversation: Conversation) -> None:
        """Finalize a single conversation by injecting context prompts.

        Uses self._conversation_index as the session index, then increments it.

        Args:
            conversation: Conversation to finalize.
        """
        self._inject_context_prompt(conversation, self._conversation_index)
        self._conversation_index += 1

    def _inject_context_prompt(
        self, conversation: Conversation, session_index: int
    ) -> None:
        """Inject shared system and user context prompts into a single conversation.

        Args:
            conversation: Conversation to inject prompts into.
            session_index: Index of this conversation in the dataset.
        """
        config = self.config.input.prompt.prefix_prompt
        has_shared_system = config.shared_system_prompt_length is not None
        has_user_context = config.user_context_prompt_length is not None

        if not (has_shared_system or has_user_context):
            return

        if has_shared_system:
            conversation.system_message = self._shared_system_prompt
            self.trace(
                lambda conv=conversation: f"Set system_message on conversation {conv.session_id}"
            )

        if has_user_context:
            user_context = self.prompt_generator.generate_user_context_prompt(
                session_index
            )
            conversation.user_context_message = user_context
            self.trace(
                lambda idx=session_index,
                conv=conversation: f"Set user_context_message for session {idx} "
                f"(conversation {conv.session_id})"
            )

    @cached_property
    def _shared_system_prompt(self) -> str | None:
        """Lazily compute the shared system prompt (same for all conversations)."""
        config = self.config.input.prompt.prefix_prompt
        if config.shared_system_prompt_length is not None:
            return self.prompt_generator.get_shared_system_prompt()
        return None

    async def _finalize_and_store(
        self,
        conversation: Conversation,
        store: DatasetBackingStoreProtocol,
        *,
        finalize_turns: bool = True,
    ) -> None:
        """Finalize a conversation and stream it into the backing store.

        Args:
            conversation: Conversation to finalize and store.
            store: Backing store to write to.
            finalize_turns: Whether to finalize turns (model name, max_tokens).
                Set to False when turns are already finalized (e.g. synthetic loaders).
        """
        if finalize_turns:
            for turn in conversation.turns:
                self._finalize_turn(turn)

        self._finalize_conversation(conversation)
        await store.add_conversation(conversation.session_id, conversation)
        self._loaded_metadata.append(conversation.metadata())

    @property
    def prefix_prompt_enabled(self) -> bool:
        """Whether prefix prompts are configured."""
        return self.config.input.prompt.prefix_prompt.length > 0
