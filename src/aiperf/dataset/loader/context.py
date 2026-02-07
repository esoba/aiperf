# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from aiperf.common.config import UserConfig
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation, Turn
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

if TYPE_CHECKING:
    pass

_MODEL_SELECTION_MAP: dict[str, type] = {
    "round_robin": RoundRobinModelSelectionStrategy,
    "random": RandomModelSelectionStrategy,
    "shuffle": ShuffleModelSelectionStrategy,
}


class LoaderContext(AIPerfLoggerMixin):
    """Shared dependencies and finalization logic for all dataset loaders.

    Built once by DatasetManager and passed to whatever loader is created.
    Loaders access shared state (prompt_generator, tokenizer, etc.) and
    finalization behavior (finalize_turn, finalize_conversation) through this.

    Args:
        config: User configuration.
        tokenizer: Tokenizer for token counting and prompt generation.
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # Shared generators
        self.session_id_generator = SessionIDGenerator(seed=config.input.random_seed)
        self.prompt_generator = PromptGenerator(config.input.prompt, tokenizer)

        # Finalization state
        self.model_selector: ModelSelectionStrategyProtocol = (
            self._create_model_selector()
        )
        self.output_tokens_sampler = OutputTokensSampler(config)
        self._seq_distribution = config.input.prompt.get_sequence_distribution()
        self._turn_sequence_cache: dict[int, tuple[int, int]] = {}

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

    # ---- Finalization ----

    def finalize_turn(self, turn: Turn) -> None:
        """Finalize a turn by setting model name and max_tokens.

        Args:
            turn: The turn to finalize.
        """
        turn.model = self.model_selector.select(turn)
        self._set_max_tokens(turn)
        self._turn_sequence_cache.pop(id(turn), None)

    def finalize_conversation(
        self, conversation: Conversation, session_index: int
    ) -> None:
        """Finalize a conversation by injecting context prompts.

        Args:
            conversation: Conversation to finalize.
            session_index: Index of this conversation in the dataset.
        """
        self._inject_context_prompt(conversation, session_index)

    def get_turn_sequence_lengths(self, turn_id: int) -> tuple[int, int]:
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

    @property
    def seq_distribution(self):
        """The sequence distribution for ISL/OSL pairing, or None."""
        return self._seq_distribution

    @property
    def prefix_prompt_enabled(self) -> bool:
        """Whether prefix prompts are configured."""
        return self.config.input.prompt.prefix_prompt.length > 0

    # ---- Private ----

    def _set_max_tokens(self, turn: Turn) -> None:
        """Set max_tokens using sequence distribution or output config.

        Args:
            turn: The turn to set max_tokens on.
        """
        if self._seq_distribution is not None:
            _, osl = self.get_turn_sequence_lengths(id(turn))
            turn.max_tokens = osl
        else:
            sampled = self.output_tokens_sampler.sample()
            if sampled is not None:
                turn.max_tokens = sampled

    def _inject_context_prompt(
        self, conversation: Conversation, session_index: int
    ) -> None:
        """Inject shared system and user context prompts into a conversation.

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
