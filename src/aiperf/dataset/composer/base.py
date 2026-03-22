# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.enums import ConversationContextMode, ModelSelectionStrategy
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.audio import AudioGenerator
from aiperf.dataset.generator.image import ImageGenerator
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.generator.video import VideoGenerator


class BaseDatasetComposer(AIPerfLoggerMixin, ABC):
    def __init__(self, config: UserConfig, tokenizer: Tokenizer | None, **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        super().__init__(config=config, tokenizer=tokenizer, **kwargs)

        # Create generators (prompt generator requires a tokenizer)
        self.prompt_generator: PromptGenerator | None = (
            PromptGenerator(config.input.prompt, tokenizer) if tokenizer else None
        )
        self.image_generator = ImageGenerator(config.input.image)
        self.audio_generator = AudioGenerator(config.input.audio)
        self.video_generator = VideoGenerator(config.input.video)

        self._model_selector_rng = rng.derive("composer.turn.model_selection")
        self._max_tokens_rng = rng.derive("composer.turn.max_tokens")

        self.turn_count = 0

        # Initialize sequence distribution
        self._seq_distribution = config.input.prompt.get_sequence_distribution()

        # Cache for turn-level sequence lengths to ensure ISL/OSL pairing consistency
        self._turn_sequence_cache: dict[int, tuple[int, int]] = {}

    @abstractmethod
    def create_dataset(self) -> Iterator[Conversation]:
        """Create conversation objects from the given configuration.

        Yields Conversation objects one at a time so callers can stream
        them directly to the backing store without materializing the full list.
        """
        ...

    def get_default_context_mode(self) -> ConversationContextMode | None:
        """Dataset-level default context mode inferred by the composer or its loader.

        Override in subclasses that delegate to a loader with format-specific defaults.
        Returns None to fall through to the global DELTAS_WITHOUT_RESPONSES default.
        """
        return None

    # TODO: This can be refactored to be similar to the DatasetSamplingStrategyProtocol in order
    # to allow for more flexible model selection strategies in the future.
    def _select_model_name(self) -> str:
        if (
            self.config.endpoint.model_selection_strategy
            == ModelSelectionStrategy.RANDOM
        ):
            return self._model_selector_rng.choice(self.config.endpoint.model_names)
        elif (
            self.config.endpoint.model_selection_strategy
            == ModelSelectionStrategy.ROUND_ROBIN
        ):
            model_name = self.config.endpoint.model_names[
                self.turn_count % len(self.config.endpoint.model_names)
            ]
            self.turn_count += 1
            return model_name
        else:
            raise ValueError(
                f"Invalid model selection strategy: {self.config.endpoint.model_selection_strategy}."
            )

    def _get_turn_sequence_lengths(self, turn_id: int) -> tuple[int, int]:
        """Get or sample ISL/OSL pair for a specific turn, ensuring consistency.

        This method caches the sequence lengths per turn to ensure that the same
        ISL/OSL pair is used for both prompt generation and max_tokens setting.

        Args:
            turn_id: Unique identifier for the turn

        Returns:
            Tuple of (input_seq_len, output_seq_len)
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
        """Clear cached sequence lengths for a specific turn.

        Args:
            turn_id: Turn identifier to remove from cache
        """
        self._turn_sequence_cache.pop(turn_id, None)

    def _set_max_tokens(self, turn: Turn) -> None:
        """Set max_tokens for the turn based on the sequence distribution or output configuration.

        Args:
            turn: The turn object to finalize.
        """
        if self._seq_distribution is not None:
            # Use cached sequence distribution to get OSL (ensures ISL/OSL pairing consistency)
            turn_id = id(turn)
            _, osl = self._get_turn_sequence_lengths(turn_id)
            turn.max_tokens = osl
        else:
            output_tokens_config = self.config.input.prompt.output_tokens
            if output_tokens_config.mean is not None:
                stddev = output_tokens_config.stddev
                turn.max_tokens = self._max_tokens_rng.sample_positive_normal_integer(
                    output_tokens_config.mean, stddev
                )

    def _finalize_turn(self, turn: Turn) -> None:
        """Finalize a turn by populating all required metadata fields.

        This method handles:
        - Model name selection
        - Max tokens sampling based on output configuration
        - Any other turn-level metadata that needs to be set

        Args:
            turn: The turn object to finalize.
        """
        turn.model = self._select_model_name()
        self._set_max_tokens(turn)

        # Clear cached sequence lengths for this turn to free memory
        turn_id = id(turn)
        self._clear_turn_cache(turn_id)

    @property
    def prefix_prompt_enabled(self) -> bool:
        return (
            self.prompt_generator is not None
            and self.config.input.prompt.prefix_prompt.length > 0
        )

    def _finalize_conversation(
        self, conversation: Conversation, session_index: int
    ) -> None:
        """Inject context prompts and coding tool-use history into a conversation.

        Sets the system_message and user_context_message fields, which
        endpoint formatters prepend to the first turn when creating payloads.
        For coding corpus multi-turn sessions, injects tool-use conversation
        history into subsequent turns' raw_messages.

        Args:
            conversation: Conversation to finalize.
            session_index: Position of this conversation in the dataset
                (used for per-session user context prompt generation).
        """
        if self.prompt_generator is None:
            return

        config = self.config.input.prompt.prefix_prompt

        if config.shared_system_prompt_length is not None:
            prompt = self._get_shared_system_prompt()
            if prompt:
                conversation.system_message = prompt

        if config.user_context_prompt_length is not None:
            conversation.user_context_message = (
                self.prompt_generator.generate_user_context_prompt(session_index)
            )

        self._inject_coding_tool_history(conversation)

    def _get_shared_system_prompt(self) -> str | None:
        """Return the shared system prompt, computing and caching on first call."""
        if not hasattr(self, "_shared_system_prompt_cache"):
            self._shared_system_prompt_cache: str | None = (
                self.prompt_generator.get_shared_system_prompt()
            )
        return self._shared_system_prompt_cache

    # -- Coding tool-use ISL injection --

    def _inject_coding_tool_history(self, conversation: Conversation) -> None:
        """Pre-generate cumulative tool-use conversation history for coding multi-turn.

        When ``--pre-generate-responses`` is enabled with coding corpus,
        builds cumulative ``raw_messages`` on each turn.  Each turn's
        ``input_length`` is delta-compressed (new tokens for that turn only),
        so the total ISL grows as history accumulates::

            Turn 0 ISL = input_length₀
            Turn 1 ISL = input_length₀ + max_tokens₀ + input_length₁
            Turn 2 ISL = input_length₀ + max_tokens₀ + input_length₁ + max_tokens₁ + input_length₂

        Assistant responses are generated at exactly ``max_tokens`` of the
        previous turn, mocking what the real LLM would have output as
        tool-use conversations (tool_calls + tool results + summary text).

        The conversation context mode is set to ``MESSAGE_ARRAY_WITH_RESPONSES``
        so the session manager sends each turn's ``raw_messages`` as-is and
        discards live LLM responses.

        Without ``--pre-generate-responses``, the normal multi-turn flow is
        used: turns accumulate and real LLM responses fill the assistant role.
        """
        if not self.config.input.prompt.pre_generate_responses:
            return

        from aiperf.dataset.generator.coding_content import CodingContentGenerator

        if not isinstance(self.prompt_generator, CodingContentGenerator):
            return
        if len(conversation.turns) <= 1:
            return

        gen: CodingContentGenerator = self.prompt_generator

        # Tell the session manager to send each turn's raw_messages as-is
        # and discard live LLM responses.
        from aiperf.common.enums import ConversationContextMode

        conversation.context_mode = ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES

        # Pre-generate an assistant response for each turn (except the last)
        # sized to that turn's max_tokens — mocking the real LLM output.
        responses: list[list[dict[str, Any]]] = []
        for turn in conversation.turns[:-1]:
            budget = turn.max_tokens or 0
            if budget > 0:
                responses.append(
                    gen.generate_response(
                        budget,
                        include_assistant_text=True,
                        assistant_text_tokens=budget,
                    )
                )
            else:
                responses.append([])

        # Build cumulative raw_messages for turns 1..N
        for i in range(1, len(conversation.turns)):
            if conversation.turns[i].raw_messages is not None:
                continue

            messages: list[dict[str, Any]] = []

            # Accumulate history: user[j] + assistant_response[j] for j=0..i-1
            for j in range(i):
                user_text = self._extract_turn_text(conversation.turns[j])
                messages.append({"role": "user", "content": user_text})
                if j < len(responses) and responses[j]:
                    messages.extend(responses[j])

            # Current turn's user prompt
            user_text = self._extract_turn_text(conversation.turns[i])
            messages.append({"role": "user", "content": user_text})

            conversation.turns[i].raw_messages = messages

    @staticmethod
    def _extract_turn_text(turn: Turn) -> str:
        """Extract text content from a turn for raw_messages construction."""
        if turn.raw_messages:
            for msg in turn.raw_messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
            return ""
        return "".join(content for text in turn.texts for content in text.contents)
