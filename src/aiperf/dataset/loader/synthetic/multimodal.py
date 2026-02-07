# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.common.config import UserConfig
from aiperf.common.models import Conversation, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader.synthetic.base import BaseSyntheticLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.dataset.protocols import DatasetBackingStoreProtocol


class SyntheticMultiModalLoader(BaseSyntheticLoader):
    """Synthetic dataset loader that generates multi-turn conversations.

    Generates conversations with synthetic text, image, audio, and video payloads
    using configurable distributions. Supports variable turn counts and delays.
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer, **kwargs: Any) -> None:
        super().__init__(config=config, tokenizer=tokenizer, **kwargs)

        if (
            not self.include_prompt
            and not self.include_image
            and not self.include_audio
        ):
            raise ValueError(
                "All synthetic data are disabled. "
                "Please enable at least one of prompt, image, or audio by "
                "setting the mean to a positive value."
            )

    @classmethod
    def can_load(
        cls,
        data: dict[str, Any] | None = None,
        filename: str | Path | None = None,
    ) -> bool:
        """Synthetic loaders don't load from files."""
        return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred sampling strategy for synthetic datasets."""
        return DatasetSamplingStrategy.SHUFFLE

    async def load(self, store: DatasetBackingStoreProtocol) -> None:
        """Generate a synthetic multi-modal conversation dataset, streaming to store.

        Args:
            store: Backing store to write conversations into.
        """
        for _ in range(self.config.input.conversation.num_dataset_entries):
            conversation = Conversation(session_id=self.session_id_generator.next())

            num_turns = self._turn_sampler_rng.sample_positive_normal_integer(
                self.config.input.conversation.turn.mean,
                self.config.input.conversation.turn.stddev,
            )
            self.logger.debug("Creating conversation with %d turns", num_turns)

            for turn_idx in range(num_turns):
                turn = self._create_turn(is_first=(turn_idx == 0))
                conversation.turns.append(turn)

            await self._finalize_and_store(conversation, store, finalize_turns=False)

    def _create_turn(self, is_first: bool) -> Turn:
        """Create a turn with synthetic multi-modal payloads.

        Args:
            is_first: Whether this is the first turn in the conversation.

        Returns:
            A Turn object with synthetic payloads.
        """
        turn = Turn()

        if self.include_prompt:
            turn.texts.append(self._generate_text_payloads(turn, is_first))
        if self.include_image:
            turn.images.append(self._generate_image_payloads())
        if self.include_audio:
            turn.audios.append(self._generate_audio_payloads())
        if self.include_video:
            turn.videos.append(self._generate_video_payloads())

        if not is_first and self.config.input.conversation.turn.delay.mean > 0:
            delay = self._delay_sampler_rng.sample_positive_normal_integer(
                self.config.input.conversation.turn.delay.mean,
                self.config.input.conversation.turn.delay.stddev,
            )
            turn.delay = delay * self.config.input.conversation.turn.delay.ratio

        if not turn.texts and not turn.images and not turn.audios and not turn.videos:
            self.logger.warning(
                "There were no synthetic payloads generated. "
                "Please enable at least one of prompt, image, or audio by "
                "setting the mean to a positive value."
            )

        self._finalize_turn(turn)

        return turn
