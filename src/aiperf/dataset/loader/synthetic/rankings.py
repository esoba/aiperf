# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader.synthetic.base import BaseSyntheticLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


class SyntheticRankingsLoader(BaseSyntheticLoader):
    """Synthetic dataset loader for ranking tasks.

    Generates synthetic data with one query and multiple passages per entry,
    designed for ranking model evaluation and benchmarking.
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer, **kwargs: Any) -> None:
        super().__init__(config=config, tokenizer=tokenizer, **kwargs)
        self._passages_rng = rng.derive("dataset.rankings.passages")
        self._passages_token_rng = rng.derive("dataset.rankings.passages.tokens")
        self._query_token_rng = rng.derive("dataset.rankings.query.tokens")

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
        """Get the preferred sampling strategy for synthetic rankings."""
        return DatasetSamplingStrategy.SHUFFLE

    async def load(self) -> AsyncIterator[Conversation]:
        """Generate synthetic dataset for the rankings endpoint.

        Each conversation contains one turn with one query and multiple passages.

        Returns:
            AsyncIterator of finalized Conversation objects.
        """
        num_entries = self.config.input.conversation.num_dataset_entries
        num_passages_mean = self.config.input.rankings.passages.mean
        num_passages_std = self.config.input.rankings.passages.stddev

        for idx in range(num_entries):
            num_passages = self._passages_rng.sample_positive_normal_integer(
                num_passages_mean, num_passages_std
            )
            conversation = Conversation(session_id=self.session_id_generator.next())
            turn = self._create_turn(num_passages=num_passages)
            conversation.turns.append(turn)

            self._finalize_conversation(conversation, idx)
            yield conversation

    def _create_turn(self, num_passages: int) -> Turn:
        """Create a single ranking turn with one synthetic query and multiple synthetic passages.

        Args:
            num_passages: Number of passages to generate.

        Returns:
            A Turn object with query and passage texts.
        """
        turn = Turn()

        query_text = self.prompt_generator.generate_prompt(
            self.prompt_generator.calculate_num_tokens(
                self.config.input.rankings.query.prompt_token_mean,
                self.config.input.rankings.query.prompt_token_stddev,
            )
        )
        query = Text(name="query", contents=[query_text])

        # Generate passages with rankings-specific token counts (per passage)
        passages = Text(name="passages")
        for _ in range(num_passages):
            passage_text = self.prompt_generator.generate_prompt(
                self.prompt_generator.calculate_num_tokens(
                    self.config.input.rankings.passages.prompt_token_mean,
                    self.config.input.rankings.passages.prompt_token_stddev,
                )
            )
            passages.contents.append(passage_text)

        turn.texts.extend([query, passages])
        self._finalize_turn(turn)

        self.debug(
            lambda: f"[rankings] query_len={len(query_text)} chars, passages={num_passages}"
        )
        return turn
