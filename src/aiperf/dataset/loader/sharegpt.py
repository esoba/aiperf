# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.utils import load_json_str
from aiperf.dataset.loader.file.base import BaseFileLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.dataset.loader.context import LoaderContext


class ShareGPTLoader(BaseFileLoader):
    """ShareGPT dataset loader for loading and processing ShareGPT conversation data.

    This loader parses a local JSON file containing ShareGPT conversations
    (downloaded via public_datasets.py). It validates entries and converts them
    to the AIPerf conversation format.

    The loader filters conversations based on:
    - Minimum conversation length (at least 2 turns required)
    - Sequence length validation for prompt and completion tokens
    - Configurable max prompt length and total sequence length
    """

    def __init__(
        self,
        *,
        filename: str,
        ctx: LoaderContext,
        **kwargs: Any,
    ) -> None:
        super().__init__(filename=filename, ctx=ctx, **kwargs)
        self.output_tokens_mean = self.ctx.config.input.prompt.output_tokens.mean

    @classmethod
    def can_load_file(
        cls, data: dict[str, Any], filename: str | Path | None = None
    ) -> bool:
        """ShareGPT format is not auto-detected from JSONL data."""
        return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred sampling strategy for ShareGPT dataset."""
        return DatasetSamplingStrategy.SEQUENTIAL

    def parse_and_validate(self) -> dict[str, list[dict]]:
        """Parse the ShareGPT JSON file.

        Returns:
            A dictionary with a single key "entries" mapping to the dataset entries.
        """
        with open(self.filename) as f:
            content = f.read()
        dataset = load_json_str(content)
        return {"entries": dataset}

    def convert_to_conversations(
        self, data: dict[str, list[dict]]
    ) -> list[Conversation]:
        """Convert ShareGPT entries to conversations.

        Filters entries based on sequence lengths and creates single-turn
        conversations from the first two messages of each valid entry.

        Args:
            data: Parsed data with "entries" key containing ShareGPT entries.

        Returns:
            A list of valid Conversation objects.
        """
        dataset = data["entries"]
        self.info("Validating ShareGPT dataset and constructing conversation dataset")
        filtered_dataset = []
        skipped_entries = 0
        for entry in dataset:
            conversations = entry.get("conversations", [])
            if not conversations or len(conversations) < 2:
                skipped_entries += 1
                continue

            prompt, completion = conversations[0]["value"], conversations[1]["value"]
            prompt_length = len(self.ctx.tokenizer.encode(prompt))
            completion_length = len(self.ctx.tokenizer.encode(completion))

            if not self._is_valid_sequence(
                prompt_len=prompt_length,
                output_len=completion_length,
                skip_min_output_len_check=self.output_tokens_mean is not None,
            ):
                skipped_entries += 1
                continue

            turn = Turn(
                texts=[Text(contents=[prompt])],
                max_tokens=completion_length,
            )
            filtered_dataset.append(
                Conversation(
                    session_id=self.ctx.session_id_generator.next(),
                    turns=[turn],
                )
            )

        self.debug(
            lambda: f"Filtered to {len(filtered_dataset)} dataset entries out of {len(dataset)} (skipped {skipped_entries})"
        )
        return filtered_dataset

    @staticmethod
    def _is_valid_sequence(
        prompt_len: int,
        output_len: int,
        min_seq_len: int = 4,
        max_prompt_len: int = 1024,
        max_total_len: int = 2048,
        skip_min_output_len_check: bool = False,
    ) -> bool:
        """Validate a sequence based on prompt and output lengths.

        Args:
            prompt_len: The length of the prompt.
            output_len: The length of the output.
            min_seq_len: The minimum length of the sequence.
            max_prompt_len: The maximum length of the prompt.
            max_total_len: The maximum length of the total sequence.
            skip_min_output_len_check: Whether to skip the minimum output length check.

        Returns:
            True if the sequence is valid, False otherwise.
        """
        prompt_too_short = prompt_len < min_seq_len
        prompt_too_long = prompt_len > max_prompt_len
        output_too_short = (not skip_min_output_len_check) and (
            output_len < min_seq_len
        )
        combined_too_long = (prompt_len + output_len) > max_total_len

        return not (
            prompt_too_short or output_too_short or prompt_too_long or combined_too_long
        )
