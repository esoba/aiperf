# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class HFInstructionResponseDatasetLoader(BaseHFDatasetLoader):
    """HuggingFace dataset loader for flat instruction/response datasets.

    Converts datasets with a prompt column into single-turn AIPerf Conversations.

    Example plugins.yaml entry:

        aimo:
          class: aiperf.dataset.loader.hf_instruction_response:HFInstructionResponseDatasetLoader
          metadata:
            hf_dataset_name: AI-MO/NuminaMath-TIR
            prompt_column: problem
    """

    def __init__(
        self,
        run: BenchmarkRun,
        prompt_column: str,
        **kwargs,
    ) -> None:
        self.prompt_column = prompt_column
        super().__init__(run=run, **kwargs)

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert each dataset row into a single-turn Conversation."""
        dataset = data["dataset"]
        conversations = []
        skipped = 0

        for row in dataset:
            prompt = row.get(self.prompt_column)
            if not prompt or not str(prompt).strip():
                skipped += 1
                continue

            conversations.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=[
                        Turn(
                            texts=[Text(contents=[str(prompt)])],
                        )
                    ],
                )
            )

        self.debug(
            lambda: f"Converted {len(conversations)} rows (skipped {skipped} empty)"
        )
        return conversations
