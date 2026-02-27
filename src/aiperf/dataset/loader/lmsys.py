# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from typing import Any

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Conversation
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader.base_public_dataset import BasePublicDatasetLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


class LMSYSLoader(BasePublicDatasetLoader):
    """Loader for LMSYS public dataset support.

    This loader wires LMSYS into the public dataset flow and loads records from
    Hugging Face datasets.
    """

    tag = "LMSYS"
    dataset_id = "lmsys/lmsys-chat-1m"
    split = "train"

    def __init__(self, user_config: UserConfig, tokenizer: Tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.user_config = user_config
        super().__init__(user_config=user_config, tokenizer=tokenizer, **kwargs)

    async def load_dataset(self) -> list[dict[str, Any]]:
        """Load LMSYS dataset from Hugging Face datasets."""
        from datasets import load_dataset

        dataset = await asyncio.to_thread(
            load_dataset,
            self.dataset_id,
            split=self.split,
            token=os.environ.get("HF_TOKEN"),
        )
        return dataset.to_list()

    async def convert_to_conversations(
        self, dataset: list[dict[str, Any]]
    ) -> list[Conversation]:
        """Convert LMSYS records into AIPerf conversations."""
        raise NotImplementedError(
            "Implement LMSYS schema mapping to Conversation objects in "
            "LMSYSLoader.convert_to_conversations()."
        )

    def get_recommended_sampling_strategy(self) -> DatasetSamplingStrategy:
        """Get the recommended sampling strategy for this dataset."""
        return DatasetSamplingStrategy.SEQUENTIAL
