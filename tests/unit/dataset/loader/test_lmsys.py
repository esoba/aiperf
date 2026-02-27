# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.dataset.loader import LMSYSLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


@pytest.mark.asyncio
class TestLMSYSLoader:
    """Test suite for LMSYSLoader class"""

    @pytest.fixture
    async def lmsys_loader(self, user_config, mock_tokenizer_cls):
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        return LMSYSLoader(user_config, tokenizer)

    async def test_initialization(self, lmsys_loader: LMSYSLoader):
        """Test initialization of LMSYSLoader"""
        assert lmsys_loader.tokenizer is not None
        assert lmsys_loader.user_config is not None
        assert lmsys_loader.tag == "LMSYS"
        assert lmsys_loader.dataset_id == "lmsys/lmsys-chat-1m"
        assert lmsys_loader.split == "train"
        assert isinstance(lmsys_loader.cache_filepath, Path)

    async def test_load_dataset_uses_hf_datasets_library(
        self, lmsys_loader: LMSYSLoader
    ):
        """Test loading LMSYS dataset from Hugging Face datasets"""
        mock_records = [{"conversation": []}]
        mock_hf_dataset = MagicMock()
        mock_hf_dataset.to_list.return_value = mock_records

        with (
            patch("datasets.load_dataset") as mock_load_dataset,
            patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
        ):
            mock_to_thread.return_value = mock_hf_dataset
            result = await lmsys_loader.load_dataset()

        mock_to_thread.assert_awaited_once()
        assert mock_to_thread.await_args.args[0] is mock_load_dataset
        assert mock_to_thread.await_args.args[1] == "lmsys/lmsys-chat-1m"
        assert mock_to_thread.await_args.kwargs["split"] == "train"
        assert result == mock_records

    async def test_convert_to_conversations_not_implemented(
        self, lmsys_loader: LMSYSLoader
    ):
        """Test LMSYS conversion raises clear implementation guidance"""
        with pytest.raises(NotImplementedError, match="Implement LMSYS schema mapping"):
            await lmsys_loader.convert_to_conversations([{"conversations": []}])

    async def test_get_recommended_sampling_strategy(self, lmsys_loader: LMSYSLoader):
        """Test that LMSYSLoader returns the correct recommended sampling strategy."""
        strategy = lmsys_loader.get_recommended_sampling_strategy()
        assert strategy == DatasetSamplingStrategy.SEQUENTIAL
