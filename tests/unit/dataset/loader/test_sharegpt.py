# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.models import Conversation
from aiperf.dataset.loader import ShareGPTLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


class TestShareGPTLoader:
    """Test suite for ShareGPTLoader class."""

    @pytest.fixture
    def sharegpt_file(self, tmp_path):
        """Create a temporary ShareGPT JSON file."""

        def _create(dataset):
            filepath = tmp_path / "sharegpt.json"
            filepath.write_text(json.dumps(dataset))
            return str(filepath)

        return _create

    @pytest.fixture
    def user_config(self):
        return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))

    @pytest.fixture
    def mock_tokenizer(self, mock_tokenizer_cls):
        return mock_tokenizer_cls.from_pretrained("test-model")

    def test_initialization(self, sharegpt_file, user_config, mock_tokenizer):
        """Test initialization of ShareGPTLoader."""
        filepath = sharegpt_file([])
        loader = ShareGPTLoader(
            filename=filepath, config=user_config, tokenizer=mock_tokenizer
        )
        assert loader.tokenizer is not None
        assert loader.config is not None
        assert loader.filename == filepath

    def test_convert_to_conversations(self, sharegpt_file, user_config, mock_tokenizer):
        """Test converting single entry dataset to conversations."""
        dataset = [
            {
                "conversations": [
                    {"value": "Hello how are you"},
                    {"value": "This is test output"},
                ]
            }
        ]
        filepath = sharegpt_file(dataset)
        loader = ShareGPTLoader(
            filename=filepath, config=user_config, tokenizer=mock_tokenizer
        )
        data = loader.parse_and_validate()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        assert isinstance(conversations[0], Conversation)

        turn = conversations[0].turns[0]
        assert turn.texts[0].contents[0] == "Hello how are you"
        assert turn.max_tokens == len(["This", "is", "test", "output"])

    def test_convert_to_conversations_validation(
        self, sharegpt_file, user_config, mock_tokenizer
    ):
        """Test converting multiple entries dataset to conversations with validation."""
        dataset = [
            {
                "conversations": [
                    {"value": "Hello"},  # 1 prompt token (too short)
                    {"value": "This is test output"},  # 4 completion tokens
                ]
            },
            {
                "conversations": [
                    {"value": "Hello how are you"},  # 4 prompt tokens
                    {"value": "This is test output"},  # 4 completion tokens
                ]
            },
            {
                "conversations": [
                    {"value": "Hello how are you"},  # 4 prompt tokens
                    {"value": "This"},  # 1 completion token (too short)
                ]
            },
        ]
        filepath = sharegpt_file(dataset)
        loader = ShareGPTLoader(
            filename=filepath, config=user_config, tokenizer=mock_tokenizer
        )
        data = loader.parse_and_validate()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        assert isinstance(conversations[0], Conversation)

        turn = conversations[0].turns[0]
        assert turn.texts[0].contents[0] == "Hello how are you"
        assert turn.max_tokens == len(["This", "is", "test", "output"])

    def test_get_preferred_sampling_strategy(self):
        """Test that ShareGPTLoader returns the correct preferred sampling strategy."""
        strategy = ShareGPTLoader.get_preferred_sampling_strategy()
        assert strategy == DatasetSamplingStrategy.SEQUENTIAL
