# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for dataset manager testing.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import aiperf.endpoints  # noqa: F401  # Import to register endpoints
import aiperf.transports  # noqa: F401  # Import to register transports
from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.models import Conversation, DatasetMetadata
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import DatasetSamplingStrategy, EndpointType


@pytest.fixture
def user_config(tmp_path: Path) -> UserConfig:
    """Create a UserConfig for testing."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            streaming=False,
            url="http://localhost:8000",
        ),
        output=OutputConfig(artifact_directory=tmp_path),
    )


@pytest.fixture
def dataset_manager(
    user_config: UserConfig,
) -> DatasetManager:
    """Create a DatasetManager instance."""
    return DatasetManager(
        service_config=ServiceConfig(),
        user_config=user_config,
        service_id="test_dataset_manager",
    )


def _create_dataset_manager_with_client(
    user_config: UserConfig,
    conversations: dict[str, Conversation],
) -> DatasetManager:
    """Create a DatasetManager with a mock dataset client backed by conversations."""
    manager = DatasetManager(
        service_config=ServiceConfig(),
        user_config=user_config,
        service_id="test_dataset_manager",
    )

    async def mock_get_conversation(conversation_id: str) -> Conversation:
        if conversation_id not in conversations:
            raise KeyError(conversation_id)
        return conversations[conversation_id]

    mock_client = AsyncMock()
    mock_client.get_conversation = AsyncMock(side_effect=mock_get_conversation)
    manager._dataset_client = mock_client

    manager.dataset_metadata = DatasetMetadata(
        conversations=[conv.metadata() for conv in conversations.values()],
        sampling_strategy=DatasetSamplingStrategy.RANDOM,
    )
    return manager


@pytest.fixture
def populated_dataset_manager(
    user_config: UserConfig,
    sample_conversations: dict[str, Conversation],
) -> DatasetManager:
    """Create a DatasetManager with a mock dataset client for payload tests."""
    return _create_dataset_manager_with_client(user_config, sample_conversations)


@pytest.fixture
def empty_dataset_manager(
    user_config: UserConfig,
) -> DatasetManager:
    """Create a DatasetManager with an empty dataset client."""
    return _create_dataset_manager_with_client(user_config, {})


@pytest.fixture
def capture_file_writes():
    """Capture file write operations for testing."""

    class FileWriteCapture:
        def __init__(self):
            self.written_content = ""

        def write_bytes(self, data: bytes):
            self.written_content = data.decode("utf-8")

    capture = FileWriteCapture()

    def mock_write_bytes(self, data):
        capture.write_bytes(data)

    with patch("pathlib.Path.write_bytes", mock_write_bytes):
        yield capture


@pytest.fixture
def conversation_ids() -> list[str]:
    """Standard list of conversation IDs for sampler testing."""
    return ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
