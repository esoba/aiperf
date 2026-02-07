# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for dataset manager testing.
"""

from pathlib import Path

import pytest

import aiperf.endpoints  # noqa: F401  # Import to register endpoints
import aiperf.transports  # noqa: F401  # Import to register transports
from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.models import Conversation, DatasetMetadata
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    DatasetSamplingStrategy,
    EndpointType,
    PluginType,
)


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


async def _build_dataset_manager_with_conversations(
    user_config: UserConfig,
    conversations: dict[str, Conversation],
) -> DatasetManager:
    """Build a DatasetManager with conversations written to a real backing store + client.

    This replaces the old pattern of directly assigning manager.dataset = {...}.
    """
    manager = DatasetManager(
        service_config=ServiceConfig(),
        user_config=user_config,
        service_id="test_dataset_manager",
    )

    # Write conversations to the backing store
    await manager._backing_store.initialize()
    for conv_id, conv in conversations.items():
        await manager._backing_store.add_conversation(conv_id, conv)
    await manager._backing_store.finalize()

    # Initialize the client so _generate_input_payloads can read from it
    client_metadata = manager._backing_store.get_client_metadata()
    ClientStoreClass = plugins.get_class(
        PluginType.DATASET_CLIENT_STORE, client_metadata.client_type
    )
    manager._dataset_client = ClientStoreClass(client_metadata=client_metadata)
    await manager._dataset_client.initialize()

    # Build dataset metadata
    manager.dataset_metadata = DatasetMetadata(
        conversations=[conv.metadata() for conv in conversations.values()],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )

    return manager


@pytest.fixture
async def empty_dataset_manager(user_config: UserConfig) -> DatasetManager:
    """Create a DatasetManager instance with empty dataset (no backing store/client)."""
    manager = DatasetManager(
        service_config=ServiceConfig(),
        user_config=user_config,
        service_id="test_dataset_manager",
    )
    manager.dataset_metadata = DatasetMetadata(
        conversations=[],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    return manager


@pytest.fixture
async def populated_dataset_manager(
    user_config: UserConfig,
    sample_conversations: dict[str, Conversation],
) -> DatasetManager:
    """Create a DatasetManager instance with sample data."""
    return await _build_dataset_manager_with_conversations(
        user_config, sample_conversations
    )


@pytest.fixture
def capture_file_writes():
    """Provide a fixture to capture file write operations for testing purposes."""
    from unittest.mock import patch

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
