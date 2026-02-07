# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.common.utils import load_json_str
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from aiperf.dataset.loader.multi_turn import MultiTurnDatasetLoader
from aiperf.dataset.loader.random_pool import RandomPoolDatasetLoader
from aiperf.dataset.loader.single_turn import SingleTurnDatasetLoader
from aiperf.plugin import plugins
from aiperf.plugin.enums import DatasetLoaderType, PluginType


def _infer_type(
    data: dict[str, Any] | None = None, filename: str | Path | None = None
) -> DatasetLoaderType:
    """Standalone version of DatasetManager._infer_type() for testing."""
    if data is not None and "type" in data:
        explicit_type = DatasetLoaderType(data["type"])
        LoaderClass = plugins.get_class(PluginType.DATASET_LOADER, explicit_type)
        if not LoaderClass.can_load(data, filename):
            raise ValueError(
                f"Explicit type field {explicit_type} specified, but loader {LoaderClass.__name__} "
                "cannot handle the data format."
            )
        return explicit_type

    detected_type = None
    for entry, LoaderClass in plugins.iter_all(PluginType.DATASET_LOADER):
        if LoaderClass.can_load(data, filename):
            dataset_type = DatasetLoaderType(entry.name)
            if detected_type is not None:
                raise ValueError(
                    f"Multiple loaders can handle the data format: {detected_type} and {dataset_type}."
                )
            detected_type = dataset_type

    if detected_type is None:
        raise ValueError("No loader can handle the data format.")

    return detected_type


def _infer_dataset_type(file_path: str) -> DatasetLoaderType | None:
    """Standalone version of DatasetManager._infer_dataset_type() for testing."""
    path = Path(file_path)

    if path.is_dir():
        return _infer_type(data=None, filename=file_path)

    with open(file_path) as f:
        for line in f:
            if not (line := line.strip()):
                continue
            data = load_json_str(line)
            return _infer_type(data=data, filename=file_path)

    return None


class TestSingleTurnCanLoad:
    """Tests for SingleTurnDatasetLoader.can_load() method.

    Note: Loaders use pydantic model validation which includes type field validation.
    The 'type' field must match the loader's expected type or be omitted (defaults to correct type).
    """

    @pytest.mark.parametrize(
        "data,expected",
        [
            param({"text": "Hello world"}, True, id="text_field"),
            param({"texts": ["Hello", "World"]}, True, id="texts_field"),
            param({"image": "/path/to/image.png"}, True, id="image_field"),
            param({"images": ["/path/1.png", "/path/2.png"]}, True, id="images_field"),
            param({"audio": "/path/to/audio.wav"}, True, id="audio_field"),
            param({"audios": ["/path/1.wav", "/path/2.wav"]}, True, id="audios_field"),
            param({"text": "Describe this", "image": "/path.png", "audio": "/audio.wav"}, True, id="multimodal"),
            # Explicit type must match (pydantic validates it)
            param({"type": "single_turn", "text": "Hello"}, True, id="with_type_field"),
            param({"type": "random_pool", "text": "Hello"}, False, id="wrong_type_rejected"),
            param({"turns": [{"text": "Hello"}]}, False, id="has_turns_field"),
            param({"session_id": "123", "metadata": "test"}, False, id="no_modality"),
            param(None, False, id="none_data"),
        ],
    )  # fmt: skip
    def test_can_load(self, data, expected):
        """Test various data formats for SingleTurn pydantic validation."""
        assert SingleTurnDatasetLoader.can_load(data) is expected


class TestMultiTurnCanLoad:
    """Tests for MultiTurnDatasetLoader.can_load() method.

    Note: Loaders use pydantic model validation which includes type field validation.
    The 'type' field must match the loader's expected type or be omitted (defaults to correct type).
    """

    @pytest.mark.parametrize(
        "data,expected",
        [
            param({"turns": [{"text": "Turn 1"}, {"text": "Turn 2"}]}, True, id="turns_list"),
            param({"session_id": "session_123", "turns": [{"text": "Hello"}]}, True, id="with_session_id"),
            # Explicit type must match (pydantic validates it)
            param({"type": "multi_turn", "turns": [{"text": "Hello"}]}, True, id="with_type_field"),
            param({"text": "Hello world"}, False, id="no_turns_field"),
            param({"turns": "not a list"}, False, id="turns_not_list_string"),
            param({"turns": {"text": "Hello"}}, False, id="turns_not_list_dict"),
            param(None, False, id="none_data"),
        ],
    )  # fmt: skip
    def test_can_load(self, data, expected):
        """Test various data formats for MultiTurn pydantic validation."""
        assert MultiTurnDatasetLoader.can_load(data) is expected


class TestRandomPoolCanLoad:
    """Tests for RandomPoolDatasetLoader.can_load() method.

    Note: Loaders use pydantic model validation. RandomPool requires either:
    1. Data with explicit type="random_pool" and valid modality fields, OR
    2. A directory/file path with at least one valid data entry
    """

    @pytest.mark.parametrize(
        "data,expected",
        [
            # RandomPool cannot distinguish from SingleTurn without explicit type
            param({"text": "Hello"}, False, id="no_explicit_type"),
            # With explicit type field, RandomPool validates via pydantic
            param({"type": "random_pool", "text": "Query"}, True, id="explicit_type_validates"),
        ],
    )  # fmt: skip
    def test_can_load_content_based(self, data, expected):
        """Test content-based detection for RandomPool.

        RandomPool.can_load() checks for explicit type field first, then validates with pydantic."""
        assert RandomPoolDatasetLoader.can_load(data) is expected

    def test_can_load_with_directory_path(self):
        """Test detection with directory path containing valid files (unique to RandomPool)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create a valid file in the directory
            file_path = temp_path / "data.jsonl"
            file_path.write_text('{"text": "Hello"}\n')
            assert (
                RandomPoolDatasetLoader.can_load(data=None, filename=temp_path) is True
            )

    def test_can_load_with_directory_path_as_string(self):
        """Test detection with directory path as string containing valid files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid file in the directory
            file_path = Path(temp_dir) / "data.jsonl"
            file_path.write_text('{"text": "Hello"}\n')
            assert (
                RandomPoolDatasetLoader.can_load(data=None, filename=temp_dir) is True
            )

    def test_cannot_load_with_file_path_no_type(self):
        """Test rejection with file path but no explicit type (ambiguous with SingleTurn)."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
            temp_path = Path(temp_file.name)
            data = {"text": "Hello"}
            # Without explicit type, ambiguous with SingleTurn
            assert RandomPoolDatasetLoader.can_load(data, filename=temp_path) is False


class TestMooncakeTraceCanLoad:
    """Tests for MooncakeTraceDatasetLoader.can_load() method.

    Note: Loaders use pydantic model validation. MooncakeTrace requires either:
    - input_length (with optional hash_ids), OR
    - text_input (hash_ids not allowed with text_input)
    """

    @pytest.mark.parametrize(
        "data,expected",
        [
            param({"input_length": 100, "output_length": 50}, True, id="input_length_with_output"),
            param({"input_length": 100}, True, id="input_length_only"),
            param({"input_length": 100, "hash_ids": [123, 456]}, True, id="input_length_with_hash_ids"),
            # Explicit type must match (pydantic validates it)
            param({"type": "mooncake_trace", "input_length": 100}, True, id="with_type_field"),
            # hash_ids only allowed with input_length, not text_input
            param({"text_input": "Hello world", "hash_ids": [123, 456]}, False, id="text_input_with_hash_ids_invalid"),
            param({"text_input": "Hello world"}, True, id="text_input_only"),
            param({"timestamp": 1000, "session_id": "abc"}, False, id="no_required_fields"),
            param({"output_length": 50}, False, id="only_output_length"),
            param(None, False, id="none_data"),
        ],
    )  # fmt: skip
    def test_can_load(self, data, expected):
        """Test various data formats for MooncakeTrace pydantic validation."""
        assert MooncakeTraceDatasetLoader.can_load(data) is expected


class TestDatasetManagerInferType:
    """Tests for _infer_type() logic (extracted from DatasetManager).

    The method first checks for explicit 'type' field, then falls back to
    querying loaders. With pydantic validation, loaders respect type fields."""

    @pytest.mark.parametrize(
        "data,filename,expected_type",
        [
            param({"text": "Hello world"}, None, DatasetLoaderType.SINGLE_TURN, id="single_turn_text"),
            param({"type": "single_turn", "text": "Hello"}, None, DatasetLoaderType.SINGLE_TURN, id="single_turn_explicit"),
            param({"image": "/path.png"}, None, DatasetLoaderType.SINGLE_TURN, id="single_turn_image"),
            param({"turns": [{"text": "Turn 1"}]}, None, DatasetLoaderType.MULTI_TURN, id="multi_turn_turns"),
            param({"type": "multi_turn", "turns": [{"text": "Turn 1"}]}, None, DatasetLoaderType.MULTI_TURN, id="multi_turn_explicit"),
            param({"input_length": 100, "output_length": 50}, None, DatasetLoaderType.MOONCAKE_TRACE, id="mooncake_input_length"),
            param({"type": "mooncake_trace", "input_length": 100}, None, DatasetLoaderType.MOONCAKE_TRACE, id="mooncake_explicit"),
            param({"text_input": "Hello"}, None, DatasetLoaderType.MOONCAKE_TRACE, id="mooncake_text_input"),
        ],
    )  # fmt: skip
    def test_infer_from_data(self, data, filename, expected_type):
        """Test inferring dataset type from various data formats."""
        result = _infer_type(data, filename=filename)
        assert result == expected_type

    def test_infer_random_pool_explicit_type(self, create_jsonl_file):
        """Test inferring RandomPool with explicit type field (requires file for validation)."""
        filepath = create_jsonl_file(['{"type": "random_pool", "text": "Query"}'])
        data = {"type": "random_pool", "text": "Query"}
        result = _infer_type(data, filename=filepath)
        assert result == DatasetLoaderType.RANDOM_POOL

    @pytest.mark.parametrize(
        "data",
        [
            param({"unknown_field": "value"}, id="unknown_format"),
            param({"metadata": "test"}, id="unknown_metadata"),
        ],
    )  # fmt: skip
    def test_infer_from_data_raises(self, data):
        """Test that unknown formats raise ValueError."""
        with pytest.raises(ValueError, match="No loader can handle"):
            _infer_type(data)

    def test_infer_random_pool_with_directory(self):
        """Test inferring RandomPool with directory path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / "data.jsonl"
            file_path.write_text('{"text": "Hello"}\n')
            result = _infer_type(data=None, filename=temp_path)
            assert result == DatasetLoaderType.RANDOM_POOL

    def test_infer_with_filename_parameter(self):
        """Test inference with filename parameter for file path."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            try:
                data = {"text": "Hello"}
                result = _infer_type(data, filename=temp_path)
                assert result == DatasetLoaderType.SINGLE_TURN
            finally:
                temp_path.unlink()


class TestDatasetManagerInferDatasetType:
    """Tests for _infer_dataset_type() logic (extracted from DatasetManager)."""

    @pytest.mark.parametrize(
        "content,expected_type",
        [
            param(['{"text": "Hello world"}'], DatasetLoaderType.SINGLE_TURN, id="single_turn_text"),
            param(['{"image": "/path.png"}'], DatasetLoaderType.SINGLE_TURN, id="single_turn_image"),
            param(['{"turns": [{"text": "Turn 1"}, {"text": "Turn 2"}]}'], DatasetLoaderType.MULTI_TURN, id="multi_turn"),
            param(['{"type": "random_pool", "text": "Query"}'], DatasetLoaderType.RANDOM_POOL, id="random_pool_explicit"),
            param(['{"input_length": 100, "output_length": 50}'], DatasetLoaderType.MOONCAKE_TRACE, id="mooncake_input_length"),
            param(['{"text_input": "Hello"}'], DatasetLoaderType.MOONCAKE_TRACE, id="mooncake_text_input"),
        ],
    )  # fmt: skip
    def test_infer_from_file(self, create_jsonl_file, content, expected_type):
        """Test inferring dataset type from file with various content."""
        filepath = create_jsonl_file(content)
        result = _infer_dataset_type(filepath)
        assert result == expected_type

    @pytest.mark.parametrize(
        "content",
        [
            param([], id="empty_file"),
            param(["", "   ", "\n"], id="only_empty_lines"),
        ],
    )  # fmt: skip
    def test_infer_from_file_empty(self, create_jsonl_file, content):
        """Test that empty files return None (no valid lines to infer from)."""
        filepath = create_jsonl_file(content)
        result = _infer_dataset_type(filepath)
        assert result is None

    def test_infer_from_file_invalid_json(self, create_jsonl_file):
        """Test that invalid JSON raises an error."""
        filepath = create_jsonl_file(["not valid json"])
        with pytest.raises((ValueError, Exception)):
            _infer_dataset_type(filepath)

    def test_infer_from_directory(self):
        """Test inferring type from directory (should be RandomPool)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file1 = temp_path / "queries.jsonl"
            file1.write_text('{"text": "Query 1"}\n')

            result = _infer_dataset_type(temp_dir)
            assert result == DatasetLoaderType.RANDOM_POOL


class TestDetectionPriorityAndAmbiguity:
    """Tests for detection priority and handling of ambiguous cases.

    Note: Loaders use pydantic model validation which validates the type field.
    The 'type' field must match the loader's expected type or be omitted.
    """

    def test_explicit_type_handled_by_validation(self):
        """Test that explicit type field is validated by loaders via pydantic."""
        data = {"type": "random_pool", "text": "Hello"}

        # Loader behavior with explicit type field:
        # - SingleTurn.can_load(data) rejects because type doesn't match
        # - RandomPool.can_load(data) validates with pydantic and returns True
        assert SingleTurnDatasetLoader.can_load(data) is False
        assert RandomPoolDatasetLoader.can_load(data) is True

        # Type inference with explicit type should return RANDOM_POOL
        result = _infer_type(data)
        assert result == DatasetLoaderType.RANDOM_POOL

    @pytest.mark.parametrize(
        "data,single_turn,random_pool",
        [
            param({"text": "Hello"}, True, False, id="text_field"),
            param({"image": "/path.png"}, True, False, id="image_field"),
        ],
    )  # fmt: skip
    def test_single_turn_vs_random_pool_ambiguity(self, data, single_turn, random_pool):
        """Test SingleTurn vs RandomPool without explicit type.

        Without explicit type or filename, SingleTurn matches, RandomPool doesn't.
        """
        assert SingleTurnDatasetLoader.can_load(data) is single_turn
        assert RandomPoolDatasetLoader.can_load(data) is random_pool

    def test_multi_turn_takes_priority_over_single_turn(self):
        """Test that MultiTurn is correctly detected over SingleTurn."""
        data = {"turns": [{"text": "Hello"}]}
        assert MultiTurnDatasetLoader.can_load(data) is True
        assert SingleTurnDatasetLoader.can_load(data) is False

    @pytest.mark.parametrize(
        "loader,should_match",
        [
            param(MooncakeTraceDatasetLoader, True, id="mooncake"),
            param(SingleTurnDatasetLoader, False, id="single_turn"),
            param(MultiTurnDatasetLoader, False, id="multi_turn"),
            param(RandomPoolDatasetLoader, False, id="random_pool"),
        ],
    )  # fmt: skip
    def test_mooncake_trace_distinct_from_others(self, loader, should_match):
        """Test that MooncakeTrace is distinct from other types."""
        data = {"input_length": 100}
        assert loader.can_load(data) is should_match

    def test_directory_path_uniquely_identifies_random_pool(self):
        """Test that directory path with valid files uniquely identifies RandomPool."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create a valid file in the directory
            file_path = temp_path / "data.jsonl"
            file_path.write_text('{"text": "Hello"}\n')
            assert RandomPoolDatasetLoader.can_load(data=None, filename=temp_path) is True  # fmt: skip
            assert SingleTurnDatasetLoader.can_load(data=None, filename=temp_path) is False  # fmt: skip
            assert MultiTurnDatasetLoader.can_load(data=None, filename=temp_path) is False  # fmt: skip
            assert MooncakeTraceDatasetLoader.can_load(data=None, filename=temp_path) is False  # fmt: skip
