# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for BaseTraceDatasetLoader.

Covers file hashing, trace_id lifecycle, parallel vs single-threaded threshold,
convert_to_conversations dispatching, and integration with parallel_convert.
"""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.common.config.config_defaults import InputTokensDefaults
from aiperf.common.models import Conversation
from aiperf.dataset.loader.base_trace_loader import (
    _MIN_TRACES_FOR_PARALLEL,
    _compute_file_hash,
)
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def create_jsonl_file():
    """Create a temporary JSONL file with custom content."""
    filenames = []

    def _create_file(content_lines):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for line in content_lines:
                f.write(line + "\n")
            filenames.append(f.name)
            return f.name

    yield _create_file

    for fn in filenames:
        Path(fn).unlink(missing_ok=True)


@pytest.fixture
def default_user_config() -> UserConfig:
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))


@pytest.fixture
def mock_prompt_generator():
    """Mock PromptGenerator with required attributes for BaseTraceDatasetLoader."""
    generator = Mock()
    generator.generate.return_value = "Generated prompt text"
    generator._cache = {}
    generator._tokenized_corpus = list(range(100, 200))
    generator._hash_id_corpus_rng = Mock()
    generator._hash_id_corpus_rng.seed = 42
    generator.tokenizer = Mock()
    generator.tokenizer.resolved_name = "test-model"
    generator.tokenizer.block_separation_token_id = None
    return generator


# -----------------------------------------------------------------------
# _compute_file_hash
# -----------------------------------------------------------------------


class TestComputeFileHash:
    """Tests for _compute_file_hash function."""

    def test_hash_is_16_hex_chars(self, create_jsonl_file):
        """Hash should be 16 hex characters."""
        filename = create_jsonl_file(["test content"])
        result = _compute_file_hash(filename)
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic_for_same_content(self, create_jsonl_file):
        """Same file content should produce same hash."""
        f1 = create_jsonl_file(["line 1", "line 2"])
        f2 = create_jsonl_file(["line 1", "line 2"])
        assert _compute_file_hash(f1) == _compute_file_hash(f2)

    def test_different_content_different_hash(self, create_jsonl_file):
        """Different file content should produce different hash."""
        f1 = create_jsonl_file(["line 1"])
        f2 = create_jsonl_file(["line 2"])
        assert _compute_file_hash(f1) != _compute_file_hash(f2)

    def test_fallback_on_file_error(self):
        """Non-existent file should fall back to hashing the filepath string."""
        result = _compute_file_hash("/nonexistent/path/file.jsonl")
        expected = hashlib.sha256(b"/nonexistent/path/file.jsonl").hexdigest()[:16]
        assert result == expected

    def test_fallback_on_type_error(self):
        """TypeError (e.g. from mock_open) should fall back to filepath hash."""
        with patch("builtins.open") as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=False)
            mock_file.read.return_value = "string not bytes"
            mock_open.return_value = mock_file

            result = _compute_file_hash("test.jsonl")
            expected = hashlib.sha256(b"test.jsonl").hexdigest()[:16]
            assert result == expected

    def test_hash_matches_sha256(self, create_jsonl_file):
        """Hash should match first 16 chars of SHA-256."""
        content = ["test line one", "test line two"]
        filename = create_jsonl_file(content)

        with open(filename, "rb") as f:
            expected = hashlib.sha256(f.read()).hexdigest()[:16]

        assert _compute_file_hash(filename) == expected


# -----------------------------------------------------------------------
# load_dataset — trace_id lifecycle
# -----------------------------------------------------------------------


class TestLoadDatasetTraceId:
    """Tests that load_dataset computes trace_id and sets it on the RNG."""

    def test_load_dataset_sets_trace_id(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """load_dataset should compute file hash and set trace_id."""
        content = ['{"input_length": 100, "hash_ids": [1], "timestamp": 1000}']
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        loader.load_dataset()

        assert loader._trace_id != ""
        assert len(loader._trace_id) == 16
        mock_prompt_generator._hash_id_corpus_rng.set_trace_id.assert_called_once_with(
            loader._trace_id
        )

    def test_trace_id_matches_file_hash(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """trace_id should match the SHA-256 hash of the file."""
        content = ['{"input_length": 100, "hash_ids": [1], "timestamp": 1000}']
        filename = create_jsonl_file(content)

        expected_hash = _compute_file_hash(filename)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        loader.load_dataset()

        assert loader._trace_id == expected_hash

    def test_different_files_different_trace_ids(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Different files should produce different trace_ids."""
        f1 = create_jsonl_file(['{"input_length": 100, "hash_ids": [1]}'])
        f2 = create_jsonl_file(['{"input_length": 200, "hash_ids": [2]}'])

        loader1 = MooncakeTraceDatasetLoader(
            filename=f1,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        loader1.load_dataset()

        loader2 = MooncakeTraceDatasetLoader(
            filename=f2,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        loader2.load_dataset()

        assert loader1._trace_id != loader2._trace_id


# -----------------------------------------------------------------------
# convert_to_conversations — threshold dispatching
# -----------------------------------------------------------------------


class TestConvertToConversationsDispatching:
    """Tests for parallel vs single-threaded dispatching."""

    def test_empty_data_returns_empty(self, mock_prompt_generator, default_user_config):
        """Empty data dict should return empty list."""
        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        result = list(loader.convert_to_conversations({}))
        assert result == []

    def test_small_dataset_uses_single_threaded(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Datasets with fewer than _MIN_TRACES_FOR_PARALLEL traces use single-threaded."""
        content = ['{"input_length": 100, "hash_ids": [1], "timestamp": 1000}']
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        data = loader.load_dataset()

        with patch(
            "aiperf.dataset.loader.base_trace_loader.parallel_convert"
        ) as mock_parallel:
            result = list(loader.convert_to_conversations(data))

            mock_parallel.assert_not_called()
            assert len(result) == 1
            assert isinstance(result[0], Conversation)

    def test_large_dataset_uses_parallel(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Datasets with >= _MIN_TRACES_FOR_PARALLEL traces use parallel conversion."""
        content = [
            f'{{"input_length": 100, "hash_ids": [{i}], "timestamp": {i * 1000}}}'
            for i in range(_MIN_TRACES_FOR_PARALLEL + 1)
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        data = loader.load_dataset()

        mock_conversations = [Conversation(session_id=f"s{i}") for i in range(3)]
        with patch(
            "aiperf.dataset.loader.base_trace_loader.parallel_convert",
            return_value=mock_conversations,
        ) as mock_parallel:
            result = list(loader.convert_to_conversations(data))

            mock_parallel.assert_called_once()
            assert result == mock_conversations

    def test_parallel_convert_receives_correct_args(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """parallel_convert should receive the right parameters from the loader."""
        content = [
            f'{{"input_length": 100, "hash_ids": [{i}], "timestamp": {i * 1000}}}'
            for i in range(_MIN_TRACES_FOR_PARALLEL + 1)
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        data = loader.load_dataset()

        with patch(
            "aiperf.dataset.loader.base_trace_loader.parallel_convert",
            return_value=[],
        ) as mock_parallel:
            list(loader.convert_to_conversations(data, num_workers=8, batch_size=50))

            call_kwargs = mock_parallel.call_args[1]
            assert call_kwargs["tokenizer_name"] == "test-model"
            assert call_kwargs["base_seed"] == 42
            assert call_kwargs["block_size"] == InputTokensDefaults.BLOCK_SIZE
            assert call_kwargs["sep_token"] is None
            assert call_kwargs["trace_id"] == loader._trace_id
            assert call_kwargs["num_workers"] == 8
            assert call_kwargs["batch_size"] == 50

    def test_exactly_threshold_uses_parallel(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Exactly _MIN_TRACES_FOR_PARALLEL traces should use parallel."""
        content = [
            f'{{"input_length": 100, "hash_ids": [{i}], "timestamp": {i * 1000}}}'
            for i in range(_MIN_TRACES_FOR_PARALLEL)
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        data = loader.load_dataset()

        with patch(
            "aiperf.dataset.loader.base_trace_loader.parallel_convert",
            return_value=[],
        ) as mock_parallel:
            list(loader.convert_to_conversations(data))
            mock_parallel.assert_called_once()

    def test_below_threshold_uses_single_threaded(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """One below _MIN_TRACES_FOR_PARALLEL uses single-threaded."""
        content = [
            f'{{"input_length": 100, "hash_ids": [{i}], "timestamp": {i * 1000}}}'
            for i in range(_MIN_TRACES_FOR_PARALLEL - 1)
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        data = loader.load_dataset()

        with patch(
            "aiperf.dataset.loader.base_trace_loader.parallel_convert"
        ) as mock_parallel:
            list(loader.convert_to_conversations(data))
            mock_parallel.assert_not_called()


# -----------------------------------------------------------------------
# _convert_single_threaded
# -----------------------------------------------------------------------


class TestConvertSingleThreaded:
    """Tests for the single-threaded conversion fallback."""

    def test_text_input_used_directly(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Traces with text_input should use the literal text."""
        content = ['{"text_input": "Hello world", "timestamp": 1000}']
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        data = loader.load_dataset()
        conversations = list(loader.convert_to_conversations(data))

        assert len(conversations) == 1
        assert conversations[0].turns[0].texts[0].contents == ["Hello world"]
        mock_prompt_generator.generate.assert_not_called()

    def test_hash_ids_calls_generate(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Traces with hash_ids should call prompt_generator.generate()."""
        content = ['{"input_length": 100, "hash_ids": [1, 2], "timestamp": 1000}']
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        data = loader.load_dataset()
        conversations = list(loader.convert_to_conversations(data))

        assert len(conversations) == 1
        mock_prompt_generator.generate.assert_called_once_with(
            mean=100,
            stddev=0,
            hash_ids=[1, 2],
            block_size=InputTokensDefaults.BLOCK_SIZE,
        )

    def test_no_input_calls_generate_with_empty_hash_ids(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Traces with input_length but no hash_ids still call generate."""
        content = ['{"input_length": 50, "timestamp": 1000}']
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        data = loader.load_dataset()
        list(loader.convert_to_conversations(data))

        mock_prompt_generator.generate.assert_called_once_with(
            mean=50,
            stddev=0,
            hash_ids=[],
            block_size=InputTokensDefaults.BLOCK_SIZE,
        )

    def test_turn_fields_populated(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Turn objects should have correct timestamp, delay, max_tokens."""
        content = [
            '{"input_length": 100, "hash_ids": [1], "timestamp": 5000, "output_length": 42}'
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        data = loader.load_dataset()
        conversations = list(loader.convert_to_conversations(data))

        turn = conversations[0].turns[0]
        assert turn.timestamp == 5000
        assert turn.max_tokens == 42

    def test_multi_session_conversion(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Multiple sessions each produce a separate Conversation."""
        content = [
            '{"session_id": "s1", "input_length": 100, "hash_ids": [1], "timestamp": 1000}',
            '{"session_id": "s2", "input_length": 200, "hash_ids": [2], "timestamp": 2000}',
            '{"session_id": "s1", "input_length": 150, "hash_ids": [3], "delay": 50}',
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        data = loader.load_dataset()
        conversations = list(loader.convert_to_conversations(data))

        session_ids = {c.session_id for c in conversations}
        assert "s1" in session_ids
        assert "s2" in session_ids

        s1_conv = next(c for c in conversations if c.session_id == "s1")
        assert len(s1_conv.turns) == 2


# -----------------------------------------------------------------------
# Block size precedence
# -----------------------------------------------------------------------


class TestBlockSizePrecedence:
    """Tests that block size follows the correct precedence chain."""

    def test_default_block_size(self, mock_prompt_generator, default_user_config):
        """Without any overrides, block_size should be InputTokensDefaults.BLOCK_SIZE."""
        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        assert loader._block_size == InputTokensDefaults.BLOCK_SIZE

    def test_plugin_default_block_size(
        self, mock_prompt_generator, default_user_config
    ):
        """Plugin metadata default should override the hardcoded fallback."""
        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
            default_block_size=16,
        )
        assert loader._block_size == 16

    def test_user_cli_overrides_plugin_default(self, mock_prompt_generator):
        """User CLI --isl-block-size should override plugin metadata default."""
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                prompt=PromptConfig(
                    input_tokens=InputTokensConfig(block_size=64),
                ),
            ),
        )
        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
            default_block_size=16,
        )
        assert loader._block_size == 64

    def test_block_size_passed_to_single_threaded(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Single-threaded path should pass block_size to generate()."""
        content = ['{"input_length": 100, "hash_ids": [1], "timestamp": 1000}']
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
            default_block_size=16,
        )
        data = loader.load_dataset()
        list(loader.convert_to_conversations(data))

        mock_prompt_generator.generate.assert_called_once_with(
            mean=100,
            stddev=0,
            hash_ids=[1],
            block_size=16,
        )

    def test_block_size_passed_to_parallel(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Parallel path should pass block_size to parallel_convert."""
        content = [
            f'{{"input_length": 100, "hash_ids": [{i}], "timestamp": {i * 1000}}}'
            for i in range(_MIN_TRACES_FOR_PARALLEL + 1)
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
            default_block_size=16,
        )
        data = loader.load_dataset()

        with patch(
            "aiperf.dataset.loader.base_trace_loader.parallel_convert",
            return_value=[],
        ) as mock_parallel:
            list(loader.convert_to_conversations(data))
            assert mock_parallel.call_args[1]["block_size"] == 16


# -----------------------------------------------------------------------
# Serialization for parallel path
# -----------------------------------------------------------------------


class TestSessionSerialization:
    """Tests that traces are correctly serialized to dicts for parallel workers."""

    def test_traces_serialized_via_model_dump(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Traces should be serialized via model_dump() before passing to parallel_convert."""
        content = [
            f'{{"input_length": {100 + i}, "hash_ids": [{i}], "timestamp": {i * 1000}, "output_length": {20 + i}}}'
            for i in range(_MIN_TRACES_FOR_PARALLEL + 1)
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        data = loader.load_dataset()

        with patch(
            "aiperf.dataset.loader.base_trace_loader.parallel_convert",
            return_value=[],
        ) as mock_parallel:
            list(loader.convert_to_conversations(data))

            sessions = mock_parallel.call_args[1]["sessions"]
            assert len(sessions) > 0

            # Each session should be (str, list[dict])
            for sid, traces in sessions:
                assert isinstance(sid, str)
                for trace_dict in traces:
                    assert isinstance(trace_dict, dict)
                    assert "input_length" in trace_dict
