# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for parallel_decode module."""

from unittest.mock import MagicMock, patch

from aiperf.dataset.generator.parallel_decode import parallel_decode


class TestParallelDecode:
    """Test suite for parallel_decode module."""

    def test_parallel_decode_empty_list(self) -> None:
        """Test parallel_decode with empty input returns empty list."""
        result = parallel_decode([], "gpt2")
        assert result == []

    @patch("aiperf.dataset.generator.parallel_decode.Tokenizer")
    def test_parallel_decode_uses_batch_decode(
        self, mock_tokenizer_class: MagicMock
    ) -> None:
        """Test that parallel_decode uses tokenizer batch_decode."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.batch_decode.return_value = [
            "decoded 1",
            "decoded 2",
            "decoded 3",
        ]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        token_sequences = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = parallel_decode(token_sequences, "gpt2")

        mock_tokenizer_class.from_pretrained.assert_called_once_with("gpt2")
        mock_tokenizer.batch_decode.assert_called_once_with(
            token_sequences, skip_special_tokens=False
        )
        assert result == ["decoded 1", "decoded 2", "decoded 3"]

    @patch("aiperf.dataset.generator.parallel_decode.Tokenizer")
    def test_parallel_decode_accepts_max_workers_chunksize(
        self, mock_tokenizer_class: MagicMock
    ) -> None:
        """Test that max_workers and chunksize are accepted for API compatibility."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.batch_decode.return_value = ["decoded"]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        result = parallel_decode([[1]], "gpt2", max_workers=4, chunksize=10)

        assert result == ["decoded"]
