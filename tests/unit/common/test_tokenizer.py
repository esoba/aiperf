# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from unittest.mock import MagicMock

import pytest

from aiperf.common.exceptions import NotInitializedError
from aiperf.common.tokenizer import Tokenizer


class TestTokenizer:
    def test_empty_tokenizer(self) -> None:
        tokenizer = Tokenizer()
        assert tokenizer._tokenizer is None

        with pytest.raises(NotInitializedError):
            tokenizer("test")
        with pytest.raises(NotInitializedError):
            tokenizer.encode("test")
        with pytest.raises(NotInitializedError):
            tokenizer.decode([1])
        with pytest.raises(NotInitializedError):
            _ = tokenizer.bos_token_id

    def test_batch_decode_not_initialized_raises(self) -> None:
        tokenizer = Tokenizer()
        with pytest.raises(NotInitializedError):
            tokenizer.batch_decode([[1, 2]])


class TestBatchDecode:
    """Tests for Tokenizer.batch_decode fast/slow path branching."""

    @pytest.fixture()
    def tokenizer_with_rust_backend(self) -> Tokenizer:
        """Tokenizer whose inner tokenizer has a backend_tokenizer (fast path)."""
        tok = Tokenizer()
        mock_inner = MagicMock()
        mock_inner.backend_tokenizer.decode_batch.return_value = ["rust_a", "rust_b"]
        mock_inner.batch_decode.return_value = ["py_a", "py_b"]
        tok._tokenizer = mock_inner
        tok._decode_args = {"skip_special_tokens": True}
        return tok

    @pytest.fixture()
    def tokenizer_without_rust_backend(self) -> Tokenizer:
        """Tokenizer whose inner tokenizer lacks backend_tokenizer (slow path)."""
        tok = Tokenizer()
        mock_inner = MagicMock(spec=["batch_decode"])
        mock_inner.batch_decode.return_value = ["py_a", "py_b"]
        tok._tokenizer = mock_inner
        tok._decode_args = {"skip_special_tokens": True}
        return tok

    def test_fast_path_used_when_no_extra_kwargs(
        self, tokenizer_with_rust_backend: Tokenizer
    ) -> None:
        seqs = [[1, 2], [3, 4]]
        result = tokenizer_with_rust_backend.batch_decode(
            seqs, skip_special_tokens=False
        )

        assert result == ["rust_a", "rust_b"]
        tokenizer_with_rust_backend._tokenizer.backend_tokenizer.decode_batch.assert_called_once_with(
            seqs, skip_special_tokens=False
        )
        tokenizer_with_rust_backend._tokenizer.batch_decode.assert_not_called()

    def test_slow_path_used_when_extra_kwargs_present(
        self, tokenizer_with_rust_backend: Tokenizer
    ) -> None:
        seqs = [[1, 2], [3, 4]]
        result = tokenizer_with_rust_backend.batch_decode(
            seqs, clean_up_tokenization_spaces=True
        )

        assert result == ["py_a", "py_b"]
        tokenizer_with_rust_backend._tokenizer.batch_decode.assert_called_once_with(
            seqs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        tokenizer_with_rust_backend._tokenizer.backend_tokenizer.decode_batch.assert_not_called()

    def test_slow_path_used_when_no_backend_tokenizer(
        self, tokenizer_without_rust_backend: Tokenizer
    ) -> None:
        seqs = [[1, 2], [3, 4]]
        result = tokenizer_without_rust_backend.batch_decode(
            seqs, skip_special_tokens=False
        )

        assert result == ["py_a", "py_b"]
        tokenizer_without_rust_backend._tokenizer.batch_decode.assert_called_once_with(
            seqs, skip_special_tokens=False
        )

    def test_default_skip_special_tokens_from_decode_args(
        self, tokenizer_with_rust_backend: Tokenizer
    ) -> None:
        """Default skip_special_tokens comes from _decode_args (True)."""
        seqs = [[1, 2]]
        tokenizer_with_rust_backend.batch_decode(seqs)

        tokenizer_with_rust_backend._tokenizer.backend_tokenizer.decode_batch.assert_called_once_with(
            seqs, skip_special_tokens=True
        )
