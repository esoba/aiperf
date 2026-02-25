# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch decode utilities for tokenizer operations.

For fast (Rust-backed) tokenizers, decoding uses the Rust tokenizer's
decode_batch with Rayon thread-based parallelism. No multiprocessing or
daemon process spawning required -- works on all platforms including macOS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.tokenizer import Tokenizer


def parallel_decode(
    token_sequences: list[list[int]],
    tokenizer_name: str,
    *,
    tokenizer: Tokenizer | None = None,
) -> list[str]:
    """Decode multiple token sequences using tokenizer batch_decode.

    For fast tokenizers, uses the Rust decode_batch with Rayon thread
    parallelism. Falls back to Python sequential decode for slow tokenizers.
    Works on all platforms including macOS.

    Args:
        token_sequences: List of token ID lists to decode.
        tokenizer_name: Name or path of the pretrained tokenizer.
            Used only when ``tokenizer`` is not provided.
        tokenizer: Pre-loaded tokenizer instance. When supplied, avoids
            the ~200ms overhead of ``Tokenizer.from_pretrained()``.

    Returns:
        List of decoded strings in the same order as input.
    """
    if not token_sequences:
        return []
    if tokenizer is None:
        from aiperf.common.tokenizer import Tokenizer

        tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    return tokenizer.batch_decode(token_sequences, skip_special_tokens=False)
