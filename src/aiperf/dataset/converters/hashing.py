# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared tokenization and hashing utilities for trace converters."""

from __future__ import annotations

from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.synthesis.rolling_hasher import RollingHasher


def texts_to_hashes_and_lengths(
    tokenizer: Tokenizer,
    texts: list[str],
    block_size: int,
) -> tuple[list[list[int]], list[int]]:
    """Tokenize texts and compute rolling hash IDs per block.

    Args:
        tokenizer: Tokenizer instance for encoding text to token IDs.
        texts: List of text strings to process.
        block_size: Number of tokens per hash block.

    Returns:
        Tuple of (hash_ids_list, token_lengths) where each entry
        corresponds to one input text.
    """
    hasher = RollingHasher(block_size=block_size)
    hash_results: list[list[int]] = []
    length_results: list[int] = []

    for text in texts:
        tokens = tokenizer.encode(text)
        length_results.append(len(tokens))

        blocks: list[list[int]] = [
            tokens[i : i + block_size] for i in range(0, len(tokens), block_size)
        ]
        if blocks:
            hash_results.append(hasher.hash_token_blocks(blocks))
        else:
            hash_results.append([])

    return hash_results, length_results
