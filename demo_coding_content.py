#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Demo script showing sample output from CodingContentGenerator templates.

Usage:
    uv run python demo_coding_content.py
    uv run python demo_coding_content.py --generators _gen_cuda_error _gen_sql_query
    uv run python demo_coding_content.py --all
"""

from __future__ import annotations

import argparse
import sys

from aiperf.common.config import PrefixPromptConfig, PromptConfig
from aiperf.dataset.generator.coding_content import (
    _TOOL_POOL_BLOCK_COUNTS,
    CodingContentGenerator,
)

SEPARATOR = "\n" + "=" * 80 + "\n"

# Generators to show by default (one from each category)
DEFAULT_GENERATORS = [
    "_gen_python_code",
    "_gen_ml_training_code",
    "_gen_ml_inference_code",
    "_gen_ml_config",
    "_gen_ml_training_log",
    "_gen_cuda_error",
    "_gen_sql_query",
    "_gen_error_traceback",
    "_gen_json_response",
    "_gen_bash_output",
    "_gen_coding_conversation",
]


def make_generator() -> CodingContentGenerator:
    """Create a CodingContentGenerator with a real tokenizer."""
    from aiperf.common.tokenizer import Tokenizer

    config = PromptConfig(
        mean=100,
        stddev=20,
        block_size=512,
        prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
    )
    tokenizer = Tokenizer.from_pretrained("gpt2")
    return CodingContentGenerator(config, tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo CodingContentGenerator output")
    parser.add_argument(
        "--generators",
        nargs="+",
        metavar="NAME",
        help="Specific generator method names to demo (e.g. _gen_cuda_error)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show one sample from every generator in _TOOL_POOL_BLOCK_COUNTS",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of samples per generator (default: 1)",
    )
    args = parser.parse_args()

    if args.all:
        gen_names = list(_TOOL_POOL_BLOCK_COUNTS.keys())
    elif args.generators:
        gen_names = args.generators
    else:
        gen_names = DEFAULT_GENERATORS

    # Validate names
    valid = set(_TOOL_POOL_BLOCK_COUNTS.keys())
    for name in gen_names:
        if name not in valid:
            print(f"Unknown generator: {name}", file=sys.stderr)
            print(f"Valid generators: {', '.join(sorted(valid))}", file=sys.stderr)
            sys.exit(1)

    print("Building generator (this tokenizes the pool, takes a few seconds)...")
    gen = make_generator()
    pool_tokens = len(gen._tool_pool)
    print(f"Tool pool: {pool_tokens:,} tokens\n")

    for name in gen_names:
        fn = getattr(gen, name)
        for i in range(args.count):
            header = f" {name} " if args.count == 1 else f" {name} (sample {i + 1}) "
            print(SEPARATOR)
            print(f"{header:=^80}")
            print(SEPARATOR)
            print(fn())

    print(SEPARATOR)
    print(
        f"Showed {len(gen_names) * args.count} sample(s) from {len(gen_names)} generator(s)."
    )
    print("Pool distribution:")
    total = sum(_TOOL_POOL_BLOCK_COUNTS.values())
    for name, count in _TOOL_POOL_BLOCK_COUNTS.items():
        marker = " <--" if name in gen_names else ""
        print(f"  {name:<30s} {count:>4d} blocks ({count / total * 100:5.1f}%){marker}")


if __name__ == "__main__":
    main()
