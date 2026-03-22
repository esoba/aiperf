#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Demo script showing sample output from ProxyInspiredCodingContentGenerator.

Usage:
    uv run python demo_proxy_inspired_coding_content.py
    uv run python demo_proxy_inspired_coding_content.py --all
    uv run python demo_proxy_inspired_coding_content.py --generators _gen_proxy_exchange
    uv run python demo_proxy_inspired_coding_content.py --prompt-tokens 256
"""

from __future__ import annotations

import argparse
import sys

from aiperf.common.config import PrefixPromptConfig, PromptConfig
from aiperf.dataset.generator.coding_content_proxy_inspired import (
    _PROXY_TOOL_POOL_BLOCK_COUNTS,
    ProxyInspiredCodingContentGenerator,
)

SEPARATOR = "\n" + "=" * 80 + "\n"

DEFAULT_GENERATORS = [
    "_gen_proxy_assistant_flat",
    "_gen_proxy_user_results_flat",
    "_gen_proxy_exchange",
    "_gen_proxy_system_stub",
]


def make_generator() -> ProxyInspiredCodingContentGenerator:
    """Create a ProxyInspiredCodingContentGenerator with a real tokenizer."""
    from aiperf.common.tokenizer import Tokenizer

    config = PromptConfig(
        mean=100,
        stddev=20,
        block_size=512,
        prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
    )
    tokenizer = Tokenizer.from_pretrained("gpt2")
    return ProxyInspiredCodingContentGenerator(config, tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo ProxyInspiredCodingContentGenerator (proxy-trace-shaped synthetic pools)"
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        metavar="NAME",
        help="Generator method names (e.g. _gen_proxy_exchange)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show one sample from every block generator in _PROXY_TOOL_POOL_BLOCK_COUNTS",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Samples per generator (default: 1)",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        metavar="N",
        help="Also print a sliding-window sample from generate_prompt(N)",
    )
    args = parser.parse_args()

    if args.all:
        gen_names = list(_PROXY_TOOL_POOL_BLOCK_COUNTS.keys())
    elif args.generators:
        gen_names = args.generators
    else:
        gen_names = DEFAULT_GENERATORS

    valid = set(_PROXY_TOOL_POOL_BLOCK_COUNTS.keys())
    for name in gen_names:
        if not hasattr(ProxyInspiredCodingContentGenerator, name):
            print(f"Unknown generator: {name}", file=sys.stderr)
            print(f"Valid generators: {', '.join(sorted(valid))}", file=sys.stderr)
            sys.exit(1)

    print("Building generator (tokenizes the pool; may take a few seconds)...")
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

    if args.prompt_tokens is not None:
        n = args.prompt_tokens
        print(SEPARATOR)
        print(f" generate_prompt({n}) ".center(80, "="))
        print(SEPARATOR)
        print(gen.generate_prompt(n))

    print(SEPARATOR)
    print(
        f"Showed {len(gen_names) * args.count} sample(s) from {len(gen_names)} generator(s)."
    )
    print("Pool block counts:")
    total = sum(_PROXY_TOOL_POOL_BLOCK_COUNTS.values())
    for name, count in _PROXY_TOOL_POOL_BLOCK_COUNTS.items():
        marker = " <--" if name in gen_names else ""
        print(f"  {name:<35s} {count:>4d} blocks ({count / total * 100:5.1f}%){marker}")


if __name__ == "__main__":
    main()
