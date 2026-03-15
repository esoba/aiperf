#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Demo script showing pre-generated coding assistant responses.

Generates realistic multi-turn tool-use conversations at various token budgets
and analyzes the output structure.

Usage:
    uv run python demo_pregenerated_responses.py
    uv run python demo_pregenerated_responses.py --tokens 1000
    uv run python demo_pregenerated_responses.py --seeds 5 --tokens 200 400 800
    uv run python demo_pregenerated_responses.py --full  # show full message content
"""

from __future__ import annotations

import argparse
from collections import Counter

from aiperf.common import random_generator as rng
from aiperf.common.config import PromptConfig
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.coding_content import CodingContentGenerator

SEPARATOR = "\n" + "=" * 80 + "\n"
THIN_SEP = "-" * 60


def make_generator(seed: int) -> CodingContentGenerator:
    rng.reset()
    rng.init(seed)
    tokenizer = Tokenizer.from_pretrained("gpt2")
    return CodingContentGenerator(config=PromptConfig(), tokenizer=tokenizer)


def _get_blocks(msg: dict) -> list[dict]:
    """Extract content blocks from a message (handles both str and list)."""
    content = msg.get("content")
    if isinstance(content, list):
        return content
    return []


def classify_message(msg: dict) -> str:
    role = msg["role"]
    blocks = _get_blocks(msg)
    if role == "user":
        has_results = any(b.get("type") == "tool_result" for b in blocks)
        return "tool_results" if has_results else "user"
    if role == "assistant":
        text_blocks = [b for b in blocks if b.get("type") == "text"]
        tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
        n = len(tool_blocks)
        if n > 0 and text_blocks:
            return f"text+{n}tool" if n == 1 else f"text+{n}tools"
        if n > 0:
            return f"silent+{n}tool" if n == 1 else f"silent+{n}tools"
        return "text_only"
    return role


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def analyze_conversation(
    msgs: list[dict], tokenizer: Tokenizer, full: bool = False
) -> dict:
    """Analyze a generated conversation and return stats."""
    stats: dict = {
        "total_messages": len(msgs),
        "styles": [],
        "tool_names": [],
        "assistant_tokens": 0,
        "tool_result_tokens": 0,
        "parallel_calls": 0,
    }

    for msg in msgs:
        style = classify_message(msg)
        stats["styles"].append(style)
        blocks = _get_blocks(msg)

        if msg["role"] == "assistant":
            for b in blocks:
                if b.get("type") == "text" and b.get("text"):
                    stats["assistant_tokens"] += count_tokens(tokenizer, b["text"])
                elif b.get("type") == "tool_use":
                    stats["tool_names"].append(b["name"])
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            if len(tool_uses) > 1:
                stats["parallel_calls"] += 1

        elif msg["role"] == "user":
            for b in blocks:
                if b.get("type") == "tool_result" and isinstance(b.get("content"), str):
                    stats["tool_result_tokens"] += count_tokens(tokenizer, b["content"])

    # Verify ordering: tool results follow their assistant message
    tool_use_ids_seen = set()
    ordering_ok = True
    for msg in msgs:
        blocks = _get_blocks(msg)
        for b in blocks:
            if b.get("type") == "tool_use":
                tool_use_ids_seen.add(b["id"])
            elif (
                b.get("type") == "tool_result"
                and b["tool_use_id"] not in tool_use_ids_seen
            ):
                ordering_ok = False

    stats["ordering_valid"] = ordering_ok
    stats["unique_call_ids"] = len(tool_use_ids_seen)
    return stats


def print_conversation(
    msgs: list[dict], tokenizer: Tokenizer, full: bool = False
) -> None:
    """Pretty-print a conversation."""
    for i, msg in enumerate(msgs):
        role = msg["role"]
        style = classify_message(msg)
        blocks = _get_blocks(msg)

        if role == "assistant" and any(b.get("type") == "tool_use" for b in blocks):
            text_parts = [b["text"] for b in blocks if b.get("type") == "text"]
            text_preview = repr(text_parts[0][:80]) + "..." if text_parts else "None"
            print(f"  [{i}] {role} ({style})")
            print(f"       text: {text_preview}")
            for b in blocks:
                if b.get("type") == "tool_use":
                    inp = b["input"]
                    inp_short = {
                        k: (v[:40] + "..." if isinstance(v, str) and len(v) > 40 else v)
                        for k, v in inp.items()
                    }
                    print(f"       call: {b['name']}({inp_short})  id={b['id']}")

        elif role == "user":
            for b in blocks:
                if b.get("type") == "tool_result":
                    content = b.get("content", "")
                    tokens = (
                        count_tokens(tokenizer, content)
                        if isinstance(content, str)
                        else 0
                    )
                    print(
                        f"  [{i}] tool_result -> {b['tool_use_id']}  ({tokens} tokens)"
                    )
                    if full:
                        preview = (
                            repr(content[:60]) + "..."
                            if len(str(content)) > 60
                            else repr(content)
                        )
                        print(f"       {preview}")

        elif role == "assistant":
            text_parts = [b["text"] for b in blocks if b.get("type") == "text"]
            text = text_parts[0] if text_parts else ""
            tokens = count_tokens(tokenizer, text) if text else 0
            print(f"  [{i}] {role} ({style}, {tokens} tokens)")
            if full:
                print(f"       {repr(text[:100])}...")
            else:
                print(f"       {repr(text[:80])}...")

        else:
            print(f"  [{i}] {role}: {repr(str(msg.get('content', ''))[:60])}...")


def run_demo(token_budgets: list[int], seeds: int, full: bool) -> None:
    tokenizer = Tokenizer.from_pretrained("gpt2")

    # Aggregate stats across all runs
    all_tool_names: Counter = Counter()
    all_styles: Counter = Counter()
    total_convos = 0
    total_parallel = 0

    for budget in token_budgets:
        print(SEPARATOR)
        print(f"TOKEN BUDGET: {budget}")
        print(SEPARATOR)

        for seed in range(seeds):
            gen = make_generator(seed)
            msgs = gen.generate_response(budget)
            stats = analyze_conversation(msgs, tokenizer, full)

            total_convos += 1
            total_parallel += stats["parallel_calls"]
            all_tool_names.update(stats["tool_names"])
            all_styles.update(stats["styles"])

            print(
                f"\n  Seed {seed}: {stats['total_messages']} messages, "
                f"~{stats['assistant_tokens']} assistant tokens, "
                f"~{stats['tool_result_tokens']} tool_result tokens, "
                f"ordering={'OK' if stats['ordering_valid'] else 'BROKEN'}"
            )
            print(f"  Flow: {' -> '.join(stats['styles'])}")

            print_conversation(msgs, tokenizer, full)

    # Summary
    print(SEPARATOR)
    print("AGGREGATE ANALYSIS")
    print(SEPARATOR)
    print(f"Conversations generated: {total_convos}")
    print(f"Parallel tool calls:     {total_parallel}")
    print()

    print("Message style distribution:")
    for style, count in all_styles.most_common():
        pct = count / sum(all_styles.values()) * 100
        print(f"  {style:<20} {count:>4}  ({pct:.1f}%)")

    print()
    print("Tool name distribution:")
    for name, count in all_tool_names.most_common():
        pct = count / sum(all_tool_names.values()) * 100
        print(f"  {name:<20} {count:>4}  ({pct:.1f}%)")

    # Validate tool_use input dicts
    print()
    input_errors = 0
    for seed in range(seeds):
        for budget in token_budgets:
            gen = make_generator(seed)
            for msg in gen.generate_response(budget):
                for b in _get_blocks(msg):
                    if b.get("type") == "tool_use" and not isinstance(
                        b.get("input"), dict
                    ):
                        input_errors += 1
    print(
        f"Tool input validation: {'ALL VALID' if input_errors == 0 else f'{input_errors} ERRORS'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[100, 300, 600, 1000],
        help="Token budgets to test (default: 100 300 600 1000)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds to test per budget (default: 3)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full message content (not just previews)",
    )
    args = parser.parse_args()

    run_demo(args.tokens, args.seeds, args.full)


if __name__ == "__main__":
    main()
