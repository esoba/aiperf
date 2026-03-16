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

import orjson

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


def classify_message(msg: dict) -> str:
    role = msg["role"]
    if role == "tool":
        return "tool_result"
    if role == "assistant":
        has_tc = "tool_calls" in msg
        has_text = isinstance(msg.get("content"), str) and msg["content"]
        if has_tc:
            n = len(msg["tool_calls"])
            if has_text:
                return f"text+{n}tool" if n == 1 else f"text+{n}tools"
            return f"silent+{n}tool" if n == 1 else f"silent+{n}tools"
        return "text_only"
    # Anthropic content-block format
    if role == "user":
        content = msg.get("content", [])
        if isinstance(content, list) and any(
            b.get("type") == "tool_result" for b in content
        ):
            return "tool_results"
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

        if msg["role"] == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content:
                stats["assistant_tokens"] += count_tokens(tokenizer, content)
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    stats["tool_names"].append(tc["function"]["name"])
                if len(msg["tool_calls"]) > 1:
                    stats["parallel_calls"] += 1

        elif msg["role"] == "tool":
            content = msg.get("content", "")
            if content:
                stats["tool_result_tokens"] += count_tokens(tokenizer, content)

    # Verify ordering: tool results follow their assistant message
    emitted_ids: set[str] = set()
    ordering_ok = True
    for msg in msgs:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                emitted_ids.add(tc["id"])
        elif msg["role"] == "tool" and msg.get("tool_call_id") not in emitted_ids:
            ordering_ok = False

    stats["ordering_valid"] = ordering_ok
    stats["unique_call_ids"] = len(emitted_ids)
    return stats


def print_conversation(
    msgs: list[dict], tokenizer: Tokenizer, full: bool = False
) -> None:
    """Pretty-print a conversation."""
    for i, msg in enumerate(msgs):
        role = msg["role"]
        style = classify_message(msg)

        if role == "assistant" and "tool_calls" in msg:
            content = msg.get("content") or ""
            text_preview = (
                repr(content[:80]) + "..." if len(content) > 80 else repr(content)
            )
            print(f"  [{i}] {role} ({style})")
            print(f"       text: {text_preview}")
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                args = orjson.loads(fn["arguments"])
                args_short = {
                    k: (v[:40] + "..." if isinstance(v, str) and len(v) > 40 else v)
                    for k, v in args.items()
                }
                print(f"       call: {fn['name']}({args_short})  id={tc['id']}")

        elif role == "tool":
            content = msg.get("content", "")
            tokens = count_tokens(tokenizer, content) if content else 0
            print(f"  [{i}] tool_result -> {msg['tool_call_id']}  ({tokens} tokens)")
            if full:
                preview = (
                    repr(content[:60]) + "..." if len(content) > 60 else repr(content)
                )
                print(f"       {preview}")

        elif role == "assistant":
            content = msg.get("content", "")
            tokens = count_tokens(tokenizer, content) if content else 0
            print(f"  [{i}] {role} ({style}, {tokens} tokens)")
            print(f"       {repr(content[:80])}...")

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

    # Validate tool call arguments
    print()
    arg_errors = 0
    for seed in range(seeds):
        for budget in token_budgets:
            gen = make_generator(seed)
            for msg in gen.generate_response(budget):
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        try:
                            parsed = orjson.loads(tc["function"]["arguments"])
                            if not isinstance(parsed, dict):
                                arg_errors += 1
                        except Exception:
                            arg_errors += 1
    print(
        f"Tool argument validation: {'ALL VALID' if arg_errors == 0 else f'{arg_errors} ERRORS'}"
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
