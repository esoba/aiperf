# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-style telemetry JSONL to mooncake JSONL converter."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from tqdm import tqdm

from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.converters.hashing import texts_to_hashes_and_lengths
from aiperf.dataset.converters.telemetry_config import TelemetryConfig

AGENT_PREFIX_MAP = {
    "You are a Deep Research agent": "deep_coordinator",
    "Gather and synthesize comprehe": "research_worker",
    "For the given task, generate a": "research_planner",
    "Current date and time:": "shallow_agent",
}


class TelemetryConverter:
    """Convert OpenAI-style telemetry traces to mooncake-style JSONL.

    Filters to llm_call events, classifies agent types from system prompt
    prefixes or user message content, tokenizes messages, and produces
    session records with agent_type and priority annotations.
    """

    def __init__(self, config: TelemetryConfig) -> None:
        self._config = config

    def convert(self) -> list[dict[str, Any]]:
        """Load telemetry JSONL, classify agents, tokenize, and produce mooncake records."""
        c = self._config

        print(f"Loading {c.input_file}...")
        events = _load_and_sort(str(c.input_file))
        print(f"Loaded {len(events)} llm_call events")
        print(f"Using tokenizer: {c.tokenizer}")
        print(f"Block size: {c.block_size}")

        for event in events:
            event["_agent_type"] = _classify_agent(event)

        all_texts: list[str] = []
        for event in tqdm(events, desc="Extracting messages"):
            all_texts.append(_messages_to_text(event))

        print(f"Tokenizing and hashing {len(all_texts)} texts...")
        tokenizer = Tokenizer.from_pretrained(c.tokenizer)
        all_hash_ids, all_input_lengths = texts_to_hashes_and_lengths(
            tokenizer, all_texts, c.block_size
        )

        records: list[dict[str, Any]] = []
        for event, input_length, hash_ids in zip(
            events, all_input_lengths, all_hash_ids, strict=True
        ):
            records.append(
                {
                    "session_id": event["session_id"],
                    "agent_type": event["_agent_type"],
                    "input_length": input_length,
                    "output_length": _get_output_tokens(event),
                    "hash_ids": hash_ids,
                    "priority": event.get("latency_priority", "MEDIUM"),
                }
            )

        if c.verbose:
            self._print_statistics(records)
        return records

    def default_output_filename(self) -> str:
        """Derive output filename from input filename."""
        return self._config.input_file.stem + "_mooncake.jsonl"

    @staticmethod
    def _print_statistics(records: list[dict[str, Any]]) -> None:
        if not records:
            print("No data to report statistics on.")
            return

        print("\n" + "=" * 60)
        print("CONVERSION STATISTICS")
        print("=" * 60)

        sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in records:
            sessions[entry["session_id"]].append(entry)

        turns = [len(t) for t in sessions.values()]
        print(f"\nSessions: {len(sessions)}")
        print(
            f"Turns per session: min={min(turns)}, "
            f"max={max(turns)}, "
            f"avg={sum(turns) / len(turns):.1f}"
        )
        print(f"Total LLM calls: {len(records)}")

        print("\nAgent types:")
        for agent, count in Counter(e["agent_type"] for e in records).most_common():
            print(f"  {agent}: {count}")

        print("\nPriorities:")
        for prio, count in Counter(e["priority"] for e in records).most_common():
            print(f"  {prio}: {count}")

        input_lengths = [e["input_length"] for e in records]
        output_lengths = [e["output_length"] for e in records]
        hash_lengths = [len(e["hash_ids"]) for e in records]

        print("\nInput Length (tokens):")
        print(f"  Min: {min(input_lengths)}")
        print(f"  Max: {max(input_lengths)}")
        print(f"  Avg: {sum(input_lengths) / len(input_lengths):.1f}")

        print("\nOutput Length (tokens):")
        print(f"  Min: {min(output_lengths)}")
        print(f"  Max: {max(output_lengths)}")
        print(f"  Avg: {sum(output_lengths) / len(output_lengths):.1f}")

        print("\nHash IDs per entry:")
        print(f"  Min: {min(hash_lengths)}")
        print(f"  Max: {max(hash_lengths)}")
        print(f"  Avg: {sum(hash_lengths) / len(hash_lengths):.1f}")

        print("=" * 60)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_and_sort(filepath: str) -> list[dict[str, Any]]:
    """Load telemetry JSONL, filter to llm_call events, sort by (session_id, timestamp)."""
    events: list[dict[str, Any]] = []
    with open(filepath) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("event_type") == "llm_call":
                events.append(obj)
    events.sort(key=lambda e: (e["session_id"], e["timestamp"]))
    return events


def _extract_system_prompt(event: dict[str, Any]) -> str | None:
    """Extract the system prompt text from an llm_call event."""
    rp = event.get("request_payload")
    if not isinstance(rp, dict):
        return None
    for msg in rp.get("messages", []):
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    return item["text"]
            return None
        return content
    return None


def _extract_first_user_message(event: dict[str, Any]) -> str:
    """Extract the first user message text from an llm_call event."""
    rp = event.get("request_payload")
    if not isinstance(rp, dict):
        return ""
    for msg in rp.get("messages", []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return str(content)
    return ""


def _classify_agent(event: dict[str, Any]) -> str:
    """Identify agent type from system prompt prefix or user message content."""
    sys_prompt = _extract_system_prompt(event)
    if sys_prompt:
        for prefix, agent_type in AGENT_PREFIX_MAP.items():
            if sys_prompt.startswith(prefix):
                return agent_type
        return "unknown"

    user_msg = _extract_first_user_message(event)
    if "Classify the user message" in user_msg:
        return "classifier"
    if "complexity analyzer" in user_msg:
        return "complexity_analyzer"
    return "unknown"


def _messages_to_text(event: dict[str, Any]) -> str:
    """Concatenate all message contents from request_payload.messages."""
    rp = event.get("request_payload")
    if not isinstance(rp, dict):
        return ""
    parts: list[str] = []
    for msg in rp.get("messages", []):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
    return "\n".join(parts)


def _get_output_tokens(event: dict[str, Any]) -> int:
    """Extract completion_tokens from response_payload.usage."""
    rp = event.get("response_payload")
    if not isinstance(rp, dict):
        return 0
    usage = rp.get("usage", {})
    if not isinstance(usage, dict):
        return 0
    return usage.get("completion_tokens", 0)
