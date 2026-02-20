# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NAT profiler JSON to mooncake JSONL converter."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from tqdm import tqdm

from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.converters.hashing import texts_to_hashes_and_lengths
from aiperf.dataset.converters.nat_config import NatConfig

_DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

_MODEL_MAPPING = {
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
}


class NatConverter:
    """Convert NAT profiler traces to mooncake-style JSONL.

    Extracts LLM calls from intermediate_steps by matching LLM_START/LLM_END
    events, tokenizes prompt text via a HuggingFace tokenizer, and produces
    multi-turn session records with hash IDs and inter-turn delay.
    """

    def __init__(self, config: NatConfig) -> None:
        self._config = config

    def convert(self) -> list[dict[str, Any]]:
        """Load NAT JSON, extract LLM calls, tokenize, and produce mooncake records."""
        c = self._config

        print(f"Loading {c.input_file}...")
        requests = _load_json_robust(str(c.input_file))
        print(f"Loaded {len(requests)} requests")

        tokenizer_name = c.tokenizer or _infer_tokenizer(requests)
        print(f"Using tokenizer: {tokenizer_name}")
        print(f"Block size: {c.block_size}")

        requests = requests[c.skip_requests :]
        if c.num_requests is not None:
            requests = requests[: c.num_requests]
        print(f"Processing {len(requests)} requests...")

        all_entries: list[tuple[str, int, str]] = []
        for req in tqdm(requests, desc="Extracting LLM calls"):
            request_number = req.get("request_number", 0)
            session_id = f"conv_{request_number}"
            for call in _extract_llm_calls(req):
                text = _chat_inputs_to_text(call["chat_inputs"])
                if not text:
                    print(f"Warning: Empty text in request {request_number}, skipping")
                    continue
                all_entries.append((session_id, call["completion_tokens"], text))

        if not all_entries:
            print("No valid LLM calls found")
            return []

        all_texts = [e[2] for e in all_entries]
        print(f"Tokenizing and hashing {len(all_texts)} texts...")
        tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        all_hash_ids, all_input_lengths = texts_to_hashes_and_lengths(
            tokenizer, all_texts, c.block_size
        )

        records: list[dict[str, Any]] = []
        seen_sessions: set[str] = set()
        for (session_id, completion_tokens, _), hash_ids, input_length in zip(
            all_entries, all_hash_ids, all_input_lengths, strict=True
        ):
            entry: dict[str, Any] = {
                "session_id": session_id,
                "input_length": input_length,
                "output_length": completion_tokens,
                "hash_ids": hash_ids,
            }
            if session_id in seen_sessions:
                entry["delay"] = c.delay
            else:
                seen_sessions.add(session_id)
            records.append(entry)

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
        print(f"\nSessions (conversations): {len(sessions)}")
        print(
            f"Turns per session: min={min(turns)}, "
            f"max={max(turns)}, "
            f"avg={sum(turns) / len(turns):.1f}"
        )
        print(f"Total LLM calls: {len(records)}")

        input_lengths = [e["input_length"] for e in records]
        output_lengths = [e["output_length"] for e in records]
        hash_lengths = [len(e["hash_ids"]) for e in records]

        print("\nInput Length (prompt_tokens):")
        print(f"  Min: {min(input_lengths)}")
        print(f"  Max: {max(input_lengths)}")
        print(f"  Avg: {sum(input_lengths) / len(input_lengths):.1f}")

        print("\nOutput Length (completion_tokens):")
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


def _load_json_robust(filepath: str) -> list[dict[str, Any]]:
    """Load JSON, handling potentially truncated files via partial parse."""
    with open(filepath) as f:
        content = f.read()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Warning: JSON file appears truncated, attempting partial parse...")

    pattern = r'"request_number":\s*(\d+)'
    matches = list(re.finditer(pattern, content))
    if not matches:
        raise ValueError("No valid requests found in file")

    requests: list[dict[str, Any]] = []
    for i, match in enumerate(matches):
        start = content.rfind("{", 0, match.start())
        end = (
            content.rfind("{", 0, matches[i + 1].start())
            if i + 1 < len(matches)
            else len(content)
        )
        chunk = content[start:end].rstrip().rstrip(",")
        try:
            requests.append(json.loads(chunk))
        except json.JSONDecodeError:
            print(f"Warning: Could not parse request {match.group(1)}, skipping...")

    print(f"Successfully parsed {len(requests)} complete requests")
    return requests


def _extract_llm_calls(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Match LLM_START/LLM_END events by UUID to extract call metadata."""
    steps = request.get("intermediate_steps", [])

    llm_starts: dict[str, dict[str, Any]] = {}
    for step in steps:
        payload = step.get("payload", {})
        if payload.get("event_type") == "LLM_START":
            uuid = payload.get("UUID")
            if uuid:
                llm_starts[uuid] = payload

    llm_calls: list[dict[str, Any]] = []
    for step in steps:
        payload = step.get("payload", {})
        if payload.get("event_type") != "LLM_END":
            continue
        uuid = payload.get("UUID")
        if not uuid or uuid not in llm_starts:
            continue
        start_payload = llm_starts[uuid]
        token_usage = payload.get("usage_info", {}).get("token_usage", {})
        llm_calls.append(
            {
                "chat_inputs": start_payload.get("metadata", {}).get("chat_inputs", []),
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "model_name": start_payload.get("name", "unknown"),
                "event_timestamp": payload.get("event_timestamp"),
            }
        )

    llm_calls.sort(key=lambda x: x.get("event_timestamp", 0) or 0)
    return llm_calls


def _chat_inputs_to_text(chat_inputs: list[Any]) -> str:
    """Concatenate chat message contents into a single string."""
    if not chat_inputs:
        return ""
    parts: list[str] = []
    for msg in chat_inputs:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if content:
                parts.append(str(content))
        elif isinstance(msg, str):
            parts.append(msg)
    return "\n".join(parts)


def _infer_tokenizer(requests: list[dict[str, Any]]) -> str:
    """Try to infer tokenizer from model name in traces."""
    for req in requests[:5]:
        for call in _extract_llm_calls(req):
            model_name = call.get("model_name", "").lower()
            for pattern, tokenizer in _MODEL_MAPPING.items():
                if pattern in model_name:
                    print(f"Inferred tokenizer from model '{model_name}': {tokenizer}")
                    return tokenizer
    print(f"Could not infer tokenizer, using default: {_DEFAULT_TOKENIZER}")
    return _DEFAULT_TOKENIZER
