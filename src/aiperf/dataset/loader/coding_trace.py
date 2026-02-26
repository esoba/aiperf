# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coding trace dataset loader for agentic coding session replay.

Loads coding traces from the kv-cache-tester format: a directory of JSON files
where each file is one agentic coding session with a sequence of LLM requests.
Flattens nested subagent requests and converts to AIPerf conversations.

Note on hash_ids: The trace format includes hash_ids per request, but these are
session-local KV cache block identifiers with no global coordination across
conversations. They are used for cache behavior analysis (hit rates, pull-backs)
but NOT for prompt generation. Each conversation generates independent prompts
sized to match the recorded input_tokens count.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.config.user_config import UserConfig
from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.models import CodingTrace, CodingTraceRequest
from aiperf.plugin.enums import DatasetSamplingStrategy


class CodingTraceLoader(BaseFileLoader):
    """Dataset loader for agentic coding traces (kv-cache-tester format).

    Loads coding traces from a directory of JSON files or a single JSON file.
    Each trace represents one coding session with a sequence of LLM requests.
    Nested subagent requests are flattened into a linear sequence.

    Prompt sizing uses deltas: in kv-cache-tester traces, input_tokens represents
    the TOTAL context at that turn (including all prior turns). Since AIPerf's
    worker accumulates previous turns when building HTTP requests, each new user
    message only needs to cover the delta: input_tokens - prev_input_tokens -
    prev_output_tokens. This prevents context inflation where cumulative turns
    far exceed the trace's intended context size.

    Example trace format:
    ```json
    {
      "id": "trace_001",
      "models": ["claude-sonnet-4-20250514"],
      "block_size": 64,
      "tool_tokens": 5000,
      "system_tokens": 3000,
      "requests": [
        {"t": 0.0, "type": "s", "in": 1000, "out": 500, "hash_ids": [1, 2, 3]}
      ]
    }
    ```
    """

    def __init__(
        self,
        *,
        filename: str,
        prompt_generator: PromptGenerator,
        user_config: UserConfig,
        **kwargs,
    ):
        super().__init__(filename=filename, user_config=user_config, **kwargs)
        self.prompt_generator = prompt_generator
        self._max_isl = user_config.input.synthesis.max_isl
        self._warm_prefix_pct = user_config.input.warm_prefix_pct
        self._skipped_max_isl = 0
        self._skipped_min_requests = 0

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data or filename.

        Returns True if:
        - filename is a directory containing *.json files, OR
        - data dict has 'requests' key with a list value and an 'id' key
        """
        if filename is not None:
            path = Path(filename)
            if path.is_dir():
                return any(path.glob("*.json"))

        if data is not None:
            return isinstance(data.get("requests"), list) and "id" in data

        return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[CodingTrace]]:
        """Load coding traces from directory or file.

        Returns:
            Dictionary of conversation_id to list of CodingTrace (one per conversation).
        """
        path = Path(self.filename)
        raw_traces: list[dict] = []

        if path.is_dir():
            json_files = sorted(path.glob("*.json"))
            self.info(f"Loading {len(json_files)} trace files from {path}")
            for json_file in json_files:
                with open(json_file, "rb") as f:
                    raw_traces.append(orjson.loads(f.read()))
        else:
            with open(path, "rb") as f:
                content = f.read()
            data = orjson.loads(content)
            if isinstance(data, list):
                raw_traces.extend(data)
            else:
                raw_traces.append(data)

        result: dict[str, list[CodingTrace]] = {}
        for raw in raw_traces:
            trace = CodingTrace.model_validate(raw)

            # Flatten nested subagent requests
            flat_requests = self._flatten_requests(trace.requests)

            # Truncate conversation at first request exceeding max_isl.
            # Unlike per-request filtering, this preserves conversation continuity:
            # all requests up to the exceeding point are kept, none after.
            if self._max_isl is not None:
                truncated: list[CodingTraceRequest] = []
                for r in flat_requests:
                    if r.input_tokens > self._max_isl:
                        break
                    truncated.append(r)
                skipped = len(flat_requests) - len(truncated)
                self._skipped_max_isl += skipped
                flat_requests = truncated

            if len(flat_requests) < 2:
                self._skipped_min_requests += 1
                continue

            # Replace trace requests with flattened version
            trace.requests = flat_requests

            # Use trace id as conversation id
            conv_id = trace.id
            result[conv_id] = [trace]

        if self._skipped_max_isl > 0:
            self.info(
                f"Skipped {self._skipped_max_isl} requests exceeding "
                f"max_isl of {self._max_isl}"
            )
        if self._skipped_min_requests > 0:
            self.info(
                f"Skipped {self._skipped_min_requests} traces with fewer than 2 requests"
            )
        self.info(f"Loaded {len(result)} coding traces with flattened requests")

        return result

    def _flatten_requests(
        self, requests: list[CodingTraceRequest]
    ) -> list[CodingTraceRequest]:
        """Recursively flatten nested subagent requests into a linear sequence."""
        flat: list[CodingTraceRequest] = []
        for req in requests:
            # Skip container-only entries (subagent wrappers with no token counts)
            if req.input_tokens > 0:
                flat.append(req)
            # Recursively flatten nested subagent requests
            if req.requests:
                flat.extend(self._flatten_requests(req.requests))
        return flat

    def convert_to_conversations(
        self, data: dict[str, list[CodingTrace]]
    ) -> list[Conversation]:
        """Convert coding traces to AIPerf conversation objects.

        Uses delta-based prompt sizing: each turn's prompt covers only the NEW
        tokens needed beyond what prior turns already contribute. Since AIPerf's
        worker accumulates all previous user + assistant turns, the delta is:
          turn 0: delta = input_tokens (full, no history)
          turn N: delta = max(1, input_tokens - prev_input_tokens - prev_output_tokens)

        A single base prompt is generated at max(deltas) and truncated for smaller
        deltas to avoid per-request tokenizer calls.

        Args:
            data: Dictionary of conversation_id to list of CodingTrace.

        Returns:
            List of Conversation objects.
        """
        warm_prefix_text = self._generate_warm_prefix(data)

        total_conversations = len(data)
        total_requests = sum(len(t[0].requests) for t in data.values())
        self.info(
            f"Converting {total_conversations} traces "
            f"({total_requests} requests) to conversations"
        )

        # Compute deltas for all requests across all conversations
        start = time.perf_counter()
        deltas_by_trace: dict[str, list[int]] = {}
        unique_deltas: set[int] = set()

        for conv_id, traces in data.items():
            trace = traces[0]
            trace_deltas: list[int] = []
            for i, req in enumerate(trace.requests):
                if i == 0:
                    delta = req.input_tokens
                else:
                    prev = trace.requests[i - 1]
                    delta = max(
                        1, req.input_tokens - prev.input_tokens - prev.output_tokens
                    )
                trace_deltas.append(delta)
                unique_deltas.add(delta)
            deltas_by_trace[conv_id] = trace_deltas

        max_delta = max(unique_deltas) if unique_deltas else 0
        self.info(
            f"Found {len(unique_deltas)} unique delta lengths (max: {max_delta} tokens)"
        )

        # Generate one base prompt at max delta, build lookup by truncation
        prompt_by_delta: dict[int, str] = {}
        if max_delta > 0:
            base_prompt = self.prompt_generator.generate_prompt(max_delta)
            chars_per_token = len(base_prompt) / max_delta
            for delta in unique_deltas:
                char_count = max(1, int(delta * chars_per_token))
                prompt_by_delta[delta] = base_prompt[:char_count]

        gen_elapsed = time.perf_counter() - start
        self.info(
            f"Generated prompts for {len(unique_deltas)} unique delta lengths "
            f"({gen_elapsed:.1f}s)"
        )

        # Build conversation objects
        self.info("Building conversation objects")
        conversations = []
        for conv_id, traces in data.items():
            trace = traces[0]
            conversation = Conversation(session_id=conv_id)

            if warm_prefix_text:
                conversation.system_message = warm_prefix_text

            trace_deltas = deltas_by_trace[conv_id]
            prev_t = None
            for i, req in enumerate(trace.requests):
                delay_ms = None
                if prev_t is not None:
                    delay_sec = req.t - prev_t
                    delay_ms = delay_sec * MILLIS_PER_SECOND
                prev_t = req.t

                delta = trace_deltas[i]
                prompt = prompt_by_delta.get(delta, "")
                turn = Turn(
                    delay=delay_ms,
                    max_tokens=req.output_tokens,
                    texts=[Text(name="text", contents=[prompt])],
                )
                conversation.turns.append(turn)

            conversations.append(conversation)

        return conversations

    def _generate_warm_prefix(self, data: dict[str, list[CodingTrace]]) -> str | None:
        """Generate a shared warm prefix for KV cache pre-fill.

        Sized to warm_prefix_pct * max(tool_tokens + system_tokens) across all traces.
        Uses a fixed seed for reproducibility.
        """
        if self._warm_prefix_pct <= 0:
            return None

        max_context_tokens = 0
        for traces in data.values():
            trace = traces[0]
            context = trace.tool_tokens + trace.system_tokens
            max_context_tokens = max(max_context_tokens, context)

        if max_context_tokens == 0:
            return None

        prefix_tokens = int(max_context_tokens * self._warm_prefix_pct)
        if prefix_tokens <= 0:
            return None

        return self.prompt_generator.generate(mean=prefix_tokens, stddev=0, hash_ids=[])
