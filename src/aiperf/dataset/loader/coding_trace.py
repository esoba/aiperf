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
from aiperf.dataset.loader.models import (
    CodingTrace,
    CodingTraceRequest,
    TraceStatistics,
)
from aiperf.plugin.enums import DatasetSamplingStrategy

# Fraction of previous hash_ids that must be removed to detect a context reset
_PULLBACK_THRESHOLD = 0.10


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
        self._min_requests = user_config.input.synthesis.min_requests
        self._warm_prefix_pct = user_config.input.warm_prefix_pct
        self._output_token_budget_ratio = user_config.input.output_token_budget_ratio
        self._skipped_max_isl = 0
        self._skipped_min_requests = 0
        # Per-conversation parallel annotations: conv_id -> {request_index -> (group_id, branch)}
        self._parallel_annotations: dict[str, dict[int, tuple[str, int]]] = {}

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

            # Flatten nested subagent requests and detect streaming/non-streaming pairs
            flat_requests = self._flatten_requests(trace.requests)
            self._detect_request_pairs(flat_requests)
            # Detect parallel groups from subagent tree structure
            parallel_annotations = self._detect_parallel_groups(
                trace.requests, flat_requests
            )

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

            # Split at context resets (pull-backs)
            segments = self._detect_pullbacks(flat_requests)

            # Build identity-based annotation lookup for O(1) per-request
            annotated_ids: dict[int, tuple[str, int]] = {}
            for flat_idx, ann in parallel_annotations.items():
                annotated_ids[id(flat_requests[flat_idx])] = ann

            for seg_idx, segment in enumerate(segments):
                if len(segment) < self._min_requests:
                    self._skipped_min_requests += 1
                    continue

                seg_trace = trace.model_copy(update={"requests": segment})

                conv_id = f"{trace.id}_seg{seg_idx}" if len(segments) > 1 else trace.id
                result[conv_id] = [seg_trace]

                # Remap parallel annotations to segment-local indices
                seg_annotations: dict[int, tuple[str, int]] = {}
                for local_idx, req in enumerate(segment):
                    if (ann := annotated_ids.get(id(req))) is not None:
                        seg_annotations[local_idx] = ann
                if seg_annotations:
                    self._parallel_annotations[conv_id] = seg_annotations

        if self._skipped_max_isl > 0:
            self.info(
                f"Skipped {self._skipped_max_isl} requests exceeding "
                f"max_isl of {self._max_isl}"
            )
        if self._skipped_min_requests > 0:
            self.info(
                f"Skipped {self._skipped_min_requests} traces with fewer than "
                f"{self._min_requests} requests"
            )
        self.info(f"Loaded {len(result)} coding traces with flattened requests")
        self._log_trace_statistics(result)

        return result

    def _flatten_requests(
        self,
        requests: list[CodingTraceRequest],
        base_time: float = 0.0,
    ) -> list[CodingTraceRequest]:
        """Recursively flatten nested subagent requests into a chronological sequence.

        Converts subagent-relative timestamps to absolute time and sorts the
        result chronologically so inter-request delays are computed correctly.
        """
        flat: list[CodingTraceRequest] = []
        for req in requests:
            abs_time = base_time + req.t
            # Skip container-only entries (subagent wrappers with no token counts)
            if req.input_tokens > 0:
                req.t = abs_time
                flat.append(req)
            # Recursively flatten nested subagent requests
            if req.requests:
                flat.extend(self._flatten_requests(req.requests, base_time=abs_time))
        flat.sort(key=lambda r: r.t)
        return flat

    @staticmethod
    def _detect_parallel_groups(
        original_requests: list[CodingTraceRequest],
        flat_requests: list[CodingTraceRequest],
    ) -> dict[int, tuple[str, int]]:
        """Detect subagent nodes with >1 child and return parallel annotations.

        Walks the original tree structure to find parent requests whose children
        were flattened. When a parent has multiple child requests, those children
        form a parallel group in the flat list.

        Returns:
            Mapping of flat request index -> (group_id, branch_index).
        """
        flat_index: dict[int, int] = {id(r): i for i, r in enumerate(flat_requests)}
        annotations: dict[int, tuple[str, int]] = {}
        group_counter = 0

        def walk(requests: list[CodingTraceRequest]) -> None:
            nonlocal group_counter
            for req in requests:
                if not req.requests:
                    continue
                child_indices: list[int] = []
                for child in req.requests:
                    if id(child) in flat_index:
                        child_indices.append(flat_index[id(child)])
                    if child.requests:
                        walk([child])

                if len(child_indices) > 1:
                    group_id = f"g{group_counter}"
                    group_counter += 1
                    for branch, idx in enumerate(child_indices):
                        annotations[idx] = (group_id, branch)

        walk(original_requests)
        return annotations

    @staticmethod
    def _detect_request_pairs(requests: list[CodingTraceRequest]) -> int:
        """Detect consecutive streaming/non-streaming pairs with identical hash_ids.

        When a trace contains a streaming request immediately followed by a
        non-streaming request with the same hash_ids, the second is a repeat.
        Mark it so convert_to_conversations() can set delta=0 (re-send same content).

        Returns the number of pairs detected.
        """
        pairs_found = 0
        for i in range(1, len(requests)):
            prev, curr = requests[i - 1], requests[i]
            if (
                prev.hash_ids
                and curr.hash_ids
                and prev.hash_ids == curr.hash_ids
                and curr.type == "n"
            ):
                curr.is_pair_repeat = True
                pairs_found += 1
        return pairs_found

    @staticmethod
    def _detect_pullbacks(
        requests: list[CodingTraceRequest],
    ) -> list[list[CodingTraceRequest]]:
        """Split requests at context resets (pull-backs).

        A pull-back occurs when >10% of the previous request's hash_ids are
        removed in the next request, indicating the context was reset.
        Each segment becomes a separate conversation.

        Returns list of request segments (at least one).
        """
        if not requests:
            return []

        segments: list[list[CodingTraceRequest]] = [[requests[0]]]
        for i in range(1, len(requests)):
            prev_ids = set(requests[i - 1].hash_ids)
            curr_ids = set(requests[i].hash_ids)

            if prev_ids and curr_ids:
                removed = prev_ids - curr_ids
                if len(removed) / len(prev_ids) > _PULLBACK_THRESHOLD:
                    segments.append([])

            segments[-1].append(requests[i])

        return segments

    def convert_to_conversations(
        self, data: dict[str, list[CodingTrace]]
    ) -> list[Conversation]:
        """Convert coding traces to AIPerf conversation objects.

        Uses delta-based prompt sizing: each turn's prompt covers only the NEW
        tokens needed beyond what prior turns already contribute. Since AIPerf's
        worker accumulates all previous user + assistant turns, the delta is:
          turn 0: delta = input_tokens - prefix_tokens
          turn N: delta = input_tokens - prev_input - (prev_output * budget_ratio)

        Expected output shortfall is compensated via output_token_budget_ratio
        (default 0.8), which inflates deltas to account for model undergeneration.

        A warm prefix (if configured) is prepended to the first turn's user
        content to enable KV cache pre-fill across conversations.

        A single base prompt is generated at max(deltas) and truncated for smaller
        deltas to avoid per-request tokenizer calls.
        """
        warm_prefix_text = self._generate_warm_prefix(data)

        total_conversations = len(data)
        total_requests = sum(len(t[0].requests) for t in data.values())
        self.info(
            f"Converting {total_conversations} traces "
            f"({total_requests} requests) to conversations"
        )

        # Compute warm prefix token count for first-turn adjustment
        prefix_tokens = self._compute_prefix_tokens(data) if warm_prefix_text else 0

        # Compute deltas for all requests across all conversations
        start = time.perf_counter()
        deltas_by_trace: dict[str, list[int]] = {}
        unique_deltas: set[int] = set()

        for conv_id, traces in data.items():
            trace = traces[0]
            trace_deltas: list[int] = []
            for i, req in enumerate(trace.requests):
                if req.is_pair_repeat:
                    # Paired request re-sends the same conversation; minimal delta
                    delta = 1
                elif i == 0:
                    delta = max(1, req.input_tokens - prefix_tokens)
                else:
                    prev = trace.requests[i - 1]
                    effective_prev_output = int(
                        prev.output_tokens * self._output_token_budget_ratio
                    )
                    delta = max(
                        1, req.input_tokens - prev.input_tokens - effective_prev_output
                    )
                trace_deltas.append(delta)
                unique_deltas.add(delta)
            deltas_by_trace[conv_id] = trace_deltas

        max_delta = max(unique_deltas) if unique_deltas else 0
        self.info(
            f"Found {len(unique_deltas)} unique delta lengths (max: {max_delta} tokens)"
        )

        # Generate base prompts and build lookup by truncation.
        # If the generator supports typed prompts (CodingContentGenerator),
        # build separate lookups for "text" and "tool_result" content types.
        has_typed = (
            getattr(self.prompt_generator, "supports_typed_prompt", False) is True
        )
        prompt_by_delta: dict[int | tuple[str, int], str] = {}
        if max_delta > 0:
            if has_typed:
                for content_type in ("text", "tool_result"):
                    base = self.prompt_generator.generate_typed_prompt(
                        max_delta, content_type
                    )
                    cpt = len(base) / max_delta
                    for delta in unique_deltas:
                        char_count = max(1, int(delta * cpt))
                        prompt_by_delta[(content_type, delta)] = base[:char_count]
            else:
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

            trace_deltas = deltas_by_trace[conv_id]
            prev_t = None
            for i, req in enumerate(trace.requests):
                delay_ms = None
                if prev_t is not None:
                    delay_sec = req.t - prev_t
                    delay_ms = delay_sec * MILLIS_PER_SECOND
                prev_t = req.t

                delta = trace_deltas[i]
                if has_typed and req.input_types:
                    # Pick content type: "text" if any text blocks, else "tool_result"
                    ct = "text" if "text" in req.input_types else "tool_result"
                    prompt = prompt_by_delta.get((ct, delta), "")
                else:
                    prompt = prompt_by_delta.get(delta, "")
                # Apply parallel annotations if available
                par_group = None
                par_branch = None
                conv_annotations = self._parallel_annotations.get(conv_id)
                if conv_annotations and i in conv_annotations:
                    par_group, par_branch = conv_annotations[i]

                turn = Turn(
                    delay=delay_ms,
                    max_tokens=req.output_tokens,
                    input_tokens=req.input_tokens,
                    texts=[Text(name="text", contents=[prompt])],
                    hash_ids=req.hash_ids,
                    parallel_group=par_group,
                    parallel_branch=par_branch,
                )
                conversation.turns.append(turn)

            if warm_prefix_text and conversation.turns:
                first_turn = conversation.turns[0]
                original_text = (
                    first_turn.texts[0].contents[0] if first_turn.texts else ""
                )
                first_turn.texts = [
                    Text(name="text", contents=[warm_prefix_text + original_text])
                ]

            conversations.append(conversation)

        return conversations

    @staticmethod
    def _compute_trace_statistics(trace: CodingTrace) -> TraceStatistics:
        """Compute derived statistics for a single trace."""
        requests = trace.requests
        total_in = sum(r.input_tokens for r in requests)
        total_out = sum(r.output_tokens for r in requests)
        max_in = max((r.input_tokens for r in requests), default=0)

        # Estimate cache hit ratio from consecutive hash_id overlap
        total_blocks = 0
        hit_blocks = 0
        prev_hash_set: set[int] = set()
        for req in requests:
            current_hash_set = set(req.hash_ids)
            if prev_hash_set and current_hash_set:
                hits = len(current_hash_set & prev_hash_set)
                total_blocks += len(current_hash_set)
                hit_blocks += hits
            elif current_hash_set:
                total_blocks += len(current_hash_set)
            prev_hash_set = current_hash_set

        cache_hit_ratio = hit_blocks / total_blocks if total_blocks > 0 else 0.0

        return TraceStatistics(
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            num_requests=len(requests),
            max_input_tokens=max_in,
            estimated_cache_hit_ratio=cache_hit_ratio,
        )

    def _log_trace_statistics(self, data: dict[str, list[CodingTrace]]) -> None:
        """Compute and log aggregate statistics across all loaded traces."""
        if not data:
            return

        stats = [self._compute_trace_statistics(t[0]) for t in data.values()]
        total_in = sum(s.total_input_tokens for s in stats)
        total_out = sum(s.total_output_tokens for s in stats)
        total_reqs = sum(s.num_requests for s in stats)
        max_in = max(s.max_input_tokens for s in stats)
        avg_cache_hit = sum(s.estimated_cache_hit_ratio for s in stats) / len(stats)

        self.info(
            f"Trace statistics: {total_reqs} total requests, "
            f"{total_in:,} total input tokens, "
            f"{total_out:,} total output tokens, "
            f"max single request={max_in:,} input tokens, "
            f"avg cache hit ratio={avg_cache_hit:.1%}"
        )

    def _compute_prefix_tokens(self, data: dict[str, list[CodingTrace]]) -> int:
        """Compute warm prefix size in tokens (same formula as _generate_warm_prefix)."""
        max_context_tokens = 0
        for traces in data.values():
            trace = traces[0]
            context = trace.tool_tokens + trace.system_tokens
            max_context_tokens = max(max_context_tokens, context)
        if max_context_tokens == 0:
            return 0
        return max(0, int(max_context_tokens * self._warm_prefix_pct))

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
