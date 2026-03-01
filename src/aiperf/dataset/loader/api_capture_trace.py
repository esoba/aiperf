# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API capture trace dataset loader for verbatim replay.

Loads captured API request bodies from mitmproxy-style capture directories containing
capture.jsonl (metadata log) and req_XXXX.json (complete request bodies). Supports
multi-thread detection (parent + subagent children) based on system prompt hashing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.config.user_config import UserConfig
from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.models import Conversation, Turn
from aiperf.common.models.dataset_models import SubagentSpawnInfo
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.models import ApiCaptureApiCall, ApiCaptureTrace
from aiperf.plugin.enums import DatasetSamplingStrategy


def _extract_system_text(system_blocks: list[dict]) -> str | None:
    """Join text from system blocks into a single string."""
    texts = [
        b.get("text", "")
        for b in system_blocks
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    joined = "\n".join(t for t in texts if t)
    return joined or None


def _thread_key(system_blocks: list[dict]) -> str:
    """Compute a grouping key from system block content."""
    return hashlib.md5(
        orjson.dumps(system_blocks, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()[:12]


def _is_prefetch(req: dict) -> bool:
    """Return True if a request is a prefetch (not a real API call)."""
    if req.get("stream") is None:
        return True
    if req.get("max_tokens") is None:
        return True
    if req.get("max_tokens", 0) <= 1:
        return True
    return not req.get("system")


class ApiCaptureTraceLoader(BaseFileLoader):
    """Dataset loader for mitmproxy-style API capture directories.

    Expects a directory containing:
    - capture.jsonl: metadata log with request/response pairs
    - req_XXXX.json: complete API request bodies
    """

    def __init__(
        self,
        *,
        filename: str,
        user_config: UserConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(filename=filename, user_config=user_config, **kwargs)
        self._traces: list[ApiCaptureTrace] = []

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Return True if filename is a directory with capture.jsonl and req_*.json."""
        if filename is None:
            return False
        path = Path(filename)
        if not path.is_dir():
            return False
        has_capture = (path / "capture.jsonl").is_file()
        has_reqs = any(path.glob("req_*.json"))
        return has_capture and has_reqs

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[ApiCaptureTrace]]:
        """Load API capture traces from a capture directory.

        1. Parse capture.jsonl for metadata
        2. Identify last session via call_index resets
        3. Correlate req files with response metadata
        4. Filter prefetches, group into threads by system prompt hash

        Returns:
            Dictionary of trace_id -> list containing one ApiCaptureTrace.
        """
        directory = Path(self.filename)
        capture_entries = self._parse_capture_jsonl(directory / "capture.jsonl")

        # Split into request/response by direction
        request_entries = [
            e for e in capture_entries if e.get("direction") == "request"
        ]
        response_entries = [
            e for e in capture_entries if e.get("direction") == "response"
        ]

        # Identify last session via call_index resets
        last_session_requests = self._extract_last_session(request_entries)

        # Build response map: call_index -> response entry
        response_map: dict[int, dict] = {}
        for resp in response_entries:
            ci = resp.get("call_index")
            if ci is not None:
                response_map[ci] = resp

        # Load req files and correlate
        api_calls_by_thread: dict[str, list[ApiCaptureApiCall]] = {}
        system_blocks_by_thread: dict[str, list[dict]] = {}

        for req_entry in last_session_requests:
            ci = req_entry["call_index"]
            req_file = directory / f"req_{ci:04d}.json"
            if not req_file.is_file():
                continue

            req_body = self._load_json(req_file)
            if req_body is None:
                continue

            if _is_prefetch(req_body):
                continue

            # Correlate with response metadata
            resp = response_map.get(ci, {})
            usage = resp.get("usage", {})

            timestamp_s = req_entry.get("timestamp")
            timestamp_ms = (
                timestamp_s * MILLIS_PER_SECOND if timestamp_s is not None else None
            )

            api_call = ApiCaptureApiCall(
                messages=req_body.get("messages", []),
                system=req_body.get("system", []),
                tools=req_body.get("tools", []),
                model=req_body.get("model"),
                max_tokens=req_body.get("max_tokens"),
                stream=req_body.get("stream"),
                thinking=req_body.get("thinking"),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                timestamp_ms=timestamp_ms,
                stop_reason=resp.get("stop_reason"),
            )

            key = _thread_key(req_body.get("system", []))
            api_calls_by_thread.setdefault(key, []).append(api_call)
            if key not in system_blocks_by_thread:
                system_blocks_by_thread[key] = req_body.get("system", [])

        # Build traces, ordered by first timestamp within each thread
        result: dict[str, list[ApiCaptureTrace]] = {}
        self._traces = []

        for key, calls in api_calls_by_thread.items():
            calls.sort(key=lambda c: c.timestamp_ms or 0)
            trace_id = f"api_capture_{key}"
            system_text = _extract_system_text(system_blocks_by_thread.get(key, []))

            trace = ApiCaptureTrace(
                id=trace_id,
                api_calls=calls,
                system_prompt_text=system_text,
                thread_key=key,
            )
            result[trace_id] = [trace]
            self._traces.append(trace)

        self.info(
            f"Loaded {len(result)} API capture threads "
            f"({sum(len(t.api_calls) for t in self._traces)} total API calls)"
        )
        return result

    def convert_to_conversations(
        self, data: dict[str, list[ApiCaptureTrace]]
    ) -> list[Conversation]:
        """Convert API capture traces to AIPerf conversation objects.

        - Identifies parent (most API calls) vs subagent children
        - Extracts system_message and tools from the first call in each thread
        - Builds turns with raw_content, assistant_prefill, and raw_payload
        - Links children to parent via SubagentSpawnInfo
        """
        traces = [t[0] for t in data.values()]
        if not traces:
            return []

        # Identify parent thread (most API calls)
        parent_trace = max(traces, key=lambda t: len(t.api_calls))
        child_traces = [t for t in traces if t.id != parent_trace.id]

        conversations: list[Conversation] = []

        # Build parent conversation
        parent_conv = self._build_conversation(parent_trace, is_child=False)

        # Build child conversations and link to parent
        child_conversations: list[Conversation] = []
        for spawn_counter, child_trace in enumerate(child_traces):
            child_conv = self._build_conversation(child_trace, is_child=True)

            # Find parent turn closest to child's first request
            join_turn_index = self._find_spawn_point(parent_trace, child_trace)
            spawn_id = f"s{spawn_counter}"

            parent_conv.subagent_spawns.append(
                SubagentSpawnInfo(
                    spawn_id=spawn_id,
                    child_conversation_ids=[child_conv.session_id],
                    join_turn_index=join_turn_index,
                )
            )

            # Mark the join turn with spawn_id
            if join_turn_index < len(parent_conv.turns):
                parent_conv.turns[join_turn_index].subagent_spawn_id = spawn_id

            child_conversations.append(child_conv)

        conversations.append(parent_conv)
        conversations.extend(child_conversations)

        total_turns = sum(len(c.turns) for c in conversations)
        self.info(
            f"Converted {len(conversations)} traces to conversations "
            f"({total_turns} total turns, {len(child_conversations)} subagent children)"
        )
        return conversations

    def _build_conversation(
        self, trace: ApiCaptureTrace, *, is_child: bool
    ) -> Conversation:
        """Build a Conversation from a single ApiCaptureTrace."""
        first_call = trace.api_calls[0]

        conversation = Conversation(
            session_id=trace.id,
            system_message=trace.system_prompt_text,
            tools=first_call.tools or None,
            is_subagent_child=is_child,
        )

        prev_timestamp_ms: float | None = None

        for call_idx, api_call in enumerate(trace.api_calls):
            delay_ms: float | None = None
            if prev_timestamp_ms is not None and api_call.timestamp_ms is not None:
                delay_ms = api_call.timestamp_ms - prev_timestamp_ms
                if delay_ms < 0:
                    delay_ms = None
            if api_call.timestamp_ms is not None:
                prev_timestamp_ms = api_call.timestamp_ms

            # Extract user content: last user-role message
            raw_content = self._extract_user_content(api_call.messages)

            # Extract assistant prefill from next request's messages
            assistant_prefill = None
            if call_idx + 1 < len(trace.api_calls):
                next_call = trace.api_calls[call_idx + 1]
                assistant_prefill = self._extract_assistant_prefill(
                    api_call.messages, next_call.messages
                )

            # Build raw_payload (complete API request for verbatim replay)
            raw_payload: dict[str, Any] = {
                "model": api_call.model,
                "messages": api_call.messages,
                "max_tokens": api_call.max_tokens,
                "stream": api_call.stream,
            }
            if api_call.system:
                raw_payload["system"] = api_call.system
            if api_call.tools:
                raw_payload["tools"] = api_call.tools
            if api_call.thinking is not None:
                raw_payload["thinking"] = api_call.thinking

            turn = Turn(
                role="user",
                model=api_call.model,
                delay=delay_ms,
                max_tokens=api_call.max_tokens or 4096,
                input_tokens=api_call.input_tokens,
                raw_content=raw_content,
                assistant_prefill=assistant_prefill,
                raw_payload=raw_payload,
            )
            conversation.turns.append(turn)

        return conversation

    @staticmethod
    def _extract_user_content(
        messages: list[dict],
    ) -> str | list[dict[str, Any]] | None:
        """Extract content from the last user-role message."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content")
        return None

    @staticmethod
    def _extract_assistant_prefill(
        current_messages: list[dict],
        next_messages: list[dict],
    ) -> list[dict[str, Any]] | None:
        """Extract assistant response that appears between current and next request.

        The next request's messages contain the current request's messages plus
        an assistant response and a new user message. The assistant response is
        the content between the end of current_messages and the last user message
        in next_messages.
        """
        current_len = len(current_messages)
        if current_len >= len(next_messages):
            return None

        # Messages after the current request's messages, before the last user message
        new_messages = next_messages[current_len:]
        assistant_blocks: list[dict[str, Any]] = []

        for msg in new_messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    assistant_blocks.extend(content)
                elif isinstance(content, str):
                    assistant_blocks.append({"type": "text", "text": content})

        return assistant_blocks or None

    @staticmethod
    def _find_spawn_point(
        parent_trace: ApiCaptureTrace, child_trace: ApiCaptureTrace
    ) -> int:
        """Find the parent turn index closest to the child's first request."""
        child_first_ts = child_trace.api_calls[0].timestamp_ms
        if child_first_ts is None:
            return 0

        best_idx = 0
        best_diff = float("inf")
        for i, call in enumerate(parent_trace.api_calls):
            if call.timestamp_ms is not None:
                diff = abs(call.timestamp_ms - child_first_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

        return min(best_idx + 1, len(parent_trace.api_calls) - 1)

    @staticmethod
    def _parse_capture_jsonl(filepath: Path) -> list[dict]:
        """Parse capture.jsonl into a list of metadata entries."""
        entries: list[dict] = []
        with open(filepath, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(orjson.loads(line))
                except Exception:
                    continue
        return entries

    @staticmethod
    def _extract_last_session(request_entries: list[dict]) -> list[dict]:
        """Extract request entries from the last session.

        Sessions are separated by call_index resets (when call_index decreases
        or resets to 0).
        """
        if not request_entries:
            return []

        session_start = 0
        prev_ci = -1
        for i, entry in enumerate(request_entries):
            ci = entry.get("call_index", 0)
            if ci <= prev_ci:
                session_start = i
            prev_ci = ci

        return request_entries[session_start:]

    @staticmethod
    def _load_json(filepath: Path) -> dict | None:
        """Load a JSON file, returning None on error."""
        try:
            with open(filepath, "rb") as f:
                return orjson.loads(f.read())
        except Exception:
            return None
