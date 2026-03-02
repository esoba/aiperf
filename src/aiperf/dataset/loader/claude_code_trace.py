# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claude Code trace dataset loader for verbatim replay of JSONL session transcripts.

Loads Claude Code JSONL session transcripts (from ~/.claude/projects/.../sessions/*.jsonl)
and converts them to AIPerf conversations. Supports two modes:

- Verbatim mode (default): Sends exact content blocks (tool_use, tool_result, text,
  thinking) to the server using raw_content on Turn, with discard_responses=True
  on the Conversation.
- Synthetic mode: Extracts timing/token metadata and generates synthetic content,
  reusing existing PromptGenerator infrastructure.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.config.user_config import UserConfig
from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.models.dataset_models import SubagentSpawnInfo
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.models import (
    ClaudeCodeApiCall,
    ClaudeCodeManifest,
    ClaudeCodeTrace,
    ClaudeCodeTraceRecord,
)
from aiperf.plugin.enums import DatasetSamplingStrategy


def _parse_timestamp_ms(ts: str | None) -> float | None:
    """Parse ISO timestamp string to milliseconds since epoch."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp() * MILLIS_PER_SECOND
    except (ValueError, TypeError):
        return None


def _merge_assistant_content(
    records: list[ClaudeCodeTraceRecord],
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None, str | None]:
    """Merge multiple assistant records (same requestId) into combined content.

    Returns:
        Tuple of (content_blocks, merged_usage, model, stop_reason).
    """
    content_blocks: list[dict[str, Any]] = []
    merged_usage: dict[str, Any] = {}
    model: str | None = None
    stop_reason: str | None = None

    for rec in records:
        msg = rec.message or {}
        if not model and msg.get("model"):
            model = msg["model"]
        if msg.get("stop_reason"):
            stop_reason = msg["stop_reason"]

        usage = msg.get("usage", {})
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            if key in usage:
                merged_usage[key] = max(merged_usage.get(key, 0), usage[key])

        msg_content = msg.get("content", [])
        if isinstance(msg_content, list):
            content_blocks.extend(msg_content)

    return content_blocks, merged_usage, model, stop_reason


def _group_records_into_api_calls(
    records: list[ClaudeCodeTraceRecord],
) -> list[ClaudeCodeApiCall]:
    """Group user/assistant record pairs into reconstructed API calls.

    Handles:
    - Multiple assistant records with the same requestId (streamed content blocks)
    - User records with string or list content
    - System records for system prompt extraction
    """
    api_calls: list[ClaudeCodeApiCall] = []
    pending_user: ClaudeCodeTraceRecord | None = None
    pending_assistants: list[ClaudeCodeTraceRecord] = []
    pending_request_id: str | None = None

    def _flush() -> None:
        nonlocal pending_user, pending_assistants, pending_request_id
        if pending_user is not None and pending_assistants:
            user_msg = pending_user.message or {}
            user_content = user_msg.get("content", "")

            content_blocks, usage, model, stop_reason = _merge_assistant_content(
                pending_assistants
            )

            timestamp_ms = _parse_timestamp_ms(pending_user.timestamp)

            api_calls.append(
                ClaudeCodeApiCall(
                    user_content=user_content,
                    assistant_content=content_blocks,
                    model=model,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_creation_input_tokens=usage.get(
                        "cache_creation_input_tokens", 0
                    ),
                    cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                    timestamp_ms=timestamp_ms,
                    stop_reason=stop_reason,
                )
            )

        pending_user = None
        pending_assistants = []
        pending_request_id = None

    for rec in records:
        if rec.type == "user":
            _flush()
            pending_user = rec
            pending_assistants = []
            pending_request_id = None

        elif rec.type == "assistant":
            req_id = rec.request_id

            # If request_id changes and we have accumulated assistants, flush
            # but preserve pending_user since the new response is for the same user turn
            if (
                pending_request_id is not None
                and req_id is not None
                and req_id != pending_request_id
                and pending_assistants
            ):
                saved_user = pending_user
                _flush()
                pending_user = saved_user

            pending_assistants.append(rec)
            if req_id is not None:
                pending_request_id = req_id

    _flush()
    return api_calls


class ClaudeCodeTraceLoader(BaseFileLoader):
    """Dataset loader for Claude Code JSONL session transcripts.

    Loads JSONL files where each line is a JSON object with type, message,
    sessionId, timestamp, requestId, etc. Supports verbatim replay (sending
    exact content blocks) and synthetic mode (using token counts for prompt
    generation).
    """

    def __init__(
        self,
        *,
        filename: str,
        user_config: UserConfig,
        prompt_generator: Any = None,
        **kwargs,
    ):
        super().__init__(filename=filename, user_config=user_config, **kwargs)
        self._prompt_generator = prompt_generator
        self._synthetic_mode = prompt_generator is not None
        self._manifest: ClaudeCodeManifest | None = None
        self._parent_trace_id: str | None = None
        self._subagent_traces: dict[str, tuple[ClaudeCodeTrace, int, bool]] = {}

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data or filename.

        Returns True if:
        - filename is a .jsonl file containing Claude Code trace records, OR
        - filename is a directory containing .jsonl files with trace records, OR
        - data dict has 'type' in ('user', 'assistant') and 'message' key
        """
        if filename is not None:
            path = Path(filename)
            if path.is_file() and path.suffix == ".jsonl":
                return cls._probe_jsonl(path)
            if path.is_dir():
                return any(cls._probe_jsonl(jsonl) for jsonl in path.glob("*.jsonl"))

        if data is not None:
            return data.get("type") in ("user", "assistant") and "message" in data

        return False

    @classmethod
    def _probe_jsonl(cls, path: Path) -> bool:
        """Check if a JSONL file contains Claude Code trace records.

        Reads the first few lines looking for records with type in
        (user, assistant, system) and a message field.
        """
        try:
            with open(path, "rb") as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    raw = orjson.loads(line)
                    if not isinstance(raw, dict):
                        continue
                    rec_type = raw.get("type")
                    if rec_type in ("user", "assistant", "system") and "message" in raw:
                        return True
        except Exception:
            pass
        return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[ClaudeCodeTrace]]:
        """Load Claude Code traces from JSONL file(s).

        When loading from a directory with a _manifest.json, uses the manifest
        to link parent and subagent sessions. Otherwise each JSONL file becomes
        an independent conversation.

        Returns:
            Dictionary of trace_id -> list containing one ClaudeCodeTrace.
        """
        path = Path(self.filename)

        if path.is_dir():
            manifest = self._load_manifest(path)
            if manifest is not None:
                return self._load_with_manifest(path, manifest)
            jsonl_files = sorted(path.glob("*.jsonl"))
            self.info(f"Loading {len(jsonl_files)} JSONL files from {path}")
        elif path.is_file():
            jsonl_files = [path]
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        result: dict[str, list[ClaudeCodeTrace]] = {}
        for jsonl_file in jsonl_files:
            trace = self._load_single_file(jsonl_file)
            if trace is not None:
                result[trace.id] = [trace]

        self.info(f"Loaded {len(result)} Claude Code traces")
        return result

    def _load_with_manifest(
        self, directory: Path, manifest: ClaudeCodeManifest
    ) -> dict[str, list[ClaudeCodeTrace]]:
        """Load traces using manifest to link parent/child sessions."""
        result: dict[str, list[ClaudeCodeTrace]] = {}

        # Load parent
        parent_path = directory / manifest.parent
        parent_trace = self._load_single_file(parent_path)
        if parent_trace is None:
            self.warning(f"Parent file {manifest.parent} could not be loaded")
            return result

        result[parent_trace.id] = [parent_trace]

        # Store manifest info for convert_to_conversations
        self._manifest = manifest
        self._parent_trace_id = parent_trace.id

        # Load subagent children
        self._subagent_traces: dict[str, tuple[ClaudeCodeTrace, int, bool]] = {}
        for link in manifest.subagents:
            child_path = directory / link.file
            child_trace = self._load_single_file(child_path)
            if child_trace is not None:
                result[child_trace.id] = [child_trace]
                self._subagent_traces[child_trace.id] = (
                    child_trace,
                    link.spawn_after_api_call,
                    link.is_background,
                )

        self.info(
            f"Loaded {len(result)} Claude Code traces "
            f"(1 parent, {len(self._subagent_traces)} subagents)"
        )
        return result

    def convert_to_conversations(
        self, data: dict[str, list[ClaudeCodeTrace]]
    ) -> list[Conversation]:
        """Convert Claude Code traces to AIPerf conversation objects.

        In verbatim mode: Creates Turns with raw_content (user content blocks)
        and sets discard_responses=True on the Conversation.
        In synthetic mode: Extracts token counts and generates synthetic content.

        When a manifest links parent/child sessions, produces SubagentSpawnInfo
        on the parent and sets agent_depth on children.
        """
        conversations: list[Conversation] = []
        child_conversations: list[Conversation] = []

        # Build spawn info mapping: parent_trace_id -> {turn_index -> [(spawn_id, child_id, is_bg)]}
        spawn_map: dict[str, dict[int, list[tuple[str, str, bool]]]] = {}
        child_trace_ids: set[str] = set()
        if self._subagent_traces:
            for spawn_counter, (child_id, (_, spawn_after, is_bg)) in enumerate(
                self._subagent_traces.items()
            ):
                spawn_id = f"s{spawn_counter}"
                child_trace_ids.add(child_id)
                parent_id = self._parent_trace_id
                spawn_map.setdefault(parent_id, {}).setdefault(spawn_after, []).append(
                    (spawn_id, child_id, is_bg)
                )

        for trace_id, traces in data.items():
            trace = traces[0]
            is_child = trace_id in child_trace_ids
            conversation = Conversation(
                session_id=trace_id,
                system_message=trace.system_prompt,
                agent_depth=1 if is_child else 0,
                discard_responses=not self._synthetic_mode,
            )

            prev_timestamp_ms: float | None = None
            # Collect spawn_id assignments for join turns (applied after loop)
            join_turn_spawn_ids: dict[int, list[str]] = {}

            for call_idx, api_call in enumerate(trace.api_calls):
                delay_ms: float | None = None
                if prev_timestamp_ms is not None and api_call.timestamp_ms is not None:
                    delay_ms = api_call.timestamp_ms - prev_timestamp_ms
                    if delay_ms < 0:
                        delay_ms = None
                if api_call.timestamp_ms is not None:
                    prev_timestamp_ms = api_call.timestamp_ms

                turn = self._build_turn(api_call, delay_ms)
                conversation.turns.append(turn)

                # Register SubagentSpawnInfo; mark the join turn with spawn_id
                spawns_at_turn = spawn_map.get(trace_id, {}).get(call_idx, [])
                for spawn_id, child_id, is_bg in spawns_at_turn:
                    join_idx = min(call_idx + 1, len(trace.api_calls) - 1)
                    join_turn_spawn_ids.setdefault(join_idx, []).append(spawn_id)
                    conversation.subagent_spawns.append(
                        SubagentSpawnInfo(
                            spawn_id=spawn_id,
                            child_conversation_ids=[child_id],
                            join_turn_index=join_idx,
                            is_background=is_bg,
                        )
                    )

            # Apply spawn_ids to join turns (adaptive_scale checks next_meta.subagent_spawn_ids)
            for join_idx, spawn_ids in join_turn_spawn_ids.items():
                if join_idx < len(conversation.turns):
                    conversation.turns[join_idx].subagent_spawn_ids = spawn_ids

            if not conversation.turns:
                continue

            if is_child:
                child_conversations.append(conversation)
            else:
                conversations.append(conversation)

        conversations.extend(child_conversations)

        total_turns = sum(len(c.turns) for c in conversations)
        self.info(
            f"Converted {len(conversations)} traces to conversations "
            f"({total_turns} total turns, {len(child_conversations)} subagent children)"
        )
        return conversations

    def _build_turn(self, api_call: ClaudeCodeApiCall, delay_ms: float | None) -> Turn:
        """Build a Turn from a single API call."""
        if self._synthetic_mode:
            return self._build_synthetic_turn(api_call, delay_ms)
        return self._build_verbatim_turn(api_call, delay_ms)

    @staticmethod
    def _build_verbatim_turn(
        api_call: ClaudeCodeApiCall, delay_ms: float | None
    ) -> Turn:
        """Build a verbatim Turn that uses raw content blocks directly."""
        return Turn(
            role="user",
            model=api_call.model,
            delay=delay_ms,
            max_tokens=api_call.output_tokens or 4096,
            input_tokens=api_call.input_tokens,
            raw_content=api_call.user_content,
        )

    def _build_synthetic_turn(
        self, api_call: ClaudeCodeApiCall, delay_ms: float | None
    ) -> Turn:
        """Build a synthetic Turn using token counts for prompt generation."""
        prompt_tokens = max(1, api_call.input_tokens)
        prompt = self._prompt_generator.generate_prompt(prompt_tokens)

        return Turn(
            role="user",
            model=api_call.model,
            delay=delay_ms,
            max_tokens=api_call.output_tokens or 4096,
            input_tokens=api_call.input_tokens,
            texts=[Text(name="text", contents=[prompt])],
        )

    def _load_single_file(self, filepath: Path) -> ClaudeCodeTrace | None:
        """Load a single JSONL file into a ClaudeCodeTrace."""
        records = self._parse_jsonl(filepath)
        if not records:
            return None

        session_id = self._extract_session_id(records, filepath)
        system_prompt = self._extract_system_prompt(records)

        conversation_records = [r for r in records if r.type in ("user", "assistant")]
        if not conversation_records:
            return None

        api_calls = _group_records_into_api_calls(conversation_records)
        if not api_calls:
            return None

        trace_id = f"cc_{filepath.stem}"
        return ClaudeCodeTrace(
            id=trace_id,
            session_id=session_id,
            api_calls=api_calls,
            system_prompt=system_prompt,
        )

    @staticmethod
    def _load_manifest(directory: Path) -> ClaudeCodeManifest | None:
        """Load _manifest.json from a directory if present."""
        manifest_path = directory / "_manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            with open(manifest_path, "rb") as f:
                raw = orjson.loads(f.read())
            return ClaudeCodeManifest.model_validate(raw)
        except Exception:
            return None

    @staticmethod
    def _parse_jsonl(filepath: Path) -> list[ClaudeCodeTraceRecord]:
        """Parse a JSONL file into a list of trace records."""
        records: list[ClaudeCodeTraceRecord] = []
        with open(filepath, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = orjson.loads(line)
                    records.append(ClaudeCodeTraceRecord.model_validate(raw))
                except Exception:
                    continue
        return records

    @staticmethod
    def _extract_session_id(
        records: list[ClaudeCodeTraceRecord], filepath: Path
    ) -> str:
        """Extract session ID from records or derive from filename."""
        for rec in records:
            if rec.session_id:
                return rec.session_id
        return filepath.stem

    @staticmethod
    def _extract_system_prompt(records: list[ClaudeCodeTraceRecord]) -> str | None:
        """Extract system prompt from system records if present."""
        for rec in records:
            if rec.type == "system":
                msg = rec.message or {}
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    return content
                if isinstance(content, list):
                    texts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    if texts:
                        return "\n".join(texts)
        return None
