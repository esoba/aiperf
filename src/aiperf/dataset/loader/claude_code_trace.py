# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claude Code trace dataset loader for verbatim replay of JSONL session transcripts.

Loads Claude Code JSONL session transcripts (from ~/.claude/projects/.../sessions/*.jsonl)
and converts them to AIPerf conversations. Supports two modes:

- Verbatim mode (default): Sends exact content blocks (tool_use, tool_result, text,
  thinking) to the server using raw_content / assistant_prefill on Turn.
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
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.models import (
    ClaudeCodeApiCall,
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
            if (
                pending_request_id is not None
                and req_id is not None
                and req_id != pending_request_id
                and pending_assistants
            ):
                _flush()

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

        Returns:
            Dictionary of trace_id -> list containing one ClaudeCodeTrace.
        """
        path = Path(self.filename)
        jsonl_files: list[Path] = []

        if path.is_dir():
            jsonl_files = sorted(path.glob("*.jsonl"))
            self.info(f"Loading {len(jsonl_files)} JSONL files from {path}")
        elif path.is_file():
            jsonl_files = [path]
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        result: dict[str, list[ClaudeCodeTrace]] = {}

        for jsonl_file in jsonl_files:
            records = self._parse_jsonl(jsonl_file)
            if not records:
                continue

            session_id = self._extract_session_id(records, jsonl_file)
            system_prompt = self._extract_system_prompt(records)

            # Filter to user/assistant records only
            conversation_records = [
                r for r in records if r.type in ("user", "assistant")
            ]
            if not conversation_records:
                continue

            api_calls = _group_records_into_api_calls(conversation_records)
            if not api_calls:
                continue

            trace_id = f"cc_{jsonl_file.stem}"
            trace = ClaudeCodeTrace(
                id=trace_id,
                session_id=session_id,
                api_calls=api_calls,
                system_prompt=system_prompt,
            )
            result[trace_id] = [trace]

        self.info(f"Loaded {len(result)} Claude Code traces")
        return result

    def convert_to_conversations(
        self, data: dict[str, list[ClaudeCodeTrace]]
    ) -> list[Conversation]:
        """Convert Claude Code traces to AIPerf conversation objects.

        In verbatim mode: Creates Turns with raw_content (user content blocks)
        and assistant_prefill (assistant response blocks).
        In synthetic mode: Extracts token counts and generates synthetic content.
        """
        conversations: list[Conversation] = []

        for trace_id, traces in data.items():
            trace = traces[0]
            conversation = Conversation(
                session_id=trace_id,
                system_message=trace.system_prompt,
            )

            prev_timestamp_ms: float | None = None

            for api_call in trace.api_calls:
                delay_ms: float | None = None
                if prev_timestamp_ms is not None and api_call.timestamp_ms is not None:
                    delay_ms = api_call.timestamp_ms - prev_timestamp_ms
                    if delay_ms < 0:
                        delay_ms = None
                if api_call.timestamp_ms is not None:
                    prev_timestamp_ms = api_call.timestamp_ms

                turn = self._build_turn(api_call, delay_ms)
                conversation.turns.append(turn)

            if conversation.turns:
                conversations.append(conversation)

        self.info(
            f"Converted {len(conversations)} traces to conversations "
            f"({sum(len(c.turns) for c in conversations)} total turns)"
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
            assistant_prefill=api_call.assistant_content,
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
