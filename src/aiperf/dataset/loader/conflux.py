# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conflux dataset loader for verbatim replay of Claude Code proxy captures.

Loads JSON files produced by the Conflux proxy containing an array of API
request/response records with explicit agent_id/is_subagent fields for
thread grouping. Supports parent + subagent hierarchies with timestamp-based
inter-turn delays.
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Conversation, Turn
from aiperf.common.models.dataset_models import SubagentSpawnInfo
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.models import ConfluxRecord
from aiperf.plugin.enums import DatasetSamplingStrategy


def _parse_timestamp_ms(iso_str: str) -> float:
    """Parse an ISO timestamp string to milliseconds since epoch."""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.timestamp() * 1000


class ConfluxLoader(BaseFileLoader):
    """Dataset loader for Conflux proxy capture JSON files.

    Expects a JSON file containing an array of API request records, each with
    agent_id, is_subagent, messages, tools, model, and timestamp fields.
    """

    def __init__(
        self,
        *,
        filename: str,
        user_config: UserConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(filename=filename, user_config=user_config, **kwargs)
        self._groups: dict[str, list[ConfluxRecord]] = {}

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Return True if filename is a JSON file with Conflux-format records."""
        if filename is None:
            return False
        path = Path(filename)
        if not path.is_file() or path.suffix != ".json":
            return False
        try:
            with open(path, "rb") as f:
                raw = orjson.loads(f.read())
            if not isinstance(raw, list) or len(raw) == 0:
                return False
            first = raw[0]
            return (
                isinstance(first, dict)
                and "agent_id" in first
                and "is_subagent" in first
                and "messages" in first
            )
        except Exception:
            return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[ConfluxRecord]]:
        """Load and group Conflux records by agent_id.

        Filters out records with agent_id=None (e.g. haiku tool-use calls),
        parses into ConfluxRecord models, and groups by agent_id sorted by
        timestamp within each group.
        """
        with open(self.filename, "rb") as f:
            raw_records: list[dict] = orjson.loads(f.read())

        # Filter out records without agent_id
        filtered = [r for r in raw_records if r.get("agent_id") is not None]

        groups: dict[str, list[ConfluxRecord]] = {}
        for raw in filtered:
            record = ConfluxRecord.model_validate(raw)
            agent_id = record.agent_id
            if agent_id is not None:
                groups.setdefault(agent_id, []).append(record)

        # Sort each group by timestamp
        for records in groups.values():
            records.sort(key=lambda r: _parse_timestamp_ms(r.timestamp))

        self._groups = groups

        total_records = sum(len(recs) for recs in groups.values())
        self.info(
            f"Loaded {len(groups)} Conflux agent threads ({total_records} total records)"
        )

        return groups

    def convert_to_conversations(
        self, data: dict[str, list[ConfluxRecord]]
    ) -> list[Conversation]:
        """Convert grouped Conflux records to Conversation objects.

        The group with is_subagent=False is the parent; groups with
        is_subagent=True are children linked via SubagentSpawnInfo.
        """
        if not data:
            return []

        parent_id: str | None = None
        child_ids: list[str] = []

        for agent_id, records in data.items():
            if not records[0].is_subagent:
                parent_id = agent_id
            else:
                child_ids.append(agent_id)

        conversations: list[Conversation] = []
        child_conversations: list[Conversation] = []

        if parent_id is not None:
            parent_records = data[parent_id]
            parent_conv = self._build_conversation(
                parent_id, parent_records, is_child=False
            )

            for spawn_counter, child_agent_id in enumerate(child_ids):
                child_records = data[child_agent_id]
                child_conv = self._build_conversation(
                    child_agent_id, child_records, is_child=True
                )

                spawn_turn_index = self._find_spawn_point(parent_records, child_records)
                spawn_id = f"s{spawn_counter}"

                parent_conv.subagent_spawns.append(
                    SubagentSpawnInfo(
                        spawn_id=spawn_id,
                        child_conversation_ids=[child_conv.session_id],
                        join_turn_index=spawn_turn_index,
                        is_background=True,
                    )
                )

                if spawn_turn_index < len(parent_conv.turns):
                    parent_conv.turns[spawn_turn_index].subagent_spawn_ids.append(
                        spawn_id
                    )

                child_conversations.append(child_conv)

            conversations.append(parent_conv)
            conversations.extend(child_conversations)
        else:
            # No parent found, treat all groups as independent
            for agent_id, records in data.items():
                conversations.append(
                    self._build_conversation(agent_id, records, is_child=False)
                )

        total_turns = sum(len(c.turns) for c in conversations)
        self.info(
            f"Converted {len(conversations)} Conflux threads to conversations "
            f"({total_turns} total turns, {len(child_conversations)} subagent children)"
        )
        return conversations

    def _build_conversation(
        self,
        agent_id: str,
        records: list[ConfluxRecord],
        *,
        is_child: bool,
    ) -> Conversation:
        """Build a Conversation from a list of ConfluxRecords for one agent."""
        first = records[0]

        conversation = Conversation(
            session_id=f"conflux_{agent_id}",
            tools=first.tools or None,
            agent_depth=1 if is_child else 0,
            discard_responses=True,
        )

        speedup_ratio = self.user_config.input.synthesis.speedup_ratio

        for record in records:
            ts_ms = _parse_timestamp_ms(record.timestamp)
            if speedup_ratio > 0 and speedup_ratio != 1.0:
                ts_ms = ts_ms / speedup_ratio

            raw_payload = self._build_raw_payload(record)

            # Extract user content for raw_messages from the API messages
            api_messages = raw_payload.get("messages", [])
            user_content = self._extract_user_content(api_messages)
            raw_messages: list[dict[str, Any]] | None = None
            if user_content is not None:
                raw_messages = [{"role": "user", "content": user_content}]

            max_tokens: int | None = raw_payload.get("max_tokens")
            input_tokens = record.tokens.input if record.tokens else 0

            turn = Turn(
                role="user",
                model=record.model,
                timestamp=ts_ms,
                max_tokens=max_tokens or 4096,
                input_tokens=input_tokens,
                raw_messages=raw_messages,
                raw_payload=raw_payload,
            )
            conversation.turns.append(turn)

        return conversation

    @staticmethod
    def _build_raw_payload(record: ConfluxRecord) -> dict[str, Any]:
        """Build the raw API payload for a record.

        Prefers the base64-encoded request_body (the actual verbatim request)
        when available. Falls back to reconstructing from top-level fields,
        splitting the synthetic system-role message from the messages array.
        """
        if record.base64 and record.base64.get("request_body"):
            payload: dict[str, Any] = orjson.loads(
                base64.b64decode(record.base64["request_body"])
            )
            payload.pop("metadata", None)
            return payload

        # Fallback: reconstruct from top-level fields.
        # record.messages[0] is a synthetic system-role message prepended
        # by the proxy; the real API messages start at index 1.
        messages = record.messages
        system: list[dict[str, Any]] | None = None
        if messages and messages[0].get("role") == "system":
            system = messages[0].get("content")
            messages = messages[1:]

        raw_payload: dict[str, Any] = {
            "model": record.model,
            "messages": messages,
        }
        if system:
            raw_payload["system"] = system

        max_tokens: int | None = None
        temperature: float | None = None
        if record.hyperparameters is not None:
            max_tokens = record.hyperparameters.max_tokens
            temperature = record.hyperparameters.temperature
        if max_tokens is not None:
            raw_payload["max_tokens"] = max_tokens
        if record.is_streaming is not None:
            raw_payload["stream"] = record.is_streaming
        if record.tools:
            raw_payload["tools"] = record.tools
        if temperature is not None:
            raw_payload["temperature"] = temperature

        return raw_payload

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
    def _find_spawn_point(
        parent_records: list[ConfluxRecord],
        child_records: list[ConfluxRecord],
    ) -> int:
        """Find the parent turn index closest to the child's first request."""
        child_first_ts = _parse_timestamp_ms(child_records[0].timestamp)

        best_idx = 0
        best_diff = float("inf")
        for i, record in enumerate(parent_records):
            ts = _parse_timestamp_ms(record.timestamp)
            diff = abs(ts - child_first_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        return min(best_idx + 1, len(parent_records) - 1)
