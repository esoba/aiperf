# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic trajectory JSONL loader for verbatim replay of multi-turn tool-calling conversations.

Each JSONL line contains one API call with cumulative messages -- turn N has
the full message history from turns 0..N. Records are grouped by conversation_id,
sorted by conversation_idx, and converted to Conversations with cumulative
raw_messages turns (replaces_history=True on every turn).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.models import AgenticTrajectoryRecord
from aiperf.plugin.enums import DatasetSamplingStrategy


class AgenticTrajectoryLoader(BaseFileLoader):
    """Dataset loader for agentic trajectory JSONL files.

    Expects a JSONL file where each line has conversation_id, conversation_idx,
    messages (cumulative), and optionally tools. Groups by conversation_id and
    converts to Turns with cumulative raw_messages (replaces_history=True).
    Post-processing in the composer pre-formats payloads via the endpoint.
    """

    def __init__(
        self,
        *,
        filename: str,
        user_config: UserConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(filename=filename, user_config=user_config, **kwargs)

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Return True if data has conversation_id (str), conversation_idx (int), messages (list)."""
        if data is not None:
            return (
                isinstance(data.get("conversation_id"), str)
                and isinstance(data.get("conversation_idx"), int)
                and isinstance(data.get("messages"), list)
            )
        if filename is not None:
            path = Path(filename)
            if not path.is_file() or path.suffix != ".jsonl":
                return False
            try:
                with open(path, "rb") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = orjson.loads(line)
                        return (
                            isinstance(record.get("conversation_id"), str)
                            and isinstance(record.get("conversation_idx"), int)
                            and isinstance(record.get("messages"), list)
                        )
            except Exception:
                return False
        return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[AgenticTrajectoryRecord]]:
        """Load JSONL line-by-line, group by conversation_id, sort by conversation_idx.

        Returns:
            Dictionary of conversation_id -> sorted list of AgenticTrajectoryRecord.
        """
        groups: dict[str, list[AgenticTrajectoryRecord]] = defaultdict(list)
        path = Path(self.filename)

        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = orjson.loads(line)
                record = AgenticTrajectoryRecord(
                    conversation_id=raw["conversation_id"],
                    conversation_idx=raw["conversation_idx"],
                    messages=raw["messages"],
                    tools=raw.get("tools", []),
                )
                groups[record.conversation_id].append(record)

        for records in groups.values():
            records.sort(key=lambda r: r.conversation_idx)

        self.info(
            f"Loaded {len(groups)} conversations "
            f"({sum(len(r) for r in groups.values())} total turns)"
        )
        return dict(groups)

    def convert_to_conversations(
        self, data: dict[str, list[AgenticTrajectoryRecord]]
    ) -> list[Conversation]:
        """Convert grouped trajectory records to AIPerf Conversations.

        For each conversation group:
        - system_message from messages[0] of first turn (role=system)
        - tools from first record with non-empty tools
        - Each turn gets cumulative raw_messages with replaces_history=True
        - discard_responses=True since each turn carries its own full context
        """
        conversations: list[Conversation] = []

        for _conv_id, records in data.items():
            if not records:
                continue

            first_record = records[0]

            system_message = _extract_system_message(first_record.messages)

            tools: list[dict[str, Any]] | None = None
            for record in records:
                if record.tools:
                    tools = record.tools
                    break

            session_id = self.session_id_generator.next()
            turns: list[Turn] = []

            for record in records:
                non_system = _strip_system(record.messages)
                turns.append(
                    Turn(
                        role="user",
                        raw_messages=non_system,
                        replaces_history=True,
                    )
                )

            conversations.append(
                Conversation(
                    session_id=session_id,
                    turns=turns,
                    system_message=system_message,
                    tools=tools,
                    discard_responses=True,
                )
            )

        total_turns = sum(len(c.turns) for c in conversations)
        self.info(
            f"Converted {len(conversations)} conversations ({total_turns} total turns)"
        )
        return conversations


def _strip_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove all leading system messages from a message list."""
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        idx += 1
    return messages[idx:] if idx > 0 else messages


def _extract_system_message(messages: list[dict[str, Any]]) -> str | None:
    """Extract and join text content from all leading system messages."""
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != "system":
            break
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.append(
                "\n".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            )
    return "\n".join(parts) if parts else None
