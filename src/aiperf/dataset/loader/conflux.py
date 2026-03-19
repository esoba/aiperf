# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conflux dataset loader for timestamp-based replay of proxy captures.

Loads JSON files containing arrays of API request records. Groups records
by agent_id into independent Conversations with timestamp-based inter-turn
delays for fixed-schedule replay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.models import ConfluxRecord
from aiperf.plugin.enums import DatasetSamplingStrategy

_EXTRA_PARAMS_SKIP = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
    }
)


class ConfluxLoader(BaseFileLoader):
    """Dataset loader for Conflux proxy capture JSON files.

    Each agent_id group becomes an independent Conversation with
    zero-aligned timestamps for fixed-schedule replay.
    """

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode:
        return ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Return True if filename is a Conflux JSON file or directory."""
        if filename is None:
            return False
        path = Path(filename)
        if path.is_dir():
            first = next(path.glob("*.json"), None)
            return first is not None and cls._probe_file(first)
        return cls._probe_file(path)

    _PROBE_BYTES = 1 << 20  # 1MB

    @classmethod
    def _probe_file(cls, path: Path) -> bool:
        """Return True if a single file looks like a Conflux JSON capture.

        Reads only a bounded prefix to avoid loading multi-MB captures
        during auto-detection.
        """
        if not path.is_file() or path.suffix != ".json":
            return False
        try:
            with path.open("rb") as f:
                head = f.read(cls._PROBE_BYTES)
        except OSError:
            return False
        if not head or head.lstrip()[:1] != b"[":
            return False
        # Extract first object from the JSON array prefix
        try:
            stripped = head.rstrip(b", \t\n\r")
            if stripped.endswith(b"]"):
                decoded = orjson.loads(stripped)
            else:
                decoded = orjson.loads(stripped + b"]")
        except orjson.JSONDecodeError:
            return False
        if not isinstance(decoded, list) or not decoded:
            return False
        try:
            ConfluxRecord.model_validate(decoded[0])
            return True
        except Exception:
            return False

    def load_dataset(self) -> dict[str, list[ConfluxRecord]]:
        """Load and group Conflux records by agent_id."""
        path = Path(self.filename)
        if path.is_dir():
            return self._load_directory(path)
        return self._load_single_file(self.filename)

    def _load_directory(self, path: Path) -> dict[str, list[ConfluxRecord]]:
        """Load all JSON files in a directory as independent sessions."""
        json_files = sorted(path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No .json files found in directory: {self.filename}"
            )

        all_groups: dict[str, list[ConfluxRecord]] = {}
        for file_idx, json_file in enumerate(json_files):
            file_groups = self._load_single_file(str(json_file), prefix=f"f{file_idx}_")
            all_groups.update(file_groups)

        total_records = sum(len(recs) for recs in all_groups.values())
        self.info(
            f"Loaded {len(all_groups)} agent threads from "
            f"{len(json_files)} files ({total_records} total records) in {path.name}/"
        )
        return all_groups

    def _load_single_file(
        self, filename: str, prefix: str = ""
    ) -> dict[str, list[ConfluxRecord]]:
        """Load and group records from a single JSON file."""
        raw_records: list[dict[str, Any]] = orjson.loads(Path(filename).read_bytes())

        include_utility = self.user_config.input.conflux_include_utility_calls

        groups: dict[str, list[ConfluxRecord]] = {}
        utility_count = 0

        for raw in raw_records:
            record = ConfluxRecord.model_validate(raw)
            if record.agent_id is not None:
                key = f"{prefix}{record.agent_id}"
                groups.setdefault(key, []).append(record)
            else:
                if include_utility:
                    groups[f"{prefix}_utility_{utility_count}"] = [record]
                utility_count += 1

        for records in groups.values():
            records.sort(key=lambda r: r.timestamp)

        if not prefix:
            total_records = sum(len(recs) for recs in groups.values())
            action = "included" if include_utility else "skipped"
            utility_label = f"{utility_count} utility calls {action}"
            self.info(
                f"Loaded {len(groups)} agent threads + {utility_label} "
                f"({total_records} total records)"
            )

        return groups

    def convert_to_conversations(
        self, data: dict[str, list[ConfluxRecord]]
    ) -> list[Conversation]:
        """Convert grouped Conflux records to Conversation objects."""
        conversations = [
            self._build_conversation(agent_id, records)
            for agent_id, records in data.items()
        ]

        total_turns = sum(len(c.turns) for c in conversations)
        self.info(
            f"Converted {len(conversations)} conversations ({total_turns} total turns)"
        )
        return conversations

    def _build_conversation(
        self,
        agent_id: str,
        records: list[ConfluxRecord],
    ) -> Conversation:
        """Build a Conversation from a list of ConfluxRecords for one agent."""
        conversation = Conversation(session_id=f"conflux_{agent_id}")

        for record in records:
            input_tokens = record.tokens.input if record.tokens else None

            max_tokens = None
            if record.tokens is not None:
                total_output = record.tokens.output + record.tokens.output_reasoning
                max_tokens = total_output or None

            turn = Turn(
                timestamp=record.timestamp,
                max_tokens=max_tokens,
                input_tokens=input_tokens,
                raw_messages=record.messages,
                raw_tools=record.tools or None,
                extra_params=self._extract_extra_params(record),
            )
            conversation.turns.append(turn)

        return conversation

    @staticmethod
    def _extract_extra_params(record: ConfluxRecord) -> dict[str, Any] | None:
        """Extract per-turn hyperparameter overrides from a ConfluxRecord."""
        if not record.hyperparameters:
            return None
        params = {
            k: v
            for k, v in record.hyperparameters.items()
            if k not in _EXTRA_PARAMS_SKIP and v is not None
        }
        return params or None
