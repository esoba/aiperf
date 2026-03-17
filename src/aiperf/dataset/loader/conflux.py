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
import functools
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import PrerequisiteKind, TurnThreadingMode
from aiperf.common.models import Conversation, Turn
from aiperf.common.models.dataset_models import (
    ConversationOrigin,
    SubagentSpawnInfo,
    TurnGroundTruth,
    TurnPrerequisite,
)
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.models import ConfluxRecord
from aiperf.dataset.message_normalizer import normalize_messages
from aiperf.plugin.enums import DatasetSamplingStrategy


@functools.lru_cache(maxsize=4096)
def _parse_timestamp_ms(iso_str: str) -> float:
    """Parse an ISO timestamp string to milliseconds since epoch."""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.timestamp() * 1000


def _messages_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build messages list from a decoded request payload, prepending system when present."""
    msgs = list(payload.get("messages", []))
    system = payload.get("system")
    if system is not None:
        msgs.insert(0, {"role": "system", "content": system})
    return msgs


def _decode_messages(record: ConfluxRecord) -> list[dict[str, Any]]:
    """Decode the messages array from a ConfluxRecord, preferring base64 payload."""
    if record.base64 and record.base64.get("request_body"):
        payload: dict[str, Any] = orjson.loads(
            base64.b64decode(record.base64["request_body"])
        )
        return _messages_from_payload(payload)
    return list(record.messages)


def _record_end_ms(record: ConfluxRecord) -> float:
    """Best-effort completion timestamp for a Conflux record."""
    start_ms = _parse_timestamp_ms(record.timestamp)
    if record.completed_at:
        return _parse_timestamp_ms(record.completed_at)
    if record.duration_ms > 0:
        return start_ms + record.duration_ms
    return start_ms


def _new_messages(
    previous_messages: list[dict[str, Any]],
    current_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return messages appended between two consecutive parent turns."""
    common_prefix_len = 0
    for previous, current in zip(previous_messages, current_messages, strict=False):
        if previous != current:
            break
        common_prefix_len += 1
    return current_messages[common_prefix_len:]


def _iter_message_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract structured content blocks from appended messages."""
    blocks: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            blocks.extend(block for block in content if isinstance(block, dict))
    return blocks


def _stringify_block_content(content: Any) -> str:
    """Flatten nested tool-result payloads for heuristic inspection."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_stringify_block_content(item) for item in content)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if "content" in content:
            return _stringify_block_content(content["content"])
        return orjson.dumps(content).decode()
    return str(content)


def _detect_join_turn_from_content(
    parent_records: list[ConfluxRecord],
    spawn_turn_index: int,
) -> tuple[int | None, bool]:
    """Infer a later join turn from appended Agent tool_result messages.

    Returns:
        (join_turn_index, saw_background_signal)
        - join_turn_index: first parent turn whose new messages contain the
          result for the Agent tool_use emitted by ``spawn_turn_index``.
        - saw_background_signal: True when only queued/async acknowledgements
          were observed for that spawn and no later full result was found.
    """
    if spawn_turn_index + 1 >= len(parent_records):
        return None, False

    # Collect all Agent tool_use IDs already present in the cumulative messages
    # at spawn_turn_index. Any such ID appearing at spawn_turn_index+1 is a
    # completion of a *previous* blocking spawn (possibly re-surfaced due to
    # metadata drift re-rooting the common-prefix diff), not a new launch.
    spawn_msgs = _decode_messages(parent_records[spawn_turn_index])
    existing_agent_ids: set[str] = set()
    for msg in spawn_msgs:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") == "Agent"
                ):
                    bid = block.get("id")
                    if isinstance(bid, str):
                        existing_agent_ids.add(bid)

    spawn_tool_use_ids: set[str] = set()
    saw_queued_result = False

    prev_msgs = spawn_msgs
    for turn_index in range(spawn_turn_index + 1, len(parent_records)):
        current_messages = _decode_messages(parent_records[turn_index])
        appended_messages = _new_messages(prev_msgs, current_messages)
        prev_msgs = current_messages

        for block in _iter_message_blocks(appended_messages):
            if (
                turn_index == spawn_turn_index + 1
                and block.get("type") == "tool_use"
                and block.get("name") == "Agent"
            ):
                block_id = block.get("id")
                if not isinstance(block_id, str):
                    continue
                if block_id in existing_agent_ids:
                    # This tool_use was already in the history — it's a previous
                    # blocking spawn completing here, not a new launch.
                    continue
                spawn_tool_use_ids.add(block_id)
                continue

            if block.get("type") != "tool_result":
                continue

            tool_use_id = block.get("tool_use_id")
            matches_spawn = (
                isinstance(tool_use_id, str) and tool_use_id in spawn_tool_use_ids
            )
            if not matches_spawn:
                continue

            result_text = _stringify_block_content(block.get("content", ""))
            if (
                "queued for running" in result_text
                or "Async agent launched" in result_text
            ):
                saw_queued_result = True
                continue

            return turn_index, False

    return None, saw_queued_result


def _find_join_turn_index(
    parent_records: list[ConfluxRecord],
    spawn_turn_index: int,
    children: list[tuple[str, list[ConfluxRecord], Conversation]],
) -> int | None:
    """Infer which parent turn waits for/consumes a spawn's child results.

    Preference order:
    1. Explicit tool_result content matched to the Agent tool_use.
    2. First later parent turn that starts after all children have completed.
    3. None if the children outlive the parent thread or content explicitly
       marks the spawn as background-only.
    """
    join_turn_from_content, saw_background_signal = _detect_join_turn_from_content(
        parent_records, spawn_turn_index
    )
    if join_turn_from_content is not None:
        return join_turn_from_content
    if saw_background_signal:
        return None

    latest_child_end_ms = max(
        _record_end_ms(child_records[-1]) for _, child_records, _ in children
    )

    for turn_index in range(spawn_turn_index + 1, len(parent_records)):
        parent_turn_start_ms = _parse_timestamp_ms(parent_records[turn_index].timestamp)
        if parent_turn_start_ms >= latest_child_end_ms:
            return turn_index

    return None


_TASK_NOTIFICATION_RE = re.compile(
    r"<task-notification>.*?<tool-use-id>(.*?)</tool-use-id>",
    re.DOTALL,
)
_AGENT_ID_RE = re.compile(r"agentId:\s*(\S+)")


def _extract_notification_joins(
    parent_records: list[ConfluxRecord],
) -> dict[str, int]:
    """Scan parent turns for <task-notification> completion signals.

    Returns {Agent_tool_use_id: first_turn_index} for each child whose
    completion was signalled via a <task-notification> user message injected
    by Claude Code when the background agent finished.
    """
    joins: dict[str, int] = {}
    prev_msgs: list[dict[str, Any]] = []
    for ti, record in enumerate(parent_records):
        curr_msgs = _decode_messages(record)
        new_msgs = _new_messages(prev_msgs, curr_msgs)
        prev_msgs = curr_msgs
        for msg in new_msgs:
            content = msg.get("content", "")
            texts: list[str] = []
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text", "")
                        if t:
                            texts.append(t)
            for text in texts:
                if "<task-notification>" not in text:
                    continue
                for m in _TASK_NOTIFICATION_RE.finditer(text):
                    tuid = m.group(1).strip()
                    if tuid not in joins:
                        joins[tuid] = ti
    return joins


def _build_spawn_tuid_to_agent_id(
    parent_records: list[ConfluxRecord],
    spawn_turn_index: int,
) -> dict[str, str]:
    """Build tool_use_id -> agent_id for newly-spawned async agents.

    At spawn_turn_index+1, finds "Async agent launched" tool_results and
    extracts the agentId for each new Agent tool_use (not previously in the
    message history).
    """
    if spawn_turn_index + 1 >= len(parent_records):
        return {}

    existing_ids: set[str] = set()
    for msg in _decode_messages(parent_records[spawn_turn_index]):
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") == "Agent"
                ):
                    bid = block.get("id")
                    if isinstance(bid, str):
                        existing_ids.add(bid)

    prev = _decode_messages(parent_records[spawn_turn_index])
    curr = _decode_messages(parent_records[spawn_turn_index + 1])
    new_msgs = _new_messages(prev, curr)

    new_spawn_ids: set[str] = set()
    for msg in new_msgs:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") == "Agent"
                ):
                    bid = block.get("id")
                    if isinstance(bid, str) and bid not in existing_ids:
                        new_spawn_ids.add(bid)

    tuid_to_agent: dict[str, str] = {}
    for msg in new_msgs:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                tuid = block.get("tool_use_id")
                if not isinstance(tuid, str) or tuid not in new_spawn_ids:
                    continue
                result_text = _stringify_block_content(block.get("content", ""))
                if "Async agent launched" not in result_text:
                    continue
                m = _AGENT_ID_RE.search(result_text)
                if m:
                    tuid_to_agent[tuid] = m.group(1)
    return tuid_to_agent


_CAN_LOAD_PROBE_BYTES = 1 << 20  # 1 MB probe limit for format detection


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
        self._orphan_ids: set[str] = set()

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
                probe = f.read(_CAN_LOAD_PROBE_BYTES)
            try:
                raw = orjson.loads(probe)
            except orjson.JSONDecodeError:
                # Truncated read of large file - fall back to byte-level detection
                if not probe or probe[0:1] != b"[":
                    return False
                has_messages = b'"messages"' in probe
                has_agent = b'"agent_id"' in probe and b'"is_subagent"' in probe
                has_proxy = b'"source"' in probe and b'"proxy"' in probe
                return has_messages and (has_agent or has_proxy)
            if not isinstance(raw, list) or len(raw) == 0:
                return False
            # Check any record for Conflux signature fields. Two detection
            # paths: explicit agent threading (agent_id + is_subagent + messages)
            # or the proxy source marker (source == "proxy" + messages).
            return any(
                isinstance(r, dict)
                and "messages" in r
                and (
                    ("agent_id" in r and "is_subagent" in r)
                    or r.get("source") == "proxy"
                )
                for r in raw[:20]
            )
        except Exception:
            return False

    @classmethod
    def get_default_threading_mode(cls) -> TurnThreadingMode:
        return TurnThreadingMode.ISOLATED_TURNS

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[ConfluxRecord]]:
        """Load and group Conflux records by agent_id.

        Records with an agent_id are grouped into multi-turn agent threads.
        Records without an agent_id (e.g. haiku tool-result processing calls)
        become single-turn background subagent children of the parent agent,
        each mapped to the closest parent turn by timestamp.
        """
        with open(self.filename, "rb") as f:
            raw_records: list[dict] = orjson.loads(f.read())

        include_orphans = self.user_config.input.conflux_include_utility_calls

        groups: dict[str, list[ConfluxRecord]] = {}
        orphan_count = 0

        for raw in raw_records:
            record = ConfluxRecord.model_validate(raw)
            if record.agent_id is not None:
                groups.setdefault(record.agent_id, []).append(record)
            elif include_orphans:
                orphan_id = f"_orphan_{orphan_count}"
                orphan_count += 1
                self._orphan_ids.add(orphan_id)
                groups[orphan_id] = [record]
            else:
                orphan_count += 1

        # Sort each group by timestamp
        for records in groups.values():
            records.sort(key=lambda r: _parse_timestamp_ms(r.timestamp))

        self._groups = groups

        threaded = len(groups) - len(self._orphan_ids)
        total_records = sum(len(recs) for recs in groups.values())
        orphan_label = (
            f"{orphan_count} utility calls included"
            if include_orphans
            else f"{orphan_count} utility calls skipped"
        )
        self.info(
            f"Loaded {threaded} agent threads + {orphan_label} "
            f"({total_records} total records)"
        )

        return groups

    def convert_to_conversations(
        self, data: dict[str, list[ConfluxRecord]]
    ) -> list[Conversation]:
        """Convert grouped Conflux records to Conversation objects.

        Three categories:
        - The group with is_subagent=False is the parent agent thread.
        - Groups with is_subagent=True are children linked via SubagentSpawnInfo.
          Children spawned at the same parent turn are grouped into a single
          SubagentSpawnInfo. Blocking joins are attached to the first later
          parent turn that consumes the child result, inferred from Agent
          tool_result content when available and child-completion timing
          otherwise.
        - Orphan groups (no agent_id) become single-turn background subagent
          children, each spawned at the closest parent turn by timestamp.
        """
        if not data:
            return []

        # Separate orphans from threaded groups
        threaded_data: dict[str, list[ConfluxRecord]] = {}
        orphan_data: dict[str, list[ConfluxRecord]] = {}
        for agent_id, records in data.items():
            if agent_id in self._orphan_ids:
                orphan_data[agent_id] = records
            else:
                threaded_data[agent_id] = records

        parent_id: str | None = None
        child_ids: list[str] = []
        unclassified_ids: list[str] = []

        for agent_id, records in threaded_data.items():
            if records[0].is_subagent is True:
                child_ids.append(agent_id)
            elif records[0].is_subagent is False:
                parent_id = agent_id
            else:
                # is_subagent=None: un-enriched record, classify heuristically
                unclassified_ids.append(agent_id)

        # For un-enriched data: if no explicit parent, elect the group with
        # the most records as parent, rest become children.
        if parent_id is None and unclassified_ids:
            unclassified_ids.sort(key=lambda aid: len(threaded_data[aid]), reverse=True)
            parent_id = unclassified_ids.pop(0)
            child_ids.extend(unclassified_ids)
        else:
            # With an explicit parent, unclassified groups become children
            child_ids.extend(unclassified_ids)

        conversations: list[Conversation] = []
        child_conversations: list[Conversation] = []

        if parent_id is not None:
            parent_records = threaded_data[parent_id]
            parent_conv = self._build_conversation(
                parent_id, parent_records, is_child=False
            )
            parent_session_id = parent_conv.session_id

            # Group children by spawn turn index, then detect blocking vs background
            spawn_turn_to_children: dict[
                int, list[tuple[str, list[ConfluxRecord], Conversation]]
            ] = {}
            for child_agent_id in child_ids:
                child_records = threaded_data[child_agent_id]
                child_conv = self._build_conversation(
                    child_agent_id,
                    child_records,
                    is_child=True,
                    parent_session_id=parent_session_id,
                )
                spawn_turn_index = self._find_spawn_point(parent_records, child_records)
                spawn_turn_to_children.setdefault(spawn_turn_index, []).append(
                    (child_agent_id, child_records, child_conv)
                )

            spawn_counter = 0
            blocking_count = 0
            background_count = 0
            notification_joins = _extract_notification_joins(parent_records)
            for spawn_turn_index in sorted(spawn_turn_to_children):
                children_at_turn = spawn_turn_to_children[spawn_turn_index]
                join_turn_index = _find_join_turn_index(
                    parent_records, spawn_turn_index, children_at_turn
                )
                is_background = join_turn_index is None

                # For background spawns, check if each child reports back via
                # <task-notification>. If so, split into per-child blocking
                # spawns with per-child join turns derived from the notification.
                if is_background and notification_joins:
                    tuid_to_agent = _build_spawn_tuid_to_agent_id(
                        parent_records, spawn_turn_index
                    )
                    agent_to_join: dict[str, int] = {
                        agent_id: notification_joins[tuid]
                        for tuid, agent_id in tuid_to_agent.items()
                        if tuid in notification_joins
                    }
                    if agent_to_join:
                        notification_children = [
                            (aid, recs, conv, agent_to_join[aid])
                            for aid, recs, conv in children_at_turn
                            if aid in agent_to_join
                        ]
                        bg_children = [
                            (aid, recs, conv)
                            for aid, recs, conv in children_at_turn
                            if aid not in agent_to_join
                        ]
                        for (
                            _,
                            _,
                            child_conv,
                            notification_turn,
                        ) in notification_children:
                            spawn_id = f"s{spawn_counter}"
                            spawn_counter += 1
                            parent_conv.subagent_spawns.append(
                                SubagentSpawnInfo(
                                    spawn_id=spawn_id,
                                    child_conversation_ids=[child_conv.session_id],
                                    is_background=False,
                                )
                            )
                            if spawn_turn_index < len(parent_conv.turns):
                                parent_conv.turns[
                                    spawn_turn_index
                                ].subagent_spawn_ids.append(spawn_id)
                            if notification_turn < len(parent_conv.turns):
                                parent_conv.turns[
                                    notification_turn
                                ].prerequisites.append(
                                    TurnPrerequisite(
                                        kind=PrerequisiteKind.SPAWN_JOIN,
                                        spawn_id=spawn_id,
                                    )
                                )
                            child_conversations.append(child_conv)
                            blocking_count += 1
                        for _, _, child_conv in bg_children:
                            spawn_id = f"s{spawn_counter}"
                            spawn_counter += 1
                            parent_conv.subagent_spawns.append(
                                SubagentSpawnInfo(
                                    spawn_id=spawn_id,
                                    child_conversation_ids=[child_conv.session_id],
                                    is_background=True,
                                )
                            )
                            if spawn_turn_index < len(parent_conv.turns):
                                parent_conv.turns[
                                    spawn_turn_index
                                ].subagent_spawn_ids.append(spawn_id)
                            child_conversations.append(child_conv)
                            background_count += 1
                        continue

                child_conv_ids = [conv.session_id for _, _, conv in children_at_turn]
                spawn_id = f"s{spawn_counter}"
                spawn_counter += 1

                parent_conv.subagent_spawns.append(
                    SubagentSpawnInfo(
                        spawn_id=spawn_id,
                        child_conversation_ids=child_conv_ids,
                        is_background=is_background,
                    )
                )

                if spawn_turn_index < len(parent_conv.turns):
                    parent_conv.turns[spawn_turn_index].subagent_spawn_ids.append(
                        spawn_id
                    )

                if join_turn_index is not None and join_turn_index < len(
                    parent_conv.turns
                ):
                    parent_conv.turns[join_turn_index].prerequisites.append(
                        TurnPrerequisite(
                            kind=PrerequisiteKind.SPAWN_JOIN,
                            spawn_id=spawn_id,
                        )
                    )

                for _, _, child_conv in children_at_turn:
                    child_conversations.append(child_conv)

                if is_background:
                    background_count += len(children_at_turn)
                else:
                    blocking_count += len(children_at_turn)

            # Attach orphan records as single-turn background subagent children
            for orphan_id, orphan_records in orphan_data.items():
                child_conv = self._build_conversation(
                    orphan_id,
                    orphan_records,
                    is_child=True,
                    parent_session_id=parent_session_id,
                )

                spawn_turn_index = self._find_spawn_point(
                    parent_records, orphan_records
                )
                spawn_id = f"s{spawn_counter}"
                spawn_counter += 1

                parent_conv.subagent_spawns.append(
                    SubagentSpawnInfo(
                        spawn_id=spawn_id,
                        child_conversation_ids=[child_conv.session_id],
                        is_background=True,
                    )
                )

                if spawn_turn_index < len(parent_conv.turns):
                    parent_conv.turns[spawn_turn_index].subagent_spawn_ids.append(
                        spawn_id
                    )

                child_conversations.append(child_conv)
                background_count += 1

            conversations.append(parent_conv)
            conversations.extend(child_conversations)
        else:
            blocking_count = 0
            background_count = 0
            for agent_id, records in threaded_data.items():
                conversations.append(
                    self._build_conversation(agent_id, records, is_child=False)
                )

        total_turns = sum(len(c.turns) for c in conversations)
        self.info(
            f"Converted {len(conversations)} conversations "
            f"({total_turns} total turns, "
            f"{len(child_conversations)} subagent children: "
            f"{blocking_count} blocking, {background_count} background, "
            f"incl. {len(orphan_data)} orphans)"
        )
        return conversations

    def _build_conversation(
        self,
        agent_id: str,
        records: list[ConfluxRecord],
        *,
        is_child: bool,
        parent_session_id: str | None = None,
    ) -> Conversation:
        """Build a Conversation from a list of ConfluxRecords for one agent."""
        first = records[0]

        origin = ConversationOrigin(
            source=first.source,
            client=first.client,
            client_version=first.client_version,
            original_session_id=first.session_id,
            original_request_ids=[
                r.request_id for r in records if r.request_id is not None
            ],
        )

        conversation = Conversation(
            session_id=f"conflux_{agent_id}",
            agent_depth=1 if is_child else 0,
            parent_conversation_id=parent_session_id if is_child else None,
            origin=origin,
        )

        for record in records:
            ts_ms = _parse_timestamp_ms(record.timestamp)
            input_tokens = record.tokens.input if record.tokens else 0

            # Extract messages and tools from best available source
            messages, tools, max_tokens = self._extract_record_fields(record)

            # Normalize to OpenAI canonical format (N+M architecture)
            provider = self._detect_conflux_provider(record)
            raw_messages, raw_tools = normalize_messages(
                messages, tools, provider=provider
            )

            # Build extra_params from hyperparameters (excluding max_tokens and nulls)
            extra_params = self._extract_extra_params(record)

            # Build ground truth from token breakdown, timing, and output
            ground_truth = self._extract_ground_truth(record)

            turn = Turn(
                role="user",
                model=record.model,
                timestamp=ts_ms,
                max_tokens=max_tokens or 4096,
                input_tokens=input_tokens,
                raw_messages=raw_messages,
                raw_tools=raw_tools,
                extra_params=extra_params,
                ground_truth=ground_truth,
            )
            conversation.turns.append(turn)

        return conversation

    @staticmethod
    def _extract_extra_params(record: ConfluxRecord) -> dict[str, Any] | None:
        """Extract per-turn hyperparameter overrides from a ConfluxRecord.

        Excludes max_tokens (already on Turn.max_tokens) and null values.
        Returns None if no non-null hyperparameters remain.
        """
        if not record.hyperparameters:
            return None
        hp = record.hyperparameters
        params: dict[str, Any] = {}
        for field_name in (
            "temperature",
            "top_p",
            "top_k",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "stop",
            "reasoning_effort",
            "reasoning_summary",
            "text_verbosity",
        ):
            value = getattr(hp, field_name, None)
            if value is not None:
                params[field_name] = value
        return params or None

    @staticmethod
    def _extract_ground_truth(record: ConfluxRecord) -> TurnGroundTruth | None:
        """Extract ground truth metadata from a ConfluxRecord.

        Returns None if no meaningful ground truth data is available.
        """
        tokens = record.tokens
        has_token_detail = tokens is not None and (
            tokens.input_cached > 0
            or tokens.input_cache_write > 0
            or tokens.output > 0
            or tokens.output_reasoning > 0
        )
        has_timing = record.ttft_ms is not None or record.duration_ms > 0
        has_streaming = record.is_streaming is not None
        if not (has_token_detail or has_timing or has_streaming):
            return None

        return TurnGroundTruth(
            input_cached_tokens=tokens.input_cached if tokens else None,
            input_cache_write_tokens=tokens.input_cache_write if tokens else None,
            output_tokens=tokens.output if tokens else None,
            output_reasoning_tokens=tokens.output_reasoning if tokens else None,
            ttft_ms=record.ttft_ms,
            duration_ms=record.duration_ms if record.duration_ms > 0 else None,
            is_streaming=record.is_streaming,
        )

    @staticmethod
    def _extract_record_fields(
        record: ConfluxRecord,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, int | None]:
        """Extract messages, tools, and max_tokens from the best available source.

        Base64 path: decode the verbatim request body for full fidelity
        (includes system, tools, thinking config, etc.).
        Fallback: use top-level Conflux fields.

        Returns:
            (messages, tools, max_tokens)
        """
        if record.base64 and record.base64.get("request_body"):
            payload: dict[str, Any] = orjson.loads(
                base64.b64decode(record.base64["request_body"])  # type: ignore[index]
            )
            payload.pop("metadata", None)
            messages = _messages_from_payload(payload)
            tools = payload.get("tools") or None
            max_tokens = payload.get("max_tokens")
            return messages, tools, max_tokens

        # Fallback: use top-level Conflux fields
        messages = list(record.messages)
        tools = record.tools if record.tools else None
        max_tokens = (
            record.hyperparameters.max_tokens if record.hyperparameters else None
        )
        return messages, tools, max_tokens

    @staticmethod
    def _detect_conflux_provider(record: ConfluxRecord) -> str | None:
        """Detect the provider from Conflux record metadata.

        Uses the ``client`` field (set by Conflux enrichment) or the
        ``provider`` field (set by the adapter pipeline) to map to a
        provider hint for the normalizer.
        """
        if record.provider:
            provider = record.provider.lower()
            if provider in ("anthropic", "openai"):
                return provider
        if record.client == "claude":
            return "anthropic"
        if record.client == "codex":
            return "openai"
        return None

    @staticmethod
    def _find_spawn_point(
        parent_records: list[ConfluxRecord],
        child_records: list[ConfluxRecord],
    ) -> int:
        """Find the parent turn that spawned a child agent.

        Three-tier matching:

        1. **In-flight overlap**: Child's first request falls within a parent
           turn's execution window (start -> completion). This handles children
           that start while the parent is still processing.

        2. **Post-completion gap**: Child starts in the gap between parent turn
           N completing and turn N+1 starting. This handles the common case
           where children are spawned immediately after the parent processes the
           response (typically within a few hundred milliseconds).

        3. **Closest timestamp** (fallback): Assigns to the nearest parent turn
           by start timestamp when neither overlap nor gap matching succeeds.
        """
        child_first_ts = _parse_timestamp_ms(child_records[0].timestamp)

        # Tier 1: In-flight overlap
        for i, record in enumerate(parent_records):
            start_ts = _parse_timestamp_ms(record.timestamp)
            if record.completed_at:
                end_ts = _parse_timestamp_ms(record.completed_at)
            elif record.duration_ms > 0:
                end_ts = start_ts + record.duration_ms
            else:
                continue
            if start_ts <= child_first_ts <= end_ts:
                return i

        # Tier 2: Post-completion gap (child starts after turn N ends but
        # before turn N+1 starts -- spawned by turn N's response processing)
        for i, record in enumerate(parent_records):
            if record.completed_at:
                end_ts = _parse_timestamp_ms(record.completed_at)
            elif record.duration_ms > 0:
                end_ts = _parse_timestamp_ms(record.timestamp) + record.duration_ms
            else:
                continue
            next_start_ts = (
                _parse_timestamp_ms(parent_records[i + 1].timestamp)
                if i + 1 < len(parent_records)
                else float("inf")
            )
            if end_ts <= child_first_ts <= next_start_ts:
                return i

        # Tier 3: Closest timestamp fallback
        best_idx = 0
        best_diff = float("inf")
        for i, record in enumerate(parent_records):
            ts = _parse_timestamp_ms(record.timestamp)
            diff = abs(ts - child_first_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        return best_idx
