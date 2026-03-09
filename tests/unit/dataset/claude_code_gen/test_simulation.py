# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for simulation dashboard logic — load_sessions hash_ids passthrough
and a Python reimplementation of the unique block refcounting logic."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import orjson

from aiperf.dataset.claude_code_gen.simulation import load_sessions


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    jsonl = path / "data.jsonl"
    with jsonl.open("wb") as f:
        for row in rows:
            f.write(orjson.dumps(row) + b"\n")
    return jsonl


class TestLoadSessionsHashIds:
    """Verify load_sessions passes hash_ids through to turn dicts."""

    def test_hash_ids_preserved(self, tmp_path: Path) -> None:
        rows = [
            {
                "session_id": "s1",
                "input_length": 100,
                "output_length": 20,
                "delay": 0,
                "group_id": 0,
                "hash_ids": [10, 11, 12],
            },
            {
                "session_id": "s1",
                "input_length": 50,
                "output_length": 10,
                "delay": 5000,
                "hash_ids": [10, 11, 12, 13, 14],
            },
        ]
        jsonl = _write_jsonl(tmp_path, rows)
        sessions = load_sessions(jsonl)

        assert len(sessions) == 1
        assert sessions[0]["turns"][0]["hash_ids"] == [10, 11, 12]
        assert sessions[0]["turns"][1]["hash_ids"] == [10, 11, 12, 13, 14]

    def test_missing_hash_ids_defaults_empty(self, tmp_path: Path) -> None:
        rows = [
            {
                "session_id": "s1",
                "input_length": 100,
                "output_length": 20,
                "delay": 0,
            },
        ]
        jsonl = _write_jsonl(tmp_path, rows)
        sessions = load_sessions(jsonl)

        assert sessions[0]["turns"][0]["hash_ids"] == []


def _simulate_unique_blocks(sessions: list[dict], concurrency: int) -> list[int]:
    """Python reimplementation of the JS unique block tracking logic.

    Returns unique_blocks count at each event for verification.
    Uses the same refcounted-map approach as the JS simulate().
    """
    block_refcount: dict[int, int] = defaultdict(int)
    session_blocks: dict[int, set[int]] = defaultdict(set)

    # Simple event-driven simulation mirroring JS logic
    # Events: (time, type, session_idx, turn_idx)
    import heapq

    events: list[tuple[float, int, int, int]] = []
    counter = 0

    active_count = 0
    next_session = 0
    unique_blocks_snapshots: list[int] = []

    def start_session(s_idx: int, time: float) -> None:
        start_turn(s_idx, 0, time)

    def start_turn(s_idx: int, t_idx: int, time: float) -> None:
        nonlocal counter
        turn = sessions[s_idx]["turns"][t_idx]
        delay = 0.0 if t_idx == 0 else turn["delay_ms"]
        prefill_start = time + delay
        # Simplified: prefill=1ms, decode=1ms per turn
        decode_start = prefill_start + 1.0
        decode_end = decode_start + 1.0
        heapq.heappush(events, (decode_start, 0, s_idx, t_idx))  # request_start
        counter += 1
        heapq.heappush(events, (decode_end, 1, s_idx, t_idx))  # request_end
        counter += 1

    # Launch initial batch
    while next_session < len(sessions) and active_count < concurrency:
        active_count += 1
        start_session(next_session, 0.0)
        next_session += 1

    while events:
        time, etype, s_idx, t_idx = heapq.heappop(events)

        if etype == 0:  # request_start
            turn = sessions[s_idx]["turns"][t_idx]
            hids = turn.get("hash_ids", [])
            s_blocks = session_blocks[s_idx]
            for bid in hids:
                if bid not in s_blocks:
                    s_blocks.add(bid)
                    block_refcount[bid] += 1

        elif etype == 1:  # request_end
            if t_idx + 1 < len(sessions[s_idx]["turns"]):
                start_turn(s_idx, t_idx + 1, time)
            else:
                # Session ends
                for bid in session_blocks[s_idx]:
                    block_refcount[bid] -= 1
                    if block_refcount[bid] <= 0:
                        del block_refcount[bid]
                session_blocks[s_idx].clear()
                active_count -= 1
                if next_session < len(sessions):
                    active_count += 1
                    start_session(next_session, time)
                    next_session += 1

        # Count unique blocks (equivalent to blockRefCount.size in JS)
        unique_blocks_snapshots.append(len(block_refcount))

    return unique_blocks_snapshots


class TestUniqueBlockTracking:
    """Verify the refcounted unique block logic using a Python reimplementation."""

    def test_shared_blocks_counted_once(self) -> None:
        """Two concurrent sessions sharing blocks 1,2,3. Shared blocks should
        only be counted once in unique_blocks."""
        sessions = [
            {
                "session_id": "s0",
                "group_id": 0,
                "turns": [
                    {
                        "input_length": 100,
                        "output_length": 10,
                        "delay_ms": 0,
                        "cumulative_input_length": 100,
                        "hash_ids": [1, 2, 3, 10, 11],
                    },
                ],
            },
            {
                "session_id": "s1",
                "group_id": 0,
                "turns": [
                    {
                        "input_length": 100,
                        "output_length": 10,
                        "delay_ms": 0,
                        "cumulative_input_length": 100,
                        "hash_ids": [1, 2, 3, 20, 21],
                    },
                ],
            },
        ]
        snapshots = _simulate_unique_blocks(sessions, concurrency=2)

        # Both sessions start concurrently. After both request_start events,
        # unique blocks = {1,2,3,10,11,20,21} = 7 (shared blocks counted once)
        # After first request_end, one session's unique blocks are removed
        # (but shared blocks stay if other session is still alive)

        # At peak (both sessions active), should be 7 unique blocks
        assert max(snapshots) == 7

    def test_blocks_freed_on_session_end(self) -> None:
        """After a session ends, its non-shared blocks are freed."""
        sessions = [
            {
                "session_id": "s0",
                "group_id": 0,
                "turns": [
                    {
                        "input_length": 100,
                        "output_length": 10,
                        "delay_ms": 0,
                        "cumulative_input_length": 100,
                        "hash_ids": [1, 2, 3],
                    },
                ],
            },
        ]
        snapshots = _simulate_unique_blocks(sessions, concurrency=1)

        # request_start: 3 blocks, request_end: session ends, 0 blocks
        assert snapshots[0] == 3
        assert snapshots[-1] == 0

    def test_sequential_sessions_no_leak(self) -> None:
        """Sessions running sequentially should fully free blocks between them."""
        sessions = [
            {
                "session_id": "s0",
                "group_id": 0,
                "turns": [
                    {
                        "input_length": 100,
                        "output_length": 10,
                        "delay_ms": 0,
                        "cumulative_input_length": 100,
                        "hash_ids": [1, 2, 3],
                    },
                ],
            },
            {
                "session_id": "s1",
                "group_id": 0,
                "turns": [
                    {
                        "input_length": 100,
                        "output_length": 10,
                        "delay_ms": 0,
                        "cumulative_input_length": 100,
                        "hash_ids": [4, 5],
                    },
                ],
            },
        ]
        snapshots = _simulate_unique_blocks(sessions, concurrency=1)

        # s0 starts: 3 blocks, s0 ends + s1 starts: blocks freed then new ones added
        # Final: s1 ends, 0 blocks
        assert snapshots[-1] == 0
        # Peak should be 3 (s0's blocks) since sequential, never both alive
        assert max(snapshots) == 3

    def test_multi_turn_accumulates_blocks(self) -> None:
        """A session with multiple turns accumulates blocks across turns.
        Duplicate block IDs across turns are not double-counted."""
        sessions = [
            {
                "session_id": "s0",
                "group_id": 0,
                "turns": [
                    {
                        "input_length": 100,
                        "output_length": 10,
                        "delay_ms": 0,
                        "cumulative_input_length": 100,
                        "hash_ids": [1, 2, 3],
                    },
                    {
                        "input_length": 50,
                        "output_length": 10,
                        "delay_ms": 1,
                        "cumulative_input_length": 160,
                        "hash_ids": [1, 2, 3, 4, 5],
                    },
                ],
            },
        ]
        snapshots = _simulate_unique_blocks(sessions, concurrency=1)

        # Turn 0 request_start: {1,2,3} = 3 blocks
        assert snapshots[0] == 3
        # Turn 1 request_start: {1,2,3,4,5} = 5 blocks (1,2,3 already in set, not re-added)
        # Find the snapshot after turn 1 starts (4th event: t0_start, t0_end, t1_start, t1_end)
        assert snapshots[2] == 5
        # After session ends: 0
        assert snapshots[-1] == 0

    def test_shared_blocks_survive_one_session_ending(self) -> None:
        """When two sessions share blocks and one ends, shared blocks remain
        because the other session still references them."""
        sessions = [
            {
                "session_id": "s0",
                "group_id": 0,
                "turns": [
                    {
                        "input_length": 100,
                        "output_length": 10,
                        "delay_ms": 0,
                        "cumulative_input_length": 100,
                        "hash_ids": [1, 2, 3],
                    },
                ],
            },
            {
                "session_id": "s1",
                "group_id": 0,
                "turns": [
                    {
                        "input_length": 100,
                        "output_length": 10,
                        "delay_ms": 0,
                        "cumulative_input_length": 100,
                        "hash_ids": [1, 2, 3, 4, 5],
                    },
                    {
                        "input_length": 50,
                        "output_length": 10,
                        "delay_ms": 1,
                        "cumulative_input_length": 160,
                        "hash_ids": [1, 2, 3, 4, 5, 6],
                    },
                ],
            },
        ]
        snapshots = _simulate_unique_blocks(sessions, concurrency=2)

        # Both start at t=0. After both request_starts:
        # s0 blocks: {1,2,3}, s1 blocks: {1,2,3,4,5}
        # unique = {1,2,3,4,5} = 5
        # s0 ends first (1 turn), s1 still alive.
        # s0's blocks {1,2,3} have refcount decremented.
        # Blocks 1,2,3 still have refcount=1 from s1. Block 4,5 refcount=1.
        # So unique blocks after s0 ends = 5 (all still referenced by s1)

        # After s0 ends (request_end for s0), before s1 turn 1:
        # s1 still has {1,2,3,4,5} -> 5 unique blocks
        # s1 turn 1 adds block 6 -> 6 unique blocks
        # s1 ends -> 0

        assert max(snapshots) == 6
        assert snapshots[-1] == 0
