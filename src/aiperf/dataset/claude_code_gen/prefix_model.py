# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV cache prefix model for hash ID generation.

Layer 1: Tools + system prompt. Identical across ALL sessions (globally cached).
Session prefix: Everything beyond L1 at turn 0 (CLAUDE.md, skills, first user msg).
              Stable within a session, size = initial_context - L1 tokens.
L3: Conversation history added from turn 1 onward. Grows each turn, unique per session.

On session reset (cache bust): new session gets fresh session prefix + L3, L1 preserved.
"""

from __future__ import annotations

import math

from aiperf.dataset.claude_code_gen.models import CacheLayerConfig

# Large gap between session ranges to avoid collisions
MAX_SESSION_BLOCKS = 4_000


class PrefixAllocator:
    """Allocates deterministic hash IDs for the prefix model.

    Hash ID layout:
        L1: [0 .. L1_blocks-1]                              (shared globally)
        Session prefix + L3: [base .. base+N]                (per session, growing)

    where base = L1_blocks + session_index * MAX_SESSION_BLOCKS

    Turn 0: all blocks beyond L1 are session prefix (no L3).
    Turn 1+: carried session blocks + new L3 blocks.

    Invariant: len(hash_ids) == ceil(input_length / block_size)
    """

    def __init__(self, config: CacheLayerConfig) -> None:
        self._block_size = config.block_size
        self._l1_blocks = math.ceil(config.layer1_tokens / config.block_size)
        self._l1_ids = list(range(self._l1_blocks))

    @property
    def l1_blocks(self) -> int:
        return self._l1_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    def session_base(self, session_index: int) -> int:
        """Compute the base hash ID offset for a session."""
        return self._l1_blocks + session_index * MAX_SESSION_BLOCKS

    def turn_hash_ids(
        self,
        session_index: int,
        input_length: int,
        prev_session_ids: list[int] | None = None,
    ) -> list[int]:
        """Generate the full hash_ids array for a turn.

        Returns L1 ++ session_ids such that len(result) == ceil(input_length / block_size).
        On turn 0 (prev_session_ids=None): everything beyond L1 is session prefix.
        On turn 1+ (prev_session_ids provided): carried blocks + new L3 growth.
        """
        total_blocks = (
            math.ceil(input_length / self._block_size) if input_length > 0 else 0
        )
        base = self.session_base(session_index)

        l1_used = min(self._l1_blocks, total_blocks)
        session_needed = max(0, total_blocks - self._l1_blocks)

        l1 = self._l1_ids[:l1_used]

        if prev_session_ids is None:
            session = list(range(base, base + session_needed))
        else:
            new_blocks = session_needed - len(prev_session_ids)
            if new_blocks > 0:
                next_id = prev_session_ids[-1] + 1 if prev_session_ids else base
                session = prev_session_ids + list(range(next_id, next_id + new_blocks))
            else:
                session = prev_session_ids[:session_needed]

        return l1 + session

    def extract_session_ids(self, hash_ids: list[int]) -> list[int]:
        """Extract everything after L1 from a full hash_ids array."""
        if len(hash_ids) <= self._l1_blocks:
            return []
        return hash_ids[self._l1_blocks :]
