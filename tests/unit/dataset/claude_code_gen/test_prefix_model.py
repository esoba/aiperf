# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the prefix model."""

from __future__ import annotations

import math

import pytest

from aiperf.dataset.claude_code_gen.models import CacheLayerConfig
from aiperf.dataset.claude_code_gen.prefix_model import (
    MAX_SESSION_BLOCKS,
    PrefixAllocator,
)


@pytest.fixture
def allocator() -> PrefixAllocator:
    return PrefixAllocator(CacheLayerConfig(layer1_tokens=32_000, block_size=512))


@pytest.fixture
def small_allocator() -> PrefixAllocator:
    return PrefixAllocator(CacheLayerConfig(layer1_tokens=200, block_size=64))


class TestPrefixAllocator:
    def test_l1_block_count(self, allocator: PrefixAllocator) -> None:
        assert allocator.l1_blocks == math.ceil(32_000 / 512)  # 63

    def test_l1_ids_identical_across_sessions(self, allocator: PrefixAllocator) -> None:
        ids_0 = allocator.turn_hash_ids(session_index=0, input_length=50_000)
        ids_1 = allocator.turn_hash_ids(session_index=1, input_length=50_000)
        l1_0 = ids_0[: allocator.l1_blocks]
        l1_1 = ids_1[: allocator.l1_blocks]
        assert l1_0 == l1_1

    def test_session_ids_grow_with_turns(self, allocator: PrefixAllocator) -> None:
        ids_t0 = allocator.turn_hash_ids(session_index=0, input_length=50_000)
        sess_t0 = allocator.extract_session_ids(ids_t0)
        ids_t1 = allocator.turn_hash_ids(
            session_index=0, input_length=60_000, prev_session_ids=sess_t0
        )
        sess_t1 = allocator.extract_session_ids(ids_t1)
        assert len(sess_t1) > len(sess_t0)
        assert sess_t1[: len(sess_t0)] == sess_t0

    def test_turn_n_is_prefix_of_turn_n_plus_1(
        self, allocator: PrefixAllocator
    ) -> None:
        ids_t0 = allocator.turn_hash_ids(session_index=0, input_length=50_000)
        sess_t0 = allocator.extract_session_ids(ids_t0)
        ids_t1 = allocator.turn_hash_ids(
            session_index=0, input_length=70_000, prev_session_ids=sess_t0
        )
        assert ids_t1[: len(ids_t0)] == ids_t0

    def test_zero_input_length_no_blocks(self, allocator: PrefixAllocator) -> None:
        ids = allocator.turn_hash_ids(session_index=0, input_length=0)
        assert ids == []

    def test_hash_ids_count_equals_ceil_isl_over_block_size(
        self, allocator: PrefixAllocator
    ) -> None:
        for isl in [10_000, 37_000, 50_000, 100_000, 200_000]:
            ids = allocator.turn_hash_ids(session_index=0, input_length=isl)
            expected = math.ceil(isl / 512)
            assert len(ids) == expected, (
                f"ISL={isl}: got {len(ids)}, expected {expected}"
            )

    def test_small_input_only_uses_l1(self, allocator: PrefixAllocator) -> None:
        """Input smaller than L1 tokens should only produce L1 blocks."""
        ids = allocator.turn_hash_ids(session_index=0, input_length=10_000)
        expected = math.ceil(10_000 / 512)  # 20
        assert len(ids) == expected
        assert all(i < allocator.l1_blocks for i in ids)

    def test_session_base_spacing(self, allocator: PrefixAllocator) -> None:
        base_0 = allocator.session_base(0)
        base_1 = allocator.session_base(1)
        assert base_1 - base_0 == MAX_SESSION_BLOCKS

    def test_no_hash_id_collisions_across_sessions(
        self, small_allocator: PrefixAllocator
    ) -> None:
        alloc = small_allocator
        ids_s0 = set(alloc.turn_hash_ids(session_index=0, input_length=500))
        ids_s1 = set(alloc.turn_hash_ids(session_index=1, input_length=500))
        l1 = set(range(alloc.l1_blocks))
        overlap = ids_s0 & ids_s1
        assert overlap == l1

    def test_turn0_no_l3(self, allocator: PrefixAllocator) -> None:
        """Turn 0 should have only L1 + session prefix blocks (no L3)."""
        ids = allocator.turn_hash_ids(session_index=0, input_length=50_000)
        l1_ids = ids[: allocator.l1_blocks]
        session_ids = allocator.extract_session_ids(ids)

        assert l1_ids == list(range(allocator.l1_blocks))
        assert len(l1_ids) + len(session_ids) == len(ids)

        base = allocator.session_base(0)
        expected_session = list(range(base, base + len(session_ids)))
        assert session_ids == expected_session

    def test_growing_input_maintains_prefix_property(
        self, allocator: PrefixAllocator
    ) -> None:
        """Simulate 5 turns of growing input, verify prefix property throughout."""
        prev_ids = None
        prev_session = None
        isl = 50_000
        for _ in range(5):
            ids = allocator.turn_hash_ids(
                session_index=0, input_length=isl, prev_session_ids=prev_session
            )
            assert len(ids) == math.ceil(isl / allocator.block_size)
            if prev_ids is not None:
                assert ids[: len(prev_ids)] == prev_ids
            prev_session = allocator.extract_session_ids(ids)
            prev_ids = ids
            isl += 10_000
