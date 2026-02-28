# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HashIdRandomGenerator parallel processing with reproducibility."""

import pytest

from aiperf.common import random_generator as rng
from aiperf.common.hash_id_random_generator import HashIdRandomGenerator


class TestHashIdRandomGenerator:
    """Test HashIdRandomGenerator for parallel processing reproducibility."""

    @pytest.fixture(autouse=True)
    def setup_rng(self):
        """Initialize global RNG before each test."""
        rng.reset()
        rng.init(42)
        yield
        rng.reset()

    def test_deterministic_seeding_per_hash_id(self):
        """Test that the same hash_id always produces the same random sequence."""
        base_rng = rng.derive("test.base")
        hash_rng = HashIdRandomGenerator.from_base_rng(base_rng)

        # Generate values for hash_id=123
        hash_rng.reseed_for_hash_id(123)
        values_1 = [hash_rng.randrange(1000) for _ in range(10)]

        # Generate values for hash_id=456
        hash_rng.reseed_for_hash_id(456)
        values_2 = [hash_rng.randrange(1000) for _ in range(10)]

        # Generate values for hash_id=123 again (should match values_1)
        hash_rng.reseed_for_hash_id(123)
        values_3 = [hash_rng.randrange(1000) for _ in range(10)]

        # Same hash_id produces same sequence
        assert values_1 == values_3
        # Different hash_ids produce different sequences
        assert values_1 != values_2

    def test_independence_across_workers(self):
        """Test that different worker instances produce identical results for same hash_id."""
        # Simulate Worker 1
        base_rng_1 = rng.derive("worker.corpus")
        hash_rng_1 = HashIdRandomGenerator.from_base_rng(base_rng_1)

        # Reset and re-init to simulate Worker 2 with same global seed
        rng.reset()
        rng.init(42)

        # Simulate Worker 2
        base_rng_2 = rng.derive("worker.corpus")
        hash_rng_2 = HashIdRandomGenerator.from_base_rng(base_rng_2)

        # Both workers process hash_id=789
        hash_rng_1.reseed_for_hash_id(789)
        worker1_values = [hash_rng_1.randrange(1000) for _ in range(10)]

        hash_rng_2.reseed_for_hash_id(789)
        worker2_values = [hash_rng_2.randrange(1000) for _ in range(10)]

        # Workers produce identical results
        assert worker1_values == worker2_values

    def test_order_independence(self):
        """Test that processing order doesn't affect reproducibility."""
        base_rng = rng.derive("test.order")

        # Process in order: 100, 200, 300
        hash_rng_1 = HashIdRandomGenerator.from_base_rng(base_rng)
        results_order1 = {}
        for hash_id in [100, 200, 300]:
            hash_rng_1.reseed_for_hash_id(hash_id)
            results_order1[hash_id] = [hash_rng_1.randrange(1000) for _ in range(5)]

        # Process in different order: 300, 100, 200
        hash_rng_2 = HashIdRandomGenerator.from_base_rng(base_rng)
        results_order2 = {}
        for hash_id in [300, 100, 200]:
            hash_rng_2.reseed_for_hash_id(hash_id)
            results_order2[hash_id] = [hash_rng_2.randrange(1000) for _ in range(5)]

        # Same hash_id produces same results regardless of order
        assert results_order1[100] == results_order2[100]
        assert results_order1[200] == results_order2[200]
        assert results_order1[300] == results_order2[300]

    def test_parallel_cache_simulation(self):
        """Simulate parallel workers with individual caches."""
        # Worker 1 processes hash_ids: 1, 2, 3
        base_rng_w1 = rng.derive("worker.corpus")
        hash_rng_w1 = HashIdRandomGenerator.from_base_rng(base_rng_w1)
        cache_w1 = {}

        for hash_id in [1, 2, 3]:
            hash_rng_w1.reseed_for_hash_id(hash_id)
            cache_w1[hash_id] = [hash_rng_w1.randrange(1000) for _ in range(5)]

        # Reset for Worker 2 simulation
        rng.reset()
        rng.init(42)

        # Worker 2 processes hash_ids: 3, 4, 5 (overlapping hash_id=3)
        base_rng_w2 = rng.derive("worker.corpus")
        hash_rng_w2 = HashIdRandomGenerator.from_base_rng(base_rng_w2)
        cache_w2 = {}

        for hash_id in [3, 4, 5]:
            hash_rng_w2.reseed_for_hash_id(hash_id)
            cache_w2[hash_id] = [hash_rng_w2.randrange(1000) for _ in range(5)]

        # Both workers produce identical results for hash_id=3
        assert cache_w1[3] == cache_w2[3]

        # Different hash_ids produce different results
        assert cache_w1[1] != cache_w1[2]
        assert cache_w2[4] != cache_w2[5]

    def test_non_deterministic_mode(self):
        """Test that non-deterministic mode works (seed=None)."""
        rng.reset()
        rng.init(None)

        base_rng = rng.derive("test.nondeterministic")
        hash_rng = HashIdRandomGenerator.from_base_rng(base_rng)

        # Should not raise errors
        hash_rng.reseed_for_hash_id(123)
        values = [hash_rng.randrange(1000) for _ in range(10)]

        assert len(values) == 10
        # Ensure that an actual seed is created for the HashIdRandomGenerator
        assert hash_rng.seed is not None

    def test_multiple_random_operations(self):
        """Test various random operations after reseeding."""
        base_rng = rng.derive("test.operations")
        hash_rng = HashIdRandomGenerator.from_base_rng(base_rng)

        # Test for hash_id=555
        hash_rng.reseed_for_hash_id(555)
        int_val = hash_rng.randrange(100, 200)
        float_val = hash_rng.uniform(0.0, 1.0)
        choice_val = hash_rng.choice([10, 20, 30, 40])

        # Re-seed with same hash_id and verify reproducibility
        hash_rng.reseed_for_hash_id(555)
        int_val_2 = hash_rng.randrange(100, 200)
        float_val_2 = hash_rng.uniform(0.0, 1.0)
        choice_val_2 = hash_rng.choice([10, 20, 30, 40])

        assert int_val == int_val_2
        assert float_val == float_val_2
        assert choice_val == choice_val_2

    def test_hash_collision_independence(self):
        """Test that different hash_ids produce independent sequences."""
        base_rng = rng.derive("test.collision")
        hash_rng = HashIdRandomGenerator.from_base_rng(base_rng)

        hash_ids = [1, 12345, 999999, 7777777, 123456789]
        sequences = {}

        for hash_id in hash_ids:
            hash_rng.reseed_for_hash_id(hash_id)
            sequences[hash_id] = [hash_rng.randrange(1000) for _ in range(20)]

        # All sequences should be different
        unique_sequences = set(tuple(seq) for seq in sequences.values())
        assert len(unique_sequences) == len(hash_ids)

    def test_reseed_state_isolation(self):
        """Test that reseeding properly isolates state between hash_ids."""
        base_rng = rng.derive("test.isolation")
        hash_rng = HashIdRandomGenerator.from_base_rng(base_rng)

        # Generate partial sequence for hash_id=111
        hash_rng.reseed_for_hash_id(111)
        partial_seq_111 = [hash_rng.randrange(1000) for _ in range(3)]

        # Switch to hash_id=222 and generate values
        hash_rng.reseed_for_hash_id(222)
        _ = [hash_rng.randrange(1000) for _ in range(5)]

        # Return to hash_id=111 and continue - should continue from fresh state
        hash_rng.reseed_for_hash_id(111)
        full_seq_111 = [hash_rng.randrange(1000) for _ in range(10)]

        # The first 3 values should match the partial sequence
        assert full_seq_111[:3] == partial_seq_111

    def test_trace_id_isolation(self):
        """Test that different trace_ids produce different sequences for same hash_id."""
        base_rng = rng.derive("test.trace_id")
        hash_rng = HashIdRandomGenerator.from_base_rng(base_rng)

        hash_rng.set_trace_id("trace_file_a")
        hash_rng.reseed_for_hash_id(100)
        values_a = [hash_rng.randrange(1000) for _ in range(10)]

        hash_rng.set_trace_id("trace_file_b")
        hash_rng.reseed_for_hash_id(100)
        values_b = [hash_rng.randrange(1000) for _ in range(10)]

        assert values_a != values_b


class TestHashIdRandomGeneratorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture(autouse=True)
    def setup_rng(self):
        """Initialize global RNG before each test."""
        rng.reset()
        rng.init(42)
        yield
        rng.reset()

    def test_zero_hash_id(self):
        """Test with hash_id=0."""
        base_rng = rng.derive("test.zero")
        hash_rng = HashIdRandomGenerator.from_base_rng(base_rng)

        hash_rng.reseed_for_hash_id(0)
        values = [hash_rng.randrange(1000) for _ in range(5)]

        assert len(values) == 5

    def test_negative_hash_id(self):
        """Test with negative hash_id."""
        base_rng = rng.derive("test.negative")
        hash_rng = HashIdRandomGenerator.from_base_rng(base_rng)

        hash_rng.reseed_for_hash_id(-123)
        values = [hash_rng.randrange(1000) for _ in range(5)]

        # Should not produce the same as positive 123
        hash_rng.reseed_for_hash_id(123)
        values_positive = [hash_rng.randrange(1000) for _ in range(5)]

        assert values != values_positive

    def test_large_hash_id(self):
        """Test with very large hash_id."""
        base_rng = rng.derive("test.large")
        hash_rng = HashIdRandomGenerator.from_base_rng(base_rng)

        large_hash_id = 999999999999999999
        hash_rng.reseed_for_hash_id(large_hash_id)
        values = [hash_rng.randrange(1000) for _ in range(5)]

        assert len(values) == 5

    @pytest.mark.parametrize(
        "operation",
        [
            lambda rng: rng.integers(0, 100, size=2),
            lambda rng: rng.random_batch(2),
            lambda rng: rng.shuffle([1, 2, 3, 4, 5]),
            lambda rng: rng.numpy_choice([1, 2, 3, 4, 5], size=2),
            lambda rng: rng.normal(0, 1, size=2),
        ],
    )
    def test_numpy_rng_raises_exception(self, operation):
        """Test that using NumPy RNG operations raises an exception."""
        base_rng = rng.derive("test.numpy")
        hash_rng = HashIdRandomGenerator.from_base_rng(base_rng)
        hash_rng.reseed_for_hash_id(123)

        with pytest.raises(
            RuntimeError, match="HashIdRandomGenerator does not support NumPy RNG"
        ):
            operation(hash_rng)
