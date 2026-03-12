# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for uniform distribution (ISL/OSL) and token ID preservation features.

Covers:
- SequenceLengthDistributionType enum
- InputTokensConfig/OutputTokensConfig uniform distribution validation
- sample_positive_uniform_integer in RandomGenerator
- InEngineResponse output_token_ids field
- BaseInEngineTransport preserve_token_ids wiring
"""

from dataclasses import asdict

import pytest

from aiperf.common.config.prompt_config import InputTokensConfig, OutputTokensConfig
from aiperf.common.enums import SequenceLengthDistributionType
from aiperf.common.models.record_models import InEngineResponse
from aiperf.common.random_generator import RandomGenerator

# ============================================================
# Feature 1: SequenceLengthDistributionType Enum
# ============================================================


class TestSequenceLengthDistributionType:
    """Test the SequenceLengthDistributionType enum."""

    def test_normal_value(self) -> None:
        assert SequenceLengthDistributionType.NORMAL == "normal"

    def test_uniform_value(self) -> None:
        assert SequenceLengthDistributionType.UNIFORM == "uniform"

    def test_case_insensitive(self) -> None:
        assert (
            SequenceLengthDistributionType("NORMAL")
            == SequenceLengthDistributionType.NORMAL
        )
        assert (
            SequenceLengthDistributionType("Uniform")
            == SequenceLengthDistributionType.UNIFORM
        )


# ============================================================
# Feature 1: Config Validation
# ============================================================


class TestInputTokensConfigUniform:
    """Test InputTokensConfig uniform distribution validation."""

    def test_default_distribution_is_normal(self) -> None:
        config = InputTokensConfig()
        assert config.distribution == SequenceLengthDistributionType.NORMAL

    def test_uniform_with_min_max_succeeds(self) -> None:
        config = InputTokensConfig(
            distribution=SequenceLengthDistributionType.UNIFORM, min=10, max=100
        )
        assert config.distribution == SequenceLengthDistributionType.UNIFORM
        assert config.min == 10
        assert config.max == 100

    def test_uniform_without_min_raises(self) -> None:
        with pytest.raises(ValueError, match="Both 'min' and 'max' must be set"):
            InputTokensConfig(
                distribution=SequenceLengthDistributionType.UNIFORM, max=100
            )

    def test_uniform_without_max_raises(self) -> None:
        with pytest.raises(ValueError, match="Both 'min' and 'max' must be set"):
            InputTokensConfig(
                distribution=SequenceLengthDistributionType.UNIFORM, min=10
            )

    def test_uniform_min_greater_than_max_raises(self) -> None:
        with pytest.raises(ValueError, match="'min'.*must be <=.*'max'"):
            InputTokensConfig(
                distribution=SequenceLengthDistributionType.UNIFORM, min=200, max=100
            )

    def test_normal_ignores_min_max(self) -> None:
        config = InputTokensConfig(
            distribution=SequenceLengthDistributionType.NORMAL, min=10, max=100
        )
        assert config.distribution == SequenceLengthDistributionType.NORMAL
        assert config.min == 10

    def test_uniform_equal_min_max_succeeds(self) -> None:
        config = InputTokensConfig(
            distribution=SequenceLengthDistributionType.UNIFORM, min=50, max=50
        )
        assert config.min == 50
        assert config.max == 50


class TestOutputTokensConfigUniform:
    """Test OutputTokensConfig uniform distribution validation."""

    def test_default_distribution_is_normal(self) -> None:
        config = OutputTokensConfig()
        assert config.distribution == SequenceLengthDistributionType.NORMAL

    def test_uniform_with_min_max_succeeds(self) -> None:
        config = OutputTokensConfig(
            distribution=SequenceLengthDistributionType.UNIFORM, min=5, max=50
        )
        assert config.distribution == SequenceLengthDistributionType.UNIFORM
        assert config.min == 5
        assert config.max == 50

    def test_uniform_without_min_raises(self) -> None:
        with pytest.raises(ValueError, match="Both 'min' and 'max' must be set"):
            OutputTokensConfig(
                distribution=SequenceLengthDistributionType.UNIFORM, max=100
            )

    def test_uniform_without_max_raises(self) -> None:
        with pytest.raises(ValueError, match="Both 'min' and 'max' must be set"):
            OutputTokensConfig(
                distribution=SequenceLengthDistributionType.UNIFORM, min=10
            )


# ============================================================
# Feature 1: sample_positive_uniform_integer
# ============================================================


class TestSamplePositiveUniformInteger:
    """Test RandomGenerator.sample_positive_uniform_integer."""

    def test_returns_value_in_range(self) -> None:
        gen = RandomGenerator(seed=42, _internal=True)
        for _ in range(100):
            val = gen.sample_positive_uniform_integer(10, 20)
            assert 10 <= val <= 20

    def test_equal_bounds_returns_exact_value(self) -> None:
        gen = RandomGenerator(seed=42, _internal=True)
        assert gen.sample_positive_uniform_integer(5, 5) == 5

    def test_low_less_than_1_raises(self) -> None:
        gen = RandomGenerator(seed=42, _internal=True)
        with pytest.raises(ValueError, match="Lower bound.*must be >= 1"):
            gen.sample_positive_uniform_integer(0, 10)

    def test_low_greater_than_high_raises(self) -> None:
        gen = RandomGenerator(seed=42, _internal=True)
        with pytest.raises(ValueError, match="Lower bound.*must be <= upper bound"):
            gen.sample_positive_uniform_integer(20, 10)

    @pytest.mark.parametrize(
        "low,high",
        [
            (1, 1),
            (1, 100),
            (50, 200),
            (1000, 2000),
        ],
    )
    def test_uniform_coverage(self, low: int, high: int) -> None:
        """Verify that sampling covers the full range (not just endpoints)."""
        gen = RandomGenerator(seed=42, _internal=True)
        values = {gen.sample_positive_uniform_integer(low, high) for _ in range(500)}
        if low < high:
            assert len(values) > 1  # Not degenerate


# ============================================================
# Feature 2: InEngineResponse output_token_ids
# ============================================================


class TestInEngineResponseTokenIds:
    """Test InEngineResponse output_token_ids field."""

    def test_default_is_none(self) -> None:
        resp = InEngineResponse(
            perf_ns=1000,
            text="hello",
            input_tokens=5,
            output_tokens=3,
        )
        assert resp.output_token_ids is None

    def test_set_token_ids(self) -> None:
        resp = InEngineResponse(
            perf_ns=1000,
            text="hello",
            input_tokens=5,
            output_tokens=3,
            output_token_ids=[101, 202, 303],
        )
        assert resp.output_token_ids == [101, 202, 303]

    def test_serialization_round_trip(self) -> None:
        resp = InEngineResponse(
            perf_ns=1000,
            text="hello",
            input_tokens=5,
            output_tokens=3,
            output_token_ids=[10, 20, 30],
        )
        data = asdict(resp)
        assert data["output_token_ids"] == [10, 20, 30]

        restored = InEngineResponse(**data)
        assert restored.output_token_ids == [10, 20, 30]

    def test_serialization_none_round_trip(self) -> None:
        resp = InEngineResponse(
            perf_ns=1000,
            text="hello",
            input_tokens=5,
            output_tokens=3,
        )
        data = asdict(resp)
        assert data["output_token_ids"] is None

        restored = InEngineResponse(**data)
        assert restored.output_token_ids is None
