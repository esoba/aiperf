# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for service tier distribution functionality."""

import pytest

from aiperf.common.models.service_tier_distribution import (
    ServiceTierDistribution,
    ServiceTierDistributionParser,
    ServiceTierEntry,
)


class TestServiceTierEntry:
    """Test ServiceTierEntry validation and behavior."""

    def test_valid_entry_creation(self):
        entry = ServiceTierEntry("default", 50.0)
        assert entry.tier == "default"
        assert entry.probability == 50.0

    def test_empty_tier_name_raises(self):
        with pytest.raises(ValueError, match="Tier name must be non-empty"):
            ServiceTierEntry("", 50.0)

    def test_whitespace_tier_name_raises(self):
        with pytest.raises(ValueError, match="Tier name must be non-empty"):
            ServiceTierEntry("   ", 50.0)

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError, match="Probability must be in"):
            ServiceTierEntry("default", -10.0)

    def test_probability_over_100_raises(self):
        with pytest.raises(ValueError, match="Probability must be in"):
            ServiceTierEntry("default", 110.0)

    def test_str_representation(self):
        entry = ServiceTierEntry("flex", 30.0)
        assert str(entry) == "flex:30.0%"


class TestServiceTierDistribution:
    """Test ServiceTierDistribution sampling and validation."""

    def test_single_tier_always_returns_that_tier(self):
        dist = ServiceTierDistribution([ServiceTierEntry("flex", 100.0)])
        for _ in range(100):
            assert dist.sample() == "flex"

    def test_empty_entries_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ServiceTierDistribution([])

    def test_probabilities_must_sum_to_100(self):
        with pytest.raises(ValueError, match="sum to 100"):
            ServiceTierDistribution(
                [
                    ServiceTierEntry("default", 50.0),
                    ServiceTierEntry("flex", 30.0),
                ]
            )

    def test_multi_tier_distribution_samples_all_tiers(self):
        dist = ServiceTierDistribution(
            [
                ServiceTierEntry("default", 50.0),
                ServiceTierEntry("flex", 30.0),
                ServiceTierEntry("priority", 20.0),
            ]
        )
        counts: dict[str, int] = {"default": 0, "flex": 0, "priority": 0}
        for _ in range(10000):
            tier = dist.sample()
            counts[tier] += 1

        # All tiers should be sampled at least once with 10k samples
        assert counts["default"] > 0
        assert counts["flex"] > 0
        assert counts["priority"] > 0

        # Rough distribution check (with wide tolerance)
        assert 4000 < counts["default"] < 6000
        assert 2000 < counts["flex"] < 4000
        assert 1000 < counts["priority"] < 3000

    def test_entries_property_returns_immutable_tuple(self):
        entries = [ServiceTierEntry("default", 100.0)]
        dist = ServiceTierDistribution(entries)
        assert isinstance(dist.entries, tuple)
        assert len(dist.entries) == 1

    def test_str_representation(self):
        dist = ServiceTierDistribution(
            [
                ServiceTierEntry("default", 60.0),
                ServiceTierEntry("flex", 40.0),
            ]
        )
        result = str(dist)
        assert "ServiceTierDistribution" in result
        assert "default" in result
        assert "flex" in result


class TestServiceTierDistributionParser:
    """Test ServiceTierDistributionParser parsing."""

    def test_parse_single_tier(self):
        dist = ServiceTierDistributionParser.parse("flex:100")
        assert len(dist.entries) == 1
        assert dist.entries[0].tier == "flex"
        assert dist.entries[0].probability == 100.0

    def test_parse_multiple_tiers(self):
        dist = ServiceTierDistributionParser.parse("default:50;flex:30;priority:20")
        assert len(dist.entries) == 3
        assert dist.entries[0].tier == "default"
        assert dist.entries[0].probability == 50.0
        assert dist.entries[1].tier == "flex"
        assert dist.entries[1].probability == 30.0
        assert dist.entries[2].tier == "priority"
        assert dist.entries[2].probability == 20.0

    def test_parse_with_whitespace(self):
        dist = ServiceTierDistributionParser.parse(" default : 50 ; flex : 50 ")
        assert len(dist.entries) == 2

    def test_parse_empty_string_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            ServiceTierDistributionParser.parse("")

    def test_parse_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid pair format"):
            ServiceTierDistributionParser.parse("just_a_string")

    def test_parse_invalid_probability_raises(self):
        with pytest.raises(ValueError, match="Invalid probability"):
            ServiceTierDistributionParser.parse("default:abc")

    def test_parse_probabilities_not_summing_to_100_raises(self):
        with pytest.raises(ValueError, match="sum to 100"):
            ServiceTierDistributionParser.parse("default:50;flex:20")

    @pytest.mark.parametrize(
        "dist_str,expected_tiers",
        [
            ("auto:100", ["auto"]),
            ("default:50;flex:50", ["default", "flex"]),
            (
                "scale:33.33;priority:33.33;default:33.34",
                ["scale", "priority", "default"],
            ),
        ],
    )
    def test_parse_various_valid_formats(self, dist_str, expected_tiers):
        dist = ServiceTierDistributionParser.parse(dist_str)
        assert [e.tier for e in dist.entries] == expected_tiers
