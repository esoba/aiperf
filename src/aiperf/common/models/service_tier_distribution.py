# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Service tier distribution for OpenAI API requests.

Allows distributing requests across different service tiers (e.g., default, flex, priority)
with configurable probabilities. Format: ``tier:prob;tier:prob`` where probabilities are
percentages 0-100 that must sum to 100.

Example:
    >>> from aiperf.common.models.service_tier_distribution import ServiceTierDistributionParser
    >>> dist = ServiceTierDistributionParser.parse("default:50;flex:30;priority:20")
    >>> tier = dist.sample()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aiperf.common import random_generator as rng
from aiperf.common.aiperf_logger import AIPerfLogger

logger = AIPerfLogger(__name__)


def _validate_probability_sum(entries: list[ServiceTierEntry]) -> None:
    """Validate that probabilities sum to approximately 100.0.

    Args:
        entries: List of ServiceTierEntry objects to validate

    Raises:
        ValueError: If probabilities don't sum to 100.0 (within floating-point tolerance)
    """
    total_prob = sum(entry.probability for entry in entries)
    if not np.isclose(total_prob, 100.0, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"Probabilities must sum to 100.0, got {total_prob:.6f}. "
            f"Entries: {[str(e) for e in entries]}"
        )


@dataclass(frozen=True)
class ServiceTierEntry:
    """Immutable representation of a service tier with probability weight."""

    tier: str
    probability: float

    def __post_init__(self) -> None:
        """Validate tier name and probability on construction."""
        if not self.tier or not self.tier.strip():
            raise ValueError("Tier name must be non-empty")
        if not 0.0 <= self.probability <= 100.0:
            raise ValueError(f"Probability must be in [0,100], got {self.probability}")

    def __str__(self) -> str:
        return f"{self.tier}:{self.probability}%"


class ServiceTierDistribution:
    """Manages probability distribution of service tiers for request sampling.

    Supports efficient O(log n) sampling using binary search on cumulative
    probability distribution.
    """

    def __init__(self, entries: list[ServiceTierEntry]) -> None:
        """Initialize distribution from list of service tier entries.

        Args:
            entries: List of ServiceTierEntry objects. Probabilities must sum to 100.

        Raises:
            ValueError: If entries is empty or probabilities don't sum to 100.
        """
        if not entries:
            raise ValueError(
                "Distribution must contain at least one service tier entry"
            )

        self._rng = rng.derive("models.service_tier.distribution")
        self._entries = tuple(entries)
        _validate_probability_sum(list(self._entries))
        self._cumulative_probs = self._compute_cumulative_probabilities()

        logger.debug(
            lambda: f"Created service tier distribution with {len(self._entries)} entries: {self}"
        )

    def _compute_cumulative_probabilities(self) -> np.ndarray:
        """Compute cumulative probability distribution for efficient sampling."""
        probs = [entry.probability / 100.0 for entry in self._entries]
        return np.cumsum(probs, dtype=np.float64)

    def sample(self) -> str:
        """Sample a service tier according to the distribution.

        Returns:
            Service tier string value
        """
        rand_val = self._rng.random()
        idx = np.searchsorted(self._cumulative_probs, rand_val, side="right")
        idx = min(idx, len(self._entries) - 1)
        return self._entries[idx].tier

    @property
    def entries(self) -> tuple[ServiceTierEntry, ...]:
        """Get immutable view of service tier entries."""
        return self._entries

    def __str__(self) -> str:
        entries_str = ";".join(str(entry) for entry in self._entries)
        return f"ServiceTierDistribution[{entries_str}]"

    def __repr__(self) -> str:
        return f"ServiceTierDistribution({list(self._entries)})"


class ServiceTierDistributionParser:
    """Parser for service tier distribution strings."""

    @classmethod
    def parse(cls, dist_str: str) -> ServiceTierDistribution:
        """Parse a service tier distribution string.

        Format: ``tier:prob;tier:prob`` where probabilities are percentages 0-100.

        Args:
            dist_str: Distribution specification string (e.g., "default:50;flex:30;priority:20")

        Returns:
            ServiceTierDistribution object

        Raises:
            ValueError: If string format is invalid
        """
        if not isinstance(dist_str, str) or not dist_str.strip():
            raise ValueError("Distribution string cannot be empty")

        dist_str = dist_str.strip()
        entries: list[ServiceTierEntry] = []

        for pair_str in dist_str.split(";"):
            pair_str = pair_str.strip()
            if not pair_str:
                continue

            parts = pair_str.rsplit(":", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid pair format: '{pair_str}'. Expected 'tier:probability'"
                )

            tier = parts[0].strip()
            try:
                probability = float(parts[1].strip())
            except ValueError as e:
                raise ValueError(
                    f"Invalid probability value in '{pair_str}': {parts[1].strip()}"
                ) from e

            entries.append(ServiceTierEntry(tier=tier, probability=probability))

        if not entries:
            raise ValueError("No valid entries found in distribution string")

        return ServiceTierDistribution(entries)
