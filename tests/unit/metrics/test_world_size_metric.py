# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for WorldSizeMetric.

Focuses on:
- Default derive value returns 1 (single GPU fallback)
- Metric flags mark it as internal / no-console
- No required_metrics dependency
"""

from aiperf.common.enums import MetricFlags
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.world_size_metric import WorldSizeMetric


class TestWorldSizeMetric:
    """Verify WorldSizeMetric behavior."""

    def test_derive_value_returns_default_of_one(self) -> None:
        """Default derive returns 1, overridden by pre-seeding in processors."""
        metric = WorldSizeMetric()
        metric_results = MetricResultsDict()

        result = metric.derive_value(metric_results)
        assert result == 1

    def test_flags_internal_and_no_console(self) -> None:
        assert WorldSizeMetric.has_flags(MetricFlags.INTERNAL)
        assert WorldSizeMetric.has_flags(MetricFlags.NO_CONSOLE)

    def test_tag(self) -> None:
        assert WorldSizeMetric.tag == "world_size"

    def test_required_metrics_is_none(self) -> None:
        """WorldSizeMetric has no upstream dependencies."""
        assert WorldSizeMetric.required_metrics is None

    def test_derive_value_ignores_existing_results(self) -> None:
        """Even if results dict has data, _derive_value still returns 1.

        The actual world_size override happens via pre-seeding in the processor,
        not inside _derive_value.
        """
        metric = WorldSizeMetric()
        metric_results = MetricResultsDict()
        metric_results["some_other_metric"] = 42

        result = metric.derive_value(metric_results)
        assert result == 1
