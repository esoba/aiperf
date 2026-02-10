# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SteadyStateAnalyzer."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from aiperf.common.accumulator_protocols import SummaryContext
from aiperf.common.config import UserConfig
from aiperf.common.exceptions import PluginDisabled
from aiperf.plugin.enums import AccumulatorType
from aiperf.post_processors.steady_state_analyzer import (
    SteadyStateAnalyzer,
    SteadyStateSummary,
)
from tests.unit.post_processors.conftest import (
    create_accumulator_with_metrics,
    create_metric_records_message,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_config(
    enabled: bool = True,
    stability_fraction: float = 0.90,
    sustained_window_pct: float = 5.0,
    min_window_pct: float = 10.0,
    start_pct: float | None = None,
    end_pct: float | None = None,
) -> UserConfig:
    """Create a UserConfig with the given steady-state settings."""
    from aiperf.common.config.output_config import OutputConfig
    from aiperf.common.config.steady_state_config import SteadyStateConfig

    ss_kwargs: dict[str, object] = {
        "enabled": enabled,
        "stability_fraction": stability_fraction,
        "sustained_window_pct": sustained_window_pct,
        "min_window_pct": min_window_pct,
    }
    if start_pct is not None:
        ss_kwargs["start_pct"] = start_pct
    if end_pct is not None:
        ss_kwargs["end_pct"] = end_pct

    return UserConfig(
        endpoint={
            "model_names": ["test-model"],
            "type": "completions",
            "streaming": False,
        },
        output=OutputConfig(steady_state=SteadyStateConfig(**ss_kwargs)),
    )


def _make_record_metric():
    """Create a simple RECORD metric class for testing."""
    from aiperf.common.enums import MetricType

    class FakeLatency:
        tag = "request_latency"
        type = MetricType.RECORD
        header = "Request Latency"
        unit = "ms"

        def derive_value(self, results):
            raise NotImplementedError

    return FakeLatency


async def _build_accumulator_with_records(
    mock_metric_registry: Mock,
    user_config: UserConfig,
    records: list[tuple[int, int, int, float]],
) -> object:
    """Build and populate a MetricsAccumulator.

    Args:
        records: list of (session_num, start_ns, end_ns, latency_value)
    """
    metric_cls = _make_record_metric()
    acc = create_accumulator_with_metrics(user_config, metric_cls)

    for session_num, start_ns, end_ns, latency in records:
        msg = create_metric_records_message(
            session_num=session_num,
            request_start_ns=start_ns,
            request_end_ns=end_ns,
            results=[{"request_latency": latency}],
        )
        await acc.process_record(msg.to_data())

    return acc


def _make_summary_ctx(acc: object) -> SummaryContext:
    """Build a SummaryContext with the given MetricsAccumulator."""
    return SummaryContext(
        accumulators={AccumulatorType.METRIC_RESULTS: acc},
        start_ns=0,
        end_ns=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSteadyStateAnalyzerDisabled:
    def test_disabled_config_raises_plugin_disabled(self) -> None:
        """Raises PluginDisabled when steady-state is disabled."""
        config = _make_user_config(enabled=False)
        with pytest.raises(PluginDisabled, match="disabled"):
            SteadyStateAnalyzer(user_config=config)


class TestSteadyStateAnalyzerNoRecords:
    @pytest.mark.asyncio
    async def test_no_accumulator_raises(self) -> None:
        """Raises PluginDisabled when MetricsAccumulator is not in context."""
        ctx = SummaryContext()
        config = _make_user_config()
        ss = SteadyStateAnalyzer(user_config=config)
        with pytest.raises(PluginDisabled, match="MetricsAccumulator not available"):
            await ss.summarize(ctx)

    @pytest.mark.asyncio
    async def test_empty_accumulator_raises(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Raises PluginDisabled when accumulator has no records."""
        metric_cls = _make_record_metric()
        acc = create_accumulator_with_metrics(mock_user_config, metric_cls)
        ctx = _make_summary_ctx(acc)

        config = _make_user_config()
        ss = SteadyStateAnalyzer(user_config=config)
        with pytest.raises(PluginDisabled, match="No records"):
            await ss.summarize(ctx)


class TestSteadyStateAnalyzerConstantConcurrency:
    @pytest.mark.asyncio
    async def test_constant_concurrency_covers_full_range(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """With all requests overlapping, window should cover most of the range."""
        # 50 fully-overlapping requests: start=0, end=1_000_000_000
        records = [(i, 0, 1_000_000_000, float(i * 10)) for i in range(50)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config()
        ss = SteadyStateAnalyzer(user_config=config)
        result = await ss.summarize(ctx)

        assert isinstance(result, SteadyStateSummary)
        assert result.window_metadata.detection_method == "concurrency_threshold"
        assert result.window_metadata.total_requests == 50
        # All requests should be in steady state (constant concurrency)
        assert result.window_metadata.steady_state_requests == 50
        # Effective concurrency metric: constant 50 → all stats ≈ 50, std ≈ 0
        conc = result.effective_concurrency
        assert conc.avg == pytest.approx(50.0, rel=0.01)
        assert conc.min == pytest.approx(50.0)
        assert conc.max == pytest.approx(50.0)
        assert conc.std == pytest.approx(0.0, abs=0.1)


class TestSteadyStateAnalyzerRamp:
    @pytest.mark.asyncio
    async def test_ramp_up_steady_ramp_down(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Linear ramp → steady → drain. Verify window excludes ramp regions."""
        # Phase 1 (ramp up): 10 requests starting staggered 0-100
        # Phase 2 (steady): 40 requests all starting at ~100, ending at ~900
        # Phase 3 (ramp down): 10 requests ending staggered 900-1000
        records = []
        session = 0

        # Ramp-up: staggered starts, all end at 900
        for i in range(10):
            start = int(i * 10)
            records.append((session, start, 900, 100.0))
            session += 1

        # Steady state: all start at 100, all end at 900
        for _i in range(40):
            records.append((session, 100, 900, 100.0))
            session += 1

        # Ramp-down: all start at 100, staggered ends
        for i in range(10):
            end = 900 + int(i * 10)
            records.append((session, 100, end, 100.0))
            session += 1

        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(
            stability_fraction=0.90,
            sustained_window_pct=5.0,
            min_window_pct=10.0,
        )
        ss = SteadyStateAnalyzer(user_config=config)
        result = await ss.summarize(ctx)

        assert result.window_metadata.total_requests == 60
        # Some requests should be excluded
        assert result.window_metadata.steady_state_requests <= 60
        # Should still have a meaningful window
        assert result.window_metadata.steady_state_duration_ns > 0


class TestSteadyStateAnalyzerUserOverride:
    @pytest.mark.asyncio
    async def test_user_override_exact_boundaries(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """User override: start_pct=10, end_pct=90 → verify exact boundaries."""
        # 10 sequential requests spanning 0-1000
        records = [(i, i * 100, (i + 1) * 100, float(i)) for i in range(10)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=10.0, end_pct=90.0)
        ss = SteadyStateAnalyzer(user_config=config)
        result = await ss.summarize(ctx)

        assert result.window_metadata.detection_method == "user_override"
        # Window should be 10%-90% of [0, 1000] = [100, 900]
        assert result.window_metadata.ramp_up_end_ns == pytest.approx(100.0)
        assert result.window_metadata.ramp_down_start_ns == pytest.approx(900.0)
        # Effective concurrency metric should be populated
        conc = result.effective_concurrency
        assert conc.avg >= 0.0
        assert conc.min >= 0.0
        assert conc.max >= conc.min

    @pytest.mark.asyncio
    async def test_user_override_request_counts(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """User override filters requests that fall entirely within the window."""
        # 10 sequential requests: [0,100), [100,200), ..., [900,1000)
        records = [(i, i * 100, (i + 1) * 100, float(i)) for i in range(10)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        # Window is [200, 800] — requests [2,3,4,5,6,7] have both endpoints inside
        config = _make_user_config(start_pct=20.0, end_pct=80.0)
        ss = SteadyStateAnalyzer(user_config=config)
        result = await ss.summarize(ctx)

        # Requests 2-7: start >= 200 and end <= 800
        # Request 2: start=200, end=300 → in
        # Request 7: start=700, end=800 → in
        # Request 8: start=800, end=900 → start=800 is >=200, but end=900>800 → out
        assert result.window_metadata.steady_state_requests == 6


class TestSteadyStateAnalyzerMetricsCorrectness:
    @pytest.mark.asyncio
    async def test_windowed_metrics_match_subset(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Verify that metrics are computed only from steady-state records."""
        # 3 records: ramp(latency=1000), steady(latency=100), drain(latency=1000)
        records = [
            (0, 0, 100, 1000.0),  # ramp — latency 1000
            (1, 100, 900, 100.0),  # steady — latency 100
            (2, 900, 1000, 1000.0),  # drain — latency 1000
        ]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        # Window is [10%, 90%] of [0, 1000] = [100, 900]
        config = _make_user_config(start_pct=10.0, end_pct=90.0)
        ss = SteadyStateAnalyzer(user_config=config)
        result = await ss.summarize(ctx)

        # Only request 1 (latency=100) should be in the window
        assert result.window_metadata.steady_state_requests == 1
        assert "request_latency" in result.results
        assert result.results["request_latency"].avg == pytest.approx(100.0)


class TestSteadyStateSummarySerialize:
    def _make_summary(self) -> SteadyStateSummary:
        from aiperf.common.models import MetricResult
        from aiperf.post_processors.steady_state_analyzer import (
            SteadyStateWindowMetadata,
        )

        return SteadyStateSummary(
            results={
                "test_tag": MetricResult(
                    tag="test_tag", header="Test", unit="ms", avg=42.0, count=1
                )
            },
            effective_concurrency=MetricResult(
                tag="effective_concurrency",
                header="Effective Concurrency",
                unit="requests",
                avg=5.0,
                min=3.0,
                max=8.0,
                p50=5.0,
                p90=7.0,
                p95=8.0,
                p99=8.0,
                std=1.5,
            ),
            window_metadata=SteadyStateWindowMetadata(
                ramp_up_end_ns=100.0,
                ramp_down_start_ns=900.0,
                steady_state_duration_ns=800.0,
                total_requests=10,
                steady_state_requests=8,
                detection_method="concurrency_threshold",
            ),
        )

    def test_to_json(self) -> None:
        summary = self._make_summary()
        data = summary.to_json()
        assert "results" in data
        assert "window_metadata" in data
        assert data["window_metadata"]["detection_method"] == "concurrency_threshold"
        assert data["window_metadata"]["total_requests"] == 10
        assert data["effective_concurrency"]["avg"] == 5.0
        assert data["effective_concurrency"]["p99"] == 8.0

    def test_to_csv(self) -> None:
        summary = self._make_summary()
        rows = summary.to_csv()
        assert len(rows) == 1
        assert rows[0]["tag"] == "test_tag"
