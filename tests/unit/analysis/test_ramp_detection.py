# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ramp boundary detection."""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.analysis.ramp_detection import (
    detect_steady_state_window,
    manual_steady_state_window,
)
from aiperf.analysis.sweep import concurrency_sweep


class TestDetectSteadyStateWindow:
    def test_empty_input(self) -> None:
        start, end = detect_steady_state_window(
            np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        )
        assert start == 0.0
        assert end == 0.0

    def test_constant_concurrency_full_window(self) -> None:
        """Constant concurrency → window should be approximately the full range."""
        # 100 requests all active for the entire duration
        n = 100
        start_ns = np.zeros(n)
        end_ns = np.full(n, 1000.0)
        sorted_ts, concurrency = concurrency_sweep(start_ns, end_ns)

        window_start, window_end = detect_steady_state_window(sorted_ts, concurrency)
        # Should cover most of the range
        assert window_start <= 100.0
        assert window_end >= 900.0

    def test_ramp_up_steady_ramp_down(self) -> None:
        """Linear ramp up → steady state → ramp down. Known boundaries."""
        # Build a scenario:
        # Phase 1 (ramp up): 0-100, requests start staggered
        # Phase 2 (steady): 100-900, all 50 requests running
        # Phase 3 (ramp down): 900-1000, requests end staggered
        n_steady = 50
        start_ns = np.concatenate(
            [
                np.linspace(0, 100, n_steady),  # ramp up
            ]
        )
        end_ns = np.concatenate(
            [
                np.linspace(900, 1000, n_steady),  # ramp down
            ]
        )
        sorted_ts, concurrency = concurrency_sweep(start_ns, end_ns)

        window_start, window_end = detect_steady_state_window(
            sorted_ts,
            concurrency,
            stability_fraction=0.90,
            sustained_window_pct=5.0,
            min_window_pct=10.0,
        )

        # Window should exclude ramp regions
        assert window_start >= 0.0
        assert window_end <= 1000.0
        # Should be a reasonable portion of the total
        window_pct = (window_end - window_start) / 1000.0 * 100
        assert window_pct >= 10.0

    def test_short_benchmark_falls_back(self) -> None:
        """Very short benchmark → window below min_window_pct → falls back to full range."""
        # 3 sequential requests, very short
        start_ns = np.array([0.0, 1.0, 2.0])
        end_ns = np.array([3.0, 4.0, 5.0])
        sorted_ts, concurrency = concurrency_sweep(start_ns, end_ns)

        window_start, window_end = detect_steady_state_window(
            sorted_ts,
            concurrency,
            min_window_pct=90.0,  # Very high threshold → force fallback
        )

        # Should fall back to full range
        assert window_start == float(sorted_ts[0])
        assert window_end == float(sorted_ts[-1])

    def test_spike_during_ramp_not_false_positive(self) -> None:
        """A transient spike during ramp should not trigger false positive."""
        # Ramp up with a brief spike
        start_ns = np.concatenate(
            [
                np.linspace(0, 500, 10),  # slow ramp
                np.linspace(500, 1000, 50),  # steady state
            ]
        )
        end_ns = np.concatenate(
            [
                np.linspace(500, 1000, 10),
                np.linspace(1000, 1500, 50),
            ]
        )
        sorted_ts, concurrency = concurrency_sweep(start_ns, end_ns)

        window_start, window_end = detect_steady_state_window(
            sorted_ts,
            concurrency,
            sustained_window_pct=10.0,
        )

        # Window should exist and cover a reasonable portion
        assert window_end > window_start


class TestManualSteadyStateWindow:
    def test_basic(self) -> None:
        start, end = manual_steady_state_window(0.0, 1000.0, 10.0, 90.0)
        assert start == pytest.approx(100.0)
        assert end == pytest.approx(900.0)

    def test_full_range(self) -> None:
        start, end = manual_steady_state_window(0.0, 1000.0, 0.0, 100.0)
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(1000.0)

    def test_narrow_window(self) -> None:
        start, end = manual_steady_state_window(100.0, 200.0, 40.0, 60.0)
        assert start == pytest.approx(140.0)
        assert end == pytest.approx(160.0)

    def test_offset_timestamps(self) -> None:
        """Verify window computation works with non-zero base timestamps."""
        start, end = manual_steady_state_window(1000.0, 2000.0, 20.0, 80.0)
        assert start == pytest.approx(1200.0)
        assert end == pytest.approx(1800.0)
