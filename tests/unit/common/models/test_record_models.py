# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.models import MetricResult, ProfileResults, TimesliceWindow


class TestProfileResults:
    """Test cases for ProfileResults model."""

    def test_profile_results_with_timeslice_metric_results(self):
        """Test ProfileResults can store timeslice metric results."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        timeslice_results = {
            0: {metric_result.tag: metric_result},
            1: {metric_result.tag: metric_result},
        }

        profile_results = ProfileResults(
            records={metric_result.tag: metric_result},
            timeslice_metric_results=timeslice_results,
            completed=1,
            start_ns=1000000000,
            end_ns=2000000000,
        )

        assert profile_results.timeslice_metric_results is not None
        assert 0 in profile_results.timeslice_metric_results
        assert 1 in profile_results.timeslice_metric_results
        assert len(profile_results.timeslice_metric_results[0]) == 1
        assert len(profile_results.timeslice_metric_results[1]) == 1

    def test_profile_results_without_timeslice_metric_results(self):
        """Test ProfileResults works without timeslice metric results."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        profile_results = ProfileResults(
            records={metric_result.tag: metric_result},
            completed=1,
            start_ns=1000000000,
            end_ns=2000000000,
        )

        assert profile_results.timeslice_metric_results is None

    def test_profile_results_with_empty_timeslice_dict(self):
        """Test ProfileResults with empty timeslice results dict."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        profile_results = ProfileResults(
            records={metric_result.tag: metric_result},
            timeslice_metric_results={},
            completed=1,
            start_ns=1000000000,
            end_ns=2000000000,
        )

        assert profile_results.timeslice_metric_results is not None
        assert len(profile_results.timeslice_metric_results) == 0

    def test_profile_results_with_multiple_timeslices_and_metrics(self):
        """Test ProfileResults with multiple timeslices containing multiple metrics."""
        latency_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        throughput_result = MetricResult(
            tag="request_throughput",
            header="Request Throughput",
            unit="requests/sec",
            avg=50.0,
            count=1,
        )

        timeslice_results = {
            0: {
                latency_result.tag: latency_result,
                throughput_result.tag: throughput_result,
            },
            1: {
                latency_result.tag: latency_result,
                throughput_result.tag: throughput_result,
            },
            2: {
                latency_result.tag: latency_result,
                throughput_result.tag: throughput_result,
            },
        }

        profile_results = ProfileResults(
            records={
                latency_result.tag: latency_result,
                throughput_result.tag: throughput_result,
            },
            timeslice_metric_results=timeslice_results,
            completed=2,
            start_ns=1000000000,
            end_ns=3000000000,
        )

        assert profile_results.timeslice_metric_results is not None
        assert len(profile_results.timeslice_metric_results) == 3
        for i in range(3):
            assert i in profile_results.timeslice_metric_results
            assert len(profile_results.timeslice_metric_results[i]) == 2

    def test_profile_results_with_timeslice_windows(self):
        """Test ProfileResults accepts timeslice_windows."""
        windows = {
            0: TimesliceWindow(start_ns=1_000_000_000, end_ns=2_000_000_000),
            1: TimesliceWindow(start_ns=2_000_000_000, end_ns=3_000_000_000),
        }

        profile_results = ProfileResults(
            records={},
            timeslice_windows=windows,
            completed=0,
            start_ns=1_000_000_000,
            end_ns=3_000_000_000,
        )

        assert profile_results.timeslice_windows is not None
        assert len(profile_results.timeslice_windows) == 2
        assert profile_results.timeslice_windows[0].start_ns == 1_000_000_000
        assert profile_results.timeslice_windows[0].end_ns == 2_000_000_000
        assert profile_results.timeslice_windows[0].is_complete is None
        assert profile_results.timeslice_windows[1].start_ns == 2_000_000_000

    def test_profile_results_without_timeslice_windows(self):
        """Test ProfileResults defaults timeslice_windows to None."""
        profile_results = ProfileResults(
            records={},
            completed=0,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        )

        assert profile_results.timeslice_windows is None
