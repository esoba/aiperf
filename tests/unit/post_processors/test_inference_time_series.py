# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for InferenceTimeSeries."""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.common.models import MetricResult
from aiperf.post_processors.inference_time_series import InferenceTimeSeries


class TestInferenceTimeSeries:
    def test_empty(self) -> None:
        ts = InferenceTimeSeries()
        assert len(ts) == 0
        assert len(ts.record_indices) == 0
        assert len(ts.values) == 0

    def test_append_single(self) -> None:
        ts = InferenceTimeSeries()
        ts.append(0, 42.0)
        assert len(ts) == 1
        assert ts.record_indices[0] == 0
        assert ts.values[0] == 42.0

    def test_append_multiple(self) -> None:
        ts = InferenceTimeSeries()
        ts.append(0, 10.0)
        ts.append(1, 20.0)
        ts.append(2, 30.0)
        assert len(ts) == 3
        np.testing.assert_array_equal(ts.record_indices, [0, 1, 2])
        np.testing.assert_array_equal(ts.values, [10.0, 20.0, 30.0])

    def test_extend(self) -> None:
        ts = InferenceTimeSeries()
        ts.extend(0, [10.0, 20.0, 30.0])
        assert len(ts) == 3
        # All three values share the same record index
        np.testing.assert_array_equal(ts.record_indices, [0, 0, 0])
        np.testing.assert_array_equal(ts.values, [10.0, 20.0, 30.0])

    def test_extend_empty_list(self) -> None:
        ts = InferenceTimeSeries()
        ts.extend(0, [])
        assert len(ts) == 0

    def test_mixed_append_extend(self) -> None:
        ts = InferenceTimeSeries()
        ts.append(0, 5.0)
        ts.extend(1, [10.0, 20.0])
        ts.append(2, 30.0)
        assert len(ts) == 4
        np.testing.assert_array_equal(ts.record_indices, [0, 1, 1, 2])
        np.testing.assert_array_equal(ts.values, [5.0, 10.0, 20.0, 30.0])


class TestGetValueMask:
    def test_empty_series(self) -> None:
        ts = InferenceTimeSeries()
        record_mask = np.array([], dtype=bool)
        value_mask = ts.get_value_mask(record_mask)
        assert len(value_mask) == 0

    def test_all_records_selected(self) -> None:
        ts = InferenceTimeSeries()
        ts.append(0, 1.0)
        ts.append(1, 2.0)
        ts.append(2, 3.0)
        record_mask = np.array([True, True, True], dtype=bool)
        value_mask = ts.get_value_mask(record_mask)
        np.testing.assert_array_equal(value_mask, [True, True, True])

    def test_no_records_selected(self) -> None:
        ts = InferenceTimeSeries()
        ts.append(0, 1.0)
        ts.append(1, 2.0)
        record_mask = np.array([False, False], dtype=bool)
        value_mask = ts.get_value_mask(record_mask)
        np.testing.assert_array_equal(value_mask, [False, False])

    def test_partial_selection(self) -> None:
        ts = InferenceTimeSeries()
        ts.append(0, 10.0)
        ts.append(1, 20.0)
        ts.append(2, 30.0)
        ts.append(3, 40.0)
        # Select records 1 and 2 only
        record_mask = np.array([False, True, True, False], dtype=bool)
        value_mask = ts.get_value_mask(record_mask)
        np.testing.assert_array_equal(value_mask, [False, True, True, False])

    def test_list_valued_metric_maps_correctly(self) -> None:
        """Multiple values per record index are all selected/deselected together."""
        ts = InferenceTimeSeries()
        ts.extend(0, [1.0, 2.0])  # record 0: 2 values
        ts.extend(1, [3.0, 4.0, 5.0])  # record 1: 3 values
        ts.append(2, 6.0)  # record 2: 1 value
        # Select only record 1
        record_mask = np.array([False, True, False], dtype=bool)
        value_mask = ts.get_value_mask(record_mask)
        np.testing.assert_array_equal(
            value_mask, [False, False, True, True, True, False]
        )

    def test_sparse_tag_with_wider_record_mask(self) -> None:
        """Tag only has values for some records; record mask covers all records."""
        ts = InferenceTimeSeries()
        ts.append(1, 10.0)  # only record 1
        ts.append(3, 30.0)  # only record 3
        # 5 total records, select records 0-2
        record_mask = np.array([True, True, True, False, False], dtype=bool)
        value_mask = ts.get_value_mask(record_mask)
        # record 1 → True, record 3 → False
        np.testing.assert_array_equal(value_mask, [True, False])


class TestToMetricResult:
    def test_basic_stats(self) -> None:
        ts = InferenceTimeSeries()
        for i, v in enumerate([10.0, 20.0, 30.0]):
            ts.append(i, v)

        result = ts.to_metric_result("test_tag", "Test Metric", "ms")
        assert isinstance(result, MetricResult)
        assert result.tag == "test_tag"
        assert result.header == "Test Metric"
        assert result.unit == "ms"
        assert result.avg == pytest.approx(20.0)
        assert result.min == pytest.approx(10.0)
        assert result.max == pytest.approx(30.0)
        assert result.count == 3

    def test_single_value(self) -> None:
        ts = InferenceTimeSeries()
        ts.append(0, 42.0)
        result = ts.to_metric_result("tag", "header", "unit")
        assert result.avg == pytest.approx(42.0)
        assert result.min == pytest.approx(42.0)
        assert result.max == pytest.approx(42.0)
        assert result.count == 1


class TestToMetricResultFiltered:
    def test_filtered_subset(self) -> None:
        ts = InferenceTimeSeries()
        ts.append(0, 10.0)
        ts.append(1, 20.0)
        ts.append(2, 30.0)
        ts.append(3, 40.0)

        # Select records 1 and 2
        record_mask = np.array([False, True, True, False], dtype=bool)
        result = ts.to_metric_result_filtered("tag", "header", "unit", record_mask)
        assert result is not None
        assert result.count == 2
        assert result.avg == pytest.approx(25.0)

    def test_filtered_empty(self) -> None:
        ts = InferenceTimeSeries()
        ts.append(0, 10.0)
        record_mask = np.array([False], dtype=bool)
        result = ts.to_metric_result_filtered("tag", "header", "unit", record_mask)
        assert result is None

    def test_filtered_all(self) -> None:
        ts = InferenceTimeSeries()
        ts.append(0, 10.0)
        ts.append(1, 20.0)
        record_mask = np.array([True, True], dtype=bool)
        result = ts.to_metric_result_filtered("tag", "header", "unit", record_mask)
        assert result is not None
        assert result.count == 2

    def test_filtered_list_valued(self) -> None:
        """Filtering works with list-valued metrics (multiple values per record)."""
        ts = InferenceTimeSeries()
        ts.extend(0, [10.0, 20.0])
        ts.extend(1, [30.0, 40.0])
        # Select only record 0
        record_mask = np.array([True, False], dtype=bool)
        result = ts.to_metric_result_filtered("tag", "header", "unit", record_mask)
        assert result is not None
        assert result.count == 2
        assert result.avg == pytest.approx(15.0)


class TestAsMetricArray:
    def test_shares_values(self) -> None:
        ts = InferenceTimeSeries()
        ts.append(0, 42.0)
        ts.append(1, 84.0)

        ga = ts.as_metric_array()
        np.testing.assert_array_equal(ga.data, [42.0, 84.0])
        # It's the same underlying GrowableArray
        assert ga is ts._values
