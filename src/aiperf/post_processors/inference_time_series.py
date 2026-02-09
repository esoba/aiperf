# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-metric numpy-backed value storage with record-index mapping for inference metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aiperf.common.growable_array import GrowableArray
from aiperf.common.models import MetricResult
from aiperf.common.types import MetricTagT


class InferenceTimeSeries:
    """Per-metric numpy-backed value storage with record-index mapping.

    Stores values + record indices (int32) pointing into a shared timestamp
    array owned by MetricsAccumulator. For list-valued metrics (e.g. ITL),
    the record index is repeated for each value, keeping arrays parallel.

    Time queries use record-level masks computed once on the shared timestamp
    array, then mapped to per-value masks via ``get_value_mask(record_mask)``.
    All operations are fully vectorized numpy.
    """

    __slots__ = ("_record_indices", "_values")

    def __init__(self) -> None:
        self._values = GrowableArray(
            initial_capacity=256, dtype=np.float64, track_sum=True
        )
        self._record_indices = GrowableArray(initial_capacity=256, dtype=np.int32)

    def append(self, record_idx: int, value: float) -> None:
        """Append a single value associated with the given record index."""
        self._values.append(value)
        self._record_indices.append(record_idx)

    def extend(self, record_idx: int, values: list[float]) -> None:
        """Append multiple values sharing the same record index (e.g. ITL list)."""
        n = len(values)
        if n == 0:
            return
        idx_arr = np.full(n, record_idx, dtype=np.int32)
        val_arr = np.asarray(values, dtype=np.float64)
        self._record_indices.extend(idx_arr)
        self._values.extend(val_arr)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def values(self) -> NDArray[np.float64]:
        """View of stored values."""
        return self._values.data

    @property
    def record_indices(self) -> NDArray[np.int32]:
        """View of stored record indices."""
        return self._record_indices.data

    def get_value_mask(self, record_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Map a record-level boolean mask to a per-value boolean mask.

        Uses fancy indexing: ``record_mask[record_indices]`` — fully vectorized.
        """
        if len(self._record_indices) == 0:
            return np.zeros(0, dtype=bool)
        return record_mask[self._record_indices.data]

    def to_metric_result(self, tag: MetricTagT, header: str, unit: str) -> MetricResult:
        """Compute full stats from all stored values (vectorized)."""
        return self._compute_result(tag, header, unit, self.values)

    def to_metric_result_filtered(
        self,
        tag: MetricTagT,
        header: str,
        unit: str,
        record_mask: NDArray[np.bool_],
    ) -> MetricResult | None:
        """Compute stats for values matching the record mask. Returns None if empty."""
        value_mask = self.get_value_mask(record_mask)
        filtered = self.values[value_mask]
        if len(filtered) == 0:
            return None
        return self._compute_result(tag, header, unit, filtered)

    @staticmethod
    def _compute_result(
        tag: MetricTagT, header: str, unit: str, arr: NDArray[np.float64]
    ) -> MetricResult:
        """Compute MetricResult stats from a numpy array."""
        p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
            arr, [1, 5, 10, 25, 50, 75, 90, 95, 99]
        )
        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            avg=float(np.mean(arr)),
            std=float(np.std(arr)),
            p1=float(p1),
            p5=float(p5),
            p10=float(p10),
            p25=float(p25),
            p50=float(p50),
            p75=float(p75),
            p90=float(p90),
            p95=float(p95),
            p99=float(p99),
            count=len(arr),
        )

    def as_metric_array(self) -> GrowableArray:
        """Return the underlying values GrowableArray for backward compat with MetricResultsDict.

        Derive functions expect MetricArray-like objects in the results dict.
        Since MetricArray wraps GrowableArray internally, we can share the same
        GrowableArray to avoid copying.
        """
        return self._values
