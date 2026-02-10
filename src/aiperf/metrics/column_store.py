# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-indexed NaN-sparse columnar storage for per-record metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from aiperf.metrics.ragged_series import RaggedSeries


class ColumnStore:
    """Request-indexed NaN-sparse columnar storage for per-record metrics.

    Uses session_num (credit issuance index) as the canonical array index.
    Pre-filled with NaN/None; records write to their slot on arrival in any order.
    """

    __slots__ = (
        "_capacity",
        "_count",
        "_numeric",
        "_string",
        "_ragged",
        "_sums",
        "_counts",
        "_metadata_numeric",
        "_metadata_string",
        "start_ns",
        "end_ns",
        "generation_start_ns",
    )

    def __init__(self, initial_capacity: int = 1024) -> None:
        self._capacity = initial_capacity
        self._count = 0
        self._numeric: dict[str, NDArray[np.float64]] = {}
        self._string: dict[str, list[str | None]] = {}
        self._ragged: dict[str, RaggedSeries] = {}
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        # Metadata columns — separate from metric columns so _compute_results()
        # doesn't pick them up. Numeric metadata uses NaN-filled float64 arrays;
        # string metadata uses None-filled lists (same layout as _string).
        self._metadata_numeric: dict[str, NDArray[np.float64]] = {}
        self._metadata_string: dict[str, list[str | None]] = {}
        # Timestamp columns — float64 for NaN support
        self.start_ns = np.full(initial_capacity, np.nan, dtype=np.float64)
        self.end_ns = np.full(initial_capacity, np.nan, dtype=np.float64)
        self.generation_start_ns = np.full(initial_capacity, np.nan, dtype=np.float64)

    # --- Public read API ---

    @property
    def count(self) -> int:
        """Number of records written (max session_num + 1)."""
        return self._count

    def numeric(self, tag: str) -> NDArray[np.float64]:
        """Return the float64 column for `tag`, sliced to count. NaN where missing."""
        col = self._numeric.get(tag)
        if col is None:
            return np.full(self._count, np.nan, dtype=np.float64)
        return col[: self._count]

    def numeric_tags(self) -> list[str]:
        """Return all numeric column tags."""
        return list(self._numeric.keys())

    def string(self, tag: str) -> list[str | None]:
        """Return the string column for `tag`, sliced to count. None where missing."""
        col = self._string.get(tag)
        if col is None:
            return [None] * self._count
        return col[: self._count]

    def ragged(self, tag: str) -> RaggedSeries:
        """Return the RaggedSeries for a list-valued metric."""
        return self._ragged[tag]

    def ragged_tags(self) -> list[str]:
        """Return all ragged column tags."""
        return list(self._ragged.keys())

    def numeric_sum(self, tag: str) -> float:
        """Return the running sum for a numeric column (O(1))."""
        return self._sums.get(tag, 0.0)

    def numeric_count(self, tag: str) -> int:
        """Return the count of values ingested for a numeric column (O(1))."""
        return self._counts.get(tag, 0)

    def metadata_numeric(self, tag: str) -> NDArray[np.float64]:
        """Return the metadata float64 column for `tag`, sliced to count. NaN where missing."""
        col = self._metadata_numeric.get(tag)
        if col is None:
            return np.full(self._count, np.nan, dtype=np.float64)
        return col[: self._count]

    def metadata_string(self, tag: str) -> list[str | None]:
        """Return the metadata string column for `tag`, sliced to count. None where missing."""
        col = self._metadata_string.get(tag)
        if col is None:
            return [None] * self._count
        return col[: self._count]

    # --- Write API (called from MetricsAccumulator.process_record) ---

    def ingest(
        self,
        idx: int,
        record_metrics: dict[str, Any],
        start_ns: float,
        end_ns: float,
        generation_start_ns: float | None,
    ) -> None:
        """Write a record's data to slot `idx` (= session_num).

        Grows capacity if idx >= _capacity. Dispatches metric values to the
        correct column type (numeric, string, ragged) based on Python type.
        """
        if idx >= self._capacity:
            self._grow(idx)

        if idx >= self._count:
            self._count = idx + 1

        self.start_ns[idx] = start_ns
        self.end_ns[idx] = end_ns
        if generation_start_ns is not None:
            self.generation_start_ns[idx] = generation_start_ns

        for tag, value in record_metrics.items():
            if isinstance(value, list):
                ragged = self._ensure_ragged_column(tag)
                ragged.extend(idx, value)
            elif isinstance(value, str):
                col = self._ensure_string_column(tag)
                col[idx] = value
            elif isinstance(value, int | float):
                col = self._ensure_numeric_column(tag)
                fval = float(value)
                col[idx] = fval
                self._sums[tag] += fval
                self._counts[tag] += 1

    def ingest_metadata(
        self,
        idx: int,
        metadata_numeric: dict[str, float | None],
        metadata_string: dict[str, str | None],
    ) -> None:
        """Write per-record metadata to slot `idx`.

        Metadata columns are kept separate from metric columns so that
        _compute_results() does not treat them as metrics.
        """
        if idx >= self._capacity:
            self._grow(idx)

        for tag, value in metadata_numeric.items():
            if value is not None:
                col = self._ensure_metadata_numeric_column(tag)
                col[idx] = float(value)

        for tag, value in metadata_string.items():
            col = self._ensure_metadata_string_column(tag)
            col[idx] = value

    # --- Internal ---

    def _grow(self, min_idx: int) -> None:
        """Double capacity until min_idx fits. New slots filled with NaN/None."""
        new_cap = self._capacity
        while new_cap <= min_idx:
            new_cap *= 2

        # Grow timestamp columns
        for attr in ("start_ns", "end_ns", "generation_start_ns"):
            old = getattr(self, attr)
            new = np.full(new_cap, np.nan, dtype=np.float64)
            new[: self._capacity] = old[: self._capacity]
            setattr(self, attr, new)

        # Grow numeric columns
        for tag, old in self._numeric.items():
            new = np.full(new_cap, np.nan, dtype=np.float64)
            new[: self._capacity] = old[: self._capacity]
            self._numeric[tag] = new

        # Grow string columns
        for tag, old in self._string.items():
            old.extend([None] * (new_cap - self._capacity))
            self._string[tag] = old

        # Grow metadata numeric columns
        for tag, old in self._metadata_numeric.items():
            new = np.full(new_cap, np.nan, dtype=np.float64)
            new[: self._capacity] = old[: self._capacity]
            self._metadata_numeric[tag] = new

        # Grow metadata string columns
        for tag, old in self._metadata_string.items():
            old.extend([None] * (new_cap - self._capacity))
            self._metadata_string[tag] = old

        self._capacity = new_cap

    def _ensure_numeric_column(self, tag: str) -> NDArray[np.float64]:
        """Lazily create a NaN-filled float64 column for `tag`."""
        col = self._numeric.get(tag)
        if col is None:
            col = np.full(self._capacity, np.nan, dtype=np.float64)
            self._numeric[tag] = col
            self._sums[tag] = 0.0
            self._counts[tag] = 0
        return col

    def _ensure_string_column(self, tag: str) -> list[str | None]:
        """Lazily create a None-filled string column for `tag`."""
        col = self._string.get(tag)
        if col is None:
            col: list[str | None] = [None] * self._capacity
            self._string[tag] = col
        return col

    def _ensure_ragged_column(self, tag: str) -> RaggedSeries:
        """Lazily create a RaggedSeries for `tag`."""
        ragged = self._ragged.get(tag)
        if ragged is None:
            ragged = RaggedSeries()
            self._ragged[tag] = ragged
        return ragged

    def _ensure_metadata_numeric_column(self, tag: str) -> NDArray[np.float64]:
        """Lazily create a NaN-filled float64 metadata column for `tag`."""
        col = self._metadata_numeric.get(tag)
        if col is None:
            col = np.full(self._capacity, np.nan, dtype=np.float64)
            self._metadata_numeric[tag] = col
        return col

    def _ensure_metadata_string_column(self, tag: str) -> list[str | None]:
        """Lazily create a None-filled metadata string column for `tag`."""
        col = self._metadata_string.get(tag)
        if col is None:
            col: list[str | None] = [None] * self._capacity
            self._metadata_string[tag] = col
        return col
