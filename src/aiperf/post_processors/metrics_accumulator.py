# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numpy-backed metrics accumulator replacing MetricResultsProcessor + TimesliceMetricResultsProcessor."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from aiperf.common.config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import (
    AggregationKind,
    MetricDictValueTypeT,
    MetricType,
    MetricValueTypeT,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models import MetricResult
from aiperf.common.types import MetricTagT, TimeSliceT
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.metric_dicts import MetricArray, MetricResultsDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor
from aiperf.post_processors.column_store import ColumnStore

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import ExportContext, SummaryContext

_AGGREGATE_FUNCS: dict[AggregationKind, Callable[[np.ndarray], float]] = {
    AggregationKind.SUM: lambda a: float(np.sum(a)),
    AggregationKind.MAX: lambda a: float(np.max(a)),
    AggregationKind.MIN: lambda a: float(np.min(a)),
}


@dataclass
class MetricsSummary:
    """Typed result from MetricsAccumulator.summarize().

    Unified summary replacing both the old MetricsSummary (results only) and
    TimesliceSummary (timeslices only). When timeslicing is configured, both
    fields are populated from a single accumulator.
    """

    results: dict[MetricTagT, MetricResult]
    timeslices: dict[TimeSliceT, dict[MetricTagT, MetricResult]] | None = field(
        default=None
    )

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "results": [r.to_json_result().model_dump() for r in self.results.values()],
        }
        if self.timeslices is not None:
            data["timeslices"] = {
                str(k): [r.to_json_result().model_dump() for r in v.values()]
                for k, v in self.timeslices.items()
            }
        return data

    def to_csv(self) -> list[dict[str, Any]]:
        rows = [r.model_dump(exclude={"current"}) for r in self.results.values()]
        if self.timeslices is not None:
            for ts, results in self.timeslices.items():
                for r in results.values():
                    row = r.model_dump(exclude={"current"})
                    row["timeslice"] = ts
                    rows.append(row)
        return rows


class MetricsAccumulator(BaseMetricsProcessor):
    """Numpy-backed accumulator for inference metrics.

    Replaces both MetricResultsProcessor and TimesliceMetricResultsProcessor.
    Uses session_num-indexed ColumnStore for NaN-sparse columnar storage,
    vectorized aggregation, and dynamic timeslicing at summarize time.

    RECORD metrics: per-value stats (percentiles, mean, etc.)
    AGGREGATE metrics: single scalar via AggregationKind (SUM/MAX/MIN) — no replay.
    DERIVED metrics: computed from the above at summarize time.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(user_config=user_config, **kwargs)

        # Session-indexed columnar storage
        self._column_store = ColumnStore(initial_capacity=1024)

        # Sparse record reference for query_time_range() / iter_requests() compat
        self._records: list[MetricRecordsData | None] = []
        self._record_count = 0

        # Derive functions for DERIVED metrics
        self._derive_funcs: dict[
            MetricTagT, Callable[[MetricResultsDict], MetricValueTypeT]
        ] = {
            metric.tag: metric.derive_value  # type: ignore
            for metric in self._setup_metrics(MetricType.DERIVED)
            if metric.type == MetricType.DERIVED
        }

        # Metric type lookup
        _all_metric_classes: list[type[BaseMetric]] = MetricRegistry.all_classes()
        self._tags_to_types: dict[MetricTagT, MetricType] = {
            metric.tag: metric.type for metric in _all_metric_classes
        }

        # Aggregation kind per AGGREGATE tag — for vectorized windowed aggregation
        self._aggregation_kinds: dict[MetricTagT, AggregationKind] = {
            metric.tag: metric.aggregation_kind
            for metric in _all_metric_classes
            if metric.type == MetricType.AGGREGATE
            and hasattr(metric, "aggregation_kind")
        }

        # Metric class metadata for result creation
        self._metric_classes: dict[MetricTagT, BaseMetric] = {
            tag: MetricRegistry.get_class(tag)() for tag in MetricRegistry.all_tags()
        }

        # Timeslice config
        slice_dur = self.user_config.output.slice_duration
        self._slice_duration_ns: int | None = (
            int(slice_dur * NANOS_PER_SECOND) if slice_dur else None
        )

    @property
    def column_store(self) -> ColumnStore:
        """Read-only access to the underlying columnar store for analyzers."""
        return self._column_store

    async def process_record(self, record: MetricRecordsData) -> None:
        """Ingest a MetricRecordsData record."""
        idx = record.metadata.session_num

        # Compute generation_start_ns from wall-clock start + TTFT duration
        ttft_ns = record.metrics.get("time_to_first_token")
        gen_start = (
            float(record.metadata.request_start_ns + int(ttft_ns))
            if ttft_ns is not None
            else None
        )

        self._column_store.ingest(
            idx=idx,
            record_metrics=record.metrics,
            start_ns=float(record.metadata.request_start_ns),
            end_ns=float(record.metadata.request_end_ns),
            generation_start_ns=gen_start,
        )

        # Keep sparse record reference for query_time_range() compat
        self._ensure_records_capacity(idx)
        self._records[idx] = record
        self._record_count += 1

    def _ensure_records_capacity(self, idx: int) -> None:
        """Extend _records list to accommodate session_num idx."""
        if idx >= len(self._records):
            self._records.extend([None] * (idx + 1 - len(self._records)))

    def query_time_range(self, start_ns: int, end_ns: int) -> list[MetricRecordsData]:
        """Return records whose request_start_ns falls within [start_ns, end_ns)."""
        n = self._column_store.count
        if n == 0:
            return []
        ts = self._column_store.start_ns[:n]
        mask = ~np.isnan(ts) & (ts >= start_ns) & (ts < end_ns)
        indices = np.where(mask)[0]
        return [self._records[i] for i in indices if self._records[i] is not None]

    def iter_requests(self) -> Iterator[MetricRecordsData]:
        """Iterate over all stored records in session_num order, skipping gaps."""
        return (r for r in self._records if r is not None)

    @property
    def record_count(self) -> int:
        """Number of records ingested so far."""
        return self._record_count

    def _aggregate_values(self, tag: MetricTagT, values: np.ndarray) -> float:
        """Apply the tag's aggregation function to an array of values."""
        kind = self._aggregation_kinds.get(tag, AggregationKind.SUM)
        return _AGGREGATE_FUNCS[kind](values)

    def build_results_for_mask(self, mask: NDArray[np.bool_]) -> MetricResultsDict:
        """Build MetricResultsDict from an arbitrary boolean mask over records.

        Works with any mask: time-range, steady-state, custom filters.
        """
        results = MetricResultsDict()
        store = self._column_store

        for tag in store.numeric_tags():
            values = store.numeric(tag)[mask]
            # Drop NaN — these are records that don't have this metric
            clean = values[~np.isnan(values)]
            if len(clean) == 0:
                continue

            metric_type = self._tags_to_types.get(tag)
            if metric_type == MetricType.RECORD:
                ma = MetricArray()
                ma.extend(clean)
                results[tag] = ma
            elif metric_type == MetricType.AGGREGATE:
                results[tag] = self._aggregate_values(tag, clean)

        for tag in store.ragged_tags():
            ragged = store.ragged(tag)
            filtered = ragged.get_values_for_mask(mask)
            if len(filtered) == 0:
                continue
            ma = MetricArray()
            ma.extend(filtered)
            results[tag] = ma

        return results

    def compute_results_for_mask(
        self, mask: NDArray[np.bool_]
    ) -> dict[MetricTagT, MetricResult]:
        """Build, derive, and convert metric results for an arbitrary boolean mask.

        Public interface for analyzers (e.g. SteadyStateAnalyzer) that
        need windowed metric computation without accessing private methods.
        """
        results = self.build_results_for_mask(mask)
        self._compute_derived(results)
        return self._build_metric_results(results)

    def _build_results_dict(self) -> MetricResultsDict:
        """Build a MetricResultsDict from all ingested data."""
        n = self._column_store.count
        if n == 0:
            return MetricResultsDict()
        ts = self._column_store.start_ns[:n]
        mask = ~np.isnan(ts)
        return self.build_results_for_mask(mask)

    def _build_results_dict_for_window(
        self, start_ns: int, end_ns: int
    ) -> MetricResultsDict:
        """Build a MetricResultsDict for a specific time window."""
        n = self._column_store.count
        if n == 0:
            return MetricResultsDict()
        ts = self._column_store.start_ns[:n]
        mask = ~np.isnan(ts) & (ts >= start_ns) & (ts < end_ns)
        return self.build_results_for_mask(mask)

    def _compute_derived(self, results: MetricResultsDict) -> None:
        """Compute derived metrics in-place on a MetricResultsDict."""
        for tag, derive_func in self._derive_funcs.items():
            try:
                results[tag] = derive_func(results)
            except NoMetricValue as e:
                self.debug(f"No metric value for derived metric '{tag}': {e!r}")
            except Exception as e:
                self.warning(f"Error deriving metric '{tag}': {e!r}")

    def _create_metric_result(
        self, tag: MetricTagT, values: MetricDictValueTypeT
    ) -> MetricResult:
        """Create a MetricResult from current values of a metric."""
        metric_class = self._metric_classes.get(tag)
        if metric_class is None:
            raise ValueError(f"Unknown metric tag: {tag}")

        if isinstance(values, MetricArray):
            return values.to_result(tag, metric_class.header, str(metric_class.unit))

        if isinstance(values, int | float):
            return MetricResult(
                tag=tag,
                header=metric_class.header,
                unit=str(metric_class.unit),
                avg=values,
                count=1,
            )

        raise ValueError(f"Unexpected values type: {type(values)}")

    def _build_metric_results(
        self, results: MetricResultsDict
    ) -> dict[MetricTagT, MetricResult]:
        """Convert a MetricResultsDict to a dict of MetricResult objects keyed by tag."""
        return {
            tag: self._create_metric_result(tag, values)
            for tag, values in results.items()
        }

    async def summarize(self, ctx: SummaryContext | None = None) -> MetricsSummary:
        """Compute and return aggregated metric results.

        If slice_duration is configured, also computes per-timeslice results
        by partitioning the data into time windows.
        """
        # Build overall results
        results_dict = self._build_results_dict()
        self._compute_derived(results_dict)
        overall_results = self._build_metric_results(results_dict)

        timeslices: dict[TimeSliceT, dict[MetricTagT, MetricResult]] | None = None

        if self._slice_duration_ns is not None and self._column_store.count > 0:
            timeslices = self._compute_timeslices()

        self.debug(lambda: f"Summarized {len(overall_results)} metric results")
        return MetricsSummary(results=overall_results, timeslices=timeslices)

    async def export_results(self, ctx: ExportContext) -> MetricsSummary:
        """Export final metrics results. Delegates to summarize()."""
        return await self.summarize()

    def _compute_timeslices(
        self,
    ) -> dict[TimeSliceT, dict[MetricTagT, MetricResult]]:
        """Compute per-timeslice results by partitioning the time range."""
        assert self._slice_duration_ns is not None

        n = self._column_store.count
        ts = self._column_store.start_ns[:n]
        filled = ~np.isnan(ts)
        filled_ts = ts[filled]

        if len(filled_ts) == 0:
            return {}

        min_ts = float(np.nanmin(filled_ts))
        max_ts = float(np.nanmax(filled_ts))

        # Build slice edges — compute n_slices first to avoid np.arange stop-exclusion issues
        n_slices = int((max_ts - min_ts) / self._slice_duration_ns) + 1
        edges = min_ts + np.arange(n_slices + 1) * self._slice_duration_ns

        # Assign each record to a bin — O(n) total via digitize
        bins = np.digitize(filled_ts, edges) - 1

        timeslice_results: dict[TimeSliceT, dict[MetricTagT, MetricResult]] = {}
        filled_indices = np.where(filled)[0]

        for bin_idx in range(len(edges) - 1):
            bin_mask_local = bins == bin_idx
            if not bin_mask_local.any():
                continue

            # Expand local mask to full-array mask
            full_mask = np.zeros(n, dtype=bool)
            full_mask[filled_indices[bin_mask_local]] = True

            window_results = self.build_results_for_mask(full_mask)
            if len(window_results) > 0:
                self._compute_derived(window_results)
                timeslice_results[bin_idx] = self._build_metric_results(window_results)

        return timeslice_results

    async def full_metrics(self) -> MetricResultsDict:
        """Returns the full metrics dict, including derived metrics."""
        results = self._build_results_dict()
        self._compute_derived(results)
        return results
