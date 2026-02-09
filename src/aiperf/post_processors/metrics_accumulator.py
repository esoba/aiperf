# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numpy-backed metrics accumulator replacing MetricResultsProcessor + TimesliceMetricResultsProcessor."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from aiperf.common.config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import (
    AggregationKind,
    MetricDictValueTypeT,
    MetricType,
    MetricValueTypeT,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.growable_array import GrowableArray
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models import MetricResult
from aiperf.common.types import MetricTagT, TimeSliceT
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.metric_dicts import MetricArray, MetricResultsDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor
from aiperf.post_processors.inference_time_series import InferenceTimeSeries

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
    Stores per-tag time series with record indices for O(log n) time queries,
    vectorized aggregation, and dynamic timeslicing at summarize time.

    RECORD metrics: per-value stats (percentiles, mean, etc.)
    AGGREGATE metrics: single scalar via AggregationKind (SUM/MAX/MIN) — no replay.
    DERIVED metrics: computed from the above at summarize time.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(user_config=user_config, **kwargs)

        # Raw record storage for query_time_range()
        self._records: list[MetricRecordsData] = []
        self._record_timestamps = GrowableArray(initial_capacity=256, dtype=np.int64)

        # Per-tag time series for RECORD and AGGREGATE metrics
        self._time_series: dict[MetricTagT, InferenceTimeSeries] = {}

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

    async def process_record(self, record: MetricRecordsData) -> None:
        """Ingest a MetricRecordsData record."""
        ts = record.metadata.request_start_ns
        record_idx = len(self._records)
        self._records.append(record)
        self._record_timestamps.append(ts)

        for tag, value in record.metrics.items():
            try:
                metric_type = self._tags_to_types.get(tag)
                if metric_type in (MetricType.RECORD, MetricType.AGGREGATE):
                    if tag not in self._time_series:
                        self._time_series[tag] = InferenceTimeSeries()
                    series = self._time_series[tag]
                    if isinstance(value, list):
                        series.extend(record_idx, value)
                    else:
                        series.append(record_idx, float(value))

                else:
                    self.warning(f"Metric '{tag}' has unexpected type: {metric_type}")
            except NoMetricValue as e:
                if self.is_trace_enabled:
                    self.trace(f"No metric value for metric '{tag}': {e!r}")
            except Exception as e:
                self.warning(f"Error processing metric '{tag}': {e!r}")

    def query_time_range(self, start_ns: int, end_ns: int) -> list[MetricRecordsData]:
        """Return records whose request_start_ns falls within [start_ns, end_ns)."""
        timestamps = self._record_timestamps.data
        if len(timestamps) == 0:
            return []
        lo = int(np.searchsorted(timestamps, start_ns, side="left"))
        hi = int(np.searchsorted(timestamps, end_ns, side="left"))
        return self._records[lo:hi]

    def iter_requests(self) -> Iterator[MetricRecordsData]:
        """Iterate over all stored records in arrival order."""
        return iter(self._records)

    @property
    def record_count(self) -> int:
        """Number of records ingested so far."""
        return len(self._records)

    def _aggregate_values(self, tag: MetricTagT, values: np.ndarray) -> float:
        """Apply the tag's aggregation function to an array of values."""
        kind = self._aggregation_kinds.get(tag, AggregationKind.SUM)
        return _AGGREGATE_FUNCS[kind](values)

    def _build_results_dict(self) -> MetricResultsDict:
        """Build a MetricResultsDict from current state for derive functions.

        RECORD metrics get a MetricArray wrapper sharing the InferenceTimeSeries values.
        AGGREGATE metrics get their vectorized scalar value.
        """
        results = MetricResultsDict()
        for tag, series in self._time_series.items():
            metric_type = self._tags_to_types.get(tag)
            if metric_type == MetricType.RECORD:
                ma = MetricArray.__new__(MetricArray)
                ma._array = series.as_metric_array()
                results[tag] = ma
            elif metric_type == MetricType.AGGREGATE and len(series) > 0:
                results[tag] = self._aggregate_values(tag, series.values)
        return results

    def _get_record_mask(self, start_ns: int, end_ns: int) -> np.ndarray:
        """Boolean mask over records for [start_ns, end_ns). O(log n) via searchsorted."""
        timestamps = self._record_timestamps.data
        n = len(timestamps)
        if n == 0:
            return np.zeros(0, dtype=bool)
        lo = int(np.searchsorted(timestamps, start_ns, side="left"))
        hi = int(np.searchsorted(timestamps, end_ns, side="left"))
        mask = np.zeros(n, dtype=bool)
        mask[lo:hi] = True
        return mask

    def _build_results_dict_for_window(
        self, start_ns: int, end_ns: int
    ) -> MetricResultsDict:
        """Build a MetricResultsDict for a specific time window.

        Both RECORD and AGGREGATE metrics are time-filtered via record mask
        and computed vectorized — no replay needed.
        """
        record_mask = self._get_record_mask(start_ns, end_ns)
        results = MetricResultsDict()

        for tag, series in self._time_series.items():
            value_mask = series.get_value_mask(record_mask)
            filtered_values = series.values[value_mask]
            if len(filtered_values) == 0:
                continue

            metric_type = self._tags_to_types.get(tag)
            if metric_type == MetricType.RECORD:
                ma = MetricArray()
                ma.extend(list(filtered_values))
                results[tag] = ma
            elif metric_type == MetricType.AGGREGATE:
                results[tag] = self._aggregate_values(tag, filtered_values)

        return results

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

        if self._slice_duration_ns is not None and len(self._records) > 0:
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

        timestamps = self._record_timestamps.data
        min_ts = int(timestamps.min())
        max_ts = int(timestamps.max())

        timeslice_results: dict[TimeSliceT, dict[MetricTagT, MetricResult]] = {}
        slice_start = min_ts
        counter = 0

        while slice_start <= max_ts:
            slice_end = slice_start + self._slice_duration_ns
            window_results = self._build_results_dict_for_window(slice_start, slice_end)

            if len(window_results) > 0:
                self._compute_derived(window_results)
                timeslice_results[counter] = self._build_metric_results(window_results)

            counter += 1
            slice_start = slice_end

        return timeslice_results

    async def full_metrics(self) -> MetricResultsDict:
        """Returns the full metrics dict, including derived metrics."""
        results = self._build_results_dict()
        self._compute_derived(results)
        return results
