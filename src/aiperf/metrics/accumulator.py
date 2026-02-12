# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numpy-backed metrics accumulator with columnar storage and dynamic timeslicing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from aiperf.analysis.sweep import (
    SweepCurves,
    concurrency_sweep,
    prefill_throughput_sweep,
    throughput_sweep,
    throughput_sweep_icl,
)
from aiperf.common.config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import (
    AggregationKind,
    MetricType,
    MetricValueTypeT,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models import MetricResult
from aiperf.common.types import MetricTagT, TimeSliceT, TimesliceWindow
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.column_store import ColumnStore
from aiperf.metrics.metric_dicts import MetricResultsDict, metric_result_from_array
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

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
    timeslice_windows: dict[TimeSliceT, TimesliceWindow] | None = field(default=None)

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
        meta = record.metadata

        # Compute generation_start_ns from wall-clock start + TTFT duration
        ttft_ns = record.metrics.get("time_to_first_token")
        gen_start = (
            float(meta.request_start_ns + int(ttft_ns)) if ttft_ns is not None else None
        )

        self._column_store.ingest(
            idx=idx,
            record_metrics=record.metrics,
            start_ns=float(meta.request_start_ns),
            end_ns=float(meta.request_end_ns),
            generation_start_ns=gen_start,
        )

        # Store per-record metadata in ColumnStore (separate from metrics)
        self._column_store.ingest_metadata(
            idx=idx,
            metadata_numeric={
                "credit_issued_ns": meta.credit_issued_ns,
                "request_ack_ns": meta.request_ack_ns,
                "cancellation_time_ns": meta.cancellation_time_ns,
                "turn_index": meta.turn_index,
                "was_cancelled": float(meta.was_cancelled),
                "has_error": float(record.error is not None),
            },
            metadata_string={
                "worker_id": meta.worker_id,
                "record_processor_id": meta.record_processor_id,
                "x_request_id": meta.x_request_id,
                "x_correlation_id": meta.x_correlation_id,
                "conversation_id": meta.conversation_id,
                "benchmark_phase": str(meta.benchmark_phase),
            },
        )

    def query_time_range(self, start_ns: int, end_ns: int) -> NDArray[np.bool_]:
        """Return a boolean mask where True marks records in [start_ns, end_ns)."""
        n = self._column_store.count
        if n == 0:
            return np.array([], dtype=bool)
        ts = self._column_store.start_ns[:n]
        return ~np.isnan(ts) & (ts >= start_ns) & (ts < end_ns)

    @property
    def record_count(self) -> int:
        """Number of records ingested so far."""
        n = self._column_store.count
        if n == 0:
            return 0
        return int(np.count_nonzero(~np.isnan(self._column_store.start_ns[:n])))

    def _aggregate_values(self, tag: MetricTagT, values: np.ndarray) -> float:
        """Apply the tag's aggregation function to an array of values."""
        kind = self._aggregation_kinds.get(tag, AggregationKind.SUM)
        return _AGGREGATE_FUNCS[kind](values)

    def _compute_results(
        self,
        mask: NDArray[np.bool_] | None = None,
        *,
        window_start_ns: int | None = None,
        window_end_ns: int | None = None,
    ) -> dict[MetricTagT, MetricResult]:
        """Compute MetricResults directly from ColumnStore columns.

        Three phases in one method:
        1. Build scalar dict for derived computation + stash record arrays
        2. Compute derived metrics
        3. Build MetricResults from scalars and record arrays
        """
        store = self._column_store

        # Phase 1 — collect scalars for derived metrics + stash record arrays
        scalar_dict: MetricResultsDict = MetricResultsDict()
        scalar_dict.window_start_ns = window_start_ns
        scalar_dict.window_end_ns = window_end_ns
        record_arrays: dict[MetricTagT, tuple[NDArray[np.float64], float]] = {}

        full_dataset = mask is None

        for tag in store.numeric_tags():
            if full_dataset:
                col = store.numeric(tag)
                clean = col[~np.isnan(col)]
            else:
                values = store.numeric(tag)[mask]
                clean = values[~np.isnan(values)]
            if len(clean) == 0:
                continue

            metric_type = self._tags_to_types.get(tag)
            if metric_type == MetricType.RECORD:
                # O(1) running sum for the full dataset; np.sum for windowed
                s = store.numeric_sum(tag) if full_dataset else float(np.sum(clean))
                scalar_dict[tag] = s
                record_arrays[tag] = (clean, s)
            elif metric_type == MetricType.AGGREGATE:
                scalar_dict[tag] = self._aggregate_values(tag, clean)

        for tag in store.ragged_tags():
            ragged = store.ragged(tag)
            filtered = (
                ragged.values if full_dataset else ragged.get_values_for_mask(mask)
            )
            if len(filtered) == 0:
                continue
            s = float(np.sum(filtered))
            scalar_dict[tag] = s
            record_arrays[tag] = (filtered, s)

        # Phase 2 — compute derived metrics
        for tag, derive_func in self._derive_funcs.items():
            try:
                scalar_dict[tag] = derive_func(scalar_dict)
            except NoMetricValue as e:
                self.debug(f"No metric value for derived metric '{tag}': {e!r}")
            except Exception as e:
                self.warning(f"Error deriving metric '{tag}': {e!r}")

        # Phase 3 — build MetricResults
        output: dict[MetricTagT, MetricResult] = {}
        for tag, value in scalar_dict.items():
            mc = self._metric_classes.get(tag)
            if mc is None:
                continue
            if tag in record_arrays:
                arr, arr_sum = record_arrays[tag]
                output[tag] = metric_result_from_array(
                    tag, mc.header, str(mc.unit), arr, arr_sum
                )
            elif isinstance(value, int | float):
                output[tag] = MetricResult(
                    tag=tag,
                    header=mc.header,
                    unit=str(mc.unit),
                    avg=value,
                    count=1,
                )
        return output

    def compute_results_for_mask(
        self,
        mask: NDArray[np.bool_],
        *,
        window_start_ns: int | None = None,
        window_end_ns: int | None = None,
    ) -> dict[MetricTagT, MetricResult]:
        """Build, derive, and convert metric results for an arbitrary boolean mask.

        Public interface for analyzers (e.g. SteadyStateAnalyzer) that
        need windowed metric computation without accessing private methods.
        """
        return self._compute_results(
            mask, window_start_ns=window_start_ns, window_end_ns=window_end_ns
        )

    async def summarize(self, ctx: SummaryContext | None = None) -> MetricsSummary:
        """Compute and return aggregated metric results.

        If slice_duration is configured, also computes per-timeslice results
        by partitioning the data into time windows.
        """
        overall_results = self._compute_results()

        timeslices: dict[TimeSliceT, dict[MetricTagT, MetricResult]] | None = None
        timeslice_windows: dict[TimeSliceT, TimesliceWindow] | None = None

        if self._column_store.count > 0:
            # Compute sweeps once for both overall and timeslice injection
            sweeps = self._compute_sweeps()
            self._inject_sweep_metrics(overall_results, sweeps)

            if self._slice_duration_ns is not None:
                timeslices, timeslice_windows = self._compute_timeslices(sweeps)

        self.debug(lambda: f"Summarized {len(overall_results)} metric results")
        return MetricsSummary(
            results=overall_results,
            timeslices=timeslices,
            timeslice_windows=timeslice_windows,
        )

    async def export_results(self, ctx: ExportContext) -> MetricsSummary:
        """Export final metrics results. Delegates to summarize()."""
        return await self.summarize()

    def _compute_sweeps(self) -> SweepCurves:
        """Compute concurrency, throughput, and prefill throughput sweep curves."""
        store = self._column_store
        n = store.count
        start_ns = store.start_ns[:n]
        end_ns = store.end_ns[:n]
        generation_start_ns = store.generation_start_ns[:n]

        sorted_c_ts, concurrency = concurrency_sweep(start_ns, end_ns)

        # Prefer ICL-aware throughput when SSE chunk timing is available
        output_tokens = store.numeric("output_tokens")
        sorted_t_ts, throughput = self._icl_aware_throughput(
            store, generation_start_ns, end_ns, output_tokens
        )

        input_tokens = store.numeric("input_sequence_length")
        sorted_p_ts, prefill_tput = prefill_throughput_sweep(
            start_ns, generation_start_ns, input_tokens
        )

        return SweepCurves(
            concurrency_ts=sorted_c_ts,
            concurrency=concurrency,
            throughput_ts=sorted_t_ts,
            throughput=throughput,
            prefill_throughput_ts=sorted_p_ts,
            prefill_throughput=prefill_tput,
        )

    def _inject_sweep_metrics(
        self,
        results: dict[MetricTagT, MetricResult],
        sweeps: SweepCurves,
    ) -> None:
        """Inject time-weighted sweep metrics into results."""
        window_start = (
            float(sweeps.concurrency_ts[0]) if len(sweeps.concurrency_ts) > 0 else 0.0
        )
        window_end = (
            float(sweeps.concurrency_ts[-1]) if len(sweeps.concurrency_ts) > 0 else 0.0
        )
        results.update(sweeps.compute_metrics(window_start, window_end))

    def _compute_timeslices(
        self,
        sweeps: SweepCurves,
    ) -> tuple[
        dict[TimeSliceT, dict[MetricTagT, MetricResult]],
        dict[TimeSliceT, TimesliceWindow],
    ]:
        """Compute per-timeslice results by partitioning the time range.

        Sweeps are pre-computed once in ``summarize()`` and windowed per
        timeslice via ``compute_time_weighted_stats`` — O(T log M) total.

        Returns:
            Tuple of (timeslice_results, timeslice_windows).
        """
        assert self._slice_duration_ns is not None

        store = self._column_store
        n = store.count
        start_ns = store.start_ns[:n]
        filled = ~np.isnan(start_ns)
        filled_ts = start_ns[filled]

        if len(filled_ts) == 0:
            return {}, {}

        min_ts = float(np.nanmin(filled_ts))
        max_ts = float(np.nanmax(filled_ts))

        # Build slice edges — compute n_slices first to avoid np.arange stop-exclusion issues
        n_slices = int((max_ts - min_ts) / self._slice_duration_ns) + 1
        edges = min_ts + np.arange(n_slices + 1) * self._slice_duration_ns

        # Assign each record to a bin — O(n) total via digitize
        bins = np.digitize(filled_ts, edges) - 1

        timeslice_results: dict[TimeSliceT, dict[MetricTagT, MetricResult]] = {}
        timeslice_windows: dict[TimeSliceT, TimesliceWindow] = {}
        filled_indices = np.where(filled)[0]

        for bin_idx in range(len(edges) - 1):
            bin_mask_local = bins == bin_idx
            if not bin_mask_local.any():
                continue

            # Expand local mask to full-array mask
            full_mask = np.zeros(n, dtype=bool)
            full_mask[filled_indices[bin_mask_local]] = True

            window_start = float(edges[bin_idx])
            window_end = float(edges[bin_idx + 1])

            results = self._compute_results(
                full_mask,
                window_start_ns=int(edges[bin_idx]),
                window_end_ns=int(edges[bin_idx + 1]),
            )
            if len(results) == 0:
                continue

            results.update(sweeps.compute_metrics(window_start, window_end))

            timeslice_results[bin_idx] = results
            timeslice_windows[bin_idx] = TimesliceWindow(
                start_ns=int(edges[bin_idx]),
                end_ns=int(edges[bin_idx + 1]),
            )

        return timeslice_results, timeslice_windows

    @staticmethod
    def _icl_aware_throughput(
        store: ColumnStore,
        generation_start_ns: NDArray[np.float64],
        end_ns: NDArray[np.float64],
        output_tokens: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute throughput sweep, preferring ICL-aware when available."""
        if "inter_chunk_latency" in store.ragged_tags():
            icl = store.ragged("inter_chunk_latency")
            if len(icl.values) > 0:
                return throughput_sweep_icl(
                    generation_start_ns,
                    output_tokens,
                    icl.values,
                    icl.record_indices,
                    icl.offsets,
                )
        return throughput_sweep(generation_start_ns, end_ns, output_tokens)

    async def full_metrics(self) -> dict[MetricTagT, MetricResult]:
        """Returns the full metrics results, including derived metrics."""
        return self._compute_results()
