# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiperf.plugin.enums import AccumulatorType


@runtime_checkable
class AccumulatorResult(Protocol):
    """Protocol for typed results from accumulator summarize()."""

    def to_json(self) -> Any:
        """Serialize to JSON-compatible structure."""
        ...

    def to_csv(self) -> list[dict[str, Any]]:
        """Serialize to list of CSV-compatible row dicts."""
        ...


@dataclass
class SummaryContext:
    """Typed cross-processor communication context for dependency-ordered summarization.

    NOT a Pydantic model — this is never serialized over the wire. It is created
    by RecordsManager._summarize_all() and passed through the topological-sort
    pipeline so each processor can read outputs from its declared dependencies.
    """

    accumulators: dict[AccumulatorType, Any] = field(default_factory=dict)
    processor_outputs: dict[str, Any] = field(default_factory=dict)
    start_ns: int = 0
    end_ns: int = 0
    cancelled: bool = False

    def get_accumulator(self, processor_type: AccumulatorType) -> Any | None:
        """Look up an accumulator by its processor_type. Returns None if not present."""
        return self.accumulators.get(processor_type)

    def get_output(self, processor_type: str) -> Any | None:
        """Look up a previously-computed processor output. Returns None if not yet available."""
        return self.processor_outputs.get(processor_type)


@runtime_checkable
class AccumulatorProtocol(Protocol):
    """Protocol for accumulators that ingest records, support time-range queries, and produce summaries.

    Accumulators are the primary data stores in the records pipeline. Each accumulator
    owns exactly one record type and is fully self-contained — no cross-accumulator
    dependencies. Derived computations belong on SubProcessorProtocol instead.
    """

    async def process_record(self, record: Any) -> None:
        """Ingest a single record into this accumulator's internal storage."""
        ...

    def query_time_range(self, start_ns: int, end_ns: int) -> list[Any]:
        """Return records whose timestamps fall within [start_ns, end_ns).

        Uses bisect for O(log n) lookup on sorted timestamp arrays.
        """
        ...

    async def summarize(self, ctx: SummaryContext | None = None) -> AccumulatorResult:
        """Compute and return aggregated metric results.

        Args:
            ctx: Optional SummaryContext for reading dependency outputs.
                 None when called for realtime metrics (no cross-processor deps).
        """
        ...


@runtime_checkable
class SubProcessorProtocol(Protocol):
    """Protocol for processors that don't ingest records directly but derive results
    from other accumulators at summarization time.

    SubProcessors declare which accumulators they need via required_accumulators
    and which outputs they depend on via summary_dependencies. They receive
    accumulator references at construction and a SummaryContext at summarize time.
    """

    required_accumulators: ClassVar[set[AccumulatorType]]
    summary_dependencies: ClassVar[list[AccumulatorType]]

    async def summarize(self, ctx: SummaryContext) -> Any:
        """Compute derived results using data from declared accumulator dependencies."""
        ...


@runtime_checkable
class StreamExporterProtocol(Protocol):
    """Protocol for processors that stream each record to an external sink (e.g. JSONL files).

    Stream exporters have no summarization dependencies and are flushed after
    all accumulators complete.
    """

    async def process_record(self, record: Any) -> None:
        """Write a single record to the export sink."""
        ...

    async def finalize(self) -> None:
        """Flush any buffered data. Called once after all records are processed."""
        ...
