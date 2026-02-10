# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Console exporter for steady-state windowed metrics."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.post_processors.steady_state_analyzer import SteadyStateSummary


class ConsoleSteadyStateExporter(AIPerfLoggerMixin):
    """Console exporter that renders steady-state windowed metrics as a Rich table."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        if exporter_config.steady_state_results is None:
            raise ConsoleExporterDisabled("No steady-state results available")
        self._summary: SteadyStateSummary = exporter_config.steady_state_results
        self._exporter_config = exporter_config

    async def export(self, console: Console) -> None:
        if not self._summary.results:
            return

        meta = self._summary.window_metadata

        conc = self._summary.effective_concurrency

        # Render info lines about the window
        info = Table.grid(padding=(0, 1))
        info.add_row(
            f"[bold]Window:[/bold] {meta.detection_method}",
            f"[bold]Requests:[/bold] {meta.steady_state_requests}/{meta.total_requests}",
            f"[bold]Duration:[/bold] {meta.steady_state_duration_ns:,.0f} ns",
        )
        info.add_row(
            f"[bold]Concurrency:[/bold] avg={conc.avg:.1f}",
            f"p50={conc.p50:.1f} p90={conc.p90:.1f}",
            f"min={conc.min:.0f} max={conc.max:.0f}",
        )

        # Reuse ConsoleMetricsExporter's table rendering
        metrics_exporter = ConsoleMetricsExporter(exporter_config=self._exporter_config)
        table = metrics_exporter.get_renderable(self._summary.results.values(), console)
        # Override the title
        if isinstance(table, Table):
            table.title = "NVIDIA AIPerf | Steady-State Metrics"

        console.print("\n")
        console.print(info)
        console.print(table)
        console.file.flush()
