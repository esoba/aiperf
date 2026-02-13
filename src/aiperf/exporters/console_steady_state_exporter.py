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
            f"[bold]Requests:[/bold] {meta.steady_state_requests}/{meta.total_requests}"
            f" ({meta.fraction_retained:.1%})",
            f"[bold]Duration:[/bold] {meta.steady_state_duration_ns:,.0f} ns",
        )
        info.add_row(
            f"[bold]Concurrency:[/bold] avg={conc.avg:.1f}",
            f"p50={conc.p50:.1f} p90={conc.p90:.1f}",
            f"min={conc.min:.0f} max={conc.max:.0f}",
        )
        tput = self._summary.effective_throughput
        info.add_row(
            f"[bold]Throughput:[/bold] avg={tput.avg:,.1f} {tput.unit}",
            f"p50={tput.p50:,.1f} p90={tput.p90:,.1f}",
            f"min={tput.min:,.1f} max={tput.max:,.1f}",
        )
        ptput = self._summary.effective_prefill_throughput
        info.add_row(
            f"[bold]Prefill Tput:[/bold] avg={ptput.avg:,.1f} {ptput.unit}",
            f"p50={ptput.p50:,.1f} p90={ptput.p90:,.1f}",
            f"min={ptput.min:,.1f} max={ptput.max:,.1f}",
        )
        total_tput = self._summary.effective_total_throughput
        info.add_row(
            f"[bold]Total Tput:[/bold] avg={total_tput.avg:,.1f} {total_tput.unit}",
            f"p50={total_tput.p50:,.1f} p90={total_tput.p90:,.1f}",
            f"min={total_tput.min:,.1f} max={total_tput.max:,.1f}",
        )
        gen_conc = self._summary.effective_generation_concurrency
        info.add_row(
            f"[bold]Gen Conc:[/bold] avg={gen_conc.avg:.1f}",
            f"p50={gen_conc.p50:.1f} p90={gen_conc.p90:.1f}",
            f"min={gen_conc.min:.0f} max={gen_conc.max:.0f}",
        )
        pre_conc = self._summary.effective_prefill_concurrency
        info.add_row(
            f"[bold]Prefill Conc:[/bold] avg={pre_conc.avg:.1f}",
            f"p50={pre_conc.p50:.1f} p90={pre_conc.p90:.1f}",
            f"min={pre_conc.min:.0f} max={pre_conc.max:.0f}",
        )
        tput_pu = self._summary.effective_throughput_per_user
        info.add_row(
            f"[bold]Tput/User:[/bold] avg={tput_pu.avg:,.1f} {tput_pu.unit}",
            f"p50={tput_pu.p50:,.1f} p90={tput_pu.p90:,.1f}",
            f"min={tput_pu.min:,.1f} max={tput_pu.max:,.1f}",
        )
        ptput_pu = self._summary.effective_prefill_throughput_per_user
        info.add_row(
            f"[bold]Prefill/User:[/bold] avg={ptput_pu.avg:,.1f} {ptput_pu.unit}",
            f"p50={ptput_pu.p50:,.1f} p90={ptput_pu.p90:,.1f}",
            f"min={ptput_pu.min:,.1f} max={ptput_pu.max:,.1f}",
        )
        tif = self._summary.tokens_in_flight
        info.add_row(
            f"[bold]Tokens In Flight:[/bold] avg={tif.avg:,.0f}",
            f"p50={tif.p50:,.0f} p90={tif.p90:,.0f}",
            f"min={tif.min:,.0f} max={tif.max:,.0f}",
        )

        # Stationarity status
        if meta.stationarity_warning:
            rho = meta.trend_correlation or 0.0
            p = meta.trend_p_value or 1.0
            info.add_row(
                f"[bold yellow]Status:[/bold yellow] Latency trend detected "
                f"(\u03c1={rho:.2f}, p={p:.3f})",
                "",
                "",
            )
        else:
            info.add_row("[bold green]Status:[/bold green] Stationary", "", "")

        if meta.sample_size_warning:
            info.add_row(
                f"[bold yellow]Warning:[/bold yellow] Small sample "
                f"(p99 from ~{meta.effective_p99_sample_size} observations)",
                "",
                "",
            )

        if meta.bootstrap_n_iterations is not None:
            ci_up = meta.bootstrap_ci_ramp_up_ns
            ci_down = meta.bootstrap_ci_ramp_down_ns
            ci_mean = meta.bootstrap_ci_mean_latency
            parts = [
                f"[bold]Bootstrap 95% CI[/bold] ({meta.bootstrap_n_iterations} iter)"
            ]
            if ci_up:
                parts.append(f"ramp-up: [{ci_up[0]:,.0f}, {ci_up[1]:,.0f}] ns")
            if ci_down:
                parts.append(f"ramp-down: [{ci_down[0]:,.0f}, {ci_down[1]:,.0f}] ns")
            if ci_mean:
                parts.append(f"mean: [{ci_mean[0]:,.2f}, {ci_mean[1]:,.2f}]")
            info.add_row(
                parts[0], " ".join(parts[1:3]), parts[3] if len(parts) > 3 else ""
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
