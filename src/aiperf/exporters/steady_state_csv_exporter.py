# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSV exporter for steady-state windowed metrics."""

from __future__ import annotations

import csv
import io
import numbers

from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.constants import STAT_KEYS
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter
from aiperf.post_processors.steady_state_analyzer import SteadyStateSummary


class SteadyStateCsvExporter(MetricsBaseExporter):
    """Exports steady-state windowed metrics to a CSV file."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        if exporter_config.steady_state_results is None:
            raise DataExporterDisabled("No steady-state results available")
        self._summary: SteadyStateSummary = exporter_config.steady_state_results
        self._file_path = (
            exporter_config.user_config.output.artifact_directory
            / OutputDefaults.PROFILE_EXPORT_AIPERF_STEADY_STATE_CSV_FILE
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Steady-State CSV Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Window metadata header
        meta = self._summary.window_metadata
        writer.writerow(["# Steady-State Window Metadata"])
        writer.writerow(["detection_method", meta.detection_method])
        writer.writerow(["ramp_up_end_ns", f"{meta.ramp_up_end_ns:.0f}"])
        writer.writerow(["ramp_down_start_ns", f"{meta.ramp_down_start_ns:.0f}"])
        writer.writerow(
            ["steady_state_duration_ns", f"{meta.steady_state_duration_ns:.0f}"]
        )
        writer.writerow(["total_requests", meta.total_requests])
        writer.writerow(["steady_state_requests", meta.steady_state_requests])
        writer.writerow(["fraction_retained", f"{meta.fraction_retained:.4f}"])
        writer.writerow(
            [
                "trend_correlation",
                f"{meta.trend_correlation:.4f}"
                if meta.trend_correlation is not None
                else "",
            ]
        )
        writer.writerow(
            [
                "trend_p_value",
                f"{meta.trend_p_value:.4f}" if meta.trend_p_value is not None else "",
            ]
        )
        writer.writerow(["stationarity_warning", meta.stationarity_warning])
        writer.writerow(
            ["variance_inflation_factor", f"{meta.variance_inflation_factor:.4f}"]
        )
        writer.writerow(["effective_p99_sample_size", meta.effective_p99_sample_size])
        writer.writerow(["sample_size_warning", meta.sample_size_warning])
        if meta.bootstrap_n_iterations is not None:
            writer.writerow(["bootstrap_n_iterations", meta.bootstrap_n_iterations])
            writer.writerow(
                [
                    "bootstrap_ci_ramp_up_ns",
                    meta.bootstrap_ci_ramp_up_ns,
                ]
            )
            writer.writerow(
                [
                    "bootstrap_ci_ramp_down_ns",
                    meta.bootstrap_ci_ramp_down_ns,
                ]
            )
            writer.writerow(
                [
                    "bootstrap_ci_mean_latency",
                    meta.bootstrap_ci_mean_latency,
                ]
            )
            writer.writerow(
                [
                    "bootstrap_ci_p99_latency",
                    meta.bootstrap_ci_p99_latency,
                ]
            )
        writer.writerow([])

        # Metrics table
        prepared = self._prepare_metrics(self._summary.results.values())
        conc = self._summary.effective_concurrency
        prepared[conc.tag] = conc
        tput = self._summary.effective_throughput
        prepared[tput.tag] = tput
        if not prepared:
            return buf.getvalue()

        # Separate request metrics (with percentiles) from system metrics
        request_metrics: list[MetricResult] = []
        system_metrics: list[MetricResult] = []
        for metric in prepared.values():
            if metric.p50 is not None:
                request_metrics.append(metric)
            else:
                system_metrics.append(metric)

        if request_metrics:
            header = ["Metric"] + list(STAT_KEYS)
            writer.writerow(header)
            for metric in request_metrics:
                row = [f"{metric.header} ({metric.unit})"]
                for key in STAT_KEYS:
                    val = getattr(metric, key, None)
                    if val is None:
                        row.append("")
                    elif isinstance(val, numbers.Integral):
                        row.append(str(val))
                    elif isinstance(val, numbers.Real):
                        row.append(f"{val:.2f}")
                    else:
                        row.append(str(val))
                writer.writerow(row)

        if system_metrics:
            writer.writerow([])
            writer.writerow(["Metric", "Value"])
            for metric in system_metrics:
                val = metric.avg
                if isinstance(val, numbers.Integral):
                    writer.writerow([f"{metric.header} ({metric.unit})", str(val)])
                elif isinstance(val, numbers.Real):
                    writer.writerow([f"{metric.header} ({metric.unit})", f"{val:.2f}"])
                else:
                    writer.writerow([f"{metric.header} ({metric.unit})", str(val)])

        return buf.getvalue()
