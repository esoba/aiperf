# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aiperf.analysis.energy_analyzer import EnergyEfficiencySummary
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.models import ProfileResults
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.post_processors.steady_state_analyzer import SteadyStateSummary


@dataclass(slots=True)
class ExporterConfig:
    """Configuration for the exporter."""

    results: ProfileResults | None
    user_config: UserConfig
    service_config: ServiceConfig | None
    telemetry_results: TelemetryExportData | None
    server_metrics_results: ServerMetricsResults | None = None
    steady_state_results: SteadyStateSummary | None = None
    energy_efficiency_results: EnergyEfficiencySummary | None = None


@dataclass(slots=True)
class FileExportInfo:
    """Information about a file export."""

    export_type: str
    file_path: Path
