# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from aiperf.common.models import ProfileResults
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults


@dataclass
class ExporterConfig:
    """Configuration for the exporter."""

    results: ProfileResults | None
    config: object
    telemetry_results: TelemetryExportData | None
    server_metrics_results: ServerMetricsResults | None = None

    @property
    def user_config(self) -> object:
        return self.config

    @property
    def service_config(self) -> object:
        return self.config


@dataclass(slots=True)
class FileExportInfo:
    """Information about a file export."""

    export_type: str
    file_path: Path
