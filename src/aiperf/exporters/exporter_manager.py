# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

from rich.console import Console

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.exceptions import (
    ArtifactPublisherDisabled,
    ConsoleExporterDisabled,
    DataExporterDisabled,
)
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ProfileResults
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.protocols import (
    ArtifactPublisherProtocol,
    ConsoleExporterProtocol,
    DataExporterProtocol,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import DataExporterType, PluginType
from aiperf.post_processors.steady_state_analyzer import SteadyStateSummary


class ExporterManager(AIPerfLoggerMixin):
    """
    ExporterManager is responsible for exporting records using all
    registered data exporters.
    """

    def __init__(
        self,
        results: ProfileResults,
        user_config: UserConfig,
        service_config: ServiceConfig,
        telemetry_results: TelemetryExportData | None,
        server_metrics_results: ServerMetricsResults | None = None,
        steady_state_results: SteadyStateSummary | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._results = results
        self._user_config = user_config
        self._tasks: set[asyncio.Task] = set()
        self._service_config = service_config
        self._exporter_config = ExporterConfig(
            results=self._results,
            user_config=self._user_config,
            service_config=self._service_config,
            telemetry_results=telemetry_results,
            server_metrics_results=server_metrics_results,
            steady_state_results=steady_state_results,
        )
        self._exported_file_infos: dict[str, FileExportInfo] = {}

    def _task_done_callback(self, task: asyncio.Task) -> None:
        self.debug(lambda: f"Task done: {task}")
        if task.exception():
            self.error(f"Error exporting records: {task.exception()}")
        else:
            self.debug(f"Exported records: {task.result()}")
        self._tasks.discard(task)

    def _instantiate_data_exporters(self) -> list[DataExporterProtocol]:
        """Instantiate all enabled data exporters, collecting file infos along the way."""
        exporters: list[DataExporterProtocol] = []
        self._exported_file_infos = {}

        for exporter_entry, ExporterClass in plugins.iter_all(PluginType.DATA_EXPORTER):
            if exporter_entry.name == DataExporterType.SERVER_METRICS_PARQUET:
                # TODO: Until the exporters move to the records manager, we need to skip the
                # parquet exporter here, as it requires the server metrics accumulator to be available.
                continue

            try:
                exporter: DataExporterProtocol = ExporterClass(
                    exporter_config=self._exporter_config
                )
            except DataExporterDisabled:
                self.debug(
                    f"Data exporter {exporter_entry.name} is disabled and will not be used"
                )
                continue
            except Exception as e:
                self.error(f"Error creating data exporter: {e!r}")
                continue

            exporters.append(exporter)
            self._exported_file_infos[ExporterClass.__name__] = (
                exporter.get_export_info()
            )

        return exporters

    async def export_data(self) -> None:
        """Export data files using all registered data exporters.

        Also populates exported_file_infos so callers can read file paths
        without re-instantiating exporters.
        """
        self.info("Exporting all records")

        for exporter in self._instantiate_data_exporters():
            self.debug(f"Creating task for exporter: {exporter.__class__.__name__}")
            task = asyncio.create_task(exporter.export())
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.debug("Exporting all records completed")

    @property
    def exported_file_infos(self) -> dict[str, FileExportInfo]:
        """File infos collected during export_data() or populated on access.

        Returns dict mapping exporter class name to FileExportInfo.
        After export_data() has run, returns the cached dict. If export_data()
        hasn't been called yet, instantiates exporters to collect the infos.
        """
        if not self._exported_file_infos:
            self._instantiate_data_exporters()
        return self._exported_file_infos

    async def export_console(self, console: Console) -> None:
        self.info("Exporting console data")

        for exporter_entry, ExporterClass in plugins.iter_all(
            PluginType.CONSOLE_EXPORTER
        ):
            try:
                exporter: ConsoleExporterProtocol = ExporterClass(
                    exporter_config=self._exporter_config
                )
            except ConsoleExporterDisabled:
                self.debug(
                    f"Console exporter {exporter_entry.name} is disabled and will not be used"
                )
                continue
            except Exception as e:
                self.error(f"Error creating console exporter: {e!r}")
                continue

            self.debug(f"Creating task for exporter: {exporter_entry.name}")
            task = asyncio.create_task(exporter.export(console=console))
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.debug("Exporting console data completed")

    async def publish_artifacts(self, artifacts: list[FileExportInfo]) -> None:
        """Publish artifacts to all registered artifact publishers.

        Iterates over all ARTIFACT_PUBLISHER plugins, instantiates each, and
        runs publish() concurrently. Errors are isolated per-publisher.
        """
        self.info("Publishing artifacts to remote storage")

        if not hasattr(PluginType, "ARTIFACT_PUBLISHER"):
            self.debug("No artifact_publisher category registered, skipping")
            return

        for entry, PublisherClass in plugins.iter_all(PluginType.ARTIFACT_PUBLISHER):
            try:
                publisher: ArtifactPublisherProtocol = PublisherClass(
                    exporter_config=self._exporter_config
                )
            except ArtifactPublisherDisabled:
                self.debug(
                    f"Artifact publisher {entry.name} is disabled and will not be used"
                )
                continue
            except Exception as e:
                self.error(f"Error creating artifact publisher: {e!r}")
                continue

            self.debug(f"Creating task for artifact publisher: {entry.name}")
            task = asyncio.create_task(publisher.publish(artifacts))
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.debug("Artifact publishing completed")
