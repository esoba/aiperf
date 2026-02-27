# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""System-wide resource monitoring collector.

Collects CPU utilization, memory usage, and per-process GPU VRAM using psutil
and optionally pynvml (via NvmlHandleManager). Produces ServerMetricsRecord
objects compatible with the existing server metrics pipeline.
"""

import asyncio
import time

import psutil

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ErrorDetails
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.common.nvml_handle_manager import NvmlHandleManager
from aiperf.server_metrics.protocols import (
    SYSTEM_METRICS_SOURCE_IDENTIFIER,
    TServerMetricsErrorCallback,
    TServerMetricsRecordCallback,
)

__all__ = ["SystemMetricsCollector"]


class SystemMetricsCollector(AIPerfLifecycleMixin):
    """Collects system-wide resource metrics using psutil and optionally pynvml.

    Provides an alternative to Prometheus scraping for environments where the
    inference server does not expose a Prometheus endpoint, or when system-level
    metrics (CPU, memory, per-process VRAM) are needed.

    Produces standard ServerMetricsRecord objects with GAUGE-type MetricFamily
    entries, so the entire downstream pipeline (accumulator, exporters, percentile
    computation) works unchanged.

    Metrics collected:
        - system_cpu_utilization_percent: System-wide CPU utilization
        - system_memory_used_bytes: Physical memory used
        - system_memory_total_bytes: Total physical memory
        - system_memory_available_bytes: Available physical memory
        - gpu_process_memory_used_bytes: Per-process GPU VRAM (if pynvml available)

    Args:
        collection_interval: Interval in seconds between metric collections
        record_callback: Optional async callback to receive collected records
        error_callback: Optional async callback to receive collection errors
        collector_id: Unique identifier for this collector instance
    """

    def __init__(
        self,
        collection_interval: float = Environment.SERVER_METRICS.COLLECTION_INTERVAL,
        record_callback: TServerMetricsRecordCallback | None = None,
        error_callback: TServerMetricsErrorCallback | None = None,
        collector_id: str = "system_metrics_collector",
    ) -> None:
        super().__init__(id=collector_id)
        self._collection_interval = collection_interval
        self._record_callback = record_callback
        self._error_callback = error_callback
        self._nvml = NvmlHandleManager()

    @property
    def endpoint_url(self) -> str:
        """Get the source identifier for this collector."""
        return SYSTEM_METRICS_SOURCE_IDENTIFIER

    @property
    def collection_interval(self) -> float:
        """Get the collection interval in seconds."""
        return self._collection_interval

    async def is_url_reachable(self) -> bool:
        """Check if system metrics collection is available.

        Always returns True since psutil works on all supported platforms.
        """
        return True

    @on_init
    async def _initialize_system_monitoring(self) -> None:
        """Initialize psutil and optionally pynvml for GPU process monitoring."""
        psutil.cpu_percent(interval=None)

        try:
            await asyncio.to_thread(self._nvml.initialize)
            self.info(
                f"pynvml initialized with {self._nvml.device_count} GPU(s) "
                "for per-process memory monitoring"
            )
        except Exception as e:
            self.info(f"pynvml not available for per-process GPU memory: {e!r}")

    @on_stop
    async def _shutdown_system_monitoring(self) -> None:
        """Clean up pynvml resources."""
        if self._nvml.initialized:
            await asyncio.to_thread(self._nvml.shutdown)
            self.debug("pynvml shutdown complete")

    @background_task(immediate=True, interval=lambda self: self.collection_interval)
    async def _collect_metrics_loop(self) -> None:
        """Background task for collecting metrics at regular intervals."""
        await self.collect_and_process_metrics()

    async def collect_and_process_metrics(self) -> None:
        """Collect system metrics and send via callback."""
        try:
            record = await asyncio.to_thread(self._collect_system_metrics)
            if record and self._record_callback:
                await self._record_callback([record], self.id)
        except Exception as e:
            if self._error_callback:
                try:
                    await self._error_callback(ErrorDetails.from_exception(e), self.id)
                except Exception as callback_error:
                    self.error(f"Failed to send error via callback: {callback_error}")
            else:
                self.error(f"System metrics collection error: {e}")

    def _collect_system_metrics(self) -> ServerMetricsRecord:
        """Collect all system metrics and build a ServerMetricsRecord.

        Runs in a thread to avoid blocking the event loop.
        """
        timestamp_ns = time.time_ns()
        metrics: dict[str, MetricFamily] = {}

        cpu_percent = psutil.cpu_percent(interval=None)
        metrics["system_cpu_utilization_percent"] = MetricFamily(
            type=PrometheusMetricType.GAUGE,
            description="System-wide CPU utilization percentage",
            samples=[MetricSample(value=cpu_percent)],
        )

        mem = psutil.virtual_memory()
        metrics["system_memory_used_bytes"] = MetricFamily(
            type=PrometheusMetricType.GAUGE,
            description="Physical memory currently in use (bytes)",
            samples=[MetricSample(value=float(mem.used))],
        )
        metrics["system_memory_total_bytes"] = MetricFamily(
            type=PrometheusMetricType.GAUGE,
            description="Total physical memory (bytes)",
            samples=[MetricSample(value=float(mem.total))],
        )
        metrics["system_memory_available_bytes"] = MetricFamily(
            type=PrometheusMetricType.GAUGE,
            description="Available physical memory (bytes)",
            samples=[MetricSample(value=float(mem.available))],
        )

        if self._nvml.available:
            gpu_samples = self._collect_gpu_process_memory()
            if gpu_samples:
                metrics["gpu_process_memory_used_bytes"] = MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="GPU memory used per process (bytes)",
                    samples=gpu_samples,
                )

        return ServerMetricsRecord(
            endpoint_url=SYSTEM_METRICS_SOURCE_IDENTIFIER,
            timestamp_ns=timestamp_ns,
            metrics=metrics,
        )

    def _collect_gpu_process_memory(self) -> list[MetricSample]:
        """Collect per-process GPU memory usage across all GPUs."""
        samples: list[MetricSample] = []
        nvml = self._nvml

        with nvml.lock:
            if not nvml.initialized:
                return samples

            pynvml = nvml.pynvml
            for gpu_idx, handle in zip(nvml.handle_indices, nvml.handles, strict=True):
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                except pynvml.NVMLError:
                    continue
                for proc in procs:
                    try:
                        used_mem = proc.usedGpuMemory
                        if used_mem is None:
                            continue
                        process_name = self._get_process_name(proc.pid)
                        samples.append(
                            MetricSample(
                                labels={
                                    "pid": str(proc.pid),
                                    "process_name": process_name,
                                    "gpu_index": str(gpu_idx),
                                },
                                value=float(used_mem),
                            )
                        )
                    except Exception:
                        continue

        return samples

    @staticmethod
    def _get_process_name(pid: int) -> str:
        """Get a descriptive process name, with Triton-aware identification."""
        try:
            proc = psutil.Process(pid)
            cmdline = proc.cmdline()
            cmdline_str = " ".join(cmdline)
            if "triton_python_backend_stub" in cmdline_str:
                return "triton_python_backend_stub"
            if "tritonserver" in cmdline_str:
                return "tritonserver"
            return proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return f"pid:{pid}"
