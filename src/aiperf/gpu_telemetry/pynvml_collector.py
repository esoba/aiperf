# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyNVML-based GPU telemetry collector.

Collects GPU metrics directly using the pynvml Python library, providing an
alternative to DCGM HTTP endpoints for local GPU monitoring. Uses the shared
NvmlHandleManager for pynvml lifecycle and handle management.

Supports dual-output: scalar per-GPU metrics flow through the telemetry pipeline
as TelemetryRecord objects, while per-process GPU VRAM flows through the server
metrics pipeline as ServerMetricsRecord objects with labeled MetricSample entries.
"""

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

import psutil
import pynvml

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    ErrorDetails,
    GpuMetadata,
    TelemetryMetrics,
    TelemetryRecord,
)
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.common.nvml_handle_manager import NvmlHandleManager
from aiperf.gpu_telemetry.constants import (
    PYNVML_SOURCE_IDENTIFIER,
)
from aiperf.gpu_telemetry.protocols import (
    TErrorCallback,
    TRecordCallback,
    TServerMetricsRecordCallback,
)

__all__ = ["PyNVMLTelemetryCollector"]


@dataclass(frozen=True)
class ScalingFactors:
    """Unit conversion scaling factors for NVML metrics."""

    gpu_power_usage = 1e-3  # mW -> W
    energy_consumption = 1e-9  # mJ -> MJ
    gpu_memory_used = 1e-9  # bytes -> GB
    gpu_memory_total = 1e-9  # bytes -> GB
    gpu_memory_free = 1e-9  # bytes -> GB
    gpu_power_limit = 1e-3  # mW -> W
    power_violation = 1e-3  # ns -> µs


@dataclass(slots=True)
class _CollectionResult:
    """Result of a single collection cycle containing both pipeline outputs."""

    telemetry_records: list[TelemetryRecord] = field(default_factory=list)
    server_metrics_record: ServerMetricsRecord | None = None


@dataclass(slots=True)
class GpuDeviceState:
    """Per-GPU state for NVML telemetry collection.

    Args:
        handle: NVML device handle
        metadata: GPU metadata
        gpm_samples: GPM samples (prev, curr) if GPM supported, else None
    """

    handle: object
    metadata: GpuMetadata
    gpm_samples: tuple[object, object] | None = None


class PyNVMLTelemetryCollector(AIPerfLifecycleMixin):
    """Collects GPU telemetry metrics using the pynvml Python library.

    Direct collector that uses NVIDIA's pynvml library to gather GPU metrics
    locally without requiring a DCGM HTTP endpoint. Useful for environments
    where DCGM is not deployed or for simple local GPU monitoring.

    Features:
        - Direct NVML API access via pynvml
        - Automatic GPU discovery and enumeration via NvmlHandleManager
        - Same TelemetryRecord output format as DCGM collector
        - Callback-based record delivery

    Requirements:
        - pynvml package installed: `pip install nvidia-ml-py`
        - NVIDIA driver installed with NVML support

    Args:
        collection_interval: Interval in seconds between metric collections (default: from Environment)
        record_callback: Optional async callback to receive collected TelemetryRecords.
            Signature: async (records: list[TelemetryRecord], collector_id: str) -> None
        error_callback: Optional async callback to receive collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        server_metrics_record_callback: Optional async callback to receive per-process GPU VRAM
            as ServerMetricsRecords. When provided, enables per-process VRAM collection which
            flows through the server metrics pipeline.
            Signature: async (records: list[ServerMetricsRecord], collector_id: str) -> None
        collector_id: Unique identifier for this collector instance

    Raises:
        RuntimeError: If pynvml package is not installed
    """

    def __init__(
        self,
        collection_interval: float = Environment.GPU.COLLECTION_INTERVAL,
        record_callback: TRecordCallback | None = None,
        error_callback: TErrorCallback | None = None,
        server_metrics_record_callback: TServerMetricsRecordCallback | None = None,
        collector_id: str = "pynvml_collector",
    ) -> None:
        super().__init__(id=collector_id)
        self._collection_interval = collection_interval
        self._record_callback = record_callback
        self._error_callback = error_callback
        self._server_metrics_record_callback = server_metrics_record_callback

        self._gpus: list[GpuDeviceState] = []
        self._nvml = NvmlHandleManager()

    @property
    def endpoint_url(self) -> str:
        """Get the source identifier for this collector.

        Returns:
            'pynvml://localhost' to identify records from pynvml collection.
        """
        return PYNVML_SOURCE_IDENTIFIER

    @property
    def collection_interval(self) -> float:
        """Get the collection interval in seconds."""
        return self._collection_interval

    async def is_url_reachable(self) -> bool:
        """Check if NVML is available and can be initialized.

        Tests NVML availability by attempting initialization if not already done.
        This allows pre-flight checks before starting collection.

        Returns:
            True if NVML is available and can access at least one GPU.
        """
        if self._nvml.initialized:
            return len(self._gpus) > 0

        try:
            return await asyncio.to_thread(self._nvml.probe)
        except Exception:
            return False

    @on_init
    async def _initialize_nvml(self) -> None:
        """Initialize NVML and discover GPUs.

        Called automatically during initialization phase.
        Uses NvmlHandleManager for NVML lifecycle, then builds per-GPU
        state with metadata and GPM support.

        Raises:
            RuntimeError: If NVML initialization or GPU discovery fails.
        """
        self._nvml.initialize()

        self._gpus = []
        for nvml_index, handle in zip(
            self._nvml.handle_indices, self._nvml.handles, strict=True
        ):
            gpu = self._create_gpu_for_handle(nvml_index, handle)
            if gpu:
                self._gpus.append(gpu)

        gpm_count = sum(1 for gpu in self._gpus if gpu.gpm_samples)
        self.info(
            f"PyNVML initialized with {len(self._gpus)} GPU(s) "
            f"({gpm_count} with GPM support)"
        )

    def _create_gpu_for_handle(
        self, index: int, handle: object
    ) -> GpuDeviceState | None:
        """Build GPU state with metadata and GPM for a device handle."""
        try:
            uuid = pynvml.nvmlDeviceGetUUID(handle)
        except pynvml.NVMLError:
            uuid = f"GPU-unknown-{index}"

        try:
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
        except pynvml.NVMLError:
            name = "Unknown GPU"

        try:
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            pci_bus_id = pci_info.busId
            if isinstance(pci_bus_id, bytes):
                pci_bus_id = pci_bus_id.decode("utf-8")
        except pynvml.NVMLError:
            pci_bus_id = None

        gpu = GpuDeviceState(
            handle=handle,
            metadata=GpuMetadata(
                gpu_index=index,
                gpu_uuid=uuid,
                gpu_model_name=name,
                pci_bus_id=pci_bus_id,
                device=f"nvidia{index}",
                hostname="localhost",
            ),
        )

        self._init_gpm_for_device(gpu)
        return gpu

    def _init_gpm_for_device(self, gpu: GpuDeviceState) -> None:
        """Initialize GPM (GPU Performance Metrics) for efficient SM utilization."""
        try:
            if not pynvml.nvmlGpmQueryDeviceSupport(gpu.handle).isSupportedDevice:
                return
            sample1 = pynvml.nvmlGpmSampleAlloc()
            sample2 = pynvml.nvmlGpmSampleAlloc()
            pynvml.nvmlGpmSampleGet(gpu.handle, sample1)
            gpu.gpm_samples = (sample1, sample2)
            self.debug(lambda: f"GPM enabled for GPU {gpu.metadata.gpu_index}")
        except pynvml.NVMLError:
            self.debug(lambda: f"GPM not supported for GPU {gpu.metadata.gpu_index}")

    def _free_gpm_samples(self) -> None:
        """Free all allocated GPM sample buffers."""
        for gpu in self._gpus:
            if gpu.gpm_samples:
                for sample in gpu.gpm_samples:
                    with contextlib.suppress(pynvml.NVMLError):
                        pynvml.nvmlGpmSampleFree(sample)
                gpu.gpm_samples = None

    def _get_sm_utilization_gpm(self, gpu: GpuDeviceState) -> float | None:
        """Get SM utilization using GPM API (device-level, more efficient)."""
        prev_sample, curr_sample = gpu.gpm_samples  # type: ignore[misc]
        try:
            pynvml.nvmlGpmSampleGet(gpu.handle, curr_sample)
            metrics_get = pynvml.c_nvmlGpmMetricsGet_t()
            metrics_get.version = pynvml.NVML_GPM_METRICS_GET_VERSION
            metrics_get.sample1 = prev_sample
            metrics_get.sample2 = curr_sample
            metrics_get.numMetrics = 1
            metrics_get.metrics[0].metricId = pynvml.NVML_GPM_METRIC_SM_UTIL
            pynvml.nvmlGpmMetricsGet(metrics_get)
            sm_util = metrics_get.metrics[0].value
        except pynvml.NVMLError:
            sm_util = None
        gpu.gpm_samples = (curr_sample, prev_sample)  # Swap for next iteration
        return sm_util

    def _shutdown_nvml_sync(self) -> None:
        """Synchronous NVML shutdown helper.

        Frees GPM samples (collector-specific) then delegates NVML shutdown
        to NvmlHandleManager.
        """
        with self._nvml.lock:
            if not self._nvml.initialized:
                return
            self._free_gpm_samples()
            self._gpus = []

        self._nvml.shutdown()

    @on_stop
    async def _shutdown_nvml(self) -> None:
        """Shutdown NVML library.

        Called automatically during shutdown phase.
        Thread-safe - waits for any in-progress collection to complete.
        """
        await asyncio.to_thread(self._shutdown_nvml_sync)
        self.debug("PyNVML shutdown complete")

    @background_task(immediate=True, interval=lambda self: self.collection_interval)
    async def _collect_metrics_loop(self) -> None:
        """Background task for collecting metrics at regular intervals.

        Runs continuously during collector's RUNNING state, triggering a metrics
        collection every collection_interval seconds.
        """
        await self._collect_and_process_metrics()

    async def _collect_and_process_metrics(self) -> None:
        """Collect metrics from all GPUs and send via callbacks.

        Gathers current metrics from all discovered GPUs using NVML APIs,
        converts them to TelemetryRecord objects (scalar GPU metrics) and
        optionally ServerMetricsRecord (per-process VRAM), then delivers
        via their respective callbacks.

        Uses asyncio.to_thread() to avoid blocking the event loop with NVML calls.
        """
        try:
            result = await asyncio.to_thread(self._collect_all_metrics)

            if result.telemetry_records and self._record_callback:
                await self._record_callback(result.telemetry_records, self.id)

            if result.server_metrics_record and self._server_metrics_record_callback:
                await self._server_metrics_record_callback(
                    [result.server_metrics_record], self.id
                )
        except Exception as e:
            if self._error_callback:
                try:
                    await self._error_callback(ErrorDetails.from_exception(e), self.id)
                except Exception as callback_error:
                    self.error(f"Failed to send error via callback: {callback_error}")
            else:
                self.error(f"Metrics collection error: {e}")

    def _collect_all_metrics(self) -> _CollectionResult:
        """Collect all metrics from GPUs: scalar telemetry + per-process VRAM.

        Thread-safe - acquires lock to prevent collection during shutdown.

        Returns:
            _CollectionResult with telemetry records and optional server metrics record.
        """
        with self._nvml.lock:
            if not self._nvml.initialized or not self._gpus:
                return _CollectionResult()

            current_timestamp = time.time_ns()
            records = self._collect_gpu_metrics(current_timestamp)

            server_metrics_record: ServerMetricsRecord | None = None
            if self._server_metrics_record_callback:
                server_metrics_record = self._collect_process_vram(current_timestamp)

            return _CollectionResult(
                telemetry_records=records,
                server_metrics_record=server_metrics_record,
            )

    def _collect_gpu_metrics(self, timestamp_ns: int) -> list[TelemetryRecord]:
        """Collect scalar per-GPU metrics using NVML APIs.

        Must be called while holding self._nvml.lock.

        Args:
            timestamp_ns: Nanosecond timestamp for all records in this cycle.

        Returns:
            List of TelemetryRecord objects, one per GPU.
        """
        records: list[TelemetryRecord] = []
        NVMLError = pynvml.NVMLError

        for gpu in self._gpus:
            handle = gpu.handle
            telemetry_data = TelemetryMetrics()

            with contextlib.suppress(NVMLError):
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                telemetry_data.gpu_power_usage = (
                    power_mw * ScalingFactors.gpu_power_usage
                )

            with contextlib.suppress(NVMLError):
                energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                telemetry_data.energy_consumption = (
                    energy_mj * ScalingFactors.energy_consumption
                )

            with contextlib.suppress(NVMLError):
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                telemetry_data.gpu_utilization = float(util.gpu)
                telemetry_data.mem_utilization = float(util.memory)

            with contextlib.suppress(NVMLError):
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                telemetry_data.gpu_memory_used = (
                    mem_info.used * ScalingFactors.gpu_memory_used
                )
                telemetry_data.gpu_memory_total = (
                    mem_info.total * ScalingFactors.gpu_memory_total
                )
                telemetry_data.gpu_memory_free = (
                    mem_info.free * ScalingFactors.gpu_memory_free
                )

            with contextlib.suppress(NVMLError):
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                telemetry_data.gpu_temperature = float(temp)

            with contextlib.suppress(NVMLError):
                dec_util, _ = pynvml.nvmlDeviceGetDecoderUtilization(handle)
                telemetry_data.decoder_utilization = float(dec_util)

            with contextlib.suppress(NVMLError):
                enc_util, _ = pynvml.nvmlDeviceGetEncoderUtilization(handle)
                telemetry_data.encoder_utilization = float(enc_util)

            with contextlib.suppress(NVMLError):
                jpg_util, _ = pynvml.nvmlDeviceGetJpgUtilization(handle)
                telemetry_data.jpg_utilization = float(jpg_util)

            sm_util: float | None = None
            if gpu.gpm_samples:
                sm_util = self._get_sm_utilization_gpm(gpu)

            if sm_util is None:
                with contextlib.suppress(NVMLError):
                    process_utils = pynvml.nvmlDeviceGetProcessesUtilizationInfo(
                        handle, 0
                    )
                    sm_util = (
                        sum(p.smUtil for p in process_utils) if process_utils else 0.0
                    )

            if sm_util is not None:
                telemetry_data.sm_utilization = min(float(sm_util), 100.0)

            with contextlib.suppress(NVMLError):
                violation = pynvml.nvmlDeviceGetViolationStatus(
                    handle, pynvml.NVML_PERF_POLICY_POWER
                )
                telemetry_data.power_violation = (
                    violation.violationTime * ScalingFactors.power_violation
                )

            with contextlib.suppress(NVMLError):
                telemetry_data.graphics_clock = float(
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                )

            with contextlib.suppress(NVMLError):
                telemetry_data.sm_clock = float(
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                )

            with contextlib.suppress(NVMLError):
                telemetry_data.memory_clock = float(
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                )

            with contextlib.suppress(NVMLError):
                power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                telemetry_data.gpu_power_limit = (
                    power_limit_mw * ScalingFactors.gpu_power_limit
                )

            with contextlib.suppress(NVMLError):
                pstate = pynvml.nvmlDeviceGetPerformanceState(handle)
                telemetry_data.performance_state = float(pstate)

            with contextlib.suppress(NVMLError):
                telemetry_data.pcie_tx_throughput = float(
                    pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                    )
                )

            with contextlib.suppress(NVMLError):
                telemetry_data.pcie_rx_throughput = float(
                    pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                    )
                )

            with contextlib.suppress(NVMLError):
                telemetry_data.fan_speed = float(pynvml.nvmlDeviceGetFanSpeed(handle))

            if telemetry_data.model_fields_set:
                record = TelemetryRecord(
                    timestamp_ns=timestamp_ns,
                    dcgm_url=PYNVML_SOURCE_IDENTIFIER,
                    **gpu.metadata.model_dump(),
                    telemetry_data=telemetry_data,
                )
                records.append(record)

        return records

    def _collect_process_vram(self, timestamp_ns: int) -> ServerMetricsRecord | None:
        """Collect per-process GPU memory usage across all GPUs.

        Must be called while holding self._nvml.lock.

        Args:
            timestamp_ns: Nanosecond timestamp for the record.

        Returns:
            ServerMetricsRecord with labeled per-process VRAM samples, or None if empty.
        """
        samples: list[MetricSample] = []

        for gpu in self._gpus:
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(gpu.handle)
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
                                "gpu_index": str(gpu.metadata.gpu_index),
                            },
                            value=float(used_mem),
                        )
                    )
                except Exception:
                    continue

        if not samples:
            return None

        return ServerMetricsRecord(
            endpoint_url=PYNVML_SOURCE_IDENTIFIER,
            timestamp_ns=timestamp_ns,
            metrics={
                "gpu_process_memory_used_bytes": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="GPU memory used per process (bytes)",
                    samples=samples,
                ),
            },
        )

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
