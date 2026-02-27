# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import ServerMetricsRecord
from aiperf.server_metrics.protocols import SYSTEM_METRICS_SOURCE_IDENTIFIER
from aiperf.server_metrics.system_collector import SystemMetricsCollector


def _make_collector_with_nvml(
    handles: list, mock_pynvml: MagicMock
) -> SystemMetricsCollector:
    """Create a collector with pre-configured NvmlHandleManager state."""
    collector = SystemMetricsCollector()
    nvml = collector._nvml
    nvml._initialized = True
    nvml._handles = handles
    nvml._device_count = len(handles)
    with patch("aiperf.common.nvml_handle_manager._pynvml", mock_pynvml):
        pass
    return collector


class TestSystemMetricsCollectorInitialization:
    """Test SystemMetricsCollector initialization."""

    def test_initialization_defaults(self):
        collector = SystemMetricsCollector()
        assert collector.id == "system_metrics_collector"
        assert collector.endpoint_url == SYSTEM_METRICS_SOURCE_IDENTIFIER
        assert collector.collection_interval == 0.333

    def test_initialization_custom_interval(self):
        collector = SystemMetricsCollector(collection_interval=1.0)
        assert collector.collection_interval == 1.0

    def test_initialization_with_callbacks(self):
        record_cb = AsyncMock()
        error_cb = AsyncMock()
        collector = SystemMetricsCollector(
            record_callback=record_cb,
            error_callback=error_cb,
        )
        assert collector._record_callback is record_cb
        assert collector._error_callback is error_cb

    def test_nvml_handle_manager_created(self):
        from aiperf.common.nvml_handle_manager import NvmlHandleManager

        collector = SystemMetricsCollector()
        assert isinstance(collector._nvml, NvmlHandleManager)
        assert collector._nvml.initialized is False

    @pytest.mark.asyncio
    async def test_is_url_reachable_always_true(self):
        collector = SystemMetricsCollector()
        assert await collector.is_url_reachable() is True


class TestSystemMetricsCollectorMetricCollection:
    """Test metric collection produces valid ServerMetricsRecord objects."""

    def test_collect_system_metrics_returns_record(self):
        collector = SystemMetricsCollector()
        record = collector._collect_system_metrics()

        assert isinstance(record, ServerMetricsRecord)
        assert record.endpoint_url == SYSTEM_METRICS_SOURCE_IDENTIFIER
        assert record.timestamp_ns > 0

    def test_record_contains_cpu_metric(self):
        collector = SystemMetricsCollector()
        record = collector._collect_system_metrics()

        assert "system_cpu_utilization_percent" in record.metrics
        metric = record.metrics["system_cpu_utilization_percent"]
        assert metric.type == PrometheusMetricType.GAUGE
        assert len(metric.samples) == 1
        assert metric.samples[0].value is not None
        assert 0.0 <= metric.samples[0].value <= 100.0

    def test_record_contains_memory_metrics(self):
        collector = SystemMetricsCollector()
        record = collector._collect_system_metrics()

        for name in [
            "system_memory_used_bytes",
            "system_memory_total_bytes",
            "system_memory_available_bytes",
        ]:
            assert name in record.metrics, f"Missing metric: {name}"
            metric = record.metrics[name]
            assert metric.type == PrometheusMetricType.GAUGE
            assert len(metric.samples) == 1
            assert metric.samples[0].value > 0

    def test_memory_total_greater_than_used(self):
        collector = SystemMetricsCollector()
        record = collector._collect_system_metrics()

        total = record.metrics["system_memory_total_bytes"].samples[0].value
        used = record.metrics["system_memory_used_bytes"].samples[0].value
        available = record.metrics["system_memory_available_bytes"].samples[0].value
        assert total >= used
        assert total >= available

    def test_no_gpu_metrics_when_nvml_not_available(self):
        collector = SystemMetricsCollector()
        record = collector._collect_system_metrics()

        assert "gpu_process_memory_used_bytes" not in record.metrics


class TestSystemMetricsCollectorGPUProcessMemory:
    """Test GPU per-process memory collection with mocked pynvml."""

    def test_collect_gpu_process_memory_with_processes(self):
        mock_handle = MagicMock()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.usedGpuMemory = 1024 * 1024 * 512

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [mock_proc]
        mock_pynvml.NVMLError = Exception

        collector = SystemMetricsCollector()
        collector._nvml._initialized = True
        collector._nvml._handles = [mock_handle]
        collector._nvml._handle_indices = [0]
        collector._nvml._device_count = 1

        with (
            patch("aiperf.common.nvml_handle_manager._pynvml", mock_pynvml),
            patch.object(
                SystemMetricsCollector,
                "_get_process_name",
                return_value="tritonserver",
            ),
        ):
            samples = collector._collect_gpu_process_memory()

        assert len(samples) == 1
        assert samples[0].labels == {
            "pid": "12345",
            "process_name": "tritonserver",
            "gpu_index": "0",
        }
        assert samples[0].value == 1024 * 1024 * 512

    def test_collect_gpu_process_memory_multiple_gpus(self):
        mock_handle_0 = MagicMock()
        mock_handle_1 = MagicMock()

        mock_proc_0 = MagicMock()
        mock_proc_0.pid = 100
        mock_proc_0.usedGpuMemory = 1000

        mock_proc_1 = MagicMock()
        mock_proc_1.pid = 200
        mock_proc_1.usedGpuMemory = 2000

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.side_effect = [
            [mock_proc_0],
            [mock_proc_1],
        ]
        mock_pynvml.NVMLError = Exception

        collector = SystemMetricsCollector()
        collector._nvml._initialized = True
        collector._nvml._handles = [mock_handle_0, mock_handle_1]
        collector._nvml._handle_indices = [0, 1]
        collector._nvml._device_count = 2

        with (
            patch("aiperf.common.nvml_handle_manager._pynvml", mock_pynvml),
            patch.object(
                SystemMetricsCollector,
                "_get_process_name",
                return_value="python",
            ),
        ):
            samples = collector._collect_gpu_process_memory()

        assert len(samples) == 2
        assert samples[0].labels["gpu_index"] == "0"
        assert samples[1].labels["gpu_index"] == "1"

    def test_collect_gpu_process_memory_empty_when_not_initialized(self):
        collector = SystemMetricsCollector()
        samples = collector._collect_gpu_process_memory()
        assert samples == []

    def test_collect_gpu_process_memory_skips_none_used_memory(self):
        """Processes with usedGpuMemory=None (NVML_VALUE_NOT_AVAILABLE) are skipped."""
        mock_handle = MagicMock()

        proc_valid = MagicMock()
        proc_valid.pid = 100
        proc_valid.usedGpuMemory = 1024

        proc_none = MagicMock()
        proc_none.pid = 200
        proc_none.usedGpuMemory = None

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [
            proc_valid,
            proc_none,
        ]
        mock_pynvml.NVMLError = Exception

        collector = SystemMetricsCollector()
        collector._nvml._initialized = True
        collector._nvml._handles = [mock_handle]
        collector._nvml._handle_indices = [0]
        collector._nvml._device_count = 1

        with (
            patch("aiperf.common.nvml_handle_manager._pynvml", mock_pynvml),
            patch.object(
                SystemMetricsCollector,
                "_get_process_name",
                return_value="python",
            ),
        ):
            samples = collector._collect_gpu_process_memory()

        assert len(samples) == 1
        assert samples[0].labels["pid"] == "100"


class TestSystemMetricsCollectorProcessName:
    """Test process name identification logic."""

    def test_get_process_name_tritonserver(self):
        with patch("psutil.Process") as mock_cls:
            mock_proc = MagicMock()
            mock_proc.cmdline.return_value = [
                "/opt/tritonserver",
                "--model-repo",
                "/models",
            ]
            mock_proc.name.return_value = "tritonserver"
            mock_cls.return_value = mock_proc

            assert SystemMetricsCollector._get_process_name(123) == "tritonserver"

    def test_get_process_name_triton_backend_stub(self):
        with patch("psutil.Process") as mock_cls:
            mock_proc = MagicMock()
            mock_proc.cmdline.return_value = [
                "/opt/triton_python_backend_stub",
                "model.py",
            ]
            mock_proc.name.return_value = "python3"
            mock_cls.return_value = mock_proc

            assert (
                SystemMetricsCollector._get_process_name(123)
                == "triton_python_backend_stub"
            )

    def test_get_process_name_regular_process(self):
        with patch("psutil.Process") as mock_cls:
            mock_proc = MagicMock()
            mock_proc.cmdline.return_value = ["python", "train.py"]
            mock_proc.name.return_value = "python"
            mock_cls.return_value = mock_proc

            assert SystemMetricsCollector._get_process_name(123) == "python"

    def test_get_process_name_no_such_process(self):
        import psutil

        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(99999)):
            assert SystemMetricsCollector._get_process_name(99999) == "pid:99999"


class TestSystemMetricsCollectorCallbacks:
    """Test callback-based record delivery."""

    @pytest.mark.asyncio
    async def test_collect_and_process_metrics_calls_callback(self):
        record_cb = AsyncMock()
        collector = SystemMetricsCollector(record_callback=record_cb)

        await collector.collect_and_process_metrics()

        record_cb.assert_called_once()
        args = record_cb.call_args
        records = args[0][0]
        collector_id = args[0][1]

        assert len(records) == 1
        assert isinstance(records[0], ServerMetricsRecord)
        assert collector_id == "system_metrics_collector"

    @pytest.mark.asyncio
    async def test_error_callback_on_failure(self):
        error_cb = AsyncMock()
        collector = SystemMetricsCollector(error_callback=error_cb)

        with patch.object(
            collector,
            "_collect_system_metrics",
            side_effect=RuntimeError("test error"),
        ):
            await collector.collect_and_process_metrics()

        error_cb.assert_called_once()
        error_details = error_cb.call_args[0][0]
        assert "test error" in str(error_details)

    @pytest.mark.asyncio
    async def test_no_callback_no_error(self):
        """Collector works without callbacks (logs instead)."""
        collector = SystemMetricsCollector()
        await collector.collect_and_process_metrics()


class TestSystemMetricsCollectorProtocolCompliance:
    """Verify SystemMetricsCollector satisfies ServerMetricsCollectorProtocol."""

    def test_satisfies_protocol(self):
        from aiperf.server_metrics.protocols import ServerMetricsCollectorProtocol

        collector = SystemMetricsCollector()
        assert isinstance(collector, ServerMetricsCollectorProtocol)
