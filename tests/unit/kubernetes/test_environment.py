# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.environment module."""

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.kubernetes.environment import (
    K8sEnvironment,
    _HealthProbeSettings,
    _JobSetSettings,
    _PortSettings,
    _resource_settings,
    _ZMQSettings,
)


class TestResourceSettingsToK8sResources:
    """Tests for _ResourceSettings.to_k8s_resources method."""

    @pytest.mark.parametrize(
        "setting_attr,cpu_request,cpu_limit,memory_request,memory_limit",
        [
            param("CONTROLLER", "600m", "3000m", "1Gi", "4Gi", id="controller"),
            param("WORKER", "350m", "500m", "1Gi", "2Gi", id="worker"),
            param("TIMING_MANAGER", "500m", "500m", "512Mi", "512Mi", id="timing_manager"),
            param("DATASET_MANAGER", "200m", "1000m", "512Mi", "1Gi", id="dataset_manager"),
            param("RECORDS_MANAGER", "600m", "3000m", "896Mi", "2304Mi", id="records_manager"),
            param("RECORD_PROCESSOR", "100m", "250m", "512Mi", "1Gi", id="record_processor"),
            param("GPU_TELEMETRY_MANAGER", "100m", "500m", "256Mi", "512Mi", id="gpu_telemetry"),
            param("SERVER_METRICS_MANAGER", "100m", "500m", "256Mi", "512Mi", id="server_metrics"),
        ],
    )  # fmt: skip
    def test_to_k8s_resources_returns_correct_structure(
        self,
        setting_attr: str,
        cpu_request: str,
        cpu_limit: str,
        memory_request: str,
        memory_limit: str,
    ) -> None:
        """Test to_k8s_resources returns correctly structured dict."""
        setting = getattr(K8sEnvironment, setting_attr)
        resources = setting.to_k8s_resources()

        assert resources == {
            "requests": {"cpu": cpu_request, "memory": memory_request},
            "limits": {"cpu": cpu_limit, "memory": memory_limit},
        }


class TestK8sEnvironmentController:
    """Tests for K8sEnvironment.CONTROLLER settings."""

    def test_controller_default_values(self) -> None:
        """Test controller has expected default values."""
        controller = K8sEnvironment.CONTROLLER
        assert controller.CPU_REQUEST == "600m"
        assert controller.CPU_LIMIT == "3000m"
        assert controller.MEMORY_REQUEST == "1Gi"
        assert controller.MEMORY_LIMIT == "4Gi"

    def test_controller_to_k8s_resources(self) -> None:
        """Test converting controller settings to Kubernetes resources."""
        resources = K8sEnvironment.CONTROLLER.to_k8s_resources()
        assert "requests" in resources
        assert "limits" in resources
        assert resources["requests"]["cpu"] == "600m"
        assert resources["limits"]["memory"] == "4Gi"


class TestK8sEnvironmentWorker:
    """Tests for K8sEnvironment.WORKER settings."""

    def test_worker_default_values(self) -> None:
        """Test worker has expected default values."""
        worker = K8sEnvironment.WORKER
        assert worker.CPU_REQUEST == "350m"
        assert worker.CPU_LIMIT == "500m"
        assert worker.MEMORY_REQUEST == "1Gi"
        assert worker.MEMORY_LIMIT == "2Gi"

    def test_worker_to_k8s_resources(self) -> None:
        """Test converting worker settings to Kubernetes resources."""
        resources = K8sEnvironment.WORKER.to_k8s_resources()
        assert resources["requests"]["cpu"] == "350m"
        assert resources["limits"]["cpu"] == "500m"


class TestK8sEnvironmentTimingManager:
    """Tests for K8sEnvironment.TIMING_MANAGER settings."""

    def test_timing_manager_default_values(self) -> None:
        """Test timing manager has expected default values."""
        timing_manager = K8sEnvironment.TIMING_MANAGER
        assert timing_manager.CPU_REQUEST == "500m"
        assert timing_manager.CPU_LIMIT == "500m"
        assert timing_manager.MEMORY_REQUEST == "512Mi"
        assert timing_manager.MEMORY_LIMIT == "512Mi"


class TestK8sEnvironmentDatasetManager:
    """Tests for K8sEnvironment.DATASET_MANAGER settings."""

    def test_dataset_manager_default_values(self) -> None:
        """Test dataset manager has expected default values."""
        dataset_manager = K8sEnvironment.DATASET_MANAGER
        assert dataset_manager.CPU_REQUEST == "200m"
        assert dataset_manager.CPU_LIMIT == "1000m"
        assert dataset_manager.MEMORY_REQUEST == "512Mi"
        assert dataset_manager.MEMORY_LIMIT == "1Gi"


class TestK8sEnvironmentRecordsManager:
    """Tests for K8sEnvironment.RECORDS_MANAGER settings."""

    def test_records_manager_default_values(self) -> None:
        """Test records manager has expected default values."""
        records_manager = K8sEnvironment.RECORDS_MANAGER
        assert records_manager.CPU_REQUEST == "600m"
        assert records_manager.CPU_LIMIT == "3000m"
        assert records_manager.MEMORY_REQUEST == "896Mi"
        assert records_manager.MEMORY_LIMIT == "2304Mi"


class TestK8sEnvironmentRecordProcessor:
    """Tests for K8sEnvironment.RECORD_PROCESSOR settings."""

    def test_record_processor_default_values(self) -> None:
        """Test record processor has expected default values."""
        record_processor = K8sEnvironment.RECORD_PROCESSOR
        assert record_processor.CPU_REQUEST == "100m"
        assert record_processor.CPU_LIMIT == "250m"
        assert record_processor.MEMORY_REQUEST == "512Mi"
        assert record_processor.MEMORY_LIMIT == "1Gi"


class TestK8sEnvironmentGPUTelemetryManager:
    """Tests for K8sEnvironment.GPU_TELEMETRY_MANAGER settings."""

    def test_gpu_telemetry_manager_default_values(self) -> None:
        """Test GPU telemetry manager has expected default values."""
        gpu_telemetry = K8sEnvironment.GPU_TELEMETRY_MANAGER
        assert gpu_telemetry.CPU_REQUEST == "100m"
        assert gpu_telemetry.CPU_LIMIT == "500m"
        assert gpu_telemetry.MEMORY_REQUEST == "256Mi"
        assert gpu_telemetry.MEMORY_LIMIT == "512Mi"


class TestK8sEnvironmentServerMetricsManager:
    """Tests for K8sEnvironment.SERVER_METRICS_MANAGER settings."""

    def test_server_metrics_manager_default_values(self) -> None:
        """Test server metrics manager has expected default values."""
        server_metrics = K8sEnvironment.SERVER_METRICS_MANAGER
        assert server_metrics.CPU_REQUEST == "100m"
        assert server_metrics.CPU_LIMIT == "500m"
        assert server_metrics.MEMORY_REQUEST == "256Mi"
        assert server_metrics.MEMORY_LIMIT == "512Mi"


class TestK8sEnvironmentHealth:
    """Tests for K8sEnvironment.HEALTH settings."""

    def test_health_default_values(self) -> None:
        """Test health probe has expected default values."""
        health = K8sEnvironment.HEALTH
        assert health.INITIAL_DELAY_SECONDS == 5
        assert health.PERIOD_SECONDS == 10
        assert health.TIMEOUT_SECONDS == 5
        assert health.FAILURE_THRESHOLD == 10
        assert health.SUCCESS_THRESHOLD == 1

    def test_health_values_within_bounds(self) -> None:
        """Test health probe values are within valid bounds."""
        health = K8sEnvironment.HEALTH
        assert 0 <= health.INITIAL_DELAY_SECONDS <= 300
        assert 1 <= health.PERIOD_SECONDS <= 300
        assert 1 <= health.TIMEOUT_SECONDS <= 60
        assert 1 <= health.FAILURE_THRESHOLD <= 20
        assert 1 <= health.SUCCESS_THRESHOLD <= 10


class TestK8sEnvironmentPorts:
    """Tests for K8sEnvironment.PORTS settings."""

    def test_ports_default_values(self) -> None:
        """Test ports have expected default values."""
        ports = K8sEnvironment.PORTS
        assert ports.SYSTEM_CONTROLLER_HEALTH == 8080
        assert ports.WORKER_MANAGER_HEALTH == 8081
        assert ports.TIMING_MANAGER_HEALTH == 8082
        assert ports.DATASET_MANAGER_HEALTH == 8083
        assert ports.RECORDS_MANAGER_HEALTH == 8084
        assert ports.API_SERVICE == 9090
        assert ports.API_SERVICE_HEALTH == 8085
        assert ports.WORKER_HEALTH == 8080
        assert ports.RECORD_PROCESSOR_HEALTH == 8081

    def test_ports_telemetry_defaults(self) -> None:
        """Test telemetry service port defaults."""
        ports = K8sEnvironment.PORTS
        assert ports.GPU_TELEMETRY_MANAGER_HEALTH == 8086
        assert ports.SERVER_METRICS_MANAGER_HEALTH == 8087

    def test_ports_unique_on_controller(self) -> None:
        """Test that controller pod ports are unique."""
        ports = K8sEnvironment.PORTS
        controller_ports = [
            ports.SYSTEM_CONTROLLER_HEALTH,
            ports.WORKER_MANAGER_HEALTH,
            ports.TIMING_MANAGER_HEALTH,
            ports.DATASET_MANAGER_HEALTH,
            ports.RECORDS_MANAGER_HEALTH,
            ports.API_SERVICE,
            ports.API_SERVICE_HEALTH,
            ports.GPU_TELEMETRY_MANAGER_HEALTH,
            ports.SERVER_METRICS_MANAGER_HEALTH,
        ]
        assert len(controller_ports) == len(set(controller_ports))

    @pytest.mark.parametrize(
        "port_attr",
        [
            param("SYSTEM_CONTROLLER_HEALTH", id="system_controller"),
            param("WORKER_MANAGER_HEALTH", id="worker_manager"),
            param("TIMING_MANAGER_HEALTH", id="timing_manager"),
            param("DATASET_MANAGER_HEALTH", id="dataset_manager"),
            param("RECORDS_MANAGER_HEALTH", id="records_manager"),
            param("API_SERVICE", id="api_service"),
            param("API_SERVICE_HEALTH", id="api_service_health"),
            param("GPU_TELEMETRY_MANAGER_HEALTH", id="gpu_telemetry"),
            param("SERVER_METRICS_MANAGER_HEALTH", id="server_metrics"),
            param("WORKER_HEALTH", id="worker_health"),
            param("RECORD_PROCESSOR_HEALTH", id="record_processor"),
        ],
    )  # fmt: skip
    def test_ports_within_valid_range(self, port_attr: str) -> None:
        """Test all port values are within valid range (1-65535)."""
        port_value = getattr(K8sEnvironment.PORTS, port_attr)
        assert 1 <= port_value <= 65535


class TestK8sEnvironmentZMQ:
    """Tests for K8sEnvironment.ZMQ settings."""

    def test_zmq_default_values(self) -> None:
        """Test ZMQ settings have expected default values."""
        zmq = K8sEnvironment.ZMQ
        assert zmq.CONTROLLER_HOST is None
        assert zmq.IPC_PATH == "/aiperf/ipc"


class TestK8sEnvironmentJobSet:
    """Tests for K8sEnvironment.JOBSET settings."""

    def test_jobset_default_values(self) -> None:
        """Test JobSet settings have expected default values."""
        jobset = K8sEnvironment.JOBSET
        assert jobset.TTL_SECONDS_AFTER_FINISHED == 300
        assert jobset.CONTROLLER_BACKOFF_LIMIT == 0
        assert jobset.WORKER_BACKOFF_LIMIT == 3
        assert jobset.CONFIG_MOUNT_PATH == "/etc/aiperf"
        assert jobset.DATASETS_PATH == "/aiperf/datasets"

    def test_jobset_ttl_can_be_zero(self) -> None:
        """Test TTL can be set to zero for immediate cleanup."""
        settings = _JobSetSettings(TTL_SECONDS_AFTER_FINISHED=0)
        assert settings.TTL_SECONDS_AFTER_FINISHED == 0

    def test_jobset_backoff_limits_within_bounds(self) -> None:
        """Test backoff limits are within expected bounds."""
        jobset = K8sEnvironment.JOBSET
        assert 0 <= jobset.CONTROLLER_BACKOFF_LIMIT <= 10
        assert 0 <= jobset.WORKER_BACKOFF_LIMIT <= 20


class TestK8sEnvironmentAllSettings:
    """Tests for K8sEnvironment comprehensive coverage."""

    @pytest.mark.parametrize(
        "setting_name",
        [
            param("CONTROLLER", id="controller"),
            param("WORKER", id="worker"),
            param("TIMING_MANAGER", id="timing_manager"),
            param("DATASET_MANAGER", id="dataset_manager"),
            param("RECORDS_MANAGER", id="records_manager"),
            param("RECORD_PROCESSOR", id="record_processor"),
            param("GPU_TELEMETRY_MANAGER", id="gpu_telemetry"),
            param("SERVER_METRICS_MANAGER", id="server_metrics"),
            param("HEALTH", id="health"),
            param("PORTS", id="ports"),
            param("ZMQ", id="zmq"),
            param("JOBSET", id="jobset"),
        ],
    )  # fmt: skip
    def test_all_settings_exist(self, setting_name: str) -> None:
        """Test all expected settings are available."""
        assert hasattr(K8sEnvironment, setting_name)
        setting = getattr(K8sEnvironment, setting_name)
        assert setting is not None

    @pytest.mark.parametrize(
        "resource_setting",
        [
            param(K8sEnvironment.CONTROLLER, id="controller"),
            param(K8sEnvironment.WORKER, id="worker"),
            param(K8sEnvironment.TIMING_MANAGER, id="timing_manager"),
            param(K8sEnvironment.DATASET_MANAGER, id="dataset_manager"),
            param(K8sEnvironment.RECORDS_MANAGER, id="records_manager"),
            param(K8sEnvironment.RECORD_PROCESSOR, id="record_processor"),
            param(K8sEnvironment.GPU_TELEMETRY_MANAGER, id="gpu_telemetry"),
            param(K8sEnvironment.SERVER_METRICS_MANAGER, id="server_metrics"),
        ],
    )  # fmt: skip
    def test_resource_settings_have_to_k8s_resources(self, resource_setting) -> None:
        """Test all resource settings have to_k8s_resources method."""
        assert hasattr(resource_setting, "to_k8s_resources")
        resources = resource_setting.to_k8s_resources()
        assert "requests" in resources
        assert "limits" in resources
        assert "cpu" in resources["requests"]
        assert "memory" in resources["requests"]
        assert "cpu" in resources["limits"]
        assert "memory" in resources["limits"]


class TestEnvironmentVariableOverrides:
    """Tests for environment variable configuration overrides."""

    def test_controller_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test controller settings can be overridden via env vars."""
        monkeypatch.setenv("AIPERF_K8S_CONTROLLER_CPU_REQUEST", "800m")
        monkeypatch.setenv("AIPERF_K8S_CONTROLLER_MEMORY_LIMIT", "8Gi")

        settings = _resource_settings("CONTROLLER_", "600m", "3000m", "1Gi", "4Gi")
        assert settings.CPU_REQUEST == "800m"
        assert settings.MEMORY_LIMIT == "8Gi"
        assert settings.CPU_LIMIT == "3000m"

    def test_worker_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test worker settings can be overridden via env vars."""
        monkeypatch.setenv("AIPERF_K8S_WORKER_CPU_LIMIT", "1000m")
        monkeypatch.setenv("AIPERF_K8S_WORKER_MEMORY_REQUEST", "2Gi")

        settings = _resource_settings("WORKER_", "350m", "500m", "1Gi", "2Gi")
        assert settings.CPU_LIMIT == "1000m"
        assert settings.MEMORY_REQUEST == "2Gi"

    def test_health_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test health settings can be overridden via env vars."""
        monkeypatch.setenv("AIPERF_K8S_HEALTH_INITIAL_DELAY_SECONDS", "15")
        monkeypatch.setenv("AIPERF_K8S_HEALTH_PERIOD_SECONDS", "30")

        settings = _HealthProbeSettings()
        assert settings.INITIAL_DELAY_SECONDS == 15
        assert settings.PERIOD_SECONDS == 30

    def test_port_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test port settings can be overridden via env vars."""
        monkeypatch.setenv("AIPERF_K8S_PORT_API_SERVICE", "8000")

        settings = _PortSettings()
        assert settings.API_SERVICE == 8000

    def test_zmq_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ZMQ settings can be overridden via env vars."""
        monkeypatch.setenv("AIPERF_K8S_ZMQ_CONTROLLER_HOST", "controller.default.svc")
        monkeypatch.setenv("AIPERF_K8S_ZMQ_IPC_PATH", "/tmp/zmq")

        settings = _ZMQSettings()
        assert settings.CONTROLLER_HOST == "controller.default.svc"
        assert settings.IPC_PATH == "/tmp/zmq"

    def test_jobset_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test JobSet settings can be overridden via env vars."""
        monkeypatch.setenv("AIPERF_K8S_JOBSET_TTL_SECONDS_AFTER_FINISHED", "600")
        monkeypatch.setenv("AIPERF_K8S_JOBSET_WORKER_BACKOFF_LIMIT", "5")
        monkeypatch.setenv("AIPERF_K8S_JOBSET_CONFIG_MOUNT_PATH", "/custom/config")

        settings = _JobSetSettings()
        assert settings.TTL_SECONDS_AFTER_FINISHED == 600
        assert settings.WORKER_BACKOFF_LIMIT == 5
        assert settings.CONFIG_MOUNT_PATH == "/custom/config"

    @pytest.mark.parametrize(
        "factory_prefix,env_prefix",
        [
            param("TIMING_MANAGER_", "AIPERF_K8S_TIMING_MANAGER_", id="timing"),
            param("DATASET_MANAGER_", "AIPERF_K8S_DATASET_MANAGER_", id="dataset"),
            param("RECORDS_MANAGER_", "AIPERF_K8S_RECORDS_MANAGER_", id="records"),
            param("RECORD_PROCESSOR_", "AIPERF_K8S_RECORD_PROCESSOR_", id="processor"),
            param("GPU_TELEMETRY_MANAGER_", "AIPERF_K8S_GPU_TELEMETRY_MANAGER_", id="gpu"),
            param("SERVER_METRICS_MANAGER_", "AIPERF_K8S_SERVER_METRICS_MANAGER_", id="server"),
        ],
    )  # fmt: skip
    def test_resource_settings_env_prefix(
        self, factory_prefix: str, env_prefix: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that resource settings use correct env prefix."""
        monkeypatch.setenv(f"{env_prefix}CPU_REQUEST", "999m")

        settings = _resource_settings(factory_prefix, "100m", "500m", "256Mi", "512Mi")
        assert settings.CPU_REQUEST == "999m"


class TestHealthProbeValidation:
    """Tests for health probe settings validation."""

    @pytest.mark.parametrize(
        "field,min_val,max_val,default_val",
        [
            param("INITIAL_DELAY_SECONDS", 0, 300, 5, id="initial_delay"),
            param("PERIOD_SECONDS", 1, 300, 10, id="period"),
            param("TIMEOUT_SECONDS", 1, 60, 5, id="timeout"),
            param("FAILURE_THRESHOLD", 1, 20, 10, id="failure"),
            param("SUCCESS_THRESHOLD", 1, 10, 1, id="success"),
        ],
    )  # fmt: skip
    def test_health_probe_bounds(
        self, field: str, min_val: int, max_val: int, default_val: int
    ) -> None:
        """Test health probe fields have correct bounds."""
        settings = _HealthProbeSettings()
        value = getattr(settings, field)

        assert value == default_val
        assert min_val <= value <= max_val

    def test_health_probe_validation_at_lower_bound(self) -> None:
        """Test health probe accepts values at lower bounds."""
        settings = _HealthProbeSettings(
            INITIAL_DELAY_SECONDS=0,
            PERIOD_SECONDS=1,
            TIMEOUT_SECONDS=1,
            FAILURE_THRESHOLD=1,
            SUCCESS_THRESHOLD=1,
        )
        assert settings.INITIAL_DELAY_SECONDS == 0
        assert settings.PERIOD_SECONDS == 1

    def test_health_probe_validation_at_upper_bound(self) -> None:
        """Test health probe accepts values at upper bounds."""
        settings = _HealthProbeSettings(
            INITIAL_DELAY_SECONDS=300,
            PERIOD_SECONDS=300,
            TIMEOUT_SECONDS=60,
            FAILURE_THRESHOLD=20,
            SUCCESS_THRESHOLD=10,
        )
        assert settings.INITIAL_DELAY_SECONDS == 300
        assert settings.SUCCESS_THRESHOLD == 10

    def test_health_probe_validation_exceeds_upper_bound_raises(self) -> None:
        """Test health probe rejects values exceeding upper bound."""
        with pytest.raises(ValidationError):
            _HealthProbeSettings(INITIAL_DELAY_SECONDS=301)

    def test_health_probe_validation_below_lower_bound_raises(self) -> None:
        """Test health probe rejects values below lower bound."""
        with pytest.raises(ValidationError):
            _HealthProbeSettings(PERIOD_SECONDS=0)


class TestPortValidation:
    """Tests for port settings validation."""

    def test_port_validation_at_bounds(self) -> None:
        """Test port accepts values at bounds."""
        settings = _PortSettings(SYSTEM_CONTROLLER_HEALTH=1, API_SERVICE=65535)
        assert settings.SYSTEM_CONTROLLER_HEALTH == 1
        assert settings.API_SERVICE == 65535

    def test_port_validation_invalid_lower_bound_raises(self) -> None:
        """Test port rejects values below 1."""
        with pytest.raises(ValidationError):
            _PortSettings(SYSTEM_CONTROLLER_HEALTH=0)

    def test_port_validation_invalid_upper_bound_raises(self) -> None:
        """Test port rejects values above 65535."""
        with pytest.raises(ValidationError):
            _PortSettings(API_SERVICE=65536)
