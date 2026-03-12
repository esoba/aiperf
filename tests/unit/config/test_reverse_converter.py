# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reverse converter: AIPerfConfig -> (UserConfig, ServiceConfig)."""

import pytest

from aiperf.common.config.zmq_config import ZMQIPCConfig, ZMQTCPConfig
from aiperf.common.enums import (
    AIPerfLogLevel,
    ExportLevel,
    GPUTelemetryMode,
)
from aiperf.config import AIPerfConfig
from aiperf.config.reverse_converter import convert_to_legacy_configs
from aiperf.plugin.enums import ArrivalPattern, TimingMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
)


def _make_config(**overrides) -> AIPerfConfig:
    """Build an AIPerfConfig with minimal defaults merged with overrides."""
    base = {
        **_MINIMAL,
        "datasets": {
            "default": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
            }
        },
        "load": {"default": {"type": "concurrency", "requests": 10, "concurrency": 4}},
    }
    base.update(overrides)
    return AIPerfConfig(**base)


# ---------------------------------------------------------------------------
# Basic synthetic dataset + concurrency phase
# ---------------------------------------------------------------------------


class TestBasicSyntheticConcurrency:
    """Test basic synthetic dataset with concurrency phase."""

    @pytest.fixture
    def result(self):
        config = _make_config()
        return convert_to_legacy_configs(config)

    def test_returns_tuple(self, result):
        uc, sc = result
        assert uc is not None
        assert sc is not None

    def test_endpoint_urls(self, result):
        uc, _ = result
        assert uc.endpoint.urls == ["http://localhost:8000/v1/chat/completions"]

    def test_endpoint_model_names(self, result):
        uc, _ = result
        assert uc.endpoint.model_names == ["test-model"]

    def test_concurrency(self, result):
        uc, _ = result
        assert uc.loadgen.concurrency == 4

    def test_request_count(self, result):
        uc, _ = result
        assert uc.loadgen.request_count == 10

    def test_arrival_pattern_concurrency(self, result):
        uc, _ = result
        assert uc.loadgen.arrival_pattern == ArrivalPattern.CONCURRENCY_BURST

    def test_concurrency_phase_does_not_set_request_rate(self, result):
        uc, _ = result
        assert uc.loadgen.request_rate is None
        assert uc.loadgen.arrival_smoothness is None

    def test_timing_mode(self, result):
        uc, _ = result
        assert uc._timing_mode == TimingMode.REQUEST_RATE

    def test_input_tokens(self, result):
        uc, _ = result
        assert uc.input.prompt.input_tokens.mean == 128
        assert uc.input.prompt.input_tokens.stddev == 0.0

    def test_output_tokens(self, result):
        uc, _ = result
        assert uc.input.prompt.output_tokens.mean == 64
        assert uc.input.prompt.output_tokens.stddev == 0.0


# ---------------------------------------------------------------------------
# Rate-based phases
# ---------------------------------------------------------------------------


class TestRateBasedPhases:
    """Test rate-based phase types (poisson, gamma, constant)."""

    @pytest.mark.parametrize(
        ("phase_type", "expected_pattern"),
        [
            ("poisson", ArrivalPattern.POISSON),
            ("gamma", ArrivalPattern.GAMMA),
            ("constant", ArrivalPattern.CONSTANT),
        ],
    )
    def test_arrival_pattern(self, phase_type, expected_pattern):
        config = _make_config(
            load={"default": {"type": phase_type, "requests": 50, "rate": 10.0}},
        )
        uc, _ = convert_to_legacy_configs(config)
        assert uc.loadgen.arrival_pattern == expected_pattern
        assert uc.loadgen.request_rate == 10.0

    def test_timing_mode_always_request_rate(self):
        config = _make_config(
            load={"default": {"type": "poisson", "requests": 50, "rate": 5.0}},
        )
        uc, _ = convert_to_legacy_configs(config)
        assert uc._timing_mode == TimingMode.REQUEST_RATE


# ---------------------------------------------------------------------------
# Warmup + profiling phases
# ---------------------------------------------------------------------------


class TestWarmupProfiling:
    """Test warmup (excluded) + profiling (included) phase separation."""

    @pytest.fixture
    def result(self):
        config = _make_config(
            load={
                "warmup": {
                    "type": "concurrency",
                    "requests": 5,
                    "concurrency": 2,
                    "exclude": True,
                },
                "profiling": {
                    "type": "poisson",
                    "requests": 100,
                    "rate": 20.0,
                },
            },
        )
        return convert_to_legacy_configs(config)

    def test_profiling_rate(self, result):
        uc, _ = result
        assert uc.loadgen.request_rate == 20.0
        assert uc.loadgen.request_count == 100

    def test_warmup_concurrency(self, result):
        uc, _ = result
        assert uc.loadgen.warmup_concurrency == 2
        assert uc.loadgen.warmup_request_count == 5

    def test_arrival_pattern_from_profiling(self, result):
        uc, _ = result
        assert uc.loadgen.arrival_pattern == ArrivalPattern.POISSON


# ---------------------------------------------------------------------------
# File dataset
# ---------------------------------------------------------------------------


class TestFileDataset:
    """Test file dataset with custom type."""

    @pytest.fixture
    def result(self):
        config = _make_config(
            datasets={
                "default": {
                    "type": "file",
                    "path": "/data/prompts.jsonl",
                    "format": "single_turn",
                },
            },
        )
        return convert_to_legacy_configs(config)

    def test_file_path(self, result):
        uc, _ = result
        assert uc.input.file == "/data/prompts.jsonl"

    def test_custom_dataset_type(self, result):
        uc, _ = result
        assert uc.input.custom_dataset_type == "single_turn"


# ---------------------------------------------------------------------------
# Public dataset
# ---------------------------------------------------------------------------


class TestPublicDataset:
    """Test public dataset."""

    @pytest.fixture
    def result(self):
        config = _make_config(
            datasets={
                "default": {
                    "type": "public",
                    "name": "sharegpt",
                },
            },
        )
        return convert_to_legacy_configs(config)

    def test_public_dataset_name(self, result):
        uc, _ = result
        assert uc.input.public_dataset == "sharegpt"


# ---------------------------------------------------------------------------
# Composed dataset with augmentation
# ---------------------------------------------------------------------------


class TestComposedDataset:
    """Test composed dataset with augmentation."""

    @pytest.fixture
    def result(self):
        config = _make_config(
            datasets={
                "default": {
                    "type": "composed",
                    "source": {
                        "type": "file",
                        "path": "/data/base.jsonl",
                        "format": "single_turn",
                    },
                    "augment": {
                        "osl": {"mean": 256, "stddev": 32},
                    },
                },
            },
        )
        return convert_to_legacy_configs(config)

    def test_file_from_source(self, result):
        uc, _ = result
        assert uc.input.file == "/data/base.jsonl"

    def test_custom_dataset_type_from_source(self, result):
        uc, _ = result
        assert uc.input.custom_dataset_type == "single_turn"

    def test_osl_from_augment(self, result):
        uc, _ = result
        assert uc.input.prompt.output_tokens.mean == 256
        assert uc.input.prompt.output_tokens.stddev == 32.0


# ---------------------------------------------------------------------------
# Communication configs (IPC / TCP)
# ---------------------------------------------------------------------------


class TestIpcCommunication:
    """Test IPC communication config mapping."""

    @pytest.fixture
    def result(self):
        config = _make_config(
            runtime={"communication": {"type": "ipc", "path": "/tmp/aiperf-test"}},
        )
        return convert_to_legacy_configs(config)

    def test_ipc_comm_config(self, result):
        _, sc = result
        assert isinstance(sc._comm_config, ZMQIPCConfig)


class TestTcpCommunication:
    """Test TCP communication config mapping."""

    @pytest.fixture
    def result(self):
        config = _make_config(
            runtime={
                "communication": {
                    "type": "tcp",
                    "host": "10.0.0.1",
                    "records_port": 5557,
                    "credit_router_port": 5564,
                    "event_bus_proxy": {"frontend_port": 5663, "backend_port": 5664},
                    "dataset_manager_proxy": {
                        "frontend_port": 5661,
                        "backend_port": 5662,
                    },
                    "raw_inference_proxy": {
                        "frontend_port": 5665,
                        "backend_port": 5666,
                    },
                },
            },
        )
        return convert_to_legacy_configs(config)

    def test_tcp_comm_config(self, result):
        _, sc = result
        assert isinstance(sc._comm_config, ZMQTCPConfig)

    def test_tcp_host(self, result):
        _, sc = result
        assert sc._comm_config.host == "10.0.0.1"


# ---------------------------------------------------------------------------
# GPU telemetry
# ---------------------------------------------------------------------------


class TestGpuTelemetryEnabled:
    """Test GPU telemetry enabled with URLs."""

    @pytest.fixture
    def result(self):
        config = _make_config(
            gpu_telemetry={
                "enabled": True,
                "urls": ["http://localhost:9400/metrics"],
            },
        )
        return convert_to_legacy_configs(config)

    def test_gpu_telemetry_not_disabled(self, result):
        uc, _ = result
        assert uc.no_gpu_telemetry is False

    def test_gpu_telemetry_urls(self, result):
        uc, _ = result
        assert uc._gpu_telemetry_urls == ["http://localhost:9400/metrics"]

    def test_gpu_telemetry_mode(self, result):
        uc, _ = result
        assert uc._gpu_telemetry_mode == GPUTelemetryMode.SUMMARY

    def test_gpu_telemetry_collector_type(self, result):
        uc, _ = result
        assert uc._gpu_telemetry_collector_type == "dcgm"


class TestGpuTelemetryDisabled:
    """Test GPU telemetry disabled."""

    @pytest.fixture
    def result(self):
        config = _make_config(gpu_telemetry={"enabled": False})
        return convert_to_legacy_configs(config)

    def test_no_gpu_telemetry_flag(self, result):
        uc, _ = result
        assert uc.no_gpu_telemetry is True

    def test_empty_urls(self, result):
        uc, _ = result
        assert uc._gpu_telemetry_urls == []


# ---------------------------------------------------------------------------
# Server metrics
# ---------------------------------------------------------------------------


class TestServerMetricsEnabled:
    """Test server metrics enabled."""

    @pytest.fixture
    def result(self):
        config = _make_config(
            server_metrics={
                "enabled": True,
                "urls": ["http://localhost:9090/metrics"],
            },
        )
        return convert_to_legacy_configs(config)

    def test_not_disabled(self, result):
        uc, _ = result
        assert uc.no_server_metrics is False

    def test_urls(self, result):
        uc, _ = result
        assert uc._server_metrics_urls == ["http://localhost:9090/metrics"]


class TestServerMetricsDisabled:
    """Test server metrics disabled."""

    @pytest.fixture
    def result(self):
        config = _make_config(server_metrics={"enabled": False})
        return convert_to_legacy_configs(config)

    def test_disabled_flag(self, result):
        uc, _ = result
        assert uc.no_server_metrics is True

    def test_empty_urls(self, result):
        uc, _ = result
        assert uc._server_metrics_urls == []


# ---------------------------------------------------------------------------
# User-centric and fixed-schedule phase types
# ---------------------------------------------------------------------------


class TestUserCentricPhase:
    """Test user-centric phase type."""

    @pytest.fixture
    def result(self):
        config = _make_config(
            load={
                "default": {
                    "type": "user_centric",
                    "requests": 50,
                    "rate": 5.0,
                    "users": 10,
                },
            },
        )
        return convert_to_legacy_configs(config)

    def test_timing_mode(self, result):
        uc, _ = result
        assert uc._timing_mode == TimingMode.USER_CENTRIC_RATE

    def test_user_centric_rate(self, result):
        uc, _ = result
        assert uc.loadgen.user_centric_rate == 5.0

    def test_num_users(self, result):
        uc, _ = result
        assert uc.loadgen.num_users == 10


class TestFixedSchedulePhase:
    """Test fixed-schedule phase type."""

    @pytest.fixture
    def result(self):
        config = _make_config(
            datasets={
                "default": {
                    "type": "file",
                    "path": "/data/schedule.jsonl",
                    "format": "single_turn",
                },
            },
            load={
                "default": {
                    "type": "fixed_schedule",
                    "auto_offset": True,
                },
            },
        )
        return convert_to_legacy_configs(config)

    def test_timing_mode(self, result):
        uc, _ = result
        assert uc._timing_mode == TimingMode.FIXED_SCHEDULE

    def test_fixed_schedule_flag(self, result):
        uc, _ = result
        assert uc.input.fixed_schedule is True

    def test_auto_offset(self, result):
        uc, _ = result
        assert uc.input.fixed_schedule_auto_offset is True


# ---------------------------------------------------------------------------
# Service config fields
# ---------------------------------------------------------------------------


class TestServiceConfig:
    """Test ServiceConfig field mapping."""

    @pytest.fixture
    def result(self):
        config = _make_config(
            logging={"level": "DEBUG"},
            runtime={"workers": 16, "record_processors": 4},
        )
        return convert_to_legacy_configs(config)

    def test_log_level(self, result):
        _, sc = result
        assert sc.log_level == AIPerfLogLevel.DEBUG

    def test_verbose(self, result):
        _, sc = result
        assert sc.verbose is True

    def test_workers_max(self, result):
        _, sc = result
        assert sc.workers.max == 16

    def test_record_processor_count(self, result):
        _, sc = result
        assert sc.record_processor_service_count == 4

    def test_comm_config_default_ipc(self, result):
        _, sc = result
        assert isinstance(sc._comm_config, ZMQIPCConfig)


# ---------------------------------------------------------------------------
# Output / artifacts
# ---------------------------------------------------------------------------


class TestOutputConfig:
    """Test output config / artifacts mapping."""

    def test_raw_export_level(self):
        config = _make_config(artifacts={"raw": True})
        uc, _ = convert_to_legacy_configs(config)
        assert uc.output.export_level == ExportLevel.RAW

    def test_records_export_level(self):
        config = _make_config(artifacts={"records": ["jsonl"]})
        uc, _ = convert_to_legacy_configs(config)
        assert uc.output.export_level == ExportLevel.RECORDS

    def test_summary_export_level(self):
        config = _make_config(artifacts={"records": False, "raw": False})
        uc, _ = convert_to_legacy_configs(config)
        assert uc.output.export_level == ExportLevel.SUMMARY
