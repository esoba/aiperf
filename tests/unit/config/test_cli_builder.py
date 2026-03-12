# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_aiperf_config() covering all major modes and edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config.cli_builder import CLIModel, build_aiperf_config
from aiperf.config.config import AIPerfConfig


@pytest.fixture()
def base_kwargs() -> dict:
    """Minimal kwargs to construct a valid CLIModel."""
    return {
        "model_names": ["test-model"],
        "urls": ["http://localhost:8000"],
    }


def _build(base_kwargs: dict, **overrides) -> AIPerfConfig:
    """Build AIPerfConfig from CLIModel with overrides."""
    kwargs = {**base_kwargs, **overrides}
    cli = CLIModel(**kwargs)
    return build_aiperf_config(cli)


class TestConcurrencyMode:
    """Concurrency-only mode (no request rate)."""

    def test_default_concurrency_mode(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, concurrency=8, request_count=100)
        profiling = config.load["profiling"]
        assert profiling.type.value == "concurrency"
        assert profiling.concurrency == 8
        assert profiling.requests == 100

    def test_concurrency_with_duration(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, concurrency=4, benchmark_duration=60.0)
        profiling = config.load["profiling"]
        assert profiling.duration == 60.0
        assert profiling.concurrency == 4


class TestRequestRateMode:
    """Request rate mode with various arrival patterns."""

    def test_poisson_rate(self, base_kwargs: dict) -> None:
        from aiperf.plugin.enums import ArrivalPattern

        config = _build(
            base_kwargs,
            request_rate=10.0,
            arrival_pattern=ArrivalPattern.POISSON,
            request_count=100,
        )
        profiling = config.load["profiling"]
        assert profiling.type.value == "poisson"
        assert profiling.rate == 10.0

    def test_constant_rate(self, base_kwargs: dict) -> None:
        from aiperf.plugin.enums import ArrivalPattern

        config = _build(
            base_kwargs,
            request_rate=5.0,
            arrival_pattern=ArrivalPattern.CONSTANT,
            request_count=50,
        )
        profiling = config.load["profiling"]
        assert profiling.type.value == "constant"
        assert profiling.rate == 5.0

    def test_gamma_rate_with_smoothness(self, base_kwargs: dict) -> None:
        from aiperf.plugin.enums import ArrivalPattern

        config = _build(
            base_kwargs,
            request_rate=20.0,
            arrival_pattern=ArrivalPattern.GAMMA,
            arrival_smoothness=1.5,
            request_count=200,
        )
        profiling = config.load["profiling"]
        assert profiling.type.value == "gamma"
        assert profiling.rate == 20.0
        assert profiling.smoothness == 1.5


class TestFixedScheduleMode:
    """Fixed schedule mode."""

    def test_fixed_schedule(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            fixed_schedule=True,
            request_count=100,
        )
        profiling = config.load["profiling"]
        assert profiling.type.value == "fixed_schedule"


class TestUserCentricMode:
    """User-centric rate mode."""

    def test_user_centric(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            user_centric_rate=5.0,
            num_users=10,
            request_count=100,
        )
        profiling = config.load["profiling"]
        assert profiling.type.value == "user_centric"
        assert profiling.rate == 5.0
        assert profiling.users == 10


class TestWarmupPhase:
    """Warmup phase creation."""

    def test_warmup_with_request_count(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            warmup_request_count=10,
            concurrency=4,
            request_count=100,
        )
        assert "warmup" in config.load
        warmup = config.load["warmup"]
        assert warmup.exclude is True
        assert warmup.requests == 10

    def test_warmup_with_duration(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            warmup_duration=30.0,
            concurrency=4,
            request_count=100,
        )
        warmup = config.load["warmup"]
        assert warmup.exclude is True
        assert warmup.duration == 30.0

    def test_warmup_inherits_concurrency(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            warmup_request_count=10,
            concurrency=8,
            request_count=100,
        )
        warmup = config.load["warmup"]
        assert warmup.concurrency == 8

    def test_warmup_overrides_concurrency(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            warmup_request_count=10,
            warmup_concurrency=2,
            concurrency=8,
            request_count=100,
        )
        warmup = config.load["warmup"]
        assert warmup.concurrency == 2

    def test_no_warmup_by_default(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, concurrency=4, request_count=100)
        assert "warmup" not in config.load


class TestDatasetInference:
    """Dataset type inference from CLI flags."""

    def test_synthetic_by_default(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, request_count=100)
        dataset = config.datasets["main"]
        assert dataset.type.value == "synthetic"

    def test_public_dataset(self, base_kwargs: dict) -> None:
        from aiperf.common.enums import PublicDatasetType

        config = _build(
            base_kwargs,
            public_dataset=PublicDatasetType.SHAREGPT,
            request_count=100,
        )
        dataset = config.datasets["main"]
        assert dataset.type.value == "public"

    def test_file_dataset(self, base_kwargs: dict, tmp_path: Path) -> None:
        input_file = tmp_path / "data.jsonl"
        input_file.write_text('{"prompt": "test"}\n')
        config = _build(
            base_kwargs,
            input_file=str(input_file),
            request_count=100,
        )
        dataset = config.datasets["main"]
        assert dataset.type.value == "file"


class TestVerboseFlags:
    """Verbose flag handling."""

    def test_verbose_sets_debug(self, base_kwargs: dict) -> None:
        from aiperf.common.enums import AIPerfLogLevel

        config = _build(base_kwargs, verbose=True, request_count=100)
        assert config.logging.level == AIPerfLogLevel.DEBUG

    def test_extra_verbose_sets_trace(self, base_kwargs: dict) -> None:
        from aiperf.common.enums import AIPerfLogLevel

        config = _build(base_kwargs, extra_verbose=True, request_count=100)
        assert config.logging.level == AIPerfLogLevel.TRACE

    def test_verbose_sets_simple_ui(self, base_kwargs: dict) -> None:
        from aiperf.plugin.enums import UIType

        config = _build(base_kwargs, verbose=True, request_count=100)
        assert config.runtime.ui == UIType.SIMPLE


class TestModelsConfig:
    """Model configuration."""

    def test_single_model(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, request_count=100)
        assert len(config.models.items) == 1
        assert config.models.items[0].name == "test-model"

    def test_multiple_models(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            model_names=["model-a", "model-b"],
            request_count=100,
        )
        assert len(config.models.items) == 2
        names = [m.name for m in config.models.items]
        assert "model-a" in names
        assert "model-b" in names


class TestEndpointConfig:
    """Endpoint configuration."""

    def test_streaming(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, streaming=True, request_count=100)
        assert config.endpoint.streaming is True

    def test_custom_endpoint(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            custom_endpoint="/v1/custom",
            request_count=100,
        )
        assert config.endpoint.path == "/v1/custom"

    def test_api_key(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            api_key="sk-test-key",
            request_count=100,
        )
        assert config.endpoint.api_key == "sk-test-key"


class TestArtifactsConfig:
    """Artifacts and output configuration."""

    def test_artifact_directory(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            artifact_directory=Path("/tmp/test-artifacts"),
            request_count=100,
        )
        assert config.artifacts.dir == Path("/tmp/test-artifacts")

    def test_benchmark_id_generated(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, request_count=100)
        assert config.artifacts.benchmark_id is not None
        assert len(config.artifacts.benchmark_id) > 0


class TestGpuTelemetry:
    """GPU telemetry configuration."""

    def test_gpu_telemetry_disabled(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, no_gpu_telemetry=True, request_count=100)
        assert not config.gpu_telemetry.enabled

    def test_gpu_telemetry_enabled_by_default(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, request_count=100)
        assert config.gpu_telemetry.enabled


class TestServerMetrics:
    """Server metrics configuration."""

    def test_server_metrics_disabled(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, no_server_metrics=True, request_count=100)
        assert not config.server_metrics.enabled

    def test_server_metrics_enabled_by_default(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, request_count=100)
        assert config.server_metrics.enabled


class TestCancellationConfig:
    """Request cancellation configuration."""

    def test_cancellation_rate(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            request_cancellation_rate=10.0,
            request_cancellation_delay=1.0,
            concurrency=4,
            request_count=100,
        )
        profiling = config.load["profiling"]
        assert profiling.cancellation is not None
        assert profiling.cancellation.rate == 10.0
        assert profiling.cancellation.delay == 1.0

    def test_no_cancellation_by_default(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, concurrency=4, request_count=100)
        profiling = config.load["profiling"]
        assert profiling.cancellation is None


class TestRampConfig:
    """Ramp configuration for gradual load increase."""

    def test_concurrency_ramp(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            concurrency=8,
            concurrency_ramp_duration=10.0,
            request_count=100,
        )
        profiling = config.load["profiling"]
        assert profiling.concurrency_ramp is not None
        assert profiling.concurrency_ramp.duration == 10.0

    def test_warmup_ramp_falls_back(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            warmup_request_count=10,
            concurrency=8,
            concurrency_ramp_duration=10.0,
            request_count=100,
        )
        warmup = config.load["warmup"]
        assert warmup.concurrency_ramp is not None
        assert warmup.concurrency_ramp.duration == 10.0
