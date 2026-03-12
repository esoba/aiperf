# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for edge cases and error handling."""

from __future__ import annotations

import pytest

from tests.kubernetes.helpers.benchmark import (
    BenchmarkConfig,
    BenchmarkDeployer,
    BenchmarkResult,
)
from tests.kubernetes.helpers.kubectl import KubectlClient


class TestMinimalConfigurations:
    """Tests for minimal/edge case configurations."""

    @pytest.mark.asyncio
    async def test_single_request(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with single request."""
        config = BenchmarkConfig(
            concurrency=1,
            request_count=1,
            warmup_request_count=1,
        )

        result = await benchmark_deployer.deploy(config, timeout=300)

        assert result.success
        assert result.metrics is not None
        assert result.metrics.request_count == 1

    @pytest.mark.asyncio
    async def test_concurrency_one(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with concurrency of 1."""
        config = BenchmarkConfig(
            concurrency=1,
            request_count=10,
            warmup_request_count=2,
        )

        result = await benchmark_deployer.deploy(config, timeout=300)

        assert result.success
        assert result.metrics is not None
        assert result.metrics.request_count == 10

    @pytest.mark.asyncio
    async def test_minimal_warmup(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with minimal warmup (1 request)."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=1,
        )

        result = await benchmark_deployer.deploy(config, timeout=300)

        assert result.success
        assert result.metrics is not None

    @pytest.mark.asyncio
    async def test_minimal_sequence_length(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with minimal sequence lengths."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
            input_sequence_min=10,
            input_sequence_max=20,
            output_tokens_min=5,
            output_tokens_max=10,
        )

        result = await benchmark_deployer.deploy(config, timeout=300)

        assert result.success


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_config_to_yaml_generates_valid_yaml(self) -> None:
        """Verify config generates valid YAML."""
        import yaml

        config = BenchmarkConfig(
            concurrency=5,
            request_count=50,
        )

        yaml_str = config.to_yaml()
        parsed = yaml.safe_load(yaml_str)

        assert parsed["loadgen"]["concurrency"] == 5
        assert parsed["loadgen"]["request_count"] == 50
        assert parsed["endpoint"]["type"] == "chat"
        assert parsed["tokenizer"]["name"] == "gpt2"

    def test_config_temp_file_is_created(self) -> None:
        """Verify config temp file is created properly."""
        config = BenchmarkConfig()
        path = config.to_temp_file()

        assert path.exists()
        assert path.suffix == ".yaml"

        # Cleanup
        path.unlink()


class TestResourceCleanup:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_after_successful_run(
        self,
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify resources are cleaned up after successful run."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
        )

        result = await benchmark_deployer.deploy(config, timeout=300)
        namespace = result.namespace

        # Namespace should exist
        assert await kubectl.namespace_exists(namespace)

        # Cleanup
        await benchmark_deployer.cleanup(result)

        # Namespace should be gone
        assert not await kubectl.namespace_exists(namespace)

    @pytest.mark.asyncio
    async def test_multiple_deployments_tracked(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Verify multiple deployments are tracked."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
        )

        initial_count = benchmark_deployer.get_deployment_count()

        await benchmark_deployer.deploy(config, timeout=300)
        assert benchmark_deployer.get_deployment_count() == initial_count + 1

        await benchmark_deployer.deploy(config, timeout=300)
        assert benchmark_deployer.get_deployment_count() == initial_count + 2


class TestDiagnostics:
    """Tests for diagnostic information collection."""

    def test_pods_collected_on_completion(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify pod information is collected."""
        assert len(deployed_small_benchmark_module.pods) > 0

    def test_controller_pod_accessible(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify controller pod is accessible."""
        controller = deployed_small_benchmark_module.controller_pod
        assert controller is not None
        assert "controller" in controller.name

    def test_worker_pods_completed_after_benchmark(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify worker pods have completed after benchmark finishes."""
        workers = deployed_small_benchmark_module.worker_pods
        for worker in workers:
            assert worker.phase == "Succeeded", (
                f"Worker pod {worker.name} in unexpected phase: {worker.phase}"
            )

    @pytest.mark.asyncio
    async def test_can_get_logs_from_control_plane_container(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
        get_pod_logs,
    ) -> None:
        """Verify logs can be retrieved from the control-plane container.

        New architecture: single control-plane container spawns all services
        as subprocesses.
        """
        logs = await get_pod_logs(
            deployed_small_benchmark_module, container="control-plane"
        )
        assert len(logs) > 0


class TestKubectlHelpers:
    """Tests for kubectl helper functions."""

    @pytest.mark.asyncio
    async def test_get_events(
        self,
        kubectl: KubectlClient,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify events can be retrieved."""
        events = await kubectl.get_events(deployed_small_benchmark_module.namespace)
        # Events string should not be empty
        assert events is not None

    @pytest.mark.asyncio
    async def test_pod_status_parsing(
        self,
        kubectl: KubectlClient,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify pod status is parsed correctly."""
        pods = await kubectl.get_pods(deployed_small_benchmark_module.namespace)

        for pod in pods:
            assert pod.name
            assert pod.namespace == deployed_small_benchmark_module.namespace
            assert pod.phase in [
                "Running",
                "Succeeded",
                "Completed",
                "Failed",
                "Pending",
                "Unknown",
            ]
            assert "/" in pod.ready  # Format: "X/Y"

    @pytest.mark.asyncio
    async def test_jobset_status_parsing(
        self,
        kubectl: KubectlClient,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify JobSet status is parsed correctly."""
        status = await kubectl.get_jobset(
            deployed_small_benchmark_module.jobset_name,
            deployed_small_benchmark_module.namespace,
        )

        assert status.name == deployed_small_benchmark_module.jobset_name
        assert status.namespace == deployed_small_benchmark_module.namespace
        assert status.is_completed
