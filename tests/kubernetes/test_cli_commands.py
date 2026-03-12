# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf kube CLI commands (status, logs, results, cancel, delete)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pytest import param

from tests.kubernetes.helpers.benchmark import (
    BenchmarkConfig,
    BenchmarkDeployer,
    BenchmarkResult,
)
from tests.kubernetes.helpers.kubectl import KubectlClient


async def run_aiperf_command(
    project_root: Path,
    *args: str,
    timeout: int = 30,
    check: bool = False,
) -> asyncio.subprocess.Process:
    """Run an aiperf CLI command.

    Args:
        project_root: Path to project root.
        *args: Command arguments after 'aiperf'.
        timeout: Command timeout in seconds.
        check: Whether to raise on non-zero exit.

    Returns:
        Completed process result with stdout, stderr, and returncode attributes.
    """
    cmd = ["uv", "run", "aiperf", *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=project_root,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise

    proc.stdout = stdout_bytes.decode() if stdout_bytes else ""
    proc.stderr = stderr_bytes.decode() if stderr_bytes else ""

    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {proc.stderr}"
        )

    return proc


class TestKubeListCommand:
    """Tests for aiperf kube list command."""

    @pytest.mark.asyncio
    async def test_list_command_shows_running_benchmark(
        self,
        benchmark_deployer: BenchmarkDeployer,
        project_root: Path,
        small_benchmark_config: BenchmarkConfig,
    ) -> None:
        """Verify list command shows running benchmarks."""
        result = await benchmark_deployer.deploy(
            small_benchmark_config, wait_for_completion=False, timeout=60
        )

        # Wait for pods to be created
        await asyncio.sleep(10)

        # Run list command
        list_result = await run_aiperf_command(
            project_root,
            "kube",
            "list",
            "--namespace",
            result.namespace,
            timeout=30,
        )

        print(f"\n{'=' * 60}")
        print("KUBE LIST OUTPUT")
        print(f"{'=' * 60}")
        print(list_result.stdout)
        if list_result.stderr:
            print(f"STDERR: {list_result.stderr}")
        print(f"{'=' * 60}\n")

        # List command should succeed
        assert list_result.returncode == 0, f"List failed: {list_result.stderr}"

        await benchmark_deployer.cleanup(result)

    @pytest.mark.asyncio
    async def test_list_command_all_namespaces(
        self,
        benchmark_deployer: BenchmarkDeployer,
        project_root: Path,
        small_benchmark_config: BenchmarkConfig,
    ) -> None:
        """Verify list command works with --all-namespaces."""
        result = await benchmark_deployer.deploy(
            small_benchmark_config, wait_for_completion=False, timeout=60
        )

        # Wait for pods to be created
        await asyncio.sleep(10)

        # Run list with --all-namespaces
        list_result = await run_aiperf_command(
            project_root,
            "kube",
            "list",
            "--all-namespaces",
            timeout=30,
        )

        assert list_result.returncode == 0, f"List failed: {list_result.stderr}"

        await benchmark_deployer.cleanup(result)


class TestKubeLogsCommand:
    """Tests for aiperf kube logs command."""

    @pytest.mark.asyncio
    async def test_logs_command_retrieves_logs(
        self,
        benchmark_deployer: BenchmarkDeployer,
        project_root: Path,
        small_benchmark_config: BenchmarkConfig,
    ) -> None:
        """Verify logs command retrieves pod logs."""
        result = await benchmark_deployer.deploy(
            small_benchmark_config, wait_for_completion=False, timeout=60
        )

        # Wait for pods to start generating logs
        await asyncio.sleep(15)

        # Run logs command with explicit namespace
        logs_result = await run_aiperf_command(
            project_root,
            "kube",
            "logs",
            "--job-id",
            result.job_id,
            "--namespace",
            result.namespace,
            "--tail",
            "50",
            timeout=30,
        )

        print(f"\n{'=' * 60}")
        print("KUBE LOGS OUTPUT (first 500 chars)")
        print(f"{'=' * 60}")
        print(logs_result.stdout[:500] if logs_result.stdout else "(empty)")
        print(f"{'=' * 60}\n")

        # Logs command should succeed (exit code 0)
        assert logs_result.returncode == 0, f"Logs failed: {logs_result.stderr}"

        await benchmark_deployer.cleanup(result)


class TestKubeResultsCommand:
    """Tests for aiperf kube results command."""

    @pytest.mark.asyncio
    async def test_results_command_after_completion(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
        project_root: Path,
    ) -> None:
        """Verify results command retrieves benchmark results."""
        result = deployed_small_benchmark_module

        # Skip if benchmark failed
        if not result.success:
            pytest.skip(f"Benchmark failed: {result.error_message}")

        # Run results command
        results_output = await run_aiperf_command(
            project_root,
            "kube",
            "results",
            "--job-id",
            result.job_id,
            "--namespace",
            result.namespace,
            timeout=30,
        )

        print(f"\n{'=' * 60}")
        print("KUBE RESULTS OUTPUT")
        print(f"{'=' * 60}")
        print(results_output.stdout[:1000] if results_output.stdout else "(empty)")
        print(f"{'=' * 60}\n")

        # Results command should succeed
        assert results_output.returncode == 0, (
            f"Results failed: {results_output.stderr}"
        )


class TestKubeDeleteCommand:
    """Tests for aiperf kube delete command."""

    @pytest.mark.asyncio
    async def test_delete_command_removes_benchmark(
        self,
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
        project_root: Path,
    ) -> None:
        """Verify delete command removes benchmark resources."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
        )

        result = await benchmark_deployer.deploy(
            config, wait_for_completion=True, timeout=300
        )
        namespace = result.namespace
        jobset_name = result.jobset_name

        # Verify namespace exists
        assert await kubectl.namespace_exists(namespace)

        # Extract job_id from namespace (format: aiperf-{job_id})
        job_id = namespace.removeprefix("aiperf-")

        # Run delete command with --force to skip confirmation
        delete_result = await run_aiperf_command(
            project_root,
            "kube",
            "delete",
            "--job-id",
            job_id,
            "--namespace",
            namespace,
            "--force",
            timeout=60,
        )

        print(f"\n{'=' * 60}")
        print("KUBE DELETE OUTPUT")
        print(f"{'=' * 60}")
        print(delete_result.stdout)
        if delete_result.stderr:
            print(f"STDERR: {delete_result.stderr}")
        print(f"{'=' * 60}\n")

        # Delete command should succeed
        assert delete_result.returncode == 0, f"Delete failed: {delete_result.stderr}"

        # Wait for deletion to complete
        await asyncio.sleep(5)

        # Verify JobSet is gone
        jobsets = await kubectl.get_jobsets(namespace)
        assert len(jobsets) == 0 or all(js.name != jobset_name for js in jobsets), (
            "JobSet should be deleted"
        )


class TestKubeGenerateCommand:
    """Tests for aiperf kube generate command."""

    @pytest.mark.asyncio
    async def test_generate_outputs_valid_yaml(self, project_root: Path) -> None:
        """Verify generate command outputs valid YAML manifests."""
        import yaml

        result = await run_aiperf_command(
            project_root,
            "kube",
            "generate",
            "--model",
            "test-model",
            "--url",
            "http://server:8000/v1",
            "--image",
            "aiperf:test",
            "--concurrency",
            "5",
            "--request-count",
            "10",
            timeout=30,
        )

        assert result.returncode == 0, f"Generate failed: {result.stderr}"

        # Parse YAML output
        manifests = list(yaml.safe_load_all(result.stdout))
        assert len(manifests) > 0

        # Check required resource types
        kinds = {m["kind"] for m in manifests if m}
        assert "ConfigMap" in kinds
        assert "JobSet" in kinds
        assert "Role" in kinds
        assert "RoleBinding" in kinds

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "workers,expected_replicas",
        [
            param(1, 1, id="1-worker-1-pod"),
            param(5, 1, id="5-workers-1-pod"),
            param(10, 1, id="10-workers-1-pod"),
        ],
    )  # fmt: skip
    async def test_generate_respects_workers_flag(
        self, project_root: Path, workers: int, expected_replicas: int
    ) -> None:
        """Verify generate command distributes workers across pods correctly.

        --workers-max sets total workers, distributed across pods based on
        --workers-per-pod (default 10). With <=10 workers, all fit in 1 pod.
        """
        import yaml

        result = await run_aiperf_command(
            project_root,
            "kube",
            "generate",
            "--model",
            "test-model",
            "--url",
            "http://server:8000/v1",
            "--image",
            "aiperf:test",
            "--workers-max",
            str(workers),
            timeout=30,
        )

        assert result.returncode == 0

        # Find JobSet manifest
        manifests = list(yaml.safe_load_all(result.stdout))
        jobset = next((m for m in manifests if m and m.get("kind") == "JobSet"), None)
        assert jobset is not None

        # Find workers replicated job
        worker_job = next(
            (j for j in jobset["spec"]["replicatedJobs"] if j["name"] == "workers"),
            None,
        )
        assert worker_job is not None
        assert worker_job["replicas"] == expected_replicas


class TestKubePreflightCommand:
    """Tests for aiperf kube preflight command."""

    @pytest.mark.asyncio
    async def test_preflight_checks_cluster_access(
        self, project_root: Path, k8s_ready
    ) -> None:
        """Verify preflight command checks cluster access."""
        result = await run_aiperf_command(
            project_root,
            "kube",
            "preflight",
            timeout=60,
        )

        print(f"\n{'=' * 60}")
        print("KUBE PREFLIGHT OUTPUT")
        print(f"{'=' * 60}")
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        print(f"{'=' * 60}\n")

        # Preflight should succeed when cluster is ready
        assert result.returncode == 0, f"Preflight failed: {result.stderr}"
        assert "cluster" in result.stdout.lower() or "pass" in result.stdout.lower()


class TestKubeCommandHelp:
    """Tests for CLI help messages."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "subcommand",
        [
            param("profile", id="profile"),
            param("status", id="status"),
            param("logs", id="logs"),
            param("results", id="results"),
            param("delete", id="delete"),
            param("generate", id="generate"),
            param("preflight", id="preflight"),
        ],
    )  # fmt: skip
    async def test_subcommand_help_available(
        self, project_root: Path, subcommand: str
    ) -> None:
        """Verify help is available for each subcommand."""
        result = await run_aiperf_command(
            project_root,
            "kube",
            subcommand,
            "--help",
            timeout=10,
        )

        assert result.returncode == 0, f"Help failed: {result.stderr}"
        assert subcommand in result.stdout.lower()
