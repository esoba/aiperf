# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark deployment and result collection for Kubernetes E2E tests."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import (
    JobSetStatus,
    KubectlClient,
    PodStatus,
    background_status,
)
from tests.kubernetes.helpers.log_streamer import PodLogStreamer
from tests.kubernetes.helpers.watchdog import BenchmarkWatchdog

logger = AIPerfLogger(__name__)


@asynccontextmanager
async def timed_operation(operation: str):
    """Context manager that logs timing information for an operation."""
    logger.info(f"[START] {operation}")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[DONE] {operation} ({elapsed:.2f}s)")


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    # Endpoint configuration
    endpoint_url: str = "http://aiperf-mock-server.default.svc.cluster.local:8000/v1"
    endpoint_type: str = "chat"
    model_name: str = "mock-model"

    # Load generation configuration
    concurrency: int = 5
    request_count: int = 50
    warmup_request_count: int = 5
    concurrency_ramp_duration: float | None = None

    # Tokenizer configuration
    tokenizer_name: str = "gpt2"

    # Dataset configuration
    input_sequence_min: int = 50
    input_sequence_max: int = 100
    output_tokens_min: int = 10
    output_tokens_max: int = 50

    # Kubernetes configuration
    image: str = "aiperf:local"
    workers: int = 1

    # Kueue / gang-scheduling
    queue_name: str | None = None
    priority_class: str | None = None

    def to_yaml(self) -> str:
        """Convert config to YAML string."""
        config = {
            "endpoint": {
                "model_names": [self.model_name],
                "urls": [self.endpoint_url],
                "type": self.endpoint_type,
            },
            "loadgen": {
                "concurrency": self.concurrency,
                "request_count": self.request_count,
                "warmup_request_count": self.warmup_request_count,
            },
            "tokenizer": {
                "name": self.tokenizer_name,
            },
            "dataset": {
                "sequence_length": {
                    "distribution": "uniform",
                    "min": self.input_sequence_min,
                    "max": self.input_sequence_max,
                },
                "output_tokens": {
                    "distribution": "uniform",
                    "min": self.output_tokens_min,
                    "max": self.output_tokens_max,
                },
            },
        }
        return yaml.dump(config, default_flow_style=False)

    def to_temp_file(self) -> Path:
        """Write config to a temporary file.

        Returns:
            Path to the temporary config file.
        """
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="benchmark-config-")
        try:
            os.write(fd, self.to_yaml().encode())
        finally:
            os.close(fd)
        return Path(path)


@dataclass
class BenchmarkMetrics:
    """Extracted benchmark metrics."""

    request_throughput: float | None = None
    output_token_throughput: float | None = None
    request_count: int | None = None
    request_latency_avg: float | None = None
    request_latency_min: float | None = None
    request_latency_max: float | None = None
    request_latency_p50: float | None = None
    request_latency_p90: float | None = None
    request_latency_p99: float | None = None
    input_sequence_length: float | None = None
    output_sequence_length_avg: float | None = None
    error_count: int = 0
    raw_logs: str = ""

    @classmethod
    def from_api_results(cls, api_results: dict[str, Any]) -> BenchmarkMetrics:
        """Parse metrics from /api/results response.

        Args:
            api_results: JSON response from /api/results endpoint.

        Returns:
            Extracted metrics.
        """
        metrics = cls()

        inner = api_results.get("results", {})
        if isinstance(inner, dict):
            inner = inner.get("results", inner)
        if not isinstance(inner, dict):
            return metrics

        records = inner.get("records", [])
        if not isinstance(records, list):
            return metrics

        # Map header/tag names to metric attributes
        tag_map: dict[str, str] = {
            "request_throughput": "request_throughput",
            "output_token_throughput": "output_token_throughput",
            "request_count": "request_count",
            "request_latency": "request_latency_avg",
            "error_request_count": "error_count",
        }

        for rec in records:
            tag = rec.get("tag", "")
            attr = tag_map.get(tag)
            if attr is None:
                continue
            avg = rec.get("avg")
            if avg is None:
                continue
            if attr in ("request_count", "error_count"):
                setattr(metrics, attr, int(avg))
            else:
                setattr(metrics, attr, float(avg))

            # Extract latency percentiles
            if tag == "request_latency":
                for key in ("min", "max", "p50", "p90", "p99"):
                    val = rec.get(key)
                    if val is not None:
                        setattr(metrics, f"request_latency_{key}", float(val))

        return metrics

    @classmethod
    def from_logs(cls, logs: str) -> BenchmarkMetrics:
        """Parse metrics from system-controller logs (fallback).

        Args:
            logs: Raw log content.

        Returns:
            Extracted metrics.
        """
        metrics = cls(raw_logs=logs)

        # Strip ANSI codes
        clean_logs = re.sub(r"\x1b\[[0-9;]*m", "", logs)

        # Parse metrics table using regex
        patterns = {
            "request_throughput": r"Request Throughput.*?│\s*([\d,]+\.?\d*)",
            "output_token_throughput": r"Output Token Throughput.*?│\s*([\d,]+\.?\d*)",
            "request_count": r"Request Count.*?│\s*([\d,]+\.?\d*)",
            "request_latency_avg": r"Request Latency.*?│\s*([\d,]+\.?\d*)",
        }

        for attr, pattern in patterns.items():
            match = re.search(pattern, clean_logs)
            if match:
                value_str = match.group(1).replace(",", "")
                try:
                    if attr == "request_count":
                        setattr(metrics, attr, int(float(value_str)))
                    else:
                        setattr(metrics, attr, float(value_str))
                except ValueError:
                    pass

        # Parse error count from "Errors: X / Y" pattern
        error_match = re.search(r"Errors:\s*(\d+)\s*/\s*\d+", clean_logs)
        if error_match:
            metrics.error_count = int(error_match.group(1))

        return metrics


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    namespace: str
    jobset_name: str
    job_id: str
    config: BenchmarkConfig
    status: JobSetStatus | None = None
    metrics: BenchmarkMetrics | None = None
    api_results: dict[str, Any] | None = None
    pods: list[PodStatus] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = False
    error_message: str | None = None

    @property
    def controller_pod(self) -> PodStatus | None:
        """Get the controller pod."""
        for pod in self.pods:
            if "controller" in pod.name:
                return pod
        return None

    @property
    def worker_pods(self) -> list[PodStatus]:
        """Get worker pods."""
        return [
            p for p in self.pods if "worker" in p.name and "controller" not in p.name
        ]

    def print_results(self, header: str = "BENCHMARK RESULTS") -> None:
        """Print a formatted summary of the benchmark results.

        Args:
            header: Header text for the output block.
        """
        sep = "=" * 70
        thin = "-" * 70
        print(f"\n{sep}")
        print(header)
        print(sep)
        print(f"  Job ID:    {self.job_id}")
        print(f"  Namespace: {self.namespace}")
        print(f"  Success:   {self.success}")
        print(f"  Duration:  {self.duration_seconds:.2f}s")
        if self.error_message:
            print(f"  Error:     {self.error_message}")
        if self.status:
            print(f"  JobSet:    {self.status.terminal_state}")

        if self.api_results:
            self._print_api_results(thin)

        if self.metrics:
            print(f"\n  {thin}")
            print("  PARSED METRICS (from controller logs)")
            print(f"  {thin}")
            self._print_metric("Request Count", self.metrics.request_count, "")
            self._print_metric(
                "Request Throughput", self.metrics.request_throughput, "req/s"
            )
            self._print_metric(
                "Output Token Throughput",
                self.metrics.output_token_throughput,
                "tok/s",
            )
            self._print_metric(
                "Request Latency Avg", self.metrics.request_latency_avg, "ms"
            )
            self._print_metric("Error Count", self.metrics.error_count, "")

        print(sep + "\n")

    def _print_api_results(self, thin: str) -> None:
        """Print parsed API results in a readable table format."""
        results = self.api_results
        if not results:
            return

        inner = results.get("results", {})
        if isinstance(inner, dict):
            inner = inner.get("results", inner)

        print(f"\n  {thin}")
        print(f"  API RESULTS (status={results.get('status', 'unknown')})")
        print(f"  {thin}")

        # Print metric records as a table
        records = inner.get("records", []) if isinstance(inner, dict) else []
        if records:
            hdr = f"  {'Metric':<30} {'Avg':>12} {'P50':>12} {'P99':>12} {'Min':>12} {'Max':>12} {'Unit':<8}"
            print(hdr)
            print(f"  {'-' * (len(hdr) - 2)}")
            for rec in records:
                tag = rec.get("header", rec.get("tag", "?"))
                unit = rec.get("unit", "")
                avg = rec.get("avg")
                p50 = rec.get("p50")
                p99 = rec.get("p99")
                mn = rec.get("min")
                mx = rec.get("max")
                print(
                    f"  {tag:<30} "
                    f"{self._fmt(avg):>12} "
                    f"{self._fmt(p50):>12} "
                    f"{self._fmt(p99):>12} "
                    f"{self._fmt(mn):>12} "
                    f"{self._fmt(mx):>12} "
                    f"{unit:<8}"
                )

        # Print summary fields
        # NOTE: "completed" is the number of metric record types, not requests.
        # "total_expected" is the configured request count.
        summary_fields = [
            ("completed", "Metric Records"),
            ("total_expected", "Total Expected"),
            ("was_cancelled", "Was Cancelled"),
        ]
        printed_summary = False
        for key, label in summary_fields:
            val = inner.get(key) if isinstance(inner, dict) else None
            if val is not None:
                if not printed_summary:
                    print()
                    printed_summary = True
                print(f"  {label:<30} {val}")

        # Print errors
        errors = results.get("errors", [])
        if not errors and isinstance(inner, dict):
            errors = inner.get("error_summary", [])
        if errors:
            print(f"\n  ERRORS ({len(errors)}):")
            for err in errors[:10]:
                print(f"    - {err}")

    @staticmethod
    def _fmt(val: float | int | None) -> str:
        """Format a numeric value for display."""
        if val is None:
            return "N/A"
        if isinstance(val, int):
            return f"{val:,}"
        if abs(val) >= 1000:
            return f"{val:,.2f}"
        if abs(val) >= 1:
            return f"{val:.4f}"
        return f"{val:.6f}"

    @staticmethod
    def _print_metric(label: str, value: float | int | None, unit: str) -> None:
        """Print a single metric line."""
        if value is None:
            print(f"    {label + ':':<30} N/A")
        elif isinstance(value, int):
            print(f"    {label + ':':<30} {value:,} {unit}")
        else:
            print(f"    {label + ':':<30} {value:,.2f} {unit}")


class BenchmarkDeployer:
    """Deploys and manages AIPerf benchmarks on Kubernetes."""

    def __init__(
        self,
        kubectl: KubectlClient,
        project_root: Path,
        default_image: str = "aiperf:local",
    ) -> None:
        """Initialize benchmark deployer.

        Args:
            kubectl: Kubectl client.
            project_root: Path to project root.
            default_image: Default image to use for benchmarks.
        """
        self.kubectl = kubectl
        self.project_root = project_root
        self.default_image = default_image
        self._deployments: list[BenchmarkResult] = []

    async def deploy(
        self,
        config: BenchmarkConfig,
        wait_for_completion: bool = True,
        timeout: int = 300,
        stream_logs: bool = False,
        pre_wait_hook: Any | None = None,
    ) -> BenchmarkResult:
        """Deploy a benchmark.

        Args:
            config: Benchmark configuration.
            wait_for_completion: Wait for benchmark to complete.
            timeout: Timeout in seconds.
            stream_logs: If True, stream pod logs in the background.
            pre_wait_hook: Optional async callable(namespace) invoked after
                the manifest is applied but before waiting for completion.
                Useful for Kueue tests that need to create a LocalQueue in the
                dynamically-created benchmark namespace.

        Returns:
            Benchmark result.
        """
        start_time = time.time()
        logger.info(
            f"[DEPLOY] Starting benchmark: concurrency={config.concurrency}, "
            f"requests={config.request_count}, image={config.image}"
        )

        # Write config to temp file
        config_path = config.to_temp_file()

        try:
            # Generate manifest using aiperf kube deploy --dry-run
            async with timed_operation("Generating Kubernetes manifest"):
                manifest = await self._generate_manifest(config, config_path)
                logger.debug(
                    lambda manifest=manifest: f"Generated manifest ({len(manifest)} bytes)"
                )

            # Patch imagePullPolicy for minikube (locally loaded images)
            manifest = self._patch_image_pull_policy(manifest, config.image)

            # Apply manifest and extract namespace
            async with timed_operation("Applying manifest to cluster"):
                output = await self.kubectl.apply(manifest)
                namespace = self._extract_namespace(output)

            if not namespace:
                raise RuntimeError("Failed to extract namespace from deployment output")

            logger.info(f"[DEPLOY] Created namespace: {namespace}")

            # Get JobSet name
            jobsets = await self.kubectl.get_jobsets(namespace)
            if not jobsets:
                raise RuntimeError(f"No JobSet found in namespace {namespace}")

            jobset_name = jobsets[0].name
            # Extract job_id from namespace (format: aiperf-{job_id})
            job_id = namespace.removeprefix("aiperf-")

            result = BenchmarkResult(
                namespace=namespace,
                jobset_name=jobset_name,
                job_id=job_id,
                config=config,
            )

            logger.info(
                f"[DEPLOY] Benchmark deployed: namespace={namespace}, jobset={jobset_name}"
            )

            if pre_wait_hook is not None:
                await pre_wait_hook(namespace)

            if wait_for_completion:
                async with PodLogStreamer(
                    self.kubectl, namespace, prefix="BENCH"
                ) as streamer:
                    if stream_logs:
                        streamer.watch()
                    async with timed_operation(
                        f"Waiting for benchmark completion (timeout={timeout}s)"
                    ):
                        result = await self._wait_and_collect(result, timeout)

            result.duration_seconds = time.time() - start_time
            logger.info(
                f"[DEPLOY] Total deployment time: {result.duration_seconds:.2f}s"
            )
            if wait_for_completion:
                result.print_results()
            self._deployments.append(result)
            return result

        finally:
            # Clean up temp config file
            config_path.unlink(missing_ok=True)

    async def _generate_manifest(
        self, config: BenchmarkConfig, config_path: Path
    ) -> str:
        """Generate Kubernetes manifest using aiperf CLI.

        Args:
            config: Benchmark configuration.
            config_path: Path to config file (not used, kept for signature compat).

        Returns:
            YAML manifest string.
        """
        cmd = [
            "uv",
            "run",
            "aiperf",
            "kube",
            "generate",
            "--model",
            config.model_name,
            "--url",
            config.endpoint_url,
            "--endpoint-type",
            config.endpoint_type,
            "--image",
            config.image,
            "--concurrency",
            str(config.concurrency),
            "--request-count",
            str(config.request_count),
            "--warmup-request-count",
            str(config.warmup_request_count),
            "--tokenizer",
            config.tokenizer_name,
            "--workers-max",
            str(config.workers),
            "--ui",
            "none",
        ]

        if config.concurrency_ramp_duration is not None:
            cmd.extend(
                ["--concurrency-ramp-duration", str(config.concurrency_ramp_duration)]
            )

        if config.queue_name is not None:
            cmd.extend(["--queue-name", config.queue_name])

        if config.priority_class is not None:
            cmd.extend(["--priority-class", config.priority_class])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.project_root),
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Failed to generate manifest: {stderr.decode()}")

        output = stdout.decode()
        # Strip any non-YAML prefix lines (e.g. warnings printed to stdout).
        # YAML manifests always start with "apiVersion:" or "---".
        lines = output.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("apiVersion:") or line.startswith("---"):
                return "\n".join(lines[i:])
        return output

    def _patch_image_pull_policy(self, manifest: str, image: str) -> str:
        """Patch manifest to use imagePullPolicy: Never for minikube.

        Args:
            manifest: YAML manifest.
            image: Image name to patch.

        Returns:
            Patched manifest.
        """
        return manifest.replace(
            f"image: {image}",
            f"image: {image}\n              imagePullPolicy: Never",
        )

    def _extract_namespace(self, apply_output: str) -> str | None:
        """Extract namespace from kubectl apply output.

        Args:
            apply_output: Output from kubectl apply.

        Returns:
            Namespace name or None.
        """
        for line in apply_output.splitlines():
            if line.startswith("namespace/"):
                parts = line.split()
                if parts:
                    return parts[0].replace("namespace/", "")
        return None

    async def _wait_and_collect(
        self,
        result: BenchmarkResult,
        timeout: int,
    ) -> BenchmarkResult:
        """Wait for benchmark completion and collect results.

        Flow:
        1. Wait for the controller pod to be running
        2. Port-forward to the API and poll /api/results until complete
        3. Call POST /api/shutdown so the controller pod exits
        4. Wait for the JobSet to reach Completed state

        Args:
            result: Partial benchmark result.
            timeout: Timeout in seconds.

        Returns:
            Updated benchmark result with metrics.
        """
        start_time = time.time()

        async with (
            BenchmarkWatchdog(
                self.kubectl,
                result.namespace,
                timeout=timeout,
                poll_interval=5.0,
                pending_threshold=30.0,
            ) as _watchdog,
            background_status(
                self.kubectl, result.namespace, label="BENCH", interval=15
            ),
        ):
            try:
                controller_pod = await self._wait_for_controller_pod(
                    result.namespace, timeout=min(120, timeout)
                )

                if not controller_pod:
                    result.success = False
                    result.error_message = "Controller pod not found"
                    return result

                remaining = max(30, timeout - int(time.time() - start_time))
                api_results = await self.kubectl.wait_for_benchmark_api(
                    pod=controller_pod.name,
                    namespace=result.namespace,
                    timeout=remaining,
                )
                result.api_results = api_results

                remaining = max(30, timeout - int(time.time() - start_time))
                status = await self.kubectl.wait_for_jobset_completion(
                    result.jobset_name,
                    result.namespace,
                    timeout=remaining,
                )
                result.status = status
                result.success = status.is_completed

            except TimeoutError as e:
                result.success = False
                result.error_message = str(e)
                logger.error(f"Benchmark timeout: {e}")

            except RuntimeError as e:
                result.success = False
                result.error_message = str(e)
                logger.error(f"Benchmark failed: {e}")

        # Collect final pods
        result.pods = await self.kubectl.get_pods(result.namespace)

        # Parse metrics: prefer API results, fall back to log parsing
        if result.api_results:
            result.metrics = BenchmarkMetrics.from_api_results(result.api_results)

        # Always try to capture raw logs from the controller pod
        controller_pod_status = result.controller_pod
        if controller_pod_status:
            logs = await self.kubectl.get_logs(
                controller_pod_status.name,
                container="control-plane",
                namespace=result.namespace,
            )
            if result.metrics is None or result.metrics.request_count is None:
                result.metrics = BenchmarkMetrics.from_logs(logs)
            elif result.metrics is not None:
                result.metrics.raw_logs = logs

        return result

    async def _wait_for_controller_pod(
        self,
        namespace: str,
        timeout: int = 120,
    ) -> PodStatus | None:
        """Wait for the controller pod to be running.

        Args:
            namespace: Kubernetes namespace.
            timeout: Timeout in seconds.

        Returns:
            Controller PodStatus or None if not found.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            pods = await self.kubectl.get_pods(namespace)
            for pod in pods:
                if "controller" in pod.name and pod.phase == "Running":
                    logger.info(
                        f"Controller pod ready: {pod.name} (waited {elapsed:.0f}s)"
                    )
                    return pod

            logger.info(
                f"Waiting for controller pod ({int(elapsed)}s, {len(pods)} pods exist)..."
            )
            await asyncio.sleep(3)
        logger.error(f"Controller pod not found in {namespace} within {timeout}s")
        return None

    async def cleanup(self, result: BenchmarkResult) -> None:
        """Clean up a benchmark deployment.

        Args:
            result: Benchmark result to clean up.
        """
        logger.info(f"Cleaning up namespace: {result.namespace}")
        await self.kubectl.delete_namespace(result.namespace, wait=True)

    async def cleanup_all(self) -> None:
        """Clean up all deployed benchmarks."""
        for result in self._deployments:
            try:
                await self.cleanup(result)
            except Exception as e:
                logger.warning(f"Failed to cleanup {result.namespace}: {e}")

        self._deployments.clear()

    def get_deployment_count(self) -> int:
        """Get number of active deployments.

        Returns:
            Number of deployments.
        """
        return len(self._deployments)
