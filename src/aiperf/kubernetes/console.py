# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logging utilities for Kubernetes CLI commands."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import orjson
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.kubernetes.enums import JobSetStatus, PodPhase

if TYPE_CHECKING:
    from collections.abc import Generator

    from aiperf.kubernetes.models import JobSetInfo, PodSummary

# Logger for kube CLI output, configured with RichHandler for colored output
logger = AIPerfLogger("aiperf.kube")

if not logger.handlers:
    _handler = RichHandler(
        show_time=False,
        show_level=False,
        show_path=False,
        markup=True,
        rich_tracebacks=False,
    )
    _handler.setLevel(logging.DEBUG)
    logger.addHandler(_handler)
    logger.set_level(logging.DEBUG)
    logger._logger.propagate = False

# Console kept only for structured output (tables)
console = Console()
stderr_console = Console(stderr=True)

# Separate logger for stderr output (deployment summary when piped)
_stderr_logger = AIPerfLogger("aiperf.kube.stderr")
if not _stderr_logger.handlers:
    _stderr_handler = RichHandler(
        show_time=False,
        show_level=False,
        show_path=False,
        markup=True,
        rich_tracebacks=False,
        console=stderr_console,
    )
    _stderr_handler.setLevel(logging.DEBUG)
    _stderr_logger.addHandler(_stderr_handler)
    _stderr_logger.set_level(logging.DEBUG)
    _stderr_logger._logger.propagate = False

# Path to store the last benchmark info
_LAST_BENCHMARK_FILE = Path.home() / ".aiperf" / "last_kube_benchmark.json"

# Rich styles for JobSet status display
_STATUS_STYLES: dict[PodPhase, str] = {
    PodPhase.RUNNING: "bold green",
    PodPhase.SUCCEEDED: "green",
    PodPhase.FAILED: "red",
    PodPhase.UNKNOWN: "yellow",
}


@dataclass(slots=True)
class LastBenchmarkInfo:
    """Information about the last used benchmark."""

    job_id: str
    """The AIPerf job identifier."""

    namespace: str
    """The Kubernetes namespace where the job was deployed."""

    name: str | None = None
    """Optional human-readable name for the benchmark."""


def save_last_benchmark(job_id: str, namespace: str, name: str | None = None) -> None:
    """Save the last used benchmark info.

    Args:
        job_id: The job ID.
        namespace: The Kubernetes namespace.
        name: Optional human-readable name.
    """
    _LAST_BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, str] = {"job_id": job_id, "namespace": namespace}
    if name:
        data["name"] = name
    _LAST_BENCHMARK_FILE.write_bytes(orjson.dumps(data))


def get_last_benchmark() -> LastBenchmarkInfo | None:
    """Get the last used benchmark info.

    Returns:
        LastBenchmarkInfo if available, None otherwise.
    """
    if not _LAST_BENCHMARK_FILE.exists():
        return None
    try:
        data = orjson.loads(_LAST_BENCHMARK_FILE.read_bytes())
        return LastBenchmarkInfo(
            job_id=data["job_id"],
            namespace=data["namespace"],
            name=data.get("name"),
        )
    except (orjson.JSONDecodeError, KeyError, OSError):
        return None


def clear_last_benchmark() -> None:
    """Clear the last benchmark info (e.g., after deletion)."""
    if _LAST_BENCHMARK_FILE.exists():
        _LAST_BENCHMARK_FILE.unlink()


def print_step(message: str, step: int | None = None, total: int | None = None) -> None:
    """Log a step message with optional step counter.

    Args:
        message: The message to display.
        step: Current step number (optional).
        total: Total number of steps (optional).
    """
    if step is not None and total is not None:
        logger.info(f"[dim cyan]\\[{step}/{total}][/dim cyan] [cyan]{message}[/cyan]")
    else:
        logger.info(f"[cyan]{message}[/cyan]")


def print_success(message: str) -> None:
    """Log a success message with checkmark."""
    logger.info(f"[green]✓ {message}[/green]")


def print_info(message: str) -> None:
    """Log an info message."""
    logger.info(f"[dim]{message}[/dim]")


def print_warning(message: str) -> None:
    """Log a warning message."""
    logger.warning(f"[yellow]! {message}[/yellow]")


def print_error(message: str) -> None:
    """Log an error message."""
    logger.error(f"[red]✗ {message}[/red]")


def print_action(message: str) -> None:
    """Log an action/instruction message."""
    logger.info(f"[blue]→ {message}[/blue]")


def print_header(title: str, style: str = "bold cyan") -> None:
    """Log a section header.

    Args:
        title: The header text.
        style: Rich markup style for the header text.
    """
    logger.info("")
    logger.info(f"[{style}]{title}[/{style}]")
    logger.info(f"[dim]{'─' * len(title)}[/dim]")


@contextmanager
def status_log(message: str) -> Generator[None, None, None]:
    """Context manager that logs a status message before an operation.

    Args:
        message: The message to display.

    Yields:
        None
    """
    logger.info(f"[cyan]... {message}[/cyan]")
    yield


def print_deployment_summary(
    job_id: str,
    namespace: str,
    image: str,
    workers: int,
    num_pods: int,
    workers_per_pod: int,
    ttl_seconds: int | None = None,
    endpoint_url: str | None = None,
    model_names: list[str] | None = None,
    to_stderr: bool = False,
    name: str | None = None,
) -> None:
    """Log a summary of the deployment configuration.

    Args:
        job_id: The job ID.
        namespace: Kubernetes namespace.
        image: Container image.
        workers: Total number of workers.
        num_pods: Number of worker pods.
        workers_per_pod: Workers per pod.
        ttl_seconds: TTL after completion (optional).
        endpoint_url: Target LLM endpoint URL (optional).
        model_names: Model names being benchmarked (optional).
        to_stderr: If True, log to stderr instead of stdout.
        name: Human-readable benchmark name (optional).
    """
    from aiperf.kubernetes import utils
    from aiperf.kubernetes.environment import K8sEnvironment

    ctrl_cpu = utils.parse_cpu(K8sEnvironment.CONTROLLER.CPU_REQUEST)
    ctrl_mem = utils.parse_memory_gib(K8sEnvironment.CONTROLLER.MEMORY_REQUEST)

    worker_cpu = utils.parse_cpu(K8sEnvironment.WORKER.CPU_REQUEST)
    worker_mem = utils.parse_memory_gib(K8sEnvironment.WORKER.MEMORY_REQUEST)
    proc_cpu = utils.parse_cpu(K8sEnvironment.RECORD_PROCESSOR.CPU_REQUEST)
    proc_mem = utils.parse_memory_gib(K8sEnvironment.RECORD_PROCESSOR.MEMORY_REQUEST)

    from aiperf.common.environment import Environment

    processors_per_pod = max(
        1, workers_per_pod // Environment.RECORD.PROCESSOR_SCALE_FACTOR
    )
    pod_cpu = (worker_cpu * workers_per_pod) + (proc_cpu * processors_per_pod)
    pod_mem = (worker_mem * workers_per_pod) + (proc_mem * processors_per_pod)

    total_cpu = ctrl_cpu + (pod_cpu * num_pods)
    total_mem = ctrl_mem + (pod_mem * num_pods)

    ttl_str = f"{ttl_seconds}s" if ttl_seconds is not None else "disabled"

    target = _stderr_logger if to_stderr else logger

    header = "AIPerf Kubernetes Deployment"
    target.info("")
    target.info(f"[bold cyan]{header}[/bold cyan]")
    target.info(f"[dim]{'─' * len(header)}[/dim]")

    fields: list[tuple[str, str]] = []
    if name:
        fields.append(("Name", name))
    fields.append(("Job ID", job_id))
    fields.append(("Namespace", namespace))
    if endpoint_url:
        fields.append(("Endpoint", endpoint_url))
    if model_names:
        fields.append(("Model", ", ".join(model_names)))
    fields.append(("Image", image))
    fields.append(("Workers", str(workers)))
    fields.append(("Pods", f"{num_pods} ({workers_per_pod} workers/pod)"))
    fields.append(("CPU", f"~{utils.format_cpu(total_cpu)}"))
    fields.append(("Memory", f"~{utils.format_memory(total_mem)}"))
    fields.append(("TTL", ttl_str))

    label_width = max(len(k) for k, _ in fields)
    for label, value in fields:
        target.info(f"  [dim]{label:<{label_width}}[/dim]  {value}")


def _print_command_hints(
    job_id: str,
    namespace: str,
    *,
    header: str | None = None,
    header_style: str = "green bold",
    attach_label: str = "Monitor progress",
) -> None:
    """Log attach/results command hints.

    Args:
        job_id: The job ID.
        namespace: Kubernetes namespace.
        header: Optional header message.
        header_style: Rich markup style for the header.
        attach_label: Label for the attach command.
    """
    if header:
        print_header(header, style=header_style)
    else:
        logger.info("")
    logger.info(f"  [dim]{attach_label}:[/dim]")
    logger.info(f"    aiperf kube attach {job_id} --namespace {namespace}")
    logger.info("")
    logger.info("  [dim]Retrieve results:[/dim]")
    logger.info(f"    aiperf kube results {job_id} --namespace {namespace}")


def print_detach_info(job_id: str, namespace: str, name: str | None = None) -> None:
    """Log detach information with helpful commands.

    Args:
        job_id: The job ID.
        namespace: Kubernetes namespace.
        name: Human-readable benchmark name (optional).
    """
    label = f"Benchmark '{name}'" if name else "Benchmark"
    _print_command_hints(job_id, namespace, header=f"{label} deployed successfully!")


def print_interrupt_info(job_id: str, namespace: str) -> None:
    """Log info after keyboard interrupt with helpful commands.

    Args:
        job_id: The job ID.
        namespace: Kubernetes namespace.
    """
    print_header("Detached from benchmark", style="yellow bold")
    logger.info("[dim]Benchmark continues running in Kubernetes[/dim]")
    _print_command_hints(
        job_id,
        namespace,
        attach_label="Reattach",
    )


def print_results_summary(output_dir: str) -> None:
    """Log a summary of where results were saved.

    Args:
        output_dir: Path to the output directory.
    """
    print_header("Results", style="bold green")
    logger.info(f"  [dim]Saved to:[/dim]  {output_dir}/")


def print_benchmark_complete() -> None:
    """Log benchmark completion message."""
    logger.info("")
    logger.info("[bold green]Benchmark complete![/bold green]")


def print_file_table(files: list[tuple[str, int]], *, verb: str = "Downloaded") -> None:
    """Log a structured list of files with sizes.

    Args:
        files: List of (filename, size_bytes) tuples.
        verb: Action verb for the title (e.g., "Downloaded", "Copied").
    """
    print_header(f"{verb} {len(files)} file(s)")
    for name, size in sorted(files):
        print_info(f"{name}: {size:,} bytes")


def print_metrics_summary(data: dict) -> None:
    """Log a summary of key metrics."""
    key_metrics = [
        "throughput",
        "latency_p50",
        "latency_p99",
        "latency_mean",
        "ttft_p50",
        "ttft_p99",
    ]
    found = [
        (
            m.get("tag", ""),
            f"{m.get('value', 0):.2f}",
            m.get("display_unit", m.get("unit", "")),
        )
        for m in data.get("metrics", [])
        if any(k in m.get("tag", "").lower() for k in key_metrics)
    ]

    if found:
        print_header("Benchmark Metrics")
        for tag, value, unit in found[:10]:
            logger.info(f"  [dim]{tag:<30}[/dim]  {value} {unit}")

    if "benchmark_id" in data:
        print_info(f"Benchmark ID: {data['benchmark_id']}")


def print_jobs_table(
    jobs: list[JobSetInfo],
    pod_summaries: dict[str, PodSummary] | None = None,
    wide: bool = False,
) -> None:
    """Print a formatted table of AIPerf jobs.

    Uses Rich Table for structured columnar output.

    Args:
        jobs: List of JobSetInfo objects.
        pod_summaries: Optional dict mapping JobSet name to PodSummary.
        wide: Show additional columns (job-id, custom-name, endpoint).
    """
    from aiperf.kubernetes.cli_helpers import format_age

    table = Table(show_header=True, header_style="bold", box=None)

    table.add_column("NAME", style="cyan")
    if wide:
        table.add_column("JOB-ID", style="dim")
        table.add_column("CUSTOM-NAME", style="bold")
    table.add_column("MODEL", style="dim")
    table.add_column("NAMESPACE", style="dim")
    table.add_column("STATUS")
    table.add_column("PROGRESS", justify="right")
    table.add_column("READY", justify="right")
    table.add_column("RESTARTS", justify="right", style="dim")
    table.add_column("AGE", style="dim")
    if wide:
        table.add_column("ENDPOINT", style="dim")

    for job in jobs:
        jobset_status = JobSetStatus.from_str(job.status)
        phase = jobset_status.to_pod_phase() if jobset_status else None
        status_text = Text(job.status, style=_STATUS_STYLES.get(phase, ""))
        progress = job.progress
        progress_text = Text(progress or "-", style="dim" if not progress else "bold")
        age = format_age(job.created)

        summary = (pod_summaries or {}).get(job.name)
        if summary:
            ready_style = "green" if summary.ready == summary.total else "yellow"
            ready_text = Text(summary.ready_str, style=ready_style)
            restart_style = "red" if summary.restarts > 0 else "dim"
            restart_text = Text(str(summary.restarts), style=restart_style)
        else:
            ready_text = Text("-", style="dim")
            restart_text = Text("-", style="dim")

        row: list[str | Text] = [job.name]
        if wide:
            row.append(job.job_id)
            row.append(job.custom_name or "")
        row.extend(
            [
                job.model or "",
                job.namespace,
                status_text,
                progress_text,
                ready_text,
                restart_text,
                age,
            ]
        )
        if wide:
            row.append(job.endpoint or "")
        table.add_row(*row)

    console.print(table)
