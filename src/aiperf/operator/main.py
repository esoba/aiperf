# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Kubernetes Operator.

Handles AIPerfJob CRD lifecycle with:
- Spec validation and endpoint health checks
- Kubernetes event emission
- Condition tracking (ConfigValid, EndpointReachable, ResourcesCreated, etc.)
- Metrics summary extraction
- Results storage with retry logic
- Job cancellation support
- Job timeout detection
- Pod restart monitoring
- Results TTL cleanup

Run: kopf run -m aiperf.operator.main --verbose
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
import kopf
import kr8s
from kr8s.asyncio.objects import ConfigMap, Pod

from aiperf.kubernetes.client import get_api
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.kubernetes.kr8s_resources import AsyncJobSet
from aiperf.kubernetes.resources import KubernetesDeployment
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.events import (
    event_cancelled,
    event_completed,
    event_created,
    event_endpoint_reachable,
    event_endpoint_unreachable,
    event_failed,
    event_job_timeout,
    event_pod_restarts,
    event_resources_created,
    event_results_cleaned,
    event_results_failed,
    event_results_stored,
    event_spec_invalid,
    event_spec_valid,
    event_started,
    event_workers_ready,
)
from aiperf.operator.models import (
    AIPerfJobSpec,
    FetchResult,
    HealthCheckResult,
    MetricsSummary,
    OwnerReference,
    PhaseProgress,
)
from aiperf.operator.progress_client import ProgressClient
from aiperf.operator.spec_converter import AIPerfJobSpecConverter
from aiperf.operator.status import (
    ConditionType,
    Phase,
    StatusBuilder,
    format_timestamp,
    parse_timestamp,
)

if TYPE_CHECKING:
    from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats

AIPERF_GROUP = "aiperf.nvidia.com"
AIPERF_VERSION = "v1alpha1"
AIPERF_PLURAL = "aiperfjobs"

logger = logging.getLogger(__name__)

# Per-job ProgressClient cache keyed by namespace/job_id.
# Avoids creating a new aiohttp session every monitor tick.
_progress_clients: dict[str, ProgressClient] = {}

# Tracks (pod_name, restart_count) pairs already warned about per job.
# Prevents emitting the same pod restart event every monitor tick.
_warned_pod_restarts: dict[str, set[tuple[str, int]]] = {}

# Tracks jobs where shutdown has already been sent to avoid duplicate signals.
_shutdown_sent: set[str] = set()


def _job_key(namespace: str, job_id: str) -> str:
    """Create a unique cache key scoped to namespace.

    CRs in different namespaces can share the same name, so cache keys
    and results directories must be namespace-scoped.
    """
    return f"{namespace}/{job_id}"


async def _get_or_create_progress_client(key: str) -> ProgressClient:
    """Get a cached ProgressClient for a job, creating one if needed."""
    client = _progress_clients.get(key)
    if client is None:
        client = ProgressClient()
        await client.__aenter__()
        _progress_clients[key] = client
    return client


async def _close_progress_client(key: str) -> None:
    """Close and remove a cached ProgressClient and dedup state for a job."""
    client = _progress_clients.pop(key, None)
    if client is not None:
        await client.__aexit__(None, None, None)
    _warned_pod_restarts.pop(key, None)
    _shutdown_sent.discard(key)


async def _get_api() -> kr8s.Api:
    """Get async Kubernetes API client.

    kr8s auto-detects in-cluster vs kubeconfig environments.
    """
    return await get_api()


def _create_owner_reference(name: str, uid: str) -> OwnerReference:
    """Create owner reference for cascade deletion."""
    return OwnerReference(
        api_version=f"{AIPERF_GROUP}/{AIPERF_VERSION}",
        kind="AIPerfJob",
        name=name,
        uid=uid,
    )


async def _check_endpoint_health(
    url: str, timeout: float = OperatorEnvironment.ENDPOINT_CHECK_TIMEOUT
) -> HealthCheckResult:
    """Check if LLM endpoint is reachable.

    Tries a single canonical health path first, falling back to alternatives
    only if the first fails.

    Args:
        url: Endpoint URL to check.
        timeout: Per-request timeout in seconds.

    Returns:
        HealthCheckResult with reachability status and error message.
    """
    from aiperf.transports.aiohttp_client import create_tcp_connector

    health_paths = ["/health", "/v1/health", "/v1/models", "/"]

    connector = create_tcp_connector()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout), connector=connector
    ) as session:
        for path in health_paths:
            try:
                check_url = url.rstrip("/") + path
                async with session.get(check_url) as response:
                    if response.status < 500:
                        return HealthCheckResult(reachable=True, error="")
            except aiohttp.ClientError:
                continue
            except Exception as e:
                return HealthCheckResult(
                    reachable=False, error=f"Unexpected error: {e}"
                )

    return HealthCheckResult(reachable=False, error="All health endpoints unreachable")


async def _fetch_results_with_retry(
    controller_host: str,
    namespace: str,
    job_id: str,
    max_retries: int = OperatorEnvironment.RESULTS.MAX_RETRIES,
    retry_delay: float = OperatorEnvironment.RESULTS.RETRY_DELAY,
) -> FetchResult:
    """Fetch results from controller pod with retry logic.

    Uses the cached ProgressClient for the job. Falls back to creating
    a temporary client if no cached one exists (e.g. after restart).

    Args:
        controller_host: Controller pod DNS name.
        namespace: Kubernetes namespace (used for results directory scoping).
        job_id: Job identifier for results directory.
        max_retries: Maximum retry attempts.
        retry_delay: Delay between retries (with exponential backoff).

    Returns:
        FetchResult with metrics dict and list of downloaded files.
    """
    # Validate components to prevent path traversal (e.g. "../../etc")
    for label, value in [("namespace", namespace), ("job_id", job_id)]:
        if not value or "/" in value or value in (".", ".."):
            logger.error(f"Invalid {label} for results storage: {value!r}")
            return FetchResult(metrics=None, downloaded=[])

    key = _job_key(namespace, job_id)
    client = await _get_or_create_progress_client(key)

    metrics = None
    downloaded: list[str] = []
    delay = retry_delay

    for attempt in range(max_retries + 1):
        try:
            if metrics is None:
                metrics = await client.get_metrics(controller_host)

            if not downloaded and OperatorEnvironment.RESULTS.DIR.exists():
                dest_dir = OperatorEnvironment.RESULTS.DIR / namespace / job_id
                downloaded = await client.download_all_results(
                    controller_host, dest_dir
                )

            if metrics and downloaded:
                return FetchResult(metrics=metrics, downloaded=downloaded)

        except Exception as e:
            if attempt < max_retries:
                logger.debug(f"Results fetch attempt {attempt + 1} failed: {e}")
            else:
                logger.warning(
                    f"Results fetch failed after {max_retries + 1} attempts: {e}"
                )

        if attempt < max_retries:
            await asyncio.sleep(delay)
            delay *= 2

    if not metrics and not downloaded:
        logger.warning(f"No metrics or files retrieved for {job_id}")
    elif not metrics:
        logger.warning(
            f"Metrics fetch failed for {job_id}, files downloaded: {len(downloaded)}"
        )
    elif not downloaded:
        logger.warning(f"File download failed for {job_id}, metrics retrieved")

    return FetchResult(metrics=metrics, downloaded=downloaded)


def _get_elapsed_seconds(status: dict[str, Any]) -> float | None:
    """Calculate elapsed seconds since startTime, or None if unavailable."""
    start_time = status.get("startTime")
    if not start_time:
        return None
    try:
        start_dt = parse_timestamp(start_time)
        return (datetime.now(timezone.utc) - start_dt).total_seconds()
    except (ValueError, TypeError):
        return None


def _get_job_timeout(spec: dict[str, Any]) -> float:
    """Get job timeout from spec or global default. 0 means no timeout."""
    return float(spec.get("timeoutSeconds", OperatorEnvironment.JOB_TIMEOUT_SECONDS))


@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_: Any) -> None:
    """Configure operator settings."""
    settings.persistence.finalizer = f"{AIPERF_GROUP}/finalizer"
    settings.posting.level = logging.INFO


@kopf.on.create(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL)
async def on_create(
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    uid: str,
    patch: kopf.Patch,
    **_: Any,
) -> dict[str, Any]:
    """Create ConfigMap and JobSet for the benchmark job."""
    job_id = name
    logger.info(f"Creating AIPerfJob {namespace}/{name}")

    status = StatusBuilder(patch)

    try:
        # Step 1: Validate spec
        try:
            validated_spec = AIPerfJobSpec.from_crd_spec(spec)
            status.conditions.set_true(
                ConditionType.CONFIG_VALID, "SpecValid", "Spec validation passed"
            )
            event_spec_valid(body)
        except ValueError as e:
            status.conditions.set_false(
                ConditionType.CONFIG_VALID, "SpecInvalid", str(e)
            )
            status.set_phase(Phase.FAILED).set_error(f"Invalid spec: {e}")
            status.finalize()
            event_spec_invalid(body, str(e))
            raise kopf.PermanentError(f"Invalid spec: {e}") from e

        # Step 2: Check endpoint health
        endpoint_url = validated_spec.get_endpoint_url()
        if endpoint_url:
            health = await _check_endpoint_health(endpoint_url)
            if health.reachable:
                status.conditions.set_true(
                    ConditionType.ENDPOINT_REACHABLE,
                    "EndpointReachable",
                    f"Endpoint {endpoint_url} is reachable",
                )
                event_endpoint_reachable(body, endpoint_url)
            else:
                status.conditions.set_false(
                    ConditionType.ENDPOINT_REACHABLE,
                    "EndpointUnreachable",
                    f"Endpoint {endpoint_url} unreachable: {health.error}",
                )
                event_endpoint_unreachable(body, endpoint_url, health.error)
                logger.warning(f"Endpoint {endpoint_url} not reachable: {health.error}")

        # Step 3: Convert spec to AIPerf configs
        converter = AIPerfJobSpecConverter(spec, name, namespace, job_id=job_id)
        aiperf_config = converter.to_aiperf_config()
        user_config, service_config = converter.to_legacy_configs()
        pod_customization = converter.to_pod_customization()
        worker_count = converter.calculate_workers()
        scheduling = converter.to_scheduling_config()

        deployment = KubernetesDeployment(
            job_id=job_id,
            namespace=namespace,
            image=validated_spec.image,
            image_pull_policy=validated_spec.image_pull_policy,
            worker_replicas=worker_count,
            ttl_seconds=validated_spec.ttl_seconds_after_finished,
            aiperf_config=aiperf_config,
            user_config=user_config,
            service_config=service_config,
            pod_customization=pod_customization,
            queue_name=scheduling["queue_name"],
            priority_class=scheduling["priority_class"],
        )

        owner_ref_dict = _create_owner_reference(name, uid).to_k8s_dict()
        api = await _get_api()

        # Step 4: Create ConfigMap
        configmap = deployment.get_configmap_spec().to_k8s_manifest()
        configmap.setdefault("metadata", {}).setdefault("ownerReferences", []).append(
            owner_ref_dict
        )
        await ConfigMap(configmap, api=api).create()
        configmap_name = configmap["metadata"]["name"]
        logger.info(f"Created ConfigMap {configmap_name}")

        # Step 5: Create JobSet
        jobset = deployment.get_jobset_spec().to_k8s_manifest()
        jobset.setdefault("metadata", {}).setdefault("ownerReferences", []).append(
            owner_ref_dict
        )
        await AsyncJobSet(jobset, api=api).create()
        jobset_name = jobset["metadata"]["name"]
        logger.info(f"Created JobSet {jobset_name}")

        # Set conditions and status
        status.conditions.set_true(
            ConditionType.RESOURCES_CREATED,
            "ResourcesCreated",
            f"Created ConfigMap/{configmap_name} and JobSet/{jobset_name}",
        )
        event_resources_created(body, configmap_name, jobset_name)
        event_created(body, job_id, worker_count)

        # Set initial status
        status.set_phase(Phase.PENDING)
        patch.status["startTime"] = format_timestamp()
        patch.status["jobId"] = job_id
        patch.status["jobSetName"] = deployment.jobset_name
        status.set_workers(0, worker_count)

        # Store results TTL if configured
        if validated_spec.results_ttl_days:
            patch.status["resultsTtlDays"] = validated_spec.results_ttl_days

        status.finalize()
        return {"jobSetName": deployment.jobset_name, "workers": worker_count}

    except kopf.PermanentError:
        raise
    except Exception as e:
        logger.exception(f"Failed to create AIPerfJob {namespace}/{name}")
        status.set_phase(Phase.FAILED).set_error(str(e))
        status.finalize()
        event_failed(body, job_id, str(e))
        raise kopf.PermanentError(f"Failed to create: {e}") from e


@kopf.on.delete(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL)
async def on_delete(
    name: str, namespace: str, status: dict[str, Any], **_: Any
) -> None:
    """Handle deletion - clean up cached ProgressClient and let K8s GC handle resources."""
    job_id = status.get("jobId", name)
    await _close_progress_client(_job_key(namespace, job_id))
    logger.info(f"Deleting AIPerfJob {namespace}/{name}")


@kopf.on.update(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL, field="spec.cancel")
async def on_cancel(
    body: dict[str, Any],
    spec: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Handle cancellation request via spec.cancel field."""
    if not spec.get("cancel"):
        return

    current_phase = status.get("phase", Phase.PENDING)
    if current_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        return  # Already terminal

    job_id = status.get("jobId", name)
    jobset_name = status.get("jobSetName")

    logger.info(f"Cancelling AIPerfJob {namespace}/{name}")

    sb = StatusBuilder(patch, status)

    # Delete the JobSet to stop the benchmark
    if jobset_name:
        try:
            api = await _get_api()
            js = await AsyncJobSet.get(jobset_name, namespace=namespace, api=api)
            await js.delete()
            logger.info(f"Deleted JobSet {jobset_name}")
        except kr8s.NotFoundError:
            pass
        except kr8s.ServerError as e:
            logger.warning(f"Failed to delete JobSet: {e}")

    await _close_progress_client(_job_key(namespace, job_id))
    sb.set_phase(Phase.CANCELLED).set_completion_time()
    sb.finalize()
    event_cancelled(body, job_id)


@kopf.timer(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_PLURAL,
    interval=OperatorEnvironment.MONITOR.INTERVAL,
    initial_delay=OperatorEnvironment.MONITOR.INITIAL_DELAY,
)
async def monitor_progress(
    body: dict[str, Any],
    status: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Monitor job progress and update status."""
    current_phase = status.get("phase", Phase.PENDING)

    # Stop monitoring terminal jobs
    if current_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        return

    jobset_name = status.get("jobSetName")
    job_id = status.get("jobId")
    if not jobset_name or not job_id:
        return

    sb = StatusBuilder(patch, status)

    try:
        key = _job_key(namespace, job_id)

        # Check job timeout
        timeout_sec = _get_job_timeout(spec)
        if timeout_sec > 0:
            elapsed = _get_elapsed_seconds(status)
            if elapsed is not None and elapsed > timeout_sec:
                sb.set_phase(Phase.FAILED).set_error(
                    f"Job timed out after {elapsed:.0f}s (limit: {timeout_sec:.0f}s)"
                )
                sb.set_completion_time()
                sb.finalize()
                event_job_timeout(body, job_id, elapsed)
                # Delete JobSet to free resources (like cancel does)
                if jobset_name:
                    try:
                        api = await _get_api()
                        js = await AsyncJobSet.get(
                            jobset_name, namespace=namespace, api=api
                        )
                        await js.delete()
                        logger.info(f"Deleted JobSet {jobset_name} after timeout")
                    except kr8s.NotFoundError:
                        pass
                    except kr8s.ServerError as e:
                        logger.warning(f"Failed to delete JobSet on timeout: {e}")
                await _close_progress_client(key)
                return

        api = await _get_api()

        # Get JobSet status
        try:
            jobset_obj = await AsyncJobSet.get(
                jobset_name, namespace=namespace, api=api
            )
            jobset = jobset_obj.raw
        except kr8s.NotFoundError:
            sb.set_phase(Phase.FAILED).set_error("JobSet not found")
            sb.finalize()
            await _close_progress_client(key)
            return

        jobset_status = jobset.get("status", {})

        # Detect Kueue-managed suspension (gang-scheduling)
        jobset_labels = jobset.get("metadata", {}).get("labels", {})
        is_kueue_managed = "kueue.x-k8s.io/queue-name" in jobset_labels
        jobset_suspended = jobset.get("spec", {}).get("suspend", False)

        if (
            is_kueue_managed
            and jobset_suspended
            and current_phase in (Phase.PENDING, Phase.QUEUED)
        ):
            sb.set_phase(Phase.QUEUED)
            sb.finalize()
            return

        # Check for terminal state
        for condition in jobset_status.get("conditions", []):
            if condition.get("status") != "True":
                continue
            if condition.get("type") == "Completed":
                await _handle_completion(
                    body, namespace, jobset_name, job_id, status, sb
                )
                await _close_progress_client(key)
                return
            if condition.get("type") == "Failed":
                sb.set_phase(Phase.FAILED)
                sb.set_error(condition.get("message", "JobSet failed"))
                sb.finalize()
                event_failed(body, job_id, condition.get("message", "JobSet failed"))
                await _close_progress_client(key)
                return

        # Update worker count and phase
        total_workers = status.get("workers", {}).get("total", 0)
        workers_ready = 0
        workers_succeeded = 0

        for rj in jobset_status.get("replicatedJobsStatus", []):
            if rj.get("name") == "workers":
                workers_ready = rj.get("ready", 0)
                workers_succeeded = rj.get("succeeded", 0)
                # Derive total from JobSet if CRD status doesn't have it yet
                if total_workers == 0:
                    total_workers = (
                        rj.get("ready", 0)
                        + rj.get("active", 0)
                        + rj.get("succeeded", 0)
                        + rj.get("failed", 0)
                        + rj.get("suspended", 0)
                    ) or 1  # Fallback to 1 if all zero
                sb.set_workers(workers_ready, total_workers)

        # Phase transitions based on worker readiness
        if current_phase in (Phase.PENDING, Phase.QUEUED) and (
            workers_ready > 0 or workers_succeeded > 0
        ):
            sb.set_phase(Phase.INITIALIZING)

        effective_phase = sb.get_phase() or current_phase
        if (
            effective_phase == Phase.INITIALIZING
            and total_workers > 0
            and (workers_ready == total_workers or workers_succeeded == total_workers)
        ):
            sb.set_phase(Phase.RUNNING)
            sb.conditions.set_true(
                ConditionType.WORKERS_READY,
                "AllWorkersReady",
                f"All {total_workers} workers are ready",
            )
            sb.conditions.set_true(
                ConditionType.BENCHMARK_RUNNING,
                "BenchmarkStarted",
                "Benchmark is running",
            )
            event_workers_ready(body, workers_ready, total_workers)
            event_started(body, job_id)

        # Check for pod restarts (CrashLoopBackOff detection)
        await _check_pod_restarts(api, body, namespace, jobset_name, key)

        # Fetch progress from controller if running or workers already completed
        effective_phase = sb.get_phase() or current_phase
        if effective_phase == Phase.RUNNING or (
            workers_succeeded > 0 and workers_succeeded >= total_workers
        ):
            client = await _get_or_create_progress_client(key)
            benchmark_complete = await _fetch_progress(
                namespace, jobset_name, patch, sb, client, key
            )

            # If benchmark is done, fetch results then shutdown controller
            if benchmark_complete:
                _shutdown_sent.add(key)
                logger.info(
                    f"Benchmark complete for {jobset_name}, "
                    f"fetching results and shutting down controller"
                )
                host = controller_dns_name(jobset_name, namespace)
                await _handle_completion(
                    body, namespace, jobset_name, job_id, status, sb
                )
                # Shutdown controller after results are fetched
                await client.send_shutdown(host)
                await _close_progress_client(key)
                return

        sb.finalize()

    except Exception:
        logger.exception(f"Error monitoring {namespace}/{name}")
        sb.finalize()


async def _check_pod_restarts(
    api: kr8s.Api,
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    key: str,
) -> None:
    """Check for excessive pod restarts and emit warning events (deduplicated)."""
    try:
        pods = [
            pod
            async for pod in Pod.list(
                namespace=namespace,
                label_selector=f"jobset.sigs.k8s.io/jobset-name={jobset_name}",
                api=api,
            )
        ]
        warned = _warned_pod_restarts.setdefault(key, set())
        for pod in pods:
            pod_status = pod.raw.get("status", {})
            all_statuses = pod_status.get("containerStatuses", []) + pod_status.get(
                "initContainerStatuses", []
            )
            for cs in all_statuses:
                restart_count = cs.get("restartCount", 0)
                if restart_count < OperatorEnvironment.POD_RESTART_THRESHOLD:
                    continue
                dedup_key = (pod.name, restart_count)
                if dedup_key in warned:
                    continue
                warned.add(dedup_key)
                reason = "Unknown"
                last_state = cs.get("lastState", {})
                terminated = last_state.get("terminated", {})
                if terminated:
                    reason = terminated.get("reason", "Unknown")
                waiting = cs.get("state", {}).get("waiting", {})
                if waiting:
                    reason = waiting.get("reason", reason)
                event_pod_restarts(body, pod.name, restart_count, reason)
    except Exception as e:
        logger.warning(f"Failed to check pod restarts: {e}")


async def _handle_completion(
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    job_id: str,
    status: dict[str, Any],
    sb: StatusBuilder,
) -> None:
    """Handle job completion: fetch results and update status."""
    # Backfill conditions for fast-completing jobs that skipped RUNNING phase
    total_workers = status.get("workers", {}).get("total", 1)
    if not sb.conditions.is_condition_true(ConditionType.WORKERS_READY):
        sb.conditions.set_true(
            ConditionType.WORKERS_READY,
            "CompletedBeforeMonitor",
            f"Job completed before workers ({total_workers}) were observed ready",
        )
    if not sb.conditions.is_condition_true(ConditionType.BENCHMARK_RUNNING):
        sb.conditions.set_true(
            ConditionType.BENCHMARK_RUNNING,
            "CompletedBeforeMonitor",
            "Job completed before running state was observed",
        )

    sb.set_phase(Phase.COMPLETED).set_completion_time()

    # Calculate duration
    duration_sec = _get_elapsed_seconds(status)

    # Fetch results with retry
    host = controller_dns_name(jobset_name, namespace)
    result = await _fetch_results_with_retry(host, namespace, job_id)

    has_metrics = bool(result.metrics and result.metrics.get("metrics"))
    has_files = bool(result.downloaded)

    if has_metrics:
        sb.set_results(result.metrics)

        summary = MetricsSummary.from_metrics(result.metrics)
        summary_dict = summary.to_status_dict()
        if summary_dict:
            sb.set_summary(summary_dict)

    if has_files:
        dest_dir = OperatorEnvironment.RESULTS.DIR / namespace / job_id
        sb.set_results_path(str(dest_dir))
        event_results_stored(body, str(dest_dir), len(result.downloaded))
        logger.info(f"Downloaded {len(result.downloaded)} result files to {dest_dir}")

    # Set condition based on what was actually retrieved
    if has_metrics and has_files:
        sb.conditions.set_true(
            ConditionType.RESULTS_AVAILABLE,
            "ResultsStored",
            f"Metrics and {len(result.downloaded)} result files stored",
        )
    elif has_metrics:
        sb.conditions.set_true(
            ConditionType.RESULTS_AVAILABLE,
            "MetricsOnly",
            "Metrics stored but result file download failed",
        )
        logger.warning(f"No result files downloaded for {jobset_name}")
    elif has_files:
        sb.conditions.set_true(
            ConditionType.RESULTS_AVAILABLE,
            "FilesOnly",
            f"{len(result.downloaded)} files stored but metrics fetch failed",
        )
        event_results_failed(body, "Could not fetch metrics")
    else:
        sb.conditions.set_false(
            ConditionType.RESULTS_AVAILABLE,
            "ResultsFetchFailed",
            "Failed to fetch both metrics and result files from controller",
        )
        event_results_failed(body, "Could not fetch metrics or result files")

    sb.finalize()
    event_completed(body, job_id, duration_sec)


async def _fetch_progress(
    namespace: str,
    jobset_name: str,
    patch: kopf.Patch,
    sb: StatusBuilder,
    client: ProgressClient,
    key: str,
) -> bool:
    """Fetch progress and live metrics from controller pod.

    Returns True if the benchmark is complete (all profiling requests done).
    """
    host = controller_dns_name(jobset_name, namespace)

    try:
        progress = await client.get_progress(host)

        if progress.connection_error:
            logger.debug(
                f"Progress API unreachable for {jobset_name}: connection error"
            )
            return False

        phases_data: dict[str, Any] = {}
        for phase, stats in progress.phases.items():
            if phase_progress := _build_phase_progress(stats):
                phases_data[phase] = phase_progress.to_k8s_dict()

        if phases_data:
            patch.status["phases"] = phases_data

        if progress.current_phase:
            patch.status["currentPhase"] = progress.current_phase

        if progress.error:
            patch.status["error"] = progress.error

        # Fetch live metrics and server metrics concurrently
        metrics_coro = client.get_metrics(host)
        server_metrics_coro = client.get_server_metrics(host)
        metrics, server_metrics = await asyncio.gather(
            metrics_coro, server_metrics_coro, return_exceptions=True
        )

        # Process metrics (may be an exception from gather)
        if isinstance(metrics, BaseException):
            logger.warning(f"Live metrics fetch failed for {jobset_name}: {metrics}")
        elif isinstance(metrics, dict) and metrics.get("metrics"):
            patch.status["liveMetrics"] = metrics

            summary = MetricsSummary.from_metrics(metrics)
            summary_dict = summary.to_status_dict()
            if summary_dict:
                patch.status["liveSummary"] = summary_dict

        # Process server metrics
        if isinstance(server_metrics, BaseException):
            logger.warning(
                f"Server metrics fetch failed for {jobset_name}: {server_metrics}"
            )
        elif isinstance(server_metrics, dict) and server_metrics.get(
            "endpoint_summaries"
        ):
            patch.status["serverMetrics"] = server_metrics

        # Return completion status for caller to handle
        if progress.is_complete and key not in _shutdown_sent:
            return True

    except Exception as e:
        logger.warning(f"Failed to fetch progress for {jobset_name}: {e}")

    return False


def _build_phase_progress(stats: CombinedPhaseStats) -> PhaseProgress | None:
    """Build PhaseProgress from CombinedPhaseStats."""
    total = stats.total_expected_requests or 0
    if total == 0 and stats.requests_sent == 0:
        return None

    elapsed = None
    if stats.start_ns is not None and stats.last_update_ns is not None:
        elapsed = round((stats.last_update_ns - stats.start_ns) / 1_000_000_000, 1)

    return PhaseProgress(
        requests_completed=stats.requests_completed,
        requests_sent=stats.requests_sent,
        requests_total=total,
        requests_cancelled=stats.requests_cancelled,
        requests_errors=stats.request_errors,
        requests_in_flight=stats.in_flight_requests,
        requests_per_second=round(stats.requests_per_second or 0, 2),
        requests_progress_percent=round(stats.requests_progress_percent or 0, 1),
        sessions_sent=stats.sent_sessions,
        sessions_completed=stats.completed_sessions,
        sessions_cancelled=stats.cancelled_sessions,
        sessions_in_flight=stats.in_flight_sessions,
        records_success=stats.success_records,
        records_error=stats.error_records,
        records_per_second=round(stats.records_per_second or 0, 2),
        records_progress_percent=round(stats.records_progress_percent or 0, 1),
        sending_complete=stats.is_requests_complete,
        timeout_triggered=stats.timeout_triggered,
        was_cancelled=stats.was_cancelled,
        requests_eta_seconds=round(stats.requests_eta_sec)
        if stats.requests_eta_sec is not None
        else None,
        records_eta_seconds=round(stats.records_eta_sec)
        if stats.records_eta_sec is not None
        else None,
        expected_duration_seconds=round(stats.expected_duration_sec, 1)
        if stats.expected_duration_sec is not None
        else None,
        elapsed_time_seconds=elapsed,
    )


@kopf.timer(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_PLURAL,
    interval=86400.0,  # Once per day
    initial_delay=3600.0,  # 1 hour after startup
    idle=3600.0,  # Run even when no changes
)
async def cleanup_old_results(
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    **_: Any,
) -> None:
    """Clean up old results based on TTL."""
    # Only run cleanup for completed jobs
    if status.get("phase") != Phase.COMPLETED:
        return

    ttl_days = status.get("resultsTtlDays", OperatorEnvironment.RESULTS.TTL_DAYS)
    job_id = status.get("jobId", name)
    results_path = status.get("resultsPath")

    if not results_path:
        return

    results_dir = Path(results_path)
    if not results_dir.exists():
        return

    # Validate that results_dir is under RESULTS_DIR to prevent path traversal
    try:
        results_dir.resolve().relative_to(OperatorEnvironment.RESULTS.DIR.resolve())
    except ValueError:
        logger.error(
            f"Results path {results_dir} is outside RESULTS_DIR "
            f"{OperatorEnvironment.RESULTS.DIR}, "
            "skipping cleanup"
        )
        return

    # Check if results are older than TTL
    try:
        mtime = results_dir.stat().st_mtime
        age_days = (datetime.now(timezone.utc).timestamp() - mtime) / 86400

        if age_days > ttl_days:
            await asyncio.to_thread(shutil.rmtree, results_dir)
            logger.info(
                f"Cleaned up old results for {job_id} (age: {age_days:.0f} days)"
            )
            event_results_cleaned(body, job_id, int(age_days))
    except Exception as e:
        logger.warning(f"Failed to clean up results for {job_id}: {e}")
