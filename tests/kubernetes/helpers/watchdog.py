# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic benchmark watchdog for Kubernetes E2E tests.

An autonomous monitoring agent that runs as a background task alongside
benchmark deployments. It continuously watches the cluster, reasons about
pod state, detects problems early, and reports actionable insights.

Monitors:
- Real-time pod status dashboard with progress bars and phase indicators
- Phase transition tracking with precise timing and role identification
- Container state analysis (CrashLoopBackOff, OOMKilled, ImagePullBackOff)
- K8s event stream for scheduling failures, volume issues, backoff errors
- Node resource pressure (CPU/memory allocation percentages)
- Timeout prediction with escalating severity warnings
- Stale namespace detection (leaked resources from previous runs)
- Container exit code analysis for non-zero exits
- Readiness probe failure tracking
- Pod age tracking relative to benchmark start

Usage::

    async with BenchmarkWatchdog(kubectl, namespace, timeout=300) as wd:
        # ... run benchmark ...
        # watchdog logs status + warnings automatically in background
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import KubectlClient, PodStatus

logger = AIPerfLogger(__name__)

# Box drawing for dashboards
_W = 72
_LINE = "+" + "-" * _W + "+"
_DLINE = "+" + "=" * _W + "+"


def _row(text: str) -> str:
    """Format a line inside the box, padded to fixed width."""
    return f"| {text:<{_W - 2}} |"


def _progress_bar(pct: float, width: int = 30) -> str:
    """Render a text progress bar."""
    filled = int(width * min(pct, 100) / 100)
    empty = width - filled
    bar = "#" * filled + "-" * empty
    return f"[{bar}] {pct:.0f}%"


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m{secs:02d}s"


def _short_pod_name(name: str, max_len: int = 38) -> str:
    """Shorten pod name, keeping the unique suffix visible."""
    if len(name) <= max_len:
        return name
    return "..." + name[-(max_len - 3) :]


def _pod_role(name: str) -> str:
    """Identify pod role from its name."""
    if "controller" in name:
        return "controller"
    if "worker" in name:
        return "worker"
    return "unknown"


def _phase_icon(phase: str) -> str:
    """Map phase to a compact visual indicator."""
    return {
        "Pending": "...",
        "Running": ">>>",
        "Succeeded": "[OK]",
        "Failed": "[!!]",
        "Unknown": "[??]",
        "Completed": "[OK]",
    }.get(phase, "   ")


class ProblemSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class WatchdogProblem:
    """A detected problem in the benchmark deployment."""

    severity: ProblemSeverity
    category: str
    message: str
    timestamp: float = field(default_factory=time.time)
    pod_name: str | None = None
    namespace: str | None = None
    suggestion: str | None = None


@dataclass
class PodTimeline:
    """Tracks a pod's phase transitions and durations."""

    name: str
    role: str = ""
    first_seen: float = field(default_factory=time.time)
    last_phase: str = "Unknown"
    phase_history: list[tuple[float, str]] = field(default_factory=list)
    restart_count: int = 0
    last_restart_count: int = 0
    pending_warned: bool = False
    pending_critical_warned: bool = False
    crashloop_warned: bool = False
    ready_str: str = "0/0"


class BenchmarkWatchdog:
    """Autonomous monitoring agent for benchmark deployments.

    Runs as a background async task that periodically:
    1. Prints a rich real-time pod status dashboard
    2. Tracks pod phase transitions with precise timing
    3. Detects CrashLoopBackOff, OOMKilled, ImagePullBackOff
    4. Monitors K8s events for scheduling failures
    5. Tracks time in Pending with escalating warnings
    6. Monitors restart counts and crash loops
    7. Checks node resource allocation
    8. Predicts timeouts with escalating urgency
    9. Detects stale namespaces from previous runs
    10. Analyzes container exit codes
    """

    def __init__(
        self,
        kubectl: KubectlClient,
        namespace: str,
        timeout: int = 300,
        poll_interval: float = 5.0,
        status_interval: float = 10.0,
        pending_threshold: float = 30.0,
        pending_critical_threshold: float = 90.0,
        crashloop_threshold: int = 2,
    ) -> None:
        self.kubectl = kubectl
        self.namespace = namespace
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.status_interval = status_interval
        self.pending_threshold = pending_threshold
        self.pending_critical_threshold = pending_critical_threshold
        self.crashloop_threshold = crashloop_threshold

        self._task: asyncio.Task | None = None
        self._problems: list[WatchdogProblem] = []
        self._pod_timelines: dict[str, PodTimeline] = {}
        self._start_time: float = 0.0
        self._event_fingerprints: set[str] = set()
        self._stopped = False
        self._tick_count: int = 0
        self._last_status_time: float = 0.0
        self._last_pod_snapshot: list[PodStatus] = []
        self._completed_pods: set[str] = set()
        self._node_check_done: bool = False
        self._stale_ns_check_done: bool = False
        self._node_cpu_pct: int | None = None
        self._node_mem_pct: int | None = None
        self._stale_ns_count: int = 0

    async def __aenter__(self) -> BenchmarkWatchdog:
        self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def start(self) -> None:
        """Start the watchdog background task."""
        self._start_time = time.time()
        self._last_status_time = self._start_time
        self._stopped = False
        logger.info(
            f"[WATCHDOG] Monitoring started for {self.namespace} "
            f"| timeout={self.timeout}s | poll={self.poll_interval}s "
            f"| status_interval={self.status_interval}s"
        )
        self._task = asyncio.create_task(
            self._monitor_loop(),
            name=f"watchdog-{self.namespace}",
        )

    async def stop(self) -> None:
        """Stop the watchdog and print final report."""
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._print_final_report()

    @property
    def problems(self) -> list[WatchdogProblem]:
        return list(self._problems)

    @property
    def has_critical(self) -> bool:
        return any(p.severity == ProblemSeverity.CRITICAL for p in self._problems)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        """Main monitoring loop - the watchdog's brain."""
        try:
            while not self._stopped:
                self._tick_count += 1
                try:
                    pods = await self._fetch_pods()
                    if pods is not None:
                        self._last_pod_snapshot = pods
                        self._analyze_pods(pods)

                        now = time.time()
                        if now - self._last_status_time >= self.status_interval:
                            self._print_status_dashboard(pods)
                            self._last_status_time = now

                    await self._check_events()
                    self._check_elapsed_time()

                    # One-time checks early in the lifecycle
                    if not self._node_check_done and self._tick_count == 2:
                        await self._check_node_resources()
                        self._node_check_done = True

                    if not self._stale_ns_check_done and self._tick_count == 3:
                        await self._check_stale_namespaces()
                        self._stale_ns_check_done = True

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug(lambda e=e: f"[WATCHDOG] Monitor error: {e}")
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            pass

    async def _fetch_pods(self) -> list[PodStatus] | None:
        try:
            return await self.kubectl.get_pods(self.namespace)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Pod analysis
    # ------------------------------------------------------------------

    def _analyze_pods(self, pods: list[PodStatus]) -> None:
        """Run all pod checks."""
        for pod in pods:
            tl = self._get_or_create_timeline(pod)
            self._track_phase_transition(pod, tl)
            self._check_pending_too_long(pod, tl)
            self._check_crash_loop(pod, tl)
            self._check_container_states(pod)
            self._check_pod_completion(pod)
            tl.ready_str = pod.ready

    def _get_or_create_timeline(self, pod: PodStatus) -> PodTimeline:
        if pod.name not in self._pod_timelines:
            self._pod_timelines[pod.name] = PodTimeline(
                name=pod.name,
                role=_pod_role(pod.name),
            )
        return self._pod_timelines[pod.name]

    def _track_phase_transition(self, pod: PodStatus, tl: PodTimeline) -> None:
        if pod.phase == tl.last_phase:
            return

        old_phase = tl.last_phase
        now = time.time()
        tl.phase_history.append((now, pod.phase))
        elapsed = now - self._start_time
        time_in_old = 0.0
        if len(tl.phase_history) >= 2:
            time_in_old = tl.phase_history[-1][0] - tl.phase_history[-2][0]

        short = pod.name.split("-")[-1] if "-" in pod.name else pod.name

        logger.info(
            f"[WATCHDOG] {tl.role}({short}): "
            f"{old_phase} -> {pod.phase}  "
            f"(in {old_phase} for {_fmt_duration(time_in_old)}, "
            f"total +{_fmt_duration(elapsed)})"
        )

        tl.last_phase = pod.phase

    def _check_pending_too_long(self, pod: PodStatus, tl: PodTimeline) -> None:
        if pod.phase != "Pending":
            return

        pending_duration = time.time() - tl.first_seen

        if (
            pending_duration > self.pending_critical_threshold
            and not tl.pending_critical_warned
        ):
            tl.pending_critical_warned = True
            self._add_problem(
                ProblemSeverity.CRITICAL,
                "pod-pending-critical",
                f"Pod {pod.name} stuck Pending for {_fmt_duration(pending_duration)}! "
                f"Likely resource exhaustion or scheduling constraint.",
                pod_name=pod.name,
                suggestion=(
                    f"1) kubectl describe pod -n {self.namespace} {pod.name} | tail -20\n"
                    f"  2) kubectl get ns | grep aiperf | wc -l  (check stale namespaces)\n"
                    f"  3) kubectl describe node | grep -A 10 'Allocated'"
                ),
            )
        elif pending_duration > self.pending_threshold and not tl.pending_warned:
            tl.pending_warned = True
            self._add_problem(
                ProblemSeverity.WARNING,
                "pod-pending",
                f"Pod {pod.name} Pending for {_fmt_duration(pending_duration)} "
                f"(threshold: {_fmt_duration(self.pending_threshold)}). "
                f"May be waiting for resources.",
                pod_name=pod.name,
                suggestion=(
                    f"kubectl describe pod -n {self.namespace} {pod.name} | tail -20"
                ),
            )

    def _check_crash_loop(self, pod: PodStatus, tl: PodTimeline) -> None:
        if pod.restarts <= tl.last_restart_count:
            tl.last_restart_count = pod.restarts
            return

        old_count = tl.last_restart_count
        tl.last_restart_count = pod.restarts
        tl.restart_count = pod.restarts
        logger.info(
            f"[WATCHDOG] Restart detected: {tl.role}({pod.name.split('-')[-1]}) "
            f"restarts {old_count} -> {pod.restarts}"
        )

        if pod.restarts >= self.crashloop_threshold and not tl.crashloop_warned:
            tl.crashloop_warned = True
            self._add_problem(
                ProblemSeverity.CRITICAL,
                "crash-loop",
                f"Pod {pod.name} restarted {pod.restarts}x - likely CrashLoopBackOff.",
                pod_name=pod.name,
                suggestion=(f"kubectl -n {self.namespace} logs {pod.name} --previous"),
            )

    def _check_container_states(self, pod: PodStatus) -> None:
        for cname, cinfo in pod.containers.items():
            state = cinfo.get("state", {})

            # Waiting states
            waiting = state.get("waiting", {})
            reason = waiting.get("reason", "")
            if reason in (
                "CrashLoopBackOff",
                "ImagePullBackOff",
                "ErrImagePull",
                "CreateContainerConfigError",
                "InvalidImageName",
            ):
                fp = f"{pod.name}/{cname}/{reason}"
                if fp not in self._event_fingerprints:
                    self._event_fingerprints.add(fp)
                    self._add_problem(
                        ProblemSeverity.CRITICAL,
                        f"container-{reason.lower()}",
                        f"{cname} in {pod.name}: {reason} - "
                        f"{waiting.get('message', 'N/A')[:100]}",
                        pod_name=pod.name,
                    )

            # Terminated states
            terminated = state.get("terminated", {})
            term_reason = terminated.get("reason", "")
            exit_code = terminated.get("exitCode")

            if term_reason == "OOMKilled":
                fp = f"{pod.name}/{cname}/OOMKilled"
                if fp not in self._event_fingerprints:
                    self._event_fingerprints.add(fp)
                    self._add_problem(
                        ProblemSeverity.CRITICAL,
                        "oom-killed",
                        f"{cname} in {pod.name}: OOMKilled. "
                        f"Process exceeded memory limits.",
                        pod_name=pod.name,
                        suggestion="Increase memory limits in benchmark config.",
                    )

            elif exit_code is not None and exit_code != 0:
                fp = f"{pod.name}/{cname}/exit-{exit_code}"
                if fp not in self._event_fingerprints:
                    self._event_fingerprints.add(fp)
                    logger.warning(
                        f"[WATCHDOG] {cname} in {pod.name}: "
                        f"exit code {exit_code} (reason: {term_reason or 'Completed'})"
                    )

    def _check_pod_completion(self, pod: PodStatus) -> None:
        """Log when a pod completes for the first time."""
        if pod.name in self._completed_pods:
            return
        if pod.phase not in ("Succeeded", "Failed"):
            return

        self._completed_pods.add(pod.name)
        elapsed = time.time() - self._start_time
        tl = self._pod_timelines.get(pod.name)
        pod_age = (time.time() - tl.first_seen) if tl else 0
        role = tl.role if tl else _pod_role(pod.name)
        short = pod.name.split("-")[-1] if "-" in pod.name else pod.name

        if pod.phase == "Succeeded":
            logger.info(
                f"[WATCHDOG] {role}({short}) completed successfully "
                f"(age={_fmt_duration(pod_age)}, +{_fmt_duration(elapsed)})"
            )
        else:
            self._add_problem(
                ProblemSeverity.CRITICAL,
                "pod-failed",
                f"Pod {pod.name} FAILED after {_fmt_duration(pod_age)}.",
                pod_name=pod.name,
                suggestion=(
                    f"kubectl -n {self.namespace} logs {pod.name} --all-containers"
                ),
            )

    # ------------------------------------------------------------------
    # Event analysis
    # ------------------------------------------------------------------

    async def _check_events(self) -> None:
        """Watch K8s events for scheduling/resource problems."""
        try:
            result = await self.kubectl.run(
                "get",
                "events",
                "-o",
                "json",
                namespace=self.namespace,
                check=False,
            )
            if result.returncode != 0:
                return

            import orjson

            data = orjson.loads(result.stdout)
            for event in data.get("items", []):
                self._process_event(event)
        except Exception:
            pass

    def _process_event(self, event: dict[str, Any]) -> None:
        reason = event.get("reason", "")
        message = event.get("message", "")
        involved = event.get("involvedObject", {})
        pod_name = involved.get("name", "")
        event_type = event.get("type", "Normal")
        fp = f"{pod_name}/{reason}/{message[:80]}"

        if fp in self._event_fingerprints:
            return

        if reason == "FailedScheduling":
            self._event_fingerprints.add(fp)
            severity = (
                ProblemSeverity.CRITICAL
                if "Insufficient" in message
                else ProblemSeverity.WARNING
            )
            self._add_problem(
                severity,
                "scheduling-failure",
                f"{pod_name}: FailedScheduling - {message[:120]}",
                pod_name=pod_name,
                suggestion=(
                    "kubectl get ns | grep aiperf- | wc -l  "
                    "(clean up stale namespaces if > 5)"
                ),
            )

        elif reason in ("FailedMount", "FailedAttachVolume"):
            self._event_fingerprints.add(fp)
            self._add_problem(
                ProblemSeverity.WARNING,
                "volume-issue",
                f"{pod_name}: {reason} - {message[:120]}",
                pod_name=pod_name,
            )

        elif reason == "BackOff" and event_type == "Warning":
            self._event_fingerprints.add(fp)
            self._add_problem(
                ProblemSeverity.WARNING,
                "container-backoff",
                f"{pod_name}: BackOff - {message[:120]}",
                pod_name=pod_name,
            )

        elif reason == "Killing":
            self._event_fingerprints.add(fp)
            logger.info(f"[WATCHDOG] Event: {pod_name} being killed - {message[:100]}")

        elif reason == "Unhealthy" and event_type == "Warning":
            self._event_fingerprints.add(fp)
            short = pod_name.split("-")[-1] if "-" in pod_name else pod_name
            logger.debug(
                lambda short=short, message=message: (
                    f"[WATCHDOG] Probe failure on {short}: {message[:80]}"
                )
            )

    # ------------------------------------------------------------------
    # Cluster-level checks
    # ------------------------------------------------------------------

    def _check_elapsed_time(self) -> None:
        """Escalating timeout warnings."""
        elapsed = time.time() - self._start_time
        remaining = self.timeout - elapsed

        if remaining < 60 and not any(
            p.category == "timeout-warning" for p in self._problems
        ):
            self._add_problem(
                ProblemSeverity.WARNING,
                "timeout-warning",
                f"<60s remaining ({_fmt_duration(remaining)} of "
                f"{_fmt_duration(self.timeout)}). Should be completing.",
            )

        if remaining < 15 and not any(
            p.category == "timeout-imminent" for p in self._problems
        ):
            self._add_problem(
                ProblemSeverity.CRITICAL,
                "timeout-imminent",
                f"TIMEOUT IMMINENT: {remaining:.0f}s left! Will be killed.",
            )

    async def _check_node_resources(self) -> None:
        """Parse node resource allocation."""
        try:
            result = await self.kubectl.run(
                "describe",
                "nodes",
                check=False,
            )
            if result.returncode != 0:
                return

            in_allocated = False
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if "Allocated resources" in stripped:
                    in_allocated = True
                    continue
                if not in_allocated:
                    continue

                if stripped.startswith("cpu"):
                    self._node_cpu_pct = self._parse_resource_pct(stripped)
                    if self._node_cpu_pct and self._node_cpu_pct > 80:
                        self._add_problem(
                            ProblemSeverity.WARNING,
                            "node-cpu-pressure",
                            f"Node CPU requests at {self._node_cpu_pct}% - "
                            f"new pods may fail to schedule.",
                            suggestion="Clean up stale aiperf-* namespaces.",
                        )
                    elif self._node_cpu_pct:
                        logger.info(
                            f"[WATCHDOG] Node CPU allocation: {self._node_cpu_pct}%"
                        )

                elif stripped.startswith("memory"):
                    self._node_mem_pct = self._parse_resource_pct(stripped)
                    if self._node_mem_pct:
                        logger.info(
                            f"[WATCHDOG] Node memory allocation: {self._node_mem_pct}%"
                        )
                    break
        except Exception:
            pass

    @staticmethod
    def _parse_resource_pct(line: str) -> int | None:
        """Parse first percentage from a describe nodes resource line."""
        import re

        match = re.search(r"\((\d+)%\)", line)
        return int(match.group(1)) if match else None

    async def _check_stale_namespaces(self) -> None:
        """Detect leftover aiperf-* namespaces from previous runs."""
        try:
            result = await self.kubectl.run(
                "get",
                "ns",
                "-o",
                "name",
                check=False,
            )
            if result.returncode != 0:
                return

            stale = [
                ns.strip().removeprefix("namespace/")
                for ns in result.stdout.splitlines()
                if ns.strip().startswith("namespace/aiperf-")
                and self.namespace not in ns
                and "aiperf-system" not in ns
            ]
            self._stale_ns_count = len(stale)

            if len(stale) > 5:
                self._add_problem(
                    ProblemSeverity.WARNING,
                    "stale-namespaces",
                    f"Found {len(stale)} stale aiperf-* namespaces. "
                    f"These consume cluster resources.",
                    suggestion=(
                        "kubectl get ns -o name | grep aiperf- | "
                        "xargs kubectl delete --wait=false"
                    ),
                )
            elif stale:
                logger.info(
                    f"[WATCHDOG] Found {len(stale)} other aiperf-* namespace(s) "
                    f"(within normal range)"
                )
            else:
                logger.info("[WATCHDOG] Cluster clean - no stale namespaces")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Status dashboard
    # ------------------------------------------------------------------

    def _print_status_dashboard(self, pods: list[PodStatus]) -> None:
        """Print a rich real-time status dashboard."""
        elapsed = time.time() - self._start_time
        remaining = max(0, self.timeout - elapsed)
        pct = min(100, (elapsed / self.timeout) * 100) if self.timeout > 0 else 0

        # Build the dashboard
        lines = [
            _DLINE,
            _row(f"WATCHDOG  |  ns={self.namespace}"),
            _row(
                f"time: {_fmt_duration(elapsed)} elapsed, "
                f"{_fmt_duration(remaining)} remaining  "
                f"{_progress_bar(pct, 20)}"
            ),
        ]

        # Node resources (if known)
        if self._node_cpu_pct is not None:
            node_line = f"node: cpu={self._node_cpu_pct}%"
            if self._node_mem_pct is not None:
                node_line += f"  mem={self._node_mem_pct}%"
            if self._stale_ns_count > 0:
                node_line += f"  stale_ns={self._stale_ns_count}"
            lines.append(_row(node_line))

        # Pod table
        lines.append(_row("-" * (_W - 2)))
        lines.append(
            _row(
                f"{'':>4} {'POD':<36} {'PHASE':<12} {'READY':<6} {'RST':>3} {'AGE':>6}"
            )
        )
        lines.append(_row("-" * (_W - 2)))

        for pod in pods:
            tl = self._pod_timelines.get(pod.name)
            age = (time.time() - tl.first_seen) if tl else 0
            icon = _phase_icon(pod.phase)
            short = _short_pod_name(pod.name, 36)

            lines.append(
                _row(
                    f"{icon} {short:<36} {pod.phase:<12} {pod.ready:<6} "
                    f"{pod.restarts:>3} {_fmt_duration(age):>6}"
                )
            )

        # Problems summary
        crits = sum(1 for p in self._problems if p.severity == ProblemSeverity.CRITICAL)
        warns = sum(1 for p in self._problems if p.severity == ProblemSeverity.WARNING)
        if crits or warns:
            lines.append(_row("-" * (_W - 2)))
            parts = []
            if crits:
                parts.append(f"{crits} CRITICAL")
            if warns:
                parts.append(f"{warns} WARNING")
            lines.append(_row(f"issues: {', '.join(parts)}"))

        lines.append(_DLINE)
        logger.info("[WATCHDOG]\n" + "\n".join(lines))

    # ------------------------------------------------------------------
    # Problem tracking
    # ------------------------------------------------------------------

    def _add_problem(
        self,
        severity: ProblemSeverity,
        category: str,
        message: str,
        pod_name: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        problem = WatchdogProblem(
            severity=severity,
            category=category,
            message=message,
            pod_name=pod_name,
            namespace=self.namespace,
            suggestion=suggestion,
        )
        self._problems.append(problem)

        if severity == ProblemSeverity.CRITICAL:
            logger.error(f"[WATCHDOG:CRITICAL] {message}")
        elif severity == ProblemSeverity.WARNING:
            logger.warning(f"[WATCHDOG:WARNING] {message}")
        else:
            logger.info(f"[WATCHDOG:INFO] {message}")

        if suggestion:
            logger.info(f"[WATCHDOG]  -> {suggestion}")

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    def _print_final_report(self) -> None:
        """Comprehensive final watchdog report."""
        elapsed = time.time() - self._start_time
        total_pods = len(self._pod_timelines)
        succeeded = sum(
            1
            for tl in self._pod_timelines.values()
            if tl.last_phase in ("Succeeded", "Completed")
        )
        failed = sum(
            1 for tl in self._pod_timelines.values() if tl.last_phase == "Failed"
        )
        total_restarts = sum(tl.restart_count for tl in self._pod_timelines.values())

        lines = [
            "",
            _DLINE,
            _row("WATCHDOG FINAL REPORT"),
            _row(f"Namespace:  {self.namespace}"),
            _row(
                f"Duration:   {_fmt_duration(elapsed)} "
                f"(timeout was {_fmt_duration(self.timeout)})"
            ),
            _row(
                f"Pods:       {total_pods} tracked, "
                f"{succeeded} succeeded, {failed} failed, "
                f"{total_restarts} total restarts"
            ),
        ]

        # Node resources
        if self._node_cpu_pct is not None:
            lines.append(
                _row(
                    f"Node:       CPU {self._node_cpu_pct}%, "
                    f"Memory {self._node_mem_pct or '?'}%"
                )
            )

        # Pod timeline
        if self._pod_timelines:
            lines.append(_row("-" * (_W - 2)))
            lines.append(_row("POD LIFECYCLE:"))
            for tl in self._pod_timelines.values():
                short = tl.name.split("-")[-1] if "-" in tl.name else tl.name

                # Calculate time in each phase
                phase_times = []
                for i, (ts, phase) in enumerate(tl.phase_history):
                    if i + 1 < len(tl.phase_history):
                        dt = tl.phase_history[i + 1][0] - ts
                        phase_times.append(f"{phase}({_fmt_duration(dt)})")
                    else:
                        phase_times.append(phase)

                timing = " -> ".join(phase_times)
                rst_str = f" [{tl.restart_count}rst]" if tl.restart_count else ""
                lines.append(_row(f"  {tl.role[:4]}({short}): {timing}{rst_str}"))

        # Problems
        crits = [p for p in self._problems if p.severity == ProblemSeverity.CRITICAL]
        warns = [p for p in self._problems if p.severity == ProblemSeverity.WARNING]

        lines.append(_row("-" * (_W - 2)))
        if crits or warns:
            lines.append(_row(f"ISSUES: {len(crits)} critical, {len(warns)} warnings"))
            for p in crits:
                msg = p.message[: (_W - 14)]
                lines.append(_row(f"  [CRIT] {msg}"))
            for p in warns:
                msg = p.message[: (_W - 14)]
                lines.append(_row(f"  [WARN] {msg}"))
        else:
            lines.append(_row("STATUS: Clean run - no problems detected"))

        lines.append(_DLINE)
        logger.info("[WATCHDOG]" + "\n".join(lines))
