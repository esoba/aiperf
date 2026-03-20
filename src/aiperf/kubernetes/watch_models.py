# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for the aiperf kube watch command."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ProgressSnapshot:
    percent: float = 0.0
    requests_completed: int = 0
    requests_total: int = 0
    eta_seconds: float | None = None
    duration_target_seconds: float | None = None


@dataclass(frozen=True)
class MetricsSnapshot:
    request_throughput_rps: float = 0.0
    request_latency_avg_ms: float = 0.0
    request_latency_p50_ms: float = 0.0
    request_latency_p99_ms: float = 0.0
    ttft_avg_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    time_to_second_token_avg_ms: float = 0.0
    inter_token_latency_avg_ms: float = 0.0
    inter_token_latency_p99_ms: float = 0.0
    output_token_throughput_tps: float = 0.0
    total_token_throughput_tps: float = 0.0
    prefill_throughput_per_user_tps: float = 0.0
    output_token_throughput_per_user_tps: float = 0.0
    request_count: int = 0
    error_count: int = 0
    goodput_rps: float = 0.0
    streaming: bool = False


@dataclass(frozen=True)
class WorkersSnapshot:
    ready: int = 0
    total: int = 0


@dataclass(frozen=True)
class PodSnapshot:
    name: str
    role: str  # "controller" or "worker"
    status: str
    restarts: int = 0
    ready: bool = False
    oom_killed: bool = False

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> PodSnapshot:
        metadata = raw.get("metadata", {})
        status = raw.get("status", {})
        name = metadata.get("name", "")
        role = "controller" if "controller" in name else "worker"
        containers = status.get("containerStatuses", [])
        restarts = sum(c.get("restartCount", 0) for c in containers)
        ready = all(c.get("ready", False) for c in containers)
        oom = any(
            c.get("lastState", {}).get("terminated", {}).get("reason") == "OOMKilled"
            for c in containers
        )
        return cls(
            name=name,
            role=role,
            status=status.get("phase", "Unknown"),
            restarts=restarts,
            ready=ready,
            oom_killed=oom,
        )


@dataclass(frozen=True)
class EventSnapshot:
    timestamp: str
    type: str
    reason: str
    object: str
    message: str
    count: int = 1


@dataclass(frozen=True)
class DiagnosisIssue:
    id: str
    severity: str  # "info", "warning", "critical"
    title: str
    detail: str
    impact: str
    suggested_fix: str
    runbook: str | None = None


@dataclass(frozen=True)
class DiagnosisResult:
    health: str = "healthy"  # healthy, degraded, stalled, failing, completed, failed
    issues: list[DiagnosisIssue] = field(default_factory=list)
    stalled: bool = False
    stall_reason: str | None = None
    error_rate: float = 0.0


@dataclass(frozen=True)
class WatchSnapshot:
    timestamp: datetime
    job_id: str
    namespace: str
    phase: str
    current_phase: str | None = None
    elapsed_seconds: float = 0.0
    progress: ProgressSnapshot | None = None
    metrics: MetricsSnapshot | None = None
    workers: WorkersSnapshot = field(default_factory=WorkersSnapshot)
    pods: list[PodSnapshot] = field(default_factory=list)
    events: list[EventSnapshot] = field(default_factory=list)
    conditions: dict[str, bool] = field(default_factory=dict)
    diagnosis: DiagnosisResult = field(default_factory=DiagnosisResult)
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    server_metrics: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    endpoint: str | None = None
    image: str | None = None
    results: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""

        def _convert_datetimes(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {k: _convert_datetimes(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert_datetimes(i) for i in obj]
            return obj

        return _convert_datetimes(dataclasses.asdict(self))
