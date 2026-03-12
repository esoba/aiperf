# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for Kubernetes operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiperf.kubernetes.constants import Annotations, Labels, ProgressAnnotations
from aiperf.kubernetes.enums import JobSetStatus


@dataclass
class JobSetInfo:
    """Information about a found JobSet.

    Use ``JobSetInfo.from_raw(raw_dict)`` to create from a Kubernetes API
    response dict.  All field extraction and status parsing is handled here.
    """

    name: str
    """Kubernetes JobSet resource name."""

    namespace: str
    """Kubernetes namespace containing the JobSet."""

    jobset: dict[str, Any]
    """Raw JobSet dict from the Kubernetes API."""

    status: str
    """Current status: "Running", "Completed", or "Failed"."""

    custom_name: str | None = None
    """User-provided benchmark name, if set."""

    model: str | None = None
    """Target model name from the endpoint, if set."""

    endpoint: str | None = None
    """Target LLM endpoint URL, if set."""

    # -- Factory ----------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> JobSetInfo:
        """Create a JobSetInfo from a raw Kubernetes JobSet dict."""
        metadata = raw.get("metadata", {})
        labels = metadata.get("labels", {})
        annotations = metadata.get("annotations", {})
        return cls(
            name=metadata["name"],
            namespace=metadata["namespace"],
            jobset=raw,
            status=cls._parse_status(raw),
            custom_name=labels.get(Labels.NAME),
            model=annotations.get(Annotations.MODEL),
            endpoint=annotations.get(Annotations.ENDPOINT),
        )

    # -- Derived properties -----------------------------------------------

    @property
    def job_id(self) -> str:
        """AIPerf job ID (falls back to the JobSet name)."""
        labels = self.jobset.get("metadata", {}).get("labels", {})
        return labels.get(Labels.JOB_ID, self.name)

    @property
    def created(self) -> str:
        """Creation timestamp from the JobSet metadata."""
        return self.jobset.get("metadata", {}).get("creationTimestamp", "")

    @property
    def progress(self) -> str | None:
        """Human-readable progress string, or None if unavailable."""
        annotations = self.jobset.get("metadata", {}).get("annotations", {})
        if not annotations.get(ProgressAnnotations.STATUS):
            return None

        parts: list[str] = []
        phase = annotations.get(ProgressAnnotations.PHASE, "")
        if phase:
            parts.append(phase)
        requests = annotations.get(ProgressAnnotations.REQUESTS)
        if requests:
            parts.append(requests)
        percent = annotations.get(ProgressAnnotations.PERCENT)
        if percent:
            parts.append(f"({percent}%)")
        return " ".join(parts) if parts else annotations.get(ProgressAnnotations.STATUS)

    # -- Private helpers --------------------------------------------------

    @staticmethod
    def _parse_status(raw: dict[str, Any]) -> str:
        """Extract status string from a raw JobSet dict."""
        conditions = raw.get("status", {}).get("conditions", [])
        condition_status = {c.get("type"): c.get("status") for c in conditions}
        if condition_status.get("Completed") == "True":
            return JobSetStatus.COMPLETED
        if condition_status.get("Failed") == "True":
            return JobSetStatus.FAILED
        return JobSetStatus.RUNNING


@dataclass
class PodSummary:
    """Summary of pod readiness for a JobSet."""

    ready: int
    """Number of pods with all containers ready and phase Running."""

    total: int
    """Total number of pods belonging to the JobSet."""

    restarts: int
    """Sum of container restart counts across all pods."""

    @property
    def ready_str(self) -> str:
        """Format as 'ready/total'."""
        return f"{self.ready}/{self.total}"
