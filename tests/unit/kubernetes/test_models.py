# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kubernetes.models module.

Focuses on:
- JobSetInfo construction via from_raw and direct instantiation
- Status parsing from Kubernetes conditions
- Derived properties (job_id, created, progress)
- PodSummary construction and ready_str formatting
"""

from __future__ import annotations

from typing import Any

import pytest

from aiperf.kubernetes.constants import Annotations, Labels, ProgressAnnotations
from aiperf.kubernetes.enums import JobSetStatus
from aiperf.kubernetes.models import JobSetInfo, PodSummary

# =============================================================================
# Helpers
# =============================================================================


def _make_raw_jobset(
    *,
    name: str = "aiperf-bench-001",
    namespace: str = "default",
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
    conditions: list[dict[str, str]] | None = None,
    creation_timestamp: str = "2026-01-15T10:30:00Z",
) -> dict[str, Any]:
    """Build a minimal raw JobSet dict for testing."""
    raw: dict[str, Any] = {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": creation_timestamp,
        },
        "status": {
            "conditions": conditions or [],
        },
    }
    if labels:
        raw["metadata"]["labels"] = labels
    if annotations:
        raw["metadata"]["annotations"] = annotations
    return raw


# =============================================================================
# JobSetInfo.from_raw — construction
# =============================================================================


class TestJobSetInfoFromRaw:
    """Verify from_raw extracts fields correctly from raw Kubernetes dicts."""

    def test_from_raw_extracts_name_and_namespace(self) -> None:
        raw = _make_raw_jobset(name="my-bench", namespace="perf")
        info = JobSetInfo.from_raw(raw)
        assert info.name == "my-bench"
        assert info.namespace == "perf"

    def test_from_raw_stores_full_jobset_dict(self) -> None:
        raw = _make_raw_jobset()
        info = JobSetInfo.from_raw(raw)
        assert info.jobset is raw

    def test_from_raw_extracts_custom_name_from_label(self) -> None:
        raw = _make_raw_jobset(labels={Labels.NAME: "nightly-gpt4"})
        info = JobSetInfo.from_raw(raw)
        assert info.custom_name == "nightly-gpt4"

    def test_from_raw_custom_name_none_when_label_absent(self) -> None:
        raw = _make_raw_jobset()
        info = JobSetInfo.from_raw(raw)
        assert info.custom_name is None

    def test_from_raw_extracts_model_from_annotation(self) -> None:
        raw = _make_raw_jobset(annotations={Annotations.MODEL: "llama-70b"})
        info = JobSetInfo.from_raw(raw)
        assert info.model == "llama-70b"

    def test_from_raw_extracts_endpoint_from_annotation(self) -> None:
        raw = _make_raw_jobset(annotations={Annotations.ENDPOINT: "http://llm:8000/v1"})
        info = JobSetInfo.from_raw(raw)
        assert info.endpoint == "http://llm:8000/v1"

    def test_from_raw_model_and_endpoint_none_when_absent(self) -> None:
        raw = _make_raw_jobset()
        info = JobSetInfo.from_raw(raw)
        assert info.model is None
        assert info.endpoint is None

    def test_from_raw_handles_missing_labels_and_annotations_keys(self) -> None:
        """metadata with no labels/annotations keys at all should not raise."""
        raw: dict[str, Any] = {
            "metadata": {"name": "x", "namespace": "ns"},
            "status": {"conditions": []},
        }
        info = JobSetInfo.from_raw(raw)
        assert info.custom_name is None
        assert info.model is None
        assert info.endpoint is None


# =============================================================================
# JobSetInfo._parse_status
# =============================================================================


class TestJobSetInfoParseStatus:
    """Verify status parsing from Kubernetes conditions."""

    def test_no_conditions_returns_running(self) -> None:
        raw = _make_raw_jobset(conditions=[])
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.RUNNING

    def test_completed_true_returns_completed(self) -> None:
        raw = _make_raw_jobset(conditions=[{"type": "Completed", "status": "True"}])
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.COMPLETED

    def test_failed_true_returns_failed(self) -> None:
        raw = _make_raw_jobset(conditions=[{"type": "Failed", "status": "True"}])
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.FAILED

    def test_completed_false_returns_running(self) -> None:
        raw = _make_raw_jobset(conditions=[{"type": "Completed", "status": "False"}])
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.RUNNING

    def test_failed_false_returns_running(self) -> None:
        raw = _make_raw_jobset(conditions=[{"type": "Failed", "status": "False"}])
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.RUNNING

    def test_both_completed_and_failed_completed_wins(self) -> None:
        """Completed is checked first, so it takes priority."""
        raw = _make_raw_jobset(
            conditions=[
                {"type": "Completed", "status": "True"},
                {"type": "Failed", "status": "True"},
            ]
        )
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.COMPLETED

    def test_missing_status_key_returns_running(self) -> None:
        raw: dict[str, Any] = {
            "metadata": {"name": "x", "namespace": "ns"},
        }
        assert JobSetInfo._parse_status(raw) == JobSetStatus.RUNNING

    def test_unrecognized_condition_type_returns_running(self) -> None:
        raw = _make_raw_jobset(conditions=[{"type": "Suspended", "status": "True"}])
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.RUNNING


# =============================================================================
# JobSetInfo derived properties
# =============================================================================


class TestJobSetInfoJobId:
    """Verify job_id property falls back to name."""

    def test_job_id_from_label(self) -> None:
        raw = _make_raw_jobset(labels={Labels.JOB_ID: "uuid-123"})
        info = JobSetInfo.from_raw(raw)
        assert info.job_id == "uuid-123"

    def test_job_id_falls_back_to_name(self) -> None:
        raw = _make_raw_jobset(name="bench-xyz")
        info = JobSetInfo.from_raw(raw)
        assert info.job_id == "bench-xyz"


class TestJobSetInfoCreated:
    """Verify created timestamp property."""

    def test_created_returns_timestamp(self) -> None:
        raw = _make_raw_jobset(creation_timestamp="2026-03-11T08:00:00Z")
        info = JobSetInfo.from_raw(raw)
        assert info.created == "2026-03-11T08:00:00Z"

    def test_created_returns_empty_when_missing(self) -> None:
        raw: dict[str, Any] = {
            "metadata": {"name": "x", "namespace": "ns"},
            "status": {"conditions": []},
        }
        info = JobSetInfo.from_raw(raw)
        assert info.created == ""


class TestJobSetInfoProgress:
    """Verify progress property builds human-readable strings."""

    def test_progress_none_when_no_status_annotation(self) -> None:
        raw = _make_raw_jobset()
        info = JobSetInfo.from_raw(raw)
        assert info.progress is None

    def test_progress_with_phase_only(self) -> None:
        raw = _make_raw_jobset(
            annotations={
                ProgressAnnotations.STATUS: "active",
                ProgressAnnotations.PHASE: "Warmup",
            }
        )
        info = JobSetInfo.from_raw(raw)
        assert info.progress == "Warmup"

    def test_progress_with_phase_and_requests(self) -> None:
        raw = _make_raw_jobset(
            annotations={
                ProgressAnnotations.STATUS: "active",
                ProgressAnnotations.PHASE: "Benchmark",
                ProgressAnnotations.REQUESTS: "500/1000",
            }
        )
        info = JobSetInfo.from_raw(raw)
        assert info.progress == "Benchmark 500/1000"

    def test_progress_with_all_fields(self) -> None:
        raw = _make_raw_jobset(
            annotations={
                ProgressAnnotations.STATUS: "active",
                ProgressAnnotations.PHASE: "Benchmark",
                ProgressAnnotations.REQUESTS: "500/1000",
                ProgressAnnotations.PERCENT: "50",
            }
        )
        info = JobSetInfo.from_raw(raw)
        assert info.progress == "Benchmark 500/1000 (50%)"

    def test_progress_with_percent_only(self) -> None:
        raw = _make_raw_jobset(
            annotations={
                ProgressAnnotations.STATUS: "active",
                ProgressAnnotations.PERCENT: "75",
            }
        )
        info = JobSetInfo.from_raw(raw)
        assert info.progress == "(75%)"

    def test_progress_falls_back_to_status_when_no_parts(self) -> None:
        """When STATUS is set but phase/requests/percent are all absent."""
        raw = _make_raw_jobset(
            annotations={
                ProgressAnnotations.STATUS: "initializing",
            }
        )
        info = JobSetInfo.from_raw(raw)
        assert info.progress == "initializing"

    def test_progress_none_when_status_empty_string(self) -> None:
        raw = _make_raw_jobset(
            annotations={
                ProgressAnnotations.STATUS: "",
            }
        )
        info = JobSetInfo.from_raw(raw)
        assert info.progress is None


# =============================================================================
# JobSetInfo — integration with conftest fixtures
# =============================================================================


class TestJobSetInfoWithFixtures:
    """Verify from_raw works with the shared sample_jobset fixtures."""

    def test_running_jobset(self, sample_running_jobset: dict[str, Any]) -> None:
        info = JobSetInfo.from_raw(sample_running_jobset)
        assert info.status == JobSetStatus.RUNNING
        assert info.name == "aiperf-test-job"

    def test_completed_jobset(self, sample_completed_jobset: dict[str, Any]) -> None:
        info = JobSetInfo.from_raw(sample_completed_jobset)
        assert info.status == JobSetStatus.COMPLETED

    def test_failed_jobset(self, sample_failed_jobset: dict[str, Any]) -> None:
        info = JobSetInfo.from_raw(sample_failed_jobset)
        assert info.status == JobSetStatus.FAILED

    def test_job_id_from_fixture(self, sample_jobset: dict[str, Any]) -> None:
        info = JobSetInfo.from_raw(sample_jobset)
        assert info.job_id == "test-job-123"


# =============================================================================
# PodSummary
# =============================================================================


class TestPodSummary:
    """Verify PodSummary construction and ready_str formatting."""

    @pytest.mark.parametrize(
        "ready,total,expected",
        [
            (0, 0, "0/0"),
            (0, 3, "0/3"),
            (2, 3, "2/3"),
            (5, 5, "5/5"),
        ],
    )  # fmt: skip
    def test_ready_str_format(self, ready: int, total: int, expected: str) -> None:
        summary = PodSummary(ready=ready, total=total, restarts=0)
        assert summary.ready_str == expected

    def test_stores_restart_count(self) -> None:
        summary = PodSummary(ready=1, total=2, restarts=7)
        assert summary.restarts == 7

    def test_all_fields_accessible(self) -> None:
        summary = PodSummary(ready=3, total=4, restarts=1)
        assert summary.ready == 3
        assert summary.total == 4
        assert summary.restarts == 1
