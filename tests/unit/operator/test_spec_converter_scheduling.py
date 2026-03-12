# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AIPerfJobSpecConverter.to_scheduling_config()."""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.operator.spec_converter import AIPerfJobSpecConverter


def _minimal_spec(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid AIPerfJob spec with optional overrides."""
    spec: dict[str, Any] = {
        "models": ["test-model"],
        "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
        "datasets": {
            "main": {"type": "synthetic", "entries": 100, "prompts": {"isl": 128}}
        },
        "load": {"default": {"type": "concurrency", "concurrency": 1, "requests": 10}},
    }
    spec.update(overrides)
    return spec


class TestToSchedulingConfig:
    """Tests for the to_scheduling_config() method."""

    def test_returns_none_when_no_scheduling_section(self) -> None:
        spec = _minimal_spec()
        converter = AIPerfJobSpecConverter(spec, "job1", "default")
        result = converter.to_scheduling_config()
        assert result == {"queue_name": None, "priority_class": None}

    def test_extracts_queue_name(self) -> None:
        spec = _minimal_spec(scheduling={"queueName": "team-queue"})
        converter = AIPerfJobSpecConverter(spec, "job1", "default")
        result = converter.to_scheduling_config()
        assert result["queue_name"] == "team-queue"
        assert result["priority_class"] is None

    def test_extracts_priority_class(self) -> None:
        spec = _minimal_spec(scheduling={"priorityClass": "high"})
        converter = AIPerfJobSpecConverter(spec, "job1", "default")
        result = converter.to_scheduling_config()
        assert result["queue_name"] is None
        assert result["priority_class"] == "high"

    def test_extracts_both_fields(self) -> None:
        spec = _minimal_spec(
            scheduling={
                "queueName": "shared-queue",
                "priorityClass": "batch-low",
            }
        )
        converter = AIPerfJobSpecConverter(spec, "job1", "default")
        result = converter.to_scheduling_config()
        assert result["queue_name"] == "shared-queue"
        assert result["priority_class"] == "batch-low"

    def test_ignores_unknown_fields(self) -> None:
        spec = _minimal_spec(
            scheduling={
                "queueName": "q",
                "priorityClass": "p",
                "unknownField": "should-be-ignored",
            }
        )
        converter = AIPerfJobSpecConverter(spec, "job1", "default")
        result = converter.to_scheduling_config()
        assert result == {"queue_name": "q", "priority_class": "p"}
        assert "unknownField" not in result

    def test_empty_scheduling_section(self) -> None:
        spec = _minimal_spec(scheduling={})
        converter = AIPerfJobSpecConverter(spec, "job1", "default")
        result = converter.to_scheduling_config()
        assert result == {"queue_name": None, "priority_class": None}

    @pytest.mark.parametrize(
        "queue_name,priority_class",
        [
            param("short", None, id="short-queue-name"),
            param("very-long-queue-name-with-many-segments", None, id="long-queue-name"),
            param(None, "default-priority", id="default-priority-class"),
            param("q", "p", id="single-char-values"),
        ],
    )  # fmt: skip
    def test_various_scheduling_values(
        self, queue_name: str | None, priority_class: str | None
    ) -> None:
        scheduling: dict[str, str] = {}
        if queue_name is not None:
            scheduling["queueName"] = queue_name
        if priority_class is not None:
            scheduling["priorityClass"] = priority_class

        spec = _minimal_spec(scheduling=scheduling)
        converter = AIPerfJobSpecConverter(spec, "job1", "default")
        result = converter.to_scheduling_config()
        assert result["queue_name"] == queue_name
        assert result["priority_class"] == priority_class
