# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.cli_commands.kube.cleanup module.

Focuses on:
- Namespace discovery and staleness filtering
- Dry-run vs force deletion behavior
- Running pod detection and skip/force logic
- Deletion error handling and continuation
- Duration formatting
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.cli_commands.kube.cleanup import _cleanup_stale_namespaces, _format_duration

# Timestamps guaranteed to be stale (hours old) or fresh (in the future)
_STALE_TS = "2020-01-01T00:00:00Z"
_CONSOLE = "aiperf.kubernetes.console"


def _fresh_timestamp() -> str:
    """Return an ISO timestamp 1 hour in the future -- always fresh."""
    future = datetime.now(tz=timezone.utc).replace(year=2099)
    return future.strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================
# Helpers
# ============================================================


def _make_namespace(name: str, created: str) -> AsyncMock:
    """Create a mock Namespace object with metadata."""
    ns = AsyncMock()
    ns.metadata = {"name": name, "creationTimestamp": created}
    ns.delete = AsyncMock()
    return ns


def _make_pod(phase: str = "Running") -> MagicMock:
    """Create a mock Pod with the given phase."""
    pod = MagicMock()
    pod.status = {"phase": phase}
    return pod


async def _run_cleanup(
    mock_kr8s: MagicMock,
    *,
    max_age: int = 3600,
    dry_run: bool = False,
    force: bool = False,
    label: str = "aiperf/job-id",
) -> dict[str, MagicMock]:
    """Run _cleanup_stale_namespaces with all console functions patched.

    Returns dict of mock console functions keyed by short name.
    """
    with (
        patch(f"{_CONSOLE}.print_info") as m_info,
        patch(f"{_CONSOLE}.print_header") as m_header,
        patch(f"{_CONSOLE}.print_success") as m_success,
        patch(f"{_CONSOLE}.print_warning") as m_warning,
        patch(f"{_CONSOLE}.print_error") as m_error,
    ):
        await _cleanup_stale_namespaces(
            max_age=max_age,
            dry_run=dry_run,
            force=force,
            label=label,
            kubeconfig=None,
            context=None,
        )

    return {
        "info": m_info,
        "header": m_header,
        "success": m_success,
        "warning": m_warning,
        "error": m_error,
    }


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_kr8s() -> MagicMock:
    """Patch kr8s.asyncio.api, Namespace.list, and Pod.list.

    Yields a holder with `.api`, `.ns_list`, `.pod_list` for test configuration.
    """
    holder = MagicMock()
    holder.api = AsyncMock()
    holder.ns_list = AsyncMock(return_value=[])
    holder.pod_list = AsyncMock(return_value=[])

    mock_api_fn = AsyncMock(return_value=holder.api)
    with (
        patch("kr8s.asyncio.api", mock_api_fn),
        patch("kr8s.asyncio.objects.Namespace.list", holder.ns_list),
        patch("kr8s.asyncio.objects.Pod.list", holder.pod_list),
    ):
        holder.api_fn = mock_api_fn
        yield holder


# ============================================================
# _format_duration Tests
# ============================================================


class TestFormatDuration:
    """Verify human-readable duration formatting."""

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "0s"),
            (30, "30s"),
            (59, "59s"),
            (60, "1m"),
            (90, "1m"),
            (3599, "59m"),
            (3600, "1h"),
            (5400, "1h 30m"),
            (7200, "2h"),
            param(7260, "2h 1m", id="hours-plus-one-minute"),
        ],
    )  # fmt: skip
    def test_format_duration_returns_expected(
        self, seconds: float, expected: str
    ) -> None:
        assert _format_duration(seconds) == expected


# ============================================================
# Happy Path Tests
# ============================================================


class TestCleanupHappyPath:
    """Verify normal cleanup operations."""

    @pytest.mark.asyncio
    async def test_no_namespaces_found_prints_info(self, mock_kr8s: MagicMock) -> None:
        mock_kr8s.ns_list.return_value = []

        out = await _run_cleanup(mock_kr8s)

        out["info"].assert_called_once_with("No aiperf benchmark namespaces found.")

    @pytest.mark.asyncio
    async def test_no_stale_namespaces_prints_info(self, mock_kr8s: MagicMock) -> None:
        mock_kr8s.ns_list.return_value = [
            _make_namespace("ns-fresh", _fresh_timestamp())
        ]
        mock_kr8s.pod_list.return_value = []

        out = await _run_cleanup(mock_kr8s)

        out["info"].assert_called_with("No stale namespaces found.")

    @pytest.mark.asyncio
    async def test_stale_namespace_dry_run_reports_without_deleting(
        self,
        mock_kr8s: MagicMock,
    ) -> None:
        ns = _make_namespace("ns-old", _STALE_TS)
        mock_kr8s.ns_list.return_value = [ns]
        mock_kr8s.pod_list.return_value = []

        out = await _run_cleanup(mock_kr8s, dry_run=True)

        ns.delete.assert_not_called()
        dry_run_calls = [c for c in out["info"].call_args_list if "Dry-run" in str(c)]
        assert len(dry_run_calls) == 1

    @pytest.mark.asyncio
    async def test_stale_namespace_force_deletes(self, mock_kr8s: MagicMock) -> None:
        ns = _make_namespace("ns-old", _STALE_TS)
        mock_kr8s.ns_list.return_value = [ns]
        mock_kr8s.pod_list.return_value = []

        out = await _run_cleanup(mock_kr8s, force=True)

        ns.delete.assert_awaited_once()
        out["success"].assert_called_once_with("Deleted namespace ns-old")

    @pytest.mark.asyncio
    async def test_multiple_stale_namespaces_all_deleted(
        self, mock_kr8s: MagicMock
    ) -> None:
        namespaces = [_make_namespace(f"ns-{i}", _STALE_TS) for i in range(3)]
        mock_kr8s.ns_list.return_value = namespaces
        mock_kr8s.pod_list.return_value = []

        out = await _run_cleanup(mock_kr8s, force=True)

        for ns in namespaces:
            ns.delete.assert_awaited_once()
        summary = [c for c in out["info"].call_args_list if "3/3" in str(c)]
        assert len(summary) == 1


# ============================================================
# Running Pod Tests
# ============================================================


class TestCleanupRunningPods:
    """Verify behavior when namespaces have running pods."""

    @pytest.mark.asyncio
    async def test_running_pods_skipped_without_force(
        self, mock_kr8s: MagicMock
    ) -> None:
        ns = _make_namespace("ns-running", _STALE_TS)
        mock_kr8s.ns_list.return_value = [ns]
        mock_kr8s.pod_list.return_value = [_make_pod("Running")]

        out = await _run_cleanup(mock_kr8s, force=False)

        out["warning"].assert_called_once()
        assert "ns-running" in str(out["warning"].call_args)
        assert "--force" in str(out["warning"].call_args)
        out["info"].assert_called_with("No stale namespaces found.")

    @pytest.mark.asyncio
    async def test_running_pods_deleted_with_force(self, mock_kr8s: MagicMock) -> None:
        ns = _make_namespace("ns-running", _STALE_TS)
        mock_kr8s.ns_list.return_value = [ns]
        mock_kr8s.pod_list.return_value = [_make_pod("Running"), _make_pod("Succeeded")]

        await _run_cleanup(mock_kr8s, force=True)

        ns.delete.assert_awaited_once()


# ============================================================
# Edge Cases
# ============================================================


class TestCleanupEdgeCases:
    """Verify boundary conditions and filtering logic."""

    @pytest.mark.asyncio
    async def test_mixed_stale_and_fresh_only_stale_deleted(
        self,
        mock_kr8s: MagicMock,
    ) -> None:
        stale_ns = _make_namespace("ns-stale", _STALE_TS)
        fresh_ns = _make_namespace("ns-fresh", _fresh_timestamp())
        mock_kr8s.ns_list.return_value = [stale_ns, fresh_ns]
        mock_kr8s.pod_list.return_value = []

        out = await _run_cleanup(mock_kr8s, force=True)

        stale_ns.delete.assert_awaited_once()
        fresh_ns.delete.assert_not_called()
        summary = [c for c in out["info"].call_args_list if "1/1" in str(c)]
        assert len(summary) == 1

    @pytest.mark.asyncio
    async def test_custom_max_age_makes_recent_namespace_stale(
        self,
        mock_kr8s: MagicMock,
    ) -> None:
        """A namespace created 2020 is stale even with max_age=1."""
        ns = _make_namespace("ns-medium", _STALE_TS)
        mock_kr8s.ns_list.return_value = [ns]
        mock_kr8s.pod_list.return_value = []

        await _run_cleanup(mock_kr8s, max_age=1, force=True)

        ns.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_custom_label_selector_forwarded(self, mock_kr8s: MagicMock) -> None:
        mock_kr8s.ns_list.return_value = []

        await _run_cleanup(mock_kr8s, label="custom/label")

        mock_kr8s.ns_list.assert_awaited_once()
        _, kwargs = mock_kr8s.ns_list.call_args
        assert kwargs["label_selector"] == "custom/label"

    @pytest.mark.asyncio
    async def test_namespace_missing_creation_timestamp_skipped(
        self,
        mock_kr8s: MagicMock,
    ) -> None:
        ns_no_ts = MagicMock()
        ns_no_ts.metadata = {"name": "ns-no-ts"}
        mock_kr8s.ns_list.return_value = [ns_no_ts]

        out = await _run_cleanup(mock_kr8s, force=True)

        out["info"].assert_called_with("No stale namespaces found.")

    @pytest.mark.asyncio
    async def test_force_without_dry_run_shows_deleting_action(
        self,
        mock_kr8s: MagicMock,
    ) -> None:
        mock_kr8s.ns_list.return_value = [_make_namespace("ns-old", _STALE_TS)]
        mock_kr8s.pod_list.return_value = []

        out = await _run_cleanup(mock_kr8s, force=True)

        action_calls = [c for c in out["info"].call_args_list if "Deleting" in str(c)]
        assert len(action_calls) == 1

    @pytest.mark.asyncio
    async def test_no_force_no_dry_run_shows_would_delete(
        self,
        mock_kr8s: MagicMock,
    ) -> None:
        mock_kr8s.ns_list.return_value = [_make_namespace("ns-old", _STALE_TS)]
        mock_kr8s.pod_list.return_value = []

        out = await _run_cleanup(mock_kr8s, force=False, dry_run=False)

        action_calls = [
            c for c in out["info"].call_args_list if "Would delete" in str(c)
        ]
        assert len(action_calls) == 1
        dry_run_calls = [c for c in out["info"].call_args_list if "Dry-run" in str(c)]
        assert len(dry_run_calls) == 1


# ============================================================
# Error Handling
# ============================================================


class TestCleanupErrors:
    """Verify deletion failure handling."""

    @pytest.mark.asyncio
    async def test_deletion_failure_continues_and_reports(
        self,
        mock_kr8s: MagicMock,
    ) -> None:
        ns_fail = _make_namespace("ns-fail", _STALE_TS)
        ns_fail.delete.side_effect = RuntimeError("API timeout")
        ns_ok = _make_namespace("ns-ok", _STALE_TS)
        mock_kr8s.ns_list.return_value = [ns_fail, ns_ok]
        mock_kr8s.pod_list.return_value = []

        out = await _run_cleanup(mock_kr8s, force=True)

        out["error"].assert_called_once()
        assert "ns-fail" in str(out["error"].call_args)
        out["success"].assert_called_once_with("Deleted namespace ns-ok")
        summary = [c for c in out["info"].call_args_list if "1/2" in str(c)]
        assert len(summary) == 1

    @pytest.mark.asyncio
    async def test_all_deletions_fail_reports_zero(self, mock_kr8s: MagicMock) -> None:
        ns = _make_namespace("ns-fail", _STALE_TS)
        ns.delete.side_effect = RuntimeError("forbidden")
        mock_kr8s.ns_list.return_value = [ns]
        mock_kr8s.pod_list.return_value = []

        out = await _run_cleanup(mock_kr8s, force=True)

        summary = [c for c in out["info"].call_args_list if "0/1" in str(c)]
        assert len(summary) == 1
