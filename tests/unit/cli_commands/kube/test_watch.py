# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.cli_commands.kube.watch module.

Focuses on:
- Namespace resolution: explicit, job-id, all-namespaces, fallback to last benchmark
- Poll loop: terminal status exits, timeout exits, keyboard interrupt handling
- Argument forwarding to AIPerfKubeClient and console functions
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.cli_commands.kube.watch import watch
from tests.harness.time_traveler import TimeTraveler

# ============================================================
# Helpers
# ============================================================


def _make_job(
    name: str = "aiperf-abc123",
    namespace: str = "bench-ns",
    status: str = "Running",
) -> MagicMock:
    """Create a mock JobSetInfo-like object."""
    job = MagicMock()
    job.name = name
    job.namespace = namespace
    job.status = status
    return job


def _make_pod_summary(ready: int = 1, total: int = 1, restarts: int = 0) -> MagicMock:
    summary = MagicMock()
    summary.ready = ready
    summary.total = total
    summary.restarts = restarts
    return summary


@dataclass
class _LastBenchmark:
    job_id: str
    namespace: str
    name: str | None = None


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_kube_client() -> AsyncMock:
    """AIPerfKubeClient mock with sensible defaults."""
    client = AsyncMock()
    client.list_jobsets.return_value = [_make_job()]
    client.get_pod_summary.return_value = _make_pod_summary()
    return client


@pytest.fixture
def patch_client(mock_kube_client: AsyncMock) -> AsyncMock:
    """Patch AIPerfKubeClient.create to return mock_kube_client."""
    with patch(
        "aiperf.kubernetes.client.AIPerfKubeClient.create",
        return_value=mock_kube_client,
    ):
        yield mock_kube_client


@pytest.fixture
def patch_console() -> dict[str, MagicMock]:
    """Patch all kube_console print functions used by watch."""
    with (
        patch("aiperf.kubernetes.console.print_header") as mock_header,
        patch("aiperf.kubernetes.console.print_info") as mock_info,
        patch("aiperf.kubernetes.console.print_warning") as mock_warning,
        patch("aiperf.kubernetes.console.print_jobs_table") as mock_table,
    ):
        yield {
            "header": mock_header,
            "info": mock_info,
            "warning": mock_warning,
            "table": mock_table,
        }


# ============================================================
# Happy Path Tests
# ============================================================


class TestWatchHappyPath:
    """Verify normal successful watch operations."""

    @pytest.mark.asyncio
    async def test_watch_explicit_namespace_lists_jobsets(
        self,
        mock_kube_client: AsyncMock,
        patch_client: AsyncMock,
        patch_console: dict,
    ) -> None:
        """Explicit --namespace forwards to list_jobsets and displays table."""
        terminal_job = _make_job(status="Completed")
        mock_kube_client.list_jobsets.return_value = [terminal_job]

        await watch(namespace="bench-ns")

        # Initial call + poll loop call
        assert mock_kube_client.list_jobsets.call_count == 2
        mock_kube_client.list_jobsets.assert_called_with(
            namespace="bench-ns",
            all_namespaces=False,
            job_id=None,
        )
        patch_console["table"].assert_called()

    @pytest.mark.asyncio
    async def test_watch_job_id_forwards_to_list_jobsets(
        self,
        mock_kube_client: AsyncMock,
        patch_client: AsyncMock,
        patch_console: dict,
    ) -> None:
        """--job-id is forwarded to list_jobsets as job_id parameter."""
        terminal_job = _make_job(status="Failed")
        mock_kube_client.list_jobsets.return_value = [terminal_job]

        await watch(job_id="xyz789")

        mock_kube_client.list_jobsets.assert_called_with(
            namespace=None,
            all_namespaces=False,
            job_id="xyz789",
        )

    @pytest.mark.asyncio
    async def test_watch_all_namespaces_flag(
        self,
        mock_kube_client: AsyncMock,
        patch_client: AsyncMock,
        patch_console: dict,
    ) -> None:
        """--all-namespaces passes all_namespaces=True to list_jobsets."""
        terminal_job = _make_job(status="Completed")
        mock_kube_client.list_jobsets.return_value = [terminal_job]

        await watch(all_namespaces=True)

        mock_kube_client.list_jobsets.assert_called_with(
            namespace=None,
            all_namespaces=True,
            job_id=None,
        )

    @pytest.mark.asyncio
    async def test_watch_kubeconfig_and_context_forwarded(
        self,
        patch_console: dict,
    ) -> None:
        """--kubeconfig and --context are forwarded to AIPerfKubeClient.create."""
        mock_client = AsyncMock()
        terminal_job = _make_job(status="Completed")
        mock_client.list_jobsets.return_value = [terminal_job]
        mock_client.get_pod_summary.return_value = _make_pod_summary()

        with patch(
            "aiperf.kubernetes.client.AIPerfKubeClient.create",
            return_value=mock_client,
        ) as mock_create:
            await watch(
                namespace="ns",
                kubeconfig="/path/to/config",
                context="prod-cluster",
            )

            mock_create.assert_called_once_with(
                kubeconfig="/path/to/config",
                kube_context="prod-cluster",
            )

    @pytest.mark.asyncio
    async def test_watch_pod_summaries_collected_for_each_job(
        self,
        mock_kube_client: AsyncMock,
        patch_client: AsyncMock,
        patch_console: dict,
    ) -> None:
        """get_pod_summary is called for each job and results passed to table."""
        job_a = _make_job(name="job-a", status="Completed")
        job_b = _make_job(name="job-b", status="Failed")
        mock_kube_client.list_jobsets.return_value = [job_a, job_b]

        summary_a = _make_pod_summary(ready=2, total=2)
        summary_b = _make_pod_summary(ready=0, total=1)
        mock_kube_client.get_pod_summary.side_effect = [
            # First call in initial list_jobsets (not in loop)
            # Actually list_jobsets is called twice: initial + loop
            summary_a,
            summary_b,
        ]

        await watch(namespace="ns")

        # print_jobs_table receives pod_summaries dict
        table_call = patch_console["table"].call_args
        assert "job-a" in table_call.kwargs["pod_summaries"]
        assert "job-b" in table_call.kwargs["pod_summaries"]


# ============================================================
# Fallback to Last Benchmark
# ============================================================


class TestWatchFallback:
    """Verify fallback to last benchmark when no args provided."""

    @pytest.mark.asyncio
    async def test_watch_no_args_uses_last_benchmark(
        self,
        mock_kube_client: AsyncMock,
        patch_client: AsyncMock,
        patch_console: dict,
    ) -> None:
        """When no namespace/job_id/all_namespaces, falls back to get_last_benchmark."""
        terminal_job = _make_job(status="Completed")
        mock_kube_client.list_jobsets.return_value = [terminal_job]
        last = _LastBenchmark(job_id="last-123", namespace="saved-ns")

        with patch(
            "aiperf.kubernetes.console.get_last_benchmark",
            return_value=last,
        ):
            await watch()

        mock_kube_client.list_jobsets.assert_called_with(
            namespace="saved-ns",
            all_namespaces=False,
            job_id="last-123",
        )
        patch_console["info"].assert_any_call(
            "Watching last benchmark: last-123 in namespace saved-ns"
        )

    @pytest.mark.asyncio
    async def test_watch_no_args_no_last_benchmark_warns_and_returns(
        self,
        patch_client: AsyncMock,
        patch_console: dict,
    ) -> None:
        """When no args and no last benchmark, prints warning and returns early."""
        with patch(
            "aiperf.kubernetes.console.get_last_benchmark",
            return_value=None,
        ):
            await watch()

        patch_console["warning"].assert_called_once_with(
            "No matching namespaces found. Use -n, -j, or -A to specify targets."
        )

    @pytest.mark.asyncio
    async def test_watch_no_jobs_found_warns_and_returns(
        self,
        mock_kube_client: AsyncMock,
        patch_client: AsyncMock,
        patch_console: dict,
    ) -> None:
        """When list_jobsets returns empty, prints warning and returns."""
        mock_kube_client.list_jobsets.return_value = []

        await watch(namespace="empty-ns")

        patch_console["warning"].assert_called_once_with("No AIPerf benchmarks found.")
        patch_console["header"].assert_not_called()


# ============================================================
# Poll Loop Behavior
# ============================================================


class TestWatchPollLoop:
    """Verify poll loop termination conditions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status",
        [
            "Completed",
            "Failed",
        ],
    )  # fmt: skip
    async def test_watch_all_terminal_exits_loop(
        self,
        status: str,
        mock_kube_client: AsyncMock,
        patch_client: AsyncMock,
        patch_console: dict,
    ) -> None:
        """Loop exits immediately when all jobs are in a terminal status."""
        terminal_job = _make_job(status=status)
        mock_kube_client.list_jobsets.return_value = [terminal_job]

        await watch(namespace="ns")

        patch_console["info"].assert_any_call("All benchmarks have finished.")

    @pytest.mark.asyncio
    async def test_watch_mixed_statuses_continues_polling(
        self,
        mock_kube_client: AsyncMock,
        patch_client: AsyncMock,
        patch_console: dict,
    ) -> None:
        """Loop continues when some jobs are non-terminal, exits when all terminal."""
        running_job = _make_job(status="Running")
        completed_job = _make_job(name="done", status="Completed")

        # First poll: one running, one completed -> continue
        # Second poll: both completed -> exit
        mock_kube_client.list_jobsets.side_effect = [
            [running_job, completed_job],  # initial call
            [running_job, completed_job],  # first poll
            [completed_job, _make_job(name="done2", status="Completed")],  # second poll
        ]

        await watch(namespace="ns")

        # 3 list_jobsets calls: initial + 2 polls
        assert mock_kube_client.list_jobsets.call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_watch_timeout_exits_loop(
        self,
        mock_kube_client: AsyncMock,
        patch_client: AsyncMock,
        patch_console: dict,
        time_traveler_no_patch_sleep: TimeTraveler,
    ) -> None:
        """Timeout causes loop to exit with warning message.

        time_traveler_no_patch_sleep patches time.monotonic to return
        virtual time synced with loop.time(). Looptime handles
        asyncio.sleep, advancing virtual time by interval each tick.
        """
        running_job = _make_job(status="Running")
        mock_kube_client.list_jobsets.return_value = [running_job]

        await watch(namespace="ns", timeout=3, interval=5)

        patch_console["warning"].assert_called_with("Watch timeout reached (3s).")

    @pytest.mark.asyncio
    async def test_watch_keyboard_interrupt_prints_stopped(
        self,
        mock_kube_client: AsyncMock,
        patch_client: AsyncMock,
        patch_console: dict,
    ) -> None:
        """Ctrl+C during poll loop prints 'Watch stopped.' and exits cleanly."""
        running_job = _make_job(status="Running")
        # Initial call succeeds, first poll raises KeyboardInterrupt
        mock_kube_client.list_jobsets.side_effect = [
            [running_job],  # initial
            KeyboardInterrupt,  # poll
        ]

        await watch(namespace="ns")

        patch_console["info"].assert_any_call("Watch stopped.")

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_watch_interval_passed_to_sleep(
        self,
        mock_kube_client: AsyncMock,
        patch_client: AsyncMock,
        patch_console: dict,
        time_traveler_no_patch_sleep: TimeTraveler,
    ) -> None:
        """Custom interval advances virtual time by interval seconds per tick."""
        running_job = _make_job(status="Running")
        completed_job = _make_job(status="Completed")

        mock_kube_client.list_jobsets.side_effect = [
            [running_job],  # initial
            [running_job],  # first poll -> sleep(30) advances clock
            [completed_job],  # second poll -> terminal
        ]

        await watch(namespace="ns", interval=30)


# ============================================================
# Error Handling
# ============================================================


class TestWatchErrors:
    """Verify error handling via exit_on_error context manager."""

    @pytest.mark.asyncio
    async def test_watch_client_create_error_exits(
        self,
        patch_console: dict,
    ) -> None:
        """Exception from AIPerfKubeClient.create is caught by exit_on_error."""
        with (
            patch(
                "aiperf.kubernetes.client.AIPerfKubeClient.create",
                side_effect=RuntimeError("k8s unreachable"),
            ),
            pytest.raises(SystemExit),
        ):
            await watch(namespace="ns")

    @pytest.mark.asyncio
    async def test_watch_list_jobsets_error_exits(
        self,
        mock_kube_client: AsyncMock,
        patch_client: AsyncMock,
        patch_console: dict,
    ) -> None:
        """Exception from list_jobsets is caught by exit_on_error."""
        mock_kube_client.list_jobsets.side_effect = ConnectionError("API timeout")

        with pytest.raises(SystemExit):
            await watch(namespace="ns")
