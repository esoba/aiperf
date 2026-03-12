# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.attach module.

Focuses on:
- attach_to_benchmark: early returns for missing/completed/failed jobs and non-running pods
- auto_attach_workflow: wait vs no-wait paths, ws vs log streaming, result retrieval
- retrieve_and_display_results: artifact retrieval, custom name handling, success/failure display
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.kubernetes.attach import (
    attach_to_benchmark,
    auto_attach_workflow,
    retrieve_and_display_results,
)
from aiperf.kubernetes.enums import JobSetStatus, PodPhase
from aiperf.kubernetes.models import JobSetInfo

# =============================================================================
# Helpers
# =============================================================================

_MODULE = "aiperf.kubernetes.attach"


def _make_jobset_info(
    status: str = JobSetStatus.RUNNING,
    name: str = "aiperf-test",
    namespace: str = "default",
    custom_name: str | None = None,
) -> JobSetInfo:
    """Create a minimal JobSetInfo for testing."""
    raw: dict = {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {},
            "annotations": {},
        },
        "status": {"conditions": []},
    }
    info = JobSetInfo(
        name=name,
        namespace=namespace,
        jobset=raw,
        status=status,
        custom_name=custom_name,
    )
    return info


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_client() -> MagicMock:
    """Mock AIPerfKubeClient with async methods."""
    client = MagicMock()
    client.find_jobset = AsyncMock(return_value=None)
    client.find_controller_pod = AsyncMock(return_value=None)
    client.wait_for_controller_pod_ready = AsyncMock(return_value="ctrl-pod-0")
    return client


@pytest.fixture
def patch_client_create(mock_client: MagicMock):
    """Patch AIPerfKubeClient.create to return mock_client."""
    with patch(f"{_MODULE}.AIPerfKubeClient") as cls_mock:
        cls_mock.create = AsyncMock(return_value=mock_client)
        yield cls_mock


@pytest.fixture
def patch_console():
    """Patch all console output functions to prevent terminal side-effects."""
    names = [
        "print_error",
        "print_warning",
        "print_action",
        "print_info",
        "print_success",
        "print_interrupt_info",
        "print_benchmark_complete",
        "print_results_summary",
        "status_log",
        "logger",
    ]
    mocks = {}
    patches = []
    for name in names:
        p = patch(f"{_MODULE}.{name}")
        mock = p.start()
        if name == "status_log":
            mock.return_value.__enter__ = MagicMock()
            mock.return_value.__exit__ = MagicMock(return_value=False)
        mocks[name] = mock
        patches.append(p)
    yield mocks
    for p in patches:
        p.stop()


# =============================================================================
# attach_to_benchmark Tests
# =============================================================================


class TestAttachToBenchmark:
    """Verify attach_to_benchmark early exits and happy path."""

    @pytest.mark.asyncio
    async def test_no_jobset_found_prints_error(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        mock_client.find_jobset.return_value = None

        await attach_to_benchmark("missing-id", "default", 8080)

        patch_console["print_error"].assert_called_once()
        assert "missing-id" in str(patch_console["print_error"].call_args)

    @pytest.mark.asyncio
    async def test_completed_job_prints_warning(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        mock_client.find_jobset.return_value = _make_jobset_info(
            status=JobSetStatus.COMPLETED
        )

        await attach_to_benchmark("job-1", "default", 8080)

        patch_console["print_warning"].assert_called_once()
        patch_console["print_action"].assert_called_once()
        mock_client.find_controller_pod.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_job_prints_error(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        mock_client.find_jobset.return_value = _make_jobset_info(
            status=JobSetStatus.FAILED
        )

        await attach_to_benchmark("job-1", "default", 8080)

        patch_console["print_error"].assert_called_once()
        patch_console["print_action"].assert_called_once()
        mock_client.find_controller_pod.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_controller_pod_prints_warning(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        mock_client.find_jobset.return_value = _make_jobset_info()
        mock_client.find_controller_pod.return_value = None

        await attach_to_benchmark("job-1", "default", 8080)

        patch_console["print_warning"].assert_called_once()
        assert "controller pod" in str(patch_console["print_warning"].call_args).lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase",
        [
            PodPhase.PENDING,
            PodPhase.SUCCEEDED,
            PodPhase.FAILED,
            PodPhase.UNKNOWN,
        ],
    )  # fmt: skip
    async def test_non_running_pod_prints_warning(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
        phase: PodPhase,
    ) -> None:
        mock_client.find_jobset.return_value = _make_jobset_info()
        mock_client.find_controller_pod.return_value = ("ctrl-0", phase)

        await attach_to_benchmark("job-1", "default", 8080)

        patch_console["print_warning"].assert_called_once()

    @pytest.mark.asyncio
    async def test_running_pod_starts_port_forward_and_streams(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        jobset = _make_jobset_info(namespace="bench-ns")
        mock_client.find_jobset.return_value = jobset
        mock_client.find_controller_pod.return_value = ("ctrl-0", PodPhase.RUNNING)

        with (
            patch(f"{_MODULE}.port_forward_with_status") as mock_pf,
            patch(f"{_MODULE}.stream_progress", new_callable=AsyncMock) as mock_stream,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = 9999
            mock_ctx.__aexit__.return_value = False
            mock_pf.return_value = mock_ctx

            await attach_to_benchmark("job-1", None, 8080)

            mock_pf.assert_called_once_with(
                "bench-ns",
                "ctrl-0",
                8080,
                kubeconfig=None,
                kube_context=None,
            )
            mock_stream.assert_awaited_once_with("ws://localhost:9999/ws")

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_prints_interrupt_info(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        jobset = _make_jobset_info(namespace="ns1")
        mock_client.find_jobset.return_value = jobset
        mock_client.find_controller_pod.return_value = ("ctrl-0", PodPhase.RUNNING)

        with patch(f"{_MODULE}.port_forward_with_status") as mock_pf:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.side_effect = KeyboardInterrupt
            mock_pf.return_value = mock_ctx

            await attach_to_benchmark("job-1", None, 8080)

            patch_console["print_interrupt_info"].assert_called_once_with(
                "job-1", "ns1"
            )

    @pytest.mark.asyncio
    async def test_passes_kube_creds_to_client(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        mock_client.find_jobset.return_value = None

        await attach_to_benchmark(
            "job-1",
            "default",
            8080,
            kubeconfig="/my/config",
            kube_context="my-ctx",
        )

        patch_client_create.create.assert_awaited_once_with(
            kubeconfig="/my/config", kube_context="my-ctx"
        )

    @pytest.mark.asyncio
    async def test_namespace_none_passes_to_find_jobset(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        mock_client.find_jobset.return_value = None

        await attach_to_benchmark("job-1", None, 8080)

        mock_client.find_jobset.assert_awaited_once_with("job-1", None)


# =============================================================================
# auto_attach_workflow Tests
# =============================================================================


class TestAutoAttachWorkflow:
    """Verify auto_attach_workflow wait/no-wait and streaming paths."""

    @pytest.mark.asyncio
    async def test_wait_for_ready_calls_wait_for_controller(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(f"{_MODULE}.stream_controller_logs", new_callable=AsyncMock),
            patch(f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock),
        ):
            await auto_attach_workflow("job-1", "ns1", 8080, wait_for_ready=True)

        mock_client.wait_for_controller_pod_ready.assert_awaited_once_with(
            "ns1", "job-1", timeout=300
        )

    @pytest.mark.asyncio
    async def test_no_wait_uses_find_controller_pod(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        mock_client.find_controller_pod.return_value = ("ctrl-0", PodPhase.RUNNING)

        with (
            patch(f"{_MODULE}.stream_controller_logs", new_callable=AsyncMock),
            patch(f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock),
        ):
            await auto_attach_workflow("job-1", "ns1", 8080, wait_for_ready=False)

        mock_client.find_controller_pod.assert_awaited_once_with("ns1", "job-1")
        mock_client.wait_for_controller_pod_ready.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_wait_no_pod_raises_runtime_error(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        mock_client.find_controller_pod.return_value = None

        with pytest.raises(RuntimeError, match="No controller pod found"):
            await auto_attach_workflow("job-1", "ns1", 8080, wait_for_ready=False)

    @pytest.mark.asyncio
    async def test_stream_ws_uses_port_forward(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(f"{_MODULE}.port_forward_with_status") as mock_pf,
            patch(f"{_MODULE}.stream_progress", new_callable=AsyncMock) as mock_stream,
            patch(f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock),
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = 9999
            mock_ctx.__aexit__.return_value = False
            mock_pf.return_value = mock_ctx

            await auto_attach_workflow("job-1", "ns1", 8080, stream_ws=True)

            mock_pf.assert_called_once()
            mock_stream.assert_awaited_once_with("ws://localhost:9999/ws")

    @pytest.mark.asyncio
    async def test_no_stream_ws_uses_controller_logs(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.stream_controller_logs", new_callable=AsyncMock
            ) as mock_logs,
            patch(f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock),
        ):
            await auto_attach_workflow("job-1", "ns1", 8080, stream_ws=False)

            mock_logs.assert_awaited_once_with(
                "ns1",
                "ctrl-pod-0",
                container="control-plane",
                kubeconfig=None,
                kube_context=None,
            )

    @pytest.mark.asyncio
    async def test_calls_retrieve_and_display_results(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(f"{_MODULE}.stream_controller_logs", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock
            ) as mock_retrieve,
        ):
            await auto_attach_workflow("job-1", "ns1", 8080)

            mock_retrieve.assert_awaited_once_with(
                "job-1", "ns1", mock_client, kubeconfig=None, kube_context=None
            )

    @pytest.mark.asyncio
    async def test_prints_benchmark_complete(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(f"{_MODULE}.stream_controller_logs", new_callable=AsyncMock),
            patch(f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock),
        ):
            await auto_attach_workflow("job-1", "ns1", 8080)

            patch_console["print_benchmark_complete"].assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_kube_creds_through(
        self,
        mock_client: MagicMock,
        patch_client_create: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(f"{_MODULE}.stream_controller_logs", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock
            ) as mock_retrieve,
        ):
            await auto_attach_workflow(
                "job-1",
                "ns1",
                8080,
                kubeconfig="/kc",
                kube_context="ctx",
            )

            patch_client_create.create.assert_awaited_once_with(
                kubeconfig="/kc", kube_context="ctx"
            )
            mock_retrieve.assert_awaited_once_with(
                "job-1", "ns1", mock_client, kubeconfig="/kc", kube_context="ctx"
            )


# =============================================================================
# retrieve_and_display_results Tests
# =============================================================================


class TestRetrieveAndDisplayResults:
    """Verify artifact retrieval and result display logic."""

    @pytest.fixture
    def mock_deps(self, mock_client: MagicMock):
        """Patch retrieve_all_artifacts and save_pod_logs."""
        with (
            patch(
                f"{_MODULE}.retrieve_all_artifacts", new_callable=AsyncMock
            ) as mock_retrieve,
            patch(f"{_MODULE}.save_pod_logs", new_callable=AsyncMock) as mock_logs,
            patch(f"{_MODULE}.print_results_summary") as mock_summary,
            patch(f"{_MODULE}.print_warning") as mock_warn,
            patch(f"{_MODULE}.print_action") as mock_action,
            patch("pathlib.Path.mkdir") as mock_mkdir,
        ):
            mock_retrieve.return_value = True
            yield {
                "retrieve": mock_retrieve,
                "logs": mock_logs,
                "summary": mock_summary,
                "warning": mock_warn,
                "action": mock_action,
                "mkdir": mock_mkdir,
            }

    @pytest.mark.asyncio
    async def test_success_prints_results_summary(
        self,
        mock_client: MagicMock,
        mock_deps: dict,
    ) -> None:
        mock_client.find_jobset.return_value = _make_jobset_info()
        mock_deps["retrieve"].return_value = True

        await retrieve_and_display_results("job-1", "ns1", mock_client)

        mock_deps["summary"].assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_prints_warning(
        self,
        mock_client: MagicMock,
        mock_deps: dict,
    ) -> None:
        mock_client.find_jobset.return_value = _make_jobset_info()
        mock_deps["retrieve"].return_value = False

        await retrieve_and_display_results("job-1", "ns1", mock_client)

        mock_deps["warning"].assert_called_once()
        mock_deps["action"].assert_called_once()
        mock_deps["summary"].assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_name_used_for_output_dir(
        self,
        mock_client: MagicMock,
        mock_deps: dict,
    ) -> None:
        mock_client.find_jobset.return_value = _make_jobset_info(
            custom_name="my-benchmark"
        )

        await retrieve_and_display_results("job-1", "ns1", mock_client)

        call_args = mock_deps["retrieve"].call_args
        output_dir = call_args[0][2]  # third positional arg
        assert "my-benchmark" in str(output_dir)

    @pytest.mark.asyncio
    async def test_job_id_used_when_no_custom_name(
        self,
        mock_client: MagicMock,
        mock_deps: dict,
    ) -> None:
        mock_client.find_jobset.return_value = _make_jobset_info(custom_name=None)

        await retrieve_and_display_results("job-1", "ns1", mock_client)

        call_args = mock_deps["retrieve"].call_args
        output_dir = call_args[0][2]
        assert "job-1" in str(output_dir)

    @pytest.mark.asyncio
    async def test_job_id_fallback_when_no_jobset_found(
        self,
        mock_client: MagicMock,
        mock_deps: dict,
    ) -> None:
        mock_client.find_jobset.return_value = None

        await retrieve_and_display_results("job-1", "ns1", mock_client)

        call_args = mock_deps["retrieve"].call_args
        output_dir = call_args[0][2]
        assert "job-1" in str(output_dir)

    @pytest.mark.asyncio
    async def test_save_pod_logs_always_called(
        self,
        mock_client: MagicMock,
        mock_deps: dict,
    ) -> None:
        mock_client.find_jobset.return_value = _make_jobset_info()
        mock_deps["retrieve"].return_value = False

        await retrieve_and_display_results("job-1", "ns1", mock_client)

        mock_deps["logs"].assert_awaited_once()

    @pytest.mark.asyncio
    async def test_passes_kube_creds_to_dependencies(
        self,
        mock_client: MagicMock,
        mock_deps: dict,
    ) -> None:
        mock_client.find_jobset.return_value = _make_jobset_info()

        await retrieve_and_display_results(
            "job-1", "ns1", mock_client, kubeconfig="/kc", kube_context="ctx"
        )

        retrieve_kwargs = mock_deps["retrieve"].call_args[1]
        assert retrieve_kwargs["kubeconfig"] == "/kc"
        assert retrieve_kwargs["kube_context"] == "ctx"

        logs_kwargs = mock_deps["logs"].call_args[1]
        assert logs_kwargs["kubeconfig"] == "/kc"
        assert logs_kwargs["kube_context"] == "ctx"

    @pytest.mark.asyncio
    async def test_creates_output_directory(
        self,
        mock_client: MagicMock,
        mock_deps: dict,
    ) -> None:
        mock_client.find_jobset.return_value = _make_jobset_info()

        await retrieve_and_display_results("job-1", "ns1", mock_client)

        mock_deps["mkdir"].assert_called_once_with(parents=True, exist_ok=True)
