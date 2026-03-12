# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.cli_commands.kube.shutdown module.

Focuses on:
- Happy path: resolve jobset then call shutdown_api_service
- Early return when resolve_jobset returns None
- Default manage_options creation when None provided
- Correct argument forwarding to resolve_jobset and shutdown_api_service
"""

from unittest.mock import AsyncMock, patch

import pytest
from pytest import param

from aiperf.cli_commands.kube.shutdown import shutdown_benchmark
from aiperf.common.config.kube_config import KubeManageOptions
from aiperf.kubernetes.cli_helpers import ResolvedJobSet
from aiperf.kubernetes.models import JobSetInfo

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def manage_options() -> KubeManageOptions:
    """Create a KubeManageOptions instance for testing."""
    return KubeManageOptions(kubeconfig=None, namespace=None)


@pytest.fixture
def completed_jobset_info() -> JobSetInfo:
    """Create a JobSetInfo with Completed status."""
    return JobSetInfo(
        name="aiperf-abc123",
        namespace="default",
        jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
        status="Completed",
    )


@pytest.fixture
def mock_resolve_jobset(completed_jobset_info: JobSetInfo) -> AsyncMock:
    """Create a mock for resolve_jobset that returns a resolved result."""
    mock_client = AsyncMock()
    resolved = ResolvedJobSet("abc123", completed_jobset_info, mock_client)
    mock = AsyncMock(return_value=resolved)
    return mock


# ============================================================
# Happy Path Tests
# ============================================================


class TestShutdownBenchmarkHappyPath:
    """Verify normal successful shutdown operations."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_shutdown_api_service(
        self,
        manage_options: KubeManageOptions,
        completed_jobset_info: JobSetInfo,
        mock_resolve_jobset: AsyncMock,
    ) -> None:
        """Shutdown resolves jobset then calls shutdown_api_service."""
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_jobset",
                new=mock_resolve_jobset,
            ),
            patch(
                "aiperf.kubernetes.results.shutdown_api_service",
                new=AsyncMock(),
            ) as mock_shutdown,
        ):
            await shutdown_benchmark("abc123", manage_options=manage_options, port=8080)

            mock_resolve_jobset.assert_called_once_with(
                "abc123",
                None,
                None,
                None,
            )
            mock_shutdown.assert_called_once_with(
                "abc123",
                "default",
                mock_resolve_jobset.return_value.client,
                8080,
                kubeconfig=None,
                kube_context=None,
            )

    @pytest.mark.asyncio
    async def test_shutdown_forwards_manage_options(
        self,
        completed_jobset_info: JobSetInfo,
    ) -> None:
        """Custom kubeconfig/namespace/context are forwarded correctly."""
        options = KubeManageOptions(
            kubeconfig="/custom/kubeconfig",
            namespace="bench-ns",
            kube_context="prod-cluster",
        )
        mock_client = AsyncMock()
        resolved = ResolvedJobSet("job42", completed_jobset_info, mock_client)

        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_jobset",
                new=AsyncMock(return_value=resolved),
            ) as mock_resolve,
            patch(
                "aiperf.kubernetes.results.shutdown_api_service",
                new=AsyncMock(),
            ) as mock_shutdown,
        ):
            await shutdown_benchmark("job42", manage_options=options, port=9090)

            mock_resolve.assert_called_once_with(
                "job42",
                "bench-ns",
                "/custom/kubeconfig",
                "prod-cluster",
            )
            mock_shutdown.assert_called_once_with(
                "job42",
                "default",
                mock_client,
                9090,
                kubeconfig="/custom/kubeconfig",
                kube_context="prod-cluster",
            )

    @pytest.mark.asyncio
    async def test_shutdown_default_port_is_zero(
        self,
        manage_options: KubeManageOptions,
        mock_resolve_jobset: AsyncMock,
    ) -> None:
        """Default port is 0 (ephemeral) when not specified."""
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_jobset",
                new=mock_resolve_jobset,
            ),
            patch(
                "aiperf.kubernetes.results.shutdown_api_service",
                new=AsyncMock(),
            ) as mock_shutdown,
        ):
            await shutdown_benchmark("abc123", manage_options=manage_options)

            assert mock_shutdown.call_args.args[3] == 0


# ============================================================
# Early Return / Edge Cases
# ============================================================


class TestShutdownBenchmarkEdgeCases:
    """Verify boundary conditions and early-return paths."""

    @pytest.mark.asyncio
    async def test_shutdown_resolve_returns_none_skips_shutdown(
        self,
        manage_options: KubeManageOptions,
    ) -> None:
        """When resolve_jobset returns None, shutdown_api_service is not called."""
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_jobset",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.results.shutdown_api_service",
                new=AsyncMock(),
            ) as mock_shutdown,
        ):
            await shutdown_benchmark("missing-job", manage_options=manage_options)

            mock_shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_no_job_id_passes_none_to_resolve(
        self,
        manage_options: KubeManageOptions,
    ) -> None:
        """When job_id is None, None is passed through to resolve_jobset."""
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_jobset",
                new=AsyncMock(return_value=None),
            ) as mock_resolve,
            patch(
                "aiperf.kubernetes.results.shutdown_api_service",
                new=AsyncMock(),
            ),
        ):
            await shutdown_benchmark(manage_options=manage_options)

            mock_resolve.assert_called_once_with(None, None, None, None)

    @pytest.mark.asyncio
    async def test_shutdown_none_manage_options_creates_default(
        self,
        completed_jobset_info: JobSetInfo,
    ) -> None:
        """When manage_options is None, a default KubeManageOptions is created."""
        mock_client = AsyncMock()
        resolved = ResolvedJobSet("abc123", completed_jobset_info, mock_client)

        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_jobset",
                new=AsyncMock(return_value=resolved),
            ) as mock_resolve,
            patch(
                "aiperf.kubernetes.results.shutdown_api_service",
                new=AsyncMock(),
            ) as mock_shutdown,
        ):
            await shutdown_benchmark("abc123", manage_options=None)

            # Default KubeManageOptions has None for all fields
            mock_resolve.assert_called_once_with("abc123", None, None, None)
            mock_shutdown.assert_called_once()
            assert mock_shutdown.call_args.kwargs["kubeconfig"] is None
            assert mock_shutdown.call_args.kwargs["kube_context"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "port",
        [
            0,
            8080,
            param(65535, id="max-port"),
        ],
    )  # fmt: skip
    async def test_shutdown_port_values_forwarded(
        self,
        port: int,
        manage_options: KubeManageOptions,
        mock_resolve_jobset: AsyncMock,
    ) -> None:
        """Various port values are forwarded to shutdown_api_service."""
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_jobset",
                new=mock_resolve_jobset,
            ),
            patch(
                "aiperf.kubernetes.results.shutdown_api_service",
                new=AsyncMock(),
            ) as mock_shutdown,
        ):
            await shutdown_benchmark("abc123", manage_options=manage_options, port=port)

            assert mock_shutdown.call_args.args[3] == port


# ============================================================
# Error Handling
# ============================================================


class TestShutdownBenchmarkErrors:
    """Verify error handling via exit_on_error context manager."""

    @pytest.mark.asyncio
    async def test_shutdown_resolve_error_caught_by_exit_on_error(
        self,
        manage_options: KubeManageOptions,
    ) -> None:
        """Exceptions from resolve_jobset are caught by exit_on_error."""
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_jobset",
                new=AsyncMock(side_effect=RuntimeError("k8s connection refused")),
            ),
            pytest.raises(SystemExit),
        ):
            await shutdown_benchmark("abc123", manage_options=manage_options)

    @pytest.mark.asyncio
    async def test_shutdown_api_service_error_caught_by_exit_on_error(
        self,
        manage_options: KubeManageOptions,
        mock_resolve_jobset: AsyncMock,
    ) -> None:
        """Exceptions from shutdown_api_service are caught by exit_on_error."""
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_jobset",
                new=mock_resolve_jobset,
            ),
            patch(
                "aiperf.kubernetes.results.shutdown_api_service",
                new=AsyncMock(side_effect=ConnectionError("port-forward failed")),
            ),
            pytest.raises(SystemExit),
        ):
            await shutdown_benchmark("abc123", manage_options=manage_options)
