# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PreflightChecker.run_quick_checks()."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.kubernetes.preflight import CheckStatus, PreflightChecker

# =============================================================================
# Fixtures
# =============================================================================


def _make_call_api_cm(response_json: dict | None = None):
    """Create a mock async context manager for api.call_api()."""
    resp = MagicMock()
    resp.json.return_value = response_json or {}
    resp.raise_for_status = MagicMock()

    @asynccontextmanager
    async def _call_api(*args, **kwargs):
        yield resp

    return _call_api


@pytest.fixture
def mock_kr8s_api():
    """Fixture that provides a factory for mocked kr8s API.

    Returns a context-manager factory that patches get_api and configures
    the mock API with version info, call_api for RBAC, and CRD checks.
    """

    @asynccontextmanager
    async def _mocks(
        *,
        jobset_crd_error=None,
        rbac_allowed=True,
        connectivity_error=None,
    ):
        mock_api = AsyncMock()

        # Version endpoint
        if connectivity_error:
            mock_api.async_version.side_effect = connectivity_error
        else:
            mock_api.async_version.return_value = {
                "major": "1",
                "minor": "28",
                "gitVersion": "v1.28.0",
            }

        # call_api for RBAC (SelfSubjectAccessReview)
        rbac_response = {
            "status": {"allowed": rbac_allowed},
        }

        if jobset_crd_error:
            # Track calls to distinguish RBAC vs CRD call_api usage
            original_call_api = _make_call_api_cm(rbac_response)

            @asynccontextmanager
            async def _switching_call_api(*args, **kwargs):
                if args and args[0] == "GET":
                    raise jobset_crd_error
                async with original_call_api(*args, **kwargs) as resp:
                    yield resp

            mock_api.call_api = _switching_call_api
        else:
            mock_api.call_api = _make_call_api_cm(rbac_response)

        with patch(
            "aiperf.kubernetes.client.get_api",
            return_value=mock_api,
        ):
            yield mock_api

    return _mocks


# =============================================================================
# Quick Checks Tests
# =============================================================================


class TestQuickChecks:
    """Tests for PreflightChecker.run_quick_checks()."""

    @pytest.mark.asyncio
    async def test_quick_checks_passes_healthy_cluster(self, mock_kr8s_api) -> None:
        """Test quick checks pass on a healthy cluster."""
        async with mock_kr8s_api():
            checker = PreflightChecker(namespace="default")
            results = await checker.run_quick_checks()

        assert results.passed is True
        assert len(results.checks) == 3

    @pytest.mark.asyncio
    async def test_quick_checks_fails_on_connectivity(self, mock_kr8s_api) -> None:
        """Test quick checks short-circuit on connectivity failure."""
        async with mock_kr8s_api(
            connectivity_error=Exception("connection refused"),
        ):
            checker = PreflightChecker(namespace="default")
            results = await checker.run_quick_checks()

        assert results.passed is False
        assert len(results.checks) == 1
        assert results.checks[0].name == "Cluster Connectivity"
        assert results.checks[0].status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_quick_checks_fails_on_jobset_crd(self, mock_kr8s_api) -> None:
        """Test quick checks fail when JobSet CRD is missing."""
        import kr8s

        async with mock_kr8s_api(jobset_crd_error=kr8s.NotFoundError()):
            checker = PreflightChecker(namespace="default")
            results = await checker.run_quick_checks()

        assert results.passed is False
        crd_check = next(c for c in results.checks if c.name == "JobSet CRD")
        assert crd_check.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_quick_checks_only_runs_three_checks(self, mock_kr8s_api) -> None:
        """Test that quick checks run exactly 3 checks on success."""
        async with mock_kr8s_api():
            checker = PreflightChecker(namespace="default")
            results = await checker.run_quick_checks()

        assert len(results.checks) == 3
        check_names = [c.name for c in results.checks]
        assert check_names == [
            "Cluster Connectivity",
            "JobSet CRD",
            "RBAC Permissions",
        ]

    @pytest.mark.asyncio
    async def test_quick_checks_fails_on_rbac(self, mock_kr8s_api) -> None:
        """Test quick checks fail when RBAC permissions are denied."""
        async with mock_kr8s_api(rbac_allowed=False):
            checker = PreflightChecker(namespace="default")
            results = await checker.run_quick_checks()

        assert results.passed is False
        rbac_check = next(c for c in results.checks if c.name == "RBAC Permissions")
        assert rbac_check.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_quick_checks_with_endpoint_runs_four_checks(
        self, mock_kr8s_api
    ) -> None:
        """Test that quick checks include endpoint when endpoint_url is set."""
        async with mock_kr8s_api():
            checker = PreflightChecker(
                namespace="default", endpoint_url="http://llm:8000/v1"
            )
            results = await checker.run_quick_checks()

        assert len(results.checks) == 4
        check_names = [c.name for c in results.checks]
        assert check_names == [
            "Cluster Connectivity",
            "JobSet CRD",
            "RBAC Permissions",
            "Endpoint Connectivity",
        ]

    @pytest.mark.asyncio
    async def test_check_results_have_duration(self, mock_kr8s_api) -> None:
        """Test that all check results have duration_ms populated."""
        async with mock_kr8s_api():
            checker = PreflightChecker(namespace="default")
            results = await checker.run_quick_checks()

        for check in results.checks:
            assert check.duration_ms is not None
            assert check.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_quick_checks_does_not_print(self, mock_kr8s_api, capsys) -> None:
        """Test that quick checks do not print anything to stdout by default."""
        async with mock_kr8s_api():
            checker = PreflightChecker(namespace="default")
            await checker.run_quick_checks()

        captured = capsys.readouterr()
        assert captured.out == ""

    @pytest.mark.asyncio
    async def test_quick_checks_show_progress_prints_output(
        self, mock_kr8s_api, capsys
    ) -> None:
        """Test that quick checks print compact results when show_progress=True."""
        async with mock_kr8s_api():
            checker = PreflightChecker(namespace="default")
            await checker.run_quick_checks(show_progress=True)

        captured = capsys.readouterr()
        assert "Cluster Connectivity" in captured.out
        assert "JobSet CRD" in captured.out
        assert "RBAC Permissions" in captured.out
