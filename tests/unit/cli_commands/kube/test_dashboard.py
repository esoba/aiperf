# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf kube dashboard command."""

from unittest.mock import AsyncMock, patch

import pytest

from aiperf.cli_commands.kube.dashboard import dashboard
from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes.enums import PodPhase


@pytest.fixture
def manage_options():
    return KubeManageOptions(kubeconfig=None, namespace=None)


@pytest.fixture
def mock_kube_client():
    from aiperf.kubernetes.client import AIPerfKubeClient

    mock_client = AsyncMock(spec=AIPerfKubeClient)
    with patch(
        "aiperf.kubernetes.client.AIPerfKubeClient.create",
        new=AsyncMock(return_value=mock_client),
    ):
        yield mock_client


class TestDashboardCommand:
    """Tests for the kube dashboard command."""

    async def test_dashboard_opens_browser(
        self, mock_kube_client, manage_options
    ) -> None:
        """Test dashboard port-forwards and opens browser."""
        mock_kube_client.find_operator_pod.return_value = (
            "aiperf-operator-abc",
            PodPhase.RUNNING,
        )

        mock_port_forward = AsyncMock()
        mock_port_forward.__aenter__ = AsyncMock(return_value=54321)
        mock_port_forward.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.kubernetes.port_forward.port_forward_with_status",
                return_value=mock_port_forward,
            ) as mock_pf,
            patch("webbrowser.open") as mock_browser,
            patch("asyncio.sleep", side_effect=KeyboardInterrupt),
        ):
            with pytest.raises(KeyboardInterrupt):
                await dashboard(manage_options=manage_options)

            mock_pf.assert_called_once()
            call_kwargs = mock_pf.call_args
            assert call_kwargs[0][0] == "aiperf-system"
            assert call_kwargs[0][1] == "aiperf-operator-abc"
            assert call_kwargs.kwargs["remote_port"] == 8081

            mock_browser.assert_called_once_with("http://localhost:54321")

    async def test_dashboard_no_browser_flag(
        self, mock_kube_client, manage_options
    ) -> None:
        """Test --no-browser prints URL instead of opening browser."""
        mock_kube_client.find_operator_pod.return_value = (
            "aiperf-operator-abc",
            PodPhase.RUNNING,
        )

        mock_port_forward = AsyncMock()
        mock_port_forward.__aenter__ = AsyncMock(return_value=54321)
        mock_port_forward.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.kubernetes.port_forward.port_forward_with_status",
                return_value=mock_port_forward,
            ),
            patch("webbrowser.open") as mock_browser,
            patch("asyncio.sleep", side_effect=KeyboardInterrupt),
        ):
            with pytest.raises(KeyboardInterrupt):
                await dashboard(manage_options=manage_options, no_browser=True)

            mock_browser.assert_not_called()

    async def test_dashboard_operator_not_found(
        self, mock_kube_client, manage_options
    ) -> None:
        """Test dashboard exits gracefully when operator pod not found."""
        mock_kube_client.find_operator_pod.return_value = None

        with patch("webbrowser.open") as mock_browser:
            await dashboard(manage_options=manage_options)

        mock_browser.assert_not_called()

    async def test_dashboard_custom_port(
        self, mock_kube_client, manage_options
    ) -> None:
        """Test dashboard with custom local port."""
        mock_kube_client.find_operator_pod.return_value = (
            "aiperf-operator-abc",
            PodPhase.RUNNING,
        )

        mock_port_forward = AsyncMock()
        mock_port_forward.__aenter__ = AsyncMock(return_value=8081)
        mock_port_forward.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.kubernetes.port_forward.port_forward_with_status",
                return_value=mock_port_forward,
            ) as mock_pf,
            patch("webbrowser.open"),
            patch("asyncio.sleep", side_effect=KeyboardInterrupt),
        ):
            with pytest.raises(KeyboardInterrupt):
                await dashboard(manage_options=manage_options, port=8081)

            assert mock_pf.call_args[0][2] == 8081

    async def test_dashboard_custom_operator_namespace(
        self, mock_kube_client, manage_options
    ) -> None:
        """Test dashboard with custom operator namespace."""
        mock_kube_client.find_operator_pod.return_value = (
            "aiperf-operator-abc",
            PodPhase.RUNNING,
        )

        mock_port_forward = AsyncMock()
        mock_port_forward.__aenter__ = AsyncMock(return_value=54321)
        mock_port_forward.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.kubernetes.port_forward.port_forward_with_status",
                return_value=mock_port_forward,
            ) as mock_pf,
            patch("webbrowser.open"),
            patch("asyncio.sleep", side_effect=KeyboardInterrupt),
        ):
            with pytest.raises(KeyboardInterrupt):
                await dashboard(
                    manage_options=manage_options,
                    operator_namespace="custom-ns",
                )

            assert mock_pf.call_args[0][0] == "custom-ns"
            mock_kube_client.find_operator_pod.assert_called_once_with(
                namespace="custom-ns",
            )
