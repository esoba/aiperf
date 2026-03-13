# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for macOS-specific terminal corruption fixes in cli_runner.py"""

import multiprocessing
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.config.config import AIPerfConfig
from aiperf.plugin.enums import UIType


@pytest.fixture(autouse=True)
def _mock_tokenizer_validation():
    """Prevent real HF network calls during CLI runner tests."""
    with patch(
        "aiperf.common.tokenizer_validator.validate_tokenizer_early",
        return_value={"test-model": "test-model"},
    ):
        yield


class TestMacOSTerminalFixes:
    """Test the macOS-specific terminal corruption fixes in cli_runner.py"""

    @pytest.fixture
    def config_dashboard(self, aiperf_config: AIPerfConfig) -> AIPerfConfig:
        """Create an AIPerfConfig with Dashboard UI type."""
        aiperf_config.ui_type = UIType.DASHBOARD
        return aiperf_config

    @pytest.fixture
    def config_simple(self, aiperf_config: AIPerfConfig) -> AIPerfConfig:
        """Create an AIPerfConfig with Simple UI type."""
        aiperf_config.ui_type = UIType.SIMPLE
        return aiperf_config

    def test_spawn_method_set_on_macos_dashboard(
        self,
        config_dashboard: AIPerfConfig,
        mock_platform_darwin: Mock,
        mock_multiprocessing_set_start_method: Mock,
        mock_bootstrap_and_run_service: Mock,
    ):
        """Test that spawn method is set when on macOS with Dashboard UI."""
        from aiperf.cli_runner import run_system_controller

        run_system_controller(config_dashboard)

        # Verify spawn method was set
        mock_multiprocessing_set_start_method.assert_called_once_with(
            "spawn", force=True
        )

    def test_no_start_method_set_on_linux(
        self,
        config_dashboard: AIPerfConfig,
        mock_platform_linux: Mock,
        mock_multiprocessing_set_start_method: Mock,
        mock_bootstrap_and_run_service: Mock,
    ):
        """Test that no start method override on Linux (uses platform default fork)."""
        from aiperf.cli_runner import run_system_controller

        run_system_controller(config_dashboard)

        mock_multiprocessing_set_start_method.assert_not_called()

    def test_no_start_method_set_for_simple_ui_on_macos(
        self,
        config_simple: AIPerfConfig,
        mock_platform_darwin: Mock,
        mock_multiprocessing_set_start_method: Mock,
        mock_bootstrap_and_run_service: Mock,
    ):
        """Test that no start method override for non-dashboard UI on macOS."""
        from aiperf.cli_runner import run_system_controller

        run_system_controller(config_simple)

        mock_multiprocessing_set_start_method.assert_not_called()

    @patch("fcntl.fcntl")
    def test_fd_cloexec_not_set_on_linux(
        self,
        mock_fcntl: Mock,
        config_dashboard: AIPerfConfig,
        mock_platform_linux: Mock,
        mock_bootstrap_and_run_service: Mock,
        mock_get_global_log_queue: Mock,
    ):
        """Test that FD_CLOEXEC is NOT set on Linux."""
        from aiperf.cli_runner import run_system_controller

        mock_get_global_log_queue.return_value = MagicMock(spec=multiprocessing.Queue)

        run_system_controller(config_dashboard)

        # fcntl should not be called on Linux
        mock_fcntl.assert_not_called()

    def test_runtime_error_in_set_start_method_is_handled(
        self,
        config_dashboard: AIPerfConfig,
        mock_platform_darwin: Mock,
        mock_multiprocessing_set_start_method: Mock,
        mock_bootstrap_and_run_service: Mock,
    ):
        """Test that RuntimeError when setting start method is gracefully handled."""
        from aiperf.cli_runner import run_system_controller

        mock_multiprocessing_set_start_method.side_effect = RuntimeError(
            "context already set"
        )

        # Should not raise an exception
        run_system_controller(config_dashboard)

        # Verify it tried to set the method
        mock_multiprocessing_set_start_method.assert_called_once()

    def test_log_queue_created_before_ui_on_dashboard(
        self,
        config_dashboard: AIPerfConfig,
        mock_platform_darwin: Mock,
        mock_bootstrap_and_run_service: Mock,
        mock_get_global_log_queue: Mock,
    ):
        """Test that log_queue is created early when using Dashboard UI."""
        from aiperf.cli_runner import run_system_controller

        mock_queue = MagicMock(spec=multiprocessing.Queue)
        mock_get_global_log_queue.return_value = mock_queue

        run_system_controller(config_dashboard)

        # Verify log queue was created
        mock_get_global_log_queue.assert_called_once()

        # Verify it was passed to bootstrap_and_run_service
        mock_bootstrap_and_run_service.assert_called_once()
        call_kwargs = mock_bootstrap_and_run_service.call_args.kwargs
        assert "log_queue" in call_kwargs
        assert call_kwargs["log_queue"] == mock_queue
