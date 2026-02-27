# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared NvmlHandleManager utility."""

from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.nvml_handle_manager import NvmlHandleManager


@pytest.fixture
def mock_pynvml():
    """Mock pynvml module with typical behavior."""
    mock = MagicMock()
    mock.NVMLError = Exception
    mock.nvmlDeviceGetCount.return_value = 2
    mock.nvmlDeviceGetHandleByIndex.side_effect = lambda i: f"handle_{i}"
    return mock


class TestNvmlHandleManagerInitialization:
    """Test NVML initialization and handle enumeration."""

    def test_default_state(self):
        mgr = NvmlHandleManager()
        assert mgr.initialized is False
        assert mgr.available is False
        assert mgr.device_count == 0
        assert mgr.handles == []
        assert mgr.handle_indices == []

    def test_initialize_success(self, mock_pynvml):
        mgr = NvmlHandleManager()
        with patch(
            "aiperf.common.nvml_handle_manager._import_pynvml", return_value=mock_pynvml
        ):
            mgr.initialize()

        assert mgr.initialized is True
        assert mgr.available is True
        assert mgr.device_count == 2
        assert mgr.handles == ["handle_0", "handle_1"]
        assert mgr.handle_indices == [0, 1]
        mock_pynvml.nvmlInit.assert_called_once()

    def test_initialize_idempotent(self, mock_pynvml):
        mgr = NvmlHandleManager()
        with patch(
            "aiperf.common.nvml_handle_manager._import_pynvml", return_value=mock_pynvml
        ):
            mgr.initialize()
            mgr.initialize()

        mock_pynvml.nvmlInit.assert_called_once()

    def test_initialize_zero_gpus(self, mock_pynvml):
        mock_pynvml.nvmlDeviceGetCount.return_value = 0
        mgr = NvmlHandleManager()
        with patch(
            "aiperf.common.nvml_handle_manager._import_pynvml", return_value=mock_pynvml
        ):
            mgr.initialize()

        assert mgr.initialized is True
        assert mgr.available is False
        assert mgr.device_count == 0
        assert mgr.handles == []

    def test_initialize_skips_failed_handles(self, mock_pynvml):
        mock_pynvml.nvmlDeviceGetCount.return_value = 3
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [
            "handle_0",
            Exception("GPU 1 failed"),
            "handle_2",
        ]
        mgr = NvmlHandleManager()
        with patch(
            "aiperf.common.nvml_handle_manager._import_pynvml", return_value=mock_pynvml
        ):
            mgr.initialize()

        assert mgr.device_count == 3
        assert len(mgr.handles) == 2
        assert mgr.handles == ["handle_0", "handle_2"]
        assert mgr.handle_indices == [0, 2]

    def test_initialize_nvml_init_failure_raises(self, mock_pynvml):
        mock_pynvml.nvmlInit.side_effect = Exception("Driver not loaded")
        mgr = NvmlHandleManager()
        with (
            patch(
                "aiperf.common.nvml_handle_manager._import_pynvml",
                return_value=mock_pynvml,
            ),
            pytest.raises(RuntimeError, match="Failed to initialize NVML"),
        ):
            mgr.initialize()

        assert mgr.initialized is False

    def test_initialize_device_count_failure_cleans_up(self, mock_pynvml):
        mock_pynvml.nvmlDeviceGetCount.side_effect = Exception("count failed")
        mgr = NvmlHandleManager()
        with (
            patch(
                "aiperf.common.nvml_handle_manager._import_pynvml",
                return_value=mock_pynvml,
            ),
            pytest.raises(RuntimeError, match="Failed to get GPU device count"),
        ):
            mgr.initialize()

        mock_pynvml.nvmlShutdown.assert_called_once()
        assert mgr.initialized is False


class TestNvmlHandleManagerShutdown:
    """Test NVML shutdown behavior."""

    def test_shutdown_clears_state(self, mock_pynvml):
        mgr = NvmlHandleManager()
        with patch(
            "aiperf.common.nvml_handle_manager._import_pynvml", return_value=mock_pynvml
        ):
            mgr.initialize()

        with patch("aiperf.common.nvml_handle_manager._pynvml", mock_pynvml):
            mgr.shutdown()

        assert mgr.initialized is False
        assert mgr.available is False
        assert mgr.device_count == 0
        assert mgr.handles == []
        assert mgr.handle_indices == []
        mock_pynvml.nvmlShutdown.assert_called()

    def test_shutdown_idempotent(self, mock_pynvml):
        mgr = NvmlHandleManager()
        with patch(
            "aiperf.common.nvml_handle_manager._import_pynvml", return_value=mock_pynvml
        ):
            mgr.initialize()

        with patch("aiperf.common.nvml_handle_manager._pynvml", mock_pynvml):
            mgr.shutdown()
            mgr.shutdown()

        mock_pynvml.nvmlShutdown.assert_called_once()

    def test_shutdown_noop_when_not_initialized(self):
        mgr = NvmlHandleManager()
        mgr.shutdown()
        assert mgr.initialized is False

    def test_shutdown_tolerates_nvml_error(self, mock_pynvml):
        mgr = NvmlHandleManager()
        with patch(
            "aiperf.common.nvml_handle_manager._import_pynvml", return_value=mock_pynvml
        ):
            mgr.initialize()

        mock_pynvml.nvmlShutdown.side_effect = Exception("shutdown error")
        with patch("aiperf.common.nvml_handle_manager._pynvml", mock_pynvml):
            mgr.shutdown()

        assert mgr.initialized is False
        assert mgr.handles == []


class TestNvmlHandleManagerProbe:
    """Test the lightweight probe method."""

    def test_probe_success(self, mock_pynvml):
        mgr = NvmlHandleManager()
        with patch(
            "aiperf.common.nvml_handle_manager._import_pynvml", return_value=mock_pynvml
        ):
            assert mgr.probe() is True

        mock_pynvml.nvmlShutdown.assert_called_once()
        assert mgr.initialized is False

    def test_probe_no_gpus(self, mock_pynvml):
        mock_pynvml.nvmlDeviceGetCount.return_value = 0
        mgr = NvmlHandleManager()
        with patch(
            "aiperf.common.nvml_handle_manager._import_pynvml", return_value=mock_pynvml
        ):
            assert mgr.probe() is False

    def test_probe_import_failure(self):
        mgr = NvmlHandleManager()
        with patch(
            "aiperf.common.nvml_handle_manager._import_pynvml",
            side_effect=ImportError("no pynvml"),
        ):
            assert mgr.probe() is False

    def test_probe_uses_existing_state_when_initialized(self, mock_pynvml):
        mgr = NvmlHandleManager()
        with patch(
            "aiperf.common.nvml_handle_manager._import_pynvml", return_value=mock_pynvml
        ):
            mgr.initialize()

        assert mgr.probe() is True
        # nvmlShutdown should NOT be called by probe (only init's nvmlInit)
        mock_pynvml.nvmlShutdown.assert_not_called()


class TestNvmlHandleManagerLock:
    """Test lock exposure for callers."""

    def test_lock_is_threading_lock(self):
        import threading

        mgr = NvmlHandleManager()
        assert type(mgr.lock) is type(threading.Lock())

    def test_lock_is_same_instance(self):
        mgr = NvmlHandleManager()
        assert mgr.lock is mgr.lock


class TestNvmlHandleManagerPynvmlProperty:
    """Test pynvml module access."""

    def test_pynvml_none_before_import(self):
        mgr = NvmlHandleManager()
        with patch("aiperf.common.nvml_handle_manager._pynvml", None):
            assert mgr.pynvml is None

    def test_pynvml_available_after_initialize(self, mock_pynvml):
        mgr = NvmlHandleManager()
        with patch(
            "aiperf.common.nvml_handle_manager._import_pynvml", return_value=mock_pynvml
        ):
            mgr.initialize()

        with patch("aiperf.common.nvml_handle_manager._pynvml", mock_pynvml):
            assert mgr.pynvml is mock_pynvml
