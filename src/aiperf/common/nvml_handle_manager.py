# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe pynvml lifecycle and handle manager.

Provides a shared resource manager for NVML initialization, shutdown, and GPU
device handle enumeration. Used by both PyNVMLTelemetryCollector and
SystemMetricsCollector to avoid duplicating pynvml boilerplate.

All methods are synchronous — callers should wrap with asyncio.to_thread()
when calling from async contexts.
"""

from __future__ import annotations

import logging
import threading

__all__ = ["NvmlHandleManager"]

_logger = logging.getLogger(__name__)

# Lazy import so pynvml remains optional
_pynvml = None


def _import_pynvml():
    global _pynvml
    if _pynvml is None:
        import pynvml

        _pynvml = pynvml
    return _pynvml


class NvmlHandleManager:
    """Thread-safe pynvml lifecycle and GPU handle manager.

    Manages the NVML library lifecycle (init/shutdown) and device handle
    enumeration. Designed as a lightweight, composable building block — no
    lifecycle hooks, no callbacks, no metrics collection.

    Both `PyNVMLTelemetryCollector` and `SystemMetricsCollector` delegate
    pynvml boilerplate to this class while keeping their collector-specific
    logic (GPM, metadata, per-process memory, etc.) in their own code.

    Thread safety:
        All public methods are protected by an internal lock. The lock is also
        exposed via the `lock` property for callers that need to hold it across
        multiple operations (e.g., iterating handles during collection).
    """

    __slots__ = (
        "_initialized",
        "_handles",
        "_handle_indices",
        "_device_count",
        "_lock",
    )

    def __init__(self) -> None:
        self._initialized: bool = False
        self._handles: list[object] = []
        self._handle_indices: list[int] = []
        self._device_count: int = 0
        self._lock = threading.Lock()

    @property
    def lock(self) -> threading.Lock:
        """The internal lock — use when callers need to hold it across operations."""
        return self._lock

    @property
    def initialized(self) -> bool:
        """Whether NVML is currently initialized."""
        return self._initialized

    @property
    def available(self) -> bool:
        """Whether NVML is initialized and at least one GPU is present."""
        return self._initialized and self._device_count > 0

    @property
    def device_count(self) -> int:
        """Number of GPU devices discovered."""
        return self._device_count

    @property
    def handles(self) -> list[object]:
        """List of NVML device handles (empty if not initialized)."""
        return self._handles

    @property
    def handle_indices(self) -> list[int]:
        """Original NVML device indices for each handle.

        Preserves the NVML device index when handles fail to enumerate.
        E.g., if GPU 0 fails and GPU 1 succeeds: handles=[h1], handle_indices=[1].
        """
        return self._handle_indices

    @property
    def pynvml(self):
        """The imported pynvml module (or None if not yet imported)."""
        return _pynvml

    def initialize(self) -> None:
        """Initialize NVML and enumerate device handles.

        Thread-safe. No-op if already initialized.

        Raises:
            ImportError: If pynvml package is not installed.
            RuntimeError: If NVML initialization or device enumeration fails.
        """
        with self._lock:
            if self._initialized:
                return

            nvml = _import_pynvml()

            try:
                nvml.nvmlInit()
            except nvml.NVMLError as e:
                raise RuntimeError(f"Failed to initialize NVML: {e}") from e

            try:
                count = nvml.nvmlDeviceGetCount()
            except nvml.NVMLError as e:
                nvml.nvmlShutdown()
                raise RuntimeError(f"Failed to get GPU device count: {e}") from e

            handles = []
            indices = []
            for i in range(count):
                try:
                    handles.append(nvml.nvmlDeviceGetHandleByIndex(i))
                    indices.append(i)
                except nvml.NVMLError:
                    _logger.warning("Failed to get handle for GPU %d, skipping", i)

            self._handles = handles
            self._handle_indices = indices
            self._device_count = count
            self._initialized = True

    def shutdown(self) -> None:
        """Shutdown NVML and clear all state.

        Thread-safe and idempotent — safe to call multiple times or when
        not initialized.
        """
        with self._lock:
            if not self._initialized:
                return
            try:
                _pynvml.nvmlShutdown()
            except Exception as e:
                _logger.warning("Error during NVML shutdown: %r", e)
            finally:
                self._initialized = False
                self._handles = []
                self._handle_indices = []
                self._device_count = 0

    def probe(self) -> bool:
        """Quick probe to check if NVML is available and GPUs are present.

        Performs a temporary init/count/shutdown cycle. Does not affect the
        manager's state (if already initialized, checks existing state instead).

        Returns:
            True if NVML is usable and at least one GPU is available.
        """
        if self._initialized:
            return self._device_count > 0

        try:
            nvml = _import_pynvml()
            nvml.nvmlInit()
            try:
                count = nvml.nvmlDeviceGetCount()
                return count > 0
            finally:
                nvml.nvmlShutdown()
        except Exception:
            return False
