# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared subprocess management utilities.

This module provides reusable subprocess spawning and lifecycle management
used by both MultiProcessServiceManager (for control-plane services) and
WorkerPodManager (for worker pod subprocesses).
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import platform
import uuid
from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.context import ForkProcess, ForkServerProcess, SpawnProcess
from typing import TYPE_CHECKING

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.environment import Environment
from aiperf.common.error_queue import ErrorQueue
from aiperf.common.logging import LogQueue
from aiperf.common.types import ServiceTypeT

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


@dataclass(slots=True)
class SubprocessInfo:
    """Information about a subprocess managed by SubprocessManager."""

    service_type: ServiceTypeT
    """Type of service running in the process"""

    service_id: str
    """ID of the service running in the process"""

    process: Process | SpawnProcess | ForkProcess | ForkServerProcess | None = None
    """The underlying multiprocessing process instance"""

    @property
    def exitcode(self) -> int | None:
        """Exit code of the process, or None if still running or no process."""
        return self.process.exitcode if self.process else None

    @property
    def pid(self) -> int | None:
        """PID of the process, or None if no process."""
        return self.process.pid if self.process else None


_SPAWN_TIMEOUT = 60.0
"""Safety-net timeout for process.start(). Normal spawns complete in
milliseconds; this guards against extreme system conditions (memory
pressure, exhausted forkserver) blocking the event loop indefinitely."""

_FORKSERVER_PRELOAD = [
    # -- aiperf core (shared by all services) --
    "aiperf.common.bootstrap",
    "aiperf.config",
    "aiperf.common.environment",
    "aiperf.common.logging",
    "aiperf.common.enums",
    "aiperf.common.hooks",
    "aiperf.common.messages",
    "aiperf.common.models",
    "aiperf.common.control_structs",
    "aiperf.common.types",
    "aiperf.plugin",
    "aiperf.plugin.enums",
    "aiperf.common.base_service",
    "aiperf.common.base_component_service",
    "aiperf.common.mixins",
    # -- Worker (replicable: num_workers instances) --
    "aiperf.workers.worker",
    "aiperf.workers.inference_client",
    "aiperf.workers.session_manager",
    "aiperf.credit",
    "aiperf.credit.issuer",
    "aiperf.transports",
    "aiperf.transports.aiohttp_client",
    # -- RecordProcessor (replicable: num_record_processors instances) --
    "aiperf.records.record_processor_service",
    "aiperf.metrics",
    "aiperf.post_processors",
    # -- heavy third-party deps --
    "pydantic",
    "numpy",
    "zmq",
    "uvloop",
    "orjson",
    "msgspec",
    "rich.console",
    "rich.logging",
    "aiohttp",
    "aiofiles",
    "psutil",
]

_mp_context: multiprocessing.context.BaseContext | None = None


def get_mp_context() -> multiprocessing.context.BaseContext:
    """Return the forkserver (Linux) or spawn (macOS) multiprocessing context.

    Lazily created on first call to avoid side-effects at import time
    (e.g. during pytest-xdist worker collection).
    """
    global _mp_context
    if _mp_context is None:
        method = "forkserver" if platform.system() == "Linux" else "spawn"
        _mp_context = multiprocessing.get_context(method)
        if platform.system() == "Linux":
            _mp_context.set_forkserver_preload(_FORKSERVER_PRELOAD)
    return _mp_context


class SubprocessManager:
    """Manages spawning and lifecycle of service subprocesses.

    This utility class provides common subprocess management functionality
    used by both service managers and service components that need to spawn
    child processes.

    Example usage:
        manager = SubprocessManager(run, log_queue)
        await manager.spawn_service(ServiceType.WORKER, "worker_0")
        await manager.stop_all()
    """

    def __init__(
        self,
        run: BenchmarkRun,
        log_queue: LogQueue | None = None,
        error_queue: ErrorQueue | None = None,
        logger: object | None = None,
    ) -> None:
        """Initialize the subprocess manager.

        Args:
            run: BenchmarkRun for spawned services.
            log_queue: Optional multiprocessing queue for centralized logging.
            error_queue: Optional multiprocessing queue for error reporting from child processes.
            logger: Optional logger object with debug/warning/error methods.
        """
        self.run = run
        self.log_queue = log_queue
        self.error_queue = error_queue
        self.subprocesses: list[SubprocessInfo] = []
        self._logger = logger

    def _debug(self, msg: str) -> None:
        """Log a debug message if logger is available."""
        if self._logger and hasattr(self._logger, "debug"):
            self._logger.debug(msg)

    def _warning(self, msg: str) -> None:
        """Log a warning message if logger is available."""
        if self._logger and hasattr(self._logger, "warning"):
            self._logger.warning(msg)

    async def spawn_service(
        self,
        service_type: ServiceTypeT,
        service_id: str | None = None,
        replicable: bool = True,
    ) -> SubprocessInfo:
        """Spawn a single service as a subprocess.

        Args:
            service_type: The type of service to spawn.
            service_id: Optional specific service ID. If None, generates one.
            replicable: Whether the service can have multiple replicas.

        Returns:
            SubprocessInfo with the spawned process details.
        """
        if service_id is None:
            service_id = (
                f"{service_type}_{uuid.uuid4().hex[:8]}"
                if replicable
                else str(service_type)
            )

        process = get_mp_context().Process(
            target=bootstrap_and_run_service,
            name=f"{service_type}_process",
            kwargs={
                "service_type": service_type,
                "service_id": service_id,
                "run": self.run,
                "log_queue": self.log_queue,
                "error_queue": self.error_queue,
            },
            daemon=True,
        )

        try:
            await asyncio.wait_for(
                asyncio.to_thread(process.start),
                timeout=_SPAWN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                process.kill()
            raise RuntimeError(
                f"Timed out spawning {service_type} subprocess "
                f"(id: {service_id}) after {_SPAWN_TIMEOUT}s"
            ) from None

        self._debug(
            f"Spawned {service_type} subprocess (pid: {process.pid}, id: {service_id})"
        )

        info = SubprocessInfo(
            process=process,
            service_type=service_type,
            service_id=service_id,
        )
        self.subprocesses.append(info)

        return info

    async def spawn_services(
        self,
        service_type: ServiceTypeT,
        num_replicas: int,
        replicable: bool = True,
    ) -> list[SubprocessInfo]:
        """Spawn multiple replicas of a service type.

        Args:
            service_type: The type of service to spawn.
            num_replicas: Number of replicas to spawn.
            replicable: Whether the service can have multiple replicas.

        Returns:
            List of SubprocessInfo for all spawned processes.
        """
        infos = []
        for _ in range(num_replicas):
            info = await self.spawn_service(service_type, replicable=replicable)
            infos.append(info)
        return infos

    async def stop_process(
        self,
        info: SubprocessInfo,
        timeout: float = Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT,
    ) -> None:
        """Stop a single subprocess gracefully, killing if necessary.

        Args:
            info: The subprocess info to stop.
            timeout: Timeout in seconds for graceful termination.
        """
        if not info.process or not info.process.is_alive():
            return

        info.process.terminate()
        await asyncio.to_thread(info.process.join, timeout=timeout)
        if info.process.is_alive():
            self._warning(
                f"Subprocess {info.service_id} did not terminate gracefully, killing"
            )
            info.process.kill()
            await asyncio.to_thread(info.process.join, timeout=timeout)
        else:
            self._debug(
                f"Subprocess {info.service_type} ({info.service_id}) stopped "
                f"(pid: {info.process.pid})"
            )

    async def stop_service(
        self,
        service_type: ServiceTypeT,
        service_id: str | None = None,
    ) -> list[BaseException | None]:
        """Stop all subprocesses of a given service type.

        Args:
            service_type: The type of service to stop.
            service_id: Optional specific service ID to stop.

        Returns:
            List of exceptions that occurred during stop, or None for success.
        """
        self._debug(f"Stopping {service_type} subprocess(es) with id: {service_id}")
        to_stop = [
            info
            for info in self.subprocesses
            if info.service_type == service_type
            and (service_id is None or info.service_id == service_id)
        ]
        for info in to_stop:
            self.subprocesses.remove(info)
        return await asyncio.gather(
            *[self.stop_process(info) for info in to_stop],
            return_exceptions=True,
        )

    async def stop_all(self) -> list[BaseException | None]:
        """Stop all managed subprocesses gracefully.

        Returns:
            List of exceptions that occurred during stop, or None for success.
        """
        self._debug("Stopping all subprocesses")
        to_stop = list(self.subprocesses)
        self.subprocesses.clear()
        return await asyncio.gather(
            *[self.stop_process(info) for info in to_stop],
            return_exceptions=True,
        )

    async def kill_all(self) -> list[BaseException | None]:
        """Kill all managed subprocesses immediately.

        Returns:
            List of exceptions that occurred during kill, or None for success.
        """
        self._debug("Killing all subprocesses")
        to_kill = list(self.subprocesses)
        self.subprocesses.clear()

        for info in to_kill:
            if info.process and info.process.is_alive():
                info.process.kill()

        async def _join(info: SubprocessInfo) -> None:
            if info.process:
                await asyncio.to_thread(
                    info.process.join,
                    timeout=Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT,
                )

        return await asyncio.gather(
            *[_join(info) for info in to_kill],
            return_exceptions=True,
        )

    def get_by_type(self, service_type: ServiceTypeT) -> list[SubprocessInfo]:
        """Get all subprocesses of a given service type.

        Args:
            service_type: The service type to filter by.

        Returns:
            List of SubprocessInfo matching the service type.
        """
        return [s for s in self.subprocesses if s.service_type == service_type]

    def check_alive(self) -> list[SubprocessInfo]:
        """Check which subprocesses have died.

        Returns:
            List of SubprocessInfo for dead subprocesses.
        """
        dead: list[SubprocessInfo] = []
        for info in self.subprocesses:
            if info.process and not info.process.is_alive():
                dead.append(info)
        return dead

    def remove(self, info: SubprocessInfo) -> None:
        """Remove a subprocess from tracking.

        Args:
            info: The subprocess info to remove.
        """
        if info in self.subprocesses:
            self.subprocesses.remove(info)

    def clear(self) -> None:
        """Clear all subprocess tracking."""
        self.subprocesses.clear()
