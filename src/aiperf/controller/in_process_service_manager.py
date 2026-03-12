# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.types import ServiceTypeT
from aiperf.controller.multiprocess_service_manager import MultiProcessServiceManager


class InProcessServiceManager(MultiProcessServiceManager):
    """Service manager for in-engine mode.

    Extends MultiProcessServiceManager but skips Worker spawning.
    In in-engine mode, the Worker is created and owned by
    InProcessCreditRouter inside TimingManager's process.
    All other services are spawned normally as separate processes.
    """

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        """Run a service, skipping Worker which is managed in-process."""
        from aiperf.plugin.enums import ServiceType

        if service_type == ServiceType.WORKER:
            self.info("Skipping Worker spawn — owned by InProcessCreditRouter")
            return
        await super().run_service(service_type, num_replicas)
