# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone worker scaling calculations.

Used by both SystemController (to pre-compute counts at startup) and
WorkerManager (to determine initial pool size). No service imports.
"""

import multiprocessing

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.environment import Environment


def calculate_worker_count(
    user_config: UserConfig,
    service_config: ServiceConfig,
) -> int:
    """Calculate the number of workers to spawn.

    Applies the same logic as WorkerManager.__init__:
    1. Default to 75% of CPU cores - 1, capped at MAX_WORKERS_CAP
    2. Cap by concurrency if set
    3. Floor at min workers (default 1)
    """
    cpu_count = multiprocessing.cpu_count()
    max_workers = service_config.workers.max
    concurrency = user_config.loadgen.concurrency

    if max_workers is None:
        max_workers = max(
            1,
            min(
                int(cpu_count * Environment.WORKER.CPU_UTILIZATION_FACTOR) - 1,
                Environment.WORKER.MAX_WORKERS_CAP,
            ),
        )

    if concurrency and concurrency < max_workers:
        max_workers = concurrency

    max_workers = max(max_workers, service_config.workers.min or 1)
    return max_workers


def calculate_record_processor_count(num_workers: int) -> int:
    """Calculate the number of record processors to scale with workers."""
    return max(1, num_workers // Environment.RECORD.PROCESSOR_SCALE_FACTOR)
