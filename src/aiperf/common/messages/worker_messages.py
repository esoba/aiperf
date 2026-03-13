# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import MessageType, WorkerStatus
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ProcessHealth, WorkerTaskStats
from aiperf.common.models.worker_models import ActiveSessionProgress
from aiperf.common.types import MessageTypeT


class WorkerHealthMessage(BaseServiceMessage):
    """Message for a worker health check."""

    message_type: MessageTypeT = MessageType.WORKER_HEALTH

    health: ProcessHealth = Field(..., description="The health of the worker process")

    # Worker specific fields
    task_stats: WorkerTaskStats = Field(
        ...,
        description="Stats for the tasks that have been sent to the worker",
    )

    @property
    def error_rate(self) -> float:
        """The error rate of the worker."""
        if self.task_stats.total == 0:
            return 0
        return self.task_stats.failed / self.task_stats.total


class RequestProgressMessage(BaseServiceMessage):
    """Per-worker snapshot of active sessions and in-flight request progress."""

    message_type: MessageTypeT = MessageType.REQUEST_PROGRESS

    sessions: list[ActiveSessionProgress] = Field(
        default_factory=list,
        description="Active sessions on this worker with current progress",
    )


class WorkerStatusSummaryMessage(BaseServiceMessage):
    """Message for a worker status summary."""

    message_type: MessageTypeT = MessageType.WORKER_STATUS_SUMMARY

    worker_statuses: dict[str, WorkerStatus] = Field(
        ...,
        description="A mapping of worker IDs to their status",
    )
