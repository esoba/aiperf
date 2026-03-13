# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ConfigDict, Field

from aiperf.common.models.base_models import AIPerfBaseModel


class WorkerTaskStats(AIPerfBaseModel):
    """Stats for the tasks that have been sent to the worker."""

    total: int = Field(
        default=0,
        description="The total number of tasks that have been sent to the worker (not all tasks will be completed)",
    )
    failed: int = Field(
        default=0,
        description="The number of tasks that returned an error",
    )
    completed: int = Field(
        default=0,
        description="The number of tasks that were completed successfully",
    )

    def task_finished(self, valid: bool) -> None:
        """Increment the task stats based on success or failure."""
        if not valid:
            self.failed += 1
        else:
            self.completed += 1

    @property
    def in_progress(self) -> int:
        """The number of tasks that are currently in progress.

        This is the total number of tasks sent to the worker minus the number of failed and successfully completed tasks.
        """
        return self.total - self.completed - self.failed


@dataclass(slots=True)
class ActiveRequestProgress:
    """Progress of a single in-flight request."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    credit_id: int
    status: str
    tokens_received: int
    tokens_expected: int | None


@dataclass(slots=True)
class ActiveSessionProgress:
    """Progress of an active user session."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    x_correlation_id: str
    conversation_id: str
    turn_index: int
    num_turns: int
    current_request: ActiveRequestProgress | None
