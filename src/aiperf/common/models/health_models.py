# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import namedtuple

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


class NumericAggregate(AIPerfBaseModel):
    """Aggregates for a single numeric value over time."""

    min: float | None = Field(default=None, description="Minimum observed value")
    max: float | None = Field(default=None, description="Maximum observed value")
    sum: float = Field(default=0.0, description="Sum of all observed values")
    count: int = Field(default=0, description="Number of observations")

    @property
    def avg(self) -> float | None:
        """Average of all observed values."""
        return self.sum / self.count if self.count > 0 else None

    def update(self, value: float | int | None) -> None:
        """Update aggregates with a new observed value."""
        if value is None:
            return
        val = float(value)
        self.min = val if self.min is None else min(self.min, val)
        self.max = val if self.max is None else max(self.max, val)
        self.sum += val
        self.count += 1


class ProcessHealthAggregates(AIPerfBaseModel):
    """Aggregated statistics for process health metrics over time."""

    memory_usage: NumericAggregate = Field(
        default_factory=NumericAggregate,
        description="Memory usage (RSS) aggregates in bytes",
    )
    cpu_usage: NumericAggregate = Field(
        default_factory=NumericAggregate,
        description="CPU usage percentage aggregates",
    )
    num_threads: NumericAggregate = Field(
        default_factory=NumericAggregate,
        description="Number of threads aggregates",
    )
    voluntary_ctx_switches: NumericAggregate = Field(
        default_factory=NumericAggregate,
        description="Voluntary context switches aggregates",
    )
    involuntary_ctx_switches: NumericAggregate = Field(
        default_factory=NumericAggregate,
        description="Involuntary context switches aggregates",
    )
    io_read_bytes: NumericAggregate = Field(
        default_factory=NumericAggregate,
        description="Disk read bytes aggregates",
    )
    io_write_bytes: NumericAggregate = Field(
        default_factory=NumericAggregate,
        description="Disk write bytes aggregates",
    )
    cpu_time_user: NumericAggregate = Field(
        default_factory=NumericAggregate,
        description="User CPU time aggregates in seconds",
    )
    cpu_time_system: NumericAggregate = Field(
        default_factory=NumericAggregate,
        description="System CPU time aggregates in seconds",
    )
    cpu_time_iowait: NumericAggregate = Field(
        default_factory=NumericAggregate,
        description="IO wait time aggregates in seconds",
    )


# TODO: These can be potentially different for each platform. (below is linux)
IOCounters = namedtuple(
    "IOCounters",
    [
        "read_count",  # system calls io read
        "write_count",  # system calls io write
        "read_bytes",  # bytes read (disk io)
        "write_bytes",  # bytes written (disk io)
        "read_chars",  # io read bytes (system calls)
        "write_chars",  # io write bytes (system calls)
    ],
)

CPUTimes = namedtuple(
    "CPUTimes",
    ["user", "system", "iowait"],
)

CtxSwitches = namedtuple("CtxSwitches", ["voluntary", "involuntary"])


class ProcessHealth(AIPerfBaseModel):
    """Model for process health data."""

    pid: int | None = Field(
        default=None,
        description="The PID of the process",
    )
    create_time: float = Field(
        ..., description="The creation time of the process in seconds"
    )
    uptime: float = Field(..., description="The uptime of the process in seconds")
    cpu_usage: float = Field(
        ..., description="The current CPU usage of the process in %"
    )
    memory_usage: int = Field(
        ..., description="The current memory usage of the process in bytes (rss)"
    )
    pss_memory: int | None = Field(
        default=None,
        description="Proportional set size in bytes (excludes shared mmap pages). Only captured at start/end.",
    )
    io_counters: IOCounters | tuple | None = Field(
        default=None,
        description="The current I/O counters of the process (read_count, write_count, read_bytes, write_bytes, read_chars, write_chars)",
    )
    cpu_times: CPUTimes | tuple | None = Field(
        default=None,
        description="The current CPU times of the process (user, system, iowait)",
    )
    num_ctx_switches: CtxSwitches | tuple | None = Field(
        default=None,
        description="The current number of context switches (voluntary, involuntary)",
    )
    num_threads: int | None = Field(
        default=None,
        description="The current number of threads",
    )
