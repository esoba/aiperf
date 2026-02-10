# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.groups import Groups


class SteadyStateConfig(BaseConfig):
    """Configuration for steady-state detection and windowed metric computation.

    When enabled, AIPerf detects the steady-state region of a benchmark run by
    analyzing concurrency curves, then re-computes metrics only over that window.
    This excludes ramp-up and ramp-down artifacts from the results.
    """

    _CLI_GROUP = Groups.OUTPUT

    enabled: Annotated[
        bool,
        Field(
            description="Enable steady-state metric computation. When enabled, AIPerf detects the steady-state "
            "region of a benchmark run and reports windowed metrics that exclude ramp-up and ramp-down periods.",
        ),
        CLIParameter(
            name="--steady-state",
            group=_CLI_GROUP,
        ),
    ] = False

    start_pct: Annotated[
        float | None,
        Field(
            ge=0.0,
            lt=100.0,
            description="Manual override: start of steady-state window as a percentage of total benchmark duration. "
            "Must be used together with --steady-state-end-pct. Overrides automatic detection.",
        ),
        CLIParameter(
            name="--steady-state-start-pct",
            group=_CLI_GROUP,
        ),
    ] = None

    end_pct: Annotated[
        float | None,
        Field(
            gt=0.0,
            le=100.0,
            description="Manual override: end of steady-state window as a percentage of total benchmark duration. "
            "Must be used together with --steady-state-start-pct. Overrides automatic detection.",
        ),
        CLIParameter(
            name="--steady-state-end-pct",
            group=_CLI_GROUP,
        ),
    ] = None

    stability_fraction: Annotated[
        float,
        Field(
            gt=0.0,
            le=1.0,
            description="Fraction of peak concurrency used as the stability threshold. "
            "A value of 0.90 means concurrency must reach 90% of the 95th-percentile peak "
            "before the region is considered steady state.",
        ),
        CLIParameter(
            name="--steady-state-stability-fraction",
            group=_CLI_GROUP,
        ),
    ] = 0.90

    sustained_window_pct: Annotated[
        float,
        Field(
            gt=0.0,
            le=50.0,
            description="Minimum sustained duration as a percentage of total benchmark duration "
            "for ramp boundary detection. Higher values require concurrency to remain above "
            "the threshold for longer before it is considered steady.",
        ),
        CLIParameter(
            name="--steady-state-sustained-window-pct",
            group=_CLI_GROUP,
        ),
    ] = 5.0

    min_window_pct: Annotated[
        float,
        Field(
            gt=0.0,
            le=100.0,
            description="Minimum steady-state window size as a percentage of total benchmark duration. "
            "If the detected window is smaller than this, AIPerf falls back to the full duration.",
        ),
        CLIParameter(
            name="--steady-state-min-window-pct",
            group=_CLI_GROUP,
        ),
    ] = 10.0
