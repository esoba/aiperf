# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration model for the sinusoidal trace generator."""

from typing import Annotated

from pydantic import Field

from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.dataset.synthesis.trace_generator import TraceGeneratorConfig


class SinTraceConfig(TraceGeneratorConfig):
    """Generate synthetic dataset with sinusoidal request rate and ISL/OSL ratio.

    Output is in mooncake-style JSONL format compatible with AIPerf's --input-file option.
    Request rate and ISL/OSL ratio follow sinusoidal patterns with configurable parameters.
    Both patterns use a phase shift of -pi/2 to start from minimum at t=0.

    Examples:
        # Varying request rate with fixed ISL/OSL
        aiperf gen-trace sin --time-duration 60 --request-rate-min 2 --request-rate-max 8 --isl1 3000 --osl1 150 --isl2 3000 --osl2 150

        # Varying ISL/OSL ratio with fixed request rate
        aiperf gen-trace sin --time-duration 60 --request-rate-min 5 --request-rate-max 5 --isl-osl-ratio-min 0.2 --isl-osl-ratio-max 0.8

        # Reproducible generation with seed
        aiperf gen-trace sin --seed 42 --time-duration 60
    """

    block_size: Annotated[
        int,
        Field(default=512, description="Block size for hashing (tokens per block)."),
        CLIParameter(name=("--block-size",)),
    ]
    total_blocks: Annotated[
        int,
        Field(
            default=10000,
            description="Pool size for random block sampling (larger = fewer duplicate prompts).",
        ),
        CLIParameter(name=("--total-blocks",)),
    ]
    time_duration: Annotated[
        int,
        Field(default=100, description="Total dataset duration in seconds."),
        CLIParameter(name=("--time-duration",)),
    ]
    process_interval: Annotated[
        int,
        Field(default=1, description="Sampling interval for generation in seconds."),
        CLIParameter(name=("--process-interval",)),
    ]
    request_rate_min: Annotated[
        float,
        Field(
            default=5.0,
            description="Minimum sinusoidal request rate (requests/second).",
        ),
        CLIParameter(name=("--request-rate-min",)),
    ]
    request_rate_max: Annotated[
        float,
        Field(
            default=10.0,
            description="Maximum sinusoidal request rate (requests/second).",
        ),
        CLIParameter(name=("--request-rate-max",)),
    ]
    request_rate_period: Annotated[
        float,
        Field(
            default=10.0,
            description="Period of the sinusoidal request rate in seconds.",
        ),
        CLIParameter(name=("--request-rate-period",)),
    ]
    isl1: Annotated[
        int,
        Field(
            default=100,
            description="Input sequence length for first ISL/OSL preset.",
        ),
        CLIParameter(name=("--isl1",)),
    ]
    osl1: Annotated[
        int,
        Field(
            default=2000,
            description="Output sequence length for first ISL/OSL preset.",
        ),
        CLIParameter(name=("--osl1",)),
    ]
    isl2: Annotated[
        int,
        Field(
            default=5000,
            description="Input sequence length for second ISL/OSL preset.",
        ),
        CLIParameter(name=("--isl2",)),
    ]
    osl2: Annotated[
        int,
        Field(
            default=100,
            description="Output sequence length for second ISL/OSL preset.",
        ),
        CLIParameter(name=("--osl2",)),
    ]
    isl_osl_ratio_min: Annotated[
        float,
        Field(
            default=0.2,
            description="Minimum ratio selecting first vs second ISL/OSL preset.",
        ),
        CLIParameter(name=("--isl-osl-ratio-min",)),
    ]
    isl_osl_ratio_max: Annotated[
        float,
        Field(
            default=0.8,
            description="Maximum ratio selecting first vs second ISL/OSL preset.",
        ),
        CLIParameter(name=("--isl-osl-ratio-max",)),
    ]
    isl_osl_ratio_period: Annotated[
        float,
        Field(
            default=10.0,
            description="Period of the sinusoidal ISL/OSL ratio in seconds.",
        ),
        CLIParameter(name=("--isl-osl-ratio-period",)),
    ]
