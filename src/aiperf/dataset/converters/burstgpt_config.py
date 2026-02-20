# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration model for the BurstGPT trace converter."""

from typing import Annotated, Literal

from pydantic import Field

from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.dataset.converters.trace_converter import TraceConverterConfig


class BurstGptConfig(TraceConverterConfig):
    """Convert BurstGPT CSV (real ChatGPT/GPT-4 logs) to mooncake JSONL.

    Reads a BurstGPT CSV with Timestamp, Request tokens, Response tokens columns.
    Supports filtering by model and log type, timestamp speed adjustment, and
    row skip/limit for sub-sampling.

    Examples:
        # Basic conversion
        aiperf convert-trace burstgpt --input-file burstgpt.csv

        # Filter to GPT-4 API logs, speed up 2x
        aiperf convert-trace burstgpt --input-file burstgpt.csv --model GPT-4 --log-type "API log" --speed-ratio 2.0

        # Sub-sample: skip first 1000, take next 500
        aiperf convert-trace burstgpt --input-file burstgpt.csv --skip-num-prompt 1000 --num-prompt 500
    """

    model: Annotated[
        Literal["ChatGPT", "GPT-4"] | None,
        Field(
            default=None,
            description="Filter by model (ChatGPT or GPT-4). If not specified, no filtering is applied.",
        ),
        CLIParameter(name=("--model",)),
    ]
    log_type: Annotated[
        Literal["Conversation log", "API log"] | None,
        Field(
            default=None,
            description="Filter by log type (Conversation log or API log). If not specified, no filtering is applied.",
        ),
        CLIParameter(name=("--log-type",)),
    ]
    num_prompt: Annotated[
        int | None,
        Field(
            default=None,
            description="Limit the number of rows to output after filtering.",
        ),
        CLIParameter(name=("--num-prompt",)),
    ]
    skip_num_prompt: Annotated[
        int,
        Field(
            default=0,
            description="Skip the first N rows after filtering (before applying --num-prompt).",
        ),
        CLIParameter(name=("--skip-num-prompt",)),
    ]
    speed_ratio: Annotated[
        float,
        Field(
            default=1.0,
            description="Timestamp speed ratio. Values > 1 speed up requests, < 1 slow down.",
        ),
        CLIParameter(name=("--speed-ratio",)),
    ]
    block_size: Annotated[
        int,
        Field(
            default=128,
            description="Block size for calculating hash array length: ceil(input_length / block_size).",
        ),
        CLIParameter(name=("--block-size",)),
    ]
    num_hash_blocks: Annotated[
        int,
        Field(
            default=10000,
            description="Maximum hash ID value for random hash generation.",
        ),
        CLIParameter(name=("--num-hash-blocks",)),
    ]
