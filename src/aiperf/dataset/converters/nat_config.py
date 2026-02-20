# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration model for the NAT profiler trace converter."""

from typing import Annotated

from pydantic import Field

from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.dataset.converters.trace_converter import TraceConverterConfig


class NatConfig(TraceConverterConfig):
    """Convert NAT profiler JSON traces to mooncake JSONL.

    Reads all_requests_profiler_traces.json from NAT profiler, extracts LLM
    calls by matching LLM_START/LLM_END events, tokenizes prompt text, and
    produces multi-turn session records with hash IDs.

    Examples:
        # Basic conversion (tokenizer inferred from trace model name)
        aiperf convert-trace nat --input-file all_requests_profiler_traces.json

        # Explicit tokenizer
        aiperf convert-trace nat --input-file traces.json --tokenizer meta-llama/Llama-3.3-70B-Instruct

        # First 50 conversations, 200ms delay between turns
        aiperf convert-trace nat --input-file traces.json --num-requests 50 --delay 200
    """

    tokenizer: Annotated[
        str | None,
        Field(
            default=None,
            description="Tokenizer name/path for hashing. If not provided, inferred from trace model name.",
        ),
        CLIParameter(name=("--tokenizer",)),
    ]
    block_size: Annotated[
        int,
        Field(
            default=64,
            description="Block size for hash generation (tokens per block).",
        ),
        CLIParameter(name=("--block-size",)),
    ]
    num_requests: Annotated[
        int | None,
        Field(
            default=None,
            description="Limit the number of requests (conversations) to process.",
        ),
        CLIParameter(name=("--num-requests",)),
    ]
    skip_requests: Annotated[
        int,
        Field(
            default=0,
            description="Skip the first N requests.",
        ),
        CLIParameter(name=("--skip-requests",)),
    ]
    delay: Annotated[
        int,
        Field(
            default=500,
            description="Delay in ms between LLM calls within a session (simulates tool call timing).",
        ),
        CLIParameter(name=("--delay",)),
    ]
