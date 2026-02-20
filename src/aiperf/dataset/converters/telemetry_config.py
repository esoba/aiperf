# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration model for the OpenAI-style telemetry trace converter."""

from typing import Annotated

from pydantic import Field

from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.dataset.converters.trace_converter import TraceConverterConfig


class TelemetryConfig(TraceConverterConfig):
    """Convert OpenAI-style telemetry JSONL to mooncake JSONL.

    Reads telemetry.jsonl with llm_call events, classifies agent types from
    system prompt prefixes, tokenizes message text, and produces session records
    with agent_type and priority fields.

    Examples:
        # Basic conversion with default tokenizer
        aiperf convert-trace telemetry --input-file telemetry.jsonl

        # Explicit tokenizer and block size
        aiperf convert-trace telemetry --input-file telemetry.jsonl --tokenizer meta-llama/Llama-3.3-70B-Instruct --block-size 128
    """

    tokenizer: Annotated[
        str,
        Field(
            default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            description="Tokenizer name/path for hashing.",
        ),
        CLIParameter(name=("--tokenizer",)),
    ]
    block_size: Annotated[
        int,
        Field(
            default=64,
            description="Block size for hash generation.",
        ),
        CLIParameter(name=("--block-size",)),
    ]
