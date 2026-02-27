# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for synthetic coding session generation."""

from typing import Annotated, Literal

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.groups import Groups


class CodingSessionConfig(BaseConfig):
    """Configuration for synthetic coding session generation.

    Generates multi-turn coding sessions with lognormal distributions for
    context growth, initial prefix, and generation length. Designed for
    adaptive_scale timing mode without requiring real trace files.
    """

    _CLI_GROUP = Groups.INPUT

    enabled: Annotated[
        bool,
        Field(
            default=False,
            description="Enable synthetic coding session generation. Generates multi-turn "
            "sessions with lognormal distributions matching real coding workload patterns. "
            "Mutually exclusive with --input-file and --public-dataset.",
        ),
        CLIParameter(name=("--coding-session",), group=_CLI_GROUP),
    ] = False

    num_sessions: Annotated[
        int,
        Field(
            default=200,
            ge=1,
            description="Number of synthetic coding sessions to generate.",
        ),
        CLIParameter(name=("--coding-session-num-sessions",), group=_CLI_GROUP),
    ] = 200

    system_prompt_tokens: Annotated[
        int,
        Field(
            default=8500,
            ge=0,
            description="Number of tokens for the system prompt prefix in each session.",
        ),
        CLIParameter(name=("--coding-session-system-prompt-tokens",), group=_CLI_GROUP),
    ] = 8500

    new_tokens_mean: Annotated[
        int,
        Field(
            default=4500,
            ge=1,
            description="Mean of the lognormal distribution for new tokens per turn.",
        ),
        CLIParameter(name=("--coding-session-new-tokens-mean",), group=_CLI_GROUP),
    ] = 4500

    new_tokens_median: Annotated[
        int,
        Field(
            default=2100,
            ge=1,
            description="Median of the lognormal distribution for new tokens per turn.",
        ),
        CLIParameter(name=("--coding-session-new-tokens-median",), group=_CLI_GROUP),
    ] = 2100

    max_prompt_tokens: Annotated[
        int,
        Field(
            default=215_000,
            ge=1,
            description="Maximum prompt tokens before a session is retired.",
        ),
        CLIParameter(name=("--coding-session-max-prompt-tokens",), group=_CLI_GROUP),
    ] = 215_000

    initial_prefix_mean: Annotated[
        int,
        Field(
            default=67_000,
            ge=1,
            description="Mean of the lognormal distribution for the initial prefix tokens.",
        ),
        CLIParameter(name=("--coding-session-initial-prefix-mean",), group=_CLI_GROUP),
    ] = 67_000

    initial_prefix_median: Annotated[
        int,
        Field(
            default=54_000,
            ge=1,
            description="Median of the lognormal distribution for the initial prefix tokens.",
        ),
        CLIParameter(
            name=("--coding-session-initial-prefix-median",), group=_CLI_GROUP
        ),
    ] = 54_000

    generation_length_mean: Annotated[
        int,
        Field(
            default=600,
            ge=1,
            description="Mean of the lognormal distribution for output generation length.",
        ),
        CLIParameter(
            name=("--coding-session-generation-length-mean",), group=_CLI_GROUP
        ),
    ] = 600

    generation_length_median: Annotated[
        int,
        Field(
            default=350,
            ge=1,
            description="Median of the lognormal distribution for output generation length.",
        ),
        CLIParameter(
            name=("--coding-session-generation-length-median",), group=_CLI_GROUP
        ),
    ] = 350

    block_size: Annotated[
        int,
        Field(
            default=64,
            ge=1,
            description="KV cache block size in tokens for hash ID generation.",
        ),
        CLIParameter(name=("--coding-session-block-size",), group=_CLI_GROUP),
    ] = 64

    tool_result_ratio: Annotated[
        float,
        Field(
            default=0.9,
            ge=0.0,
            le=1.0,
            description="Probability a turn uses tool_result content vs text content. "
            "Real trace data shows ~90%% tool_result and ~10%% text by token count.",
        ),
        CLIParameter(name=("--coding-session-tool-result-ratio",), group=_CLI_GROUP),
    ] = 0.9

    language: Annotated[
        Literal["python", "go", "rust", "typescript", "mixed"],
        Field(
            default="mixed",
            description="Programming language for session content. 'mixed' randomly assigns "
            "a language per session using weighted distribution. A specific language forces "
            "all sessions to use that language's tool pool.",
        ),
        CLIParameter(name=("--coding-session-language",), group=_CLI_GROUP),
    ] = "mixed"
