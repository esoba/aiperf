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

    max_turns_mean: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Mean session turn count (lognormal). 0 disables turn-count limiting "
            "(sessions grow until token ceiling only).",
        ),
        CLIParameter(name=("--coding-session-max-turns-mean",), group=_CLI_GROUP),
    ] = 0

    max_turns_median: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Median session turn count (lognormal). 0 disables turn-count limiting.",
        ),
        CLIParameter(name=("--coding-session-max-turns-median",), group=_CLI_GROUP),
    ] = 0

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

    parallel_probability: Annotated[
        float,
        Field(
            default=0.3,
            ge=0.0,
            le=1.0,
            description="Probability a turn fans out into parallel tool calls. "
            "0.0 disables parallel generation entirely.",
        ),
        CLIParameter(name=("--coding-session-parallel-probability",), group=_CLI_GROUP),
    ] = 0.3

    parallel_fan_out_mean: Annotated[
        float,
        Field(
            default=3.0,
            ge=2.0,
            description="Mean number of parallel branches per fan-out.",
        ),
        CLIParameter(
            name=("--coding-session-parallel-fan-out-mean",), group=_CLI_GROUP
        ),
    ] = 3.0

    parallel_fan_out_max: Annotated[
        int,
        Field(
            default=8,
            ge=2,
            description="Maximum number of parallel branches per fan-out.",
        ),
        CLIParameter(name=("--coding-session-parallel-fan-out-max",), group=_CLI_GROUP),
    ] = 8

    parallel_branch_tokens_mean: Annotated[
        int,
        Field(
            default=800,
            ge=1,
            description="Mean new tokens per parallel branch (lognormal distribution).",
        ),
        CLIParameter(
            name=("--coding-session-parallel-branch-tokens-mean",), group=_CLI_GROUP
        ),
    ] = 800

    parallel_branch_tokens_median: Annotated[
        int,
        Field(
            default=400,
            ge=1,
            description="Median new tokens per parallel branch (lognormal distribution).",
        ),
        CLIParameter(
            name=("--coding-session-parallel-branch-tokens-median",), group=_CLI_GROUP
        ),
    ] = 400

    subagent_probability: Annotated[
        float,
        Field(
            default=0.15,
            ge=0.0,
            le=1.0,
            description="Probability a turn spawns subagent children (mutually exclusive "
            "with parallel_probability per turn). 0.0 disables subagent generation.",
        ),
        CLIParameter(name=("--coding-session-subagent-probability",), group=_CLI_GROUP),
    ] = 0.15

    subagent_count_mean: Annotated[
        float,
        Field(
            default=1.5,
            ge=1.0,
            description="Mean number of subagent children per spawn (Poisson).",
        ),
        CLIParameter(name=("--coding-session-subagent-count-mean",), group=_CLI_GROUP),
    ] = 1.5

    subagent_count_max: Annotated[
        int,
        Field(
            default=4,
            ge=1,
            description="Maximum number of subagent children per spawn.",
        ),
        CLIParameter(name=("--coding-session-subagent-count-max",), group=_CLI_GROUP),
    ] = 4

    subagent_turns_mean: Annotated[
        int,
        Field(
            default=8,
            ge=1,
            description="Mean number of turns per subagent child session (lognormal).",
        ),
        CLIParameter(name=("--coding-session-subagent-turns-mean",), group=_CLI_GROUP),
    ] = 8

    subagent_turns_median: Annotated[
        int,
        Field(
            default=5,
            ge=1,
            description="Median number of turns per subagent child session (lognormal).",
        ),
        CLIParameter(
            name=("--coding-session-subagent-turns-median",), group=_CLI_GROUP
        ),
    ] = 5

    subagent_system_tokens: Annotated[
        int,
        Field(
            default=4000,
            ge=0,
            description="System prompt tokens for subagent children (smaller tool set).",
        ),
        CLIParameter(
            name=("--coding-session-subagent-system-tokens",), group=_CLI_GROUP
        ),
    ] = 4000

    subagent_new_tokens_mean: Annotated[
        int,
        Field(
            default=2500,
            ge=1,
            description="Mean new tokens per turn in subagent children.",
        ),
        CLIParameter(
            name=("--coding-session-subagent-new-tokens-mean",), group=_CLI_GROUP
        ),
    ] = 2500

    subagent_new_tokens_median: Annotated[
        int,
        Field(
            default=1200,
            ge=1,
            description="Median new tokens per turn in subagent children.",
        ),
        CLIParameter(
            name=("--coding-session-subagent-new-tokens-median",), group=_CLI_GROUP
        ),
    ] = 1200

    subagent_max_prompt_tokens: Annotated[
        int,
        Field(
            default=50000,
            ge=1,
            description="Maximum prompt tokens for subagent children (shorter-lived).",
        ),
        CLIParameter(
            name=("--coding-session-subagent-max-prompt-tokens",), group=_CLI_GROUP
        ),
    ] = 50000

    # Cache lifetime mechanics
    l1_tokens: Annotated[
        int,
        Field(
            default=32000,
            ge=0,
            description="L1 (tools+system) tokens. Maps to deterministic hash_ids shared "
            "across all sessions. 0 disables L1 layer.",
        ),
        CLIParameter(name=("--coding-session-l1-tokens",), group=_CLI_GROUP),
    ] = 32000

    l2_tokens: Annotated[
        int,
        Field(
            default=1500,
            ge=0,
            description="L2 (CLAUDE.md+skills) tokens. Random per session, stable across turns. "
            "0 disables L2 layer.",
        ),
        CLIParameter(name=("--coding-session-l2-tokens",), group=_CLI_GROUP),
    ] = 1500

    restart_probability: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Per-turn probability of a --continue restart. Preserves L1, "
            "regenerates L2+L3 hash_ids. 0.0 disables.",
        ),
        CLIParameter(name=("--coding-session-restart-probability",), group=_CLI_GROUP),
    ] = 0.0

    compression_threshold: Annotated[
        float,
        Field(
            default=0.85,
            ge=0.0,
            le=1.0,
            description="Fraction of max_prompt_tokens that triggers context compression.",
        ),
        CLIParameter(
            name=("--coding-session-compression-threshold",), group=_CLI_GROUP
        ),
    ] = 0.85

    compression_ratio: Annotated[
        float,
        Field(
            default=0.3,
            ge=0.0,
            le=1.0,
            description="Fraction of L3 blocks retained after compression.",
        ),
        CLIParameter(name=("--coding-session-compression-ratio",), group=_CLI_GROUP),
    ] = 0.3

    max_compressions: Annotated[
        int,
        Field(
            default=3,
            ge=0,
            description="Maximum compression events per session. 0 disables compression.",
        ),
        CLIParameter(name=("--coding-session-max-compressions",), group=_CLI_GROUP),
    ] = 3

    thinking_tokens_mean: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Mean thinking tokens per tool-use turn (lognormal). 0 disables.",
        ),
        CLIParameter(name=("--coding-session-thinking-tokens-mean",), group=_CLI_GROUP),
    ] = 0

    thinking_tokens_median: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Median thinking tokens per tool-use turn (lognormal).",
        ),
        CLIParameter(
            name=("--coding-session-thinking-tokens-median",), group=_CLI_GROUP
        ),
    ] = 0

    thinking_strip_probability: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            le=1.0,
            description="Probability of stripping thinking blocks at non-tool-result boundary. "
            "Causes L2+L3 hash_id regeneration (cache invalidation).",
        ),
        CLIParameter(
            name=("--coding-session-thinking-strip-probability",), group=_CLI_GROUP
        ),
    ] = 1.0

    cache_ttl_sec: Annotated[
        float,
        Field(
            default=3600.0,
            gt=0,
            description="Main agent KV cache TTL in seconds for working set eviction.",
        ),
        CLIParameter(name=("--coding-session-cache-ttl-sec",), group=_CLI_GROUP),
    ] = 3600.0

    subagent_cache_ttl_sec: Annotated[
        float,
        Field(
            default=300.0,
            gt=0,
            description="Subagent KV cache TTL in seconds for working set eviction.",
        ),
        CLIParameter(
            name=("--coding-session-subagent-cache-ttl-sec",), group=_CLI_GROUP
        ),
    ] = 300.0

    delay_mean_ms: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Mean inter-turn delay in milliseconds (lognormal). "
            "0 disables delays (back-to-back turns).",
        ),
        CLIParameter(name=("--coding-session-delay-mean-ms",), group=_CLI_GROUP),
    ] = 0

    delay_median_ms: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Median inter-turn delay in milliseconds (lognormal). "
            "0 disables delays.",
        ),
        CLIParameter(name=("--coding-session-delay-median-ms",), group=_CLI_GROUP),
    ] = 0
