# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI-to-AIPerfConfig mapping table.

Defines CLI flags and their relationship to AIPerfConfig field paths.
Descriptions, types, and defaults are read from AIPerfConfig sub-models
via introspection at import time — they are NOT duplicated here.

Only CLI-specific metadata lives in this file:
- Flag names and aliases
- Help groups
- Warmup override relationships
- Custom parsers and cyclopts behavior
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiperf.common.enums import CaseInsensitiveStrEnum

_UNSET = object()


class CLIGroup(CaseInsensitiveStrEnum):
    """CLI help groups — controls display order in --help."""

    ENDPOINT = "Endpoint"
    INPUT = "Input"
    FIXED_SCHEDULE = "Fixed Schedule"
    GOODPUT = "Goodput"
    OUTPUT = "Output"
    HTTP_TRACE = "HTTP Trace"
    TOKENIZER = "Tokenizer"
    LOAD_GENERATOR = "Load Generator"
    WARMUP = "Warmup"
    USER_CENTRIC = "User-Centric Rate"
    REQUEST_CANCELLATION = "Request Cancellation"
    CONVERSATION_INPUT = "Conversation Input"
    ISL = "Input Sequence Length (ISL)"
    OSL = "Output Sequence Length (OSL)"
    PROMPT = "Prompt"
    PREFIX_PROMPT = "Prefix Prompt"
    RANKINGS = "Rankings"
    SYNTHESIS = "Synthesis"
    AUDIO_INPUT = "Audio Input"
    IMAGE_INPUT = "Image Input"
    VIDEO_INPUT = "Video Input"
    SERVICE = "Service"
    SERVER_METRICS = "Server Metrics"
    GPU_TELEMETRY = "GPU Telemetry"
    UI = "UI"
    WORKERS = "Workers"
    ZMQ_COMMUNICATION = "ZMQ Communication"
    MULTI_RUN = "Multi-Run"
    ACCURACY = "Accuracy"


@dataclass(frozen=True, slots=True)
class CLIField:
    """Maps a CLI flag to an AIPerfConfig field path.

    For most fields, descriptions/types/defaults are introspected from AIPerfConfig.
    Override fields are provided for CLI-only flags or when CLI types differ from YAML.

    Attributes:
        flags: CLI flag name(s). First is primary, rest are aliases.
        field_name: Python attribute name in the generated CLI model.
        group: Help group for --help display.
        path: Dot-path into AIPerfConfig for introspection. None for CLI-only fields.
        warmup_of: If this is a warmup override, the flag it overrides.
        consume_multiple: Cyclopts: collect repeated flag values into a list.
        negative: Cyclopts: generate --no-{flag} for booleans.
        show_choices: Cyclopts: show enum choices in help.
        parse: False to disable CLI parsing (auto-generated fields).
        type_override: Override introspected type when CLI type differs.
        default_override: Override introspected default.
        description_override: Override introspected description (CLI-only fields).
        before_validators: Names of BeforeValidator functions to attach.
    """

    flags: str | tuple[str, ...]
    field_name: str
    group: CLIGroup
    path: str | None = None
    warmup_of: str | None = None
    consume_multiple: bool = False
    negative: bool = False
    show_choices: bool = True
    parse: bool = True
    type_override: Any = None
    default_override: Any = _UNSET
    description_override: str | None = None
    before_validators: tuple[str, ...] = ()

    @property
    def primary_flag(self) -> str:
        return self.flags if isinstance(self.flags, str) else self.flags[0]

    @property
    def is_warmup_override(self) -> bool:
        return self.warmup_of is not None

    @property
    def is_cli_only(self) -> bool:
        return self.path is None


# =============================================================================
# MAPPING TABLE
# =============================================================================

CLI_FIELDS: list[CLIField] = [
    # =========================================================================
    # ENDPOINT
    # =========================================================================
    CLIField(
        flags=("--model-names", "--model", "-m"),
        field_name="model_names",
        group=CLIGroup.ENDPOINT,
        path="models.items",
        type_override="list[str]",
        before_validators=("parse_str_or_list",),
    ),
    CLIField(
        flags="--model-selection-strategy",
        field_name="model_selection_strategy",
        group=CLIGroup.ENDPOINT,
        path="models.strategy",
    ),
    CLIField(
        flags=("--url", "-u"),
        field_name="urls",
        group=CLIGroup.ENDPOINT,
        path="endpoint.urls",
        consume_multiple=True,
        default_override=["localhost:8000"],
        before_validators=("parse_str_or_list",),
    ),
    CLIField(
        flags="--url-strategy",
        field_name="url_selection_strategy",
        group=CLIGroup.ENDPOINT,
        path="endpoint.url_strategy",
    ),
    CLIField(
        flags="--endpoint-type",
        field_name="endpoint_type",
        group=CLIGroup.ENDPOINT,
        path="endpoint.type",
    ),
    CLIField(
        flags="--streaming",
        field_name="streaming",
        group=CLIGroup.ENDPOINT,
        path="endpoint.streaming",
    ),
    CLIField(
        flags=("--custom-endpoint", "--endpoint"),
        field_name="custom_endpoint",
        group=CLIGroup.ENDPOINT,
        path="endpoint.path",
    ),
    CLIField(
        flags="--api-key",
        field_name="api_key",
        group=CLIGroup.ENDPOINT,
        path="endpoint.api_key",
    ),
    CLIField(
        flags="--request-timeout-seconds",
        field_name="request_timeout_seconds",
        group=CLIGroup.ENDPOINT,
        path="endpoint.timeout",
    ),
    CLIField(
        flags=("--transport", "--transport-type"),
        field_name="transport_type",
        group=CLIGroup.ENDPOINT,
        path="endpoint.transport",
    ),
    CLIField(
        flags="--use-legacy-max-tokens",
        field_name="use_legacy_max_tokens",
        group=CLIGroup.ENDPOINT,
        path="endpoint.use_legacy_max_tokens",
    ),
    CLIField(
        flags="--use-server-token-count",
        field_name="use_server_token_count",
        group=CLIGroup.ENDPOINT,
        path="endpoint.use_server_token_count",
    ),
    CLIField(
        flags="--connection-reuse-strategy",
        field_name="connection_reuse_strategy",
        group=CLIGroup.ENDPOINT,
        path="endpoint.connection_reuse",
    ),
    CLIField(
        flags="--download-video-content",
        field_name="download_video_content",
        group=CLIGroup.ENDPOINT,
        path="endpoint.download_video_content",
    ),
    CLIField(
        flags="--extra-inputs",
        field_name="extra_inputs",
        group=CLIGroup.ENDPOINT,
        path="endpoint.extra",
        consume_multiple=True,
        type_override="Any",
        before_validators=("parse_str_or_dict_as_tuple_list",),
    ),
    CLIField(
        flags=("--header", "-H"),
        field_name="headers",
        group=CLIGroup.ENDPOINT,
        path="endpoint.headers",
        consume_multiple=True,
        type_override="Any",
        before_validators=("parse_str_or_dict_as_tuple_list",),
    ),
    # =========================================================================
    # INPUT
    # =========================================================================
    CLIField(
        flags="--input-file",
        field_name="input_file",
        group=CLIGroup.INPUT,
        path="datasets.main.path",
        before_validators=("parse_file",),
    ),
    CLIField(
        flags="--public-dataset",
        field_name="public_dataset",
        group=CLIGroup.INPUT,
        path="datasets.main.name",
    ),
    CLIField(
        flags="--custom-dataset-type",
        field_name="custom_dataset_type",
        group=CLIGroup.INPUT,
        path="datasets.main.format",
    ),
    CLIField(
        flags="--dataset-sampling-strategy",
        field_name="dataset_sampling_strategy",
        group=CLIGroup.INPUT,
        path="datasets.main.sampling",
    ),
    CLIField(
        flags="--random-seed",
        field_name="random_seed",
        group=CLIGroup.INPUT,
        path="random_seed",
    ),
    # =========================================================================
    # FIXED SCHEDULE
    # =========================================================================
    CLIField(
        flags="--fixed-schedule",
        field_name="fixed_schedule",
        group=CLIGroup.FIXED_SCHEDULE,
        negative=True,
        description_override=(
            "Run requests according to timestamps specified in the input dataset. "
            "When enabled, AIPerf replays the exact timing pattern from the dataset. "
            "This mode is automatically enabled for mooncake_trace datasets."
        ),
        type_override="bool",
        default_override=False,
    ),
    CLIField(
        flags="--fixed-schedule-auto-offset",
        field_name="fixed_schedule_auto_offset",
        group=CLIGroup.FIXED_SCHEDULE,
        path="load.profiling.auto_offset",
    ),
    CLIField(
        flags="--fixed-schedule-start-offset",
        field_name="fixed_schedule_start_offset",
        group=CLIGroup.FIXED_SCHEDULE,
        path="load.profiling.start_offset",
    ),
    CLIField(
        flags="--fixed-schedule-end-offset",
        field_name="fixed_schedule_end_offset",
        group=CLIGroup.FIXED_SCHEDULE,
        path="load.profiling.end_offset",
    ),
    # =========================================================================
    # GOODPUT
    # =========================================================================
    CLIField(
        flags="--goodput",
        field_name="goodput",
        group=CLIGroup.GOODPUT,
        path="slos",
        type_override="Any | None",
        default_override=None,
        before_validators=("parse_str_as_numeric_dict",),
    ),
    # =========================================================================
    # OUTPUT
    # =========================================================================
    CLIField(
        flags=("--output-artifact-dir", "--artifact-dir"),
        field_name="artifact_directory",
        group=CLIGroup.OUTPUT,
        path="artifacts.dir",
    ),
    CLIField(
        flags=("--profile-export-prefix", "--profile-export-file"),
        field_name="profile_export_prefix",
        group=CLIGroup.OUTPUT,
        path="artifacts.prefix",
        type_override="Path | None",
        default_override=None,
    ),
    CLIField(
        flags=("--export-level", "--profile-export-level"),
        field_name="export_level",
        group=CLIGroup.OUTPUT,
        description_override=(
            "Controls which output files are generated. "
            "summary: Only aggregate metrics files. "
            "records: Includes per-request metrics. "
            "raw: Includes raw request/response data."
        ),
        type_override="ExportLevel",
        default_override="ExportLevel.SUMMARY",
    ),
    CLIField(
        flags="--slice-duration",
        field_name="slice_duration",
        group=CLIGroup.OUTPUT,
        path="artifacts.slice_duration",
    ),
    # =========================================================================
    # HTTP TRACE
    # =========================================================================
    CLIField(
        flags="--export-http-trace",
        field_name="export_http_trace",
        group=CLIGroup.HTTP_TRACE,
        path="artifacts.trace",
    ),
    CLIField(
        flags="--show-trace-timing",
        field_name="show_trace_timing",
        group=CLIGroup.HTTP_TRACE,
        path="artifacts.show_trace_timing",
    ),
    # =========================================================================
    # TOKENIZER
    # =========================================================================
    CLIField(
        flags="--tokenizer",
        field_name="tokenizer_name",
        group=CLIGroup.TOKENIZER,
        path="tokenizer.name",
    ),
    CLIField(
        flags="--tokenizer-revision",
        field_name="tokenizer_revision",
        group=CLIGroup.TOKENIZER,
        path="tokenizer.revision",
    ),
    CLIField(
        flags="--tokenizer-trust-remote-code",
        field_name="tokenizer_trust_remote_code",
        group=CLIGroup.TOKENIZER,
        path="tokenizer.trust_remote_code",
    ),
    # =========================================================================
    # LOAD GENERATOR → load.profiling
    # =========================================================================
    CLIField(
        flags="--benchmark-duration",
        field_name="benchmark_duration",
        group=CLIGroup.LOAD_GENERATOR,
        path="load.profiling.duration",
    ),
    CLIField(
        flags="--benchmark-grace-period",
        field_name="benchmark_grace_period",
        group=CLIGroup.LOAD_GENERATOR,
        path="load.profiling.grace_period",
    ),
    CLIField(
        flags="--concurrency",
        field_name="concurrency",
        group=CLIGroup.LOAD_GENERATOR,
        path="load.profiling.concurrency",
    ),
    CLIField(
        flags="--prefill-concurrency",
        field_name="prefill_concurrency",
        group=CLIGroup.LOAD_GENERATOR,
        path="load.profiling.prefill_concurrency",
    ),
    CLIField(
        flags="--request-rate",
        field_name="request_rate",
        group=CLIGroup.LOAD_GENERATOR,
        path="load.profiling.rate",
    ),
    CLIField(
        flags=("--arrival-pattern", "--request-rate-mode"),
        field_name="arrival_pattern",
        group=CLIGroup.LOAD_GENERATOR,
        description_override=(
            "Sets the arrival pattern for the load generated by AIPerf. "
            "Valid values: constant, poisson, gamma.\n"
            "constant: Generate requests at a fixed rate.\n"
            "poisson: Generate requests using a poisson distribution.\n"
            "gamma: Generate requests using a gamma distribution with tunable smoothness."
        ),
        type_override="ArrivalPattern",
        default_override="ArrivalPattern.POISSON",
    ),
    CLIField(
        flags=("--arrival-smoothness", "--vllm-burstiness"),
        field_name="arrival_smoothness",
        group=CLIGroup.LOAD_GENERATOR,
        path="load.profiling.smoothness",
    ),
    CLIField(
        flags=("--request-count", "--num-requests"),
        field_name="request_count",
        group=CLIGroup.LOAD_GENERATOR,
        path="load.profiling.requests",
    ),
    CLIField(
        flags="--concurrency-ramp-duration",
        field_name="concurrency_ramp_duration",
        group=CLIGroup.LOAD_GENERATOR,
        path="load.profiling.concurrency_ramp.duration",
    ),
    CLIField(
        flags="--prefill-concurrency-ramp-duration",
        field_name="prefill_concurrency_ramp_duration",
        group=CLIGroup.LOAD_GENERATOR,
        path="load.profiling.prefill_ramp.duration",
    ),
    CLIField(
        flags="--request-rate-ramp-duration",
        field_name="request_rate_ramp_duration",
        group=CLIGroup.WARMUP,
        path="load.profiling.rate_ramp.duration",
    ),
    # =========================================================================
    # WARMUP — same PhaseConfig fields, descriptions auto-derived
    # =========================================================================
    CLIField(
        flags=("--warmup-request-count", "--num-warmup-requests"),
        field_name="warmup_request_count",
        group=CLIGroup.WARMUP,
        path="load.warmup.requests",
        warmup_of="--request-count",
    ),
    CLIField(
        flags="--warmup-duration",
        field_name="warmup_duration",
        group=CLIGroup.WARMUP,
        path="load.warmup.duration",
        warmup_of="--benchmark-duration",
    ),
    CLIField(
        flags="--num-warmup-sessions",
        field_name="warmup_num_sessions",
        group=CLIGroup.WARMUP,
        path="load.warmup.sessions",
        warmup_of="--conversation-num",
    ),
    CLIField(
        flags="--warmup-concurrency",
        field_name="warmup_concurrency",
        group=CLIGroup.WARMUP,
        path="load.warmup.concurrency",
        warmup_of="--concurrency",
    ),
    CLIField(
        flags="--warmup-prefill-concurrency",
        field_name="warmup_prefill_concurrency",
        group=CLIGroup.WARMUP,
        path="load.warmup.prefill_concurrency",
        warmup_of="--prefill-concurrency",
    ),
    CLIField(
        flags="--warmup-request-rate",
        field_name="warmup_request_rate",
        group=CLIGroup.WARMUP,
        path="load.warmup.rate",
        warmup_of="--request-rate",
    ),
    CLIField(
        flags="--warmup-arrival-pattern",
        field_name="warmup_arrival_pattern",
        group=CLIGroup.WARMUP,
        path="load.warmup.type",
        warmup_of="--arrival-pattern",
        show_choices=False,
    ),
    CLIField(
        flags="--warmup-grace-period",
        field_name="warmup_grace_period",
        group=CLIGroup.WARMUP,
        path="load.warmup.grace_period",
        warmup_of="--benchmark-grace-period",
    ),
    CLIField(
        flags="--warmup-concurrency-ramp-duration",
        field_name="warmup_concurrency_ramp_duration",
        group=CLIGroup.WARMUP,
        path="load.warmup.concurrency_ramp.duration",
        warmup_of="--concurrency-ramp-duration",
    ),
    CLIField(
        flags="--warmup-prefill-concurrency-ramp-duration",
        field_name="warmup_prefill_concurrency_ramp_duration",
        group=CLIGroup.WARMUP,
        path="load.warmup.prefill_ramp.duration",
        warmup_of="--prefill-concurrency-ramp-duration",
    ),
    CLIField(
        flags="--warmup-request-rate-ramp-duration",
        field_name="warmup_request_rate_ramp_duration",
        group=CLIGroup.WARMUP,
        path="load.warmup.rate_ramp.duration",
        warmup_of="--request-rate-ramp-duration",
    ),
    # =========================================================================
    # USER-CENTRIC RATE
    # =========================================================================
    CLIField(
        flags="--user-centric-rate",
        field_name="user_centric_rate",
        group=CLIGroup.USER_CENTRIC,
        path="load.profiling.rate",
        description_override=(
            "Enable user-centric rate limiting mode with the specified request rate (QPS). "
            "Each user has a gap = num_users / qps between turns. "
            "Designed for KV cache benchmarking with realistic multi-user patterns. "
            "Requires --num-users to be set."
        ),
        type_override="float | None",
        default_override=None,
    ),
    CLIField(
        flags="--num-users",
        field_name="num_users",
        group=CLIGroup.USER_CENTRIC,
        path="load.profiling.users",
    ),
    # =========================================================================
    # REQUEST CANCELLATION
    # =========================================================================
    CLIField(
        flags="--request-cancellation-rate",
        field_name="request_cancellation_rate",
        group=CLIGroup.REQUEST_CANCELLATION,
        description_override=(
            "Percentage (0-100) of requests to cancel for testing cancellation handling. "
            "Cancelled requests are sent normally but aborted after --request-cancellation-delay seconds."
        ),
        type_override="float | None",
        default_override=None,
    ),
    CLIField(
        flags="--request-cancellation-delay",
        field_name="request_cancellation_delay",
        group=CLIGroup.REQUEST_CANCELLATION,
        description_override=(
            "Seconds to wait after the request is fully sent before cancelling. "
            "A delay of 0 means send the full request, then immediately disconnect. "
            "Requires --request-cancellation-rate to be set."
        ),
        type_override="float",
        default_override=0.0,
    ),
    # =========================================================================
    # CONVERSATION INPUT
    # =========================================================================
    CLIField(
        flags=("--conversation-turn-delay-mean", "--session-turn-delay-mean"),
        field_name="turn_delay_mean",
        group=CLIGroup.CONVERSATION_INPUT,
        description_override=(
            "Mean delay in milliseconds between consecutive turns within a multi-turn conversation. "
            "Simulates user think time. Set to 0 for back-to-back turns."
        ),
        type_override="float",
        default_override=0.0,
    ),
    CLIField(
        flags=("--conversation-turn-delay-stddev", "--session-turn-delay-stddev"),
        field_name="turn_delay_stddev",
        group=CLIGroup.CONVERSATION_INPUT,
        description_override="Standard deviation for turn delays in milliseconds.",
        type_override="float",
        default_override=0.0,
    ),
    CLIField(
        flags=("--conversation-turn-delay-ratio", "--session-delay-ratio"),
        field_name="turn_delay_ratio",
        group=CLIGroup.CONVERSATION_INPUT,
        description_override="Multiplier for scaling all turn delays. Values < 1 speed up, > 1 slow down.",
        type_override="float",
        default_override=1.0,
    ),
    CLIField(
        flags=("--conversation-turn-mean", "--session-turns-mean"),
        field_name="num_turns_mean",
        group=CLIGroup.CONVERSATION_INPUT,
        description_override="Mean number of request-response turns per conversation. Set to 1 for single-turn.",
        type_override="int",
        default_override=1,
    ),
    CLIField(
        flags=("--conversation-turn-stddev", "--session-turns-stddev"),
        field_name="num_turns_stddev",
        group=CLIGroup.CONVERSATION_INPUT,
        description_override="Standard deviation for number of turns per conversation.",
        type_override="int",
        default_override=0,
    ),
    CLIField(
        flags=("--conversation-num", "--num-conversations", "--num-sessions"),
        field_name="num_sessions",
        group=CLIGroup.CONVERSATION_INPUT,
        path="load.profiling.sessions",
    ),
    CLIField(
        flags=("--num-dataset-entries", "--num-prompts"),
        field_name="num_dataset_entries",
        group=CLIGroup.CONVERSATION_INPUT,
        path="datasets.main.entries",
    ),
    # =========================================================================
    # ISL
    # =========================================================================
    CLIField(
        flags=("--prompt-input-tokens-mean", "--synthetic-input-tokens-mean", "--isl"),
        field_name="isl_mean",
        group=CLIGroup.ISL,
        path="datasets.main.prompts.isl.mean",
        type_override="int",
        default_override=550,
    ),
    CLIField(
        flags=(
            "--prompt-input-tokens-stddev",
            "--synthetic-input-tokens-stddev",
            "--isl-stddev",
        ),
        field_name="isl_stddev",
        group=CLIGroup.ISL,
        path="datasets.main.prompts.isl.stddev",
        type_override="float",
        default_override=0.0,
    ),
    CLIField(
        flags=(
            "--prompt-input-tokens-block-size",
            "--synthetic-input-tokens-block-size",
            "--isl-block-size",
        ),
        field_name="isl_block_size",
        group=CLIGroup.ISL,
        path="datasets.main.prompts.block_size",
    ),
    CLIField(
        flags=("--seq-dist", "--sequence-distribution"),
        field_name="sequence_distribution",
        group=CLIGroup.ISL,
        description_override=(
            "Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. "
            "Format: ISL,OSL:prob;ISL,OSL:prob (probabilities 0-100 summing to 100)."
        ),
        type_override="str | None",
        default_override=None,
        before_validators=("validate_sequence_distribution",),
    ),
    # =========================================================================
    # OSL
    # =========================================================================
    CLIField(
        flags=("--prompt-output-tokens-mean", "--output-tokens-mean", "--osl"),
        field_name="osl_mean",
        group=CLIGroup.OSL,
        path="datasets.main.prompts.osl.mean",
        type_override="int | None",
        default_override=None,
    ),
    CLIField(
        flags=(
            "--prompt-output-tokens-stddev",
            "--output-tokens-stddev",
            "--osl-stddev",
        ),
        field_name="osl_stddev",
        group=CLIGroup.OSL,
        path="datasets.main.prompts.osl.stddev",
        type_override="float | None",
        default_override=0.0,
    ),
    # =========================================================================
    # PROMPT
    # =========================================================================
    CLIField(
        flags=("--prompt-batch-size", "--batch-size-text", "--batch-size", "-b"),
        field_name="prompt_batch_size",
        group=CLIGroup.PROMPT,
        path="datasets.main.prompts.batch_size",
    ),
    # =========================================================================
    # PREFIX PROMPT
    # =========================================================================
    CLIField(
        flags=(
            "--prompt-prefix-pool-size",
            "--prefix-prompt-pool-size",
            "--num-prefix-prompts",
        ),
        field_name="num_prefix_prompts",
        group=CLIGroup.PREFIX_PROMPT,
        path="datasets.main.prefix_prompts.pool_size",
        type_override="int",
        default_override=0,
    ),
    CLIField(
        flags=("--prompt-prefix-length", "--prefix-prompt-length"),
        field_name="prefix_prompt_length",
        group=CLIGroup.PREFIX_PROMPT,
        path="datasets.main.prefix_prompts.length",
        type_override="int",
        default_override=0,
    ),
    CLIField(
        flags="--shared-system-prompt-length",
        field_name="shared_system_prompt_length",
        group=CLIGroup.PREFIX_PROMPT,
        path="datasets.main.prefix_prompts.shared_system_length",
    ),
    CLIField(
        flags="--user-context-prompt-length",
        field_name="user_context_prompt_length",
        group=CLIGroup.PREFIX_PROMPT,
        path="datasets.main.prefix_prompts.user_context_length",
    ),
    # =========================================================================
    # RANKINGS
    # =========================================================================
    CLIField(
        flags="--rankings-passages-mean",
        field_name="passages_mean",
        group=CLIGroup.RANKINGS,
        path="datasets.main.rankings.passages.mean",
        type_override="int",
        default_override=10,
    ),
    CLIField(
        flags="--rankings-passages-stddev",
        field_name="passages_stddev",
        group=CLIGroup.RANKINGS,
        path="datasets.main.rankings.passages.stddev",
        type_override="int",
        default_override=0,
    ),
    CLIField(
        flags="--rankings-passages-prompt-token-mean",
        field_name="passages_prompt_token_mean",
        group=CLIGroup.RANKINGS,
        path="datasets.main.rankings.passage_tokens.mean",
        type_override="int",
        default_override=128,
    ),
    CLIField(
        flags="--rankings-passages-prompt-token-stddev",
        field_name="passages_prompt_token_stddev",
        group=CLIGroup.RANKINGS,
        path="datasets.main.rankings.passage_tokens.stddev",
        type_override="int",
        default_override=0,
    ),
    CLIField(
        flags="--rankings-query-prompt-token-mean",
        field_name="query_prompt_token_mean",
        group=CLIGroup.RANKINGS,
        path="datasets.main.rankings.query_tokens.mean",
        type_override="int",
        default_override=32,
    ),
    CLIField(
        flags="--rankings-query-prompt-token-stddev",
        field_name="query_prompt_token_stddev",
        group=CLIGroup.RANKINGS,
        path="datasets.main.rankings.query_tokens.stddev",
        type_override="int",
        default_override=0,
    ),
    # =========================================================================
    # SYNTHESIS
    # =========================================================================
    CLIField(
        flags="--synthesis-speedup-ratio",
        field_name="synthesis_speedup_ratio",
        group=CLIGroup.SYNTHESIS,
        path="datasets.main.synthesis.speedup_ratio",
        type_override="float",
        default_override=1.0,
    ),
    CLIField(
        flags="--synthesis-prefix-len-multiplier",
        field_name="synthesis_prefix_len_multiplier",
        group=CLIGroup.SYNTHESIS,
        path="datasets.main.synthesis.prefix_len_multiplier",
        type_override="float",
        default_override=1.0,
    ),
    CLIField(
        flags="--synthesis-prefix-root-multiplier",
        field_name="synthesis_prefix_root_multiplier",
        group=CLIGroup.SYNTHESIS,
        path="datasets.main.synthesis.prefix_root_multiplier",
        type_override="int",
        default_override=1,
    ),
    CLIField(
        flags="--synthesis-prompt-len-multiplier",
        field_name="synthesis_prompt_len_multiplier",
        group=CLIGroup.SYNTHESIS,
        path="datasets.main.synthesis.prompt_len_multiplier",
        type_override="float",
        default_override=1.0,
    ),
    CLIField(
        flags="--synthesis-max-isl",
        field_name="synthesis_max_isl",
        group=CLIGroup.SYNTHESIS,
        path="datasets.main.synthesis.max_isl",
    ),
    CLIField(
        flags="--synthesis-max-osl",
        field_name="synthesis_max_osl",
        group=CLIGroup.SYNTHESIS,
        path="datasets.main.synthesis.max_osl",
    ),
    # =========================================================================
    # AUDIO INPUT
    # =========================================================================
    CLIField(
        flags="--audio-length-mean",
        field_name="audio_length_mean",
        group=CLIGroup.AUDIO_INPUT,
        path="datasets.main.audio.length.mean",
        type_override="float",
        default_override=10.0,
    ),
    CLIField(
        flags="--audio-length-stddev",
        field_name="audio_length_stddev",
        group=CLIGroup.AUDIO_INPUT,
        path="datasets.main.audio.length.stddev",
        type_override="float",
        default_override=0.0,
    ),
    CLIField(
        flags=("--audio-batch-size", "--batch-size-audio"),
        field_name="audio_batch_size",
        group=CLIGroup.AUDIO_INPUT,
        path="datasets.main.audio.batch_size",
    ),
    CLIField(
        flags="--audio-format",
        field_name="audio_format",
        group=CLIGroup.AUDIO_INPUT,
        path="datasets.main.audio.format",
    ),
    CLIField(
        flags="--audio-depths",
        field_name="audio_depths",
        group=CLIGroup.AUDIO_INPUT,
        path="datasets.main.audio.depths",
        before_validators=("parse_str_or_list_of_positive_values",),
    ),
    CLIField(
        flags="--audio-sample-rates",
        field_name="audio_sample_rates",
        group=CLIGroup.AUDIO_INPUT,
        path="datasets.main.audio.sample_rates",
        before_validators=("parse_str_or_list_of_positive_values",),
    ),
    CLIField(
        flags="--audio-num-channels",
        field_name="audio_num_channels",
        group=CLIGroup.AUDIO_INPUT,
        path="datasets.main.audio.channels",
    ),
    # =========================================================================
    # IMAGE INPUT
    # =========================================================================
    CLIField(
        flags="--image-height-mean",
        field_name="image_height_mean",
        group=CLIGroup.IMAGE_INPUT,
        path="datasets.main.images.height.mean",
        type_override="float",
        default_override=512.0,
    ),
    CLIField(
        flags="--image-height-stddev",
        field_name="image_height_stddev",
        group=CLIGroup.IMAGE_INPUT,
        path="datasets.main.images.height.stddev",
        type_override="float",
        default_override=0.0,
    ),
    CLIField(
        flags="--image-width-mean",
        field_name="image_width_mean",
        group=CLIGroup.IMAGE_INPUT,
        path="datasets.main.images.width.mean",
        type_override="float",
        default_override=512.0,
    ),
    CLIField(
        flags="--image-width-stddev",
        field_name="image_width_stddev",
        group=CLIGroup.IMAGE_INPUT,
        path="datasets.main.images.width.stddev",
        type_override="float",
        default_override=0.0,
    ),
    CLIField(
        flags=("--image-batch-size", "--batch-size-image"),
        field_name="image_batch_size",
        group=CLIGroup.IMAGE_INPUT,
        path="datasets.main.images.batch_size",
    ),
    CLIField(
        flags="--image-format",
        field_name="image_format",
        group=CLIGroup.IMAGE_INPUT,
        path="datasets.main.images.format",
    ),
    # =========================================================================
    # VIDEO INPUT
    # =========================================================================
    CLIField(
        flags=("--video-batch-size", "--batch-size-video"),
        field_name="video_batch_size",
        group=CLIGroup.VIDEO_INPUT,
        path="datasets.main.video.batch_size",
    ),
    CLIField(
        flags="--video-duration",
        field_name="video_duration",
        group=CLIGroup.VIDEO_INPUT,
        path="datasets.main.video.duration",
    ),
    CLIField(
        flags="--video-fps",
        field_name="video_fps",
        group=CLIGroup.VIDEO_INPUT,
        path="datasets.main.video.fps",
    ),
    CLIField(
        flags="--video-width",
        field_name="video_width",
        group=CLIGroup.VIDEO_INPUT,
        path="datasets.main.video.width",
    ),
    CLIField(
        flags="--video-height",
        field_name="video_height",
        group=CLIGroup.VIDEO_INPUT,
        path="datasets.main.video.height",
    ),
    CLIField(
        flags="--video-synth-type",
        field_name="video_synth_type",
        group=CLIGroup.VIDEO_INPUT,
        path="datasets.main.video.synth_type",
    ),
    CLIField(
        flags="--video-format",
        field_name="video_format",
        group=CLIGroup.VIDEO_INPUT,
        path="datasets.main.video.format",
    ),
    CLIField(
        flags="--video-codec",
        field_name="video_codec",
        group=CLIGroup.VIDEO_INPUT,
        path="datasets.main.video.codec",
    ),
    # =========================================================================
    # SERVICE
    # =========================================================================
    CLIField(
        flags="--log-level",
        field_name="log_level",
        group=CLIGroup.SERVICE,
        path="logging.level",
    ),
    CLIField(
        flags=("--verbose", "-v"),
        field_name="verbose",
        group=CLIGroup.SERVICE,
        description_override=(
            "Equivalent to --log-level DEBUG. Enables detailed logging and switches UI to simple mode."
        ),
        type_override="bool",
        default_override=False,
    ),
    CLIField(
        flags=("--extra-verbose", "-vv"),
        field_name="extra_verbose",
        group=CLIGroup.SERVICE,
        description_override=(
            "Equivalent to --log-level TRACE. Most verbose logging including ZMQ messages. Switches UI to simple mode."
        ),
        type_override="bool",
        default_override=False,
    ),
    CLIField(
        flags=("--record-processor-service-count", "--record-processors"),
        field_name="record_processor_service_count",
        group=CLIGroup.SERVICE,
        path="runtime.record_processors",
    ),
    # =========================================================================
    # SERVER METRICS
    # =========================================================================
    CLIField(
        flags="--server-metrics",
        field_name="server_metrics",
        group=CLIGroup.SERVER_METRICS,
        consume_multiple=True,
        description_override=(
            "Server metrics collection (ENABLED BY DEFAULT). "
            "Optionally specify additional Prometheus endpoint URLs. "
            "Use --no-server-metrics to disable."
        ),
        type_override="list[str] | None",
        default_override=None,
        before_validators=("parse_str_or_list",),
    ),
    CLIField(
        flags="--no-server-metrics",
        field_name="no_server_metrics",
        group=CLIGroup.SERVER_METRICS,
        description_override="Disable server metrics collection entirely.",
        type_override="bool",
        default_override=False,
    ),
    CLIField(
        flags="--server-metrics-formats",
        field_name="server_metrics_formats",
        group=CLIGroup.SERVER_METRICS,
        path="server_metrics.formats",
        consume_multiple=True,
        before_validators=("parse_str_or_list",),
    ),
    # =========================================================================
    # GPU TELEMETRY
    # =========================================================================
    CLIField(
        flags="--gpu-telemetry",
        field_name="gpu_telemetry",
        group=CLIGroup.GPU_TELEMETRY,
        consume_multiple=True,
        description_override=(
            "Enable GPU telemetry and optionally specify: "
            "'dashboard' for realtime mode, "
            "custom DCGM URLs, or a metrics CSV file."
        ),
        type_override="list[str] | None",
        default_override=None,
        before_validators=("parse_str_or_list",),
    ),
    CLIField(
        flags="--no-gpu-telemetry",
        field_name="no_gpu_telemetry",
        group=CLIGroup.GPU_TELEMETRY,
        description_override="Disable GPU telemetry collection entirely.",
        type_override="bool",
        default_override=False,
    ),
    # =========================================================================
    # UI
    # =========================================================================
    CLIField(
        flags=("--ui-type", "--ui"),
        field_name="ui_type",
        group=CLIGroup.UI,
        path="runtime.ui",
    ),
    # =========================================================================
    # WORKERS
    # =========================================================================
    CLIField(
        flags=("--workers-max", "--max-workers"),
        field_name="workers_max",
        group=CLIGroup.WORKERS,
        path="runtime.workers",
    ),
    # =========================================================================
    # ZMQ COMMUNICATION
    # =========================================================================
    CLIField(
        flags="--zmq-host",
        field_name="zmq_host",
        group=CLIGroup.ZMQ_COMMUNICATION,
        description_override=(
            "Host address for internal ZMQ TCP communication between AIPerf services."
        ),
        type_override="str | None",
        default_override=None,
    ),
    CLIField(
        flags="--zmq-ipc-path",
        field_name="zmq_ipc_path",
        group=CLIGroup.ZMQ_COMMUNICATION,
        description_override=(
            "Directory path for ZMQ IPC socket files for local inter-process communication."
        ),
        type_override="Path | None",
        default_override=None,
    ),
    # =========================================================================
    # MULTI-RUN
    # =========================================================================
    CLIField(
        flags="--num-profile-runs",
        field_name="num_profile_runs",
        group=CLIGroup.MULTI_RUN,
        path="multi_run.num_runs",
    ),
    CLIField(
        flags="--profile-run-cooldown-seconds",
        field_name="profile_run_cooldown_seconds",
        group=CLIGroup.MULTI_RUN,
        path="multi_run.cooldown_seconds",
    ),
    CLIField(
        flags="--confidence-level",
        field_name="confidence_level",
        group=CLIGroup.MULTI_RUN,
        path="multi_run.confidence_level",
    ),
    CLIField(
        flags="--profile-run-disable-warmup-after-first",
        field_name="profile_run_disable_warmup_after_first",
        group=CLIGroup.MULTI_RUN,
        path="multi_run.disable_warmup_after_first",
        negative=True,
    ),
    CLIField(
        flags="--set-consistent-seed",
        field_name="set_consistent_seed",
        group=CLIGroup.MULTI_RUN,
        path="multi_run.set_consistent_seed",
        negative=True,
    ),
    # =========================================================================
    # ACCURACY
    # =========================================================================
    CLIField(
        flags="--accuracy-benchmark",
        field_name="accuracy_benchmark",
        group=CLIGroup.ACCURACY,
        path="accuracy.benchmark",
    ),
    CLIField(
        flags="--accuracy-tasks",
        field_name="accuracy_tasks",
        group=CLIGroup.ACCURACY,
        path="accuracy.tasks",
    ),
    CLIField(
        flags="--accuracy-n-shots",
        field_name="accuracy_n_shots",
        group=CLIGroup.ACCURACY,
        path="accuracy.n_shots",
    ),
    CLIField(
        flags="--accuracy-enable-cot",
        field_name="accuracy_enable_cot",
        group=CLIGroup.ACCURACY,
        path="accuracy.enable_cot",
    ),
    CLIField(
        flags="--accuracy-grader",
        field_name="accuracy_grader",
        group=CLIGroup.ACCURACY,
        path="accuracy.grader",
    ),
    CLIField(
        flags="--accuracy-system-prompt",
        field_name="accuracy_system_prompt",
        group=CLIGroup.ACCURACY,
        path="accuracy.system_prompt",
    ),
    CLIField(
        flags="--accuracy-verbose",
        field_name="accuracy_verbose",
        group=CLIGroup.ACCURACY,
        path="accuracy.verbose",
    ),
    # =========================================================================
    # GENERATED (disabled in CLI)
    # =========================================================================
    CLIField(
        flags="--cli-command",
        field_name="cli_command",
        group=CLIGroup.SERVICE,
        parse=False,
        description_override="The CLI command (auto-generated).",
        type_override="str | None",
        default_override=None,
    ),
    CLIField(
        flags="--benchmark-id",
        field_name="benchmark_id",
        group=CLIGroup.SERVICE,
        parse=False,
        description_override="Unique benchmark run identifier (auto-generated UUID).",
        type_override="str | None",
        default_override=None,
    ),
]

# Build lookup indices
CLI_FIELD_BY_NAME: dict[str, CLIField] = {f.field_name: f for f in CLI_FIELDS}
CLI_FIELD_BY_FLAG: dict[str, CLIField] = {}
for _f in CLI_FIELDS:
    flags = (_f.flags,) if isinstance(_f.flags, str) else _f.flags
    for _flag in flags:
        CLI_FIELD_BY_FLAG[_flag] = _f
