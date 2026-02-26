# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

from pydantic import BeforeValidator, Field, model_validator
from typing_extensions import Self

from aiperf.common import random_generator as rng
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.audio_config import AudioConfig
from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import InputDefaults
from aiperf.common.config.config_validators import (
    parse_file,
    parse_str_as_numeric_dict,
    parse_str_or_dict_as_tuple_list,
)
from aiperf.common.config.conversation_config import ConversationConfig
from aiperf.common.config.groups import Groups
from aiperf.common.config.image_config import ImageConfig
from aiperf.common.config.prompt_config import PromptConfig
from aiperf.common.config.rankings_config import RankingsConfig
from aiperf.common.config.synthesis_config import SynthesisConfig
from aiperf.common.config.video_config import VideoConfig
from aiperf.common.enums import PublicDatasetType
from aiperf.common.exceptions import InvalidStateError, MetricTypeError
from aiperf.plugin.enums import (
    CustomDatasetType,
    DatasetSamplingStrategy,
)

_logger = AIPerfLogger(__name__)


class InputConfig(BaseConfig):
    """
    A configuration class for defining input related settings.
    """

    _CLI_GROUP = Groups.INPUT

    @model_validator(mode="before")
    @classmethod
    def initialize_rng(cls, data: dict) -> dict:
        """Initialize RNG with random seed before any field validation."""
        if isinstance(data, dict):
            seed = data.get("random_seed")
            # Initialize RNG if not already initialized
            try:
                rng.init(seed)
            except InvalidStateError:
                # Already initialized, that's fine - skip reinitialization
                _logger.debug("RNG already initialized, skipping reinitialization")
        return data

    @model_validator(mode="after")
    def validate_fixed_schedule(self) -> Self:
        """Validate the fixed schedule configuration."""
        if self.fixed_schedule and self.file is None:
            raise ValueError("Fixed schedule requires a file to be provided")
        return self

    @model_validator(mode="after")
    def validate_fixed_schedule_start_offset(self) -> Self:
        """Validate the fixed schedule start offset configuration."""
        if (
            self.fixed_schedule_start_offset is not None
            and self.fixed_schedule_auto_offset
        ):
            raise ValueError(
                "The --fixed-schedule-start-offset and --fixed-schedule-auto-offset options cannot be used together"
            )
        return self

    @model_validator(mode="after")
    def validate_fixed_schedule_start_and_end_offset(self) -> Self:
        """Validate the fixed schedule start and end offset configuration."""
        if (
            self.fixed_schedule_start_offset is not None
            and self.fixed_schedule_end_offset is not None
            and self.fixed_schedule_start_offset > self.fixed_schedule_end_offset
        ):
            raise ValueError(
                "The --fixed-schedule-start-offset must be less than or equal to the --fixed-schedule-end-offset"
            )
        return self

    @model_validator(mode="after")
    def validate_dataset_type(self) -> Self:
        """Validate the different dataset type configuration."""
        if self.public_dataset is not None and self.custom_dataset_type is not None:
            raise ValueError(
                "The --public-dataset and --custom-dataset-type options cannot be set together"
            )
        return self

    @model_validator(mode="after")
    def validate_custom_dataset_file(self) -> Self:
        """Validate that custom dataset type has a file."""
        if self.custom_dataset_type is not None and self.file is None:
            raise ValueError("Custom dataset type requires --input-file to be provided")
        return self

    @model_validator(mode="after")
    def validate_synthesis_requires_mooncake_trace(self) -> Self:
        """Validate that synthesis options require mooncake_trace dataset type.

        Only validates when custom_dataset_type is explicitly set to a non-mooncake
        type. If custom_dataset_type is None (auto-detect), we allow synthesis
        options and defer validation to runtime when the actual type is determined.
        """
        if (
            (
                self.synthesis.should_synthesize()
                or self.synthesis.max_isl is not None
                or self.synthesis.max_osl is not None
            )
            and self.custom_dataset_type is not None
            and self.custom_dataset_type
            not in (CustomDatasetType.MOONCAKE_TRACE, CustomDatasetType.CODING_TRACE)
        ):
            raise ValueError(
                "Synthesis options (--synthesis-speedup-ratio, --synthesis-prefix-len-multiplier, "
                "--synthesis-prefix-root-multiplier, --synthesis-prompt-len-multiplier, "
                "--synthesis-max-isl, --synthesis-max-osl) "
                "require --custom-dataset-type mooncake_trace"
            )
        return self

    @model_validator(mode="after")
    def validate_goodput(self) -> Self:
        """
        Validate that all keys provided to --goodput are known metric tags.
        Runs after the model is constructed so we can inspect self.goodput directly.
        """
        if self.goodput:
            from aiperf.common.enums import MetricType
            from aiperf.metrics.metric_registry import MetricRegistry

            for tag in self.goodput:
                try:
                    metric_cls = MetricRegistry.get_class(tag)
                except MetricTypeError as e:
                    raise ValueError(f"Unknown metric tag in --goodput: {tag}") from e
                if metric_cls.type == MetricType.DERIVED:
                    raise ValueError(
                        f"Metric '{tag}' is a Derived metric and cannot be used for --goodput. "
                        "Use a per-record metric instead (e.g., 'inter_token_latency', 'time_to_first_token')."
                    )

        return self

    extra: Annotated[
        Any,
        Field(
            description="Additional input parameters to include in every API request payload. Specify as `key:value` pairs "
            "(e.g., `--extra-inputs temperature:0.7 top_p:0.9`) or as JSON string (e.g., `'{\"temperature\": 0.7}'`). "
            "These parameters are merged with request-specific inputs and sent directly to the endpoint API.",
        ),
        CLIParameter(
            name=(
                "--extra-inputs",  # GenAI-Perf
            ),
            consume_multiple=True,
            group=_CLI_GROUP,
        ),
        BeforeValidator(parse_str_or_dict_as_tuple_list),
    ] = InputDefaults.EXTRA

    headers: Annotated[
        Any,
        Field(
            description="Custom HTTP headers to include with every request. Specify as `Header:Value` pairs "
            "(e.g., `--header X-Custom-Header:value`) or as JSON string. Can be specified multiple times. "
            "Useful for custom authentication, tracking, or API-specific requirements. Combined with auto-generated headers "
            "(e.g., `Authorization` from `--api-key`).",
        ),
        BeforeValidator(parse_str_or_dict_as_tuple_list),
        CLIParameter(
            name=(
                "--header",  # GenAI-Perf
                "-H",  # GenAI-Perf
            ),
            consume_multiple=True,
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.HEADERS

    file: Annotated[
        Any,
        Field(
            description="Path to file or directory containing benchmark dataset. Required when using `--custom-dataset-type`. "
            "Supported formats depend on dataset type: JSONL for `single_turn`/`multi_turn`, JSONL trace files for `mooncake_trace`, "
            "directories for `random_pool`. File is parsed according to `--custom-dataset-type` specification.",
        ),
        BeforeValidator(parse_file),
        CLIParameter(
            name=(
                "--input-file",  # GenAI-Perf,
            ),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.FILE

    fixed_schedule: Annotated[
        bool,
        Field(
            description="Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays "
            "the exact timing pattern from the dataset. This mode is automatically enabled for `mooncake_trace` datasets."
        ),
        CLIParameter(
            name=(
                "--fixed-schedule",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.FIXED_SCHEDULE

    fixed_schedule_auto_offset: Annotated[
        bool,
        Field(
            description="Automatically normalize timestamps in fixed schedule by shifting all timestamps so the first timestamp becomes 0. "
            "When enabled, benchmark starts immediately with the timing pattern preserved. When disabled, timestamps are used as absolute "
            "offsets from benchmark start. Mutually exclusive with `--fixed-schedule-start-offset`.",
        ),
        CLIParameter(
            name=("--fixed-schedule-auto-offset",),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.FIXED_SCHEDULE_AUTO_OFFSET

    fixed_schedule_start_offset: Annotated[
        int | None,
        Field(
            ge=0,
            description="Start offset in milliseconds for fixed schedule replay. Skips all requests before this timestamp, allowing "
            "benchmark to start from a specific point in the trace. Requests at exactly the start offset are included. "
            "Useful for analyzing specific time windows. Mutually exclusive with `--fixed-schedule-auto-offset`. "
            "Must be ≤ `--fixed-schedule-end-offset` if both specified.",
        ),
        CLIParameter(
            name=("--fixed-schedule-start-offset",),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.FIXED_SCHEDULE_START_OFFSET

    fixed_schedule_end_offset: Annotated[
        int | None,
        Field(
            ge=0,
            description="End offset in milliseconds for fixed schedule replay. Stops issuing requests after this timestamp, allowing "
            "benchmark of specific trace subsets. Requests at exactly the end offset are included. Defaults to last timestamp in dataset. "
            "Must be ≥ `--fixed-schedule-start-offset` if both specified.",
        ),
        CLIParameter(
            name=("--fixed-schedule-end-offset",),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.FIXED_SCHEDULE_END_OFFSET

    public_dataset: Annotated[
        PublicDatasetType | None,
        Field(
            description="Pre-configured public dataset to download and use for benchmarking (e.g., `sharegpt`). "
            "AIPerf automatically downloads and parses these datasets. Mutually exclusive with `--custom-dataset-type`. "
            "See `PublicDatasetType` enum for available datasets.",
        ),
        CLIParameter(
            name=("--public-dataset"),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.PUBLIC_DATASET

    custom_dataset_type: Annotated[
        CustomDatasetType | None,
        Field(
            description="Format specification for custom dataset provided via `--input-file`. Determines parsing logic and expected file structure. "
            "Options: `single_turn` (JSONL with single exchanges), `multi_turn` (JSONL with conversation history), "
            "`mooncake_trace` (timestamped trace files), `random_pool` (directory of reusable prompts). "
            "Requires `--input-file`. Mutually exclusive with `--public-dataset`.",
        ),
        CLIParameter(
            name=("--custom-dataset-type"),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.CUSTOM_DATASET_TYPE

    dataset_sampling_strategy: Annotated[
        DatasetSamplingStrategy | None,
        Field(
            description="Strategy for selecting entries from dataset during benchmarking. "
            "`sequential`: Iterate through dataset in order, wrapping to start after end. "
            "`random`: Randomly sample with replacement (entries may repeat before all are used). "
            "`shuffle`: Shuffle dataset and iterate without replacement, re-shuffling after exhaustion. "
            "Default behavior depends on dataset type (e.g., `sequential` for traces, `shuffle` for synthetic).",
        ),
        CLIParameter(
            name=("--dataset-sampling-strategy",),
            group=_CLI_GROUP,
        ),
    ] = None

    random_seed: Annotated[
        int | None,
        Field(
            description="Random seed for deterministic data generation. When set, makes synthetic prompts, sampling, delays, and other "
            "random operations reproducible across runs. Essential for A/B testing and debugging. Uses system entropy if not specified. "
            "Initialized globally at config creation.",
        ),
        CLIParameter(
            name=(
                "--random-seed",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.RANDOM_SEED

    goodput: Annotated[
        Any | None,
        Field(
            default=None,
            description="Specify service level objectives (SLOs) for goodput as space-separated "
            "'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the "
            "metric's display unit (falls back to its base unit if no display unit is defined). "
            "Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), "
            "`output_token_throughput_per_user:600` (tokens/s).\n"
            "Only metrics applicable to the current endpoint/config are considered. "
            "For more context on the definition of goodput, "
            "refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
            "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
        ),
        BeforeValidator(parse_str_as_numeric_dict),
        CLIParameter(
            name=("--goodput",),
            group=_CLI_GROUP,
        ),
    ] = InputDefaults.GOODPUT

    adaptive_scale: Annotated[
        bool,
        Field(
            description="Enable adaptive user scaling based on TTFT headroom. Automatically scales up the number of "
            "concurrent users while TTFT remains below the threshold. Designed for coding trace replay workloads."
        ),
        CLIParameter(
            name=("--adaptive-scale",),
            group=_CLI_GROUP,
        ),
    ] = False

    adaptive_scale_start_users: Annotated[
        int,
        Field(
            ge=1,
            description="Initial number of concurrent users for adaptive scale mode.",
        ),
        CLIParameter(
            name=("--adaptive-scale-start-users",),
            group=_CLI_GROUP,
        ),
    ] = 1

    adaptive_scale_max_users: Annotated[
        int | None,
        Field(
            ge=1,
            description="Maximum number of concurrent users for adaptive scale mode. "
            "Default is 50. Set to None to limit only by available conversations.",
        ),
        CLIParameter(
            name=("--adaptive-scale-max-users",),
            group=_CLI_GROUP,
        ),
    ] = 50

    adaptive_scale_max_ttft: Annotated[
        float,
        Field(
            gt=0,
            description="Maximum TTFT threshold in seconds for adaptive scale mode. "
            "When the measured TTFT metric exceeds this value, user scaling stops.",
        ),
        CLIParameter(
            name=("--adaptive-scale-max-ttft",),
            group=_CLI_GROUP,
        ),
    ] = 2.0

    adaptive_scale_ttft_metric: Annotated[
        str,
        Field(
            description="TTFT metric to use for scaling decisions: 'p95', 'avg', or 'max'.",
        ),
        CLIParameter(
            name=("--adaptive-scale-ttft-metric",),
            group=_CLI_GROUP,
        ),
    ] = "p95"

    adaptive_scale_assessment_period: Annotated[
        float,
        Field(
            gt=0,
            description="Period in seconds between scaling assessments.",
        ),
        CLIParameter(
            name=("--adaptive-scale-assessment-period",),
            group=_CLI_GROUP,
        ),
    ] = 30.0

    adaptive_scale_max_delay: Annotated[
        float | None,
        Field(
            ge=0,
            description="Maximum inter-request delay in seconds. Delays from traces exceeding "
            "this value are clamped. Default is 60.0s. Set to None for unclamped trace delays.",
        ),
        CLIParameter(
            name=("--adaptive-scale-max-delay",),
            group=_CLI_GROUP,
        ),
    ] = 60.0

    adaptive_scale_time_scale: Annotated[
        float,
        Field(
            gt=0,
            description="Time scale factor applied to trace delays. Values < 1.0 compress "
            "delays (faster replay), > 1.0 stretch them.",
        ),
        CLIParameter(
            name=("--adaptive-scale-time-scale",),
            group=_CLI_GROUP,
        ),
    ] = 1.0

    adaptive_scale_recycle: Annotated[
        bool,
        Field(
            description="Recycle completed sessions by sampling new conversations. "
            "When disabled, each conversation is replayed at most once.",
        ),
        CLIParameter(
            name=("--adaptive-scale-recycle",),
            group=_CLI_GROUP,
        ),
    ] = False

    adaptive_scale_stagger_ms: Annotated[
        float,
        Field(
            ge=0,
            description="Delay in milliseconds between launching new users during adaptive scaling. "
            "Lower values ramp up faster and collect more data per assessment period.",
        ),
        CLIParameter(
            name=("--adaptive-scale-stagger-ms",),
            group=_CLI_GROUP,
        ),
    ] = 50.0

    adaptive_scale_formula: Annotated[
        str,
        Field(
            description="Scaling formula for computing users to add per assessment period. "
            "'conservative': max(1, active * headroom * 0.5) - scales proportional to current users. "
            "'aggressive': max(2, 2 + headroom_pct / 10) - fast ramp regardless of current users. "
            "'linear': max(1, headroom_pct / 5) - linear ramp based on headroom percentage.",
        ),
        CLIParameter(
            name=("--adaptive-scale-formula",),
            group=_CLI_GROUP,
        ),
    ] = "conservative"

    warm_prefix_pct: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Percentage of max(tool_tokens + system_tokens) to use as a warm prefix "
            "for KV cache pre-fill in coding trace replay. Set to 0 to disable.",
        ),
        CLIParameter(
            name=("--warm-prefix-pct",),
            group=_CLI_GROUP,
        ),
    ] = 0.5

    audio: AudioConfig = AudioConfig()
    image: ImageConfig = ImageConfig()
    video: VideoConfig = VideoConfig()
    prompt: PromptConfig = PromptConfig()
    rankings: RankingsConfig = RankingsConfig()
    synthesis: SynthesisConfig = SynthesisConfig()
    conversation: ConversationConfig = ConversationConfig()
