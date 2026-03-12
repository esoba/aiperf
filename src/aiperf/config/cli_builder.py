# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamic CLI model generation and AIPerfConfig builder.

This module:
1. Introspects AIPerfConfig sub-models to extract field metadata
2. Generates a flat Pydantic model (CLIModel) for cyclopts at import time
3. Converts parsed CLIModel instances into AIPerfConfig objects

Replaces both cli_config.py (~2600 lines) and converter.py (~800 lines).
"""

from __future__ import annotations

import contextlib
import sys
import types as pytypes
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Union, get_args, get_origin

from cyclopts import Group
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.cli_parameter import CLIParameter, DisableCLI
from aiperf.common.config.config_validators import (
    coerce_value,
    parse_file,
    parse_str_as_numeric_dict,
    parse_str_or_dict_as_tuple_list,
    parse_str_or_list,
    parse_str_or_list_of_positive_values,
    validate_sequence_distribution,
)
from aiperf.config.cli_mapping import _UNSET, CLI_FIELDS, CLIField, CLIGroup

if TYPE_CHECKING:
    from aiperf.config.config import AIPerfConfig

_logger = AIPerfLogger(__name__)

__all__ = [
    "CLIModel",
    "build_aiperf_config",
]


# =============================================================================
# VALIDATOR FUNCTION REGISTRY
# =============================================================================

_VALIDATOR_REGISTRY: dict[str, Any] = {
    "parse_str_or_list": parse_str_or_list,
    "parse_str_or_dict_as_tuple_list": parse_str_or_dict_as_tuple_list,
    "parse_str_as_numeric_dict": parse_str_as_numeric_dict,
    "parse_str_or_list_of_positive_values": parse_str_or_list_of_positive_values,
    "validate_sequence_distribution": validate_sequence_distribution,
    "parse_file": parse_file,
}


# =============================================================================
# TYPE STRING RESOLUTION
# =============================================================================


def _resolve_type_string(type_str: str) -> Any:
    """Resolve a type string like 'float | None' or 'list[str]' to a real type."""
    from aiperf.common.enums import (
        AIPerfLogLevel,
        AudioFormat,
        ConnectionReuseStrategy,
        ExportLevel,
        ImageFormat,
        ModelSelectionStrategy,
        PublicDatasetType,
        ServerMetricsFormat,
        VideoFormat,
        VideoSynthType,
    )
    from aiperf.plugin.enums import (
        ArrivalPattern,
        CustomDatasetType,
        DatasetSamplingStrategy,
        EndpointType,
        TransportType,
        UIType,
        URLSelectionStrategy,
    )

    ns: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "None": type(None),
        "AIPerfLogLevel": AIPerfLogLevel,
        "AudioFormat": AudioFormat,
        "ConnectionReuseStrategy": ConnectionReuseStrategy,
        "ExportLevel": ExportLevel,
        "ImageFormat": ImageFormat,
        "ModelSelectionStrategy": ModelSelectionStrategy,
        "PublicDatasetType": PublicDatasetType,
        "ServerMetricsFormat": ServerMetricsFormat,
        "VideoFormat": VideoFormat,
        "VideoSynthType": VideoSynthType,
        "ArrivalPattern": ArrivalPattern,
        "CustomDatasetType": CustomDatasetType,
        "DatasetSamplingStrategy": DatasetSamplingStrategy,
        "EndpointType": EndpointType,
        "TransportType": TransportType,
        "UIType": UIType,
        "URLSelectionStrategy": URLSelectionStrategy,
    }
    return eval(type_str, ns)  # noqa: S307


def _resolve_default_string(default_str: str) -> Any:
    """Resolve a default value string like 'ArrivalPattern.POISSON'."""
    from aiperf.common.enums import ExportLevel
    from aiperf.plugin.enums import ArrivalPattern

    ns = {
        "ArrivalPattern": ArrivalPattern,
        "ExportLevel": ExportLevel,
    }
    return eval(default_str, ns)  # noqa: S307


# =============================================================================
# INTROSPECTION: Resolve dot-paths to AIPerfConfig field metadata
# =============================================================================

_DICT_KEY_SEGMENTS = frozenset({"main", "profiling", "warmup", "default"})


def _unwrap_type(annotation: Any) -> Any:
    """Unwrap Optional, Annotated, Union to get the core type."""
    origin = get_origin(annotation)

    if origin is not None and hasattr(annotation, "__metadata__"):
        return _unwrap_type(get_args(annotation)[0])

    if origin is Union or isinstance(annotation, pytypes.UnionType):
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]

    return annotation


def _extract_model_types(annotation: Any) -> list[type[BaseModel]]:
    """Extract all BaseModel subclasses from a type annotation."""
    inner = _unwrap_type(annotation)

    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return [inner]

    origin = get_origin(inner)

    if origin is dict:
        _, value_type = get_args(inner)
        return _extract_model_types(value_type)

    if origin is Union or isinstance(inner, pytypes.UnionType):
        models = []
        for arg in get_args(inner):
            unwrapped = _unwrap_type(arg)
            if isinstance(unwrapped, type) and issubclass(unwrapped, BaseModel):
                models.append(unwrapped)
        return models

    return []


def resolve_field(path: str) -> tuple[type[BaseModel], str, FieldInfo]:
    """Resolve a dot-path to (model_class, field_name, field_info).

    Handles dict[str, X], discriminated unions, and nested sub-models.

    Raises:
        ValueError: If the path cannot be resolved.
    """
    from aiperf.config.config import AIPerfConfig

    parts = path.split(".")
    field_name = parts[-1]
    candidates: list[type[BaseModel]] = [AIPerfConfig]

    for part in parts[:-1]:
        if part in _DICT_KEY_SEGMENTS:
            continue

        next_candidates: list[type[BaseModel]] = []
        for candidate in candidates:
            if part in candidate.model_fields:
                finfo = candidate.model_fields[part]
                next_candidates.extend(_extract_model_types(finfo.annotation))

        if not next_candidates:
            names = [c.__name__ for c in candidates]
            raise ValueError(f"Path segment '{part}' not found in {names}")
        candidates = next_candidates

    for candidate in candidates:
        if field_name in candidate.model_fields:
            return candidate, field_name, candidate.model_fields[field_name]

    names = [c.__name__ for c in candidates]
    raise ValueError(f"Field '{field_name}' not found in {names}")


def get_field_description(cli_field: CLIField) -> str:
    """Get description for a CLI field, introspecting AIPerfConfig when possible."""
    if cli_field.description_override:
        return cli_field.description_override

    if cli_field.path is None:
        return f"Set {cli_field.field_name}"

    _, _, field_info = resolve_field(cli_field.path)
    base = field_info.description or f"Set {cli_field.field_name}"

    if cli_field.is_warmup_override:
        return f"Warmup phase: {base} If not set, uses the {cli_field.warmup_of} value."

    return base


def get_field_type_and_default(cli_field: CLIField) -> tuple[Any, Any]:
    """Get (type, default) for a CLI field."""
    if cli_field.type_override is not None:
        field_type = (
            _resolve_type_string(cli_field.type_override)
            if isinstance(cli_field.type_override, str)
            else cli_field.type_override
        )
    elif cli_field.path is not None:
        _, _, field_info = resolve_field(cli_field.path)
        field_type = _unwrap_type(field_info.annotation)
    else:
        field_type = str

    from pydantic_core import PydanticUndefined

    if cli_field.default_override is not _UNSET:
        default = cli_field.default_override
    elif cli_field.path is not None:
        _, _, field_info = resolve_field(cli_field.path)
        default = (
            field_info.default if field_info.default is not PydanticUndefined else None
        )
    else:
        default = None

    if isinstance(default, str) and "." in default and default[0].isupper():
        with contextlib.suppress(Exception):
            default = _resolve_default_string(default)

    return field_type, default


# =============================================================================
# DYNAMIC MODEL GENERATION
# =============================================================================

_CLI_GROUPS: dict[str, Group] = {}


def _get_group(group: CLIGroup) -> Group:
    """Get or create a cyclopts Group for display ordering."""
    if group.value not in _CLI_GROUPS:
        _CLI_GROUPS[group.value] = Group.create_ordered(group.value)
    return _CLI_GROUPS[group.value]


def generate_cli_model() -> type[BaseModel]:
    """Generate a flat Pydantic model for cyclopts from the CLI mapping table.

    Walks CLI_FIELDS, introspects AIPerfConfig for metadata, and calls
    pydantic.create_model() to build a model with CLIParameter annotations.
    """
    field_definitions: dict[str, Any] = {}

    for cli_field in CLI_FIELDS:
        field_type, default = get_field_type_and_default(cli_field)
        description = get_field_description(cli_field)

        metadata: list[Any] = [
            Field(default=default, description=description),
        ]

        for validator_name in cli_field.before_validators:
            fn = _VALIDATOR_REGISTRY[validator_name]
            metadata.append(BeforeValidator(fn))

        if not cli_field.parse:
            metadata.append(DisableCLI())
        else:
            cli_param_kwargs: dict[str, Any] = {
                "name": cli_field.flags,
                "group": _get_group(cli_field.group),
            }
            if not cli_field.show_choices:
                cli_param_kwargs["show_choices"] = False
            if cli_field.negative:
                cli_param_kwargs["negative"] = None
            if cli_field.consume_multiple:
                cli_param_kwargs["consume_multiple"] = True
            metadata.append(CLIParameter(**cli_param_kwargs))

        annotated_type = Annotated[tuple([field_type, *metadata])]

        field_definitions[cli_field.field_name] = (annotated_type, default)

    model = create_model(
        "CLIModel",
        __config__=ConfigDict(extra="forbid"),
        **field_definitions,
    )
    return model


CLIModel: type[BaseModel] = generate_cli_model()


# =============================================================================
# CONFIG BUILDER: CLIModel → AIPerfConfig
# =============================================================================


def build_aiperf_config(cli: BaseModel) -> AIPerfConfig:
    """Build an AIPerfConfig from a parsed CLIModel instance.

    Absorbs timing-mode inference, warmup phase creation, dataset type
    inference, and all other CLI-specific transformation logic.
    """
    from aiperf.common import random_generator as rng
    from aiperf.common.enums import ExportFormat, ExportLevel
    from aiperf.common.exceptions import InvalidStateError
    from aiperf.config.artifacts import (
        ArtifactsConfig,
    )
    from aiperf.config.config import AIPerfConfig
    from aiperf.config.endpoint import EndpointConfig
    from aiperf.config.models import (
        LoggingConfig,
        ModelItem,
        ModelsAdvanced,
        TokenizerConfig,
    )

    # --- Initialize RNG ---
    with contextlib.suppress(InvalidStateError):
        rng.init(cli.random_seed)

    # --- Verbose flag handling ---
    from aiperf.common.enums import AIPerfLogLevel

    log_level = cli.log_level
    if cli.extra_verbose:
        log_level = AIPerfLogLevel.TRACE
    elif cli.verbose:
        log_level = AIPerfLogLevel.DEBUG

    # --- CLI command ---
    args = [coerce_value(x) for x in sys.argv[1:]]
    cli_command = " ".join(
        ["aiperf"]
        + [
            f"'{x}'"
            if isinstance(x, str) and not x.startswith("-") and x != "profile"
            else str(x)
            for x in args
        ]
    )

    # --- Benchmark ID ---
    benchmark_id = str(uuid.uuid4())

    # --- Models ---
    models = ModelsAdvanced(
        strategy=cli.model_selection_strategy,
        items=[ModelItem(name=name) for name in cli.model_names],
    )

    # --- Endpoint ---
    endpoint_kwargs: dict[str, Any] = {}
    if cli.headers:
        endpoint_kwargs["headers"] = dict(cli.headers)
    if cli.extra_inputs:
        endpoint_kwargs["extra"] = dict(cli.extra_inputs)

    endpoint = EndpointConfig(
        urls=cli.urls,
        url_strategy=cli.url_selection_strategy,
        type=cli.endpoint_type,
        path=cli.custom_endpoint,
        api_key=cli.api_key,
        timeout=cli.request_timeout_seconds,
        streaming=cli.streaming,
        transport=cli.transport_type,
        connection_reuse=cli.connection_reuse_strategy,
        use_legacy_max_tokens=cli.use_legacy_max_tokens,
        use_server_token_count=cli.use_server_token_count,
        **endpoint_kwargs,
    )

    # --- Dataset ---
    dataset = _build_dataset(cli)

    # --- Load phases ---
    load = _build_load(cli)

    # --- Artifacts ---
    records: list[str] | bool = False
    if cli.export_level in (ExportLevel.RECORDS, ExportLevel.RAW):
        records = [ExportFormat.JSONL]

    prefix = "profile"
    if cli.profile_export_prefix:
        prefix = Path(cli.profile_export_prefix).stem

    artifacts = ArtifactsConfig(
        dir=cli.artifact_directory,
        prefix=prefix,
        summary=[ExportFormat.JSON],
        records=records,
        raw=cli.export_level == ExportLevel.RAW,
        trace=cli.export_http_trace,
        slice_duration=cli.slice_duration,
        show_trace_timing=cli.show_trace_timing,
        cli_command=cli_command,
        benchmark_id=benchmark_id,
    )

    # --- SLOs ---
    slos = dict(cli.goodput) if cli.goodput else None

    # --- Tokenizer ---
    tokenizer = None
    if cli.tokenizer_name:
        tokenizer = TokenizerConfig(
            name=cli.tokenizer_name,
            revision=cli.tokenizer_revision,
            trust_remote_code=cli.tokenizer_trust_remote_code,
        )

    # --- GPU Telemetry ---
    gpu_telemetry = _build_gpu_telemetry(cli)

    # --- Server Metrics ---
    server_metrics = _build_server_metrics(cli)

    # --- Runtime ---
    runtime = _build_runtime(cli)

    return AIPerfConfig(
        models=models,
        endpoint=endpoint,
        datasets={"main": dataset},
        load=load,
        artifacts=artifacts,
        slos=slos,
        tokenizer=tokenizer,
        gpu_telemetry=gpu_telemetry,
        server_metrics=server_metrics,
        runtime=runtime,
        logging=LoggingConfig(level=log_level),
        random_seed=cli.random_seed,
    )


# =============================================================================
# PRIVATE BUILDER HELPERS
# =============================================================================


def _set_if_not_none(d: dict, key: str, value: Any) -> None:
    """Only add key to dict if value is not None."""
    if value is not None:
        d[key] = value


def _make_mean_stddev(
    mean: float | None,
    stddev: float | None,
    *,
    min_mean: float = 0,
    as_int: bool = False,
) -> Any:
    from aiperf.config.types import MeanStddev

    if mean is None or mean <= min_mean:
        return None
    if stddev is not None and stddev > 0:
        return MeanStddev(mean=mean, stddev=stddev)
    return int(mean) if as_int else mean


def _build_ramp(primary: float | None, fallback: float | None = None) -> Any:
    from aiperf.config.phases import RampConfig

    duration = primary if primary is not None else fallback
    return RampConfig(duration=duration) if duration is not None else None


def _build_dataset(cli: BaseModel) -> Any:
    from aiperf.common.enums import DatasetType
    from aiperf.config.dataset import (
        PublicDataset,
    )

    if cli.public_dataset:
        return PublicDataset(
            type=DatasetType.PUBLIC,
            name=cli.public_dataset,
            entries=cli.request_count or cli.num_sessions,
        )

    if cli.input_file:
        if _needs_augmentation(cli):
            return _build_composed_dataset(cli)
        return _build_file_dataset(cli)

    return _build_synthetic_dataset(cli)


def _build_synthetic_dataset(cli: BaseModel) -> Any:
    from aiperf.common.enums import DatasetType
    from aiperf.config.dataset import SyntheticDataset

    entries = cli.request_count or cli.num_sessions or cli.num_dataset_entries
    return SyntheticDataset(
        type=DatasetType.SYNTHETIC,
        entries=entries,
        random_seed=cli.random_seed,
        prompts=_build_prompts(cli),
        prefix_prompts=_build_prefix_prompts(cli),
        turns=_make_mean_stddev(
            cli.num_turns_mean, cli.num_turns_stddev, min_mean=1, as_int=True
        ),
        turn_delay=_make_mean_stddev(cli.turn_delay_mean, cli.turn_delay_stddev),
        images=_build_images(cli),
        audio=_build_audio(cli),
        video=_build_video(cli),
        rankings=_build_rankings(cli),
    )


def _build_prompts(cli: BaseModel) -> Any:
    from aiperf.config.dataset import PromptConfig
    from aiperf.config.types import MeanStddev, SequenceDistributionEntry

    isl = MeanStddev(mean=cli.isl_mean, stddev=cli.isl_stddev) if cli.isl_mean else None
    osl = MeanStddev(mean=cli.osl_mean, stddev=cli.osl_stddev) if cli.osl_mean else None

    seq_pairs = None
    if cli.sequence_distribution:
        from aiperf.common.models.sequence_distribution import DistributionParser

        distribution = DistributionParser.parse(cli.sequence_distribution)
        seq_pairs = [
            SequenceDistributionEntry(
                isl=MeanStddev(mean=p.input_seq_len, stddev=p.input_seq_len_stddev),
                osl=MeanStddev(mean=p.output_seq_len, stddev=p.output_seq_len_stddev),
                probability=p.probability,
            )
            for p in distribution.pairs
        ]

    return PromptConfig(
        isl=isl,
        osl=osl,
        block_size=cli.isl_block_size,
        batch_size=cli.prompt_batch_size,
        sequence_distribution=seq_pairs,
    )


def _build_prefix_prompts(cli: BaseModel) -> Any:
    from aiperf.config.dataset import PrefixPromptConfig

    has_prefix = (
        cli.num_prefix_prompts > 0
        or cli.prefix_prompt_length > 0
        or cli.shared_system_prompt_length is not None
        or cli.user_context_prompt_length is not None
    )
    if not has_prefix:
        return None

    return PrefixPromptConfig(
        pool_size=cli.num_prefix_prompts or 1,
        length=cli.prefix_prompt_length or 128,
        shared_system_length=cli.shared_system_prompt_length,
        user_context_length=cli.user_context_prompt_length,
    )


def _build_images(cli: BaseModel) -> Any:
    from aiperf.config.dataset import ImageConfig

    if cli.image_batch_size == 0:
        return None
    return ImageConfig(
        batch_size=cli.image_batch_size,
        width=cli.image_width_mean,
        height=cli.image_height_mean,
        format=cli.image_format,
    )


def _build_audio(cli: BaseModel) -> Any:
    from aiperf.config.dataset import AudioConfig

    if cli.audio_batch_size == 0:
        return None
    return AudioConfig(
        batch_size=cli.audio_batch_size,
        length=cli.audio_length_mean,
        format=cli.audio_format,
    )


def _build_video(cli: BaseModel) -> Any:
    from aiperf.config.dataset import VideoConfig

    if cli.video_batch_size == 0:
        return None
    return VideoConfig(
        batch_size=cli.video_batch_size,
        duration=cli.video_duration,
        fps=cli.video_fps,
        width=cli.video_width,
        height=cli.video_height,
        format=cli.video_format,
        codec=cli.video_codec,
        synth_type=cli.video_synth_type,
    )


def _build_rankings(cli: BaseModel) -> Any:
    from aiperf.config.dataset import RankingsConfig
    from aiperf.config.types import MeanStddev

    return RankingsConfig(
        passages=MeanStddev(mean=cli.passages_mean, stddev=cli.passages_stddev),
        passage_tokens=MeanStddev(
            mean=cli.passages_prompt_token_mean, stddev=cli.passages_prompt_token_stddev
        ),
        query_tokens=MeanStddev(
            mean=cli.query_prompt_token_mean, stddev=cli.query_prompt_token_stddev
        ),
    )


def _needs_augmentation(cli: BaseModel) -> bool:
    return (
        cli.num_prefix_prompts > 0
        or cli.prefix_prompt_length > 0
        or cli.shared_system_prompt_length is not None
        or cli.user_context_prompt_length is not None
        or cli.image_batch_size > 0
        or cli.audio_batch_size > 0
        or cli.video_batch_size > 0
        or cli.osl_mean is not None
    )


def _build_file_dataset(cli: BaseModel) -> Any:
    from aiperf.common.enums import DatasetType
    from aiperf.config.dataset import FileDataset, SynthesisConfig

    synthesis = None
    if _has_synthesis(cli):
        synthesis = SynthesisConfig(
            speedup_ratio=cli.synthesis_speedup_ratio,
            prefix_len_multiplier=cli.synthesis_prefix_len_multiplier,
            prefix_root_multiplier=cli.synthesis_prefix_root_multiplier,
            prompt_len_multiplier=cli.synthesis_prompt_len_multiplier,
            max_isl=cli.synthesis_max_isl,
            max_osl=cli.synthesis_max_osl,
        )

    return FileDataset(
        type=DatasetType.FILE,
        path=cli.input_file,
        format=cli.custom_dataset_type,
        sampling=cli.dataset_sampling_strategy,
        synthesis=synthesis,
    )


def _has_synthesis(cli: BaseModel) -> bool:
    return (
        cli.synthesis_speedup_ratio != 1.0
        or cli.synthesis_prefix_len_multiplier != 1.0
        or cli.synthesis_prefix_root_multiplier != 1
        or cli.synthesis_prompt_len_multiplier != 1.0
        or cli.synthesis_max_isl is not None
        or cli.synthesis_max_osl is not None
    )


def _build_composed_dataset(cli: BaseModel) -> Any:
    from aiperf.common.enums import DatasetType
    from aiperf.config.dataset import (
        AugmentConfig,
        AugmentPrefixConfig,
        ComposedDataset,
        FileSourceConfig,
    )

    source = FileSourceConfig(
        type=DatasetType.FILE,
        path=cli.input_file,
        format=cli.custom_dataset_type,
        sampling=cli.dataset_sampling_strategy,
    )

    prefix = None
    if cli.prefix_prompt_length > 0 or cli.num_prefix_prompts > 0:
        prefix = AugmentPrefixConfig(
            length=cli.prefix_prompt_length or 128,
            pool_size=cli.num_prefix_prompts or 1,
        )
    elif cli.shared_system_prompt_length is not None:
        prefix = AugmentPrefixConfig(
            length=cli.shared_system_prompt_length,
            pool_size=1,
        )

    augment = AugmentConfig(
        prefix=prefix,
        osl=_make_mean_stddev(cli.osl_mean, cli.osl_stddev, as_int=True),
        images=_build_images(cli),
        audio=_build_audio(cli),
        video=_build_video(cli),
    )

    return ComposedDataset(
        source=source,
        augment=augment,
        entries=cli.request_count or cli.num_sessions,
        random_seed=cli.random_seed,
    )


def _build_load(cli: BaseModel) -> dict[str, Any]:
    from aiperf.common.enums import CreditPhase
    from aiperf.config.phases import (
        CancellationConfig,
        PhaseConfig,
        PhaseType,
    )

    load: dict[str, Any] = {}

    # --- Warmup phase ---
    warmup = _build_warmup_phase(cli)
    if warmup:
        load[CreditPhase.WARMUP] = warmup

    # --- Main profiling phase ---
    phase_type, rate, smoothness, users = _determine_phase_type(cli)

    cancellation = None
    if cli.request_cancellation_rate:
        cancellation = CancellationConfig(
            rate=cli.request_cancellation_rate,
            delay=cli.request_cancellation_delay or 0.0,
        )

    concurrency = cli.concurrency
    if phase_type == PhaseType.CONCURRENCY and concurrency is None:
        concurrency = 1

    phase_kwargs: dict[str, Any] = {
        "type": phase_type,
    }
    _set_if_not_none(phase_kwargs, "requests", cli.request_count)
    _set_if_not_none(phase_kwargs, "sessions", cli.num_sessions)
    _set_if_not_none(phase_kwargs, "duration", cli.benchmark_duration)
    _set_if_not_none(phase_kwargs, "concurrency", concurrency)
    _set_if_not_none(
        phase_kwargs, "concurrency_ramp", _build_ramp(cli.concurrency_ramp_duration)
    )
    _set_if_not_none(phase_kwargs, "prefill_concurrency", cli.prefill_concurrency)
    _set_if_not_none(
        phase_kwargs, "prefill_ramp", _build_ramp(cli.prefill_concurrency_ramp_duration)
    )
    _set_if_not_none(phase_kwargs, "rate", rate)
    _set_if_not_none(
        phase_kwargs, "rate_ramp", _build_ramp(cli.request_rate_ramp_duration)
    )
    _set_if_not_none(phase_kwargs, "smoothness", smoothness)
    _set_if_not_none(phase_kwargs, "users", users)
    _set_if_not_none(phase_kwargs, "grace_period", cli.benchmark_grace_period)
    _set_if_not_none(phase_kwargs, "cancellation", cancellation)
    _set_if_not_none(phase_kwargs, "auto_offset", cli.fixed_schedule_auto_offset)
    _set_if_not_none(phase_kwargs, "start_offset", cli.fixed_schedule_start_offset)
    _set_if_not_none(phase_kwargs, "end_offset", cli.fixed_schedule_end_offset)

    load[CreditPhase.PROFILING] = PhaseConfig(**phase_kwargs)

    return load


def _determine_phase_type(cli: BaseModel) -> tuple:
    from aiperf.config.phases import PhaseType
    from aiperf.plugin.enums import ArrivalPattern

    if cli.fixed_schedule:
        return PhaseType.FIXED_SCHEDULE, None, None, None

    if cli.user_centric_rate is not None:
        return PhaseType.USER_CENTRIC, cli.user_centric_rate, None, cli.num_users

    if cli.request_rate is not None:
        match cli.arrival_pattern:
            case ArrivalPattern.GAMMA:
                return PhaseType.GAMMA, cli.request_rate, cli.arrival_smoothness, None
            case ArrivalPattern.CONSTANT:
                return PhaseType.CONSTANT, cli.request_rate, None, None
            case _:
                return PhaseType.POISSON, cli.request_rate, None, None

    return PhaseType.CONCURRENCY, None, None, None


def _build_warmup_phase(cli: BaseModel) -> Any:
    from aiperf.config.phases import PhaseConfig, PhaseType
    from aiperf.plugin.enums import ArrivalPattern

    has_warmup = (
        cli.warmup_request_count is not None
        or cli.warmup_num_sessions is not None
        or cli.warmup_duration is not None
    )
    if not has_warmup:
        return None

    requests = cli.warmup_request_count
    sessions = cli.warmup_num_sessions if not requests else None
    duration = cli.warmup_duration if not requests and not sessions else None

    warmup_rate = cli.warmup_request_rate or cli.request_rate
    warmup_pattern = cli.warmup_arrival_pattern or cli.arrival_pattern
    warmup_concurrency = cli.warmup_concurrency or cli.concurrency or 1

    if warmup_rate is not None:
        match warmup_pattern:
            case ArrivalPattern.GAMMA:
                phase_type = PhaseType.GAMMA
                rate = warmup_rate
                smoothness = cli.arrival_smoothness
            case ArrivalPattern.CONSTANT:
                phase_type = PhaseType.CONSTANT
                rate = warmup_rate
                smoothness = None
            case _:
                phase_type = PhaseType.POISSON
                rate = warmup_rate
                smoothness = None
    else:
        phase_type = PhaseType.CONCURRENCY
        rate = None
        smoothness = None

    warmup_kwargs: dict[str, Any] = {
        "type": phase_type,
        "exclude": True,
    }
    _set_if_not_none(warmup_kwargs, "requests", requests)
    _set_if_not_none(warmup_kwargs, "sessions", sessions)
    _set_if_not_none(warmup_kwargs, "duration", duration)
    _set_if_not_none(warmup_kwargs, "concurrency", warmup_concurrency)
    _set_if_not_none(
        warmup_kwargs,
        "concurrency_ramp",
        _build_ramp(
            cli.warmup_concurrency_ramp_duration, cli.concurrency_ramp_duration
        ),
    )
    _set_if_not_none(
        warmup_kwargs, "prefill_concurrency", cli.warmup_prefill_concurrency
    )
    _set_if_not_none(
        warmup_kwargs,
        "prefill_ramp",
        _build_ramp(
            cli.warmup_prefill_concurrency_ramp_duration,
            cli.prefill_concurrency_ramp_duration,
        ),
    )
    _set_if_not_none(warmup_kwargs, "rate", rate)
    _set_if_not_none(
        warmup_kwargs,
        "rate_ramp",
        _build_ramp(
            cli.warmup_request_rate_ramp_duration, cli.request_rate_ramp_duration
        ),
    )
    _set_if_not_none(warmup_kwargs, "smoothness", smoothness)
    _set_if_not_none(warmup_kwargs, "grace_period", cli.warmup_grace_period)

    return PhaseConfig(**warmup_kwargs)


def _build_gpu_telemetry(cli: BaseModel) -> Any:
    from aiperf.config.artifacts import GpuTelemetryConfig

    if cli.no_gpu_telemetry:
        return GpuTelemetryConfig(enabled=False)

    if not cli.gpu_telemetry:
        return GpuTelemetryConfig(enabled=True)

    urls = []
    metrics_file = None
    for item in cli.gpu_telemetry:
        if item.endswith(".csv"):
            metrics_file = Path(item)
        elif item.startswith("http") or ":" in item:
            urls.append(item if item.startswith("http") else f"http://{item}")
        # 'dashboard' mode handled at runtime

    return GpuTelemetryConfig(
        enabled=True,
        urls=urls,
        metrics_file=metrics_file,
    )


def _build_server_metrics(cli: BaseModel) -> Any:
    from aiperf.common.metric_utils import normalize_metrics_endpoint_url
    from aiperf.config.artifacts import ServerMetricsConfig

    if cli.no_server_metrics:
        return ServerMetricsConfig(enabled=False)

    urls = []
    for item in cli.server_metrics or []:
        if item.startswith("http") or ":" in item:
            normalized = item if item.startswith("http") else f"http://{item}"
            normalized = normalize_metrics_endpoint_url(normalized)
            urls.append(normalized)

    return ServerMetricsConfig(
        enabled=True,
        urls=urls,
        formats=list(cli.server_metrics_formats) if cli.server_metrics_formats else [],
    )


def _build_runtime(cli: BaseModel) -> Any:
    from aiperf.common.enums import CommunicationType
    from aiperf.config.models import (
        IpcCommunicationConfig,
        RuntimeConfig,
        TcpCommunicationConfig,
    )
    from aiperf.plugin.enums import UIType

    ui_type = cli.ui_type
    if cli.extra_verbose or cli.verbose:
        ui_type = UIType.SIMPLE

    communication = None
    if cli.zmq_ipc_path is not None:
        communication = IpcCommunicationConfig(
            type=CommunicationType.IPC,
            path=str(cli.zmq_ipc_path),
        )
    elif cli.zmq_host is not None:
        communication = TcpCommunicationConfig(
            type=CommunicationType.TCP,
            host=cli.zmq_host,
        )

    return RuntimeConfig(
        ui=ui_type,
        workers=cli.workers_max,
        record_processors=cli.record_processor_service_count,
        communication=communication,
    )
