# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field, model_validator
from typing_extensions import Self

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.config.config_validators import parse_str_or_list
from aiperf.common.config.groups import Groups
from aiperf.common.enums import ExportLevel, RecordExportFormat


class OutputConfig(BaseConfig):
    """
    A configuration class for defining output related settings.
    """

    _CLI_GROUP = Groups.OUTPUT

    artifact_directory: Annotated[
        Path,
        Field(
            description="Output directory for all benchmark artifacts including metrics (`.csv`, `.json`, `.jsonl`), per-record CSV (`_records.csv`), "
            "raw data (`_raw.jsonl`), GPU telemetry (`_gpu_telemetry.jsonl`), and time-sliced metrics (`_timeslices.csv/json`). "
            "Directory created if it doesn't exist. All output file paths are constructed relative to this directory.",
        ),
        CLIParameter(
            name=(
                "--output-artifact-dir",
                "--artifact-dir",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = OutputDefaults.ARTIFACT_DIRECTORY

    profile_export_prefix: Annotated[
        Path | None,
        Field(
            description="Custom prefix for profile export file names. AIPerf generates multiple output files with different formats: "
            "`.csv` (summary metrics), `.json` (summary with metadata), `.jsonl` (per-record metrics), and `_raw.jsonl` (raw request/response data). "
            "If not specified, defaults to `profile_export_aiperf` for summary files and `profile_export` for detailed files.",
        ),
        CLIParameter(
            name=(
                "--profile-export-prefix",
                "--profile-export-file",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = None

    export_level: Annotated[
        ExportLevel,
        Field(
            description="Controls which output files are generated. "
            "`summary`: Only aggregate metrics files (`.csv`, `.json`). "
            "`records`: Includes per-request metrics (`.jsonl`). "
            "`raw`: Includes raw request/response data (`_raw.jsonl`).",
        ),
        CLIParameter(
            name=("--export-level", "--profile-export-level"),
            group=_CLI_GROUP,
        ),
    ] = OutputDefaults.EXPORT_LEVEL

    record_export_formats: Annotated[
        list[RecordExportFormat],
        Field(
            description="Specify which output formats to generate for per-record metrics. "
            "Multiple formats can be specified (e.g., `--record-export-formats jsonl csv`). "
            "Requires --export-level >= records.",
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--record-export-formats",),
            consume_multiple=True,
            group=_CLI_GROUP,
        ),
    ] = OutputDefaults.RECORD_EXPORT_FORMATS

    slice_duration: Annotated[
        float | None,
        Field(
            description="Duration in seconds for time-sliced metric analysis. When set, AIPerf divides the benchmark timeline into fixed-length "
            "windows and computes metrics separately for each window. This enables analysis of performance trends and variations over time "
            "(e.g., warmup effects, degradation under sustained load).",
        ),
        CLIParameter(
            name=("--slice-duration"),
            group=_CLI_GROUP,
        ),
    ] = OutputDefaults.SLICE_DURATION

    export_http_trace: Annotated[
        bool,
        Field(
            description="Include HTTP trace data (timestamps, chunks, headers, socket info) in profile_export.jsonl. "
            "Computed metrics (http_req_duration, http_req_waiting, etc.) are always included regardless of this setting. "
            "See the HTTP Trace Metrics guide for details on trace data fields.",
        ),
        CLIParameter(
            name="--export-http-trace",
            group=_CLI_GROUP,
        ),
    ] = OutputDefaults.EXPORT_HTTP_TRACE

    export_per_chunk_data: Annotated[
        bool,
        Field(
            description="Include per-chunk list data (e.g., inter_chunk_latency arrays) in per-record exports "
            "(profile_export.jsonl and profile_export_records.csv). These arrays contain one timing value per SSE "
            "chunk and can be very large for long responses. When disabled (default), only scalar aggregate metrics "
            "are included. The aggregate inter_token_latency scalar is always exported regardless of this setting.",
        ),
        CLIParameter(
            name="--export-per-chunk-data",
            group=_CLI_GROUP,
        ),
    ] = OutputDefaults.EXPORT_PER_CHUNK_DATA

    show_trace_timing: Annotated[
        bool,
        Field(
            description="Display HTTP trace timing metrics in the console at the end of the benchmark. "
            "Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, "
            "and total duration following k6 naming conventions.",
        ),
        CLIParameter(
            name="--show-trace-timing",
            group=_CLI_GROUP,
        ),
    ] = OutputDefaults.SHOW_TRACE_TIMING

    _profile_export_csv_file: Path = OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
    _profile_export_json_file: Path = OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
    _profile_export_timeslices_csv_file: Path = (
        OutputDefaults.PROFILE_EXPORT_AIPERF_TIMESLICES_CSV_FILE
    )
    _profile_export_timeslices_json_file: Path = (
        OutputDefaults.PROFILE_EXPORT_AIPERF_TIMESLICES_JSON_FILE
    )
    _profile_export_records_csv_file: Path = (
        OutputDefaults.PROFILE_EXPORT_RECORDS_CSV_FILE
    )
    _profile_export_jsonl_file: Path = OutputDefaults.PROFILE_EXPORT_JSONL_FILE
    _profile_export_raw_jsonl_file: Path = OutputDefaults.PROFILE_EXPORT_RAW_JSONL_FILE
    _profile_export_gpu_telemetry_jsonl_file: Path = (
        OutputDefaults.PROFILE_EXPORT_GPU_TELEMETRY_JSONL_FILE
    )
    _server_metrics_export_jsonl_file: Path = (
        OutputDefaults.SERVER_METRICS_EXPORT_JSONL_FILE
    )
    _server_metrics_export_json_file: Path = (
        OutputDefaults.SERVER_METRICS_EXPORT_JSON_FILE
    )
    _server_metrics_export_csv_file: Path = (
        OutputDefaults.SERVER_METRICS_EXPORT_CSV_FILE
    )
    _server_metrics_export_parquet_file: Path = (
        OutputDefaults.SERVER_METRICS_EXPORT_PARQUET_FILE
    )

    @model_validator(mode="after")
    def validate_record_export_formats(self) -> Self:
        """Validate record_export_formats against export_level."""
        if not self.record_export_formats:
            raise ValueError(
                "--record-export-formats cannot be empty; "
                "specify at least one format (e.g., jsonl, csv)"
            )
        if (
            "record_export_formats" in self.model_fields_set
            and self.export_level == ExportLevel.SUMMARY
        ):
            raise ValueError(
                "--record-export-formats requires --export-level >= records"
            )
        return self

    @model_validator(mode="after")
    def set_export_filenames(self) -> Self:
        """Set export filename variants by stripping suffixes and adding proper ones."""
        if self.profile_export_prefix is None:
            return self

        base_path = self.profile_export_prefix
        base_str = str(base_path)

        # Check complex suffixes first (longest to shortest) to avoid double-suffixing
        # e.g., if user passes "foo_raw.jsonl", we want "foo" not "foo_raw"
        suffixes_to_strip = [
            "_server_metrics.parquet",
            "_server_metrics.jsonl",
            "_server_metrics.json",
            "_server_metrics.csv",
            "_gpu_telemetry.jsonl",
            "_timeslices.csv",
            "_timeslices.json",
            "_records.csv",
            "_raw.jsonl",
            ".parquet",
            ".csv",
            ".json",
            ".jsonl",
        ]
        for suffix in suffixes_to_strip:
            if base_str.endswith(suffix):
                base_str = base_str[: -len(suffix)]
                break

        self._profile_export_csv_file = Path(f"{base_str}.csv")
        self._profile_export_json_file = Path(f"{base_str}.json")
        self._profile_export_timeslices_csv_file = Path(f"{base_str}_timeslices.csv")
        self._profile_export_timeslices_json_file = Path(f"{base_str}_timeslices.json")
        self._profile_export_records_csv_file = Path(f"{base_str}_records.csv")
        self._profile_export_jsonl_file = Path(f"{base_str}.jsonl")
        self._profile_export_raw_jsonl_file = Path(f"{base_str}_raw.jsonl")
        self._profile_export_gpu_telemetry_jsonl_file = Path(
            f"{base_str}_gpu_telemetry.jsonl"
        )
        self._server_metrics_export_jsonl_file = Path(
            f"{base_str}_server_metrics.jsonl"
        )
        self._server_metrics_export_json_file = Path(f"{base_str}_server_metrics.json")
        self._server_metrics_export_csv_file = Path(f"{base_str}_server_metrics.csv")
        self._server_metrics_export_parquet_file = Path(
            f"{base_str}_server_metrics.parquet"
        )
        return self

    @property
    def profile_export_csv_file(self) -> Path:
        return self.artifact_directory / self._profile_export_csv_file

    @property
    def profile_export_json_file(self) -> Path:
        return self.artifact_directory / self._profile_export_json_file

    @property
    def profile_export_timeslices_csv_file(self) -> Path:
        return self.artifact_directory / self._profile_export_timeslices_csv_file

    @property
    def profile_export_timeslices_json_file(self) -> Path:
        return self.artifact_directory / self._profile_export_timeslices_json_file

    @property
    def profile_export_records_csv_file(self) -> Path:
        return self.artifact_directory / self._profile_export_records_csv_file

    @property
    def profile_export_jsonl_file(self) -> Path:
        return self.artifact_directory / self._profile_export_jsonl_file

    @property
    def profile_export_raw_jsonl_file(self) -> Path:
        return self.artifact_directory / self._profile_export_raw_jsonl_file

    @property
    def profile_export_gpu_telemetry_jsonl_file(self) -> Path:
        return self.artifact_directory / self._profile_export_gpu_telemetry_jsonl_file

    @property
    def server_metrics_export_jsonl_file(self) -> Path:
        return self.artifact_directory / self._server_metrics_export_jsonl_file

    @property
    def server_metrics_export_json_file(self) -> Path:
        return self.artifact_directory / self._server_metrics_export_json_file

    @property
    def server_metrics_export_csv_file(self) -> Path:
        return self.artifact_directory / self._server_metrics_export_csv_file

    @property
    def server_metrics_export_parquet_file(self) -> Path:
        return self.artifact_directory / self._server_metrics_export_parquet_file
