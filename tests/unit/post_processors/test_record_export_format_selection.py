# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for record export format selection feature."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from aiperf.common.config import (
    EndpointConfig,
    OutputConfig,
    ServiceConfig,
    UserConfig,
)
from aiperf.common.enums import ExportLevel, RecordExportFormat
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.plugin.enums import EndpointType
from aiperf.post_processors.record_export_csv_processor import (
    RecordExportCSVProcessor,
)
from aiperf.post_processors.record_export_results_processor import (
    RecordExportResultsProcessor,
)


@pytest.fixture
def tmp_artifact_dir(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


@pytest.fixture
def service_config() -> ServiceConfig:
    return ServiceConfig()


def _make_user_config(
    tmp_artifact_dir: Path,
    export_level: ExportLevel = ExportLevel.RECORDS,
    record_export_formats: list[RecordExportFormat] | None = None,
) -> UserConfig:
    """Helper to create a UserConfig with specified export settings."""
    output_kwargs: dict = {
        "artifact_directory": tmp_artifact_dir,
        "export_level": export_level,
    }
    if record_export_formats is not None:
        output_kwargs["record_export_formats"] = record_export_formats
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
        ),
        output=OutputConfig(**output_kwargs),
    )


class TestRecordExportFormatConfig:
    """Test record export format configuration."""

    def test_default_is_jsonl_only(self, tmp_artifact_dir: Path):
        config = _make_user_config(tmp_artifact_dir)
        assert config.output.record_export_formats == [RecordExportFormat.JSONL]

    def test_single_format_csv(self, tmp_artifact_dir: Path):
        config = _make_user_config(
            tmp_artifact_dir, record_export_formats=[RecordExportFormat.CSV]
        )
        assert RecordExportFormat.CSV in config.output.record_export_formats
        assert RecordExportFormat.JSONL not in config.output.record_export_formats

    def test_multiple_formats(self, tmp_artifact_dir: Path):
        config = _make_user_config(
            tmp_artifact_dir,
            record_export_formats=[RecordExportFormat.JSONL, RecordExportFormat.CSV],
        )
        assert RecordExportFormat.JSONL in config.output.record_export_formats
        assert RecordExportFormat.CSV in config.output.record_export_formats

    def test_case_insensitive_string_input(self, tmp_artifact_dir: Path):
        config = _make_user_config(
            tmp_artifact_dir, record_export_formats=["CSV", "Jsonl"]
        )
        assert RecordExportFormat.CSV in config.output.record_export_formats
        assert RecordExportFormat.JSONL in config.output.record_export_formats

    def test_invalid_format_raises_error(self, tmp_artifact_dir: Path):
        with pytest.raises(ValidationError, match="Input should be 'csv' or 'jsonl'"):
            _make_user_config(tmp_artifact_dir, record_export_formats=["parquet"])

    def test_empty_list_raises_error(self, tmp_artifact_dir: Path):
        with pytest.raises(ValidationError, match="cannot be empty"):
            _make_user_config(tmp_artifact_dir, record_export_formats=[])

    def test_formats_with_summary_export_level_raises_error(
        self, tmp_artifact_dir: Path
    ):
        with pytest.raises(ValidationError, match="requires --export-level >= records"):
            _make_user_config(
                tmp_artifact_dir,
                export_level=ExportLevel.SUMMARY,
                record_export_formats=[RecordExportFormat.JSONL],
            )

    def test_default_formats_with_summary_export_level_ok(self, tmp_artifact_dir: Path):
        """Default formats + summary is fine (user did not explicitly set formats)."""
        config = _make_user_config(tmp_artifact_dir, export_level=ExportLevel.SUMMARY)
        assert config.output.export_level == ExportLevel.SUMMARY


class TestCSVProcessorFormatGating:
    """Test CSV processor respects format selection."""

    def test_csv_enabled_when_format_selected(
        self, tmp_artifact_dir: Path, service_config: ServiceConfig
    ):
        config = _make_user_config(
            tmp_artifact_dir,
            record_export_formats=[RecordExportFormat.CSV],
        )
        processor = RecordExportCSVProcessor(
            service_id="test",
            service_config=service_config,
            user_config=config,
        )
        assert processor is not None

    def test_csv_disabled_when_format_not_selected(
        self, tmp_artifact_dir: Path, service_config: ServiceConfig
    ):
        config = _make_user_config(
            tmp_artifact_dir,
            record_export_formats=[RecordExportFormat.JSONL],
        )
        with pytest.raises(PostProcessorDisabled, match="format not selected"):
            RecordExportCSVProcessor(
                service_id="test",
                service_config=service_config,
                user_config=config,
            )

    def test_csv_disabled_by_default(
        self, tmp_artifact_dir: Path, service_config: ServiceConfig
    ):
        """Default config (JSONL only) disables CSV processor."""
        config = _make_user_config(tmp_artifact_dir)
        with pytest.raises(PostProcessorDisabled, match="format not selected"):
            RecordExportCSVProcessor(
                service_id="test",
                service_config=service_config,
                user_config=config,
            )

    def test_csv_export_level_checked_before_format(
        self, tmp_artifact_dir: Path, service_config: ServiceConfig
    ):
        """export_level=summary produces the export-level error, not format error."""
        config = _make_user_config(tmp_artifact_dir, export_level=ExportLevel.SUMMARY)
        with pytest.raises(PostProcessorDisabled, match="export level"):
            RecordExportCSVProcessor(
                service_id="test",
                service_config=service_config,
                user_config=config,
            )


class TestJSONLProcessorFormatGating:
    """Test JSONL processor respects format selection."""

    def test_jsonl_enabled_when_format_selected(
        self, tmp_artifact_dir: Path, service_config: ServiceConfig
    ):
        config = _make_user_config(
            tmp_artifact_dir,
            record_export_formats=[RecordExportFormat.JSONL],
        )
        processor = RecordExportResultsProcessor(
            service_id="test",
            service_config=service_config,
            user_config=config,
        )
        assert processor is not None

    def test_jsonl_disabled_when_format_not_selected(
        self, tmp_artifact_dir: Path, service_config: ServiceConfig
    ):
        config = _make_user_config(
            tmp_artifact_dir,
            record_export_formats=[RecordExportFormat.CSV],
        )
        with pytest.raises(PostProcessorDisabled, match="format not selected"):
            RecordExportResultsProcessor(
                service_id="test",
                service_config=service_config,
                user_config=config,
            )

    def test_jsonl_enabled_by_default(
        self, tmp_artifact_dir: Path, service_config: ServiceConfig
    ):
        """Default config enables JSONL processor."""
        config = _make_user_config(tmp_artifact_dir)
        processor = RecordExportResultsProcessor(
            service_id="test",
            service_config=service_config,
            user_config=config,
        )
        assert processor is not None

    def test_jsonl_export_level_checked_before_format(
        self, tmp_artifact_dir: Path, service_config: ServiceConfig
    ):
        """export_level=summary produces the export-level error, not format error."""
        config = _make_user_config(tmp_artifact_dir, export_level=ExportLevel.SUMMARY)
        with pytest.raises(PostProcessorDisabled, match="export level"):
            RecordExportResultsProcessor(
                service_id="test",
                service_config=service_config,
                user_config=config,
            )


class TestBothProcessorsFormatGating:
    """Test both processors with all formats selected."""

    def test_both_enabled_when_all_formats_selected(
        self, tmp_artifact_dir: Path, service_config: ServiceConfig
    ):
        config = _make_user_config(
            tmp_artifact_dir,
            record_export_formats=[RecordExportFormat.JSONL, RecordExportFormat.CSV],
        )
        csv_proc = RecordExportCSVProcessor(
            service_id="test-csv",
            service_config=service_config,
            user_config=config,
        )
        jsonl_proc = RecordExportResultsProcessor(
            service_id="test-jsonl",
            service_config=service_config,
            user_config=config,
        )
        assert csv_proc is not None
        assert jsonl_proc is not None
