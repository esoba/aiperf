# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the writer module."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.claude_code_gen.models import SessionDistributionConfig
from aiperf.dataset.claude_code_gen.session_synthesizer import SessionSynthesizer
from aiperf.dataset.claude_code_gen.writer import compute_quality_report, write_dataset
from aiperf.dataset.loader.models import MooncakeTrace


@pytest.fixture
def sessions(coding_config: SessionDistributionConfig):
    synth = SessionSynthesizer(coding_config, seed=42)
    return synth.synthesize_sessions(10)


class TestWriteDataset:
    def test_creates_three_files(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        run_dir = tmp_path / "run"
        jsonl_path, manifest_path, quality_path = write_dataset(
            sessions, run_dir, coding_config, seed=42, config_name="default"
        )
        assert jsonl_path.exists()
        assert manifest_path.exists()
        assert quality_path.exists()
        assert jsonl_path.name == "dataset.jsonl"
        assert manifest_path.name == "manifest.json"
        assert quality_path.name == "quality.json"

    def test_jsonl_rows_are_mooncake_compatible(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, coding_config, seed=42)
        jsonl = run_dir / "dataset.jsonl"
        with jsonl.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = orjson.loads(line)
                MooncakeTrace(**data)

    def test_manifest_contains_seed_and_config_name(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        run_dir = tmp_path / "run"
        _, manifest_path, _ = write_dataset(
            sessions, run_dir, coding_config, seed=42, config_name="default"
        )
        manifest = orjson.loads(manifest_path.read_bytes())
        assert manifest["seed"] == 42
        assert manifest["config_name"] == "default"
        assert manifest["num_sessions"] == 10

    def test_first_turn_has_no_delay_field(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, coding_config, seed=42)
        jsonl = run_dir / "dataset.jsonl"
        with jsonl.open("rb") as f:
            first_line = orjson.loads(f.readline())
            assert "delay" not in first_line

    def test_session_ids_present_in_output(
        self, tmp_path: Path, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, coding_config, seed=42)
        jsonl = run_dir / "dataset.jsonl"
        session_ids = set()
        with jsonl.open("rb") as f:
            for line in f:
                data = orjson.loads(line.strip())
                session_ids.add(data["session_id"])
        assert len(session_ids) == 10


class TestQualityReport:
    def test_report_has_expected_metrics(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        assert "initial_context" in report.observed_vs_target
        assert "generation_length" in report.observed_vs_target
        assert "new_tokens_per_turn" in report.observed_vs_target
        assert "inter_turn_delay_ms" in report.observed_vs_target
        assert "turns_per_session" in report.observed_vs_target

    def test_report_observed_has_percentile_stats(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        ic = report.observed_vs_target["initial_context"]
        assert ic.observed.count == 10
        assert ic.observed.mean > 0
        assert ic.observed.median > 0
        assert ic.observed.p05 <= ic.observed.p25 <= ic.observed.median
        assert ic.observed.median <= ic.observed.p75 <= ic.observed.p95

    def test_report_has_config_summary(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        assert report.config_summary["system_prompt_tokens"] == 8000
        assert report.config_summary["initial_context_mean"] == 50000
        assert report.config_summary["max_prompt_tokens"] == 200000

    def test_report_has_session_end_stats(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        stats = report.session_end_stats
        assert stats.total_sessions == 10
        assert stats.forced_retires + stats.probabilistic_resets == 10
        assert stats.retire_fraction + stats.reset_fraction == pytest.approx(1.0)
        assert stats.final_context_utilization.count == 10

    def test_report_session_stats_is_percentile_stats(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        assert report.session_stats.count == 10
        assert report.session_stats.mean > 0

    def test_report_pct_error_mean_is_non_negative(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        for metric in report.observed_vs_target.values():
            if metric.pct_error_mean is not None:
                assert metric.pct_error_mean >= 0

    def test_report_pct_error_median_is_non_negative(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        for metric in report.observed_vs_target.values():
            if metric.pct_error_median is not None:
                assert metric.pct_error_median >= 0

    def test_report_target_values_from_config(
        self, sessions, coding_config: SessionDistributionConfig
    ) -> None:
        report = compute_quality_report(sessions, coding_config)
        ic = report.observed_vs_target["initial_context"]
        assert ic.target_mean == coding_config.initial_context.mean
        assert ic.target_median == coding_config.initial_context.median
        delay = report.observed_vs_target["inter_turn_delay_ms"]
        assert delay.target_mean is None
        assert delay.target_median is None
