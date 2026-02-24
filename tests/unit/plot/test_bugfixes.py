# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for plot bugfixes:

1. experiment_classification silently overriding user's groups setting
2. flatten_config not recursing into config dicts (e.g. endpoint with 'type' key)
3. Metadata columns being overwritten by flatten_config in _runs_to_dataframe
4. Legend title derived from group_by column name
"""

from pathlib import Path

import pandas as pd
import pytest

from aiperf.plot.config import PlotConfig
from aiperf.plot.core.data_loader import RunData, RunMetadata
from aiperf.plot.core.data_preparation import flatten_config
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import ExperimentClassificationConfig
from aiperf.plot.exporters.png import MultiRunPNGExporter


class TestHasClassificationPatterns:
    """Tests for PlotConfig._has_classification_patterns."""

    @pytest.mark.parametrize(
        "baselines,treatments,expected",
        [
            (["*baseline*"], ["*treatment*"], True),
            (["*baseline*"], [], True),
            ([], ["*treatment*"], True),
            ([], [], False),
        ],
        ids=[
            "both_defined",
            "only_baselines",
            "only_treatments",
            "neither_defined",
        ],
    )
    def test_has_classification_patterns(
        self,
        baselines: list[str],
        treatments: list[str],
        expected: bool,
    ) -> None:
        config = ExperimentClassificationConfig(
            baselines=baselines, treatments=treatments
        )
        assert PlotConfig._has_classification_patterns(config) is expected

    def test_default_config_has_no_patterns(self) -> None:
        """Default ExperimentClassificationConfig has empty pattern lists."""
        config = ExperimentClassificationConfig()
        assert PlotConfig._has_classification_patterns(config) is False


class TestExperimentClassificationGroupsDefault:
    """Tests that experiment_classification provides a default for groups, never an override."""

    @staticmethod
    def _write_config(
        path: Path,
        *,
        groups: str | None = None,
        exp_class_section: str = "",
    ) -> Path:
        groups_line = f"      groups: [{groups}]" if groups else ""
        config_file = path / "config.yaml"
        config_file.write_text(
            f"""\
{exp_class_section}
visualization:
  multi_run_defaults:
    - test_plot
  multi_run_plots:
    test_plot:
      type: scatter_line
      x: request_latency_avg
      y: request_throughput_avg
{groups_line}
  single_run_defaults: []
  single_run_plots: {{}}
"""
        )
        return config_file

    def test_explicit_groups_wins_over_classification_patterns(
        self, tmp_path: Path
    ) -> None:
        """Explicit groups must never be overridden, even when classification patterns exist."""
        config_file = self._write_config(
            tmp_path,
            groups="concurrency",
            exp_class_section=(
                "experiment_classification:\n"
                "  baselines:\n"
                "    - '*baseline*'\n"
                "  treatments:\n"
                "    - '*treatment*'\n"
            ),
        )
        specs = PlotConfig(config_file).get_multi_run_plot_specs()
        assert specs[0].group_by == "concurrency"

    def test_empty_classification_section_respects_user_groups(
        self, tmp_path: Path
    ) -> None:
        """Empty experiment_classification (no patterns) must NOT override groups."""
        config_file = self._write_config(
            tmp_path,
            groups="concurrency",
            exp_class_section=(
                "experiment_classification:\n"
                "  baselines: []\n"
                "  treatments: []\n"
                "  default: treatment\n"
            ),
        )
        specs = PlotConfig(config_file).get_multi_run_plot_specs()
        assert specs[0].group_by == "concurrency"

    def test_no_classification_section_respects_user_groups(
        self, tmp_path: Path
    ) -> None:
        """No experiment_classification section at all respects user groups."""
        config_file = self._write_config(tmp_path, groups="model")
        specs = PlotConfig(config_file).get_multi_run_plot_specs()
        assert specs[0].group_by == "model"

    def test_no_classification_no_groups_defaults_to_run_name(
        self, tmp_path: Path
    ) -> None:
        """No classification and no groups → defaults to run_name."""
        config_file = self._write_config(tmp_path)
        specs = PlotConfig(config_file).get_multi_run_plot_specs()
        assert specs[0].group_by == "run_name"

    def test_classification_defaults_only_does_not_override(
        self, tmp_path: Path
    ) -> None:
        """Classification with only 'default' field (no patterns) respects user groups."""
        config_file = self._write_config(
            tmp_path,
            groups="concurrency",
            exp_class_section="experiment_classification:\n  default: baseline\n",
        )
        specs = PlotConfig(config_file).get_multi_run_plot_specs()
        assert specs[0].group_by == "concurrency"

    def test_no_groups_with_patterns_defaults_to_experiment_group(
        self, tmp_path: Path
    ) -> None:
        """When groups is omitted and classification patterns exist, default to experiment_group."""
        config_file = self._write_config(
            tmp_path,
            exp_class_section=(
                "experiment_classification:\n"
                "  baselines:\n"
                "    - '*baseline*'\n"
                "  treatments:\n"
                "    - '*treatment*'\n"
            ),
        )
        specs = PlotConfig(config_file).get_multi_run_plot_specs()
        assert specs[0].group_by == "experiment_group"


class TestFlattenConfig:
    """Tests for flatten_config unconditionally recursing into all nested dicts."""

    def test_flattens_endpoint_config_with_type(self) -> None:
        """Endpoint configs with 'type' field must be recursed into."""
        config = {
            "endpoint": {
                "type": "openai-chat",
                "model_names": ["Qwen3-32B"],
                "streaming": True,
            }
        }
        result = flatten_config(config)
        assert result["endpoint.type"] == "openai-chat"
        assert (
            result["endpoint.model_names"] == "Qwen3-32B"
        )  # single-element list unwrapped
        assert result["endpoint.streaming"] is True

    def test_flattens_loadgen_config(self) -> None:
        config = {
            "loadgen": {
                "concurrency": 10,
                "benchmark_duration": 60,
            }
        }
        result = flatten_config(config)
        assert result["loadgen.concurrency"] == 10
        assert result["loadgen.benchmark_duration"] == 60

    def test_deeply_nested_dicts_fully_flattened(self) -> None:
        """All nested dicts are recursed into regardless of their keys."""
        config = {
            "outer": {
                "unit": "ms",
                "avg": 42.0,
                "nested": {"deep_key": "deep_value"},
            }
        }
        result = flatten_config(config)
        assert result["outer.unit"] == "ms"
        assert result["outer.avg"] == 42.0
        assert result["outer.nested.deep_key"] == "deep_value"

    def test_nested_flattening(self) -> None:
        config = {
            "endpoint": {
                "type": "chat",
                "urls": ["http://localhost:8000"],
            },
            "loadgen": {
                "concurrency": 4,
            },
        }
        result = flatten_config(config)
        assert result == {
            "endpoint.type": "chat",
            "endpoint.urls": "http://localhost:8000",  # single-element list unwrapped
            "loadgen.concurrency": 4,
        }

    def test_single_element_list_unwrapped(self) -> None:
        config = {"model_names": ["Qwen3-32B"]}
        result = flatten_config(config)
        assert result["model_names"] == "Qwen3-32B"

    def test_multi_element_list_preserved(self) -> None:
        config = {"model_names": ["model-a", "model-b"]}
        result = flatten_config(config)
        assert result["model_names"] == ["model-a", "model-b"]

    def test_empty_dict_produces_no_keys(self) -> None:
        assert flatten_config({}) == {}

    def test_empty_nested_dict_produces_no_keys(self) -> None:
        config = {"section": {}}
        assert flatten_config(config) == {}


class TestMetadataColumnProtection:
    """Tests that _runs_to_dataframe protects metadata columns from flatten_config overwrites."""

    @pytest.fixture
    def exporter(self, tmp_path: Path) -> MultiRunPNGExporter:
        return MultiRunPNGExporter(tmp_path / "plots")

    def _make_run(
        self,
        tmp_path: Path,
        *,
        run_name: str = "run_001",
        model: str = "Qwen3-32B",
        concurrency: int = 10,
        input_config: dict | None = None,
    ) -> RunData:
        return RunData(
            metadata=RunMetadata(
                run_name=run_name,
                run_path=tmp_path / run_name,
                model=model,
                concurrency=concurrency,
            ),
            requests=None,
            aggregated={"input_config": input_config} if input_config else {},
            timeslices=None,
            slice_duration=None,
        )

    def test_metadata_not_overwritten_by_flattened_config(
        self, exporter: MultiRunPNGExporter, tmp_path: Path
    ) -> None:
        """Metadata columns must win over identically-named flatten_config keys."""
        run = self._make_run(
            tmp_path,
            model="Qwen3-32B",
            concurrency=10,
            input_config={
                "model": "should-not-overwrite",
                "concurrency": 999,
                "run_name": "should-not-overwrite",
            },
        )
        df = exporter._runs_to_dataframe([run], {"display_names": {}, "units": {}})
        assert df["model"].iloc[0] == "Qwen3-32B"
        assert df["concurrency"].iloc[0] == 10
        assert df["run_name"].iloc[0] == "run_001"

    def test_non_conflicting_config_keys_are_added(
        self, exporter: MultiRunPNGExporter, tmp_path: Path
    ) -> None:
        """Config keys that don't conflict with metadata should appear in the DataFrame."""
        run = self._make_run(
            tmp_path,
            input_config={
                "endpoint": {
                    "type": "openai-chat",
                    "streaming": True,
                },
                "loadgen": {
                    "benchmark_duration": 60,
                },
            },
        )
        df = exporter._runs_to_dataframe([run], {"display_names": {}, "units": {}})
        assert "endpoint.type" in df.columns
        assert df["endpoint.type"].iloc[0] == "openai-chat"
        assert df["endpoint.streaming"].iloc[0] == True  # noqa: E712
        assert df["loadgen.benchmark_duration"].iloc[0] == 60


class TestFormatLegendTitle:
    """Tests for PlotGenerator._format_legend_title."""

    @pytest.mark.parametrize(
        "column,expected",
        [
            ("concurrency", "Concurrency"),
            ("model", "Model"),
            ("run_name", "Run Name"),
            ("experiment_group", "Experiment Group"),
            ("endpoint_type", "Endpoint Type"),
        ],
    )
    def test_format_legend_title(self, column: str, expected: str) -> None:
        assert PlotGenerator._format_legend_title(column) == expected


class TestLegendTitle:
    """Tests that multi-run plots include a legend title derived from group_by."""

    @pytest.fixture
    def generator(self) -> PlotGenerator:
        return PlotGenerator()

    @pytest.fixture
    def multi_run_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": ["ModelA"] * 3 + ["ModelB"] * 3,
                "concurrency": [1, 10, 50, 1, 10, 50],
                "request_latency": [100, 200, 400, 120, 250, 500],
                "request_throughput": [10, 20, 30, 8, 18, 25],
                "time_to_first_token": [50, 80, 150, 60, 90, 170],
                "output_token_throughput_per_user": [90, 70, 50, 85, 65, 45],
            }
        )

    def test_pareto_plot_has_legend_title(
        self, generator: PlotGenerator, multi_run_df: pd.DataFrame
    ) -> None:
        fig = generator.create_pareto_plot(
            df=multi_run_df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by="model",
        )
        assert fig.layout.legend.title.text == "<b>Model</b>"

    def test_scatter_line_plot_has_legend_title(
        self, generator: PlotGenerator, multi_run_df: pd.DataFrame
    ) -> None:
        fig = generator.create_scatter_line_plot(
            df=multi_run_df,
            x_metric="time_to_first_token",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by="model",
        )
        assert fig.layout.legend.title.text == "<b>Model</b>"

    def test_legend_title_from_concurrency(
        self, generator: PlotGenerator, multi_run_df: pd.DataFrame
    ) -> None:
        fig = generator.create_scatter_line_plot(
            df=multi_run_df,
            x_metric="time_to_first_token",
            y_metric="request_throughput",
            label_by="model",
            group_by="concurrency",
        )
        assert fig.layout.legend.title.text == "<b>Concurrency</b>"

    def test_no_legend_title_without_group_by(self, generator: PlotGenerator) -> None:
        df = pd.DataFrame(
            {
                "concurrency": [1, 10, 50],
                "request_latency": [100, 200, 400],
                "request_throughput": [10, 20, 30],
            }
        )
        fig = generator.create_pareto_plot(
            df=df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by=None,
        )
        assert fig.layout.legend.title.text is None

    def test_legend_title_font_styling(
        self, generator: PlotGenerator, multi_run_df: pd.DataFrame
    ) -> None:
        fig = generator.create_pareto_plot(
            df=multi_run_df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by="model",
        )
        assert fig.layout.legend.title.font.size == 12
