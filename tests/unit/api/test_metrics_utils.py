# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the metrics utilities."""

from unittest.mock import patch

import pytest
from pytest import param

from aiperf.api.metrics_utils import build_info_labels, format_metrics_json
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.exceptions import MetricTypeError
from aiperf.common.models import MetricResult

from .conftest import (
    make_info_labels,
    make_latency_metric,
    make_metric_result,
    make_throughput_metric,
    make_user_config,
)


class TestBuildInfoLabels:
    """Test info label building from UserConfig."""

    def test_basic_labels(self) -> None:
        """Test basic label extraction from config."""
        config = make_user_config(model_names=["gpt-4"])
        labels = build_info_labels(config)

        assert labels["model"] == "gpt-4"
        assert labels["endpoint_type"] == "chat"
        assert labels["streaming"] == "false"
        assert "config" in labels

    def test_multiple_models(self) -> None:
        """Test multiple model names are comma-separated."""
        config = make_user_config(model_names=["gpt-4", "gpt-3.5-turbo"])
        labels = build_info_labels(config)
        assert labels["model"] == "gpt-4,gpt-3.5-turbo"

    def test_benchmark_id_included(self) -> None:
        """Test benchmark_id is included when set."""
        config = make_user_config(benchmark_id="test-bench-123")
        labels = build_info_labels(config)
        assert labels["benchmark_id"] == "test-bench-123"

    def test_benchmark_id_auto_generated(self) -> None:
        """Test benchmark_id is auto-generated when not explicitly set."""
        config = make_user_config()
        labels = build_info_labels(config)
        assert "benchmark_id" in labels

    @pytest.mark.parametrize(
        "loadgen_kwargs,label_key,expected",
        [
            param({"concurrency": 10}, "concurrency", "10", id="concurrency-int"),
            param({"concurrency": 1}, "concurrency", "1", id="concurrency-one"),
            param({"request_rate": 5.0}, "request_rate", "5.0", id="request-rate-float"),
            param({"request_rate": 0.5}, "request_rate", "0.5", id="request-rate-decimal"),
        ],
    )  # fmt: skip
    def test_loadgen_labels(
        self, loadgen_kwargs: dict, label_key: str, expected: str
    ) -> None:
        """Test loadgen parameters are included in labels."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["model"]),
            loadgen=LoadGeneratorConfig(**loadgen_kwargs),
        )
        labels = build_info_labels(config)
        assert labels[label_key] == expected

    @pytest.mark.parametrize(
        "streaming,expected",
        [
            param(False, "false", id="streaming-false"),
            param(True, "true", id="streaming-true"),
        ],
    )  # fmt: skip
    def test_streaming_label(self, streaming: bool, expected: str) -> None:
        """Test streaming label reflects endpoint configuration."""
        config = make_user_config(streaming=streaming)
        labels = build_info_labels(config)
        assert labels["streaming"] == expected

    @pytest.mark.parametrize(
        "endpoint_type",
        [
            param("chat", id="chat"),
            param("completions", id="completions"),
            param("embeddings", id="embeddings"),
        ],
    )  # fmt: skip
    def test_endpoint_type_label(self, endpoint_type: str) -> None:
        """Test endpoint_type label reflects configuration."""
        config = make_user_config(endpoint_type=endpoint_type)
        labels = build_info_labels(config)
        assert labels["endpoint_type"] == endpoint_type

    def test_config_contains_serialized_user_config(self) -> None:
        """Test that config label contains serialized UserConfig."""
        config = make_user_config(
            benchmark_id="test-123",
            model_names=["test-model"],
            streaming=True,
        )
        labels = build_info_labels(config)

        # config should be a dict (model_dump result)
        assert isinstance(labels["config"], dict)
        assert labels["config"]["benchmark_id"] == "test-123"

    def test_default_loadgen_params_in_labels(self) -> None:
        """Test that default loadgen params appear in labels when set by UserConfig."""
        config = make_user_config()  # UserConfig defaults concurrency to 1
        labels = build_info_labels(config)

        # Concurrency is defaulted by UserConfig, so it should appear
        assert "concurrency" in labels
        assert labels["concurrency"] == "1"
        # request_rate is not defaulted, so it should not appear
        assert "request_rate" not in labels

    def test_benchmark_id_excluded_when_empty(self) -> None:
        """Test benchmark_id is not included when it's an empty string."""
        config = make_user_config(benchmark_id="will-be-replaced")
        # Override the auto-generated benchmark_id with empty string
        config.benchmark_id = ""
        labels = build_info_labels(config)
        assert "benchmark_id" not in labels

    def test_benchmark_id_excluded_when_none(self) -> None:
        """Test benchmark_id is not included when it's None."""
        config = make_user_config(benchmark_id="will-be-replaced")
        # Override the auto-generated benchmark_id with None
        config.benchmark_id = None
        labels = build_info_labels(config)
        assert "benchmark_id" not in labels

    def test_loadgen_concurrency_none_excluded(self) -> None:
        """Test that concurrency is not included when explicitly set to None."""
        config = make_user_config()
        # Override concurrency to None
        config.loadgen.concurrency = None
        labels = build_info_labels(config)
        assert "concurrency" not in labels

    def test_loadgen_request_rate_none_excluded(self) -> None:
        """Test that request_rate is not included when it's None."""
        config = make_user_config()
        # Ensure request_rate is None (it should be by default)
        config.loadgen.request_rate = None
        labels = build_info_labels(config)
        assert "request_rate" not in labels


class TestFormatMetricsJson:
    """Test JSON metrics formatting."""

    def test_empty_metrics(self) -> None:
        """Test formatting empty metrics list."""
        data = format_metrics_json([])

        assert "aiperf_version" in data
        assert data["metrics"] == {}

    def test_single_metric(self) -> None:
        """Test formatting single metric."""
        metric = make_metric_result(avg=100.0, min=50.0, max=150.0)
        data = format_metrics_json([metric])

        assert "test_metric" in data["metrics"]
        assert data["metrics"]["test_metric"]["avg"] == 100.0
        assert data["metrics"]["test_metric"]["min"] == 50.0
        assert data["metrics"]["test_metric"]["max"] == 150.0
        assert "tag" not in data["metrics"]["test_metric"]

    def test_benchmark_id_included(self) -> None:
        """Test benchmark_id is included when provided."""
        data = format_metrics_json([], benchmark_id="bench-123")
        assert data["benchmark_id"] == "bench-123"

    def test_benchmark_id_excluded_when_none(self) -> None:
        """Test benchmark_id is not included when None."""
        data = format_metrics_json([])
        assert "benchmark_id" not in data

    def test_info_labels_included(self) -> None:
        """Test info labels are merged into response."""
        labels = make_info_labels(model="gpt-4", endpoint_type="openai")
        data = format_metrics_json([], info_labels=labels)

        assert data["model"] == "gpt-4"
        assert data["endpoint_type"] == "openai"

    @pytest.mark.parametrize(
        "excluded_key",
        [
            param("config", id="config-excluded"),
            param("version", id="version-excluded"),
        ],
    )  # fmt: skip
    def test_excluded_labels(self, excluded_key: str) -> None:
        """Test that certain labels are excluded from output."""
        data = format_metrics_json(
            [], info_labels={"model": "gpt-4", excluded_key: "some_value"}
        )

        assert excluded_key not in data or data.get(excluded_key) != "some_value"
        assert data["model"] == "gpt-4"

    def test_multiple_metrics(self) -> None:
        """Test formatting multiple metrics."""
        metrics = [
            make_latency_metric(avg=100.0),
            make_throughput_metric(avg=50.0),
        ]
        data = format_metrics_json(metrics)

        assert data["metrics"]["latency"]["avg"] == 100.0
        assert data["metrics"]["throughput"]["avg"] == 50.0

    def test_metric_type_error_uses_raw_metric(self) -> None:
        """Test that MetricTypeError falls back to raw metric values."""
        metric = make_metric_result(tag="unknown_metric", avg=100.0, min=50.0)

        with patch.object(
            MetricResult,
            "to_display_unit",
            side_effect=MetricTypeError("Unknown metric"),
        ):
            data = format_metrics_json([metric])

        # Should still include the metric with raw values
        assert "unknown_metric" in data["metrics"]
        assert data["metrics"]["unknown_metric"]["avg"] == 100.0

    def test_none_values_excluded_from_metric_dump(self) -> None:
        """Test that None values are excluded from metric output."""
        metric = make_metric_result(avg=100.0, min=None, max=None)
        data = format_metrics_json([metric])

        metric_data = data["metrics"]["test_metric"]
        assert "avg" in metric_data
        assert "min" not in metric_data
        assert "max" not in metric_data

    def test_all_stat_values_included(self) -> None:
        """Test that all provided stat values are included."""
        metric = MetricResult(
            tag="full_metric",
            header="Full Metric",
            unit="ms",
            avg=100.0,
            min=10.0,
            max=200.0,
            sum=5000.0,
            std=25.0,
            p50=95.0,
            p95=180.0,
            p99=195.0,
        )
        data = format_metrics_json([metric])

        metric_data = data["metrics"]["full_metric"]
        assert metric_data["avg"] == 100.0
        assert metric_data["min"] == 10.0
        assert metric_data["max"] == 200.0
        assert metric_data["sum"] == 5000.0
        assert metric_data["std"] == 25.0
        assert metric_data["p50"] == 95.0
        assert metric_data["p95"] == 180.0
        assert metric_data["p99"] == 195.0

    def test_info_labels_coerced(self) -> None:
        """Test that info label values are coerced via coerce_value."""
        # Pass a dict value that should be coerced
        labels = {"model": "test", "numeric": 42, "bool": True}
        data = format_metrics_json([], info_labels=labels)

        assert data["model"] == "test"
        assert data["numeric"] == 42
        assert data["bool"] is True

    def test_version_always_included(self) -> None:
        """Test that aiperf_version is always included."""
        data = format_metrics_json([])
        assert "aiperf_version" in data
        assert isinstance(data["aiperf_version"], str)
        assert len(data["aiperf_version"]) > 0

    def test_empty_info_labels_dict(self) -> None:
        """Test that empty info_labels dict does not add any labels."""
        data = format_metrics_json([], info_labels={})
        # Only aiperf_version and metrics should be present
        assert "aiperf_version" in data
        assert "metrics" in data
        # No additional labels from empty dict
        assert len(data) == 2

    def test_info_labels_with_string_values(self) -> None:
        """Test that string values in info_labels are coerced correctly."""
        labels = {
            "model": "gpt-4",
            "str_true": "true",
            "str_false": "false",
            "str_none": "none",
            "str_null": "null",
            "str_int": "42",
            "str_negative_int": "-10",
            "str_float": "3.14",
            "str_negative_float": "-2.5",
            "plain_string": "hello",
        }
        data = format_metrics_json([], info_labels=labels)

        assert data["model"] == "gpt-4"
        assert data["str_true"] is True
        assert data["str_false"] is False
        assert data["str_none"] is None
        assert data["str_null"] is None
        assert data["str_int"] == 42
        assert data["str_negative_int"] == -10
        assert data["str_float"] == 3.14
        assert data["str_negative_float"] == -2.5
        assert data["plain_string"] == "hello"

    def test_benchmark_id_and_info_labels_combined(self) -> None:
        """Test both benchmark_id and info_labels provided together."""
        labels = make_info_labels(model="test-model", benchmark_id="label-bench-id")
        data = format_metrics_json(
            [], info_labels=labels, benchmark_id="param-bench-id"
        )

        # benchmark_id from info_labels takes precedence (it's copied from labels dict)
        assert data["benchmark_id"] == "label-bench-id"
        # model from info_labels should be present
        assert data["model"] == "test-model"

    def test_metric_with_all_percentiles(self) -> None:
        """Test metric with all percentile values included."""
        metric = MetricResult(
            tag="percentile_metric",
            header="Percentile Metric",
            unit="ms",
            p50=50.0,
            p90=90.0,
            p95=95.0,
            p99=99.0,
        )
        data = format_metrics_json([metric])

        metric_data = data["metrics"]["percentile_metric"]
        assert metric_data["p50"] == 50.0
        assert metric_data["p90"] == 90.0
        assert metric_data["p95"] == 95.0
        assert metric_data["p99"] == 99.0

    def test_metric_display_unit_conversion_success(self) -> None:
        """Test that metrics are converted to display units when possible."""
        # Create a metric that can be converted to display units
        metric = make_latency_metric(avg=1000.0, min=500.0, max=2000.0)

        # When to_display_unit succeeds, it should use the converted values
        data = format_metrics_json([metric])

        # Verify the metric is included (conversion may or may not change values
        # depending on the metric type and registry)
        assert "latency" in data["metrics"]
        assert "avg" in data["metrics"]["latency"]
