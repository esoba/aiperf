# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AIPerfConfig <-> CRD integration.

Validates:
- CRD spec -> AIPerfConfig conversion
- Reverse converter (AIPerfConfig -> UserConfig/ServiceConfig)
- CRD validation
- ConfigMap generation with AIPerfConfig
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.config.config import AIPerfConfig
from aiperf.config.reverse_converter import convert_to_legacy_configs
from aiperf.operator.spec_converter import AIPerfJobSpecConverter

# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------


def _spec() -> dict[str, Any]:
    """Create a CRD spec with AIPerfConfig fields inline."""
    return {
        "image": "nvcr.io/nvidia/aiperf:latest",
        "models": ["meta-llama/Llama-3.1-8B-Instruct"],
        "endpoint": {
            "urls": ["http://localhost:8000/v1/chat/completions"],
            "type": "chat",
            "streaming": True,
        },
        "datasets": {
            "main": {
                "type": "synthetic",
                "entries": 1000,
                "prompts": {"isl": {"mean": 512, "stddev": 50}, "osl": 128},
            },
        },
        "load": {
            "profiling": {
                "type": "concurrency",
                "dataset": "main",
                "requests": 100,
                "concurrency": 8,
            },
        },
    }


def _spec_multi_phase() -> dict[str, Any]:
    """Create a spec with warmup + profiling phases."""
    spec = _spec()
    spec["datasets"]["warmup"] = {
        "type": "synthetic",
        "entries": 100,
        "prompts": {"isl": 256, "osl": 64},
    }
    spec["load"] = {
        "warmup": {
            "type": "concurrency",
            "dataset": "warmup",
            "exclude": True,
            "requests": 50,
            "concurrency": 4,
        },
        "profiling": {
            "type": "gamma",
            "dataset": "main",
            "duration": 300,
            "rate": 50.0,
            "concurrency": 64,
            "smoothness": 1.5,
        },
    }
    return spec


# ---------------------------------------------------------------------------
# CRD spec -> AIPerfConfig
# ---------------------------------------------------------------------------


class TestSpecConversion:
    """Tests for converting CRD specs to AIPerfConfig."""

    def test_basic_conversion(self) -> None:
        converter = AIPerfJobSpecConverter(
            spec=_spec(), name="test-job", namespace="default"
        )
        config = converter.to_aiperf_config()
        assert config.get_model_names() == ["meta-llama/Llama-3.1-8B-Instruct"]
        assert len(config.endpoint.urls) == 1
        assert "main" in config.datasets
        assert "profiling" in config.load

    def test_multi_phase_conversion(self) -> None:
        converter = AIPerfJobSpecConverter(
            spec=_spec_multi_phase(), name="test-job", namespace="default"
        )
        config = converter.to_aiperf_config()
        assert "warmup" in config.load
        assert "profiling" in config.load
        assert config.load["warmup"].exclude is True
        assert config.load["profiling"].exclude is False

    def test_artifacts_default_dir(self) -> None:
        converter = AIPerfJobSpecConverter(
            spec=_spec(), name="test-job", namespace="default"
        )
        config = converter.to_aiperf_config()
        assert str(config.artifacts.dir) == "/results"

    def test_cli_command_traceability(self) -> None:
        converter = AIPerfJobSpecConverter(
            spec=_spec(), name="my-bench", namespace="default"
        )
        config = converter.to_aiperf_config()
        assert "my-bench" in config.artifacts.cli_command

    def test_with_slos(self) -> None:
        spec = _spec()
        spec["slos"] = {"time_to_first_token": 100, "inter_token_latency": 10}
        converter = AIPerfJobSpecConverter(
            spec=spec, name="test-job", namespace="default"
        )
        config = converter.to_aiperf_config()
        assert config.slos is not None

    def test_with_runtime_config(self) -> None:
        spec = _spec()
        spec["runtime"] = {"workers": 4}
        converter = AIPerfJobSpecConverter(
            spec=spec, name="test-job", namespace="default"
        )
        config = converter.to_aiperf_config()
        assert config.runtime.workers == 4

    def test_with_random_seed(self) -> None:
        spec = _spec()
        spec["random_seed"] = 42
        converter = AIPerfJobSpecConverter(
            spec=spec, name="test-job", namespace="default"
        )
        config = converter.to_aiperf_config()
        assert config.random_seed == 42

    def test_worker_count_from_phases(self) -> None:
        spec = _spec()
        spec["load"]["profiling"]["concurrency"] = 1000
        spec["connectionsPerWorker"] = 200
        converter = AIPerfJobSpecConverter(
            spec=spec, name="test-job", namespace="default"
        )
        assert converter.calculate_workers() == 5  # ceil(1000/200)


# ---------------------------------------------------------------------------
# Reverse converter
# ---------------------------------------------------------------------------


class TestReverseConverter:
    """Tests that AIPerfConfig correctly bridges to legacy configs."""

    def test_basic_reverse_conversion(self) -> None:
        converter = AIPerfJobSpecConverter(
            spec=_spec(), name="test-job", namespace="default"
        )
        config = converter.to_aiperf_config()
        uc, sc = convert_to_legacy_configs(config)

        assert uc.endpoint.model_names == ["meta-llama/Llama-3.1-8B-Instruct"]
        assert uc.endpoint.urls == ["http://localhost:8000/v1/chat/completions"]
        assert sc is not None

    def test_reverse_preserves_concurrency(self) -> None:
        converter = AIPerfJobSpecConverter(
            spec=_spec(), name="test-job", namespace="default"
        )
        config = converter.to_aiperf_config()
        uc, _ = convert_to_legacy_configs(config)
        assert uc.loadgen.concurrency == 8

    def test_reverse_preserves_request_count(self) -> None:
        converter = AIPerfJobSpecConverter(
            spec=_spec(), name="test-job", namespace="default"
        )
        config = converter.to_aiperf_config()
        uc, _ = convert_to_legacy_configs(config)
        assert uc.loadgen.request_count == 100

    def test_reverse_preserves_streaming(self) -> None:
        converter = AIPerfJobSpecConverter(
            spec=_spec(), name="test-job", namespace="default"
        )
        config = converter.to_aiperf_config()
        uc, _ = convert_to_legacy_configs(config)
        assert uc.endpoint.streaming is True


# ---------------------------------------------------------------------------
# ConfigMap generation
# ---------------------------------------------------------------------------


class TestConfigMapGeneration:
    """Tests that ConfigMap is generated correctly with AIPerfConfig."""

    def test_configmap_from_aiperf_config(self) -> None:
        from aiperf.kubernetes.resources import ConfigMapSpec

        converter = AIPerfJobSpecConverter(
            spec=_spec(), name="test-job", namespace="default"
        )
        config = converter.to_aiperf_config()

        cm = ConfigMapSpec.from_aiperf_config(
            name="test-config",
            namespace="default",
            config=config,
            job_id="abc123",
        )

        assert "aiperf_config.json" in cm.data

    def test_configmap_contains_model_names(self) -> None:
        from aiperf.kubernetes.resources import ConfigMapSpec

        converter = AIPerfJobSpecConverter(
            spec=_spec(), name="test-job", namespace="default"
        )
        config = converter.to_aiperf_config()

        cm = ConfigMapSpec.from_aiperf_config(
            name="test-config",
            namespace="default",
            config=config,
            job_id="abc123",
        )

        assert "Llama-3.1-8B-Instruct" in cm.data["aiperf_config.json"]


# ---------------------------------------------------------------------------
# CRD validation
# ---------------------------------------------------------------------------


class TestCRDValidation:
    """Tests that CRD validation works."""

    def test_validate_spec(self, tmp_path: Path) -> None:
        import yaml

        from aiperf.kubernetes.validate import validate_file

        doc = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": {"name": "test-job"},
            "spec": _spec(),
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(doc))
        result = validate_file(path)
        assert result.passed, f"Validation errors: {result.errors}"

    def test_validate_multi_phase_spec(self, tmp_path: Path) -> None:
        import yaml

        from aiperf.kubernetes.validate import validate_file

        doc = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": {"name": "multi-phase"},
            "spec": _spec_multi_phase(),
        }
        path = tmp_path / "multi-phase.yaml"
        path.write_text(yaml.dump(doc))
        result = validate_file(path)
        assert result.passed, f"Validation errors: {result.errors}"


# ---------------------------------------------------------------------------
# Template validation
# ---------------------------------------------------------------------------


class TestTemplateValidation:
    """Test that YAML templates validate correctly."""

    @pytest.fixture
    def template_dir(self) -> Path:
        return Path("src/aiperf/config/templates")

    def test_templates_exist(self, template_dir: Path) -> None:
        assert template_dir.exists()
        yamls = list(template_dir.glob("*.yaml"))
        assert len(yamls) > 0

    @pytest.mark.parametrize(
        "template",
        [
            param("audio_multimodal.yaml", id="audio_multimodal"),
            param("basic_throughput.yaml", id="basic_throughput"),
            param("composed_dataset.yaml", id="composed_dataset"),
            param("embeddings.yaml", id="embeddings"),
            param("goodput_slo.yaml", id="goodput_slo"),
            param("kv_cache_test.yaml", id="kv_cache_test"),
            param("latency_test.yaml", id="latency_test"),
            param("long_context.yaml", id="long_context"),
            param("minimal.yaml", id="minimal"),
            param("multi_turn_conversation.yaml", id="multi_turn_conversation"),
            param("multi_url_load_balancing.yaml", id="multi_url_load_balancing"),
            param("multimodal_vision.yaml", id="multimodal_vision"),
            param("public_dataset.yaml", id="public_dataset"),
            param("rate_limited.yaml", id="rate_limited"),
            param("request_cancellation.yaml", id="request_cancellation"),
            param("trace_replay.yaml", id="trace_replay"),
            param("warmup_profiling.yaml", id="warmup_profiling"),
            param("multi_run.yaml", id="multi_run"),
            param("accuracy.yaml", id="accuracy"),
        ],
    )  # fmt: skip
    def test_template_loads_as_config(self, template_dir: Path, template: str) -> None:
        from aiperf.config.loader import load_config

        template_path = template_dir / template
        config = load_config(template_path, substitute_env=False)
        assert isinstance(config, AIPerfConfig)
        assert len(config.get_model_names()) > 0

    @pytest.fixture
    def schema(self) -> dict[str, Any]:
        import json

        schema_path = Path("src/aiperf/config/schema/aiperf-config.schema.json")
        return json.loads(schema_path.read_text())

    @pytest.mark.parametrize(
        "template",
        [
            param("audio_multimodal.yaml", id="audio_multimodal"),
            param("basic_throughput.yaml", id="basic_throughput"),
            param("composed_dataset.yaml", id="composed_dataset"),
            param("embeddings.yaml", id="embeddings"),
            param("goodput_slo.yaml", id="goodput_slo"),
            param("kv_cache_test.yaml", id="kv_cache_test"),
            param("latency_test.yaml", id="latency_test"),
            param("long_context.yaml", id="long_context"),
            param("minimal.yaml", id="minimal"),
            param("multi_turn_conversation.yaml", id="multi_turn_conversation"),
            param("multi_url_load_balancing.yaml", id="multi_url_load_balancing"),
            param("multimodal_vision.yaml", id="multimodal_vision"),
            param("public_dataset.yaml", id="public_dataset"),
            param("rate_limited.yaml", id="rate_limited"),
            param("request_cancellation.yaml", id="request_cancellation"),
            param("trace_replay.yaml", id="trace_replay"),
            param("warmup_profiling.yaml", id="warmup_profiling"),
            param("multi_run.yaml", id="multi_run"),
            param("accuracy.yaml", id="accuracy"),
        ],
    )  # fmt: skip
    def test_template_validates_against_json_schema(
        self,
        template_dir: Path,
        template: str,
        schema: dict[str, Any],
    ) -> None:
        import yaml
        from jsonschema import Draft202012Validator

        template_path = template_dir / template
        data = yaml.safe_load(template_path.read_text())

        validator = Draft202012Validator(schema)
        errors = list(validator.iter_errors(data))
        assert not errors, f"{template} has schema validation errors:\n" + "\n".join(
            f"  - {e.json_path}: {e.message}" for e in errors
        )
