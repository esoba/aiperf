# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.cli_commands.config module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pytest import param

from aiperf.cli_commands.config import (
    _build_cr_yaml,
    _dump_clean_yaml,
    _remove_empty,
    _strip_runtime_fields,
    generate,
)
from aiperf.config.config import AIPerfConfig


def _make_config(**overrides: Any) -> AIPerfConfig:
    """Create a minimal valid AIPerfConfig for testing."""
    base: dict[str, Any] = {
        "models": ["test-model"],
        "endpoint": {"urls": ["http://localhost:8000"], "streaming": True},
        "datasets": {
            "main": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 512, "osl": 128},
            }
        },
        "load": {
            "profiling": {
                "type": "concurrency",
                "requests": 100,
                "concurrency": 8,
            }
        },
    }
    base.update(overrides)
    return AIPerfConfig.model_validate(base)


# =============================================================================
# Strip Runtime Fields
# =============================================================================


class TestStripRuntimeFields:
    """Tests for _strip_runtime_fields."""

    def test_removes_cli_command(self) -> None:
        data: dict[str, Any] = {
            "models": {},
            "artifacts": {
                "cli_command": "aiperf profile --model foo",
                "dir": "./results",
            },
        }
        _strip_runtime_fields(data)
        assert "cli_command" not in data["artifacts"]

    def test_removes_benchmark_id(self) -> None:
        data: dict[str, Any] = {
            "artifacts": {
                "benchmark_id": "abc-123",
                "dir": "./results",
            },
        }
        _strip_runtime_fields(data)
        assert "benchmark_id" not in data["artifacts"]

    def test_removes_empty_artifacts(self) -> None:
        data: dict[str, Any] = {
            "artifacts": {"cli_command": "x", "benchmark_id": "y"},
            "models": {},
        }
        _strip_runtime_fields(data)
        assert "artifacts" not in data

    def test_preserves_non_runtime_artifacts(self) -> None:
        data: dict[str, Any] = {
            "artifacts": {
                "cli_command": "x",
                "benchmark_id": "y",
                "dir": "./results",
            },
        }
        _strip_runtime_fields(data)
        assert data["artifacts"] == {"dir": "./results"}


# =============================================================================
# Remove Empty
# =============================================================================


class TestRemoveEmpty:
    """Tests for _remove_empty."""

    def test_removes_empty_dicts(self) -> None:
        data: dict[str, Any] = {"a": {}, "b": "value"}
        _remove_empty(data)
        assert data == {"b": "value"}

    def test_removes_empty_lists(self) -> None:
        data: dict[str, Any] = {"formats": [], "name": "test"}
        _remove_empty(data)
        assert data == {"name": "test"}

    def test_removes_nested_empty(self) -> None:
        data: dict[str, Any] = {"outer": {"inner": {}}}
        _remove_empty(data)
        assert data == {}

    def test_preserves_non_empty(self) -> None:
        data: dict[str, Any] = {"a": {"b": 1}, "c": [1, 2]}
        _remove_empty(data)
        assert data == {"a": {"b": 1}, "c": [1, 2]}

    def test_preserves_zero_and_false(self) -> None:
        data: dict[str, Any] = {"count": 0, "enabled": False}
        _remove_empty(data)
        assert data == {"count": 0, "enabled": False}


# =============================================================================
# YAML Generation
# =============================================================================


class TestDumpCleanYaml:
    """Tests for _dump_clean_yaml."""

    def test_produces_valid_yaml(self) -> None:
        config = _make_config()
        result = _dump_clean_yaml(config)
        parsed = yaml.safe_load(result)
        assert "models" in parsed
        assert "endpoint" in parsed

    def test_excludes_runtime_fields(self) -> None:
        config = _make_config()
        result = _dump_clean_yaml(config)
        assert "cli_command" not in result
        assert "benchmark_id" not in result

    def test_roundtrip_validation(self) -> None:
        """Generated YAML should be loadable back as a valid AIPerfConfig."""
        config = _make_config()
        result = _dump_clean_yaml(config)
        parsed = yaml.safe_load(result)
        reloaded = AIPerfConfig.model_validate(parsed)
        assert reloaded.endpoint.streaming is True
        assert len(list(reloaded.load.values())) > 0


# =============================================================================
# CR Generation
# =============================================================================


class TestBuildCrYaml:
    """Tests for _build_cr_yaml."""

    def test_produces_valid_cr_structure(self) -> None:
        config = _make_config()
        result = _build_cr_yaml(config, name="test-job", namespace=None, image=None)
        parsed = yaml.safe_load(result)
        assert parsed["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
        assert parsed["kind"] == "AIPerfJob"
        assert parsed["metadata"]["name"] == "test-job"
        assert "spec" in parsed

    def test_includes_namespace(self) -> None:
        config = _make_config()
        result = _build_cr_yaml(
            config, name="test-job", namespace="benchmarks", image=None
        )
        parsed = yaml.safe_load(result)
        assert parsed["metadata"]["namespace"] == "benchmarks"

    def test_omits_namespace_when_none(self) -> None:
        config = _make_config()
        result = _build_cr_yaml(config, name="test-job", namespace=None, image=None)
        parsed = yaml.safe_load(result)
        assert "namespace" not in parsed["metadata"]

    def test_includes_image(self) -> None:
        config = _make_config()
        result = _build_cr_yaml(
            config, name="test-job", namespace=None, image="aiperf:v2"
        )
        parsed = yaml.safe_load(result)
        assert parsed["spec"]["image"] == "aiperf:v2"

    def test_omits_image_when_none(self) -> None:
        config = _make_config()
        result = _build_cr_yaml(config, name="test-job", namespace=None, image=None)
        parsed = yaml.safe_load(result)
        assert "image" not in parsed["spec"]

    def test_excludes_runtime_fields(self) -> None:
        config = _make_config()
        result = _build_cr_yaml(config, name="test-job", namespace=None, image=None)
        assert "cli_command" not in result
        assert "benchmark_id" not in result

    def test_spec_contains_config_fields(self) -> None:
        config = _make_config()
        result = _build_cr_yaml(config, name="test-job", namespace=None, image=None)
        parsed = yaml.safe_load(result)
        spec = parsed["spec"]
        assert "models" in spec
        assert "endpoint" in spec
        assert "datasets" in spec
        assert "load" in spec

    @pytest.mark.parametrize(
        "name,namespace,image",
        [
            param("my-job", "prod", "img:latest", id="all-fields"),
            param("minimal", None, None, id="minimal"),
            param("with-image", None, "ghcr.io/nvidia/aiperf:v2", id="image-only"),
        ],
    )  # fmt: skip
    def test_cr_metadata_variants(
        self, name: str, namespace: str | None, image: str | None
    ) -> None:
        config = _make_config()
        result = _build_cr_yaml(config, name=name, namespace=namespace, image=image)
        parsed = yaml.safe_load(result)
        assert parsed["metadata"]["name"] == name
        if namespace:
            assert parsed["metadata"]["namespace"] == namespace
        else:
            assert "namespace" not in parsed["metadata"]
        if image:
            assert parsed["spec"]["image"] == image


# =============================================================================
# Generate Command Integration
# =============================================================================


class TestGenerateCommand:
    """Tests for the generate command function."""

    def test_yaml_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        from aiperf.config.cli_builder import CLIModel

        cli = CLIModel(
            model_names=["test-model"],
            urls=["http://localhost:8000"],
            streaming=True,
            request_count=100,
            concurrency=8,
        )
        generate(cli, format="yaml", output=None)

        captured = capsys.readouterr()
        parsed = yaml.safe_load(captured.out)
        assert "models" in parsed
        assert "endpoint" in parsed

    def test_cr_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        from aiperf.config.cli_builder import CLIModel

        cli = CLIModel(
            model_names=["test-model"],
            urls=["http://localhost:8000"],
            streaming=True,
            request_count=100,
            concurrency=8,
        )
        generate(
            cli,
            format="cr",
            output=None,
            cr_name="my-job",
            cr_namespace="default",
            cr_image="aiperf:latest",
        )

        captured = capsys.readouterr()
        parsed = yaml.safe_load(captured.out)
        assert parsed["kind"] == "AIPerfJob"
        assert parsed["spec"]["image"] == "aiperf:latest"

    def test_yaml_to_file(self, tmp_path: Path) -> None:
        from aiperf.config.cli_builder import CLIModel

        output_file = tmp_path / "config.yaml"
        cli = CLIModel(
            model_names=["test-model"],
            urls=["http://localhost:8000"],
            streaming=True,
            request_count=100,
            concurrency=8,
        )
        generate(cli, format="yaml", output=output_file)

        assert output_file.exists()
        parsed = yaml.safe_load(output_file.read_text())
        assert "models" in parsed

    def test_cr_to_file(self, tmp_path: Path) -> None:
        from aiperf.config.cli_builder import CLIModel

        output_file = tmp_path / "cr.yaml"
        cli = CLIModel(
            model_names=["test-model"],
            urls=["http://localhost:8000"],
            streaming=True,
            request_count=100,
            concurrency=8,
        )
        generate(
            cli,
            format="cr",
            output=output_file,
            cr_name="file-job",
            cr_image="aiperf:v2",
        )

        assert output_file.exists()
        parsed = yaml.safe_load(output_file.read_text())
        assert parsed["kind"] == "AIPerfJob"
        assert parsed["metadata"]["name"] == "file-job"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        from aiperf.config.cli_builder import CLIModel

        output_file = tmp_path / "deep" / "nested" / "config.yaml"
        cli = CLIModel(
            model_names=["test-model"],
            urls=["http://localhost:8000"],
            request_count=100,
            concurrency=8,
        )
        generate(cli, format="yaml", output=output_file)

        assert output_file.exists()

    def test_yaml_roundtrip_valid(self, tmp_path: Path) -> None:
        """Generated YAML can be loaded back and passes AIPerfConfig validation."""
        from aiperf.config.cli_builder import CLIModel

        output_file = tmp_path / "roundtrip.yaml"
        cli = CLIModel(
            model_names=["test-model"],
            urls=["http://localhost:8000"],
            streaming=True,
            request_count=500,
            concurrency=32,
        )
        generate(cli, format="yaml", output=output_file)

        parsed = yaml.safe_load(output_file.read_text())
        reloaded = AIPerfConfig.model_validate(parsed)
        assert reloaded.endpoint.streaming is True

    def test_success_message_yaml(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiperf.config.cli_builder import CLIModel

        output_file = tmp_path / "msg.yaml"
        cli = CLIModel(
            model_names=["test-model"],
            urls=["http://localhost:8000"],
            request_count=100,
            concurrency=8,
        )
        generate(cli, format="yaml", output=output_file)

        captured = capsys.readouterr()
        assert "Configuration written to" in captured.err
        assert "aiperf profile --config" in captured.err

    def test_success_message_cr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiperf.config.cli_builder import CLIModel

        output_file = tmp_path / "msg.yaml"
        cli = CLIModel(
            model_names=["test-model"],
            urls=["http://localhost:8000"],
            request_count=100,
            concurrency=8,
        )
        generate(cli, format="cr", output=output_file, cr_name="x")

        captured = capsys.readouterr()
        assert "kubectl apply -f" in captured.err
