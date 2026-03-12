# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate all AIPerfJob recipe YAML files against the CRD schema and UserConfig model."""

from pathlib import Path

import pytest
import yaml

from aiperf.operator.spec_converter import AIPerfJobSpecConverter

RECIPES_DIR = Path(__file__).parents[3] / "recipes"


def _discover_recipes() -> list[Path]:
    """Find all aiperfjob.yaml files in the recipes directory."""
    if not RECIPES_DIR.exists():
        return []
    return sorted(RECIPES_DIR.rglob("aiperfjob.yaml"))


RECIPE_FILES = _discover_recipes()


@pytest.mark.parametrize(
    "recipe_path",
    RECIPE_FILES,
    ids=[str(p.relative_to(RECIPES_DIR)) for p in RECIPE_FILES],
)
class TestRecipeValidation:
    """Validate each recipe YAML against the AIPerfJob CRD schema."""

    def test_yaml_structure(self, recipe_path: Path) -> None:
        """Verify required YAML structure: apiVersion, kind, metadata.name, spec.userConfig."""
        doc = yaml.safe_load(recipe_path.read_text())
        assert doc["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
        assert doc["kind"] == "AIPerfJob"
        assert "name" in doc["metadata"]
        assert "userConfig" in doc["spec"]

    def test_user_config_validates(self, recipe_path: Path) -> None:
        """Verify spec.userConfig passes UserConfig.model_validate()."""
        doc = yaml.safe_load(recipe_path.read_text())
        spec = doc["spec"]
        name = doc["metadata"]["name"]

        converter = AIPerfJobSpecConverter(spec=spec, name=name, namespace="default")
        user_config = converter.to_user_config()

        assert user_config.endpoint.model_names
        assert user_config.endpoint.urls
        for url in user_config.endpoint.urls:
            assert url.startswith("http://") or url.startswith("https://")

    def test_service_config_validates(self, recipe_path: Path) -> None:
        """Verify spec converts to valid ServiceConfig."""
        doc = yaml.safe_load(recipe_path.read_text())
        spec = doc["spec"]
        name = doc["metadata"]["name"]

        converter = AIPerfJobSpecConverter(spec=spec, name=name, namespace="default")
        service_config = converter.to_service_config()

        assert service_config is not None

    def test_pod_customization(self, recipe_path: Path) -> None:
        """Verify podTemplate converts to valid PodCustomization."""
        doc = yaml.safe_load(recipe_path.read_text())
        spec = doc["spec"]
        name = doc["metadata"]["name"]

        converter = AIPerfJobSpecConverter(spec=spec, name=name, namespace="default")
        pod = converter.to_pod_customization()

        assert pod is not None

    def test_worker_calculation(self, recipe_path: Path) -> None:
        """Verify worker count calculation produces >= 1."""
        doc = yaml.safe_load(recipe_path.read_text())
        spec = doc["spec"]
        name = doc["metadata"]["name"]

        converter = AIPerfJobSpecConverter(spec=spec, name=name, namespace="default")
        workers = converter.calculate_workers()

        assert workers >= 1

    def test_metadata_name_is_valid_k8s_name(self, recipe_path: Path) -> None:
        """Verify metadata.name is a valid Kubernetes resource name."""
        import re

        doc = yaml.safe_load(recipe_path.read_text())
        name = doc["metadata"]["name"]

        assert len(name) <= 253
        assert re.match(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$", name), (
            f"Invalid K8s name: {name}"
        )

    def test_no_unknown_top_level_spec_fields(self, recipe_path: Path) -> None:
        """Verify spec only contains known CRD fields."""
        doc = yaml.safe_load(recipe_path.read_text())
        spec = doc["spec"]

        known_fields = {
            "image",
            "imagePullPolicy",
            "userConfig",
            "connectionsPerWorker",
            "timeoutSeconds",
            "ttlSecondsAfterFinished",
            "resultsTtlDays",
            "cancel",
            "podTemplate",
            "scheduling",
        }
        unknown = set(spec.keys()) - known_fields
        assert not unknown, f"Unknown spec fields: {unknown}"


class TestRecipeCompleteness:
    """Verify all expected recipes exist."""

    def test_recipe_count(self) -> None:
        """Verify we have all 15 expected recipe files."""
        assert len(RECIPE_FILES) == 15, (
            f"Expected 15 recipes, found {len(RECIPE_FILES)}: "
            + ", ".join(str(p.relative_to(RECIPES_DIR)) for p in RECIPE_FILES)
        )

    def test_expected_recipes_exist(self) -> None:
        """Verify the expected directory structure exists."""
        expected = [
            "deepseek-r1/trtllm/disagg/wide_ep/gb200/aiperfjob.yaml",
            "deepseek-v32-fp4/trtllm/agg-round-robin/aiperfjob.yaml",
            "deepseek-v32-fp4/trtllm/disagg-kv-router/aiperfjob.yaml",
            "gpt-oss-120b/trtllm/agg/aiperfjob.yaml",
            "gpt-oss-120b/trtllm/disagg/aiperfjob.yaml",
            "llama-3-70b/vllm/agg/aiperfjob.yaml",
            "llama-3-70b/vllm/disagg-multi-node/aiperfjob.yaml",
            "llama-3-70b/vllm/disagg-single-node/aiperfjob.yaml",
            "qwen3-235b-a22b-fp8/trtllm/agg/aiperfjob.yaml",
            "qwen3-235b-a22b-fp8/trtllm/disagg/aiperfjob.yaml",
            "qwen3-32b-fp8/trtllm/agg/aiperfjob.yaml",
            "qwen3-32b-fp8/trtllm/disagg/aiperfjob.yaml",
            "qwen3-32b/vllm/agg-round-robin/aiperfjob.yaml",
            "qwen3-32b/vllm/disagg-kv-router/aiperfjob.yaml",
            "qwen3-vl-30b/vllm/agg-embedding-cache/aiperfjob.yaml",
        ]
        for path_str in expected:
            full_path = RECIPES_DIR / path_str
            assert full_path.exists(), f"Missing recipe: {path_str}"
