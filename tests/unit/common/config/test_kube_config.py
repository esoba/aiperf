# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.common.config.kube_config module."""

from pathlib import Path

import pytest
from pytest import param

from aiperf.common.config.kube_config import (
    KubeManageOptions,
    KubeOptions,
    SecretMountConfig,
)

# =============================================================================
# SecretMountConfig Tests
# =============================================================================


class TestSecretMountConfig:
    """Tests for SecretMountConfig model."""

    def test_required_fields(self) -> None:
        """Test SecretMountConfig requires name and mount_path."""
        config = SecretMountConfig(name="my-secret", mount_path="/secrets/creds")

        assert config.name == "my-secret"
        assert config.mount_path == "/secrets/creds"
        assert config.sub_path is None

    def test_with_sub_path(self) -> None:
        """Test SecretMountConfig with sub_path."""
        config = SecretMountConfig(
            name="api-keys", mount_path="/secrets/api", sub_path="api_key"
        )

        assert config.name == "api-keys"
        assert config.mount_path == "/secrets/api"
        assert config.sub_path == "api_key"

    def test_missing_name_raises_error(self) -> None:
        """Test SecretMountConfig raises error when name is missing."""
        with pytest.raises(ValueError, match="name"):
            SecretMountConfig(mount_path="/secrets/creds")  # type: ignore

    def test_missing_mount_path_raises_error(self) -> None:
        """Test SecretMountConfig raises error when mount_path is missing."""
        with pytest.raises(ValueError, match="mount_path"):
            SecretMountConfig(name="my-secret")  # type: ignore


# =============================================================================
# KubeManageOptions Tests
# =============================================================================


class TestKubeManageOptions:
    """Tests for KubeManageOptions model."""

    def test_default_values(self) -> None:
        """Test KubeManageOptions has expected default values."""
        options = KubeManageOptions()

        assert options.kubeconfig is None
        assert options.namespace is None

    def test_custom_kubeconfig(self) -> None:
        """Test KubeManageOptions accepts custom kubeconfig."""
        options = KubeManageOptions(kubeconfig="/path/to/kubeconfig")

        assert options.kubeconfig == "/path/to/kubeconfig"

    def test_custom_namespace(self) -> None:
        """Test KubeManageOptions accepts custom namespace."""
        options = KubeManageOptions(namespace="my-namespace")

        assert options.namespace == "my-namespace"

    def test_both_fields_set(self) -> None:
        """Test KubeManageOptions with both fields set."""
        options = KubeManageOptions(
            kubeconfig="~/.kube/prod-config", namespace="production"
        )

        assert options.kubeconfig == "~/.kube/prod-config"
        assert options.namespace == "production"


# =============================================================================
# KubeOptions Inheritance Tests
# =============================================================================


class TestKubeOptionsInheritance:
    """Tests for KubeOptions inheritance from KubeManageOptions."""

    def test_kube_options_inherits_from_manage_options(self) -> None:
        """Test KubeOptions inherits from KubeManageOptions."""
        assert issubclass(KubeOptions, KubeManageOptions)

    def test_kube_options_has_manage_fields(self) -> None:
        """Test KubeOptions includes fields from KubeManageOptions."""
        options = KubeOptions(
            image="aiperf:latest", kubeconfig="/path", namespace="test"
        )

        assert options.kubeconfig == "/path"
        assert options.namespace == "test"
        assert options.image == "aiperf:latest"

    def test_kube_options_isinstance_of_manage_options(self) -> None:
        """Test KubeOptions instance is also instance of KubeManageOptions."""
        options = KubeOptions(image="aiperf:latest")

        assert isinstance(options, KubeManageOptions)

    def test_manage_options_fields_have_defaults_in_kube_options(self) -> None:
        """Test inherited fields have same defaults in KubeOptions."""
        options = KubeOptions(image="aiperf:latest")

        assert options.kubeconfig is None
        assert options.namespace is None


# =============================================================================
# KubeOptions Name Field Tests
# =============================================================================


class TestKubeOptionsNameField:
    """Tests for the KubeOptions name field validation."""

    @pytest.mark.parametrize(
        "name",
        [
            param("my-benchmark", id="simple-name"),
            param("a", id="single-char"),
            param("test-run-2024-01-15", id="with-date"),
            param("a" * 40, id="max-length"),
        ],
    )  # fmt: skip
    def test_name_valid_dns_label(self, name: str) -> None:
        """Test valid DNS label names are accepted."""
        options = KubeOptions(image="aiperf:latest", name=name)
        assert options.name == name

    def test_name_none_is_valid(self) -> None:
        """Test name=None is the default and valid."""
        options = KubeOptions(image="aiperf:latest")
        assert options.name is None

    @pytest.mark.parametrize(
        "name",
        [
            param("UPPERCASE", id="uppercase"),
            param("has spaces", id="spaces"),
            param("-starts-with-dash", id="leading-dash"),
            param("ends-with-dash-", id="trailing-dash"),
            param("has_underscores", id="underscores"),
            param("has.dots", id="dots"),
        ],
    )  # fmt: skip
    def test_name_invalid_raises(self, name: str) -> None:
        """Test invalid DNS label names are rejected."""
        with pytest.raises(ValueError, match="must be a valid DNS label"):
            KubeOptions(image="aiperf:latest", name=name)

    def test_name_too_long_raises(self) -> None:
        """Test names exceeding 40 chars are rejected."""
        with pytest.raises(ValueError, match="exceeds maximum length of 40"):
            KubeOptions(image="aiperf:latest", name="a" * 41)


# =============================================================================
# KubeOptions Edge Cases
# =============================================================================


class TestKubeOptionsEdgeCases:
    """Tests for KubeOptions boundary/edge-case values."""

    def test_workers_zero_accepted(self) -> None:
        """Test workers=0 is accepted (no minimum validation)."""
        options = KubeOptions(image="aiperf:latest", workers=0)
        assert options.workers == 0

    def test_workers_negative_accepted(self) -> None:
        """Test workers=-1 is accepted (no minimum validation)."""
        options = KubeOptions(image="aiperf:latest", workers=-1)
        assert options.workers == -1

    def test_empty_image_accepted(self) -> None:
        """Test image='' is accepted (no non-empty validation)."""
        options = KubeOptions(image="")
        assert options.image == ""

    def test_ttl_seconds_zero_accepted(self) -> None:
        """Test ttl_seconds=0 is accepted (immediate cleanup)."""
        options = KubeOptions(image="aiperf:latest", ttl_seconds=0)
        assert options.ttl_seconds == 0


# =============================================================================
# KubeOptions Tests - Default Values
# =============================================================================


class TestKubeOptionsDefaults:
    """Tests for KubeOptions default values."""

    def test_image_is_required(self) -> None:
        """Test KubeOptions requires image field."""
        with pytest.raises(ValueError, match="image"):
            KubeOptions()  # type: ignore

    def test_default_values(self) -> None:
        """Test KubeOptions has expected default values."""
        options = KubeOptions(image="aiperf:latest")

        assert options.image == "aiperf:latest"
        assert options.namespace is None
        assert options.workers == 10  # Default workers per pod
        assert options.ttl_seconds == 300
        assert options.node_selector == {}
        assert options.tolerations == []
        assert options.annotations == {}
        assert options.labels == {}
        assert options.image_pull_secrets == []
        assert options.env_vars == {}
        assert options.env_from_secrets == {}
        assert options.secret_mounts == []
        assert options.service_account is None


# =============================================================================
# KubeOptions Tests - Custom Values
# =============================================================================


class TestKubeOptionsCustomValues:
    """Tests for KubeOptions with custom values."""

    def test_custom_image(self) -> None:
        """Test KubeOptions accepts custom image."""
        options = KubeOptions(image="myregistry.io/aiperf:v2.0.0")

        assert options.image == "myregistry.io/aiperf:v2.0.0"

    def test_custom_namespace(self) -> None:
        """Test KubeOptions accepts custom namespace."""
        options = KubeOptions(image="aiperf:latest", namespace="benchmark-ns")

        assert options.namespace == "benchmark-ns"

    def test_custom_workers(self) -> None:
        """Test KubeOptions accepts custom worker count."""
        options = KubeOptions(image="aiperf:latest", workers=100)

        assert options.workers == 100

    def test_custom_ttl_seconds(self) -> None:
        """Test KubeOptions accepts custom TTL."""
        options = KubeOptions(image="aiperf:latest", ttl_seconds=600)

        assert options.ttl_seconds == 600

    def test_ttl_seconds_none_disables_ttl(self) -> None:
        """Test KubeOptions with None TTL disables auto-cleanup."""
        options = KubeOptions(image="aiperf:latest", ttl_seconds=None)

        assert options.ttl_seconds is None

    def test_custom_node_selector(self) -> None:
        """Test KubeOptions accepts custom node selector."""
        options = KubeOptions(
            image="aiperf:latest",
            node_selector={"gpu": "true", "zone": "us-west-1"},
        )

        assert options.node_selector == {"gpu": "true", "zone": "us-west-1"}

    def test_custom_tolerations(self) -> None:
        """Test KubeOptions accepts custom tolerations."""
        tolerations = [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
            {
                "key": "dedicated",
                "operator": "Equal",
                "value": "ml",
                "effect": "NoSchedule",
            },
        ]
        options = KubeOptions(image="aiperf:latest", tolerations=tolerations)

        assert options.tolerations == tolerations
        assert len(options.tolerations) == 2

    def test_custom_annotations(self) -> None:
        """Test KubeOptions accepts custom annotations."""
        options = KubeOptions(
            image="aiperf:latest",
            annotations={
                "prometheus.io/scrape": "true",
                "prometheus.io/port": "9090",
            },
        )

        assert options.annotations == {
            "prometheus.io/scrape": "true",
            "prometheus.io/port": "9090",
        }

    def test_custom_labels(self) -> None:
        """Test KubeOptions accepts custom labels."""
        options = KubeOptions(
            image="aiperf:latest",
            labels={"team": "ml-platform", "environment": "benchmark"},
        )

        assert options.labels == {"team": "ml-platform", "environment": "benchmark"}

    def test_custom_image_pull_secrets(self) -> None:
        """Test KubeOptions accepts custom image pull secrets."""
        options = KubeOptions(
            image="aiperf:latest",
            image_pull_secrets=["registry-creds", "backup-creds"],
        )

        assert options.image_pull_secrets == ["registry-creds", "backup-creds"]

    def test_custom_env_vars(self) -> None:
        """Test KubeOptions accepts custom environment variables."""
        options = KubeOptions(
            image="aiperf:latest",
            env_vars={"DEBUG": "true", "LOG_LEVEL": "DEBUG"},
        )

        assert options.env_vars == {"DEBUG": "true", "LOG_LEVEL": "DEBUG"}

    def test_custom_env_from_secrets(self) -> None:
        """Test KubeOptions accepts custom env vars from secrets."""
        options = KubeOptions(
            image="aiperf:latest",
            env_from_secrets={
                "API_KEY": "api-secret/key",
                "TOKEN": "auth-secret/token",
            },
        )

        assert options.env_from_secrets == {
            "API_KEY": "api-secret/key",
            "TOKEN": "auth-secret/token",
        }

    def test_custom_secret_mounts(self) -> None:
        """Test KubeOptions accepts custom secret mounts."""
        options = KubeOptions(
            image="aiperf:latest",
            secret_mounts=[
                SecretMountConfig(name="api-creds", mount_path="/secrets/api"),
                SecretMountConfig(
                    name="tls-certs", mount_path="/certs/tls", sub_path="cert.pem"
                ),
            ],
        )

        assert len(options.secret_mounts) == 2
        assert options.secret_mounts[0].name == "api-creds"
        assert options.secret_mounts[1].sub_path == "cert.pem"

    def test_custom_service_account(self) -> None:
        """Test KubeOptions accepts custom service account."""
        options = KubeOptions(image="aiperf:latest", service_account="aiperf-runner")

        assert options.service_account == "aiperf-runner"


# =============================================================================
# KubeOptions Tests - Full Configuration
# =============================================================================


class TestKubeOptionsFullConfiguration:
    """Tests for KubeOptions with all fields populated."""

    def test_all_fields_populated(self) -> None:
        """Test KubeOptions with all fields set."""
        options = KubeOptions(
            image="myregistry.io/aiperf:v2.0.0",
            namespace="aiperf-benchmark",
            workers=50,
            ttl_seconds=1800,
            node_selector={"gpu": "a100", "zone": "us-central1-a"},
            tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists"}],
            annotations={"owner": "ml-team"},
            labels={"project": "llm-benchmark"},
            image_pull_secrets=["registry-creds"],
            env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            env_from_secrets={"HF_TOKEN": "hf-secret/token"},
            secret_mounts=[
                SecretMountConfig(name="api-keys", mount_path="/secrets/api")
            ],
            service_account="benchmark-sa",
        )

        assert options.image == "myregistry.io/aiperf:v2.0.0"
        assert options.namespace == "aiperf-benchmark"
        assert options.workers == 50
        assert options.ttl_seconds == 1800
        assert options.node_selector == {"gpu": "a100", "zone": "us-central1-a"}
        assert len(options.tolerations) == 1
        assert options.annotations == {"owner": "ml-team"}
        assert options.labels == {"project": "llm-benchmark"}
        assert options.image_pull_secrets == ["registry-creds"]
        assert options.env_vars == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        assert options.env_from_secrets == {"HF_TOKEN": "hf-secret/token"}
        assert len(options.secret_mounts) == 1
        assert options.service_account == "benchmark-sa"


# =============================================================================
# KubeOptions Tests - Serialization
# =============================================================================


class TestKubeOptionsSerialization:
    """Tests for KubeOptions serialization."""

    def test_model_dump_minimal(self) -> None:
        """Test model_dump with minimal configuration."""
        options = KubeOptions(image="aiperf:latest")
        data = options.model_dump()

        assert data["image"] == "aiperf:latest"
        assert data["namespace"] is None
        assert data["workers"] == 10  # Default workers per pod
        assert data["ttl_seconds"] == 300
        assert data["node_selector"] == {}
        assert data["tolerations"] == []

    def test_model_dump_with_secret_mounts(self) -> None:
        """Test model_dump includes nested secret mounts."""
        options = KubeOptions(
            image="aiperf:latest",
            secret_mounts=[
                SecretMountConfig(name="test", mount_path="/test", sub_path="key")
            ],
        )
        data = options.model_dump()

        assert len(data["secret_mounts"]) == 1
        assert data["secret_mounts"][0]["name"] == "test"
        assert data["secret_mounts"][0]["mount_path"] == "/test"
        assert data["secret_mounts"][0]["sub_path"] == "key"

    def test_roundtrip_serialization(self) -> None:
        """Test KubeOptions can be serialized and deserialized."""
        original = KubeOptions(
            image="aiperf:v1",
            namespace="test-ns",
            workers=5,
            node_selector={"gpu": "true"},
        )

        data = original.model_dump()
        restored = KubeOptions(**data)

        assert restored.image == original.image
        assert restored.namespace == original.namespace
        assert restored.workers == original.workers
        assert restored.node_selector == original.node_selector


# =============================================================================
# load_kube_options Tests
# =============================================================================


class TestLoadKubeOptions:
    """Tests for load_kube_options function."""

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        """Test loading KubeOptions from YAML file."""
        from aiperf.common.config.loader import load_kube_options

        yaml_content = """
image: aiperf:v2.0.0
namespace: benchmarks
workers: 25
ttl_seconds: 600
node_selector:
  gpu: "true"
"""
        config_file = tmp_path / "kube.yaml"
        config_file.write_text(yaml_content)

        options = load_kube_options(config_file)

        assert options.image == "aiperf:v2.0.0"
        assert options.namespace == "benchmarks"
        assert options.workers == 25
        assert options.ttl_seconds == 600
        assert options.node_selector == {"gpu": "true"}

    def test_load_from_json_file(self, tmp_path: Path) -> None:
        """Test loading KubeOptions from JSON file."""
        from aiperf.common.config.loader import load_kube_options

        json_content = """{
    "image": "aiperf:latest",
    "workers": 10,
    "env_vars": {"DEBUG": "true"}
}"""
        config_file = tmp_path / "kube.json"
        config_file.write_text(json_content)

        options = load_kube_options(config_file)

        assert options.image == "aiperf:latest"
        assert options.workers == 10
        assert options.env_vars == {"DEBUG": "true"}

    def test_load_with_tolerations(self, tmp_path: Path) -> None:
        """Test loading KubeOptions with tolerations."""
        from aiperf.common.config.loader import load_kube_options

        yaml_content = """
image: aiperf:latest
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  - key: dedicated
    operator: Equal
    value: ml
    effect: NoSchedule
"""
        config_file = tmp_path / "kube.yaml"
        config_file.write_text(yaml_content)

        options = load_kube_options(config_file)

        assert len(options.tolerations) == 2
        assert options.tolerations[0]["key"] == "nvidia.com/gpu"
        assert options.tolerations[1]["value"] == "ml"

    def test_load_with_secret_mounts(self, tmp_path: Path) -> None:
        """Test loading KubeOptions with secret mounts."""
        from aiperf.common.config.loader import load_kube_options

        yaml_content = """
image: aiperf:latest
secret_mounts:
  - name: api-creds
    mount_path: /secrets/api
  - name: tls-certs
    mount_path: /certs/tls
    sub_path: cert.pem
"""
        config_file = tmp_path / "kube.yaml"
        config_file.write_text(yaml_content)

        options = load_kube_options(config_file)

        assert len(options.secret_mounts) == 2
        assert options.secret_mounts[0].name == "api-creds"
        assert options.secret_mounts[0].sub_path is None
        assert options.secret_mounts[1].sub_path == "cert.pem"

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Test load_kube_options raises error when file not found."""
        from aiperf.common.config.loader import load_kube_options

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_kube_options(tmp_path / "nonexistent.yaml")

    def test_load_unsupported_format(self, tmp_path: Path) -> None:
        """Test load_kube_options raises error for unsupported format."""
        from aiperf.common.config.loader import load_kube_options

        config_file = tmp_path / "kube.txt"
        config_file.write_text("image: aiperf:latest")

        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_kube_options(config_file)

    def test_load_missing_required_field(self, tmp_path: Path) -> None:
        """Test load_kube_options raises error when image is missing."""
        from aiperf.common.config.loader import load_kube_options

        yaml_content = """
workers: 10
namespace: test
"""
        config_file = tmp_path / "kube.yaml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="image"):
            load_kube_options(config_file)

    def test_load_yml_extension(self, tmp_path: Path) -> None:
        """Test loading KubeOptions from .yml file."""
        from aiperf.common.config.loader import load_kube_options

        yaml_content = """
image: aiperf:latest
workers: 5
"""
        config_file = tmp_path / "kube.yml"
        config_file.write_text(yaml_content)

        options = load_kube_options(config_file)

        assert options.image == "aiperf:latest"
        assert options.workers == 5
