# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.operator.spec_converter module."""

from typing import Any

import pytest

from aiperf.operator.spec_converter import (
    DEFAULT_CONNECTIONS_PER_WORKER,
    AIPerfJobSpecConverter,
)
from aiperf.plugin.enums import ServiceRunType, UIType


class TestAIPerfJobSpecConverterInit:
    """Tests for AIPerfJobSpecConverter initialization."""

    def test_init_stores_parameters(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test converter stores spec, name, and namespace."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "test-ns"
        )
        assert converter.spec == minimal_aiperfjob_spec
        assert converter.name == "test-job"
        assert converter.namespace == "test-ns"

    def test_job_id_defaults_to_name(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test job_id defaults to name when not provided."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "test-ns"
        )
        assert converter.job_id == "test-job"

    def test_job_id_can_be_overridden(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test job_id can be overridden via parameter."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "test-ns", job_id="custom-id-abc123"
        )
        assert converter.job_id == "custom-id-abc123"
        assert converter.name == "test-job"  # Name still stored separately

    def test_default_connections_per_worker(self) -> None:
        """Test default connections per worker constant."""
        assert DEFAULT_CONNECTIONS_PER_WORKER == 500


class TestToUserConfig:
    """Tests for to_user_config method."""

    def test_minimal_user_config(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        """Test converting minimal spec to UserConfig."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        user_config = converter.to_user_config()

        assert user_config.endpoint.model_names == ["test-model"]

    def test_sets_cli_command(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        """Test that cli_command is set for traceability."""
        converter = AIPerfJobSpecConverter(minimal_aiperfjob_spec, "my-job", "my-ns")
        user_config = converter.to_user_config()

        assert "kubectl apply" in user_config.cli_command
        assert "my-job" in user_config.cli_command

    def test_sets_artifact_directory(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test that artifact_directory is set to /results."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        user_config = converter.to_user_config()

        assert user_config.output.artifact_directory.as_posix() == "/results"

    def test_full_user_config(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        """Test converting full spec to UserConfig."""
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        user_config = converter.to_user_config()

        assert user_config.endpoint.model_names == ["gpt-4"]
        assert "http://api.example.com" in user_config.endpoint.urls
        assert user_config.loadgen.concurrency == 500
        assert user_config.loadgen.request_count == 1000
        assert user_config.loadgen.warmup_request_count == 50


class TestToServiceConfig:
    """Tests for to_service_config method."""

    def test_service_config_kubernetes_mode(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test ServiceConfig is set for Kubernetes mode."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        service_config = converter.to_service_config()

        assert service_config.service_run_type == ServiceRunType.KUBERNETES
        assert service_config.ui_type == UIType.SIMPLE

    def test_service_config_has_zmq_dual(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test ServiceConfig has ZMQ dual-bind config."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        service_config = converter.to_service_config()

        assert service_config.zmq_dual is not None
        assert service_config.zmq_dual.ipc_path.as_posix() == "/aiperf/ipc"

    def test_service_config_api_settings(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test ServiceConfig has correct API settings."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "my-namespace"
        )
        service_config = converter.to_service_config()

        assert service_config.api_host == "0.0.0.0"
        assert service_config.api_port == 9090  # Default from K8sEnvironment

    def test_service_config_dataset_api_url(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test ServiceConfig has correct dataset API URL."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "my-namespace"
        )
        service_config = converter.to_service_config()

        # URL should include job_id (defaults to name), namespace, and port
        assert "test-job" in service_config.dataset_api_base_url
        assert "my-namespace" in service_config.dataset_api_base_url
        assert "/api/dataset" in service_config.dataset_api_base_url

    def test_service_config_dataset_api_url_uses_job_id(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test ServiceConfig dataset API URL uses job_id when provided."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec,
            "original-name",
            "my-namespace",
            job_id="unique-abc123",
        )
        service_config = converter.to_service_config()

        # URL should use job_id, not the CR name
        assert "unique-abc123" in service_config.dataset_api_base_url
        assert "original-name" not in service_config.dataset_api_base_url
        assert "aiperf-unique-abc123" in service_config.dataset_api_base_url


class TestToPodCustomization:
    """Tests for to_pod_customization method."""

    def test_empty_pod_template(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        """Test pod customization with no podTemplate."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        custom = converter.to_pod_customization()

        assert custom.node_selector == {}
        assert custom.tolerations == []
        assert custom.annotations == {}
        assert custom.labels == {}
        assert custom.image_pull_secrets == []
        assert custom.env_vars == {}
        assert custom.env_from_secrets == {}
        assert custom.secret_mounts == []
        assert custom.service_account is None

    def test_full_pod_customization(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        """Test pod customization with full podTemplate."""
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        custom = converter.to_pod_customization()

        assert custom.node_selector == {"gpu": "true"}
        assert len(custom.tolerations) == 1
        assert custom.tolerations[0]["key"] == "nvidia.com/gpu"
        assert custom.annotations == {"prometheus.io/scrape": "true"}
        assert custom.labels == {"team": "ml-platform"}
        assert custom.image_pull_secrets == ["my-registry-secret"]
        assert custom.service_account == "aiperf-sa"

    def test_env_vars_direct(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        """Test converting direct env vars."""
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        custom = converter.to_pod_customization()

        assert custom.env_vars["DEBUG"] == "true"

    def test_env_vars_from_secrets(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        """Test converting env vars from secrets."""
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        custom = converter.to_pod_customization()

        assert custom.env_from_secrets["API_KEY"] == "api-secrets/api-key"

    def test_secret_mounts(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        """Test converting secret volume mounts."""
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        custom = converter.to_pod_customization()

        assert len(custom.secret_mounts) == 1
        assert custom.secret_mounts[0].name == "my-creds"
        assert custom.secret_mounts[0].mount_path == "/etc/creds"


class TestCalculateWorkers:
    """Tests for calculate_workers method."""

    def test_default_concurrency(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        """Test worker calculation with default concurrency."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        workers = converter.calculate_workers()

        # Default concurrency is 1, should give 1 worker
        assert workers == 1

    def test_high_concurrency(
        self, aiperfjob_spec_high_concurrency: dict[str, Any]
    ) -> None:
        """Test worker calculation with high concurrency."""
        converter = AIPerfJobSpecConverter(
            aiperfjob_spec_high_concurrency, "test-job", "default"
        )
        workers = converter.calculate_workers()

        # 1000 concurrency / 100 connections per worker = 10 workers
        assert workers == 10

    @pytest.mark.parametrize(
        "concurrency,connections_per_worker,expected",
        [
            (1, 500, 1),  # Minimum 1 worker
            (500, 500, 1),  # Exactly 1 worker
            (501, 500, 2),  # Rounds up to 2 workers
            (1000, 100, 10),  # 10 workers
            (1001, 100, 11),  # Rounds up to 11 workers
            (50, 100, 1),  # Less than 1 rounds to 1
        ],  # fmt: skip
    )
    def test_worker_calculation_formula(
        self, concurrency: int, connections_per_worker: int, expected: int
    ) -> None:
        """Test worker calculation with various parameters."""
        spec = {
            "connectionsPerWorker": connections_per_worker,
            "userConfig": {
                "endpoint": {"model_names": ["test"]},
                "loadgen": {"concurrency": concurrency},
            },
        }
        converter = AIPerfJobSpecConverter(spec, "test-job", "default")
        workers = converter.calculate_workers()

        assert workers == expected

    def test_minimum_one_worker(self) -> None:
        """Test that at least 1 worker is always returned."""
        spec = {
            "connectionsPerWorker": 1000,
            "userConfig": {
                "endpoint": {"model_names": ["test"]},
                "loadgen": {"concurrency": 1},  # Very low concurrency
            },
        }
        converter = AIPerfJobSpecConverter(spec, "test-job", "default")
        workers = converter.calculate_workers()

        assert workers >= 1


class TestConvertEnvVars:
    """Tests for _convert_env_vars helper method."""

    def test_empty_list(self) -> None:
        """Test converting empty env list."""
        converter = AIPerfJobSpecConverter({}, "test", "default")
        result = converter._convert_env_vars([])
        assert result == {}

    def test_direct_values_only(self) -> None:
        """Test converting only direct values."""
        converter = AIPerfJobSpecConverter({}, "test", "default")
        env_list = [
            {"name": "FOO", "value": "bar"},
            {"name": "BAZ", "value": "qux"},
        ]
        result = converter._convert_env_vars(env_list)

        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_ignores_value_from(self) -> None:
        """Test that valueFrom references are ignored."""
        converter = AIPerfJobSpecConverter({}, "test", "default")
        env_list = [
            {"name": "FOO", "value": "bar"},
            {
                "name": "SECRET",
                "valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}},
            },
        ]
        result = converter._convert_env_vars(env_list)

        assert "FOO" in result
        assert "SECRET" not in result


class TestConvertEnvFromSecrets:
    """Tests for _convert_env_from_secrets helper method."""

    def test_empty_list(self) -> None:
        """Test converting empty env list."""
        converter = AIPerfJobSpecConverter({}, "test", "default")
        result = converter._convert_env_from_secrets([])
        assert result == {}

    def test_secret_refs_only(self) -> None:
        """Test converting only secret references."""
        converter = AIPerfJobSpecConverter({}, "test", "default")
        env_list = [
            {
                "name": "API_KEY",
                "valueFrom": {"secretKeyRef": {"name": "my-secret", "key": "api-key"}},
            },
            {
                "name": "TOKEN",
                "valueFrom": {"secretKeyRef": {"name": "auth", "key": "token"}},
            },
        ]
        result = converter._convert_env_from_secrets(env_list)

        assert result == {
            "API_KEY": "my-secret/api-key",
            "TOKEN": "auth/token",
        }

    def test_ignores_direct_values(self) -> None:
        """Test that direct values are ignored."""
        converter = AIPerfJobSpecConverter({}, "test", "default")
        env_list = [
            {"name": "FOO", "value": "bar"},
            {
                "name": "SECRET",
                "valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}},
            },
        ]
        result = converter._convert_env_from_secrets(env_list)

        assert "FOO" not in result
        assert "SECRET" in result


class TestConvertSecretMounts:
    """Tests for _convert_secret_mounts helper method."""

    def test_empty_lists(self) -> None:
        """Test converting empty volume lists."""
        converter = AIPerfJobSpecConverter({}, "test", "default")
        result = converter._convert_secret_mounts([], [])
        assert result == []

    def test_secret_volume_with_mount(self) -> None:
        """Test converting secret volume with matching mount."""
        converter = AIPerfJobSpecConverter({}, "test", "default")
        volumes = [
            {"name": "my-vol", "secret": {"secretName": "my-secret"}},
        ]
        mounts = [
            {"name": "my-vol", "mountPath": "/etc/secrets"},
        ]
        result = converter._convert_secret_mounts(volumes, mounts)

        assert len(result) == 1
        assert result[0].name == "my-secret"
        assert result[0].mount_path == "/etc/secrets"
        assert result[0].sub_path is None

    def test_secret_mount_with_subpath(self) -> None:
        """Test converting secret mount with subPath."""
        converter = AIPerfJobSpecConverter({}, "test", "default")
        volumes = [
            {"name": "tls", "secret": {"secretName": "tls-secret"}},
        ]
        mounts = [
            {"name": "tls", "mountPath": "/etc/tls/cert.pem", "subPath": "tls.crt"},
        ]
        result = converter._convert_secret_mounts(volumes, mounts)

        assert len(result) == 1
        assert result[0].sub_path == "tls.crt"

    def test_ignores_non_secret_volumes(self) -> None:
        """Test that non-secret volumes are ignored."""
        converter = AIPerfJobSpecConverter({}, "test", "default")
        volumes = [
            {"name": "data", "emptyDir": {}},
            {"name": "config", "configMap": {"name": "my-config"}},
            {"name": "secret", "secret": {"secretName": "my-secret"}},
        ]
        mounts = [
            {"name": "data", "mountPath": "/data"},
            {"name": "config", "mountPath": "/config"},
            {"name": "secret", "mountPath": "/secret"},
        ]
        result = converter._convert_secret_mounts(volumes, mounts)

        # Only the secret volume should be converted
        assert len(result) == 1
        assert result[0].name == "my-secret"

    def test_unmatched_mount_ignored(self) -> None:
        """Test that mounts without matching secret volumes are ignored."""
        converter = AIPerfJobSpecConverter({}, "test", "default")
        volumes = [
            {"name": "secret", "secret": {"secretName": "my-secret"}},
        ]
        mounts = [
            {"name": "other", "mountPath": "/other"},  # No matching volume
            {"name": "secret", "mountPath": "/secret"},
        ]
        result = converter._convert_secret_mounts(volumes, mounts)

        assert len(result) == 1
        assert result[0].name == "my-secret"
