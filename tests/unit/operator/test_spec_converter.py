# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.operator.spec_converter module."""

from typing import Any

import pytest

from aiperf.operator.spec_converter import (
    DEFAULT_CONNECTIONS_PER_WORKER,
    AIPerfJobSpecConverter,
)


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


class TestToAIPerfConfig:
    """Tests for to_aiperf_config method."""

    def test_minimal_config_models(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test converting minimal spec produces correct models."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()

        assert config.get_model_names() == ["test-model"]

    def test_minimal_config_endpoint(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test converting minimal spec produces correct endpoint."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()

        assert "http://localhost:8000/v1/chat/completions" in config.endpoint.urls

    def test_sets_cli_command(self, minimal_aiperfjob_spec: dict[str, Any]) -> None:
        """Test that cli_command is set for traceability."""
        converter = AIPerfJobSpecConverter(minimal_aiperfjob_spec, "my-job", "my-ns")
        config = converter.to_aiperf_config()

        assert "kubectl apply" in config.artifacts.cli_command
        assert "my-job" in config.artifacts.cli_command

    def test_sets_artifact_directory(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test that artifact directory is set to /results."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        config = converter.to_aiperf_config()

        assert str(config.artifacts.dir) == "/results"

    def test_full_config_models(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        """Test converting full spec produces correct models."""
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        config = converter.to_aiperf_config()

        assert config.get_model_names() == ["meta-llama/Llama-3.1-8B-Instruct"]

    def test_full_config_endpoint(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        """Test converting full spec produces correct endpoint."""
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        config = converter.to_aiperf_config()

        assert "http://api.example.com/v1/chat/completions" in config.endpoint.urls
        assert config.endpoint.streaming is True

    def test_full_config_load_phases(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        """Test converting full spec produces correct load phases."""
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        config = converter.to_aiperf_config()

        assert "profiling" in config.load
        phase = config.load["profiling"]
        assert phase.concurrency == 500
        assert phase.requests == 1000

    def test_full_config_datasets(self, full_aiperfjob_spec: dict[str, Any]) -> None:
        """Test converting full spec produces correct datasets."""
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        config = converter.to_aiperf_config()

        assert "main" in config.datasets
        ds = config.datasets["main"]
        assert ds.type == "synthetic"
        assert ds.entries == 1000


class TestToLegacyConfigs:
    """Tests for to_legacy_configs method."""

    def test_legacy_configs_returns_tuple(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test to_legacy_configs returns a (UserConfig, ServiceConfig) tuple."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        result = converter.to_legacy_configs()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_legacy_user_config_model_names(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test legacy UserConfig has correct model names."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        user_config, _ = converter.to_legacy_configs()

        assert user_config.endpoint.model_names == ["test-model"]

    def test_legacy_user_config_endpoint_url(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test legacy UserConfig has correct endpoint URL."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        user_config, _ = converter.to_legacy_configs()

        assert "http://localhost:8000/v1/chat/completions" in user_config.endpoint.urls

    def test_legacy_service_config_type(
        self, minimal_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test legacy ServiceConfig is produced."""
        converter = AIPerfJobSpecConverter(
            minimal_aiperfjob_spec, "test-job", "default"
        )
        _, service_config = converter.to_legacy_configs()

        from aiperf.common.config import ServiceConfig

        assert isinstance(service_config, ServiceConfig)

    def test_full_spec_legacy_configs(
        self, full_aiperfjob_spec: dict[str, Any]
    ) -> None:
        """Test legacy configs from full spec have correct values."""
        converter = AIPerfJobSpecConverter(full_aiperfjob_spec, "test-job", "default")
        user_config, _ = converter.to_legacy_configs()

        assert user_config.endpoint.model_names == ["meta-llama/Llama-3.1-8B-Instruct"]
        assert "http://api.example.com/v1/chat/completions" in user_config.endpoint.urls


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
            "models": ["test"],
            "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
            "datasets": {
                "main": {"type": "synthetic", "entries": 100, "prompts": {"isl": 128}}
            },
            "load": {
                "default": {
                    "type": "concurrency",
                    "concurrency": concurrency,
                    "requests": 10,
                }
            },
        }
        converter = AIPerfJobSpecConverter(spec, "test-job", "default")
        workers = converter.calculate_workers()

        assert workers == expected

    def test_minimum_one_worker(self) -> None:
        """Test that at least 1 worker is always returned."""
        spec = {
            "connectionsPerWorker": 1000,
            "models": ["test"],
            "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
            "datasets": {
                "main": {"type": "synthetic", "entries": 100, "prompts": {"isl": 128}}
            },
            "load": {
                "default": {"type": "concurrency", "concurrency": 1, "requests": 10}
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
