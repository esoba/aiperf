# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.runner module."""

from unittest.mock import AsyncMock, MagicMock, patch

import kr8s
import orjson
import pytest

from aiperf.common.config import ServiceConfig
from aiperf.common.config.kube_config import KubeOptions, SecretMountConfig
from aiperf.common.environment import Environment
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.runner import (
    _BACKOFF_MULTIPLIER,
    _INITIAL_BACKOFF_SEC,
    _MAX_RETRIES,
    _RETRYABLE_STATUS_CODES,
    K8sDeploymentError,
    _apply_manifests,
    _format_api_error,
    _kube_options_to_pod_customization,
    _output_manifests_yaml,
    _with_retry,
    run_kubernetes_deployment,
)
from aiperf.plugin.enums import CommunicationBackend, ServiceRunType, UIType
from tests.unit.kubernetes.conftest import create_server_error

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock()
    return logger


@pytest.fixture
def mock_server_error():
    """Factory for creating kr8s ServerError instances."""

    def _create_error(status_code: int, reason: str = "Error"):
        return create_server_error(status_code, reason)

    return _create_error


@pytest.fixture
def mock_kr8s_apply():
    """Mock get_api and kr8s resource classes for _apply_manifests tests."""
    mock_api = MagicMock()

    async def _get_api(**kwargs):
        return mock_api

    return mock_api, _get_api


# =============================================================================
# Retry Configuration Tests
# =============================================================================


class TestRetryConfiguration:
    """Tests for retry configuration constants."""

    def test_max_retries_value(self) -> None:
        """Test _MAX_RETRIES has expected value."""
        assert _MAX_RETRIES == 3

    def test_initial_backoff_value(self) -> None:
        """Test _INITIAL_BACKOFF_SEC has expected value."""
        assert _INITIAL_BACKOFF_SEC == 1.0

    def test_backoff_multiplier_value(self) -> None:
        """Test _BACKOFF_MULTIPLIER has expected value."""
        assert _BACKOFF_MULTIPLIER == 2.0

    @pytest.mark.parametrize(
        "status_code",
        [408, 429, 500, 502, 503, 504],
    )
    def test_retryable_status_codes(self, status_code: int) -> None:
        """Test _RETRYABLE_STATUS_CODES contains expected codes."""
        assert status_code in _RETRYABLE_STATUS_CODES

    @pytest.mark.parametrize(
        "status_code",
        [400, 401, 403, 404, 409, 422],
    )
    def test_non_retryable_status_codes(self, status_code: int) -> None:
        """Test _RETRYABLE_STATUS_CODES does not contain non-retryable codes."""
        assert status_code not in _RETRYABLE_STATUS_CODES


# =============================================================================
# _with_retry Function Tests
# =============================================================================


class TestWithRetry:
    """Tests for _with_retry function."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self, mock_logger) -> None:
        """Test operation succeeds on first attempt."""
        operation = AsyncMock(return_value="success")

        result = await _with_retry(operation, mock_logger, "test operation")

        assert result == "success"
        operation.assert_called_once()
        mock_logger.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_operation_result(self, mock_logger) -> None:
        """Test _with_retry returns the operation result."""
        expected = {"data": [1, 2, 3]}
        operation = AsyncMock(return_value=expected)

        result = await _with_retry(operation, mock_logger, "test operation")

        assert result == expected

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(
        self, mock_logger, mock_server_error
    ) -> None:
        """Test non-retryable errors raise immediately without retry."""
        operation = AsyncMock(side_effect=mock_server_error(404, "Not Found"))

        with pytest.raises(kr8s.ServerError) as exc_info:
            await _with_retry(operation, mock_logger, "test operation")

        assert exc_info.value.response.status_code == 404
        operation.assert_called_once()  # No retries
        mock_logger.warning.assert_not_called()

    @pytest.mark.parametrize(
        "status_code,reason",
        [
            (400, "Bad Request"),
            (401, "Unauthorized"),
            (403, "Forbidden"),
            (404, "Not Found"),
            (409, "Conflict"),
            (422, "Unprocessable Entity"),
        ],
    )
    @pytest.mark.asyncio
    async def test_client_errors_not_retried(
        self, status_code: int, reason: str, mock_logger, mock_server_error
    ) -> None:
        """Test client errors (4xx except 408/429) are not retried."""
        operation = AsyncMock(side_effect=mock_server_error(status_code, reason))

        with pytest.raises(kr8s.ServerError):
            await _with_retry(operation, mock_logger, "test operation")

        operation.assert_called_once()

    @pytest.mark.asyncio
    async def test_retryable_error_retries(
        self, mock_logger, mock_server_error
    ) -> None:
        """Test retryable errors trigger retry attempts."""

        # Fail twice with 503, then succeed
        operation = AsyncMock(
            side_effect=[
                mock_server_error(503, "Service Unavailable"),
                mock_server_error(503, "Service Unavailable"),
                "success",
            ]
        )

        result = await _with_retry(operation, mock_logger, "test operation")

        assert result == "success"
        assert operation.call_count == 3
        assert mock_logger.warning.call_count == 2

    @pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
    @pytest.mark.asyncio
    async def test_all_retryable_codes_trigger_retry(
        self, status_code: int, mock_logger, mock_server_error
    ) -> None:
        """Test all retryable status codes trigger retry."""
        # Fail once, then succeed
        operation = AsyncMock(
            side_effect=[mock_server_error(status_code, "Error"), "success"]
        )

        result = await _with_retry(operation, mock_logger, "test operation")

        assert result == "success"
        assert operation.call_count == 2
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, mock_logger, mock_server_error) -> None:
        """Test error is raised when max retries are exhausted."""
        # Always fail with retryable error
        operation = AsyncMock(side_effect=mock_server_error(503, "Service Unavailable"))

        with pytest.raises(kr8s.ServerError) as exc_info:
            await _with_retry(operation, mock_logger, "test operation")

        assert exc_info.value.response.status_code == 503
        assert operation.call_count == _MAX_RETRIES + 1  # Initial + retries
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_exponential_backoff(self, mock_logger, mock_server_error) -> None:
        """Test exponential backoff timing."""
        # Fail 3 times, then succeed
        operation = AsyncMock(
            side_effect=[
                mock_server_error(503, "Service Unavailable"),
                mock_server_error(503, "Service Unavailable"),
                mock_server_error(503, "Service Unavailable"),
                "success",
            ]
        )

        sleep_times: list[float] = []

        async def mock_sleep(t: float) -> None:
            sleep_times.append(t)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await _with_retry(operation, mock_logger, "test operation")

        # Verify exponential backoff: 1.0, 2.0, 4.0
        assert sleep_times == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_warning_log_includes_details(
        self, mock_logger, mock_server_error
    ) -> None:
        """Test warning log includes operation name and attempt info."""
        operation = AsyncMock(
            side_effect=[mock_server_error(503, "Service Unavailable"), "success"]
        )

        await _with_retry(operation, mock_logger, "create ConfigMap/test")

        warning_msg = mock_logger.warning.call_args[0][0]
        assert "create ConfigMap/test" in warning_msg
        assert "attempt 1" in warning_msg
        assert "503" in warning_msg

    @pytest.mark.asyncio
    async def test_error_log_on_exhausted_retries(
        self, mock_logger, mock_server_error
    ) -> None:
        """Test error log when retries exhausted includes details."""
        operation = AsyncMock(side_effect=mock_server_error(503, "Service Unavailable"))

        with pytest.raises(kr8s.ServerError):
            await _with_retry(operation, mock_logger, "create JobSet/bench")

        error_msg = mock_logger.error.call_args[0][0]
        assert "create JobSet/bench" in error_msg
        assert "4 attempts" in error_msg  # _MAX_RETRIES + 1


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestRetryWithRealExceptions:
    """Tests using real kr8s ServerError class."""

    @pytest.mark.asyncio
    async def test_with_real_server_error(self, mock_logger) -> None:
        """Test with real kr8s ServerError."""
        exc = create_server_error(503, "Service Unavailable")

        operation = AsyncMock(side_effect=[exc, "success"])

        result = await _with_retry(operation, mock_logger, "test op")

        assert result == "success"
        assert operation.call_count == 2


# =============================================================================
# _kube_options_to_pod_customization Tests
# =============================================================================


class TestKubeOptionsToPodCustomization:
    """Tests for _kube_options_to_pod_customization function."""

    def test_converts_basic_options(self) -> None:
        """Test basic conversion from KubeOptions to PodCustomization."""

        kube_options = KubeOptions(
            image="aiperf:latest",
            node_selector={"zone": "us-west"},
            tolerations=[{"key": "gpu", "operator": "Exists"}],
            annotations={"prometheus.io/scrape": "true"},
            labels={"team": "ml"},
            image_pull_secrets=["my-secret"],
            env_vars={"LOG_LEVEL": "DEBUG"},
            service_account="bench-sa",
        )

        pod_customization = _kube_options_to_pod_customization(kube_options)

        assert pod_customization.node_selector == {"zone": "us-west"}
        assert pod_customization.tolerations == [{"key": "gpu", "operator": "Exists"}]
        assert pod_customization.annotations == {"prometheus.io/scrape": "true"}
        assert pod_customization.labels == {"team": "ml"}
        assert pod_customization.image_pull_secrets == ["my-secret"]
        assert pod_customization.env_vars == {"LOG_LEVEL": "DEBUG"}
        assert pod_customization.service_account == "bench-sa"

    def test_converts_empty_options(self) -> None:
        """Test conversion with minimal KubeOptions."""

        kube_options = KubeOptions(image="aiperf:latest")

        pod_customization = _kube_options_to_pod_customization(kube_options)

        assert pod_customization.node_selector == {}
        assert pod_customization.tolerations == []
        assert pod_customization.annotations == {}
        assert pod_customization.labels == {}
        assert pod_customization.image_pull_secrets == []
        assert pod_customization.env_vars == {}
        assert pod_customization.env_from_secrets == {}
        assert pod_customization.secret_mounts == []
        assert pod_customization.service_account is None

    def test_converts_secret_mounts(self) -> None:
        """Test conversion of SecretMountConfig to SecretMount."""

        kube_options = KubeOptions(
            image="aiperf:latest",
            secret_mounts=[
                SecretMountConfig(name="api-keys", mount_path="/secrets/api"),
                SecretMountConfig(
                    name="tls-certs", mount_path="/certs/tls", sub_path="cert.pem"
                ),
            ],
        )

        pod_customization = _kube_options_to_pod_customization(kube_options)

        assert len(pod_customization.secret_mounts) == 2
        assert pod_customization.secret_mounts[0].name == "api-keys"
        assert pod_customization.secret_mounts[0].mount_path == "/secrets/api"
        assert pod_customization.secret_mounts[0].sub_path is None
        assert pod_customization.secret_mounts[1].name == "tls-certs"
        assert pod_customization.secret_mounts[1].sub_path == "cert.pem"

    def test_converts_env_from_secrets(self) -> None:
        """Test conversion of env_from_secrets."""

        kube_options = KubeOptions(
            image="aiperf:latest",
            env_from_secrets={"API_KEY": "secret/key", "TOKEN": "auth/token"},
        )

        pod_customization = _kube_options_to_pod_customization(kube_options)

        assert pod_customization.env_from_secrets == {
            "API_KEY": "secret/key",
            "TOKEN": "auth/token",
        }


# =============================================================================
# run_kubernetes_deployment Tests
# =============================================================================


class TestRunKubernetesDeployment:
    """Tests for run_kubernetes_deployment function."""

    @pytest.fixture
    def fresh_service_config(self):
        """Create a fresh service config to avoid state leakage between tests."""

        return ServiceConfig(
            service_run_type=ServiceRunType.MULTIPROCESSING,
            comm_backend=CommunicationBackend.ZMQ_IPC,
        )

    @pytest.mark.asyncio
    async def test_dry_run_outputs_yaml(
        self, sample_user_config, fresh_service_config, sample_aiperf_config, capsys
    ) -> None:
        """Test dry run mode outputs YAML manifests."""

        kube_options = KubeOptions(
            image="aiperf:latest",
            namespace="test-ns",
            workers=2,
        )

        job_id, namespace = await run_kubernetes_deployment(
            sample_user_config,
            fresh_service_config,
            kube_options,
            aiperf_config=sample_aiperf_config,
            dry_run=True,
        )

        # Check return values
        assert len(job_id) == 8  # UUID hex prefix
        assert namespace == "test-ns"

        # Check YAML was output (when namespace is specified, no Namespace manifest is created)
        captured = capsys.readouterr()
        assert "kind: Role" in captured.out
        assert "kind: ConfigMap" in captured.out
        assert "kind: JobSet" in captured.out

    @pytest.mark.asyncio
    async def test_dry_run_generates_namespace_when_not_specified(
        self, sample_user_config, fresh_service_config, sample_aiperf_config, capsys
    ) -> None:
        """Test dry run auto-generates namespace from job_id."""

        kube_options = KubeOptions(
            image="aiperf:latest",
            namespace=None,  # Not specified
            workers=2,
        )

        job_id, namespace = await run_kubernetes_deployment(
            sample_user_config,
            fresh_service_config,
            kube_options,
            aiperf_config=sample_aiperf_config,
            dry_run=True,
        )

        assert namespace == f"aiperf-{job_id}"

        # When namespace is auto-generated, a Namespace manifest IS created
        captured = capsys.readouterr()
        assert "kind: Namespace" in captured.out

    @pytest.mark.asyncio
    async def test_apply_mode_creates_resources(
        self, sample_user_config, fresh_service_config, sample_aiperf_config
    ) -> None:
        """Test apply mode creates Kubernetes resources."""

        kube_options = KubeOptions(
            image="aiperf:latest",
            namespace=None,  # Auto-generate namespace to trigger namespace creation
            workers=2,
        )

        mock_api = MagicMock()

        # Track which classes were instantiated and what .create was called on
        created_kinds: list[str] = []

        def make_mock_cls(kind_name):
            def cls_init(manifest, api=None):
                obj = MagicMock()
                obj.create = AsyncMock()
                created_kinds.append(kind_name)
                return obj

            return cls_init

        kind_mocks = {
            "Namespace": make_mock_cls("Namespace"),
            "ConfigMap": make_mock_cls("ConfigMap"),
            "Role": make_mock_cls("Role"),
            "RoleBinding": make_mock_cls("RoleBinding"),
            "JobSet": make_mock_cls("JobSet"),
        }

        with (
            patch(
                "aiperf.kubernetes.client.get_api",
                new=AsyncMock(return_value=mock_api),
            ),
            patch.dict("aiperf.kubernetes.runner._KIND_TO_CLASS", kind_mocks),
        ):
            job_id, namespace = await run_kubernetes_deployment(
                sample_user_config,
                fresh_service_config,
                kube_options,
                aiperf_config=sample_aiperf_config,
                dry_run=False,
            )

        # Verify resources were created (Namespace + ConfigMap(s) + Role + RoleBinding + JobSet)
        assert "Namespace" in created_kinds
        assert "ConfigMap" in created_kinds
        assert "JobSet" in created_kinds

    @pytest.mark.asyncio
    async def test_worker_scaling_single_pod(
        self, sample_user_config, fresh_service_config, sample_aiperf_config
    ) -> None:
        """Test that few workers results in single pod."""

        kube_options = KubeOptions(
            image="aiperf:latest",
            namespace="test-ns",
            workers=2,  # Less than default workers_per_pod
        )

        await run_kubernetes_deployment(
            sample_user_config,
            fresh_service_config,
            kube_options,
            aiperf_config=sample_aiperf_config,
            dry_run=True,
        )

        # workers_per_pod should be set to the total workers
        assert fresh_service_config.workers_per_pod == 2

    @pytest.mark.asyncio
    async def test_worker_scaling_multiple_pods(
        self, sample_user_config, fresh_service_config, sample_aiperf_config
    ) -> None:
        """Test that many workers results in multiple pods."""

        kube_options = KubeOptions(
            image="aiperf:latest",
            namespace="test-ns",
            workers=20,  # More than default workers_per_pod
        )

        await run_kubernetes_deployment(
            sample_user_config,
            fresh_service_config,
            kube_options,
            aiperf_config=sample_aiperf_config,
            dry_run=True,
        )

        # workers_per_pod should be the default
        assert (
            fresh_service_config.workers_per_pod
            == Environment.WORKER.DEFAULT_WORKERS_PER_POD
        )


# =============================================================================
# _output_manifests_yaml Tests
# =============================================================================


class TestOutputManifestsYaml:
    """Tests for _output_manifests_yaml function."""

    def test_outputs_single_manifest(self, capsys) -> None:
        """Test outputting a single manifest."""

        manifests = [
            {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "test"}}
        ]

        _output_manifests_yaml(manifests)

        captured = capsys.readouterr()
        assert "apiVersion: v1" in captured.out
        assert "kind: ConfigMap" in captured.out

    def test_outputs_multiple_manifests_with_separator(self, capsys) -> None:
        """Test outputting multiple manifests with --- separator."""

        manifests = [
            {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": "ns1"}},
            {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "cm1"}},
        ]

        _output_manifests_yaml(manifests)

        captured = capsys.readouterr()
        assert "---" in captured.out
        assert "kind: Namespace" in captured.out
        assert "kind: ConfigMap" in captured.out


# =============================================================================
# _apply_manifests Tests
# =============================================================================


class TestApplyManifests:
    """Tests for _apply_manifests function."""

    def _make_mock_cls(self):
        """Create a mock kr8s resource class that tracks .create() calls."""
        mock_obj = MagicMock()
        mock_obj.create = AsyncMock()

        def cls_init(manifest, api=None):
            return mock_obj

        cls_init._mock_obj = mock_obj
        return cls_init

    @pytest.mark.asyncio
    async def test_apply_namespace(self, mock_logger) -> None:
        """Test applying a Namespace manifest."""

        mock_cls = self._make_mock_cls()
        manifests = [{"kind": "Namespace", "metadata": {"name": "test-ns"}}]

        with (
            patch(
                "aiperf.kubernetes.client.get_api",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch.dict(
                "aiperf.kubernetes.runner._KIND_TO_CLASS", {"Namespace": mock_cls}
            ),
        ):
            await _apply_manifests(manifests, mock_logger)

        mock_cls._mock_obj.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_apply_configmap(self, mock_logger) -> None:
        """Test applying a ConfigMap manifest."""

        mock_cls = self._make_mock_cls()
        manifests = [
            {
                "kind": "ConfigMap",
                "metadata": {"name": "test-cm", "namespace": "default"},
            }
        ]

        with (
            patch(
                "aiperf.kubernetes.client.get_api",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch.dict(
                "aiperf.kubernetes.runner._KIND_TO_CLASS", {"ConfigMap": mock_cls}
            ),
        ):
            await _apply_manifests(manifests, mock_logger)

        mock_cls._mock_obj.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_apply_role(self, mock_logger) -> None:
        """Test applying a Role manifest."""

        mock_cls = self._make_mock_cls()
        manifests = [
            {
                "kind": "Role",
                "metadata": {"name": "test-role", "namespace": "default"},
            }
        ]

        with (
            patch(
                "aiperf.kubernetes.client.get_api",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch.dict("aiperf.kubernetes.runner._KIND_TO_CLASS", {"Role": mock_cls}),
        ):
            await _apply_manifests(manifests, mock_logger)

        mock_cls._mock_obj.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_apply_role_binding(self, mock_logger) -> None:
        """Test applying a RoleBinding manifest."""

        mock_cls = self._make_mock_cls()
        manifests = [
            {
                "kind": "RoleBinding",
                "metadata": {"name": "test-rb", "namespace": "default"},
            }
        ]

        with (
            patch(
                "aiperf.kubernetes.client.get_api",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch.dict(
                "aiperf.kubernetes.runner._KIND_TO_CLASS", {"RoleBinding": mock_cls}
            ),
        ):
            await _apply_manifests(manifests, mock_logger)

        mock_cls._mock_obj.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_apply_jobset(self, mock_logger) -> None:
        """Test applying a JobSet manifest."""

        mock_cls = self._make_mock_cls()
        manifests = [
            {
                "kind": "JobSet",
                "metadata": {"name": "test-jobset", "namespace": "default"},
            }
        ]

        with (
            patch(
                "aiperf.kubernetes.client.get_api",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch.dict("aiperf.kubernetes.runner._KIND_TO_CLASS", {"JobSet": mock_cls}),
        ):
            await _apply_manifests(manifests, mock_logger)

        mock_cls._mock_obj.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_apply_unknown_kind_logs_warning(self, mock_logger) -> None:
        """Test that unknown resource kind logs a warning."""

        manifests = [{"kind": "UnknownKind", "metadata": {"name": "test"}}]

        with patch(
            "aiperf.kubernetes.client.get_api",
            new=AsyncMock(return_value=MagicMock()),
        ):
            await _apply_manifests(manifests, mock_logger)

        mock_logger.warning.assert_called()
        assert "Unknown resource kind" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_apply_handles_409_conflict(self, mock_logger) -> None:
        """Test that 409 Conflict is handled as 'already exists'."""

        error = create_server_error(409, "Conflict")
        mock_obj = MagicMock()
        mock_obj.create = AsyncMock(side_effect=error)

        def mock_cls(manifest, api=None):
            return mock_obj

        manifests = [
            {
                "kind": "ConfigMap",
                "metadata": {"name": "existing-cm", "namespace": "default"},
            }
        ]

        with (
            patch(
                "aiperf.kubernetes.client.get_api",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch.dict(
                "aiperf.kubernetes.runner._KIND_TO_CLASS", {"ConfigMap": mock_cls}
            ),
        ):
            # Should not raise
            await _apply_manifests(manifests, mock_logger)

        mock_logger.warning.assert_called()
        assert "already exists" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_apply_raises_on_other_errors(self, mock_logger) -> None:
        """Test that non-409 errors are raised as K8sDeploymentError."""

        error = create_server_error(403, "Forbidden")
        mock_obj = MagicMock()
        mock_obj.create = AsyncMock(side_effect=error)

        def mock_cls(manifest, api=None):
            return mock_obj

        manifests = [
            {
                "kind": "ConfigMap",
                "metadata": {"name": "test-cm", "namespace": "default"},
            }
        ]

        with (
            patch(
                "aiperf.kubernetes.client.get_api",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch.dict(
                "aiperf.kubernetes.runner._KIND_TO_CLASS", {"ConfigMap": mock_cls}
            ),
            pytest.raises(K8sDeploymentError) as exc_info,
        ):
            await _apply_manifests(manifests, mock_logger)

        # Check that error message contains helpful info
        error_msg = str(exc_info.value)
        assert "ConfigMap/test-cm" in error_msg
        assert "namespace 'default'" in error_msg
        assert "permission" in error_msg.lower() or "rbac" in error_msg.lower()


# =============================================================================
# _format_api_error and K8sDeploymentError Tests
# =============================================================================


class TestFormatApiError:
    """Tests for _format_api_error function."""

    def test_format_403_forbidden(self, mock_server_error) -> None:
        """Test 403 error provides permission-related suggestion."""

        exc = mock_server_error(403, "Forbidden")

        msg = _format_api_error(exc, "ConfigMap", "my-config", "default")

        assert "ConfigMap/my-config" in msg
        assert "namespace 'default'" in msg
        assert "permission" in msg.lower() or "rbac" in msg.lower()

    def test_format_404_jobset_crd_missing(self, mock_server_error) -> None:
        """Test 404 error for JobSet suggests CRD installation."""

        exc = mock_server_error(404, "Not Found")

        msg = _format_api_error(exc, "JobSet", "my-jobset", "default")

        assert "JobSet/my-jobset" in msg
        assert "CRD" in msg
        assert "kubectl apply" in msg

    def test_format_404_other_resource(self, mock_server_error) -> None:
        """Test 404 error for non-JobSet resource."""

        exc = mock_server_error(404, "Not Found")

        msg = _format_api_error(exc, "ConfigMap", "missing", "default")

        assert "ConfigMap/missing" in msg
        assert "not found" in msg.lower()

    def test_format_422_validation_error(self, mock_server_error) -> None:
        """Test 422 error suggests using generate command."""

        exc = mock_server_error(422, "Unprocessable Entity")

        msg = _format_api_error(exc, "JobSet", "my-jobset", "default")

        assert "aiperf kube generate" in msg

    def test_format_401_auth_error(self, mock_server_error) -> None:
        """Test 401 error suggests checking credentials."""

        exc = mock_server_error(401, "Unauthorized")

        msg = _format_api_error(exc, "Namespace", "my-ns", None)

        assert "Namespace/my-ns" in msg
        assert "kubeconfig" in msg.lower() or "credential" in msg.lower()

    def test_format_includes_body_message(self) -> None:
        """Test error message includes details from API response body."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.reason_phrase = "Bad Request"
        mock_response.text = orjson.dumps(
            {"message": "spec.containers: Required value"}
        ).decode()
        exc = kr8s.ServerError("Bad Request", response=mock_response)

        msg = _format_api_error(exc, "Pod", "my-pod", "default")

        assert "spec.containers: Required value" in msg

    def test_format_without_namespace(self, mock_server_error) -> None:
        """Test formatting for cluster-scoped resources."""

        exc = mock_server_error(403, "Forbidden")

        msg = _format_api_error(exc, "Namespace", "my-ns", None)

        assert "Namespace/my-ns" in msg
        assert "namespace" not in msg.lower() or "Namespace/my-ns" in msg


class TestK8sDeploymentError:
    """Tests for K8sDeploymentError exception."""

    def test_is_exception(self) -> None:
        """Test K8sDeploymentError is an Exception."""

        assert issubclass(K8sDeploymentError, Exception)

    def test_can_be_raised_with_message(self) -> None:
        """Test K8sDeploymentError can be raised with a message."""

        with pytest.raises(K8sDeploymentError) as exc_info:
            raise K8sDeploymentError("Test error message")

        assert "Test error message" in str(exc_info.value)


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestFormatApiErrorEdgeCases:
    """Additional tests for _format_api_error edge cases."""

    def test_format_body_invalid_json(self) -> None:
        """Test handling of invalid JSON in error body."""

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.reason_phrase = "Bad Request"
        mock_response.text = "not valid json {"
        exc = kr8s.ServerError("Bad Request", response=mock_response)

        msg = _format_api_error(exc, "Pod", "my-pod", "default")

        # Should include truncated body as detail
        assert "not valid json" in msg

    def test_format_body_none_response(self) -> None:
        """Test handling of None response."""

        exc = kr8s.ServerError("Internal Server Error", response=None)

        msg = _format_api_error(exc, "Pod", "my-pod", "default")

        # Should not raise, should include status info
        assert "Pod/my-pod" in msg

    def test_format_body_empty_text(self) -> None:
        """Test handling of empty text in response."""

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.reason_phrase = "Internal Server Error"
        mock_response.text = ""
        exc = kr8s.ServerError("Internal Server Error", response=mock_response)

        msg = _format_api_error(exc, "Pod", "my-pod", "default")

        assert "Pod/my-pod" in msg

    def test_format_body_truncation_long_body(self) -> None:
        """Test that very long error bodies are truncated."""

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.reason_phrase = "Bad Request"
        mock_response.text = "x" * 500  # Long non-JSON body
        exc = kr8s.ServerError("Bad Request", response=mock_response)

        msg = _format_api_error(exc, "Pod", "my-pod", "default")

        # Body should be truncated to first 200 chars
        assert len(msg) < 1000

    def test_format_generic_status_code(self, mock_server_error) -> None:
        """Test handling of unrecognized status codes."""

        exc = mock_server_error(418, "I'm a teapot")

        msg = _format_api_error(exc, "Pod", "my-pod", "default")

        # Should include status code and reason
        assert "HTTP 418" in msg
        assert "I'm a teapot" in msg


class TestWithRetryAsync:
    """Tests for _with_retry with async operations."""

    @pytest.mark.asyncio
    async def test_async_operation_success(self, mock_logger) -> None:
        """Test _with_retry with async operation that succeeds."""

        async def async_operation():
            return "async_success"

        result = await _with_retry(async_operation, mock_logger, "async test")

        assert result == "async_success"
        mock_logger.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_operation_with_retry(
        self, mock_logger, mock_server_error
    ) -> None:
        """Test _with_retry with async operation that fails then succeeds."""

        attempt_count = 0

        async def async_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise mock_server_error(503, "Service Unavailable")
            return "async_success_after_retry"

        result = await _with_retry(async_operation, mock_logger, "async retry test")

        assert result == "async_success_after_retry"
        assert attempt_count == 2
        mock_logger.warning.assert_called_once()


class TestRunKubernetesDeploymentAdvanced:
    """Additional tests for run_kubernetes_deployment."""

    @pytest.fixture
    def fresh_service_config(self):
        """Create a fresh service config to avoid state leakage between tests."""

        return ServiceConfig(
            service_run_type=ServiceRunType.MULTIPROCESSING,
            comm_backend=CommunicationBackend.ZMQ_IPC,
        )

    @pytest.mark.asyncio
    async def test_preserves_existing_workers_per_pod(
        self, sample_user_config, fresh_service_config, sample_aiperf_config
    ) -> None:
        """Test that existing workers_per_pod config is respected."""

        # Set a custom workers_per_pod
        fresh_service_config.workers_per_pod = 5
        fresh_service_config.model_fields_set.add("workers_per_pod")

        kube_options = KubeOptions(
            image="aiperf:latest",
            namespace="test-ns",
            workers=15,  # 3 pods with 5 workers each
        )

        await run_kubernetes_deployment(
            sample_user_config,
            fresh_service_config,
            kube_options,
            aiperf_config=sample_aiperf_config,
            dry_run=True,
        )

        # workers_per_pod should use the custom value
        assert fresh_service_config.workers_per_pod == 5

    @pytest.mark.asyncio
    async def test_sets_api_config_for_kubernetes(
        self, sample_user_config, fresh_service_config, sample_aiperf_config
    ) -> None:
        """Test that API port and host are set correctly for Kubernetes."""

        kube_options = KubeOptions(
            image="aiperf:latest",
            namespace="test-ns",
            workers=2,
        )

        await run_kubernetes_deployment(
            sample_user_config,
            fresh_service_config,
            kube_options,
            aiperf_config=sample_aiperf_config,
            dry_run=True,
        )

        # API should be enabled
        assert fresh_service_config.api_port == K8sEnvironment.PORTS.API_SERVICE
        assert fresh_service_config.api_host == "0.0.0.0"

    @pytest.mark.asyncio
    async def test_sets_dataset_api_url(
        self, sample_user_config, fresh_service_config, sample_aiperf_config
    ) -> None:
        """Test that dataset API URL is constructed correctly."""

        kube_options = KubeOptions(
            image="aiperf:latest",
            namespace="test-ns",
            workers=2,
        )

        job_id, namespace = await run_kubernetes_deployment(
            sample_user_config,
            fresh_service_config,
            kube_options,
            aiperf_config=sample_aiperf_config,
            dry_run=True,
        )

        # Dataset API URL should use the controller DNS
        assert fresh_service_config.dataset_api_base_url is not None
        assert f"aiperf-{job_id}" in fresh_service_config.dataset_api_base_url
        assert "test-ns" in fresh_service_config.dataset_api_base_url

    @pytest.mark.asyncio
    async def test_apply_with_kubeconfig(
        self, sample_user_config, fresh_service_config, sample_aiperf_config
    ) -> None:
        """Test apply mode with custom kubeconfig path."""

        kube_options = KubeOptions(
            image="aiperf:latest",
            namespace="test-ns",
            workers=2,
            kubeconfig="/custom/path/kubeconfig",
        )

        mock_api = MagicMock()
        mock_get_api = AsyncMock(return_value=mock_api)

        def make_mock_cls():
            def cls_init(manifest, api=None):
                obj = MagicMock()
                obj.create = AsyncMock()
                return obj

            return cls_init

        kind_mocks = {
            "ConfigMap": make_mock_cls(),
            "Role": make_mock_cls(),
            "RoleBinding": make_mock_cls(),
            "JobSet": make_mock_cls(),
        }

        with (
            patch("aiperf.kubernetes.client.get_api", mock_get_api),
            patch.dict("aiperf.kubernetes.runner._KIND_TO_CLASS", kind_mocks),
        ):
            await run_kubernetes_deployment(
                sample_user_config,
                fresh_service_config,
                kube_options,
                aiperf_config=sample_aiperf_config,
                dry_run=False,
            )

        # Verify kubeconfig was passed to get_api
        mock_get_api.assert_called_once_with(
            kubeconfig="/custom/path/kubeconfig", kube_context=None
        )

    @pytest.mark.asyncio
    async def test_worker_scaling_exact_divisible(
        self, sample_user_config, fresh_service_config, sample_aiperf_config
    ) -> None:
        """Test worker scaling when workers divide evenly into pods."""

        # Set a known workers_per_pod default
        default_workers_per_pod = Environment.WORKER.DEFAULT_WORKERS_PER_POD

        kube_options = KubeOptions(
            image="aiperf:latest",
            namespace="test-ns",
            workers=default_workers_per_pod * 2,  # Exactly 2 pods worth
        )

        await run_kubernetes_deployment(
            sample_user_config,
            fresh_service_config,
            kube_options,
            aiperf_config=sample_aiperf_config,
            dry_run=True,
        )

        # Should use default workers_per_pod
        assert fresh_service_config.workers_per_pod == default_workers_per_pod

    @pytest.mark.asyncio
    async def test_sets_zmq_dual_config(
        self, sample_user_config, fresh_service_config, sample_aiperf_config
    ) -> None:
        """Test that ZMQ dual-bind config is set correctly."""

        kube_options = KubeOptions(
            image="aiperf:latest",
            namespace="test-ns",
            workers=2,
        )

        await run_kubernetes_deployment(
            sample_user_config,
            fresh_service_config,
            kube_options,
            aiperf_config=sample_aiperf_config,
            dry_run=True,
        )

        # ZMQ dual config should be set
        assert fresh_service_config.zmq_dual is not None
        # IPC path should match K8s environment
        assert (
            str(fresh_service_config.zmq_dual.ipc_path) == K8sEnvironment.ZMQ.IPC_PATH
        )

    @pytest.mark.asyncio
    async def test_sets_service_run_type_and_ui_type(
        self, sample_user_config, fresh_service_config, sample_aiperf_config
    ) -> None:
        """Test that service_run_type and ui_type are set for Kubernetes."""
        kube_options = KubeOptions(
            image="aiperf:latest",
            namespace="test-ns",
            workers=2,
        )

        await run_kubernetes_deployment(
            sample_user_config,
            fresh_service_config,
            kube_options,
            aiperf_config=sample_aiperf_config,
            dry_run=True,
        )

        # Service run type should be KUBERNETES
        assert fresh_service_config.service_run_type == ServiceRunType.KUBERNETES
        # UI should be disabled in pods
        assert fresh_service_config.ui_type == UIType.NONE


class TestApplyManifestsRetry:
    """Tests for _apply_manifests retry behavior via inlined creation."""

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self, mock_logger) -> None:
        """Test resource creation retries on transient errors."""

        call_count = 0
        error = create_server_error(503, "Service Unavailable")

        async def mock_create():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise error

        mock_obj = MagicMock()
        mock_obj.create = mock_create

        def mock_cls(manifest, api=None):
            return mock_obj

        manifests = [{"kind": "Namespace", "metadata": {"name": "test-ns"}}]

        with (
            patch(
                "aiperf.kubernetes.client.get_api",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch.dict(
                "aiperf.kubernetes.runner._KIND_TO_CLASS", {"Namespace": mock_cls}
            ),
        ):
            await _apply_manifests(manifests, mock_logger)

        assert call_count == 2
        mock_logger.warning.assert_called()
        mock_logger.info.assert_called()
