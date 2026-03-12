# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.cli_commands.kube module."""

import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.cli_commands.kube.attach import attach
from aiperf.cli_commands.kube.cancel import cancel
from aiperf.cli_commands.kube.delete import delete
from aiperf.cli_commands.kube.generate import generate
from aiperf.cli_commands.kube.list_ import list_jobs
from aiperf.cli_commands.kube.logs import logs
from aiperf.cli_commands.kube.preflight import preflight
from aiperf.cli_commands.kube.profile import profile
from aiperf.cli_commands.kube.results import results
from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.config.kube_config import KubeManageOptions, KubeOptions
from aiperf.config.cli_builder import CLIModel
from aiperf.kubernetes.cli_helpers import resolve_job_id_and_namespace
from aiperf.kubernetes.console import LastBenchmarkInfo
from aiperf.kubernetes.models import JobSetInfo
from aiperf.kubernetes.preflight import (
    CheckResult,
    CheckStatus,
    PreflightResults,
)
from aiperf.kubernetes.ui_dispatch import print_progress_message, print_realtime_metrics

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def manage_options():
    """Create a KubeManageOptions instance for testing."""
    return KubeManageOptions(kubeconfig=None, namespace=None)


@pytest.fixture
def mock_kube_client():
    """Mock AIPerfKubeClient for CLI command tests.

    Patches AIPerfKubeClient.create to return a mock client with all
    methods available as AsyncMocks. Individual tests configure return
    values as needed.
    """
    from aiperf.kubernetes.client import AIPerfKubeClient

    mock_client = AsyncMock(spec=AIPerfKubeClient)
    mock_client.job_selector = AIPerfKubeClient.job_selector
    mock_client.controller_selector = AIPerfKubeClient.controller_selector

    with patch(
        "aiperf.kubernetes.client.AIPerfKubeClient.create",
        new=AsyncMock(return_value=mock_client),
    ):
        yield mock_client


@pytest.fixture
def sample_jobset_item() -> JobSetInfo:
    """Create a sample JobSetInfo from Kubernetes API."""
    raw = {
        "metadata": {
            "name": "aiperf-abc123",
            "namespace": "default",
            "creationTimestamp": "2026-01-15T10:00:00Z",
            "labels": {
                "app": "aiperf",
                "aiperf.nvidia.com/job-id": "abc123",
            },
        },
        "status": {
            "conditions": [{"type": "Completed", "status": "True"}],
        },
    }
    return JobSetInfo.from_raw(raw)


@pytest.fixture
def running_jobset_info():
    """Create a JobSetInfo with Running status."""
    return JobSetInfo(
        name="aiperf-abc123",
        namespace="default",
        jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
        status="Running",
    )


@pytest.fixture
def completed_jobset_info():
    """Create a JobSetInfo with Completed status."""
    return JobSetInfo(
        name="aiperf-abc123",
        namespace="default",
        jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
        status="Completed",
    )


@pytest.fixture
def failed_jobset_info():
    """Create a JobSetInfo with Failed status."""
    return JobSetInfo(
        name="aiperf-abc123",
        namespace="default",
        jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
        status="Failed",
    )


# =============================================================================
# Status Command Tests
# =============================================================================


class TestListJobsCommand:
    """Tests for the kube list_jobs command."""

    async def test_status_no_jobs_found(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test list_jobs when no jobs are found."""
        mock_kube_client.list_jobsets.return_value = []
        await list_jobs(manage_options=manage_options)

        captured = capsys.readouterr()
        assert "No AIPerf jobs found" in captured.out

    async def test_status_lists_jobs(
        self, mock_kube_client, manage_options, sample_jobset_item, capsys
    ) -> None:
        """Test status lists found jobs."""
        from aiperf.kubernetes.console import console as _console
        from aiperf.kubernetes.models import PodSummary

        _console.width = 200
        try:
            mock_kube_client.list_jobsets.return_value = [sample_jobset_item]
            mock_kube_client.get_pod_summary.return_value = PodSummary(
                ready=0, total=0, restarts=0
            )
            await list_jobs(manage_options=manage_options)
        finally:
            _console.width = None

        captured = capsys.readouterr()
        assert "aiperf-abc123" in captured.out
        assert "default" in captured.out

    async def test_status_with_namespace(
        self, mock_kube_client, manage_options
    ) -> None:
        """Test status with specific namespace."""
        opts = KubeManageOptions(namespace="my-namespace")

        mock_kube_client.list_jobsets.return_value = []
        await list_jobs(manage_options=opts)

        mock_kube_client.list_jobsets.assert_called_once_with(
            "my-namespace", False, None, None
        )

    async def test_status_all_namespaces(
        self, mock_kube_client, manage_options
    ) -> None:
        """Test status with all namespaces flag."""
        mock_kube_client.list_jobsets.return_value = []
        await list_jobs(manage_options=manage_options, all_namespaces=True)

        mock_kube_client.list_jobsets.assert_called_once_with(None, True, None, None)

    async def test_status_with_job_id(self, mock_kube_client, manage_options) -> None:
        """Test status with specific job ID."""
        mock_kube_client.list_jobsets.return_value = []
        await list_jobs(manage_options=manage_options, job_id="abc123")

        mock_kube_client.list_jobsets.assert_called_once_with(
            None, True, "abc123", None
        )

    async def test_status_running_filter(
        self, mock_kube_client, manage_options
    ) -> None:
        """Test status with --running filter."""
        mock_kube_client.list_jobsets.return_value = []
        await list_jobs(manage_options=manage_options, running=True)

        mock_kube_client.list_jobsets.assert_called_once_with(
            None, True, None, "Running"
        )

    async def test_status_completed_filter(
        self, mock_kube_client, manage_options
    ) -> None:
        """Test status with --completed filter."""
        mock_kube_client.list_jobsets.return_value = []
        await list_jobs(manage_options=manage_options, completed=True)

        mock_kube_client.list_jobsets.assert_called_once_with(
            None, True, None, "Completed"
        )

    async def test_status_failed_filter(self, mock_kube_client, manage_options) -> None:
        """Test status with --failed filter."""
        mock_kube_client.list_jobsets.return_value = []
        await list_jobs(manage_options=manage_options, failed=True)

        mock_kube_client.list_jobsets.assert_called_once_with(
            None, True, None, "Failed"
        )


# =============================================================================
# Delete Command Tests
# =============================================================================


class TestDeleteCommand:
    """Tests for the kube delete command."""

    async def test_delete_job_not_found(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test delete when job is not found."""
        mock_kube_client.find_jobset.return_value = None
        await delete("nonexistent", manage_options=manage_options, force=True)

        captured = capsys.readouterr()
        assert "No AIPerf job found" in captured.out

    async def test_delete_force(
        self, mock_kube_client, manage_options, running_jobset_info, capsys
    ) -> None:
        """Test delete with force flag skips confirmation."""
        mock_kube_client.find_jobset.return_value = running_jobset_info
        await delete("abc123", manage_options=manage_options, force=True)

        mock_kube_client.delete_jobset.assert_called_once()

        captured = capsys.readouterr()
        assert "deleted successfully" in captured.out

    async def test_delete_prompts_without_force(
        self, mock_kube_client, manage_options, running_jobset_info, monkeypatch
    ) -> None:
        """Test delete prompts for confirmation without force."""
        monkeypatch.setattr("builtins.input", lambda _: "n")

        mock_kube_client.find_jobset.return_value = running_jobset_info
        await delete("abc123", manage_options=manage_options, force=False)

        mock_kube_client.delete_jobset.assert_not_called()


# =============================================================================
# Logs Command Tests
# =============================================================================


class TestLogsCommand:
    """Tests for the kube logs command."""

    async def test_logs_no_pods_found(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test logs when no pods are found."""
        mock_kube_client.get_pods.return_value = []
        await logs("nonexistent", manage_options=manage_options)

        captured = capsys.readouterr()
        assert "No pods found" in captured.out

    async def test_logs_with_namespace(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test logs with specific namespace."""
        opts = KubeManageOptions(namespace="default")

        mock_pod = MagicMock()
        mock_pod.name = "aiperf-abc123-controller-0-0"
        mock_pod.raw = {
            "spec": {"containers": [{"name": "control-plane"}]},
        }

        async def _mock_logs(*args, **kwargs):
            yield "log output"

        mock_pod.logs = _mock_logs
        mock_kube_client.get_pods.return_value = [mock_pod]

        await logs("abc123", manage_options=opts)


# =============================================================================
# Results Command Tests
# =============================================================================


class TestResultsCommand:
    """Tests for the kube results command."""

    async def test_results_from_api(
        self, mock_kube_client, manage_options, completed_jobset_info, capsys, tmp_path
    ) -> None:
        """Test results retrieval from API service (summary-only mode)."""
        output_dir = tmp_path / "results"

        mock_kube_client.find_jobset.return_value = completed_jobset_info
        with patch(
            "aiperf.kubernetes.results.retrieve_results_from_api",
            new=AsyncMock(return_value=True),
        ):
            await results(
                "abc123",
                manage_options=manage_options,
                output=output_dir,
                all_artifacts=False,
            )

        captured = capsys.readouterr()
        assert "Results" in captured.out
        assert "Saved to" in captured.out

    async def test_results_api_fails_tries_pod(
        self, mock_kube_client, manage_options, running_jobset_info, capsys, tmp_path
    ) -> None:
        """Test results falls back to pod copy when API fails."""
        output_dir = tmp_path / "results"

        mock_kube_client.find_jobset.return_value = running_jobset_info
        mock_kube_client.find_controller_pod.return_value = None
        with patch(
            "aiperf.kubernetes.results.retrieve_results_from_api",
            new=AsyncMock(return_value=False),
        ):
            await results(
                "abc123",
                manage_options=manage_options,
                output=output_dir,
                all_artifacts=False,
            )

        captured = capsys.readouterr()
        assert "No controller pod found" in captured.out


# =============================================================================
# Cancel Command Tests
# =============================================================================


class TestCancelCommand:
    """Tests for the kube cancel command."""

    async def test_cancel_job_not_found(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test cancel when job is not found."""
        mock_kube_client.find_jobset.return_value = None
        await cancel("nonexistent", manage_options=manage_options, force=True)

        captured = capsys.readouterr()
        assert "No AIPerf job found" in captured.out

    async def test_cancel_completed_job(
        self,
        mock_kube_client,
        manage_options,
        completed_jobset_info,
        capsys,
        monkeypatch,
    ) -> None:
        """Test cancel on already completed job."""
        monkeypatch.setattr("builtins.input", lambda _: "n")

        mock_kube_client.find_jobset.return_value = completed_jobset_info
        await cancel("abc123", manage_options=manage_options, force=False)

        mock_kube_client.delete_jobset.assert_not_called()

    async def test_cancel_running_job_force(
        self, mock_kube_client, manage_options, running_jobset_info, capsys
    ) -> None:
        """Test cancel on running job with force flag."""
        mock_kube_client.find_jobset.return_value = running_jobset_info
        await cancel("abc123", manage_options=manage_options, force=True)

        mock_kube_client.delete_jobset.assert_called_once()

        captured = capsys.readouterr()
        assert "cancelled" in captured.out


# =============================================================================
# Preflight Command Tests
# =============================================================================


def _make_passing_preflight_results(
    *,
    secret_names: list[str] | None = None,
    image_pull_secret_names: list[str] | None = None,
) -> PreflightResults:
    """Build a PreflightResults that passes all checks.

    Args:
        secret_names: Optional secrets to include in output details.
        image_pull_secret_names: Optional image pull secrets to include.
    """
    results = PreflightResults()
    results.add(
        CheckResult(
            name="Cluster Connectivity",
            status=CheckStatus.PASS,
            message="Connected to Kubernetes cluster",
        )
    )
    results.add(
        CheckResult(
            name="Kubernetes Version",
            status=CheckStatus.PASS,
            message="v1.31.0",
        )
    )
    results.add(
        CheckResult(
            name="Namespace",
            status=CheckStatus.PASS,
            message="Namespace default exists",
        )
    )
    results.add(
        CheckResult(
            name="RBAC Permissions",
            status=CheckStatus.PASS,
            message="All required permissions granted",
        )
    )
    results.add(
        CheckResult(
            name="JobSet CRD",
            status=CheckStatus.PASS,
            message="JobSet CRD installed",
        )
    )
    results.add(
        CheckResult(
            name="JobSet Controller",
            status=CheckStatus.PASS,
            message="JobSet controller running",
        )
    )
    results.add(
        CheckResult(
            name="Resource Quotas",
            status=CheckStatus.PASS,
            message="No restrictive quotas",
        )
    )
    results.add(
        CheckResult(
            name="Node Resources",
            status=CheckStatus.PASS,
            message="Sufficient resources",
        )
    )
    all_secrets = list(secret_names or []) + list(image_pull_secret_names or [])
    if all_secrets:
        results.add(
            CheckResult(
                name="Secrets",
                status=CheckStatus.PASS,
                message=f"All secrets found: {', '.join(all_secrets)}",
                details=all_secrets,
            )
        )
    else:
        results.add(
            CheckResult(
                name="Secrets",
                status=CheckStatus.SKIP,
                message="No secrets specified to verify",
            )
        )
    results.add(
        CheckResult(
            name="Image Pull",
            status=CheckStatus.SKIP,
            message="No image specified",
        )
    )
    results.add(
        CheckResult(
            name="Network Policies",
            status=CheckStatus.PASS,
            message="No restrictive policies",
        )
    )
    results.add(
        CheckResult(
            name="DNS Resolution",
            status=CheckStatus.PASS,
            message="CoreDNS running",
        )
    )
    results.add(
        CheckResult(
            name="Endpoint Connectivity",
            status=CheckStatus.SKIP,
            message="No endpoint specified",
        )
    )
    return results


def _mock_run_all_checks(preflight_results: PreflightResults) -> AsyncMock:
    """Create a mock run_all_checks that prints results like the real one."""
    from aiperf.kubernetes.preflight import _print_check_result

    async def _run(self=None) -> PreflightResults:
        total = len(preflight_results.checks)
        for i, check in enumerate(preflight_results.checks, 1):
            _print_check_result(check, i, total)
        preflight_results.print_summary()
        return preflight_results

    return AsyncMock(side_effect=_run)


class TestPreflightCommand:
    """Tests for the kube preflight command."""

    async def test_preflight_cluster_connectivity(self, manage_options, capsys) -> None:
        """Test preflight checks cluster connectivity."""
        pf_results = _make_passing_preflight_results()

        with patch("aiperf.kubernetes.preflight.PreflightChecker") as mock_cls:
            mock_cls.return_value.run_all_checks = _mock_run_all_checks(pf_results)
            await preflight(manage_options=manage_options)

        captured = capsys.readouterr()
        assert "Connected to Kubernetes" in captured.out
        assert "v1.31.0" in captured.out
        assert "pre-flight checks passed" in captured.out.lower()

    async def test_preflight_jobset_crd_missing(self, manage_options, capsys) -> None:
        """Test preflight reports missing JobSet CRD."""
        pf_results = PreflightResults()
        pf_results.add(
            CheckResult(
                name="Cluster Connectivity",
                status=CheckStatus.PASS,
                message="Connected to Kubernetes cluster",
            )
        )
        pf_results.add(
            CheckResult(
                name="JobSet CRD",
                status=CheckStatus.FAIL,
                message="JobSet CRD not found",
                hints=["Install JobSet"],
            )
        )

        with (
            patch("aiperf.kubernetes.preflight.PreflightChecker") as mock_cls,
            pytest.raises(SystemExit) as exc_info,
        ):
            mock_cls.return_value.run_all_checks = _mock_run_all_checks(pf_results)
            await preflight(manage_options=manage_options)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "JobSet CRD not found" in captured.out

    @pytest.mark.parametrize(
        "secret_param,secret_name,expected_output",
        [
            ({"secret": ["my-secret"]}, "my-secret", "my-secret"),
            (
                {"image_pull_secret": ["registry-creds"]},
                "registry-creds",
                "registry-creds",
            ),
        ],
    )  # fmt: skip
    async def test_preflight_secret_exists(
        self,
        secret_param: dict,
        secret_name: str,
        expected_output: str,
        manage_options,
        capsys,
    ) -> None:
        """Test preflight reports when secret exists."""
        pf_results = _make_passing_preflight_results(secret_names=[secret_name])

        with patch("aiperf.kubernetes.preflight.PreflightChecker") as mock_cls:
            mock_cls.return_value.run_all_checks = _mock_run_all_checks(pf_results)
            await preflight(manage_options=manage_options, **secret_param)

        captured = capsys.readouterr()
        assert expected_output in captured.out
        assert "pre-flight checks passed" in captured.out.lower()

    @pytest.mark.parametrize(
        "status_code,secret_name,expected_output,should_fail",
        [
            (404, "missing-secret", "not found", True),
            (403, "protected-secret", "permission denied", False),
        ],
    )  # fmt: skip
    async def test_preflight_secret_errors(
        self,
        status_code: int,
        secret_name: str,
        expected_output: str,
        should_fail: bool,
        manage_options,
        capsys,
    ) -> None:
        """Test preflight handles secret lookup errors."""
        pf_results = PreflightResults()
        pf_results.add(
            CheckResult(
                name="Cluster Connectivity",
                status=CheckStatus.PASS,
                message="Connected to Kubernetes cluster",
            )
        )
        if status_code == 404:
            pf_results.add(
                CheckResult(
                    name="Secrets",
                    status=CheckStatus.FAIL,
                    message=f"Secret {secret_name} not found",
                )
            )
        else:
            pf_results.add(
                CheckResult(
                    name="Secrets",
                    status=CheckStatus.WARN,
                    message=f"Secret {secret_name}: permission denied",
                )
            )

        if should_fail:
            with (
                patch("aiperf.kubernetes.preflight.PreflightChecker") as mock_cls,
                pytest.raises(SystemExit) as exc_info,
            ):
                mock_cls.return_value.run_all_checks = _mock_run_all_checks(pf_results)
                await preflight(manage_options=manage_options, secret=[secret_name])
            assert exc_info.value.code == 1
        else:
            with patch("aiperf.kubernetes.preflight.PreflightChecker") as mock_cls:
                mock_cls.return_value.run_all_checks = _mock_run_all_checks(pf_results)
                await preflight(manage_options=manage_options, secret=[secret_name])

        captured = capsys.readouterr()
        assert expected_output in captured.out

    async def test_preflight_multiple_secrets(
        self,
        manage_options,
        capsys,
    ) -> None:
        """Test preflight checks multiple secrets."""
        secrets = ["s1", "s2", "s3"]
        pf_results = _make_passing_preflight_results(secret_names=secrets)

        with patch("aiperf.kubernetes.preflight.PreflightChecker") as mock_cls:
            mock_cls.return_value.run_all_checks = _mock_run_all_checks(pf_results)
            await preflight(manage_options=manage_options, secret=secrets)

            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args
            assert list(call_kwargs.kwargs.get("secrets", [])) == secrets

        captured = capsys.readouterr()
        assert "pre-flight checks passed" in captured.out.lower()


# =============================================================================
# Profile Command Tests
# =============================================================================


class TestProfileCommand:
    """Tests for the kube profile command."""

    @pytest.fixture
    def _mock_config_conversion(self):
        """Mock build_aiperf_config and convert_to_legacy_configs."""
        user_config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        service_config = ServiceConfig()
        with (
            patch("aiperf.config.cli_builder.build_aiperf_config") as mock_build,
            patch(
                "aiperf.config.reverse_converter.convert_to_legacy_configs"
            ) as mock_convert,
        ):
            mock_aiperf_config = MagicMock()
            mock_aiperf_config.endpoint.urls = [
                "http://localhost:8000/v1/chat/completions"
            ]
            mock_build.return_value = mock_aiperf_config
            mock_convert.return_value = (user_config, service_config)
            yield mock_build, mock_convert, user_config, service_config

    async def test_profile_calls_runner(self, _mock_config_conversion) -> None:
        """Test profile calls run_kubernetes_deployment with correct args."""
        _, _, user_config, service_config = _mock_config_conversion
        cli = CLIModel(model_names=["test-model"])
        kube_options = KubeOptions(image="aiperf:latest", workers=10)

        with patch("aiperf.kubernetes.runner.run_kubernetes_deployment") as mock_run:
            mock_run.return_value = ("test-job-id", "test-namespace")

            await profile(
                cli=cli,
                kube_options=kube_options,
                detach=True,
                skip_endpoint_check=True,
                skip_preflight=True,
            )

            mock_run.assert_called_once()

    async def test_profile_derives_configs_from_cli(
        self, _mock_config_conversion
    ) -> None:
        """Test profile derives UserConfig/ServiceConfig via build_aiperf_config."""
        mock_build, mock_convert, _, _ = _mock_config_conversion
        cli = CLIModel(model_names=["test-model"])
        kube_options = KubeOptions(image="aiperf:latest")

        with patch("aiperf.kubernetes.runner.run_kubernetes_deployment") as mock_run:
            mock_run.return_value = ("test-job-id", "test-namespace")

            await profile(
                cli=cli,
                kube_options=kube_options,
                detach=True,
                skip_endpoint_check=True,
                skip_preflight=True,
            )

            mock_build.assert_called_once_with(cli)
            mock_convert.assert_called_once()
            mock_run.assert_called_once()

    async def test_profile_with_kube_options(self, _mock_config_conversion) -> None:
        """Test profile with custom KubeOptions."""
        cli = CLIModel(model_names=["test-model"])
        kube_options = KubeOptions(
            image="aiperf:v2",
            namespace="benchmarks",
            workers=50,
            node_selector={"gpu": "true", "zone": "us-west"},
            env_vars={"DEBUG": "true"},
            tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists"}],
        )

        with patch("aiperf.kubernetes.runner.run_kubernetes_deployment") as mock_run:
            mock_run.return_value = ("test-job-id", "benchmarks")

            await profile(
                cli=cli,
                kube_options=kube_options,
                detach=True,
                skip_endpoint_check=True,
                skip_preflight=True,
            )

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            passed_kube_options = call_kwargs.args[2]
            assert passed_kube_options.image == "aiperf:v2"
            assert passed_kube_options.namespace == "benchmarks"
            assert passed_kube_options.workers == 50
            assert passed_kube_options.node_selector == {
                "gpu": "true",
                "zone": "us-west",
            }
            assert passed_kube_options.env_vars == {"DEBUG": "true"}
            assert len(passed_kube_options.tolerations) == 1


# =============================================================================
# Attach Command Tests (Async)
# =============================================================================


class TestAttachCommand:
    """Tests for the kube attach command."""

    async def test_attach_job_not_found(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test attach when job is not found."""
        mock_kube_client.find_jobset.return_value = None
        await attach("nonexistent", manage_options=manage_options)

        captured = capsys.readouterr()
        assert "No running job found" in captured.out

    async def test_attach_completed_job(
        self, mock_kube_client, manage_options, completed_jobset_info, capsys
    ) -> None:
        """Test attach on completed job."""
        mock_kube_client.find_jobset.return_value = completed_jobset_info
        await attach("abc123", manage_options=manage_options)

        captured = capsys.readouterr()
        assert "already completed" in captured.out

    async def test_attach_failed_job(
        self, mock_kube_client, manage_options, failed_jobset_info, capsys
    ) -> None:
        """Test attach on failed job."""
        mock_kube_client.find_jobset.return_value = failed_jobset_info
        await attach("abc123", manage_options=manage_options)

        captured = capsys.readouterr()
        assert "has failed" in captured.out


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestPrintProgressMessage:
    """Tests for print_progress_message helper."""

    def test_credit_phase_start(self, capsys) -> None:
        """Test printing CREDIT_PHASE_START message."""
        print_progress_message(
            {
                "message_type": "CREDIT_PHASE_START",
                "phase": "profiling",
            }
        )

        captured = capsys.readouterr()
        assert "[PHASE]" in captured.out
        assert "profiling" in captured.out

    def test_credit_phase_progress(self, capsys) -> None:
        """Test printing CREDIT_PHASE_PROGRESS message."""
        print_progress_message(
            {
                "message_type": "CREDIT_PHASE_PROGRESS",
                "phase": "profiling",
                "requests": {
                    "completed": 500,
                    "total_expected_requests": 1000,
                },
            }
        )

        captured = capsys.readouterr()
        assert "500/1000" in captured.out
        assert "50.0%" in captured.out

    def test_credit_phase_complete(self, capsys) -> None:
        """Test printing CREDIT_PHASE_COMPLETE message."""
        print_progress_message(
            {
                "message_type": "CREDIT_PHASE_COMPLETE",
                "phase": "warmup",
            }
        )

        captured = capsys.readouterr()
        assert "[PHASE]" in captured.out
        assert "Completed" in captured.out
        assert "warmup" in captured.out

    def test_all_records_received(self, capsys) -> None:
        """Test printing ALL_RECORDS_RECEIVED message."""
        print_progress_message({"message_type": "ALL_RECORDS_RECEIVED"})

        captured = capsys.readouterr()
        assert "[COMPLETE]" in captured.out

    def test_subscribed_message_ignored(self, capsys) -> None:
        """Test subscribed message is ignored."""
        print_progress_message({"message_type": "subscribed"})

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_worker_status_summary(self, capsys) -> None:
        """Test printing WORKER_STATUS_SUMMARY message."""
        print_progress_message(
            {
                "message_type": "WORKER_STATUS_SUMMARY",
                "workers": {
                    "worker-1": {"status": "HEALTHY"},
                    "worker-2": {"status": "HEALTHY"},
                    "worker-3": {"status": "UNHEALTHY"},
                },
            }
        )

        captured = capsys.readouterr()
        assert "[WORKERS]" in captured.out
        assert "2/3" in captured.out


class TestPrintRealtimeMetrics:
    """Tests for print_realtime_metrics helper."""

    def test_prints_key_metrics(self, capsys) -> None:
        """Test printing key metrics from realtime metrics message."""
        print_realtime_metrics(
            {
                "metrics": [
                    {"tag": "throughput", "value": 100.5, "display_unit": "req/s"},
                    {"tag": "latency_p50", "value": 50.2, "unit": "ms"},
                    {"tag": "latency_p99", "value": 120.3, "display_unit": "ms"},
                    {"tag": "ttft_p50", "value": 25.0, "unit": "ms"},
                ]
            }
        )

        captured = capsys.readouterr()
        assert "[METRIC]" in captured.out
        assert "throughput: 100.50" in captured.out
        assert "latency_p50: 50.20" in captured.out

    def test_no_output_when_no_key_metrics(self, capsys) -> None:
        """Test no output when no key metrics are present."""
        print_realtime_metrics(
            {
                "metrics": [
                    {"tag": "other_metric", "value": 42.0, "unit": "things"},
                ]
            }
        )

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_empty_metrics_list(self, capsys) -> None:
        """Test handling empty metrics list."""
        print_realtime_metrics({"metrics": []})

        captured = capsys.readouterr()
        assert captured.out == ""


# =============================================================================
# Resolve Job ID and Namespace Tests
# =============================================================================


class TestResolveJobIdAndNamespace:
    """Tests for resolve_job_id_and_namespace helper."""

    def test_returns_provided_job_id(self) -> None:
        """Test returns provided job_id and namespace."""
        result = resolve_job_id_and_namespace("abc123", "my-namespace")
        assert result == ("abc123", "my-namespace")

    def test_returns_job_id_with_none_namespace(self) -> None:
        """Test returns job_id with None namespace when only job_id provided."""
        result = resolve_job_id_and_namespace("abc123", None)
        assert result == ("abc123", None)

    def test_uses_last_benchmark_when_no_job_id(self, capsys) -> None:
        """Test uses last benchmark info when no job_id specified."""
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(job_id="last-job", namespace="last-ns"),
        ):
            result = resolve_job_id_and_namespace(None, None)

        assert result == ("last-job", "last-ns")
        captured = capsys.readouterr()
        assert "Using last benchmark" in captured.out

    def test_uses_provided_namespace_over_last_benchmark(self, capsys) -> None:
        """Test provided namespace overrides last benchmark namespace."""
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(job_id="last-job", namespace="last-ns"),
        ):
            result = resolve_job_id_and_namespace(None, "override-ns")

        assert result == ("last-job", "override-ns")

    def test_returns_none_when_no_job_id_and_no_last_benchmark(self, capsys) -> None:
        """Test returns None when no job_id and no last benchmark."""
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark", return_value=None
        ):
            result = resolve_job_id_and_namespace(None, None)

        assert result is None
        captured = capsys.readouterr()
        assert "No job_id specified" in captured.out
        assert "no previous benchmark found" in captured.out


# =============================================================================
# Generate Command Tests
# =============================================================================


class TestGenerateCommand:
    """Tests for the kube generate command."""

    @pytest.fixture
    def _mock_config_conversion(self):
        """Mock build_aiperf_config and convert_to_legacy_configs."""
        user_config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        service_config = ServiceConfig()
        with (
            patch("aiperf.config.cli_builder.build_aiperf_config") as mock_build,
            patch(
                "aiperf.config.reverse_converter.convert_to_legacy_configs"
            ) as mock_convert,
        ):
            mock_aiperf_config = MagicMock()
            mock_aiperf_config.endpoint.urls = [
                "http://localhost:8000/v1/chat/completions"
            ]
            mock_build.return_value = mock_aiperf_config
            mock_convert.return_value = (user_config, service_config)
            yield mock_build, mock_convert, user_config, service_config

    async def test_generate_calls_runner_with_dry_run(
        self, _mock_config_conversion
    ) -> None:
        """Test generate calls run_kubernetes_deployment with dry_run=True."""
        cli = CLIModel(model_names=["test-model"])
        kube_options = KubeOptions(image="aiperf:latest", workers=10)

        with patch("aiperf.kubernetes.runner.run_kubernetes_deployment") as mock_run:
            mock_run.return_value = ("test-job-id", "test-namespace")

            await generate(cli=cli, kube_options=kube_options)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs.get("dry_run") is True

    async def test_generate_derives_configs_from_cli(
        self, _mock_config_conversion
    ) -> None:
        """Test generate derives configs via build_aiperf_config."""
        mock_build, mock_convert, _, _ = _mock_config_conversion
        cli = CLIModel(model_names=["test-model"])
        kube_options = KubeOptions(image="aiperf:latest")

        with patch("aiperf.kubernetes.runner.run_kubernetes_deployment") as mock_run:
            mock_run.return_value = ("test-job-id", "test-namespace")

            await generate(cli=cli, kube_options=kube_options)

            mock_build.assert_called_once_with(cli)
            mock_convert.assert_called_once()
            mock_run.assert_called_once()


# =============================================================================
# Delete Namespace Tests (client.delete_namespace method)
# =============================================================================


class TestDeleteNamespace:
    """Tests for AIPerfKubeClient.delete_namespace method."""

    async def test_delete_namespace_success(self, capsys) -> None:
        """Test successful namespace deletion."""
        from aiperf.kubernetes.client import AIPerfKubeClient

        mock_api = AsyncMock()
        mock_ns = AsyncMock()

        client = AIPerfKubeClient(mock_api)
        with patch(
            "kr8s.asyncio.objects.Namespace.get",
            new=AsyncMock(return_value=mock_ns),
        ):
            await client.delete_namespace("test-namespace")

        mock_ns.delete.assert_called_once()
        captured = capsys.readouterr()
        assert "Deleted Namespace/test-namespace" in captured.out

    async def test_delete_namespace_not_found(self, capsys) -> None:
        """Test namespace deletion when namespace not found."""
        import kr8s

        from aiperf.kubernetes.client import AIPerfKubeClient

        mock_api = AsyncMock()
        client = AIPerfKubeClient(mock_api)

        with patch(
            "kr8s.asyncio.objects.Namespace.get",
            new=AsyncMock(side_effect=kr8s.NotFoundError(MagicMock())),
        ):
            await client.delete_namespace("missing-namespace")

        captured = capsys.readouterr()
        assert "not found" in captured.out
        assert "may already be deleted" in captured.out

    async def test_delete_namespace_other_error(self, capsys) -> None:
        """Test namespace deletion with other API errors."""
        import kr8s

        from aiperf.kubernetes.client import AIPerfKubeClient

        mock_api = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 403

        client = AIPerfKubeClient(mock_api)
        with patch(
            "kr8s.asyncio.objects.Namespace.get",
            new=AsyncMock(
                side_effect=kr8s.ServerError("Forbidden", response=mock_response)
            ),
        ):
            await client.delete_namespace("protected-namespace")

        captured = capsys.readouterr()
        assert "Failed to delete namespace" in captured.out


# =============================================================================
# Delete Command with Namespace Tests
# =============================================================================


class TestDeleteCommandWithNamespace:
    """Additional tests for delete command with namespace deletion."""

    async def test_delete_with_auto_generated_namespace(
        self, mock_kube_client, monkeypatch, capsys
    ) -> None:
        """Test delete with delete_namespace flag on auto-generated namespace."""
        jobset_info = JobSetInfo(
            name="aiperf-abc123",
            namespace="aiperf-abc123",  # auto-generated format
            jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
            status="Completed",
        )

        manage_options = KubeManageOptions()

        mock_kube_client.find_jobset.return_value = jobset_info
        await delete(
            "abc123",
            manage_options=manage_options,
            force=True,
            delete_namespace=True,
        )

        mock_kube_client.delete_namespace.assert_called_once_with("aiperf-abc123")

    async def test_delete_skips_non_auto_namespace(
        self, mock_kube_client, capsys
    ) -> None:
        """Test delete skips namespace deletion for non-auto-generated namespace."""
        jobset_info = JobSetInfo(
            name="aiperf-abc123",
            namespace="custom-namespace",  # not auto-generated
            jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
            status="Completed",
        )

        manage_options = KubeManageOptions()

        mock_kube_client.find_jobset.return_value = jobset_info
        await delete(
            "abc123",
            manage_options=manage_options,
            force=True,
            delete_namespace=True,
        )

        mock_kube_client.delete_namespace.assert_not_called()

        captured = capsys.readouterr()
        assert "was not auto-generated" in captured.out

    async def test_delete_clears_last_benchmark(self, mock_kube_client) -> None:
        """Test delete clears last benchmark when deleting the last used job."""
        jobset_info = JobSetInfo(
            name="aiperf-abc123",
            namespace="default",
            jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
            status="Running",
        )

        manage_options = KubeManageOptions()

        mock_kube_client.find_jobset.return_value = jobset_info
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.get_last_benchmark",
                return_value=LastBenchmarkInfo(job_id="abc123", namespace="default"),
            ),
            patch("aiperf.kubernetes.cli_helpers.clear_last_benchmark") as mock_clear,
        ):
            await delete("abc123", manage_options=manage_options, force=True)

            mock_clear.assert_called_once()

    async def test_delete_confirms_with_user(
        self, mock_kube_client, monkeypatch, capsys
    ) -> None:
        """Test delete prompts user and proceeds on 'y'."""
        jobset_info = JobSetInfo(
            name="aiperf-abc123",
            namespace="default",
            jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
            status="Running",
        )

        manage_options = KubeManageOptions()
        monkeypatch.setattr("builtins.input", lambda _: "y")

        mock_kube_client.find_jobset.return_value = jobset_info
        await delete("abc123", manage_options=manage_options, force=False)

        mock_kube_client.delete_jobset.assert_called_once()


# =============================================================================
# Profile Command Additional Tests
# =============================================================================


class TestProfileCommandAdditional:
    """Additional tests for the kube profile command."""

    @pytest.fixture
    def _mock_config_conversion(self):
        """Mock build_aiperf_config and convert_to_legacy_configs."""
        user_config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        service_config = ServiceConfig()
        with (
            patch("aiperf.config.cli_builder.build_aiperf_config") as mock_build,
            patch(
                "aiperf.config.reverse_converter.convert_to_legacy_configs"
            ) as mock_convert,
        ):
            mock_aiperf_config = MagicMock()
            mock_aiperf_config.endpoint.urls = [
                "http://localhost:8000/v1/chat/completions"
            ]
            mock_build.return_value = mock_aiperf_config
            mock_convert.return_value = (user_config, service_config)
            yield mock_build, mock_convert, user_config, service_config

    async def test_profile_non_interactive_auto_detach(
        self, _mock_config_conversion, capsys, monkeypatch
    ) -> None:
        """Test profile auto-detaches in non-interactive environment."""
        cli = CLIModel(model_names=["test-model"])
        kube_options = KubeOptions(image="aiperf:latest")

        mock_stdout = MagicMock(spec=StringIO)
        mock_stdout.isatty.return_value = False
        monkeypatch.setattr(sys, "stdout", mock_stdout)

        with patch("aiperf.kubernetes.runner.run_kubernetes_deployment") as mock_run:
            mock_run.return_value = ("test-job-id", "test-namespace")

            await profile(
                cli=cli,
                kube_options=kube_options,
                skip_endpoint_check=True,
                skip_preflight=True,
            )

            mock_run.assert_called_once()

    async def test_profile_saves_last_benchmark(self, _mock_config_conversion) -> None:
        """Test profile saves last benchmark info after deployment."""
        cli = CLIModel(model_names=["test-model"])
        kube_options = KubeOptions(image="aiperf:latest")

        with (
            patch("aiperf.kubernetes.runner.run_kubernetes_deployment") as mock_run,
            patch("aiperf.kubernetes.console.save_last_benchmark") as mock_save,
        ):
            mock_run.return_value = ("deployed-job-id", "deployed-namespace")

            await profile(
                cli=cli,
                kube_options=kube_options,
                detach=True,
                skip_endpoint_check=True,
                skip_preflight=True,
            )

            mock_save.assert_called_once_with(
                "deployed-job-id", "deployed-namespace", name=None
            )


# =============================================================================
# Cancel Command Additional Tests
# =============================================================================


class TestCancelCommandAdditional:
    """Additional tests for cancel command."""

    async def test_cancel_clears_last_benchmark(self, mock_kube_client) -> None:
        """Test cancel clears last benchmark when cancelling the last used job."""
        jobset_info = JobSetInfo(
            name="aiperf-abc123",
            namespace="default",
            jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
            status="Running",
        )

        manage_options = KubeManageOptions()

        mock_kube_client.find_jobset.return_value = jobset_info
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.get_last_benchmark",
                return_value=LastBenchmarkInfo(job_id="abc123", namespace="default"),
            ),
            patch("aiperf.kubernetes.cli_helpers.clear_last_benchmark") as mock_clear,
        ):
            await cancel("abc123", manage_options=manage_options, force=True)

            mock_clear.assert_called_once()

    async def test_cancel_failed_job_with_force(
        self, mock_kube_client, failed_jobset_info, capsys
    ) -> None:
        """Test cancel on failed job with force flag."""
        manage_options = KubeManageOptions()

        mock_kube_client.find_jobset.return_value = failed_jobset_info
        await cancel("abc123", manage_options=manage_options, force=True)

        mock_kube_client.delete_jobset.assert_called_once()

        captured = capsys.readouterr()
        assert "cancelled" in captured.out

    async def test_cancel_uses_last_benchmark(self, mock_kube_client, capsys) -> None:
        """Test cancel uses last benchmark when no job_id specified."""
        jobset_info = JobSetInfo(
            name="aiperf-last123",
            namespace="last-ns",
            jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
            status="Running",
        )

        manage_options = KubeManageOptions()

        mock_kube_client.find_jobset.return_value = jobset_info
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(job_id="last123", namespace="last-ns"),
        ):
            await cancel(manage_options=manage_options, force=True)

        mock_kube_client.delete_jobset.assert_called_once()

        captured = capsys.readouterr()
        assert "Using last benchmark" in captured.out


# =============================================================================
# Logs Command Additional Tests
# =============================================================================


class TestLogsCommandAdditional:
    """Additional tests for logs command."""

    async def test_logs_uses_last_benchmark(self, mock_kube_client, capsys) -> None:
        """Test logs uses last benchmark when no job_id specified."""
        manage_options = KubeManageOptions()

        mock_kube_client.get_pods.return_value = []

        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(job_id="last123", namespace="last-ns"),
        ):
            await logs(manage_options=manage_options)

        captured = capsys.readouterr()
        assert "Using last benchmark" in captured.out

    async def test_logs_returns_early_when_no_job_found(
        self, mock_kube_client, capsys
    ) -> None:
        """Test logs returns early when no job_id and no last benchmark."""
        manage_options = KubeManageOptions()

        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark", return_value=None
        ):
            await logs(manage_options=manage_options)

        captured = capsys.readouterr()
        assert "No job_id specified" in captured.out


# =============================================================================
# Results Command Additional Tests
# =============================================================================


class TestResultsCommandAdditional:
    """Additional tests for results command."""

    async def test_results_uses_last_benchmark(
        self, mock_kube_client, capsys, tmp_path
    ) -> None:
        """Test results uses last benchmark when no job_id specified."""
        manage_options = KubeManageOptions()
        output_dir = tmp_path / "results"

        mock_kube_client.find_jobset.return_value = None
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.get_last_benchmark",
                return_value=LastBenchmarkInfo(job_id="last123", namespace="last-ns"),
            ),
            patch(
                "aiperf.kubernetes.results.retrieve_results_from_api",
                new=AsyncMock(return_value=False),
            ),
        ):
            await results(
                manage_options=manage_options,
                output=output_dir,
                all_artifacts=False,
            )

        captured = capsys.readouterr()
        assert "Using last benchmark" in captured.out

    async def test_results_from_pod_flag(
        self, mock_kube_client, running_jobset_info, tmp_path
    ) -> None:
        """Test results with --from-pod flag."""
        manage_options = KubeManageOptions()
        output_dir = tmp_path / "results"

        mock_kube_client.find_jobset.return_value = running_jobset_info
        with patch(
            "aiperf.kubernetes.results.retrieve_results_from_pod",
            new_callable=AsyncMock,
        ) as mock_pod:
            mock_pod.return_value = True
            await results(
                "abc123",
                manage_options=manage_options,
                output=output_dir,
                from_pod=True,
                all_artifacts=False,
            )

            mock_pod.assert_called_once()

    async def test_results_all_artifacts_flag(
        self, mock_kube_client, running_jobset_info, tmp_path
    ) -> None:
        """Test results with --all flag for archive download."""
        manage_options = KubeManageOptions()
        output_dir = tmp_path / "results"

        mock_kube_client.find_jobset.return_value = running_jobset_info
        with patch(
            "aiperf.kubernetes.results.retrieve_all_artifacts",
            new_callable=AsyncMock,
        ) as mock_archive:
            await results(
                "abc123",
                manage_options=manage_options,
                output=output_dir,
                all_artifacts=True,
            )

            mock_archive.assert_called_once()


# =============================================================================
# Attach Command Additional Tests
# =============================================================================


class TestAttachCommandAdditional:
    """Additional tests for attach command."""

    async def test_attach_uses_last_benchmark(
        self, mock_kube_client, completed_jobset_info, capsys
    ) -> None:
        """Test attach uses last benchmark when no job_id specified."""
        manage_options = KubeManageOptions()

        mock_kube_client.find_jobset.return_value = completed_jobset_info
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(job_id="abc123", namespace="default"),
        ):
            await attach(manage_options=manage_options)

        captured = capsys.readouterr()
        assert "Using last benchmark" in captured.out

    async def test_attach_no_controller_pod(
        self, mock_kube_client, running_jobset_info, capsys
    ) -> None:
        """Test attach when no controller pod found."""
        manage_options = KubeManageOptions()

        mock_kube_client.find_jobset.return_value = running_jobset_info
        mock_kube_client.find_controller_pod.return_value = None
        await attach("abc123", manage_options=manage_options)

        captured = capsys.readouterr()
        assert "No controller pod found" in captured.out


# =============================================================================
# Status Command with Wide Output Tests
# =============================================================================


class TestProfilePreflight:
    """Tests for the auto-preflight in kube profile command."""

    @pytest.fixture
    def _mock_config_conversion(self):
        """Mock build_aiperf_config and convert_to_legacy_configs."""
        user_config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        service_config = ServiceConfig()
        with (
            patch("aiperf.config.cli_builder.build_aiperf_config") as mock_build,
            patch(
                "aiperf.config.reverse_converter.convert_to_legacy_configs"
            ) as mock_convert,
        ):
            mock_aiperf_config = MagicMock()
            mock_aiperf_config.endpoint.urls = [
                "http://localhost:8000/v1/chat/completions"
            ]
            mock_build.return_value = mock_aiperf_config
            mock_convert.return_value = (user_config, service_config)
            yield mock_build, mock_convert, user_config, service_config

    async def test_profile_runs_preflight_before_deploy(
        self, _mock_config_conversion
    ) -> None:
        """Test profile runs preflight checks before deployment (passing)."""
        cli = CLIModel(model_names=["test-model"])
        kube_options = KubeOptions(image="aiperf:latest")

        passing_results = PreflightResults()
        passing_results.add(
            CheckResult(
                name="Cluster Connectivity", status=CheckStatus.PASS, message="OK"
            )
        )
        passing_results.add(
            CheckResult(name="JobSet CRD", status=CheckStatus.PASS, message="OK")
        )
        passing_results.add(
            CheckResult(name="RBAC Permissions", status=CheckStatus.PASS, message="OK")
        )

        with (
            patch("aiperf.kubernetes.preflight.PreflightChecker") as mock_checker_cls,
            patch("aiperf.kubernetes.runner.run_kubernetes_deployment") as mock_run,
        ):
            mock_checker_cls.return_value.run_quick_checks = AsyncMock(
                return_value=passing_results
            )
            mock_run.return_value = ("test-job-id", "test-namespace")

            await profile(
                cli=cli,
                kube_options=kube_options,
                detach=True,
                skip_endpoint_check=True,
            )

            mock_checker_cls.return_value.run_quick_checks.assert_called_once()
            mock_run.assert_called_once()

    async def test_profile_preflight_failure_blocks_deploy(
        self, _mock_config_conversion, capsys
    ) -> None:
        """Test profile exits when preflight fails."""
        cli = CLIModel(model_names=["test-model"])
        kube_options = KubeOptions(image="aiperf:latest")

        failing_results = PreflightResults()
        failing_results.add(
            CheckResult(
                name="Cluster Connectivity", status=CheckStatus.PASS, message="OK"
            )
        )
        failing_results.add(
            CheckResult(
                name="JobSet CRD",
                status=CheckStatus.FAIL,
                message="JobSet CRD not found",
                hints=["Install JobSet"],
            )
        )

        with (
            patch("aiperf.kubernetes.preflight.PreflightChecker") as mock_checker_cls,
            patch("aiperf.kubernetes.runner.run_kubernetes_deployment") as mock_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            mock_checker_cls.return_value.run_quick_checks = AsyncMock(
                return_value=failing_results
            )

            await profile(
                cli=cli,
                kube_options=kube_options,
                detach=True,
                skip_endpoint_check=True,
            )

        assert exc_info.value.code == 1
        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "JobSet CRD" in captured.out
        assert "--skip-preflight" in captured.out

    async def test_profile_skip_preflight_bypasses_checks(
        self, _mock_config_conversion
    ) -> None:
        """Test --skip-preflight flag bypasses preflight checks."""
        cli = CLIModel(model_names=["test-model"])
        kube_options = KubeOptions(image="aiperf:latest")

        with (
            patch("aiperf.kubernetes.preflight.PreflightChecker") as mock_checker_cls,
            patch("aiperf.kubernetes.runner.run_kubernetes_deployment") as mock_run,
        ):
            mock_run.return_value = ("test-job-id", "test-namespace")

            await profile(
                cli=cli,
                kube_options=kube_options,
                detach=True,
                skip_endpoint_check=True,
                skip_preflight=True,
            )

            # PreflightChecker should NOT have been instantiated
            mock_checker_cls.assert_not_called()
            mock_run.assert_called_once()


class TestStatusCommandWide:
    """Tests for status command with wide output."""

    async def test_status_wide_output(
        self, mock_kube_client, manage_options, sample_jobset_item, capsys
    ) -> None:
        """Test status with wide output shows job-id column."""
        from aiperf.kubernetes.console import console as _console
        from aiperf.kubernetes.models import PodSummary

        _console.width = 200
        try:
            mock_kube_client.list_jobsets.return_value = [sample_jobset_item]
            mock_kube_client.get_pod_summary.return_value = PodSummary(
                ready=2, total=2, restarts=0
            )
            await list_jobs(manage_options=manage_options, wide=True)
        finally:
            _console.width = None

        captured = capsys.readouterr()
        # Wide mode shows JOB-ID, READY, and RESTARTS columns
        assert "abc123" in captured.out
        assert "JOB-ID" in captured.out
        assert "2/2" in captured.out

    async def test_status_no_filter_shows_all(
        self, mock_kube_client, manage_options
    ) -> None:
        """Test status with no filter passes None as status_filter."""
        mock_kube_client.list_jobsets.return_value = []
        await list_jobs(manage_options=manage_options)

        mock_kube_client.list_jobsets.assert_called_once_with(None, True, None, None)

    async def test_status_filter_with_status_message(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test status with filter shows filter in message."""
        mock_kube_client.list_jobsets.return_value = []
        await list_jobs(manage_options=manage_options, running=True)

        captured = capsys.readouterr()
        assert "No AIPerf jobs found with status 'Running'" in captured.out
