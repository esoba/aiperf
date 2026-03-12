# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.setup module.

Focuses on:
- CLI arg building for kubectl/helm subprocesses
- Binary discovery with SystemExit on missing
- JobSet CRD detection and installation flow
- Operator install/upgrade via Helm
- Namespace creation
- run_setup orchestration with all combinations of flags
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import kr8s
import pytest

from aiperf.kubernetes.jobset import JOBSET_FALLBACK_VERSION
from aiperf.kubernetes.setup import (
    OPERATOR_DEFAULT_CHART,
    OPERATOR_DEFAULT_NAMESPACE,
    OPERATOR_RELEASE_NAME,
    _check_jobset_installed,
    _check_namespace_exists,
    _check_operator_installed,
    _create_namespace,
    _find_binary,
    _find_helm,
    _find_kubectl,
    _install_jobset,
    _install_operator,
    _kube_cli_args,
    _resolve_jobset_version,
    _run_and_report,
    _upgrade_operator,
    run_setup,
)
from aiperf.kubernetes.subproc import CommandResult
from tests.unit.kubernetes.conftest import create_not_found_error, create_server_error

# Module path prefix for patching
_MOD = "aiperf.kubernetes.setup"


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def ok_result() -> CommandResult:
    """Successful command result with stdout."""
    return CommandResult(returncode=0, stdout="applied\n", stderr="")


@pytest.fixture
def fail_result() -> CommandResult:
    """Failed command result with stderr."""
    return CommandResult(returncode=1, stdout="", stderr="something broke\n")


@pytest.fixture
def mock_api() -> MagicMock:
    """Mock kr8s API client."""
    api = MagicMock(spec=kr8s.Api)
    api.async_get = AsyncMock(return_value=[])
    api.async_version = AsyncMock(return_value={"gitVersion": "v1.30.0"})
    return api


# ============================================================
# _kube_cli_args
# ============================================================


class TestKubeCliArgs:
    """Verify CLI arg construction for kubectl/helm subprocesses."""

    @pytest.mark.parametrize(
        "kubeconfig,kube_context,expected",
        [
            (None, None, []),
            ("/path/to/config", None, ["--kubeconfig", "/path/to/config"]),
            (None, "my-ctx", ["--context", "my-ctx"]),
            (
                "/cfg",
                "ctx",
                ["--kubeconfig", "/cfg", "--context", "ctx"],
            ),
        ],
    )  # fmt: skip
    def test_kube_cli_args_builds_correct_flags(
        self,
        kubeconfig: str | None,
        kube_context: str | None,
        expected: list[str],
    ) -> None:
        assert _kube_cli_args(kubeconfig, kube_context) == expected


# ============================================================
# _find_binary / _find_kubectl / _find_helm
# ============================================================


class TestFindBinary:
    """Verify binary discovery raises SystemExit when not found."""

    def test_find_binary_returns_path_when_found(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/kubectl"):
            assert _find_binary("kubectl", "https://example.com") == "/usr/bin/kubectl"

    def test_find_binary_raises_system_exit_when_missing(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(SystemExit, match="1"),
        ):
            _find_binary("kubectl", "https://example.com")

    def test_find_kubectl_delegates_to_find_binary(self) -> None:
        with patch("shutil.which", return_value="/usr/local/bin/kubectl"):
            assert _find_kubectl() == "/usr/local/bin/kubectl"

    def test_find_helm_delegates_to_find_binary(self) -> None:
        with patch("shutil.which", return_value="/usr/local/bin/helm"):
            assert _find_helm() == "/usr/local/bin/helm"

    def test_find_kubectl_raises_when_missing(self) -> None:
        with patch("shutil.which", return_value=None), pytest.raises(SystemExit):
            _find_kubectl()

    def test_find_helm_raises_when_missing(self) -> None:
        with patch("shutil.which", return_value=None), pytest.raises(SystemExit):
            _find_helm()


# ============================================================
# _check_jobset_installed
# ============================================================


class TestCheckJobSetInstalled:
    """Verify JobSet CRD detection via kr8s API."""

    @pytest.mark.asyncio
    async def test_check_jobset_installed_returns_true_when_crd_exists(
        self, mock_api: MagicMock
    ) -> None:
        mock_api.async_get = MagicMock(return_value=_async_iter([]))
        result = await _check_jobset_installed(mock_api)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_jobset_installed_returns_false_on_404(
        self, mock_api: MagicMock
    ) -> None:
        mock_api.async_get = MagicMock(
            return_value=_async_iter_raise(create_server_error(404, "Not Found"))
        )
        result = await _check_jobset_installed(mock_api)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_jobset_installed_returns_false_on_not_found_error(
        self, mock_api: MagicMock
    ) -> None:
        mock_api.async_get = MagicMock(
            return_value=_async_iter_raise(create_not_found_error("jobset"))
        )
        result = await _check_jobset_installed(mock_api)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_jobset_installed_reraises_non_404_server_error(
        self, mock_api: MagicMock
    ) -> None:
        mock_api.async_get = MagicMock(
            return_value=_async_iter_raise(create_server_error(500, "Internal"))
        )
        with pytest.raises(kr8s.ServerError):
            await _check_jobset_installed(mock_api)


# ============================================================
# _resolve_jobset_version
# ============================================================


class TestResolveJobSetVersion:
    """Verify version resolution: explicit > latest > fallback."""

    @pytest.mark.asyncio
    async def test_resolve_explicit_version_returned_immediately(self) -> None:
        result = await _resolve_jobset_version("v0.8.0")
        assert result == "v0.8.0"

    @pytest.mark.asyncio
    async def test_resolve_none_fetches_latest(self) -> None:
        with patch(
            f"{_MOD}.get_latest_jobset_version", new_callable=AsyncMock
        ) as mock_latest:
            mock_latest.return_value = "v0.9.0"
            result = await _resolve_jobset_version(None)
        assert result == "v0.9.0"

    @pytest.mark.asyncio
    async def test_resolve_none_falls_back_when_latest_unavailable(self) -> None:
        with patch(
            f"{_MOD}.get_latest_jobset_version", new_callable=AsyncMock
        ) as mock_latest:
            mock_latest.return_value = None
            result = await _resolve_jobset_version(None)
        assert result == JOBSET_FALLBACK_VERSION


# ============================================================
# _run_and_report
# ============================================================


class TestRunAndReport:
    """Verify command execution with console output."""

    @pytest.mark.asyncio
    async def test_run_and_report_returns_true_on_success(
        self, ok_result: CommandResult
    ) -> None:
        with patch(f"{_MOD}.run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ok_result
            result = await _run_and_report(["kubectl", "apply"], "install failed")
        assert result is True

    @pytest.mark.asyncio
    async def test_run_and_report_returns_false_on_failure(
        self, fail_result: CommandResult
    ) -> None:
        with patch(f"{_MOD}.run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = fail_result
            result = await _run_and_report(["kubectl", "apply"], "install failed")
        assert result is False

    @pytest.mark.asyncio
    async def test_run_and_report_handles_empty_stdout(self) -> None:
        result = CommandResult(returncode=0, stdout="", stderr="")
        with patch(f"{_MOD}.run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = result
            assert await _run_and_report(["cmd"], "err") is True


# ============================================================
# _install_jobset
# ============================================================


class TestInstallJobSet:
    """Verify JobSet installation builds correct commands."""

    @pytest.mark.asyncio
    async def test_install_jobset_dry_run_returns_true_without_running(self) -> None:
        with patch(
            f"{_MOD}._resolve_jobset_version", new_callable=AsyncMock
        ) as mock_ver:
            mock_ver.return_value = "v0.7.1"
            result = await _install_jobset("/usr/bin/kubectl", dry_run=True)
        assert result is True

    @pytest.mark.asyncio
    async def test_install_jobset_builds_correct_command(
        self, ok_result: CommandResult
    ) -> None:
        with (
            patch(
                f"{_MOD}._resolve_jobset_version", new_callable=AsyncMock
            ) as mock_ver,
            patch(f"{_MOD}.run_command", new_callable=AsyncMock) as mock_run,
        ):
            mock_ver.return_value = "v0.7.1"
            mock_run.return_value = ok_result
            await _install_jobset(
                "/usr/bin/kubectl",
                version="v0.7.1",
                kubeconfig="/cfg",
                kube_context="ctx",
            )
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "/usr/bin/kubectl"
            assert "apply" in cmd
            assert "--server-side" in cmd
            assert "--kubeconfig" in cmd
            assert "/cfg" in cmd
            assert "--context" in cmd
            assert "ctx" in cmd

    @pytest.mark.asyncio
    async def test_install_jobset_returns_false_on_command_failure(
        self, fail_result: CommandResult
    ) -> None:
        with (
            patch(
                f"{_MOD}._resolve_jobset_version", new_callable=AsyncMock
            ) as mock_ver,
            patch(f"{_MOD}.run_command", new_callable=AsyncMock) as mock_run,
        ):
            mock_ver.return_value = "v0.7.1"
            mock_run.return_value = fail_result
            result = await _install_jobset("/usr/bin/kubectl")
        assert result is False


# ============================================================
# _check_namespace_exists
# ============================================================


class TestCheckNamespaceExists:
    """Verify namespace existence check."""

    @pytest.mark.asyncio
    async def test_check_namespace_returns_true_when_exists(
        self, mock_api: MagicMock
    ) -> None:
        mock_ns = AsyncMock()
        mock_ns.exists = AsyncMock(return_value=True)
        with patch(
            "kr8s.asyncio.objects.Namespace.get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_ns
            result = await _check_namespace_exists(mock_api, "default")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_namespace_returns_false_on_not_found(
        self, mock_api: MagicMock
    ) -> None:
        with patch(
            "kr8s.asyncio.objects.Namespace.get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = create_not_found_error("ns")
            result = await _check_namespace_exists(mock_api, "missing-ns")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_namespace_returns_false_on_server_error(
        self, mock_api: MagicMock
    ) -> None:
        with patch(
            "kr8s.asyncio.objects.Namespace.get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = create_server_error(500, "Internal")
            result = await _check_namespace_exists(mock_api, "ns")
        assert result is False


# ============================================================
# _create_namespace
# ============================================================


class TestCreateNamespace:
    """Verify namespace creation and dry run."""

    @pytest.mark.asyncio
    async def test_create_namespace_dry_run_returns_true(
        self, mock_api: MagicMock
    ) -> None:
        result = await _create_namespace(mock_api, "test-ns", dry_run=True)
        assert result is True

    @pytest.mark.asyncio
    async def test_create_namespace_calls_kr8s_create(
        self, mock_api: MagicMock
    ) -> None:
        mock_ns_instance = MagicMock()
        mock_ns_instance.create = AsyncMock()
        with patch("kr8s.asyncio.objects.Namespace", return_value=mock_ns_instance):
            result = await _create_namespace(mock_api, "new-ns")
        assert result is True
        mock_ns_instance.create.assert_awaited_once()


# ============================================================
# _check_operator_installed
# ============================================================


class TestCheckOperatorInstalled:
    """Verify operator Helm release detection."""

    @pytest.mark.asyncio
    async def test_check_operator_returns_true_when_installed(self) -> None:
        with patch(f"{_MOD}.check_command", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True
            result = await _check_operator_installed("/usr/bin/helm", "aiperf-system")
        assert result is True
        cmd = mock_check.call_args[0][0]
        assert OPERATOR_RELEASE_NAME in cmd
        assert "-n" in cmd
        assert "aiperf-system" in cmd

    @pytest.mark.asyncio
    async def test_check_operator_returns_false_when_not_installed(self) -> None:
        with patch(f"{_MOD}.check_command", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False
            result = await _check_operator_installed("/usr/bin/helm", "aiperf-system")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_operator_passes_kube_credentials(self) -> None:
        with patch(f"{_MOD}.check_command", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True
            await _check_operator_installed(
                "/usr/bin/helm", "ns", kubeconfig="/cfg", kube_context="ctx"
            )
            cmd = mock_check.call_args[0][0]
            assert "--kubeconfig" in cmd
            assert "--context" in cmd


# ============================================================
# _install_operator
# ============================================================


class TestInstallOperator:
    """Verify operator Helm install command construction."""

    @pytest.mark.asyncio
    async def test_install_operator_dry_run_returns_true(self) -> None:
        result = await _install_operator("/usr/bin/helm", dry_run=True)
        assert result is True

    @pytest.mark.asyncio
    async def test_install_operator_builds_correct_command(
        self, ok_result: CommandResult
    ) -> None:
        with patch(f"{_MOD}.run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ok_result
            await _install_operator(
                "/usr/bin/helm",
                chart="my-chart",
                namespace="my-ns",
                values_file="/values.yaml",
                set_values=["key1=val1", "key2=val2"],
                kubeconfig="/cfg",
                kube_context="ctx",
            )
            cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/helm"
        assert "install" in cmd
        assert OPERATOR_RELEASE_NAME in cmd
        assert "my-chart" in cmd
        assert "-n" in cmd
        assert "my-ns" in cmd
        assert "--create-namespace" in cmd
        assert "-f" in cmd
        assert "/values.yaml" in cmd
        assert cmd.count("--set") == 2
        assert "--kubeconfig" in cmd
        assert "--context" in cmd

    @pytest.mark.asyncio
    async def test_install_operator_omits_values_file_when_none(
        self, ok_result: CommandResult
    ) -> None:
        with patch(f"{_MOD}.run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ok_result
            await _install_operator("/usr/bin/helm")
            cmd = mock_run.call_args[0][0]
        assert "-f" not in cmd

    @pytest.mark.asyncio
    async def test_install_operator_omits_set_when_none(
        self, ok_result: CommandResult
    ) -> None:
        with patch(f"{_MOD}.run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ok_result
            await _install_operator("/usr/bin/helm")
            cmd = mock_run.call_args[0][0]
        assert "--set" not in cmd

    @pytest.mark.asyncio
    async def test_install_operator_returns_false_on_failure(
        self, fail_result: CommandResult
    ) -> None:
        with patch(f"{_MOD}.run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = fail_result
            result = await _install_operator("/usr/bin/helm")
        assert result is False


# ============================================================
# _upgrade_operator
# ============================================================


class TestUpgradeOperator:
    """Verify operator Helm upgrade command construction."""

    @pytest.mark.asyncio
    async def test_upgrade_operator_dry_run_returns_true(self) -> None:
        result = await _upgrade_operator("/usr/bin/helm", dry_run=True)
        assert result is True

    @pytest.mark.asyncio
    async def test_upgrade_operator_uses_upgrade_subcommand(
        self, ok_result: CommandResult
    ) -> None:
        with patch(f"{_MOD}.run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ok_result
            await _upgrade_operator("/usr/bin/helm")
            cmd = mock_run.call_args[0][0]
        assert "upgrade" in cmd
        assert "--create-namespace" not in cmd

    @pytest.mark.asyncio
    async def test_upgrade_operator_includes_values_and_set(
        self, ok_result: CommandResult
    ) -> None:
        with patch(f"{_MOD}.run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ok_result
            await _upgrade_operator(
                "/usr/bin/helm",
                values_file="/v.yaml",
                set_values=["a=b"],
            )
            cmd = mock_run.call_args[0][0]
        assert "-f" in cmd
        assert "/v.yaml" in cmd
        assert "--set" in cmd
        assert "a=b" in cmd

    @pytest.mark.asyncio
    async def test_upgrade_operator_returns_false_on_failure(
        self, fail_result: CommandResult
    ) -> None:
        with patch(f"{_MOD}.run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = fail_result
            result = await _upgrade_operator("/usr/bin/helm")
        assert result is False


# ============================================================
# run_setup - Orchestration
# ============================================================


class TestRunSetup:
    """Verify full setup orchestration under various flag combinations."""

    @pytest.fixture
    def patch_get_api(self, mock_api: MagicMock):
        """Patch get_api to return mock_api."""
        with patch(
            "aiperf.kubernetes.client.get_api", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_api
            yield mock_get

    @pytest.fixture
    def patch_binaries(self):
        """Patch binary discovery to return fake paths."""
        with (
            patch(f"{_MOD}._find_kubectl", return_value="/usr/bin/kubectl"),
            patch(f"{_MOD}._find_helm", return_value="/usr/bin/helm"),
        ):
            yield

    @pytest.mark.asyncio
    async def test_run_setup_returns_false_when_cluster_unreachable(self) -> None:
        with patch(
            "aiperf.kubernetes.client.get_api", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = ConnectionError("refused")
            result = await run_setup()
        assert result is False

    @pytest.mark.asyncio
    async def test_run_setup_skips_jobset_when_flag_set(
        self, patch_get_api: AsyncMock, mock_api: MagicMock
    ) -> None:
        with patch(
            f"{_MOD}._check_jobset_installed", new_callable=AsyncMock
        ) as mock_check:
            result = await run_setup(skip_jobset=True)
        mock_check.assert_not_awaited()
        assert result is True

    @pytest.mark.asyncio
    async def test_run_setup_skips_jobset_install_when_already_present(
        self, patch_get_api: AsyncMock, mock_api: MagicMock
    ) -> None:
        with (
            patch(
                f"{_MOD}._check_jobset_installed", new_callable=AsyncMock
            ) as mock_check,
            patch(f"{_MOD}._install_jobset", new_callable=AsyncMock) as mock_install,
        ):
            mock_check.return_value = True
            result = await run_setup()
        mock_install.assert_not_awaited()
        assert result is True

    @pytest.mark.asyncio
    async def test_run_setup_installs_jobset_when_missing(
        self, patch_get_api: AsyncMock, patch_binaries: None, mock_api: MagicMock
    ) -> None:
        with (
            patch(
                f"{_MOD}._check_jobset_installed", new_callable=AsyncMock
            ) as mock_check,
            patch(f"{_MOD}._install_jobset", new_callable=AsyncMock) as mock_install,
        ):
            mock_check.return_value = False
            mock_install.return_value = True
            result = await run_setup()
        mock_install.assert_awaited_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_run_setup_returns_false_when_jobset_install_fails(
        self, patch_get_api: AsyncMock, patch_binaries: None, mock_api: MagicMock
    ) -> None:
        with (
            patch(
                f"{_MOD}._check_jobset_installed", new_callable=AsyncMock
            ) as mock_check,
            patch(f"{_MOD}._install_jobset", new_callable=AsyncMock) as mock_install,
        ):
            mock_check.return_value = False
            mock_install.return_value = False
            result = await run_setup()
        assert result is False

    @pytest.mark.asyncio
    async def test_run_setup_installs_operator_when_not_present(
        self, patch_get_api: AsyncMock, patch_binaries: None, mock_api: MagicMock
    ) -> None:
        with (
            patch(f"{_MOD}._check_jobset_installed", new_callable=AsyncMock) as mock_js,
            patch(
                f"{_MOD}._check_operator_installed", new_callable=AsyncMock
            ) as mock_op_check,
            patch(
                f"{_MOD}._install_operator", new_callable=AsyncMock
            ) as mock_op_install,
        ):
            mock_js.return_value = True
            mock_op_check.return_value = False
            mock_op_install.return_value = True
            result = await run_setup(operator=True)
        mock_op_install.assert_awaited_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_run_setup_upgrades_operator_when_already_installed(
        self, patch_get_api: AsyncMock, patch_binaries: None, mock_api: MagicMock
    ) -> None:
        with (
            patch(f"{_MOD}._check_jobset_installed", new_callable=AsyncMock) as mock_js,
            patch(
                f"{_MOD}._check_operator_installed", new_callable=AsyncMock
            ) as mock_op_check,
            patch(
                f"{_MOD}._upgrade_operator", new_callable=AsyncMock
            ) as mock_op_upgrade,
        ):
            mock_js.return_value = True
            mock_op_check.return_value = True
            mock_op_upgrade.return_value = True
            result = await run_setup(operator=True)
        mock_op_upgrade.assert_awaited_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_run_setup_returns_false_when_operator_install_fails(
        self, patch_get_api: AsyncMock, patch_binaries: None, mock_api: MagicMock
    ) -> None:
        with (
            patch(f"{_MOD}._check_jobset_installed", new_callable=AsyncMock) as mock_js,
            patch(
                f"{_MOD}._check_operator_installed", new_callable=AsyncMock
            ) as mock_op_check,
            patch(
                f"{_MOD}._install_operator", new_callable=AsyncMock
            ) as mock_op_install,
        ):
            mock_js.return_value = True
            mock_op_check.return_value = False
            mock_op_install.return_value = False
            result = await run_setup(operator=True)
        assert result is False

    @pytest.mark.asyncio
    async def test_run_setup_returns_false_when_operator_upgrade_fails(
        self, patch_get_api: AsyncMock, patch_binaries: None, mock_api: MagicMock
    ) -> None:
        with (
            patch(f"{_MOD}._check_jobset_installed", new_callable=AsyncMock) as mock_js,
            patch(
                f"{_MOD}._check_operator_installed", new_callable=AsyncMock
            ) as mock_op_check,
            patch(
                f"{_MOD}._upgrade_operator", new_callable=AsyncMock
            ) as mock_op_upgrade,
        ):
            mock_js.return_value = True
            mock_op_check.return_value = True
            mock_op_upgrade.return_value = False
            result = await run_setup(operator=True)
        assert result is False

    @pytest.mark.asyncio
    async def test_run_setup_creates_namespace_when_missing(
        self, patch_get_api: AsyncMock, mock_api: MagicMock
    ) -> None:
        with (
            patch(f"{_MOD}._check_jobset_installed", new_callable=AsyncMock) as mock_js,
            patch(
                f"{_MOD}._check_namespace_exists", new_callable=AsyncMock
            ) as mock_ns_check,
            patch(
                f"{_MOD}._create_namespace", new_callable=AsyncMock
            ) as mock_ns_create,
        ):
            mock_js.return_value = True
            mock_ns_check.return_value = False
            mock_ns_create.return_value = True
            result = await run_setup(namespace="bench-ns", skip_jobset=True)
        mock_ns_create.assert_awaited_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_run_setup_skips_namespace_creation_when_exists(
        self, patch_get_api: AsyncMock, mock_api: MagicMock
    ) -> None:
        with (
            patch(f"{_MOD}._check_jobset_installed", new_callable=AsyncMock) as mock_js,
            patch(
                f"{_MOD}._check_namespace_exists", new_callable=AsyncMock
            ) as mock_ns_check,
            patch(
                f"{_MOD}._create_namespace", new_callable=AsyncMock
            ) as mock_ns_create,
        ):
            mock_js.return_value = True
            mock_ns_check.return_value = True
            result = await run_setup(namespace="bench-ns", skip_jobset=True)
        mock_ns_create.assert_not_awaited()
        assert result is True

    @pytest.mark.asyncio
    async def test_run_setup_returns_false_when_namespace_creation_raises(
        self, patch_get_api: AsyncMock, mock_api: MagicMock
    ) -> None:
        with (
            patch(f"{_MOD}._check_jobset_installed", new_callable=AsyncMock) as mock_js,
            patch(
                f"{_MOD}._check_namespace_exists", new_callable=AsyncMock
            ) as mock_ns_check,
            patch(
                f"{_MOD}._create_namespace", new_callable=AsyncMock
            ) as mock_ns_create,
        ):
            mock_js.return_value = True
            mock_ns_check.return_value = False
            mock_ns_create.side_effect = RuntimeError("API error")
            result = await run_setup(namespace="bench-ns", skip_jobset=True)
        assert result is False

    @pytest.mark.asyncio
    async def test_run_setup_no_namespace_arg_skips_namespace_step(
        self, patch_get_api: AsyncMock, mock_api: MagicMock
    ) -> None:
        with (
            patch(f"{_MOD}._check_jobset_installed", new_callable=AsyncMock) as mock_js,
            patch(
                f"{_MOD}._check_namespace_exists", new_callable=AsyncMock
            ) as mock_ns_check,
        ):
            mock_js.return_value = True
            result = await run_setup(skip_jobset=True)
        mock_ns_check.assert_not_awaited()
        assert result is True

    @pytest.mark.asyncio
    async def test_run_setup_dry_run_returns_true(
        self, patch_get_api: AsyncMock, patch_binaries: None, mock_api: MagicMock
    ) -> None:
        with (
            patch(f"{_MOD}._check_jobset_installed", new_callable=AsyncMock) as mock_js,
            patch(f"{_MOD}._install_jobset", new_callable=AsyncMock) as mock_install,
        ):
            mock_js.return_value = False
            mock_install.return_value = True
            result = await run_setup(dry_run=True)
        assert result is True

    @pytest.mark.asyncio
    async def test_run_setup_passes_kube_credentials_through(
        self, patch_binaries: None
    ) -> None:
        mock_api = MagicMock()
        mock_api.async_version = AsyncMock(return_value={"gitVersion": "v1.30.0"})
        with (
            patch(
                "aiperf.kubernetes.client.get_api", new_callable=AsyncMock
            ) as mock_get_api,
            patch(f"{_MOD}._check_jobset_installed", new_callable=AsyncMock) as mock_js,
            patch(f"{_MOD}._install_jobset", new_callable=AsyncMock) as mock_install,
        ):
            mock_get_api.return_value = mock_api
            mock_js.return_value = False
            mock_install.return_value = True
            await run_setup(kubeconfig="/my/cfg", kube_context="my-ctx")
        mock_get_api.assert_awaited_once_with(
            kubeconfig="/my/cfg", kube_context="my-ctx"
        )
        _, kwargs = mock_install.call_args
        assert kwargs["kubeconfig"] == "/my/cfg"
        assert kwargs["kube_context"] == "my-ctx"


# ============================================================
# Constants
# ============================================================


class TestModuleConstants:
    """Verify module-level constants have expected values."""

    def test_operator_release_name(self) -> None:
        assert OPERATOR_RELEASE_NAME == "aiperf-operator"

    def test_operator_default_namespace(self) -> None:
        assert OPERATOR_DEFAULT_NAMESPACE == "aiperf-system"

    def test_operator_default_chart(self) -> None:
        assert OPERATOR_DEFAULT_CHART == "oci://nvcr.io/nvidia/aiperf-operator"


# ============================================================
# Async Helpers (for mocking async iterators)
# ============================================================


async def _async_iter(items: list[Any]) -> Any:
    """Async generator from a list (for mocking api.async_get)."""
    for item in items:
        yield item


async def _async_iter_raise(exc: Exception) -> Any:
    """Async generator that raises an exception on iteration."""
    raise exc
    yield  # noqa: RET503 - makes this a generator
