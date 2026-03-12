# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.setup module."""

from unittest.mock import AsyncMock, MagicMock, patch

import kr8s
import pytest
from pytest import param

from aiperf.cli_commands.kube.setup import setup
from aiperf.common.config.kube_config import KubeManageOptions
from aiperf.kubernetes.jobset import (
    JOBSET_FALLBACK_VERSION,
    get_jobset_manifest_url,
)
from aiperf.kubernetes.setup import (
    OPERATOR_DEFAULT_CHART,
    OPERATOR_DEFAULT_NAMESPACE,
    OPERATOR_RELEASE_NAME,
    _check_jobset_installed,
    _check_namespace_exists,
    _check_operator_installed,
    _create_namespace,
    _find_helm,
    _find_kubectl,
    _install_jobset,
    _install_operator,
    _kube_cli_args,
    _resolve_jobset_version,
    _upgrade_operator,
    run_setup,
)
from aiperf.kubernetes.subproc import CommandResult


def _mock_server_error(status_code: int = 500) -> kr8s.ServerError:
    """Create a kr8s.ServerError with a mock response carrying the given status."""
    resp = MagicMock()
    resp.status_code = status_code
    return kr8s.ServerError("error", response=resp)


# Default operator kwargs for TestSetupCommand assertions
_OPERATOR_DEFAULTS = {
    "operator": False,
    "operator_namespace": OPERATOR_DEFAULT_NAMESPACE,
    "operator_chart": OPERATOR_DEFAULT_CHART,
    "operator_values": None,
    "operator_set": None,
}

# =============================================================================
# _kube_cli_args
# =============================================================================


class TestKubeCliArgs:
    def test_empty_when_none(self) -> None:
        assert _kube_cli_args(None, None) == []

    def test_kubeconfig_only(self) -> None:
        assert _kube_cli_args("/path/to/kc", None) == ["--kubeconfig", "/path/to/kc"]

    def test_context_only(self) -> None:
        assert _kube_cli_args(None, "my-ctx") == ["--context", "my-ctx"]

    def test_both(self) -> None:
        assert _kube_cli_args("/kc", "ctx") == [
            "--kubeconfig",
            "/kc",
            "--context",
            "ctx",
        ]


# =============================================================================
# _find_kubectl
# =============================================================================


class TestFindKubectl:
    def test_returns_path_when_found(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.shutil.which",
            return_value="/usr/bin/kubectl",
        ):
            assert _find_kubectl() == "/usr/bin/kubectl"

    def test_returns_alternative_path(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.shutil.which",
            return_value="/usr/local/bin/kubectl",
        ):
            assert _find_kubectl() == "/usr/local/bin/kubectl"

    def test_exits_when_not_found(self) -> None:
        with (
            patch("aiperf.kubernetes.setup.shutil.which", return_value=None),
            pytest.raises(SystemExit, match="1"),
        ):
            _find_kubectl()


# =============================================================================
# _find_helm
# =============================================================================


class TestFindHelm:
    def test_returns_path_when_found(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.shutil.which",
            return_value="/usr/bin/helm",
        ):
            assert _find_helm() == "/usr/bin/helm"

    def test_returns_alternative_path(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.shutil.which",
            return_value="/usr/local/bin/helm",
        ):
            assert _find_helm() == "/usr/local/bin/helm"

    def test_exits_when_not_found(self) -> None:
        with (
            patch("aiperf.kubernetes.setup.shutil.which", return_value=None),
            pytest.raises(SystemExit, match="1"),
        ):
            _find_helm()


# =============================================================================
# _check_jobset_installed
# =============================================================================


class TestCheckJobsetInstalled:
    @staticmethod
    def _make_api(results: list | None = None, error: Exception | None = None):
        """Build a mock kr8s api whose async_get returns an async iterator."""
        api = MagicMock()

        async def _async_get(*args, **kwargs):
            if error is not None:
                raise error
            for item in results or []:
                yield item

        api.async_get = _async_get
        return api

    async def test_returns_true_when_installed(self) -> None:
        api = self._make_api(results=[])
        assert await _check_jobset_installed(api) is True

    async def test_returns_true_with_existing_jobsets(self) -> None:
        api = self._make_api(results=[MagicMock()])
        assert await _check_jobset_installed(api) is True

    async def test_returns_false_when_not_found(self) -> None:
        api = self._make_api(error=kr8s.NotFoundError("not found"))
        assert await _check_jobset_installed(api) is False

    async def test_returns_false_on_404_server_error(self) -> None:
        api = self._make_api(error=_mock_server_error(404))
        assert await _check_jobset_installed(api) is False

    async def test_raises_on_non_404_server_error(self) -> None:
        api = self._make_api(error=_mock_server_error(500))
        with pytest.raises(kr8s.ServerError):
            await _check_jobset_installed(api)


# =============================================================================
# _resolve_jobset_version
# =============================================================================


class TestResolveJobsetVersion:
    async def test_returns_explicit_version(self) -> None:
        assert await _resolve_jobset_version("v0.5.2") == "v0.5.2"

    async def test_returns_explicit_version_without_github_call(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.get_latest_jobset_version",
            new_callable=AsyncMock,
        ) as mock_latest:
            await _resolve_jobset_version("v0.6.0")
            mock_latest.assert_not_called()

    async def test_queries_github_when_none(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.get_latest_jobset_version",
            new_callable=AsyncMock,
            return_value="v0.7.1",
        ):
            assert await _resolve_jobset_version(None) == "v0.7.1"

    async def test_falls_back_when_github_fails(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.get_latest_jobset_version",
            new_callable=AsyncMock,
            return_value=None,
        ):
            assert await _resolve_jobset_version(None) == JOBSET_FALLBACK_VERSION

    @pytest.mark.parametrize(
        "explicit_version",
        [
            param("v0.5.0", id="older"),
            param("v1.0.0", id="major"),
            param("v0.5.2-rc1", id="prerelease"),
        ],
    )  # fmt: skip
    async def test_accepts_any_version_string(self, explicit_version: str) -> None:
        assert await _resolve_jobset_version(explicit_version) == explicit_version


# =============================================================================
# _install_jobset
# =============================================================================


class TestInstallJobset:
    @pytest.fixture(autouse=True)
    def _mock_resolve(self):
        """Mock version resolution for all install tests."""
        with patch(
            "aiperf.kubernetes.setup._resolve_jobset_version",
            new_callable=AsyncMock,
            return_value="v0.5.2",
        ) as mock:
            self.mock_resolve = mock
            yield

    async def test_dry_run_does_not_execute(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            result = await _install_jobset(
                "/usr/bin/kubectl", version="v0.5.2", dry_run=True
            )
            assert result is True
            mock_run.assert_not_called()

    async def test_dry_run_still_resolves_version(self) -> None:
        with patch("aiperf.kubernetes.setup.run_command", new_callable=AsyncMock):
            await _install_jobset("/usr/bin/kubectl", dry_run=True)
            self.mock_resolve.assert_called_once()

    async def test_runs_kubectl_apply_with_resolved_version(self) -> None:
        self.mock_resolve.return_value = "v0.7.1"
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(
                returncode=0, stdout="created", stderr=""
            )
            result = await _install_jobset("/usr/bin/kubectl")
            assert result is True
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "/usr/bin/kubectl"
            assert args[1] == "apply"
            assert "--server-side" in args
            assert get_jobset_manifest_url("v0.7.1") in args

    async def test_passes_version_to_resolve(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _install_jobset("/usr/bin/kubectl", version="v0.6.0")
            self.mock_resolve.assert_called_once_with("v0.6.0")

    async def test_passes_none_version_to_resolve(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _install_jobset("/usr/bin/kubectl")
            self.mock_resolve.assert_called_once_with(None)

    async def test_returns_false_on_failure(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(
                returncode=1, stdout="", stderr="permission denied"
            )
            result = await _install_jobset("/usr/bin/kubectl")
            assert result is False

    async def test_handles_empty_stdout(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            assert await _install_jobset("/usr/bin/kubectl") is True

    async def test_forwards_kubeconfig_and_context(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _install_jobset(
                "/usr/bin/kubectl", kubeconfig="/kc", kube_context="ctx"
            )
            args = mock_run.call_args[0][0]
            assert "--kubeconfig" in args
            assert "/kc" in args
            assert "--context" in args
            assert "ctx" in args

    async def test_no_kube_args_when_none(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _install_jobset("/usr/bin/kubectl")
            args = mock_run.call_args[0][0]
            assert "--kubeconfig" not in args
            assert "--context" not in args


# =============================================================================
# _check_namespace_exists
# =============================================================================


class TestCheckNamespaceExists:
    async def test_returns_true_when_exists(self) -> None:
        mock_ns = AsyncMock()
        mock_ns.exists.return_value = True
        with patch(
            "kr8s.asyncio.objects.Namespace.get",
            new=AsyncMock(return_value=mock_ns),
        ):
            api = MagicMock()
            assert await _check_namespace_exists(api, "test-ns") is True

    async def test_returns_false_when_not_found(self) -> None:
        with patch(
            "kr8s.asyncio.objects.Namespace.get",
            new=AsyncMock(side_effect=kr8s.NotFoundError("not found")),
        ):
            api = MagicMock()
            assert await _check_namespace_exists(api, "test-ns") is False

    async def test_returns_false_on_server_error(self) -> None:
        with patch(
            "kr8s.asyncio.objects.Namespace.get",
            new=AsyncMock(side_effect=kr8s.ServerError("error", response=MagicMock())),
        ):
            api = MagicMock()
            assert await _check_namespace_exists(api, "test-ns") is False


# =============================================================================
# _create_namespace
# =============================================================================


class TestCreateNamespace:
    async def test_dry_run_does_not_create(self) -> None:
        mock_ns_cls = MagicMock()
        with patch("kr8s.asyncio.objects.Namespace", mock_ns_cls):
            api = MagicMock()
            result = await _create_namespace(api, "test-ns", dry_run=True)
            assert result is True
            mock_ns_cls.assert_not_called()

    async def test_creates_namespace(self) -> None:
        mock_ns = AsyncMock()
        mock_ns_cls = MagicMock(return_value=mock_ns)
        with patch("kr8s.asyncio.objects.Namespace", mock_ns_cls):
            api = MagicMock()
            result = await _create_namespace(api, "test-ns")
            assert result is True
            mock_ns.create.assert_called_once()

    async def test_creates_with_correct_name(self) -> None:
        mock_ns = AsyncMock()
        mock_ns_cls = MagicMock(return_value=mock_ns)
        with patch("kr8s.asyncio.objects.Namespace", mock_ns_cls):
            api = MagicMock()
            await _create_namespace(api, "my-benchmark-ns")
            call_args = mock_ns_cls.call_args
            assert call_args[0][0]["metadata"]["name"] == "my-benchmark-ns"


# =============================================================================
# _check_operator_installed
# =============================================================================


class TestCheckOperatorInstalled:
    async def test_returns_true_when_installed(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.check_command", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = True
            assert (
                await _check_operator_installed("/usr/bin/helm", "aiperf-system")
                is True
            )

    async def test_returns_false_when_not_installed(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.check_command", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = False
            assert (
                await _check_operator_installed("/usr/bin/helm", "aiperf-system")
                is False
            )

    async def test_calls_helm_status_with_correct_args(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.check_command", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = True
            await _check_operator_installed("/usr/bin/helm", "my-namespace")
            mock_check.assert_called_once_with(
                [
                    "/usr/bin/helm",
                    "status",
                    OPERATOR_RELEASE_NAME,
                    "-n",
                    "my-namespace",
                ]
            )

    async def test_forwards_kubeconfig_and_context(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.check_command", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = True
            await _check_operator_installed(
                "/usr/bin/helm", "ns", kubeconfig="/kc", kube_context="ctx"
            )
            args = mock_check.call_args[0][0]
            assert "--kubeconfig" in args
            assert "/kc" in args
            assert "--context" in args
            assert "ctx" in args


# =============================================================================
# _install_operator
# =============================================================================


class TestInstallOperator:
    async def test_dry_run_does_not_execute(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            result = await _install_operator("/usr/bin/helm", dry_run=True)
            assert result is True
            mock_run.assert_not_called()

    async def test_runs_helm_install_with_defaults(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(
                returncode=0, stdout="deployed", stderr=""
            )
            result = await _install_operator("/usr/bin/helm")
            assert result is True
            args = mock_run.call_args[0][0]
            assert args[0] == "/usr/bin/helm"
            assert args[1] == "install"
            assert OPERATOR_RELEASE_NAME in args
            assert OPERATOR_DEFAULT_CHART in args
            assert "-n" in args
            assert OPERATOR_DEFAULT_NAMESPACE in args
            assert "--create-namespace" in args

    async def test_uses_custom_chart(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _install_operator("/usr/bin/helm", chart="./local/chart")
            args = mock_run.call_args[0][0]
            assert "./local/chart" in args

    async def test_uses_custom_namespace(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _install_operator("/usr/bin/helm", namespace="custom-ns")
            args = mock_run.call_args[0][0]
            idx = args.index("-n")
            assert args[idx + 1] == "custom-ns"

    async def test_adds_values_file(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _install_operator("/usr/bin/helm", values_file="/path/to/values.yaml")
            args = mock_run.call_args[0][0]
            assert "-f" in args
            assert "/path/to/values.yaml" in args

    async def test_adds_set_values(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _install_operator(
                "/usr/bin/helm", set_values=["image.tag=v1.0", "replicas=2"]
            )
            args = mock_run.call_args[0][0]
            assert "--set" in args
            assert "image.tag=v1.0" in args
            assert "replicas=2" in args

    async def test_returns_false_on_failure(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(
                returncode=1, stdout="", stderr="chart not found"
            )
            assert await _install_operator("/usr/bin/helm") is False

    async def test_handles_empty_stdout(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            assert await _install_operator("/usr/bin/helm") is True

    async def test_no_values_file_when_none(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _install_operator("/usr/bin/helm")
            args = mock_run.call_args[0][0]
            assert "-f" not in args

    async def test_no_set_values_when_none(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _install_operator("/usr/bin/helm")
            args = mock_run.call_args[0][0]
            assert "--set" not in args

    async def test_forwards_kubeconfig_and_context(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _install_operator(
                "/usr/bin/helm", kubeconfig="/kc", kube_context="ctx"
            )
            args = mock_run.call_args[0][0]
            assert "--kubeconfig" in args
            assert "/kc" in args
            assert "--context" in args
            assert "ctx" in args


# =============================================================================
# _upgrade_operator
# =============================================================================


class TestUpgradeOperator:
    async def test_dry_run_does_not_execute(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            result = await _upgrade_operator("/usr/bin/helm", dry_run=True)
            assert result is True
            mock_run.assert_not_called()

    async def test_runs_helm_upgrade_with_defaults(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(
                returncode=0, stdout="upgraded", stderr=""
            )
            result = await _upgrade_operator("/usr/bin/helm")
            assert result is True
            args = mock_run.call_args[0][0]
            assert args[0] == "/usr/bin/helm"
            assert args[1] == "upgrade"
            assert OPERATOR_RELEASE_NAME in args
            assert OPERATOR_DEFAULT_CHART in args

    async def test_upgrade_does_not_include_create_namespace(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _upgrade_operator("/usr/bin/helm")
            args = mock_run.call_args[0][0]
            assert "--create-namespace" not in args

    async def test_uses_custom_chart(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _upgrade_operator("/usr/bin/helm", chart="./local/chart")
            args = mock_run.call_args[0][0]
            assert "./local/chart" in args

    async def test_adds_values_file(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _upgrade_operator("/usr/bin/helm", values_file="/path/to/values.yaml")
            args = mock_run.call_args[0][0]
            assert "-f" in args
            assert "/path/to/values.yaml" in args

    async def test_adds_set_values(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _upgrade_operator(
                "/usr/bin/helm", set_values=["image.tag=v2.0", "replicas=3"]
            )
            args = mock_run.call_args[0][0]
            assert "--set" in args
            assert "image.tag=v2.0" in args
            assert "replicas=3" in args

    async def test_returns_false_on_failure(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(
                returncode=1, stdout="", stderr="upgrade failed"
            )
            assert await _upgrade_operator("/usr/bin/helm") is False

    async def test_forwards_kubeconfig_and_context(self) -> None:
        with patch(
            "aiperf.kubernetes.setup.run_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            await _upgrade_operator(
                "/usr/bin/helm", kubeconfig="/kc", kube_context="ctx"
            )
            args = mock_run.call_args[0][0]
            assert "--kubeconfig" in args
            assert "/kc" in args
            assert "--context" in args
            assert "ctx" in args


# =============================================================================
# run_setup
# =============================================================================


class TestRunSetup:
    @pytest.fixture
    def mock_k8s(self):
        """Mock kr8s API client returned by get_api."""
        mock_api = AsyncMock()
        mock_api.async_version = AsyncMock(return_value={"gitVersion": "v1.28.0"})

        with patch(
            "aiperf.kubernetes.client.get_api",
            new=AsyncMock(return_value=mock_api),
        ) as mock_get_api:
            yield {
                "api": mock_api,
                "get_api": mock_get_api,
            }

    @pytest.fixture(autouse=True)
    def _default_helpers(self, request):
        """Provide default mocks for helper functions used by run_setup.

        Tests that use mock_k8s get auto-patched helpers.
        Tests without mock_k8s (e.g. connection error tests) are skipped.
        """
        if "mock_k8s" not in request.fixturenames:
            yield
            return

        with (
            patch(
                "aiperf.kubernetes.setup._check_jobset_installed",
                new=AsyncMock(return_value=True),
            ) as mock_check_js,
            patch(
                "aiperf.kubernetes.setup._check_namespace_exists",
                new=AsyncMock(return_value=False),
            ) as mock_check_ns,
            patch(
                "aiperf.kubernetes.setup._create_namespace",
                new=AsyncMock(return_value=True),
            ) as mock_create_ns,
        ):
            self._mock_check_jobset = mock_check_js
            self._mock_check_ns = mock_check_ns
            self._mock_create_ns = mock_create_ns
            yield

    # -- Connectivity --

    async def test_setup_fails_on_connection_error(self) -> None:
        """Setup fails gracefully when cluster is unreachable."""
        with patch(
            "aiperf.kubernetes.client.get_api",
            new=AsyncMock(side_effect=Exception("connection refused")),
        ):
            assert await run_setup() is False

    async def test_setup_forwards_kubeconfig(self, mock_k8s) -> None:
        """Setup passes kubeconfig to get_api."""
        await run_setup(kubeconfig="/path/to/kubeconfig")
        mock_k8s["get_api"].assert_called_once_with(
            kubeconfig="/path/to/kubeconfig",
            kube_context=None,
        )

    async def test_setup_forwards_context(self, mock_k8s) -> None:
        """Setup passes context to get_api."""
        await run_setup(kube_context="my-context")
        mock_k8s["get_api"].assert_called_once_with(
            kubeconfig=None,
            kube_context="my-context",
        )

    async def test_setup_forwards_kubeconfig_and_context(self, mock_k8s) -> None:
        """Setup passes both kubeconfig and context."""
        await run_setup(kubeconfig="/kc", kube_context="ctx")
        mock_k8s["get_api"].assert_called_once_with(
            kubeconfig="/kc",
            kube_context="ctx",
        )

    # -- JobSet CRD --

    async def test_setup_succeeds_jobset_already_installed(self, mock_k8s) -> None:
        """Setup succeeds when JobSet is already installed."""
        assert await run_setup() is True

    async def test_setup_installs_jobset_when_missing(self, mock_k8s) -> None:
        """Setup installs JobSet CRD when not present."""
        self._mock_check_jobset.return_value = False
        with (
            patch(
                "aiperf.kubernetes.setup._find_kubectl",
                return_value="/usr/bin/kubectl",
            ),
            patch(
                "aiperf.kubernetes.setup._install_jobset",
                new=AsyncMock(return_value=True),
            ),
        ):
            assert await run_setup() is True

    async def test_setup_forwards_kubeconfig_to_jobset_install(self, mock_k8s) -> None:
        """Setup passes kubeconfig/context to _install_jobset."""
        self._mock_check_jobset.return_value = False
        with (
            patch(
                "aiperf.kubernetes.setup._find_kubectl",
                return_value="/usr/bin/kubectl",
            ),
            patch(
                "aiperf.kubernetes.setup._install_jobset",
                new=AsyncMock(return_value=True),
            ) as mock_install,
        ):
            await run_setup(kubeconfig="/kc", kube_context="ctx")
            mock_install.assert_called_once_with(
                "/usr/bin/kubectl",
                version=None,
                dry_run=False,
                kubeconfig="/kc",
                kube_context="ctx",
            )

    async def test_setup_forwards_kubeconfig_to_operator_install(
        self, mock_k8s
    ) -> None:
        """Setup passes kubeconfig/context to operator install/check."""
        with (
            patch(
                "aiperf.kubernetes.setup._find_helm",
                return_value="/usr/bin/helm",
            ),
            patch(
                "aiperf.kubernetes.setup._check_operator_installed",
                new=AsyncMock(return_value=False),
            ) as mock_check,
            patch(
                "aiperf.kubernetes.setup._install_operator",
                new=AsyncMock(return_value=True),
            ) as mock_install,
        ):
            await run_setup(operator=True, kubeconfig="/kc", kube_context="ctx")
            mock_check.assert_called_once_with(
                "/usr/bin/helm",
                OPERATOR_DEFAULT_NAMESPACE,
                kubeconfig="/kc",
                kube_context="ctx",
            )
            mock_install.assert_called_once_with(
                "/usr/bin/helm",
                chart=OPERATOR_DEFAULT_CHART,
                namespace=OPERATOR_DEFAULT_NAMESPACE,
                values_file=None,
                set_values=None,
                dry_run=False,
                kubeconfig="/kc",
                kube_context="ctx",
            )

    async def test_setup_forwards_kubeconfig_to_operator_upgrade(
        self, mock_k8s
    ) -> None:
        """Setup passes kubeconfig/context to operator upgrade."""
        with (
            patch(
                "aiperf.kubernetes.setup._find_helm",
                return_value="/usr/bin/helm",
            ),
            patch(
                "aiperf.kubernetes.setup._check_operator_installed",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "aiperf.kubernetes.setup._upgrade_operator",
                new=AsyncMock(return_value=True),
            ) as mock_upgrade,
        ):
            await run_setup(operator=True, kubeconfig="/kc", kube_context="ctx")
            mock_upgrade.assert_called_once_with(
                "/usr/bin/helm",
                chart=OPERATOR_DEFAULT_CHART,
                namespace=OPERATOR_DEFAULT_NAMESPACE,
                values_file=None,
                set_values=None,
                dry_run=False,
                kubeconfig="/kc",
                kube_context="ctx",
            )

    async def test_setup_passes_jobset_version(self, mock_k8s) -> None:
        """Setup passes --jobset-version to _install_jobset."""
        self._mock_check_jobset.return_value = False
        with (
            patch(
                "aiperf.kubernetes.setup._find_kubectl",
                return_value="/usr/bin/kubectl",
            ),
            patch(
                "aiperf.kubernetes.setup._install_jobset",
                new=AsyncMock(return_value=True),
            ) as mock_install,
        ):
            await run_setup(jobset_version="v0.6.0")
            mock_install.assert_called_once_with(
                "/usr/bin/kubectl",
                version="v0.6.0",
                dry_run=False,
                kubeconfig=None,
                kube_context=None,
            )

    async def test_setup_passes_none_version_by_default(self, mock_k8s) -> None:
        """Setup passes None version when --jobset-version not specified."""
        self._mock_check_jobset.return_value = False
        with (
            patch(
                "aiperf.kubernetes.setup._find_kubectl",
                return_value="/usr/bin/kubectl",
            ),
            patch(
                "aiperf.kubernetes.setup._install_jobset",
                new=AsyncMock(return_value=True),
            ) as mock_install,
        ):
            await run_setup()
            mock_install.assert_called_once_with(
                "/usr/bin/kubectl",
                version=None,
                dry_run=False,
                kubeconfig=None,
                kube_context=None,
            )

    async def test_setup_fails_when_jobset_install_fails(self, mock_k8s) -> None:
        """Setup returns False when JobSet installation fails."""
        self._mock_check_jobset.return_value = False
        with (
            patch(
                "aiperf.kubernetes.setup._find_kubectl",
                return_value="/usr/bin/kubectl",
            ),
            patch(
                "aiperf.kubernetes.setup._install_jobset",
                new=AsyncMock(return_value=False),
            ),
        ):
            assert await run_setup() is False

    async def test_setup_fails_when_kubectl_not_found(self, mock_k8s) -> None:
        """Setup fails when kubectl binary is not available."""
        self._mock_check_jobset.return_value = False
        with (
            patch(
                "aiperf.kubernetes.setup._find_kubectl",
                side_effect=SystemExit(1),
            ),
            pytest.raises(SystemExit),
        ):
            await run_setup()

    async def test_setup_skips_jobset_when_flag_set(self, mock_k8s) -> None:
        """Setup skips JobSet installation with --skip-jobset."""
        assert await run_setup(skip_jobset=True) is True
        self._mock_check_jobset.assert_not_called()

    # -- Namespace --

    async def test_setup_creates_namespace(self, mock_k8s) -> None:
        """Setup creates namespace when it doesn't exist."""
        # _check_namespace_exists defaults to False, _create_namespace defaults to True
        assert await run_setup(namespace="bench") is True
        self._mock_create_ns.assert_called_once()

    async def test_setup_skips_existing_namespace(self, mock_k8s) -> None:
        """Setup skips namespace creation if it already exists."""
        self._mock_check_ns.return_value = True
        assert await run_setup(namespace="existing") is True
        self._mock_create_ns.assert_not_called()

    async def test_setup_no_namespace_skips_namespace_check(self, mock_k8s) -> None:
        """Setup doesn't check namespace when not specified."""
        assert await run_setup(namespace=None) is True
        self._mock_check_ns.assert_not_called()

    async def test_setup_namespace_create_failure(self, mock_k8s) -> None:
        """Setup returns False when namespace creation fails."""
        self._mock_create_ns.side_effect = kr8s.ServerError(
            "forbidden", response=MagicMock(status_code=403)
        )
        assert await run_setup(namespace="restricted") is False

    async def test_setup_namespace_create_generic_exception(self, mock_k8s) -> None:
        """Setup handles generic exceptions during namespace creation."""
        self._mock_create_ns.side_effect = RuntimeError("quota exceeded")
        assert await run_setup(namespace="over-quota") is False

    # -- Dry run --

    async def test_setup_dry_run_makes_no_changes(self, mock_k8s) -> None:
        """Dry run does not install or create anything."""
        self._mock_check_jobset.return_value = False
        with (
            patch(
                "aiperf.kubernetes.setup._find_kubectl",
                return_value="/usr/bin/kubectl",
            ),
            patch(
                "aiperf.kubernetes.setup._install_jobset",
                new=AsyncMock(return_value=True),
            ) as mock_install,
        ):
            assert await run_setup(namespace="bench", dry_run=True) is True
            mock_install.assert_called_once_with(
                "/usr/bin/kubectl",
                version=None,
                dry_run=True,
                kubeconfig=None,
                kube_context=None,
            )

    async def test_setup_dry_run_passes_flag_to_install(self, mock_k8s) -> None:
        """Dry run forwards dry_run=True to _install_jobset."""
        self._mock_check_jobset.return_value = False
        with (
            patch(
                "aiperf.kubernetes.setup._find_kubectl",
                return_value="/usr/bin/kubectl",
            ),
            patch(
                "aiperf.kubernetes.setup._install_jobset",
                new=AsyncMock(return_value=True),
            ) as mock_install,
        ):
            await run_setup(dry_run=True)
            mock_install.assert_called_once_with(
                "/usr/bin/kubectl",
                version=None,
                dry_run=True,
                kubeconfig=None,
                kube_context=None,
            )

    # -- Combined scenarios --

    async def test_setup_jobset_fails_but_namespace_still_attempted(
        self, mock_k8s
    ) -> None:
        """Namespace creation is still attempted even if JobSet install fails."""
        self._mock_check_jobset.return_value = False
        with (
            patch(
                "aiperf.kubernetes.setup._find_kubectl",
                return_value="/usr/bin/kubectl",
            ),
            patch(
                "aiperf.kubernetes.setup._install_jobset",
                new=AsyncMock(return_value=False),
            ),
        ):
            result = await run_setup(namespace="bench")
            assert result is False
            self._mock_create_ns.assert_called_once()

    async def test_setup_all_succeed(self, mock_k8s) -> None:
        """Full setup with all steps succeeding."""
        self._mock_check_jobset.return_value = False
        with (
            patch(
                "aiperf.kubernetes.setup._find_kubectl",
                return_value="/usr/bin/kubectl",
            ),
            patch(
                "aiperf.kubernetes.setup._install_jobset",
                new=AsyncMock(return_value=True),
            ),
        ):
            assert await run_setup(namespace="bench", jobset_version="v0.7.1") is True
            self._mock_create_ns.assert_called_once()

    # -- Operator --

    async def test_setup_skips_operator_by_default(self, mock_k8s) -> None:
        """Operator is not installed unless --operator flag is set."""
        with patch("aiperf.kubernetes.setup._find_helm") as mock_helm:
            await run_setup()
            mock_helm.assert_not_called()

    async def test_setup_installs_operator_when_not_present(self, mock_k8s) -> None:
        """Setup installs operator via helm when --operator and not installed."""
        with (
            patch(
                "aiperf.kubernetes.setup._find_helm",
                return_value="/usr/bin/helm",
            ),
            patch(
                "aiperf.kubernetes.setup._check_operator_installed",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "aiperf.kubernetes.setup._install_operator",
                new=AsyncMock(return_value=True),
            ) as mock_install,
        ):
            assert await run_setup(operator=True) is True
            mock_install.assert_called_once_with(
                "/usr/bin/helm",
                chart=OPERATOR_DEFAULT_CHART,
                namespace=OPERATOR_DEFAULT_NAMESPACE,
                values_file=None,
                set_values=None,
                dry_run=False,
                kubeconfig=None,
                kube_context=None,
            )

    async def test_setup_upgrades_operator_when_already_installed(
        self, mock_k8s
    ) -> None:
        """Setup upgrades operator when it's already installed."""
        with (
            patch(
                "aiperf.kubernetes.setup._find_helm",
                return_value="/usr/bin/helm",
            ),
            patch(
                "aiperf.kubernetes.setup._check_operator_installed",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "aiperf.kubernetes.setup._upgrade_operator",
                new=AsyncMock(return_value=True),
            ) as mock_upgrade,
        ):
            assert await run_setup(operator=True) is True
            mock_upgrade.assert_called_once_with(
                "/usr/bin/helm",
                chart=OPERATOR_DEFAULT_CHART,
                namespace=OPERATOR_DEFAULT_NAMESPACE,
                values_file=None,
                set_values=None,
                dry_run=False,
                kubeconfig=None,
                kube_context=None,
            )

    async def test_setup_passes_operator_options(self, mock_k8s) -> None:
        """Setup forwards operator options to _install_operator."""
        with (
            patch(
                "aiperf.kubernetes.setup._find_helm",
                return_value="/usr/bin/helm",
            ),
            patch(
                "aiperf.kubernetes.setup._check_operator_installed",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "aiperf.kubernetes.setup._install_operator",
                new=AsyncMock(return_value=True),
            ) as mock_install,
        ):
            await run_setup(
                operator=True,
                operator_namespace="custom-ns",
                operator_chart="./local/chart",
                operator_values="/path/to/values.yaml",
                operator_set=["image.tag=v1.0"],
            )
            mock_install.assert_called_once_with(
                "/usr/bin/helm",
                chart="./local/chart",
                namespace="custom-ns",
                values_file="/path/to/values.yaml",
                set_values=["image.tag=v1.0"],
                dry_run=False,
                kubeconfig=None,
                kube_context=None,
            )

    async def test_setup_fails_when_operator_install_fails(self, mock_k8s) -> None:
        """Setup returns False when operator installation fails."""
        with (
            patch(
                "aiperf.kubernetes.setup._find_helm",
                return_value="/usr/bin/helm",
            ),
            patch(
                "aiperf.kubernetes.setup._check_operator_installed",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "aiperf.kubernetes.setup._install_operator",
                new=AsyncMock(return_value=False),
            ),
        ):
            assert await run_setup(operator=True) is False

    async def test_setup_fails_when_operator_upgrade_fails(self, mock_k8s) -> None:
        """Setup returns False when operator upgrade fails."""
        with (
            patch(
                "aiperf.kubernetes.setup._find_helm",
                return_value="/usr/bin/helm",
            ),
            patch(
                "aiperf.kubernetes.setup._check_operator_installed",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "aiperf.kubernetes.setup._upgrade_operator",
                new=AsyncMock(return_value=False),
            ),
        ):
            assert await run_setup(operator=True) is False

    async def test_setup_operator_dry_run(self, mock_k8s) -> None:
        """Operator dry run passes dry_run=True to _install_operator."""
        with (
            patch(
                "aiperf.kubernetes.setup._find_helm",
                return_value="/usr/bin/helm",
            ),
            patch(
                "aiperf.kubernetes.setup._check_operator_installed",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "aiperf.kubernetes.setup._install_operator",
                new=AsyncMock(return_value=True),
            ) as mock_install,
        ):
            await run_setup(operator=True, dry_run=True)
            mock_install.assert_called_once_with(
                "/usr/bin/helm",
                chart=OPERATOR_DEFAULT_CHART,
                namespace=OPERATOR_DEFAULT_NAMESPACE,
                values_file=None,
                set_values=None,
                dry_run=True,
                kubeconfig=None,
                kube_context=None,
            )

    async def test_setup_operator_next_steps(self, mock_k8s, capsys) -> None:
        """Setup shows operator-specific next steps when --operator used."""
        with (
            patch(
                "aiperf.kubernetes.setup._find_helm",
                return_value="/usr/bin/helm",
            ),
            patch(
                "aiperf.kubernetes.setup._check_operator_installed",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "aiperf.kubernetes.setup._install_operator",
                new=AsyncMock(return_value=True),
            ),
        ):
            await run_setup(operator=True)
        captured = capsys.readouterr()
        assert "kubectl get pods" in captured.out
        assert OPERATOR_DEFAULT_NAMESPACE in captured.out

    # -- Output verification --

    async def test_setup_outputs_success_message(self, mock_k8s, capsys) -> None:
        """Setup prints success message when everything passes."""
        await run_setup()
        captured = capsys.readouterr()
        assert "Cluster is ready for AIPerf benchmarks" in captured.out

    async def test_setup_outputs_next_steps(self, mock_k8s, capsys) -> None:
        """Setup prints next steps on success."""
        await run_setup()
        captured = capsys.readouterr()
        assert "aiperf kube init" in captured.out
        assert "aiperf kube profile" in captured.out

    async def test_setup_outputs_namespace_in_next_steps(
        self, mock_k8s, capsys
    ) -> None:
        """Setup includes namespace in next-step profile command."""
        self._mock_check_ns.return_value = True
        await run_setup(namespace="my-ns")
        captured = capsys.readouterr()
        assert "--namespace my-ns" in captured.out

    async def test_setup_outputs_dry_run_header(self, mock_k8s, capsys) -> None:
        """Dry run shows (dry run) in header."""
        await run_setup(dry_run=True)
        captured = capsys.readouterr()
        assert "dry run" in captured.out.lower()

    async def test_setup_outputs_dry_run_no_changes(self, mock_k8s, capsys) -> None:
        """Dry run says no changes were made."""
        await run_setup(dry_run=True)
        captured = capsys.readouterr()
        assert "No changes were made" in captured.out

    async def test_setup_outputs_error_message_on_failure(
        self, mock_k8s, capsys
    ) -> None:
        """Setup prints warning when steps fail."""
        self._mock_check_jobset.return_value = False
        with (
            patch(
                "aiperf.kubernetes.setup._find_kubectl",
                return_value="/usr/bin/kubectl",
            ),
            patch(
                "aiperf.kubernetes.setup._install_jobset",
                new=AsyncMock(return_value=False),
            ),
        ):
            await run_setup()
        captured = capsys.readouterr()
        assert "aiperf kube preflight" in captured.out

    async def test_setup_outputs_connection_info(self, mock_k8s, capsys) -> None:
        """Setup prints cluster version on successful connect."""
        await run_setup()
        captured = capsys.readouterr()
        assert "v1.28.0" in captured.out

    async def test_setup_outputs_connection_error(self, capsys) -> None:
        """Setup prints connection error details."""
        with patch(
            "aiperf.kubernetes.client.get_api",
            new=AsyncMock(side_effect=Exception("no route to host")),
        ):
            await run_setup()
        captured = capsys.readouterr()
        assert "no route to host" in captured.out

    async def test_setup_outputs_jobset_already_installed(
        self, mock_k8s, capsys
    ) -> None:
        """Setup indicates JobSet is already installed."""
        await run_setup()
        captured = capsys.readouterr()
        assert "already installed" in captured.out.lower()

    async def test_setup_outputs_skip_jobset_message(self, mock_k8s, capsys) -> None:
        """Setup indicates JobSet was skipped."""
        await run_setup(skip_jobset=True)
        captured = capsys.readouterr()
        assert "Skipping" in captured.out

    async def test_setup_outputs_namespace_already_exists(
        self, mock_k8s, capsys
    ) -> None:
        """Setup indicates namespace already exists."""
        self._mock_check_ns.return_value = True
        await run_setup(namespace="existing")
        captured = capsys.readouterr()
        assert "already exists" in captured.out.lower()


# =============================================================================
# setup CLI command (kube.py entry point)
# =============================================================================


class TestSetupCommand:
    """Tests for the kube setup CLI command entry point."""

    async def test_setup_calls_run_setup(self) -> None:
        """CLI setup delegates to run_setup."""
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup()
            mock_run.assert_called_once_with(
                kubeconfig=None,
                kube_context=None,
                namespace=None,
                jobset_version=None,
                dry_run=False,
                skip_jobset=False,
                **_OPERATOR_DEFAULTS,
            )

    async def test_setup_exits_on_failure(self) -> None:
        """CLI setup exits with code 1 when run_setup returns False."""
        with (
            patch(
                "aiperf.kubernetes.setup.run_setup",
                new=AsyncMock(return_value=False),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await setup()
        assert exc_info.value.code == 1

    async def test_setup_passes_namespace(self) -> None:
        """CLI setup forwards --namespace via manage_options."""
        opts = KubeManageOptions(namespace="my-ns")
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(manage_options=opts)
            mock_run.assert_called_once_with(
                kubeconfig=None,
                kube_context=None,
                namespace="my-ns",
                jobset_version=None,
                dry_run=False,
                skip_jobset=False,
                **_OPERATOR_DEFAULTS,
            )

    async def test_setup_passes_jobset_version(self) -> None:
        """CLI setup forwards --jobset-version."""
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(jobset_version="v0.7.1")
            mock_run.assert_called_once_with(
                kubeconfig=None,
                kube_context=None,
                namespace=None,
                jobset_version="v0.7.1",
                dry_run=False,
                skip_jobset=False,
                **_OPERATOR_DEFAULTS,
            )

    async def test_setup_passes_dry_run(self) -> None:
        """CLI setup forwards --dry-run."""
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(dry_run=True)
            mock_run.assert_called_once_with(
                kubeconfig=None,
                kube_context=None,
                namespace=None,
                jobset_version=None,
                dry_run=True,
                skip_jobset=False,
                **_OPERATOR_DEFAULTS,
            )

    async def test_setup_passes_skip_jobset(self) -> None:
        """CLI setup forwards --skip-jobset."""
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(skip_jobset=True)
            mock_run.assert_called_once_with(
                kubeconfig=None,
                kube_context=None,
                namespace=None,
                jobset_version=None,
                dry_run=False,
                skip_jobset=True,
                **_OPERATOR_DEFAULTS,
            )

    async def test_setup_passes_manage_options_kubeconfig(self) -> None:
        """CLI setup forwards kubeconfig from manage_options."""
        opts = KubeManageOptions(kubeconfig="/my/kubeconfig")
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(manage_options=opts)
            mock_run.assert_called_once_with(
                kubeconfig="/my/kubeconfig",
                kube_context=None,
                namespace=None,
                jobset_version=None,
                dry_run=False,
                skip_jobset=False,
                **_OPERATOR_DEFAULTS,
            )

    async def test_setup_namespace_from_manage_options(self) -> None:
        """CLI setup uses manage_options.namespace."""
        opts = KubeManageOptions(namespace="from-opts")
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(manage_options=opts)
            mock_run.assert_called_once_with(
                kubeconfig=None,
                kube_context=None,
                namespace="from-opts",
                jobset_version=None,
                dry_run=False,
                skip_jobset=False,
                **_OPERATOR_DEFAULTS,
            )

    async def test_setup_all_flags_combined(self) -> None:
        """CLI setup forwards all flags together."""
        opts = KubeManageOptions(kubeconfig="/kc", kube_context="ctx", namespace="ns")
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(
                jobset_version="v0.6.0",
                dry_run=True,
                skip_jobset=True,
                manage_options=opts,
            )
            mock_run.assert_called_once_with(
                kubeconfig="/kc",
                kube_context="ctx",
                namespace="ns",
                jobset_version="v0.6.0",
                dry_run=True,
                skip_jobset=True,
                **_OPERATOR_DEFAULTS,
            )

    # -- Operator CLI flags --

    async def test_setup_passes_operator_flag(self) -> None:
        """CLI setup forwards --operator."""
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(operator=True)
            mock_run.assert_called_once_with(
                kubeconfig=None,
                kube_context=None,
                namespace=None,
                jobset_version=None,
                dry_run=False,
                skip_jobset=False,
                operator=True,
                operator_namespace=OPERATOR_DEFAULT_NAMESPACE,
                operator_chart=OPERATOR_DEFAULT_CHART,
                operator_values=None,
                operator_set=None,
            )

    async def test_setup_passes_operator_namespace(self) -> None:
        """CLI setup forwards --operator-namespace."""
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(operator=True, operator_namespace="custom-ns")
            mock_run.assert_called_once_with(
                kubeconfig=None,
                kube_context=None,
                namespace=None,
                jobset_version=None,
                dry_run=False,
                skip_jobset=False,
                operator=True,
                operator_namespace="custom-ns",
                operator_chart=OPERATOR_DEFAULT_CHART,
                operator_values=None,
                operator_set=None,
            )

    async def test_setup_passes_operator_chart(self) -> None:
        """CLI setup forwards --operator-chart."""
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(operator=True, operator_chart="./local/chart")
            mock_run.assert_called_once_with(
                kubeconfig=None,
                kube_context=None,
                namespace=None,
                jobset_version=None,
                dry_run=False,
                skip_jobset=False,
                operator=True,
                operator_namespace=OPERATOR_DEFAULT_NAMESPACE,
                operator_chart="./local/chart",
                operator_values=None,
                operator_set=None,
            )

    async def test_setup_passes_operator_values(self) -> None:
        """CLI setup forwards --operator-values."""
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(operator=True, operator_values="/path/to/values.yaml")
            mock_run.assert_called_once_with(
                kubeconfig=None,
                kube_context=None,
                namespace=None,
                jobset_version=None,
                dry_run=False,
                skip_jobset=False,
                operator=True,
                operator_namespace=OPERATOR_DEFAULT_NAMESPACE,
                operator_chart=OPERATOR_DEFAULT_CHART,
                operator_values="/path/to/values.yaml",
                operator_set=None,
            )

    async def test_setup_passes_operator_set(self) -> None:
        """CLI setup forwards --operator-set."""
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(operator=True, operator_set=["image.tag=v1.0", "replicas=2"])
            mock_run.assert_called_once_with(
                kubeconfig=None,
                kube_context=None,
                namespace=None,
                jobset_version=None,
                dry_run=False,
                skip_jobset=False,
                operator=True,
                operator_namespace=OPERATOR_DEFAULT_NAMESPACE,
                operator_chart=OPERATOR_DEFAULT_CHART,
                operator_values=None,
                operator_set=["image.tag=v1.0", "replicas=2"],
            )

    async def test_setup_all_operator_flags_combined(self) -> None:
        """CLI setup forwards all operator flags together."""
        opts = KubeManageOptions(kubeconfig="/kc", kube_context="ctx", namespace="ns")
        with patch(
            "aiperf.kubernetes.setup.run_setup", new=AsyncMock(return_value=True)
        ) as mock_run:
            await setup(
                jobset_version="v0.6.0",
                dry_run=True,
                skip_jobset=True,
                operator=True,
                operator_namespace="custom-ns",
                operator_chart="./local/chart",
                operator_values="/path/to/values.yaml",
                operator_set=["image.tag=v1.0"],
                manage_options=opts,
            )
            mock_run.assert_called_once_with(
                kubeconfig="/kc",
                kube_context="ctx",
                namespace="ns",
                jobset_version="v0.6.0",
                dry_run=True,
                skip_jobset=True,
                operator=True,
                operator_namespace="custom-ns",
                operator_chart="./local/chart",
                operator_values="/path/to/values.yaml",
                operator_set=["image.tag=v1.0"],
            )
