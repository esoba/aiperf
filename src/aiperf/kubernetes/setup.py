# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cluster setup for AIPerf Kubernetes deployments.

Installs prerequisites (JobSet CRD) and optionally the AIPerf operator
so the cluster is ready to run AIPerf benchmarks.
"""

from __future__ import annotations

import shutil
from typing import Any

from aiperf.kubernetes.console import (
    console,
    print_error,
    print_info,
    print_step,
    print_success,
    print_warning,
)
from aiperf.kubernetes.jobset import (
    JOBSET_FALLBACK_VERSION,
    get_jobset_manifest_url,
    get_latest_jobset_version,
)
from aiperf.kubernetes.subproc import check_command, run_command

OPERATOR_RELEASE_NAME = "aiperf-operator"
OPERATOR_DEFAULT_NAMESPACE = "aiperf-system"
OPERATOR_DEFAULT_CHART = "oci://nvcr.io/nvidia/aiperf-operator"


def _kube_cli_args(kubeconfig: str | None, kube_context: str | None) -> list[str]:
    """Build --kubeconfig/--context args for kubectl and helm subprocesses."""
    args: list[str] = []
    if kubeconfig:
        args.extend(["--kubeconfig", kubeconfig])
    if kube_context:
        args.extend(["--context", kube_context])
    return args


def _find_binary(name: str, install_hint: str) -> str:
    """Find a binary in PATH or raise.

    Args:
        name: Binary name to find.
        install_hint: URL or instruction to show if not found.

    Returns:
        Path to the binary.

    Raises:
        SystemExit: If the binary is not found.
    """
    path = shutil.which(name)
    if path is None:
        print_error(f"{name} not found in PATH")
        print_info(f"Install {name}: {install_hint}")
        raise SystemExit(1)
    return path


def _find_kubectl() -> str:
    """Find kubectl binary or raise."""
    return _find_binary("kubectl", "https://kubernetes.io/docs/tasks/tools/")


def _find_helm() -> str:
    """Find helm binary or raise."""
    return _find_binary("helm", "https://helm.sh/docs/intro/install/")


async def _check_jobset_installed(api: Any) -> bool:
    """Check if JobSet CRD is already installed.

    Args:
        api: kr8s async API client.

    Returns:
        True if JobSet CRD is installed.
    """
    import kr8s

    from aiperf.kubernetes.kr8s_resources import AsyncJobSet

    try:
        _ = [js async for js in api.async_get(AsyncJobSet, namespace=kr8s.ALL)]
        return True
    except kr8s.ServerError as e:
        if e.response and e.response.status_code == 404:
            return False
        raise
    except kr8s.NotFoundError:
        return False


async def _resolve_jobset_version(version: str | None) -> str:
    """Resolve the JobSet version to install.

    Uses the provided version, or queries GitHub for the latest release,
    falling back to a known-good version if the lookup fails.

    Args:
        version: Explicit version tag, or None for auto-detect.

    Returns:
        Resolved version tag (e.g. "v0.7.1").
    """
    if version is not None:
        return version

    print_step("Checking for latest JobSet release...")
    latest = await get_latest_jobset_version()
    if latest is not None:
        print_info(f"Latest JobSet version: {latest}")
        return latest

    print_warning(
        f"Could not determine latest version, using fallback {JOBSET_FALLBACK_VERSION}"
    )
    return JOBSET_FALLBACK_VERSION


async def _run_and_report(cmd: list[str], error_msg: str) -> bool:
    """Run a command, print stdout on success or error on failure.

    Args:
        cmd: Command and arguments to execute.
        error_msg: Prefix for the error message on failure.

    Returns:
        True if the command succeeded.
    """
    result = await run_command(cmd)
    if result.ok:
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                console.print(f"  [dim]{line}[/dim]")
        return True
    print_error(f"{error_msg}: {result.stderr.strip()}")
    return False


async def _install_jobset(
    kubectl: str,
    *,
    version: str | None = None,
    dry_run: bool = False,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> bool:
    """Install JobSet CRD and controller via kubectl.

    Args:
        kubectl: Path to kubectl binary.
        version: JobSet version tag to install, or None for latest.
        dry_run: If True, only print the command without executing.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.

    Returns:
        True if installation succeeded (or dry_run).
    """
    resolved = await _resolve_jobset_version(version)
    url = get_jobset_manifest_url(resolved)
    cmd = [kubectl, "apply", "--server-side", "-f", url]
    cmd.extend(_kube_cli_args(kubeconfig, kube_context))

    if dry_run:
        print_info(f"Would run: {' '.join(cmd)}")
        return True

    print_step(f"Installing JobSet {resolved}...")
    return await _run_and_report(cmd, "Failed to install JobSet")


async def _check_namespace_exists(api: Any, namespace: str) -> bool:
    """Check if a namespace exists.

    Args:
        api: kr8s async API client.
        namespace: Namespace name.

    Returns:
        True if namespace exists.
    """
    import kr8s
    from kr8s.asyncio.objects import Namespace

    try:
        ns = await Namespace.get(namespace, api=api)
        return await ns.exists()
    except (kr8s.NotFoundError, kr8s.ServerError):
        return False


async def _create_namespace(api: Any, namespace: str, *, dry_run: bool = False) -> bool:
    """Create a Kubernetes namespace.

    Args:
        api: kr8s async API client.
        namespace: Namespace name to create.
        dry_run: If True, only print what would be done.

    Returns:
        True if namespace was created (or dry_run).
    """
    from kr8s.asyncio.objects import Namespace

    if dry_run:
        print_info(f"Would create namespace '{namespace}'")
        return True

    print_step(f"Creating namespace '{namespace}'...")
    ns = Namespace({"metadata": {"name": namespace}}, api=api)
    await ns.create()
    return True


async def _check_operator_installed(
    helm: str,
    namespace: str,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> bool:
    """Check if the AIPerf operator Helm release is already installed.

    Args:
        helm: Path to helm binary.
        namespace: Namespace to check for the release.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.

    Returns:
        True if the operator release exists.
    """
    cmd = [helm, "status", OPERATOR_RELEASE_NAME, "-n", namespace]
    cmd.extend(_kube_cli_args(kubeconfig, kube_context))
    return await check_command(cmd)


async def _install_operator(
    helm: str,
    *,
    chart: str = OPERATOR_DEFAULT_CHART,
    namespace: str = OPERATOR_DEFAULT_NAMESPACE,
    values_file: str | None = None,
    set_values: list[str] | None = None,
    dry_run: bool = False,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> bool:
    """Install the AIPerf operator via Helm.

    Args:
        helm: Path to helm binary.
        chart: Helm chart reference (OCI URL or local path).
        namespace: Namespace to install the operator in.
        values_file: Path to a values.yaml override file.
        set_values: List of --set key=value overrides.
        dry_run: If True, only print what would be done.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.

    Returns:
        True if installation succeeded (or dry_run).
    """
    cmd = [
        helm,
        "install",
        OPERATOR_RELEASE_NAME,
        chart,
        "-n",
        namespace,
        "--create-namespace",
    ]

    if values_file:
        cmd.extend(["-f", values_file])

    for sv in set_values or []:
        cmd.extend(["--set", sv])

    cmd.extend(_kube_cli_args(kubeconfig, kube_context))

    if dry_run:
        print_info(f"Would run: {' '.join(cmd)}")
        return True

    print_step("Installing AIPerf operator...")
    return await _run_and_report(cmd, "Failed to install operator")


async def _upgrade_operator(
    helm: str,
    *,
    chart: str = OPERATOR_DEFAULT_CHART,
    namespace: str = OPERATOR_DEFAULT_NAMESPACE,
    values_file: str | None = None,
    set_values: list[str] | None = None,
    dry_run: bool = False,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> bool:
    """Upgrade the AIPerf operator via Helm.

    Args:
        helm: Path to helm binary.
        chart: Helm chart reference (OCI URL or local path).
        namespace: Namespace where the operator is installed.
        values_file: Path to a values.yaml override file.
        set_values: List of --set key=value overrides.
        dry_run: If True, only print what would be done.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.

    Returns:
        True if upgrade succeeded (or dry_run).
    """
    cmd = [
        helm,
        "upgrade",
        OPERATOR_RELEASE_NAME,
        chart,
        "-n",
        namespace,
    ]

    if values_file:
        cmd.extend(["-f", values_file])

    for sv in set_values or []:
        cmd.extend(["--set", sv])

    cmd.extend(_kube_cli_args(kubeconfig, kube_context))

    if dry_run:
        print_info(f"Would run: {' '.join(cmd)}")
        return True

    print_step("Upgrading AIPerf operator...")
    return await _run_and_report(cmd, "Failed to upgrade operator")


async def run_setup(
    *,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
    namespace: str | None = None,
    jobset_version: str | None = None,
    dry_run: bool = False,
    skip_jobset: bool = False,
    operator: bool = False,
    operator_namespace: str = OPERATOR_DEFAULT_NAMESPACE,
    operator_chart: str = OPERATOR_DEFAULT_CHART,
    operator_values: str | None = None,
    operator_set: list[str] | None = None,
) -> bool:
    """Run cluster setup for AIPerf.

    Args:
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context to use.
        namespace: Namespace to create for benchmarks (if it doesn't exist).
        jobset_version: Specific JobSet version to install, or None for latest.
        dry_run: Print what would be done without making changes.
        skip_jobset: Skip JobSet CRD installation.
        operator: Install the AIPerf Kubernetes operator.
        operator_namespace: Namespace for the operator (default: aiperf-system).
        operator_chart: Helm chart reference for the operator.
        operator_values: Path to a Helm values.yaml override file.
        operator_set: List of Helm --set key=value overrides.

    Returns:
        True if all setup steps succeeded.
    """
    from aiperf.kubernetes.client import get_api

    if dry_run:
        console.print("[bold cyan]AIPerf Cluster Setup (dry run)[/bold cyan]\n")
    else:
        console.print("[bold cyan]AIPerf Cluster Setup[/bold cyan]\n")

    # Check connectivity
    print_step("Connecting to cluster...")
    try:
        api = await get_api(kubeconfig=kubeconfig, kube_context=kube_context)
        version_info = await api.async_version()
        print_success(f"Connected to Kubernetes {version_info['gitVersion']}")
    except Exception as e:
        print_error(f"Cannot connect to cluster: {e}")
        return False

    success = True

    kube_creds = {"kubeconfig": kubeconfig, "kube_context": kube_context}

    # JobSet CRD
    if skip_jobset:
        print_info("Skipping JobSet CRD installation (--skip-jobset)")
    elif await _check_jobset_installed(api):
        print_success("JobSet CRD already installed")
    else:
        kubectl = _find_kubectl()
        if not await _install_jobset(
            kubectl, version=jobset_version, dry_run=dry_run, **kube_creds
        ):
            success = False
        else:
            print_success("JobSet CRD installed")

    # Operator
    if operator:
        helm = _find_helm()
        if await _check_operator_installed(helm, operator_namespace, **kube_creds):
            print_success("AIPerf operator already installed")
            print_info("Upgrading to ensure latest version...")
            if not await _upgrade_operator(
                helm,
                chart=operator_chart,
                namespace=operator_namespace,
                values_file=operator_values,
                set_values=operator_set,
                dry_run=dry_run,
                **kube_creds,
            ):
                success = False
            else:
                print_success("AIPerf operator upgraded")
        else:
            if not await _install_operator(
                helm,
                chart=operator_chart,
                namespace=operator_namespace,
                values_file=operator_values,
                set_values=operator_set,
                dry_run=dry_run,
                **kube_creds,
            ):
                success = False
            else:
                print_success("AIPerf operator installed")

    # Namespace (for benchmarks, separate from operator namespace)
    if namespace is not None:
        if await _check_namespace_exists(api, namespace):
            print_success(f"Namespace '{namespace}' already exists")
        else:
            try:
                await _create_namespace(api, namespace, dry_run=dry_run)
                if not dry_run:
                    print_success(f"Namespace '{namespace}' created")
            except Exception as e:
                print_error(f"Failed to create namespace '{namespace}': {e}")
                success = False

    # Summary
    console.print()
    if dry_run:
        print_info("Dry run complete. No changes were made.")
    elif success:
        print_success("Cluster is ready for AIPerf benchmarks")
        print_info("Next steps:")
        if operator:
            console.print(
                f"  1. Verify operator: kubectl get pods -n {operator_namespace}\n"
                "  2. Create a benchmark: kubectl apply -f aiperfjob.yaml"
            )
        elif namespace:
            console.print(
                f"  1. aiperf kube init -o benchmark.yaml\n"
                f"  2. aiperf kube profile --user-config benchmark.yaml "
                f"--image <your-image> --namespace {namespace}"
            )
        else:
            console.print(
                "  1. aiperf kube init -o benchmark.yaml\n"
                "  2. aiperf kube profile --user-config benchmark.yaml --image <your-image>"
            )
    else:
        print_warning(
            "Setup completed with errors. Run 'aiperf kube preflight' to diagnose."
        )

    return success
