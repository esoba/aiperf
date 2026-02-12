#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Minikube development CLI.

Single-file script for building, deploying, and running AIPerf in a local
minikube cluster. On Linux with NVIDIA GPUs, GPU support is enabled; on macOS
the cluster is CPU-only. Invoke directly::

    uv run python dev/kube.py setup
    ./dev/kube.py setup
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import orjson
import yaml
from cyclopts import App, Group, Parameter
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from aiperf.common.config.user_config import UserConfig

_ANSI_ESCAPE = re.compile(r"\033\[[0-9;]*m")

# ---------------------------------------------------------------------------
# Constants (overridable via env vars)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_NAME = os.environ.get("CLUSTER_NAME") or "aiperf"
AIPERF_IMAGE = os.environ.get("AIPERF_IMAGE") or "aiperf:local"
MOCK_SERVER_IMAGE = os.environ.get("MOCK_SERVER_IMAGE") or "aiperf-mock-server:local"
JOBSET_VERSION = os.environ.get("JOBSET_VERSION") or "v0.8.0"
MOCK_SERVER_MANIFEST = PROJECT_ROOT / "dev" / "deploy" / "mock-server.yaml"
DEFAULT_BENCHMARK_CONFIG = (
    PROJECT_ROOT / "dev" / "deploy" / "test-benchmark-config.yaml"
)
DYNAMO_BENCHMARK_CONFIG = (
    PROJECT_ROOT / "dev" / "deploy" / "dynamo-benchmark-config.yaml"
)

VLLM_IMAGE = os.environ.get("VLLM_IMAGE") or "vllm/vllm-openai:latest"
VLLM_MODEL = os.environ.get("MODEL") or "Qwen/Qwen3-0.6B"
VLLM_GPUS = int(os.environ.get("GPUS") or "1")
VLLM_MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN") or "4096")
VLLM_GPU_MEM_UTIL = os.environ.get("GPU_MEM_UTIL") or "0.5"
VLLM_NAMESPACE = "vllm-server"

DYNAMO_IMAGE = (
    os.environ.get("DYNAMO_IMAGE") or "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0"
)
DYNAMO_MODE = os.environ.get("DYNAMO_MODE") or "agg"
DYNAMO_NAMESPACE = "dynamo-server"
DYNAMO_VERSION = os.environ.get("DYNAMO_VERSION") or "0.9.0"
# Single-GPU disaggregated: both prefill+decode share 1 GPU at low memory util
DYNAMO_1GPU_MEM_UTIL = os.environ.get("DYNAMO_1GPU_MEM_UTIL") or "0.3"
# KV router mode for frontend (e.g. "kv" for KV-aware routing, "round-robin")
DYNAMO_ROUTER_MODE: str | None = os.environ.get("DYNAMO_ROUTER_MODE") or None
# KVBM CPU cache size in GB for prefill workers (enables KV cache offloading)
DYNAMO_KVBM_CPU_CACHE_GB: int | None = (
    int(v) if (v := os.environ.get("DYNAMO_KVBM_CPU_CACHE_GB")) else None
)
# Connectors for prefill workers (comma-separated, e.g. "kvbm,nixl")
DYNAMO_CONNECTORS: list[str] = (
    [c.strip() for c in v.split(",") if c.strip()]
    if (v := os.environ.get("DYNAMO_CONNECTORS"))
    else []
)

MINIKUBE_MEMORY = os.environ.get("MINIKUBE_MEMORY") or "16000mb"
MINIKUBE_CPUS = os.environ.get("MINIKUBE_CPUS") or "8"

# fmt: off
_BANNER = (
    "\n"
    "       ░▒▓ █▀█ █ █▀█ █▀▀ █▀█ █▀▀ ▓▒░\n"
    "    ░░▒▒▓▓ █▀█ █ █▀▀ ██▄ █▀▄ █▀  ▓▓▒▒░░\n"
    "\n"
    "         From prefill to production.\n"
    " Minikube · Docker · GPU · Mock · vLLM · Dynamo\n"
)
# fmt: on

DEFAULT_CONFIG: str | None = os.environ.get("CONFIG") or None
DEFAULT_WORKERS = int(os.environ.get("WORKERS") or "10")
FOLLOW_DEFAULT = bool(os.environ.get("FOLLOW"))

# Reusable cyclopts type alias for --follow/-f flags
FollowFlag = Annotated[
    bool, Parameter(name=["--follow", "-f"], help="Follow log output.")
]

# ---------------------------------------------------------------------------
# CLI app + groups + shared option models
# ---------------------------------------------------------------------------

_ENV_HELP = """\
Environment variables:
  CLUSTER_NAME       Minikube profile name (default: aiperf)
  AIPERF_IMAGE       AIPerf image (default: aiperf:local)
  CONFIG             Benchmark config file
  WORKERS            Number of workers (default: 10)
  MODEL              Model name (default: Qwen/Qwen3-0.6B)
  GPUS               GPUs per instance (default: 1)
  VLLM_IMAGE         vLLM image (default: vllm/vllm-openai:latest)
  MAX_MODEL_LEN      Max context length (default: 4096)
  GPU_MEM_UTIL       GPU memory utilization (default: 0.5)
  HF_TOKEN           Hugging Face token (for gated models)
  DYNAMO_IMAGE       Dynamo image (default: nvcr.io/.../vllm-runtime:0.9.0)
  DYNAMO_MODE        agg|disagg|disagg-1gpu (default: agg)
  DYNAMO_VERSION     Dynamo operator version (default: 0.9.0)
  PLATFORM           Override platform detection (arch|debian|fedora|linux|mac)
  ARCH               Override Linux binary arch (amd64|arm64)

Examples:
  ./dev/kube.py setup
  ./dev/kube.py deploy-dynamo
  ./dev/kube.py deploy-dynamo --mode disagg-1gpu
  ./dev/kube.py deploy-vllm --model facebook/opt-125m
  ./dev/kube.py reload && ./dev/kube.py run
  ./dev/kube.py run -c my.yaml -w 20
  ./dev/kube.py deploy-lora --name my-lora --base-model Qwen/Qwen3-0.6B --source hf://org/repo
"""

app = App(
    name="aiperf-kube",
    help="AIPerf Minikube development CLI.",
    help_format="plaintext",
    help_epilogue=_ENV_HELP,
)
app.register_install_completion_command()

workflow = Group.create_ordered("Workflow")
server = Group.create_ordered("Server")
benchmark = Group.create_ordered("Benchmark")
lowlevel = Group.create_ordered("Low-level")


# fmt: off
@Parameter(name="*")
class ModelOptions(BaseModel):
    """Shared model configuration for deploy commands."""
    model:         Annotated[str, Parameter(name=["--model", "-m"])] = Field(default=VLLM_MODEL,         description="Model name.")
    gpus:          Annotated[int, Parameter(name=["--gpus", "-g"])]  = Field(default=VLLM_GPUS,          description="GPUs per instance.")
    max_model_len: int                                               = Field(default=VLLM_MAX_MODEL_LEN, description="Max context length.")

@Parameter(name="*")
class RunOptions(BaseModel):
    """Benchmark run configuration."""
    config:  Annotated[str | None, Parameter(name=["--config", "-c"])] = Field(default=DEFAULT_CONFIG,  description="Benchmark config file.")
    workers: Annotated[int,        Parameter(name=["--workers", "-w"])] = Field(default=DEFAULT_WORKERS, description="Number of workers.")

@Parameter(name="*")
class DynamoDeployOptions(BaseModel):
    """Dynamo-specific deploy configuration."""
    dynamo_image:      str              = Field(default=DYNAMO_IMAGE,              description="Dynamo container image.")
    mode:              str              = Field(default=DYNAMO_MODE,               description="Deployment mode: agg|disagg|disagg-1gpu.")
    router_mode:       str | None       = Field(default=DYNAMO_ROUTER_MODE,        description='KV router mode (e.g. "kv", "round-robin").')
    kvbm_cpu_cache_gb: int | None       = Field(default=DYNAMO_KVBM_CPU_CACHE_GB,  description="KVBM CPU cache GB for prefill workers.")
    connectors:        list[str] | None = Field(default=DYNAMO_CONNECTORS or None, description="Connectors for prefill workers (e.g. kvbm nixl).")
# fmt: on


# B008: module-level defaults for Pydantic models used in function signatures
_DEFAULT_MODEL_OPTS = ModelOptions()
_DEFAULT_RUN_OPTS = RunOptions()
_DEFAULT_DYNAMO_OPTS = DynamoDeployOptions()

# ---------------------------------------------------------------------------
# ANSI / formatting helpers
# ---------------------------------------------------------------------------

# fmt: off
_CYAN   = "\033[0;36m"
_GREEN  = "\033[0;32m"
_BLUE   = "\033[0;34m"
_YELLOW = "\033[1;33m"
_RED    = "\033[0;31m"
_NC     = "\033[0m"

def log_step(msg: str) -> None:    print(f"\n{_CYAN}>>> {msg}{_NC}")
def log_info(msg: str) -> None:    print(f"{_BLUE}[INFO]{_NC} {msg}")
def log_success(msg: str) -> None: print(f"{_GREEN}[OK]{_NC} {msg}")
def log_warn(msg: str) -> None:    print(f"{_YELLOW}[WARN]{_NC} {msg}")
def log_error(msg: str) -> None:   print(f"{_RED}[ERROR]{_NC} {msg}", file=sys.stderr)

def _ok(msg: str) -> str:   return f"  {_GREEN}\u2713{_NC} {msg}"
def _miss(msg: str) -> str: return f"  {_YELLOW}\u25cb{_NC} {msg}"
def _fail(msg: str) -> str: return f"  {_RED}\u2717{_NC} {msg}"
# fmt: on


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------


def sh(
    *cmd: str, check: bool = True, capture: bool = False, **kwargs
) -> subprocess.CompletedProcess:
    """Run a command, forwarding stdout/stderr unless *capture* is set."""
    kwargs.setdefault("text", capture)
    return subprocess.run(cmd, check=check, capture_output=capture, **kwargs)


def kubectl(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run kubectl against the minikube cluster."""
    return sh("kubectl", "--context", CLUSTER_NAME, *args, **kwargs)


def minikube(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run minikube with the cluster profile."""
    return sh("minikube", "-p", CLUSTER_NAME, *args, **kwargs)


def cluster_exists() -> bool:
    r = minikube("status", "--format={{.Host}}", capture=True, check=False)
    return r.returncode == 0 and r.stdout.strip() in ("Running", "Stopped")


def cluster_running() -> bool:
    r = minikube("status", "--format={{.Host}}", capture=True, check=False)
    return r.returncode == 0 and r.stdout.strip() == "Running"


def require(*cmds: str) -> None:
    missing = [c for c in cmds if shutil.which(c) is None]
    if missing:
        log_error(f"Required command(s) not found: {', '.join(missing)}")
        raise SystemExit(1)


def require_cluster() -> None:
    if not cluster_running():
        log_error(
            f"Cluster {CLUSTER_NAME} is not running. Run 'setup' or 'cluster-create' first."
        )
        raise SystemExit(1)


def _require_kubectl_and_cluster() -> None:
    """Require kubectl and a running cluster. Exits on failure."""
    require("kubectl")
    require_cluster()


def _require_docker_running() -> None:
    """Require Docker daemon to be running. Exits on failure."""
    r = sh("docker", "info", capture=True, check=False)
    if r.returncode != 0:
        log_error("Docker is not running. Please start Docker first.")
        raise SystemExit(1)


def _skip_if_deployment_ready(
    name: str,
    namespace: str | None,
    message: str,
) -> bool:
    """Return True if deployment exists and has ready replicas (caller should return)."""
    ready = _deployment_ready_replicas(name, namespace)
    if ready is not None and ready > 0:
        log_success(message)
        return True
    return False


def _deployment_ready_replicas(name: str, namespace: str | None = None) -> int | None:
    """Return ready replicas for a deployment, or None if it doesn't exist."""
    ns_args = ("-n", namespace) if namespace else ()
    r = kubectl(
        "get",
        "deployment",
        name,
        *ns_args,
        "-o",
        "jsonpath={.status.readyReplicas}",
        capture=True,
        check=False,
    )
    if r.returncode != 0:
        return None
    return int(r.stdout or "0")


def _get_aiperf_namespaces(*, latest_only: bool = False) -> list[str]:
    """Return names of aiperf-managed namespaces."""
    extra = (
        [
            "--sort-by=.metadata.creationTimestamp",
            "-o",
            "jsonpath={.items[-1].metadata.name}",
        ]
        if latest_only
        else ["-o", "jsonpath={.items[*].metadata.name}"]
    )
    r = kubectl(
        "get", "namespaces", "-l", "app=aiperf", *extra, capture=True, check=False
    )
    return r.stdout.split() if r.stdout.strip() else []


def _docker_build(image: str, dockerfile: str, *extra_args: str) -> None:
    """Build a Docker image from PROJECT_ROOT."""
    log_step(f"Building {image}")
    sh(
        "docker",
        "build",
        *extra_args,
        "-t",
        image,
        "-f",
        dockerfile,
        ".",
        cwd=PROJECT_ROOT,
    )
    log_success(f"Built {image}")


def _confirm(
    prompt: str, default_yes: bool = False, console: Console | None = None
) -> bool:
    """Prompt for confirmation. default_yes: [Y/n] vs [y/N]."""
    con = console or Console()
    try:
        return Confirm.ask(prompt, default=default_yes, console=con)
    except (EOFError, KeyboardInterrupt):
        return default_yes


def _run_install(cmds: list[str], *, console: Console | None = None) -> bool:
    """Run a list of shell commands, returning True if all succeed."""
    for cmd in cmds:
        if console is not None:
            console.print(Syntax(cmd, "bash", theme="monokai", line_numbers=False))
        else:
            print(f"  $ {cmd}")
        r = subprocess.run(cmd, shell=True, check=False)
        if r.returncode != 0:
            log_error(f"Command failed (exit {r.returncode}): {cmd}")
            return False
    return True


def _ensure_hf_token_secret(namespace: str) -> str | None:
    """Create HF token secret if HF_TOKEN env var is set. Returns secret name or None."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return None
    log_info("Creating HF token secret...")
    kubectl("create", "namespace", namespace, check=False)
    kubectl(
        "create",
        "secret",
        "generic",
        "hf-token",
        f"--from-literal=token={hf_token}",
        "-n",
        namespace,
        check=False,
    )
    return "hf-token"


def _remove_server(label: str, namespace: str) -> None:
    """Remove a server by deleting its namespace."""
    log_step(f"Removing {label}")
    _require_kubectl_and_cluster()
    kubectl("delete", "namespace", namespace, "--ignore-not-found")
    log_success(f"{label} removed")


def _format_bytes_iec(size_bytes: int) -> str:
    """Format byte count in IEC units (KiB, MiB, ...). Portable (no numfmt)."""
    if size_bytes < 0:
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}" if unit != "B" else f"{size_bytes}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PiB"


def _get_gpu_allocatable() -> str:
    """Return the nvidia.com/gpu allocatable count from the cluster, or empty string."""
    r = kubectl(
        "get",
        "nodes",
        "-o",
        r"jsonpath={.items[0].status.allocatable.nvidia\.com/gpu}",
        capture=True,
        check=False,
    )
    return r.stdout.strip() if r.stdout else ""


# ---------------------------------------------------------------------------
# Doctor: platform detection + install recipes
# ---------------------------------------------------------------------------


def _detect_platform() -> str:
    """Return a platform key: 'mac', 'arch', 'debian', 'fedora', or 'linux'.

    Override with PLATFORM=arch|debian|fedora|linux|mac env var.
    """
    override = os.environ.get("PLATFORM", "").lower()
    if override in ("mac", "arch", "debian", "fedora", "linux"):
        return override
    if sys.platform == "darwin":
        return "mac"
    if sys.platform.startswith("linux"):
        try:
            os_release = Path("/etc/os-release").read_text().lower()
        except OSError:
            return "linux"
        if "arch" in os_release:
            return "arch"
        if any(d in os_release for d in ("debian", "ubuntu", "mint", "pop")):
            return "debian"
        if any(d in os_release for d in ("fedora", "rhel", "centos", "rocky", "alma")):
            return "fedora"
    return "linux"


def _detect_linux_arch() -> str:
    """Return Linux binary arch: 'amd64' or 'arm64'. Override with ARCH=amd64|arm64."""
    override = os.environ.get("ARCH", "").lower()
    if override in ("amd64", "arm64"):
        return override
    r = subprocess.run(["uname", "-m"], capture_output=True, text=True, check=False)
    machine = (r.stdout or "").strip().lower()
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine in ("x86_64", "amd64"):
        return "amd64"
    return "amd64"


def _resolve_recipe_cmds(cmds: list[str], platform: str) -> list[str]:
    """Substitute {arch} and {install_prefix} in Linux recipe commands."""
    if platform != "linux":
        return cmds
    arch = _detect_linux_arch()
    install_prefix = os.environ.get("INSTALL_PREFIX", "/usr/local")
    return [
        c.replace("{arch}", arch).replace("{install_prefix}", install_prefix)
        for c in cmds
    ]


_MINIKUBE_LINUX_RECIPE: dict[str, list[str]] = {
    "cmds": [
        "curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-{arch}",
        "sudo install minikube-linux-{arch} {install_prefix}/bin/minikube",
        "rm -f minikube-linux-{arch}",
    ],
    "post": [],
}
_KUBECTL_LINUX_RECIPE: dict[str, list[str]] = {
    "cmds": [
        'curl -LO "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/{arch}/kubectl"',
        "sudo install -o root -g root -m 0755 kubectl {install_prefix}/bin/kubectl",
        "rm -f kubectl",
    ],
    "post": [],
}
_HELM_LINUX_RECIPE: dict[str, list[str]] = {
    "cmds": [
        "curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash"
    ],
    "post": [],
}
_K9S_LINUX_RECIPE: dict[str, list[str]] = {
    "cmds": [
        "curl -sL https://github.com/derailed/k9s/releases/latest/download/k9s_Linux_{arch}.tar.gz -o k9s.tar.gz",
        "tar xzf k9s.tar.gz",
        "sudo install k9s {install_prefix}/bin/k9s",
        "rm -f k9s k9s.tar.gz",
    ],
    "post": [],
}
_INSTALL_RECIPES: dict[str, dict[str, dict[str, list[str]]]] = {
    "docker": {
        "mac": {
            "cmds": ["brew install --cask docker"],
            "post": ["Open Docker Desktop to finish setup."],
        },
        "arch": {
            "cmds": ["sudo pacman -S --noconfirm docker"],
            "post": ["sudo systemctl enable --now docker"],
        },
        "debian": {
            "cmds": ["curl -fsSL https://get.docker.com | sh"],
            "post": ["sudo systemctl enable --now docker"],
        },
        "fedora": {
            "cmds": ["curl -fsSL https://get.docker.com | sh"],
            "post": ["sudo systemctl enable --now docker"],
        },
        "linux": {"cmds": ["curl -fsSL https://get.docker.com | sh"], "post": []},
    },
    "minikube": {
        "mac": {"cmds": ["brew install minikube"], "post": []},
        "arch": {"cmds": ["sudo pacman -S --noconfirm minikube"], "post": []},
        "debian": _MINIKUBE_LINUX_RECIPE,
        "fedora": _MINIKUBE_LINUX_RECIPE,
        "linux": _MINIKUBE_LINUX_RECIPE,
    },
    "kubectl": {
        "mac": {"cmds": ["brew install kubectl"], "post": []},
        "arch": {"cmds": ["sudo pacman -S --noconfirm kubectl"], "post": []},
        "debian": _KUBECTL_LINUX_RECIPE,
        "fedora": _KUBECTL_LINUX_RECIPE,
        "linux": _KUBECTL_LINUX_RECIPE,
    },
    "helm": {
        "mac": {"cmds": ["brew install helm"], "post": []},
        "arch": {"cmds": ["sudo pacman -S --noconfirm helm"], "post": []},
        "debian": _HELM_LINUX_RECIPE,
        "fedora": _HELM_LINUX_RECIPE,
        "linux": _HELM_LINUX_RECIPE,
    },
    "k9s": {
        "mac": {"cmds": ["brew install k9s"], "post": []},
        "arch": {"cmds": ["sudo pacman -S --noconfirm k9s"], "post": []},
        "debian": _K9S_LINUX_RECIPE,
        "fedora": {"cmds": ["sudo dnf install -y k9s"], "post": []},
        "linux": _K9S_LINUX_RECIPE,
    },
}

_PLATFORM_LABELS = {
    "mac": "macOS",
    "arch": "Arch Linux",
    "debian": "Debian/Ubuntu",
    "fedora": "Fedora/RHEL",
    "linux": "Linux",
}

_VERSION_CMDS: dict[str, tuple[list[str], str]] = {
    "docker": (["docker", "version", "--format", "{{.Client.Version}}"], "v{}"),
    "kubectl": (["kubectl", "version", "--client"], "{after_colon}"),
    "minikube": (["minikube", "version", "--short"], "{}"),
    "helm": (["helm", "version", "--short"], "{}"),
    "k9s": (["k9s", "version"], "{}"),
}


def _tool_version(tool: str) -> str:
    """Get a short version string for a tool, or empty string."""
    cmd, fmt = _VERSION_CMDS.get(tool, ([tool, "version"], "{}"))
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    stdout = (r.stdout or "").strip()
    if not stdout:
        return ""
    if tool == "k9s":
        for line in stdout.splitlines():
            if "Version:" in line:
                return line.split("Version:", 1)[-1].strip()
        return ""
    line = stdout.splitlines()[0]
    if "{after_colon}" in fmt:
        line = line.split(": ", 1)[-1] if ": " in line else line
        return line
    return fmt.format(line)


_REQUIRED_TOOLS = ["docker", "minikube", "kubectl", "helm"]
_OPTIONAL_TOOLS = ["k9s"]


def _tools_status_rows(tools: list[str]) -> list[tuple[str, bool, str]]:
    """Return (tool, found, version) for each tool."""
    rows: list[tuple[str, bool, str]] = []
    for tool in tools:
        found = shutil.which(tool) is not None
        version = _tool_version(tool) if found else ""
        rows.append((tool, found, version or ("—" if not found else "")))
    return rows


def _doctor_tools_panel(
    rows: list[tuple[str, bool, str]],
    *,
    title: str,
    border_style: str,
    header_style: str,
    found_mark: str = "[green]✓[/]",
    missing_mark: str = "[red]✗[/]",
    detail_column: str = "Version",
) -> Panel:
    """Build a Rich Panel with a tool status table."""
    table = Table(show_header=True, header_style=header_style, box=None)
    table.add_column("Tool", style="bold")
    table.add_column("Status", width=6)
    table.add_column(detail_column)
    for tool, found, detail in rows:
        status = found_mark if found else missing_mark
        table.add_row(tool, status, detail or "—")
    return Panel(table, title=title, border_style=border_style)


def _check_tools(
    tools: list[str],
    *,
    title: str,
    border_style: str,
    header_style: str,
    found_mark: str = "[green]✓[/]",
    missing_mark: str = "[red]✗[/]",
    optional: bool = False,
    verbose: bool = True,
    console: Any = None,
) -> list[str]:
    """Check tools and optionally render status. Returns list of missing tool names."""
    rows = _tools_status_rows(tools)
    missing = [t for t, found, _ in rows if not found]
    if verbose and console is not None:
        console.print(
            _doctor_tools_panel(
                rows,
                title=title,
                border_style=border_style,
                header_style=header_style,
                found_mark=found_mark,
                missing_mark=missing_mark,
            )
        )
    elif verbose:
        for tool, found, version in rows:
            if found:
                print(_ok(f"{tool:10s} {version or ''}"))
            else:
                print(
                    _miss(f"{tool:10s} not found (optional cluster UI)")
                    if optional
                    else _fail(f"{tool:10s} not found")
                )
    return missing


def _check_prerequisites(*, verbose: bool = True, console: Any = None) -> list[str]:
    """Check for required tools. Returns list of missing tool names."""
    return _check_tools(
        _REQUIRED_TOOLS,
        title="[bold cyan]Required[/]",
        border_style="cyan",
        header_style="bold cyan",
        verbose=verbose,
        console=console,
    )


def _check_optional_tools(*, verbose: bool = True, console: Any = None) -> list[str]:
    """Check optional tools (e.g. k9s). Returns list of missing."""
    return _check_tools(
        _OPTIONAL_TOOLS,
        title="[bold yellow]Optional (cluster UI)[/]",
        border_style="yellow",
        header_style="bold yellow",
        found_mark="[green]✓[/]",
        missing_mark="[yellow]○[/] optional",
        optional=True,
        verbose=verbose,
        console=console,
    )


def _preflight() -> None:
    """Quick prerequisite check for cmd_setup. Exits with hint on failure."""
    missing = _check_prerequisites(verbose=False)
    if missing:
        log_error(f"Missing prerequisites: {', '.join(missing)}")
        log_info("Run './dev/kube.py doctor' to install them.")
        raise SystemExit(1)


def _doctor_gpu_section(*, console: Any = None) -> None:
    """Show GPU prerequisites in doctor output."""
    nvidia_smi_ok = shutil.which("nvidia-smi") is not None
    nvidia_ctk_ok = shutil.which("nvidia-ctk") is not None
    nvidia_smi_detail = ""
    nvidia_ctk_detail = ""
    if nvidia_smi_ok:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        nvidia_smi_detail = (
            r.stdout.strip().splitlines()[0]
            if r.returncode == 0 and r.stdout.strip()
            else "installed"
        )
    if nvidia_ctk_ok:
        r = subprocess.run(
            ["nvidia-ctk", "--version"], capture_output=True, text=True, check=False
        )
        nvidia_ctk_detail = (
            r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "installed"
        )

    if console is not None:
        gpu_rows: list[tuple[str, bool, str]] = [
            (
                "nvidia-smi",
                nvidia_smi_ok,
                nvidia_smi_detail
                or "not found (install NVIDIA drivers for GPU support)",
            ),
            (
                "nvidia-ctk",
                nvidia_ctk_ok,
                nvidia_ctk_detail
                or "not found (install nvidia-container-toolkit for GPU support)",
            ),
        ]
        console.print(
            _doctor_tools_panel(
                gpu_rows,
                title="[bold blue]GPU[/]",
                border_style="blue",
                header_style="bold blue",
                missing_mark="[yellow]○[/]",
                detail_column="Details",
            )
        )
    else:
        print(f"\n{_CYAN}GPU Prerequisites:{_NC}")
        if nvidia_smi_ok:
            print(_ok(f"nvidia-smi  {nvidia_smi_detail}"))
        else:
            print(
                _miss("nvidia-smi  not found (install NVIDIA drivers for GPU support)")
            )
        if nvidia_ctk_ok:
            print(_ok(f"nvidia-ctk  {nvidia_ctk_detail}"))
        else:
            print(
                _miss(
                    "nvidia-ctk  not found (install nvidia-container-toolkit for GPU support)"
                )
            )


def _doctor_install_prompt(
    tool: str,
    cmds: list[str],
    default_yes: bool,
    recipe: dict,
    console: Any,
) -> bool:
    """Show install prompt and run install if confirmed. Return True if installed ok."""
    block = "\n".join(cmds)
    console.print(
        Panel(
            Syntax(block, "bash", theme="monokai", line_numbers=False),
            title=f"[bold green]Install {tool}?[/]",
            border_style="green",
        )
    )
    if not _confirm("\nRun the above?", default_yes=default_yes, console=console):
        return False
    if _run_install(cmds, console=console):
        log_success(f"{tool} installed")
        for hint in recipe.get("post", []):
            log_info(f"  Note: {hint}")
        return True
    log_error(f"Failed to install {tool}")
    return False


def _doctor_offer_installs(
    tools: list[str],
    platform: str,
    default_yes: bool,
    console: Console,
    *,
    warn_no_recipe: bool = False,
) -> tuple[list[str], list[str]]:
    """Offer to install each tool. Returns (installed, skipped)."""
    installed: list[str] = []
    skipped: list[str] = []
    for tool in tools:
        recipe = _INSTALL_RECIPES.get(tool, {}).get(platform, {})
        cmds = _resolve_recipe_cmds(recipe.get("cmds", []), platform)
        if not cmds:
            if warn_no_recipe:
                log_warn(f"No auto-install recipe for {tool} on this platform.")
                skipped.append(tool)
            continue
        if _doctor_install_prompt(tool, cmds, default_yes, recipe, console):
            installed.append(tool)
        else:
            if warn_no_recipe:
                log_info(f"Skipping {tool}")
            skipped.append(tool)
        if warn_no_recipe:
            print()
    return (installed, skipped)


def cmd_doctor() -> None:
    """Check prerequisites interactively. Offer to install missing tools."""
    con = Console()
    con.print(Rule("[bold cyan]Doctor — Prerequisites[/]", style="cyan"))
    platform = _detect_platform()
    con.print(
        Panel(
            _PLATFORM_LABELS.get(platform, platform),
            title="[bold]Platform[/]",
            border_style="dim",
        )
    )

    missing = _check_prerequisites(verbose=True, console=con)
    missing_optional = _check_optional_tools(verbose=True, console=con)

    if not missing:
        r = sh("docker", "info", capture=True, check=False)
        if r.returncode != 0:
            print()
            log_warn("Docker is installed but not running. Start it with:")
            if platform == "mac":
                print("  open -a Docker")
            elif shutil.which("systemctl"):
                print("  sudo systemctl start docker")
            else:
                print("  Start Docker (see your distro docs, e.g. systemd or service)")
        else:
            print()
            log_success(
                "All prerequisites met! Run './dev/kube.py setup' to get started."
            )
        _doctor_gpu_section(console=con)
        _doctor_offer_installs(missing_optional, platform, True, con)
        return

    # Interactively install missing tools
    print()
    installed, skipped = _doctor_offer_installs(
        missing, platform, False, con, warn_no_recipe=True
    )
    if installed:
        log_success(f"Installed: {', '.join(installed)}")
    if skipped:
        log_warn(f"Still missing: {', '.join(skipped)}")
        print("\nRun './dev/kube.py doctor' again after installing manually.")
    elif not skipped:
        log_success("All prerequisites met! Run './dev/kube.py setup' to get started.")

    _doctor_offer_installs(missing_optional, platform, True, con)


# ---------------------------------------------------------------------------
# Cluster management
# ---------------------------------------------------------------------------


def cmd_cluster_create() -> None:
    log_step(f"Creating minikube cluster: {CLUSTER_NAME}")
    require("minikube", "kubectl", "docker")
    _require_docker_running()

    if cluster_running():
        log_success(f"Cluster {CLUSTER_NAME} already running")
        return

    if cluster_exists():
        log_info(f"Cluster {CLUSTER_NAME} exists but stopped, starting...")
        minikube("start")
        log_success(f"Cluster {CLUSTER_NAME} started")
        return

    # Detect GPU availability for --gpus flag
    has_gpu = shutil.which("nvidia-smi") is not None
    gpu_args = ["--gpus", "all"] if has_gpu else []
    if has_gpu:
        log_info("GPU detected — enabling GPU passthrough")
    else:
        log_info("No GPU detected — creating CPU-only cluster")

    minikube(
        "start",
        "--driver",
        "docker",
        "--container-runtime",
        "docker",
        *gpu_args,
        "--memory",
        MINIKUBE_MEMORY,
        "--cpus",
        MINIKUBE_CPUS,
        "--profile",
        CLUSTER_NAME,
    )
    kubectl("cluster-info", capture=True)

    # Create nvidia RuntimeClass (minikube --gpus enables the device plugin but
    # doesn't create the RuntimeClass that pods with runtimeClassName: nvidia need)
    if has_gpu:
        _runtimeclass = """\
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
"""
        kubectl("apply", "-f", "-", input=_runtimeclass, text=True)
        log_info("Created nvidia RuntimeClass")

    log_success(f"Cluster {CLUSTER_NAME} created (context: {CLUSTER_NAME})")


def cmd_cluster_delete() -> None:
    log_step(f"Deleting minikube cluster: {CLUSTER_NAME}")
    require("minikube")

    if not cluster_exists():
        log_warn(f"Cluster {CLUSTER_NAME} does not exist")
        return

    minikube("delete")
    log_success(f"Cluster {CLUSTER_NAME} deleted")


# ---------------------------------------------------------------------------
# Build & load
# ---------------------------------------------------------------------------


def cmd_build(*, aiperf: bool = True, mock: bool = True) -> None:
    require("docker")
    if aiperf:
        _docker_build(AIPERF_IMAGE, "Dockerfile", "--target", "runtime")
    if mock:
        _docker_build(MOCK_SERVER_IMAGE, "dev/deploy/Dockerfile.mock-server")


def cmd_load(*, aiperf: bool = True, mock: bool = True) -> None:
    require("minikube")
    require_cluster()

    def _load(image: str) -> None:
        log_step(f"Loading {image} into minikube")
        r = sh("docker", "image", "inspect", image, capture=True, check=False)
        if r.returncode != 0:
            log_error(f"Image {image} not found. Run 'build' first.")
            raise SystemExit(1)
        minikube("image", "load", image)
        log_success(f"Loaded {image}")

    if aiperf:
        _load(AIPERF_IMAGE)
    if mock:
        _load(MOCK_SERVER_IMAGE)


# ---------------------------------------------------------------------------
# JobSet controller
# ---------------------------------------------------------------------------


def cmd_install_jobset() -> None:
    log_step(f"Installing JobSet controller {JOBSET_VERSION}")
    _require_kubectl_and_cluster()

    if _skip_if_deployment_ready(
        "jobset-controller-manager",
        "jobset-system",
        "JobSet controller already installed",
    ):
        return

    log_info("Applying JobSet manifests...")
    kubectl(
        "apply",
        "--server-side",
        "-f",
        f"https://github.com/kubernetes-sigs/jobset/releases/download/{JOBSET_VERSION}/manifests.yaml",
    )
    log_info("Waiting for JobSet controller to be ready...")
    kubectl(
        "wait",
        "--for=condition=available",
        "--timeout=120s",
        "deployment/jobset-controller-manager",
        "-n",
        "jobset-system",
    )
    log_success(f"JobSet controller {JOBSET_VERSION} installed and ready")


# ---------------------------------------------------------------------------
# Dynamo operator (Helm from NGC)
# ---------------------------------------------------------------------------


def cmd_install_dynamo() -> None:
    """Install Dynamo operator via Helm from NGC."""
    log_step(f"Installing Dynamo operator {DYNAMO_VERSION}")
    require("helm")
    _require_kubectl_and_cluster()

    # Check if already installed
    r = kubectl(
        "get", "crd", "dynamographdeployments.nvidia.com", capture=True, check=False
    )
    if r.returncode == 0:
        log_success("Dynamo operator already installed")
        return

    ngc_base = "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts"
    crds_tgz = f"dynamo-crds-{DYNAMO_VERSION}.tgz"
    platform_tgz = f"dynamo-platform-{DYNAMO_VERSION}.tgz"

    # Fetch and install CRDs
    log_info("Fetching Dynamo CRDs chart...")
    sh("helm", "fetch", f"{ngc_base}/{crds_tgz}")
    sh(
        "helm",
        "install",
        "dynamo-crds",
        crds_tgz,
        "--kube-context",
        CLUSTER_NAME,
        "--namespace",
        "default",
    )
    os.unlink(crds_tgz)

    # Fetch and install platform
    log_info("Fetching Dynamo platform chart...")
    sh("helm", "fetch", f"{ngc_base}/{platform_tgz}")
    sh(
        "helm",
        "install",
        "dynamo-platform",
        platform_tgz,
        "--kube-context",
        CLUSTER_NAME,
        "--namespace",
        "dynamo-system",
        "--create-namespace",
        "--set",
        "dynamo-operator.webhook.enabled=false",
        "--set",
        "grove.enabled=false",
        "--set",
        "kai-scheduler.enabled=false",
    )
    os.unlink(platform_tgz)

    # Wait for operator deployment
    log_info("Waiting for Dynamo operator to be ready...")
    for i in range(60):
        r = kubectl(
            "get",
            "pods",
            "-n",
            "dynamo-system",
            "-l",
            "app.kubernetes.io/name=dynamo-operator",
            "--no-headers",
            capture=True,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip() and "Running" in r.stdout:
            break
        if (i + 1) % 10 == 0:
            log_info(f"Still waiting... ({i + 1}/60)")
        time.sleep(5)
    else:
        log_warn(
            "Dynamo operator may not be fully ready — check 'kubectl get pods -n dynamo-system'"
        )

    log_success(f"Dynamo operator {DYNAMO_VERSION} installed")


# ---------------------------------------------------------------------------
# Mock server
# ---------------------------------------------------------------------------


def cmd_deploy_mock() -> None:
    log_step("Deploying mock server")
    _require_kubectl_and_cluster()

    if _skip_if_deployment_ready(
        "aiperf-mock-server", None, "Mock server already deployed and ready"
    ):
        return

    if not MOCK_SERVER_MANIFEST.exists():
        log_error(f"Mock server manifest not found: {MOCK_SERVER_MANIFEST}")
        raise SystemExit(1)

    log_info("Applying mock server manifest...")
    manifest = MOCK_SERVER_MANIFEST.read_text().replace(
        "image: aiperf-mock-server:latest",
        f"image: {MOCK_SERVER_IMAGE}",
    )
    kubectl("apply", "-f", "-", input=manifest, text=True)
    log_info("Waiting for mock server to be ready...")
    kubectl("rollout", "status", "deployment/aiperf-mock-server", "--timeout=120s")
    log_success("Mock server deployed")
    kubectl("get", "pods", "-l", "app=aiperf-mock-server")
    log_info("Endpoint: http://aiperf-mock-server.default.svc.cluster.local:8000")


def cmd_remove_mock() -> None:
    log_step("Removing mock server")
    _require_kubectl_and_cluster()
    kubectl("delete", "-f", str(MOCK_SERVER_MANIFEST), "--ignore-not-found")
    log_success("Mock server removed")


# ---------------------------------------------------------------------------
# vLLM deployment
# ---------------------------------------------------------------------------


def _generate_vllm_manifest(
    model: str,
    gpus: int,
    vllm_image: str,
    max_model_len: int,
    *,
    hf_token: bool,
) -> str:
    """Generate Namespace + Service + Deployment YAML for vLLM."""
    env_block = ""
    if hf_token:
        env_block = """
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: token"""

    gpu_resources = (
        f"""
          resources:
            limits:
              nvidia.com/gpu: "{gpus}"
            requests:
              nvidia.com/gpu: "{gpus}" """
        if gpus > 0
        else ""
    )

    return f"""\
apiVersion: v1
kind: Namespace
metadata:
  name: {VLLM_NAMESPACE}
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-server
  namespace: {VLLM_NAMESPACE}
spec:
  selector:
    app: vllm-server
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: {VLLM_NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      runtimeClassName: nvidia
      containers:
        - name: vllm
          image: {vllm_image}
          args:
            - "--model"
            - "{model}"
            - "--port"
            - "8000"
            - "--max-model-len"
            - "{max_model_len}"
            - "--dtype"
            - "auto"
            - "--tensor-parallel-size"
            - "{gpus}"
            - "--gpu-memory-utilization"
            - "{VLLM_GPU_MEM_UTIL}"
            - "--enforce-eager"
          ports:
            - containerPort: 8000
              name: http{gpu_resources}{env_block}
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 30
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 15
            timeoutSeconds: 5
            failureThreshold: 5
"""


@app.command(name="deploy-vllm", group=server, help="Deploy standalone vLLM server.")
def cmd_deploy_vllm(
    *,
    model_opts: ModelOptions = _DEFAULT_MODEL_OPTS,
    vllm_image: Annotated[str, Parameter(help="vLLM container image.")] = VLLM_IMAGE,
) -> None:
    """Deploy vLLM inference server."""
    log_step(
        f"Deploying vLLM server (model={model_opts.model}, gpus={model_opts.gpus})"
    )
    _require_kubectl_and_cluster()

    if _skip_if_deployment_ready(
        "vllm-server", VLLM_NAMESPACE, "vLLM server already deployed and ready"
    ):
        return

    hf_secret = _ensure_hf_token_secret(VLLM_NAMESPACE)
    manifest = _generate_vllm_manifest(
        model_opts.model,
        model_opts.gpus,
        vllm_image,
        model_opts.max_model_len,
        hf_token=bool(hf_secret),
    )
    kubectl("apply", "-f", "-", input=manifest, text=True)

    log_info(
        "Waiting for vLLM to be ready (this may take several minutes for model download)..."
    )
    kubectl(
        "rollout",
        "status",
        "deployment/vllm-server",
        "-n",
        VLLM_NAMESPACE,
        "--timeout=600s",
    )
    log_success("vLLM server deployed")
    kubectl("get", "pods", "-n", VLLM_NAMESPACE, "-l", "app=vllm-server")
    log_info(f"Endpoint: http://vllm-server.{VLLM_NAMESPACE}.svc.cluster.local:8000/v1")


def cmd_remove_vllm() -> None:
    _remove_server("vLLM server", VLLM_NAMESPACE)


@app.command(name="vllm-logs", group=server, help="View vLLM logs.")
def cmd_vllm_logs(*, follow: FollowFlag = FOLLOW_DEFAULT) -> None:
    """View vLLM server logs."""
    _require_kubectl_and_cluster()
    kubectl(
        "logs",
        "-n",
        VLLM_NAMESPACE,
        "-l",
        "app=vllm-server",
        *(["-f"] if follow else []),
    )


# ---------------------------------------------------------------------------
# Dynamo deployment (DynamoGraphDeployment CRD via Dynamo operator)
# ---------------------------------------------------------------------------

_DYNAMO_HEALTH_PORT = 9090
_DYNAMO_METRICS_PORT = 8081  # DYN_SYSTEM_PORT: worker /metrics endpoint
_DYNAMO_1GPU_KVBM_CPU_CACHE_GB = (
    1  # conservative default for single-GPU (pinned memory)
)
_DYNAMO_WORKER_WORKING_DIR = "/workspace/examples/backends/vllm"
_DYNAMO_WORKER_COMMAND = ["python3", "-m", "dynamo.vllm"]


def _check_dynamo_operator() -> None:
    """Verify the Dynamo operator CRD is installed. Exits with hint on failure."""
    r = kubectl(
        "get", "crd", "dynamographdeployments.nvidia.com", capture=True, check=False
    )
    if r.returncode != 0:
        log_error(
            "Dynamo operator not found (CRD dynamographdeployments.nvidia.com missing)"
        )
        log_info("Run './dev/kube.py install-dynamo' to install the Dynamo operator.")
        raise SystemExit(1)


def _generate_dynamo_manifest(
    model: str,
    gpus: int,
    dynamo_image: str,
    max_model_len: int,
    *,
    mode: str,
    hf_token_secret: str | None,
    router_mode: str | None = None,
    kvbm_cpu_cache_gb: int | None = None,
    connectors: list[str] | None = None,
) -> str:
    """Generate Namespace + DynamoGraphDeployment CRD YAML.

    Modes: agg, disagg, disagg-1gpu.
    disagg-1gpu shares 1 physical GPU between prefill+decode workers
    by requesting 0 GPUs in K8s resources (runtimeClassName: nvidia gives access)
    and using low gpu_memory_utilization so both fit.

    Optional KV features:
      router_mode: sets DYN_ROUTER_MODE on frontend (e.g. "kv", "round-robin")
      kvbm_cpu_cache_gb: sets DYN_KVBM_CPU_CACHE_GB on prefill workers
      connectors: adds --connector flags to prefill workers (e.g. ["kvbm", "nixl"])
    """
    single_gpu = mode == "disagg-1gpu"
    deploy_name = f"dynamo-{mode}"

    # Single-GPU disagg: override to low mem util, force gpus=0 for K8s scheduling
    gpu_mem_util = DYNAMO_1GPU_MEM_UTIL if single_gpu else VLLM_GPU_MEM_UTIL
    gpu_request = 0 if single_gpu else gpus

    # Auto-default KVBM CPU cache for single-GPU when kvbm connector is used
    has_kvbm = connectors and "kvbm" in connectors
    if has_kvbm and kvbm_cpu_cache_gb is None and single_gpu:
        kvbm_cpu_cache_gb = _DYNAMO_1GPU_KVBM_CPU_CACHE_GB
        log_info(
            f"KVBM: defaulting CPU cache to {kvbm_cpu_cache_gb}GB "
            f"(pinned memory, conservative for single-GPU)"
        )

    worker_args = ["--model", model]
    if max_model_len:
        worker_args.extend(["--max-model-len", str(max_model_len)])
    worker_args.extend(["--gpu-memory-utilization", gpu_mem_util, "--enforce-eager"])

    probes = {
        "startupProbe": {
            "httpGet": {"path": "/live", "port": _DYNAMO_HEALTH_PORT},
            "initialDelaySeconds": 60,
            "periodSeconds": 10,
            "failureThreshold": 60,
            "timeoutSeconds": 5,
        },
        "livenessProbe": {
            "httpGet": {"path": "/live", "port": _DYNAMO_HEALTH_PORT},
            "initialDelaySeconds": 0,
            "periodSeconds": 10,
            "failureThreshold": 15,
            "timeoutSeconds": 5,
        },
    }

    worker_container: dict = {
        "image": dynamo_image,
        "workingDir": _DYNAMO_WORKER_WORKING_DIR,
        "command": _DYNAMO_WORKER_COMMAND,
        "args": worker_args,
        **probes,
    }

    pod_uid_env = {
        "name": "POD_UID",
        "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
    }
    metrics_port_env = {
        "name": "DYN_SYSTEM_PORT",
        "value": str(_DYNAMO_METRICS_PORT),
    }

    frontend_envs: list[dict] = [pod_uid_env]
    if router_mode:
        frontend_envs.append({"name": "DYN_ROUTER_MODE", "value": router_mode})

    frontend: dict = {
        "dynamoNamespace": deploy_name,
        "componentType": "frontend",
        "replicas": 1,
        "extraPodSpec": {
            "mainContainer": {"image": dynamo_image, "env": frontend_envs}
        },
    }

    decode_envs: list[dict] = [pod_uid_env, metrics_port_env]

    decode_worker: dict = {
        "dynamoNamespace": deploy_name,
        "componentType": "worker",
        "replicas": 1,
        "resources": {"limits": {"gpu": str(gpu_request)}},
        "extraPodSpec": {
            "runtimeClassName": "nvidia",
            "mainContainer": {**worker_container, "env": decode_envs},
        },
    }

    if hf_token_secret:
        decode_worker["envFromSecret"] = hf_token_secret

    services: dict = {"Frontend": frontend, "VllmDecodeWorker": decode_worker}

    if mode in ("disagg", "disagg-1gpu"):
        decode_worker["subComponentType"] = "decode"
        decode_worker["extraPodSpec"]["mainContainer"]["args"] = [
            *worker_args,
            "--is-decode-worker",
        ]

        prefill_args = [*worker_args, "--is-prefill-worker"]
        for connector in connectors or []:
            prefill_args.extend(["--connector", connector])

        prefill_envs: list[dict] = [pod_uid_env, metrics_port_env]
        if kvbm_cpu_cache_gb is not None:
            prefill_envs.append(
                {"name": "DYN_KVBM_CPU_CACHE_GB", "value": str(kvbm_cpu_cache_gb)}
            )
            prefill_envs.append({"name": "DYN_KVBM_METRICS", "value": "true"})

        prefill_container = {
            **worker_container,
            "args": prefill_args,
            "env": prefill_envs,
        }
        prefill_worker: dict = {
            "dynamoNamespace": deploy_name,
            "componentType": "worker",
            "subComponentType": "prefill",
            "replicas": 1,
            "resources": {"limits": {"gpu": str(gpu_request)}},
            "extraPodSpec": {
                "runtimeClassName": "nvidia",
                "mainContainer": prefill_container,
            },
        }
        if hf_token_secret:
            prefill_worker["envFromSecret"] = hf_token_secret
        services["VllmPrefillWorker"] = prefill_worker

    sa_name = f"{deploy_name}-k8s-service-discovery"

    documents = [
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": DYNAMO_NAMESPACE},
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {"name": "dynamo-discovery", "namespace": DYNAMO_NAMESPACE},
            "rules": [
                {
                    "apiGroups": ["nvidia.com"],
                    "resources": ["dynamoworkermetadatas"],
                    "verbs": [
                        "get",
                        "list",
                        "watch",
                        "create",
                        "update",
                        "patch",
                        "delete",
                    ],
                }
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {"name": "dynamo-discovery", "namespace": DYNAMO_NAMESPACE},
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": sa_name,
                    "namespace": DYNAMO_NAMESPACE,
                }
            ],
            "roleRef": {
                "kind": "Role",
                "name": "dynamo-discovery",
                "apiGroup": "rbac.authorization.k8s.io",
            },
        },
        {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": deploy_name, "namespace": DYNAMO_NAMESPACE},
            "spec": {"services": services},
        },
    ]

    return "\n---\n".join(yaml.dump(doc, default_flow_style=False) for doc in documents)


def _wait_for_dynamo_ready(timeout: int = 600) -> None:
    """Wait for all Dynamo pods to be Ready."""
    deploy_name = f"dynamo-{DYNAMO_MODE}"
    log_info(f"Waiting for Dynamo pods to be ready (timeout={timeout}s)...")
    deadline = time.time() + timeout

    while time.time() < deadline:
        r = kubectl(
            "get",
            "pods",
            "-n",
            DYNAMO_NAMESPACE,
            "-o",
            "json",
            capture=True,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            data = orjson.loads(r.stdout)
            pods = [
                p for p in data.get("items", []) if deploy_name in p["metadata"]["name"]
            ]
            if pods:
                ready_count = sum(
                    1
                    for p in pods
                    if p.get("status", {}).get("phase") == "Running"
                    and all(
                        cs.get("ready", False)
                        for cs in p.get("status", {}).get("containerStatuses", [])
                    )
                )
                if ready_count == len(pods):
                    log_success(f"All {len(pods)} Dynamo pod(s) ready")
                    return
                log_info(f"  {ready_count}/{len(pods)} pods ready...")
        time.sleep(10)

    log_error(f"Dynamo pods not ready within {timeout}s")
    kubectl("get", "pods", "-n", DYNAMO_NAMESPACE, "-o", "wide", check=False)
    raise SystemExit(1)


@app.command(
    name="deploy-dynamo",
    group=server,
    help="Deploy Dynamo inference server (agg/disagg/disagg-1gpu).",
)
def cmd_deploy_dynamo(
    *,
    model_opts: ModelOptions = _DEFAULT_MODEL_OPTS,
    dynamo_opts: DynamoDeployOptions = _DEFAULT_DYNAMO_OPTS,
) -> None:
    """Deploy Dynamo inference server."""
    extras: list[str] = []
    if dynamo_opts.router_mode:
        extras.append(f"router={dynamo_opts.router_mode}")
    if dynamo_opts.kvbm_cpu_cache_gb is not None:
        extras.append(f"kvbm={dynamo_opts.kvbm_cpu_cache_gb}GB")
    if dynamo_opts.connectors:
        extras.append(f"connectors={','.join(dynamo_opts.connectors)}")
    extras_str = f", {', '.join(extras)}" if extras else ""

    if dynamo_opts.mode == "disagg-1gpu":
        log_step(
            f"Deploying Dynamo server ({dynamo_opts.mode}, model={model_opts.model}, "
            f"shared 1 GPU, mem_util={DYNAMO_1GPU_MEM_UTIL}{extras_str})"
        )
    else:
        log_step(
            f"Deploying Dynamo server ({dynamo_opts.mode}, model={model_opts.model}, "
            f"gpus={model_opts.gpus}{extras_str})"
        )
    _require_kubectl_and_cluster()
    _check_dynamo_operator()

    hf_secret = _ensure_hf_token_secret(DYNAMO_NAMESPACE)
    manifest = _generate_dynamo_manifest(
        model_opts.model,
        model_opts.gpus,
        dynamo_opts.dynamo_image,
        model_opts.max_model_len,
        mode=dynamo_opts.mode,
        hf_token_secret=hf_secret,
        router_mode=dynamo_opts.router_mode,
        kvbm_cpu_cache_gb=dynamo_opts.kvbm_cpu_cache_gb,
        connectors=dynamo_opts.connectors,
    )
    kubectl("apply", "-f", "-", input=manifest, text=True)
    _wait_for_dynamo_ready(timeout=600)

    deploy_name = f"dynamo-{dynamo_opts.mode}"
    endpoint = (
        f"http://{deploy_name}-frontend.{DYNAMO_NAMESPACE}.svc.cluster.local:8000/v1"
    )
    log_success("Dynamo server deployed")
    kubectl("get", "pods", "-n", DYNAMO_NAMESPACE, "-o", "wide")
    log_info(f"Endpoint: {endpoint}")


def cmd_remove_dynamo() -> None:
    _remove_server("Dynamo server", DYNAMO_NAMESPACE)


@app.command(name="dynamo-logs", group=server, help="View Dynamo pod logs.")
def cmd_dynamo_logs(*, follow: FollowFlag = FOLLOW_DEFAULT) -> None:
    """View Dynamo server logs."""
    _require_kubectl_and_cluster()

    r = kubectl(
        "get", "pods", "-n", DYNAMO_NAMESPACE, "-o", "name", capture=True, check=False
    )
    pods = [p.strip() for p in (r.stdout or "").splitlines() if p.strip()]
    if not pods:
        log_warn("No pods found in dynamo-server namespace")
        return

    if follow:
        worker = next((p for p in pods if "worker" in p.lower()), pods[-1])
        kubectl("logs", "-n", DYNAMO_NAMESPACE, worker, "--all-containers", "-f")
    else:
        for pod in pods:
            print(f"\n{_CYAN}--- {pod} ---{_NC}")
            kubectl(
                "logs",
                "-n",
                DYNAMO_NAMESPACE,
                pod,
                "--all-containers",
                "--tail=50",
                check=False,
            )


# ---------------------------------------------------------------------------
# LoRA adapter deployment (DynamoModel CRD)
# ---------------------------------------------------------------------------


def _generate_lora_manifest(name: str, base_model: str, source: str) -> str:
    """Generate DynamoModel CRD YAML for a LoRA adapter."""
    doc = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoModel",
        "metadata": {"name": name, "namespace": DYNAMO_NAMESPACE},
        "spec": {
            "modelName": name,
            "baseModelName": base_model,
            "modelType": "lora",
            "source": {"uri": source},
        },
    }
    return yaml.dump(doc, default_flow_style=False)


@app.command(
    name="deploy-lora",
    group=server,
    help="Deploy LoRA adapter on running Dynamo base model.",
)
def cmd_deploy_lora(
    *,
    name: Annotated[str, Parameter(help="LoRA adapter name.")],
    base_model: Annotated[str, Parameter(help="Base model name.")],
    source: Annotated[str, Parameter(help="LoRA source URI (e.g. hf://org/repo).")],
) -> None:
    """Deploy a LoRA adapter on a running Dynamo base model."""
    log_step(f"Deploying LoRA adapter: {name}")
    _require_kubectl_and_cluster()
    _check_dynamo_operator()

    manifest = _generate_lora_manifest(name, base_model, source)
    kubectl("apply", "-f", "-", input=manifest, text=True)
    log_success(
        f"LoRA adapter '{name}' deployed (base: {base_model}, source: {source})"
    )


@app.command(name="remove-lora", group=server, help="Remove LoRA adapter.")
def cmd_remove_lora(
    *, name: Annotated[str, Parameter(help="LoRA adapter name.")]
) -> None:
    """Remove a LoRA adapter."""
    log_step(f"Removing LoRA adapter: {name}")
    _require_kubectl_and_cluster()
    kubectl("delete", "dynamomodel", name, "-n", DYNAMO_NAMESPACE, "--ignore-not-found")
    log_success(f"LoRA adapter '{name}' removed")


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------


def cmd_run(*, opts: RunOptions, detach: bool, dry_run: bool) -> None:
    import asyncio

    from aiperf.common.config import load_service_config, load_user_config
    from aiperf.common.config.kube_config import KubeOptions
    from aiperf.kubernetes.runner import run_kubernetes_deployment

    config_path = Path(opts.config) if opts.config else DEFAULT_BENCHMARK_CONFIG
    if not config_path.exists():
        log_error(f"Config file not found: {config_path}")
        raise SystemExit(1)

    user_config = load_user_config(config_path)
    service_config = load_service_config()
    kube_options = KubeOptions(
        image=AIPERF_IMAGE, image_pull_policy="Never", workers=opts.workers
    )

    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        job_id, namespace = asyncio.run(
            run_kubernetes_deployment(
                user_config, service_config, kube_options, dry_run=True
            )
        )
    finally:
        sys.stdout = old_stdout

    manifest = buf.getvalue()

    if dry_run:
        print(manifest, end="")
        return

    _require_kubectl_and_cluster()
    kubectl("apply", "-f", "-", input=manifest, text=True)
    log_success(f"Deployed to namespace: {namespace} (job_id: {job_id})")

    if detach:
        log_info("Detached mode — benchmark running in background")
        log_info(f"Check status: uv run aiperf kube status {job_id}")
        log_info(f"View logs:    uv run aiperf kube logs {job_id}")
    else:
        if shutil.which("uv") is None:
            log_error("uv not found on PATH. Install uv or activate the project venv.")
            raise SystemExit(1)
        log_info("Attaching to benchmark...")
        os.execvp(
            "uv",
            ["uv", "run", "aiperf", "kube", "attach", job_id, "--namespace", namespace],
        )


# ---------------------------------------------------------------------------
# Single-pod benchmark (run-local)
# ---------------------------------------------------------------------------


def _user_config_to_cli_args(config: UserConfig) -> list[str]:
    """Convert key UserConfig fields to ``aiperf profile`` CLI arguments.

    Only a pre-configured subset of flags is supported — enough for the
    common single-pod benchmarking scenarios against mock / vLLM servers.
    """
    args: list[str] = []

    # -- Endpoint (required) --------------------------------------------------
    for model in config.endpoint.model_names:
        args.extend(["--model", model])
    for url in config.endpoint.urls:
        args.extend(["--url", url])
    args.extend(["--endpoint-type", str(config.endpoint.type)])
    if config.endpoint.streaming:
        args.append("--streaming")

    # -- Load generator -------------------------------------------------------
    if config.loadgen.concurrency is not None:
        args.extend(["--concurrency", str(config.loadgen.concurrency)])
    if config.loadgen.request_count is not None:
        args.extend(["--request-count", str(config.loadgen.request_count)])
    if config.loadgen.request_rate is not None:
        args.extend(["--request-rate", str(config.loadgen.request_rate)])
    if config.loadgen.warmup_request_count is not None:
        args.extend(
            ["--warmup-request-count", str(config.loadgen.warmup_request_count)]
        )
    if config.loadgen.benchmark_duration is not None:
        args.extend(["--benchmark-duration", str(config.loadgen.benchmark_duration)])

    # -- Tokenizer ------------------------------------------------------------
    if config.tokenizer.name:
        args.extend(["--tokenizer", config.tokenizer.name])

    # -- Telemetry (disabled — no DCGM / Prometheus in single-pod) ------------
    args.append("--no-gpu-telemetry")
    args.append("--no-server-metrics")

    return args


def _generate_single_pod_manifest(
    cli_args: list[str],
    namespace: str,
    image: str,
) -> str:
    """Generate Namespace + Job YAML for a single-pod ``aiperf profile`` run."""
    documents: list[dict] = [
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": namespace,
                "labels": {"app": "aiperf"},
            },
        },
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": "aiperf-benchmark",
                "namespace": namespace,
                "labels": {"app": "aiperf"},
            },
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 300,
                "template": {
                    "metadata": {"labels": {"app": "aiperf"}},
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "aiperf",
                                "image": image,
                                "imagePullPolicy": "Never",
                                "command": ["aiperf"],
                                "args": ["profile", *cli_args],
                            }
                        ],
                    },
                },
            },
        },
    ]
    return "\n---\n".join(yaml.dump(doc, default_flow_style=False) for doc in documents)


def cmd_run_local(*, opts: RunOptions, detach: bool, dry_run: bool) -> None:
    """Run benchmark as a single pod via ``aiperf profile`` CLI flags."""
    from aiperf.common.config.user_config import UserConfig

    config_path = Path(opts.config) if opts.config else DEFAULT_BENCHMARK_CONFIG
    if not config_path.exists():
        log_error(f"Config file not found: {config_path}")
        raise SystemExit(1)

    # Load YAML → validate through UserConfig → extract CLI args
    user_data = yaml.safe_load(config_path.read_text())
    user_config = UserConfig(**(user_data or {}))
    cli_args = _user_config_to_cli_args(user_config)
    cli_args.extend(["--workers-max", str(opts.workers)])
    cli_args.extend(["--ui-type", "none"])

    namespace = f"aiperf-local-{int(time.time())}"
    manifest = _generate_single_pod_manifest(cli_args, namespace, AIPERF_IMAGE)

    if dry_run:
        print(manifest, end="")
        return

    _require_kubectl_and_cluster()
    log_step("Deploying single-pod benchmark")
    kubectl("apply", "-f", "-", input=manifest, text=True)
    log_success(f"Deployed to namespace: {namespace}")

    if detach:
        log_info("Detached mode — benchmark running in background")
        log_info(f"View logs: kubectl logs -n {namespace} -l app=aiperf -f")
        log_info(f"Cleanup:   kubectl delete namespace {namespace}")
    else:
        log_info("Waiting for pod to start...")
        kubectl(
            "wait",
            "--for=condition=Ready",
            "pod",
            "-l",
            "app=aiperf",
            "-n",
            namespace,
            "--timeout=120s",
            check=False,
        )
        log_info("Attaching to benchmark logs (Ctrl-C to detach)...")
        kubectl(
            "logs",
            "-n",
            namespace,
            "-l",
            "app=aiperf",
            "-f",
            "--all-containers",
            check=False,
        )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _status_deployment(label: str, name: str, namespace: str | None = None) -> None:
    """Print a status line for a Kubernetes deployment."""
    print(f"\n{_CYAN}{label}:{_NC}")
    ready = _deployment_ready_replicas(name, namespace)
    if ready is not None:
        verb = "Installed" if namespace else "Deployed"
        print(_ok(f"{verb} ({ready} ready)"))
    else:
        verb = "Not installed" if namespace else "Not deployed"
        print(_miss(verb))


def cmd_status() -> None:
    require("kubectl")

    print()
    print("\u2554" + "\u2550" * 60 + "\u2557")
    print("\u2551" + "AIPerf Minikube Cluster Status".center(60) + "\u2551")
    print("\u255a" + "\u2550" * 60 + "\u255d")
    print()

    # --- Cluster ---
    print(f"{_CYAN}Cluster:{_NC}")
    if not cluster_exists():
        print(_miss(f"{CLUSTER_NAME} not found") + "\n")
        print("Run: ./dev/kube.py cluster-create")
        return

    if cluster_running():
        print(_ok(f"{CLUSTER_NAME} running"))
    else:
        print(_miss(f"{CLUSTER_NAME} stopped"))
        print("\nRun: minikube start -p " + CLUSTER_NAME)
        return

    r = kubectl("cluster-info", capture=True, check=False)
    print(
        _ok("kubectl connected")
        if r.returncode == 0
        else _fail("kubectl cannot connect")
    )

    # --- Docker images ---
    print(f"\n{_CYAN}Docker Images:{_NC}")
    for image in (AIPERF_IMAGE, MOCK_SERVER_IMAGE):
        r = sh(
            "docker",
            "image",
            "inspect",
            image,
            "--format",
            "{{.Size}}",
            capture=True,
            check=False,
        )
        if r.returncode != 0:
            print(_miss(f"{image} not built"))
            continue
        try:
            label = _format_bytes_iec(int(r.stdout.strip()))
        except (ValueError, TypeError):
            label = "?"
        print(_ok(f"{image} ({label})"))

    # --- Deployments ---
    _status_deployment(
        "JobSet Controller", "jobset-controller-manager", "jobset-system"
    )
    _status_deployment("Mock Server", "aiperf-mock-server")

    # --- GPU ---
    print(f"\n{_CYAN}GPU:{_NC}")
    gpu_count = _get_gpu_allocatable()
    if gpu_count and gpu_count != "0":
        print(_ok(f"{gpu_count} nvidia.com/gpu allocatable"))
    else:
        print(_miss("No GPU resources detected"))

    # --- Dynamo operator ---
    print(f"\n{_CYAN}Dynamo Operator:{_NC}")
    r = kubectl(
        "get", "crd", "dynamographdeployments.nvidia.com", capture=True, check=False
    )
    if r.returncode == 0:
        print(_ok("Installed"))
    else:
        print(_miss("Not installed"))

    # --- vLLM ---
    _status_deployment("vLLM Server", "vllm-server", VLLM_NAMESPACE)

    # --- Dynamo server ---
    print(f"\n{_CYAN}Dynamo Server:{_NC}")
    r = kubectl(
        "get", "pods", "-n", DYNAMO_NAMESPACE, "--no-headers", capture=True, check=False
    )
    if r.returncode == 0 and r.stdout.strip():
        lines = r.stdout.strip().splitlines()
        running = sum(1 for line in lines if "Running" in line)
        print(
            _ok(f"Deployed ({running}/{len(lines)} pods running) — mode: {DYNAMO_MODE}")
        )
    else:
        print(_miss("Not deployed"))

    # --- LoRA adapters ---
    print(f"\n{_CYAN}LoRA Adapters:{_NC}")
    r = kubectl(
        "get",
        "dynamomodel",
        "-n",
        DYNAMO_NAMESPACE,
        "--no-headers",
        capture=True,
        check=False,
    )
    if r.returncode == 0 and r.stdout.strip():
        for line in r.stdout.strip().splitlines():
            print(_ok(line.split()[0]))
    else:
        print(_miss("None"))

    # --- Running benchmarks ---
    print(f"\n{_CYAN}Running Benchmarks:{_NC}")
    r = kubectl(
        "get",
        "jobsets",
        "--all-namespaces",
        "-o",
        "jsonpath={.items[*].metadata.name}",
        capture=True,
        check=False,
    )
    if r.stdout.strip():
        kubectl("get", "jobsets", "--all-namespaces", "-o", "wide", check=False)
    else:
        print(_miss("No active benchmarks"))

    # --- AIPerf namespaces ---
    print(f"\n{_CYAN}AIPerf Namespaces:{_NC}")
    namespaces = _get_aiperf_namespaces()
    if namespaces:
        for ns in namespaces:
            pods = kubectl(
                "get", "pods", "-n", ns, "--no-headers", capture=True, check=False
            )
            count = len(pods.stdout.strip().splitlines()) if pods.stdout.strip() else 0
            print(f"  - {ns} ({count} pods)")
    else:
        print(_miss("No aiperf namespaces"))
    print()


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


@app.command(name="logs", group=lowlevel, help="View AIPerf benchmark pod logs.")
def cmd_logs(
    *,
    namespace: Annotated[
        str | None,
        Parameter(name=["--namespace", "-n"], help="Kubernetes namespace."),
    ] = os.environ.get("NS") or None,
    pod: Annotated[
        str, Parameter(name=["--pod", "-p"], help="Pod name filter.")
    ] = os.environ.get("POD") or "controller",
    follow: FollowFlag = FOLLOW_DEFAULT,
) -> None:
    _require_kubectl_and_cluster()

    if not namespace:
        ns_list = _get_aiperf_namespaces(latest_only=True)
        if not ns_list:
            log_error("No AIPerf namespace found. Run a benchmark first.")
            raise SystemExit(1)
        namespace = ns_list[0]
        log_info(f"Using namespace: {namespace}")

    follow_flag = ["-f"] if follow else []

    if pod == "all":
        kubectl(
            "logs",
            "-n",
            namespace,
            *follow_flag,
            "-l",
            "job-name",
            "--all-containers",
            "--prefix",
        )
        return

    r = kubectl(
        "get", "pods", "-n", namespace, "-l", "job-name", "-o", "name", capture=True
    )
    match = [p for p in r.stdout.strip().splitlines() if pod in p]

    if not match:
        log_error(f"Pod '{pod}' not found in {namespace}")
        raise SystemExit(1)

    extra = ["-c", "system-controller"] if pod == "controller" else []
    kubectl("logs", "-n", namespace, *follow_flag, match[0], *extra)


# ---------------------------------------------------------------------------
# Cleanup / teardown / reload
# ---------------------------------------------------------------------------


def cmd_cleanup() -> None:
    log_step("Cleaning up AIPerf resources")
    require("kubectl")

    if not cluster_running():
        log_warn(f"Cluster {CLUSTER_NAME} is not running")
        return

    log_info("Deleting all AIPerf namespaces...")
    namespaces = _get_aiperf_namespaces()
    if namespaces:
        for ns in namespaces:
            log_info(f"  Deleting: {ns}")
            kubectl("delete", "namespace", ns, "--ignore-not-found", check=False)
    else:
        log_info("  No AIPerf namespaces found")

    log_info("Deleting any orphaned JobSets...")
    kubectl(
        "delete",
        "jobsets",
        "--all",
        "--all-namespaces",
        "--ignore-not-found",
        check=False,
    )
    log_success("Cleanup complete")


def cmd_teardown() -> None:
    if cluster_running():
        cmd_cleanup()
    cmd_cluster_delete()


def cmd_reload() -> None:
    cmd_build(aiperf=True, mock=False)
    cmd_load(aiperf=True, mock=False)
    log_success("AIPerf image rebuilt and reloaded")


# ---------------------------------------------------------------------------
# Setup pipeline
# ---------------------------------------------------------------------------


def cmd_setup() -> None:
    """Full setup: cluster + GPU + Dynamo operator + JobSet + images + servers."""
    _preflight()
    cmd_cluster_create()
    cmd_install_dynamo()
    cmd_install_jobset()
    cmd_build(aiperf=True, mock=True)
    cmd_load(aiperf=True, mock=True)
    cmd_deploy_mock()

    print()
    log_success("Setup complete!")
    log_info("Next steps:")
    log_info("  ./dev/kube.py deploy-dynamo   # Deploy Dynamo inference server")
    log_info("  ./dev/kube.py deploy-vllm     # Deploy vLLM inference server")
    log_info("  ./dev/kube.py run             # Run benchmark against mock server")


# ---------------------------------------------------------------------------
# Command registration — no-arg commands (table-driven)
# ---------------------------------------------------------------------------

# fmt: off
for _func, _name, _group, _help in [
    (cmd_doctor,         "doctor",         workflow, "Check prerequisites (minikube, kubectl, helm, docker, nvidia, k9s)."),
    (cmd_setup,          "setup",          workflow, "Full setup: cluster + GPU + Dynamo operator + JobSet + mock server."),
    (cmd_teardown,       "teardown",       workflow, "Delete entire minikube cluster."),
    (cmd_status,         "status",         workflow, "Show cluster status (GPU, Dynamo, vLLM, benchmarks)."),
    (cmd_reload,         "reload",         workflow, "Rebuild + load AIPerf image."),
    (cmd_remove_dynamo,  "remove-dynamo",  server,   "Remove Dynamo server."),
    (cmd_remove_vllm,    "remove-vllm",    server,   "Remove vLLM server."),
    (cmd_deploy_mock,    "deploy-mock",    server,   "Deploy mock LLM server."),
    (cmd_remove_mock,    "remove-mock",    server,   "Remove mock server."),
    (cmd_cluster_create, "cluster-create", lowlevel, "Create minikube cluster with GPU."),
    (cmd_cluster_delete, "cluster-delete", lowlevel, "Delete minikube cluster."),
    (cmd_install_dynamo, "install-dynamo", lowlevel, "Install Dynamo operator (Helm)."),
    (cmd_install_jobset, "install-jobset", lowlevel, "Install JobSet controller."),
    (cmd_cleanup,        "cleanup",        lowlevel, "Remove benchmark namespaces."),
]:
    app.command(_func, name=_name, group=_group, help=_help)
# fmt: on

# ---------------------------------------------------------------------------
# Thin wrappers — commands that delegate with hardcoded args
# ---------------------------------------------------------------------------


# fmt: off
@app.command(name="run",        group=benchmark, help="Run AIPerf benchmark (attach).")
def _cli_run(*, opts: RunOptions = _DEFAULT_RUN_OPTS) -> None:
    cmd_run(opts=opts, detach=False, dry_run=False)

@app.command(name="run-detach", group=benchmark, help="Run benchmark (detach).")
def _cli_run_detach(*, opts: RunOptions = _DEFAULT_RUN_OPTS) -> None:
    cmd_run(opts=opts, detach=True,  dry_run=False)

@app.command(name="dry-run", group=benchmark, help="Print manifest only.")
def _cli_dry_run(*, opts: RunOptions = _DEFAULT_RUN_OPTS) -> None:
    cmd_run(opts=opts, detach=False, dry_run=True)

@app.command(name="run-local",        group=benchmark, help="Run single-pod benchmark (attach).")
def _cli_run_local(*, opts: RunOptions = _DEFAULT_RUN_OPTS) -> None:
    cmd_run_local(opts=opts, detach=False, dry_run=False)

@app.command(name="run-local-detach", group=benchmark, help="Run single-pod benchmark (detach).")
def _cli_run_local_detach(*, opts: RunOptions = _DEFAULT_RUN_OPTS) -> None:
    cmd_run_local(opts=opts, detach=True,  dry_run=False)

@app.command(name="dry-run-local",    group=benchmark, help="Print single-pod manifest only.")
def _cli_dry_run_local(*, opts: RunOptions = _DEFAULT_RUN_OPTS) -> None:
    cmd_run_local(opts=opts, detach=False, dry_run=True)

@app.command(name="build", group=lowlevel, help="Build Docker images.")
def _cli_build() -> None:
    cmd_build(aiperf=True, mock=True)

@app.command(name="load",  group=lowlevel, help="Load images into minikube.")
def _cli_load() -> None:
    cmd_load(aiperf=True, mock=True)
# fmt: on


@app.meta.default
def _cli_meta(
    *tokens: Annotated[str, Parameter(show=False)],
    no_banner: Annotated[
        bool,
        Parameter(name="--no-banner", negative="", help="Suppress the startup banner."),
    ] = False,
) -> None:
    if not no_banner:
        print(_BANNER)
    app(tokens)


if __name__ == "__main__":
    app.meta()
