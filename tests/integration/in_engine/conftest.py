# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Watchdog fixture for in-engine integration tests.

Monitors aiperf subprocess tree during test execution, detecting:
- Worker process crashes / zombie processes
- GPU memory usage changes
- Log file progress (new lines = forward progress)
- Hang detection (no log progress for N seconds)

Prints periodic status lines to stdout for real-time visibility.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfRunnerResult

# ---------------------------------------------------------------------------
# CUDA JIT environment setup for SGLang (flashinfer / sglang jit_kernel)
# ---------------------------------------------------------------------------
# SGLang requires nvcc for JIT compilation of flashinfer and custom kernels.
# On systems with GCC > 14 (e.g., Arch Linux rolling), nvcc 12.x fails to
# compile against the C++ stdlib. We use a standalone nvcc + GCC 14 from
# conda-forge to work around this.
#
# Setup (one-time, run manually):
#   1. Download CUDA nvcc redistributable:
#      curl -sL "https://developer.download.nvidia.com/compute/cuda/redist/\
#      cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz" \
#      -o /tmp/nvcc.tar.xz && cd /tmp && tar xf nvcc.tar.xz
#   2. Download CCCL headers:
#      curl -sL "https://developer.download.nvidia.com/compute/cuda/redist/\
#      cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz" \
#      -o /tmp/cccl.tar.xz && cd /tmp && tar xf cccl.tar.xz
#   3. Build CUDA_HOME:
#      mkdir -p /tmp/cuda_home/{bin,include,lib64}
#      cp nvcc-archive/bin/* /tmp/cuda_home/bin/
#      cp -a cccl-archive/include/* /tmp/cuda_home/include/
#      # Copy cuda_runtime headers from pip nvidia packages
#   4. Install GCC 14:
#      curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
#      tar -xvj -C /tmp bin/micromamba
#      MAMBA_ROOT_PREFIX=/tmp/mamba /tmp/bin/micromamba create -n gcc14 \
#      -c conda-forge gcc_linux-64=14 gxx_linux-64=14 sysroot_linux-64 --yes
#   5. Create nvcc wrapper at /tmp/cuda_home/bin/nvcc that uses GCC 14
# ---------------------------------------------------------------------------

_CUDA_HOME = "/tmp/cuda_home"
_GCC14_CC = "/tmp/mamba/envs/gcc14/bin/x86_64-conda-linux-gnu-gcc"
_GCC14_CXX = "/tmp/mamba/envs/gcc14/bin/x86_64-conda-linux-gnu-g++"


@pytest.fixture(scope="session", autouse=True)
def _setup_cuda_jit_env() -> None:
    """Set CUDA_HOME and GCC 14 for SGLang JIT compilation.

    These env vars are inherited by aiperf subprocesses so that
    flashinfer and sglang jit_kernel can compile CUDA code.
    Only activates if the CUDA_HOME directory exists.
    """
    if os.path.isdir(_CUDA_HOME) and os.path.isfile(_GCC14_CC):
        os.environ.setdefault("CUDA_HOME", _CUDA_HOME)
        os.environ.setdefault("CC", _GCC14_CC)
        os.environ.setdefault("CXX", _GCC14_CXX)


def _get_child_pids(parent_pid: int) -> list[dict[str, str]]:
    """Get child processes of a given PID using /proc filesystem.

    Returns list of dicts with keys: pid, name, state, state_char.
    """
    children = []
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                stat_path = f"/proc/{entry}/stat"
                with open(stat_path) as f:
                    stat = f.read()
                # Parse: pid (comm) state ppid ...
                match = re.match(r"(\d+) \((.+?)\) (\S) (\d+)", stat)
                if match and int(match.group(4)) == parent_pid:
                    state_char = match.group(3)
                    state_map = {
                        "R": "running",
                        "S": "sleeping",
                        "D": "disk-wait",
                        "Z": "zombie",
                        "T": "stopped",
                        "X": "dead",
                    }
                    children.append(
                        {
                            "pid": match.group(1),
                            "name": match.group(2),
                            "state_char": state_char,
                            "state": state_map.get(state_char, state_char),
                        }
                    )
            except (FileNotFoundError, PermissionError):
                continue
    except FileNotFoundError:
        pass
    return children


def _get_gpu_memory_mb() -> str:
    """Get GPU memory usage via nvidia-smi (single line summary)."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(",")
            if len(parts) == 2:
                return f"{parts[0].strip()}MB/{parts[1].strip()}MB"
    except Exception:
        pass
    return "N/A"


def _count_log_lines(log_dir: Path) -> int:
    """Count total lines across all log files in a directory."""
    total = 0
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            try:
                with open(log_file) as f:
                    total += sum(1 for _ in f)
            except Exception:
                pass
    return total


def _tail_log_errors(log_dir: Path, last_n: int = 5) -> list[str]:
    """Get last N error/warning lines from log files."""
    errors = []
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            try:
                with open(log_file) as f:
                    for line in f:
                        if any(
                            kw in line
                            for kw in ["ERROR", "CRITICAL", "Exception", "Traceback"]
                        ):
                            errors.append(line.strip()[:200])
            except Exception:
                pass
    return errors[-last_n:]


class InEngineWatchdog:
    """Background watchdog that monitors aiperf subprocess health.

    Attributes:
        poll_interval: Seconds between status checks.
        hang_timeout: Seconds of no log progress before declaring a hang.
        status_log: List of all status messages for post-mortem analysis.
    """

    def __init__(
        self,
        output_dir: Path,
        poll_interval: float = 5.0,
        hang_timeout: float = 120.0,
    ) -> None:
        self._output_dir = output_dir
        self._poll_interval = poll_interval
        self._hang_timeout = hang_timeout
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self.status_log: list[str] = []
        self._zombies_detected: list[str] = []
        self._errors_detected: list[str] = []

    async def start(self) -> None:
        """Start the watchdog background task."""
        self._stop_event.clear()
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop the watchdog and print summary."""
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._print_summary()

    def _log(self, msg: str) -> None:
        """Log a status message to both stdout and the status log."""
        ts = time.strftime("%H:%M:%S")
        line = f"[watchdog {ts}] {msg}"
        print(line, flush=True)
        self.status_log.append(line)

    def _print_summary(self) -> None:
        """Print post-mortem summary."""
        if self._zombies_detected:
            print(
                f"\n[watchdog] ZOMBIES DETECTED: {self._zombies_detected}", flush=True
            )
        if self._errors_detected:
            print("\n[watchdog] ERRORS DETECTED:", flush=True)
            for err in self._errors_detected[-10:]:
                print(f"  {err}", flush=True)

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        last_log_lines = 0
        last_progress_time = time.monotonic()
        log_dir = self._output_dir / "logs"
        tick = 0

        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                return

            tick += 1
            now = time.monotonic()

            # Find aiperf processes (look for our output dir to identify the right tree)
            aiperf_pids = self._find_aiperf_pids()

            # Check process tree
            all_children = []
            zombies = []
            for pid in aiperf_pids:
                children = _get_child_pids(pid)
                all_children.extend(children)
                for child in children:
                    if child["state_char"] == "Z":
                        zombies.append(f"{child['name']}(pid={child['pid']})")

            if zombies:
                self._zombies_detected.extend(zombies)
                self._log(f"ZOMBIE PROCESSES: {zombies}")

            # Check log progress
            current_log_lines = _count_log_lines(log_dir)
            if current_log_lines > last_log_lines:
                last_log_lines = current_log_lines
                last_progress_time = now

            stall_secs = now - last_progress_time

            # Check for errors in logs
            errors = _tail_log_errors(log_dir)
            new_errors = [e for e in errors if e not in self._errors_detected]
            if new_errors:
                self._errors_detected.extend(new_errors)
                for err in new_errors:
                    self._log(f"LOG ERROR: {err}")

            # GPU memory
            gpu_mem = _get_gpu_memory_mb()

            # Build status line
            proc_names = [f"{c['name']}({c['state_char']})" for c in all_children[:8]]
            proc_summary = ", ".join(proc_names) if proc_names else "no children"

            # Print status every tick
            self._log(
                f"gpu={gpu_mem} | log_lines={current_log_lines} | "
                f"stall={stall_secs:.0f}s | procs=[{proc_summary}]"
            )

            # Hang detection
            if stall_secs > self._hang_timeout and current_log_lines > 0:
                self._log(
                    f"HANG DETECTED: No log progress for {stall_secs:.0f}s "
                    f"(threshold={self._hang_timeout}s)"
                )

    def _find_aiperf_pids(self) -> list[int]:
        """Find aiperf main process PIDs by scanning /proc."""
        pids = []
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    cmdline_path = f"/proc/{entry}/cmdline"
                    with open(cmdline_path, "rb") as f:
                        cmdline = f.read().decode("utf-8", errors="replace")
                    if "aiperf" in cmdline and "system_controller" in cmdline:
                        pids.append(int(entry))
                except (FileNotFoundError, PermissionError):
                    continue
        except FileNotFoundError:
            pass
        return pids


# ---------------------------------------------------------------------------
# TRT-LLM Docker-based runner
# ---------------------------------------------------------------------------
# TRT-LLM requires a specific container environment (torch 2.10, transformers
# 4.57, tensorrt_llm 1.3.0rc7) that is incompatible with the host venv.
# We run aiperf inside a persistent Docker container using `docker exec`.
# ---------------------------------------------------------------------------

TRTLLM_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7"
_TRTLLM_CONTAINER_NAME = f"aiperf-trtllm-test-{os.getpid()}"


@pytest.fixture(scope="session")
def _trtllm_container() -> str:
    """Start a persistent TRT-LLM container with aiperf installed.

    The container runs for the entire test session. aiperf is installed
    once via `pip install -e .` and all tests use `docker exec` to run
    aiperf commands inside it.

    Yields:
        Container name for use with `docker exec`.
    """
    project_root = Path(__file__).resolve().parents[3]
    hf_cache = Path.home() / ".cache" / "huggingface"
    container_name = _TRTLLM_CONTAINER_NAME

    # Check Docker is available
    r = subprocess.run(["docker", "info"], capture_output=True)
    if r.returncode != 0:
        pytest.skip("Docker not available")

    # Check image is pulled
    r = subprocess.run(
        ["docker", "image", "inspect", TRTLLM_IMAGE], capture_output=True
    )
    if r.returncode != 0:
        pytest.skip(f"{TRTLLM_IMAGE} not pulled")

    # Remove any stale container from a previous crashed run
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    # The container needs CUDA forward-compat libs (for newer CUDA than host
    # driver supports), TensorRT libs, and tensorrt pip package libs.
    trtllm_ld_path = (
        "/usr/local/cuda/compat/lib"
        ":/usr/local/tensorrt/lib"
        ":/usr/local/lib/python3.12/dist-packages/tensorrt_libs"
    )

    print(f"\n[TRT-LLM] Starting persistent container {container_name}...")
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--init",  # tini as PID 1 — reaps MPI zombie processes
            "--name",
            container_name,
            "--gpus",
            "all",
            "--ipc=host",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "-v",
            f"{project_root}:/app",
            "-v",
            f"{hf_cache}:/root/.cache/huggingface",
            "-v",
            "/tmp:/tmp",
            "-w",
            "/app",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "HF_HOME=/root/.cache/huggingface",
            "-e",
            f"LD_LIBRARY_PATH={trtllm_ld_path}",
            TRTLLM_IMAGE,
            "sleep",
            "infinity",
        ],
        check=True,
    )

    # Install aiperf inside the container.
    # 1. Install hatchling build backend (not in container by default)
    # 2. Install aiperf in editable mode
    print("[TRT-LLM] Installing aiperf in container...")
    r = subprocess.run(
        [
            "docker",
            "exec",
            container_name,
            "pip",
            "install",
            "hatchling",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        print(f"[TRT-LLM] hatchling install failed:\n{r.stderr}")

    r = subprocess.run(
        [
            "docker",
            "exec",
            container_name,
            "pip",
            "install",
            "-e",
            ".[dev]",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r.returncode != 0:
        print(f"[TRT-LLM] pip install stderr:\n{r.stderr}")
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        pytest.fail(f"Failed to install aiperf in TRT-LLM container:\n{r.stderr}")

    # Clear stale __pycache__ bytecode to prevent AttributeError on edited files.
    # Race condition: .pyc same-second timestamps but compiled from older source.
    subprocess.run(
        [
            "docker",
            "exec",
            container_name,
            "find",
            "src",
            "tests",
            "-name",
            "__pycache__",
            "-type",
            "d",
            "-exec",
            "rm",
            "-rf",
            "{}",
            "+",
        ],
        capture_output=True,
    )

    # Verify tensorrt_llm is still importable after pip install
    r = subprocess.run(
        [
            "docker",
            "exec",
            "-e",
            f"LD_LIBRARY_PATH={trtllm_ld_path}",
            container_name,
            "python3",
            "-c",
            "import tensorrt_llm; print(tensorrt_llm.__version__)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        print(f"[TRT-LLM] tensorrt_llm import failed: {r.stderr}")
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        pytest.fail(f"tensorrt_llm not importable after install:\n{r.stderr}")
    print(f"[TRT-LLM] tensorrt_llm version: {r.stdout.strip()}")
    print("[TRT-LLM] Container ready!")

    yield container_name

    # Cleanup
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


@pytest.fixture
async def trtllm_cli(_trtllm_container: str, temp_output_dir: Path) -> AIPerfCLI:
    """CLI that runs aiperf inside the TRT-LLM Docker container.

    Uses `docker exec` on the persistent container to run aiperf commands.
    The temp_output_dir is shared via /tmp mount between host and container.
    """
    container_name = _trtllm_container

    async def runner(args: list[str], timeout: float = 300.0) -> AIPerfRunnerResult:
        full_args = list(args)
        if full_args and full_args[0] == "profile":
            full_args += ["--artifact-dir", str(temp_output_dir)]
            if "--tokenizer" not in full_args:
                full_args += ["--tokenizer", "Qwen/Qwen3-0.6B"]

        cmd = [
            "docker",
            "exec",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/tensorrt/lib:/usr/local/lib/python3.12/dist-packages/tensorrt_libs",
            "-e",
            "AIPERF_SERVICE__REGISTRATION_TIMEOUT=120",
            "-e",
            "AIPERF_SERVICE__START_TIMEOUT=120",
            container_name,
            "python3",
            "-m",
            "aiperf",
        ] + full_args

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=None,
            stderr=None,
        )

        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Kill aiperf processes inside the container
            await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                container_name,
                "pkill",
                "-f",
                "python.*aiperf",
            )
            raise RuntimeError(f"TRT-LLM test timed out after {timeout}s") from None

        return AIPerfRunnerResult(
            exit_code=process.returncode or 0,
            output_dir=temp_output_dir,
        )

    return AIPerfCLI(runner)


@pytest.fixture
async def watchdog(tmp_path: Path) -> InEngineWatchdog:
    """Pytest fixture that provides a background watchdog for in-engine tests.

    The watchdog starts automatically and monitors the aiperf subprocess tree,
    GPU memory, and log files throughout the test. It prints periodic status
    updates and detects crashes, zombies, and hangs.

    Usage in tests:
        async def test_something(self, cli: AIPerfCLI, watchdog: InEngineWatchdog) -> None:
            result = await cli.run(cmd, timeout=300.0)
            assert result.request_count == 5
    """
    output_dir = tmp_path / "aiperf_output"
    wd = InEngineWatchdog(output_dir, poll_interval=5.0, hang_timeout=120.0)
    await wd.start()
    yield wd
    await wd.stop()
