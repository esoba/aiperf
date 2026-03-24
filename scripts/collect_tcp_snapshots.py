#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect TCP state snapshots from AIPerf worker and mock-LLM pods in parallel.

This script captures a near-simultaneous snapshot from all matching pods by:
1. Listing pods in a namespace
2. Matching worker and mock pods by regex
3. Running ``kubectl exec`` across all matched pods concurrently
4. Storing raw output per pod without overwriting an existing trace directory
5. Writing per-pod summary files with TCP state counts

Examples:
    uv run python scripts/collect_tcp_snapshots.py \
        --namespace acasagrande-aiperf-bench \
        --worker-pattern 'mock-250k-99p-nr-osl20k-132913-workers' \
        --mock-pattern 'mock-llm-'

    uv run python scripts/collect_tcp_snapshots.py \
        --namespace acasagrande-aiperf-bench \
        --worker-pattern 'my-job-workers' \
        --output-dir /tmp/tcp-snap-custom
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TCP_STATE_MAP = {
    "01": "ESTAB",
    "02": "SYN-SENT",
    "03": "SYN-RECV",
    "04": "FIN-WAIT-1",
    "05": "FIN-WAIT-2",
    "06": "TIME-WAIT",
    "07": "CLOSE",
    "08": "CLOSE-WAIT",
    "09": "LAST-ACK",
    "0A": "LISTEN",
    "0B": "CLOSING",
    "0C": "NEW-SYN-RECV",
}

TCP_CAPTURE_CMD = (
    "if command -v ss >/dev/null 2>&1; then "
    "ss -tan; "
    "else "
    "echo '__AIPERF_PROC_NET_TCP__'; "
    "cat /proc/net/tcp; "
    "echo '__AIPERF_PROC_NET_TCP6__'; "
    "cat /proc/net/tcp6; "
    "fi"
)


@dataclass(frozen=True)
class PodTarget:
    group: str
    pod: str


@dataclass
class CaptureResult:
    target: PodTarget
    output_path: Path
    success: bool
    returncode: int
    stderr: str
    state_counts: Counter[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        default="acasagrande-aiperf-bench",
        help="Kubernetes namespace containing the pods.",
    )
    parser.add_argument(
        "--worker-pattern",
        default=r"mock-250k-99p-nr-osl20k-132913-workers",
        help="Regex matched against pod names for AIPerf worker pods.",
    )
    parser.add_argument(
        "--mock-pattern",
        default=r"mock-llm-",
        help="Regex matched against pod names for mock-LLM server pods.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Target snapshot directory. If omitted, a fresh timestamped directory is "
            "created under /tmp. If the path already exists, a numeric suffix is added."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(32, (os.cpu_count() or 4) * 4),
        help="Maximum concurrent kubectl exec operations.",
    )
    parser.add_argument(
        "--kubectl",
        default="kubectl",
        help="kubectl executable to use.",
    )
    return parser.parse_args()


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )


def list_pods(kubectl: str, namespace: str) -> list[str]:
    result = run_command([kubectl, "-n", namespace, "get", "pods", "-o", "name"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "kubectl get pods failed")

    pods: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pods.append(line.removeprefix("pod/"))
    return pods


def build_targets(
    pods: list[str],
    worker_pattern: str,
    mock_pattern: str,
) -> list[PodTarget]:
    worker_re = re.compile(worker_pattern)
    mock_re = re.compile(mock_pattern)

    workers = sorted(p for p in pods if worker_re.search(p))
    mocks = sorted(p for p in pods if mock_re.search(p))

    targets = [PodTarget(group="workers", pod=pod) for pod in workers]
    targets.extend(PodTarget(group="mock", pod=pod) for pod in mocks)
    return targets


def ensure_unique_dir(requested: str | None) -> Path:
    base = (
        Path(requested)
        if requested
        else Path("/tmp") / f"tcp-snap-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}-{suffix}")
        suffix += 1

    (candidate / "workers").mkdir(parents=True, exist_ok=False)
    (candidate / "mock").mkdir(parents=True, exist_ok=False)
    return candidate


def summarize_states(raw_text: str) -> Counter[str]:
    lines = [line.rstrip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return Counter()

    if any(line.startswith("__AIPERF_PROC_NET_TCP__") for line in lines):
        return summarize_proc_net(lines)

    return summarize_ss(lines)


def summarize_ss(lines: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in lines:
        if line.startswith("State") or line.startswith("Recv-Q"):
            continue
        state = line.split(None, 1)[0]
        if state:
            counts[state] += 1
    return counts


def summarize_proc_net(lines: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in lines:
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("__AIPERF_PROC_NET_TCP__")
            or stripped.startswith("__AIPERF_PROC_NET_TCP6__")
            or stripped.startswith("sl")
        ):
            continue

        parts = stripped.split()
        if len(parts) < 4:
            continue
        state_hex = parts[3].upper()
        counts[TCP_STATE_MAP.get(state_hex, f"UNKNOWN-{state_hex}")] += 1
    return counts


def capture_target(
    kubectl: str,
    namespace: str,
    output_dir: Path,
    target: PodTarget,
) -> CaptureResult:
    output_path = output_dir / target.group / f"{target.pod}.txt"
    cmd = [
        kubectl,
        "-n",
        namespace,
        "exec",
        target.pod,
        "--",
        "sh",
        "-lc",
        TCP_CAPTURE_CMD,
    ]
    result = run_command(cmd)

    raw_text = result.stdout
    if result.stderr:
        if raw_text and not raw_text.endswith("\n"):
            raw_text += "\n"
        raw_text += f"\n[stderr]\n{result.stderr}"

    output_path.write_text(raw_text, encoding="utf-8")

    state_counts = (
        summarize_states(result.stdout) if result.returncode == 0 else Counter()
    )
    return CaptureResult(
        target=target,
        output_path=output_path,
        success=result.returncode == 0,
        returncode=result.returncode,
        stderr=result.stderr.strip(),
        state_counts=state_counts,
    )


def write_summary(path: Path, results: list[CaptureResult]) -> None:
    lines: list[str] = []
    aggregate: Counter[str] = Counter()

    for result in sorted(results, key=lambda item: item.target.pod):
        lines.append(f"=== {result.target.pod} ===")
        if result.success:
            if result.state_counts:
                for state, count in sorted(result.state_counts.items()):
                    lines.append(f"{count:>8} {state}")
                    aggregate[state] += count
            else:
                lines.append("       0 NO-STATES")
        else:
            lines.append(f"ERROR returncode={result.returncode}")
            if result.stderr:
                lines.append(result.stderr)
        lines.append("")

    lines.append("=== aggregate ===")
    if aggregate:
        for state, count in sorted(aggregate.items()):
            lines.append(f"{count:>8} {state}")
    else:
        lines.append("       0 NO-STATES")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    namespace: str,
    worker_pattern: str,
    mock_pattern: str,
    targets: list[PodTarget],
) -> None:
    lines = [
        f"namespace: {namespace}",
        f"worker_pattern: {worker_pattern}",
        f"mock_pattern: {mock_pattern}",
        f"captured_at: {datetime.now().isoformat()}",
        "",
        "targets:",
    ]
    for target in targets:
        lines.append(f"  - {target.group}: {target.pod}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = ensure_unique_dir(args.output_dir)

    try:
        pods = list_pods(args.kubectl, args.namespace)
    except RuntimeError as exc:
        print(f"failed to list pods: {exc}", file=sys.stderr)
        return 1

    targets = build_targets(pods, args.worker_pattern, args.mock_pattern)
    if not targets:
        print("no matching pods found", file=sys.stderr)
        return 1

    write_manifest(
        output_dir / "manifest.txt",
        args.namespace,
        args.worker_pattern,
        args.mock_pattern,
        targets,
    )

    print(f"snapshot_dir={output_dir}")
    print(f"matched_workers={sum(1 for t in targets if t.group == 'workers')}")
    print(f"matched_mock={sum(1 for t in targets if t.group == 'mock')}")
    print("starting parallel capture...", file=sys.stderr)

    results: list[CaptureResult] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        future_map = {
            executor.submit(
                capture_target,
                args.kubectl,
                args.namespace,
                output_dir,
                target,
            ): target
            for target in targets
        }
        for future in concurrent.futures.as_completed(future_map):
            target = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive CLI handling
                failed_path = output_dir / target.group / f"{target.pod}.txt"
                failed_path.write_text(f"unexpected error: {exc}\n", encoding="utf-8")
                results.append(
                    CaptureResult(
                        target=target,
                        output_path=failed_path,
                        success=False,
                        returncode=1,
                        stderr=str(exc),
                        state_counts=Counter(),
                    )
                )
                continue

            results.append(result)
            status = "ok" if result.success else f"failed({result.returncode})"
            print(f"{target.group}:{target.pod} {status}", file=sys.stderr)

    worker_results = [result for result in results if result.target.group == "workers"]
    mock_results = [result for result in results if result.target.group == "mock"]
    write_summary(output_dir / "workers-summary.txt", worker_results)
    write_summary(output_dir / "mock-summary.txt", mock_results)

    failures = [result for result in results if not result.success]
    if failures:
        print(f"completed with {len(failures)} failures", file=sys.stderr)
        return 2

    print(str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
