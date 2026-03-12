# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clean up stale benchmark namespaces from Kubernetes."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

app = App(name="cleanup")


@app.default
async def cleanup(
    *,
    max_age: Annotated[int, Parameter(name="--max-age")] = 3600,
    dry_run: Annotated[bool, Parameter(name="--dry-run")] = False,
    force: Annotated[bool, Parameter(name=["-f", "--force"])] = False,
    label: Annotated[str, Parameter(name=["-l", "--label"])] = "aiperf/job-id",
    kubeconfig: Annotated[str | None, Parameter(name="--kubeconfig")] = None,
    context: Annotated[str | None, Parameter(name="--context")] = None,
) -> None:
    """Remove stale benchmark namespaces older than max-age seconds.

    Finds namespaces with aiperf labels that are older than the specified
    max-age and removes them. By default, only shows what would be deleted
    (dry-run mode). Use --force to actually delete.

    Examples:
        aiperf kube cleanup                    # Dry-run, show stale namespaces
        aiperf kube cleanup --force            # Actually delete stale namespaces
        aiperf kube cleanup --max-age 7200     # 2 hour threshold
        aiperf kube cleanup --dry-run          # Preview what would be deleted
    """
    from aiperf import cli_utils

    with cli_utils.exit_on_error(title="Error Cleaning Up Namespaces"):
        await _cleanup_stale_namespaces(
            max_age=max_age,
            dry_run=dry_run,
            force=force,
            label=label,
            kubeconfig=kubeconfig,
            context=context,
        )


async def _cleanup_stale_namespaces(
    *,
    max_age: int,
    dry_run: bool,
    force: bool,
    label: str,
    kubeconfig: str | None,
    context: str | None,
) -> None:
    """Find and remove stale benchmark namespaces.

    Args:
        max_age: Maximum namespace age in seconds before considered stale.
        dry_run: If True, only report what would be deleted.
        force: Delete even if running pods exist.
        label: Label selector for finding aiperf namespaces.
        kubeconfig: Path to kubeconfig file.
        context: Kubernetes context name.
    """
    from datetime import datetime, timezone

    import kr8s.asyncio
    from kr8s.asyncio.objects import Namespace, Pod

    from aiperf.kubernetes.console import (
        print_error,
        print_header,
        print_info,
        print_success,
        print_warning,
    )

    api = await kr8s.asyncio.api(kubeconfig=kubeconfig, context=context)
    namespaces: list[Namespace] = await Namespace.list(api=api, label_selector=label)

    if not namespaces:
        print_info("No aiperf benchmark namespaces found.")
        return

    now = datetime.now(tz=timezone.utc)
    stale: list[tuple[Namespace, float, int, int]] = []

    for ns in namespaces:
        created_str = ns.metadata.get("creationTimestamp", "")
        if not created_str:
            continue

        created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        age_seconds = (now - created).total_seconds()

        if age_seconds < max_age:
            continue

        ns_name = ns.metadata.get("name", "")
        pods: list[Pod] = await Pod.list(api=api, namespace=ns_name)
        running_count = sum(1 for p in pods if p.status.get("phase") == "Running")
        total_pods = len(pods)

        if running_count > 0 and not force:
            print_warning(
                f"Skipping {ns_name}: {running_count} running pod(s) "
                f"(use --force to override)"
            )
            continue

        stale.append((ns, age_seconds, running_count, total_pods))

    if not stale:
        print_info("No stale namespaces found.")
        return

    print_header("Stale Benchmark Namespaces")
    for ns, age_seconds, running_count, total_pods in stale:
        ns_name = ns.metadata.get("name", "")
        age_str = _format_duration(age_seconds)
        pod_info = f"{total_pods} pod(s), {running_count} running"
        print_info(f"  {ns_name}  age={age_str}  ({pod_info})")

    action = "Deleting" if force and not dry_run else "Would delete"
    print_info(f"{action} {len(stale)} namespace(s)")

    if dry_run or not force:
        print_info("Dry-run mode. Use --force to delete.")
        return

    deleted = 0
    for ns, _, _, _ in stale:
        ns_name = ns.metadata.get("name", "")
        try:
            await ns.delete()
            print_success(f"Deleted namespace {ns_name}")
            deleted += 1
        except Exception as e:
            print_error(f"Failed to delete {ns_name}: {e}")

    print_info(f"Cleaned up {deleted}/{len(stale)} namespace(s)")


def _format_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string (e.g., "2h 15m", "45m", "30s").
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining_mins = minutes % 60
    if remaining_mins:
        return f"{hours}h {remaining_mins}m"
    return f"{hours}h"
