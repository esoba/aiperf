# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UI dispatch and progress streaming helpers for kube commands."""

from __future__ import annotations

from aiperf.common.enums import MessageType

# WebSocket subscription message types for progress streaming
WS_MESSAGE_TYPES = [
    MessageType.CREDIT_PHASE_START,
    MessageType.CREDIT_PHASE_PROGRESS,
    MessageType.CREDIT_PHASE_COMPLETE,
    MessageType.REALTIME_METRICS,
    MessageType.WORKER_STATUS_SUMMARY,
    MessageType.ALL_RECORDS_RECEIVED,
]

# WebSocket reconnection settings
WS_MAX_RETRIES = 10

# API path segments used by CLI commands
API_WS_PATH = "/ws"


async def stream_progress(ws_url: str) -> None:
    """Stream progress messages from the benchmark via WebSocket.

    Args:
        ws_url: WebSocket URL for progress streaming.
    """
    from aiperf.kubernetes.console import logger, print_step
    from aiperf.kubernetes.port_forward import stream_progress_from_api

    print_step("Streaming progress...")
    logger.info("")

    async def handle_message(data: dict) -> bool:
        print_progress_message(data)
        return data.get("message_type") == MessageType.ALL_RECORDS_RECEIVED

    await stream_progress_from_api(
        ws_url,
        on_message=handle_message,
        message_types=WS_MESSAGE_TYPES,
        max_retries=WS_MAX_RETRIES,
    )


def print_progress_message(data: dict) -> None:
    """Log a progress message."""
    from aiperf.kubernetes.console import logger

    msg_type = data.get("message_type", "")

    if msg_type == "subscribed":
        return

    if msg_type == MessageType.CREDIT_PHASE_START:
        stats = data.get("stats", {})
        phase = stats.get("phase", "unknown")
        logger.info(f"[bold cyan]\\[PHASE][/bold cyan] Starting {phase} phase")

    elif msg_type == MessageType.CREDIT_PHASE_PROGRESS:
        stats = data.get("stats", {})
        phase = stats.get("phase", "")
        completed = stats.get("requests_completed", 0)
        total = stats.get("total_expected_requests", 0)
        percent = (completed / total * 100) if total > 0 else 0
        logger.info(
            f"[bold cyan]\\[PROGRESS][/bold cyan] {phase} "
            f"{completed}/{total} requests ({percent:.1f}%)"
        )

    elif msg_type == MessageType.CREDIT_PHASE_COMPLETE:
        stats = data.get("stats", {})
        phase = stats.get("phase", "unknown")
        logger.info(f"[bold cyan]\\[PHASE][/bold cyan] Completed {phase} phase")

    elif msg_type == MessageType.REALTIME_METRICS:
        print_realtime_metrics(data)

    elif msg_type == MessageType.WORKER_STATUS_SUMMARY:
        workers = data.get("workers", {})
        total = len(workers)
        healthy = sum(
            1
            for w in workers.values()
            if isinstance(w, dict) and w.get("status", "").upper() == "HEALTHY"
        )
        logger.info(f"[bold cyan]\\[WORKERS][/bold cyan] {healthy}/{total} healthy")

    elif msg_type == MessageType.ALL_RECORDS_RECEIVED:
        logger.info(
            "[bold green]\\[COMPLETE][/bold green] All records received, benchmark finishing..."
        )


def print_realtime_metrics(data: dict) -> None:
    """Log key metrics from realtime metrics message."""
    from aiperf.kubernetes.console import logger

    metrics = data.get("metrics", [])
    key_keywords = ["throughput", "latency", "ttft", "token"]
    found = []
    for m in metrics:
        tag = m.get("tag", "")
        tag_lower = tag.lower()
        if not any(kw in tag_lower for kw in key_keywords):
            continue
        value = m.get("value", m.get("avg", m.get("current", 0))) or 0
        unit = m.get("display_unit", m.get("unit", ""))
        found.append((tag, value, unit))
    for tag, value, unit in found:
        logger.info(f"[dim]\\[METRIC][/dim] {tag}: {value:.2f} {unit}".rstrip())
