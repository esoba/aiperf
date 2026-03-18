# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for analyzing trace datasets."""

from __future__ import annotations

from pathlib import Path

from cyclopts import App

app = App(name="analyze-trace")


@app.default
def analyze_trace(
    input_file: Path,
    block_size: int = 512,
    output_file: Path | None = None,
) -> None:
    """Analyze a trace file or directory for distributions and statistics.

    Auto-detects the format:
    - Conflux JSON (file or directory of files): conversation structure, token distributions, timing
    - JSONL traces (Mooncake/Bailian): ISL/OSL distributions, prefix cache hit rates

    Args:
        input_file: Path to trace file or directory
        block_size: KV cache block size for JSONL prefix analysis (default: 512)
        output_file: Optional output path for analysis report (JSON)
    """
    from aiperf.dataset.synthesis.cli import analyze_trace as _analyze_trace

    _analyze_trace(input_file, block_size=block_size, output_file=output_file)
