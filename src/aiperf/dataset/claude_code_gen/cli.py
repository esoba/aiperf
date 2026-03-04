# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI subcommands for Claude Code dataset generation."""

from __future__ import annotations

import functools
import http.server
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import orjson
from cyclopts import App
from rich.console import Console

from aiperf.dataset.claude_code_gen.config_loader import load_config
from aiperf.dataset.claude_code_gen.models import SessionDistributionConfig
from aiperf.dataset.claude_code_gen.session_synthesizer import SessionSynthesizer
from aiperf.dataset.claude_code_gen.simulation import load_sessions, render_simulation
from aiperf.dataset.claude_code_gen.writer import write_dataset
from aiperf.dataset.loader.models import MooncakeTrace

claude_code_gen_app = App(
    name="claude-code-gen",
    help="Synthesize Claude Code single-thread session datasets for AIPerf load generation.",
)


def _serve_html(directory: Path, html_files: list[str], port: int) -> None:
    """Start a local HTTP server, print URLs for *html_files*, and open the first."""
    console = Console()
    existing = [f for f in html_files if (directory / f).exists()]
    if not existing:
        console.print("[red]No HTML files found.[/red]")
        return

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    try:
        server = http.server.HTTPServer(("", port), handler)
    except OSError:
        console.print(f"[red]Port {port} is already in use. Try --port <other>.[/red]")
        return

    base = f"http://localhost:{port}"
    for name in existing:
        console.print(f"[green]{base}/{name}[/green]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    threading.Timer(0.5, lambda: webbrowser.open(f"{base}/{existing[0]}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped.[/dim]")
    finally:
        server.server_close()


@claude_code_gen_app.command(name="synthesize")
def synthesize(
    num_sessions: int = 1000,
    output: Path = Path("."),
    config: str | None = None,
    seed: int = 42,
    max_isl: int | None = None,
) -> None:
    """Synthesize multi-turn session dataset into a unique run directory.

    --config accepts a path to a config JSON or a manifest.json from a previous run.
    If omitted, built-in defaults are used.

    Examples:
        aiperf claude-code-gen synthesize --num-sessions 1000 --output .test/
        aiperf claude-code-gen synthesize --config custom.json --num-sessions 500
        aiperf claude-code-gen synthesize --config .test/prev_run/manifest.json --num-sessions 1000
        aiperf claude-code-gen synthesize --max-isl 262144 --num-sessions 1000

    Args:
        num_sessions: Number of sessions to generate.
        output: Parent directory for the run directory (default: current dir).
        config: Path to config/manifest JSON (default: built-in defaults).
        seed: Random seed for reproducibility.
        max_isl: Maximum input sequence length — overrides max_prompt_tokens to clip context.
    """
    console = Console()

    if config:
        dist_config = load_config(config)
        config_name = Path(config).stem if Path(config).is_file() else config
    else:
        dist_config = SessionDistributionConfig()
        config_name = "default"

    if max_isl is not None:
        dist_config = dist_config.model_copy(update={"max_prompt_tokens": max_isl})

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir_name = f"{config_name}_{num_sessions}s_seed{seed}_{timestamp}"
    run_dir = Path(output) / run_dir_name

    synth = SessionSynthesizer(dist_config, seed=seed)
    console.print(f"Generating {num_sessions} sessions (seed={seed})...")
    sessions = synth.synthesize_sessions(num_sessions)

    jsonl_path, manifest_path, quality_path = write_dataset(
        sessions, run_dir, dist_config, seed=seed, config_name=config_name
    )

    total_turns = sum(len(s.turns) for s in sessions)
    console.print(f"[green]Run directory: {run_dir}[/green]")
    console.print(f"  JSONL:           {jsonl_path} ({total_turns} turns)")
    console.print(f"  Manifest:        {manifest_path}")
    console.print(f"  Quality:         {quality_path}")
    console.print(f"  Dashboard:       {run_dir / 'report.html'}")
    console.print(f"  Cache explorer:  {run_dir / 'cache_explorer.html'}")
    console.print()

    comparison_path = run_dir / "comparison.txt"
    if comparison_path.exists():
        console.print(comparison_path.read_text())

    console.print(f"[dim]View: open {run_dir / 'report.html'} in a browser[/dim]")


@claude_code_gen_app.command(name="validate")
def validate(
    input: Path,
) -> None:
    """Validate a generated JSONL dataset for Mooncake compatibility.

    Examples:
        aiperf claude-code-gen validate --input dataset.jsonl

    Args:
        input: Path to JSONL dataset file.
    """
    console = Console()
    errors: list[str] = []
    line_count = 0

    with input.open("rb") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                data = orjson.loads(line)
                MooncakeTrace(**data)
            except Exception as e:
                errors.append(f"Line {line_num}: {e}")
                if len(errors) >= 10:
                    break

    if errors:
        console.print(f"[red]Validation failed with {len(errors)} error(s):[/red]")
        for err in errors:
            console.print(f"  {err}")
        raise SystemExit(1)
    else:
        console.print(
            f"[green]Validation passed: {line_count} rows are Mooncake-compatible.[/green]"
        )


@claude_code_gen_app.command(name="simulate")
def simulate(
    input: Path,
    output: Path | None = None,
    port: int | None = None,
) -> None:
    """Generate a time-based simulation dashboard from a synthesized dataset.

    Reads dataset.jsonl and generates simulation.html showing how sessions
    play out over time at a given concurrency level with interactive controls.

    Examples:
        aiperf claude-code-gen simulate --input .test/run/dataset.jsonl
        aiperf claude-code-gen simulate --input .test/run/dataset.jsonl --port 8091

    Args:
        input: Path to dataset.jsonl file.
        output: Output HTML path (default: simulation.html next to JSONL).
        port: Serve on this port after generating (default: just write file).
    """
    console = Console()
    jsonl_path = Path(input)

    if not jsonl_path.exists():
        console.print(f"[red]File not found: {jsonl_path}[/red]")
        raise SystemExit(1)

    out_path = Path(output) if output else jsonl_path.parent / "simulation.html"
    sessions = load_sessions(jsonl_path)
    render_simulation(sessions, out_path)

    if port is not None:
        _serve_html(out_path.parent, [out_path.name], port)
