# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube init command: generate starter configuration template."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

app = App(name="init")


@app.default
async def init_config(
    *,
    output: Annotated[
        Path | None,
        Parameter(
            name=["-o", "--output"],
            help="Output file path. If not specified, prints to stdout.",
        ),
    ] = None,
) -> None:
    """Generate a starter configuration template for Kubernetes benchmarks.

    Outputs a commented YAML template with common configuration sections.
    Without --output, prints to stdout (suitable for piping).
    With --output, writes to a file.

    Examples:
        # Print template to stdout
        aiperf kube init

        # Pipe to a file
        aiperf kube init > benchmark.yaml

        # Write to a specific file
        aiperf kube init --output benchmark.yaml
    """
    from aiperf import cli_utils
    from aiperf.kubernetes import cli_helpers, init_template
    from aiperf.kubernetes import console as kube_console

    with cli_utils.exit_on_error(title="Error Generating Config Template"):
        filename = output.name if output else "benchmark.yaml"
        template = init_template.generate_init_template(filename)

        if output is None:
            print(template, end="")
            return

        if output.exists() and not await cli_helpers.confirm_action(
            f"File '{output}' already exists. Overwrite?"
        ):
            return

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(template)
        kube_console.print_success(f"Config template written to {output}")
        kube_console.print_info("Next steps:")
        kube_console.print_action(
            f"1. Edit {output} with your endpoint and model settings"
        )
        kube_console.print_action(
            f"2. Run: aiperf kube profile --config {output} --image <your-image>"
        )
