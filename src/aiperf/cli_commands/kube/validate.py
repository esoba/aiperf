# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube validate command: validate AIPerfJob YAML files against the CRD schema."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

app = App(name="validate")


@app.default
async def validate(
    files: Annotated[list[Path], Parameter(name="files")],
    *,
    strict: Annotated[bool, Parameter(name=["-s", "--strict"])] = False,
) -> None:
    """Validate AIPerfJob YAML files against the CRD schema and AIPerfConfig model.

    Performs comprehensive validation including:
    - YAML parsing and structure verification
    - Required fields: apiVersion, kind, metadata.name, spec (with AIPerfConfig fields)
    - Kubernetes resource name validation (RFC 1123)
    - AIPerfConfig validation via AIPerfJobSpecConverter
    - PodCustomization extraction validation
    - Worker count calculation (>= 1)
    - Unknown spec field detection (warning or error with --strict)

    Examples:
        aiperf kube validate aiperfjob.yaml
        aiperf kube validate recipes/llama/*.yaml recipes/qwen/*.yaml
        aiperf kube validate --strict aiperfjob.yaml

    Args:
        files: One or more AIPerfJob YAML file paths to validate.
        strict: Fail on warnings such as unknown spec fields.
    """
    from aiperf import cli_utils

    with cli_utils.exit_on_error(title="Error Running Validation"):
        from aiperf.kubernetes import validate as kube_validate

        passed, failed = await kube_validate.validate_files(files, strict=strict)

        if failed:
            raise SystemExit(1)
