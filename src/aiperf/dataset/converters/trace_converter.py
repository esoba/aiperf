# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protocol and base config for trace converter plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Protocol, runtime_checkable

from pydantic import Field

from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.models.base_models import AIPerfBaseModel


class TraceConverterConfig(AIPerfBaseModel):
    """Base configuration shared by all trace converters."""

    input_file: Annotated[
        Path,
        Field(
            ...,
            description="Path to the input trace file.",
        ),
        CLIParameter(name=("--input-file", "-i")),
    ]
    output_file: Annotated[
        Path | None,
        Field(
            default=None,
            description="Output JSONL file path (auto-generated from input filename if not specified).",
        ),
        CLIParameter(name=("--output-file", "-o")),
    ]
    verbose: Annotated[
        bool,
        Field(
            default=False,
            description="Print detailed conversion statistics.",
        ),
        CLIParameter(name=("--verbose", "-v")),
    ]


@runtime_checkable
class TraceConverterProtocol(Protocol):
    """Protocol for trace converters that transform external formats to mooncake JSONL.

    Implementations accept a ``TraceConverterConfig`` subclass and produce
    a list of dicts representing mooncake trace records via ``convert()``.
    """

    def __init__(self, config: TraceConverterConfig) -> None: ...

    def convert(self) -> list[dict[str, Any]]:
        """Convert input trace data to mooncake-format records.

        Returns:
            List of dicts, each representing a single trace record.
        """
        ...

    def default_output_filename(self) -> str:
        """Generate a default output filename from the input filename.

        Returns:
            Filename string ending in .jsonl.
        """
        ...
