# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protocol and base config for trace generator plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Protocol, runtime_checkable

from pydantic import Field

from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.models.base_models import AIPerfBaseModel


class TraceGeneratorConfig(AIPerfBaseModel):
    """Base configuration shared by all trace generators."""

    output_file: Annotated[
        Path | None,
        Field(
            default=None,
            description="Output JSONL file path (auto-generated from parameters if not specified).",
        ),
        CLIParameter(name=("--output-file",)),
    ]
    seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed for reproducibility. If not specified, uses non-deterministic entropy.",
        ),
        CLIParameter(name=("--seed",)),
    ]
    verbose: Annotated[
        bool,
        Field(
            default=False,
            description="Log detailed generation info at each time step.",
        ),
        CLIParameter(name=("--verbose",)),
    ]


@runtime_checkable
class TraceGeneratorProtocol(Protocol):
    """Protocol for trace generators that produce mooncake-style JSONL datasets.

    Implementations accept a ``TraceGeneratorConfig`` subclass and produce
    orjson-serialized JSONL lines via ``generate()``.
    """

    def __init__(self, config: TraceGeneratorConfig) -> None: ...

    def generate(self) -> list[bytes]:
        """Generate trace records as orjson-serialized JSONL lines.

        Returns:
            List of bytes, each element is a single JSON-serialized trace record.
        """
        ...

    def default_output_filename(self) -> str:
        """Generate a default output filename from the config parameters.

        Returns:
            Filename string ending in .jsonl.
        """
        ...
