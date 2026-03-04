# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load SessionDistributionConfig from file paths."""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.dataset.claude_code_gen.models import SessionDistributionConfig


def load_config(path_or_name: str) -> SessionDistributionConfig:
    """Load a config from a file path.

    Supports both raw config JSON and manifest.json (has generation_params wrapper).

    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    path = Path(path_or_name)
    if path.is_file():
        data = orjson.loads(path.read_bytes())
        if "generation_params" in data:
            data = data["generation_params"]
        return SessionDistributionConfig(**data)

    raise FileNotFoundError(f"Config '{path_or_name}' not found as file.")
