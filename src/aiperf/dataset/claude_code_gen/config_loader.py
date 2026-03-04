# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load SessionDistributionConfig from bundled names or file paths."""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.dataset.claude_code_gen.models import SessionDistributionConfig

_CONFIGS_DIR = Path(__file__).parent / "configs"


def list_bundled_configs() -> list[str]:
    """Return names of bundled config files (without .json extension)."""
    return sorted(p.stem for p in _CONFIGS_DIR.glob("*.json"))


def load_config(path_or_name: str) -> SessionDistributionConfig:
    """Load a config from a bundled name or file path.

    Resolution order:
    1. If *path_or_name* matches a bundled config name, load it.
    2. If it is a path to an existing file, load it.
    3. Otherwise raise FileNotFoundError.

    Supports both raw config JSON and manifest.json (has generation_params wrapper).
    """
    bundled = _CONFIGS_DIR / f"{path_or_name}.json"
    if bundled.is_file():
        return _load_json(bundled)

    path = Path(path_or_name)
    if path.is_file():
        return _load_json(path)

    available = list_bundled_configs()
    raise FileNotFoundError(
        f"Config '{path_or_name}' not found. "
        f"Bundled configs: {available}. Or provide a file path."
    )


def _load_json(path: Path) -> SessionDistributionConfig:
    data = orjson.loads(path.read_bytes())
    if "generation_params" in data:
        data = data["generation_params"]
    return SessionDistributionConfig(**data)
