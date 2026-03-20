# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared test configuration and fixtures for all test types.

ONLY ADD FIXTURES HERE THAT ARE USED IN ALL TEST TYPES.
DO NOT ADD FIXTURES THAT ARE ONLY USED IN A SPECIFIC TEST TYPE.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Path prefix -> markers to auto-enable (remove from default exclusions).
# When a user targets a path starting with the prefix, the listed markers are
# stripped from the ``-m 'not X and not Y ...'`` expression in addopts so the
# tests actually run instead of being silently deselected.
# Matching is bidirectional: targeting ``tests/kubernetes/gpu/vllm`` enables
# markers up the tree (k8s, gpu), and targeting ``tests/kubernetes`` enables
# all descendant markers (gpu, vllm, dynamo).
# Each entry only needs its own markers.
_PATH_MARKER_MAP: list[tuple[str, list[str]]] = [
    ("tests/kubernetes/gpu/vllm", ["vllm"]),
    ("tests/kubernetes/gpu/dynamo", ["dynamo"]),
    ("tests/kubernetes/gpu", ["gpu"]),
    ("tests/kubernetes", ["k8s"]),
    ("tests/integration", ["integration"]),
    ("tests/component_integration", ["component_integration"]),
]


def pytest_configure(config: pytest.Config) -> None:
    """Auto-enable markers when the user targets a specific test path.

    ``addopts`` in pyproject.toml excludes heavy test suites by default via
    ``-m 'not k8s and not gpu and ...'``.  When the user explicitly runs
    ``pytest tests/kubernetes/`` (or any other excluded path), this hook
    detects the target and strips the corresponding ``not <marker>`` clauses
    so the tests are collected instead of silently skipped.
    """
    markexpr = getattr(config.option, "markexpr", "") or ""
    if not markexpr:
        return

    raw_args = [str(a) for a in config.invocation_params.args]
    if not raw_args:
        return

    # Normalize args to project-relative paths (handles absolute paths too)
    rootpath = config.invocation_params.dir
    rel_args: list[str] = []
    for arg in raw_args:
        # Strip ::TestClass::test_method node ids for path matching
        path_part = arg.split("::")[0]
        try:
            rel_args.append(str(Path(path_part).resolve().relative_to(rootpath)))
        except (ValueError, OSError):
            rel_args.append(path_part)

    # Collect all markers to enable based on targeted paths (bidirectional).
    # "tests/kubernetes/gpu/vllm/test_foo.py" starts with "tests/kubernetes"
    # so the k8s marker is enabled.  "tests/kubernetes" is a prefix of
    # "tests/kubernetes/gpu", so gpu is also enabled.
    enable: set[str] = set()
    for path_prefix, markers in _PATH_MARKER_MAP:
        if any(
            a.startswith(path_prefix) or path_prefix.startswith(a) for a in rel_args
        ):
            enable.update(markers)

    if not enable:
        return

    # Strip matching 'not <marker>' clauses from the expression
    exclude = {f"not {m}" for m in enable}
    parts = [p for p in re.split(r"\s+and\s+", markexpr) if p.strip() not in exclude]
    config.option.markexpr = " and ".join(parts) if parts else ""
