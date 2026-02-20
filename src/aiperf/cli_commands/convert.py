# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for converting external trace formats to mooncake JSONL.

Subcommands are dynamically registered from ``trace_converter`` plugins.
Only lightweight Pydantic config models are imported at startup; the actual
converter classes (with heavy deps like pandas/tokenizers/tqdm) are lazy-loaded
when a subcommand is invoked.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cyclopts import App

from aiperf.dataset.converters.trace_converter import TraceConverterConfig
from aiperf.plugin import plugins
from aiperf.plugin.types import PluginEntry

convert_app = App(
    name="convert-trace",
    help="Convert external trace formats to mooncake-style JSONL for use with aiperf profile --input-file",
)


def _import_class(class_path: str) -> type:
    """Import a class from a 'module.path:ClassName' string."""
    module_path, _, class_name = class_path.rpartition(":")
    return getattr(importlib.import_module(module_path), class_name)


def _make_handler(
    entry: PluginEntry, config_class: type[TraceConverterConfig]
) -> Callable[..., None]:
    """Build a CLI handler that lazy-loads the converter on invocation."""

    def handler(config: TraceConverterConfig) -> None:
        converter_class = entry.load()
        converter = converter_class(config)

        output_file = config.output_file
        if output_file is None:
            output_file = Path(converter.default_output_filename())

        records: list[dict[str, Any]] = converter.convert()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        print(f"Converted {len(records):,} records -> {output_file}")

    handler.__annotations__["config"] = config_class
    handler.__doc__ = config_class.__doc__
    return handler


for _entry in plugins.iter_entries("trace_converter"):
    _config_path = _entry.metadata.get("config_class")
    if _config_path is None:
        continue
    _config_class = _import_class(_config_path)
    _sub_app = App(name=_entry.name, help=_entry.description.strip())
    _sub_app.default(_make_handler(_entry, _config_class))
    convert_app.command(_sub_app)
