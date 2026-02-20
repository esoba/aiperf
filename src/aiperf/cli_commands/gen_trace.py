# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for generating synthetic trace datasets.

Subcommands are dynamically registered from ``trace_generator`` plugins.
Only lightweight Pydantic config models are imported at startup; the actual
generator classes (with heavy deps like numpy/orjson/tqdm) are lazy-loaded
when a subcommand is invoked.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path

from cyclopts import App

from aiperf.common import random_generator as rng
from aiperf.dataset.synthesis.trace_generator import TraceGeneratorConfig
from aiperf.plugin import plugins
from aiperf.plugin.types import PluginEntry

gen_trace_app = App(
    name="gen-trace",
    help="Generate synthetic trace datasets in mooncake-style JSONL format",
)


def _import_class(class_path: str) -> type:
    """Import a class from a 'module.path:ClassName' string."""
    module_path, _, class_name = class_path.rpartition(":")
    return getattr(importlib.import_module(module_path), class_name)


def _make_handler(
    entry: PluginEntry, config_class: type[TraceGeneratorConfig]
) -> Callable[..., None]:
    """Build a CLI handler that lazy-loads the generator on invocation."""

    def handler(config: TraceGeneratorConfig) -> None:
        if config.verbose:
            logging.basicConfig(level=logging.INFO)

        rng.reset()
        rng.init(config.seed)

        generator_class = entry.load()
        generator = generator_class(config)

        output_file = config.output_file
        if output_file is None:
            output_file = Path(generator.default_output_filename())

        output_data = generator.generate()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "wb") as f:
            for line in output_data:
                f.write(line + b"\n")

        print(f"Generated {len(output_data):,} requests → {output_file}")

    handler.__annotations__["config"] = config_class
    handler.__doc__ = config_class.__doc__
    return handler


for _entry in plugins.iter_entries("trace_generator"):
    _config_path = _entry.metadata.get("config_class")
    if _config_path is None:
        continue
    _config_class = _import_class(_config_path)
    _sub_app = App(name=_entry.name, help=_entry.description.strip())
    _sub_app.default(_make_handler(_entry, _config_class))
    gen_trace_app.command(_sub_app)
