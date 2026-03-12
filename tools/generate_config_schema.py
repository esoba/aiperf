#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate JSON Schema for AIPerf YAML configuration files.

Produces the JSON Schema from the AIPerfConfig Pydantic model, then
post-processes it to accept the shorthand input forms that
AIPerfConfig.normalize_before_validation() handles at runtime:

  - models: string | list[string] | ModelsAdvanced
  - model (singular alias for models)
  - dataset (singular alias for datasets)
  - load: single phase config (with 'type' key) in addition to named phases
  - endpoint.url (singular alias for endpoint.urls)
  - MeanStddev: int/float shorthand (e.g. isl: 512)

Usage:
    ./tools/generate_config_schema.py
    ./tools/generate_config_schema.py --check
    ./tools/generate_config_schema.py --verbose
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

# Allow direct execution: add repo root to path for 'tools' package imports
if __name__ == "__main__" and "tools" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Any

from tools._core import (
    GeneratedFile,
    Generator,
    GeneratorResult,
    main,
    print_step,
)

# =============================================================================
# Configuration
# =============================================================================

SCHEMA_FILE = Path("src/aiperf/config/schema/aiperf-config.schema.json")

SCHEMA_ID = "https://nvidia.github.io/aiperf/schemas/aiperf-config.schema.json"
JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"


# =============================================================================
# Schema Post-Processing
# =============================================================================


def _patch_models_shorthand(schema: dict[str, Any]) -> None:
    """Allow models to accept string, list[string], or ModelsAdvanced.

    The Pydantic schema only knows about ModelsAdvanced (the post-validation
    type). At runtime, normalize_before_validation() converts:
      - "model-name" → ModelsAdvanced
      - ["model-a", "model-b"] → ModelsAdvanced
    """
    props = schema.get("properties", {})
    if "models" not in props:
        return

    models_ref = props["models"]
    desc = models_ref.get("description", "")

    # Replace $ref with anyOf that accepts all input forms
    props["models"] = {
        "anyOf": [
            # String shorthand: "model-name"
            {"type": "string"},
            # List of strings: ["model-a", "model-b"]
            {"type": "array", "items": {"type": "string"}},
            # Full ModelsAdvanced object
            {"$ref": "#/$defs/ModelsAdvanced"},
        ],
        "description": desc,
    }


def _add_singular_aliases(schema: dict[str, Any]) -> None:
    """Add 'model' and 'dataset' as accepted aliases.

    At runtime, normalize_before_validation() converts:
      - model → models
      - dataset → datasets (wrapped as {"default": dataset_config})
    """
    props = schema.get("properties", {})

    # model (singular) → same schema as models
    if "models" in props:
        props["model"] = {
            **props["models"],
            "description": "Shorthand for 'models' (singular model name).",
        }

    # dataset (singular) → accepts a single dataset config
    if "datasets" in props:
        datasets_schema = props["datasets"]
        # Get the dataset union type from additionalProperties
        ap = datasets_schema.get("additionalProperties", {})
        if ap:
            props["dataset"] = {
                **ap,
                "description": "Shorthand for 'datasets'. "
                "Becomes the 'default' dataset.",
            }

    # Relax required: models/datasets not required when model/dataset present
    # JSON Schema can't express "require models OR model", so just make
    # none of the aliased fields strictly required in the schema.
    # Runtime validation handles the actual requirement.
    if "required" in schema:
        schema["required"] = [
            r for r in schema["required"] if r not in ("models", "datasets")
        ]


def _patch_load_shorthand(schema: dict[str, Any]) -> None:
    """Allow load to accept a single phase config (with 'type' key).

    At runtime, normalize_before_validation() converts:
      load: {type: concurrency, ...} → load: {default: {type: concurrency, ...}}
    """
    props = schema.get("properties", {})
    if "load" not in props:
        return

    load_schema = props["load"]
    desc = load_schema.get("description", "")

    # Keep the existing dict-of-phases schema and add single-phase as alternative
    props["load"] = {
        "anyOf": [
            # Single phase config (has 'type' key) — shorthand
            {"$ref": "#/$defs/PhaseConfig"},
            # Named phases dict — canonical form
            load_schema,
        ],
        "description": desc,
    }


# Duration string schema: matches _DURATION_PATTERN from phases.py
# Accepts: "30s", "5m", "2h", "30 sec", "5.5 min", "2 hour", or bare "300"
_DURATION_STRING_SCHEMA: dict[str, Any] = {
    "type": "string",
    "pattern": r"^\d+(?:\.\d+)?\s*(?:s|sec|m|min|h|hr|hour)?$",
}


def _patch_duration_fields(schema: dict[str, Any]) -> None:
    """Allow duration fields to accept string shorthand (e.g. '30s', '5m', '2h').

    At runtime, _parse_duration() converts strings like '30s' to float seconds.
    Fields: PhaseConfig.duration, PhaseConfig.grace_period,
    ArtifactsConfig.slice_duration, and ramp fields.
    """
    defs = schema.get("$defs", {})

    # PhaseConfig: duration, grace_period
    phase = defs.get("PhaseConfig", {})
    for field_name in ("duration", "grace_period"):
        _add_duration_string_to_anyof(phase.get("properties", {}), field_name)

    # ArtifactsConfig: slice_duration
    artifacts = defs.get("ArtifactsConfig", {})
    _add_duration_string_to_anyof(artifacts.get("properties", {}), "slice_duration")

    # RampConfig: duration (ramp accepts string too)
    ramp = defs.get("RampConfig", {})
    _add_duration_string_to_anyof(ramp.get("properties", {}), "duration")

    # PhaseConfig: rate_ramp, concurrency_ramp, prefill_ramp accept bare string/number shorthand
    for field_name in ("rate_ramp", "concurrency_ramp", "prefill_ramp"):
        prop = phase.get("properties", {}).get(field_name)
        if not prop or "anyOf" not in prop:
            continue
        any_of = prop["anyOf"]
        has_duration = any(a.get("pattern") for a in any_of)
        has_number = any(a.get("type") == "number" for a in any_of)
        if not has_duration:
            any_of.insert(0, {**_DURATION_STRING_SCHEMA})
        if not has_number:
            any_of.insert(0, {"type": "number"})


def _add_duration_string_to_anyof(props: dict[str, Any], field_name: str) -> None:
    """Add duration string type to an anyOf field that currently only accepts number|null."""
    prop = props.get(field_name)
    if not prop:
        return

    if "anyOf" in prop:
        any_of = prop["anyOf"]
        if not any(a.get("pattern") for a in any_of):
            # Insert duration string before null
            null_idx = next(
                (i for i, a in enumerate(any_of) if a.get("type") == "null"),
                len(any_of),
            )
            any_of.insert(null_idx, {**_DURATION_STRING_SCHEMA})
    elif prop.get("type") == "number":
        desc = prop.pop("description", None)
        default = prop.pop("default", None)
        props[field_name] = {"anyOf": [{"type": "number"}, {**_DURATION_STRING_SCHEMA}]}
        if desc:
            props[field_name]["description"] = desc
        if default is not None:
            props[field_name]["default"] = default


def _patch_mean_stddev_shorthand(schema: dict[str, Any]) -> None:
    """Allow MeanStddev to accept int/float as shorthand.

    At runtime, MeanStddev.coerce_scalar_to_distribution() converts:
      512 → {mean: 512.0, stddev: 0.0}
    """
    defs = schema.get("$defs", {})
    ms = defs.get("MeanStddev")
    if not ms:
        return

    # Replace the object-only schema with anyOf that also accepts scalars
    original = {k: v for k, v in ms.items()}
    ms.clear()
    ms["anyOf"] = [
        {"type": "number"},
        {"type": "integer"},
        {k: v for k, v in original.items() if k not in ("title",)},
    ]
    if "title" in original:
        ms["title"] = original["title"]
    if "description" in original:
        ms["description"] = original["description"]


def _patch_endpoint_url_shorthand(schema: dict[str, Any]) -> None:
    """Allow EndpointConfig to accept 'url' (singular) as alias for 'urls'.

    At runtime, EndpointConfig.normalize_before_validation() converts:
      url: "http://..." → urls: ["http://..."]
    """
    defs = schema.get("$defs", {})
    endpoint = defs.get("EndpointConfig")
    if not endpoint:
        return

    props = endpoint.get("properties", {})
    if "urls" not in props:
        return

    # Add 'url' as a string alias
    props["url"] = {
        "type": "string",
        "description": "Shorthand for 'urls' (single endpoint URL).",
    }

    # urls is required in the Pydantic model, but not when url is used
    if "required" in endpoint:
        endpoint["required"] = [r for r in endpoint["required"] if r != "urls"]


def postprocess_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Apply all post-processing patches to accept shorthand input forms."""
    _patch_models_shorthand(schema)
    _add_singular_aliases(schema)
    _patch_load_shorthand(schema)
    _patch_endpoint_url_shorthand(schema)
    _patch_duration_fields(schema)
    _patch_mean_stddev_shorthand(schema)
    return schema


# =============================================================================
# Generator
# =============================================================================


class ConfigSchemaGenerator(Generator):
    """Generate JSON Schema for AIPerf YAML configuration."""

    name = "Config Schema"
    description = "Generate JSON Schema from AIPerfConfig Pydantic model"

    def generate(self) -> GeneratorResult:
        sys.path.insert(0, "src")
        from aiperf.config.config import AIPerfConfig

        # Generate base schema from Pydantic
        schema = AIPerfConfig.model_json_schema()
        if self.verbose:
            defs = schema.get("$defs", {})
            props = schema.get("properties", {})
            print_step(f"Base schema: {len(defs)} definitions, {len(props)} properties")

        # Post-process for shorthand input forms
        schema = postprocess_schema(schema)
        if self.verbose:
            print_step(
                "Applied shorthand patches (models, model/dataset aliases, load)"
            )

        # Ensure $schema and $id are first in output
        ordered: dict[str, Any] = OrderedDict()
        ordered["$schema"] = JSON_SCHEMA_DRAFT
        ordered["$id"] = SCHEMA_ID
        for k, v in schema.items():
            ordered[k] = v

        content = json.dumps(ordered, indent=2) + "\n"

        return GeneratorResult(
            files=[GeneratedFile(SCHEMA_FILE, content)],
            summary=f"schema with {len(schema.get('$defs', {}))} definitions",
        )


if __name__ == "__main__":
    main(ConfigSchemaGenerator)
