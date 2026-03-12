#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Kubernetes CRD schema from AIPerfConfig Pydantic model.

Introspects the AIPerfConfig model to produce a complete CRD YAML that
stays in sync with the Python configuration schema. Operator-specific
fields (image, podTemplate, scheduling, etc.) and the status sub-schema
are defined statically.

Usage:
    ./tools/generate_crd.py
    ./tools/generate_crd.py --check
    ./tools/generate_crd.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution: add repo root to path for 'tools' package imports
if __name__ == "__main__" and "tools" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Any

import yaml

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

CRD_FILE = Path("deploy/crd.yaml")
HELM_CRD_FILE = Path("deploy/helm/aiperf-operator/templates/crd.yaml")

SPDX_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
    "# SPDX-License-Identifier: Apache-2.0",
)

# Keys to strip from JSON Schema that K8s CRDs don't support
_STRIP_KEYS = frozenset({"title", "examples", "$defs", "$schema"})

# Maximum recursion depth before falling back to preserve-unknown-fields.
# Keeps the CRD from exploding for deeply nested models.
_MAX_DEPTH = 6


# =============================================================================
# JSON Schema -> K8s OpenAPI v3 Converter
# =============================================================================


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a $ref string to its definition."""
    # "#/$defs/FooBar" -> "FooBar"
    name = ref.rsplit("/", 1)[-1]
    if name not in defs:
        return {}
    return defs[name]


def _is_nullable_anyof(schema: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    """Check if schema is anyOf: [{real_type}, {type: null}].

    Returns (is_nullable, real_type_schema) if it matches the pattern.
    """
    any_of = schema.get("anyOf")
    if not any_of or len(any_of) != 2:
        return False, None

    null_idx = None
    for i, item in enumerate(any_of):
        if item.get("type") == "null":
            null_idx = i

    if null_idx is None:
        return False, None

    real_schema = any_of[1 - null_idx]
    return True, real_schema


def _convert_schema(
    schema: dict[str, Any],
    defs: dict[str, Any],
    depth: int = 0,
) -> dict[str, Any]:
    """Convert a JSON Schema node to K8s-compatible OpenAPI v3.

    Recursively resolves $ref, handles anyOf-with-null (nullable),
    converts discriminated unions, and strips unsupported keys.
    Falls back to x-kubernetes-preserve-unknown-fields at max depth.
    """
    if not schema:
        return {}

    # Resolve $ref first
    if "$ref" in schema:
        resolved = _resolve_ref(schema["$ref"], defs)
        # Carry over description from the referencing schema
        merged = _convert_schema(resolved, defs, depth)
        if "description" in schema and schema["description"]:
            merged["description"] = schema["description"]
        return merged

    # Depth guard: fall back to flexible schema for deeply nested types
    if depth > _MAX_DEPTH:
        result: dict[str, Any] = {
            "type": "object",
            "x-kubernetes-preserve-unknown-fields": True,
        }
        if "description" in schema:
            result["description"] = schema["description"]
        return result

    # Handle anyOf: [{type: X}, {type: null}] -> just the non-null type
    is_nullable, real_type = _is_nullable_anyof(schema)
    if is_nullable and real_type:
        result = _convert_schema(real_type, defs, depth)
        # Carry over description from wrapper (skip None defaults)
        if "description" in schema and "description" not in result:
            result["description"] = schema["description"]
        if (
            "default" in schema
            and schema["default"] is not None
            and "default" not in result
        ):
            result["default"] = schema["default"]
        return result

    # Handle anyOf with non-null alternatives (complex unions)
    if "anyOf" in schema and not is_nullable:
        any_of = schema["anyOf"]
        # If all alternatives are simple scalar types, merge them
        scalar_types = []
        for alt in any_of:
            if "type" in alt and alt["type"] not in ("object", "array"):
                scalar_types.append(alt["type"])
            elif "const" in alt:
                scalar_types.append(type(alt["const"]).__name__)
        if scalar_types and len(scalar_types) == len(any_of):
            # Multiple scalar types - use the first non-null one
            result = _convert_schema(any_of[0], defs, depth)
            for key in ("default", "description"):
                if key in schema:
                    result[key] = schema[key]
            return result

        # Complex union -> preserve unknown fields
        result = {"x-kubernetes-preserve-unknown-fields": True}
        if "description" in schema:
            result["description"] = schema["description"]
        if "default" in schema:
            result["default"] = schema["default"]
        return result

    # Handle oneOf (discriminated unions) -> preserve unknown fields
    if "oneOf" in schema:
        result = {"x-kubernetes-preserve-unknown-fields": True}
        if "description" in schema:
            result["description"] = schema["description"]
        return result

    # Handle discriminator in additionalProperties (e.g. datasets)
    ap = schema.get("additionalProperties", {})
    if isinstance(ap, dict) and "discriminator" in ap:
        result = {"type": "object", "x-kubernetes-preserve-unknown-fields": True}
        if "description" in schema:
            result["description"] = schema["description"]
        return result

    # Build the result node
    result = {}

    # Copy type
    if "type" in schema:
        result["type"] = schema["type"]

    # Description
    if "description" in schema:
        # Use first line/paragraph for CRD description
        desc = schema["description"]
        result["description"] = desc

    # Enum values
    if "enum" in schema:
        result["enum"] = schema["enum"]

    # const -> enum with single value
    if "const" in schema:
        result["enum"] = [schema["const"]]
        if "type" not in result:
            val = schema["const"]
            if isinstance(val, str):
                result["type"] = "string"
            elif isinstance(val, bool):
                result["type"] = "boolean"
            elif isinstance(val, int):
                result["type"] = "integer"

    # Default value (skip None defaults - they render as empty YAML and aren't useful in CRDs)
    if "default" in schema and schema["default"] is not None:
        result["default"] = schema["default"]

    # Numeric constraints
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema:
            result[key] = schema[key]

    # String constraints
    for key in ("minLength", "maxLength", "pattern"):
        if key in schema:
            result[key] = schema[key]

    # Format
    if "format" in schema and schema["format"] != "path":
        result["format"] = schema["format"]

    # Object properties
    if schema.get("type") == "object" or "properties" in schema:
        result["type"] = "object"

        if "properties" in schema:
            props = {}
            for prop_name, prop_schema in schema["properties"].items():
                props[prop_name] = _convert_schema(prop_schema, defs, depth + 1)
            if props:
                result["properties"] = props

        if "required" in schema:
            result["required"] = schema["required"]

        # additionalProperties
        if "additionalProperties" in schema:
            ap = schema["additionalProperties"]
            if isinstance(ap, bool):
                result["additionalProperties"] = ap
            elif isinstance(ap, dict):
                if "$ref" in ap or "type" in ap:
                    converted = _convert_schema(ap, defs, depth + 1)
                    if converted:
                        result["additionalProperties"] = converted
                elif "discriminator" in ap:
                    result["x-kubernetes-preserve-unknown-fields"] = True
                else:
                    result["additionalProperties"] = _convert_schema(
                        ap, defs, depth + 1
                    )

        # If it has additionalProperties: false from Pydantic, don't include it
        # in the CRD (K8s CRD handles this differently)
        if schema.get("additionalProperties") is False:
            result.pop("additionalProperties", None)

    # Array items
    if schema.get("type") == "array" and "items" in schema:
        result["items"] = _convert_schema(schema["items"], defs, depth + 1)

    # minItems / maxItems
    for key in ("minItems", "maxItems"):
        if key in schema:
            result[key] = schema[key]

    # Strip unsupported keys
    for key in _STRIP_KEYS:
        result.pop(key, None)

    return result


def convert_aiperf_config_fields(
    schema: dict[str, Any], verbose: bool = False
) -> dict[str, Any]:
    """Convert AIPerfConfig's JSON Schema properties to K8s CRD spec properties.

    Returns a dict of property_name -> K8s OpenAPI v3 schema.
    """
    defs = schema.get("$defs", {})
    properties = schema.get("properties", {})

    result = {}
    for name, prop_schema in properties.items():
        converted = _convert_schema(prop_schema, defs, depth=0)
        if verbose:
            print_step(f"Converted field: {name}")
        result[name] = converted

    return result


# =============================================================================
# Static Schema Definitions (operator-specific)
# =============================================================================


def _operator_fields() -> dict[str, Any]:
    """Return operator-specific fields that are not part of AIPerfConfig."""
    return {
        "image": {
            "type": "string",
            "description": "AIPerf container image",
            "default": "nvcr.io/nvidia/aiperf:latest",
        },
        "imagePullPolicy": {
            "type": "string",
            "description": "Image pull policy (Always, Never, IfNotPresent)",
            "enum": ["Always", "Never", "IfNotPresent"],
        },
        "connectionsPerWorker": {
            "type": "integer",
            "description": "Connections per worker for auto-scaling calculation",
            "minimum": 1,
            "default": 500,
        },
        "timeoutSeconds": {
            "type": "number",
            "description": "Job timeout in seconds (0 = no timeout, overrides AIPERF_JOB_TIMEOUT_SECONDS)",
            "minimum": 0,
            "default": 0,
        },
        "ttlSecondsAfterFinished": {
            "type": "integer",
            "description": "TTL in seconds for JobSet cleanup after completion",
            "minimum": 0,
        },
        "resultsTtlDays": {
            "type": "integer",
            "description": "Days to retain result files before cleanup",
            "minimum": 1,
        },
        "cancel": {
            "type": "boolean",
            "description": "Set to true to cancel a running benchmark",
        },
        "podTemplate": {
            "type": "object",
            "properties": {
                "nodeSelector": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Node selector labels",
                },
                "tolerations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "operator": {
                                "type": "string",
                                "enum": ["Exists", "Equal"],
                            },
                            "value": {"type": "string"},
                            "effect": {
                                "type": "string",
                                "enum": [
                                    "NoSchedule",
                                    "PreferNoSchedule",
                                    "NoExecute",
                                ],
                            },
                            "tolerationSeconds": {"type": "integer"},
                        },
                    },
                    "description": "Pod tolerations",
                },
                "annotations": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Additional pod annotations",
                },
                "labels": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Additional pod labels",
                },
                "imagePullSecrets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Image pull secret names",
                },
                "serviceAccountName": {
                    "type": "string",
                    "description": "Service account name",
                },
                "env": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "string"},
                            "valueFrom": {
                                "type": "object",
                                "properties": {
                                    "secretKeyRef": {
                                        "type": "object",
                                        "required": ["name", "key"],
                                        "properties": {
                                            "name": {"type": "string"},
                                            "key": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "description": "Environment variables",
                },
                "volumes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "x-kubernetes-preserve-unknown-fields": True,
                    },
                    "description": "Additional volumes",
                },
                "volumeMounts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "mountPath"],
                        "properties": {
                            "name": {"type": "string"},
                            "mountPath": {"type": "string"},
                            "subPath": {"type": "string"},
                            "readOnly": {"type": "boolean"},
                        },
                    },
                    "description": "Volume mounts for containers",
                },
            },
        },
        "scheduling": {
            "type": "object",
            "properties": {
                "queueName": {
                    "type": "string",
                    "description": "Kueue LocalQueue name for gang-scheduling admission",
                },
                "priorityClass": {
                    "type": "string",
                    "description": "Kueue WorkloadPriorityClass for scheduling priority",
                },
            },
            "description": "Kueue scheduling configuration for gang-scheduling",
        },
    }


def _status_schema() -> dict[str, Any]:
    """Return the status sub-schema."""
    return {
        "type": "object",
        "properties": {
            "observedGeneration": {
                "type": "integer",
                "format": "int64",
                "description": "Generation of the spec that was last processed",
            },
            "phase": {
                "type": "string",
                "description": "Current job phase",
                "enum": [
                    "Pending",
                    "Queued",
                    "Initializing",
                    "Running",
                    "Completed",
                    "Failed",
                    "Cancelled",
                ],
            },
            "jobId": {
                "type": "string",
                "description": "Unique job identifier",
            },
            "startTime": {
                "type": "string",
                "format": "date-time",
                "description": "Time when job started",
            },
            "completionTime": {
                "type": "string",
                "format": "date-time",
                "description": "Time when job completed",
            },
            "jobSetName": {
                "type": "string",
                "description": "Name of the managed JobSet",
            },
            "error": {
                "type": "string",
                "description": "Error message if failed",
            },
            "workers": {
                "type": "object",
                "properties": {
                    "ready": {
                        "type": "integer",
                        "description": "Number of ready workers",
                    },
                    "total": {
                        "type": "integer",
                        "description": "Total number of workers",
                    },
                },
            },
            "phases": {
                "type": "object",
                "description": "Progress tracking for each benchmark phase (keyed by phase name)",
                "additionalProperties": {
                    "type": "object",
                    "description": "Phase progress stats",
                    "x-kubernetes-preserve-unknown-fields": True,
                },
            },
            "currentPhase": {
                "type": "string",
                "description": "Current benchmark phase (warmup, profiling, etc)",
            },
            "liveMetrics": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Live metrics updated during benchmark run",
            },
            "serverMetrics": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Server-side metrics from inference server Prometheus endpoint",
            },
            "results": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Final benchmark results and metrics",
            },
            "resultsPath": {
                "type": "string",
                "description": "Path to stored results on operator PVC",
            },
            "liveSummary": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Live summary metrics updated during benchmark run",
            },
            "summary": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Final summary metrics after benchmark completion",
            },
            "resultsTtlDays": {
                "type": "integer",
                "description": "Days to retain result files before cleanup",
            },
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["True", "False", "Unknown"],
                        },
                        "reason": {"type": "string"},
                        "message": {"type": "string"},
                        "lastTransitionTime": {
                            "type": "string",
                            "format": "date-time",
                        },
                    },
                },
                "description": "Detailed status conditions",
            },
        },
    }


def _printer_columns() -> list[dict[str, Any]]:
    """Return additionalPrinterColumns for kubectl output."""
    return [
        {
            "name": "Phase",
            "type": "string",
            "jsonPath": ".status.phase",
        },
        {
            "name": "Stage",
            "type": "string",
            "jsonPath": ".status.currentPhase",
            "description": "Current benchmark stage (warmup, profiling)",
        },
        {
            "name": "Progress",
            "type": "string",
            "jsonPath": ".status.phases.profiling.requestsCompleted",
            "description": "Requests completed in profiling phase",
        },
        {
            "name": "QPS",
            "type": "number",
            "jsonPath": ".status.phases.profiling.requestsPerSecond",
            "description": "Requests per second in profiling phase",
        },
        {
            "name": "Age",
            "type": "date",
            "jsonPath": ".metadata.creationTimestamp",
        },
    ]


# =============================================================================
# CRD Assembly
# =============================================================================


def _build_crd(config_properties: dict[str, Any]) -> dict[str, Any]:
    """Assemble the full CRD document."""
    # Merge config fields with operator-specific fields
    spec_properties: dict[str, Any] = {}

    # Operator fields first (image, imagePullPolicy)
    operator = _operator_fields()
    spec_properties["image"] = operator.pop("image")
    spec_properties["imagePullPolicy"] = operator.pop("imagePullPolicy")

    # AIPerfConfig fields
    spec_properties.update(config_properties)

    # Remaining operator fields (connectionsPerWorker, timeoutSeconds, etc.)
    spec_properties.update(operator)

    return {
        "apiVersion": "apiextensions.k8s.io/v1",
        "kind": "CustomResourceDefinition",
        "metadata": {
            "name": "aiperfjobs.aiperf.nvidia.com",
        },
        "spec": {
            "group": "aiperf.nvidia.com",
            "names": {
                "kind": "AIPerfJob",
                "listKind": "AIPerfJobList",
                "plural": "aiperfjobs",
                "singular": "aiperfjob",
                "shortNames": ["apj", "aiperf"],
            },
            "scope": "Namespaced",
            "versions": [
                {
                    "name": "v1alpha1",
                    "served": True,
                    "storage": True,
                    "additionalPrinterColumns": _printer_columns(),
                    "subresources": {"status": {}},
                    "schema": {
                        "openAPIV3Schema": {
                            "type": "object",
                            "required": ["spec"],
                            "properties": {
                                "spec": {
                                    "type": "object",
                                    "properties": spec_properties,
                                },
                                "status": _status_schema(),
                            },
                        },
                    },
                },
            ],
        },
    }


# =============================================================================
# YAML Rendering
# =============================================================================


class _CRDDumper(yaml.SafeDumper):
    """Custom YAML dumper for CRD output."""


def _str_representer(dumper: yaml.SafeDumper, data: str) -> Any:
    """Use literal block style for multi-line strings."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def _bool_representer(dumper: yaml.SafeDumper, data: bool) -> Any:
    """Represent bools as true/false (not True/False)."""
    return dumper.represent_scalar(
        "tag:yaml.org,2002:bool", "true" if data else "false"
    )


def _none_representer(dumper: yaml.SafeDumper, data: None) -> Any:
    """Represent None as empty mapping for status: {}."""
    return dumper.represent_scalar("tag:yaml.org,2002:null", "")


_CRDDumper.add_representer(str, _str_representer)
_CRDDumper.add_representer(bool, _bool_representer)
_CRDDumper.add_representer(type(None), _none_representer)


def render_crd_yaml(crd: dict[str, Any]) -> str:
    """Render the CRD dict as YAML with SPDX header."""
    lines = list(SPDX_HEADER) + ["---"]
    yaml_str = yaml.dump(
        crd,
        Dumper=_CRDDumper,
        default_flow_style=False,
        sort_keys=False,
        width=120,
        allow_unicode=True,
    )
    lines.append(yaml_str.rstrip())
    return "\n".join(lines) + "\n"


def render_helm_crd_yaml(crd: dict[str, Any]) -> str:
    """Render the Helm-templated CRD variant.

    Wraps with {{- if .Values.crd.install }}, adds Helm labels,
    and templates the default image value.
    """
    # Deep copy and modify for Helm
    import copy

    helm_crd = copy.deepcopy(crd)

    # The Helm version uses a template expression for labels and image default.
    # We render the base YAML and then do string replacements for template expressions.
    yaml_str = yaml.dump(
        helm_crd,
        Dumper=_CRDDumper,
        default_flow_style=False,
        sort_keys=False,
        width=120,
        allow_unicode=True,
    )

    # Replace the static image default with Helm template
    yaml_str = yaml_str.replace(
        "default: nvcr.io/nvidia/aiperf:latest",
        "default: {{ .Values.defaults.image | quote }}",
    )

    # Add Helm labels to metadata
    yaml_str = yaml_str.replace(
        "  name: aiperfjobs.aiperf.nvidia.com\n",
        "  name: aiperfjobs.aiperf.nvidia.com\n"
        "  labels:\n"
        '    {{- include "aiperf-operator.labels" . | nindent 4 }}\n',
    )

    lines = list(SPDX_HEADER)
    lines.append("{{- if .Values.crd.install }}")
    lines.append(yaml_str.rstrip())
    lines.append("{{- end }}")
    return "\n".join(lines) + "\n"


# =============================================================================
# Generator
# =============================================================================


class CRDGenerator(Generator):
    """Generate Kubernetes CRD from AIPerfConfig schema."""

    name = "CRD Schema"
    description = "Generate Kubernetes CRD YAML from AIPerfConfig Pydantic model"

    def generate(self) -> GeneratorResult:
        sys.path.insert(0, "src")
        from aiperf.config.config import AIPerfConfig

        # Get JSON Schema from Pydantic model
        schema = AIPerfConfig.model_json_schema()
        if self.verbose:
            defs = schema.get("$defs", {})
            props = schema.get("properties", {})
            print_step(
                f"JSON Schema: {len(defs)} definitions, {len(props)} top-level properties"
            )

        # Convert to K8s-compatible OpenAPI v3
        config_properties = convert_aiperf_config_fields(schema, verbose=self.verbose)

        # Build full CRD
        crd = _build_crd(config_properties)

        # Render YAML
        crd_yaml = render_crd_yaml(crd)
        helm_yaml = render_helm_crd_yaml(crd)

        field_count = len(config_properties)
        return GeneratorResult(
            files=[
                GeneratedFile(CRD_FILE, crd_yaml),
                GeneratedFile(HELM_CRD_FILE, helm_yaml),
            ],
            summary=f"CRD with {field_count} AIPerfConfig fields",
        )


if __name__ == "__main__":
    main(CRDGenerator)
