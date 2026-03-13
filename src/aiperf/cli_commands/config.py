# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for generating configuration files from CLI options."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from cyclopts import App, Parameter

from aiperf.config.cli_builder import CLIModel

if TYPE_CHECKING:
    from aiperf.config.config import AIPerfConfig

app = App(name="config", help="Configuration file generation and management")


@app.command(name="generate")
def generate(
    cli: CLIModel,
    *,
    format: Annotated[
        Literal["yaml", "cr"],
        Parameter(name=["-f", "--format"]),
    ] = "yaml",
    output: Annotated[
        Path | None,
        Parameter(name=["-o", "--output"]),
    ] = None,
    cr_name: Annotated[
        str,
        Parameter(name="--cr-name"),
    ] = "my-benchmark",
    cr_namespace: Annotated[
        str | None,
        Parameter(name="--cr-namespace"),
    ] = None,
    cr_image: Annotated[
        str | None,
        Parameter(name="--cr-image"),
    ] = None,
) -> None:
    """Generate a YAML config file or Kubernetes CR from CLI options.

    Converts profile CLI flags into a reusable configuration file.
    Supports plain YAML config or Kubernetes AIPerfJob custom resource format.

    Examples:
        # Generate YAML config from CLI options
        aiperf config generate --model Qwen/Qwen3-0.6B --url localhost:8000 --streaming

        # Save to file
        aiperf config generate --model Qwen/Qwen3-0.6B --url localhost:8000 -o benchmark.yaml

        # Generate Kubernetes AIPerfJob CR
        aiperf config generate --model Qwen/Qwen3-0.6B --url localhost:8000 \\
            --format cr --cr-name my-bench --cr-image aiperf:latest

        # Full example with load options
        aiperf config generate --model Qwen/Qwen3-0.6B --url localhost:8000 \\
            --concurrency 64 --request-count 1000 --streaming -o config.yaml

    Args:
        cli: Benchmark configuration (parsed from CLI flags).
        format: Output format. 'yaml' for config file, 'cr' for Kubernetes AIPerfJob custom resource.
        output: Output file path. Prints to stdout if not specified.
        cr_name: Name for the AIPerfJob custom resource (only used with --format cr).
        cr_namespace: Namespace for the custom resource (only used with --format cr). Omitted if not set.
        cr_image: Container image for the AIPerfJob (only used with --format cr).
    """
    from aiperf import cli_utils

    with cli_utils.exit_on_error(title="Error Generating Configuration"):
        from aiperf.config import cli_builder

        aiperf_config = cli_builder.build_aiperf_config(cli)

        if format == "yaml":
            content = _dump_clean_yaml(aiperf_config)
        else:
            content = _build_cr_yaml(
                aiperf_config,
                name=cr_name,
                namespace=cr_namespace,
                image=cr_image,
            )

        if output is None:
            sys.stdout.write(content)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
            _print_success_message(output, format)


def _dump_clean_yaml(config: AIPerfConfig) -> str:
    """Dump AIPerfConfig to clean YAML, stripping runtime-only fields.

    Args:
        config: The validated AIPerf configuration.

    Returns:
        YAML string suitable for use as a reusable config file.
    """
    import yaml

    data: dict[str, Any] = config.model_dump(
        exclude_defaults=True,
        exclude_none=True,
        mode="json",
    )
    _strip_runtime_fields(data)

    return yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def _strip_runtime_fields(data: dict[str, Any]) -> None:
    """Remove runtime-only fields that shouldn't appear in saved configs.

    Modifies data in-place: removes cli_command, benchmark_id, and
    empty containers that are artifacts of the CLI builder.

    Args:
        data: The config dict to clean up.
    """
    artifacts = data.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts.pop("cli_command", None)
        artifacts.pop("benchmark_id", None)
        if not artifacts:
            data.pop("artifacts")

    _remove_empty(data)


def _remove_empty(data: dict[str, Any]) -> None:
    """Recursively remove empty dicts and lists from a nested structure.

    Args:
        data: The dict to clean in-place.
    """
    keys_to_remove = []
    for key, value in data.items():
        if isinstance(value, dict):
            _remove_empty(value)
            if not value:
                keys_to_remove.append(key)
        elif isinstance(value, list) and not value:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del data[key]


def _build_cr_yaml(
    config: AIPerfConfig,
    *,
    name: str,
    namespace: str | None,
    image: str | None,
) -> str:
    """Build a Kubernetes AIPerfJob custom resource YAML from config.

    Args:
        config: The validated AIPerf configuration.
        name: CR metadata.name.
        namespace: CR metadata.namespace (omitted if None).
        image: Container image for spec.image (omitted if None).

    Returns:
        YAML string of the complete AIPerfJob CR.
    """
    import yaml

    spec: dict[str, Any] = config.model_dump(
        exclude_defaults=True,
        exclude_none=True,
        mode="json",
    )
    _strip_runtime_fields(spec)

    if image:
        spec["image"] = image

    metadata: dict[str, Any] = {"name": name}
    if namespace:
        metadata["namespace"] = namespace

    cr: dict[str, Any] = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": metadata,
        "spec": spec,
    }

    return yaml.dump(
        cr,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def _print_success_message(output: Path, format: str) -> None:
    """Print contextual success message after file generation."""
    sys.stderr.write(f"Configuration written to {output}\n")
    if format == "yaml":
        sys.stderr.write(f"  aiperf profile --config {output}\n")
    else:
        sys.stderr.write(f"  kubectl apply -f {output}\n")
