# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - YAML Loader

This module provides functions for loading AIPerf configuration from
YAML files with support for environment variable substitution.

Key Features:
    - YAML file loading with validation
    - Environment variable substitution (${VAR} syntax)
    - Default value support (${VAR:default} syntax)
    - Detailed error messages for configuration issues

Example Usage:
    >>> from config_v2 import load_config
    >>> config = load_config("benchmark.yaml")
    >>> print(config.models)

    With environment variables:
    >>> # YAML: api_key: ${OPENAI_API_KEY}
    >>> import os
    >>> os.environ["OPENAI_API_KEY"] = "sk-..."
    >>> config = load_config("benchmark.yaml")

Environment Variable Syntax:
    ${VAR}           - Required variable, error if not set
    ${VAR:default}   - Optional with default value
    ${VAR:}          - Optional with empty string default
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from aiperf.config.config import AIPerfConfig

__all__ = [
    # Constants
    "ENV_VAR_PATTERN",
    # Exceptions
    "ConfigurationError",
    "MissingEnvironmentVariableError",
    # Core loading functions
    "load_config",
    "load_config_from_env",
    "load_config_from_file_env",
    "load_config_from_string",
    "dump_config",
    "save_config",
    "validate_config_file",
    "merge_configs",
    "substitute_env_vars",
]

# Regex pattern for environment variable substitution
# Matches: ${VAR} or ${VAR:default}
ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


class ConfigurationError(Exception):
    """
    Exception raised for configuration loading errors.

    Attributes:
        message: Human-readable error description.
        file_path: Path to the configuration file (if applicable).
        line: Line number where the error occurred (if known).
        context: Additional context about the error.
    """

    def __init__(
        self,
        message: str,
        file_path: Path | str | None = None,
        line: int | None = None,
        context: str | None = None,
    ):
        self.message = message
        self.file_path = file_path
        self.line = line
        self.context = context

        # Build full message
        parts = [message]
        if file_path:
            parts.append(f"File: {file_path}")
        if line:
            parts.append(f"Line: {line}")
        if context:
            parts.append(f"Context: {context}")

        super().__init__("\n".join(parts))


class MissingEnvironmentVariableError(ConfigurationError):
    """
    Exception raised when a required environment variable is not set.

    Attributes:
        variable_name: Name of the missing variable.
        file_path: Path to the configuration file.
    """

    def __init__(
        self,
        variable_name: str,
        file_path: Path | str | None = None,
    ):
        self.variable_name = variable_name
        super().__init__(
            f"Required environment variable '{variable_name}' is not set",
            file_path=file_path,
            context="Use ${{VAR:default}} syntax to provide a default value",
        )


def substitute_env_vars(
    value: Any,
    file_path: Path | str | None = None,
) -> Any:
    """
    Recursively substitute environment variables in configuration values.

    Processes strings, lists, and dictionaries recursively, replacing
    ${VAR} and ${VAR:default} patterns with environment variable values.

    Args:
        value: Configuration value to process. Can be string, list, dict,
            or any other type (non-string/list/dict types pass through unchanged).
        file_path: Path to config file for error messages.

    Returns:
        Value with environment variables substituted.

    Raises:
        MissingEnvironmentVariableError: If a required variable (no default)
            is not set in the environment.

    Examples:
        >>> os.environ["MY_VAR"] = "hello"
        >>> substitute_env_vars("${MY_VAR}")
        'hello'

        >>> substitute_env_vars("${UNSET:default_value}")
        'default_value'

        >>> substitute_env_vars({"key": "${MY_VAR}"})
        {'key': 'hello'}

        >>> substitute_env_vars(["${MY_VAR}", "static"])
        ['hello', 'static']
    """
    if isinstance(value, str):
        return _substitute_string(value, file_path)
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v, file_path) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_vars(item, file_path) for item in value]
    else:
        # Pass through non-string/dict/list values unchanged
        return value


def _substitute_string(
    text: str,
    file_path: Path | str | None = None,
) -> str:
    """
    Substitute environment variables in a single string.

    Args:
        text: String containing ${VAR} or ${VAR:default} patterns.
        file_path: Path to config file for error messages.

    Returns:
        String with variables substituted.

    Raises:
        MissingEnvironmentVariableError: If required variable not set.
    """

    def replace_match(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)

        env_value = os.environ.get(var_name)

        if env_value is not None:
            return env_value
        elif default is not None:
            # Default was specified (even if empty string)
            return default
        else:
            # No value and no default - error
            raise MissingEnvironmentVariableError(var_name, file_path)

    return ENV_VAR_PATTERN.sub(replace_match, text)


def load_config(
    file_path: Path | str,
    *,
    substitute_env: bool = True,
) -> AIPerfConfig:
    """
    Load and validate AIPerf configuration from a YAML file.

    This is the primary function for loading configuration files. It reads
    the YAML file, optionally substitutes environment variables, and
    validates the configuration against the schema.

    Args:
        file_path: Path to the YAML configuration file.
        substitute_env: Whether to process environment variable substitution.
            Defaults to True.

    Returns:
        Validated AIPerfConfig object.

    Raises:
        ConfigurationError: If the file cannot be read or parsed.
        MissingEnvironmentVariableError: If a required env var is missing.
        pydantic.ValidationError: If the configuration fails validation.

    Example:
        >>> config = load_config("benchmark.yaml")
        >>> print(config.models)
        ['meta-llama/Llama-3.1-8B-Instruct']

        >>> print(list(config.load.keys())[0])
        'warmup'
    """
    file_path = Path(file_path)

    # Check file exists
    if not file_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {file_path}",
            file_path=file_path,
        )

    if not file_path.is_file():
        raise ConfigurationError(
            f"Path is not a file: {file_path}",
            file_path=file_path,
        )

    # Read file contents
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigurationError(
            f"Failed to read configuration file: {e}",
            file_path=file_path,
        ) from e

    # Load and validate
    return load_config_from_string(
        content,
        file_path=file_path,
        substitute_env=substitute_env,
    )


def load_config_from_string(
    yaml_content: str,
    *,
    file_path: Path | str | None = None,
    substitute_env: bool = True,
) -> AIPerfConfig:
    """
    Load and validate AIPerf configuration from a YAML string.

    Useful for programmatic configuration or testing without files.

    Args:
        yaml_content: YAML configuration as a string.
        file_path: Optional file path for error messages.
        substitute_env: Whether to process environment variable substitution.

    Returns:
        Validated AIPerfConfig object.

    Raises:
        ConfigurationError: If the YAML cannot be parsed.
        MissingEnvironmentVariableError: If a required env var is missing.
        pydantic.ValidationError: If the configuration fails validation.

    Example:
        >>> yaml_str = '''
        ... models:
        ...   - llama-3-8b
        ... endpoint:
        ...   urls: ["http://localhost:8000/v1/chat/completions"]
        ... datasets:
        ...   main:
        ...     type: synthetic
        ...     entries: 100
        ... load:
        ...   type: concurrency
        ...   dataset: main
        ...   duration: 10
        ...   concurrency: 1
        ... '''
        >>> config = load_config_from_string(yaml_str)
    """
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML syntax: {e}",
            file_path=file_path,
        ) from e

    if data is None:
        raise ConfigurationError(
            "Configuration file is empty",
            file_path=file_path,
        )

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration must be a YAML mapping, got {type(data).__name__}",
            file_path=file_path,
        )

    # Substitute environment variables
    if substitute_env:
        data = substitute_env_vars(data, file_path)

    # Validate and create config
    try:
        return AIPerfConfig.model_validate(data)
    except Exception as e:
        # Re-raise with file context if not already a ConfigurationError
        if isinstance(e, ConfigurationError):
            raise
        # Pydantic ValidationError has good messages, just add file context
        if file_path:
            raise ConfigurationError(
                f"Configuration validation failed: {e}",
                file_path=file_path,
            ) from e
        raise


def dump_config(
    config: AIPerfConfig,
    *,
    exclude_defaults: bool = True,
    exclude_none: bool = True,
) -> str:
    """
    Dump an AIPerfConfig object to YAML string.

    Useful for generating configuration templates or debugging.

    Args:
        config: The configuration object to dump.
        exclude_defaults: Exclude fields that have default values.
        exclude_none: Exclude fields that are None.

    Returns:
        YAML string representation of the configuration.

    Example:
        >>> config = AIPerfConfig(...)
        >>> print(dump_config(config))
    """
    data = config.model_dump(
        exclude_defaults=exclude_defaults,
        exclude_none=exclude_none,
        mode="json",  # Use JSON-compatible types
    )
    return yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def save_config(
    config: AIPerfConfig,
    file_path: Path | str,
    *,
    exclude_defaults: bool = True,
    exclude_none: bool = True,
) -> None:
    """
    Save an AIPerfConfig object to a YAML file.

    Args:
        config: The configuration object to save.
        file_path: Path to the output YAML file.
        exclude_defaults: Exclude fields that have default values.
        exclude_none: Exclude fields that are None.

    Raises:
        OSError: If the file cannot be written.

    Example:
        >>> config = AIPerfConfig(...)
        >>> save_config(config, "output.yaml")
    """
    file_path = Path(file_path)
    yaml_content = dump_config(
        config,
        exclude_defaults=exclude_defaults,
        exclude_none=exclude_none,
    )
    file_path.write_text(yaml_content, encoding="utf-8")


def validate_config_file(file_path: Path | str) -> list[str]:
    """
    Validate a configuration file and return any warnings.

    Unlike load_config, this function collects warnings rather than
    raising exceptions immediately, making it useful for linting.

    Args:
        file_path: Path to the YAML configuration file.

    Returns:
        List of warning messages (empty if no issues).

    Raises:
        ConfigurationError: If the file has fatal errors.

    Example:
        >>> warnings = validate_config_file("benchmark.yaml")
        >>> for w in warnings:
        ...     print(f"Warning: {w}")
    """
    warnings: list[str] = []

    # Load the config (will raise on fatal errors)
    config = load_config(file_path)

    # Check for potential issues

    # Warn if no profiling load configs
    profiling_phases = config.get_profiling_phases()
    if not profiling_phases:
        warnings.append(
            "All load configs have exclude=True. Final results will be empty."
        )

    # Warn if streaming disabled but TTFT goodput set
    if config.slos:
        if config.slos.time_to_first_token and not config.endpoint.streaming:
            warnings.append(
                "slos.time_to_first_token is set but streaming is disabled. "
                "TTFT measurement requires streaming=true."
            )
        if config.slos.inter_token_latency and not config.endpoint.streaming:
            warnings.append(
                "slos.inter_token_latency is set but streaming is disabled. "
                "ITL measurement requires streaming=true."
            )

    # Warn if prefill_concurrency set without streaming
    for name, phase in config.load.items():
        if phase.prefill_concurrency and not config.endpoint.streaming:
            warnings.append(
                f"Load config '{name}' has prefill_concurrency set but "
                "streaming is disabled. Prefill concurrency requires streaming=true."
            )

    return warnings


def load_config_from_file_env() -> AIPerfConfig:
    """
    Load AIPerf configuration from the AIPERF_CONFIG_FILE environment variable.

    This function is used by Kubernetes child processes to load the
    configuration file that was mounted into the pod via a ConfigMap.

    Returns:
        AIPerfConfig object.

    Raises:
        ConfigurationError: If the env var is not set or the file cannot be loaded.
    """
    config_file = os.environ.get("AIPERF_CONFIG_FILE")
    if config_file is None:
        raise ConfigurationError(
            "AIPERF_CONFIG_FILE environment variable not set. "
            "This function is meant to be called from Kubernetes pods "
            "that receive configuration via a mounted ConfigMap."
        )

    return load_config(config_file, substitute_env=True)


def load_config_from_env() -> AIPerfConfig:
    """
    Load AIPerf configuration from environment variables.

    This function is used by child processes to deserialize the configuration
    that was passed from the parent process via environment variables.

    The configuration is expected to be serialized as JSON in the
    AIPERF_CONFIG environment variable.

    Returns:
        AIPerfConfig object.

    Raises:
        ConfigurationError: If the config cannot be loaded from environment.

    Example:
        >>> # In parent process:
        >>> os.environ["AIPERF_CONFIG"] = config.model_dump_json()
        >>>
        >>> # In child process:
        >>> config = load_config_from_env()
    """
    import orjson

    config_json = os.environ.get("AIPERF_CONFIG")
    if config_json is None:
        raise ConfigurationError(
            "AIPERF_CONFIG environment variable not set. "
            "This function is meant to be called from child processes "
            "that receive configuration from the parent process."
        )

    try:
        data = orjson.loads(config_json)
        return AIPerfConfig.model_validate(data)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration from environment: {e}"
        ) from e


def merge_configs(
    base: AIPerfConfig,
    override: dict[str, Any],
) -> AIPerfConfig:
    """
    Merge override values into a base configuration.

    Useful for applying CLI overrides to a file-based configuration.

    Args:
        base: The base configuration.
        override: Dictionary of override values.

    Returns:
        New AIPerfConfig with merged values.

    Example:
        >>> config = load_config("benchmark.yaml")
        >>> config = merge_configs(config, {"random_seed": 123})
    """
    base_dict = base.model_dump()

    def deep_merge(base_dict: dict, override_dict: dict) -> dict:
        """Recursively merge override into base."""
        result = base_dict.copy()
        for key, value in override_dict.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = deep_merge(base_dict, override)
    return AIPerfConfig.model_validate(merged)
