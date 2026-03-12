# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for validating and parsing configuration inputs.

Includes:
- Parsing utilities (strings, lists, dicts)
- Validation helpers for Pydantic model validators
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.utils import load_json_str
from aiperf.plugin.enums import ServiceType

if TYPE_CHECKING:
    from pydantic import BaseModel


# -----------------------------------------------------------------------------
# Validation Helpers
# -----------------------------------------------------------------------------


def raise_if(*conditions: bool, error: str) -> None:
    """Raise ValueError if ALL conditions are True.

    Use this to consolidate conditional validation logic in model validators.

    Args:
        *conditions: Boolean conditions to check. Error is raised only if ALL are True.
        error: Error message to raise.

    Example:
        # Instead of:
        if "num_users" in self.model_fields_set and self.user_centric_rate is None:
            raise ValueError("--num-users requires --user-centric-rate")

        # Use:
        raise_if(
            is_field_set(self.loadgen, "num_users"),
            self.user_centric_rate is None,
            error="--num-users requires --user-centric-rate"
        )
    """
    if all(conditions):
        raise ValueError(error)


def is_field_set(model: BaseModel, field_name: str) -> bool:
    """Check if a field was explicitly set on a Pydantic model.

    Args:
        model: The Pydantic model instance.
        field_name: The name of the field to check.

    Returns:
        True if the field is in model_fields_set, False otherwise.
    """
    return field_name in model.model_fields_set


def are_fields_set(model: BaseModel, *field_names: str) -> bool:
    """Check if ANY of the specified fields were explicitly set.

    Args:
        model: The Pydantic model instance.
        *field_names: Field names to check.

    Returns:
        True if any field is in model_fields_set, False otherwise.
    """
    return any(f in model.model_fields_set for f in field_names)


def check_mutually_exclusive(
    model: BaseModel,
    *field_names: str,
    error: str | None = None,
) -> None:
    """Raise ValueError if more than one of the specified fields is set.

    Args:
        model: The Pydantic model instance.
        *field_names: Field names that are mutually exclusive.
        error: Custom error message. If None, generates a default message.

    Example:
        check_mutually_exclusive(
            self,
            "no_gpu_telemetry",
            "gpu_telemetry",
            error="Cannot use both --no-gpu-telemetry and --gpu-telemetry"
        )
    """
    set_fields = [f for f in field_names if f in model.model_fields_set]
    if len(set_fields) > 1:
        if error is None:
            options = ", ".join(f"--{f.replace('_', '-')}" for f in set_fields)
            error = f"Cannot use {options} together. Use only one of these options."
        raise ValueError(error)


def check_requires(
    model: BaseModel,
    field: str,
    required_field: str,
    *,
    error: str | None = None,
) -> None:
    """Raise ValueError if field is set but required_field is not.

    Args:
        model: The Pydantic model instance.
        field: The field that has a requirement.
        required_field: The field that must also be set.
        error: Custom error message.

    Example:
        check_requires(
            self.loadgen,
            "num_users",
            "user_centric_rate",
            error="--num-users can only be used with --user-centric-rate"
        )
    """
    if field in model.model_fields_set and required_field not in model.model_fields_set:
        if error is None:
            opt = f"--{field.replace('_', '-')}"
            req = f"--{required_field.replace('_', '-')}"
            error = f"{opt} can only be used with {req}."
        raise ValueError(error)


def parse_str_or_list(input: Any) -> list[Any] | None:
    """
    Parses the input to ensure it is either a string, a list, or None. If the input is a string,
    it splits the string by commas and trims any whitespace around each element, returning
    the result as a list. If the input is already a list, it is returned as-is. If the input
    is None, returns None. If the input is any other type, a ValueError is raised.
    Args:
        input (Any): The input to be parsed. Expected to be a string, list, or None.
    Returns:
        list | None: A list of strings derived from the input, or None.
    Raises:
        ValueError: If the input is neither a string, list, nor None.
    """
    if input is None:
        return None
    if isinstance(input, str):
        output = [item.strip() for item in input.split(",")]
    elif isinstance(input, list):
        # TODO: When using cyclopts, the values are already lists, so we have to split them by commas.
        output = []
        for item in input:
            if isinstance(item, str):
                output.extend([token.strip() for token in item.split(",")])
            else:
                output.append(item)
    else:
        raise ValueError(f"User Config: {input} - must be a string, list, or None")

    return output


def parse_str_or_csv_list(input: Any) -> list[Any]:
    """
    Parses the input to ensure it is either a string or a list. If the input is a string,
    it splits the string by commas and trims any whitespace around each element, returning
    the result as a list. If the input is already a list, it will split each item by commas
    and trim any whitespace around each element, returning the combined result as a list.
    If the input is neither a string nor a list, a ValueError is raised.

    [1, 2, 3] -> [1, 2, 3]
    "1,2,3" -> ["1", "2", "3"]
    ["1,2,3", "4,5,6"] -> ["1", "2", "3", "4", "5", "6"]
    ["1,2,3", 4, 5] -> ["1", "2", "3", 4, 5]
    """
    if isinstance(input, str):
        output = [item.strip() for item in input.split(",")]
    elif isinstance(input, list):
        output = []
        for item in input:
            if isinstance(item, str):
                output.extend([token.strip() for token in item.split(",")])
            else:
                output.append(item)
    else:
        raise ValueError(f"User Config: {input} - must be a string or list")

    return output


def parse_service_types(input: Any | None) -> set[ServiceType] | None:
    """Parses the input to ensure it is a set of service types.
    Will replace hyphens with underscores for user convenience."""
    if input is None:
        return None

    return {
        ServiceType(service_type.replace("-", "_"))
        for service_type in parse_str_or_csv_list(input)
    }


def coerce_value(value: Any) -> Any:
    """Coerce the value to the correct type."""
    if not isinstance(value, str):
        return value
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("none", "null"):
        return None
    if value.isdigit() and (not value.startswith("0") or value == "0"):
        return int(value)
    if (
        value.startswith("-")
        and value[1:].isdigit()
        and (not value.startswith("-0") or value == "-0")
    ):
        return int(value)
    if value.count(".") == 1 and (
        value.replace(".", "").isdigit()
        or (value.startswith("-") and value[1:].replace(".", "").isdigit())
    ):
        return float(value)
    return value


def parse_str_or_dict_as_tuple_list(input: Any | None) -> list[tuple[str, Any]] | None:
    """
    Parses the input to ensure it is a list of tuples. (key, value) pairs.

    - If the input is a string:
        - If the string starts with a '{', it is parsed as a JSON string.
        - Otherwise, it splits the string by commas and then for each item, it splits the item by colons
        into key and value, trims any whitespace, and coerces the value to the correct type.
    - If the input is a dictionary, it is converted to a list of tuples by key and value pairs.
    - If the input is a list, it recursively calls this function on each item, and aggregates the results.
        - If the item is already a 2-element sequence (key-value pair), it is converted directly to a tuple.
    - Otherwise, a ValueError is raised.

    Args:
        input (Any): The input to be parsed. Expected to be a string, list, or dictionary.
    Returns:
        list[tuple[str, Any]]: A list of tuples derived from the input.
    Raises:
        ValueError: If the input is neither a string, list, nor dictionary, or if the parsing fails.
    """
    if input is None:
        return None

    if isinstance(input, list | tuple | set):
        output = []
        for item in input:
            # If item is already a 2-element sequence (key-value pair), convert directly to tuple
            if isinstance(item, list | tuple) and len(item) == 2:
                key, value = item
                output.append((str(key), coerce_value(value)))
            else:
                res = parse_str_or_dict_as_tuple_list(item)
                if res is not None:
                    output.extend(res)
        return output

    if isinstance(input, dict):
        return [(key, coerce_value(value)) for key, value in input.items()]

    if isinstance(input, str):
        if input.startswith("{"):
            try:
                return [(key, value) for key, value in load_json_str(input).items()]
            except orjson.JSONDecodeError as e:
                raise ValueError(
                    f"User Config: {input} - must be a valid JSON string"
                ) from e
        else:
            result = []
            for item in input.split(","):
                parts = item.split(":", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"User Config: {input} - each item must be in 'key:value' format"
                    )
                key, value = parts
                result.append((key.strip(), coerce_value(value.strip())))
            return result

    raise ValueError(f"User Config: {input} - must be a valid string, list, or dict")


def print_str_or_list(input: Any) -> str:
    if isinstance(input, list):
        return ", ".join(map(str, input))
    elif isinstance(input, Enum):
        return str(input).lower()
    return str(input)


def parse_str_or_list_of_positive_values(input: Any) -> list[Any]:
    """
    Parses the input to ensure it is a list of positive integers or floats.
    This function first converts the input into a list using `parse_str_or_list`.
    It then validates that each value in the list is either an integer or a float
    and that all values are strictly greater than zero. If any value fails this
    validation, a `ValueError` is raised.
    Args:
        input (Any): The input to be parsed. It can be a string or a list.
    Returns:
        List[Any]: A list of positive integers or floats.
    Raises:
        ValueError: If any value in the parsed list is not a positive integer or float,
                    or if the input is None.
    """
    # Guard against None before calling parse_str_or_list to provide clear error
    if input is None:
        raise ValueError("input must be a string or list of strings, not None")

    output = parse_str_or_list(input)

    # Additional safety check (should not be reached due to above check, but defensive)
    if output is None:
        raise ValueError("input must be a string or list of strings, not None")

    try:
        output = [
            float(x) if "." in str(x) or "e" in str(x).lower() else int(x)
            for x in output
        ]
    except ValueError as e:
        raise ValueError(f"User Config: {output} - all values must be numeric") from e

    if not all(isinstance(x, int | float) and x > 0 for x in output):
        raise ValueError(f"User Config: {output} - all values must be positive numbers")

    return output


def parse_file(value: str | Path | None) -> Path | None:
    """
    Parses the given string value and returns a Path object if the value represents
    a valid file or directory. Returns None if the input value is empty.

    Absolute paths that don't exist locally are accepted without validation
    to support Kubernetes deployments where files reside on PVC mounts.

    Args:
        value: The string or Path value to parse.
    Returns:
        A Path object if the value is valid, or None if the value is empty.
    Raises:
        ValueError: If the value is not a valid file or directory and is a relative path.
    """

    if not value:
        return None
    if isinstance(value, Path):
        return value
    elif not isinstance(value, str):
        raise ValueError(f"Expected a string, but got {type(value).__name__}")
    else:
        path = Path(value)
        if path.is_file() or path.is_dir() or path.is_absolute():
            return path
        else:
            raise ValueError(f"'{value}' is not a valid file or directory")


def parse_str_as_numeric_dict(
    input_value: str | dict[str, float] | None,
) -> dict[str, float] | None:
    """
    Parse a string of key:value pairs such as 'k:v x:y' into {k: v, x: y}.
    Also accepts a dict directly (e.g., from YAML/JSON config).
    """
    if input_value is None:
        return None
    if isinstance(input_value, dict):
        return {k: float(v) for k, v in input_value.items()}
    if not isinstance(input_value, str):
        raise ValueError(
            f"User Config: expected a string of space-separated 'key:value' pairs or a dict, got {type(input_value).__name__}"
        )
    input_string = input_value

    input_string = input_string.strip()
    if not input_string:
        raise ValueError(
            "User Config: expected space-separated 'key:value' pairs (e.g., 'k:v x:y'), got empty string"
        )

    output: dict[str, float] = {}
    for item in input_string.split():
        if not item or ":" not in item:
            raise ValueError(f"User Config: '{item}' is not in 'key:value' format")
        key, val = item.split(":", 1)
        key, val = key.strip(), val.strip()
        if not key:
            raise ValueError(f"User Config: '{item}' has an empty key")
        if not val:
            raise ValueError(f"User Config: '{item}' has an empty value")
        try:
            output[key] = float(val)
        except ValueError as e:
            raise ValueError(
                f"User Config: value for '{key}' must be numeric, got '{val}'"
            ) from e
    return output


def validate_sequence_distribution(v: str | None) -> str | None:
    """Validate sequence distribution format, returns original value if valid."""
    if v is not None:
        from aiperf.common.models.sequence_distribution import DistributionParser

        DistributionParser.validate(v)
    return v
