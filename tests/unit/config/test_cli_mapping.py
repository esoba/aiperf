# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CLI mapping table integrity.

Validates that all CLIField entries in CLI_FIELDS resolve correctly
against AIPerfConfig and produce valid metadata for model generation.
"""

from __future__ import annotations

import pytest

from aiperf.config.cli_builder import (
    CLIModel,
    get_field_description,
    get_field_type_and_default,
    resolve_field,
)
from aiperf.config.cli_mapping import CLI_FIELD_BY_NAME, CLI_FIELDS, CLIField


class TestCLIFieldsIntegrity:
    """Validate structural integrity of the CLI_FIELDS mapping table."""

    def test_no_duplicate_field_names(self) -> None:
        names = [f.field_name for f in CLI_FIELDS]
        assert len(names) == len(set(names)), (
            f"Duplicate field_name entries: {[n for n in names if names.count(n) > 1]}"
        )

    def test_no_duplicate_primary_flags(self) -> None:
        flags = [f.primary_flag for f in CLI_FIELDS]
        assert len(flags) == len(set(flags)), (
            f"Duplicate primary flags: {[f for f in flags if flags.count(f) > 1]}"
        )

    def test_field_by_name_index_complete(self) -> None:
        assert len(CLI_FIELD_BY_NAME) == len(CLI_FIELDS)

    def test_all_flags_start_with_double_dash(self) -> None:
        for cli_field in CLI_FIELDS:
            flags = (
                (cli_field.flags,)
                if isinstance(cli_field.flags, str)
                else cli_field.flags
            )
            for flag in flags:
                assert flag.startswith("-"), (
                    f"{cli_field.field_name}: flag {flag!r} must start with -"
                )


class TestPathResolution:
    """Validate that all CLIField paths resolve against AIPerfConfig."""

    @pytest.fixture(
        params=[f for f in CLI_FIELDS if f.path is not None], ids=lambda f: f.field_name
    )
    def cli_field_with_path(self, request: pytest.FixtureRequest) -> CLIField:
        return request.param

    def test_path_resolves(self, cli_field_with_path: CLIField) -> None:
        model_cls, field_name, field_info = resolve_field(cli_field_with_path.path)
        assert field_info is not None
        assert field_name == cli_field_with_path.path.split(".")[-1]


class TestDescriptions:
    """Validate that all CLIField entries produce non-empty descriptions."""

    @pytest.fixture(params=CLI_FIELDS, ids=lambda f: f.field_name)
    def cli_field(self, request: pytest.FixtureRequest) -> CLIField:
        return request.param

    def test_description_non_empty(self, cli_field: CLIField) -> None:
        description = get_field_description(cli_field)
        assert description, f"{cli_field.field_name} has empty description"
        assert len(description) > 10, (
            f"{cli_field.field_name} description too short: {description!r}"
        )


class TestTypeAndDefault:
    """Validate type/default resolution for all fields."""

    @pytest.fixture(params=CLI_FIELDS, ids=lambda f: f.field_name)
    def cli_field(self, request: pytest.FixtureRequest) -> CLIField:
        return request.param

    def test_type_resolves(self, cli_field: CLIField) -> None:
        field_type, default = get_field_type_and_default(cli_field)
        assert field_type is not None, f"{cli_field.field_name} has None type"


class TestCLIModelGeneration:
    """Validate the dynamically generated CLIModel."""

    def test_model_is_generated(self) -> None:
        assert CLIModel is not None
        assert CLIModel.__name__ == "CLIModel"

    def test_model_has_all_fields(self) -> None:
        for cli_field in CLI_FIELDS:
            assert cli_field.field_name in CLIModel.model_fields, (
                f"Missing field {cli_field.field_name} in CLIModel"
            )

    def test_model_field_count_matches(self) -> None:
        assert len(CLIModel.model_fields) == len(CLI_FIELDS)

    def test_model_fields_have_descriptions(self) -> None:
        for name, field_info in CLIModel.model_fields.items():
            assert field_info.description, f"CLIModel.{name} has no description"

    def test_warmup_fields_have_warmup_descriptions(self) -> None:
        warmup_fields = [f for f in CLI_FIELDS if f.is_warmup_override]
        for cli_field in warmup_fields:
            field_info = CLIModel.model_fields[cli_field.field_name]
            assert "warmup" in field_info.description.lower(), (
                f"{cli_field.field_name} description should mention 'warmup'"
            )
