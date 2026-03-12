# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.init_template module.

Focuses on:
- Filename substitution into template usage instructions
- Template structural integrity (required YAML sections present)
- Edge-case filenames (spaces, special characters)
"""

import pytest
from pytest import param

from aiperf.kubernetes.init_template import KUBE_INIT_TEMPLATE, generate_init_template

# ============================================================
# Template Generation
# ============================================================


class TestGenerateInitTemplate:
    """Verify filename substitution and template content."""

    def test_generate_init_template_standard_filename_substitutes_correctly(
        self,
    ) -> None:
        result = generate_init_template("my-config.yaml")
        assert "aiperf kube profile --user-config my-config.yaml --image" in result

    def test_generate_init_template_different_filename_substitutes_correctly(
        self,
    ) -> None:
        result = generate_init_template("benchmark.yml")
        assert "aiperf kube profile --user-config benchmark.yml --image" in result

    @pytest.mark.parametrize(
        "filename",
        [
            param("config with spaces.yaml", id="spaces-in-name"),
            param("path/to/config.yaml", id="path-separators"),
            param("config_v2.0.yaml", id="dots-in-name"),
            param("a", id="single-char"),
        ],
    )  # fmt: skip
    def test_generate_init_template_special_filenames_substitutes_correctly(
        self, filename: str
    ) -> None:
        result = generate_init_template(filename)
        assert filename in result


# ============================================================
# Template Structure
# ============================================================


class TestTemplateStructure:
    """Verify the template contains required YAML sections and documentation."""

    @pytest.fixture
    def rendered(self) -> str:
        return generate_init_template("config.yaml")

    def test_template_contains_endpoint_section(self, rendered: str) -> None:
        assert "endpoint:" in rendered

    def test_template_contains_model_names_field(self, rendered: str) -> None:
        assert "model_names:" in rendered

    def test_template_contains_urls_field(self, rendered: str) -> None:
        assert "urls:" in rendered

    def test_template_contains_loadgen_section_commented(self, rendered: str) -> None:
        assert "# loadgen:" in rendered

    def test_template_contains_input_section_commented(self, rendered: str) -> None:
        assert "# input:" in rendered

    def test_template_documents_cli_flags(self, rendered: str) -> None:
        for flag in ("--image", "--workers-max", "--namespace", "--detach"):
            assert flag in rendered, f"Missing CLI flag documentation: {flag}"

    def test_template_is_valid_yaml_comment_structure(self, rendered: str) -> None:
        """Every non-empty line should be a YAML value or comment."""
        for line in rendered.splitlines():
            stripped = line.strip()
            if stripped:
                is_comment = stripped.startswith("#")
                is_yaml_content = (
                    ":" in stripped
                    or stripped.startswith("-")
                    or stripped.startswith('"')
                )
                assert is_comment or is_yaml_content, (
                    f"Unexpected line format: {line!r}"
                )


# ============================================================
# Raw Template Constant
# ============================================================


class TestKubeInitTemplateConstant:
    """Verify properties of the raw template string."""

    def test_template_has_single_format_placeholder(self) -> None:
        """Template should only use {filename} - no other placeholders."""
        # Substitute filename, then verify no remaining placeholders
        result = KUBE_INIT_TEMPLATE.format(filename="TEST")
        assert "{" not in result
        assert "}" not in result

    def test_template_does_not_end_with_newline(self) -> None:
        """Trailing newline is stripped by backslash continuation on the quote."""
        assert not KUBE_INIT_TEMPLATE.endswith("\n\n")

    def test_template_starts_with_comment(self) -> None:
        assert KUBE_INIT_TEMPLATE.startswith("#")
