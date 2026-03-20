# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.cli_commands.kube_init module."""

from pathlib import Path

import ruamel.yaml

from aiperf.cli_commands.kube.init import init_config
from aiperf.kubernetes.init_template import generate_init_template

# =============================================================================
# Template Content Tests
# =============================================================================


class TestTemplateContent:
    """Tests for the KUBE_INIT_TEMPLATE content."""

    def test_template_contains_required_sections(self) -> None:
        """Test template has AIPerfJob CR structure."""
        template = generate_init_template("test.yaml")
        assert "apiVersion: aiperf.nvidia.com/v1alpha1" in template
        assert "kind: AIPerfJob" in template
        assert "models:" in template
        assert "endpoint:" in template
        assert "phases:" in template

    def test_template_substitutes_filename(self) -> None:
        """Test that {filename} is replaced in the template."""
        template = generate_init_template("my-benchmark.yaml")
        assert "my-benchmark.yaml" in template
        assert "{filename}" not in template

    def test_template_is_valid_yaml(self) -> None:
        """Test uncommented lines parse as valid YAML."""
        template = generate_init_template("test.yaml")
        # Extract only non-comment, non-empty lines
        yaml_lines = []
        for line in template.splitlines():
            stripped = line.lstrip()
            if stripped and not stripped.startswith("#"):
                yaml_lines.append(line)

        yaml_str = "\n".join(yaml_lines)
        yaml = ruamel.yaml.YAML()
        parsed = yaml.load(yaml_str)
        assert parsed is not None
        assert "apiVersion" in parsed
        assert "spec" in parsed

    def test_template_mentions_deployment_options(self) -> None:
        """Test template mentions deployment configuration."""
        template = generate_init_template("test.yaml")
        assert "podTemplate" in template
        assert "scheduling" in template


# =============================================================================
# Init Command Tests
# =============================================================================


class TestInitCommand:
    """Tests for the kube init command."""

    async def test_init_stdout(self, capsys) -> None:
        """Test init prints template to stdout when no output specified."""

        await init_config(output=None)

        captured = capsys.readouterr()
        assert "apiVersion:" in captured.out
        assert "models:" in captured.out

    async def test_init_writes_to_file(self, tmp_path: Path) -> None:
        """Test init writes template to specified file."""

        output_file = tmp_path / "benchmark.yaml"
        await init_config(output=output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "endpoint:" in content
        assert "benchmark.yaml" in content

    async def test_init_prompts_on_overwrite(self, tmp_path: Path, monkeypatch) -> None:
        """Test init prompts before overwriting existing file."""
        from unittest.mock import AsyncMock, patch

        output_file = tmp_path / "existing.yaml"
        output_file.write_text("old content")

        with patch(
            "aiperf.kubernetes.cli_helpers.confirm_action",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await init_config(output=output_file)

        # File should not be overwritten
        assert output_file.read_text() == "old content"

    async def test_init_overwrites_on_confirm(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test init overwrites file when user confirms."""
        from unittest.mock import AsyncMock, patch

        output_file = tmp_path / "existing.yaml"
        output_file.write_text("old content")

        with patch(
            "aiperf.kubernetes.cli_helpers.confirm_action",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await init_config(output=output_file)

        content = output_file.read_text()
        assert "endpoint:" in content
        assert "old content" not in content

    async def test_init_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test init creates parent directories if needed."""

        output_file = tmp_path / "subdir" / "deep" / "benchmark.yaml"
        await init_config(output=output_file)

        assert output_file.exists()
