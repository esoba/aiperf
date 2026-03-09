# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the config_loader module."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.claude_code_gen.config_loader import (
    list_bundled_configs,
    load_config,
)


class TestLoadConfig:
    def test_load_bundled_default(self) -> None:
        config = load_config("default")
        assert config.system_prompt_tokens == 8_000
        assert config.max_prompt_tokens == 200_000

    def test_list_bundled_configs(self) -> None:
        names = list_bundled_configs()
        assert "default" in names

    def test_load_from_file_path(self, tmp_path: Path) -> None:
        data = {
            "system_prompt_tokens": 5000,
            "new_tokens_per_turn": {"mean": 1000, "median": 500},
            "generation_length": {"mean": 300, "median": 200},
            "inter_turn_delay": {
                "agentic_fraction": 0.5,
                "agentic_delay": {"mean": 2000, "median": 1500},
                "human_delay": {"mean": 30000, "median": 20000},
            },
            "cache": {
                "layer1_tokens": 3000,
                "layer1_5_tokens": 2000,
                "layer2": {"mean": 5000, "median": 4000},
                "block_size": 512,
            },
        }
        path = tmp_path / "custom.json"
        path.write_bytes(orjson.dumps(data))

        config = load_config(str(path))
        assert config.system_prompt_tokens == 5000
        assert config.cache.layer2.mean == 5000
        assert config.cache.layer2.mu is not None
        assert config.cache.layer2.sigma is not None

    def test_load_manifest_as_config(self, tmp_path: Path) -> None:
        """manifest.json wraps config under generation_params."""
        inner = {
            "system_prompt_tokens": 8500,
            "new_tokens_per_turn": {"mean": 4500, "median": 2100},
            "generation_length": {"mean": 600, "median": 350},
            "inter_turn_delay": {
                "agentic_fraction": 0.7,
                "agentic_delay": {"mean": 3000, "median": 2000},
                "human_delay": {"mean": 45000, "median": 30000},
            },
            "cache": {
                "layer1_tokens": 32000,
                "layer1_5_tokens": 20000,
                "layer2": {"mean": 15000, "median": 2000},
                "block_size": 512,
            },
        }
        manifest = {
            "seed": 42,
            "block_size": 512,
            "num_sessions": 100,
            "config_name": "custom",
            "generation_params": inner,
        }
        path = tmp_path / "manifest.json"
        path.write_bytes(orjson.dumps(manifest))

        config = load_config(str(path))
        assert config.system_prompt_tokens == 8500

    def test_unknown_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config("nonexistent-config")
