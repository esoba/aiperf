# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProxyInspiredCodingContentGenerator."""

import json

import pytest

from aiperf.common.config import PrefixPromptConfig, PromptConfig
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.generator.coding_content_proxy_inspired import (
    ProxyInspiredCodingContentGenerator,
)


class TestProxyInspiredCodingContentGenerator:
    @pytest.fixture
    def config(self) -> PromptConfig:
        return PromptConfig(
            mean=80,
            stddev=10,
            block_size=128,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )

    def test_pools_built(self, config: PromptConfig, mock_tokenizer_cls: type) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        gen = ProxyInspiredCodingContentGenerator(config, tokenizer)
        assert len(gen._tool_pool) > 0
        pool = gen._ensure_text_pool()
        assert len(pool) > 0

    def test_isinstance_base_coding_generator(
        self, config: PromptConfig, mock_tokenizer_cls: type
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        gen = ProxyInspiredCodingContentGenerator(config, tokenizer)
        assert isinstance(gen, CodingContentGenerator)

    def test_assistant_flat_contains_tool_json(
        self, config: PromptConfig, mock_tokenizer_cls: type
    ) -> None:
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        gen = ProxyInspiredCodingContentGenerator(config, tokenizer)
        for _ in range(12):
            s = gen._gen_proxy_assistant_flat()
            if '"name"' in s and '"input"' in s:
                for line in s.splitlines():
                    line = line.strip()
                    if line.startswith("{"):
                        obj = json.loads(line)
                        assert "name" in obj and "input" in obj
                        return
        pytest.fail("expected at least one JSON tool line in samples")

    def test_tool_weights_skew_read(self, mock_tokenizer_cls: type) -> None:
        config = PromptConfig(
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        gen = ProxyInspiredCodingContentGenerator(config, tokenizer)
        counts: dict[str, int] = {}
        for _ in range(400):
            name, _ = gen._make_tool_call()
            counts[name] = counts.get(name, 0) + 1
        assert counts.get("Read", 0) > counts.get("Glob", 0)
