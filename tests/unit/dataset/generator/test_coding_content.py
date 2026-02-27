# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CodingContentGenerator."""

import pytest

from aiperf.common.config import PrefixPromptConfig, PromptConfig
from aiperf.common.exceptions import ConfigurationError, NotInitializedError
from aiperf.dataset.generator.coding_content import CodingContentGenerator


class TestCodingContentGenerator:
    """Tests for CodingContentGenerator."""

    @pytest.fixture
    def mock_tokenizer(self, mock_tokenizer_cls):
        return mock_tokenizer_cls.from_pretrained("gpt2")

    @pytest.fixture
    def config(self):
        return PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )

    @pytest.fixture
    def generator(self, config, mock_tokenizer):
        return CodingContentGenerator(config=config, tokenizer=mock_tokenizer)

    # -- Pool construction --

    def test_pools_built_with_nonzero_tokens(self, generator):
        assert len(generator._text_pool) > 0
        assert len(generator._tool_pool) > 0

    def test_tool_pool_larger_than_text_pool(self, generator):
        assert len(generator._tool_pool) > len(generator._text_pool)

    # -- generate_prompt --

    def test_generate_prompt_returns_string(self, generator):
        result = generator.generate_prompt(10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_prompt_zero_tokens(self, generator):
        result = generator.generate_prompt(0)
        assert result == ""

    # -- generate_typed_prompt --

    def test_generate_typed_prompt_text_vs_tool(self, generator):
        text_result = generator.generate_typed_prompt(50, "text")
        tool_result = generator.generate_typed_prompt(50, "tool_result")
        assert isinstance(text_result, str)
        assert isinstance(tool_result, str)
        assert len(text_result) > 0
        assert len(tool_result) > 0

    def test_generate_typed_prompt_defaults_to_tool_pool(self, generator):
        result = generator.generate_typed_prompt(10, "unknown_type")
        assert isinstance(result, str)
        assert len(result) > 0

    # -- _sample_tokens --

    def test_sample_tokens_exact_count(self, generator):
        tokens = generator._sample_tokens(20, generator._tool_pool)
        assert len(tokens) == 20

    def test_sample_tokens_wraps_at_boundary(self, generator):
        pool = generator._tool_pool
        pool_size = len(pool)
        # Force start near end to trigger wrap
        from unittest.mock import patch

        with patch.object(
            generator._corpus_rng, "randrange", return_value=pool_size - 3
        ):
            tokens = generator._sample_tokens(10, pool)
            assert len(tokens) == 10
            expected = pool[pool_size - 3 :] + pool[:7]
            assert tokens == expected

    def test_sample_tokens_empty_pool_raises(self, generator):
        with pytest.raises(NotInitializedError):
            generator._sample_tokens(5, [])

    # -- Determinism --

    def test_deterministic_output(self, config, mock_tokenizer):
        gen1 = CodingContentGenerator(config=config, tokenizer=mock_tokenizer)
        gen2 = CodingContentGenerator(config=config, tokenizer=mock_tokenizer)
        assert gen1._text_pool == gen2._text_pool
        assert gen1._tool_pool == gen2._tool_pool

    # -- Cache support --

    def test_cached_prompt_with_hash_ids(self, generator):
        result = generator._generate_cached_prompt(
            num_tokens=10, hash_ids=[1, 2], block_size=5
        )
        assert isinstance(result, str)
        assert 1 in generator._cache
        assert 2 in generator._cache

    def test_cache_reuse(self, generator):
        generator._generate_cached_prompt(10, [1, 2], 5)
        first_1 = generator._cache[1].copy()
        generator._generate_cached_prompt(10, [1, 2], 5)
        assert generator._cache[1] == first_1

    def test_cached_prompt_invalid_config_raises(self, generator):
        with pytest.raises(ConfigurationError):
            generator._generate_cached_prompt(10, [1, 2, 3], 5)

    # -- generate() interface --

    def test_generate_without_hash_ids(self, generator):
        result = generator.generate(mean=50, stddev=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_hash_ids(self, generator):
        result = generator.generate(mean=520, hash_ids=[1, 2], stddev=0)
        assert isinstance(result, str)

    # -- Structural plausibility --

    def test_structural_plausibility_python(self, generator):
        block = generator._gen_python_code()
        assert "class " in block
        assert "def " in block
        assert "import " in block

    def test_structural_plausibility_go(self, generator):
        block = generator._gen_go_code()
        assert "func " in block
        assert "package " in block

    def test_structural_plausibility_rust(self, generator):
        block = generator._gen_rust_code()
        assert "fn " in block
        assert "struct " in block
        assert "use " in block

    def test_structural_plausibility_diff(self, generator):
        block = generator._gen_git_diff()
        assert "diff --git" in block
        assert "---" in block
        assert "+++" in block

    def test_structural_plausibility_json(self, generator):
        block = generator._gen_json_response()
        assert '"status"' in block
        assert '"data"' in block

    def test_structural_plausibility_error(self, generator):
        block = generator._gen_error_traceback()
        assert any(kw in block for kw in ["Traceback", "panic:", "panicked", "Error:"])

    # -- Tool pool variety --

    def test_tool_pool_variety(self, generator):
        blocks = [
            generator._gen_python_code(),
            generator._gen_go_code(),
            generator._gen_git_diff(),
            generator._gen_json_response(),
        ]
        combined = "\n".join(blocks)
        patterns = ["class ", "func ", "diff --git", '"status"', "def "]
        found = [p for p in patterns if p in combined]
        assert len(found) >= 3, f"Expected variety in blocks, found only: {found}"

    # -- calculate_num_tokens --

    def test_calculate_num_tokens(self, generator):
        result = generator.calculate_num_tokens(mean=100, stddev=0)
        assert result == 100

    # -- Language pools --

    @pytest.mark.parametrize("language", ["python", "go", "rust", "typescript"])
    def test_build_language_pool_creates_pool(self, generator, language):
        pool = generator.build_language_pool(language)
        assert len(pool) > 0

    def test_language_pool_cached(self, generator):
        pool1 = generator.build_language_pool("python")
        pool2 = generator.build_language_pool("python")
        assert pool1 is pool2

    @pytest.mark.parametrize("language", ["python", "go", "rust", "typescript"])
    def test_generate_language_prompt_returns_string(self, generator, language):
        result = generator.generate_language_prompt(50, "tool_result", language)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_language_prompt_text_uses_text_pool(self, generator):
        result = generator.generate_language_prompt(50, "text", "python")
        assert isinstance(result, str)
        assert len(result) > 0

    # -- Pool scaling --

    def test_pool_scale_default(self, config, mock_tokenizer):
        gen = CodingContentGenerator(config=config, tokenizer=mock_tokenizer)
        assert gen._pool_scale == 1.0

    def test_pool_scale_increases_with_target(self, config, mock_tokenizer):
        gen = CodingContentGenerator(
            config=config, tokenizer=mock_tokenizer, pool_tokens_target=500_000
        )
        assert gen._pool_scale == 2.5

    def test_scaled_pool_is_larger(self, config, mock_tokenizer):
        gen_default = CodingContentGenerator(config=config, tokenizer=mock_tokenizer)
        gen_scaled = CodingContentGenerator(
            config=config, tokenizer=mock_tokenizer, pool_tokens_target=500_000
        )
        default_pool = gen_default.build_language_pool("python")
        scaled_pool = gen_scaled.build_language_pool("python")
        assert len(scaled_pool) > len(default_pool)

    # -- Language-specific generators --

    @pytest.mark.parametrize(
        "language,expected",
        [
            ("python", "Traceback"),
            ("go", "goroutine"),
            ("rust", "panicked"),
            ("typescript", "processTicksAndRejections"),
        ],
    )
    def test_gen_error_traceback_with_language(self, generator, language, expected):
        block = generator._gen_error_traceback(language=language)
        assert expected in block

    @pytest.mark.parametrize(
        "language,expected",
        [
            ("python", "test session starts"),
            ("go", "Test"),
            ("rust", "Compiling"),
            ("typescript", "test suites matching"),
        ],
    )
    def test_gen_test_output_with_language(self, generator, language, expected):
        block = generator._gen_test_output(language=language)
        assert expected in block

    @pytest.mark.parametrize(
        "language,expected_kinds",
        [
            ("python", {"yaml", "toml", "dockerfile"}),
            ("go", {"yaml", "makefile"}),
            ("rust", {"toml"}),
            ("typescript", {"yaml", "dockerfile"}),
        ],
    )
    def test_gen_config_file_with_language(self, generator, language, expected_kinds):
        seen = set()
        for _ in range(100):
            block = generator._gen_config_file(language=language)
            if "service:" in block or "logging:" in block:
                seen.add("yaml")
            elif "[project]" in block or "[dependencies]" in block:
                seen.add("toml")
            elif "FROM " in block:
                seen.add("dockerfile")
            elif ".PHONY" in block:
                seen.add("makefile")
        assert seen <= expected_kinds
        assert len(seen) > 0

    @pytest.mark.parametrize(
        "language,expected_tool",
        [
            ("python", "ruff"),
            ("go", "golangci-lint"),
            ("rust", "cargo clippy"),
            ("typescript", "eslint"),
        ],
    )
    def test_gen_cicd_output_with_language(self, generator, language, expected_tool):
        block = generator._gen_cicd_output(language=language)
        assert expected_tool in block

    @pytest.mark.parametrize(
        "language,expected_fence",
        [
            ("python", "```python"),
            ("go", "```go"),
            ("rust", "```rust"),
            ("typescript", "```typescript"),
        ],
    )
    def test_gen_markdown_doc_with_language(self, generator, language, expected_fence):
        block = generator._gen_markdown_doc(language=language)
        assert expected_fence in block
