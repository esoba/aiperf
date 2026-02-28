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
        assert (1, 5) in generator._cache
        assert (2, 5) in generator._cache

    def test_cache_reuse(self, generator):
        generator._generate_cached_prompt(10, [1, 2], 5)
        first_1 = generator._cache[(1, 5)].copy()
        generator._generate_cached_prompt(10, [1, 2], 5)
        assert generator._cache[(1, 5)] == first_1

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
        blocks = [generator._gen_python_code() for _ in range(20)]
        combined = "\n".join(blocks)
        assert "def " in combined
        assert "import " in combined

    def test_structural_plausibility_go(self, generator):
        blocks = [generator._gen_go_code() for _ in range(20)]
        combined = "\n".join(blocks)
        assert "func " in combined or "func(" in combined
        assert "package " in combined

    def test_structural_plausibility_rust(self, generator):
        blocks = [generator._gen_rust_code() for _ in range(20)]
        combined = "\n".join(blocks)
        assert "fn " in combined
        assert "use " in combined

    def test_structural_plausibility_typescript(self, generator):
        blocks = [generator._gen_typescript_code() for _ in range(20)]
        combined = "\n".join(blocks)
        assert any(kw in combined for kw in ["import ", "export "])

    def test_structural_plausibility_diff(self, generator):
        block = generator._gen_git_diff()
        assert "diff --git" in block
        assert "---" in block
        assert "+++" in block

    def test_structural_plausibility_json(self, generator):
        blocks = [generator._gen_json_response() for _ in range(10)]
        combined = "\n".join(blocks)
        assert '"status"' in combined or '"data"' in combined or '"error"' in combined

    def test_structural_plausibility_error(self, generator):
        block = generator._gen_error_traceback()
        assert any(kw in block for kw in ["Traceback", "panic:", "panicked", "Error:"])

    # -- Variant diversity --

    def test_python_variant_diversity(self, generator):
        blocks = [generator._gen_python_code() for _ in range(50)]
        combined = "\n".join(blocks)
        patterns = ["class ", "pytest", "FastAPI", "BaseModel", "asynccontextmanager"]
        found = [p for p in patterns if p in combined]
        assert len(found) >= 2, f"Expected multiple Python patterns, found: {found}"

    def test_go_variant_diversity(self, generator):
        blocks = [generator._gen_go_code() for _ in range(50)]
        combined = "\n".join(blocks)
        patterns = ["struct {", "func Test", "errors.New", "json.NewDecoder"]
        found = [p for p in patterns if p in combined]
        assert len(found) >= 2, f"Expected multiple Go patterns, found: {found}"

    def test_rust_variant_diversity(self, generator):
        blocks = [generator._gen_rust_code() for _ in range(50)]
        combined = "\n".join(blocks)
        patterns = ["pub struct", "#[tokio::test]", "thiserror", "axum"]
        found = [p for p in patterns if p in combined]
        assert len(found) >= 2, f"Expected multiple Rust patterns, found: {found}"

    def test_typescript_variant_diversity(self, generator):
        blocks = [generator._gen_typescript_code() for _ in range(50)]
        combined = "\n".join(blocks)
        patterns = ["class ", "describe(", "z.object", "type "]
        found = [p for p in patterns if p in combined]
        assert len(found) >= 2, f"Expected multiple TS patterns, found: {found}"

    # -- Tool use blocks --

    def test_tool_use_block_structure(self, generator):
        blocks = [generator._gen_tool_use_block() for _ in range(20)]
        combined = "\n".join(blocks)
        assert "<tool_name>" in combined
        assert "</result>" in combined

    def test_tool_use_block_variant_diversity(self, generator):
        blocks = [generator._gen_tool_use_block() for _ in range(30)]
        combined = "\n".join(blocks)
        tools = ["read", "edit", "search", "bash"]
        found = [t for t in tools if f"<tool_name>{t}</tool_name>" in combined]
        assert len(found) >= 2, f"Expected multiple tool types, found: {found}"

    # -- Richer prompts --

    def test_user_prompt_average_length(self, generator):
        prompts = [generator._gen_user_prompt() for _ in range(100)]
        avg_len = sum(len(p) for p in prompts) / len(prompts)
        assert avg_len > 60, f"Average prompt length {avg_len:.0f} too short"

    def test_user_prompt_some_have_context(self, generator):
        prompts = [generator._gen_user_prompt() for _ in range(100)]
        multiline = [p for p in prompts if "\n\n" in p]
        assert len(multiline) >= 5, "Expected some prompts with context paragraphs"

    # -- Bash output variants --

    def test_bash_output_variant_diversity(self, generator):
        blocks = [generator._gen_bash_output() for _ in range(30)]
        combined = "\n".join(blocks)
        patterns = ["git checkout", "make build", "wc -l", "du -sh"]
        found = [p for p in patterns if p in combined]
        assert len(found) >= 2, f"Expected multiple bash patterns, found: {found}"

    # -- JSON response variants --

    def test_json_response_variant_diversity(self, generator):
        blocks = [generator._gen_json_response() for _ in range(30)]
        combined = "\n".join(blocks)
        patterns = ['"pagination"', '"error"', '"items"']
        found = [p for p in patterns if p in combined]
        assert len(found) >= 2, f"Expected multiple JSON shapes, found: {found}"

    # -- Tool pool variety --

    def test_tool_pool_variety(self, generator):
        blocks = [
            generator._gen_python_code(),
            generator._gen_go_code(),
            generator._gen_git_diff(),
            generator._gen_json_response(),
        ]
        combined = "\n".join(blocks)
        patterns = ["def ", "func ", "diff --git", '"status"', '"data"', '"error"']
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

    def test_gen_markdown_doc_has_config_and_errors(self, generator):
        block = generator._gen_markdown_doc(language="python")
        assert "## Configuration" in block
        assert "## Errors" in block

    # -- Git diff expanded --

    def test_gen_git_diff_has_commit_header(self, generator):
        block = generator._gen_git_diff()
        assert "commit " in block
        assert "Author:" in block

    # -- Language-aware generators --

    @pytest.mark.parametrize(
        "language,expected_path_fragment",
        [
            ("python", ".py"),
            ("go", ".go"),
            ("rust", ".rs"),
            ("typescript", ".ts"),
        ],
    )
    def test_tool_read_uses_language_files(
        self, generator, language, expected_path_fragment
    ):
        blocks = [generator._gen_tool_read(language=language) for _ in range(10)]
        combined = "\n".join(blocks)
        assert expected_path_fragment in combined

    @pytest.mark.parametrize(
        "language,expected_syntax",
        [
            ("python", "def "),
            ("go", "func "),
            ("rust", "fn "),
            ("typescript", "async "),
        ],
    )
    def test_tool_read_uses_language_syntax(self, generator, language, expected_syntax):
        blocks = [generator._gen_tool_read(language=language) for _ in range(10)]
        combined = "\n".join(blocks)
        assert expected_syntax in combined

    @pytest.mark.parametrize(
        "language,expected_path_fragment",
        [
            ("python", ".py"),
            ("go", ".go"),
            ("rust", ".rs"),
            ("typescript", ".ts"),
        ],
    )
    def test_tool_edit_uses_language_files(
        self, generator, language, expected_path_fragment
    ):
        blocks = [generator._gen_tool_edit(language=language) for _ in range(10)]
        combined = "\n".join(blocks)
        assert expected_path_fragment in combined

    @pytest.mark.parametrize(
        "language,expected_pattern",
        [
            ("python", "def "),
            ("go", "func "),
            ("rust", "fn "),
            ("typescript", "interface "),
        ],
    )
    def test_tool_search_uses_language_patterns(
        self, generator, language, expected_pattern
    ):
        blocks = [generator._gen_tool_search(language=language) for _ in range(20)]
        combined = "\n".join(blocks)
        assert expected_pattern in combined

    @pytest.mark.parametrize(
        "language,expected_cmd",
        [
            ("python", "pytest"),
            ("go", "go test"),
            ("rust", "cargo test"),
            ("typescript", "vitest"),
        ],
    )
    def test_tool_bash_uses_language_commands(self, generator, language, expected_cmd):
        block = generator._gen_tool_bash(language=language)
        assert expected_cmd in block

    @pytest.mark.parametrize(
        "language,expected_path_fragment",
        [
            ("python", ".py"),
            ("go", ".go"),
            ("rust", ".rs"),
            ("typescript", ".ts"),
        ],
    )
    def test_bash_output_uses_language_files(
        self, generator, language, expected_path_fragment
    ):
        blocks = [generator._gen_bash_output(language=language) for _ in range(20)]
        combined = "\n".join(blocks)
        assert expected_path_fragment in combined

    @pytest.mark.parametrize(
        "language,expected_path_fragment",
        [
            ("python", ".py"),
            ("go", ".go"),
            ("rust", ".rs"),
            ("typescript", ".ts"),
        ],
    )
    def test_git_diff_uses_language_files(
        self, generator, language, expected_path_fragment
    ):
        block = generator._gen_git_diff(language=language)
        assert expected_path_fragment in block
