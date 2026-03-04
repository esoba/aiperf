# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CodingContentGenerator."""

import pytest

from aiperf.common.config import PrefixPromptConfig, PromptConfig
from aiperf.common.exceptions import ConfigurationError, NotInitializedError
from aiperf.dataset.generator.coding_content import (
    _FILE_PATHS,
    _LANG_FILE_PATHS,
    _TOOL_POOL_BLOCK_COUNTS,
    CodingContentGenerator,
)


class TestCodingContentGeneratorInit:
    @pytest.fixture
    def config(self):
        return PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )

    @pytest.fixture
    def generator(self, config, mock_tokenizer_cls):
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_pools_built(self, generator):
        assert generator._text_pool is None
        assert len(generator._tool_pool) > 0

    def test_text_pool_lazy_build(self, generator):
        assert generator._text_pool is None
        pool = generator._ensure_text_pool()
        assert len(pool) > 0
        assert generator._text_pool is pool

    def test_tokenized_corpus_aliases_tool_pool(self, generator):
        assert generator._tokenized_corpus is generator._tool_pool

    def test_hash_id_corpus_rng_exists(self, generator):
        assert generator._hash_id_corpus_rng is not None

    def test_pool_scale(self, mock_tokenizer_cls):
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        gen_default = CodingContentGenerator(config, tokenizer)
        gen_2x = CodingContentGenerator(
            config, tokenizer, pool_tokens_target=20_000_000
        )
        assert gen_2x._pool_scale == pytest.approx(2.0)
        assert gen_default._pool_scale == pytest.approx(1.0)


class TestGenerate:
    @pytest.fixture
    def config(self):
        return PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )

    @pytest.fixture
    def generator(self, config, mock_tokenizer_cls):
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_generate_without_hash_ids(self, generator):
        result = generator.generate(mean=100, stddev=20)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_hash_ids(self, generator):
        result = generator.generate(mean=100, hash_ids=[1, 2], block_size=50)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_missing_mean_raises_value_error(self, generator):
        with pytest.raises(ValueError, match="mean must be provided"):
            generator.generate(hash_ids=[1, 2])

    def test_generate_empty_hash_ids_uses_normal_path(self, generator):
        result = generator.generate(mean=50, stddev=10, hash_ids=[])
        assert isinstance(result, str)


class TestGeneratePrompt:
    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_returns_decoded_string(self, generator):
        result = generator.generate_prompt(50)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_zero_tokens(self, generator):
        result = generator.generate_prompt(0)
        assert result == ""


class TestBuildTokenSequence:
    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_correct_total_length(self, generator):
        tokens = generator._build_token_sequence(100, [1, 2], 50)
        assert len(tokens) == 100

    def test_caches_per_hash_id(self, generator):
        generator._build_token_sequence(100, [10, 20], 50)
        assert 10 in generator._cache
        assert 20 in generator._cache

    def test_reuses_cache(self, generator):
        generator._build_token_sequence(100, [10, 20], 50)
        cached_10 = generator._cache[10]
        generator._build_token_sequence(100, [10, 30], 50)
        assert generator._cache[10] is cached_10

    def test_incompatible_params_raise_configuration_error(self, generator):
        with pytest.raises(ConfigurationError):
            generator._build_token_sequence(10, [1, 2, 3, 4, 5], 50)

    def test_deterministic_per_hash_id(self, mock_tokenizer_cls):
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        gen1 = CodingContentGenerator(config, tokenizer)
        gen2 = CodingContentGenerator(config, tokenizer)

        gen1._build_token_sequence(100, [42, 99], 50)
        gen2._build_token_sequence(100, [99, 42], 50)

        assert gen1._cache[42] == gen2._cache[42]
        assert gen1._cache[99] == gen2._cache[99]

    def test_uses_hash_id_corpus_rng(self, generator):
        generator._build_token_sequence(100, [7, 8], 50)
        assert 7 in generator._cache


class TestSampleTokens:
    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_empty_pool_raises(self, generator):
        with pytest.raises(NotInitializedError):
            generator._sample_tokens(10, [])

    def test_wraps_around_pool_boundary(self, generator):
        pool = [1, 2, 3, 4, 5]
        tokens = generator._sample_tokens(7, pool)
        assert len(tokens) == 7


class TestTemplateSmoke:
    """Smoke tests for all template generators."""

    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    @pytest.mark.parametrize("gen_name", list(_TOOL_POOL_BLOCK_COUNTS.keys()))
    def test_tool_pool_generators(self, generator, gen_name):
        gen_fn = getattr(generator, gen_name)
        result = gen_fn()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_gen_user_prompt(self, generator):
        result = generator._gen_user_prompt()
        assert isinstance(result, str)
        assert len(result) > 0


class TestMLTemplates:
    """Tests for ML-specific generators produce realistic ML content."""

    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_ml_training_contains_torch(self, generator):
        result = generator._gen_ml_training_code()
        assert "torch" in result

    def test_ml_inference_contains_generate(self, generator):
        result = generator._gen_ml_inference_code()
        assert "generate" in result
        assert "torch" in result

    def test_ml_config_contains_model_path(self, generator):
        result = generator._gen_ml_config()
        assert "model_name_or_path" in result

    def test_ml_training_log_contains_loss(self, generator):
        result = generator._gen_ml_training_log()
        assert "loss" in result

    def test_cuda_error_contains_cuda(self, generator):
        result = generator._gen_cuda_error()
        assert "CUDA" in result or "cuda" in result

    def test_sql_query_contains_sql_keywords(self, generator):
        result = generator._gen_sql_query()
        text_upper = result.upper()
        has_sql = any(
            kw in text_upper for kw in ("SELECT", "INSERT", "CREATE", "ALTER")
        )
        assert has_sql


class TestFilePool:
    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_language_specific_paths(self, generator):
        for lang in ("python", "go", "rust", "typescript"):
            pool = generator._file_pool(lang)
            assert pool is _LANG_FILE_PATHS[lang]

    def test_generic_paths(self, generator):
        assert generator._file_pool(None) is _FILE_PATHS
        assert generator._file_pool("unknown") is _FILE_PATHS


class TestCodingConversation:
    @pytest.fixture
    def generator(self, mock_tokenizer_cls):
        config = PromptConfig(
            mean=100,
            stddev=20,
            block_size=512,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        tokenizer = mock_tokenizer_cls.from_pretrained("gpt2")
        return CodingContentGenerator(config, tokenizer)

    def test_coding_conversation_has_role_markers(self, generator):
        result = generator._gen_coding_conversation()
        assert "[User]" in result
        assert "[Assistant]" in result

    def test_coding_conversation_has_tool_calls(self, generator):
        result = generator._gen_coding_conversation()
        assert "<tool_name>" in result

    @pytest.mark.parametrize(
        "pattern_name",
        [
            "_gen_conv_bugfix",
            "_gen_conv_review",
            "_gen_conv_feature",
            "_gen_conv_debug",
            "_gen_conv_qa",
            "_gen_conv_refactor",
            "_gen_conv_perf",
            "_gen_conv_cicd",
            "_gen_conv_ml_debug",
            "_gen_conv_test_write",
            "_gen_conv_migration",
            "_gen_conv_deploy",
            "_gen_conv_security",
            "_gen_conv_distributed",
            "_gen_conv_observability",
            "_gen_conv_db_optimize",
            "_gen_conv_architecture_review",
            "_gen_conv_incident_response",
        ],
    )
    def test_coding_conversation_patterns_all_produce_output(
        self, generator, pattern_name
    ):
        gen_fn = getattr(generator, pattern_name)
        result = gen_fn()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "[User]" in result
        assert "[Assistant]" in result

    @pytest.mark.parametrize(
        "pattern_name",
        ["_gen_conv_architecture_review", "_gen_conv_incident_response"],
    )
    def test_coding_conversation_deep_patterns_have_long_turns(
        self, generator, pattern_name
    ):
        gen_fn = getattr(generator, pattern_name)
        result = gen_fn()
        assert len(result) > 2000
