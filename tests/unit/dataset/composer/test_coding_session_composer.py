# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.config import (
    CodingSessionConfig,
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.dataset.composer.coding_session import CodingSessionComposer


@pytest.fixture
def coding_session_config():
    return UserConfig(
        endpoint=EndpointConfig(model_names=["test_model"], streaming=True),
        loadgen=LoadGeneratorConfig(benchmark_duration=300),
        input=InputConfig(
            coding_session=CodingSessionConfig(
                enabled=True,
                num_sessions=5,
                system_prompt_tokens=100,
                new_tokens_mean=500,
                new_tokens_median=300,
                max_prompt_tokens=5000,
                initial_prefix_mean=1000,
                initial_prefix_median=800,
                generation_length_mean=100,
                generation_length_median=80,
                block_size=64,
                subagent_probability=0.0,
            ),
            prompt=PromptConfig(),
        ),
    )


class TestCodingSessionComposer:
    def test_create_dataset_returns_correct_num_sessions(
        self, coding_session_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(coding_session_config, mock_tokenizer)
        conversations = composer.create_dataset()
        assert len(conversations) == 5

    def test_session_turns_have_growing_input_tokens(
        self, coding_session_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(coding_session_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            # Filter to sequential turns only (parallel branches fork from parent context)
            sequential_tokens = [
                t.input_tokens for t in conv.turns if t.parallel_group is None
            ]
            assert len(sequential_tokens) >= 2
            for i in range(1, len(sequential_tokens)):
                assert sequential_tokens[i] >= sequential_tokens[i - 1]

    def test_session_retires_at_max_prompt_tokens(
        self, coding_session_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(coding_session_config, mock_tokenizer)
        conversations = composer.create_dataset()

        max_tokens = coding_session_config.input.coding_session.max_prompt_tokens
        for conv in conversations:
            for turn in conv.turns:
                assert turn.input_tokens <= max_tokens

    def test_hash_ids_grow_across_sequential_turns(
        self, coding_session_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(coding_session_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            sequential_hash_lens = [
                len(t.hash_ids) for t in conv.turns if t.parallel_group is None
            ]
            for i in range(1, len(sequential_hash_lens)):
                assert sequential_hash_lens[i] >= sequential_hash_lens[i - 1]

    def test_system_message_set_on_conversation(
        self, coding_session_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(coding_session_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            assert conv.system_message is not None
            assert len(conv.system_message) > 0

    def test_input_tokens_metadata_on_turns(
        self, coding_session_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(coding_session_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns:
                assert turn.input_tokens is not None
                assert turn.input_tokens > 0

    def test_deterministic_output(self, coding_session_config, mock_tokenizer):
        composer1 = CodingSessionComposer(coding_session_config, mock_tokenizer)
        result1 = composer1.create_dataset()

        composer2 = CodingSessionComposer(coding_session_config, mock_tokenizer)
        result2 = composer2.create_dataset()

        assert len(result1) == len(result2)
        for c1, c2 in zip(result1, result2, strict=True):
            assert len(c1.turns) == len(c2.turns)
            for t1, t2 in zip(c1.turns, c2.turns, strict=True):
                assert t1.input_tokens == t2.input_tokens
                assert t1.hash_ids == t2.hash_ids
                assert t1.max_tokens == t2.max_tokens

    def test_turns_have_text_content(self, coding_session_config, mock_tokenizer):
        composer = CodingSessionComposer(coding_session_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns:
                assert len(turn.texts) == 1
                assert len(turn.texts[0].contents) == 1
                assert len(turn.texts[0].contents[0]) > 0

    def test_turns_have_model_set(self, coding_session_config, mock_tokenizer):
        composer = CodingSessionComposer(coding_session_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns:
                assert turn.model == "test_model"

    def test_session_ids_are_unique(self, coding_session_config, mock_tokenizer):
        composer = CodingSessionComposer(coding_session_config, mock_tokenizer)
        conversations = composer.create_dataset()

        session_ids = [c.session_id for c in conversations]
        assert len(session_ids) == len(set(session_ids))


class TestCodingSessionLanguageAndContentType:
    """Tests for language affinity and content type distribution."""

    @pytest.fixture
    def make_config(self):
        def _make(**overrides):
            session_kwargs = {
                "enabled": True,
                "num_sessions": 10,
                "system_prompt_tokens": 100,
                "new_tokens_mean": 500,
                "new_tokens_median": 300,
                "max_prompt_tokens": 3000,
                "initial_prefix_mean": 500,
                "initial_prefix_median": 400,
                "generation_length_mean": 100,
                "generation_length_median": 80,
                "block_size": 64,
            }
            session_kwargs.update(overrides)
            return UserConfig(
                endpoint=EndpointConfig(model_names=["test_model"], streaming=True),
                loadgen=LoadGeneratorConfig(benchmark_duration=300),
                input=InputConfig(
                    coding_session=CodingSessionConfig(**session_kwargs),
                    prompt=PromptConfig(),
                ),
            )

        return _make

    def test_language_affinity_single(self, make_config, mock_tokenizer):
        """Single language mode uses that language's pool for all sessions."""
        config = make_config(language="go")
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()
        # The go pool should have been built, others should not
        gen = composer._content_generator
        assert "go" in gen._language_pools
        assert "python" not in gen._language_pools
        assert "rust" not in gen._language_pools

    def test_language_affinity_mixed(self, make_config, mock_tokenizer):
        """Mixed mode uses varied languages across sessions."""
        config = make_config(language="mixed", num_sessions=50)
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()
        gen = composer._content_generator
        assert len(gen._language_pools) >= 2

    def test_large_context_scales_pools(self, make_config, mock_tokenizer):
        """Large initial_prefix_mean causes pool auto-scaling."""
        config = make_config(
            initial_prefix_mean=500_000,
            initial_prefix_median=400_000,
            max_prompt_tokens=1_000_000,
            num_sessions=1,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        assert composer._content_generator._pool_scale > 1.0

    def test_content_type_distribution(self, make_config, mock_tokenizer):
        """With tool_result_ratio=0.9, ~90% of turns use tool_result content."""
        config = make_config(
            tool_result_ratio=0.5,
            language="python",
            num_sessions=20,
            max_prompt_tokens=5000,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)

        # Count content type selections via the RNG
        from unittest.mock import patch

        content_types = []
        original_select = composer._select_content_type

        def tracking_select(cfg):
            result = original_select(cfg)
            content_types.append(result)
            return result

        with patch.object(
            composer, "_select_content_type", side_effect=tracking_select
        ):
            composer.create_dataset()

        total = len(content_types)
        assert total > 20
        tool_ratio = sum(1 for ct in content_types if ct == "tool_result") / total
        # With ratio=0.5, expect roughly 50% (+/- 20% tolerance for small sample)
        assert 0.3 < tool_ratio < 0.7, (
            f"Expected ~50% tool_result, got {tool_ratio:.2%}"
        )
