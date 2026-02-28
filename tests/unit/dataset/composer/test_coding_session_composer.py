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
from aiperf.common.models.dataset_models import CacheLayerSizes
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


# ============================================================================
# Cache Layer Structure Tests
# ============================================================================


def _make_cache_config(**overrides):
    """Helper to build a UserConfig with cache layer defaults."""
    session_kwargs = {
        "enabled": True,
        "num_sessions": 3,
        "system_prompt_tokens": 100,
        "new_tokens_mean": 500,
        "new_tokens_median": 300,
        "max_prompt_tokens": 5000,
        "initial_prefix_mean": 1000,
        "initial_prefix_median": 800,
        "generation_length_mean": 100,
        "generation_length_median": 80,
        "block_size": 64,
        "l1_tokens": 640,
        "l2_tokens": 128,
        "subagent_probability": 0.0,
        "parallel_probability": 0.0,
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


class TestCacheLayerStructure:
    """Tests for L1/L2/L3 cache layer structure in generated sessions."""

    def test_all_turns_have_cache_layer_sizes(self, mock_tokenizer):
        config = _make_cache_config()
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns:
                assert turn.cache_layer_sizes is not None
                assert isinstance(turn.cache_layer_sizes, CacheLayerSizes)

    def test_l1_shared_across_sessions(self, mock_tokenizer):
        """L1 hash_ids are deterministic range(0, N), identical across sessions."""
        config = _make_cache_config(l1_tokens=640, block_size=64)
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        l1_count = 640 // 64  # 10 blocks
        for conv in conversations:
            first_turn = conv.turns[0]
            assert first_turn.cache_layer_sizes.l1 == l1_count
            # L1 IDs are range(0, 10)
            l1_ids = first_turn.hash_ids[:l1_count]
            assert l1_ids == list(range(l1_count))

    def test_l2_stable_within_session(self, mock_tokenizer):
        """L2 hash_ids remain the same across turns within a session."""
        config = _make_cache_config(
            l2_tokens=128, block_size=64, max_compressions=0, restart_probability=0.0
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            l1_count = conv.turns[0].cache_layer_sizes.l1
            l2_count = conv.turns[0].cache_layer_sizes.l2
            if l2_count == 0:
                continue
            l2_first = conv.turns[0].hash_ids[l1_count : l1_count + l2_count]
            for turn in conv.turns[1:]:
                if turn.parallel_group is not None:
                    continue
                l2_this = turn.hash_ids[l1_count : l1_count + l2_count]
                assert l2_this == l2_first

    def test_l3_grows_across_turns(self, mock_tokenizer):
        """L3 block count grows as conversation context accumulates."""
        config = _make_cache_config(max_compressions=0, restart_probability=0.0)
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            sequential_l3 = [
                t.cache_layer_sizes.l3 for t in conv.turns if t.parallel_group is None
            ]
            for i in range(1, len(sequential_l3)):
                assert sequential_l3[i] >= sequential_l3[i - 1]

    def test_layer_sizes_sum_to_hash_ids_length(self, mock_tokenizer):
        """L1+L2+L3 matches len(hash_ids) (thinking blocks may add extra)."""
        config = _make_cache_config(thinking_tokens_mean=0)
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns:
                sizes = turn.cache_layer_sizes
                expected = sizes.l1 + sizes.l2 + sizes.l3
                assert len(turn.hash_ids) == expected


class TestSessionRestart:
    """Tests for --continue restart behavior."""

    def test_restart_preserves_l1(self, mock_tokenizer):
        """After restart, L1 IDs remain unchanged."""
        config = _make_cache_config(
            restart_probability=1.0,
            l1_tokens=640,
            block_size=64,
            num_sessions=1,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        conv = conversations[0]
        l1_count = 640 // 64
        for turn in conv.turns:
            assert turn.hash_ids[:l1_count] == list(range(l1_count))

    def test_restart_regenerates_l2_l3(self, mock_tokenizer):
        """After restart, L2+L3 IDs change."""
        config = _make_cache_config(
            restart_probability=1.0,
            l1_tokens=640,
            l2_tokens=128,
            block_size=64,
            num_sessions=1,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        conv = conversations[0]
        if len(conv.turns) < 2:
            pytest.skip("Not enough turns for restart test")

        l1_count = 640 // 64
        l2_count = 128 // 64
        l2_t0 = conv.turns[0].hash_ids[l1_count : l1_count + l2_count]
        l2_t1 = conv.turns[1].hash_ids[l1_count : l1_count + l2_count]
        # With restart_probability=1.0, every turn regenerates L2
        assert l2_t0 != l2_t1

    def test_restart_disabled_by_default(self, mock_tokenizer):
        """With restart_probability=0, L2 stays stable across turns."""
        config = _make_cache_config(
            restart_probability=0.0,
            max_compressions=0,
            l1_tokens=640,
            l2_tokens=128,
            block_size=64,
            num_sessions=1,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        conv = conversations[0]
        l1_count = 640 // 64
        l2_count = 128 // 64
        l2_t0 = conv.turns[0].hash_ids[l1_count : l1_count + l2_count]
        for turn in conv.turns[1:]:
            if turn.parallel_group is not None:
                continue
            l2_this = turn.hash_ids[l1_count : l1_count + l2_count]
            assert l2_this == l2_t0


class TestContextCompression:
    """Tests for context compression events."""

    def test_compression_reduces_l3(self, mock_tokenizer):
        """Compression should reduce L3 block count."""
        config = _make_cache_config(
            max_compressions=3,
            compression_threshold=0.5,
            compression_ratio=0.3,
            max_prompt_tokens=5000,
            num_sessions=1,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        conv = conversations[0]
        l3_sizes = [
            t.cache_layer_sizes.l3 for t in conv.turns if t.parallel_group is None
        ]
        # After compression, L3 should drop
        has_drop = any(l3_sizes[i] < l3_sizes[i - 1] for i in range(1, len(l3_sizes)))
        assert has_drop, f"Expected L3 drop from compression, got {l3_sizes}"

    def test_max_compressions_respected(self, mock_tokenizer):
        """No more than max_compressions events occur."""
        config = _make_cache_config(
            max_compressions=1,
            compression_threshold=0.3,
            compression_ratio=0.3,
            max_prompt_tokens=5000,
            num_sessions=1,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        conv = conversations[0]
        l3_sizes = [
            t.cache_layer_sizes.l3 for t in conv.turns if t.parallel_group is None
        ]
        drops = sum(1 for i in range(1, len(l3_sizes)) if l3_sizes[i] < l3_sizes[i - 1])
        assert drops <= 1

    def test_compression_disabled_when_zero(self, mock_tokenizer):
        """max_compressions=0 disables compression."""
        config = _make_cache_config(
            max_compressions=0,
            compression_threshold=0.1,
            compression_ratio=0.3,
            num_sessions=1,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        conv = conversations[0]
        l3_sizes = [
            t.cache_layer_sizes.l3 for t in conv.turns if t.parallel_group is None
        ]
        for i in range(1, len(l3_sizes)):
            assert l3_sizes[i] >= l3_sizes[i - 1]

    def test_l1_preserved_during_compression(self, mock_tokenizer):
        """L1 IDs remain unchanged through compression events."""
        config = _make_cache_config(
            max_compressions=3,
            compression_threshold=0.3,
            compression_ratio=0.3,
            l1_tokens=640,
            block_size=64,
            num_sessions=1,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        l1_count = 640 // 64
        conv = conversations[0]
        for turn in conv.turns:
            assert turn.hash_ids[:l1_count] == list(range(l1_count))


class TestThinkingBlocks:
    """Tests for thinking block accumulation and stripping."""

    def test_thinking_accumulates_on_tool_result(self, mock_tokenizer):
        """With thinking enabled and all tool_result, hash_ids grow beyond L1+L2+L3."""
        config = _make_cache_config(
            thinking_tokens_mean=500,
            thinking_tokens_median=300,
            tool_result_ratio=1.0,
            thinking_strip_probability=0.0,
            num_sessions=1,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        conv = conversations[0]
        # At least some turns should have thinking blocks (hash_ids > L1+L2+L3)
        has_thinking = False
        for turn in conv.turns[1:]:
            if turn.parallel_group is not None:
                continue
            sizes = turn.cache_layer_sizes
            layer_total = sizes.l1 + sizes.l2 + sizes.l3
            if len(turn.hash_ids) > layer_total:
                has_thinking = True
                break
        assert has_thinking, "Expected thinking blocks to add extra hash_ids"

    def test_thinking_disabled_when_zero(self, mock_tokenizer):
        """thinking_tokens_mean=0 means no extra hash_ids beyond layers."""
        config = _make_cache_config(
            thinking_tokens_mean=0,
            thinking_tokens_median=0,
            num_sessions=1,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        conv = conversations[0]
        for turn in conv.turns:
            sizes = turn.cache_layer_sizes
            assert len(turn.hash_ids) == sizes.l1 + sizes.l2 + sizes.l3


class TestWorkingSetL1Sharing:
    """Tests for L1 sharing across sessions in working set counting."""

    def test_l1_blocks_shared_across_sessions(self, mock_tokenizer):
        """Multiple sessions should share the same L1 hash_ids."""
        config = _make_cache_config(
            l1_tokens=640,
            block_size=64,
            num_sessions=5,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        l1_count = 640 // 64
        all_l1_sets: list[set[int]] = []
        for conv in conversations:
            first_turn = conv.turns[0]
            l1_set = set(first_turn.hash_ids[:l1_count])
            all_l1_sets.append(l1_set)

        # All L1 sets should be identical
        for l1_set in all_l1_sets[1:]:
            assert l1_set == all_l1_sets[0]

    def test_working_set_dedup_with_l1(self, mock_tokenizer):
        """Union of all hash_ids across sessions should count L1 only once."""
        config = _make_cache_config(
            l1_tokens=640,
            block_size=64,
            num_sessions=10,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        # Compute naive (overcounting) and deduped working sets
        all_ids: set[int] = set()
        naive_total = 0
        for conv in conversations:
            first_turn = conv.turns[0]
            all_ids.update(first_turn.hash_ids)
            naive_total += len(first_turn.hash_ids)

        # Deduped should be strictly less than naive (L1 counted once vs N times)
        assert len(all_ids) < naive_total
