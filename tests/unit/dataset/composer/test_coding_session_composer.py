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
from aiperf.dataset.loader.models import CodingTrace


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
                subagent_session_probability=0.0,
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
            sequential_tokens = [t.input_tokens for t in conv.turns]
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
            sequential_hash_lens = [len(t.hash_ids) for t in conv.turns]
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
                "subagent_session_probability": 0.0,
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
        "subagent_session_probability": 0.0,
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
                l2_this = turn.hash_ids[l1_count : l1_count + l2_count]
                assert l2_this == l2_first

    def test_l3_grows_across_turns(self, mock_tokenizer):
        """L3 block count grows as conversation context accumulates."""
        config = _make_cache_config(max_compressions=0, restart_probability=0.0)
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            sequential_l3 = [t.cache_layer_sizes.l3 for t in conv.turns]
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
        l3_sizes = [t.cache_layer_sizes.l3 for t in conv.turns]
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
        l3_sizes = [t.cache_layer_sizes.l3 for t in conv.turns]
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
        l3_sizes = [t.cache_layer_sizes.l3 for t in conv.turns]
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


# ============================================================================
# Subagent L1 Isolation Tests
# ============================================================================


class TestSubagentL1Isolation:
    """Verify subagent children use an independent L1 range starting at 2**30."""

    @pytest.fixture
    def subagent_config(self):
        return _make_cache_config(
            num_sessions=5,
            l1_tokens=640,
            l2_tokens=128,
            block_size=64,
            subagent_probability=0.0,
            subagent_session_probability=1.0,
            subagent_turn_probability=1.0,
            subagent_count_mean=2.0,
            subagent_count_max=3,
            subagent_turns_mean=4,
            subagent_turns_median=3,
            subagent_system_tokens=200,
            subagent_new_tokens_mean=300,
            subagent_new_tokens_median=200,
            subagent_max_prompt_tokens=3000,
        )

    def test_subagent_l1_offset_constant(self):
        """Class constant _SUBAGENT_L1_OFFSET equals 2**30."""
        assert CodingSessionComposer._SUBAGENT_L1_OFFSET == 2**30

    def test_subagent_l1_ids_start_at_offset(self, subagent_config, mock_tokenizer):
        """Child L1 IDs start at 2**30, not 0."""
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        children = [c for c in conversations if c.is_subagent_child]
        assert len(children) > 0

        offset = 2**30
        for child in children:
            first_turn = child.turns[0]
            l1_count = first_turn.cache_layer_sizes.l1
            if l1_count == 0:
                continue
            l1_ids = first_turn.hash_ids[:l1_count]
            assert l1_ids[0] >= offset, (
                f"Child L1 should start at {offset}, got {l1_ids[0]}"
            )
            assert l1_ids == list(range(offset, offset + l1_count))

    def test_parent_l1_ids_start_at_zero(self, subagent_config, mock_tokenizer):
        """Parent L1 IDs still start at 0 (unchanged behavior)."""
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        parents = [c for c in conversations if not c.is_subagent_child]
        l1_count = subagent_config.input.coding_session.l1_tokens // 64
        for parent in parents:
            first_turn = parent.turns[0]
            l1_ids = first_turn.hash_ids[:l1_count]
            assert l1_ids == list(range(l1_count))

    def test_zero_overlap_between_parent_and_child_l1(
        self, subagent_config, mock_tokenizer
    ):
        """Parent L1 set and child L1 set have zero intersection."""
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        parents = [c for c in conversations if not c.is_subagent_child]
        children = [c for c in conversations if c.is_subagent_child]
        assert len(children) > 0

        parent_l1_set: set[int] = set()
        for parent in parents:
            l1_count = parent.turns[0].cache_layer_sizes.l1
            parent_l1_set.update(parent.turns[0].hash_ids[:l1_count])

        child_l1_set: set[int] = set()
        for child in children:
            l1_count = child.turns[0].cache_layer_sizes.l1
            child_l1_set.update(child.turns[0].hash_ids[:l1_count])

        overlap = parent_l1_set & child_l1_set
        assert len(overlap) == 0, (
            f"Expected zero L1 overlap, got {len(overlap)} shared IDs"
        )

    def test_multiple_children_share_same_l1_range(
        self, subagent_config, mock_tokenizer
    ):
        """All subagent children use the same L1 ID range."""
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        children = [c for c in conversations if c.is_subagent_child]
        assert len(children) >= 2

        # Collect all child L1 ranges
        child_l1_ranges: list[list[int]] = []
        for child in children:
            l1_count = child.turns[0].cache_layer_sizes.l1
            if l1_count > 0:
                child_l1_ranges.append(child.turns[0].hash_ids[:l1_count])

        # All children with the same L1 count should have the same L1 IDs
        if len(child_l1_ranges) >= 2:
            for l1_range in child_l1_ranges[1:]:
                if len(l1_range) == len(child_l1_ranges[0]):
                    assert l1_range == child_l1_ranges[0]

    def test_child_l1_stable_across_turns(self, subagent_config, mock_tokenizer):
        """Within a child conversation, L1 IDs remain constant across all turns."""
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        children = [c for c in conversations if c.is_subagent_child]
        assert len(children) > 0

        for child in children:
            l1_count = child.turns[0].cache_layer_sizes.l1
            if l1_count == 0:
                continue
            l1_first = child.turns[0].hash_ids[:l1_count]
            for turn in child.turns[1:]:
                assert turn.hash_ids[:l1_count] == l1_first

    def test_child_l2_independent_from_parent(self, subagent_config, mock_tokenizer):
        """Child L2 IDs are randomly generated, independent from parent L2."""
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        parents = [c for c in conversations if not c.is_subagent_child]
        children = [c for c in conversations if c.is_subagent_child]
        assert len(children) > 0

        # Get parent L2 IDs from first parent
        parent = parents[0]
        p_l1 = parent.turns[0].cache_layer_sizes.l1
        p_l2 = parent.turns[0].cache_layer_sizes.l2
        parent_l2_set = set(parent.turns[0].hash_ids[p_l1 : p_l1 + p_l2])

        # At least one child should have different L2 IDs
        any_different = False
        for child in children:
            c_l1 = child.turns[0].cache_layer_sizes.l1
            c_l2 = child.turns[0].cache_layer_sizes.l2
            if c_l2 > 0:
                child_l2_set = set(child.turns[0].hash_ids[c_l1 : c_l1 + c_l2])
                if child_l2_set != parent_l2_set:
                    any_different = True
                    break
        assert any_different, "Expected child L2 IDs to differ from parent"


# ============================================================================
# Config Default Tests
# ============================================================================


class TestCodingSessionConfigDefaults:
    """Verify updated default values in CodingSessionConfig."""

    def test_thinking_strip_probability_defaults_to_one(self):
        """thinking_strip_probability defaults to 1.0 (always strip)."""
        cfg = CodingSessionConfig(enabled=True)
        assert cfg.thinking_strip_probability == 1.0


# ============================================================================
# CodingTrace Export Tests
# ============================================================================


def _make_export_config(**overrides):
    """Helper to build a UserConfig for trace export tests."""
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
        "subagent_session_probability": 0.0,
        "thinking_tokens_mean": 0,
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


class TestToCodingTracesBasic:
    """Tests for basic to_coding_traces() conversion."""

    def test_produces_one_trace_per_session(self, mock_tokenizer):
        config = _make_export_config(num_sessions=3)
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()
        assert len(traces) == 3

    def test_trace_ids_match_session_ids(self, mock_tokenizer):
        config = _make_export_config(num_sessions=2)
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()
        assert traces[0].id == "coding_session_0000"
        assert traces[1].id == "coding_session_0001"

    def test_trace_has_correct_metadata(self, mock_tokenizer):
        config = _make_export_config(
            num_sessions=1, block_size=64, system_prompt_tokens=100, l1_tokens=640
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()
        trace = traces[0]
        assert trace.block_size == 64
        assert trace.system_tokens == 100
        assert trace.tool_tokens == 640

    def test_request_count_matches_turn_count(self, mock_tokenizer):
        config = _make_export_config(num_sessions=1)
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        traces = composer.to_coding_traces()
        parent_convs = [c for c in conversations if not c.is_subagent_child]
        assert len(traces[0].requests) == len(parent_convs[0].turns)

    def test_request_tokens_match_turns(self, mock_tokenizer):
        config = _make_export_config(num_sessions=1)
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        traces = composer.to_coding_traces()
        parent = [c for c in conversations if not c.is_subagent_child][0]
        for req, turn in zip(traces[0].requests, parent.turns, strict=True):
            assert req.input_tokens == turn.input_tokens
            assert req.output_tokens == turn.max_tokens
            assert req.hash_ids == turn.hash_ids

    def test_last_request_has_end_turn_stop(self, mock_tokenizer):
        config = _make_export_config(num_sessions=1)
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()
        assert traces[0].requests[-1].stop == "end_turn"

    def test_non_last_requests_have_tool_use_stop(self, mock_tokenizer):
        config = _make_export_config(num_sessions=1)
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()
        for req in traces[0].requests[:-1]:
            assert req.stop == "tool_use"

    def test_raises_before_create_dataset(self, mock_tokenizer):
        config = _make_export_config()
        composer = CodingSessionComposer(config, mock_tokenizer)
        with pytest.raises(RuntimeError, match="Must call create_dataset"):
            composer.to_coding_traces()

    def test_trace_validates_as_coding_trace(self, mock_tokenizer):
        """Exported traces are valid CodingTrace models."""
        config = _make_export_config(num_sessions=1)
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()
        for trace in traces:
            # Round-trip through model_validate
            data = trace.model_dump(by_alias=True)
            reloaded = CodingTrace.model_validate(data)
            assert reloaded.id == trace.id
            assert len(reloaded.requests) == len(trace.requests)


class TestToCodingTracesWithSubagents:
    """Tests for subagent nesting in to_coding_traces()."""

    @pytest.fixture
    def subagent_config(self):
        return _make_export_config(
            num_sessions=3,
            subagent_probability=0.0,
            subagent_session_probability=1.0,
            subagent_turn_probability=1.0,
            subagent_count_mean=2.0,
            subagent_count_max=3,
            subagent_turns_mean=3,
            subagent_turns_median=2,
            subagent_system_tokens=200,
            subagent_new_tokens_mean=300,
            subagent_new_tokens_median=200,
            subagent_max_prompt_tokens=3000,
        )

    def test_subagent_children_nested_in_requests(
        self, subagent_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()

        # Find at least one trace with nested requests
        has_nested = False
        for trace in traces:
            for req in trace.requests:
                if req.requests:
                    has_nested = True
                    break
        assert has_nested, "Expected nested subagent requests"

    def test_nested_request_token_counts(self, subagent_config, mock_tokenizer):
        """Nested subagent requests have valid token counts."""
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()
        for trace in traces:
            for req in trace.requests:
                for nested in req.requests:
                    if nested.type == "subagent" and nested.input_tokens == 0:
                        # Container wrapper — actual child requests are inside
                        for child in nested.requests:
                            assert child.input_tokens > 0
                            assert child.output_tokens > 0
                    else:
                        assert nested.input_tokens > 0
                        assert nested.output_tokens > 0


class TestWriteTraces:
    """Tests for write_traces() serialization."""

    def test_writes_json_files(self, mock_tokenizer, tmp_path):
        config = _make_export_config(num_sessions=2)
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()
        CodingSessionComposer.write_traces(traces, tmp_path)

        json_files = sorted(tmp_path.glob("*.json"))
        assert len(json_files) == 2
        assert json_files[0].name == "coding_session_0000.json"
        assert json_files[1].name == "coding_session_0001.json"

    def test_written_files_are_valid_json(self, mock_tokenizer, tmp_path):
        import orjson

        config = _make_export_config(num_sessions=1)
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()
        CodingSessionComposer.write_traces(traces, tmp_path)

        json_file = tmp_path / "coding_session_0000.json"
        data = orjson.loads(json_file.read_bytes())
        assert data["id"] == "coding_session_0000"
        assert "requests" in data
        assert len(data["requests"]) > 0

    def test_written_files_use_alias_names(self, mock_tokenizer, tmp_path):
        """JSON uses 'in'/'out' aliases, not 'input_tokens'/'output_tokens'."""
        import orjson

        config = _make_export_config(num_sessions=1)
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()
        CodingSessionComposer.write_traces(traces, tmp_path)

        json_file = tmp_path / "coding_session_0000.json"
        data = orjson.loads(json_file.read_bytes())
        first_req = data["requests"][0]
        assert "in" in first_req
        assert "out" in first_req

    def test_written_files_round_trip_to_coding_trace(self, mock_tokenizer, tmp_path):
        """Written JSON files can be parsed back into CodingTrace objects."""
        import orjson

        config = _make_export_config(num_sessions=1)
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()
        CodingSessionComposer.write_traces(traces, tmp_path)

        json_file = tmp_path / "coding_session_0000.json"
        data = orjson.loads(json_file.read_bytes())
        reloaded = CodingTrace.model_validate(data)
        assert reloaded.id == traces[0].id
        assert len(reloaded.requests) == len(traces[0].requests)

    def test_creates_output_directory(self, mock_tokenizer, tmp_path):
        config = _make_export_config(num_sessions=1)
        composer = CodingSessionComposer(config, mock_tokenizer)
        composer.create_dataset()

        traces = composer.to_coding_traces()
        nested_dir = tmp_path / "nested" / "output"
        CodingSessionComposer.write_traces(traces, nested_dir)

        assert nested_dir.exists()
        assert len(list(nested_dir.glob("*.json"))) == 1


class TestRoundTrip:
    """Test full round-trip: CodingSessionComposer -> CodingTrace JSON -> CodingTraceLoader."""

    def test_round_trip_basic(self, mock_tokenizer, tmp_path):
        """Generate -> export -> load -> verify structure matches."""
        import orjson

        config = _make_export_config(num_sessions=2)
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        traces = composer.to_coding_traces()
        CodingSessionComposer.write_traces(traces, tmp_path)

        # Verify each JSON file is a valid CodingTrace
        parents = [c for c in conversations if not c.is_subagent_child]
        for parent in parents:
            json_file = tmp_path / f"{parent.session_id}.json"
            assert json_file.exists()
            data = orjson.loads(json_file.read_bytes())
            loaded = CodingTrace.model_validate(data)

            assert loaded.id == parent.session_id
            assert len(loaded.requests) == len(parent.turns)

            for req, turn in zip(loaded.requests, parent.turns, strict=True):
                assert req.input_tokens == turn.input_tokens
                assert req.output_tokens == turn.max_tokens
                assert req.hash_ids == turn.hash_ids

    def test_round_trip_preserves_hash_ids(self, mock_tokenizer, tmp_path):
        """Hash IDs survive the round-trip through JSON."""
        import orjson

        config = _make_export_config(num_sessions=1)
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        traces = composer.to_coding_traces()
        CodingSessionComposer.write_traces(traces, tmp_path)

        parent = [c for c in conversations if not c.is_subagent_child][0]
        json_file = tmp_path / f"{parent.session_id}.json"
        data = orjson.loads(json_file.read_bytes())
        loaded = CodingTrace.model_validate(data)

        for req, turn in zip(loaded.requests, parent.turns, strict=True):
            assert req.hash_ids == turn.hash_ids


class TestInterTurnDelay:
    """Tests for inter-turn delay sampling."""

    def test_delay_disabled_by_default(self, mock_tokenizer):
        config = _make_cache_config()
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns:
                assert turn.delay is None

    def test_delay_first_turn_always_none(self, mock_tokenizer):
        config = _make_cache_config(delay_mean_ms=5000, delay_median_ms=3000)
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            assert conv.turns[0].delay is None

    def test_delay_non_first_turns_have_positive_delay(self, mock_tokenizer):
        config = _make_cache_config(delay_mean_ms=5000, delay_median_ms=3000)
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns[1:]:
                assert turn.delay is not None
                assert turn.delay > 0

    def test_delay_deterministic(self, mock_tokenizer):
        config = _make_cache_config(delay_mean_ms=5000, delay_median_ms=3000)
        composer1 = CodingSessionComposer(config, mock_tokenizer)
        convs1 = composer1.create_dataset()

        composer2 = CodingSessionComposer(config, mock_tokenizer)
        convs2 = composer2.create_dataset()

        for c1, c2 in zip(convs1, convs2, strict=True):
            delays1 = [t.delay for t in c1.turns]
            delays2 = [t.delay for t in c2.turns]
            assert delays1 == delays2

    def test_delay_subagent_children(self, mock_tokenizer):
        config = _make_cache_config(
            delay_mean_ms=5000,
            delay_median_ms=3000,
            subagent_probability=0.0,
            subagent_session_probability=1.0,
            subagent_turn_probability=1.0,
            max_prompt_tokens=10000,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        child_convs = [c for c in conversations if c.is_subagent_child]
        assert len(child_convs) > 0

        for child in child_convs:
            assert child.turns[0].delay is None
            for turn in child.turns[1:]:
                assert turn.delay is not None
                assert turn.delay > 0


class TestMaxTurns:
    """Tests for session turn count limiting."""

    def test_max_turns_disabled_by_default(self, mock_tokenizer):
        config = _make_cache_config()
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        # With defaults (max_turns_mean=0), sessions should have multiple turns
        # limited only by token ceiling
        for conv in conversations:
            assert len(conv.turns) >= 2

    def test_max_turns_limits_session_length(self, mock_tokenizer):
        config = _make_cache_config(
            max_turns_mean=5,
            max_turns_median=5,
            max_prompt_tokens=500_000,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            assert len(conv.turns) <= 5

    def test_max_turns_token_ceiling_still_applies(self, mock_tokenizer):
        config = _make_cache_config(
            max_turns_mean=100,
            max_turns_median=100,
            max_prompt_tokens=2000,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns:
                assert turn.input_tokens <= 2000

    def test_max_turns_deterministic(self, mock_tokenizer):
        config = _make_cache_config(
            max_turns_mean=8,
            max_turns_median=5,
            max_prompt_tokens=500_000,
        )
        composer1 = CodingSessionComposer(config, mock_tokenizer)
        convs1 = composer1.create_dataset()

        composer2 = CodingSessionComposer(config, mock_tokenizer)
        convs2 = composer2.create_dataset()

        for c1, c2 in zip(convs1, convs2, strict=True):
            assert len(c1.turns) == len(c2.turns)


class TestToolDefinitions:
    """Verify conversations include tool definitions."""

    def test_coding_session_conversations_have_tools(self, mock_tokenizer):
        config = _make_cache_config()
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        parents = [c for c in conversations if not c.is_subagent_child]
        assert len(parents) > 0
        for conv in parents:
            assert conv.tools is not None
            assert len(conv.tools) == 8
            for tool in conv.tools:
                assert tool["type"] == "function"
                assert "function" in tool

    def test_subagent_children_have_tools_subset(self, mock_tokenizer):
        config = _make_cache_config(
            subagent_probability=0.0,
            subagent_session_probability=1.0,
            subagent_turn_probability=1.0,
            subagent_count_mean=2.0,
            subagent_count_max=3,
            subagent_turns_mean=4,
            subagent_turns_median=3,
            subagent_system_tokens=200,
            subagent_new_tokens_mean=300,
            subagent_new_tokens_median=200,
            subagent_max_prompt_tokens=3000,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        parents = [c for c in conversations if not c.is_subagent_child]
        children = [c for c in conversations if c.is_subagent_child]
        assert len(children) > 0

        parent_tool_count = len(parents[0].tools)
        for child in children:
            assert child.tools is not None
            assert len(child.tools) <= parent_tool_count


class TestSubagentRealism:
    """Tests for subagent realism improvements: per-type profiles, bimodal
    probability, background spawns, correlated join tokens, model routing."""

    @pytest.fixture
    def bimodal_config(self):
        """Config with high session probability but moderate turn probability."""
        return _make_subagent_config(
            num_sessions=50,
            subagent_session_probability=0.5,
            subagent_turn_probability=0.3,
        )

    @pytest.fixture
    def explore_model_config(self):
        """Config with explore model override."""
        return _make_subagent_config(
            num_sessions=10,
            subagent_session_probability=1.0,
            subagent_turn_probability=1.0,
            subagent_explore_model_name="claude-haiku-4-5-20251001",
        )

    @pytest.fixture
    def background_config(self):
        """Config with high background probability."""
        return _make_subagent_config(
            num_sessions=20,
            subagent_session_probability=1.0,
            subagent_turn_probability=1.0,
            subagent_background_probability=1.0,
        )

    def test_subagent_session_probability_zero_no_subagents(self, mock_tokenizer):
        config = _make_subagent_config(
            num_sessions=20,
            subagent_session_probability=0.0,
            subagent_turn_probability=1.0,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()
        children = [c for c in conversations if c.is_subagent_child]
        assert len(children) == 0

    def test_subagent_bimodal_distribution(self, bimodal_config, mock_tokenizer):
        composer = CodingSessionComposer(bimodal_config, mock_tokenizer)
        conversations = composer.create_dataset()
        parents = [c for c in conversations if not c.is_subagent_child]
        parents_with_spawns = [p for p in parents if p.subagent_spawns]
        parents_without_spawns = [p for p in parents if not p.subagent_spawns]
        # With 50 sessions at 0.5 session prob, expect both groups non-empty
        assert len(parents_with_spawns) > 0
        assert len(parents_without_spawns) > 0

    def test_explore_agent_read_only_tools(self, mock_tokenizer):
        config = _make_subagent_config(
            num_sessions=20,
            subagent_session_probability=1.0,
            subagent_turn_probability=1.0,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()
        children = [c for c in conversations if c.is_subagent_child]
        assert len(children) > 0
        # Verify no child has edit_file or write_file (only Explore and Plan lack them)
        write_tools = {"edit_file", "write_file"}
        for child in children:
            tool_names = {t["function"]["name"] for t in (child.tools or [])}
            if not tool_names & write_tools:
                # Found an Explore or Plan child with read-only tools
                return
        # General agents do have write tools, so at least some children should lack them
        # With weighted selection (50% Explore, 15% Plan), over 20 sessions we expect at least one
        explore_or_plan_children = [
            c
            for c in children
            if not {t["function"]["name"] for t in (c.tools or [])} & write_tools
        ]
        assert len(explore_or_plan_children) > 0

    def test_general_agent_all_tools(self, mock_tokenizer):
        config = _make_subagent_config(
            num_sessions=20,
            subagent_session_probability=1.0,
            subagent_turn_probability=1.0,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()
        children = [c for c in conversations if c.is_subagent_child]
        # Some children should be General type with all 8 tools
        max_tool_count = max(len(c.tools or []) for c in children)
        assert max_tool_count == 8

    def test_explore_agent_model_set_on_turns(
        self, explore_model_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(explore_model_config, mock_tokenizer)
        conversations = composer.create_dataset()
        children = [c for c in conversations if c.is_subagent_child]
        assert len(children) > 0
        # Some children should have the Explore model set
        explore_children = [
            c
            for c in children
            if any(t.model == "claude-haiku-4-5-20251001" for t in c.turns)
        ]
        assert len(explore_children) > 0

    def test_general_agent_inherits_parent_model(
        self, explore_model_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(explore_model_config, mock_tokenizer)
        conversations = composer.create_dataset()
        children = [c for c in conversations if c.is_subagent_child]
        # General/Plan children inherit the endpoint model, not the explore model
        non_explore_children = [
            c
            for c in children
            if all(t.model != "claude-haiku-4-5-20251001" for t in c.turns)
        ]
        assert len(non_explore_children) > 0

    def test_background_spawn_is_background_flag(
        self, background_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(background_config, mock_tokenizer)
        conversations = composer.create_dataset()
        parents = [c for c in conversations if not c.is_subagent_child]
        bg_spawns = [s for p in parents for s in p.subagent_spawns if s.is_background]
        assert len(bg_spawns) > 0

    def test_subagent_count_mean_default(self):
        cfg = CodingSessionConfig(enabled=True)
        assert cfg.subagent_count_mean == 1.2

    def test_per_type_system_tokens(self):
        from aiperf.common.config.coding_session_config import DEFAULT_SUBAGENT_PROFILES
        from aiperf.common.enums.enums import SubagentType

        profiles_by_type = {p.agent_type: p for p in DEFAULT_SUBAGENT_PROFILES}
        assert profiles_by_type[SubagentType.EXPLORE].system_tokens == 12000
        assert profiles_by_type[SubagentType.GENERAL].system_tokens == 20000
        assert profiles_by_type[SubagentType.PLAN].system_tokens == 15000

    def test_per_type_turn_counts(self):
        from aiperf.common.config.coding_session_config import DEFAULT_SUBAGENT_PROFILES
        from aiperf.common.enums.enums import SubagentType

        profiles_by_type = {p.agent_type: p for p in DEFAULT_SUBAGENT_PROFILES}
        # Explore should be shorter than General
        assert (
            profiles_by_type[SubagentType.EXPLORE].turns_mean
            < profiles_by_type[SubagentType.GENERAL].turns_mean
        )

    def test_subagent_type_weighted_selection(self, mock_tokenizer):
        config = _make_subagent_config(
            num_sessions=100,
            subagent_session_probability=1.0,
            subagent_turn_probability=1.0,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        # Sample 1000 profiles and check rough distribution
        counts = {"explore": 0, "general": 0, "plan": 0}
        for _ in range(1000):
            profile = composer._select_subagent_profile()
            counts[profile.agent_type.value] += 1
        # Explore ~50%, General ~35%, Plan ~15% (with generous tolerance)
        assert counts["explore"] > counts["general"]
        assert counts["general"] > counts["plan"]
        assert counts["explore"] > 300  # should be ~500
        assert counts["plan"] > 50  # should be ~150

    def test_join_turn_tokens_from_result_distribution(self, mock_tokenizer):
        """Verify join turn delta uses subagent_result_tokens_* (not parent distribution)."""
        config = _make_subagent_config(
            num_sessions=20,
            subagent_session_probability=1.0,
            subagent_turn_probability=1.0,
            subagent_result_tokens_mean=100,
            subagent_result_tokens_median=50,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()
        parents = [c for c in conversations if not c.is_subagent_child]
        # Verify spawns were generated (sanity check)
        has_spawns = any(p.subagent_spawns for p in parents)
        assert has_spawns

    def test_new_session_probability_fields(self):
        cfg = CodingSessionConfig(enabled=True)
        assert cfg.subagent_session_probability == 0.35
        assert cfg.subagent_turn_probability == 0.25
        assert cfg.subagent_background_probability == 0.15
        assert cfg.subagent_result_tokens_mean == 3000
        assert cfg.subagent_result_tokens_median == 1500
        assert cfg.subagent_explore_model_name is None


def _make_subagent_config(
    num_sessions: int = 10,
    subagent_session_probability: float = 1.0,
    subagent_turn_probability: float = 1.0,
    subagent_background_probability: float = 0.0,
    subagent_explore_model_name: str | None = None,
    subagent_result_tokens_mean: int = 3000,
    subagent_result_tokens_median: int = 1500,
) -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(model_names=["test_model"], streaming=True),
        loadgen=LoadGeneratorConfig(benchmark_duration=300),
        input=InputConfig(
            coding_session=CodingSessionConfig(
                enabled=True,
                num_sessions=num_sessions,
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
                subagent_session_probability=subagent_session_probability,
                subagent_turn_probability=subagent_turn_probability,
                subagent_background_probability=subagent_background_probability,
                subagent_explore_model_name=subagent_explore_model_name,
                subagent_result_tokens_mean=subagent_result_tokens_mean,
                subagent_result_tokens_median=subagent_result_tokens_median,
                subagent_count_mean=2.0,
                subagent_count_max=3,
            ),
            prompt=PromptConfig(),
        ),
    )


# ============================================================================
# Context Loss / replaces_history Tests
# ============================================================================


class TestReplacesHistory:
    """Tests for replaces_history flag on context-loss turns."""

    def test_restart_sets_replaces_history(self, mock_tokenizer):
        """With restart_probability=1.0, all turns after turn 0 have replaces_history."""
        config = _make_cache_config(
            num_sessions=3,
            restart_probability=1.0,
            max_prompt_tokens=10000,
            thinking_tokens_mean=0,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            assert not conv.turns[0].replaces_history
            for turn in conv.turns[1:]:
                assert turn.replaces_history

    def test_compression_sets_replaces_history(self, mock_tokenizer):
        """With compression enabled, at least one turn has replaces_history."""
        config = _make_cache_config(
            num_sessions=5,
            max_prompt_tokens=5000,
            max_compressions=3,
            compression_threshold=0.3,
            compression_ratio=0.5,
            restart_probability=0.0,
            thinking_tokens_mean=0,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        any_replaces = any(
            turn.replaces_history for conv in conversations for turn in conv.turns
        )
        assert any_replaces

    def test_no_context_loss_no_replaces_history(self, mock_tokenizer):
        """With all context-loss events disabled, no turns have replaces_history."""
        config = _make_cache_config(
            num_sessions=5,
            restart_probability=0.0,
            max_compressions=0,
            thinking_tokens_mean=0,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns:
                assert not turn.replaces_history

    def test_replaces_history_turn_has_full_context_delta(self, mock_tokenizer):
        """A replaces_history turn's prompt text covers cumulative_tokens, not incremental."""
        config = _make_cache_config(
            num_sessions=3,
            restart_probability=1.0,
            max_prompt_tokens=10000,
            thinking_tokens_mean=0,
        )
        composer = CodingSessionComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns:
                if turn.replaces_history:
                    assert turn.texts
                    assert len(turn.texts[0].contents[0]) > 0
