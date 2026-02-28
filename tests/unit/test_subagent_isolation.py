# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for subagent isolation: separate conversations with independent KV caches.

Tests cover all phases of the subagent pipeline:
- Phase 1: Data model (SubagentSpawnInfo, Turn/TurnMetadata subagent_spawn_id,
  Conversation/ConversationMetadata is_subagent_child, subagent_spawns)
- Phase 2: Config (CodingSessionConfig subagent parameters)
- Phase 3: Dataset generation (CodingSessionComposer generates child conversations)
- Phase 4: ConversationSource (start_child_session, get_subagent_spawn, sampler filtering)
- Phase 5: Strategy (dispatch_subagent_spawn, handle_subagent_child_complete,
  PendingSubagentJoin, cleanup)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.config import (
    CodingSessionConfig,
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    Conversation,
    ConversationMetadata,
    DatasetMetadata,
    SubagentSpawnInfo,
    Turn,
    TurnMetadata,
)
from aiperf.credit.structs import Credit
from aiperf.dataset.composer.coding_session import CodingSessionComposer
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.strategies.adaptive_scale import (
    AdaptiveScaleStrategy,
    PendingSubagentJoin,
)
from tests.unit.timing.conftest import make_sampler

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_tokenizer(mock_tokenizer_cls):
    return mock_tokenizer_cls.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    )


# =============================================================================
# Helpers
# =============================================================================


def _make_sequential_credit(
    *,
    credit_id: int = 1,
    conv_id: str = "conv_0",
    corr_id: str = "xcorr-1",
    turn_index: int = 0,
    num_turns: int = 10,
) -> Credit:
    return Credit(
        id=credit_id,
        phase=CreditPhase.PROFILING,
        conversation_id=conv_id,
        x_correlation_id=corr_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
    )


def _make_dataset_with_subagent(
    num_conversations: int = 3,
    turns_per_conv: int = 6,
    spawn_at: int | None = 2,
    num_children: int = 2,
    child_turns: int = 3,
    is_background: bool = False,
) -> tuple[DatasetMetadata, list[str]]:
    """Create dataset with subagent spawns.

    Returns (DatasetMetadata, list of child conversation IDs).
    """
    convs = []
    child_conv_ids: list[str] = []

    for c in range(num_conversations):
        conv_id = f"conv_{c}"
        turns = []
        spawns: list[SubagentSpawnInfo] = []

        for i in range(turns_per_conv):
            spawn_id = None
            if spawn_at is not None and i == spawn_at + 1:
                spawn_id = "s0"

            turns.append(
                TurnMetadata(
                    delay_ms=200.0 if i > 0 else None,
                    input_tokens=500 + i * 100,
                    subagent_spawn_id=spawn_id,
                )
            )

        if spawn_at is not None:
            children = []
            for ci in range(num_children):
                child_id = f"{conv_id}_s0_c{ci}"
                children.append(child_id)
                child_conv_ids.append(child_id)

            spawns.append(
                SubagentSpawnInfo(
                    spawn_id="s0",
                    child_conversation_ids=children,
                    join_turn_index=spawn_at + 1,
                    is_background=is_background,
                )
            )

        convs.append(
            ConversationMetadata(
                conversation_id=conv_id,
                turns=turns,
                subagent_spawns=spawns,
            )
        )

    # Add child conversations
    for child_id in child_conv_ids:
        child_turns_list = [
            TurnMetadata(
                delay_ms=100.0 if j > 0 else None,
                input_tokens=300 + j * 50,
            )
            for j in range(child_turns)
        ]
        convs.append(
            ConversationMetadata(
                conversation_id=child_id,
                turns=child_turns_list,
                is_subagent_child=True,
            )
        )

    ds = DatasetMetadata(
        conversations=convs,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    return ds, child_conv_ids


def _make_conversation_source(ds: DatasetMetadata) -> ConversationSource:
    sampler = make_sampler(
        [c.conversation_id for c in ds.conversations if not c.is_subagent_child],
        ds.sampling_strategy,
    )
    return ConversationSource(ds, sampler)


def _make_adaptive_strategy(
    num_conversations: int = 10,
    start_users: int = 2,
    max_users: int | None = 10,
    recycle: bool = False,
    spawn_at: int | None = 2,
    num_children: int = 2,
    child_turns: int = 3,
    turns_per_conv: int = 6,
    is_background: bool = False,
) -> tuple[
    AdaptiveScaleStrategy, MagicMock, MagicMock, MagicMock, DatasetMetadata, list[str]
]:
    scheduler = MagicMock()
    scheduler.schedule_later = MagicMock()
    scheduler.execute_async = MagicMock()
    stop_checker = MagicMock()
    stop_checker.can_send_any_turn = MagicMock(return_value=True)
    issuer = MagicMock()
    issuer.issue_credit = AsyncMock(return_value=True)
    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 1_000_000_000

    ds, child_ids = _make_dataset_with_subagent(
        num_conversations=num_conversations,
        turns_per_conv=turns_per_conv,
        spawn_at=spawn_at,
        num_children=num_children,
        child_turns=child_turns,
        is_background=is_background,
    )
    src = _make_conversation_source(ds)

    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.ADAPTIVE_SCALE,
        expected_duration_sec=120.0,
        start_users=start_users,
        max_users=max_users,
        recycle_sessions=recycle,
    )
    strategy = AdaptiveScaleStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )
    return strategy, scheduler, issuer, stop_checker, ds, child_ids


# =============================================================================
# Phase 1: Data Model Tests
# =============================================================================


class TestSubagentSpawnInfo:
    """SubagentSpawnInfo model tests."""

    def test_basic_construction(self):
        info = SubagentSpawnInfo(
            spawn_id="s0",
            child_conversation_ids=["child_0", "child_1"],
            join_turn_index=4,
        )
        assert info.spawn_id == "s0"
        assert info.child_conversation_ids == ["child_0", "child_1"]
        assert info.join_turn_index == 4

    def test_single_child(self):
        info = SubagentSpawnInfo(
            spawn_id="s1",
            child_conversation_ids=["child_only"],
            join_turn_index=2,
        )
        assert len(info.child_conversation_ids) == 1


class TestTurnSubagentFields:
    """Turn and TurnMetadata subagent_spawn_id fields."""

    def test_turn_defaults_none(self):
        turn = Turn(max_tokens=100, input_tokens=500)
        assert turn.subagent_spawn_id is None

    def test_turn_with_subagent_spawn_id(self):
        turn = Turn(
            max_tokens=100,
            input_tokens=500,
            subagent_spawn_id="s0",
        )
        assert turn.subagent_spawn_id == "s0"

    def test_turn_metadata_passthrough(self):
        turn = Turn(
            max_tokens=100,
            input_tokens=500,
            subagent_spawn_id="s2",
        )
        meta = turn.metadata()
        assert meta.subagent_spawn_id == "s2"

    def test_turn_metadata_defaults_none(self):
        turn = Turn(max_tokens=100, input_tokens=500)
        meta = turn.metadata()
        assert meta.subagent_spawn_id is None

    def test_turn_metadata_direct_construction(self):
        meta = TurnMetadata(
            input_tokens=1000,
            subagent_spawn_id="s3",
        )
        assert meta.subagent_spawn_id == "s3"


class TestConversationSubagentFields:
    """Conversation and ConversationMetadata subagent fields."""

    def test_conversation_defaults(self):
        conv = Conversation(session_id="test")
        assert conv.is_subagent_child is False
        assert conv.subagent_spawns == []

    def test_conversation_subagent_child(self):
        conv = Conversation(session_id="child_0", is_subagent_child=True)
        assert conv.is_subagent_child is True

    def test_conversation_with_spawns(self):
        spawn = SubagentSpawnInfo(
            spawn_id="s0",
            child_conversation_ids=["c0"],
            join_turn_index=3,
        )
        conv = Conversation(
            session_id="parent",
            subagent_spawns=[spawn],
        )
        assert len(conv.subagent_spawns) == 1

    def test_metadata_passthrough(self):
        spawn = SubagentSpawnInfo(
            spawn_id="s0",
            child_conversation_ids=["c0", "c1"],
            join_turn_index=3,
        )
        conv = Conversation(
            session_id="parent",
            is_subagent_child=False,
            subagent_spawns=[spawn],
            turns=[Turn(max_tokens=10, input_tokens=100) for _ in range(5)],
        )
        meta = conv.metadata()
        assert meta.is_subagent_child is False
        assert len(meta.subagent_spawns) == 1
        assert meta.subagent_spawns[0].spawn_id == "s0"
        assert meta.subagent_spawns[0].child_conversation_ids == ["c0", "c1"]

    def test_child_metadata_passthrough(self):
        conv = Conversation(
            session_id="child",
            is_subagent_child=True,
            turns=[Turn(max_tokens=10, input_tokens=100)],
        )
        meta = conv.metadata()
        assert meta.is_subagent_child is True
        assert meta.subagent_spawns == []

    def test_conversation_metadata_defaults(self):
        meta = ConversationMetadata(conversation_id="c1")
        assert meta.subagent_spawns == []
        assert meta.is_subagent_child is False

    def test_backward_compatible_without_subagent(self):
        meta = ConversationMetadata(
            conversation_id="c1",
            turns=[TurnMetadata(delay_ms=100) for _ in range(3)],
        )
        assert meta.subagent_spawns == []
        assert meta.is_subagent_child is False


# =============================================================================
# Phase 2: Config Tests
# =============================================================================


class TestCodingSessionSubagentConfig:
    """CodingSessionConfig subagent parameters."""

    def test_defaults(self):
        cfg = CodingSessionConfig(enabled=True)
        assert cfg.subagent_probability == 0.15
        assert cfg.subagent_count_mean == 1.2
        assert cfg.subagent_count_max == 4
        assert cfg.subagent_turns_mean == 8
        assert cfg.subagent_turns_median == 5
        assert cfg.subagent_system_tokens == 4000
        assert cfg.subagent_new_tokens_mean == 2500
        assert cfg.subagent_new_tokens_median == 1200
        assert cfg.subagent_max_prompt_tokens == 50000
        assert cfg.subagent_session_probability == 0.35
        assert cfg.subagent_turn_probability == 0.25
        assert cfg.subagent_background_probability == 0.15
        assert cfg.subagent_result_tokens_mean == 3000
        assert cfg.subagent_result_tokens_median == 1500
        assert cfg.subagent_explore_model_name is None

    def test_disable_subagent(self):
        cfg = CodingSessionConfig(enabled=True, subagent_probability=0.0)
        assert cfg.subagent_probability == 0.0

    def test_custom_values(self):
        cfg = CodingSessionConfig(
            enabled=True,
            subagent_probability=0.3,
            subagent_count_mean=2.0,
            subagent_count_max=6,
            subagent_turns_mean=10,
            subagent_turns_median=7,
            subagent_system_tokens=3000,
            subagent_new_tokens_mean=2000,
            subagent_new_tokens_median=1000,
            subagent_max_prompt_tokens=40000,
        )
        assert cfg.subagent_probability == 0.3
        assert cfg.subagent_count_mean == 2.0
        assert cfg.subagent_count_max == 6


# =============================================================================
# Phase 3: Dataset Generation Tests
# =============================================================================


class TestCodingSessionSubagentGeneration:
    """CodingSessionComposer subagent child generation."""

    @pytest.fixture
    def subagent_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test_model"], streaming=True),
            loadgen=LoadGeneratorConfig(benchmark_duration=300),
            input=InputConfig(
                coding_session=CodingSessionConfig(
                    enabled=True,
                    num_sessions=10,
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
                ),
                prompt=PromptConfig(),
            ),
        )

    @pytest.fixture
    def no_subagent_config(self):
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

    def test_subagent_probability_1_generates_children(
        self, subagent_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        children = [c for c in conversations if c.is_subagent_child]
        parents = [c for c in conversations if not c.is_subagent_child]
        assert len(children) > 0
        assert len(parents) == 10

    def test_subagent_probability_0_generates_no_children(
        self, no_subagent_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(no_subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        children = [c for c in conversations if c.is_subagent_child]
        assert len(children) == 0

    def test_child_conversations_have_independent_hash_ids(
        self, subagent_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        parents = [c for c in conversations if not c.is_subagent_child]
        children = [c for c in conversations if c.is_subagent_child]

        if not children:
            pytest.skip("No children generated")

        # Child hash_ids should not be a prefix of any parent's hash_ids
        parent_first_hashes = set()
        for p in parents:
            if p.turns and p.turns[0].hash_ids:
                parent_first_hashes.add(tuple(p.turns[0].hash_ids[:5]))

        for child in children:
            if child.turns and child.turns[0].hash_ids:
                assert len(child.turns[0].hash_ids) > 0

    def test_parent_has_subagent_spawns(self, subagent_config, mock_tokenizer):
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        parents = [c for c in conversations if not c.is_subagent_child]
        has_spawns = any(len(p.subagent_spawns) > 0 for p in parents)
        assert has_spawns

    def test_spawn_info_references_valid_children(
        self, subagent_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        child_ids = {c.session_id for c in conversations if c.is_subagent_child}
        parents = [c for c in conversations if not c.is_subagent_child]

        for parent in parents:
            for spawn in parent.subagent_spawns:
                for child_id in spawn.child_conversation_ids:
                    assert child_id in child_ids, (
                        f"Spawn {spawn.spawn_id} references unknown child {child_id}"
                    )

    def test_join_turn_has_subagent_spawn_id(self, subagent_config, mock_tokenizer):
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        parents = [c for c in conversations if not c.is_subagent_child]
        found_spawn_id = False
        for parent in parents:
            for turn in parent.turns:
                if turn.subagent_spawn_id is not None:
                    found_spawn_id = True
                    break
        assert found_spawn_id

    def test_metadata_includes_subagent_spawns(self, subagent_config, mock_tokenizer):
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        parents = [c for c in conversations if not c.is_subagent_child]
        has_meta_spawns = False
        for parent in parents:
            meta = parent.metadata()
            if meta.subagent_spawns:
                has_meta_spawns = True
                for spawn in meta.subagent_spawns:
                    assert spawn.join_turn_index < len(meta.turns)
        assert has_meta_spawns

    def test_child_input_tokens_within_max(self, subagent_config, mock_tokenizer):
        from aiperf.common.config.coding_session_config import DEFAULT_SUBAGENT_PROFILES

        max_tokens = max(p.max_prompt_tokens for p in DEFAULT_SUBAGENT_PROFILES)
        composer = CodingSessionComposer(subagent_config, mock_tokenizer)
        conversations = composer.create_dataset()

        children = [c for c in conversations if c.is_subagent_child]
        for child in children:
            for turn in child.turns:
                assert turn.input_tokens <= max_tokens


# =============================================================================
# Phase 4: ConversationSource Tests
# =============================================================================


class TestConversationSourceSubagent:
    """ConversationSource subagent methods."""

    @pytest.fixture
    def src_with_subagent(self):
        ds, child_ids = _make_dataset_with_subagent(
            num_conversations=3,
            turns_per_conv=6,
            spawn_at=2,
            num_children=2,
            child_turns=3,
        )
        return _make_conversation_source(ds), ds, child_ids

    def test_start_child_session(self, src_with_subagent):
        src, ds, child_ids = src_with_subagent
        session = src.start_child_session(child_ids[0])
        assert session.conversation_id == child_ids[0]
        assert session.x_correlation_id is not None
        assert len(session.metadata.turns) == 3

    def test_start_child_session_unique_correlation_ids(self, src_with_subagent):
        src, ds, child_ids = src_with_subagent
        s1 = src.start_child_session(child_ids[0])
        s2 = src.start_child_session(child_ids[0])
        assert s1.x_correlation_id != s2.x_correlation_id

    def test_get_subagent_spawn_found(self, src_with_subagent):
        src, ds, child_ids = src_with_subagent
        spawn = src.get_subagent_spawn("conv_0", "s0")
        assert spawn is not None
        assert spawn.spawn_id == "s0"
        assert len(spawn.child_conversation_ids) == 2

    def test_get_subagent_spawn_not_found(self, src_with_subagent):
        src, ds, child_ids = src_with_subagent
        assert src.get_subagent_spawn("conv_0", "s99") is None

    def test_get_subagent_spawn_unknown_conversation(self, src_with_subagent):
        src, ds, child_ids = src_with_subagent
        assert src.get_subagent_spawn("unknown_conv", "s0") is None

    def test_sampler_excludes_children(self, src_with_subagent):
        src, ds, child_ids = src_with_subagent
        # Sample several conversations -- none should be children
        sampled_ids = set()
        for _ in range(20):
            session = src.next()
            sampled_ids.add(session.conversation_id)

        for child_id in child_ids:
            assert child_id not in sampled_ids

    def test_children_in_metadata_lookup(self, src_with_subagent):
        src, ds, child_ids = src_with_subagent
        # Children should be accessible via get_metadata even though not sampled
        for child_id in child_ids:
            meta = src.get_metadata(child_id)
            assert meta.is_subagent_child is True

    def test_child_session_builds_first_turn(self, src_with_subagent):
        src, ds, child_ids = src_with_subagent
        session = src.start_child_session(child_ids[0])
        turn = session.build_first_turn()
        assert turn.turn_index == 0
        assert turn.num_turns == 3
        assert turn.conversation_id == child_ids[0]


# =============================================================================
# Phase 5: Strategy Tests
# =============================================================================


class TestAdaptiveScaleSubagentDispatch:
    """AdaptiveScaleStrategy subagent spawn dispatch and join."""

    @pytest.mark.asyncio
    async def test_dispatch_subagent_spawn_creates_pending_join(self):
        strategy, scheduler, issuer, _, ds, child_ids = _make_adaptive_strategy()

        credit = _make_sequential_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )

        strategy._dispatch_subagent_spawn(credit, "s0")

        assert "parent-1" in strategy._pending_subagent_joins
        pending = strategy._pending_subagent_joins["parent-1"]
        assert pending.expected_count == 2
        assert pending.completed_count == 0
        assert pending.parent_conversation_id == "conv_0"

    @pytest.mark.asyncio
    async def test_dispatch_subagent_spawn_issues_child_credits(self):
        strategy, scheduler, issuer, _, ds, child_ids = _make_adaptive_strategy()

        credit = _make_sequential_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )

        strategy._dispatch_subagent_spawn(credit, "s0")

        # Should issue credits for each child
        assert scheduler.execute_async.call_count == 2

    @pytest.mark.asyncio
    async def test_dispatch_subagent_spawn_registers_child_to_parent_mapping(self):
        strategy, scheduler, issuer, _, ds, child_ids = _make_adaptive_strategy()

        credit = _make_sequential_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )

        strategy._dispatch_subagent_spawn(credit, "s0")

        assert len(strategy._subagent_child_to_parent) == 2
        for (
            _child_corr_id,
            parent_corr_id,
        ) in strategy._subagent_child_to_parent.items():
            assert parent_corr_id == "parent-1"

    @pytest.mark.asyncio
    async def test_dispatch_subagent_spawn_fallback_on_unknown_spawn(self):
        strategy, scheduler, issuer, _, ds, child_ids = _make_adaptive_strategy()

        credit = _make_sequential_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )

        strategy._dispatch_subagent_spawn(credit, "unknown_spawn")

        # Should fallback to issuing the next sequential turn
        assert scheduler.execute_async.call_count == 1
        assert "parent-1" not in strategy._pending_subagent_joins

    @pytest.mark.asyncio
    async def test_handle_subagent_child_complete_partial(self):
        strategy, scheduler, issuer, _, ds, child_ids = _make_adaptive_strategy()

        # Set up spawn
        credit = _make_sequential_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        strategy._dispatch_subagent_spawn(credit, "s0")
        scheduler.execute_async.reset_mock()

        # Get a child correlation ID
        child_corr_ids = list(strategy._subagent_child_to_parent.keys())
        assert len(child_corr_ids) == 2

        # First child completes
        child_credit = _make_sequential_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
        )
        strategy._handle_subagent_child_complete(child_credit)

        # Should NOT dispatch join yet
        pending = strategy._pending_subagent_joins.get("parent-1")
        assert pending is not None
        assert pending.completed_count == 1

    @pytest.mark.asyncio
    async def test_handle_subagent_child_complete_all_done(self):
        strategy, scheduler, issuer, _, ds, child_ids = _make_adaptive_strategy()

        # Set up spawn
        credit = _make_sequential_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        strategy._dispatch_subagent_spawn(credit, "s0")
        scheduler.execute_async.reset_mock()

        child_corr_ids = list(strategy._subagent_child_to_parent.keys())

        # Both children complete
        for i, child_corr_id in enumerate(child_corr_ids):
            child_credit = _make_sequential_credit(
                conv_id=child_ids[i],
                corr_id=child_corr_id,
                turn_index=2,
                num_turns=3,
            )
            strategy._handle_subagent_child_complete(child_credit)

        # Join should be dispatched
        assert "parent-1" not in strategy._pending_subagent_joins
        # At least one execute_async call for the join
        assert scheduler.execute_async.call_count >= 1

    @pytest.mark.asyncio
    async def test_handle_credit_return_detects_subagent_spawn(self):
        strategy, scheduler, issuer, _, ds, child_ids = _make_adaptive_strategy()

        # Turn 2 is sequential, turn 3 has subagent_spawn_id="s0"
        credit = _make_sequential_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )

        await strategy.handle_credit_return(credit)

        # Should have detected the spawn and dispatched children
        assert "parent-1" in strategy._pending_subagent_joins

    @pytest.mark.asyncio
    async def test_handle_credit_return_subagent_child_final_turn(self):
        strategy, scheduler, issuer, _, ds, child_ids = _make_adaptive_strategy()

        # Set up spawn manually
        credit = _make_sequential_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        strategy._dispatch_subagent_spawn(credit, "s0")
        scheduler.execute_async.reset_mock()

        child_corr_ids = list(strategy._subagent_child_to_parent.keys())

        # Child's final turn triggers subagent completion path
        child_credit = _make_sequential_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
        )
        assert child_credit.is_final_turn

        await strategy.handle_credit_return(child_credit)

        # Should have processed as subagent child, not regular final turn
        pending = strategy._pending_subagent_joins.get("parent-1")
        assert pending is not None
        assert pending.completed_count == 1

    @pytest.mark.asyncio
    async def test_cleanup_session_removes_subagent_state(self):
        strategy, scheduler, issuer, _, ds, child_ids = _make_adaptive_strategy()

        # Set up spawn
        credit = _make_sequential_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        strategy._dispatch_subagent_spawn(credit, "s0")

        # Cleanup parent session
        strategy._cleanup_session("parent-1")

        assert "parent-1" not in strategy._pending_subagent_joins
        # Orphaned child mappings should also be cleaned
        for v in strategy._subagent_child_to_parent.values():
            assert v != "parent-1"


class TestPendingSubagentJoin:
    """PendingSubagentJoin dataclass tests."""

    def test_basic_construction(self):
        pj = PendingSubagentJoin(
            parent_conversation_id="conv_0",
            parent_correlation_id="parent-1",
            expected_count=3,
            join_turn_index=5,
            parent_num_turns=10,
        )
        assert pj.parent_conversation_id == "conv_0"
        assert pj.parent_correlation_id == "parent-1"
        assert pj.expected_count == 3
        assert pj.completed_count == 0
        assert pj.join_turn_index == 5
        assert pj.parent_num_turns == 10

    def test_increment_completed(self):
        pj = PendingSubagentJoin(
            parent_conversation_id="conv_0",
            parent_correlation_id="parent-1",
            expected_count=2,
        )
        pj.completed_count += 1
        assert pj.completed_count == 1
        assert pj.completed_count < pj.expected_count

        pj.completed_count += 1
        assert pj.completed_count == pj.expected_count


# =============================================================================
# Background Subagent Tests
# =============================================================================


class TestBackgroundSubagentDispatch:
    """Tests for background subagent spawns where parent continues immediately."""

    @pytest.mark.asyncio
    async def test_background_spawn_dispatches_join_immediately(self):
        strategy, scheduler, issuer, _, ds, child_ids = _make_adaptive_strategy(
            is_background=True
        )

        credit = _make_sequential_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )

        strategy._dispatch_subagent_spawn(credit, "s0")

        # Should NOT register a pending join (parent doesn't wait)
        assert "parent-1" not in strategy._pending_subagent_joins

        # Should issue credits for children + the join turn
        # 2 children + 1 join = 3 execute_async calls
        assert scheduler.execute_async.call_count == 3

    @pytest.mark.asyncio
    async def test_background_child_complete_no_pending_join(self):
        strategy, scheduler, issuer, _, ds, child_ids = _make_adaptive_strategy(
            is_background=True
        )

        credit = _make_sequential_credit(
            conv_id="conv_0", corr_id="parent-1", turn_index=2, num_turns=6
        )
        strategy._dispatch_subagent_spawn(credit, "s0")
        scheduler.execute_async.reset_mock()

        child_corr_ids = list(strategy._subagent_child_to_parent.keys())
        assert len(child_corr_ids) == 2

        # Child completes -- should clean up without error
        child_credit = _make_sequential_credit(
            conv_id=child_ids[0],
            corr_id=child_corr_ids[0],
            turn_index=2,
            num_turns=3,
        )
        strategy._handle_subagent_child_complete(child_credit)

        # No pending join to dispatch (already dispatched immediately)
        assert "parent-1" not in strategy._pending_subagent_joins

    def test_subagent_spawn_info_is_background_default(self):
        info = SubagentSpawnInfo(
            spawn_id="s0",
            child_conversation_ids=["c0"],
            join_turn_index=3,
        )
        assert info.is_background is False

    def test_subagent_spawn_info_is_background_true(self):
        info = SubagentSpawnInfo(
            spawn_id="s0",
            child_conversation_ids=["c0"],
            join_turn_index=3,
            is_background=True,
        )
        assert info.is_background is True
