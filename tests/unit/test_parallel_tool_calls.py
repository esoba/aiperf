# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for parallel tool call / sub-agent support.

Tests cover all phases of the parallel dispatch pipeline:
- Phase 1: Data model (Turn, TurnMetadata, ParallelGroupInfo, ConversationMetadata)
- Phase 2: Config (CodingSessionConfig parallel parameters)
- Phase 3: Dataset generation (CodingSessionComposer, CodingTraceLoader)
- Phase 4: ConversationSource (get_parallel_group, get_turn_metadata_at)
- Phase 5: Credit issuer (parallel branch slot behavior)
- Phase 6: Sticky router (parent correlation routing)
- Phase 7: Adaptive scale strategy (fan-out, join tracking, branch returns)
- Phase 8: Worker session manager (child-to-parent mapping)
- Credit/TurnToSend structs (parallel fields, for_parallel_branch factory)
"""

import asyncio
import time
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
    ParallelGroupInfo,
    Turn,
    TurnMetadata,
)
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.dataset.composer.coding_session import CodingSessionComposer
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.strategies.adaptive_scale import (
    AdaptiveScaleStrategy,
    PendingJoin,
)
from aiperf.workers.session_manager import UserSessionManager
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


def _make_parallel_credit(
    *,
    credit_id: int = 1,
    conv_id: str = "conv_0",
    corr_id: str = "parent-corr",
    turn_index: int = 1,
    num_turns: int = 10,
    parallel_group: str = "g0",
    parallel_branch: int = 0,
    parent_correlation_id: str = "parent-corr",
) -> Credit:
    return Credit(
        id=credit_id,
        phase=CreditPhase.PROFILING,
        conversation_id=conv_id,
        x_correlation_id=f"{parent_correlation_id}_b{parallel_branch}",
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        parallel_group=parallel_group,
        parallel_branch=parallel_branch,
        parent_correlation_id=parent_correlation_id,
    )


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


def _compute_parallel_groups(
    turns: list[TurnMetadata],
) -> list[ParallelGroupInfo]:
    """Compute parallel group info from turn metadata (mirrors Conversation._compute_parallel_groups)."""
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(turns):
        if t.parallel_group is not None:
            groups.setdefault(t.parallel_group, []).append(i)

    result: list[ParallelGroupInfo] = []
    for group_id, indices in groups.items():
        join_index = indices[-1] + 1
        if join_index >= len(turns):
            join_index = len(turns) - 1
        result.append(
            ParallelGroupInfo(
                group_id=group_id,
                turn_indices=indices,
                join_turn_index=join_index,
            )
        )
    return result


def _make_dataset_with_parallel(
    num_conversations: int = 5,
    turns_per_conv: int = 8,
    parallel_at: int | None = 2,
    num_branches: int = 3,
) -> DatasetMetadata:
    """Create dataset with parallel groups embedded in turn metadata."""
    convs = []
    for c in range(num_conversations):
        turns = []
        group_counter = 0
        i = 0
        while i < turns_per_conv:
            if parallel_at is not None and i == parallel_at:
                group_id = f"g{group_counter}"
                for b in range(num_branches):
                    turns.append(
                        TurnMetadata(
                            delay_ms=100.0,
                            input_tokens=1000 + b * 200,
                            parallel_group=group_id,
                            parallel_branch=b,
                        )
                    )
                # Join turn
                turns.append(TurnMetadata(delay_ms=500.0, input_tokens=2000))
                group_counter += 1
                i += num_branches + 1
            else:
                turns.append(
                    TurnMetadata(
                        delay_ms=200.0 if i > 0 else None,
                        input_tokens=500 + i * 100,
                    )
                )
                i += 1
        parallel_groups = _compute_parallel_groups(turns)
        convs.append(
            ConversationMetadata(
                conversation_id=f"conv_{c}",
                turns=turns,
                parallel_groups=parallel_groups,
            )
        )
    return DatasetMetadata(
        conversations=convs,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _make_conversation_source(
    ds: DatasetMetadata,
) -> ConversationSource:
    sampler = make_sampler(
        [c.conversation_id for c in ds.conversations],
        ds.sampling_strategy,
    )
    return ConversationSource(ds, sampler)


def _make_adaptive_strategy(
    num_conversations: int = 20,
    start_users: int = 2,
    max_users: int | None = 10,
    recycle: bool = False,
    parallel_at: int | None = 2,
    num_branches: int = 3,
    turns_per_conv: int = 8,
) -> tuple[AdaptiveScaleStrategy, MagicMock, MagicMock, MagicMock, DatasetMetadata]:
    scheduler = MagicMock()
    scheduler.schedule_later = MagicMock()
    scheduler.execute_async = MagicMock()
    stop_checker = MagicMock()
    stop_checker.can_send_any_turn = MagicMock(return_value=True)
    issuer = MagicMock()
    issuer.issue_credit = AsyncMock(return_value=True)
    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 1_000_000_000

    ds = _make_dataset_with_parallel(
        num_conversations=num_conversations,
        turns_per_conv=turns_per_conv,
        parallel_at=parallel_at,
        num_branches=num_branches,
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
    return strategy, scheduler, issuer, stop_checker, ds


# =============================================================================
# Phase 1: Data Model Tests
# =============================================================================


class TestTurnParallelFields:
    """Turn and TurnMetadata parallel_group / parallel_branch fields."""

    def test_turn_defaults_none(self):
        turn = Turn(max_tokens=100, input_tokens=500)
        assert turn.parallel_group is None
        assert turn.parallel_branch is None

    def test_turn_with_parallel_fields(self):
        turn = Turn(
            max_tokens=100,
            input_tokens=500,
            parallel_group="g0",
            parallel_branch=2,
        )
        assert turn.parallel_group == "g0"
        assert turn.parallel_branch == 2

    def test_turn_metadata_passthrough(self):
        turn = Turn(
            max_tokens=100,
            input_tokens=500,
            hash_ids=[1, 2, 3],
            parallel_group="g1",
            parallel_branch=0,
        )
        meta = turn.metadata()
        assert meta.parallel_group == "g1"
        assert meta.parallel_branch == 0
        assert meta.input_tokens == 500
        assert meta.hash_ids == [1, 2, 3]

    def test_turn_metadata_defaults_none(self):
        turn = Turn(max_tokens=100, input_tokens=500)
        meta = turn.metadata()
        assert meta.parallel_group is None
        assert meta.parallel_branch is None

    def test_turn_metadata_direct_construction(self):
        meta = TurnMetadata(
            input_tokens=1000,
            parallel_group="g5",
            parallel_branch=3,
        )
        assert meta.parallel_group == "g5"
        assert meta.parallel_branch == 3


class TestParallelGroupInfo:
    """ParallelGroupInfo model tests."""

    def test_basic_construction(self):
        pgi = ParallelGroupInfo(
            group_id="g0",
            turn_indices=[1, 2, 3],
            join_turn_index=4,
        )
        assert pgi.group_id == "g0"
        assert pgi.turn_indices == [1, 2, 3]
        assert pgi.join_turn_index == 4

    def test_single_group_from_turns(self):
        turns = [
            TurnMetadata(input_tokens=100),
            TurnMetadata(input_tokens=200, parallel_group="g0", parallel_branch=0),
            TurnMetadata(input_tokens=200, parallel_group="g0", parallel_branch=1),
            TurnMetadata(input_tokens=200, parallel_group="g0", parallel_branch=2),
            TurnMetadata(input_tokens=400),
        ]
        conv = Conversation(
            session_id="test",
            turns=[
                Turn(
                    max_tokens=10,
                    input_tokens=t.input_tokens,
                    parallel_group=t.parallel_group,
                    parallel_branch=t.parallel_branch,
                )
                for t in turns
            ],
        )
        meta = conv.metadata()
        assert len(meta.parallel_groups) == 1
        pg = meta.parallel_groups[0]
        assert pg.group_id == "g0"
        assert pg.turn_indices == [1, 2, 3]
        assert pg.join_turn_index == 4

    def test_multiple_groups_from_turns(self):
        turns = [
            Turn(max_tokens=10, input_tokens=100),
            Turn(
                max_tokens=10, input_tokens=200, parallel_group="g0", parallel_branch=0
            ),
            Turn(
                max_tokens=10, input_tokens=200, parallel_group="g0", parallel_branch=1
            ),
            Turn(max_tokens=10, input_tokens=400),
            Turn(
                max_tokens=10, input_tokens=500, parallel_group="g1", parallel_branch=0
            ),
            Turn(
                max_tokens=10, input_tokens=500, parallel_group="g1", parallel_branch=1
            ),
            Turn(max_tokens=10, input_tokens=600),
        ]
        conv = Conversation(session_id="test", turns=turns)
        meta = conv.metadata()
        assert len(meta.parallel_groups) == 2
        assert meta.parallel_groups[0].group_id == "g0"
        assert meta.parallel_groups[0].turn_indices == [1, 2]
        assert meta.parallel_groups[0].join_turn_index == 3
        assert meta.parallel_groups[1].group_id == "g1"
        assert meta.parallel_groups[1].turn_indices == [4, 5]
        assert meta.parallel_groups[1].join_turn_index == 6

    def test_no_parallel_groups(self):
        conv = Conversation(
            session_id="test",
            turns=[Turn(max_tokens=10, input_tokens=100) for _ in range(5)],
        )
        meta = conv.metadata()
        assert meta.parallel_groups == []


class TestConversationMetadataParallelGroups:
    """ConversationMetadata.parallel_groups population."""

    def test_parallel_groups_default_empty(self):
        meta = ConversationMetadata(conversation_id="c1")
        assert meta.parallel_groups == []

    def test_parallel_groups_from_metadata(self):
        meta = ConversationMetadata(
            conversation_id="c1",
            parallel_groups=[
                ParallelGroupInfo(group_id="g0", turn_indices=[1, 2], join_turn_index=3)
            ],
        )
        assert len(meta.parallel_groups) == 1

    def test_backward_compatible_without_parallel(self):
        meta = ConversationMetadata(
            conversation_id="c1",
            turns=[TurnMetadata(delay_ms=100) for _ in range(3)],
        )
        assert meta.parallel_groups == []


# =============================================================================
# Phase 1b: Credit Structs Tests
# =============================================================================


class TestCreditParallelFields:
    """Credit parallel_group, parallel_branch, parent_correlation_id."""

    def test_credit_defaults_none(self):
        c = _make_sequential_credit()
        assert c.parallel_group is None
        assert c.parallel_branch is None
        assert c.parent_correlation_id is None
        assert c.is_parallel_branch is False

    def test_credit_with_parallel_fields(self):
        c = _make_parallel_credit()
        assert c.parallel_group == "g0"
        assert c.parallel_branch == 0
        assert c.parent_correlation_id == "parent-corr"
        assert c.is_parallel_branch is True

    def test_credit_is_parallel_branch_property(self):
        c1 = _make_sequential_credit()
        assert not c1.is_parallel_branch

        c2 = _make_parallel_credit(parallel_group="g2", parallel_branch=3)
        assert c2.is_parallel_branch


class TestTurnToSendParallelFields:
    """TurnToSend parallel fields and for_parallel_branch factory."""

    def test_defaults_none(self):
        t = TurnToSend(
            conversation_id="c1",
            x_correlation_id="x1",
            turn_index=0,
            num_turns=5,
        )
        assert t.parallel_group is None
        assert t.parallel_branch is None
        assert t.parent_correlation_id is None
        assert not t.is_parallel_branch

    def test_for_parallel_branch_factory(self):
        t = TurnToSend.for_parallel_branch(
            conversation_id="conv_0",
            parent_correlation_id="parent-uuid",
            turn_index=3,
            num_turns=10,
            parallel_group="g0",
            parallel_branch=2,
        )
        assert t.conversation_id == "conv_0"
        assert t.x_correlation_id == "parent-uuid_b2"
        assert t.turn_index == 3
        assert t.num_turns == 10
        assert t.parallel_group == "g0"
        assert t.parallel_branch == 2
        assert t.parent_correlation_id == "parent-uuid"
        assert t.is_parallel_branch

    def test_for_parallel_branch_correlation_id_format(self):
        for i in range(5):
            t = TurnToSend.for_parallel_branch(
                conversation_id="c",
                parent_correlation_id="abc-123",
                turn_index=1,
                num_turns=10,
                parallel_group="g0",
                parallel_branch=i,
            )
            assert t.x_correlation_id == f"abc-123_b{i}"

    def test_from_previous_credit_no_parallel_fields(self):
        """from_previous_credit should NOT carry parallel fields (join is sequential)."""
        c = _make_parallel_credit(turn_index=3, num_turns=10)
        t = TurnToSend.from_previous_credit(c)
        assert t.turn_index == 4
        assert t.parallel_group is None
        assert t.parallel_branch is None
        assert t.parent_correlation_id is None

    def test_is_final_turn(self):
        t = TurnToSend.for_parallel_branch(
            conversation_id="c",
            parent_correlation_id="p",
            turn_index=9,
            num_turns=10,
            parallel_group="g0",
            parallel_branch=0,
        )
        assert t.is_final_turn


# =============================================================================
# Phase 2: Config Tests
# =============================================================================


class TestCodingSessionParallelConfig:
    """CodingSessionConfig parallel parameters."""

    def test_defaults(self):
        cfg = CodingSessionConfig(enabled=True)
        assert cfg.parallel_probability == 0.3
        assert cfg.parallel_fan_out_mean == 3.0
        assert cfg.parallel_fan_out_max == 8
        assert cfg.parallel_branch_tokens_mean == 800
        assert cfg.parallel_branch_tokens_median == 400

    def test_disable_parallel(self):
        cfg = CodingSessionConfig(enabled=True, parallel_probability=0.0)
        assert cfg.parallel_probability == 0.0

    def test_custom_values(self):
        cfg = CodingSessionConfig(
            enabled=True,
            parallel_probability=0.5,
            parallel_fan_out_mean=4.0,
            parallel_fan_out_max=6,
            parallel_branch_tokens_mean=1200,
            parallel_branch_tokens_median=600,
        )
        assert cfg.parallel_probability == 0.5
        assert cfg.parallel_fan_out_mean == 4.0
        assert cfg.parallel_fan_out_max == 6
        assert cfg.parallel_branch_tokens_mean == 1200
        assert cfg.parallel_branch_tokens_median == 600


# =============================================================================
# Phase 3: Dataset Generation Tests
# =============================================================================


class TestCodingSessionParallelGeneration:
    """CodingSessionComposer parallel group generation."""

    @pytest.fixture
    def parallel_config(self):
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
                    parallel_probability=1.0,
                    parallel_fan_out_mean=3.0,
                    parallel_fan_out_max=5,
                    parallel_branch_tokens_mean=200,
                    parallel_branch_tokens_median=100,
                ),
                prompt=PromptConfig(),
            ),
        )

    @pytest.fixture
    def no_parallel_config(self):
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
                    parallel_probability=0.0,
                ),
                prompt=PromptConfig(),
            ),
        )

    def test_parallel_probability_1_generates_parallel_turns(
        self, parallel_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(parallel_config, mock_tokenizer)
        conversations = composer.create_dataset()

        has_parallel = False
        for conv in conversations:
            for turn in conv.turns:
                if turn.parallel_group is not None:
                    has_parallel = True
                    break
        assert has_parallel

    def test_parallel_probability_0_generates_no_parallel_turns(
        self, no_parallel_config, mock_tokenizer
    ):
        composer = CodingSessionComposer(no_parallel_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns:
                assert turn.parallel_group is None
                assert turn.parallel_branch is None

    def test_parallel_branches_share_parent_prefix(
        self, parallel_config, mock_tokenizer
    ):
        """All branches in a group should share the parent's hash_id prefix."""
        composer = CodingSessionComposer(parallel_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            # Group branches by parallel_group
            groups: dict[str, list[Turn]] = {}
            for turn in conv.turns:
                if turn.parallel_group is not None:
                    groups.setdefault(turn.parallel_group, []).append(turn)

            for _group_id, branches in groups.items():
                if len(branches) < 2:
                    continue
                # All branches should share a common prefix in hash_ids
                min_len = min(len(b.hash_ids) for b in branches)
                if min_len == 0:
                    continue
                # Find the preceding sequential turn's hash_ids count
                idx = conv.turns.index(branches[0])
                if idx > 0:
                    parent_hash_count = len(conv.turns[idx - 1].hash_ids)
                    for branch in branches:
                        # Branch hash_ids should start with parent prefix
                        assert len(branch.hash_ids) >= parent_hash_count

    def test_parallel_branches_have_correct_annotations(
        self, parallel_config, mock_tokenizer
    ):
        """Parallel turns have group and branch set, sequential turns don't."""
        composer = CodingSessionComposer(parallel_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns:
                if turn.parallel_group is not None:
                    assert turn.parallel_branch is not None
                    assert turn.parallel_branch >= 0
                else:
                    assert turn.parallel_branch is None

    def test_parallel_branch_input_tokens_within_max(
        self, parallel_config, mock_tokenizer
    ):
        """Parallel branch input_tokens should be clamped to max_prompt_tokens."""
        max_tokens = parallel_config.input.coding_session.max_prompt_tokens
        composer = CodingSessionComposer(parallel_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            for turn in conv.turns:
                assert turn.input_tokens <= max_tokens

    def test_join_turn_follows_parallel_group(self, parallel_config, mock_tokenizer):
        """A sequential (join) turn should follow each parallel group."""
        composer = CodingSessionComposer(parallel_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conv in conversations:
            seen_groups: set[str] = set()
            for i, turn in enumerate(conv.turns):
                if turn.parallel_group is not None:
                    seen_groups.add(turn.parallel_group)
                elif (
                    seen_groups
                    and i > 0
                    and conv.turns[i - 1].parallel_group is not None
                ):
                    assert turn.parallel_group is None

    def test_conversation_metadata_has_parallel_groups(
        self, parallel_config, mock_tokenizer
    ):
        """ConversationMetadata.parallel_groups populated from generated turns."""
        composer = CodingSessionComposer(parallel_config, mock_tokenizer)
        conversations = composer.create_dataset()

        has_groups = False
        for conv in conversations:
            meta = conv.metadata()
            if meta.parallel_groups:
                has_groups = True
                for pg in meta.parallel_groups:
                    assert len(pg.turn_indices) >= 2
                    assert pg.join_turn_index > max(pg.turn_indices)
        assert has_groups

    def test_fan_out_clamped_to_range(self, parallel_config, mock_tokenizer):
        """Fan-out should be at least 2 and at most parallel_fan_out_max."""
        composer = CodingSessionComposer(parallel_config, mock_tokenizer)
        conversations = composer.create_dataset()
        max_branches = parallel_config.input.coding_session.parallel_fan_out_max

        for conv in conversations:
            groups: dict[str, int] = {}
            for turn in conv.turns:
                if turn.parallel_group is not None:
                    groups[turn.parallel_group] = groups.get(turn.parallel_group, 0) + 1
            for count in groups.values():
                assert count >= 2
                assert count <= max_branches


# =============================================================================
# Phase 4: ConversationSource Tests
# =============================================================================


class TestConversationSourceParallel:
    """ConversationSource parallel group and turn metadata access."""

    @pytest.fixture
    def src_with_parallel(self):
        ds = _make_dataset_with_parallel(
            num_conversations=3,
            turns_per_conv=8,
            parallel_at=2,
            num_branches=3,
        )
        return _make_conversation_source(ds), ds

    def test_get_parallel_group_found(self, src_with_parallel):
        src, ds = src_with_parallel
        # The dataset should have parallel groups
        conv = ds.conversations[0]
        if conv.parallel_groups:
            pg = src.get_parallel_group(
                conv.conversation_id, conv.parallel_groups[0].group_id
            )
            assert pg is not None
            assert len(pg.turn_indices) == 3

    def test_get_parallel_group_not_found(self, src_with_parallel):
        src, ds = src_with_parallel
        result = src.get_parallel_group("conv_0", "nonexistent")
        assert result is None

    def test_get_turn_metadata_at_valid(self, src_with_parallel):
        src, ds = src_with_parallel
        meta = src.get_turn_metadata_at("conv_0", 0)
        assert meta is not None
        assert meta.input_tokens is not None

    def test_get_turn_metadata_at_out_of_range(self, src_with_parallel):
        src, ds = src_with_parallel
        with pytest.raises(ValueError, match="No turn"):
            src.get_turn_metadata_at("conv_0", 999)

    def test_get_turn_metadata_at_parallel_turn(self, src_with_parallel):
        src, ds = src_with_parallel
        conv = ds.conversations[0]
        # Find a parallel turn
        for i, t in enumerate(conv.turns):
            if t.parallel_group is not None:
                meta = src.get_turn_metadata_at(conv.conversation_id, i)
                assert meta.parallel_group is not None
                assert meta.parallel_branch is not None
                break


# =============================================================================
# Phase 5: Credit Issuer Tests
# =============================================================================


class TestCreditIssuerParallelBranches:
    """CreditIssuer parallel branch slot and field handling."""

    @pytest.fixture
    def issuer_deps(self):
        stop_checker = MagicMock()
        stop_checker.can_send_any_turn = MagicMock(return_value=True)
        stop_checker.can_start_new_session = MagicMock(return_value=True)

        progress = MagicMock()
        progress.increment_sent = MagicMock(return_value=(1, False))
        progress.freeze_sent_counts = MagicMock()
        progress.all_credits_sent_event = asyncio.Event()

        concurrency = MagicMock()
        concurrency.acquire_session_slot = AsyncMock(return_value=True)
        concurrency.acquire_prefill_slot = AsyncMock(return_value=True)
        concurrency.release_session_slot = MagicMock()
        concurrency.try_acquire_session_slot = MagicMock(return_value=True)
        concurrency.try_acquire_prefill_slot = MagicMock(return_value=True)
        concurrency.session_slot_available = MagicMock(return_value=True)

        router = MagicMock()
        router.send_credit = AsyncMock()

        cancellation = MagicMock()
        cancellation.next_cancellation_delay_ns = MagicMock(return_value=None)

        lifecycle = MagicMock()
        lifecycle.started_at_ns = time.time_ns()
        lifecycle.started_at_perf_ns = time.perf_counter_ns()

        issuer = CreditIssuer(
            phase=CreditPhase.PROFILING,
            stop_checker=stop_checker,
            progress=progress,
            concurrency_manager=concurrency,
            credit_router=router,
            cancellation_policy=cancellation,
            lifecycle=lifecycle,
        )
        return issuer, concurrency, router, progress

    @pytest.mark.asyncio
    async def test_parallel_branch_skips_session_slot(self, issuer_deps):
        """Parallel branches should NOT acquire session slots."""
        issuer, concurrency, router, _ = issuer_deps

        turn = TurnToSend.for_parallel_branch(
            conversation_id="conv_0",
            parent_correlation_id="parent-uuid",
            turn_index=2,
            num_turns=10,
            parallel_group="g0",
            parallel_branch=0,
        )
        await issuer.issue_credit(turn)

        concurrency.acquire_session_slot.assert_not_awaited()
        concurrency.acquire_prefill_slot.assert_awaited_once()
        router.send_credit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_parallel_branch_acquires_prefill_slot(self, issuer_deps):
        """Each parallel branch should acquire its own prefill slot."""
        issuer, concurrency, _, _ = issuer_deps

        for i in range(3):
            turn = TurnToSend.for_parallel_branch(
                conversation_id="conv_0",
                parent_correlation_id="parent-uuid",
                turn_index=2 + i,
                num_turns=10,
                parallel_group="g0",
                parallel_branch=i,
            )
            await issuer.issue_credit(turn)

        assert concurrency.acquire_prefill_slot.await_count == 3
        assert concurrency.acquire_session_slot.await_count == 0

    @pytest.mark.asyncio
    async def test_first_sequential_turn_acquires_session_slot(self, issuer_deps):
        """A normal first turn (turn_index=0, no parallel) acquires session slot."""
        issuer, concurrency, _, _ = issuer_deps

        turn = TurnToSend(
            conversation_id="conv_0",
            x_correlation_id="corr-1",
            turn_index=0,
            num_turns=5,
        )
        await issuer.issue_credit(turn)

        concurrency.acquire_session_slot.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_credit_carries_parallel_fields(self, issuer_deps):
        """Credit sent to router should carry parallel fields from TurnToSend."""
        issuer, _, router, _ = issuer_deps

        turn = TurnToSend.for_parallel_branch(
            conversation_id="conv_0",
            parent_correlation_id="parent-uuid",
            turn_index=3,
            num_turns=10,
            parallel_group="g0",
            parallel_branch=1,
        )
        await issuer.issue_credit(turn)

        sent_credit = router.send_credit.call_args[1]["credit"]
        assert sent_credit.parallel_group == "g0"
        assert sent_credit.parallel_branch == 1
        assert sent_credit.parent_correlation_id == "parent-uuid"

    @pytest.mark.asyncio
    async def test_try_issue_credit_parallel_branch_skips_session(self, issuer_deps):
        """try_issue_credit for parallel branch skips session slot."""
        issuer, concurrency, _, _ = issuer_deps

        turn = TurnToSend.for_parallel_branch(
            conversation_id="conv_0",
            parent_correlation_id="parent-uuid",
            turn_index=2,
            num_turns=10,
            parallel_group="g0",
            parallel_branch=0,
        )
        result = await issuer.try_issue_credit(turn)
        assert result is True

        concurrency.try_acquire_session_slot.assert_not_called()
        concurrency.try_acquire_prefill_slot.assert_called_once()


# =============================================================================
# Phase 6: Sticky Router Tests
# =============================================================================


class TestStickyRouterParallelBranches:
    """StickyCreditRouter parallel branch routing via parent_correlation_id."""

    @pytest.mark.asyncio
    async def test_parallel_branch_routes_to_parent_worker(self, service_config):
        from aiperf.credit.sticky_router import StickyCreditRouter

        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-1")
        router._register_worker("worker-2")

        # First, route parent session to worker-1
        parent_credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv_0",
            x_correlation_id="parent-corr",
            turn_index=0,
            num_turns=10,
            issued_at_ns=0,
        )
        await router.send_credit(parent_credit)
        parent_worker = router._router_client.send_to.call_args[0][0]

        # Now route a parallel branch with parent_correlation_id
        branch_credit = _make_parallel_credit(
            credit_id=2,
            parent_correlation_id="parent-corr",
            parallel_branch=0,
        )
        await router.send_credit(branch_credit)

        branch_worker = router._router_client.send_to.call_args[0][0]
        assert branch_worker == parent_worker

    @pytest.mark.asyncio
    async def test_parallel_branch_does_not_create_sticky_session(self, service_config):
        from aiperf.credit.sticky_router import StickyCreditRouter

        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-1")

        # Route parent first
        parent_credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv_0",
            x_correlation_id="parent-corr",
            turn_index=0,
            num_turns=10,
            issued_at_ns=0,
        )
        await router.send_credit(parent_credit)
        sessions_before = len(router._sticky_sessions)

        # Route parallel branch
        branch_credit = _make_parallel_credit(
            credit_id=2,
            parent_correlation_id="parent-corr",
            parallel_branch=0,
        )
        await router.send_credit(branch_credit)

        # Branch should not add a new sticky session
        assert len(router._sticky_sessions) == sessions_before

    @pytest.mark.asyncio
    async def test_multiple_branches_route_to_same_worker(self, service_config):
        from aiperf.credit.sticky_router import StickyCreditRouter

        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-1")
        router._register_worker("worker-2")

        parent_credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv_0",
            x_correlation_id="parent-corr",
            turn_index=0,
            num_turns=10,
            issued_at_ns=0,
        )
        await router.send_credit(parent_credit)
        parent_worker = router._router_client.send_to.call_args[0][0]

        workers_used = set()
        for i in range(5):
            branch = _make_parallel_credit(
                credit_id=10 + i,
                parent_correlation_id="parent-corr",
                parallel_branch=i,
            )
            await router.send_credit(branch)
            workers_used.add(router._router_client.send_to.call_args[0][0])

        assert workers_used == {parent_worker}


# =============================================================================
# Phase 7: Adaptive Scale Strategy Tests
# =============================================================================


class TestAdaptiveScaleParallelDispatch:
    """AdaptiveScaleStrategy parallel fan-out, join, and branch return."""

    @pytest.mark.asyncio
    async def test_parallel_branch_return_tracks_completion(self):
        """Returning a parallel branch increments pending join counter."""
        strategy, scheduler, issuer, _, ds = _make_adaptive_strategy()
        await strategy.setup_phase()

        parent_corr = "parent-corr-1"
        strategy._pending_joins[parent_corr] = PendingJoin(
            conversation_id="conv_0",
            parent_correlation_id=parent_corr,
            expected_count=3,
            join_turn_index=5,
            num_turns=10,
        )

        # Return first branch
        branch_credit = _make_parallel_credit(
            credit_id=1,
            parent_correlation_id=parent_corr,
            parallel_branch=0,
        )
        await strategy.handle_credit_return(branch_credit)

        assert strategy._pending_joins[parent_corr].completed_count == 1
        # Join not dispatched yet
        scheduler.execute_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_branches_complete_dispatches_join(self):
        """When all branches complete, the join turn is dispatched."""
        strategy, scheduler, issuer, _, ds = _make_adaptive_strategy()
        await strategy.setup_phase()

        parent_corr = "parent-corr-2"
        strategy._pending_joins[parent_corr] = PendingJoin(
            conversation_id="conv_0",
            parent_correlation_id=parent_corr,
            expected_count=3,
            join_turn_index=5,
            num_turns=10,
        )

        # Return all 3 branches
        for i in range(3):
            branch_credit = _make_parallel_credit(
                credit_id=10 + i,
                parent_correlation_id=parent_corr,
                parallel_branch=i,
            )
            await strategy.handle_credit_return(branch_credit)

        # Join should be dispatched
        assert scheduler.execute_async.call_count == 1
        # Pending join should be cleaned up
        assert parent_corr not in strategy._pending_joins

    @pytest.mark.asyncio
    async def test_partial_branch_completion_waits(self):
        """Returning 2 of 3 branches should NOT dispatch join."""
        strategy, scheduler, issuer, _, ds = _make_adaptive_strategy()
        await strategy.setup_phase()

        parent_corr = "parent-corr-3"
        strategy._pending_joins[parent_corr] = PendingJoin(
            conversation_id="conv_0",
            parent_correlation_id=parent_corr,
            expected_count=3,
            join_turn_index=5,
            num_turns=10,
        )

        for i in range(2):
            branch_credit = _make_parallel_credit(
                credit_id=20 + i,
                parent_correlation_id=parent_corr,
                parallel_branch=i,
            )
            await strategy.handle_credit_return(branch_credit)

        assert scheduler.execute_async.call_count == 0
        assert parent_corr in strategy._pending_joins
        assert strategy._pending_joins[parent_corr].completed_count == 2

    @pytest.mark.asyncio
    async def test_sequential_return_before_parallel_dispatches_group(self):
        """When a sequential credit returns and next turn is parallel, fan-out."""
        strategy, scheduler, issuer, _, ds = _make_adaptive_strategy(
            num_branches=3, parallel_at=2, turns_per_conv=8
        )
        await strategy.setup_phase()

        # The next turn (index 2) should be in a parallel group
        conv = ds.conversations[0]
        assert conv.turns[2].parallel_group is not None

        # Return credit for turn 1 (sequential, before parallel group)
        credit = _make_sequential_credit(
            conv_id="conv_0",
            corr_id="xcorr-dispatch",
            turn_index=1,
            num_turns=len(conv.turns),
        )
        await strategy.handle_credit_return(credit)

        # Should dispatch parallel branches (3 calls to execute_async)
        assert scheduler.execute_async.call_count == 3
        # Should register a pending join
        assert "xcorr-dispatch" in strategy._pending_joins
        pending = strategy._pending_joins["xcorr-dispatch"]
        assert pending.expected_count == 3

    @pytest.mark.asyncio
    async def test_sequential_return_normal_dispatches_next(self):
        """Normal sequential return dispatches next turn normally."""
        strategy, scheduler, issuer, _, ds = _make_adaptive_strategy(
            parallel_at=None  # No parallel groups
        )
        await strategy.setup_phase()

        credit = _make_sequential_credit(
            conv_id="conv_0",
            corr_id="xcorr-normal",
            turn_index=0,
            num_turns=8,
        )
        await strategy.handle_credit_return(credit)

        # Should schedule or execute next turn
        assert (
            scheduler.schedule_later.call_count == 1
            or scheduler.execute_async.call_count == 1
        )

    @pytest.mark.asyncio
    async def test_final_turn_cleans_up_pending_joins(self):
        """Session cleanup on final turn removes pending join state."""
        strategy, _, _, _, _ = _make_adaptive_strategy()
        strategy._active_users = 3

        corr_id = "xcorr-cleanup"
        strategy._pending_joins[corr_id] = PendingJoin(
            conversation_id="conv_0",
            parent_correlation_id=corr_id,
            expected_count=3,
            join_turn_index=5,
            num_turns=10,
        )

        credit = _make_sequential_credit(
            conv_id="conv_0",
            corr_id=corr_id,
            turn_index=7,
            num_turns=8,
        )
        await strategy.handle_credit_return(credit)

        assert corr_id not in strategy._pending_joins
        assert strategy._active_users == 2

    @pytest.mark.asyncio
    async def test_parallel_branch_return_without_pending_join_noop(self):
        """Branch return without pending join is a no-op (no crash)."""
        strategy, scheduler, _, _, _ = _make_adaptive_strategy()
        await strategy.setup_phase()

        branch_credit = _make_parallel_credit(
            credit_id=99,
            parent_correlation_id="nonexistent-parent",
            parallel_branch=0,
        )
        await strategy.handle_credit_return(branch_credit)

        scheduler.execute_async.assert_not_called()


class TestPendingJoin:
    """PendingJoin dataclass tests."""

    def test_defaults(self):
        pj = PendingJoin(
            conversation_id="c1",
            parent_correlation_id="p1",
            expected_count=4,
            join_turn_index=6,
            num_turns=10,
        )
        assert pj.completed_count == 0

    def test_increment(self):
        pj = PendingJoin(
            conversation_id="c1",
            parent_correlation_id="p1",
            expected_count=3,
            join_turn_index=5,
            num_turns=10,
        )
        pj.completed_count += 1
        assert pj.completed_count == 1
        pj.completed_count += 1
        assert pj.completed_count == 2


# =============================================================================
# Phase 8: Worker Session Manager Tests
# =============================================================================


class TestSessionManagerChildParentMapping:
    """UserSessionManager child-to-parent mapping for parallel branches."""

    @pytest.fixture
    def manager(self):
        return UserSessionManager()

    @pytest.fixture
    def conversation(self):
        return Conversation(
            session_id="test-conv",
            turns=[Turn(max_tokens=100, input_tokens=500 + i * 100) for i in range(10)],
        )

    def test_register_child_enables_parent_lookup(self, manager, conversation):
        parent_session = manager.create_and_store(
            "parent-corr", conversation, num_turns=10
        )
        manager.register_child("parent-corr_b0", "parent-corr")

        found = manager.get("parent-corr_b0")
        assert found is parent_session

    def test_get_returns_direct_session_first(self, manager, conversation):
        """Direct cache lookup takes priority over child mapping."""
        manager.create_and_store("parent-corr", conversation, num_turns=10)
        child_conv = Conversation(
            session_id="child-conv",
            turns=[Turn(max_tokens=10, input_tokens=100)],
        )
        child = manager.create_and_store("child-corr", child_conv, num_turns=1)

        manager.register_child("child-corr", "parent-corr")

        # Direct lookup should return the child session, not parent
        found = manager.get("child-corr")
        assert found is child

    def test_evict_child_removes_mapping(self, manager, conversation):
        manager.create_and_store("parent-corr", conversation, num_turns=10)
        manager.register_child("parent-corr_b0", "parent-corr")

        manager.evict_child("parent-corr_b0")

        # After eviction, child ID should not resolve
        found = manager.get("parent-corr_b0")
        assert found is None

    def test_evict_child_preserves_parent(self, manager, conversation):
        parent = manager.create_and_store("parent-corr", conversation, num_turns=10)
        manager.register_child("parent-corr_b0", "parent-corr")
        manager.register_child("parent-corr_b1", "parent-corr")

        manager.evict_child("parent-corr_b0")

        # Parent still accessible directly
        assert manager.get("parent-corr") is parent
        # Other child still accessible
        assert manager.get("parent-corr_b1") is parent

    def test_multiple_children_map_to_same_parent(self, manager, conversation):
        parent = manager.create_and_store("parent-corr", conversation, num_turns=10)

        for i in range(5):
            manager.register_child(f"parent-corr_b{i}", "parent-corr")

        for i in range(5):
            assert manager.get(f"parent-corr_b{i}") is parent

    def test_evict_parent_does_not_affect_child_mapping(self, manager, conversation):
        """Evicting parent removes session but child mapping still exists (returns None)."""
        manager.create_and_store("parent-corr", conversation, num_turns=10)
        manager.register_child("parent-corr_b0", "parent-corr")

        manager.evict("parent-corr")

        # Child mapping exists but parent is gone, so returns None
        assert manager.get("parent-corr_b0") is None

    def test_get_unknown_returns_none(self, manager):
        assert manager.get("nonexistent") is None

    def test_evict_child_unknown_noop(self, manager):
        """Evicting unknown child is a no-op."""
        manager.evict_child("nonexistent")  # Should not raise


# =============================================================================
# End-to-End Integration: Data Model -> Metadata -> Strategy
# =============================================================================


class TestParallelEndToEnd:
    """End-to-end tests verifying the full pipeline from data model to strategy."""

    def test_conversation_with_parallel_produces_correct_metadata(self):
        """Build a conversation with parallel turns and verify metadata is correct."""
        turns = [
            Turn(max_tokens=50, input_tokens=100),
            Turn(
                max_tokens=50, input_tokens=200, parallel_group="g0", parallel_branch=0
            ),
            Turn(
                max_tokens=50, input_tokens=200, parallel_group="g0", parallel_branch=1
            ),
            Turn(
                max_tokens=50, input_tokens=200, parallel_group="g0", parallel_branch=2
            ),
            Turn(max_tokens=50, input_tokens=400),
            Turn(max_tokens=50, input_tokens=500),
        ]
        conv = Conversation(session_id="e2e", turns=turns)
        meta = conv.metadata()

        assert len(meta.turns) == 6
        assert len(meta.parallel_groups) == 1

        pg = meta.parallel_groups[0]
        assert pg.group_id == "g0"
        assert pg.turn_indices == [1, 2, 3]
        assert pg.join_turn_index == 4

        # Verify turn metadata has parallel annotations
        assert meta.turns[0].parallel_group is None
        assert meta.turns[1].parallel_group == "g0"
        assert meta.turns[1].parallel_branch == 0
        assert meta.turns[4].parallel_group is None

    def test_dataset_metadata_preserves_parallel_groups(self):
        """DatasetMetadata from conversations preserves parallel group info."""
        turns = [
            Turn(max_tokens=50, input_tokens=100),
            Turn(
                max_tokens=50, input_tokens=200, parallel_group="g0", parallel_branch=0
            ),
            Turn(
                max_tokens=50, input_tokens=200, parallel_group="g0", parallel_branch=1
            ),
            Turn(max_tokens=50, input_tokens=400),
        ]
        conv = Conversation(session_id="e2e-ds", turns=turns)
        ds = DatasetMetadata(
            conversations=[conv.metadata()],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )

        assert len(ds.conversations) == 1
        assert len(ds.conversations[0].parallel_groups) == 1

    @pytest.mark.asyncio
    async def test_full_parallel_dispatch_cycle(self):
        """Full cycle: sequential -> fan-out -> branch returns -> join dispatch."""
        strategy, scheduler, issuer, _, ds = _make_adaptive_strategy(
            num_conversations=5,
            turns_per_conv=8,
            parallel_at=2,
            num_branches=3,
        )
        await strategy.setup_phase()

        conv = ds.conversations[0]
        num_turns = len(conv.turns)

        # Step 1: Return sequential credit at turn_index=1
        # Next turn (2) should be in a parallel group
        credit = _make_sequential_credit(
            conv_id="conv_0",
            corr_id="xcorr-e2e",
            turn_index=1,
            num_turns=num_turns,
        )
        await strategy.handle_credit_return(credit)

        # Should have dispatched 3 parallel branches
        assert scheduler.execute_async.call_count == 3
        assert "xcorr-e2e" in strategy._pending_joins
        pending = strategy._pending_joins["xcorr-e2e"]
        assert pending.expected_count == 3
        assert pending.completed_count == 0

        # Step 2: Return all branches
        scheduler.execute_async.reset_mock()
        for i in range(3):
            branch = _make_parallel_credit(
                credit_id=100 + i,
                conv_id="conv_0",
                parent_correlation_id="xcorr-e2e",
                parallel_branch=i,
                turn_index=2 + i,
                num_turns=num_turns,
            )
            await strategy.handle_credit_return(branch)

        # Join should be dispatched after all branches return
        assert scheduler.execute_async.call_count == 1
        assert "xcorr-e2e" not in strategy._pending_joins
