# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pytest import param

from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    SubagentSpawnInfo,
    TurnMetadata,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import DatasetSamplingStrategy, PluginType
from aiperf.timing.conversation_source import ConversationSource, SampledSession
from tests.unit.timing.conftest import make_credit


def _mk_source(ds: DatasetMetadata) -> ConversationSource:
    SamplerClass = plugins.get_class(PluginType.DATASET_SAMPLER, ds.sampling_strategy)
    sampler = SamplerClass(
        conversation_ids=[c.conversation_id for c in ds.conversations],
    )
    return ConversationSource(ds, sampler)


@pytest.fixture
def ds():
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="c1",
                turns=[TurnMetadata(timestamp_ms=0.0), TurnMetadata(delay_ms=100.0)],
            ),
            ConversationMetadata(
                conversation_id="c2", turns=[TurnMetadata(timestamp_ms=50.0)]
            ),
            ConversationMetadata(
                conversation_id="c3",
                turns=[
                    TurnMetadata(timestamp_ms=100.0),
                    TurnMetadata(delay_ms=50.0),
                    TurnMetadata(delay_ms=75.0),
                ],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


@pytest.fixture
def src(ds):
    return _mk_source(ds)


class TestConversationSource:
    def test_next_returns_sampled_session(self, src):
        s = src.next()
        assert isinstance(s, SampledSession)
        assert s.conversation_id in ["c1", "c2", "c3"]
        assert s.metadata is not None
        assert len(s.x_correlation_id) == 36

    def test_unique_correlation_ids(self, src):
        assert src.next().x_correlation_id != src.next().x_correlation_id

    def test_sequential_order(self, ds):
        src = _mk_source(ds)
        assert [src.next().conversation_id for _ in range(3)] == ["c1", "c2", "c3"]

    def test_get_metadata_returns_conversation(self, src):
        m = src.get_metadata("c1")
        assert m.conversation_id == "c1"
        assert len(m.turns) == 2

    def test_get_metadata_raises_for_invalid(self, src):
        with pytest.raises(KeyError, match="No metadata for conversation bad"):
            src.get_metadata("bad")


class TestMultiTurn:
    @pytest.fixture
    def mt_src(self):
        ds = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="mt",
                    turns=[
                        TurnMetadata(timestamp_ms=1000.0),
                        TurnMetadata(delay_ms=50.0),
                        TurnMetadata(delay_ms=100.0),
                    ],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        return _mk_source(ds)

    @pytest.mark.parametrize("turn,exp_delay", [(0, 50.0), (1, 100.0)])  # fmt: skip
    def test_get_next_turn_metadata(self, mt_src, turn, exp_delay):
        cr = make_credit(conv_id="mt", turn=turn, is_final=False)
        assert mt_src.get_next_turn_metadata(cr).delay_ms == exp_delay

    def test_raises_when_no_next_turn(self, mt_src):
        cr = make_credit(conv_id="mt", turn=2, is_final=True)
        with pytest.raises(ValueError, match="No turn 3"):
            mt_src.get_next_turn_metadata(cr)


# ============================================================
# get_turn_metadata_at
# ============================================================


class TestGetTurnMetadataAt:
    """Verify get_turn_metadata_at index lookup and guard rails."""

    @pytest.fixture
    def src_3turn(self) -> ConversationSource:
        ds = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="t3",
                    turns=[
                        TurnMetadata(timestamp_ms=0.0),
                        TurnMetadata(delay_ms=100.0),
                        TurnMetadata(delay_ms=200.0),
                    ],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        return _mk_source(ds)

    @pytest.mark.parametrize(
        "index,expected_delay",
        [
            (0, None),
            (1, 100.0),
            (2, 200.0),
        ],
    )  # fmt: skip
    def test_valid_index_returns_correct_turn(
        self, src_3turn: ConversationSource, index: int, expected_delay: float | None
    ) -> None:
        turn = src_3turn.get_turn_metadata_at("t3", index)
        assert turn.delay_ms == expected_delay

    @pytest.mark.parametrize(
        "index",
        [
            param(-1, id="negative-index"),
            param(-100, id="large-negative-index"),
        ],
    )  # fmt: skip
    def test_negative_index_raises_value_error(
        self, src_3turn: ConversationSource, index: int
    ) -> None:
        with pytest.raises(ValueError, match="No turn"):
            src_3turn.get_turn_metadata_at("t3", index)

    @pytest.mark.parametrize(
        "index",
        [
            param(3, id="one-past-end"),
            param(100, id="far-past-end"),
        ],
    )  # fmt: skip
    def test_out_of_bounds_index_raises_value_error(
        self, src_3turn: ConversationSource, index: int
    ) -> None:
        with pytest.raises(ValueError, match="only 3 turns exist"):
            src_3turn.get_turn_metadata_at("t3", index)

    def test_unknown_conversation_raises_key_error(
        self, src_3turn: ConversationSource
    ) -> None:
        with pytest.raises(KeyError, match="No metadata for conversation"):
            src_3turn.get_turn_metadata_at("nonexistent", 0)


# ============================================================
# SampledSession.build_first_turn
# ============================================================


class TestBuildFirstTurn:
    """Verify build_first_turn agent_depth and max_turns truncation."""

    @pytest.fixture
    def session_5turn(self) -> SampledSession:
        meta = ConversationMetadata(
            conversation_id="bft",
            turns=[TurnMetadata(delay_ms=float(i * 10)) for i in range(5)],
        )
        return SampledSession(
            conversation_id="bft",
            metadata=meta,
            x_correlation_id="fixed-xcorr",
        )

    def test_agent_depth_zero_default(self, session_5turn: SampledSession) -> None:
        turn = session_5turn.build_first_turn()
        assert turn.agent_depth == 0

    @pytest.mark.parametrize("depth", [1, 2, 5])  # fmt: skip
    def test_nonzero_agent_depth_propagates(
        self, session_5turn: SampledSession, depth: int
    ) -> None:
        turn = session_5turn.build_first_turn(agent_depth=depth)
        assert turn.agent_depth == depth
        assert turn.turn_index == 0
        assert turn.conversation_id == "bft"

    def test_max_turns_none_uses_conversation_length(
        self, session_5turn: SampledSession
    ) -> None:
        turn = session_5turn.build_first_turn(max_turns=None)
        assert turn.num_turns == 5

    @pytest.mark.parametrize(
        "max_turns,expected",
        [
            (1, 1),
            (3, 3),
            param(10, 10, id="max-exceeds-conversation-length"),
        ],
    )  # fmt: skip
    def test_max_turns_truncates_num_turns(
        self, session_5turn: SampledSession, max_turns: int, expected: int
    ) -> None:
        turn = session_5turn.build_first_turn(max_turns=max_turns)
        assert turn.num_turns == expected

    def test_max_turns_and_agent_depth_combined(
        self, session_5turn: SampledSession
    ) -> None:
        turn = session_5turn.build_first_turn(max_turns=2, agent_depth=1)
        assert turn.num_turns == 2
        assert turn.agent_depth == 1
        assert turn.x_correlation_id == "fixed-xcorr"


# ============================================================
# next() with explicit x_correlation_id
# ============================================================


class TestNextExplicitCorrelationId:
    """Verify next() respects an explicit x_correlation_id parameter."""

    def test_explicit_correlation_id_used(self, src: ConversationSource) -> None:
        session = src.next(x_correlation_id="my-custom-id")
        assert session.x_correlation_id == "my-custom-id"

    def test_none_correlation_id_generates_uuid(self, src: ConversationSource) -> None:
        session = src.next(x_correlation_id=None)
        assert len(session.x_correlation_id) == 36

    def test_explicit_id_preserves_metadata(self, src: ConversationSource) -> None:
        session = src.next(x_correlation_id="explicit-123")
        assert session.metadata is not None
        assert session.conversation_id in ["c1", "c2", "c3"]


# ============================================================
# get_subagent_spawn: multiple spawns on a single conversation
# ============================================================


class TestGetSubagentSpawnMultiple:
    """Verify get_subagent_spawn when a conversation has multiple spawns."""

    @pytest.fixture
    def src_multi_spawn(self) -> ConversationSource:
        ds = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="parent",
                    turns=[TurnMetadata(delay_ms=float(i * 10)) for i in range(6)],
                    subagent_spawns=[
                        SubagentSpawnInfo(
                            spawn_id="s0",
                            child_conversation_ids=["child_a", "child_b"],
                            join_turn_index=2,
                        ),
                        SubagentSpawnInfo(
                            spawn_id="s1",
                            child_conversation_ids=["child_c"],
                            join_turn_index=4,
                        ),
                        SubagentSpawnInfo(
                            spawn_id="s2",
                            child_conversation_ids=["child_d", "child_e", "child_f"],
                            join_turn_index=5,
                        ),
                    ],
                ),
                ConversationMetadata(
                    conversation_id="child_a",
                    turns=[TurnMetadata()],
                    agent_depth=1,
                ),
                ConversationMetadata(
                    conversation_id="child_b",
                    turns=[TurnMetadata()],
                    agent_depth=1,
                ),
                ConversationMetadata(
                    conversation_id="child_c",
                    turns=[TurnMetadata()],
                    agent_depth=1,
                ),
                ConversationMetadata(
                    conversation_id="child_d",
                    turns=[TurnMetadata()],
                    agent_depth=1,
                ),
                ConversationMetadata(
                    conversation_id="child_e",
                    turns=[TurnMetadata()],
                    agent_depth=1,
                ),
                ConversationMetadata(
                    conversation_id="child_f",
                    turns=[TurnMetadata()],
                    agent_depth=1,
                ),
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        return _mk_source(ds)

    @pytest.mark.parametrize(
        "spawn_id,expected_children",
        [
            ("s0", ["child_a", "child_b"]),
            ("s1", ["child_c"]),
            ("s2", ["child_d", "child_e", "child_f"]),
        ],
    )  # fmt: skip
    def test_each_spawn_resolved_independently(
        self,
        src_multi_spawn: ConversationSource,
        spawn_id: str,
        expected_children: list[str],
    ) -> None:
        spawn = src_multi_spawn.get_subagent_spawn("parent", spawn_id)
        assert spawn is not None
        assert spawn.child_conversation_ids == expected_children

    def test_nonexistent_spawn_returns_none(
        self, src_multi_spawn: ConversationSource
    ) -> None:
        assert src_multi_spawn.get_subagent_spawn("parent", "s99") is None


# ============================================================
# start_child_session: invalid/unknown conversation_id
# ============================================================


class TestStartChildSessionErrors:
    """Verify start_child_session propagates errors for unknown conversations."""

    def test_unknown_conversation_id_raises_key_error(
        self, src: ConversationSource
    ) -> None:
        with pytest.raises(
            KeyError, match="No metadata for conversation unknown_child"
        ):
            src.start_child_session("unknown_child")

    def test_valid_child_returns_session(self, src: ConversationSource) -> None:
        session = src.start_child_session("c1")
        assert session.conversation_id == "c1"
        assert len(session.x_correlation_id) == 36
