# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.agentic_load import (
    AgenticLoadStrategy,
    AgenticUser,
)
from tests.unit.timing.conftest import OrchestratorHarness

TWO_TURN = [("c1", 2), ("c2", 2), ("c3", 2), ("c4", 2), ("c5", 2)]
THREE_TURN = [("c1", 3), ("c2", 3), ("c3", 3), ("c4", 3)]


class TestAgenticLoadInit:
    def test_missing_concurrency_raises(self) -> None:
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENTIC_LOAD,
            concurrency=None,
            total_expected_requests=10,
        )
        with pytest.raises(ValueError, match="concurrency must be set"):
            AgenticLoadStrategy(
                config=cfg,
                conversation_source=MagicMock(),
                scheduler=MagicMock(),
                stop_checker=MagicMock(),
                credit_issuer=MagicMock(),
                lifecycle=MagicMock(),
            )


class TestAgenticUser:
    def test_advance_trajectory_wraps_around(self) -> None:
        user = AgenticUser(user_id=0, trajectory_index=4)
        user.x_correlation_id = "old-session"
        user.advance_trajectory(num_conversations=5)
        assert user.trajectory_index == 0
        assert user.x_correlation_id is None

    def test_advance_trajectory_increments(self) -> None:
        user = AgenticUser(user_id=0, trajectory_index=2)
        user.advance_trajectory(num_conversations=5)
        assert user.trajectory_index == 3

    def test_dataclass_fields(self) -> None:
        user = AgenticUser(user_id=7, trajectory_index=3, x_correlation_id="abc")
        assert user.user_id == 7
        assert user.trajectory_index == 3
        assert user.x_correlation_id == "abc"


@pytest.mark.asyncio
class TestAgenticLoadExecution:
    @pytest.mark.parametrize(
        "convs,concurrency,count,expected",
        [
            (TWO_TURN * 4, 5, 10, 10),
            (THREE_TURN * 3, 4, 12, 12),
            (TWO_TURN * 2, 2, 4, 4),
            (TWO_TURN * 10, 10, 50, 50),
            (THREE_TURN * 5, 8, 30, 30),
            (TWO_TURN * 4, 1, 5, 5),
        ],
    )
    async def test_issues_expected_credits(
        self, create_orchestrator_harness, convs, concurrency, count, expected
    ) -> None:
        """Verify strategy issues correct number of credits across configurations."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=convs,
            timing_mode=TimingMode.AGENTIC_LOAD,
            concurrency=concurrency,
            request_count=count,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) >= expected

    async def test_spawns_all_users(self, create_orchestrator_harness) -> None:
        """Verify all N users get first turns issued."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * 4,
            timing_mode=TimingMode.AGENTIC_LOAD,
            concurrency=5,
            request_count=20,
        )
        await h.run_with_auto_return()
        # At least 5 unique correlation IDs (one per user)
        assert len({c.x_correlation_id for c in h.sent_credits}) >= 5


@pytest.mark.asyncio
class TestSessionTracking:
    async def test_multi_turn_shares_correlation(
        self, create_orchestrator_harness
    ) -> None:
        """Turns within a trajectory share the same correlation ID with sequential indices."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=THREE_TURN * 3,
            timing_mode=TimingMode.AGENTIC_LOAD,
            concurrency=3,
            request_count=20,
        )
        await h.run_with_auto_return()
        sessions: dict[str, list] = {}
        for c in h.sent_credits:
            sessions.setdefault(c.x_correlation_id, []).append(c)
        multi = [s for s in sessions.values() if len(s) > 1]
        assert len(multi) > 0
        for credits in multi:
            indices = [c.turn_index for c in credits]
            assert indices == sorted(indices)
            assert indices[0] == 0

    async def test_new_correlation_id_between_trajectories(
        self, create_orchestrator_harness
    ) -> None:
        """Each new trajectory gets a fresh x_correlation_id for sticky routing."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * 4,
            timing_mode=TimingMode.AGENTIC_LOAD,
            concurrency=2,
            request_count=12,
        )
        await h.run_with_auto_return()
        # Group by correlation ID: each session should be a single trajectory
        sessions: dict[str, list] = {}
        for c in h.sent_credits:
            sessions.setdefault(c.x_correlation_id, []).append(c)
        # Each session should have sequential turns starting at 0
        for credits in sessions.values():
            assert credits[0].turn_index == 0
            for i, c in enumerate(credits):
                assert c.turn_index == i


@pytest.mark.asyncio
class TestStopConditions:
    async def test_stops_at_request_count(self, create_orchestrator_harness) -> None:
        """Strategy should stop issuing credits when request count is reached."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * 10,
            timing_mode=TimingMode.AGENTIC_LOAD,
            concurrency=5,
            request_count=15,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == 15

    async def test_stops_at_session_count(self, create_orchestrator_harness) -> None:
        """Strategy should stop starting new sessions when session count is reached."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=THREE_TURN * 5,
            timing_mode=TimingMode.AGENTIC_LOAD,
            concurrency=4,
            num_sessions=6,
        )
        await h.run_with_auto_return()
        first_turns = [c for c in h.sent_credits if c.turn_index == 0]
        assert len(first_turns) == 6


@pytest.mark.asyncio
class TestDeterministicUserOffsets:
    async def test_offsets_independent_of_concurrency(
        self, create_orchestrator_harness
    ) -> None:
        """User i should get the same starting offset regardless of total concurrency.

        Run with different concurrency levels and verify user 0's first conversation
        is the same.
        """
        convs = [(f"c{i}", 2) for i in range(20)]

        h1: OrchestratorHarness = create_orchestrator_harness(
            conversations=convs,
            timing_mode=TimingMode.AGENTIC_LOAD,
            concurrency=5,
            request_count=5,
        )
        await h1.run_with_auto_return()
        first_conv_5 = h1.sent_credits[0].conversation_id

        h2: OrchestratorHarness = create_orchestrator_harness(
            conversations=convs,
            timing_mode=TimingMode.AGENTIC_LOAD,
            concurrency=15,
            request_count=15,
        )
        await h2.run_with_auto_return()
        first_conv_15 = h2.sent_credits[0].conversation_id

        # User 0's first conversation should be the same regardless of concurrency
        assert first_conv_5 == first_conv_15


@pytest.mark.asyncio
class TestWrapAround:
    async def test_user_wraps_after_all_conversations(
        self, create_orchestrator_harness
    ) -> None:
        """Users should wrap around to the beginning after finishing all conversations."""
        # 3 conversations, 1 user, enough credits to wrap around
        convs = [("c0", 1), ("c1", 1), ("c2", 1)]
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=convs,
            timing_mode=TimingMode.AGENTIC_LOAD,
            concurrency=1,
            request_count=6,
        )
        await h.run_with_auto_return()
        # With wrap-around, after 3 conversations the user should cycle back
        conv_ids = [c.conversation_id for c in h.sent_credits]
        assert len(conv_ids) == 6
        # The sequence should repeat
        assert conv_ids[:3] == conv_ids[3:]


@pytest.mark.asyncio
class TestZeroInterTurnDelay:
    async def test_credits_issued_immediately_on_return(
        self, create_orchestrator_harness
    ) -> None:
        """Credits should be issued immediately when a turn completes (no scheduling delay)."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=THREE_TURN * 3,
            timing_mode=TimingMode.AGENTIC_LOAD,
            concurrency=3,
            request_count=15,
        )
        await h.run_with_auto_return()
        # All credits should be issued (strategy doesn't use scheduling delays)
        assert len(h.sent_credits) >= 15
