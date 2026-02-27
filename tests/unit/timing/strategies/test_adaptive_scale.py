# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.strategies.adaptive_scale import AdaptiveScaleStrategy
from tests.unit.timing.conftest import make_sampler


@pytest.fixture
async def time_traveler(time_traveler_no_patch_sleep):
    return time_traveler_no_patch_sleep


def make_conversations(n: int) -> DatasetMetadata:
    """Create dataset metadata with n conversations, each with 5 turns."""
    convs = []
    for i in range(n):
        turns = [TurnMetadata(delay_ms=None)]
        turns.extend(TurnMetadata(delay_ms=2000.0) for _ in range(4))
        convs.append(ConversationMetadata(conversation_id=f"conv_{i}", turns=turns))
    return DatasetMetadata(
        conversations=convs,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def make_strategy(
    num_conversations: int = 20,
    start_users: int = 2,
    max_users: int | None = 10,
    max_ttft_sec: float = 2.0,
    assessment_period_sec: float = 5.0,
    recycle: bool = False,
    adaptive_scale_slo: dict[str, float] | None = None,
    min_goodput_ratio: float = 0.95,
) -> tuple[AdaptiveScaleStrategy, MagicMock, MagicMock, MagicMock]:
    scheduler = MagicMock()
    scheduler.schedule_later = MagicMock()
    scheduler.execute_async = MagicMock()
    stop_checker = MagicMock()
    stop_checker.can_send_any_turn = MagicMock(return_value=True)
    issuer = MagicMock()
    issuer.issue_credit = AsyncMock(return_value=True)
    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 1_000_000_000
    lifecycle.started_at_perf_sec = 1.0

    ds = make_conversations(num_conversations)
    sampler = make_sampler(
        [c.conversation_id for c in ds.conversations],
        DatasetSamplingStrategy.SEQUENTIAL,
    )
    src = ConversationSource(ds, sampler)

    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.ADAPTIVE_SCALE,
        expected_duration_sec=120.0,
        start_users=start_users,
        max_users=max_users,
        max_ttft_sec=max_ttft_sec,
        ttft_metric="p95",
        assessment_period_sec=assessment_period_sec,
        recycle_sessions=recycle,
        adaptive_scale_slo=adaptive_scale_slo,
        min_goodput_ratio=min_goodput_ratio,
    )

    strategy = AdaptiveScaleStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )

    return strategy, scheduler, issuer, stop_checker


def _make_credit(credit_id: int = 1, corr_id: str = "xcorr-1") -> Credit:
    return Credit(
        id=credit_id,
        phase=CreditPhase.PROFILING,
        conversation_id="conv_0",
        x_correlation_id=corr_id,
        turn_index=0,
        num_turns=5,
        issued_at_ns=0,
    )


def _make_credit_return(
    credit_id: int = 1,
    ttft_ns: int | None = None,
    request_latency_ns: int | None = None,
    cancelled: bool = False,
    corr_id: str = "xcorr-1",
) -> CreditReturn:
    return CreditReturn(
        credit=_make_credit(credit_id=credit_id, corr_id=corr_id),
        cancelled=cancelled,
        ttft_ns=ttft_ns,
        request_latency_ns=request_latency_ns,
    )


class TestAdaptiveScaleStrategy:
    def test_init_defaults(self):
        strategy, _, _, _ = make_strategy()
        assert strategy._start_users == 2
        assert strategy._max_users == 10
        assert strategy._max_ttft_sec == 2.0
        assert strategy._ttft_metric == "p95"

    @pytest.mark.asyncio
    async def test_setup_phase(self):
        strategy, _, _, _ = make_strategy()
        await strategy.setup_phase()
        # setup_phase is a no-op, just ensure it doesn't raise

    @pytest.mark.asyncio
    async def test_execute_phase_issues_initial_users(self):
        """Verify initial credits are issued for start_users."""
        strategy, scheduler, issuer, stop_checker = make_strategy(start_users=3)
        # Stop after initial credits by making sleep raise
        stop_checker.can_send_any_turn = MagicMock(
            side_effect=[True, True, True, False]
        )

        await strategy.setup_phase()
        await strategy.execute_phase()

        # All 3 users get issue_credit called; first is awaited directly,
        # remaining 2 are passed to scheduler.schedule_later
        assert issuer.issue_credit.call_count == 3
        assert issuer.issue_credit.await_count == 1  # only first awaited directly
        assert scheduler.schedule_later.call_count == 2  # staggered

    def test_on_ttft_sample(self):
        strategy, _, _, _ = make_strategy()
        strategy.on_ttft_sample(1_500_000_000)  # 1.5s
        strategy.on_ttft_sample(500_000_000)  # 0.5s
        assert len(strategy._period_ttft_samples) == 2
        assert strategy._period_ttft_samples[0] == pytest.approx(1.5)
        assert strategy._period_ttft_samples[1] == pytest.approx(0.5)

    def test_compute_ttft_metric_p95(self):
        strategy, _, _, _ = make_strategy()
        samples = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
        ]
        strategy._ttft_metric = "p95"
        result = strategy._compute_ttft_metric(samples)
        assert result == 1.9  # 95th percentile of 20 values

    def test_compute_ttft_metric_avg(self):
        strategy, _, _, _ = make_strategy()
        strategy._ttft_metric = "avg"
        result = strategy._compute_ttft_metric([1.0, 2.0, 3.0])
        assert result == pytest.approx(2.0)

    def test_compute_ttft_metric_max(self):
        strategy, _, _, _ = make_strategy()
        strategy._ttft_metric = "max"
        result = strategy._compute_ttft_metric([1.0, 2.0, 3.0])
        assert result == 3.0

    def test_compute_ttft_metric_empty(self):
        strategy, _, _, _ = make_strategy()
        assert strategy._compute_ttft_metric([]) == 0.0

    def test_assess_and_scale_adds_users_when_headroom(self):
        """When TTFT is well below threshold, users should be added."""
        strategy, scheduler, issuer, _ = make_strategy(
            start_users=2, max_users=20, max_ttft_sec=2.0
        )
        strategy._active_users = 2
        strategy._period_ttft_samples = [0.5, 0.6, 0.4, 0.5, 0.5]

        strategy._assess_and_scale()

        # Should have added some users
        assert strategy._active_users > 2
        assert scheduler.schedule_later.call_count > 0

    def test_assess_and_scale_stops_at_max_users(self):
        """Should not exceed max_users."""
        strategy, _, _, _ = make_strategy(max_users=5, max_ttft_sec=2.0)
        strategy._active_users = 5
        strategy._period_ttft_samples = [0.5, 0.6]

        strategy._assess_and_scale()

        assert strategy._active_users == 5

    def test_assess_and_scale_no_add_when_exceeded(self):
        """When TTFT exceeds threshold, no users should be added."""
        strategy, scheduler, _, _ = make_strategy(max_ttft_sec=1.0)
        strategy._active_users = 5
        strategy._period_ttft_samples = [1.5, 1.2, 1.8]

        strategy._assess_and_scale()

        assert strategy._active_users == 5
        assert scheduler.schedule_later.call_count == 0

    def test_assess_and_scale_no_samples_skips(self):
        strategy, scheduler, _, _ = make_strategy()
        strategy._period_ttft_samples = []
        strategy._assess_and_scale()
        assert scheduler.schedule_later.call_count == 0

    @pytest.mark.asyncio
    async def test_handle_credit_return_dispatches_next_turn(self):
        """Non-final credit should dispatch next turn with delay."""
        strategy, scheduler, issuer, _ = make_strategy()
        await strategy.setup_phase()

        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv_0",
            x_correlation_id="xcorr-1",
            turn_index=0,
            num_turns=5,
            issued_at_ns=0,
        )
        await strategy.handle_credit_return(credit)

        # Should schedule next turn with delay
        assert (
            scheduler.schedule_later.call_count == 1
            or scheduler.execute_async.call_count == 1
        )

    @pytest.mark.asyncio
    async def test_handle_credit_return_final_turn_decrements(self):
        """Final turn without recycle should decrement active users."""
        strategy, _, _, _ = make_strategy(recycle=False)
        strategy._active_users = 3

        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv_0",
            x_correlation_id="xcorr-1",
            turn_index=4,
            num_turns=5,
            issued_at_ns=0,
        )
        await strategy.handle_credit_return(credit)
        assert strategy._active_users == 2

    @pytest.mark.asyncio
    async def test_handle_credit_return_recycle(self):
        """Final turn with recycle should sample new conversation."""
        strategy, scheduler, issuer, _ = make_strategy(recycle=True)
        strategy._active_users = 3

        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv_0",
            x_correlation_id="xcorr-1",
            turn_index=4,
            num_turns=5,
            issued_at_ns=0,
        )
        await strategy.handle_credit_return(credit)

        # Should issue new credit for recycled session
        assert scheduler.execute_async.call_count == 1
        assert strategy._active_users == 3  # unchanged when recycling

    def test_stagger_from_config(self):
        """Stagger delay is read from config and converted to seconds."""
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.ADAPTIVE_SCALE,
            expected_duration_sec=120.0,
            stagger_ms=200.0,
        )
        strategy = AdaptiveScaleStrategy(
            config=cfg,
            conversation_source=MagicMock(),
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=MagicMock(),
            lifecycle=MagicMock(),
        )
        assert strategy._stagger_sec == pytest.approx(0.2)

    def test_stagger_default_50ms(self):
        """Default stagger is 50ms when not specified in config."""
        strategy, _, _, _ = make_strategy()
        # make_strategy doesn't pass stagger_ms, so config has None,
        # which falls back to 50.0ms -> 0.05s
        assert strategy._stagger_sec == pytest.approx(0.05)

    def test_scaling_formula_default_conservative(self):
        """Default scaling formula is conservative."""
        strategy, _, _, _ = make_strategy()
        assert strategy._scaling_formula == "conservative"

    def test_compute_users_to_add_conservative(self):
        """Conservative formula: max(1, active * headroom * 0.5)."""
        strategy, _, _, _ = make_strategy()
        strategy._active_users = 10
        # 50% headroom -> max(1, 10 * 0.5 * 0.5) = max(1, 2) = 2
        result = strategy._compute_users_to_add(0.5, 50.0)
        assert result == 2

    def test_compute_users_to_add_aggressive(self):
        """Aggressive formula: max(2, 2 + headroom_pct / 10)."""
        strategy, _, _, _ = make_strategy()
        strategy._scaling_formula = "aggressive"
        # 50% headroom -> max(2, 2 + 50/10) = max(2, 7) = 7
        result = strategy._compute_users_to_add(0.5, 50.0)
        assert result == 7

    def test_compute_users_to_add_linear(self):
        """Linear formula: max(1, headroom_pct / 5)."""
        strategy, _, _, _ = make_strategy()
        strategy._scaling_formula = "linear"
        # 50% headroom -> max(1, 50/5) = max(1, 10) = 10
        result = strategy._compute_users_to_add(0.5, 50.0)
        assert result == 10

    def test_compute_users_to_add_conservative_low_headroom(self):
        """Conservative with low headroom floors at 1."""
        strategy, _, _, _ = make_strategy()
        strategy._active_users = 2
        # 5% headroom -> max(1, 2 * 0.05 * 0.5) = max(1, 0) = 1
        result = strategy._compute_users_to_add(0.05, 5.0)
        assert result == 1

    def test_compute_users_to_add_aggressive_low_headroom(self):
        """Aggressive with low headroom floors at 2."""
        strategy, _, _, _ = make_strategy()
        strategy._scaling_formula = "aggressive"
        # 5% headroom -> max(2, 2 + 5/10) = max(2, 2) = 2
        result = strategy._compute_users_to_add(0.05, 5.0)
        assert result == 2

    @pytest.mark.asyncio
    async def test_execute_phase_uses_stagger_delay(self):
        """Initial users beyond the first are staggered by stagger_sec."""
        strategy, scheduler, issuer, stop_checker = make_strategy(start_users=3)
        strategy._stagger_sec = 0.1
        stop_checker.can_send_any_turn = MagicMock(
            side_effect=[True, True, True, False]
        )

        await strategy.setup_phase()
        await strategy.execute_phase()

        # First user awaited directly, remaining 2 scheduled with stagger
        assert scheduler.schedule_later.call_count == 2
        # Verify stagger delays: 0.1*1=0.1 and 0.1*2=0.2
        first_call_delay = scheduler.schedule_later.call_args_list[0][0][0]
        second_call_delay = scheduler.schedule_later.call_args_list[1][0][0]
        assert first_call_delay == pytest.approx(0.1)
        assert second_call_delay == pytest.approx(0.2)

    def test_ttft_rate_limiting_applies_backoff(self):
        """Backoff applied when TTFT exceeds threshold."""
        strategy, scheduler, issuer, _ = make_strategy(max_ttft_sec=2.0)
        strategy._enable_rate_limiting = True

        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv_0",
            x_correlation_id="xcorr-1",
            turn_index=0,
            num_turns=5,
            issued_at_ns=0,
        )
        # TTFT of 4s exceeds 2s threshold
        strategy.on_ttft_sample(4_000_000_000, credit=credit)

        assert "xcorr-1" in strategy._session_backoffs
        assert strategy._session_backoffs["xcorr-1"] > 0

    def test_ttft_rate_limiting_resets_on_good_ttft(self):
        """Backoff resets when TTFT drops below threshold."""
        strategy, _, _, _ = make_strategy(max_ttft_sec=2.0)
        strategy._enable_rate_limiting = True

        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv_0",
            x_correlation_id="xcorr-1",
            turn_index=0,
            num_turns=5,
            issued_at_ns=0,
        )
        # First sample: TTFT exceeds threshold
        strategy.on_ttft_sample(4_000_000_000, credit=credit)
        assert "xcorr-1" in strategy._session_backoffs

        # Second sample: TTFT drops below threshold
        strategy.on_ttft_sample(1_000_000_000, credit=credit)
        assert "xcorr-1" not in strategy._session_backoffs
        assert "xcorr-1" not in strategy._rate_limit_counts

    @pytest.mark.asyncio
    async def test_ttft_backoff_added_to_delay(self):
        """Session backoff is added to trace delay in handle_credit_return."""
        strategy, scheduler, issuer, _ = make_strategy()
        await strategy.setup_phase()

        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv_0",
            x_correlation_id="xcorr-1",
            turn_index=0,
            num_turns=5,
            issued_at_ns=0,
        )
        strategy._session_backoffs["xcorr-1"] = 5.0

        await strategy.handle_credit_return(credit)

        # Should schedule with delay including backoff
        assert scheduler.schedule_later.call_count == 1
        scheduled_delay = scheduler.schedule_later.call_args_list[0][0][0]
        # Delay should include the session backoff (at least 5.0)
        assert scheduled_delay >= 5.0

    @pytest.mark.asyncio
    async def test_session_cleanup_on_final_turn(self):
        """Rate limiting and working set state cleaned up on final turn."""
        strategy, _, _, _ = make_strategy(recycle=False)
        strategy._active_users = 3

        corr_id = "xcorr-cleanup"
        strategy._rate_limit_counts[corr_id] = 3
        strategy._session_backoffs[corr_id] = 2.5
        strategy._session_hash_ids[corr_id] = {1, 2, 3}
        strategy._active_hash_ids = {1, 2, 3, 4, 5}
        strategy._session_hash_ids["other"] = {4, 5}

        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv_0",
            x_correlation_id=corr_id,
            turn_index=4,
            num_turns=5,
            issued_at_ns=0,
        )
        await strategy.handle_credit_return(credit)

        assert corr_id not in strategy._rate_limit_counts
        assert corr_id not in strategy._session_backoffs
        assert corr_id not in strategy._session_hash_ids
        assert strategy._active_hash_ids == {4, 5}
        assert strategy._active_users == 2

    def test_working_set_budget_rejects_over_budget(self):
        """New session rejected when hash_ids would exceed working set budget."""
        strategy, _, _, _ = make_strategy()
        strategy._max_working_set_tokens = 640  # 10 blocks * 64 tokens/block
        strategy._block_size = 64
        strategy._active_hash_ids = set(range(8))  # 8 blocks = 512 tokens

        ds = make_conversations(20)
        # Give first turn hash_ids that would push over budget
        ds.conversations[0].turns[0].hash_ids = list(range(20, 24))  # 4 new blocks

        from aiperf.timing.conversation_source import SampledSession

        session = SampledSession(
            conversation_id="conv_0",
            metadata=ds.conversations[0],
            x_correlation_id="xcorr-budget",
        )

        # 8 existing + 4 new = 12 blocks * 64 = 768 > 640
        assert not strategy._check_token_budget(session)

    def test_working_set_budget_accepts_within_budget(self):
        """New session accepted when hash_ids fit within working set budget."""
        strategy, _, _, _ = make_strategy()
        strategy._max_working_set_tokens = 1280  # 20 blocks * 64 tokens/block
        strategy._block_size = 64
        strategy._active_hash_ids = set(range(8))  # 8 blocks = 512 tokens

        ds = make_conversations(20)
        ds.conversations[0].turns[0].hash_ids = list(range(20, 24))  # 4 new blocks

        from aiperf.timing.conversation_source import SampledSession

        session = SampledSession(
            conversation_id="conv_0",
            metadata=ds.conversations[0],
            x_correlation_id="xcorr-budget",
        )

        # 8 existing + 4 new = 12 blocks * 64 = 768 < 1280
        assert strategy._check_token_budget(session)
        assert strategy._active_hash_ids == set(range(8)) | set(range(20, 24))


NS_PER_MS = 1_000_000


class TestSLOBasedScaling:
    """Tests for SLO-based goodput scaling mode."""

    def test_no_slo_uses_ttft_headroom(self):
        """Without SLOs, strategy uses TTFT headroom mode."""
        strategy, scheduler, _, _ = make_strategy()
        assert strategy._slo_thresholds is None

        strategy._active_users = 2
        strategy._period_ttft_samples = [0.5, 0.6, 0.4]
        strategy._assess_and_scale()
        assert strategy._active_users > 2

    def test_slo_thresholds_normalized_to_ns(self):
        """SLO thresholds in ms are normalized to nanoseconds."""
        strategy, _, _, _ = make_strategy(
            adaptive_scale_slo={"time_to_first_token": 500, "request_latency": 2000},
        )
        assert strategy._slo_thresholds is not None
        assert strategy._slo_thresholds["time_to_first_token"] == 500 * NS_PER_MS
        assert strategy._slo_thresholds["request_latency"] == 2000 * NS_PER_MS

    def test_slo_ttft_threshold_pass(self):
        """Request with TTFT below SLO threshold passes."""
        strategy, _, _, _ = make_strategy(
            adaptive_scale_slo={"time_to_first_token": 500},
        )
        cr = _make_credit_return(
            ttft_ns=400 * NS_PER_MS, request_latency_ns=1000 * NS_PER_MS
        )
        assert strategy._evaluate_slos(cr) is True

    def test_slo_ttft_threshold_fail(self):
        """Request with TTFT above SLO threshold fails."""
        strategy, _, _, _ = make_strategy(
            adaptive_scale_slo={"time_to_first_token": 500},
        )
        cr = _make_credit_return(
            ttft_ns=600 * NS_PER_MS, request_latency_ns=1000 * NS_PER_MS
        )
        assert strategy._evaluate_slos(cr) is False

    def test_slo_request_latency_threshold_pass(self):
        """Request with latency below SLO threshold passes."""
        strategy, _, _, _ = make_strategy(
            adaptive_scale_slo={"request_latency": 2000},
        )
        cr = _make_credit_return(
            ttft_ns=400 * NS_PER_MS, request_latency_ns=1500 * NS_PER_MS
        )
        assert strategy._evaluate_slos(cr) is True

    def test_slo_request_latency_threshold_fail(self):
        """Request with latency above SLO threshold fails."""
        strategy, _, _, _ = make_strategy(
            adaptive_scale_slo={"request_latency": 2000},
        )
        cr = _make_credit_return(
            ttft_ns=400 * NS_PER_MS, request_latency_ns=2500 * NS_PER_MS
        )
        assert strategy._evaluate_slos(cr) is False

    def test_slo_combined_thresholds(self):
        """All SLOs must pass for a request to be 'good'."""
        strategy, _, _, _ = make_strategy(
            adaptive_scale_slo={"time_to_first_token": 500, "request_latency": 2000},
        )
        # Both pass
        cr = _make_credit_return(
            ttft_ns=400 * NS_PER_MS, request_latency_ns=1500 * NS_PER_MS
        )
        assert strategy._evaluate_slos(cr) is True

        # TTFT fails
        cr = _make_credit_return(
            ttft_ns=600 * NS_PER_MS, request_latency_ns=1500 * NS_PER_MS
        )
        assert strategy._evaluate_slos(cr) is False

        # Latency fails
        cr = _make_credit_return(
            ttft_ns=400 * NS_PER_MS, request_latency_ns=2500 * NS_PER_MS
        )
        assert strategy._evaluate_slos(cr) is False

    def test_slo_missing_data_fails(self):
        """Missing metric data causes SLO check to fail."""
        strategy, _, _, _ = make_strategy(
            adaptive_scale_slo={"time_to_first_token": 500, "request_latency": 2000},
        )
        # Missing ttft_ns
        cr = _make_credit_return(ttft_ns=None, request_latency_ns=1500 * NS_PER_MS)
        assert strategy._evaluate_slos(cr) is False

        # Missing request_latency_ns
        cr = _make_credit_return(ttft_ns=400 * NS_PER_MS, request_latency_ns=None)
        assert strategy._evaluate_slos(cr) is False

    def test_goodput_scaling_all_good_adds_users(self):
        """When all requests meet SLOs, goodput ratio is 1.0 and users are added."""
        strategy, scheduler, _, _ = make_strategy(
            adaptive_scale_slo={"time_to_first_token": 500},
            min_goodput_ratio=0.9,
        )
        strategy._active_users = 2

        for i in range(10):
            cr = _make_credit_return(credit_id=i, ttft_ns=300 * NS_PER_MS)
            strategy.on_request_complete(cr)

        strategy._assess_and_scale()
        assert strategy._active_users > 2

    def test_goodput_scaling_below_ratio_holds(self):
        """When goodput ratio is below threshold, no users are added."""
        strategy, scheduler, _, _ = make_strategy(
            adaptive_scale_slo={"time_to_first_token": 500},
            min_goodput_ratio=0.9,
        )
        strategy._active_users = 5

        # 5 good, 5 bad -> ratio 0.5 < 0.9
        for i in range(5):
            cr = _make_credit_return(credit_id=i, ttft_ns=300 * NS_PER_MS)
            strategy.on_request_complete(cr)
        for i in range(5, 10):
            cr = _make_credit_return(credit_id=i, ttft_ns=600 * NS_PER_MS)
            strategy.on_request_complete(cr)

        strategy._assess_and_scale()
        assert strategy._active_users == 5
        assert scheduler.schedule_later.call_count == 0

    def test_goodput_scaling_mixed_results(self):
        """With ratio just above threshold, users are added."""
        strategy, scheduler, _, _ = make_strategy(
            adaptive_scale_slo={"time_to_first_token": 500},
            min_goodput_ratio=0.8,
        )
        strategy._active_users = 4

        # 9 good, 1 bad -> ratio 0.9 >= 0.8
        for i in range(9):
            cr = _make_credit_return(credit_id=i, ttft_ns=300 * NS_PER_MS)
            strategy.on_request_complete(cr)
        cr = _make_credit_return(credit_id=9, ttft_ns=600 * NS_PER_MS)
        strategy.on_request_complete(cr)

        strategy._assess_and_scale()
        assert strategy._active_users > 4

    def test_goodput_period_counters_reset(self):
        """Period counters reset after assessment."""
        strategy, _, _, _ = make_strategy(
            adaptive_scale_slo={"time_to_first_token": 500},
        )
        strategy._active_users = 2

        for i in range(5):
            cr = _make_credit_return(credit_id=i, ttft_ns=300 * NS_PER_MS)
            strategy.on_request_complete(cr)

        assert strategy._period_total_count == 5
        assert strategy._period_good_count == 5

        strategy._assess_and_scale()

        assert strategy._period_total_count == 0
        assert strategy._period_good_count == 0

    def test_on_request_complete_noop_without_slo(self):
        """on_request_complete is a no-op when no SLOs configured."""
        strategy, _, _, _ = make_strategy()
        cr = _make_credit_return(ttft_ns=300 * NS_PER_MS)
        strategy.on_request_complete(cr)
        assert strategy._period_total_count == 0

    def test_slo_rate_limiting_on_failure(self):
        """Failed SLO applies per-session backoff when rate limiting enabled."""
        strategy, _, _, _ = make_strategy(
            adaptive_scale_slo={"time_to_first_token": 500},
        )
        strategy._enable_rate_limiting = True

        cr = _make_credit_return(ttft_ns=600 * NS_PER_MS, corr_id="xcorr-rl")
        strategy.on_request_complete(cr)

        assert "xcorr-rl" in strategy._session_backoffs
        assert strategy._session_backoffs["xcorr-rl"] > 0
