# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cache lifetime TTL eviction in AdaptiveScaleStrategy."""

import time
from unittest.mock import MagicMock

from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.adaptive_scale import AdaptiveScaleStrategy


def _make_strategy(
    cache_ttl_sec: float = 3600.0,
    subagent_cache_ttl_sec: float = 300.0,
    max_working_set_tokens: int | None = 1_000_000,
) -> AdaptiveScaleStrategy:
    """Build a strategy with minimal mocks for TTL testing."""
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.ADAPTIVE_SCALE,
        start_users=1,
        max_working_set_tokens=max_working_set_tokens,
        cache_ttl_sec=cache_ttl_sec,
        subagent_cache_ttl_sec=subagent_cache_ttl_sec,
    )
    strategy = AdaptiveScaleStrategy(
        config=config,
        conversation_source=MagicMock(),
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=MagicMock(),
        lifecycle=MagicMock(),
    )
    return strategy


class TestTTLEviction:
    """Tests for TTL-based working set eviction."""

    def test_expired_session_evicted(self):
        """Sessions past their TTL are removed from the working set."""
        strategy = _make_strategy(cache_ttl_sec=0.001)

        # Add a session to working set
        session_ids = {100, 101, 102}
        strategy._session_hash_ids["sess1"] = session_ids
        strategy._active_hash_ids = set(session_ids)
        strategy._session_last_active_ns["sess1"] = (
            time.perf_counter_ns() - 10_000_000_000
        )  # 10s ago

        strategy._evict_expired_sessions()

        assert "sess1" not in strategy._session_hash_ids
        assert "sess1" not in strategy._session_last_active_ns
        assert len(strategy._active_hash_ids) == 0

    def test_active_session_preserved(self):
        """Sessions within their TTL are not evicted."""
        strategy = _make_strategy(cache_ttl_sec=3600.0)

        session_ids = {200, 201, 202}
        strategy._session_hash_ids["sess1"] = session_ids
        strategy._active_hash_ids = set(session_ids)
        strategy._session_last_active_ns["sess1"] = time.perf_counter_ns()

        strategy._evict_expired_sessions()

        assert "sess1" in strategy._session_hash_ids
        assert len(strategy._active_hash_ids) == 3

    def test_subagent_shorter_ttl(self):
        """Subagent sessions use shorter TTL."""
        strategy = _make_strategy(
            cache_ttl_sec=3600.0,
            subagent_cache_ttl_sec=0.001,
        )

        # Main session (recent) and subagent (old)
        now = time.perf_counter_ns()
        strategy._session_hash_ids["main"] = {1, 2, 3}
        strategy._session_last_active_ns["main"] = now
        strategy._session_depth["main"] = 0

        strategy._session_hash_ids["child"] = {4, 5, 6}
        strategy._session_last_active_ns["child"] = now - 10_000_000_000  # 10s ago
        strategy._session_depth["child"] = 1

        strategy._active_hash_ids = {1, 2, 3, 4, 5, 6}

        strategy._evict_expired_sessions()

        assert "main" in strategy._session_hash_ids
        assert "child" not in strategy._session_hash_ids
        assert strategy._active_hash_ids == {1, 2, 3}

    def test_no_sessions_no_error(self):
        """Eviction with empty state doesn't error."""
        strategy = _make_strategy()
        strategy._evict_expired_sessions()

    def test_eviction_rebuilds_active_ids(self):
        """After eviction, active_hash_ids is rebuilt from remaining sessions."""
        strategy = _make_strategy(cache_ttl_sec=0.001)

        now = time.perf_counter_ns()
        old = now - 10_000_000_000

        strategy._session_hash_ids["expired"] = {1, 2}
        strategy._session_last_active_ns["expired"] = old

        strategy._session_hash_ids["active"] = {2, 3}
        strategy._session_last_active_ns["active"] = now

        strategy._active_hash_ids = {1, 2, 3}

        strategy._evict_expired_sessions()

        assert "expired" not in strategy._session_hash_ids
        assert "active" in strategy._session_hash_ids
        assert strategy._active_hash_ids == {2, 3}
