# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for engine telemetry collection infrastructure.

Covers:
- EngineIterationStats model serialization, defaults, and all-fields populated
- Telemetry loop with mock engine returning canned stats
- Telemetry disabled by default (no overhead)
- Telemetry log accumulates entries correctly
- _stop_telemetry_loop cancels cleanly and double-stop is idempotent
- get_telemetry_log returns an independent copy
- Telemetry loop timing (interval_ms respected)
- Telemetry loop concurrent with request processing (no interference)
- configure() starts telemetry when enabled
- _on_stop_engine() stops telemetry cleanly
- TRT-LLM _get_engine_stats with mock engine (get_stats_async present/absent)
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import EngineIterationStats, RequestInfo
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.plugin.enums import EndpointType
from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import FirstTokenCallback
from aiperf.transports.in_engine.base_in_engine_transport import BaseInEngineTransport

# ============================================================
# Concrete subclass with configurable _get_engine_stats
# ============================================================


class TelemetryTestTransport(BaseInEngineTransport):
    """Concrete subclass for testing telemetry infrastructure."""

    def __init__(
        self,
        *,
        stats_results: list[dict[str, Any] | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._stats_results = stats_results or []
        self._stats_call_count = 0

    @classmethod
    def metadata(cls) -> TransportMetadata:
        return TransportMetadata(
            transport_type="test-telem", url_schemes=["test-telem"]
        )

    async def _init_engine(self) -> None:
        pass

    async def _start_engine(self) -> None:
        pass

    async def _stop_engine(self) -> None:
        pass

    async def _warmup_single(
        self, prompt: str, max_tokens: int, *, streaming: bool
    ) -> None:
        pass

    async def _generate(
        self,
        *,
        messages: list[dict[str, Any]],
        sampling_params: Any,
        request_id: str,
        first_token_callback: FirstTokenCallback | None = None,
        input_ids: list[int] | None = None,
    ) -> tuple[str, int, int, str]:
        return ("text", 5, 3, "stop")

    async def _get_engine_stats(self) -> dict[str, Any] | None:
        if self._stats_call_count < len(self._stats_results):
            result = self._stats_results[self._stats_call_count]
            self._stats_call_count += 1
            return result
        return None


async def _await_telemetry_entries(
    transport: TelemetryTestTransport, min_entries: int, max_yields: int = 200
) -> None:
    """Yield control to the event loop until the telemetry log has enough entries.

    The unit-test autouse ``no_sleep`` fixture makes ``asyncio.sleep`` instant,
    so a single ``await asyncio.sleep(X)`` only yields once. This helper yields
    multiple times so the background telemetry task gets enough scheduling to
    collect the expected entries.
    """
    for _ in range(max_yields):
        if len(transport._telemetry_log) >= min_entries:
            return
        await asyncio.sleep(0)


# ============================================================
# Fixtures
# ============================================================


def _make_model_endpoint(
    engine_params: list[tuple[str, str]] | None = None,
) -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test/model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_urls=["test-telem://test/model"],
            engine_params=engine_params or [],
        ),
    )


def _make_request_info(model_endpoint: ModelEndpointInfo) -> RequestInfo:
    """Create a RequestInfo with sensible defaults for telemetry tests."""
    return RequestInfo(
        model_endpoint=model_endpoint,
        turns=[],
        endpoint_headers={},
        endpoint_params={},
        turn_index=0,
        credit_num=1,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="req-telem-001",
        x_correlation_id="corr-telem-001",
        conversation_id="conv-telem-001",
    )


@pytest.fixture
def model_endpoint() -> ModelEndpointInfo:
    return _make_model_endpoint()


@pytest.fixture
def telemetry_endpoint() -> ModelEndpointInfo:
    return _make_model_endpoint(
        engine_params=[("telemetry", "true"), ("telemetry_interval_ms", "50")]
    )


def _make_telemetry_transport(
    endpoint: ModelEndpointInfo,
    stats_results: list[dict[str, Any] | None] | None = None,
) -> TelemetryTestTransport:
    """Create a TelemetryTestTransport with telemetry config parsed from engine_params."""
    transport = TelemetryTestTransport(
        model_endpoint=endpoint,
        stats_results=stats_results,
    )
    params = transport._get_raw_engine_params()
    transport._pop_warmup_config(params)
    return transport


# ============================================================
# EngineIterationStats model tests
# ============================================================


class TestEngineIterationStats:
    """Verify EngineIterationStats serialization and defaults."""

    def test_required_timestamp_only(self) -> None:
        stats = EngineIterationStats(timestamp_ns=1_000_000_000)
        assert stats.timestamp_ns == 1_000_000_000
        assert stats.batch_size is None
        assert stats.num_tokens is None
        assert stats.queue_depth is None
        assert stats.raw == {}

    def test_all_fields_populated(self) -> None:
        raw = {"custom_key": "value", "batch_size": 4}
        stats = EngineIterationStats(
            timestamp_ns=2_000_000_000,
            batch_size=4,
            num_tokens=128,
            queue_depth=10,
            raw=raw,
        )
        assert stats.batch_size == 4
        assert stats.num_tokens == 128
        assert stats.queue_depth == 10
        assert stats.raw == raw

    def test_all_fields_populated_with_engine_specific_raw(self) -> None:
        """Raw dict can contain engine-specific keys alongside promoted fields."""
        raw = {
            "batch_size": 16,
            "num_tokens": 512,
            "queue_depth": 3,
            "gpu_utilization": 0.95,
            "kv_cache_usage": 0.72,
            "inflight_requests": 8,
        }
        stats = EngineIterationStats(
            timestamp_ns=9_999_999_999,
            batch_size=16,
            num_tokens=512,
            queue_depth=3,
            raw=raw,
        )
        assert stats.timestamp_ns == 9_999_999_999
        assert stats.batch_size == 16
        assert stats.num_tokens == 512
        assert stats.queue_depth == 3
        assert stats.raw["gpu_utilization"] == 0.95
        assert stats.raw["kv_cache_usage"] == 0.72
        assert stats.raw["inflight_requests"] == 8

    def test_serialization_roundtrip(self) -> None:
        stats = EngineIterationStats(
            timestamp_ns=123, batch_size=2, num_tokens=64, queue_depth=5, raw={"k": 1}
        )
        data = stats.model_dump()
        restored = EngineIterationStats(**data)
        assert restored == stats

    @pytest.mark.parametrize(
        "field,value",
        [
            ("batch_size", 8),
            ("num_tokens", 256),
            ("queue_depth", 0),
        ],
    )  # fmt: skip
    def test_individual_optional_fields(self, field: str, value: int) -> None:
        stats = EngineIterationStats(timestamp_ns=1, **{field: value})
        assert getattr(stats, field) == value


# ============================================================
# Telemetry disabled by default
# ============================================================


class TestTelemetryDisabledByDefault:
    """Verify telemetry adds zero overhead when not enabled."""

    def test_telemetry_disabled_by_default(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        transport = TelemetryTestTransport(model_endpoint=model_endpoint)
        assert transport._telemetry_enabled is False
        assert transport._telemetry_task is None

    def test_start_telemetry_loop_noop_when_disabled(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        transport = TelemetryTestTransport(model_endpoint=model_endpoint)
        transport._start_telemetry_loop()
        assert transport._telemetry_task is None

    @pytest.mark.asyncio
    async def test_stop_telemetry_loop_noop_when_no_task(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        transport = TelemetryTestTransport(model_endpoint=model_endpoint)
        await transport._stop_telemetry_loop()
        assert transport._telemetry_task is None


# ============================================================
# Telemetry config parsing
# ============================================================


class TestTelemetryConfigParsing:
    """Verify telemetry params are extracted from engine_params."""

    @pytest.mark.parametrize(
        "telemetry_val,expected",
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
            (True, True),
            (False, False),
        ],
    )  # fmt: skip
    def test_telemetry_flag_parsing(self, telemetry_val: Any, expected: bool) -> None:
        endpoint = _make_model_endpoint(engine_params=[("telemetry", telemetry_val)])
        transport = TelemetryTestTransport(model_endpoint=endpoint)
        params = transport._get_raw_engine_params()
        transport._pop_warmup_config(params)
        assert transport._telemetry_enabled is expected
        assert "telemetry" not in params

    def test_telemetry_interval_ms_parsed(self) -> None:
        endpoint = _make_model_endpoint(
            engine_params=[("telemetry", "true"), ("telemetry_interval_ms", "200")]
        )
        transport = TelemetryTestTransport(model_endpoint=endpoint)
        params = transport._get_raw_engine_params()
        transport._pop_warmup_config(params)
        assert transport._telemetry_interval_ms == 200
        assert "telemetry_interval_ms" not in params

    def test_default_interval_is_500ms(self, model_endpoint: ModelEndpointInfo) -> None:
        transport = TelemetryTestTransport(model_endpoint=model_endpoint)
        assert transport._telemetry_interval_ms == 500


# ============================================================
# Telemetry loop accumulates entries
# ============================================================


class TestTelemetryLoop:
    """Verify the telemetry loop polls stats and appends to the log."""

    @pytest.mark.asyncio
    async def test_telemetry_loop_collects_stats(
        self, telemetry_endpoint: ModelEndpointInfo
    ) -> None:
        canned_stats = [
            {"batch_size": 4, "num_tokens": 128, "queue_depth": 2},
            {"batch_size": 8, "num_tokens": 256, "queue_depth": 0},
        ]
        transport = _make_telemetry_transport(telemetry_endpoint, canned_stats)

        transport._start_telemetry_loop()
        assert transport._telemetry_task is not None

        # Yield enough times for the background task to process all entries
        await _await_telemetry_entries(transport, min_entries=2)
        await transport._stop_telemetry_loop()

        log = transport.get_telemetry_log()
        assert len(log) >= 2
        assert log[0].batch_size == 4
        assert log[0].num_tokens == 128
        assert log[0].queue_depth == 2
        assert log[1].batch_size == 8
        assert log[1].num_tokens == 256

    @pytest.mark.asyncio
    async def test_telemetry_loop_skips_none_stats(
        self, telemetry_endpoint: ModelEndpointInfo
    ) -> None:
        transport = _make_telemetry_transport(
            telemetry_endpoint, [None, {"batch_size": 1}, None]
        )

        transport._start_telemetry_loop()
        # Wait for the single real entry to be collected
        await _await_telemetry_entries(transport, min_entries=1)
        # Yield a few more times so the loop processes past the trailing None
        for _ in range(20):
            await asyncio.sleep(0)
        await transport._stop_telemetry_loop()

        log = transport.get_telemetry_log()
        # Only the non-None entry should be logged
        assert len(log) == 1
        assert log[0].batch_size == 1

    @pytest.mark.asyncio
    async def test_telemetry_loop_handles_stats_error_gracefully(
        self, telemetry_endpoint: ModelEndpointInfo
    ) -> None:
        transport = _make_telemetry_transport(telemetry_endpoint, [{"batch_size": 1}])

        # Make _get_engine_stats raise after the first call
        original = transport._get_engine_stats
        call_count = 0

        async def flaky_stats() -> dict[str, Any] | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return await original()
            raise RuntimeError("stats API unavailable")

        transport._get_engine_stats = flaky_stats  # type: ignore[method-assign]

        transport._start_telemetry_loop()
        await _await_telemetry_entries(transport, min_entries=1)
        await transport._stop_telemetry_loop()

        # First entry collected, errors swallowed
        log = transport.get_telemetry_log()
        assert len(log) >= 1
        assert log[0].batch_size == 1

    @pytest.mark.asyncio
    async def test_telemetry_loop_entries_have_timestamp(
        self, telemetry_endpoint: ModelEndpointInfo
    ) -> None:
        """Each telemetry entry has a wall-clock timestamp_ns set by the loop."""
        before_ns = time.time_ns()
        transport = _make_telemetry_transport(telemetry_endpoint, [{"batch_size": 1}])

        transport._start_telemetry_loop()
        await _await_telemetry_entries(transport, min_entries=1)
        await transport._stop_telemetry_loop()
        after_ns = time.time_ns()

        log = transport.get_telemetry_log()
        assert len(log) >= 1
        assert before_ns <= log[0].timestamp_ns <= after_ns

    @pytest.mark.asyncio
    async def test_telemetry_loop_raw_dict_stored(
        self, telemetry_endpoint: ModelEndpointInfo
    ) -> None:
        """The raw dict from _get_engine_stats is stored in the entry."""
        raw = {"batch_size": 2, "custom_metric": 42.5}
        transport = _make_telemetry_transport(telemetry_endpoint, [raw])

        transport._start_telemetry_loop()
        await _await_telemetry_entries(transport, min_entries=1)
        await transport._stop_telemetry_loop()

        log = transport.get_telemetry_log()
        assert len(log) >= 1
        assert log[0].raw == raw
        assert log[0].raw["custom_metric"] == 42.5


# ============================================================
# Telemetry loop timing (interval_ms respected)
# ============================================================


class TestTelemetryLoopTiming:
    """Verify the telemetry loop calls _get_engine_stats on each iteration."""

    @pytest.mark.asyncio
    async def test_loop_consumes_all_available_stats(self) -> None:
        """The loop polls _get_engine_stats on every iteration until stats are exhausted."""
        endpoint = _make_model_endpoint(
            engine_params=[("telemetry", "true"), ("telemetry_interval_ms", "100")]
        )
        stats = [{"batch_size": i} for i in range(5)]
        transport = _make_telemetry_transport(endpoint, stats)

        transport._start_telemetry_loop()
        await _await_telemetry_entries(transport, min_entries=5)
        await transport._stop_telemetry_loop()

        log = transport.get_telemetry_log()
        assert len(log) == 5
        assert [e.batch_size for e in log] == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_interval_ms_is_stored_correctly(self) -> None:
        """Verify the interval_ms from engine_params is correctly stored on the transport."""
        endpoint_100 = _make_model_endpoint(
            engine_params=[("telemetry", "true"), ("telemetry_interval_ms", "100")]
        )
        endpoint_25 = _make_model_endpoint(
            engine_params=[("telemetry", "true"), ("telemetry_interval_ms", "25")]
        )

        transport_100 = _make_telemetry_transport(endpoint_100, [])
        transport_25 = _make_telemetry_transport(endpoint_25, [])

        assert transport_100._telemetry_interval_ms == 100
        assert transport_25._telemetry_interval_ms == 25


# ============================================================
# Telemetry concurrent with request processing
# ============================================================


class TestTelemetryConcurrentWithRequests:
    """Verify telemetry loop does not interfere with request processing."""

    @pytest.mark.asyncio
    async def test_telemetry_loop_does_not_block_send_request(
        self, telemetry_endpoint: ModelEndpointInfo
    ) -> None:
        """send_request completes normally while the telemetry loop is running."""
        transport = _make_telemetry_transport(
            telemetry_endpoint,
            [{"batch_size": i} for i in range(100)],
        )

        transport._start_telemetry_loop()

        # Send a request while telemetry is running
        request_info = _make_request_info(telemetry_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "sampling_params": {},
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        assert record.error is None
        assert len(record.responses) == 1
        assert record.responses[0].text == "text"

        await transport._stop_telemetry_loop()

    @pytest.mark.asyncio
    async def test_multiple_requests_with_telemetry_running(
        self, telemetry_endpoint: ModelEndpointInfo
    ) -> None:
        """Multiple concurrent requests succeed while telemetry polls."""
        transport = _make_telemetry_transport(
            telemetry_endpoint,
            [{"batch_size": i} for i in range(100)],
        )

        transport._start_telemetry_loop()

        request_info = _make_request_info(telemetry_endpoint)
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "sampling_params": {},
        }

        # Fire off multiple requests concurrently
        results = await asyncio.gather(
            transport.send_request(request_info, payload),
            transport.send_request(request_info, payload),
            transport.send_request(request_info, payload),
        )

        for record in results:
            assert record.status == 200
            assert record.error is None

        await transport._stop_telemetry_loop()


# ============================================================
# Stop telemetry loop cancels cleanly
# ============================================================


class TestStopTelemetryLoop:
    """Verify _stop_telemetry_loop cancels without errors."""

    @pytest.mark.asyncio
    async def test_stop_cancels_running_task(
        self, telemetry_endpoint: ModelEndpointInfo
    ) -> None:
        transport = _make_telemetry_transport(
            telemetry_endpoint, [{"batch_size": 1}] * 100
        )

        transport._start_telemetry_loop()
        task = transport._telemetry_task
        assert task is not None
        assert not task.done()

        await transport._stop_telemetry_loop()
        assert transport._telemetry_task is None
        assert task.done()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, telemetry_endpoint: ModelEndpointInfo) -> None:
        transport = _make_telemetry_transport(telemetry_endpoint, [])

        transport._start_telemetry_loop()
        await transport._stop_telemetry_loop()
        # Second stop is a no-op
        await transport._stop_telemetry_loop()
        assert transport._telemetry_task is None

    @pytest.mark.asyncio
    async def test_double_stop_without_start_is_idempotent(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        """Calling _stop_telemetry_loop twice without ever starting is safe."""
        transport = TelemetryTestTransport(model_endpoint=model_endpoint)
        await transport._stop_telemetry_loop()
        await transport._stop_telemetry_loop()
        assert transport._telemetry_task is None


# ============================================================
# get_telemetry_log returns a copy
# ============================================================


class TestGetTelemetryLog:
    """Verify get_telemetry_log returns an independent copy."""

    def test_returns_copy_not_reference(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        transport = TelemetryTestTransport(model_endpoint=model_endpoint)
        transport._telemetry_log.append(
            EngineIterationStats(timestamp_ns=1, batch_size=2, raw={})
        )

        log = transport.get_telemetry_log()
        assert len(log) == 1
        log.clear()
        assert len(transport._telemetry_log) == 1

    def test_empty_log_when_no_telemetry(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        transport = TelemetryTestTransport(model_endpoint=model_endpoint)
        assert transport.get_telemetry_log() == []

    def test_modifying_returned_list_does_not_affect_internal(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        """Appending to the returned list does not change internal state."""
        transport = TelemetryTestTransport(model_endpoint=model_endpoint)
        transport._telemetry_log.append(EngineIterationStats(timestamp_ns=100, raw={}))

        log = transport.get_telemetry_log()
        log.append(EngineIterationStats(timestamp_ns=200, raw={}))
        log.append(EngineIterationStats(timestamp_ns=300, raw={}))

        # Internal log still has only 1 entry
        assert len(transport._telemetry_log) == 1
        assert len(transport.get_telemetry_log()) == 1


# ============================================================
# Base class _get_engine_stats default returns None
# ============================================================


class TestBaseGetEngineStats:
    """Verify the default _get_engine_stats returns None."""

    @pytest.mark.asyncio
    async def test_default_returns_none(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        # Use BaseInEngineTransport's default via super
        transport = TelemetryTestTransport(model_endpoint=model_endpoint)
        result = await BaseInEngineTransport._get_engine_stats(transport)
        assert result is None


# ============================================================
# configure() starts telemetry when enabled
# ============================================================


class TestConfigureStartsTelemetry:
    """Verify configure() starts the telemetry loop when telemetry is enabled."""

    @pytest.mark.asyncio
    async def test_configure_starts_telemetry_when_enabled(
        self, telemetry_endpoint: ModelEndpointInfo
    ) -> None:
        """After configure(), the telemetry task is running."""
        transport = _make_telemetry_transport(
            telemetry_endpoint, [{"batch_size": 1}] * 100
        )

        await transport.configure()

        assert transport._telemetry_task is not None
        assert not transport._telemetry_task.done()

        # Cleanup
        await transport._stop_telemetry_loop()

    @pytest.mark.asyncio
    async def test_configure_does_not_start_telemetry_when_disabled(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        """When telemetry is disabled, configure() does not create a task."""
        transport = TelemetryTestTransport(model_endpoint=model_endpoint)

        await transport.configure()

        assert transport._telemetry_task is None


# ============================================================
# _on_stop_engine() stops telemetry cleanly
# ============================================================


class TestOnStopEngineStopsTelemetry:
    """Verify _on_stop_engine() stops the telemetry loop and cleans up."""

    @pytest.mark.asyncio
    async def test_on_stop_engine_stops_telemetry_task(
        self, telemetry_endpoint: ModelEndpointInfo
    ) -> None:
        """_on_stop_engine cancels the telemetry task and sets engine to None."""
        transport = _make_telemetry_transport(
            telemetry_endpoint, [{"batch_size": 1}] * 100
        )

        # Simulate engine running with telemetry
        transport._engine = MagicMock()
        transport._start_telemetry_loop()
        assert transport._telemetry_task is not None

        await transport._on_stop_engine()

        assert transport._telemetry_task is None
        assert transport._engine is None

    @pytest.mark.asyncio
    async def test_on_stop_engine_safe_without_telemetry(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        """_on_stop_engine works when telemetry was never started."""
        transport = TelemetryTestTransport(model_endpoint=model_endpoint)
        transport._engine = MagicMock()

        await transport._on_stop_engine()

        assert transport._telemetry_task is None
        assert transport._engine is None

    @pytest.mark.asyncio
    async def test_on_stop_engine_preserves_telemetry_log(
        self, telemetry_endpoint: ModelEndpointInfo
    ) -> None:
        """Stopping the engine does not clear already-collected telemetry entries."""
        transport = _make_telemetry_transport(telemetry_endpoint, [{"batch_size": 5}])
        transport._engine = MagicMock()

        transport._start_telemetry_loop()
        await _await_telemetry_entries(transport, min_entries=1)
        await transport._on_stop_engine()

        log = transport.get_telemetry_log()
        assert len(log) >= 1
        assert log[0].batch_size == 5


# ============================================================
# TRT-LLM _get_engine_stats with mock engine
# ============================================================


class TestTRTLLMGetEngineStats:
    """Verify TRTLLMTransport._get_engine_stats with mock engines."""

    @pytest.fixture
    def mock_trtllm_module(self) -> types.ModuleType:
        """Build a mock tensorrt_llm module with needed classes."""
        mock_trtllm = types.ModuleType("tensorrt_llm")
        mock_trtllm.SamplingParams = MagicMock  # type: ignore[attr-defined]
        mock_trtllm.LLM = MagicMock  # type: ignore[attr-defined]

        mock_llmapi = types.ModuleType("tensorrt_llm.llmapi")
        mock_llmapi.CapacitySchedulerPolicy = MagicMock()  # type: ignore[attr-defined]
        mock_llmapi.SchedulerConfig = MagicMock()  # type: ignore[attr-defined]
        mock_llmapi.KvCacheConfig = MagicMock()  # type: ignore[attr-defined]
        mock_llmapi.ExtendedRuntimePerfKnobConfig = MagicMock()  # type: ignore[attr-defined]
        mock_trtllm.llmapi = mock_llmapi  # type: ignore[attr-defined]
        return mock_trtllm

    def _make_trtllm_endpoint(
        self, engine_params: list[tuple[str, Any]] | None = None
    ) -> ModelEndpointInfo:
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="meta-llama/Llama-3.1-8B")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_urls=["trtllm://meta-llama/Llama-3.1-8B"],
                engine_params=engine_params or [],
            ),
        )

    @pytest.mark.asyncio
    async def test_get_engine_stats_with_get_stats_async(
        self, mock_trtllm_module: types.ModuleType
    ) -> None:
        """When engine has get_stats_async, returns the first yielded stats dict."""
        mock_modules = {
            "tensorrt_llm": mock_trtllm_module,
            "tensorrt_llm.llmapi": mock_trtllm_module.llmapi,  # type: ignore[attr-defined]
        }
        with patch.dict(sys.modules, mock_modules):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = self._make_trtllm_endpoint()
            transport = TRTLLMTransport(model_endpoint=endpoint)

            # Build a mock engine with get_stats_async that yields a dict
            expected_stats = {
                "batch_size": 8,
                "num_tokens": 256,
                "queue_depth": 4,
                "gpu_util": 0.9,
            }

            async def mock_get_stats_async(timeout: int = 1) -> Any:
                yield expected_stats

            mock_engine = MagicMock()
            mock_engine.get_stats_async = mock_get_stats_async
            transport._engine = mock_engine

            result = await transport._get_engine_stats()

            assert result == expected_stats

    @pytest.mark.asyncio
    async def test_get_engine_stats_without_get_stats_async(
        self, mock_trtllm_module: types.ModuleType
    ) -> None:
        """When engine lacks get_stats_async, returns None."""
        mock_modules = {
            "tensorrt_llm": mock_trtllm_module,
            "tensorrt_llm.llmapi": mock_trtllm_module.llmapi,  # type: ignore[attr-defined]
        }
        with patch.dict(sys.modules, mock_modules):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = self._make_trtllm_endpoint()
            transport = TRTLLMTransport(model_endpoint=endpoint)

            # Engine without get_stats_async
            mock_engine = MagicMock(spec=["generate_async", "shutdown"])
            transport._engine = mock_engine

            result = await transport._get_engine_stats()

            assert result is None

    @pytest.mark.asyncio
    async def test_get_engine_stats_when_engine_is_none(
        self, mock_trtllm_module: types.ModuleType
    ) -> None:
        """When engine is None (not yet started), returns None."""
        mock_modules = {
            "tensorrt_llm": mock_trtllm_module,
            "tensorrt_llm.llmapi": mock_trtllm_module.llmapi,  # type: ignore[attr-defined]
        }
        with patch.dict(sys.modules, mock_modules):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = self._make_trtllm_endpoint()
            transport = TRTLLMTransport(model_endpoint=endpoint)
            transport._engine = None

            result = await transport._get_engine_stats()

            assert result is None

    @pytest.mark.asyncio
    async def test_get_engine_stats_empty_generator(
        self, mock_trtllm_module: types.ModuleType
    ) -> None:
        """When get_stats_async yields nothing, returns None."""
        mock_modules = {
            "tensorrt_llm": mock_trtllm_module,
            "tensorrt_llm.llmapi": mock_trtllm_module.llmapi,  # type: ignore[attr-defined]
        }
        with patch.dict(sys.modules, mock_modules):
            from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

            endpoint = self._make_trtllm_endpoint()
            transport = TRTLLMTransport(model_endpoint=endpoint)

            async def empty_stats_async(timeout: int = 1) -> Any:
                if False:
                    yield  # pragma: no cover - makes this an async generator

            mock_engine = MagicMock()
            mock_engine.get_stats_async = empty_stats_async
            transport._engine = mock_engine

            result = await transport._get_engine_stats()

            assert result is None
