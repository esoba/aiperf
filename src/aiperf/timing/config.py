# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import ConfigDict, Field

from aiperf.common.config import InputDefaults, UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.plugin.enums import (
    ArrivalPattern,
    TimingMode,
    URLSelectionStrategy,
)
from aiperf.timing.request_cancellation import RequestCancellationConfig


class TimingConfig(AIPerfBaseModel):
    """Configuration for TimingManager and timing strategies.

    Controls timing mode (REQUEST_RATE, FIXED_SCHEDULE, or USER_CENTRIC_RATE),
    rate/concurrency settings, warmup/profiling phase stop conditions, and
    request cancellation behavior.
    """

    model_config = ConfigDict(frozen=True)

    phase_configs: list[CreditPhaseConfig] = Field(
        ...,
        description="List of phase configs to execute in order. These specify the exact behavior of each phase.",
    )
    request_cancellation: RequestCancellationConfig = Field(
        default_factory=RequestCancellationConfig,
        description="Configuration for request cancellation policy.",
    )
    urls: list[str] = Field(
        default_factory=list,
        description="List of endpoint URLs for load balancing. If multiple URLs provided, "
        "requests are distributed according to url_selection_strategy.",
    )
    url_selection_strategy: URLSelectionStrategy = Field(
        default=URLSelectionStrategy.ROUND_ROBIN,
        description="Strategy for selecting URLs when multiple URLs are provided.",
    )

    @classmethod
    def from_user_config(cls, user_config: UserConfig) -> TimingConfig:
        """Build ordered list of phase configs based on user config: [warmup?, profiling].

        Warmup (if enabled) executes first to prepare system,
        then profiling for actual measurement.
        """
        loadgen = user_config.loadgen
        configs: list[CreditPhaseConfig] = []

        warmup = _build_warmup_config(user_config)
        if warmup:
            configs.append(warmup)

        configs.append(_build_profiling_config(user_config))

        return cls(
            phase_configs=configs,
            request_cancellation=RequestCancellationConfig(
                rate=loadgen.request_cancellation_rate,
                delay=loadgen.request_cancellation_delay,
            ),
            urls=user_config.endpoint.urls,
            url_selection_strategy=user_config.endpoint.url_selection_strategy,
        )


class CreditPhaseConfig(AIPerfBaseModel):
    """Model for credit phase config. This is used to configure a credit phase.

    Stop conditions (first one reached wins):
    - total_expected_requests: Stop after sending this many total requests
    - expected_num_sessions: Stop starting NEW user sessions after this many (complete ongoing ones)
    - expected_duration_sec: Stop after this time
    """

    model_config = ConfigDict(frozen=True)

    phase: CreditPhase = Field(..., description="The phase of the credit phase.")
    timing_mode: TimingMode = Field(
        ...,
        description="The timing mode of the credit phase. Used to determine "
        "how to send requests to the workers.",
    )
    total_expected_requests: int | None = Field(
        default=None, gt=0, description="The total number of expected requests to send."
    )
    expected_num_sessions: int | None = Field(
        default=None, gt=0, description="The total number of expected sessions to send."
    )
    expected_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="The expected duration of the credit phase in seconds.",
    )
    seamless: bool = Field(
        default=False,
        description="Whether the credit phase should be seamless. "
        "Seamless phases start immediately after the previous phase sends all credits, "
        "without waiting for all credits to return. This can be used to maintain concurrency "
        "during phase transitions.",
    )
    concurrency: int | None = Field(
        default=None,
        gt=0,
        description="The max concurrency of the credit phase. "
        "This is the max number of requests that can be in flight at once. "
        "If None, the concurrency is unlimited.",
    )
    prefill_concurrency: int | None = Field(
        default=None,
        gt=0,
        description="The max concurrency of the prefill phase. "
        "This is the max number of requests that can be waiting for the first token at once. "
        "If None, the prefill concurrency is unlimited.",
    )
    request_rate: float | None = Field(
        default=None, gt=0, description="The request rate of the credit phase."
    )
    arrival_pattern: ArrivalPattern = Field(
        default=ArrivalPattern.POISSON,
        description="The arrival pattern of the credit phase.",
    )
    arrival_smoothness: float | None = Field(
        default=None,
        gt=0,
        description="The smoothness parameter for gamma distribution arrivals. "
        "Only used when arrival_pattern is GAMMA. Controls the shape of the distribution: "
        "1.0 = Poisson-like (exponential), <1.0 = bursty, >1.0 = smooth/regular. "
        "If None, defaults to 1.0 when using GAMMA arrival pattern.",
    )
    grace_period_sec: float | None = Field(
        default=None,
        ge=0,
        description="The grace period of the credit phase in seconds. "
        "This is the time to wait after the expected duration of the phase has elapsed "
        "before the phase is considered complete. This can be used to ensure that all requests "
        "have returned before the phase is considered complete. "
        "If None, the grace period is disabled.",
    )
    num_users: int | None = Field(
        default=None,
        ge=1,
        description="The number of concurrent users to use for the credit phase. "
        "This is only applicable when using user-centric rate limiting mode. ",
    )
    concurrency_ramp_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="Duration in seconds to ramp session concurrency from 1 to target. "
        "If None, concurrency starts at target immediately.",
    )
    request_concurrency: int | None = Field(
        default=None,
        gt=0,
        description="The max number of concurrent request streams (prefill + decode). "
        "Acquired every turn, released on credit return. "
        "If None, request concurrency is unlimited.",
    )
    request_concurrency_ramp_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="Duration in seconds to ramp request concurrency from 1 to target. "
        "If None, request concurrency starts at target immediately.",
    )
    prefill_concurrency_ramp_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="Duration in seconds to ramp prefill concurrency from 1 to target. "
        "If None, prefill concurrency starts at target immediately.",
    )
    request_rate_ramp_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="Duration in seconds to ramp request rate from 1 QPS to target. "
        "If None, request rate starts at target immediately.",
    )
    auto_offset_timestamps: bool = Field(
        default=InputDefaults.FIXED_SCHEDULE_AUTO_OFFSET,
        description="The auto offset timestamps of the timing manager.",
    )
    fixed_schedule_start_offset: int | None = Field(
        default=None,
        ge=0,
        description="The fixed schedule start offset of the timing manager.",
    )
    fixed_schedule_end_offset: int | None = Field(
        default=None,
        ge=0,
        description="The fixed schedule end offset of the timing manager.",
    )
    # Adaptive scale config
    start_users: int | None = Field(
        default=None,
        ge=1,
        description="Initial number of concurrent users for adaptive scale mode.",
    )
    max_users: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of concurrent users for adaptive scale mode.",
    )
    max_ttft_sec: float | None = Field(
        default=None,
        gt=0,
        description="Maximum TTFT threshold in seconds for adaptive scale scaling decisions.",
    )
    ttft_metric: str | None = Field(
        default=None,
        description="TTFT metric to use for scaling: 'p95', 'avg', or 'max'.",
    )
    assessment_period_sec: float | None = Field(
        default=None,
        gt=0,
        description="Period in seconds between scaling assessments.",
    )
    max_delay_sec: float | None = Field(
        default=None,
        ge=0,
        description="Maximum inter-request delay in seconds (clamp trace delays).",
    )
    time_scale: float | None = Field(
        default=None,
        gt=0,
        description="Time scale factor for trace delays.",
    )
    recycle_sessions: bool = Field(
        default=False,
        description="Whether to recycle completed sessions by sampling new conversations.",
    )
    stagger_ms: float | None = Field(
        default=None,
        ge=0,
        description="Delay in milliseconds between launching new users in adaptive scale.",
    )
    scaling_formula: str | None = Field(
        default=None,
        description="Scaling formula preset: 'conservative', 'aggressive', or 'linear'.",
    )
    max_new_tokens_per_period: int | None = Field(
        default=None,
        ge=0,
        description="Maximum new input tokens from new sessions per assessment period.",
    )
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable per-session TTFT-responsive rate limiting in adaptive scale mode.",
    )
    max_working_set_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Maximum total KV cache working set in tokens across all active sessions.",
    )
    block_size: int = Field(
        default=64,
        description="KV cache block size in tokens for working set calculation.",
    )
    adaptive_scale_slo: dict[str, float] | None = Field(
        default=None,
        description="SLO thresholds for goodput-based adaptive scaling (display units, e.g. ms).",
    )
    min_goodput_ratio: float = Field(
        default=0.95,
        description="Minimum goodput ratio required to continue scaling up.",
    )
    cache_ttl_sec: float = Field(
        default=3600.0,
        gt=0,
        description="Main agent KV cache TTL in seconds for working set eviction.",
    )
    subagent_cache_ttl_sec: float = Field(
        default=300.0,
        gt=0,
        description="Subagent KV cache TTL in seconds for working set eviction.",
    )
    # Agentic load config
    user_spawn_rate: float | None = Field(
        default=None,
        gt=0,
        description="Users to spawn per second during ramp-up for agentic load mode.",
    )
    settling_time_sec: float | None = Field(
        default=None,
        ge=0,
        description="Wait time in seconds after all users spawned before measurement starts.",
    )
    trajectories_per_user: int | None = Field(
        default=None,
        ge=1,
        description="Number of conversations assigned to each user in agentic load mode.",
    )
    max_isl_offset: int | None = Field(
        default=None,
        ge=0,
        description="Maximum initial starting line offset for first trajectory to prevent bias.",
    )
    agentic_seed: int | None = Field(
        default=None,
        description="Random seed for deterministic trajectory assignment in agentic load mode.",
    )
    benchmark_id: str | None = Field(
        default=None,
        description="Unique benchmark run ID for cache bust uniqueness across runs.",
    )


def _build_warmup_config(user_config: UserConfig) -> CreditPhaseConfig | None:
    """Build warmup phase config if any warmup stop condition is set.

    Returns None if warmup disabled (no stop conditions).
    Warmup triggers JIT compilation, memory allocation, and connection pool
    initialization so profiling measurements aren't polluted by cold-start effects.

    Note:
        When warmup_grace_period is not specified, defaults to infinity (wait forever
        for in-flight requests). This differs from the CreditPhaseConfig field default
        of None (disabled) because warmup should always complete all requests.
    """
    loadgen = user_config.loadgen
    if not (
        loadgen.warmup_request_count
        or loadgen.warmup_duration
        or loadgen.warmup_num_sessions
    ):
        return None

    request_rate = loadgen.warmup_request_rate or loadgen.request_rate
    arrival_pattern = loadgen.warmup_arrival_pattern or loadgen.arrival_pattern
    concurrency = loadgen.warmup_concurrency or loadgen.concurrency
    request_concurrency = (
        loadgen.warmup_request_concurrency or loadgen.request_concurrency
    )
    prefill_concurrency = (
        loadgen.warmup_prefill_concurrency or loadgen.prefill_concurrency
    )
    if request_rate is None:
        arrival_pattern = ArrivalPattern.CONCURRENCY_BURST
        if concurrency is None and prefill_concurrency is None:
            concurrency = 1
            # TODO: We should add a warning here

    return CreditPhaseConfig(
        phase=CreditPhase.WARMUP,
        # Warmup phase is always request rate timing mode
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=loadgen.warmup_request_count,
        expected_duration_sec=loadgen.warmup_duration,
        expected_num_sessions=loadgen.warmup_num_sessions,
        concurrency=concurrency,
        request_concurrency=request_concurrency,
        prefill_concurrency=prefill_concurrency,
        request_rate=request_rate,
        arrival_pattern=arrival_pattern,
        arrival_smoothness=loadgen.arrival_smoothness,
        seamless=False,
        grace_period_sec=loadgen.warmup_grace_period if loadgen.warmup_grace_period is not None else float('inf'),
        concurrency_ramp_duration_sec=loadgen.warmup_concurrency_ramp_duration or loadgen.concurrency_ramp_duration,
        request_concurrency_ramp_duration_sec=loadgen.warmup_request_concurrency_ramp_duration or loadgen.request_concurrency_ramp_duration,
        prefill_concurrency_ramp_duration_sec=loadgen.warmup_prefill_concurrency_ramp_duration or loadgen.prefill_concurrency_ramp_duration,
        request_rate_ramp_duration_sec=loadgen.warmup_request_rate_ramp_duration or loadgen.request_rate_ramp_duration,
    )  # fmt: skip


def _build_profiling_config(user_config: UserConfig) -> CreditPhaseConfig:
    """Build profiling phase config (always created).

    Main benchmark phase where all performance metrics are collected.
    Grace period allows in-flight requests to complete after the stop condition
    is met, ensuring metrics include requests that were sent before the deadline.
    """

    loadgen = user_config.loadgen
    input = user_config.input

    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=user_config.timing_mode,
        expected_duration_sec=loadgen.benchmark_duration,
        total_expected_requests=loadgen.request_count,
        expected_num_sessions=input.conversation.num,
        concurrency=loadgen.concurrency,
        request_concurrency=loadgen.request_concurrency,
        prefill_concurrency=loadgen.prefill_concurrency,
        request_rate=loadgen.request_rate or loadgen.user_centric_rate,
        arrival_pattern=loadgen.arrival_pattern,
        arrival_smoothness=loadgen.arrival_smoothness,
        grace_period_sec=loadgen.benchmark_grace_period,
        num_users=loadgen.num_users,
        concurrency_ramp_duration_sec=loadgen.concurrency_ramp_duration,
        request_concurrency_ramp_duration_sec=loadgen.request_concurrency_ramp_duration,
        prefill_concurrency_ramp_duration_sec=loadgen.prefill_concurrency_ramp_duration,
        request_rate_ramp_duration_sec=loadgen.request_rate_ramp_duration,
        # Fixed schedule config
        auto_offset_timestamps=input.fixed_schedule_auto_offset,
        fixed_schedule_start_offset=input.fixed_schedule_start_offset,
        fixed_schedule_end_offset=input.fixed_schedule_end_offset,
        # Adaptive scale config
        start_users=input.adaptive_scale_start_users if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else None,
        max_users=input.adaptive_scale_max_users if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else None,
        max_ttft_sec=input.adaptive_scale_max_ttft if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else None,
        ttft_metric=input.adaptive_scale_ttft_metric if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else None,
        assessment_period_sec=input.adaptive_scale_assessment_period if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else None,
        max_delay_sec=input.adaptive_scale_max_delay if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else None,
        time_scale=input.adaptive_scale_time_scale if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else None,
        recycle_sessions=input.adaptive_scale_recycle if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else False,
        stagger_ms=input.adaptive_scale_stagger_ms if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else None,
        scaling_formula=input.adaptive_scale_formula if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else None,
        max_new_tokens_per_period=input.adaptive_scale_max_new_tokens_per_period if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else None,
        enable_rate_limiting=input.adaptive_scale_enable_rate_limiting if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else True,
        max_working_set_tokens=input.adaptive_scale_max_working_set_tokens if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else None,
        adaptive_scale_slo=input.adaptive_scale_slo if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else None,
        min_goodput_ratio=input.adaptive_scale_min_goodput_ratio if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else 0.95,
        cache_ttl_sec=input.coding_session.cache_ttl_sec if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else 3600.0,
        subagent_cache_ttl_sec=input.coding_session.subagent_cache_ttl_sec if user_config.timing_mode == TimingMode.ADAPTIVE_SCALE else 300.0,
        # Agentic load config
        user_spawn_rate=loadgen.agentic_user_spawn_rate if user_config.timing_mode == TimingMode.AGENTIC_LOAD else None,
        settling_time_sec=loadgen.agentic_settling_time if user_config.timing_mode == TimingMode.AGENTIC_LOAD else None,
        trajectories_per_user=input.agentic_trajectories_per_user if user_config.timing_mode == TimingMode.AGENTIC_LOAD else None,
        max_isl_offset=input.agentic_max_isl_offset if user_config.timing_mode == TimingMode.AGENTIC_LOAD else None,
        agentic_seed=input.random_seed if user_config.timing_mode == TimingMode.AGENTIC_LOAD else None,
        benchmark_id=user_config.benchmark_id if user_config.timing_mode == TimingMode.AGENTIC_LOAD else None,
    )  # fmt: skip
