# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field

from aiperf.common.enums import CreditPhase
from aiperf.config import InputDefaults

if TYPE_CHECKING:
    from aiperf.config import BenchmarkConfig
    from aiperf.config.phases import BasePhaseConfig

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.plugin.enums import (
    ArrivalPattern,
    PhaseType,
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
    def from_user_config(cls, config: BenchmarkConfig) -> TimingConfig:
        """Alias for from_config (backward compatibility)."""
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: BenchmarkConfig) -> TimingConfig:
        """Build TimingConfig from AIPerfConfig phases in config order.

        Each phase uses its dict key as the phase name and preserves
        exclude_from_results from the config.
        """
        phase_configs: list[CreditPhaseConfig] = []
        cancellation = RequestCancellationConfig()

        for name, phase in config.phases.items():
            phase_config = _build_credit_phase_config(
                phase, phase_name=name, exclude_from_results=phase.exclude_from_results
            )
            phase_configs.append(phase_config)

            # Use first non-excluded phase's cancellation as global cancellation
            if (
                not phase.exclude_from_results
                and phase.cancellation
                and cancellation.rate is None
            ):
                cancellation = RequestCancellationConfig(
                    rate=phase.cancellation.rate,
                    delay=phase.cancellation.delay,
                )

        return cls(
            phase_configs=phase_configs,
            request_cancellation=cancellation,
            urls=config.endpoint.urls,
            url_selection_strategy=config.endpoint.url_strategy,
        )


class CreditPhaseConfig(AIPerfBaseModel):
    """Model for credit phase config. This is used to configure a credit phase.

    Stop conditions (first one reached wins):
    - total_expected_requests: Stop after sending this many total requests
    - expected_num_sessions: Stop starting NEW user sessions after this many (complete ongoing ones)
    - expected_duration_sec: Stop after this time
    """

    model_config = ConfigDict(frozen=True)

    phase: CreditPhase = Field(
        ..., description="The name of the credit phase (e.g. 'warmup', 'main')."
    )
    exclude_from_results: bool = Field(
        default=False,
        description="Whether this phase is excluded from final results.",
    )
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


def _phase_type_to_timing(phase_type: PhaseType) -> tuple[TimingMode, ArrivalPattern]:
    """Map PhaseType to (TimingMode, ArrivalPattern).

    Delegates to the shared resolution function in config.resolved.
    """
    from aiperf.config.resolved import get_phase_timing

    return get_phase_timing(phase_type)


def _build_credit_phase_config(
    phase: BasePhaseConfig,
    *,
    phase_name: str,
    exclude_from_results: bool,
) -> CreditPhaseConfig:
    """Build a CreditPhaseConfig from a phase config.

    Maps the AIPerfConfig phase structure to the internal
    CreditPhaseConfig used by the timing system. Uses getattr for
    fields that only exist on specific phase types.

    For excluded phases (exclude_from_results=True), grace_period defaults to infinity
    to ensure all in-flight requests complete before the next phase begins.
    """
    timing_mode, arrival_pattern = _phase_type_to_timing(phase.type)

    grace_period = phase.grace_period
    if exclude_from_results and grace_period is None:
        grace_period = float("inf")

    rate_ramp = getattr(phase, "rate_ramp", None)

    return CreditPhaseConfig(
        phase=phase_name,
        exclude_from_results=exclude_from_results,
        timing_mode=timing_mode,
        arrival_pattern=arrival_pattern,
        total_expected_requests=phase.requests,
        expected_duration_sec=phase.duration,
        expected_num_sessions=phase.sessions,
        concurrency=phase.concurrency,
        prefill_concurrency=phase.prefill_concurrency,
        request_rate=getattr(phase, "rate", None),
        arrival_smoothness=getattr(phase, "smoothness", None),
        num_users=getattr(phase, "users", None),
        grace_period_sec=grace_period,
        seamless=phase.seamless,
        auto_offset_timestamps=getattr(phase, "auto_offset", True),
        fixed_schedule_start_offset=getattr(phase, "start_offset", None),
        fixed_schedule_end_offset=getattr(phase, "end_offset", None),
        concurrency_ramp_duration_sec=phase.concurrency_ramp.duration if phase.concurrency_ramp else None,
        prefill_concurrency_ramp_duration_sec=phase.prefill_ramp.duration if phase.prefill_ramp else None,
        request_rate_ramp_duration_sec=rate_ramp.duration if rate_ramp else None,
    )  # fmt: skip
