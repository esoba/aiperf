# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Phase Configuration

Discriminated union of phase types. Each concrete phase type only exposes
fields it supports; ``extra="forbid"`` rejects unknown fields structurally,
making invalid states unrepresentable.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    PrivateAttr,
    model_validator,
)
from typing_extensions import Self

from aiperf.plugin.enums import PhaseType, PhaseTypeStr, RampType

__all__ = [
    "BasePhaseConfig",
    "CancellationConfig",
    "ConcurrencyPhase",
    "ConstantPhase",
    "FixedSchedulePhase",
    "GammaPhase",
    "PhaseConfig",
    "PhaseType",
    "PhaseTypeStr",
    "PoissonPhase",
    "RampConfig",
    "RatePhaseConfig",
    "UserCentricPhase",
]


# =============================================================================
# DURATION PARSING
# =============================================================================

_DURATION_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(s|sec|m|min|h|hr|hour)?$", re.IGNORECASE
)


def _parse_duration(v: Any) -> float | None:
    """Parse duration from various formats to seconds.

    Supports:
        - Numbers: 30, 5.5 (interpreted as seconds)
        - Strings: "30s", "5m", "2h", "30 sec", "5 min"

    Returns:
        Duration in seconds, or None if input is None.

    Raises:
        ValueError: If string format is invalid.
    """
    if v is None:
        return None
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        match = _DURATION_PATTERN.match(v.strip())
        if not match:
            raise ValueError(
                f"Invalid duration format: {v!r}. Use number or '30s', '5m', '2h'."
            )
        value = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        if unit in ("s", "sec"):
            return value
        elif unit in ("m", "min"):
            return value * 60
        elif unit in ("h", "hr", "hour"):
            return value * 3600
    return v


def _normalize_duration(v: Any) -> Any:
    """Normalize duration fields to float seconds."""
    return _parse_duration(v)


# Type alias for duration that supports shorthand strings
DurationSpec = Annotated[float | None, BeforeValidator(_normalize_duration)]


# =============================================================================
# RAMP CONFIGURATION
# =============================================================================


class RampConfig(BaseModel):
    """
    Configuration for gradual value ramping.

    Controls how a value (concurrency, rate, etc.) transitions from
    start to target over time.
    """

    model_config = ConfigDict(extra="forbid")

    duration: Annotated[
        float,
        Field(
            gt=0.0,
            description="Seconds to ramp from start to target value.",
        ),
    ]

    strategy: Annotated[
        RampType,
        Field(
            default=RampType.LINEAR,
            description="Ramp curve shape: "
            "linear (constant rate), "
            "exponential (slow start, fast finish), "
            "poisson (stochastic with guaranteed completion).",
        ),
    ]


def _normalize_ramp(v: Any) -> Any:
    """Normalize ramp shorthand to RampConfig dict."""
    if v is None:
        return None
    if isinstance(v, int | float | str):
        duration = _parse_duration(v)
        return {"duration": duration}
    if isinstance(v, dict) and "duration" in v:
        v["duration"] = _parse_duration(v["duration"])
    return v


# Type alias for ramp that supports shorthand (just duration as number or string)
RampSpec = Annotated[RampConfig | None, BeforeValidator(_normalize_ramp)]


# =============================================================================
# CANCELLATION CONFIGURATION
# =============================================================================


class CancellationConfig(BaseModel):
    """
    Configuration for request cancellation testing.

    Enables testing server behavior when clients cancel requests mid-flight.
    """

    model_config = ConfigDict(extra="forbid")

    # TODO: We should add a warning for values below 1.0, to ensure the user is aware
    # that the value is a percentage.
    rate: Annotated[
        float,
        Field(
            ge=0.0,
            le=100.0,
            description="Percentage of requests to cancel (0-100). "
            "10.0 means 10%% of requests will be cancelled.",
        ),
    ]

    delay: Annotated[
        float,
        Field(
            ge=0.0,
            default=0.0,
            description="Seconds to wait after sending before cancelling. "
            "0.0 means cancel immediately after send.",
        ),
    ]


# =============================================================================
# PHASE HIERARCHY
# =============================================================================


class BasePhaseConfig(BaseModel):
    """Base configuration shared by all phase types.

    Not instantiated directly -- use a concrete type via the
    :data:`PhaseConfig` discriminated union.
    """

    model_config = ConfigDict(extra="forbid")

    # Narrowed to Literal in each concrete class; declared here so that
    # code holding a BasePhaseConfig reference can always access .type.
    type: Annotated[
        PhaseType,
        Field(
            description="Load generation type. "
            "concurrency: concurrency-controlled immediate dispatch, "
            "poisson/gamma/constant: rate-controlled with arrival distribution, "
            "user_centric: N users sharing global rate, "
            "fixed_schedule: replay from timestamps.",
        ),
    ]

    _name: str | None = PrivateAttr(default=None)

    # =========================================================================
    # UNIVERSAL FIELDS
    # =========================================================================

    dataset: Annotated[
        str | None,
        Field(
            default=None,
            description="Name of dataset to use (from datasets section). "
            "If not specified, uses first defined dataset.",
        ),
    ]

    exclude_from_results: Annotated[
        bool,
        Field(
            default=False,
            description="Exclude this phase's metrics from final results. "
            "Set to true for warmup or cooldown phases.",
        ),
    ]

    # -------------------------------------------------------------------------
    # Stop Conditions (at least one required unless _stop_condition_required=False)
    # -------------------------------------------------------------------------

    requests: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Stop after this many requests sent (must be >= 1).",
        ),
    ]

    duration: Annotated[
        DurationSpec,
        Field(
            gt=0,
            default=None,
            description="Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'.",
        ),
    ]

    sessions: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Stop after this many sessions completed (must be >= 1).",
        ),
    ]

    # -------------------------------------------------------------------------
    # Concurrency Control
    # -------------------------------------------------------------------------

    concurrency: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Max concurrent in-flight requests (must be >= 1). "
            "For concurrency type: primary control. "
            "For rate types: acts as a cap.",
        ),
    ]

    concurrency_ramp: Annotated[
        RampSpec,
        Field(
            default=None,
            description="Ramp concurrency from lower value. "
            "Can be number (seconds) or {duration, strategy}.",
        ),
    ]

    prefill_concurrency: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Max concurrent requests in prefill stage (must be >= 1). "
            "Limits requests before first token received.",
        ),
    ]

    prefill_ramp: Annotated[
        RampSpec,
        Field(
            default=None,
            description="Ramp prefill_concurrency from lower value. "
            "Can be number (seconds) or {duration, strategy}.",
        ),
    ]

    # -------------------------------------------------------------------------
    # Transition Settings
    # -------------------------------------------------------------------------

    grace_period: Annotated[
        DurationSpec,
        Field(
            ge=0,
            default=None,
            description="Seconds to wait for in-flight requests after duration expires (must be >= 0). "
            "Requires 'duration' to be set. Supports: 30, '30s', '2m'.",
        ),
    ]

    cancellation: Annotated[
        CancellationConfig | None,
        Field(
            default=None,
            description="Request cancellation testing configuration.",
        ),
    ]

    seamless: Annotated[
        bool,
        Field(
            default=False,
            description="Start this phase immediately when previous phase stops, "
            "without waiting for in-flight requests to complete. "
            "Cannot be True for the first phase.",
        ),
    ]

    # Subclasses set False to opt out (e.g. FixedSchedulePhase where
    # the stop condition is inferred from the dataset).
    _stop_condition_required: ClassVar[bool] = True

    # =========================================================================
    # HELPERS
    # =========================================================================

    @property
    def name(self) -> str:
        """Phase name (injected from dict key)."""
        return self._name or "unnamed"

    @property
    def _display_name(self) -> str:
        """Name for error messages (handles None case)."""
        return self.name

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @model_validator(mode="after")
    def _validate_phase_constraints(self) -> Self:
        """Validate stop condition and cross-field constraints."""
        if (
            self._stop_condition_required
            and self.requests is None
            and self.duration is None
            and self.sessions is None
        ):
            raise ValueError(
                f"Phase '{self._display_name}': at least one of "
                "'requests', 'duration', or 'sessions' must be specified"
            )
        if (
            self.prefill_concurrency is not None
            and self.concurrency is not None
            and self.prefill_concurrency > self.concurrency
        ):
            raise ValueError(
                f"Phase '{self._display_name}': "
                "prefill_concurrency must be <= concurrency"
            )
        if self.grace_period is not None and self.duration is None:
            raise ValueError(
                f"Phase '{self._display_name}': "
                "grace_period requires duration to be set"
            )
        return self


# =============================================================================
# CONCURRENCY PHASE
# =============================================================================


class ConcurrencyPhase(BasePhaseConfig):
    """Concurrency-controlled load: dispatch immediately when a slot opens.

    Primary control is ``concurrency`` (defaults to 1).
    No rate limiting -- pure concurrency-based throughput.
    """

    type: Annotated[
        Literal[PhaseType.CONCURRENCY],
        Field(description="Concurrency-controlled immediate dispatch."),
    ]

    concurrency: Annotated[
        int,
        Field(
            ge=1,
            default=1,
            description="Max concurrent in-flight requests (must be >= 1). "
            "Primary control for concurrency phases.",
        ),
    ]


# =============================================================================
# RATE-CONTROLLED PHASES
# =============================================================================


class RatePhaseConfig(BasePhaseConfig):
    """Base for rate-controlled phases. Not instantiated directly."""

    rate: Annotated[
        float,
        Field(
            gt=0,
            description="Target request rate in requests per second (must be > 0).",
        ),
    ]

    rate_ramp: Annotated[
        RampSpec,
        Field(
            default=None,
            description="Ramp rate from lower value. "
            "Can be number (seconds) or {duration, strategy}.",
        ),
    ]


class PoissonPhase(RatePhaseConfig):
    """Poisson-distributed request arrivals at the target rate."""

    type: Annotated[
        Literal[PhaseType.POISSON],
        Field(description="Poisson-distributed rate-controlled arrivals."),
    ]


class GammaPhase(RatePhaseConfig):
    """Gamma-distributed request arrivals with configurable smoothness."""

    type: Annotated[
        Literal[PhaseType.GAMMA],
        Field(description="Gamma-distributed rate-controlled arrivals."),
    ]

    smoothness: Annotated[
        float | None,
        Field(
            gt=0,
            default=None,
            description="Gamma distribution shape parameter (must be > 0). "
            "1.0 = Poisson, <1 = bursty, >1 = regular.",
        ),
    ]


class ConstantPhase(RatePhaseConfig):
    """Constant-rate request arrivals (fixed inter-arrival time)."""

    type: Annotated[
        Literal[PhaseType.CONSTANT],
        Field(description="Constant rate-controlled arrivals."),
    ]


class UserCentricPhase(RatePhaseConfig):
    """N concurrent users sharing a global request rate.

    Requires multi-turn conversations. Each user gets a proportional
    share of the global ``rate``.
    """

    type: Annotated[
        Literal[PhaseType.USER_CENTRIC],
        Field(description="N users sharing a global request rate."),
    ]

    users: Annotated[
        int,
        Field(
            ge=1,
            description="Number of simulated concurrent users (must be >= 1). "
            "Requests distributed across users to achieve global rate.",
        ),
    ]

    @model_validator(mode="after")
    def validate_user_centric_constraints(self) -> UserCentricPhase:
        """Validate user-centric mode constraints."""
        if self.sessions is not None and self.sessions < self.users:
            raise ValueError(
                f"Phase '{self._display_name}': --num-sessions ({self.sessions}) must be "
                f">= --num-users ({self.users}). Each user needs at least one session."
            )

        if self.requests is not None and self.requests < self.users:
            raise ValueError(
                f"Phase '{self._display_name}': --request-count ({self.requests}) must be "
                f">= --num-users ({self.users}). Each user needs at least one request."
            )

        return self


# =============================================================================
# FIXED SCHEDULE PHASE
# =============================================================================


class FixedSchedulePhase(BasePhaseConfig):
    """Replay requests at predetermined timestamps from a trace dataset.

    Stop condition not required -- the trace dataset determines when the
    phase ends.
    """

    _stop_condition_required: ClassVar[bool] = False

    type: Annotated[
        Literal[PhaseType.FIXED_SCHEDULE],
        Field(description="Replay requests at trace timestamps."),
    ]

    auto_offset: Annotated[
        bool,
        Field(
            default=True,
            description="Normalize trace timestamps to start at 0. "
            "Subtracts minimum timestamp from all entries.",
        ),
    ]

    start_offset: Annotated[
        int | None,
        Field(
            ge=0,
            default=None,
            description="Filter out trace requests before this timestamp in ms (must be >= 0).",
        ),
    ]

    end_offset: Annotated[
        int | None,
        Field(
            ge=0,
            default=None,
            description="Filter out trace requests after this timestamp in ms (must be >= 0).",
        ),
    ]

    @model_validator(mode="after")
    def _validate_fixed_schedule_constraints(self) -> Self:
        if self.auto_offset and self.start_offset is not None:
            raise ValueError("auto_offset cannot be True when start_offset is set")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.start_offset > self.end_offset
        ):
            raise ValueError("start_offset must be <= end_offset")
        return self


# =============================================================================
# DISCRIMINATED UNION
# =============================================================================

PhaseConfig = Annotated[
    ConcurrencyPhase
    | PoissonPhase
    | GammaPhase
    | ConstantPhase
    | UserCentricPhase
    | FixedSchedulePhase,
    Discriminator("type"),
]
