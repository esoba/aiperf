# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Phase Configuration

Unified phase model with type-based load generation strategies.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from annotated_types import Ge, Gt
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

from aiperf.common.enums import CaseInsensitiveStrEnum
from aiperf.config.constraints import (
    AllowedOnlyWith,
    AtLeastOneUnless,
    ConstraintsMixin,
    ForbiddenWith,
    LessThanOrEqual,
    RequiredIf,
    RequiredIfIn,
)
from aiperf.plugin.enums import RampType


class PhaseType(CaseInsensitiveStrEnum):
    """Load generation strategy for a benchmark phase."""

    CONCURRENCY = "concurrency"
    CONSTANT = "constant"
    GAMMA = "gamma"
    POISSON = "poisson"
    FIXED_SCHEDULE = "fixed_schedule"
    USER_CENTRIC = "user_centric"


PhaseTypeStr = str

__all__ = [
    "PhaseType",
    "PhaseTypeStr",
    "RampConfig",
    "CancellationConfig",
    "PhaseConfig",
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
            default=0.0,
            ge=0.0,
            description="Seconds to wait after sending before cancelling. "
            "0.0 means cancel immediately after send.",
        ),
    ]


# =============================================================================
# UNIFIED PHASE CONFIGURATION
# =============================================================================


class PhaseConfig(BaseModel, ConstraintsMixin):
    """
    Unified phase configuration with type-based load generation.

    The `type` field determines the load generation strategy and which
    fields are required/relevant:

    **concurrency**: Dispatch immediately when concurrency slot opens.
        - Primary control: `concurrency`
        - No rate limiting, pure concurrency-based throughput

    **poisson/gamma/constant**: Rate-controlled request generation.
        - Primary control: `rate` (required)
        - `concurrency` acts as a cap on in-flight requests
        - `smoothness` only used with gamma type

    **user_centric**: Simulate N concurrent users sharing a global rate.
        - Required: `users` and `rate`
        - Requests distributed across simulated users

    **fixed_schedule**: Replay requests at predetermined timestamps.
        - Timestamps from trace dataset
        - `auto_offset`, `start_offset`, `end_offset` control replay
    """

    model_config = ConfigDict(extra="forbid")

    # =========================================================================
    # REQUIRED FIELDS
    # =========================================================================

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

    # Name is injected from dict key via _name attribute, accessed via property
    _name: str | None = PrivateAttr(default=None)

    # =========================================================================
    # UNIVERSAL FIELDS (all modes)
    # =========================================================================

    dataset: Annotated[
        str | None,
        Field(
            default=None,
            description="Name of dataset to use (from datasets section). "
            "If not specified, uses first defined dataset.",
        ),
    ]

    exclude: Annotated[
        bool,
        Field(
            default=False,
            description="Exclude this phase's metrics from final results. "
            "Set to true for warmup or cooldown phases.",
        ),
    ]

    # -------------------------------------------------------------------------
    # Stop Conditions (at least one required for non-fixed_schedule types)
    # -------------------------------------------------------------------------

    requests: Annotated[
        int | None,
        Ge(1),
        AtLeastOneUnless("stop", "type", PhaseType.FIXED_SCHEDULE),
        Field(
            default=None,
            description="Stop after this many requests sent (must be >= 1).",
        ),
    ]

    duration: Annotated[
        DurationSpec,
        Gt(0),
        AtLeastOneUnless("stop", "type", PhaseType.FIXED_SCHEDULE),
        Field(
            default=None,
            description="Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'.",
        ),
    ]

    sessions: Annotated[
        int | None,
        Ge(1),
        AtLeastOneUnless("stop", "type", PhaseType.FIXED_SCHEDULE),
        Field(
            default=None,
            description="Stop after this many sessions completed (must be >= 1).",
        ),
    ]

    # -------------------------------------------------------------------------
    # Concurrency Control (all types)
    # -------------------------------------------------------------------------

    concurrency: Annotated[
        int | None,
        Ge(1),
        RequiredIf("type", PhaseType.CONCURRENCY),
        Field(
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
        Ge(1),
        LessThanOrEqual("concurrency"),
        Field(
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
    # Rate Control (poisson, gamma, constant, user_centric types)
    # -------------------------------------------------------------------------

    rate: Annotated[
        float | None,
        Gt(0),
        ForbiddenWith("type", PhaseType.CONCURRENCY),
        RequiredIfIn("type", [PhaseType.POISSON, PhaseType.GAMMA, PhaseType.CONSTANT]),
        RequiredIf("type", PhaseType.USER_CENTRIC),
        Field(
            default=None,
            description="Target request rate in requests per second (must be > 0). "
            "Required for poisson/gamma/constant types. "
            "For user_centric: global rate shared across all users.",
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

    smoothness: Annotated[
        float | None,
        Gt(0),
        AllowedOnlyWith("type", PhaseType.GAMMA),
        Field(
            default=None,
            description="Gamma distribution shape parameter (must be > 0). "
            "Only used with type='gamma'. "
            "1.0 = Poisson, <1 = bursty, >1 = regular.",
        ),
    ]

    # -------------------------------------------------------------------------
    # User-Centric Type Fields
    # -------------------------------------------------------------------------

    users: Annotated[
        int | None,
        Ge(1),
        RequiredIf("type", PhaseType.USER_CENTRIC),
        Field(
            default=None,
            description="Number of simulated concurrent users (must be >= 1). "
            "Required for user_centric type. "
            "Requests distributed across users to achieve global rate.",
        ),
    ]

    # -------------------------------------------------------------------------
    # Fixed Schedule Type Fields
    # -------------------------------------------------------------------------

    auto_offset: Annotated[
        bool,
        Field(
            default=True,
            description="Normalize trace timestamps to start at 0. "
            "Subtracts minimum timestamp from all entries. "
            "Only used with type='fixed_schedule'.",
        ),
    ]

    start_offset: Annotated[
        int | None,
        Ge(0),
        LessThanOrEqual("end_offset"),
        Field(
            default=None,
            description="Filter out trace requests before this timestamp in ms (must be >= 0). "
            "Only used with type='fixed_schedule'.",
        ),
    ]

    end_offset: Annotated[
        int | None,
        Ge(0),
        Field(
            default=None,
            description="Filter out trace requests after this timestamp in ms (must be >= 0). "
            "Only used with type='fixed_schedule'.",
        ),
    ]

    # -------------------------------------------------------------------------
    # Transition Settings (all types)
    # -------------------------------------------------------------------------

    grace_period: Annotated[
        DurationSpec,
        Ge(0),
        Field(
            default=None,
            description="Seconds to wait for in-flight requests at phase end (must be >= 0). "
            "Supports: 30, '30s', '2m'.",
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
    # MODEL VALIDATORS
    # =========================================================================

    @model_validator(mode="after")
    def validate_fixed_schedule_offsets(self) -> PhaseConfig:
        """Validate auto_offset and start_offset are mutually exclusive for fixed_schedule."""
        if (
            self.type == PhaseType.FIXED_SCHEDULE
            and self.auto_offset
            and self.start_offset is not None
        ):
            raise ValueError(
                f"Phase '{self._display_name}': 'auto_offset' and 'start_offset' are "
                "mutually exclusive. Use one or the other."
            )
        return self
