# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Reusable Type Definitions

This module defines reusable type aliases and Pydantic models that are
used across multiple configuration sections. These types provide
consistent handling of statistical distributions, token length
specifications, and other common patterns.

Key Types:
    - MeanStddev: Distribution with mean/stddev (coerces int/float)
    - SequenceDistributionEntry: ISL/OSL distribution entry with probability

Design Philosophy:
    The type system supports multiple input formats for user convenience
    while normalizing to a consistent internal representation. For example,
    an ISL can be specified as:
        - Simple integer: 512 (coerced to MeanStddev(mean=512, stddev=0))
        - Dictionary: {mean: 512, stddev: 50}

    Validators automatically normalize these to MeanStddev objects for
    consistent downstream processing.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "MeanStddev",
    "SequenceDistributionEntry",
    "validate_probability_distribution",
]


class MeanStddev(BaseModel):
    """
    Represents a statistical distribution with mean and standard deviation.

    This is the fundamental building block for specifying token lengths,
    dimensions, and other quantities that can have natural variation.

    The distribution is assumed to be normal (Gaussian). Values are sampled
    and then clamped to valid ranges (e.g., positive integers for token counts).

    Attributes:
        mean: The average/expected value.
            For token counts, this is the target number of tokens.
            For dimensions, this is the target size in pixels.

        stddev: The standard deviation (spread) of the distribution.
            A value of 0 means fixed/deterministic values.
            Higher values produce more variation between samples.

    Input Forms (all normalized to MeanStddev):
        - Integer: 512 → MeanStddev(mean=512.0, stddev=0.0)
        - Float: 512.5 → MeanStddev(mean=512.5, stddev=0.0)
        - Dict: {mean: 512, stddev: 50} → MeanStddev(mean=512.0, stddev=50.0)

    YAML Representations:
        # Shorthand (stddev defaults to 0):
        isl: 512

        # Explicit:
        isl:
          mean: 512
          stddev: 50
    """

    mean: Annotated[
        float,
        Field(
            description="The mean (average) value of the distribution. "
            "For token counts, represents the target number of tokens."
        ),
    ]

    stddev: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description="The standard deviation of the distribution. "
            "A value of 0 means deterministic (no variation). "
            "Higher values produce more spread in sampled values.",
        ),
    ]

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"mean": 512, "stddev": 0},
                {"mean": 550, "stddev": 50},
            ]
        },
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_scalar_to_distribution(cls, data: Any) -> Any:
        """Coerce int/float to distribution dict format."""
        if isinstance(data, int | float):
            return {"mean": float(data), "stddev": 0.0}
        return data


class SequenceDistributionEntry(BaseModel):
    """
    Defines a single entry in an ISL/OSL probability distribution.

    AIPerf uniquely supports multi-modal token length distributions,
    allowing you to specify different ISL/OSL combinations with
    their relative frequencies. This enables realistic workload
    modeling where different request types have different sizes.

    Attributes:
        isl: Input sequence length for this distribution bucket.
            Can be a fixed value or a distribution itself.

        isl_stddev: Standard deviation for ISL (shorthand).
            If provided, creates a distribution around isl mean.
            Mutually exclusive with isl being a MeanStddev.

        osl: Output sequence length for this distribution bucket.
            Can be a fixed value or a distribution itself.

        osl_stddev: Standard deviation for OSL (shorthand).
            If provided, creates a distribution around osl mean.
            Mutually exclusive with osl being a MeanStddev.

        probability: Relative probability weight (0-100).
            Weights are normalized across all entries.
            A weight of 40 means 40% of requests will use this bucket.

    Examples:
        Production traffic simulation:
            >>> entries = [
            ...     SequenceDistributionEntry(isl=128, osl=64, probability=40),
            ...     SequenceDistributionEntry(isl=512, osl=256, probability=35),
            ...     SequenceDistributionEntry(isl=2048, osl=512, probability=20),
            ...     SequenceDistributionEntry(isl=8192, osl=1024, probability=5),
            ... ]

    YAML Representation:
        sequence_distribution:
          - {isl: 128, osl: 64, probability: 40}
          - {isl: 512, osl: 256, probability: 35}
          - {isl: 2048, osl: 512, probability: 20}
          - {isl: 8192, osl: 1024, probability: 5}

        # With variance:
          - {isl: 128, isl_stddev: 20, osl: 64, osl_stddev: 10, probability: 40}
    """

    isl: Annotated[
        MeanStddev,
        Field(
            description="Input sequence length (tokens). "
            "Can be a fixed integer or a {mean, stddev} distribution."
        ),
    ]

    isl_stddev: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description="Shorthand for ISL standard deviation. "
            "If provided when isl is an integer, creates a distribution. "
            "Cannot be used when isl is already a {mean, stddev} dict.",
        ),
    ]

    osl: Annotated[
        MeanStddev,
        Field(
            description="Output sequence length (tokens). "
            "Can be a fixed integer or a {mean, stddev} distribution."
        ),
    ]

    osl_stddev: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description="Shorthand for OSL standard deviation. "
            "If provided when osl is an integer, creates a distribution. "
            "Cannot be used when osl is already a {mean, stddev} dict.",
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def merge_stddev_shorthand(cls, data: Any) -> Any:
        """Merge isl_stddev/osl_stddev shorthand into isl/osl fields."""
        if not isinstance(data, dict):
            return data

        # Handle isl + isl_stddev shorthand
        if "isl_stddev" in data and data["isl_stddev"] is not None:
            isl = data.get("isl")
            if isinstance(isl, int | float):
                data["isl"] = {"mean": float(isl), "stddev": data["isl_stddev"]}
            data["isl_stddev"] = None

        # Handle osl + osl_stddev shorthand
        if "osl_stddev" in data and data["osl_stddev"] is not None:
            osl = data.get("osl")
            if isinstance(osl, int | float):
                data["osl"] = {"mean": float(osl), "stddev": data["osl_stddev"]}
            data["osl_stddev"] = None

        return data

    probability: Annotated[
        float,
        Field(
            ge=0.0,
            le=100.0,
            description="Relative probability weight for this distribution bucket (0-100). "
            "Weights are normalized across all entries. "
            "Example: probability=40 means 40%% of requests use this ISL/OSL.",
        ),
    ]

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"isl": 128, "osl": 64, "probability": 40},
                {
                    "isl": 512,
                    "isl_stddev": 50,
                    "osl": 256,
                    "osl_stddev": 25,
                    "probability": 35,
                },
            ]
        },
    )


def validate_probability_distribution(
    entries: list[SequenceDistributionEntry],
) -> list[SequenceDistributionEntry]:
    """
    Validate that a probability distribution sums to approximately 100.

    Args:
        entries: List of distribution entries with probability weights.

    Returns:
        The validated entries (unchanged).

    Raises:
        ValueError: If probabilities don't sum to approximately 100.
    """
    total = sum(entry.probability for entry in entries)
    if not (99.0 <= total <= 101.0):  # Allow small floating point variance
        raise ValueError(
            f"Sequence distribution probabilities must sum to ~100, got {total}. "
            f"Individual probabilities: {[e.probability for e in entries]}"
        )
    return entries
