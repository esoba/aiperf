# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import (
    LifecycleState,
    ServiceRegistrationStatus,
)
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.types import ServiceTypeT


class ServiceRunInfo(AIPerfBaseModel):
    """Base model for tracking service run information."""

    service_type: ServiceTypeT = Field(
        ...,
        description="The type of service",
    )
    registration_status: ServiceRegistrationStatus = Field(
        ...,
        description="The registration status of the service",
    )
    service_id: str = Field(
        ...,
        description="The ID of the service",
    )
    first_seen_ns: int | None = Field(
        default=None,
        description="The first time the service was seen (nanoseconds)",
    )
    last_seen_ns: int | None = Field(
        default=None,
        description="The last time the service was seen (nanoseconds)",
    )
    state: LifecycleState = Field(
        default=LifecycleState.CREATED,
        description="The current state of the service",
    )
    pod_name: str | None = Field(
        default=None,
        description="Kubernetes pod name hosting this service",
    )
    pod_index: str | None = Field(
        default=None,
        description="Kubernetes pod index from JobSet job-index label",
    )
