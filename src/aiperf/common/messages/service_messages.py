# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

from pydantic import Field

from aiperf.common.enums import LifecycleState, MessageType
from aiperf.common.memory_tracker import MemoryPhase
from aiperf.common.messages.base_messages import Message
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.types import MessageTypeT, ServiceTypeT


class BaseServiceMessage(Message):
    """Base message that is sent from a service. Requires a service_id field to specify
    the service that sent the message."""

    service_id: str = Field(
        ...,
        description="ID of the service sending the message",
    )


class BaseStatusMessage(BaseServiceMessage):
    """Base message containing status data.
    This message is sent by a service to the system controller to report its status.
    """

    # override request_ns to be auto-filled if not provided
    request_ns: int | None = Field(
        default_factory=time.time_ns,
        description="Timestamp of the request",
    )
    state: LifecycleState = Field(
        ...,
        description="Current state of the service",
    )
    service_type: ServiceTypeT = Field(
        ...,
        description="Type of service",
    )


class StatusMessage(BaseStatusMessage):
    """Message containing status data.
    This message is sent by a service to the system controller to report its status.
    """

    message_type: MessageTypeT = MessageType.STATUS


class RegistrationMessage(BaseStatusMessage):
    """Message containing registration data.
    This message is sent by a service to the system controller to register itself.
    """

    message_type: MessageTypeT = MessageType.REGISTRATION


class HeartbeatMessage(BaseStatusMessage):
    """Message containing heartbeat data.
    This message is sent by a service to the system controller to indicate that it is
    still running.
    """

    message_type: MessageTypeT = MessageType.HEARTBEAT


class MemoryReportMessage(BaseServiceMessage):
    """Self-reported memory snapshot from a service process.

    Each service reads its own memory at startup and shutdown and publishes this
    message so the SystemController can aggregate accurate memory data without
    needing cross-process reads (which fail when processes have already exited).
    """

    message_type: MessageTypeT = MessageType.MEMORY_REPORT

    pid: int = Field(description="Process ID that measured its own memory")
    service_type: ServiceTypeT = Field(description="Type of service reporting")
    phase: MemoryPhase = Field(description="Lifecycle phase when memory was captured")
    pss_bytes: int = Field(description="PSS (Proportional Set Size) in bytes")
    rss_bytes: int | None = Field(
        default=None, description="RSS (Resident Set Size) in bytes"
    )
    uss_bytes: int | None = Field(
        default=None, description="USS (Unique Set Size) in bytes"
    )
    shared_bytes: int | None = Field(default=None, description="Shared memory in bytes")


class ConnectionProbeMessage(BaseServiceMessage):
    """Self-echo message that mitigates ZMQ's "slow joiner" problem.

    Each service publishes a probe and subscribes to the connection_probe topic,
    filtering by its own service_id in the callback. A successful round-trip
    through the XPUB/XSUB proxy proves that subscriptions are active and the
    service will not miss broadcast messages. See MessageBusClientMixin._run_connection_probes."""

    message_type: MessageTypeT = MessageType.CONNECTION_PROBE


class BaseServiceErrorMessage(BaseServiceMessage):
    """Base message containing error data."""

    message_type: MessageTypeT = MessageType.SERVICE_ERROR

    error: ErrorDetails = Field(..., description="Error information")
