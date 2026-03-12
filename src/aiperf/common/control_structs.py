# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native msgspec structs for the DEALER/ROUTER control channel.

All over-the-wire structs use tag_field="t" for efficient polymorphic decoding via tagged unions.
Tag values are short strings for minimal wire overhead.

Service -> Controller (ControllerBoundMessage):
    Registration ("reg")       - service registration / connection probe, expects RegistrationAck back
    Heartbeat ("hb")           - periodic heartbeat, fire-and-forget
    StatusUpdate ("su")        - state change notification, fire-and-forget
    MemoryReport ("mr")        - self-reported memory snapshot, fire-and-forget
    TelemetryStatus ("ts")     - telemetry availability from TelemetryManager, fire-and-forget
    ServerMetricsStatus ("sm") - server metrics availability from ServerMetricsManager, fire-and-forget

Bidirectional (both unions):
    Command ("cmd")   - command request (controller->service or service->controller)
    CommandAck ("ca") - acknowledged, no data
    CommandOk ("co")  - success with optional payload
    CommandErr ("ce") - failure with error message

Controller -> Service (ServiceBoundMessage):
    RegistrationAck ("ack") - response to Registration
"""

from typing import TypeAlias

from msgspec import Struct

# ---------------------------------------------------------------------------
# Service -> Controller: status & telemetry
# ---------------------------------------------------------------------------


class Registration(Struct, frozen=True, kw_only=True, tag_field="t", tag="reg"):
    """Service registration / connection probe. Expects RegistrationAck back.

    In Kubernetes mode, services populate pod_name and pod_index from their
    environment for controller visibility. WorkerPodManagers additionally set
    num_workers and num_record_processors so the controller knows how many
    child services to expect from each pod.
    """

    sid: str
    rid: str
    stype: str
    state: str
    pod_name: str | None = None
    pod_index: str | None = None
    num_workers: int | None = None
    num_record_processors: int | None = None


class Heartbeat(Struct, frozen=True, kw_only=True, tag_field="t", tag="hb"):
    """Periodic heartbeat (fire-and-forget)."""

    sid: str
    stype: str
    state: str


class StatusUpdate(Struct, frozen=True, kw_only=True, tag_field="t", tag="su"):
    """State change notification (fire-and-forget)."""

    sid: str
    stype: str
    state: str


class MemoryReport(Struct, frozen=True, kw_only=True, tag_field="t", tag="mr"):
    """Self-reported memory snapshot (fire-and-forget)."""

    sid: str
    stype: str
    pid: int
    phase: str
    pss_bytes: int
    rss_bytes: int | None = None
    uss_bytes: int | None = None
    shared_bytes: int | None = None


class TelemetryStatus(Struct, frozen=True, kw_only=True, tag_field="t", tag="ts"):
    """Telemetry availability status from TelemetryManager (fire-and-forget)."""

    sid: str
    enabled: bool
    reason: str | None = None
    endpoints_configured: tuple[str, ...] = ()
    endpoints_reachable: tuple[str, ...] = ()


class ServerMetricsStatus(Struct, frozen=True, kw_only=True, tag_field="t", tag="sm"):
    """Server metrics availability status from ServerMetricsManager (fire-and-forget)."""

    sid: str
    enabled: bool
    reason: str | None = None
    endpoints_configured: tuple[str, ...] = ()
    endpoints_reachable: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Bidirectional: command request-reply
# ---------------------------------------------------------------------------


class Command(Struct, frozen=True, kw_only=True, tag_field="t", tag="cmd"):
    """Command request. Sent in either direction (controller->service or service->controller).

    The ``cmd`` field is a CommandType string value. The ``payload`` carries
    command-specific data encoded with ``orjson.dumps``.
    """

    cid: str
    cmd: str
    payload: bytes = b""


class CommandAck(Struct, frozen=True, kw_only=True, tag_field="t", tag="ca"):
    """Command acknowledged, no result data."""

    cid: str
    sid: str = ""


class CommandOk(Struct, frozen=True, kw_only=True, tag_field="t", tag="co"):
    """Command succeeded with optional result payload (orjson-encoded)."""

    cid: str
    sid: str = ""
    payload: bytes = b""


class CommandErr(Struct, frozen=True, kw_only=True, tag_field="t", tag="ce"):
    """Command failed."""

    cid: str
    sid: str = ""
    error: str = ""
    traceback: str = ""


CommandResponse: TypeAlias = CommandAck | CommandOk | CommandErr


# ---------------------------------------------------------------------------
# Controller -> Service only
# ---------------------------------------------------------------------------


class RegistrationAck(Struct, frozen=True, kw_only=True, tag_field="t", tag="ack"):
    """Acknowledgement of a Registration."""

    rid: str


# ---------------------------------------------------------------------------
# Union types for polymorphic decoding
# ---------------------------------------------------------------------------

ControllerBoundMessage: TypeAlias = (
    Registration
    | Heartbeat
    | StatusUpdate
    | MemoryReport
    | TelemetryStatus
    | ServerMetricsStatus
    | Command
    | CommandAck
    | CommandOk
    | CommandErr
)

ServiceBoundMessage: TypeAlias = (
    RegistrationAck | Command | CommandAck | CommandOk | CommandErr
)
