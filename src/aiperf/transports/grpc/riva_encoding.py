# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Riva audio encoding map for ASR and TTS serializers."""

from __future__ import annotations

from aiperf.transports.grpc.proto.riva import riva_audio_pb2

ENCODING_MAP: dict[str, int] = {
    "LINEAR_PCM": riva_audio_pb2.LINEAR_PCM,
    "FLAC": riva_audio_pb2.FLAC,
    "MULAW": riva_audio_pb2.MULAW,
    "OGGOPUS": riva_audio_pb2.OGGOPUS,
    "ALAW": riva_audio_pb2.ALAW,
}
