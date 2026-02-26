# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ttft_ns field on CreditReturn and CreditContext."""

import msgspec

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit, CreditContext


class TestCreditContextTtftNs:
    def test_default_none(self):
        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv-1",
            x_correlation_id="xcorr-1",
            turn_index=0,
            num_turns=1,
            issued_at_ns=0,
        )
        ctx = CreditContext(credit=credit, drop_perf_ns=0)
        assert ctx.ttft_ns is None

    def test_set_ttft_ns(self):
        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv-1",
            x_correlation_id="xcorr-1",
            turn_index=0,
            num_turns=1,
            issued_at_ns=0,
        )
        ctx = CreditContext(credit=credit, drop_perf_ns=0)
        ctx.ttft_ns = 1_500_000_000
        assert ctx.ttft_ns == 1_500_000_000


class TestCreditReturnTtftNs:
    def _make_credit(self) -> Credit:
        return Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv-1",
            x_correlation_id="xcorr-1",
            turn_index=0,
            num_turns=1,
            issued_at_ns=0,
        )

    def test_default_none(self):
        cr = CreditReturn(credit=self._make_credit())
        assert cr.ttft_ns is None

    def test_with_ttft_ns(self):
        cr = CreditReturn(credit=self._make_credit(), ttft_ns=500_000_000)
        assert cr.ttft_ns == 500_000_000

    def test_serialization_roundtrip_with_ttft(self):
        """Verify ttft_ns survives msgspec encode/decode."""
        cr = CreditReturn(
            credit=self._make_credit(),
            first_token_sent=True,
            ttft_ns=1_234_567_890,
        )
        encoded = msgspec.msgpack.encode(cr)
        decoded = msgspec.msgpack.decode(encoded, type=CreditReturn)
        assert decoded.ttft_ns == 1_234_567_890
        assert decoded.first_token_sent is True

    def test_serialization_roundtrip_without_ttft(self):
        """Verify backward compat: ttft_ns=None is omitted from wire format."""
        cr = CreditReturn(credit=self._make_credit())
        encoded = msgspec.msgpack.encode(cr)
        decoded = msgspec.msgpack.decode(encoded, type=CreditReturn)
        assert decoded.ttft_ns is None
        assert decoded.cancelled is False
