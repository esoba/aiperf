# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core session synthesis engine.

Generates multi-turn Claude Code sessions using a state machine that models
context growth, mixture delays, and probabilistic resets.
"""

from __future__ import annotations

import math
import uuid

import numpy as np

from aiperf.dataset.claude_code_gen.distributions import (
    sample_lognormal,
    sample_mixture_delay,
)
from aiperf.dataset.claude_code_gen.models import (
    LognormalParams,
    SessionDistributionConfig,
    SessionEndReason,
    SynthesizedSession,
    SynthesizedTurn,
)
from aiperf.dataset.claude_code_gen.prefix_model import PrefixAllocator

OUTPUT_MIN = 30


class SessionSynthesizer:
    """Synthesizes multi-turn sessions from distribution config.

    State machine per session:
        START -> derive initial_context (L1 + L1.5 + sampled L2) -> Turn 0
        TURN_LOOP:
            1. Sample delay (mixture: agentic 70% / human 30%)
            2. Sample new_tokens (lognormal)
            3. input_length = prev_input + prev_output + new_tokens
            4. Check RESET:
               a. input_length >= max_prompt_tokens -> forced retire
               b. P(reset) based on context scaling -> if triggered, end session
            5. Sample output_length (lognormal)
            6. Generate hash_ids (prefix_model)
            7. -> TURN_LOOP
    """

    def __init__(self, config: SessionDistributionConfig, seed: int = 42) -> None:
        self._config = config
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._allocator = PrefixAllocator(config.cache)
        self._session_counter = 0

        # Pre-compute Zipf weights for group assignment
        ng = config.group.num_groups
        weights = np.array(
            [1.0 / (k**config.group.zipf_alpha) for k in range(1, ng + 1)]
        )
        self._group_weights = weights / weights.sum()

        # Fixed prefix size (L1 + L1.5)
        self._fixed_prefix = config.cache.layer1_tokens + config.cache.layer1_5_tokens

        # Pre-compute bias-corrected new_tokens params: shift mu by log(bias)
        # to compensate for right-tail truncation at context limit
        ntp = config.new_tokens_per_turn
        if config.new_tokens_bias != 1.0:
            shifted_mu = ntp.mu + math.log(config.new_tokens_bias)
            self._new_tokens_params = LognormalParams(
                mu=shifted_mu,
                sigma=ntp.sigma,
                mean=ntp.mean * config.new_tokens_bias,
                median=ntp.median * config.new_tokens_bias,
            )
        else:
            self._new_tokens_params = ntp

    @property
    def config(self) -> SessionDistributionConfig:
        return self._config

    @property
    def allocator(self) -> PrefixAllocator:
        return self._allocator

    def _next_session_index(self) -> int:
        idx = self._session_counter
        self._session_counter += 1
        return idx

    def _should_reset(self, input_length: int) -> bool:
        """Check probabilistic reset based on context utilization."""
        cfg = self._config.reset
        ratio = input_length / self._config.max_prompt_tokens
        p = cfg.base_probability * (1.0 + (cfg.context_scaling - 1.0) * ratio)
        return bool(self._rng.random() < p)

    def synthesize_session(
        self, inject_restart: bool = False
    ) -> SynthesizedSession:
        """Generate a single multi-turn session.

        If inject_restart is True, one turn mid-session gets a very long delay
        (10-60 min) modeling a user who pauses and returns later.
        """
        session_index = self._next_session_index()
        rand_bytes = self._rng.bytes(16)
        session_id = f"sess-{uuid.UUID(bytes=rand_bytes).hex[:12]}"
        turns: list[SynthesizedTurn] = []

        # Pick turn for restart pause (if applicable)
        restart_at_turn = int(self._rng.integers(3, 12)) if inject_restart else -1

        # Assign group via Zipf distribution
        group_id = int(
            self._rng.choice(self._config.group.num_groups, p=self._group_weights)
        )

        # Turn 0: derive initial_context = L1 + L1.5 + sampled L2
        l2_tokens = int(
            sample_lognormal(self._config.cache.layer2, self._rng, size=1)[0]
        )
        l2_tokens = max(l2_tokens, 1)
        initial_ctx = self._fixed_prefix + l2_tokens
        initial_ctx = max(initial_ctx, self._config.system_prompt_tokens + 1)
        initial_ctx = min(initial_ctx, self._config.max_prompt_tokens)

        output_len = int(
            sample_lognormal(
                self._config.generation_length,
                self._rng,
                size=1,
                clip_min=OUTPUT_MIN,
            )[0]
        )

        timestamp_ms = 0.0
        hash_ids = self._allocator.turn_hash_ids(
            session_index,
            group_id=group_id,
            input_length=initial_ctx,
            prev_session_ids=None,
        )

        turns.append(
            SynthesizedTurn(
                turn_index=0,
                input_length=initial_ctx,
                output_length=output_len,
                new_tokens=initial_ctx,
                delay_ms=0.0,
                timestamp_ms=timestamp_ms,
                hash_ids=hash_ids,
            )
        )

        prev_input = initial_ctx
        prev_output = output_len

        turn_idx = 1
        end_reason = SessionEndReason.FORCED_RETIRE
        while True:
            # 1. Sample delay (or inject restart pause)
            if turn_idx == restart_at_turn:
                # Restart pause: lognormal ~5-20 min
                restart_params = LognormalParams(mean=600_000, median=300_000)
                delay_ms = float(
                    sample_lognormal(restart_params, self._rng, size=1)[0]
                )
            else:
                delay_ms = float(
                    sample_mixture_delay(
                        self._config.inter_turn_delay, self._rng, size=1
                    )[0]
                )
                # Scale delay down as context fills up (deeper in session = more agentic)
                context_ratio = prev_input / self._config.max_prompt_tokens
                delay_ms *= max(0.2, 1.0 - 0.8 * context_ratio)
            timestamp_ms += delay_ms

            # 2. Sample new tokens (bias-corrected for truncation)
            new_tokens = int(
                sample_lognormal(self._new_tokens_params, self._rng, size=1)[0]
            )
            new_tokens = max(new_tokens, 1)

            # 3. Compute input length
            input_length = prev_input + prev_output + new_tokens

            # 4a. Forced retire if over context limit
            if input_length >= self._config.max_prompt_tokens:
                end_reason = SessionEndReason.FORCED_RETIRE
                break

            # 4b. Probabilistic reset
            if self._should_reset(input_length):
                end_reason = SessionEndReason.PROBABILISTIC_RESET
                break

            # 5. Sample output length
            output_len = int(
                sample_lognormal(
                    self._config.generation_length,
                    self._rng,
                    size=1,
                    clip_min=OUTPUT_MIN,
                )[0]
            )

            # 6. Generate hash_ids (extend previous session ids)
            prev_session = self._allocator.extract_session_ids(turns[-1].hash_ids)
            hash_ids = self._allocator.turn_hash_ids(
                session_index,
                group_id=group_id,
                input_length=input_length,
                prev_session_ids=prev_session,
            )

            turns.append(
                SynthesizedTurn(
                    turn_index=turn_idx,
                    input_length=input_length,
                    output_length=output_len,
                    new_tokens=new_tokens,
                    delay_ms=delay_ms,
                    timestamp_ms=timestamp_ms,
                    hash_ids=hash_ids,
                )
            )

            prev_input = input_length
            prev_output = output_len
            turn_idx += 1

        return SynthesizedSession(
            session_id=session_id,
            group_id=group_id,
            turns=turns,
            end_reason=end_reason,
        )

    def synthesize_sessions(self, num_sessions: int) -> list[SynthesizedSession]:
        """Generate multiple sessions with optional restart pauses.

        Restart probability decreases linearly from restart_fraction to 0
        so early sessions are more likely to pause and resume (modeling
        returning users), while late sessions run without pausing.
        """
        restart_frac = self._config.restart_fraction
        cutoff = 0.75
        sessions: list[SynthesizedSession] = []
        for i in range(num_sessions):
            progress = i / max(1, num_sessions - 1)
            if progress >= cutoff:
                p_restart = 0.0
            else:
                p_restart = restart_frac * (1.0 - progress / cutoff)
            inject = float(self._rng.random()) < p_restart
            sessions.append(self.synthesize_session(inject_restart=inject))
        return sessions
