# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic coding session composer.

Generates multi-turn coding sessions with lognormal distributions for context
growth, initial prefix, and generation length. Each session grows its context
monotonically until reaching max_prompt_tokens, mimicking real agentic coding
patterns (tool calls, edits, test output accumulating in context).

Designed for adaptive_scale timing mode without requiring real trace files.
"""

from __future__ import annotations

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.config.coding_session_config import CodingSessionConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.composer.base import BaseDatasetComposer
from aiperf.dataset.generator.coding_content import CodingContentGenerator

_LANGUAGE_WEIGHTS = {"python": 0.50, "go": 0.15, "rust": 0.15, "typescript": 0.20}
_LANGUAGES = list(_LANGUAGE_WEIGHTS.keys())
_WEIGHTS = list(_LANGUAGE_WEIGHTS.values())


class CodingSessionComposer(BaseDatasetComposer):
    """Generates synthetic multi-turn coding sessions from statistical parameters.

    Each session starts with an initial prefix sampled from a lognormal
    distribution, then grows context by lognormal-sampled deltas each turn
    until the cumulative context reaches max_prompt_tokens.

    Produces Turn objects with input_tokens and hash_ids metadata, compatible
    with AdaptiveScaleStrategy for token budget and working set tracking.
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer, **kwargs):
        super().__init__(config, tokenizer, **kwargs)

        self._coding_config: CodingSessionConfig = config.input.coding_session
        self._output_token_budget_ratio = config.input.output_token_budget_ratio

        self._content_generator = CodingContentGenerator(
            config=config.input.prompt,
            tokenizer=self.prompt_generator.tokenizer,
        )

        self._new_tokens_rng = rng.derive("coding_session.new_tokens")
        self._prefix_rng = rng.derive("coding_session.initial_prefix")
        self._gen_length_rng = rng.derive("coding_session.generation_length")
        self._hash_id_rng = rng.derive("coding_session.hash_ids")
        self._language_rng = rng.derive("coding_session.language")
        self._content_type_rng = rng.derive("coding_session.content_type")

    def create_dataset(self) -> list[Conversation]:
        cfg = self._coding_config
        conversations: list[Conversation] = []

        for session_idx in range(cfg.num_sessions):
            conversation = self._generate_session(session_idx, cfg)
            conversations.append(conversation)

        self._finalize_conversations(conversations)
        self._log_statistics(conversations)
        return conversations

    def _select_session_language(self, cfg: CodingSessionConfig) -> str:
        """Select a language for this session based on config."""
        if cfg.language != "mixed":
            return cfg.language
        return self._language_rng.choice(
            [
                lang
                for lang, w in zip(_LANGUAGES, _WEIGHTS, strict=True)
                for _ in range(int(w * 100))
            ]
        )

    def _select_content_type(self, cfg: CodingSessionConfig) -> str:
        """Select content type for a turn based on tool_result_ratio."""
        return (
            "tool_result"
            if self._content_type_rng.random() < cfg.tool_result_ratio
            else "text"
        )

    def _generate_session(
        self, session_idx: int, cfg: CodingSessionConfig
    ) -> Conversation:
        conversation = Conversation(session_id=f"coding_session_{session_idx:04d}")

        block_size = cfg.block_size
        session_language = self._select_session_language(cfg)

        # Sample initial prefix size
        initial_prefix = self._prefix_rng.sample_lognormal_integer(
            cfg.initial_prefix_mean, cfg.initial_prefix_median
        )
        initial_prefix = min(initial_prefix, cfg.max_prompt_tokens)

        # Turn 0: system_prompt + initial_prefix
        cumulative_tokens = cfg.system_prompt_tokens + initial_prefix
        num_blocks = max(1, cumulative_tokens // block_size)
        hash_ids = self._generate_hash_ids(num_blocks)

        gen_length = self._gen_length_rng.sample_lognormal_integer(
            cfg.generation_length_mean, cfg.generation_length_median
        )

        # Delta for turn 0: the entire initial content beyond system prompt
        delta = initial_prefix
        content_type = self._select_content_type(cfg)
        prompt_text = self._content_generator.generate_language_prompt(
            delta, content_type, session_language
        )
        system_text = self._content_generator.generate_language_prompt(
            cfg.system_prompt_tokens, "tool_result", session_language
        )
        conversation.system_message = system_text

        turn = Turn(
            max_tokens=gen_length,
            input_tokens=cumulative_tokens,
            texts=[Text(name="text", contents=[prompt_text])],
            hash_ids=list(hash_ids),
        )
        self._finalize_turn(turn)
        conversation.turns.append(turn)

        # Subsequent turns: grow context until max_prompt_tokens
        while cumulative_tokens < cfg.max_prompt_tokens:
            new_tokens = self._new_tokens_rng.sample_lognormal_integer(
                cfg.new_tokens_mean, cfg.new_tokens_median
            )
            gen_length = self._gen_length_rng.sample_lognormal_integer(
                cfg.generation_length_mean, cfg.generation_length_median
            )

            # Account for output shortfall compensation
            effective_prev_output = int(gen_length * self._output_token_budget_ratio)
            delta = max(1, new_tokens - effective_prev_output)

            cumulative_tokens += new_tokens
            if cumulative_tokens > cfg.max_prompt_tokens:
                cumulative_tokens = cfg.max_prompt_tokens

            # Grow hash_ids for new blocks
            num_blocks = max(1, cumulative_tokens // block_size)
            new_block_count = num_blocks - len(hash_ids)
            if new_block_count > 0:
                new_ids = self._generate_hash_ids(new_block_count, offset=len(hash_ids))
                hash_ids.extend(new_ids)

            content_type = self._select_content_type(cfg)
            prompt_text = self._content_generator.generate_language_prompt(
                delta, content_type, session_language
            )
            turn = Turn(
                max_tokens=gen_length,
                input_tokens=cumulative_tokens,
                texts=[Text(name="text", contents=[prompt_text])],
                hash_ids=list(hash_ids),
            )
            self._finalize_turn(turn)
            conversation.turns.append(turn)

        return conversation

    def _generate_hash_ids(self, count: int, offset: int = 0) -> list[int]:
        """Generate deterministic hash IDs for KV cache blocks."""
        return [self._hash_id_rng.randint(0, 2**31 - 1) for _ in range(count)]

    def _log_statistics(self, conversations: list[Conversation]) -> None:
        if not conversations:
            return

        turn_counts = [len(c.turns) for c in conversations]
        max_inputs = [
            max((t.input_tokens or 0) for t in c.turns) for c in conversations
        ]

        self.info(
            f"Generated {len(conversations)} coding sessions: "
            f"turns min={min(turn_counts)} mean={sum(turn_counts) / len(turn_counts):.1f} "
            f"max={max(turn_counts)}, "
            f"max_input min={min(max_inputs):,} mean={sum(max_inputs) / len(max_inputs):,.0f} "
            f"max={max(max_inputs):,}"
        )
