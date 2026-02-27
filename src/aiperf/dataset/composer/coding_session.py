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

import math

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.config.coding_session_config import CodingSessionConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.models.dataset_models import SubagentSpawnInfo
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

        pool_target = max(
            self._coding_config.initial_prefix_mean * 2,
            self._coding_config.new_tokens_mean * 3,
            200_000,
        )
        self._content_generator = CodingContentGenerator(
            config=config.input.prompt,
            tokenizer=self.prompt_generator.tokenizer,
            pool_tokens_target=pool_target,
        )

        self._new_tokens_rng = rng.derive("coding_session.new_tokens")
        self._prefix_rng = rng.derive("coding_session.initial_prefix")
        self._gen_length_rng = rng.derive("coding_session.generation_length")
        self._hash_id_rng = rng.derive("coding_session.hash_ids")
        self._language_rng = rng.derive("coding_session.language")
        self._content_type_rng = rng.derive("coding_session.content_type")
        self._parallel_rng = rng.derive("coding_session.parallel")
        self._branch_tokens_rng = rng.derive("coding_session.branch_tokens")
        self._subagent_rng = rng.derive("coding_session.subagent")
        self._subagent_hash_rng = rng.derive("coding_session.subagent_hash")
        self._subagent_tokens_rng = rng.derive("coding_session.subagent_tokens")
        self._subagent_gen_rng = rng.derive("coding_session.subagent_gen")
        self._subagent_turns_rng = rng.derive("coding_session.subagent_turns")

    def create_dataset(self) -> list[Conversation]:
        cfg = self._coding_config
        conversations: list[Conversation] = []
        child_conversations: list[Conversation] = []

        for session_idx in range(cfg.num_sessions):
            conversation, children = self._generate_session(session_idx, cfg)
            conversations.append(conversation)
            child_conversations.extend(children)

        all_conversations = conversations + child_conversations
        self._finalize_conversations(all_conversations)
        self._log_statistics(conversations)
        if child_conversations:
            self.info(
                f"Generated {len(child_conversations)} subagent child conversations"
            )
        return all_conversations

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
    ) -> tuple[Conversation, list[Conversation]]:
        session_id = f"coding_session_{session_idx:04d}"
        conversation = Conversation(session_id=session_id)
        child_conversations: list[Conversation] = []

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

        parallel_group_counter = 0
        subagent_spawn_counter = 0

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

            if cumulative_tokens >= cfg.max_prompt_tokens:
                break

            # Subagent spawn (checked first, mutually exclusive with parallel)
            if (
                cfg.subagent_probability > 0
                and self._subagent_rng.random() < cfg.subagent_probability
            ):
                spawn_id = f"s{subagent_spawn_counter}"
                subagent_spawn_counter += 1

                num_children = self._sample_subagent_count(cfg)
                child_conv_ids: list[str] = []

                for child_idx in range(num_children):
                    child_conv = self._generate_subagent_child(
                        session_id, spawn_id, child_idx, cfg, session_language
                    )
                    child_conversations.append(child_conv)
                    child_conv_ids.append(child_conv.session_id)

                # Join turn after subagent spawn
                join_delta = self._new_tokens_rng.sample_lognormal_integer(
                    cfg.new_tokens_mean, cfg.new_tokens_median
                )
                cumulative_tokens += join_delta
                if cumulative_tokens > cfg.max_prompt_tokens:
                    cumulative_tokens = cfg.max_prompt_tokens

                num_blocks = max(1, cumulative_tokens // block_size)
                new_block_count = num_blocks - len(hash_ids)
                if new_block_count > 0:
                    new_ids = self._generate_hash_ids(
                        new_block_count, offset=len(hash_ids)
                    )
                    hash_ids.extend(new_ids)

                join_turn_index = len(conversation.turns)
                join_gen = self._gen_length_rng.sample_lognormal_integer(
                    cfg.generation_length_mean, cfg.generation_length_median
                )
                join_content_type = self._select_content_type(cfg)
                join_prompt = self._content_generator.generate_language_prompt(
                    max(1, join_delta), join_content_type, session_language
                )
                join_turn = Turn(
                    max_tokens=join_gen,
                    input_tokens=cumulative_tokens,
                    texts=[Text(name="text", contents=[join_prompt])],
                    hash_ids=list(hash_ids),
                    subagent_spawn_id=spawn_id,
                )
                self._finalize_turn(join_turn)
                conversation.turns.append(join_turn)

                conversation.subagent_spawns.append(
                    SubagentSpawnInfo(
                        spawn_id=spawn_id,
                        child_conversation_ids=child_conv_ids,
                        join_turn_index=join_turn_index,
                    )
                )
                continue

            # Parallel fan-out: after each sequential turn, potentially spawn branches
            if (
                cfg.parallel_probability > 0
                and self._parallel_rng.random() < cfg.parallel_probability
            ):
                group_id = f"g{parallel_group_counter}"
                parallel_group_counter += 1

                fan_out = self._sample_fan_out(cfg)
                branch_token_sum = 0

                for branch_idx in range(fan_out):
                    branch_tokens = self._branch_tokens_rng.sample_lognormal_integer(
                        cfg.parallel_branch_tokens_mean,
                        cfg.parallel_branch_tokens_median,
                    )
                    branch_input = min(
                        cumulative_tokens + branch_tokens, cfg.max_prompt_tokens
                    )

                    # Each branch extends parent hash_ids with a unique tail
                    branch_blocks = max(1, branch_input // block_size)
                    branch_new = branch_blocks - len(hash_ids)
                    branch_hash_ids = list(hash_ids)
                    if branch_new > 0:
                        branch_hash_ids.extend(
                            self._generate_hash_ids(branch_new, offset=len(hash_ids))
                        )

                    branch_gen = self._gen_length_rng.sample_lognormal_integer(
                        cfg.generation_length_mean, cfg.generation_length_median
                    )
                    branch_content_type = self._select_content_type(cfg)
                    branch_prompt = self._content_generator.generate_language_prompt(
                        branch_tokens, branch_content_type, session_language
                    )

                    branch_turn = Turn(
                        max_tokens=branch_gen,
                        input_tokens=branch_input,
                        texts=[Text(name="text", contents=[branch_prompt])],
                        hash_ids=branch_hash_ids,
                        parallel_group=group_id,
                        parallel_branch=branch_idx,
                    )
                    self._finalize_turn(branch_turn)
                    conversation.turns.append(branch_turn)
                    branch_token_sum += branch_tokens

                # Join turn: cumulative grows by sum of branch deltas + join delta
                join_delta = self._new_tokens_rng.sample_lognormal_integer(
                    cfg.new_tokens_mean, cfg.new_tokens_median
                )
                cumulative_tokens += branch_token_sum + join_delta
                if cumulative_tokens > cfg.max_prompt_tokens:
                    cumulative_tokens = cfg.max_prompt_tokens

                num_blocks = max(1, cumulative_tokens // block_size)
                new_block_count = num_blocks - len(hash_ids)
                if new_block_count > 0:
                    new_ids = self._generate_hash_ids(
                        new_block_count, offset=len(hash_ids)
                    )
                    hash_ids.extend(new_ids)

                join_gen = self._gen_length_rng.sample_lognormal_integer(
                    cfg.generation_length_mean, cfg.generation_length_median
                )
                join_content_type = self._select_content_type(cfg)
                join_prompt = self._content_generator.generate_language_prompt(
                    max(1, join_delta), join_content_type, session_language
                )
                join_turn = Turn(
                    max_tokens=join_gen,
                    input_tokens=cumulative_tokens,
                    texts=[Text(name="text", contents=[join_prompt])],
                    hash_ids=list(hash_ids),
                )
                self._finalize_turn(join_turn)
                conversation.turns.append(join_turn)

        return conversation, child_conversations

    def _sample_subagent_count(self, cfg: CodingSessionConfig) -> int:
        """Sample subagent count clamped to [1, subagent_count_max]."""
        lam = cfg.subagent_count_mean
        limit = math.exp(-lam)
        k = 0
        p = 1.0
        while p > limit:
            k += 1
            p *= self._subagent_rng.random()
        return max(1, min(k - 1, cfg.subagent_count_max))

    def _generate_subagent_child(
        self,
        parent_session_id: str,
        spawn_id: str,
        child_idx: int,
        cfg: CodingSessionConfig,
        session_language: str,
    ) -> Conversation:
        """Generate a subagent child conversation with independent hash_ids."""
        child_id = f"{parent_session_id}_{spawn_id}_c{child_idx}"
        child = Conversation(session_id=child_id, is_subagent_child=True)

        block_size = cfg.block_size

        # Sample number of turns for this child
        num_turns = self._subagent_turns_rng.sample_lognormal_integer(
            cfg.subagent_turns_mean, cfg.subagent_turns_median
        )
        num_turns = max(1, num_turns)

        # Fresh hash_ids for the child (independent KV cache)
        cumulative_tokens = cfg.subagent_system_tokens
        initial_tokens = self._subagent_tokens_rng.sample_lognormal_integer(
            cfg.subagent_new_tokens_mean, cfg.subagent_new_tokens_median
        )
        cumulative_tokens += initial_tokens
        cumulative_tokens = min(cumulative_tokens, cfg.subagent_max_prompt_tokens)

        num_blocks = max(1, cumulative_tokens // block_size)
        hash_ids = [
            self._subagent_hash_rng.randint(0, 2**31 - 1) for _ in range(num_blocks)
        ]

        gen_length = self._subagent_gen_rng.sample_lognormal_integer(
            cfg.generation_length_mean, cfg.generation_length_median
        )

        content_type = self._select_content_type(cfg)
        prompt_text = self._content_generator.generate_language_prompt(
            initial_tokens, content_type, session_language
        )
        system_text = self._content_generator.generate_language_prompt(
            cfg.subagent_system_tokens, "tool_result", session_language
        )
        child.system_message = system_text

        turn = Turn(
            max_tokens=gen_length,
            input_tokens=cumulative_tokens,
            texts=[Text(name="text", contents=[prompt_text])],
            hash_ids=list(hash_ids),
        )
        self._finalize_turn(turn)
        child.turns.append(turn)

        for _ in range(num_turns - 1):
            if cumulative_tokens >= cfg.subagent_max_prompt_tokens:
                break

            new_tokens = self._subagent_tokens_rng.sample_lognormal_integer(
                cfg.subagent_new_tokens_mean, cfg.subagent_new_tokens_median
            )
            gen_length = self._subagent_gen_rng.sample_lognormal_integer(
                cfg.generation_length_mean, cfg.generation_length_median
            )

            effective_prev_output = int(gen_length * self._output_token_budget_ratio)
            delta = max(1, new_tokens - effective_prev_output)

            cumulative_tokens += new_tokens
            if cumulative_tokens > cfg.subagent_max_prompt_tokens:
                cumulative_tokens = cfg.subagent_max_prompt_tokens

            num_blocks = max(1, cumulative_tokens // block_size)
            new_block_count = num_blocks - len(hash_ids)
            if new_block_count > 0:
                new_ids = [
                    self._subagent_hash_rng.randint(0, 2**31 - 1)
                    for _ in range(new_block_count)
                ]
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
            child.turns.append(turn)

        return child

    def _sample_fan_out(self, cfg: CodingSessionConfig) -> int:
        """Sample fan-out count clamped to [2, parallel_fan_out_max].

        Uses Knuth's algorithm for Poisson sampling with the underlying RNG.
        """
        lam = cfg.parallel_fan_out_mean
        limit = math.exp(-lam)
        k = 0
        p = 1.0
        while p > limit:
            k += 1
            p *= self._parallel_rng.random()
        return max(2, min(k - 1, cfg.parallel_fan_out_max))

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
