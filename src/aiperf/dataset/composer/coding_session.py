# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic coding session composer.

Generates multi-turn coding sessions with lognormal distributions for context
growth, initial prefix, and generation length. Each session grows its context
monotonically until reaching max_prompt_tokens, mimicking real agentic coding
patterns (tool calls, edits, test output accumulating in context).

Hash IDs are structured into three cache layers matching real prompt caching:
- L1 (tools+system): deterministic, shared across all sessions
- L2 (CLAUDE.md+skills): random per session, stable across turns
- L3 (conversation history): random, grows each turn

Session restarts, context compression, and thinking block invalidation events
regenerate affected layer IDs to model cache misses observed in production.

Designed for adaptive_scale timing mode without requiring real trace files.
"""

from __future__ import annotations

import math
from pathlib import Path

import orjson

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.config.coding_session_config import CodingSessionConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.models.dataset_models import CacheLayerSizes, SubagentSpawnInfo
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.composer.base import BaseDatasetComposer
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.loader.models import CodingTrace, CodingTraceRequest

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
        self._l2_hash_rng = rng.derive("coding_session.l2_hash")
        self._restart_rng = rng.derive("coding_session.restart")
        self._thinking_rng = rng.derive("coding_session.thinking")
        self._delay_rng = rng.derive("coding_session.delay")
        self._max_turns_rng = rng.derive("coding_session.max_turns")

    def create_dataset(self) -> list[Conversation]:
        cfg = self._coding_config
        conversations: list[Conversation] = []
        child_conversations: list[Conversation] = []

        for session_idx in range(cfg.num_sessions):
            conversation, children = self._generate_session(session_idx, cfg)
            conversations.append(conversation)
            child_conversations.extend(children)

        # Preserve parent/child mapping for to_coding_traces()
        self._last_parents = list(conversations)
        self._last_children = list(child_conversations)

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

    def _sample_delay(self, cfg: CodingSessionConfig) -> int | None:
        """Sample an inter-turn delay in milliseconds from lognormal distribution."""
        if cfg.delay_mean_ms <= 0 or cfg.delay_median_ms <= 0:
            return None
        return self._delay_rng.sample_lognormal_integer(
            cfg.delay_mean_ms, cfg.delay_median_ms
        )

    def _generate_l1_hash_ids(self, cfg: CodingSessionConfig) -> list[int]:
        """Generate deterministic L1 hash IDs shared across all sessions.

        L1 count is capped so L1+L2 don't exceed max_prompt_tokens.
        """
        max_blocks = cfg.max_prompt_tokens // cfg.block_size
        count = min(cfg.l1_tokens // cfg.block_size, max_blocks)
        return list(range(count))

    def _generate_l2_hash_ids(
        self, cfg: CodingSessionConfig, l1_count: int = 0
    ) -> list[int]:
        """Generate random L2 hash IDs (session-stable).

        L2 count is capped so L1+L2 don't exceed max_prompt_tokens.
        """
        max_blocks = cfg.max_prompt_tokens // cfg.block_size
        count = min(cfg.l2_tokens // cfg.block_size, max(0, max_blocks - l1_count))
        return [self._l2_hash_rng.randint(0, 2**31 - 1) for _ in range(count)]

    def _generate_session(
        self, session_idx: int, cfg: CodingSessionConfig
    ) -> tuple[Conversation, list[Conversation]]:
        session_id = f"coding_session_{session_idx:04d}"
        conversation = Conversation(session_id=session_id)
        child_conversations: list[Conversation] = []

        block_size = cfg.block_size
        session_language = self._select_session_language(cfg)

        # L1/L2 layer setup
        l1_ids = self._generate_l1_hash_ids(cfg)
        l2_ids = self._generate_l2_hash_ids(cfg, len(l1_ids))

        # Sample initial prefix size
        initial_prefix = self._prefix_rng.sample_lognormal_integer(
            cfg.initial_prefix_mean, cfg.initial_prefix_median
        )
        initial_prefix = min(initial_prefix, cfg.max_prompt_tokens)

        # Turn 0: system_prompt + initial_prefix
        cumulative_tokens = cfg.system_prompt_tokens + initial_prefix
        total_blocks = max(1, cumulative_tokens // block_size)
        l3_count = max(0, total_blocks - len(l1_ids) - len(l2_ids))
        l3_ids = self._generate_hash_ids(l3_count)
        thinking_ids: list[int] = []

        gen_length = self._gen_length_rng.sample_lognormal_integer(
            cfg.generation_length_mean, cfg.generation_length_median
        )

        delta = initial_prefix
        content_type = self._select_content_type(cfg)
        prompt_text = self._content_generator.generate_language_prompt(
            delta, content_type, session_language
        )
        system_text = self._content_generator.generate_language_prompt(
            cfg.system_prompt_tokens, "tool_result", session_language
        )
        conversation.system_message = system_text

        hash_ids = l1_ids + l2_ids + l3_ids
        layer_sizes = CacheLayerSizes(l1=len(l1_ids), l2=len(l2_ids), l3=len(l3_ids))
        turn = Turn(
            max_tokens=gen_length,
            input_tokens=cumulative_tokens,
            texts=[Text(name="text", contents=[prompt_text])],
            hash_ids=list(hash_ids),
            cache_layer_sizes=layer_sizes,
        )
        self._finalize_turn(turn)
        conversation.turns.append(turn)

        # Sample session turn limit (None = unlimited, token ceiling only)
        max_turns: int | None = None
        if cfg.max_turns_mean > 0 and cfg.max_turns_median > 0:
            max_turns = max(
                1,
                self._max_turns_rng.sample_lognormal_integer(
                    cfg.max_turns_mean, cfg.max_turns_median
                ),
            )

        parallel_group_counter = 0
        subagent_spawn_counter = 0
        compressions_used = 0
        turn_count = 1

        while cumulative_tokens < cfg.max_prompt_tokens:
            turn_count += 1
            if max_turns is not None and turn_count > max_turns:
                break
            # Session restart check
            if (
                cfg.restart_probability > 0
                and self._restart_rng.random() < cfg.restart_probability
            ):
                l2_ids = self._generate_l2_hash_ids(cfg, len(l1_ids))
                l3_ids = [
                    self._hash_id_rng.randint(0, 2**31 - 1) for _ in range(len(l3_ids))
                ]
                thinking_ids = []

            new_tokens = self._new_tokens_rng.sample_lognormal_integer(
                cfg.new_tokens_mean, cfg.new_tokens_median
            )
            gen_length = self._gen_length_rng.sample_lognormal_integer(
                cfg.generation_length_mean, cfg.generation_length_median
            )

            effective_prev_output = int(gen_length * self._output_token_budget_ratio)
            delta = max(1, new_tokens - effective_prev_output)

            cumulative_tokens += new_tokens
            if cumulative_tokens > cfg.max_prompt_tokens:
                cumulative_tokens = cfg.max_prompt_tokens

            # Compression check
            if (
                cfg.max_compressions > 0
                and compressions_used < cfg.max_compressions
                and cumulative_tokens / cfg.max_prompt_tokens
                >= cfg.compression_threshold
            ):
                l2_ids = self._generate_l2_hash_ids(cfg, len(l1_ids))
                compressed_l3 = max(1, int(len(l3_ids) * cfg.compression_ratio))
                l3_ids = [
                    self._hash_id_rng.randint(0, 2**31 - 1)
                    for _ in range(compressed_l3)
                ]
                thinking_ids = []
                cumulative_tokens = min(
                    (len(l1_ids) + len(l2_ids) + compressed_l3) * block_size,
                    cfg.max_prompt_tokens,
                )
                compressions_used += 1
            else:
                # Normal L3 growth
                total_blocks = max(1, cumulative_tokens // block_size)
                current_count = (
                    len(l1_ids) + len(l2_ids) + len(l3_ids) + len(thinking_ids)
                )
                new_l3_count = total_blocks - current_count
                if new_l3_count > 0:
                    l3_ids.extend(
                        self._hash_id_rng.randint(0, 2**31 - 1)
                        for _ in range(new_l3_count)
                    )

            # Thinking blocks
            content_type = self._select_content_type(cfg)
            if cfg.thinking_tokens_mean > 0 and cfg.thinking_tokens_median > 0:
                if content_type == "tool_result":
                    thinking_tokens = self._thinking_rng.sample_lognormal_integer(
                        cfg.thinking_tokens_mean, cfg.thinking_tokens_median
                    )
                    thinking_block_count = max(0, thinking_tokens // block_size)
                    thinking_ids.extend(
                        self._thinking_rng.randint(0, 2**31 - 1)
                        for _ in range(thinking_block_count)
                    )
                elif content_type == "text" and thinking_ids:
                    if self._thinking_rng.random() < cfg.thinking_strip_probability:
                        l2_ids = self._generate_l2_hash_ids(cfg, len(l1_ids))
                        l3_ids = [
                            self._hash_id_rng.randint(0, 2**31 - 1)
                            for _ in range(len(l3_ids))
                        ]
                        thinking_ids = []

            hash_ids = l1_ids + l2_ids + l3_ids + thinking_ids
            layer_sizes = CacheLayerSizes(
                l1=len(l1_ids), l2=len(l2_ids), l3=len(l3_ids)
            )

            prompt_text = self._content_generator.generate_language_prompt(
                delta, content_type, session_language
            )
            turn = Turn(
                max_tokens=gen_length,
                input_tokens=cumulative_tokens,
                texts=[Text(name="text", contents=[prompt_text])],
                hash_ids=list(hash_ids),
                cache_layer_sizes=layer_sizes,
                delay=self._sample_delay(cfg),
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

                join_delta = self._new_tokens_rng.sample_lognormal_integer(
                    cfg.new_tokens_mean, cfg.new_tokens_median
                )
                cumulative_tokens += join_delta
                if cumulative_tokens > cfg.max_prompt_tokens:
                    cumulative_tokens = cfg.max_prompt_tokens

                # Grow L3 for join turn
                total_blocks = max(1, cumulative_tokens // block_size)
                current_count = (
                    len(l1_ids) + len(l2_ids) + len(l3_ids) + len(thinking_ids)
                )
                new_l3_count = total_blocks - current_count
                if new_l3_count > 0:
                    l3_ids.extend(
                        self._hash_id_rng.randint(0, 2**31 - 1)
                        for _ in range(new_l3_count)
                    )

                join_turn_index = len(conversation.turns)
                join_gen = self._gen_length_rng.sample_lognormal_integer(
                    cfg.generation_length_mean, cfg.generation_length_median
                )
                join_content_type = self._select_content_type(cfg)
                join_prompt = self._content_generator.generate_language_prompt(
                    max(1, join_delta), join_content_type, session_language
                )
                join_hash_ids = l1_ids + l2_ids + l3_ids + thinking_ids
                join_layer_sizes = CacheLayerSizes(
                    l1=len(l1_ids), l2=len(l2_ids), l3=len(l3_ids)
                )
                join_turn = Turn(
                    max_tokens=join_gen,
                    input_tokens=cumulative_tokens,
                    texts=[Text(name="text", contents=[join_prompt])],
                    hash_ids=list(join_hash_ids),
                    cache_layer_sizes=join_layer_sizes,
                    subagent_spawn_id=spawn_id,
                    delay=self._sample_delay(cfg),
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

            # Parallel fan-out
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

                    # Each branch extends parent L3 with branch-specific IDs
                    branch_blocks = max(1, branch_input // block_size)
                    parent_count = (
                        len(l1_ids) + len(l2_ids) + len(l3_ids) + len(thinking_ids)
                    )
                    branch_new = branch_blocks - parent_count
                    branch_l3_extra = max(0, branch_new)
                    branch_extra_ids = [
                        self._hash_id_rng.randint(0, 2**31 - 1)
                        for _ in range(branch_l3_extra)
                    ]
                    branch_hash_ids = (
                        l1_ids + l2_ids + l3_ids + thinking_ids + branch_extra_ids
                    )
                    branch_layer_sizes = CacheLayerSizes(
                        l1=len(l1_ids),
                        l2=len(l2_ids),
                        l3=len(l3_ids) + branch_l3_extra,
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
                        hash_ids=list(branch_hash_ids),
                        cache_layer_sizes=branch_layer_sizes,
                        parallel_group=group_id,
                        parallel_branch=branch_idx,
                        delay=self._sample_delay(cfg),
                    )
                    self._finalize_turn(branch_turn)
                    conversation.turns.append(branch_turn)
                    branch_token_sum += branch_tokens

                join_delta = self._new_tokens_rng.sample_lognormal_integer(
                    cfg.new_tokens_mean, cfg.new_tokens_median
                )
                cumulative_tokens += branch_token_sum + join_delta
                if cumulative_tokens > cfg.max_prompt_tokens:
                    cumulative_tokens = cfg.max_prompt_tokens

                total_blocks = max(1, cumulative_tokens // block_size)
                current_count = (
                    len(l1_ids) + len(l2_ids) + len(l3_ids) + len(thinking_ids)
                )
                new_l3_count = total_blocks - current_count
                if new_l3_count > 0:
                    l3_ids.extend(
                        self._hash_id_rng.randint(0, 2**31 - 1)
                        for _ in range(new_l3_count)
                    )

                join_gen = self._gen_length_rng.sample_lognormal_integer(
                    cfg.generation_length_mean, cfg.generation_length_median
                )
                join_content_type = self._select_content_type(cfg)
                join_prompt = self._content_generator.generate_language_prompt(
                    max(1, join_delta), join_content_type, session_language
                )
                join_hash_ids = l1_ids + l2_ids + l3_ids + thinking_ids
                join_layer_sizes = CacheLayerSizes(
                    l1=len(l1_ids), l2=len(l2_ids), l3=len(l3_ids)
                )
                join_turn = Turn(
                    max_tokens=join_gen,
                    input_tokens=cumulative_tokens,
                    texts=[Text(name="text", contents=[join_prompt])],
                    hash_ids=list(join_hash_ids),
                    cache_layer_sizes=join_layer_sizes,
                    delay=self._sample_delay(cfg),
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

    # Subagent L1 IDs start at this offset so zero blocks overlap with parent.
    # Real subagent tool sets diverge from byte 0 (different tool definitions).
    _SUBAGENT_L1_OFFSET = 2**30

    def _generate_subagent_child(
        self,
        parent_session_id: str,
        spawn_id: str,
        child_idx: int,
        cfg: CodingSessionConfig,
        session_language: str,
    ) -> Conversation:
        """Generate a subagent child conversation with independent L1 range."""
        child_id = f"{parent_session_id}_{spawn_id}_c{child_idx}"
        child = Conversation(session_id=child_id, is_subagent_child=True)

        block_size = cfg.block_size

        # L1: independent range (subagent has different tool set, diverges from byte 0)
        max_blocks = cfg.subagent_max_prompt_tokens // block_size
        l1_count = min(
            min(cfg.subagent_system_tokens, cfg.l1_tokens) // block_size, max_blocks
        )
        offset = self._SUBAGENT_L1_OFFSET
        child_l1_ids = list(range(offset, offset + l1_count))

        # L2: own random IDs (capped so L1+L2 don't exceed budget)
        l2_count = min(cfg.l2_tokens // block_size, max(0, max_blocks - l1_count))
        child_l2_ids = [
            self._subagent_hash_rng.randint(0, 2**31 - 1) for _ in range(l2_count)
        ]

        num_turns = self._subagent_turns_rng.sample_lognormal_integer(
            cfg.subagent_turns_mean, cfg.subagent_turns_median
        )
        num_turns = max(1, num_turns)

        cumulative_tokens = cfg.subagent_system_tokens
        initial_tokens = self._subagent_tokens_rng.sample_lognormal_integer(
            cfg.subagent_new_tokens_mean, cfg.subagent_new_tokens_median
        )
        cumulative_tokens += initial_tokens
        cumulative_tokens = min(cumulative_tokens, cfg.subagent_max_prompt_tokens)

        total_blocks = max(1, cumulative_tokens // block_size)
        l3_count = max(0, total_blocks - len(child_l1_ids) - len(child_l2_ids))
        child_l3_ids = [
            self._subagent_hash_rng.randint(0, 2**31 - 1) for _ in range(l3_count)
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

        hash_ids = child_l1_ids + child_l2_ids + child_l3_ids
        layer_sizes = CacheLayerSizes(
            l1=len(child_l1_ids), l2=len(child_l2_ids), l3=len(child_l3_ids)
        )
        turn = Turn(
            max_tokens=gen_length,
            input_tokens=cumulative_tokens,
            texts=[Text(name="text", contents=[prompt_text])],
            hash_ids=list(hash_ids),
            cache_layer_sizes=layer_sizes,
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

            total_blocks = max(1, cumulative_tokens // block_size)
            current_count = len(child_l1_ids) + len(child_l2_ids) + len(child_l3_ids)
            new_l3_count = total_blocks - current_count
            if new_l3_count > 0:
                child_l3_ids.extend(
                    self._subagent_hash_rng.randint(0, 2**31 - 1)
                    for _ in range(new_l3_count)
                )

            content_type = self._select_content_type(cfg)
            prompt_text = self._content_generator.generate_language_prompt(
                delta, content_type, session_language
            )
            hash_ids = child_l1_ids + child_l2_ids + child_l3_ids
            layer_sizes = CacheLayerSizes(
                l1=len(child_l1_ids), l2=len(child_l2_ids), l3=len(child_l3_ids)
            )
            turn = Turn(
                max_tokens=gen_length,
                input_tokens=cumulative_tokens,
                texts=[Text(name="text", contents=[prompt_text])],
                hash_ids=list(hash_ids),
                cache_layer_sizes=layer_sizes,
                delay=self._sample_delay(cfg),
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

    def _generate_hash_ids(self, count: int) -> list[int]:
        """Generate deterministic hash IDs for KV cache blocks."""
        return [self._hash_id_rng.randint(0, 2**31 - 1) for _ in range(count)]

    def to_coding_traces(self) -> list[CodingTrace]:
        """Convert the last generated dataset to CodingTrace objects.

        Must be called after create_dataset(). Reconstructs the nested request
        tree structure that CodingTraceLoader expects, including subagent
        children as nested requests and parallel groups as sibling requests
        under a synthetic parent.

        Returns:
            List of CodingTrace objects, one per parent session.
        """
        parents = getattr(self, "_last_parents", None)
        if parents is None:
            raise RuntimeError("Must call create_dataset() before to_coding_traces()")

        children = self._last_children
        cfg = self._coding_config

        # Index child conversations by session_id for lookup
        child_by_id: dict[str, Conversation] = {c.session_id: c for c in children}

        traces: list[CodingTrace] = []
        for conv in parents:
            requests = self._conversation_to_requests(conv, child_by_id, cfg)
            trace = CodingTrace(
                id=conv.session_id,
                block_size=cfg.block_size,
                system_tokens=cfg.system_prompt_tokens,
                tool_tokens=cfg.l1_tokens,
                requests=requests,
            )
            traces.append(trace)

        return traces

    def _conversation_to_requests(
        self,
        conv: Conversation,
        child_by_id: dict[str, Conversation],
        cfg: CodingSessionConfig,
    ) -> list[CodingTraceRequest]:
        """Convert a parent Conversation's turns into CodingTraceRequest list.

        Handles sequential turns, parallel groups (nested under synthetic parent),
        and subagent spawns (nested child conversation requests).
        """
        # Build spawn_id -> child conversations lookup
        spawn_children: dict[str, list[str]] = {}
        spawn_join_idx: dict[str, int] = {}
        for spawn in conv.subagent_spawns:
            spawn_children[spawn.spawn_id] = spawn.child_conversation_ids
            spawn_join_idx[spawn.spawn_id] = spawn.join_turn_index

        requests: list[CodingTraceRequest] = []
        cumulative_t = 0.0
        i = 0
        turns = conv.turns

        while i < len(turns):
            turn = turns[i]

            # Check for parallel group start
            if turn.parallel_group is not None:
                group_id = turn.parallel_group
                branch_requests: list[CodingTraceRequest] = []

                # Collect all turns in this parallel group
                while i < len(turns) and turns[i].parallel_group == group_id:
                    branch_turn = turns[i]
                    branch_req = self._turn_to_request(
                        branch_turn, 0.0, i == len(turns) - 1
                    )
                    branch_requests.append(branch_req)
                    i += 1

                # Emit a synthetic container (input_tokens=0) so the loader
                # sees a parent with multiple leaf children and detects them
                # as a parallel group. The container is skipped during
                # flattening; only the leaf branch requests enter the flat list.
                container = CodingTraceRequest(
                    t=cumulative_t,
                    type="n",
                    input_tokens=0,
                    output_tokens=0,
                    requests=branch_requests,
                )
                requests.append(container)

                # The join turn (if present) becomes a separate sequential request
                if i < len(turns) and turns[i].parallel_group is None:
                    # Don't advance i here — the join turn will be processed
                    # as a normal sequential turn in the next iteration
                    pass
            else:
                req = self._turn_to_request(turn, cumulative_t, i == len(turns) - 1)

                # Check if this turn is a subagent spawn point
                if turn.subagent_spawn_id is not None:
                    spawn_id = turn.subagent_spawn_id
                    child_ids = spawn_children.get(spawn_id, [])
                    for child_id in child_ids:
                        child_conv = child_by_id.get(child_id)
                        if child_conv:
                            child_reqs = self._child_conversation_to_requests(
                                child_conv
                            )
                            if child_reqs:
                                # Wrap as a container with input_tokens=0 (the
                                # loader skips containers) and nest the actual
                                # child requests inside. This gives depth > 1
                                # so _extract_subagent_subtrees recognizes it.
                                subtree = CodingTraceRequest(
                                    t=0.0,
                                    type="subagent",
                                    input_tokens=0,
                                    output_tokens=0,
                                    requests=child_reqs,
                                )
                                req.requests.append(subtree)

                requests.append(req)
                cumulative_t += (turn.delay or 0.0) / 1000.0
                i += 1

        return requests

    def _child_conversation_to_requests(
        self, conv: Conversation
    ) -> list[CodingTraceRequest]:
        """Convert a child subagent conversation to a flat request list."""
        requests: list[CodingTraceRequest] = []
        cumulative_t = 0.0
        for i, turn in enumerate(conv.turns):
            req = self._turn_to_request(turn, cumulative_t, i == len(conv.turns) - 1)
            requests.append(req)
            cumulative_t += (turn.delay or 0.0) / 1000.0
        return requests

    @staticmethod
    def _turn_to_request(turn: Turn, t: float, is_last: bool) -> CodingTraceRequest:
        """Convert a single Turn to a CodingTraceRequest."""
        # Determine type from content: tool_result turns use "s", text turns use "n"
        req_type = "s"
        if turn.texts and turn.texts[0].contents:
            content = turn.texts[0].contents[0]
            if not any(kw in content for kw in ("def ", "func ", "fn ", "function ")):
                req_type = "n"

        stop = "end_turn" if is_last else "tool_use"

        return CodingTraceRequest(
            t=t,
            type=req_type,
            input_tokens=turn.input_tokens or 0,
            output_tokens=turn.max_tokens or 0,
            hash_ids=list(turn.hash_ids),
            stop=stop,
        )

    @staticmethod
    def write_traces(traces: list[CodingTrace], output_dir: str | Path) -> None:
        """Write CodingTrace objects to a directory of JSON files.

        Each trace is written as {trace.id}.json using orjson with by_alias=True
        to produce the "in"/"out" field aliases expected by CodingTraceLoader.

        Args:
            traces: List of CodingTrace objects to serialize.
            output_dir: Directory to write JSON files into.
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        for trace in traces:
            data = trace.model_dump(by_alias=True)
            file_path = path / f"{trace.id}.json"
            file_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

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
