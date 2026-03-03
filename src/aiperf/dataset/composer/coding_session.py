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
from typing import Any

import orjson

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.config.coding_session_config import (
    DEFAULT_SUBAGENT_PROFILES,
    CodingSessionConfig,
    SubagentTypeProfile,
)
from aiperf.common.enums.enums import SubagentType
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.models.dataset_models import CacheLayerSizes, SubagentSpawnInfo
from aiperf.common.random_generator import RandomGenerator
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

    def __init__(self, config: UserConfig, tokenizer: Tokenizer, **kwargs: Any) -> None:
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

        self._tool_definitions = self._content_generator.generate_tool_definitions()
        self._tool_definitions_by_name: dict[str, dict[str, Any]] = {
            t["function"]["name"]: t for t in self._tool_definitions
        }

        # Build per-type profiles from defaults, applying config overrides
        self._subagent_profiles = self._build_subagent_profiles(self._coding_config)

    def _build_subagent_profiles(
        self, cfg: CodingSessionConfig
    ) -> list[SubagentTypeProfile]:
        """Build subagent type profiles with config overrides applied."""
        profiles: list[SubagentTypeProfile] = []
        for default in DEFAULT_SUBAGENT_PROFILES:
            model_name = default.model_name
            if (
                default.agent_type == SubagentType.EXPLORE
                and cfg.subagent_explore_model_name
            ):
                model_name = cfg.subagent_explore_model_name
            profiles.append(
                SubagentTypeProfile(
                    agent_type=default.agent_type,
                    model_name=model_name,
                    system_tokens=default.system_tokens,
                    turns_mean=default.turns_mean,
                    turns_median=default.turns_median,
                    new_tokens_mean=default.new_tokens_mean,
                    new_tokens_median=default.new_tokens_median,
                    max_prompt_tokens=default.max_prompt_tokens,
                    tool_names=list(default.tool_names),
                    weight=default.weight,
                    cache_ttl_sec=default.cache_ttl_sec,
                )
            )
        return profiles

    def _select_subagent_profile(self) -> SubagentTypeProfile:
        """Select a subagent type profile using weighted random choice."""
        total = sum(p.weight for p in self._subagent_profiles)
        r = self._subagent_rng.random() * total
        cumulative = 0.0
        for profile in self._subagent_profiles:
            cumulative += profile.weight
            if r <= cumulative:
                return profile
        return self._subagent_profiles[-1]

    def _filter_tools_for_profile(
        self, profile: SubagentTypeProfile
    ) -> list[dict[str, Any]]:
        """Filter tool definitions to those allowed for a subagent type."""
        return [
            self._tool_definitions_by_name[name]
            for name in profile.tool_names
            if name in self._tool_definitions_by_name
        ]

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

    def _should_session_use_subagents(self, cfg: CodingSessionConfig) -> bool:
        """Decide at session creation whether this session uses subagents.

        Two-level bimodal distribution: first decide if session uses subagents
        at all, then per-turn probability is checked separately.
        """
        if cfg.subagent_session_probability <= 0:
            return False
        return self._subagent_rng.random() < cfg.subagent_session_probability

    def _should_spawn_subagent(
        self, cfg: CodingSessionConfig, session_uses_subagents: bool
    ) -> bool:
        """Per-turn spawn check, conditional on session using subagents."""
        if not session_uses_subagents:
            return False
        if cfg.subagent_turn_probability <= 0:
            return False
        return self._subagent_rng.random() < cfg.subagent_turn_probability

    def _generate_session(
        self, session_idx: int, cfg: CodingSessionConfig
    ) -> tuple[Conversation, list[Conversation]]:
        session_id = f"coding_session_{session_idx:04d}"
        conversation = Conversation(session_id=session_id, tools=self._tool_definitions)
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

        # Session-level subagent decision (bimodal distribution)
        session_uses_subagents = self._should_session_use_subagents(cfg)

        subagent_spawn_counter = 0
        compressions_used = 0
        turn_count = 1
        context_loss = False

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
                context_loss = True

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
                context_loss = True
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
                        context_loss = True

            hash_ids = l1_ids + l2_ids + l3_ids + thinking_ids
            layer_sizes = CacheLayerSizes(
                l1=len(l1_ids), l2=len(l2_ids), l3=len(l3_ids)
            )

            replaces_history = False
            if context_loss:
                delta = cumulative_tokens
                replaces_history = True
                context_loss = False

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
                replaces_history=replaces_history,
            )
            self._finalize_turn(turn)
            conversation.turns.append(turn)

            if cumulative_tokens >= cfg.max_prompt_tokens:
                break

            # Subagent spawn (two-level probability)
            if self._should_spawn_subagent(cfg, session_uses_subagents):
                spawn_id = f"s{subagent_spawn_counter}"
                subagent_spawn_counter += 1

                profile = self._select_subagent_profile()
                num_children = self._sample_subagent_count(cfg)
                child_conv_ids: list[str] = []

                for child_idx in range(num_children):
                    child_conv, descendants = self._generate_subagent_child(
                        session_id, spawn_id, child_idx, cfg, session_language, profile
                    )
                    child_conversations.append(child_conv)
                    child_conversations.extend(descendants)
                    child_conv_ids.append(child_conv.session_id)

                is_background = (
                    cfg.subagent_background_probability > 0
                    and self._subagent_rng.random()
                    < cfg.subagent_background_probability
                )

                join_turn_index = len(conversation.turns)
                join_turn, cumulative_tokens = self._build_join_turn(
                    is_background=is_background,
                    cumulative_tokens=cumulative_tokens,
                    max_prompt_tokens=cfg.max_prompt_tokens,
                    l1_ids=l1_ids,
                    l2_ids=l2_ids,
                    l3_ids=l3_ids,
                    thinking_ids=thinking_ids,
                    block_size=block_size,
                    cfg=cfg,
                    session_language=session_language,
                    gen_rng=self._gen_length_rng,
                    hash_rng=self._hash_id_rng,
                    spawn_id=spawn_id,
                )
                conversation.turns.append(join_turn)
                conversation.subagent_spawns.append(
                    SubagentSpawnInfo(
                        spawn_id=spawn_id,
                        child_conversation_ids=child_conv_ids,
                        join_turn_index=join_turn_index,
                        is_background=is_background,
                    )
                )
                continue

        return conversation, child_conversations

    def _build_join_turn(
        self,
        *,
        is_background: bool,
        cumulative_tokens: int,
        max_prompt_tokens: int,
        l1_ids: list[int],
        l2_ids: list[int],
        l3_ids: list[int],
        thinking_ids: list[int],
        block_size: int,
        cfg: CodingSessionConfig,
        session_language: str,
        gen_rng: RandomGenerator,
        hash_rng: RandomGenerator,
        spawn_id: str,
        model: str | None = None,
    ) -> tuple[Turn, int]:
        """Build a subagent join turn and return it with updated cumulative tokens.

        For blocking spawns, samples result tokens and grows L3 cache layer.
        For background spawns, uses a minimal single-block prompt.
        Mutates l3_ids in place when new blocks are needed.

        Returns:
            (join_turn, updated_cumulative_tokens)
        """
        if not is_background:
            join_delta = self._subagent_rng.sample_lognormal_integer(
                cfg.subagent_result_tokens_mean,
                cfg.subagent_result_tokens_median,
            )
            cumulative_tokens += join_delta
            if cumulative_tokens > max_prompt_tokens:
                cumulative_tokens = max_prompt_tokens

            total_blocks = max(1, cumulative_tokens // block_size)
            current_count = len(l1_ids) + len(l2_ids) + len(l3_ids) + len(thinking_ids)
            new_l3_count = total_blocks - current_count
            if new_l3_count > 0:
                l3_ids.extend(
                    hash_rng.randint(0, 2**31 - 1) for _ in range(new_l3_count)
                )
            prompt_tokens = join_delta
        else:
            prompt_tokens = block_size

        join_gen = gen_rng.sample_lognormal_integer(
            cfg.generation_length_mean, cfg.generation_length_median
        )
        join_content_type = self._select_content_type(cfg)
        join_prompt = self._content_generator.generate_language_prompt(
            max(1, prompt_tokens), join_content_type, session_language
        )

        join_hash_ids = l1_ids + l2_ids + l3_ids + thinking_ids
        join_layer_sizes = CacheLayerSizes(
            l1=len(l1_ids), l2=len(l2_ids), l3=len(l3_ids)
        )
        join_turn = Turn(
            model=model,
            max_tokens=join_gen,
            input_tokens=cumulative_tokens,
            texts=[Text(name="text", contents=[join_prompt])],
            hash_ids=list(join_hash_ids),
            cache_layer_sizes=join_layer_sizes,
            subagent_spawn_ids=[spawn_id],
            delay=self._sample_delay(cfg),
        )
        self._finalize_turn(join_turn)
        return join_turn, cumulative_tokens

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
        profile: SubagentTypeProfile,
        depth: int = 1,
    ) -> tuple[Conversation, list[Conversation]]:
        """Generate a subagent child conversation using per-type profile parameters.

        Returns:
            Tuple of (child_conversation, descendant_conversations).
            Descendants are grandchildren (and deeper) that should be added
            to the flat child_conversations list.
        """
        child_id = f"{parent_session_id}_{spawn_id}_c{child_idx}"
        child_tools = self._filter_tools_for_profile(profile)
        child = Conversation(
            session_id=child_id,
            agent_depth=depth,
            subagent_type=profile.agent_type,
            parent_conversation_id=parent_session_id,
            tools=child_tools,
        )

        block_size = cfg.block_size

        # L1: independent range (subagent has different tool set, diverges from byte 0)
        max_blocks = profile.max_prompt_tokens // block_size
        l1_count = min(
            min(profile.system_tokens, cfg.l1_tokens) // block_size, max_blocks
        )
        offset = self._SUBAGENT_L1_OFFSET * depth
        child_l1_ids = list(range(offset, offset + l1_count))

        # L2: own random IDs (capped so L1+L2 don't exceed budget)
        l2_count = min(cfg.l2_tokens // block_size, max(0, max_blocks - l1_count))
        child_l2_ids = [
            self._subagent_hash_rng.randint(0, 2**31 - 1) for _ in range(l2_count)
        ]

        num_turns = self._subagent_turns_rng.sample_lognormal_integer(
            profile.turns_mean, profile.turns_median
        )
        num_turns = max(1, num_turns)

        cumulative_tokens = profile.system_tokens
        initial_tokens = self._subagent_tokens_rng.sample_lognormal_integer(
            profile.new_tokens_mean, profile.new_tokens_median
        )
        cumulative_tokens += initial_tokens
        cumulative_tokens = min(cumulative_tokens, profile.max_prompt_tokens)

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
            profile.system_tokens, "tool_result", session_language
        )
        child.system_message = system_text

        hash_ids = child_l1_ids + child_l2_ids + child_l3_ids
        layer_sizes = CacheLayerSizes(
            l1=len(child_l1_ids), l2=len(child_l2_ids), l3=len(child_l3_ids)
        )
        turn = Turn(
            model=profile.model_name,
            max_tokens=gen_length,
            input_tokens=cumulative_tokens,
            texts=[Text(name="text", contents=[prompt_text])],
            hash_ids=list(hash_ids),
            cache_layer_sizes=layer_sizes,
        )
        self._finalize_turn(turn)
        child.turns.append(turn)

        descendant_conversations: list[Conversation] = []
        subagent_spawn_counter = 0

        for _ in range(num_turns - 1):
            if cumulative_tokens >= profile.max_prompt_tokens:
                break

            new_tokens = self._subagent_tokens_rng.sample_lognormal_integer(
                profile.new_tokens_mean, profile.new_tokens_median
            )
            gen_length = self._subagent_gen_rng.sample_lognormal_integer(
                cfg.generation_length_mean, cfg.generation_length_median
            )

            effective_prev_output = int(gen_length * self._output_token_budget_ratio)
            delta = max(1, new_tokens - effective_prev_output)

            cumulative_tokens += new_tokens
            if cumulative_tokens > profile.max_prompt_tokens:
                cumulative_tokens = profile.max_prompt_tokens

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
                model=profile.model_name,
                max_tokens=gen_length,
                input_tokens=cumulative_tokens,
                texts=[Text(name="text", contents=[prompt_text])],
                hash_ids=list(hash_ids),
                cache_layer_sizes=layer_sizes,
                delay=self._sample_delay(cfg),
            )
            self._finalize_turn(turn)
            child.turns.append(turn)

            if cumulative_tokens >= profile.max_prompt_tokens:
                break

            # Recursive subagent spawn (depth < max)
            if depth < cfg.max_subagent_depth:
                decay = cfg.subagent_depth_spawn_decay**depth
                effective_prob = cfg.subagent_turn_probability * decay
                if effective_prob > 0 and self._subagent_rng.random() < effective_prob:
                    gc_spawn_id = f"s{subagent_spawn_counter}"
                    subagent_spawn_counter += 1
                    gc_profile = self._select_subagent_profile()
                    gc_count = self._sample_subagent_count(cfg)
                    gc_conv_ids: list[str] = []

                    for gc_idx in range(gc_count):
                        gc_conv, gc_descendants = self._generate_subagent_child(
                            child_id,
                            gc_spawn_id,
                            gc_idx,
                            cfg,
                            session_language,
                            gc_profile,
                            depth=depth + 1,
                        )
                        descendant_conversations.append(gc_conv)
                        descendant_conversations.extend(gc_descendants)
                        gc_conv_ids.append(gc_conv.session_id)

                    is_background = (
                        cfg.subagent_background_probability > 0
                        and self._subagent_rng.random()
                        < cfg.subagent_background_probability
                    )

                    gc_join_turn_index = len(child.turns)
                    gc_join_turn, cumulative_tokens = self._build_join_turn(
                        is_background=is_background,
                        cumulative_tokens=cumulative_tokens,
                        max_prompt_tokens=profile.max_prompt_tokens,
                        l1_ids=child_l1_ids,
                        l2_ids=child_l2_ids,
                        l3_ids=child_l3_ids,
                        thinking_ids=[],
                        block_size=block_size,
                        cfg=cfg,
                        session_language=session_language,
                        gen_rng=self._subagent_gen_rng,
                        hash_rng=self._subagent_hash_rng,
                        spawn_id=gc_spawn_id,
                        model=profile.model_name,
                    )
                    child.turns.append(gc_join_turn)
                    child.subagent_spawns.append(
                        SubagentSpawnInfo(
                            spawn_id=gc_spawn_id,
                            child_conversation_ids=gc_conv_ids,
                            join_turn_index=gc_join_turn_index,
                            is_background=is_background,
                        )
                    )

        return child, descendant_conversations

    def _generate_hash_ids(self, count: int) -> list[int]:
        """Generate deterministic hash IDs for KV cache blocks."""
        return [self._hash_id_rng.randint(0, 2**31 - 1) for _ in range(count)]

    def to_coding_traces(self) -> list[CodingTrace]:
        """Convert the last generated dataset to CodingTrace objects.

        Must be called after create_dataset(). Reconstructs the nested request
        tree structure that CodingTraceLoader expects, including subagent
        children as nested requests.

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

        Handles sequential turns and subagent spawns (nested child conversation requests).
        """
        # Build spawn_id -> child conversations lookup
        spawn_children: dict[str, list[str]] = {}
        for spawn in conv.subagent_spawns:
            spawn_children[spawn.spawn_id] = spawn.child_conversation_ids

        requests: list[CodingTraceRequest] = []
        cumulative_t = 0.0

        for i, turn in enumerate(conv.turns):
            req = self._turn_to_request(turn, cumulative_t, i == len(conv.turns) - 1)

            # Check if this turn is a subagent spawn point
            for spawn_id in turn.subagent_spawn_ids:
                child_ids = spawn_children.get(spawn_id, [])
                for child_id in child_ids:
                    child_conv = child_by_id.get(child_id)
                    if child_conv:
                        child_reqs = self._child_conversation_to_requests(
                            child_conv, child_by_id
                        )
                        if child_reqs:
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

        return requests

    def _child_conversation_to_requests(
        self,
        conv: Conversation,
        child_by_id: dict[str, Conversation],
    ) -> list[CodingTraceRequest]:
        """Convert a child subagent conversation to a request list with nested spawns."""
        spawn_children: dict[str, list[str]] = {}
        for spawn in conv.subagent_spawns:
            spawn_children[spawn.spawn_id] = spawn.child_conversation_ids

        requests: list[CodingTraceRequest] = []
        cumulative_t = 0.0
        for i, turn in enumerate(conv.turns):
            req = self._turn_to_request(turn, cumulative_t, i == len(conv.turns) - 1)

            for spawn_id in turn.subagent_spawn_ids:
                gc_ids = spawn_children.get(spawn_id, [])
                for gc_id in gc_ids:
                    gc_conv = child_by_id.get(gc_id)
                    if gc_conv:
                        gc_reqs = self._child_conversation_to_requests(
                            gc_conv, child_by_id
                        )
                        if gc_reqs:
                            subtree = CodingTraceRequest(
                                t=0.0,
                                type="subagent",
                                input_tokens=0,
                                output_tokens=0,
                                requests=gc_reqs,
                            )
                            req.requests.append(subtree)

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
