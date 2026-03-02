<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Synthetic Coding Sessions

Generate multi-turn agentic coding workloads with realistic context growth, KV cache structure, and subagent hierarchies -- no trace files required.

## Overview

The `coding_session` dataset composer generates synthetic multi-turn coding sessions that replicate the request patterns of agentic coding tools. Each session starts with an initial context (system prompt, tool definitions, project files), then grows monotonically as the agent reads files, runs commands, and edits code. Sessions retire when they reach a token ceiling, just like real coding agents that hit context limits.

Use synthetic coding sessions when you need to:

- **Stress-test KV cache behavior** under agentic workloads without collecting real traces
- **Benchmark prefix caching** with realistic shared/session-stable/growing cache layers
- **Evaluate context scaling** as sessions grow from 50K to 200K+ tokens
- **Test subagent hierarchies** where parent agents spawn child agents with independent contexts
- **Reproduce production patterns** with configurable lognormal distributions for context growth, generation length, and turn counts

The composer is registered as a plugin in `plugins.yaml`:

```yaml
coding_session:
  class: aiperf.dataset.composer.coding_session:CodingSessionComposer
  description: |
    Synthetic coding session composer that generates multi-turn sessions with
    lognormal distributions for context growth, initial prefix, and generation
    length. Designed for adaptive_scale timing mode.
```

---

## Quick Start

Enable coding sessions with a single flag. AIPerf auto-sets `adaptive_scale` mode:

```bash
aiperf profile \
    --model your-model-name \
    --endpoint-type chat \
    --coding-session \
    --streaming \
    --url localhost:8000 \
    --concurrency 10
```

This generates 200 sessions (the default) with contexts growing from ~54K to 215K tokens, using mixed programming languages and a 90/10 tool-result-to-text ratio.

---

## Session Structure

Each coding session is a `Conversation` with sequentially growing turns:

```
Turn 0:  system_prompt (8,500 tokens) + initial_prefix (~54K-67K tokens)
Turn 1:  cumulative_tokens += new_tokens (~2,100-4,500 tokens)
Turn 2:  cumulative_tokens += new_tokens
  ...    (context grows monotonically)
Turn N:  cumulative_tokens reaches max_prompt_tokens (215,000)
         -> session retires
```

**Turn content** is drawn from language-specific token pools built from template-based code, bash output, JSON responses, error tracebacks, git diffs, CI output, configs, markdown docs, and test output. The `tool_result_ratio` controls the split: by default, 90% of turns carry tool-result content (code, command output) and 10% carry text content (user prompts).

**Generation length** (output tokens per turn) is sampled independently from its own lognormal distribution, defaulting to mean=600, median=350.

---

## Three-Layer Cache Model

Hash IDs on each turn decompose into three cache layers that model real prompt-caching behavior:

```
hash_ids = [ L1 blocks | L2 blocks | L3 blocks | thinking blocks ]
```

| Layer | What it models | Scope | Default tokens |
|-------|---------------|-------|----------------|
| **L1** | Tool definitions + system prompt | Deterministic `range(0, N)`, shared across **all** sessions | 32,000 |
| **L2** | CLAUDE.md + skills + project context | Random per session, **stable across turns** within a session | 1,500 |
| **L3** | Conversation history | Random, **grows each turn** as context accumulates | Remainder |

L1 blocks use deterministic sequential IDs (`0, 1, 2, ...`), so the server's prefix cache recognizes them as shared across sessions. L2 blocks are randomly generated once per session and remain constant across turns (unless a restart or compression event invalidates them). L3 blocks grow with every turn as new tool results and user messages accumulate.

### Cache invalidation events

Three events regenerate L2+L3 hash IDs, modeling cache misses:

1. **Session restart** (`--coding-session-restart-probability`): Simulates a `--continue` restart. L1 is preserved, but L2 and L3 get new random IDs.
2. **Context compression** (`--coding-session-compression-threshold`): When cumulative tokens reach the threshold fraction of `max_prompt_tokens`, L3 is compressed to `compression_ratio` of its block count. L2 is regenerated.
3. **Thinking block strip** (`--coding-session-thinking-strip-probability`): When a text turn follows tool-result turns that accumulated thinking blocks, thinking blocks are stripped and L2+L3 are regenerated.

---

## CLI Options Reference

All options use the `--coding-session-*` prefix. Enable the feature with `--coding-session`.

### Core Session Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session` | `false` | Enable synthetic coding session generation. Mutually exclusive with `--input-file` and `--public-dataset`. |
| `--coding-session-num-sessions` | `200` | Number of sessions to generate. |
| `--coding-session-system-prompt-tokens` | `8500` | Token count for the system prompt prefix in each session. |
| `--coding-session-max-prompt-tokens` | `215000` | Maximum prompt tokens before a session retires. |
| `--coding-session-block-size` | `64` | KV cache block size in tokens for hash ID generation. |
| `--coding-session-language` | `mixed` | Programming language for content. One of: `python`, `go`, `rust`, `typescript`, `mixed`. |

### Context Growth Distribution

New tokens per turn follow a lognormal distribution parameterized by mean and median:

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-new-tokens-mean` | `4500` | Mean of lognormal distribution for new tokens per turn. |
| `--coding-session-new-tokens-median` | `2100` | Median of lognormal distribution for new tokens per turn. |
| `--coding-session-initial-prefix-mean` | `67000` | Mean of lognormal distribution for initial prefix tokens. |
| `--coding-session-initial-prefix-median` | `54000` | Median of lognormal distribution for initial prefix tokens. |

### Output Generation Length

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-generation-length-mean` | `600` | Mean of lognormal distribution for output generation length. |
| `--coding-session-generation-length-median` | `350` | Median of lognormal distribution for output generation length. |

### Turn Count Limits

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-max-turns-mean` | `0` | Mean session turn count (lognormal). `0` disables turn-count limiting -- sessions grow until the token ceiling only. |
| `--coding-session-max-turns-median` | `0` | Median session turn count (lognormal). `0` disables. |

### Content Type

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-tool-result-ratio` | `0.9` | Probability a turn uses tool-result content vs text content. Real traces show ~90% tool-result by token count. |

### Cache Layer Sizes

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-l1-tokens` | `32000` | L1 (tools+system) tokens. Deterministic hash IDs shared across all sessions. `0` disables. |
| `--coding-session-l2-tokens` | `1500` | L2 (CLAUDE.md+skills) tokens. Random per session, stable across turns. `0` disables. |

### Compression

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-compression-threshold` | `0.85` | Fraction of `max_prompt_tokens` that triggers context compression. |
| `--coding-session-compression-ratio` | `0.3` | Fraction of L3 blocks retained after compression. |
| `--coding-session-max-compressions` | `3` | Maximum compression events per session. `0` disables compression entirely. |

### Session Restart

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-restart-probability` | `0.0` | Per-turn probability of a `--continue` restart. Preserves L1, regenerates L2+L3 hash IDs. `0.0` disables. |

### Thinking Blocks

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-thinking-tokens-mean` | `0` | Mean thinking tokens per tool-use turn (lognormal). `0` disables. |
| `--coding-session-thinking-tokens-median` | `0` | Median thinking tokens per tool-use turn (lognormal). |
| `--coding-session-thinking-strip-probability` | `1.0` | Probability of stripping thinking blocks at a non-tool-result boundary. Causes L2+L3 hash ID regeneration. |

### Cache TTL

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-cache-ttl-sec` | `3600.0` | Main agent KV cache TTL in seconds for working set eviction. |
| `--coding-session-subagent-cache-ttl-sec` | `300.0` | Subagent KV cache TTL in seconds. |

### Inter-Turn Delays

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-delay-mean-ms` | `0` | Mean inter-turn delay in milliseconds (lognormal). `0` disables delays (back-to-back turns). |
| `--coding-session-delay-median-ms` | `0` | Median inter-turn delay in milliseconds (lognormal). `0` disables. |

### Subagent Spawn Probabilities

Subagent spawning uses a two-level bimodal distribution: first, a per-session coin flip decides whether a session uses subagents at all; then, per-turn probability controls spawning within sessions that opted in.

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-subagent-probability` | `0.15` | Legacy per-turn subagent probability. Superseded by the session+turn pair below. Kept as fallback. |
| `--coding-session-subagent-session-probability` | `0.35` | Probability a session uses subagents at all. First level of the bimodal distribution. |
| `--coding-session-subagent-turn-probability` | `0.25` | Per-turn spawn probability, conditional on the session using subagents. Second level. |
| `--coding-session-subagent-background-probability` | `0.15` | Fraction of spawns that run in background (parent continues without waiting). |

### Subagent Hierarchy

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--coding-session-max-subagent-depth` | `1` | 1-5 | Maximum nesting depth. `1` = children only, `2` = children can spawn grandchildren. |
| `--coding-session-subagent-depth-spawn-decay` | `0.5` | 0.0-1.0 | Spawn probability decay per depth level. At depth `d`, effective probability = `base * decay^d`. |

### Subagent Count

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-subagent-count-mean` | `1.2` | Mean number of children per spawn (Poisson). |
| `--coding-session-subagent-count-max` | `4` | Maximum children per spawn. |

### Subagent Session Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-subagent-turns-mean` | `8` | Mean turn count per child session (lognormal). |
| `--coding-session-subagent-turns-median` | `5` | Median turn count per child session (lognormal). |
| `--coding-session-subagent-system-tokens` | `4000` | System prompt tokens for subagent children. |
| `--coding-session-subagent-new-tokens-mean` | `2500` | Mean new tokens per turn in child sessions. |
| `--coding-session-subagent-new-tokens-median` | `1200` | Median new tokens per turn in child sessions. |
| `--coding-session-subagent-max-prompt-tokens` | `50000` | Maximum prompt tokens for child sessions. |

### Subagent Result Injection

When a subagent completes, its output is injected into the parent as a join turn:

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-subagent-result-tokens-mean` | `3000` | Mean tool-result tokens added to the parent join turn from subagent output. |
| `--coding-session-subagent-result-tokens-median` | `1500` | Median tool-result tokens added to the parent join turn. |

### Subagent Model Override

| Option | Default | Description |
|--------|---------|-------------|
| `--coding-session-subagent-explore-model-name` | `None` | Model name for Explore-type subagents. `None` inherits the parent model. |

---

## Subagent Hierarchy

Parent sessions can spawn subagent children that run as independent conversations with their own context windows, cache layers, and tool subsets. This models real agentic systems where a primary agent delegates tasks (file exploration, planning, code review) to specialized sub-agents.

### Subagent Types

Three built-in subagent type profiles are selected by weighted random choice:

| Type | Weight | System Tokens | Turns (mean/median) | New Tokens (mean/median) | Max Prompt | Tools |
|------|--------|---------------|---------------------|--------------------------|------------|-------|
| **Explore** | 0.50 | 12,000 | 5/4 | 2,000/1,000 | 30,000 | `read_file`, `search_files`, `list_directory`, `run_command` |
| **General** | 0.35 | 20,000 | 10/7 | 3,000/1,500 | 80,000 | `read_file`, `edit_file`, `search_files`, `run_command`, `list_directory`, `write_file`, `find_references`, `get_diagnostics` |
| **Plan** | 0.15 | 15,000 | 4/3 | 2,500/1,200 | 50,000 | `read_file`, `search_files`, `list_directory`, `run_command`, `find_references` |

The type profile determines the child's tool set, system prompt size, turn budget, and maximum context. Explore agents are short-lived read-only scouts. General agents have the full tool set and longer sessions. Plan agents are short deliberation sessions with intermediate context budgets.

### Cache isolation

Subagent children have independent cache namespaces. Their L1 hash IDs start at an offset of `2^30 * depth`, so no blocks overlap with the parent or with children at other depths. This matches real subagent behavior where tool definitions diverge from byte zero.

### Recursive spawning

When `max_subagent_depth > 1`, children can recursively spawn their own subagents. The effective spawn probability decays exponentially:

```
effective_probability = subagent_turn_probability * subagent_depth_spawn_decay ^ depth
```

At the default decay of 0.5, a depth-1 child has half the base spawn probability, a depth-2 grandchild has one quarter, and so on.

---

## Language-Specific Content

The content generator builds per-language token pools from structural templates. When `--coding-session-language` is set to a specific language, all sessions draw from that language's pool. When set to `mixed` (the default), each session randomly selects a language with these weights:

| Language | Weight |
|----------|--------|
| Python | 50% |
| TypeScript | 20% |
| Go | 15% |
| Rust | 15% |

Each pool contains language-specific code templates, file paths, imports, and idioms mixed with language-agnostic content (bash output, JSON, git diffs, tool-use blocks). The content type mix within each pool follows real trace data distribution: ~35% code, ~25% bash output, ~15% JSON, ~10% errors, ~15% other (diffs, CI, configs, docs, tests).

---

## Token Pool Auto-Scaling

The content generator builds pre-tokenized pools and uses window slicing for O(1) prompt generation. The pool size auto-scales based on the configured context sizes:

```
pool_target = max(initial_prefix_mean * 2, new_tokens_mean * 3, 200_000)
pool_scale  = max(1.0, pool_target / 200_000)
```

When `initial_prefix_mean` is large (e.g., 500K for a million-token context window), the pool scales up proportionally so that window slices remain diverse. You can verify scaling by checking that `_pool_scale > 1.0` in debug logs.

---

## Practical Examples

### Minimal test run

Generate 10 short sessions to verify setup:

```bash
aiperf profile \
    --model your-model-name \
    --endpoint-type chat \
    --coding-session \
    --coding-session-num-sessions 10 \
    --coding-session-max-prompt-tokens 10000 \
    --coding-session-initial-prefix-mean 2000 \
    --coding-session-initial-prefix-median 1500 \
    --streaming \
    --url localhost:8000 \
    --concurrency 5
```

### Production-scale KV cache stress test

200 sessions growing to 215K tokens with subagents and compression:

```bash
aiperf profile \
    --model your-model-name \
    --endpoint-type chat \
    --coding-session \
    --coding-session-num-sessions 200 \
    --coding-session-max-prompt-tokens 215000 \
    --coding-session-subagent-session-probability 0.35 \
    --coding-session-subagent-turn-probability 0.25 \
    --coding-session-compression-threshold 0.85 \
    --coding-session-max-compressions 3 \
    --streaming \
    --url localhost:8000 \
    --concurrency 50
```

### Python-only sessions with thinking tokens

Force all sessions to Python and enable extended thinking:

```bash
aiperf profile \
    --model your-model-name \
    --endpoint-type chat \
    --coding-session \
    --coding-session-language python \
    --coding-session-thinking-tokens-mean 500 \
    --coding-session-thinking-tokens-median 300 \
    --coding-session-thinking-strip-probability 1.0 \
    --streaming \
    --url localhost:8000 \
    --concurrency 20
```

### Deep subagent hierarchy

Enable two levels of nesting (children can spawn grandchildren):

```bash
aiperf profile \
    --model your-model-name \
    --endpoint-type chat \
    --coding-session \
    --coding-session-max-subagent-depth 2 \
    --coding-session-subagent-session-probability 0.5 \
    --coding-session-subagent-turn-probability 0.3 \
    --coding-session-subagent-depth-spawn-decay 0.5 \
    --coding-session-subagent-background-probability 0.15 \
    --streaming \
    --url localhost:8000 \
    --concurrency 30
```

### Different model for Explore subagents

Route lightweight file-reading subagents to a smaller model:

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type chat \
    --coding-session \
    --coding-session-subagent-explore-model-name claude-haiku-4-5-20251001 \
    --coding-session-subagent-session-probability 0.4 \
    --streaming \
    --url localhost:8000 \
    --concurrency 20
```

### Simulating session restarts with inter-turn delays

Add realistic user think-time and occasional session restarts:

```bash
aiperf profile \
    --model your-model-name \
    --endpoint-type chat \
    --coding-session \
    --coding-session-restart-probability 0.05 \
    --coding-session-delay-mean-ms 5000 \
    --coding-session-delay-median-ms 3000 \
    --streaming \
    --url localhost:8000 \
    --concurrency 20
```

### Bounded session length

Limit sessions to a turn count instead of (or in addition to) the token ceiling:

```bash
aiperf profile \
    --model your-model-name \
    --endpoint-type chat \
    --coding-session \
    --coding-session-max-turns-mean 20 \
    --coding-session-max-turns-median 15 \
    --streaming \
    --url localhost:8000 \
    --concurrency 20
```

---

## Tips

- **Deterministic output**: Sessions are generated using seeded RNG streams derived from the global `--random-seed`. The same seed produces identical sessions across runs.
- **Auto-enabled adaptive_scale**: When `--coding-session` is enabled, `adaptive_scale` and `adaptive_scale_recycle` are automatically set to `true` unless you override them. This is the intended timing mode.
- **Mutually exclusive**: `--coding-session` cannot be combined with `--input-file` or `--public-dataset`. It is a standalone dataset source.
- **Compression reduces context**: When compression fires, cumulative tokens drop to `(L1 + L2 + compressed_L3) * block_size`. This means the session may grow for additional turns before re-hitting the ceiling, up to `max_compressions` times.
- **Subagent background spawns**: When `subagent_background_probability > 0`, some subagents run in the background -- the parent continues its next turn without waiting. This models fire-and-forget file exploration patterns.
- **Tool definitions**: The composer generates 8 OpenAI-format tool schemas (read_file, edit_file, search_files, run_command, list_directory, write_file, find_references, get_diagnostics). Subagent types receive a filtered subset based on their profile.
- **Output token budget ratio**: The composer respects `--output-token-budget-ratio` (default 0.8) when computing per-turn context deltas. This compensates for model undergeneration, making deltas larger so cumulative context matches trace expectations.
- **Export to trace files**: The composer supports `to_coding_traces()` and `write_traces()` methods for exporting generated sessions to JSON files compatible with `CodingTraceLoader`, enabling offline dataset inspection.
