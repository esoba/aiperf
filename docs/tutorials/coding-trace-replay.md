<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Coding Trace Replay

Benchmark LLM inference servers by replaying real agentic coding sessions captured in the kv-cache-tester trace format.

## Overview

Coding trace replay loads pre-recorded agentic coding sessions and replays them against an inference server. Each trace captures a complete multi-turn coding session -- the sequence of LLM requests, their token counts, inter-request timing, and KV cache block hashes. AIPerf converts these traces into multi-turn conversations with delta-based prompt sizing, so the server sees realistic context growth and prefix sharing patterns.

### When to Use

- **KV cache benchmarking**: Measure prefix cache hit rates and eviction behavior under realistic workloads
- **Capacity planning**: Find the maximum number of concurrent coding sessions a server can sustain before TTFT degrades
- **Server comparison**: Replay identical traces against different server configurations for apples-to-apples comparison
- **Context window testing**: Validate server behavior as context windows grow from ~20K to 100K+ tokens across a session

### When Not to Use

- For single-request throughput testing, use [Mooncake trace replay](../tutorials/prefix-synthesis.md) or synthetic datasets
- For fixed-rate load testing without multi-turn context, use [request rate modes](../tutorials/request-rate-concurrency.md)

---

## Trace Format

Coding traces use the [kv-cache-tester format](https://github.com/callanjfox/kv-cache-tester) -- a directory of JSON files where each file represents one agentic coding session.

### Trace File Structure

Each JSON file contains a single trace object:

```json
{
  "id": "trace_0001",
  "models": ["claude-sonnet-4-20250514"],
  "block_size": 64,
  "tool_tokens": 12524,
  "system_tokens": 3189,
  "requests": [
    {
      "t": 0.0,
      "type": "s",
      "in": 20064,
      "out": 797,
      "hash_ids": [101, 102, 103, 104, 105],
      "input_types": ["text"],
      "output_types": ["text"],
      "stop": "end_turn"
    },
    {
      "t": 45.2,
      "type": "n",
      "in": 25600,
      "out": 1523,
      "hash_ids": [101, 102, 103, 104, 105, 106, 107],
      "input_types": ["tool_result"],
      "output_types": ["text", "tool_use"],
      "stop": "tool_use"
    }
  ]
}
```

### Trace-Level Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | required | Unique trace identifier |
| `models` | list[string] | `[]` | Model names used in the original session |
| `block_size` | int | `64` | KV cache block size in tokens |
| `tool_tokens` | int | `0` | Estimated token count for tool definitions |
| `system_tokens` | int | `0` | Estimated token count for system prompt |
| `requests` | list[object] | required | Sequence of LLM requests (at least one) |

### Request-Level Fields

| Field | Type | Alias | Default | Description |
|-------|------|-------|---------|-------------|
| `t` | float | -- | required | Relative timestamp in seconds from session start (or parent for subagents) |
| `type` | string | -- | required | Request type: `s` (streaming), `n` (non-streaming), `tool_result` |
| `model` | string | -- | `null` | Model name for this request |
| `in` | int | `input_tokens` | `0` | Total context tokens at this turn (cumulative, not delta) |
| `out` | int | `output_tokens` | `0` | Expected output token count |
| `hash_ids` | list[int] | -- | `[]` | KV cache block hashes for prefix sharing analysis |
| `input_types` | list[string] | -- | `[]` | Input content types (e.g., `text`, `tool_result`) |
| `output_types` | list[string] | -- | `[]` | Output content types (e.g., `text`, `tool_use`) |
| `stop` | string | -- | `null` | Stop reason (e.g., `end_turn`, `tool_use`) |
| `requests` | list[object] | -- | `[]` | Nested subagent requests (recursive structure) |

The `in` field is the **total** context size at that turn, not the incremental delta. AIPerf computes deltas internally (see [Delta-Based Prompt Sizing](#delta-based-prompt-sizing)).

---

## Basic Usage

### 1. Start an Inference Server

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B
```

### 2. Obtain Traces

Download traces from the kv-cache-tester repository:

```bash
git clone https://github.com/callanjfox/kv-cache-tester.git
```

Or point to your own directory of JSON trace files.

### 3. Run Coding Trace Replay

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file kv-cache-tester/traces/ \
    --custom-dataset-type coding_trace \
    --adaptive-scale \
    --benchmark-duration 300
```

This will:
- Load all JSON trace files from the directory
- Flatten nested subagent requests into linear sequences
- Compute delta-based prompts sized to match each turn's token count
- Generate a warm prefix from system/tool token counts (default 50% of max)
- Start with 1 concurrent user and scale up while TTFT stays below 2 seconds
- Run for 5 minutes, reporting metrics at the end

You can also load a single JSON file instead of a directory:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file trace_0001.json \
    --custom-dataset-type coding_trace \
    --adaptive-scale \
    --benchmark-duration 60
```

---

## Configuration Options

### Dataset Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-file` | required | Path to trace directory or single JSON file |
| `--custom-dataset-type coding_trace` | -- | Selects the coding trace loader |
| `--warm-prefix-pct` | `0.5` | Warm prefix size as fraction of max(tool_tokens + system_tokens) |
| `--output-token-budget-ratio` | `0.8` | Expected ratio of actual to requested output tokens |
| `--synthesis-max-isl` | None | Truncate conversations at first request exceeding this ISL |
| `--synthesis-min-requests` | `2` | Skip traces with fewer requests after truncation |

### Adaptive Scale Options

Adaptive scale is the recommended execution mode for coding trace replay. It automatically scales the number of concurrent users based on TTFT headroom.

| Option | Default | Description |
|--------|---------|-------------|
| `--adaptive-scale` | `false` | Enable adaptive user scaling |
| `--adaptive-scale-start-users` | `1` | Initial number of concurrent users |
| `--adaptive-scale-max-users` | `50` | Maximum concurrent users |
| `--adaptive-scale-max-ttft` | `2.0` | TTFT threshold in seconds; scaling stops when exceeded |
| `--adaptive-scale-ttft-metric` | `p95` | TTFT metric for scaling: `p95`, `avg`, or `max` |
| `--adaptive-scale-assessment-period` | `30.0` | Seconds between scaling assessments |
| `--adaptive-scale-max-delay` | `60.0` | Clamp inter-request delays at this value (seconds) |
| `--adaptive-scale-time-scale` | `1.0` | Multiplier for trace delays; `0.5` = 2x faster replay |
| `--adaptive-scale-recycle` | `false` | Recycle completed sessions with new traces |
| `--adaptive-scale-stagger-ms` | `50.0` | Milliseconds between launching new users |
| `--adaptive-scale-formula` | `conservative` | Scaling formula: `conservative`, `aggressive`, or `linear` |
| `--adaptive-scale-enable-rate-limiting` | `true` | Per-session rate limiting with exponential backoff |
| `--adaptive-scale-max-new-tokens-per-period` | `500000` | Token budget per assessment period for new sessions |
| `--adaptive-scale-max-working-set-tokens` | None | Maximum total KV cache working set in tokens |

### Stop Conditions

At least one stop condition is required:

| Option | Description |
|--------|-------------|
| `--benchmark-duration` | Run for this many seconds |
| `--request-count` | Stop after this many total requests |

---

## How the Loader Works

### Delta-Based Prompt Sizing

In kv-cache-tester traces, the `in` field (aliased as `input_tokens`) is the **total** context at that turn -- system prompt, tool definitions, all prior turns, and new content. Since AIPerf's worker accumulates previous user and assistant turns when building HTTP requests, each turn's synthetic prompt only needs to cover the **delta** (new content):

```
Turn 0: delta = max(1, input_tokens - prefix_tokens)
Turn N: delta = max(1, input_tokens - prev_input_tokens - prev_output_tokens * budget_ratio)
```

Where `prefix_tokens` is the warm prefix size (see [Warm Prefix](#warm-prefix)), and `budget_ratio` is `--output-token-budget-ratio` (default `0.8`). The budget ratio compensates for model undergeneration: at 0.8, AIPerf expects the model to produce only 80% of the trace's recorded output tokens, so deltas are inflated by ~20% to keep total context aligned with the trace's intent.

A single base prompt is generated at the maximum delta size across all conversations. Smaller deltas are derived by truncating this base prompt by character ratio, avoiding per-request tokenizer calls.

### Subagent Flattening

Traces can contain nested `requests` arrays representing subagent calls spawned by the parent session. The loader handles these in two ways:

**Leaf children** (nested requests with no further nesting) are flattened inline into the parent's chronological sequence. Their timestamps are converted from parent-relative to absolute: `absolute_t = parent_t + child.t`.

**Subtree children** (nested requests that themselves contain nested requests, i.e., depth > 1) are extracted as independent child conversations linked to the parent via `SubagentSpawnInfo`. This preserves the tree structure for realistic replay where subagent sessions run concurrently with the parent.

After flattening, all requests are sorted by absolute timestamp to produce a chronological sequence with correct inter-request delays.

### Context Reset Detection (Pull-backs)

The loader detects context resets by examining consecutive hash_id sets. When more than 10% of the previous request's hash_ids are removed in the next request, the loader treats this as a context reset and splits the trace into separate conversation segments. Each segment becomes an independent conversation.

### Request Pair Detection

Some traces contain consecutive requests where a streaming request is immediately followed by a non-streaming request with identical hash_ids. This pattern represents the same conversation being re-sent (typically for tool-use confirmation after a streaming response). The loader detects these pairs and assigns `delta = 1` to the repeat request, since no new content is being added.

### Conversation Truncation

When `--synthesis-max-isl` is set, the loader truncates each conversation at the first request whose `input_tokens` exceeds the limit. All requests up to that point are kept; none after. This preserves conversation continuity rather than filtering individual requests.

After truncation, traces with fewer requests than `--synthesis-min-requests` (default: 2) are skipped entirely.

---

## Cache Layers

The loader annotates each turn with L1/L2/L3 cache layer sizes derived from hash_id stability patterns across the conversation:

| Layer | Meaning | How Computed |
|-------|---------|--------------|
| **L1** | Global prefix (system prompt, tools) | Intersection of all hash_id sets across all turns |
| **L2** | Session-stable prefix (CLAUDE.md, skills) | Estimated from `l2_tokens // block_size` in coding session config (default: `1500 // 64 = 23` blocks) |
| **L3** | Turn-specific content | Remainder: `total_blocks - L1 - L2` |

These annotations are used for cache behavior analysis and working set estimation. They do not affect prompt generation.

---

## Warm Prefix

The warm prefix is a shared system message prepended to all conversations. It simulates cross-conversation KV cache sharing that occurs in production when multiple users share tool definitions and system prompts.

**Size calculation**: `warm_prefix_pct * max(tool_tokens + system_tokens)` across all loaded traces.

**Default behavior**: With `--warm-prefix-pct 0.5` (the default) and traces averaging 15,000 combined tool/system tokens, the warm prefix is approximately 7,500 tokens.

**First-turn adjustment**: The first turn's delta is reduced by the prefix token count to keep total context aligned with the trace's intended `input_tokens`.

**Disabling**: Set `--warm-prefix-pct 0.0` to benchmark cold-cache performance with no shared prefix.

---

## Examples

### Find Maximum Sustainable Concurrency

The primary use case: scale up users until TTFT hits the threshold, finding the server's capacity ceiling.

```bash
aiperf profile \
    --model your-model \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file traces/ \
    --custom-dataset-type coding_trace \
    --adaptive-scale \
    --adaptive-scale-max-ttft 2.0 \
    --adaptive-scale-ttft-metric p95 \
    --benchmark-duration 600
```

### Aggressive Scaling for Quick Capacity Finding

Use the aggressive formula with shorter assessment periods to find the ceiling faster.

```bash
aiperf profile \
    --model your-model \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file traces/ \
    --custom-dataset-type coding_trace \
    --adaptive-scale \
    --adaptive-scale-formula aggressive \
    --adaptive-scale-assessment-period 15 \
    --adaptive-scale-max-users 100 \
    --benchmark-duration 300
```

### Accelerated Replay

Compress inter-request delays by 2x to stress-test the server with faster-than-real-time request pacing.

```bash
aiperf profile \
    --model your-model \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file traces/ \
    --custom-dataset-type coding_trace \
    --adaptive-scale \
    --adaptive-scale-time-scale 0.5 \
    --adaptive-scale-max-delay 30 \
    --benchmark-duration 300
```

A `time-scale` of `0.5` halves all inter-request delays, effectively replaying sessions at 2x speed. The `max-delay` of 30 seconds clamps any remaining long pauses.

### Long-Running with Session Recycling

Recycle completed sessions to maintain load over extended benchmarks. When a user finishes their trace, they are assigned a new one.

```bash
aiperf profile \
    --model your-model \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file traces/ \
    --custom-dataset-type coding_trace \
    --adaptive-scale \
    --adaptive-scale-recycle \
    --benchmark-duration 1800
```

### Constrained Context Window

Limit traces to fit a smaller context window, preventing out-of-memory errors on servers with limited KV cache capacity.

```bash
aiperf profile \
    --model your-model \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file traces/ \
    --custom-dataset-type coding_trace \
    --adaptive-scale \
    --synthesis-max-isl 32000 \
    --benchmark-duration 300
```

Conversations are truncated at the first request exceeding 32K input tokens. Traces that fall below `--synthesis-min-requests` (default: 2) after truncation are skipped.

### Cold-Cache Benchmarking

Disable the warm prefix to measure performance without cross-conversation prefix sharing.

```bash
aiperf profile \
    --model your-model \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file traces/ \
    --custom-dataset-type coding_trace \
    --adaptive-scale \
    --warm-prefix-pct 0.0 \
    --benchmark-duration 300
```

### SLO-Based Scaling

Scale up while a goodput ratio (fraction of requests meeting all SLOs) stays above a minimum threshold.

```bash
aiperf profile \
    --model your-model \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file traces/ \
    --custom-dataset-type coding_trace \
    --adaptive-scale \
    --adaptive-scale-slo time_to_first_token:500 request_latency:2000 \
    --adaptive-scale-min-goodput-ratio 0.95 \
    --benchmark-duration 600
```

This scales up while at least 95% of requests have TTFT under 500ms and total latency under 2000ms.

### Adjusted Output Token Budget

If your model consistently underproduces output tokens relative to the trace, lower the budget ratio to inflate prompt deltas further. If your model produces close to the requested token count, raise it toward 1.0.

```bash
aiperf profile \
    --model your-model \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --input-file traces/ \
    --custom-dataset-type coding_trace \
    --adaptive-scale \
    --output-token-budget-ratio 0.6 \
    --benchmark-duration 300
```

At `0.6`, AIPerf expects only 60% of the trace's output tokens, making each turn's delta ~40% larger to compensate.

---

## Tips

- **Start with defaults**: The default configuration (`--adaptive-scale` with default options) works well for most servers. Tune only after an initial run.
- **Sampling strategy**: The coding trace loader uses `sequential` sampling by default. Traces are played in order, wrapping around after all traces are consumed.
- **Reproducibility**: Set `--random-seed 42` for reproducible prompt generation and session assignment across runs.
- **Scaling formula choice**: Use `conservative` (default) for production servers where overshooting is expensive. Use `aggressive` when you want to find the ceiling quickly and do not mind brief periods of elevated TTFT.
- **Token budget control**: If you see TTFT spikes when many users start simultaneously, lower `--adaptive-scale-max-new-tokens-per-period` to throttle how many cache-cold tokens enter the system per assessment period.
- **Working set limits**: Use `--adaptive-scale-max-working-set-tokens` to cap total KV cache usage when your server has a known memory budget.
