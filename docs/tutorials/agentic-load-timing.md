<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic Load Timing Strategy

Closed-loop load testing for multi-turn agentic conversations with deterministic trajectory assignment, staggered user ramp-up, and automatic cache busting.

## Overview

The `agentic_load` timing strategy simulates realistic multi-user agentic workloads where each user drives a sequence of multi-turn conversations in a closed loop. Unlike open-loop strategies that inject requests at a fixed rate, agentic load waits for each response before sending the next turn, matching the behavior of real agentic systems like coding assistants and tool-calling agents.

### When to Use

- **KV cache benchmarking**: Measure server performance under realistic agentic conversation patterns where each user maintains context across turns
- **Multi-turn tool-calling workloads**: Replay captured agentic trajectories with proper conversation sequencing
- **Closed-loop load testing**: Test server behavior when request arrival depends on response completion
- **Subagent-aware benchmarking**: Workloads where parent conversations spawn child (subagent) sessions

### How It Differs from Other Strategies

| Strategy | Loop Type | Scheduling | Best For |
|----------|-----------|------------|----------|
| `request_rate` | Open | Fixed rate (constant, poisson, gamma) | Throughput testing at controlled QPS |
| `fixed_schedule` | Open | Timestamp replay | Trace replay with exact timing |
| `user_centric_rate` | Semi-closed | Per-user gap = num_users / QPS | KV cache testing with rate control |
| **`agentic_load`** | **Closed** | **Response-driven (send next turn on completion)** | **Agentic conversation replay** |
| `adaptive_scale` | Closed | TTFT-driven auto-scaling | Finding maximum sustainable concurrency |

---

## How It Works

### Phase Timeline

The agentic load strategy organizes execution into three phases within a single benchmark duration:

```
|--- ramp-up ---|--- settling ---|--- measurement window ---|
^               ^                ^                          ^
phase start  all spawned    measure start              phase end
```

- **Ramp-up**: Users are spawned at a configurable rate (`--agentic-user-spawn-rate`). The ramp-up duration is `(num_users - 1) / spawn_rate` seconds.
- **Settling**: A configurable wait period (`--agentic-settling-time`) after all users are spawned, allowing the system to reach steady state before measurement begins.
- **Measurement window**: The remaining time within `--benchmark-duration` where performance metrics are collected.

### Trajectory Assignment

Before the phase starts, conversations are pre-assigned to users deterministically:

1. All top-level conversations (those with `agent_depth == 0`) are collected from the dataset.
2. The conversation list is shuffled using the seeded RNG (controlled by `--random-seed`).
3. Each user is assigned `--agentic-trajectories-per-user` conversations in non-overlapping round-robin fashion where possible.
4. When there are more assignments than unique conversations, IDs wrap around and may overlap across users.

### Closed-Loop Execution

Each user operates independently in a tight closed loop:

1. User sends the first turn of their current conversation.
2. On response completion, the next turn is issued immediately (no delay).
3. When all turns in a conversation are complete, the user advances to their next assigned conversation (with a fresh cache bust suffix).
4. After exhausting all assigned conversations, the user loops back to the first one (incrementing a pass counter to generate new cache bust suffixes).

### Subagent Integration

When a credit returns with `agent_depth > 0` (indicating a subagent child session), the strategy continues issuing subsequent turns for that child session but does not track it as a primary user trajectory. The parent user's trajectory index only advances when the parent conversation's final turn completes.

---

## CLI Options

### Required Options

| Option | Description |
|--------|-------------|
| `--agentic-load` | Enable closed-loop agentic load mode. |
| `--num-users` | Number of concurrent users. Each user runs through their assigned conversations in a closed loop. |
| `--benchmark-duration` | Total benchmark duration in seconds. Must be long enough for ramp-up + settling + at least 60 seconds of measurement. |

### Agentic Load Options

| Option | Default | Description |
|--------|---------|-------------|
| `--agentic-trajectories-per-user` | `20` | Number of conversations assigned to each user. Users loop through these until the phase ends. |
| `--agentic-max-isl-offset` | `10` | Maximum initial starting line (turn) offset. Each user skips a random number of turns (0 to this value) in their first conversation to stagger starting points. |
| `--agentic-user-spawn-rate` | `1.0` | Users to spawn per second during ramp-up. Controls how quickly users are introduced. |
| `--agentic-settling-time` | `5.0` | Wait time in seconds after all users are spawned before the measurement window starts. |

### Commonly Used Options

| Option | Default | Description |
|--------|---------|-------------|
| `--random-seed` | None | Seed for deterministic trajectory assignment and ISL offset randomization. |
| `--benchmark-grace-period` | `inf` (auto-set) | Grace period after benchmark duration ends. Auto-set to infinity for agentic load mode when not explicitly provided. |
| `--concurrency` | `--num-users` (auto-set) | Session concurrency. Auto-set to match `--num-users` when not explicitly provided. |
| `--input-file` | Required | Path to the agentic trajectory JSONL file. |

---

## Cache Busting

Each trajectory execution generates a unique cache bust suffix appended to the system prompt. The suffix is a SHA-256 hash derived from four values:

- **benchmark_id**: Unique per benchmark run, ensuring different runs never share cached prefixes.
- **pass_count**: Increments each time a user loops back through all their assigned conversations.
- **user_id**: The user's numeric identifier.
- **trajectory_index**: Which conversation in the user's assignment list is active.

The suffix is constant for all turns within a single trajectory execution, preserving intra-trajectory KV cache reuse (the server can cache the shared prefix across turns of the same conversation). It changes on trajectory and pass boundaries, preventing stale cache hits from a previous pass.

The generated suffix looks like: `\n\n[rid:a1b2c3d4e5f6]`

---

## ISL Offset Randomization

The `--agentic-max-isl-offset` option prevents all users from starting at turn 0 of their first conversation, which would create an unrealistic synchronized burst of short-context requests.

When `--agentic-max-isl-offset` is set to a value greater than 0:

1. Each user gets a per-user random offset between 0 and the max value (inclusive), derived from a seeded RNG.
2. The offset is applied only to the first trajectory of the first pass.
3. The starting turn is clamped to `min(offset, num_turns - 1)` to avoid skipping past the end of the conversation.
4. Subsequent conversations for the same user always start at turn 0.

This creates a spread of initial input sequence lengths across users, producing a more realistic distribution of KV cache prefill sizes at benchmark start.

---

## Dataset Format

Agentic load works with the `agentic_trajectory` custom dataset loader. Each line in the JSONL file represents one API call with cumulative messages:

```json
{"conversation_id": "conv_1", "conversation_idx": 0, "messages": [{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": "Write a Python function to sort a list."}]}
{"conversation_id": "conv_1", "conversation_idx": 1, "messages": [{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": "Write a Python function to sort a list."}, {"role": "assistant", "content": "def sort_list(lst): ..."}, {"role": "user", "content": "Now add type hints."}]}
```

Key format details:

- **conversation_id**: Groups turns into conversations. Only top-level conversations (`agent_depth == 0`) are assigned to users.
- **conversation_idx**: Ordering index within a conversation (0-based).
- **messages**: Cumulative message history -- turn N contains the full history from turns 0..N.
- **tools** (optional): Tool definitions for tool-calling conversations.

---

## Practical Examples

### Basic Agentic Load Test

Run 10 users through agentic trajectories for 5 minutes:

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --input-file trajectories.jsonl \
    --agentic-load \
    --num-users 10 \
    --benchmark-duration 300 \
    --streaming \
    --url localhost:8000
```

With the defaults, this will:
- Assign 20 conversations per user from the dataset
- Spawn 1 user per second (9-second ramp-up)
- Wait 5 seconds for settling
- Measure for the remaining ~286 seconds
- Auto-set concurrency to 10 (matching `--num-users`)
- Auto-set grace period to infinity

### High-Concurrency Test with Fast Ramp-Up

Spawn 50 users quickly with a larger trajectory pool:

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --input-file trajectories.jsonl \
    --agentic-load \
    --num-users 50 \
    --agentic-user-spawn-rate 10.0 \
    --agentic-settling-time 10.0 \
    --agentic-trajectories-per-user 50 \
    --benchmark-duration 600 \
    --streaming \
    --url localhost:8000
```

This ramp-up completes in `(50 - 1) / 10 = 4.9` seconds, followed by a 10-second settling period, leaving ~585 seconds for measurement.

### Deterministic, Reproducible Benchmark

Pin the random seed for fully reproducible trajectory assignment and ISL offsets:

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --input-file trajectories.jsonl \
    --agentic-load \
    --num-users 20 \
    --benchmark-duration 300 \
    --agentic-max-isl-offset 5 \
    --random-seed 42 \
    --streaming \
    --url localhost:8000
```

Running this command twice produces identical trajectory assignments, ISL offsets, and user spawn ordering.

### Disable ISL Offset

Start all users at turn 0 of their first conversation (useful for controlled experiments):

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --input-file trajectories.jsonl \
    --agentic-load \
    --num-users 10 \
    --benchmark-duration 300 \
    --agentic-max-isl-offset 0 \
    --streaming \
    --url localhost:8000
```

---

## Duration Requirements

The benchmark duration must satisfy:

```
benchmark_duration >= ramp_up + settling + 60 seconds
```

Where `ramp_up = (num_users - 1) / agentic_user_spawn_rate`.

If the duration is too short, AIPerf raises an error showing the breakdown:

```
--benchmark-duration (120s) too short: ramp-up (49.0s) + settling (5.0s) leaves
only 66.0s for measurement (minimum 60s)
```

Planning examples:

| Users | Spawn Rate | Settling | Min Duration |
|-------|-----------|----------|--------------|
| 10 | 1.0/s | 5s | 74s |
| 10 | 10.0/s | 5s | 66s |
| 50 | 1.0/s | 5s | 114s |
| 50 | 10.0/s | 10s | 75s |
| 100 | 5.0/s | 10s | 90s |

---

## Tips

- **Dataset size**: Ensure your dataset has enough conversations to cover `num_users * trajectories_per_user` assignments. Conversations wrap around if the dataset is smaller, but more unique conversations produce more realistic workload diversity.
- **Settling time**: Increase `--agentic-settling-time` for large user counts to ensure all users have active in-flight requests before measurement begins.
- **ISL offset tuning**: Set `--agentic-max-isl-offset` based on the typical conversation length in your dataset. A value of 10 works well for conversations with 15+ turns.
- **Grace period**: The grace period is automatically set to infinity for agentic load mode, ensuring all in-flight requests complete and are included in metrics. Override with `--benchmark-grace-period` if needed.
- **Subagent conversations**: The strategy automatically filters out child conversations (`agent_depth > 0`) during trajectory assignment. Subagent sessions are handled transparently when credits return with `agent_depth > 0`.
- **Incompatible options**: `--dataset-sampling-strategy` cannot be used with agentic load mode, as trajectory assignment handles dataset selection internally.
