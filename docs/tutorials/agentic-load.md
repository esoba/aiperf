<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic Load Generation

Benchmark LLM inference servers with pre-recorded agentic trajectories using concurrency-based load generation with deterministic user assignment.

## When to Use This Mode

Use agentic load generation when you need to:

- **Replay agentic trajectories** --- Benchmark with real multi-turn conversations captured from AI agents (coding assistants, research agents, etc.)
- **Concurrency-based load** --- N concurrent users = N concurrent requests max, with no rate limiting
- **Zero inter-turn delay** --- Next turn fires immediately when the previous one completes, matching how agents actually execute
- **Deterministic user assignment** --- Same user-to-trajectory mapping regardless of concurrency level, enabling reproducible A/B comparisons
- **Full dataset coverage** --- Every user works through all conversations in the dataset, starting at a random offset to avoid input sequence length (ISL) bias

### The Real-World Scenario

Imagine 10 AI coding agents working concurrently against the same inference server. Each agent:
1. Sends a prompt (e.g., "write a function to sort a list")
2. Receives the response immediately
3. Sends the next prompt in the trajectory without delay
4. Finishes the conversation and starts the next one

Agentic load generation recreates this pattern exactly. Unlike rate-based modes, there is no artificial pacing between requests --- concurrency is the only throttle, matching real agent execution.

### Contrast with Other Modes

| Mode | Turn Timing | Load Control | Best For |
|------|-------------|--------------|----------|
| **Agentic load** | Immediate (zero delay) | Concurrency only | Agentic trajectory replay, real-world agent simulation |
| **User-centric rate** | Fixed per-user gap | Rate + concurrency | KV cache TTL testing, controlled multi-turn timing |
| **Request rate** | Next rate interval | Rate (+ optional concurrency) | Throughput testing, arrival pattern simulation |
| **Concurrency** | Immediate | Concurrency only | Max throughput discovery, stress testing |

## Quick Start

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --input-file trajectories.jsonl \
    --custom-dataset-type multi_turn \
    --agentic-load \
    --concurrency 10 \
    --benchmark-duration 300
```

This configures 10 concurrent users working through trajectories from `trajectories.jsonl`:

- **10 users** each working through conversations sequentially
- **Zero inter-turn delay**: next turn fires immediately on completion
- **Random start offsets**: each user begins at a different conversation to avoid ISL bias
- **Wrap-around**: users loop back to the first conversation after finishing the last one
- **Duration-based stop**: benchmark ends after 300 seconds

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--agentic-load` | Enable agentic load generation mode |
| `--concurrency` | Number of concurrent users (required) |
| `--input-file` | Path to JSONL file with multi-turn trajectories |
| `--custom-dataset-type` | Dataset format (`multi_turn` for agentic trajectories) |

## How It Works

### User Assignment

During setup, each user is assigned a **deterministic random start offset** into the conversation list:

```
Dataset conversations: [C0, C1, C2, C3, C4, C5, C6, C7]

User 0 starts at → C3 → C4 → C5 → C6 → C7 → C0 → C1 → C2 → C3 → ...
User 1 starts at → C6 → C7 → C0 → C1 → C2 → C3 → C4 → C5 → C6 → ...
User 2 starts at → C1 → C2 → C3 → C4 → C5 → C6 → C7 → C0 → C1 → ...
```

Key properties:
- **Deterministic**: User i always gets the same offset regardless of total concurrency (User 5 starts at the same conversation whether you run with 10 or 100 users)
- **Distributed**: Random offsets prevent all users from starting at the same conversation, avoiding ISL bias from correlated prompt lengths
- **Wrap-around**: After the last conversation, users loop back to the beginning

### Execution Flow

```
User 0:  [C3 Turn0][C3 Turn1][C3 Turn2] → [C4 Turn0][C4 Turn1] → [C5 Turn0]...
User 1:  [C6 Turn0][C6 Turn1] → [C7 Turn0][C7 Turn1][C7 Turn2][C7 Turn3] → ...
User 2:  [C1 Turn0][C1 Turn1][C1 Turn2] → [C2 Turn0][C2 Turn1] → ...
              │         │         │
              └────┬────┘         │
           Zero delay between     │
           turns in same          │
           conversation           │
                                  │
                      New x_correlation_id
                      for next conversation
                      (fresh sticky routing)
```

1. **Within a conversation**: turns fire immediately on completion (zero delay), sharing the same `x_correlation_id` for sticky routing
2. **Between conversations**: the user advances to the next trajectory with a new `x_correlation_id`, enabling fresh server-side session state

### Concurrency Ramp Integration

The existing `--concurrency-ramp-duration` option works with agentic load. During `execute_phase()`, each user's first turn blocks on the session concurrency semaphore, so ramping naturally staggers user spawning:

```
    Concurrency (users active)
    10 ─────────────────────────────── ●━━━━━━━━━━━━━━━━
                                  ●───┘
                             ●───┘
                        ●───┘
                   ●───┘
     1 ───────●───┘
    └─────────┬───────────────────────┬──────────────────▶
              0s                    ramp               Time
                                  duration
```

## Preparing Trajectory Data

Agentic load works with `multi_turn` custom datasets. Each conversation in the JSONL file represents one trajectory:

```bash
cat > trajectories.jsonl << 'EOF'
{"turns": [{"text": "Write a Python function to sort a list"}, {"text": "Add type hints to the function"}]}
{"turns": [{"text": "Explain quicksort"}, {"text": "Show me the implementation"}, {"text": "What is the time complexity?"}]}
{"turns": [{"text": "What is a binary tree?"}]}
EOF
```

Each line is a JSON object with a `turns` array, where each turn has a `text` field. The file can contain conversations of different lengths --- the strategy handles variable turn counts naturally.

> [!NOTE]
> The trajectory file should contain pre-recorded conversations captured from real agent interactions. Each conversation represents a complete agentic session (e.g., a coding task from start to finish). See the [Custom Dataset Guide](./custom-dataset.md) for detailed format documentation.

## Examples

### Basic Trajectory Replay

Replay agentic trajectories with 10 concurrent users for 5 minutes:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --input-file trajectories.jsonl \
    --custom-dataset-type multi_turn \
    --agentic-load \
    --concurrency 10 \
    --benchmark-duration 300
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Agentic load: 10 users, 50 conversations
INFO     AIPerf System is PROFILING

Profiling: [05:00] - Running for 300 seconds...

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-openai-chat-c10-agentic/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │  456.78 │  123.45 │ 1234.56 │ 1098.23 │  423.12 │
│    Time to First Token (ms) │   89.12 │   45.67 │  234.56 │  212.34 │   78.90 │
│    Inter Token Latency (ms) │   12.34 │    8.90 │   23.45 │   21.23 │   11.89 │
│ Output Token Count (tokens) │  156.00 │   12.00 │  512.00 │  489.00 │  134.00 │
│  Request Throughput (req/s) │   21.45 │       - │       - │       - │       - │
└─────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-openai-chat-c10-agentic/profile_export_aiperf.json
```

### With Warmup Phase

Add a warmup phase to prepare the system before measurement:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --input-file trajectories.jsonl \
    --custom-dataset-type multi_turn \
    --agentic-load \
    --concurrency 10 \
    --warmup-duration 30 \
    --benchmark-duration 300
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is WARMING UP

Warming Up: [00:30] - Running for 30 seconds...

INFO     Warmup completed, starting profiling phase
INFO     Agentic load: 10 users, 50 conversations
INFO     AIPerf System is PROFILING

Profiling: [05:00] - Running for 300 seconds...

INFO     Benchmark completed successfully
```

Warmup requests are discarded from metrics, ensuring cold-start effects don't pollute results.

### With Concurrency Ramp

Gradually increase the number of active users to avoid overwhelming the server at startup:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --input-file trajectories.jsonl \
    --custom-dataset-type multi_turn \
    --agentic-load \
    --concurrency 10 \
    --concurrency-ramp-duration 5 \
    --warmup-duration 30 \
    --benchmark-duration 300
```

Users ramp from 1 to 10 over 5 seconds, then maintain full concurrency for the rest of the benchmark.

### Request Count Stop Condition

Stop after a specific number of total requests instead of a duration:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --input-file trajectories.jsonl \
    --custom-dataset-type multi_turn \
    --agentic-load \
    --concurrency 5 \
    --request-count 500
```

### Session Count Stop Condition

Stop after a specific number of conversations have been started:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --input-file trajectories.jsonl \
    --custom-dataset-type multi_turn \
    --agentic-load \
    --concurrency 5 \
    --conversation-num 100
```

### High Concurrency Stress Test

Test how the server handles many concurrent agentic sessions:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --input-file trajectories.jsonl \
    --custom-dataset-type multi_turn \
    --agentic-load \
    --concurrency 50 \
    --concurrency-ramp-duration 10 \
    --benchmark-duration 600
```

### Reproducible A/B Comparison

Compare two server configurations with identical load patterns:

```bash
# Server A
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url server-a:8000 \
    --streaming \
    --input-file trajectories.jsonl \
    --custom-dataset-type multi_turn \
    --agentic-load \
    --concurrency 10 \
    --random-seed 42 \
    --benchmark-duration 300

# Server B (same flags, different URL)
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url server-b:8000 \
    --streaming \
    --input-file trajectories.jsonl \
    --custom-dataset-type multi_turn \
    --agentic-load \
    --concurrency 10 \
    --random-seed 42 \
    --benchmark-duration 300
```

With `--random-seed 42`, both runs use identical user-to-trajectory assignments, so any difference in metrics is attributable to the server, not the load pattern.

## CLI Reference

### Required Options

| Option | Type | Description |
|--------|------|-------------|
| `--agentic-load` | flag | Enable agentic load generation mode |
| `--concurrency` | int | Number of concurrent users |

### Stop Conditions (at least one required)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--benchmark-duration` | float | None | Stop after this many seconds |
| `--request-count` | int | None | Stop after this many total requests |
| `--conversation-num` | int | None | Stop after starting this many conversations |

### Optional Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--concurrency-ramp-duration` | float | None | Ramp concurrency from 1 to target over this many seconds |
| `--benchmark-grace-period` | float | 30.0 | Seconds to wait for in-flight responses after stop condition |
| `--random-seed` | int | None | Seed for deterministic user offset assignment (global setting, also affects dataset generation) |

### Dataset Options

| Option | Type | Description |
|--------|------|-------------|
| `--input-file` | path | Path to JSONL file with trajectory data |
| `--custom-dataset-type` | str | Dataset format (use `multi_turn` for trajectories) |

## Incompatible Options

| Option | Reason |
|--------|--------|
| `--request-rate` | Agentic load is concurrency-only (no rate limiting) |
| `--user-centric-rate` | Different timing model (per-user gaps vs. zero delay) |
| `--num-users` | Only valid with `--user-centric-rate`; agentic load uses `--concurrency` for user count |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `--agentic-load requires --concurrency to be set` | Missing required option | Add `--concurrency N` |
| `--agentic-load cannot be used with --request-rate` | Conflicting timing modes | Remove `--request-rate` |
| `--agentic-load cannot be used with --user-centric-rate` | Conflicting timing modes | Remove `--user-centric-rate` |
| All users start at the same conversation | No random seed or only one conversation | Add more conversations to the dataset or use `--random-seed` |
| Low throughput despite high concurrency | Server bottleneck | Check server GPU utilization and queue depth |
| `At least one stop condition must be set (--request-count, --num-sessions, or --benchmark-duration)` | No stop condition configured | Add `--benchmark-duration`, `--request-count`, or `--conversation-num` |

## Related Documentation

- [Agentic Benchmarking](./agentic-benchmarking.md) --- Agentic Coding dataset format with pre-recorded trajectories
- [Custom Dataset Guide](./custom-dataset.md) --- Preparing trajectory JSONL files
- [Multi-Turn Conversations](./multi-turn.md) --- General multi-turn conversation benchmarking
- [Warmup Phase](./warmup.md) --- Configuring warmup for accurate measurement
- [Gradual Ramping](./ramping.md) --- Smooth concurrency ramp-up
- [User-Centric Timing](./user-centric-timing.md) --- Alternative mode with per-user turn gaps
- [Request Rate with Concurrency](./request-rate-concurrency.md) --- Rate-based load generation
