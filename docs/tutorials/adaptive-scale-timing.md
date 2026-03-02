<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adaptive Scale Timing

Find the maximum sustainable concurrency for your LLM server by automatically scaling users while monitoring SLO compliance.

## Overview

The `adaptive_scale` timing strategy starts with a small number of concurrent users and progressively adds more as long as the server meets your performance targets. It is designed for trace replay workloads, especially coding session and coding trace datasets, where the goal is to discover how many users a server can support before latency degrades.

**Use adaptive scale when you need to:**

- Find the maximum concurrency a server sustains under SLO constraints
- Benchmark KV cache efficiency under growing multi-turn workloads
- Stress-test inference servers with realistic agentic coding patterns
- Avoid manual binary-search tuning of concurrency parameters

**How it works:**

1. Launch `--adaptive-scale-start-users` (default: `1`) concurrent sessions
2. Every `--adaptive-scale-assessment-period` seconds (default: `30.0`), measure server health
3. If the health metric has headroom below the threshold, add more users
4. Stop adding users when the threshold is breached or `--adaptive-scale-max-users` (default: `50`) is reached
5. Continue running until `--benchmark-duration` expires

**Requirements:**

- `--streaming` must be enabled (TTFT measurement requires streaming responses)
- `--benchmark-duration` must be set (adaptive scale runs indefinitely otherwise)
- Automatically enabled for `--custom-dataset-type coding_trace` datasets

---

## Two Scaling Modes

Adaptive scale supports two modes that determine how the server health is assessed at each period.

### TTFT Headroom Mode (Default)

Measures the Time to First Token (TTFT) across all requests in the assessment period and compares the configured percentile against a threshold.

| Parameter | CLI Option | Default |
|-----------|-----------|---------|
| TTFT threshold | `--adaptive-scale-max-ttft` | `2.0` seconds |
| TTFT metric | `--adaptive-scale-ttft-metric` | `p95` |

The headroom ratio is computed as:

```
headroom_ratio = 1.0 - (measured_metric / max_ttft)
```

If the measured metric equals or exceeds the threshold, no users are added.

**Example:** If p95 TTFT is 0.8s with a 2.0s threshold, headroom is `1.0 - 0.4 = 0.6` (60%).

### Goodput/SLO Mode

Activated by providing `--adaptive-scale-slo` thresholds. Instead of TTFT headroom, scaling decisions are based on the fraction of completed requests that meet all SLO targets.

| Parameter | CLI Option | Default |
|-----------|-----------|---------|
| SLO thresholds | `--adaptive-scale-slo` | None (disabled) |
| Minimum goodput ratio | `--adaptive-scale-min-goodput-ratio` | `0.95` |

Supported SLO metrics:

- `time_to_first_token` -- threshold in milliseconds
- `request_latency` -- threshold in milliseconds

All configured SLOs must pass for a request to count as "good." The headroom ratio is:

```
goodput_ratio = good_requests / total_requests
headroom_ratio = goodput_ratio - min_goodput_ratio
```

If the goodput ratio drops below the minimum, no users are added.

**Example:** With `--adaptive-scale-slo time_to_first_token:500 request_latency:2000` and `--adaptive-scale-min-goodput-ratio 0.9`, a period where 95% of requests meet both SLOs yields headroom of `0.95 - 0.9 = 0.05` (5%).

---

## Scaling Formulas

The `--adaptive-scale-formula` option (default: `conservative`) controls how many users are added when headroom is available.

| Formula | Computation | Minimum | Best For |
|---------|------------|---------|----------|
| `conservative` | `max(1, int(active_users * headroom_ratio * 0.5))` | 1 | Steady, proportional growth |
| `aggressive` | `max(2, 2 + int(headroom_pct / 10))` | 2 | Fast ramp regardless of user count |
| `linear` | `max(1, int(headroom_pct / 5))` | 1 | Predictable linear ramp |

Where `headroom_pct = headroom_ratio * 100`.

**Examples with 10 active users and 50% headroom:**

- `conservative`: `max(1, 10 * 0.5 * 0.5)` = 2 users
- `aggressive`: `max(2, 2 + 50/10)` = 7 users
- `linear`: `max(1, 50/5)` = 10 users

The number of users actually added is capped at `max_users - active_users`. When using `coding_trace` datasets, the formula defaults to `aggressive` unless explicitly overridden.

---

## Per-Session Rate Limiting

Enabled by default (`--adaptive-scale-enable-rate-limiting`), rate limiting applies exponential backoff to individual sessions that violate performance targets. This slows down "hot" sessions without stopping the entire benchmark.

**In TTFT headroom mode**, when a session's TTFT exceeds the threshold:

```
backoff = min(ttft / max_ttft - 1.0, 10.0)
actual_delay = min(backoff * 1.5^violation_count, 30.0)
```

**In SLO mode**, when a session fails any SLO:

```
actual_delay = min(2.0 * 1.5^violation_count, 30.0)
```

The backoff delay is added to the normal trace delay for the session's next turn. When a session's performance recovers (TTFT drops below threshold or SLO passes), the backoff is cleared immediately.

---

## Working Set Budget Tracking

Two mechanisms prevent new sessions from overwhelming the server's KV cache.

### Token Budget Per Period

`--adaptive-scale-max-new-tokens-per-period` (default: `500000`) limits the total input tokens from newly started sessions in each assessment period. This prevents TTFT spikes from bursts of cache-miss tokens. Initial `start_users` bypass this check.

### KV Cache Working Set Budget

`--adaptive-scale-max-working-set-tokens` (default: None, disabled) limits the total KV cache footprint across all active sessions. Each session's footprint is estimated from its hash block IDs multiplied by `--block-size` (internal default: `64` tokens). New sessions are rejected if they would push the total beyond the budget.

Sessions are tracked with TTL-based eviction:

- Main agent sessions: `cache_ttl_sec` (default: `3600.0` seconds)
- Subagent sessions: `subagent_cache_ttl_sec` (default: `300.0` seconds)

Expired sessions are evicted from the working set before each budget check and before each assessment.

---

## Session Recycling

When `--adaptive-scale-recycle` is enabled, completed sessions are replaced by sampling a new conversation from the dataset. This maintains the active user count at steady state. Without recycling (default), each conversation is replayed at most once and the active user count decreases as sessions finish.

---

## CLI Reference

### Required Options

| Option | Description |
|--------|------------|
| `--adaptive-scale` | Enable adaptive scale timing mode |
| `--benchmark-duration` | Maximum runtime in seconds |
| `--streaming` | Required for TTFT measurement |

### Scaling Options

| Option | Default | Description |
|--------|---------|------------|
| `--adaptive-scale-start-users` | `1` | Initial concurrent users |
| `--adaptive-scale-max-users` | `50` | Maximum concurrent users |
| `--adaptive-scale-assessment-period` | `30.0` | Seconds between assessments |
| `--adaptive-scale-formula` | `conservative` | Scaling formula: `conservative`, `aggressive`, `linear` |
| `--adaptive-scale-stagger-ms` | `50.0` | Milliseconds between launching new users |
| `--adaptive-scale-recycle` | `false` | Recycle completed sessions |

### TTFT Headroom Options

| Option | Default | Description |
|--------|---------|------------|
| `--adaptive-scale-max-ttft` | `2.0` | TTFT threshold in seconds |
| `--adaptive-scale-ttft-metric` | `p95` | Metric to evaluate: `p95`, `avg`, `max` |

### SLO/Goodput Options

| Option | Default | Description |
|--------|---------|------------|
| `--adaptive-scale-slo` | None | SLO thresholds as `KEY:VALUE` pairs (ms) |
| `--adaptive-scale-min-goodput-ratio` | `0.95` | Minimum goodput ratio to continue scaling |

### Rate Limiting and Budget Options

| Option | Default | Description |
|--------|---------|------------|
| `--adaptive-scale-enable-rate-limiting` | `true` | Per-session exponential backoff |
| `--adaptive-scale-max-new-tokens-per-period` | `500000` | Max new input tokens per period |
| `--adaptive-scale-max-working-set-tokens` | None | Max KV cache working set in tokens |

### Trace Replay Options

| Option | Default | Description |
|--------|---------|------------|
| `--adaptive-scale-max-delay` | `60.0` | Max inter-request delay in seconds |
| `--adaptive-scale-time-scale` | `1.0` | Multiplier for trace delays (`<1.0` = faster) |

---

## Examples

### Find Maximum Concurrency with TTFT Headroom

Start with 2 users, scale up while p95 TTFT stays under 1 second, cap at 100 users:

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --input-file traces.jsonl \
    --custom-dataset-type coding_trace \
    --benchmark-duration 600 \
    --adaptive-scale \
    --adaptive-scale-start-users 2 \
    --adaptive-scale-max-users 100 \
    --adaptive-scale-max-ttft 1.0 \
    --adaptive-scale-ttft-metric p95 \
    --adaptive-scale-formula aggressive
```

### SLO-Based Scaling with Goodput Targets

Scale up while 95% of requests meet both TTFT < 500ms and request latency < 2000ms:

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --input-file traces.jsonl \
    --custom-dataset-type coding_trace \
    --benchmark-duration 600 \
    --adaptive-scale \
    --adaptive-scale-start-users 5 \
    --adaptive-scale-max-users 200 \
    --adaptive-scale-slo time_to_first_token:500 request_latency:2000 \
    --adaptive-scale-min-goodput-ratio 0.95
```

### Conservative Scaling with Token Budget

Slow, steady scaling with a tight token budget to avoid TTFT spikes:

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --input-file traces.jsonl \
    --custom-dataset-type coding_trace \
    --benchmark-duration 300 \
    --adaptive-scale \
    --adaptive-scale-start-users 1 \
    --adaptive-scale-max-users 50 \
    --adaptive-scale-formula conservative \
    --adaptive-scale-assessment-period 60 \
    --adaptive-scale-max-new-tokens-per-period 200000
```

### Fast Replay with Compressed Delays

Replay traces at 2x speed with recycling to maintain steady state:

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --input-file traces.jsonl \
    --custom-dataset-type coding_trace \
    --benchmark-duration 300 \
    --adaptive-scale \
    --adaptive-scale-time-scale 0.5 \
    --adaptive-scale-max-delay 10.0 \
    --adaptive-scale-recycle
```

### Automatic Mode with Coding Traces

When using `--custom-dataset-type coding_trace`, adaptive scale is enabled automatically with the `aggressive` formula. You only need to set the duration:

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --input-file coding_traces/ \
    --custom-dataset-type coding_trace \
    --benchmark-duration 600
```

---

## Tips

- **Start conservative, then tune.** Begin with the `conservative` formula and short assessment periods to understand your server's scaling curve before switching to `aggressive`.
- **Use SLO mode for production validation.** TTFT headroom mode is simpler, but SLO mode with `--adaptive-scale-slo` gives you direct control over what "good enough" means for your deployment.
- **Set `--adaptive-scale-max-delay`** to avoid long idle gaps from trace replay. The default of 60 seconds already clamps most outlier delays.
- **Enable recycling for sustained load.** Without `--adaptive-scale-recycle`, active users decrease as conversations finish. Enable it to maintain steady-state concurrency for the full benchmark duration.
- **Watch the token budget.** If scaling appears to stall before `max_users`, check whether `--adaptive-scale-max-new-tokens-per-period` is limiting new session starts. Increase the budget or set to None to disable.
- **Use `--adaptive-scale-time-scale`** values below 1.0 to compress trace delays for faster benchmarks. A value of `0.5` replays at 2x speed.
- **Combine with `--goodput`** to get both adaptive scaling decisions and final goodput metrics in the benchmark output. The `--adaptive-scale-slo` thresholds control scaling while `--goodput` controls metric reporting independently.
