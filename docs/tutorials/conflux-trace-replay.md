<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Conflux Proxy Trace Replay

Replay Claude Code proxy captures with full parent/subagent hierarchy and absolute timestamp scheduling.

## Overview

The `conflux` dataset loader reads JSON files produced by the Conflux proxy — a transparent HTTP proxy that records all API requests made during a Claude Code session. Each record contains the complete request payload, agent thread identifiers, token usage, and precise timestamps.

Unlike the [Claude Code Trace Replay](claude-code-trace-replay.md) loader (which reconstructs API calls from JSONL session transcripts), the Conflux loader replays **verbatim API request bodies** captured at the HTTP layer. This includes the exact messages array, system prompts, tool definitions, thinking configuration, and streaming flags as they were sent to the API.

### When to Use

- **Verbatim production replay**: Send the exact same HTTP request bodies captured from a real session
- **Absolute timestamp scheduling**: Replay every request at its precise original timestamp (with optional speedup)
- **Multi-agent benchmarking**: Automatically detects parent/subagent thread hierarchies from proxy metadata
- **KV cache evaluation**: Test prefix caching with real growing message arrays and tool-use patterns

### How It Works

1. Conflux proxy records each API request with `agent_id`, `is_subagent`, timestamps, and (optionally) base64-encoded request bodies
2. AIPerf groups records by `agent_id` into conversation threads
3. The thread with `is_subagent=False` becomes the parent; threads with `is_subagent=True` become background subagent children
4. Records without an `agent_id` (e.g. lightweight haiku tool-processing calls) are attached as single-turn background subagent children at the closest parent turn
5. All requests are dispatched at their original absolute timestamps using [Fixed Schedule](fixed-schedule.md) mode

---

## Capture File Format

The loader expects a single JSON file containing an array of API request records:

```json
[
  {
    "id": "req_001",
    "session_id": "735d8f1b-...",
    "agent_id": "claude",
    "is_subagent": false,
    "timestamp": "2026-02-25T02:02:00.000Z",
    "duration_ms": 15000,
    "model": "claude-opus-4-6",
    "messages": [...],
    "tools": [...],
    "tokens": {
      "input": 5000,
      "input_cached": 3000,
      "input_cache_write": 1500,
      "output": 800
    },
    "hyperparameters": {"max_tokens": 32000},
    "is_streaming": true,
    "base64": {
      "request_body": "<base64-encoded complete request body>"
    }
  },
  ...
]
```

### Record Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `string` | Yes | Unique record identifier |
| `session_id` | `string` | Yes | Claude Code session identifier |
| `agent_id` | `string \| null` | No | Agent thread identifier (e.g. `"claude"` for parent, hex string for subagents, `null` for orphan calls) |
| `is_subagent` | `bool` | No | Whether this record belongs to a subagent thread (default: `false`) |
| `timestamp` | `string` | Yes | ISO 8601 timestamp of the API request |
| `duration_ms` | `int \| float` | No | Server-side response duration in milliseconds |
| `model` | `string` | No | Model name |
| `messages` | `list[dict]` | No | Messages array, defaults to `[]` (may include a synthetic system-role message at index 0). Auto-detection requires this field in at least one of the first 20 records. |
| `tools` | `list[dict]` | No | Tool definitions |
| `tokens` | `object` | No | Token usage: `input`, `input_cached`, `input_cache_write`, `output` |
| `hyperparameters` | `object` | No | Request parameters: `max_tokens`, `temperature` |
| `is_streaming` | `bool \| null` | No | Whether the request used streaming |
| `ttft_ms` | `int \| float \| null` | No | Time to first token in milliseconds |
| `base64.request_body` | `string` | No | Base64-encoded verbatim request body (preferred over reconstructed payload) |
| `output` | `list[dict]` | No | Response content blocks |

### Payload Resolution

When `base64.request_body` is present, the loader decodes it and uses the verbatim request body as the turn's `raw_payload`. This preserves the exact request structure including thinking configuration, system prompts, and all parameters.

When `base64.request_body` is absent, the loader reconstructs the payload from top-level fields (`messages`, `model`, `tools`, `hyperparameters`, `is_streaming`). The first message is checked for a synthetic `system` role prepended by the proxy and split into a separate `system` key if found.

---

## Thread Detection

Records are grouped by `agent_id`:

| `agent_id` | `is_subagent` | Classification |
|-------------|---------------|----------------|
| `"claude"` (or any value) | `false` | Parent agent thread |
| Hex string | `true` | Subagent child thread |
| `null` | N/A | Orphan record (attached as background child at closest parent turn) |

### Parent and Child Linking

Child threads are attached to the parent via `SubagentSpawnInfo` with `is_background=True`. The spawn point is the parent turn whose timestamp is closest to the child's first request. The child is dispatched when the parent turn *preceding* the spawn point completes (i.e., when the spawn point becomes the next turn to send).

Orphan records (those without `agent_id`, typically lightweight haiku model calls for file-path extraction) are each treated as a single-turn background subagent child, spawned at the closest parent turn by timestamp.

---

## Basic Usage

### Quick Start

```bash
aiperf profile \
    claude-opus-4-6,claude-haiku-4-5-20251001 \
    --url localhost:8080 \
    --endpoint-type anthropic-messages \
    --input-file ~/captures/unified-export.json \
    --custom-dataset-type conflux \
    --fixed-schedule \
    --fixed-schedule-auto-offset \
    --streaming \
    --tokenizer gpt2
```

### Flags Explained

| Flag | Purpose |
|------|---------|
| `claude-opus-4-6,claude-haiku-4-5-20251001` | Model names matching the models in the capture (comma-separated for multi-model traces) |
| `--endpoint-type anthropic-messages` | Must match the API format of the captured requests (e.g., `anthropic-messages` for Claude API captures). The loader replays verbatim request bodies, so the endpoint must accept the same format. |
| `--input-file` | Path to the Conflux JSON capture file |
| `--custom-dataset-type conflux` | Explicitly select the Conflux loader (auto-detection also works) |
| `--fixed-schedule` | Enable absolute timestamp scheduling |
| `--fixed-schedule-auto-offset` | Normalize timestamps so the first request starts at t=0 |
| `--streaming` | Enable streaming responses |
| `--tokenizer gpt2` | Use a placeholder tokenizer (Anthropic models don't have public tokenizers) |

---

## Speedup Ratio

Use `--synthesis-speedup-ratio` to compress or stretch the replay timeline. This divides all timestamps by the given factor:

```bash
# Replay at 10x speed (a 20-minute session completes in ~2 minutes)
aiperf profile \
    claude-opus-4-6,claude-haiku-4-5-20251001 \
    --url localhost:8080 \
    --endpoint-type anthropic-messages \
    --input-file ~/captures/unified-export.json \
    --custom-dataset-type conflux \
    --fixed-schedule \
    --fixed-schedule-auto-offset \
    --synthesis-speedup-ratio 10 \
    --streaming \
    --tokenizer gpt2
```

| Ratio | Effect |
|-------|--------|
| `1` | Real-time replay (original timing) |
| `10` | 10x faster (compresses inter-request gaps) |
| `20` | 20x faster |

The speedup ratio affects the timestamps used for scheduling but does not change the actual request payloads. Server response times are not affected.

---

## Request Budget

In fixed schedule mode, the request count is **automatically set** to the total number of dataset entries when neither `--num-requests` nor `--num-sessions` is provided. This means you can replay a full Conflux trace without specifying a request budget — the tool counts the records for you.

To override the auto-detected count (e.g., to replay only a subset of the trace), use `--num-requests`:

```bash
# Stop after 100 requests (partial replay)
aiperf profile \
    claude-opus-4-6,claude-haiku-4-5-20251001 \
    --url localhost:8080 \
    --endpoint-type anthropic-messages \
    --input-file ~/captures/unified-export.json \
    --custom-dataset-type conflux \
    --fixed-schedule \
    --fixed-schedule-auto-offset \
    --synthesis-speedup-ratio 10 \
    --num-requests 100 \
    --streaming \
    --tokenizer gpt2
```

If `--num-requests` is lower than the total record count, the benchmark stops early and some turns will not be sent.

---

## Understanding the Output

A typical Conflux replay produces output like:

```
Loaded 9 agent threads + 158 orphan requests (336 total records)
Merged 158 orphan records into parent agent 'claude'
Converted 167 conversations (336 total turns, 166 subagent children incl. 158 orphans)
Built schedule with 1 timestamps, zero_ms=177198492783, auto_offset=True
Phase profiling started | target: 336 requests, 167 sessions
Phase profiling complete | completed=336, cancelled=0, errors=0 | elapsed=117.96s
```

| Log Line | Meaning |
|----------|---------|
| `9 agent threads + 158 orphan requests` | 1 parent + 8 subagent threads + 158 untagged haiku calls |
| `167 conversations` | 1 parent + 8 explicit subagents + 158 orphan subagents |
| `Built schedule with 1 timestamps` | Only root conversations (`agent_depth=0`) have their first turns scheduled directly. This trace has one root conversation (the parent), so 1 timestamp. All 166 children are spawned by the subagent manager when the parent reaches the corresponding turn. |
| `336 requests` | Total API calls replayed |

---

## Tips

- **Match `--endpoint-type` to the capture format.** The loader replays verbatim request bodies, so the endpoint must accept the same API format used during capture. For Claude API captures, use `--endpoint-type anthropic-messages`.
- **Auto-detection works.** The loader checks if the first 20 records contain `agent_id`, `is_subagent`, and `messages` fields. You can omit `--custom-dataset-type conflux` if auto-detection succeeds.
- **Base64 payloads are preferred.** When `base64.request_body` is present, the loader uses the exact captured request body. This preserves thinking configuration, temperature, and all parameters that might be lost in the top-level field reconstruction.
- **Metadata is stripped.** The `metadata` key (which may contain user IDs) is removed from base64 payloads before replay.
- **Orphan records are background children.** Records without `agent_id` do not block the parent's progress. They fire concurrently at their original timestamps.
- **The parent is the only scheduled conversation.** `setup_phase` only schedules root conversations (`agent_depth=0`). All subagent children (explicit and orphan) are spawned by the `SubagentSessionManager` when the parent reaches the corresponding turn.
- **Responses are discarded.** All conversations are created with `discard_responses=True` since the replay sends captured messages verbatim rather than building on server responses.
- **Request count is auto-detected.** In fixed schedule mode, omitting `--num-requests` and `--num-sessions` automatically sets the request budget to the dataset entry count. Use `--num-requests` only to cap a partial replay.

---

## Related Documentation

- [Fixed Schedule Benchmarking](fixed-schedule.md) -- Timestamp-based scheduling mode used by Conflux replay
- [Claude Code Trace Replay](claude-code-trace-replay.md) -- Alternative loader for JSONL session transcripts
- [API Capture Trace Replay](api-capture-trace-replay.md) -- Alternative loader for mitmproxy captures
- [Subagent Sessions](subagent-sessions.md) -- How parent/child session hierarchies work in AIPerf
- [Raw Payload Replay](raw-payload-replay.md) -- How `raw_payload` turns bypass prompt generation
