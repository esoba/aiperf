<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# API Capture Trace Replay

Replay captured API traffic from mitmproxy or similar HTTP proxies to benchmark LLM inference servers under production-realistic workloads.

## Overview

The `api_capture_trace` dataset loader reads directories of captured API request bodies and replays them verbatim against a target server. This preserves the exact request structure from production traffic: full message history, system prompts, tool definitions, thinking configuration, and streaming flags.

### When to Use

- **Production replay**: Benchmark a server with the exact same requests your application makes in production
- **KV cache evaluation**: Test prefix caching behavior with real conversation patterns where messages grow incrementally
- **Agentic workloads**: Replay multi-thread sessions with parent and subagent conversations running concurrently
- **A/B testing**: Compare server configurations under identical workloads captured once and replayed many times

### How It Works

1. A capture proxy (mitmproxy, custom middleware, etc.) records API requests to a directory
2. Each request body is saved as `req_XXXX.json`; metadata goes into `capture.jsonl`
3. AIPerf loads the directory, detects conversation threads by hashing system prompts, and groups API calls into threads
4. The thread with the most API calls becomes the parent; remaining threads become background subagent children
5. Each API call is replayed as a complete `raw_payload` turn with the original timing preserved as inter-turn delays

---

## Capture Directory Format

The loader expects a directory containing two kinds of files:

```
my_capture/
  capture.jsonl       # Metadata log (one JSON object per line)
  req_0000.json       # Complete API request body for call_index 0
  req_0001.json       # Complete API request body for call_index 1
  req_0002.json       # ...
  ...
```

### capture.jsonl

Each line is a JSON object recording either a request or response event. The `call_index` field correlates request/response pairs and maps to the `req_XXXX.json` filename.

**Request entry fields:**

| Field | Type | Description |
|-------|------|-------------|
| `call_index` | `int` | Sequence number, zero-padded to 4 digits in the filename (`req_0000.json`) |
| `direction` | `"request"` | Marks this as a request entry |
| `timestamp` | `float` | Unix timestamp in seconds |
| `model` | `string` | Model name |
| `max_tokens` | `int \| null` | Max tokens from the request |
| `stream` | `bool \| null` | Streaming flag |
| `tool_count` | `int` | Number of tools in the request |

**Response entry fields:**

| Field | Type | Description |
|-------|------|-------------|
| `call_index` | `int` | Matches the corresponding request |
| `direction` | `"response"` | Marks this as a response entry |
| `timestamp` | `float` | Unix timestamp in seconds |
| `stop_reason` | `string` | Stop reason (`"end_turn"`, `"tool_use"`, etc.) |
| `usage` | `object` | Token usage with `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` |

**Example capture.jsonl:**

```jsonl
{"call_index": 0, "direction": "request", "timestamp": 1771454900.0, "model": "claude-opus-4-6", "max_tokens": 32000, "stream": true, "tool_count": 35}
{"call_index": 0, "direction": "response", "timestamp": 1771454902.0, "stop_reason": "tool_use", "usage": {"input_tokens": 5000, "output_tokens": 800, "cache_creation_input_tokens": 3000, "cache_read_input_tokens": 1500}}
{"call_index": 1, "direction": "request", "timestamp": 1771454905.0, "model": "claude-opus-4-6", "max_tokens": 32000, "stream": true, "tool_count": 35}
{"call_index": 1, "direction": "response", "timestamp": 1771454908.0, "stop_reason": "end_turn", "usage": {"input_tokens": 6500, "output_tokens": 1200, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 4500}}
```

### req_XXXX.json

Each file contains the complete API request body as sent to the server. The loader reads these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | `list[dict]` | Yes | Full messages array (user, assistant, tool_result, etc.) |
| `system` | `list[dict]` | No | System prompt blocks (e.g., `[{"type": "text", "text": "..."}]`) |
| `tools` | `list[dict]` | No | Tool definitions |
| `model` | `string` | No | Model name |
| `max_tokens` | `int` | No | Max tokens for generation |
| `stream` | `bool` | No | Streaming flag |
| `thinking` | `dict` | No | Thinking configuration (e.g., `{"type": "adaptive"}`) |

**Example req_0000.json:**

```json
{
  "model": "claude-opus-4-6",
  "system": [
    {"type": "text", "text": "You are an expert coding assistant."},
    {"type": "text", "text": "Follow these rules carefully."}
  ],
  "messages": [
    {"role": "user", "content": "Refactor the authentication module to use JWT tokens."}
  ],
  "tools": [
    {"name": "read_file", "description": "Read a file from disk"},
    {"name": "write_file", "description": "Write a file to disk"}
  ],
  "max_tokens": 32000,
  "stream": true,
  "thinking": {"type": "adaptive"}
}
```

---

## Thread Detection

The loader groups API calls into conversation threads by hashing the system prompt blocks from each request. Requests that share the same system prompt content belong to the same thread.

### How Grouping Works

1. **System prompt hashing**: For each request, the `system` blocks are serialized (with keys sorted) and hashed with MD5. The first 12 hex characters form the thread key.
2. **Billing header filtering**: System blocks whose text starts with `x-anthropic-billing-header` are excluded from hashing, since these contain volatile per-request data that would incorrectly split a single conversation into multiple threads.
3. **Independent requests**: Requests with no system blocks (empty `system` array or missing) are treated as independent single-turn threads. Each gets its own thread key: `independent_XXXX`.

### Parent vs. Child Classification

After grouping, threads are classified into three categories:

| Category | Criteria | Behavior |
|----------|----------|----------|
| **Parent** | The thread with the most API calls among those with system prompts | Runs as the primary conversation (`agent_depth=0`) |
| **Children** | All other threads with system prompts | Spawned as background subagent children (`agent_depth=1`) |
| **Independent** | Threads without system prompts | Standalone single-turn conversations (`agent_depth=0`) |

Child threads are attached to the parent via `SubagentSpawnInfo` with `is_background=True`. The spawn point is determined by finding the parent turn whose timestamp is closest to the child's first request, then advancing by one index (so the child spawns just after that parent turn).

---

## Prefetch Filtering

Requests with `max_tokens` set to 0 or 1 are treated as prefetch/health-check requests and are automatically filtered out. Cache-warming prefetches (where `max_tokens` is `null` or missing) are kept for replay.

## Session Detection

When a capture directory contains multiple recording sessions (e.g., the proxy was restarted), the loader uses only the **last session**. Sessions are detected by monitoring `call_index` values in request entries: when `call_index` does not increase (i.e., stays the same, decreases, or resets to 0), a new session boundary is identified.

---

## Basic Usage

Point `--input-file` at a capture directory. The loader auto-detects the format when it finds both `capture.jsonl` and `req_*.json` files:

```bash
aiperf profile \
    --model claude-opus-4-6 \
    --endpoint-type chat \
    --input-file ./my_capture/ \
    --url api.anthropic.com \
    --api-key $ANTHROPIC_API_KEY \
    --streaming \
    --concurrency 1
```

You can also explicitly specify the dataset type:

```bash
aiperf profile \
    --model claude-opus-4-6 \
    --endpoint-type chat \
    --input-file ./my_capture/ \
    --custom-dataset-type api_capture_trace \
    --url api.anthropic.com \
    --api-key $ANTHROPIC_API_KEY \
    --streaming \
    --concurrency 1
```

### With Adaptive Scaling

To find the maximum sustainable load, use adaptive scaling. This starts with a single user and adds more while TTFT remains under the threshold:

```bash
aiperf profile \
    --model claude-opus-4-6 \
    --endpoint-type chat \
    --input-file ./my_capture/ \
    --url localhost:8080 \
    --streaming \
    --adaptive-scale \
    --adaptive-scale-start-users 1 \
    --adaptive-scale-max-users 50 \
    --adaptive-scale-max-ttft 2.0 \
    --adaptive-scale-recycle \
    --benchmark-duration 300
```

---

## Configuration Options

### Dataset Sampling Strategy

The default sampling strategy for `api_capture_trace` is `sequential`. You can override this:

```bash
--dataset-sampling-strategy sequential   # Default: iterate traces in order
--dataset-sampling-strategy shuffle      # Shuffle traces, iterate without replacement
--dataset-sampling-strategy random       # Random sampling with replacement
```

### Timing Replay

Inter-turn delays are computed from the `timestamp` field in `capture.jsonl`. The delay for each turn is the difference between its timestamp and the previous turn's timestamp (within the same thread). The first turn in each thread has no delay.

To compress or stretch replay timing, use `--adaptive-scale-time-scale`:

```bash
--adaptive-scale-time-scale 0.5    # Replay at 2x speed (halved delays)
--adaptive-scale-time-scale 1.0    # Replay at original speed (default)
--adaptive-scale-time-scale 2.0    # Replay at half speed (doubled delays)
```

To cap long idle periods between turns:

```bash
--adaptive-scale-max-delay 60.0    # Cap delays at 60 seconds (default)
--adaptive-scale-max-delay 10.0    # Cap delays at 10 seconds for faster replay
```

### Verbatim Replay

Every turn carries a `raw_payload` containing the complete API request body (model, messages, system, tools, max_tokens, stream, thinking). During replay, AIPerf sends this payload directly to the server, preserving the exact request structure from the capture. Conversations are created with `discard_responses=True` since the replay uses captured messages rather than building on server responses.

---

## Preparing Capture Data

### From mitmproxy

Set up mitmproxy to intercept API traffic and save request bodies:

```python
# mitmproxy addon: save_api_requests.py
import json
import os
from mitmproxy import http

capture_dir = "./my_capture"
os.makedirs(capture_dir, exist_ok=True)
call_index = 0
capture_log = open(os.path.join(capture_dir, "capture.jsonl"), "a")

def request(flow: http.HTTPFlow):
    global call_index
    if "/v1/messages" not in flow.request.pretty_url:
        return

    body = json.loads(flow.request.get_text())
    req_file = os.path.join(capture_dir, f"req_{call_index:04d}.json")
    with open(req_file, "w") as f:
        json.dump(body, f)

    entry = {
        "call_index": call_index,
        "direction": "request",
        "timestamp": flow.request.timestamp_start,
        "model": body.get("model"),
        "max_tokens": body.get("max_tokens"),
        "stream": body.get("stream"),
        "tool_count": len(body.get("tools", [])),
    }
    capture_log.write(json.dumps(entry) + "\n")
    capture_log.flush()
    flow.metadata["call_index"] = call_index
    call_index += 1

def response(flow: http.HTTPFlow):
    if "call_index" not in flow.metadata:
        return

    ci = flow.metadata["call_index"]
    # For streaming responses, parse the final event for usage
    # For non-streaming, parse the response body directly
    body = json.loads(flow.response.get_text()) if flow.response else {}

    entry = {
        "call_index": ci,
        "direction": "response",
        "timestamp": flow.response.timestamp_end if flow.response else None,
        "stop_reason": body.get("stop_reason"),
        "usage": body.get("usage", {}),
    }
    capture_log.write(json.dumps(entry) + "\n")
    capture_log.flush()
```

Run the proxy:

```bash
mitmdump -s save_api_requests.py --listen-port 8888
```

Configure your application to route API traffic through the proxy, then run the capture through AIPerf:

```bash
aiperf profile \
    --model claude-opus-4-6 \
    --endpoint-type chat \
    --input-file ./my_capture/ \
    --url localhost:8080 \
    --streaming \
    --concurrency 1
```

### From Custom Middleware

If you have server-side middleware or logging, format the output to match the expected directory structure. The minimum requirements are:

1. `capture.jsonl` with `call_index`, `direction`, and `timestamp` for each request/response
2. `req_XXXX.json` files with `messages` (required) and optionally `system`, `tools`, `model`, `max_tokens`, `stream`, `thinking`
3. Response `usage` data in `capture.jsonl` (optional, used for token count metadata)

---

## Multi-Thread vs. Single-Thread Sessions

### Single-Thread Session

A session where all requests share the same system prompt produces one conversation:

```
my_capture/
  capture.jsonl
  req_0000.json   # system: [{"type": "text", "text": "You are a helpful assistant."}]
  req_0001.json   # system: [{"type": "text", "text": "You are a helpful assistant."}]
  req_0002.json   # system: [{"type": "text", "text": "You are a helpful assistant."}]
```

Result: 1 conversation with 3 turns, `agent_depth=0`.

### Multi-Thread Session (Agentic)

A session where different system prompts appear produces a parent/child hierarchy. For example, an agentic coding tool that spawns subagents:

```
my_capture/
  capture.jsonl
  req_0000.json   # system: "Expert coding assistant..." (parent)
  req_0001.json   # system: "Expert coding assistant..." (parent)
  req_0002.json   # system: "Research subagent..."       (child A)
  req_0003.json   # system: "Expert coding assistant..." (parent)
  req_0004.json   # system: "Research subagent..."       (child A)
  req_0005.json   # system: "Classification task..."     (child B)
  req_0006.json   # system: "Expert coding assistant..." (parent)
```

Result:
- Parent conversation: 4 turns (`req_0000`, `req_0001`, `req_0003`, `req_0006`), `agent_depth=0`
- Child A conversation: 2 turns (`req_0002`, `req_0004`), `agent_depth=1`, spawned as background
- Child B conversation: 1 turn (`req_0005`), `agent_depth=1`, spawned as background

### Independent Requests

Requests without system prompts are treated as independent single-turn conversations:

```
req_0010.json   # system: [] (empty)
req_0011.json   # system: [] (empty)
```

Result: 2 independent conversations, each with 1 turn.

---

## Tips

- **Response metadata is optional.** If `capture.jsonl` has no response entries, the loader still works. Token counts default to 0 and stop reason defaults to `null`.
- **Growing message arrays are expected.** In agentic workloads, each turn's `messages` array includes the full conversation history up to that point. The loader preserves this structure in `raw_payload` for verbatim replay.
- **Prefetch requests are filtered automatically.** You do not need to remove health-check or quota-probe requests (those with `max_tokens` of 0 or 1) from your capture data.
- **Multiple sessions in one directory are fine.** If the proxy was restarted mid-capture, only the last session (detected by `call_index` resets) is used.
- **Tools are set per-conversation, not per-turn.** The tool definitions from the first API call in each thread are used as the conversation-level `tools` field.
- **System prompt text is extracted for convenience.** The `system_prompt_text` on each trace joins all non-billing text blocks with newlines. The full structured `system` blocks are preserved in each turn's `raw_payload`.
- **Use `--concurrency 1` for faithful replay.** When replaying a single captured session, use concurrency 1 to match the original request pattern. Use higher concurrency or adaptive scaling to stress-test the server.
