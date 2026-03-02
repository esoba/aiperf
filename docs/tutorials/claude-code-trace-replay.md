<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Claude Code Trace Replay

Replay Claude Code session transcripts against an LLM inference server to benchmark real-world agentic coding workloads.

## Overview

The `claude_code_trace` dataset loader reads JSONL session transcripts exported from Claude Code (typically found at `~/.claude/projects/.../sessions/*.jsonl`) and converts them into AIPerf conversations. Each transcript captures a full coding session: user messages, assistant responses with tool calls and thinking blocks, token usage, and timestamps.

This loader supports two modes:

| Mode | Description | Response Handling |
|------|-------------|-------------------|
| **Verbatim** (default) | Sends exact content blocks (`tool_use`, `tool_result`, `text`, `thinking`) to the server via `raw_messages` on each Turn | Discards server responses (`discard_responses=True`) |
| **Synthetic** | Extracts token counts from the trace and generates synthetic content using a `PromptGenerator` | Processes server responses normally |

**When to use this loader:**

- Measure server performance under realistic agentic coding traffic patterns
- Replay recorded sessions with original inter-turn timing
- Test KV cache behavior with real tool-use conversation structures
- Benchmark multi-agent workloads with parent/subagent session linking

---

## File Format

### JSONL Record Structure

Each line in the JSONL file is a JSON object with the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `string` | Yes | Record type: `user`, `assistant`, or `system` |
| `message` | `object` | Yes | Message payload containing `content`, `usage`, `model`, etc. |
| `sessionId` | `string` | No | Claude Code session identifier |
| `timestamp` | `string` | No | ISO 8601 timestamp (e.g., `2025-06-01T10:00:01Z`) |
| `requestId` | `string` | No | Groups assistant records from the same API response |
| `uuid` | `string` | No | Unique identifier for this record |
| `parentUuid` | `string` | No | Parent record UUID for threading |

### Record Types

**System records** contain the system prompt:

```json
{"type": "system", "message": {"content": "You are Claude Code, an AI coding assistant."}, "sessionId": "sess-abc", "timestamp": "2025-06-01T10:00:00Z"}
```

**User records** contain user messages or tool results. Content can be a plain string or a list of content blocks:

```json
{"type": "user", "message": {"content": "Refactor the auth module."}, "sessionId": "sess-abc", "timestamp": "2025-06-01T10:00:01Z", "requestId": "req-001"}
```

```json
{"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "tu-read1", "content": "class AuthHandler: ..."}]}, "sessionId": "sess-abc", "timestamp": "2025-06-01T10:00:05Z", "requestId": "req-002"}
```

**Assistant records** contain the model response with content blocks and usage metadata:

```json
{"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "Let me read the file first."}, {"type": "tool_use", "id": "tu-read1", "name": "Read", "input": {"file_path": "src/auth.py"}}], "model": "claude-sonnet-4-20250514", "usage": {"input_tokens": 800, "output_tokens": 120, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, "stop_reason": "tool_use"}, "sessionId": "sess-abc", "timestamp": "2025-06-01T10:00:04Z", "requestId": "req-001"}
```

### Content Block Types

Assistant response content blocks can include any of the following types:

| Block Type | Description |
|------------|-------------|
| `text` | Plain text response from the model |
| `tool_use` | Tool invocation with `id`, `name`, and `input` fields |
| `tool_result` | Result returned from a tool execution (in user messages) |
| `thinking` | Extended thinking / chain-of-thought reasoning |

### How Records Are Grouped

The loader pairs user and assistant records into reconstructed API calls:

1. A `user` record starts a new API call
2. Subsequent `assistant` records with the same `requestId` are merged into a single response (handling streamed content blocks)
3. If a new `requestId` appears on an assistant record, the previous pair is flushed
4. `system` records are used to extract the system prompt but do not produce API calls
5. Usage fields (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`) are merged by taking the maximum across streamed chunks

### Timestamps and Delays

When records include ISO 8601 timestamps, the loader computes inter-turn delays automatically. Each Turn's `delay` is the time difference in milliseconds between consecutive user record timestamps. Negative deltas (from out-of-order timestamps) are discarded.

---

## Subagent Linking with Manifests

When loading from a directory, the loader looks for a `_manifest.json` file that declares parent/child session relationships. This enables realistic multi-agent replay where subagent sessions run alongside the parent session.

### Manifest Format

```json
{
  "parent": "parent.jsonl",
  "subagents": [
    {
      "file": "child.jsonl",
      "spawn_after_api_call": 1,
      "is_background": false
    }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `parent` | `string` | Yes | JSONL filename for the parent session |
| `subagents` | `array` | No | List of subagent link objects (defaults to empty) |
| `subagents[].file` | `string` | Yes | JSONL filename for the subagent session |
| `subagents[].spawn_after_api_call` | `int` | Yes | 0-based index of the parent API call after which this subagent spawns |
| `subagents[].is_background` | `bool` | No | If `true`, the parent continued without waiting for this subagent (default: `false`) |

### How Subagents Are Replayed

- The parent conversation gets `SubagentSpawnInfo` entries attached at the appropriate turn indices
- Child conversations have `agent_depth=1` (parent has `agent_depth=0`)
- The `join_turn_index` is set to `spawn_after_api_call + 1` (clamped to the last turn)
- Background subagents (`is_background: true`) allow the parent to proceed without blocking

### Directory Structure Example

```
my-session/
  _manifest.json
  parent.jsonl
  task-runner.jsonl
  code-reviewer.jsonl
```

With manifest:

```json
{
  "parent": "parent.jsonl",
  "subagents": [
    {"file": "task-runner.jsonl", "spawn_after_api_call": 2, "is_background": false},
    {"file": "code-reviewer.jsonl", "spawn_after_api_call": 3, "is_background": true}
  ]
}
```

---

## Basic Usage

### Single Session File

Load a single JSONL transcript. The dataset type is auto-detected from the file contents (the loader probes the first 5 lines for records with `type` in `user`/`assistant`/`system` and a `message` field):

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type chat \
    --input-file ~/session.jsonl \
    --streaming \
    --url localhost:8080
```

You can also explicitly set the dataset type:

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type chat \
    --input-file ~/session.jsonl \
    --custom-dataset-type claude_code_trace \
    --streaming \
    --url localhost:8080
```

### Directory of Sessions

Point `--input-file` at a directory containing multiple JSONL files. Without a manifest, each file becomes an independent conversation:

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type chat \
    --input-file ~/claude-sessions/ \
    --streaming \
    --url localhost:8080
```

### Directory with Manifest (Parent + Subagents)

When the directory contains a `_manifest.json`, the loader uses it to link parent and child sessions:

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type chat \
    --input-file ~/my-session/ \
    --streaming \
    --url localhost:8080
```

---

## Adaptive Scaling

Claude Code trace replay pairs naturally with adaptive scaling, which ramps up concurrent users while monitoring TTFT. Use `--adaptive-scale` to enable it:

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type chat \
    --input-file ~/claude-sessions/ \
    --streaming \
    --url localhost:8080 \
    --adaptive-scale \
    --adaptive-scale-start-users 1 \
    --adaptive-scale-max-users 50 \
    --adaptive-scale-max-ttft 2.0 \
    --adaptive-scale-recycle \
    --benchmark-duration 300
```

Key options for trace replay with adaptive scaling:

| Option | Default | Description |
|--------|---------|-------------|
| `--adaptive-scale` | `false` | Enable adaptive user scaling based on TTFT headroom |
| `--adaptive-scale-start-users` | `1` | Initial concurrent users |
| `--adaptive-scale-max-users` | `50` | Maximum concurrent users |
| `--adaptive-scale-max-ttft` | `2.0` | Maximum TTFT threshold in seconds; scaling stops when exceeded |
| `--adaptive-scale-ttft-metric` | `p95` | TTFT metric for scaling decisions: `p95`, `avg`, or `max` |
| `--adaptive-scale-recycle` | `false` | Re-sample completed sessions (required for long benchmarks with few traces) |
| `--adaptive-scale-max-delay` | `60.0` | Maximum inter-request delay in seconds; longer trace delays are clamped |
| `--adaptive-scale-time-scale` | `1.0` | Time scale factor for trace delays; `< 1.0` compresses, `> 1.0` stretches |
| `--adaptive-scale-assessment-period` | `30.0` | Seconds between scaling assessments |
| `--adaptive-scale-formula` | `conservative` | Scaling formula: `conservative`, `aggressive`, or `linear` |

---

## Dataset Sampling

The `claude_code_trace` loader defaults to `sequential` sampling (conversations replay in order, wrapping after the last one). Override with `--dataset-sampling-strategy`:

```bash
# Shuffle conversations before replay
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type chat \
    --input-file ~/claude-sessions/ \
    --dataset-sampling-strategy shuffle \
    --streaming \
    --url localhost:8080
```

Available strategies:

| Strategy | Behavior |
|----------|----------|
| `sequential` (default for traces) | Iterate in order, wrap to start after end |
| `random` | Random sampling with replacement |
| `shuffle` | Shuffle and iterate without replacement, re-shuffle after exhaustion |

---

## Tips and Common Patterns

### Exporting Sessions from Claude Code

Claude Code stores session transcripts as JSONL files under `~/.claude/projects/<project-hash>/sessions/`. Each file represents one coding session:

```bash
# Find recent session files
ls -lt ~/.claude/projects/*/sessions/*.jsonl | head -20

# Copy a session for replay
cp ~/.claude/projects/<hash>/sessions/<session-id>.jsonl ~/trace.jsonl
```

### Building a Manifest for Multi-Agent Sessions

If your session spawned subagents, you need to identify which JSONL files correspond to child sessions and at which parent turn they were spawned. Create a `_manifest.json` linking them:

```bash
mkdir ~/my-session
cp parent-session.jsonl ~/my-session/parent.jsonl
cp child-session.jsonl ~/my-session/child.jsonl

cat > ~/my-session/_manifest.json << 'EOF'
{
  "parent": "parent.jsonl",
  "subagents": [
    {"file": "child.jsonl", "spawn_after_api_call": 1}
  ]
}
EOF
```

### Combining Multiple Sessions in a Directory

Place multiple independent JSONL files in a directory (without a manifest) to replay them as separate conversations. The loader sorts files alphabetically and loads each one:

```bash
mkdir ~/sessions
cp session-a.jsonl ~/sessions/
cp session-b.jsonl ~/sessions/
cp session-c.jsonl ~/sessions/

aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type chat \
    --input-file ~/sessions/ \
    --streaming \
    --url localhost:8080
```

### Verbatim vs. Synthetic Mode

In verbatim mode (the default), the loader sends the exact content blocks from the transcript as `raw_messages` on each Turn and sets `discard_responses=True` on the Conversation. The server receives the real tool-use and tool-result messages but its responses are not stored in conversation history.

Synthetic mode is activated internally when a `PromptGenerator` is provided to the loader. In this mode, token counts from the trace drive synthetic prompt generation while preserving the original timing pattern. This is useful when you want to test with realistic token distributions without sending the actual session content.

### Trace ID Format

Each loaded trace gets an ID of the form `cc_<filename_stem>`. For example, `parent.jsonl` becomes `cc_parent`. These IDs appear in logs and are used as `session_id` values on the resulting conversations.
