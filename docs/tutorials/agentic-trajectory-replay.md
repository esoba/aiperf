<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic Trajectory Replay

Benchmark LLM inference servers by replaying recorded multi-turn tool-calling conversations with cumulative message history.

## Overview

The `agentic_trajectory` dataset loader replays captured agentic conversations verbatim. Each JSONL line represents one API call with the **full cumulative message history** up to that point in the conversation. This is the format you get when you record an AI coding assistant or tool-calling agent making successive LLM calls, where each call includes the entire conversation so far.

### When to Use

- **Agent workload simulation**: Replay recorded tool-calling sessions (coding assistants, research agents, etc.) against your inference server
- **KV cache stress testing**: Cumulative message histories grow with each turn, exercising prefix caching and memory management
- **Multi-turn fidelity**: Each turn carries the exact messages the original agent sent, including assistant responses, tool calls, and tool results
- **Regression benchmarking**: Compare inference server performance across versions using the same recorded trajectories

### How It Differs from Other Loaders

| Loader | Input Format | History Model | Payload Construction |
|--------|-------------|---------------|---------------------|
| **agentic_trajectory** | JSONL with cumulative messages per turn | Each turn replaces full history | Pre-formatted via endpoint |
| **multi_turn** | JSONL with per-turn text/media | Turns accumulate incrementally | Built from Turn text/media fields |
| **raw_payload** | JSONL with complete API request bodies | N/A (single-turn) | Sent verbatim, no formatting |
| **inputs_json** | AIPerf inputs.json with pre-built payloads | Per-session payload lists | Sent verbatim from inputs.json |

The key distinction: `agentic_trajectory` groups JSONL records by `conversation_id`, sorts by `conversation_idx`, and converts each record's cumulative `messages` array into a turn with `replaces_history=True`. The endpoint then formats these messages into proper API payloads during a pre-formatting step.

---

## File Format

The input is a JSONL file (`.jsonl`) where each line is a JSON object with these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `conversation_id` | `str` | Yes | Groups records into conversations |
| `conversation_idx` | `int` | Yes | 0-based turn index within the conversation |
| `messages` | `list[dict]` | Yes | Cumulative message history for this turn |
| `tools` | `list[dict]` | No | Tool definitions (defaults to `[]` if omitted) |

### Cumulative vs Delta Messages

Each record's `messages` array contains the **full conversation history** up to that turn, not just the new messages. This is the cumulative model:

```
Turn 0 messages: [system, user_0]
Turn 1 messages: [system, user_0, assistant_0, tool_result_0, user_1]
Turn 2 messages: [system, user_0, assistant_0, tool_result_0, user_1, assistant_1, user_2]
```

The loader strips leading system messages from each turn's `messages` and stores them separately on the `Conversation.system_message` field. The remaining non-system messages become the turn's `raw_messages` with `replaces_history=True`, meaning each turn replaces the entire prior history rather than appending to it.

Server responses are discarded (`discard_responses=True` on each conversation) because each turn already carries its own full context from the recorded trajectory.

---

## Example JSONL File

```jsonl
{"conversation_id": "session-1", "conversation_idx": 0, "messages": [{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": "Read the file src/main.py"}], "tools": [{"name": "read_file", "description": "Read a file from disk"}]}
{"conversation_id": "session-1", "conversation_idx": 1, "messages": [{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": "Read the file src/main.py"}, {"role": "assistant", "content": [{"type": "text", "text": "I'll read that file."}], "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "def main(): pass"}], "tools": [{"name": "read_file", "description": "Read a file from disk"}]}
{"conversation_id": "session-2", "conversation_idx": 0, "messages": [{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": "What does this project do?"}]}
```

This file contains two conversations: `session-1` with 2 turns and `session-2` with 1 turn. Records can appear in any order in the file -- the loader groups by `conversation_id` and sorts by `conversation_idx`.

---

## Basic Usage

The loader is auto-detected when the JSONL file contains `conversation_id` (string), `conversation_idx` (integer), and `messages` (list) fields. You can also specify it explicitly.

### Auto-Detection

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --input-file trajectories.jsonl \
    --streaming \
    --url localhost:8000 \
    --concurrency 4
```

### Explicit Dataset Type

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --input-file trajectories.jsonl \
    --custom-dataset-type agentic_trajectory \
    --streaming \
    --url localhost:8000 \
    --concurrency 4
```

### With Agentic Load Mode

The `agentic_trajectory` loader pairs naturally with `--agentic-load` mode, which pre-assigns conversations to simulated users and runs each user in a closed loop:

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --input-file trajectories.jsonl \
    --custom-dataset-type agentic_trajectory \
    --agentic-load \
    --num-users 10 \
    --benchmark-duration 300 \
    --agentic-trajectories-per-user 20 \
    --streaming \
    --url localhost:8000
```

---

## Configuration Options

### Dataset Sampling Strategy

The loader's preferred sampling strategy is `sequential`. You can override this:

```bash
--dataset-sampling-strategy sequential  # Default for this loader
--dataset-sampling-strategy shuffle     # Shuffle conversations, iterate without replacement
--dataset-sampling-strategy random      # Random sampling with replacement
```

### Agentic Load Options

When using `--agentic-load`, these options control user behavior:

| Option | Default | Description |
|--------|---------|-------------|
| `--agentic-load` | `false` | Enable closed-loop agentic load mode |
| `--agentic-trajectories-per-user` | `20` | Conversations assigned to each user |
| `--agentic-max-isl-offset` | `10` | Max turns to skip in first conversation per user |

### Extra Inputs

Add parameters to every API request:

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --input-file trajectories.jsonl \
    --extra-inputs temperature:0.0 max_tokens:4096 \
    --streaming \
    --url localhost:8000 \
    --concurrency 4
```

---

## How the Loader Works

The loading pipeline has two stages:

### 1. Load and Group

The `load_dataset()` method reads the JSONL file line by line, parses each record into an `AgenticTrajectoryRecord`, groups records by `conversation_id`, and sorts each group by `conversation_idx`.

### 2. Convert to Conversations

The `convert_to_conversations()` method transforms each group into a `Conversation`:

1. **System message extraction**: Leading `system` role messages from the first turn's `messages` are extracted and joined with newlines into `Conversation.system_message`. Both string content and structured content blocks (text type) are supported. Multiple leading system messages are concatenated.

2. **Tool definitions**: The first record with a non-empty `tools` list provides the conversation's tool definitions stored on `Conversation.tools`.

3. **Turn creation**: For each record, leading system messages are stripped from `messages`, and the remaining messages become `Turn.raw_messages` with `replaces_history=True`.

4. **Response discarding**: `discard_responses=True` is set on each conversation because turns carry their own cumulative context.

### 3. Payload Pre-Formatting

After conversion, the custom dataset composer detects that `agentic_trajectory` has `preformat_payload: true` in its plugin metadata. It then runs each turn through the endpoint's `format_payload()` method, storing the result as `turn.raw_payload`. This means workers send the pre-built payloads directly to the server without any per-request formatting overhead.

---

## Preparing Trajectory Data

### From API Logs

If you have API call logs, convert them to the cumulative format:

```python
import json

# Each entry is one API call with its full messages array
api_calls = [
    {
        "conversation_id": "session-001",
        "turn_index": 0,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
    },
    {
        "conversation_id": "session-001",
        "turn_index": 1,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Write a function to sort a list"},
        ],
    },
]

with open("trajectories.jsonl", "w") as f:
    for call in api_calls:
        record = {
            "conversation_id": call["conversation_id"],
            "conversation_idx": call["turn_index"],
            "messages": call["messages"],
        }
        f.write(json.dumps(record) + "\n")
```

### Validation Checklist

Before running a benchmark, verify your JSONL file:

- Every line has `conversation_id` (string), `conversation_idx` (integer), and `messages` (list)
- `conversation_idx` values are 0-based and contiguous within each `conversation_id`
- Each turn's `messages` array is cumulative (turn N contains all messages from turns 0 through N)
- System messages appear at the start of the `messages` array (not interleaved)
- The file extension is `.jsonl`

---

## Tips and Common Patterns

**Interleaved conversations**: Records from different conversations can appear in any order in the JSONL file. The loader groups by `conversation_id` and sorts by `conversation_idx`, so you do not need to pre-sort the file.

**Tool definitions on any turn**: The `tools` field can appear on any record. The loader uses the first record (by `conversation_idx` order) that has a non-empty `tools` list. If no records have tools, `Conversation.tools` is `None`.

**Multiple system messages**: If the first turn has multiple leading system-role messages, they are concatenated with newlines. Structured content blocks (list of `{"type": "text", "text": "..."}` dicts) are also supported.

**Empty lines**: Blank lines in the JSONL file are silently skipped.

**Missing tools key**: If a record omits the `tools` key entirely, it defaults to an empty list `[]`.

**Reproducibility**: Use `--random-seed` for deterministic sampling when combined with `shuffle` or `random` sampling strategies:

```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --input-file trajectories.jsonl \
    --dataset-sampling-strategy shuffle \
    --random-seed 42 \
    --streaming \
    --url localhost:8000 \
    --concurrency 4
```
