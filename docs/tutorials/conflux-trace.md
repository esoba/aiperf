---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Replaying AI Coding Sessions
---

# Replaying AI Coding Sessions with Conflux Traces

Benchmark your LLM inference server using real-world traffic captured from AI coding assistants like Claude Code and OpenAI Codex. Conflux trace files record every API call made during a coding session, including the full conversation history, tool definitions, model parameters, and timing data. AIPerf replays these traces against your server to measure how it handles realistic agentic workloads.

---

## What Is a Conflux Trace?

When you use an AI coding assistant, it makes many API calls behind the scenes. A single user request ("fix this bug") can trigger a chain of calls: the main agent reasons about the problem, spawns subagents to search code or run tests, and each of those subagents makes its own API calls. A Conflux trace captures all of this activity.

Each trace file is a JSON array where every element represents one API call. The key fields are:

| Field | What it tells AIPerf |
|---|---|
| `messages` | The full conversation sent to the model (system prompt, user messages, tool results) |
| `tools` | Tool definitions available to the model (file search, code execution, etc.) |
| `model` | Which model was used (e.g. `claude-opus-4-6`, `gpt-4o`) |
| `agent_id` | Which agent thread made this call |
| `is_subagent` | Whether this was a background subagent or the main agent |
| `timestamp` | When the call was made (used for replay timing) |
| `tokens` | How many tokens were used (input, output, cached, reasoning) |
| `hyperparameters` | Generation settings (temperature, top_p, max_tokens, etc.) |

---

## How Agents and Subagents Work

A coding session typically has a tree-shaped structure:

```
Parent Agent (main conversation)
  Turn 1: User asks "refactor the auth module"
  Turn 2: Agent reads files, thinks about approach
    +-- Subagent A: searches codebase for auth references
    +-- Subagent B: runs existing tests to check current behavior
  Turn 3: Agent writes new code
    +-- Subagent C: runs updated tests
  Turn 4: Agent summarizes changes
```

The **parent agent** is the main conversation thread that the user interacts with. It has multiple turns, each representing one API call in sequence.

**Subagents** are background tasks spawned by the parent. When the coding assistant needs to search files or run a command, it often launches a separate agent to handle that work in parallel. Each subagent has its own conversation thread with its own API calls.

AIPerf preserves this structure during replay:

1. Each agent thread becomes a separate **conversation** with its own sequence of turns
2. Subagent conversations are linked to the parent via **spawn points** and, for blocking work, a later **join point** where the parent consumes the child result
3. Turns within each conversation replay in order with the original timing gaps between them

There are also **utility calls** -- lightweight API calls for housekeeping tasks like generating conversation titles or detecting topics. These lack an `agent_id` and are excluded by default since they happen outside the main coding workflow.

---

## Trace File Format

A minimal Conflux trace looks like this:

```json
[
  {
    "id": "req_abc123",
    "session_id": "sess_001",
    "agent_id": "agent_main",
    "is_subagent": false,
    "timestamp": "2026-03-15T10:00:00Z",
    "model": "claude-opus-4-6",
    "messages": [
      {"role": "system", "content": "You are a coding assistant."},
      {"role": "user", "content": "Fix the login bug in auth.py"}
    ],
    "tools": [
      {"type": "function", "function": {"name": "read_file", "parameters": {}}}
    ],
    "tokens": {"input": 1500, "output": 800, "input_cached": 200},
    "duration_ms": 3200,
    "hyperparameters": {"temperature": 1.0, "max_tokens": 4096}
  },
  {
    "id": "req_def456",
    "session_id": "sess_001",
    "agent_id": "agent_search_1",
    "is_subagent": true,
    "timestamp": "2026-03-15T10:00:01.500Z",
    "model": "claude-opus-4-6",
    "messages": [
      {"role": "user", "content": "Search for all files importing auth module"}
    ],
    "tokens": {"input": 500, "output": 300},
    "duration_ms": 1100
  }
]
```

The first record is a parent agent turn. The second is a subagent that was spawned 1.5 seconds into the parent's first turn. AIPerf detects this overlap automatically and links them together.

For highest fidelity, traces can include a `base64` field containing the raw request body. When present, AIPerf uses it instead of the top-level `messages` and `tools` fields, preserving every detail of the original API call exactly as it was sent.

---

## Getting Started

### Auto-Detection

AIPerf automatically detects Conflux format when your JSON file contains records with `messages` and either `agent_id` + `is_subagent` fields or `source: "proxy"`. No `--custom-dataset-type` flag is needed:

```bash
aiperf profile \
    --url localhost:8000 \
    --model your-model-name \
    --endpoint-type chat \
    --streaming \
    --input-file my-session.json \
    --fixed-schedule
```

You can also be explicit:

```bash
aiperf profile \
    --url localhost:8000 \
    --model your-model-name \
    --endpoint-type chat \
    --streaming \
    --input-file my-session.json \
    --custom-dataset-type conflux \
    --fixed-schedule
```

<Warning>
Use `--fixed-schedule` to replay requests at their original timestamps. This is automatically enabled when you specify `--custom-dataset-type conflux`, but is recommended when relying on auto-detection.
</Warning>

### Speed Up or Slow Down Replay

Real coding sessions can last minutes or hours. Use `--fixed-schedule-speedup` to compress or expand the timeline:

```bash
# Replay at 10x speed (a 10-minute session completes in 1 minute)
aiperf profile \
    --url localhost:8000 \
    --model your-model-name \
    --endpoint-type chat \
    --streaming \
    --input-file my-session.json \
    --fixed-schedule \
    --fixed-schedule-speedup 10.0
```

A value of `2.0` replays at double speed, `0.5` replays at half speed (useful for stress testing at lower rates). This works with any dataset that uses `--fixed-schedule`, not just Conflux traces.

### Include Utility Calls

By default, AIPerf skips utility calls (API requests without an `agent_id`, typically used for title generation or topic detection). To include them:

```bash
aiperf profile \
    --url localhost:8000 \
    --model your-model-name \
    --endpoint-type chat \
    --streaming \
    --input-file my-session.json \
    --fixed-schedule \
    --conflux-include-utility-calls
```

---

## What Gets Replayed

For each API call in the trace, AIPerf sends:

- **Messages**: The full conversation history (system prompt, user messages, assistant responses, tool results) normalized to OpenAI-compatible format
- **Tools**: All tool definitions that were available to the model
- **Model**: The model identifier from the trace (override with `--model` if targeting a different model)
- **Hyperparameters**: Per-turn settings like temperature, top_p, and reasoning effort, sent as request parameters
- **Timing**: Original inter-request delays preserved via `--fixed-schedule`, including later subagent joins when a child result is consumed several parent turns after it was spawned
- **Max tokens**: The generation limit from the original request

AIPerf also records **ground truth** metadata from the trace (original token counts, TTFT, duration) so you can compare your server's performance against the captured baseline.

---

## Spawn and Join Patterns

Across real Claude Code sessions, two distinct subagent coordination patterns appear:

### Blocking immediate (gap = 1 parent turn)

The parent spawns agents via the `Agent` tool and the results come back inline — the tool_result containing the child's output is already present in the cumulative message history by the time the parent makes its next API call. AIPerf detects this by finding the first parent turn whose new messages contain a non-acknowledgement tool_result for the spawn's `Agent` tool_use ID.

```
Parent turn N:   Agent tool_use  ──► child runs
                 tool_result: "child output here"
Parent turn N+1: JOIN — reads child result, continues
```

This is the most common pattern. Children often take longer to complete than the gap implies — the parent reads an early partial result or the child was fast enough. The join turn is N+1 regardless of child wall-clock completion time.

### Blocking notification (gap = 1 to 183 parent turns)

The parent spawns agents asynchronously (`"Async agent launched successfully"` acknowledgement) and continues doing other work. When each child finishes, Claude Code injects a `<task-notification>` user message into the parent's context:

```xml
<task-notification>
  <task-id>ae2c5aa3243b040ea</task-id>
  <tool-use-id>toolu_019BxppEYd3m7AaX5Hc8BLJq</tool-use-id>
  <output-file>/tmp/.../tasks/ae2c5aa3243b040ea.output</output-file>
</task-notification>
```

The `tool-use-id` matches the original `Agent` tool_use, allowing AIPerf to identify which spawn each notification belongs to. Each child gets its own join turn — the first parent turn where its notification appears.

Two sub-cases observed in the wild:

- **Parent idle** (e.g. pr-review): parent had nothing else to do, so each child's completion immediately triggered a new parent turn (~135 ms gap). Four children → four separate join turns spaced 6–9 parent turns after spawn.
- **Parent busy** (e.g. long-horizon): parent continued active work (182 turns of code porting across files) while children ran in parallel. Notifications all arrived at the final summary turn once the parent finished its own work.

### Join gate behavior during replay

When replaying, AIPerf fires the join turn at:

```
join_dispatch = max(last_child_end, ideal_join_timestamp) + ~few ms
```

Whether the gate or the original schedule is the binding constraint depends on how
fast the server responds relative to the inter-turn gaps.

```
GATE-bound  (slow server: response time > inter-turn gap, children can't keep up)

  PARENT  ──[spawn]──────────────────────────────[gate: waiting]──[JOIN]──▶
                │                                                     ↑
  CHILD         └──[=======================================done]──────┘
                                          ↑
               ideal join timestamp ──────┘  (already passed, irrelevant)


SCHED-bound  (fast server: response time < inter-turn gap, children finish early)

  PARENT  ──[spawn]──────────────────[sched wait]──[JOIN]──▶
                │                                ↑
  CHILD         └──[================done]........│
                          child early,           │
                      wait for schedule          │
                                                 │
               ideal join timestamp ─────────────┘
```

With a slow server, children cannot finish within the inter-turn gaps and the gate
drives the join. With a fast server, children finish with time to spare and the join
waits for the original timestamp. The crossover point is when
`server_response_time ≈ original_inter_turn_gap / speedup`.

### Why timestamps alone are insufficient

For idle-parent notification joins, timestamp detection (first parent turn starting after child ends) gives the correct answer since the gap is ~135 ms. For busy-parent notification joins it gives the wrong turn — the parent was actively running and the notification arrived much later. AIPerf uses `<task-notification>` content matching as the authoritative signal.

---

## Understanding the Output

When AIPerf loads a Conflux trace, it reports the conversation structure:

```
Loaded 3 agent threads + 5 utility calls skipped (28 total records)
Converted 3 conversations (28 total turns, 2 subagent children incl. 0 orphans)
```

This tells you:
- **3 agent threads**: one parent + two subagents
- **5 utility calls skipped**: housekeeping calls excluded (use `--conflux-include-utility-calls` to include them)
- **3 conversations**: the parent and its two children, each with their own turn sequence

After the benchmark completes, the standard AIPerf metrics apply: throughput, latency percentiles, time-to-first-token, and inter-token latency, all measured under the realistic traffic pattern from the original session.

---

## Related Tutorials

- [Profile with Bailian Traces](bailian-trace.md) - Replay Alibaba production traces
- [Trace Replay with Mooncake Traces](../benchmark-modes/trace-replay.md) - Mooncake FAST'25 trace replay
- [Fixed Schedule](fixed-schedule.md) - Precise timestamp-based execution for any dataset
- [Multi-Turn Conversations](multi-turn.md) - Multi-turn conversation benchmarking
