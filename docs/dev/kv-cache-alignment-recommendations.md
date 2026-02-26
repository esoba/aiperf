<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Recommended Changes: Aligning AIPerf with kv-cache-tester

Actionable recommendations for improving AIPerf's coding trace replay to match and
exceed the kv-cache-tester reference implementation. Organized by priority (P0 = critical
accuracy issues, P1 = important missing features, P2 = nice-to-have improvements).

Reference: [`kv-cache-tester-comparison.md`](kv-cache-tester-comparison.md) for the
detailed feature-by-feature comparison this document builds on.

---

## P0: Accuracy Fixes

These affect the correctness of benchmark results. Without them, AIPerf's coding trace
replay produces measurements that don't reflect the intended workload.

### P0-1: Runtime Response Token Accounting (Shortfall Tracking)

**Problem:** AIPerf computes deltas at load time using `prev.output_tokens` from the trace.
If the model generates fewer tokens than expected, cumulative context drifts shorter than
intended. Over a 20-turn conversation, a consistent 20% undergeneration compounds into
a ~60% context deficit by the final turn.

**kv-cache-tester approach:** `UserSession.record_shortfall()` fires when actual output is
< 80% of expected. The full deficit (`expected - actual`) carries forward as
`token_shortfall` and is added to the next turn's user message budget. Reset to 0 after
each compensation.

**Recommended implementation:**

1. Track `stored_response_tokens` per session in `UserSession` (actual token count from
   the model's response, not the trace's expected count).
2. After each response, compare actual vs expected. If `actual < expected * 0.8`, set
   `shortfall = expected - actual`.
3. On the next turn, increase the user message size by `shortfall` tokens.
4. This requires **runtime prompt generation** for compensating turns, not pre-computed
   prompts. Two options:
   - **Option A (minimal change):** Pre-compute prompts at `delta + max_possible_shortfall`
     and truncate at runtime based on actual shortfall. Wastes some mmap space but
     preserves the pre-computation architecture.
   - **Option B (full runtime):** Generate prompts at request time like kv-cache-tester.
     More accurate but requires tokenizer access in workers (already available via
     `InferenceClient.endpoint.tokenizer`).

**Files to modify:**
- `src/aiperf/workers/session_manager.py` — add `stored_response_tokens: int` and
  `token_shortfall: int` to `UserSession`
- `src/aiperf/workers/worker.py` — after `store_response()`, count actual tokens and
  call `record_shortfall(expected, actual)`
- `src/aiperf/dataset/loader/coding_trace.py` — if Option A, generate oversized prompts;
  if Option B, defer prompt generation to runtime
- `src/aiperf/dataset/loader/models.py` — store expected `output_tokens` per turn in
  `Conversation` metadata for runtime comparison

**Complexity:** Medium (Option A) / High (Option B)

---

### P0-2: First-Turn Token Accounting with Warm Prefix

**Problem:** AIPerf adds the warm prefix as `conversation.system_message` on top of the
full first-turn delta. The first turn's total context is `prefix_tokens + delta_tokens`,
overshooting the trace's intended `input_tokens` by `prefix_tokens`.

**kv-cache-tester approach:** `build_messages()` subtracts `prefix_tokens` from the first
turn's content budget: `remaining_tokens = max(0, current_input_tokens - prefix_tokens)`.

**Recommended implementation:**

In `CodingTraceLoader.convert_to_conversations()`, when warm prefix is enabled, reduce the
first turn's delta:

```python
if i == 0 and prefix_tokens > 0:
    delta = max(1, req.input_tokens - prefix_tokens)
else:
    # normal delta computation
```

**Files to modify:**
- `src/aiperf/dataset/loader/coding_trace.py` — adjust first-turn delta in
  `convert_to_conversations()`

**Complexity:** Low

---

### P0-3: Subagent Timestamp Normalization and Sorting

**Problem:** AIPerf preserves original relative timestamps from nested subagents without
converting to absolute time. After flattening, requests are in parent-first order rather
than chronological order. This means inter-request delays are computed from unsorted
timestamps, producing incorrect timing.

**kv-cache-tester approach:** `flatten_requests()` converts subagent-relative timestamps
to absolute: `base_time + req['t']`, then sorts all flattened requests by absolute time.
Also normalizes all types to `'streaming'`.

**Recommended implementation:**

In `CodingTraceLoader._flatten_requests()`:
1. Pass `base_time` parameter for recursive calls
2. Convert each request's `t` to absolute: `t = base_time + req.t`
3. After flattening, sort by absolute `t`
4. Optionally normalize `type` to a single value (or keep for metadata)

**Files to modify:**
- `src/aiperf/dataset/loader/coding_trace.py` — update `_flatten_requests()` to accept
  `base_time`, convert timestamps, sort result

**Complexity:** Low

---

## P1: Important Missing Features

These are features that meaningfully improve the quality and realism of the benchmark.

### P1-1: Admission Control (Max Concurrent Requests)

**Problem:** AIPerf has no global cap on in-flight requests. Against fast mock servers,
adaptive scaling can launch hundreds of concurrent requests, overwhelming the benchmark
infrastructure rather than finding the server's TTFT ceiling.

**kv-cache-tester approach:** `--max-concurrent-requests` (default 50) gates every dispatch.
Tracks `in_flight_requests`, `in_flight_prefilling` (= in_flight - in_flight_decoding), and
`in_flight_decoding` separately. User scaling is also suppressed when at capacity.

**Recommended implementation:**

Add admission control to `AdaptiveScaleStrategy`:
1. New config: `--adaptive-scale-max-concurrent-requests` (default 50)
2. Track `in_flight_requests` counter, incremented on credit issue, decremented on
   `CreditReturn`
3. In `execute_phase()` / `handle_credit_return()`, check capacity before issuing next
   credit
4. Track prefill vs decode counts using existing `FirstToken` messages:
   `in_flight_prefilling = in_flight - in_flight_decoding`
5. In `_assess_and_scale()`, skip scaling when at capacity

**Files to modify:**
- `src/aiperf/timing/strategies/adaptive_scale.py` — add counter and gate logic
- `src/aiperf/timing/config.py` — add `max_concurrent_requests` to `CreditPhaseConfig`
- `src/aiperf/common/config/user_config.py` — add CLI option

**Complexity:** Medium

---

### P1-2: Working Set Budget

**Problem:** Without working set tracking, AIPerf can't control KV cache pressure. This
matters for benchmarks targeting specific cache utilization levels.

**kv-cache-tester approach:** `--max-working-set-tokens` (default 0 = unlimited). Tracks
union of all users' hash_ids * chunk_size. New users are rejected if estimated first-request
tokens would exceed the budget. Up to 5 traces are tried before giving up.

**Recommended implementation:**

1. Track global hash_id working set in `AdaptiveScaleStrategy` or a new
   `WorkingSetTracker` component
2. Each `CreditReturn` includes the conversation's hash_ids (already in trace metadata)
3. Before issuing a new session's first credit, estimate tokens from trace's first request
   hash_ids
4. Reject and try another trace if budget exceeded
5. New config: `--adaptive-scale-max-working-set-tokens` (default 0 = unlimited)

**Challenge:** AIPerf's multi-process architecture means hash_ids are distributed across
workers. Options:
- **Option A:** Track in TimingManager (single process). Workers send hash_id updates
  via message bus. Adds latency.
- **Option B:** Approximate using pre-computed per-trace estimates at load time. Less
  accurate but no runtime overhead.

**Files to modify:**
- New: `src/aiperf/timing/strategies/working_set_tracker.py`
- `src/aiperf/timing/strategies/adaptive_scale.py` — integrate tracker
- `src/aiperf/timing/config.py` — add config field
- `src/aiperf/common/config/user_config.py` — add CLI option
- `src/aiperf/dataset/loader/coding_trace.py` — compute per-trace first-request estimates

**Complexity:** Medium-High

---

### P1-3: Per-Period Token Budget

**Problem:** Without per-period token budgets, a burst of new users can flood the server
with cache-miss tokens in a single assessment period, causing temporary TTFT spikes that
don't reflect steady-state performance.

**kv-cache-tester approach:** `--max-new-tokens-per-period` (default 500K). Tracks
`period_new_tokens`, resets each period. New users are blocked if their first request's
`input_tokens` would exceed remaining budget. Initial users bypass this check.

**Recommended implementation:**

1. Add `period_new_tokens` counter to `AdaptiveScaleStrategy`
2. Reset at each `_assess_and_scale()` call
3. Before issuing a new session, check `first_request.input_tokens <= remaining_budget`
4. Initial `start_users` bypass the check
5. New config: `--adaptive-scale-max-new-tokens-per-period` (default 500000)

**Files to modify:**
- `src/aiperf/timing/strategies/adaptive_scale.py` — add budget tracking
- `src/aiperf/timing/config.py` — add config field
- `src/aiperf/common/config/user_config.py` — add CLI option

**Complexity:** Low-Medium

---

### P1-4: Connection Circuit Breaker

**Problem:** AIPerf continues sending requests even when the target server is down,
accumulating errors without stopping the benchmark. Wastes time and produces meaningless
results.

**kv-cache-tester approach:** 10 consecutive connection errors (classified by string
matching: "connection", "connect", "refused") triggers `self.running = False`. Counter
resets on any successful request.

**Recommended implementation:**

Add circuit breaker to `Worker`:
1. Track `consecutive_connection_errors` per worker (atomic counter)
2. On connection-class error (classify in `InferenceClient`), increment
3. On success, reset to 0
4. At threshold (configurable, default 10), publish `WorkerShutdown` with error reason
5. `WorkerManager` detects all workers in error state and publishes `ProfileCancel`

Alternative: implement at `WorkerManager` level by aggregating `WorkerHealth` messages
that report error streaks.

**Files to modify:**
- `src/aiperf/workers/worker.py` — add counter and classification logic
- `src/aiperf/workers/worker_manager.py` — aggregate error states, trigger cancel

**Complexity:** Low-Medium

---

### P1-5: Pull-Back Detection

**Problem:** AIPerf always appends to the conversation, even when the trace shows the
context shrinking (user abandoned a branch). This produces incorrect context sizes for
post-pull-back requests.

**kv-cache-tester approach:** Detects when >10% of previous hash_ids are removed. On
pull-back, clears the entire conversation and regenerates a single user message sized to
`input_tokens - kept_tokens` (the kept portion is assumed cached server-side).

**Recommended implementation:**

This requires runtime prompt generation (same as P0-1 Option B). If implementing P0-1
with Option B:
1. Store previous request's `hash_ids` in `UserSession`
2. On each turn, compute `removed = prev_hash_ids - current_hash_ids`
3. If `len(removed) > 0.1 * len(prev_hash_ids)`: pull-back detected
4. Clear `turn_list`, generate new user message at `input_tokens - kept_tokens`
5. Store `prev_hash_ids` for next comparison

If using Option A (pre-computed prompts), pull-back can be approximated by pre-computing
a "reset prompt" at the pull-back turn's full `input_tokens - estimated_kept_tokens`.

**Files to modify:**
- `src/aiperf/workers/session_manager.py` — add `prev_hash_ids` tracking
- `src/aiperf/workers/worker.py` — detect pull-back, reset conversation
- `src/aiperf/dataset/loader/coding_trace.py` — store hash_ids per turn in conversation
  metadata

**Complexity:** High (depends on P0-1)

---

### P1-6: Coding-Specific Prompt Content

**Problem:** Shakespeare text has a very different token distribution from real coding
sessions. This affects KV cache behavior (different attention patterns), model output
length (Shakespeare doesn't encourage long code responses), and tokenizer efficiency
(code tokens vs literary tokens).

**kv-cache-tester approach:** Two synthetic pools (500K tokens each):
- **User text pool:** 40% coding prompts + 30% technical questions + 30% tech vocabulary
- **Tool result pool:** 35% file contents (with line numbers) + 25% bash output +
  15% JSON structures + 10% error tracebacks + 15% file path listings

Plus a 20-question technical question bank appended to every user message to encourage
long model responses.

**Recommended implementation:**

Create a new `CodingPromptGenerator` as a plugin alternative to `PromptGenerator`:
1. Register as a plugin in `plugins.yaml` under the `generator` category
2. Generate two token pools at initialization (user text + tool results)
3. Route between pools based on `input_types` from the trace (`tool_result` -> tool pool)
4. Append a question from the bank to each user message
5. Use per-request seeding: `hash(f"{conv_id}_{turn_index}")` for deterministic content

**Files to modify:**
- New: `src/aiperf/dataset/generator/coding_prompt.py`
- `src/aiperf/plugin/plugins.yaml` — register new generator
- `src/aiperf/dataset/loader/coding_trace.py` — use `CodingPromptGenerator` when
  available (or configurable via `--prompt-generator coding`)

**Complexity:** Medium

---

### P1-7: Request Pair Detection

**Problem:** Coding traces contain consecutive requests with identical `hash_ids`
(streaming + non-streaming pairs). Without detection, AIPerf treats them as separate
turns, generating new user content and double-counting the assistant response.

**kv-cache-tester approach:** `build_messages()` detects identical hash_ids and re-sends
the same conversation. `store_assistant_response()` only stores after the non-streaming
(second) request.

**Recommended implementation:**

At load time in `convert_to_conversations()`:
1. Detect consecutive requests where `req[i].hash_ids == req[i+1].hash_ids` and
   `req[i+1].type == 'n'` (non-streaming)
2. Mark the pair in conversation metadata
3. For the paired request, use the same user turn content (delta = 0, re-send)
4. Only count assistant response tokens after the second request

**Files to modify:**
- `src/aiperf/dataset/loader/coding_trace.py` — detect pairs, mark in conversation
- `src/aiperf/dataset/loader/models.py` — add `is_paired_request` field to turn metadata

**Complexity:** Medium

---

## P2: Improvements and Polish

Nice-to-have features that improve observability and benchmark quality.

### P2-1: More Aggressive Scaling Formula

**Problem:** AIPerf's `max(1, active * headroom * 0.5)` formula starts slow. At 1 user
with 95% headroom, it adds 1 user per period. kv-cache-tester's `2 + headroom/10` adds
11 users.

**Recommendation:** Make the scaling formula configurable or offer presets:
- `conservative` (current): `max(1, active * headroom * 0.5)`
- `aggressive` (kv-cache-tester-like): `max(2, 2 + int(headroom_pct / 10))`
- `linear`: `max(1, int(headroom_pct / 5))`

New config: `--adaptive-scale-formula` with default `conservative`.

**Files to modify:**
- `src/aiperf/timing/strategies/adaptive_scale.py` — parameterize formula
- `src/aiperf/timing/config.py` — add formula selection
- `src/aiperf/common/config/user_config.py` — add CLI option

**Complexity:** Low

---

### P2-2: Tighter User Stagger

**Problem:** 500ms stagger between new users is 10x slower than kv-cache-tester's 50ms.
Fewer data points per assessment period, slower ramp-up.

**Recommendation:** Reduce default to 50ms. Make configurable:
`--adaptive-scale-stagger-ms` (default 50).

**Files to modify:**
- `src/aiperf/timing/strategies/adaptive_scale.py` — replace hardcoded 500ms
- `src/aiperf/timing/config.py` — add `stagger_ms` field

**Complexity:** Low

---

### P2-3: Default Max Users Cap

**Problem:** No default cap allows runaway scaling against fast servers.

**Recommendation:** Set `--adaptive-scale-max-users` default to 50 (matching
kv-cache-tester). Users can override with `--adaptive-scale-max-users 0` for unlimited.

**Files to modify:**
- `src/aiperf/timing/config.py` — change default
- `src/aiperf/common/config/user_config.py` — update help text

**Complexity:** Trivial

---

### P2-4: Default Max Delay Cap

**Problem:** No default delay cap means trace delays are used as-is. A trace with a 300s
gap between requests wastes benchmark time.

**Recommendation:** Set `--adaptive-scale-max-delay` default to 60.0 seconds (matching
kv-cache-tester).

**Files to modify:**
- `src/aiperf/timing/config.py` — change default

**Complexity:** Trivial

---

### P2-5: Cache Hit Rate Metrics

**Problem:** AIPerf loads `hash_ids` from traces but doesn't use them for any metrics.
Cache hit rate is a key indicator for KV cache benchmarks.

**Recommended implementation:**

1. Compute per-request cache hits/misses using set intersection of current vs previous
   request's hash_ids (same as kv-cache-tester's runtime approach)
2. Store in `RequestRecord` as new fields: `cache_hit_blocks`, `cache_miss_blocks`
3. Aggregate as `cache_hit_rate` in existing metrics pipeline
4. Export in CSV/JSON outputs

**Files to modify:**
- `src/aiperf/records/inference_result_parser.py` — compute cache metrics from hash_ids
- `src/aiperf/records/records_manager.py` — propagate to exports
- `src/aiperf/exporters/` — add cache hit columns/fields
- `src/aiperf/dataset/loader/coding_trace.py` — store hash_ids per turn in conversation

**Complexity:** Medium

---

### P2-6: Assessment Period Detailed Metrics

**Problem:** AIPerf's `AdaptiveScaleStrategy` only tracks TTFT samples per period. No
visibility into requests launched/completed, in-flight breakdown, dispatch delays, or
idle time.

**Recommended implementation:**

Add `AssessmentPeriodMetrics` dataclass:
- `active_users`, `requests_launched`, `requests_completed`
- `in_flight_prefilling`, `in_flight_decoding`
- `input_tokens_per_second`, `output_tokens_per_second`
- `idle_time_pct`, `dispatch_delay_avg`, `dispatch_delay_max`
- `admission_blocked_events` (if P1-1 implemented)
- `ttft_headroom_pct`

Log at each assessment period boundary. Export as `assessment_periods.csv`.

**Files to modify:**
- `src/aiperf/timing/strategies/adaptive_scale.py` — add metrics tracking and export
- New: `src/aiperf/timing/strategies/assessment_metrics.py` — dataclass definition

**Complexity:** Medium

---

### P2-7: User Lifecycle Events

**Problem:** No per-session lifecycle tracking. Hard to debug why specific conversations
succeeded, were truncated, or errored.

**Recommended implementation:**

Track events: `started`, `completed`, `truncated`, `error` per session.
Store as `(timestamp, conversation_id, event_type, details)`.
Export as `user_lifecycle.csv`.

Hook into existing credit system events:
- `started` on first credit issue
- `completed` on final `CreditReturn` without error
- `truncated` if conversation was truncated by max_isl
- `error` on `CreditReturn` with error

**Files to modify:**
- `src/aiperf/timing/strategies/adaptive_scale.py` — log lifecycle events
- `src/aiperf/exporters/` — new `user_lifecycle_csv_exporter.py`

**Complexity:** Low-Medium

---

### P2-8: TTFT-Responsive Rate Limiting

**Problem:** When TTFT exceeds the threshold, AIPerf stops scaling but doesn't back off
existing users. The server may still be overloaded from current request rate.

**kv-cache-tester approach:** Per-user exponential backoff:
`actual_backoff = min(overage_ratio, 10.0) * 1.5^retry_count`. Users enter `rate_limited`
state, resume after backoff expires. Counter resets on successful dispatch.

**Recommended implementation:**

Add opt-in rate limiting to `AdaptiveScaleStrategy`:
1. New config: `--adaptive-scale-rate-limit` (flag, default off)
2. When TTFT > threshold, compute backoff for new credit dispatches
3. Delay next-turn credit issuance by backoff duration using `scheduler.schedule_later()`
4. Track `rate_limit_count` per session in credit metadata

**Files to modify:**
- `src/aiperf/timing/strategies/adaptive_scale.py` — add rate limit logic
- `src/aiperf/timing/config.py` — add flag

**Complexity:** Medium

---

### P2-9: Trace Metadata Enrichment at Load Time

**Problem:** AIPerf stores only raw trace fields. Useful derived metrics like
`cache_hit_rate`, `total_input_tokens`, `max_shared_prefix_tokens` aren't computed until
needed (if ever).

**Recommended implementation:**

In `CodingTraceLoader.load_dataset()`, after validation:
1. Compute `total_input_tokens = sum(r.input_tokens for r in trace.requests)`
2. Compute `total_output_tokens = sum(r.output_tokens for r in trace.requests)`
3. Compute `max_shared_prefix_tokens` using prefix-based cache hit algorithm
4. Store in `DatasetMetadata.extra` or a new `TraceStatistics` model

These statistics enable better trace selection (e.g., skip traces that are mostly cache
hits) and more informative logging.

**Files to modify:**
- `src/aiperf/dataset/loader/coding_trace.py` — compute stats after loading
- `src/aiperf/dataset/loader/models.py` — add `TraceStatistics` model

**Complexity:** Low

---

### P2-10: Interactive Performance Graphs

**Problem:** AIPerf's `aiperf plot` command generates different visualizations than what's
useful for adaptive scaling analysis. No TTFT-over-time, cache-hit-over-time, or user
lifecycle timeline.

**Recommended implementation:**

Add an `--adaptive-scale-graphs` flag (or auto-generate for adaptive scale runs):
1. TTFT over time: p50, p95, p99 traces with threshold line
2. Throughput + users over time: input/output tok/s with user count overlay
3. User lifecycle scatter plot (if P2-7 implemented): user timeline with start/complete/
   truncate markers

Use Plotly (already available via `aiperf plot`) for interactive HTML output.

**Files to modify:**
- New: `src/aiperf/plot/adaptive_scale_graphs.py`
- `src/aiperf/timing/strategies/adaptive_scale.py` — emit time-series data for graphs

**Complexity:** Medium

---

### P2-11: Configurable Minimum Requests Filter

**Problem:** AIPerf hardcodes minimum 2 requests per trace. kv-cache-tester makes this
configurable via `--min-requests` (default 1).

**Recommendation:** Add `--synthesis-min-requests` (default 2) to make this configurable.

**Files to modify:**
- `src/aiperf/dataset/loader/coding_trace.py` — parameterize the `< 2` check
- `src/aiperf/common/config/user_config.py` — add CLI option

**Complexity:** Trivial

---

## Implementation Roadmap

### Phase 1: Accuracy (P0) — Do First

| ID | Change | Effort | Depends On |
|---|---|---|---|
| P0-2 | First-turn warm prefix accounting | Low | -- |
| P0-3 | Subagent timestamp normalization | Low | -- |
| P0-1 | Shortfall tracking | Medium-High | Architecture decision (Option A vs B) |

P0-2 and P0-3 are independent quick wins. P0-1 is the most impactful and should be
prototyped early to inform the architecture decision for P1-5 and P1-6.

### Phase 2: Core Missing Features (P1) — Unblock Real Benchmarks

| ID | Change | Effort | Depends On |
|---|---|---|---|
| P1-1 | Admission control | Medium | -- |
| P1-3 | Per-period token budget | Low-Medium | P1-1 (shares counter infrastructure) |
| P1-4 | Circuit breaker | Low-Medium | -- |
| P1-7 | Request pair detection | Medium | -- |
| P1-2 | Working set budget | Medium-High | P1-1 |
| P1-5 | Pull-back detection | High | P0-1 (Option B) |
| P1-6 | Coding prompt content | Medium | -- |

P1-1 (admission control) and P1-4 (circuit breaker) are independent and high-value.
P1-3 builds on P1-1's counter infrastructure. P1-5 depends on P0-1's runtime prompt
generation.

### Phase 3: Observability and Polish (P2)

| ID | Change | Effort | Depends On |
|---|---|---|---|
| P2-2 | Tighter stagger (50ms) | Low | -- |
| P2-3 | Default max users (50) | Trivial | -- |
| P2-4 | Default max delay (60s) | Trivial | -- |
| P2-11 | Configurable min requests | Trivial | -- |
| P2-1 | Configurable scaling formula | Low | -- |
| P2-5 | Cache hit rate metrics | Medium | -- |
| P2-9 | Trace metadata enrichment | Low | -- |
| P2-6 | Assessment period metrics | Medium | P1-1 |
| P2-7 | User lifecycle events | Low-Medium | -- |
| P2-8 | Rate limiting | Medium | P1-1 |
| P2-10 | Interactive graphs | Medium | P2-6, P2-7 |

P2-2 through P2-4 and P2-11 are trivial config changes that can ship immediately.
P2-5 and P2-9 are independent data enrichment. P2-6/7/8/10 form an observability
cluster.

---

## Architecture Considerations

### Runtime vs Pre-Computed Prompts

The biggest architectural decision is whether to move from pre-computed (mmap) prompts
to runtime prompt generation. This affects P0-1, P1-5, and P1-6.

**Keep pre-computed (Option A):**
- Pro: No tokenizer in hot path, deterministic, memory-efficient via mmap sharing
- Con: Can't adapt to runtime model behavior, shortfall compensation is approximate
- Approach: Over-allocate prompts by max expected shortfall, truncate at runtime

**Move to runtime (Option B):**
- Pro: Exact shortfall compensation, pull-back support, coding-specific content
- Con: Tokenizer overhead per request, requires prompt generator in worker processes
- Approach: Workers generate prompts on demand using conversation state

**Hybrid recommendation:** Keep mmap for the base conversation structure. Add a
`RuntimePromptAdjuster` that can extend the last user turn by shortfall tokens at
request time. This avoids full runtime generation while supporting compensation:

```
stored_prompt = mmap_conversation.get_turn(i)  # pre-computed base
if shortfall > 0:
    extension = prompt_generator.generate_prompt(shortfall)
    stored_prompt.texts.append(extension)
```

### Multi-Process Working Set Tracking

Hash_id working set tracking in a multi-process architecture requires either:
1. **Centralized (TimingManager):** Workers report hash_ids via message bus. Accurate
   but adds latency to admission decisions.
2. **Estimated (load time):** Compute per-trace estimates at dataset load. Fast but
   approximate — doesn't account for runtime cross-user deduplication.
3. **Hybrid:** Use load-time estimates for admission, accumulate actuals for reporting.

Recommendation: Start with option 3 (estimates for admission, actuals for metrics).

### Credit System Integration

Most P1 features integrate through the existing credit system:
- **Admission control:** Gate in `AdaptiveScaleStrategy` before `issue_credit()`
- **Working set/token budgets:** Check in `AdaptiveScaleStrategy.handle_credit_return()`
  before issuing next session
- **Circuit breaker:** Worker-level, publishes `WorkerShutdown` message
- **Rate limiting:** Delay `scheduler.schedule_later()` calls by backoff duration
- **Lifecycle events:** Hook into credit issue and `CreditReturn` processing

No new message types needed for most features — they fit within the existing
`CreditReturn.ttft_ns` pattern (add fields to `CreditReturn` for cache metrics, error
classification, etc.).
