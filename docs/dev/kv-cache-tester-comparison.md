<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf vs kv-cache-tester: Coding Trace Replay Comparison

Detailed feature-by-feature comparison of AIPerf's coding trace replay implementation
against the reference `trace_replay_tester.py` from kv-cache-tester.

**Compared files:**
- kv-cache-tester: `trace_replay_tester.py` (v1.0, 2830 lines, single-file)
- AIPerf: `coding_trace.py`, `adaptive_scale.py`, `worker.py`, `session_manager.py`, `prompt.py`, `user_config.py`, `input_config.py`, `callback_handler.py`, `credit/structs.py`, `models.py`

---

## 1. Trace Loading

### Same

| Behavior | kv-cache-tester | AIPerf |
|---|---|---|
| Input format | Directory of `*.json` files | Directory of `*.json` files or single JSON file/array |
| JSON parsing | `json.load()` per file | `orjson.loads()` per file |
| Trace identity | Both use the trace `id` field. kv-cache-tester stores it as `metadata.conversation_id` internally. | `id` field from trace root, used directly as `conv_id`. |
| Minimum request filter | Configurable `min_requests` (default 1) | Hardcoded minimum of 2 requests |

### AIPerf Better

| Feature | Detail |
|---|---|
| Single-file support | AIPerf also accepts a single JSON file containing one trace or an array of traces. kv-cache-tester only supports directories. |
| Pydantic validation | AIPerf validates traces via `CodingTrace.model_validate()` with typed fields, aliases, and validators. kv-cache-tester uses raw dict access with `.get()` defaults. |
| Detection heuristic | `CodingTraceLoader.can_load()` checks filename (directory with `*.json`) or data dict structure (`requests` list + `id` key). kv-cache-tester has no auto-detection -- it's always the trace loader. |

### kv-cache-tester Better

| Feature | Detail |
|---|---|
| Trace metadata enrichment | `normalize_trace()` computes `cache_hit_rate`, `total_input_tokens`, `request_count`, and `max_shared_prefix_tokens` at load time. AIPerf stores only the raw trace fields. |
| Cache hit rate pre-computation | Uses prefix-only cache semantics (matching hash_ids from start until first miss) per request. AIPerf does not compute cache hit rates. |

---

## 2. Subagent Flattening

### Same

| Behavior | kv-cache-tester | AIPerf |
|---|---|---|
| Recursive flattening | Descends into nested `requests` arrays | Descends into nested `requests` arrays |

### Different

| Aspect | kv-cache-tester | AIPerf |
|---|---|---|
| Container skipping | Type-based: entries with `type == 'subagent'` are recursed into but the wrapper itself is excluded regardless of token count | Token-based: entries with `input_tokens == 0` are skipped. A subagent with non-zero tokens would be kept. |
| Timestamp handling | Converts relative subagent timestamps to absolute: `base_time + req['t']`, then sorts all by absolute time | Preserves original `t` values, no re-sorting after flattening |
| Type normalization | Forces all requests to `type='streaming'` | Preserves original `type` field (`s`, `n`, `tool_result`, etc.) |
| Sort after flatten | Sorts flattened requests by absolute timestamp | No sort -- order is parent-first, then children in original order |
| File loading order | `list(trace_dir.glob("*.json"))` -- unsorted glob, traces later sorted by `conversation_id` then shuffled with seeded RNG | `sorted(path.glob("*.json"))` -- deterministic alphabetical order |

---

## 3. Context Filtering / Truncation

### Same

| Behavior | Detail |
|---|---|
| Purpose | Limit conversations to a maximum context size |

### Different

| Aspect | kv-cache-tester | AIPerf |
|---|---|---|
| Filter field | `max_context` (CLI arg, default 128000) | `max_isl` (from `--synthesis-max-isl`) |
| Trace-level filter | First request's `input_tokens <= max_context` at load time | No trace-level pre-filter |
| Mid-conversation truncation | In `UserSession.get_next_request()`: if `request['input_tokens'] > max_context`, sets `state = "truncated"` and returns `None`. Conversation stops there. | In `load_dataset()`: iterates requests, `break` on first `r.input_tokens > self._max_isl`. All requests before the exceeding one are kept. |
| Post-truncation requests | Never processed -- session is done | Never loaded into the trace |
| Minimum after truncation | kv-cache-tester filters at load: `first_input <= max_context and num_requests >= min_requests` | AIPerf filters after truncation: `len(flat_requests) < 2` skips the trace |

### kv-cache-tester Better

| Feature | Detail |
|---|---|
| Runtime truncation | Truncation happens at request time, so the session can process N-1 requests successfully before hitting the limit. AIPerf truncates at load time, which means a trace with requests `[100, 200, 50000, 300]` and `max_isl=1000` keeps `[100, 200]` but never gets to `300`. kv-cache-tester would process 100, 200, then truncate at 50000 -- same result, but the decision is made later with full runtime context. |

---

## 4. Delta-Based Prompt Sizing

### Same

| Behavior | kv-cache-tester | AIPerf |
|---|---|---|
| Core concept | `input_tokens` = total context at this turn, not new content | Same understanding, same delta approach |
| First turn | Full `input_tokens` used | `delta = req.input_tokens` |
| Normal growth formula | `token_delta = current_input_tokens - prev_input_tokens` minus `stored_response_tokens` | `delta = max(1, input_tokens - prev.input_tokens - prev.output_tokens)` |
| Floor at minimum | `max(0, ...)` for tokens_to_generate | `max(1, ...)` -- floors at 1, not 0 |

### Different

| Aspect | kv-cache-tester | AIPerf |
|---|---|---|
| Response token accounting | Uses `stored_response_tokens` -- the ACTUAL token count from the model's response (counted at runtime) | Uses `prev.output_tokens` -- the EXPECTED token count from the trace file |
| When delta is computed | At runtime in `build_messages()`, after receiving the model's actual response | At dataset load time in `convert_to_conversations()`, before any requests are sent |
| Shortfall compensation | `token_shortfall` carries forward deficit when model generates < 80% of expected: `tokens_to_generate = max(0, token_delta - stored_response_tokens + token_shortfall)` | No shortfall tracking -- assumes model generates exactly `output_tokens` from the trace |

### kv-cache-tester Better

| Feature | Detail |
|---|---|
| Runtime response accounting | kv-cache-tester tracks the ACTUAL model output tokens and adjusts the next delta accordingly. If the model generates 300 tokens instead of expected 500, the next user message is 200 tokens larger to compensate. AIPerf uses the trace's expected output_tokens, so runtime model undergeneration causes cumulative context drift. |
| Shortfall tracking | The `token_shortfall` mechanism (triggered at < 80% of expected) prevents context size from drifting when models consistently undergenerate. AIPerf has no equivalent. |
| Pull-back handling | kv-cache-tester detects "pull-backs" (>10% hash_ids removed) and resets the conversation entirely, regenerating from scratch based on `kept_hash_ids * block_size`. AIPerf has no pull-back detection -- it always appends to the conversation. |
| Request pair detection | kv-cache-tester detects consecutive requests with identical `hash_ids` and re-sends the same conversation unchanged. AIPerf treats every request independently. |

### AIPerf Better

| Feature | Detail |
|---|---|
| Pre-computation efficiency | AIPerf computes all deltas at load time and generates a single base prompt at `max(deltas)`, truncating by character ratio. This avoids per-request tokenizer calls. kv-cache-tester generates content per-request at runtime. |
| Deterministic prompts | AIPerf's prompts are deterministic from the dataset -- same trace always produces the same conversations. kv-cache-tester's prompts depend on runtime model responses, making results harder to reproduce. |

---

## 5. Prompt Content Generation

### Different

| Aspect | kv-cache-tester | AIPerf |
|---|---|---|
| Content source | Two synthetic pools: user text (coding prompts, 40% prompts + 30% questions + 30% tech words) and tool results (35% file contents + 25% bash output + 15% JSON + 10% tracebacks + 15% paths). 500K tokens each. | Shakespeare corpus tokenized once, sampled by offset. Single pool. |
| Question bank | 25 detailed technical questions appended to every user message to encourage long responses. Question tokens subtracted from content budget. | No question bank. |
| Seeding | Per-request seed: `hash(f"{user_id}_{request_idx}_...")` for deterministic-per-user content | Global corpus with seeded RNG (`seed=42`). Offset-based sampling. |
| Token pool size | 500K tokens per pool | Full Shakespeare text (~900K tokens depending on tokenizer) |
| Content realism | Mimics real coding session content (code, errors, JSON, file listings) | Generic literary text (Shakespeare) |

### kv-cache-tester Better

| Feature | Detail |
|---|---|
| Content realism | The dual-pool approach with coding-specific content (file contents with line numbers, bash output, JSON, error tracebacks) better approximates real agentic coding sessions. Shakespeare text has very different token distribution. |
| Question bank | Appending technical questions encourages models to generate longer responses, helping hit the expected `output_tokens` targets. Without this, models may generate shorter responses, exacerbating the shortfall problem. |
| Per-request determinism | Content is deterministic per `(user_id, request_idx)` tuple, so different users get different content but each user's content is reproducible. AIPerf's offset-based sampling from a single corpus is reproducible but all conversations share the same text pool pattern. |

---

## 6. Warm Prefix

### Same

| Behavior | kv-cache-tester | AIPerf |
|---|---|---|
| Purpose | Shared prefix across all conversations for cross-conversation KV cache hits | Same |
| Sizing | `warm_prefix_pct * max(tool_tokens + system_tokens)` across all traces | Same formula |
| Default percentage | 50% (`--warm-prefix-pct 0.5`) | 50% (`--warm-prefix-pct 0.5`) |
| Shared across users | Yes -- identical content for all | Yes -- set as `conversation.system_message` |

### Different

| Aspect | kv-cache-tester | AIPerf |
|---|---|---|
| Content source | Fixed seed: `hash("canonical_warm_prefix_v1") % 2**32`. Generated from user text pool. Cached in `_canonical_prefix_content`. | Generated via `prompt_generator.generate(mean=prefix_tokens, stddev=0, hash_ids=[])` from Shakespeare corpus. |
| Integration | Prepended as part of first user message content | Set as `conversation.system_message` (separate from user turns) |
| First-turn adjustment | First turn's user content is reduced by prefix tokens: `remaining_tokens = max(0, current_input_tokens - prefix_tokens)` | No adjustment -- warm prefix is additive (system message + full first-turn delta). This means the first turn's total context exceeds the trace's intended `input_tokens` by `prefix_tokens`. |

### kv-cache-tester Better

| Feature | Detail |
|---|---|
| First-turn token accounting | kv-cache-tester subtracts `prefix_tokens` from the first request's content budget, so total context stays close to the trace's `input_tokens`. AIPerf adds the warm prefix as a system message on top of the full first-turn delta, inflating the first turn's context. |

---

## 7. Adaptive User Scaling

### Same

| Behavior | kv-cache-tester | AIPerf |
|---|---|---|
| Core concept | Start with N users, periodically assess TTFT, add users while headroom exists | Same |
| TTFT metrics | `max`, `avg`, `p95` | Same three options |
| Default metric | `p95` | `p95` |
| Default threshold | 2.0 seconds | 2.0 seconds |
| Default assessment period | 30 seconds | 30 seconds |
| Time scaling | `time_scale` factor on inter-request delays | Same (`adaptive_scale_time_scale`) |
| Delay capping | `max_delay` caps inter-request delays | Same (`adaptive_scale_max_delay`) |
| Session recycling | `--recycle` flag replaces completed sessions | `--adaptive-scale-recycle` flag |
| Scaling stops at threshold | When TTFT >= max_ttft, no new users added | Same behavior |

### Different

| Aspect | kv-cache-tester | AIPerf |
|---|---|---|
| Scaling formula | `max(2, 2 + int(headroom_pct / 10))`. Example: 80% headroom = 10 users. Linear, aggressive. | `max(1, int(active_users * headroom_ratio * 0.5))`. Proportional to current user count, conservative (0.5 damping). |
| User stagger | 50ms between batch-created users | 500ms between new users |
| Initial user stagger | 50ms stagger via `create_users_batch(delay_ms=50)` | 500ms stagger for initial users (first issued immediately) |
| Scale-down | Users are removed on completion/truncation. `active_users` decreases naturally. No active scale-down based on TTFT. | `active_users -= 1` on final turn (no recycle). No active scale-down. Same natural attrition. |
| No-data handling | Returns 0 users to add (no data = don't scale) | Skips assessment with debug log "No TTFT samples in assessment period" |
| TTFT sample scope | Per-period only (current assessment period) | Per-period samples used for metric, also accumulated in `_all_ttft_samples` (currently unused) |
| Default max users | 50 (`--max-users 50`) | None / unlimited (`--adaptive-scale-max-users` is optional) |

### kv-cache-tester Better

| Feature | Detail |
|---|---|
| More aggressive scaling | The `2 + headroom/10` formula adds more users faster. At 95% headroom (mock server), kv-cache-tester adds 11 users per period vs AIPerf's `max(1, active * 0.475)` which starts slow and accelerates. For real servers where TTFT matters, kv-cache-tester finds the ceiling faster. |
| Tighter stagger | 50ms stagger vs 500ms means new users start sending requests sooner, giving the assessment period more data points. |
| Default max users cap | Having a default of 50 prevents runaway scaling against fast mock servers. AIPerf with no cap scaled to 60 users in our tests. |

### AIPerf Better

| Feature | Detail |
|---|---|
| Proportional scaling | The `active_users * headroom * 0.5` formula is self-regulating -- it adds more users as the fleet grows, but the 0.5 damping prevents overshooting. kv-cache-tester's linear formula doesn't account for current load. |
| Streaming validation | AIPerf validates at config time that `--streaming` is enabled for adaptive scale (TTFT requires streaming SSE parsing). kv-cache-tester always uses streaming but doesn't validate it. |
| Auto-detection | AIPerf auto-enables `ADAPTIVE_SCALE` mode when `coding_trace` dataset is detected, even without `--adaptive-scale` flag. kv-cache-tester is always in adaptive mode. |

---

## 8. TTFT Measurement

### Same

| Behavior | Detail |
|---|---|
| Source | First SSE chunk with actual content (not empty or metadata) |
| Metric computation | p95, avg, or max of samples within an assessment period |

### Different

| Aspect | kv-cache-tester | AIPerf |
|---|---|---|
| Measurement point | `time.time()` at first chunk with `delta.content` in the streaming response | `time.perf_counter_ns()` subtracted from request start in `InferenceClient`. Stored as `ttft_ns` on `CreditContext`. |
| Clock type | Wall clock (`time.time()`) | Monotonic high-resolution (`perf_counter_ns`) |
| Transport | First-token callback fires `on_first_token()` on the `TestOrchestrator` which increments `in_flight_decoding` | First-token callback stores `ttft_ns` on credit context. For prefill concurrency, also sends `FirstToken` message via DEALER socket. |
| Routing to scaler | Direct method call on `TestOrchestrator.calculate_users_to_add()` within the same event loop | `CreditReturn.ttft_ns` flows back through `CreditCallbackHandler.on_credit_return()` which calls `strategy.on_ttft_sample(ttft_ns)` via duck-typed dispatch |

### AIPerf Better

| Feature | Detail |
|---|---|
| Clock precision | `perf_counter_ns` (nanosecond monotonic) vs `time.time()` (microsecond wall clock, subject to NTP adjustments) |
| Separation of concerns | TTFT flows through the credit system cleanly. kv-cache-tester mixes TTFT measurement with admission control state (`in_flight_decoding`) in the same callback. |

---

## 9. Working Set and Admission Control

### Missing from AIPerf

| Feature | kv-cache-tester | AIPerf |
|---|---|---|
| Working set budget | Tracks union of all users' `all_hash_ids * chunk_size`. New user rejected if `current_working_set + estimated_new_tokens > max_working_set_tokens`. | Not implemented. |
| Per-period token budget | Limits new cache-miss tokens per assessment period: `max_new_tokens_per_period` (default 500K). Prevents burst-loading too many new KV blocks. | Not implemented. |
| Admission control | `max_concurrent_requests` caps in-flight requests. Dispatch skipped when at capacity. Tracks `in_flight_prefilling` and `in_flight_decoding` separately. | Not implemented as a standalone feature. Prefill concurrency limiter (`--prefill-concurrency`) limits concurrent prefills specifically, but no general admission control. |
| Hash ID tracking | Each user's hash_ids are prefixed with user_id for global uniqueness. Working set = union of all users' global hash_ids * chunk_size. | Hash IDs are loaded from traces but not used for any runtime decisions. |

---

## 10. Rate Limiting

### Missing from AIPerf

| Feature | kv-cache-tester | AIPerf |
|---|---|---|
| Request rate limiting | When `--enable-request-rate-limiting` is set and TTFT exceeds threshold: `backoff = min(1.0 * overage_ratio, 10.0)`, with exponential backoff per user: `actual_backoff = backoff * (1.5 ** rate_limit_count)`. Affected users enter `rate_limited` state. | Not implemented. AIPerf uses inter-turn delays from traces but has no TTFT-responsive rate limiting. |

---

## 11. Error Handling

### Different

| Aspect | kv-cache-tester | AIPerf |
|---|---|---|
| Connection error circuit breaker | Stops test after 10 consecutive connection errors | No circuit breaker. Individual request errors are recorded but don't abort the run. |
| Error classification | Classifies as `"connection"` (connection/connect/refused) vs `"other"` | Error stored as `ErrorDetails` with full exception info. No classification. |
| Retry on rate limit | `rate_limited` state with exponential backoff | No explicit rate-limit handling |
| Request timeout | No explicit timeout (relies on aiohttp/openai client defaults) | No explicit per-request timeout (configurable via `cancel_after_ns` on credits) |

### AIPerf Better

| Feature | Detail |
|---|---|
| Error propagation | Errors flow through `CreditReturn.error` and `RequestRecord.error` to the metrics pipeline. Every error is recorded. kv-cache-tester logs errors but only tracks `success` boolean per request. |
| Graceful shutdown | Worker sends `WorkerShutdown` message. Credit system guarantees every credit is returned (via `finally` block + done callback). kv-cache-tester cancels remaining tasks after 60s timeout. |

---

## 12. Request Pairs / Streaming Detection

### Missing from AIPerf

| Feature | kv-cache-tester | AIPerf |
|---|---|---|
| Request pair detection | Detects consecutive requests with identical `hash_ids` (streaming + non-streaming pair). Re-sends same conversation for the paired request. Only stores assistant response after the non-streaming request to avoid double-counting. | Not implemented. Every request is treated independently. The `type` field is preserved in the model but not used for any behavioral decisions. |

---

## 13. Metrics and Output

### Same

| Feature | Detail |
|---|---|
| Per-request metrics | Both record TTFT, request latency, input/output token counts, error status |
| CSV export | Both export per-request metrics to CSV |
| JSON export | Both export aggregate metrics to JSON |

### kv-cache-tester Better

| Feature | Detail |
|---|---|
| Cache hit metrics | Per-request `cache_hit_blocks`, `cache_miss_blocks`. Assessment period `cache_hit_rate`. AIPerf does not compute cache metrics. |
| Proportional output attribution | Output tokens are proportionally attributed to assessment periods based on when each chunk was generated (handles requests spanning multiple periods). AIPerf attributes entire request to its completion time. |
| Interactive graphs | Three Plotly HTML graphs: TTFT over time with threshold, throughput + users, cache hit timeline, user lifecycle scatter plot. AIPerf uses a separate `aiperf plot` command with different visualizations. |
| User lifecycle tracking | `UserLifecycleEvent` records with timestamps for started/completed/truncated/error per user. Exported to `user_lifecycle.csv`. AIPerf tracks session completion implicitly through the credit system but doesn't export per-session lifecycle data. |
| Idle time computation | `max(0, 1 - total_request_time / (duration * total_users)) * 100` measures how much time users spent waiting vs making requests. Not computed in AIPerf. |
| Dispatch delay tracking | Tracks time between a user becoming ready and actually being dispatched (measures scheduling overhead). Not tracked in AIPerf. |
| Assessment period metrics | Detailed per-period stats including requests_launched, requests_completed (new vs prior), requests_in_progress (new vs prior), admission_blocked_events. AIPerf's `AdaptiveScaleStrategy` only tracks TTFT samples per period. |

### AIPerf Better

| Feature | Detail |
|---|---|
| Comprehensive metrics pipeline | Tokenizer-based token counting (not 1-token-per-chunk approximation), ITL computation, TPOT, output token throughput per user, time-sliced metrics. kv-cache-tester approximates 1 token per SSE chunk. |
| Multiple export formats | CSV, JSON, Parquet for server metrics. Configurable via plugin system. |
| Real-time dashboard | `aiperf dashboard` provides live Textual TUI during profiling. kv-cache-tester only has terminal log output. |

---

## 14. Architecture

### Different

| Aspect | kv-cache-tester | AIPerf |
|---|---|---|
| Architecture | Single-file, single-process, asyncio event loop | Multi-process: SystemController orchestrates 9 services (DatasetManager, TimingManager, Workers, RecordProcessors, etc.) via ZMQ message bus |
| Concurrency model | Single asyncio loop manages all users and requests | Multiple worker processes, each with own event loop. Credits distributed via DEALER/ROUTER sockets. |
| Dataset storage | In-memory (conversation lists per user) | Memory-mapped files (mmap) shared across worker processes |
| Session affinity | Implicit -- single process, all state in memory | Explicit sticky routing via `StickyCreditRouter` ensuring all turns of a conversation go to the same worker |
| Configuration | `argparse` with `TestConfig` dataclass | Pydantic `UserConfig` with nested models, validators, CLI via `click` |

### AIPerf Better

| Feature | Detail |
|---|---|
| Scalability | Multi-process architecture can saturate high-throughput servers. kv-cache-tester is limited by single-process GIL and event loop capacity. |
| Memory efficiency | mmap-backed dataset avoids duplicating conversation data across workers. kv-cache-tester keeps all user conversations in memory. |
| Extensibility | Plugin system for endpoints, transports, timing strategies, dataset loaders. kv-cache-tester is monolithic. |

### kv-cache-tester Better

| Feature | Detail |
|---|---|
| Simplicity | Single file, easy to read and modify. AIPerf's distributed architecture adds complexity (ZMQ, credit system, process management). |
| Runtime adaptability | Can modify conversation content based on actual model responses (shortfall tracking, pull-back detection). AIPerf's pre-computed prompts can't adapt to runtime model behavior. |

---

## 15. Configuration Comparison

| Parameter | kv-cache-tester | AIPerf | Notes |
|---|---|---|---|
| API endpoint | `--api-endpoint` (required) | `--url` | Same concept |
| Trace directory | `--trace-directory` (required) | `--input-file` | AIPerf also accepts single files |
| Max context | `--max-context` (default 128000) | `--synthesis-max-isl` | Different name, same purpose |
| Max TTFT | `--max-ttft` (default 2.0) | `--adaptive-scale-max-ttft` (default 2.0) | Same default |
| TTFT metric | `--ttft-metric` (default p95) | `--adaptive-scale-ttft-metric` (default p95) | Same default |
| Start users | `--start-users` (default 1) | `--adaptive-scale-start-users` (default 1) | Same default |
| Max users | `--max-users` (default 50) | `--adaptive-scale-max-users` (default None) | AIPerf has no cap by default |
| Recycle | `--recycle` (flag) | `--adaptive-scale-recycle` (flag) | Same |
| Max delay | `--max-delay` (default 60.0) | `--adaptive-scale-max-delay` (default None) | kv-cache-tester caps at 60s by default |
| Time scale | `--time-scale` (default 1.0) | `--adaptive-scale-time-scale` (default 1.0) | Same default |
| Test duration | `--test-duration` (optional) | `--benchmark-duration` (required for adaptive) | AIPerf requires it |
| Assessment period | `--assessment-period` (default 30) | `--adaptive-scale-assessment-period` (default 30.0) | Same default |
| Warm prefix | `--warm-prefix-pct` (default 0.5) | `--warm-prefix-pct` (default 0.5) | Same |
| Tokenizer | `--tokenizer` (default Qwen/Qwen2.5-Coder-32B-Instruct) | `--tokenizer` | Different defaults |
| Seed | `--seed` (optional) | N/A (uses fixed seed 42 for prompt generation) | kv-cache-tester has more seed control |
| Working set budget | `--max-working-set-tokens` (default 0=unlimited) | N/A | Missing from AIPerf |
| Period token budget | `--max-new-tokens-per-period` (default 500K) | N/A | Missing from AIPerf |
| Admission control | `--max-concurrent-requests` (default 50, caps total in-flight requests) | `--prefill-concurrency` (limits concurrent prefills only, not total in-flight) | Not equivalent -- kv-cache-tester limits all in-flight requests, AIPerf limits only prefill phase |
| Rate limiting | `--enable-request-rate-limiting` (flag) | N/A | Missing from AIPerf |
| Min requests | `--min-requests` (default 1) | Hardcoded at 2 | Not configurable in AIPerf |
| Chunk size | `--chunk-size` (default 256) | `block_size` from trace (default 64) | kv-cache-tester uses for working set math |
| Temperature | `--temperature` (optional override) | N/A (uses endpoint defaults) | Not exposed |
| Model-specific defaults | Auto-detects qwen3-coder, applies temp/top_p/top_k/rep_penalty | N/A | Missing from AIPerf |

---

## 16. Summary Scorecard

| Category | Same | AIPerf Better | kv-cache-tester Better | Missing from AIPerf |
|---|---|---|---|---|
| Trace loading | Format | Single-file support, validation | Metadata enrichment, file loading order | -- |
| Flattening | Recursive descent | -- | Timestamp re-sort | -- |
| Context truncation | Mid-conversation break | Load-time efficiency | Runtime truncation | -- |
| Delta computation | Core formula | Pre-computation, determinism | Runtime response accounting, shortfall, pull-back | Shortfall tracking |
| Prompt content | -- | Efficiency (single base prompt) | Realism (coding content, question bank) | -- |
| Warm prefix | Sizing formula | -- | First-turn token accounting | First-turn accounting |
| Adaptive scaling | Concept, metrics, defaults | Auto-detection, streaming validation, proportional formula | Aggressive scaling, tight stagger | -- |
| TTFT measurement | SSE-based, per-period | Clock precision, separation of concerns | -- | -- |
| Working set tracking | -- | -- | -- | Working set budget, per-period budget |
| Admission control | -- | -- | -- | Max concurrent requests |
| Rate limiting | -- | -- | -- | TTFT-responsive rate limiting |
| Request pairs | -- | -- | -- | Paired request detection |
| Error handling | -- | Error propagation, graceful shutdown | Circuit breaker | Connection circuit breaker |
| Metrics | Per-request, CSV, JSON | Tokenizer counting, dashboard, multi-format | Cache metrics, proportional attribution, graphs, lifecycle | Cache hit metrics, user lifecycle |
| Architecture | -- | Multi-process, mmap, plugin system | Simplicity, runtime adaptability | -- |
