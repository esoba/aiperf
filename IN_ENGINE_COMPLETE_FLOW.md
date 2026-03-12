# In-Engine Transport: Complete Feature Flow

Everything new on the `ajc/in-engine-transport` branch, end to end.

---

## 1. The Big Picture

This branch adds **direct Python API access** to LLM engines (vLLM, SGLang, TRT-LLM)
running in the same process as AIPerf. Instead of HTTP requests to a server, AIPerf calls
`engine.generate_async()` directly — eliminating network overhead and enabling access to
engine-internal metrics like speculative decoding stats and per-iteration telemetry.

```
BEFORE (HTTP mode):
  AIPerf Worker ──[HTTP/SSE]──> vLLM Server ──> GPU
                                   ↑
                           separate process

AFTER (In-Engine mode):
  AIPerf Worker ──[Python API]──> vLLM Engine ──> GPU
                                     ↑
                              same process, same event loop
```

---

## 2. Transport Layer

### Base Class: `BaseInEngineTransport`
**File**: `src/aiperf/transports/in_engine/base_in_engine_transport.py`

Abstract base providing:
- **Engine lifecycle**: `_start_engine()` / `_stop_engine()` (implemented by each engine)
- **Warmup infrastructure**: `_run_warmup()` drives N iterations, each engine implements `_warmup_single()`
- **Request handling**: `send_request()` extracts messages + sampling params, calls `_generate()`, builds `RequestRecord`
- **Concurrency control**: Optional `asyncio.Semaphore` from `concurrency` engine param
- **Telemetry loop**: Background task polling `_get_engine_stats()` for per-iteration metrics
- **Pre-tokenized support**: Threads `input_ids` through to `_generate()` when `pre_tokenized=True`
- **Streaming TTFT**: First-token callback for time-to-first-token measurement

### vLLM Transport
**File**: `src/aiperf/transports/in_engine/vllm_transport.py`

- `AsyncLLMEngine.from_engine_args()` for engine creation
- Streaming: `RequestOutputKind.DELTA` for incremental tokens
- Non-streaming: `RequestOutputKind.FINAL_ONLY` for efficiency
- Pre-tokenized: `{"prompt_token_ids": ids}` prompt format
- Warmup: Generates with `max_tokens=1` to warm GPU caches

### SGLang Transport
**File**: `src/aiperf/transports/in_engine/sglang_transport.py`

- `sglang.Engine` with `async_generate()` API
- Pre-tokenized: `input_ids=ids` parameter
- Warmup: `async_generate()` with minimal output

### TRT-LLM Transport
**File**: `src/aiperf/transports/in_engine/trtllm_transport.py`

- `tensorrt_llm.LLM` with `generate_async()` + `await output.aresult()`
- **Low-latency preset** (`latency_optimized: true`):
  - CUDA graphs, chunked prefill disabled
  - `GUARANTEED_NO_EVICT` scheduler
  - KV cache: `free_gpu_memory_fraction=0.90`, block reuse disabled
  - Env vars: `TRTLLM_ENABLE_PDL`, `FORCE_MULTI_BLOCK_MODE`, `TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG`, `TRTLLM_MMHA_KERNEL_BLOCK_SIZE`
- **Speculative decoding capture**: Extracts `decode_iterations` and `max_draft_len` from response
- **Engine stats reset**: `_reset_engine_stats()` after warmup to prevent telemetry contamination
- Pre-tokenized: `list[int]` passed directly as prompt

### Engine Generate Endpoints
**File**: `src/aiperf/endpoints/engine_generate.py`

Three endpoint classes (VLLMGenerateEndpoint, SGLangGenerateEndpoint, TRTLLMGenerateEndpoint):
- Format payloads with messages + sampling params
- Thread `input_ids` and spec decode metadata into payload
- Parse `InEngineResponse` into `ParsedResponse` with metadata propagation

---

## 3. Credit Fast-Path (In-Process)

### The Problem
Standard mode: credits traverse ZMQ ROUTER/DEALER sockets between TimingManager (Process A)
and Worker (Process B). For in-engine mode, both live in the same process — ZMQ is pure overhead.

### The Solution

**InProcessCreditRouter** (`src/aiperf/credit/in_process_credit_router.py`):
- Implements `CreditRouterProtocol` (same interface as `StickyCreditRouter`)
- `send_credit(credit)` → `worker.receive_credit(credit)` (direct method call)
- CreditReturn/FirstToken flow back via async callbacks
- No ZMQ, no serialization, no background tasks

**Worker fast-path** (`src/aiperf/workers/worker.py` modifications):
- `receive_credit(credit)` — accept credit without ZMQ
- `set_credit_return_callback(cb)` / `set_first_token_callback(cb)` — route returns via callback
- `_send_credit_return()` / `_send_first_token()` — dispatch to callback or ZMQ fallback

**TimingManager auto-detection** (`src/aiperf/timing/manager.py`):
- `_is_in_engine_mode()` checks URL scheme (`vllm://`, `sglang://`, `trtllm://`)
- In-engine → creates InProcessCreditRouter + Worker in-process
- HTTP → creates StickyCreditRouter as before

**InProcessServiceManager** (`src/aiperf/controller/in_process_service_manager.py`):
- Extends `MultiProcessServiceManager`
- Blocks Worker spawning (Worker is owned by TimingManager)
- Everything else spawns normally

### Credit Flow (In-Engine)
```
CreditIssuer.issue_credit()
  └── InProcessCreditRouter.send_credit(credit)
        └── worker.receive_credit(credit)                    # direct call
              └── _schedule_credit_drop_task(credit)
                    └── _process_credit(credit_context)
                          ├── fetch conversation (ZMQ to DatasetManager)
                          ├── inference_client.send_request()
                          │     └── transport._generate()
                          │           └── engine.generate_async()  # GPU inference
                          ├── _send_first_token(ft)               # callback to PhaseOrchestrator
                          └── _send_credit_return(cr)             # callback to PhaseOrchestrator
```

---

## 4. New Metrics

### Speculative Decoding (10 metrics)
**File**: `src/aiperf/metrics/types/speculative_decoding_metrics.py`

Per-request (from engine response):
- `DecodeIterationCountMetric` — number of decode iterations
- `DraftTokenCountMetric` — `max_draft_len * (decode_iter + 1)`
- `AcceptedDraftTokenCountMetric` — `osl - decode_iter - 1`
- `DraftAcceptanceRateMetric` — `accepted / draft`
- `AcceptanceLengthMetric` — `osl / (decode_iter + 1)` (tau)

Derived aggregates (computed from arrays):
- `AvgDraftTokenCountMetric`
- `AvgAcceptedDraftTokenCountMetric`
- `AvgDraftAcceptanceRateMetric`
- `AvgAcceptanceLengthMetric`
- `TotalDraftTokenCountMetric`

Formulas match trtllm-bench `statistics.py:131-155` exactly.

### Avg Concurrent Requests (Little's Law)
**File**: `src/aiperf/metrics/types/avg_concurrent_requests_metric.py`

```
avg_concurrent_requests = sum(request_latencies) / benchmark_duration
```

### Per-GPU Output Throughput
**File**: `src/aiperf/metrics/types/per_gpu_output_throughput_metric.py`

```
per_gpu_output_throughput = output_token_throughput / world_size
```

### World Size (Config Injection)
**File**: `src/aiperf/metrics/types/world_size_metric.py`

- `BaseDerivedMetric[int]`, default 1, `INTERNAL | NO_CONSOLE`
- Pre-seeded by `MetricResultsProcessor` from `--world-size` config
- Used by PerGPUOutputThroughputMetric

---

## 5. Energy Efficiency
**File**: `src/aiperf/post_processors/energy_efficiency_processor.py`

Cross-pipeline post-processor bridging GPU telemetry with inference metrics:

| Metric | Formula |
|--------|---------|
| `total_energy_joules` | Sum of GPU energy from telemetry |
| `tokens_per_joule` | `total_output_tokens / total_energy_joules` |
| `avg_gpu_power_watts` | `total_energy_joules / benchmark_duration_seconds` |
| `tokens_per_second_per_watt` | `output_token_throughput / avg_gpu_power_watts` |

Integrated into `RecordsManager._process_results()` when GPU telemetry is available.

---

## 6. Pre-Tokenized Dataset Path
**Files**: `input_config.py`, `prompt.py`, `synthetic.py`, `base.py`, `engine_generate.py`, all transports

When `--pre-tokenized` is set:
1. `PromptGenerator.generate_token_ids()` creates random token IDs with EOS filtering: `(eos_id + 1) % vocab_size`
2. `SyntheticDatasetComposer` populates `Text.token_ids` instead of text
3. `EngineGenerateEndpoint` includes `input_ids` in payload
4. Each transport sends token IDs directly to engine:
   - vLLM: `{"prompt_token_ids": ids}`
   - SGLang: `input_ids=ids`
   - TRT-LLM: `list[int]` as prompt

Bypasses re-tokenization for exact ISL control.

---

## 7. Uniform Distribution for ISL/OSL
**Files**: `prompt_config.py`, `enums.py`, `random_generator.py`, `base.py`

- New enum: `SequenceLengthDistributionType.UNIFORM`
- Config: `--isl-distribution uniform --isl-min 128 --isl-max 512`
- `RandomGenerator.sample_positive_uniform_integer(low, high)` — `[low, high]` inclusive
- Validator: uniform requires min/max to be set

---

## 8. Engine Telemetry Collection
**Files**: `engine_telemetry_models.py`, `base_in_engine_transport.py`

- `EngineIterationStats`: timestamp_ns, batch_size, num_tokens, queue_depth, raw dict
- Background `_telemetry_loop` polls `_get_engine_stats()` from engine API
- TRT-LLM: `engine.get_stats_async(timeout=1)` — the cleanest API
- Opt-in via `--engine-params telemetry:true`
- `get_telemetry_log()` returns collected stats for export

---

## 9. Realistic Warmup
**Files**: `base_in_engine_transport.py`, all 3 transport files

- Configurable via `--engine-params warmup_iterations:2,warmup_max_tokens:16`
- `_generate_warmup_prompt()` creates synthetic text of configurable length
- Each engine implements `_warmup_single(prompt, sampling_params)`:
  - vLLM: `FINAL_ONLY` mode, `max_tokens=1` probe
  - SGLang: `async_generate()` with minimal output
  - TRT-LLM: `generate_async()` + `aresult()`, then `_reset_engine_stats()`
- Warmup defaults to 0 unless `latency_optimized=True` (then 2)

---

## 10. Configuration Changes

### InputConfig
- `world_size: int` (default 1, `--world-size`)
- `pre_tokenized: bool` (default False, `--pre-tokenized`)

### PromptConfig
- `InputTokensConfig.distribution` / `OutputTokensConfig.distribution`: `SequenceLengthDistributionType`
- `.min` / `.max` fields for uniform distribution bounds

### InEngineResponse (record_models.py)
- `decode_iterations: int | None` — spec decode iteration count
- `max_draft_len: int | None` — max draft tokens per iteration
- `output_token_ids: list[int] | None` — preserved token IDs

### MetricEnums
- `IN_ENGINE_SPEC_DECODE = 1 << 17` — metric flag
- `TOKENS_PER_JOULE`, `TOKENS_PER_SECOND_PER_WATT` — energy units

---

## 11. Plugin Registration (plugins.yaml)

### Transports
```yaml
transport:
  vllm:
    class: aiperf.transports.in_engine.vllm_transport:VLLMTransport
    metadata: { transport_type: vllm, url_schemes: [vllm] }
  sglang:
    class: aiperf.transports.in_engine.sglang_transport:SGLangTransport
    metadata: { transport_type: sglang, url_schemes: [sglang] }
  trtllm:
    class: aiperf.transports.in_engine.trtllm_transport:TRTLLMTransport
    metadata: { transport_type: trtllm }
```

### Endpoints
```yaml
endpoint:
  vllm_generate: { supported_transports: [vllm] }
  sglang_generate: { supported_transports: [sglang] }
  trtllm_generate: { supported_transports: [trtllm] }
```

### Service Manager
```yaml
service_manager:
  in_engine:
    class: aiperf.controller.in_process_service_manager:InProcessServiceManager
```

## 13. End-to-End Request Flow

Here's what happens when you run `aiperf profile --url trtllm://model-path`:

```
1. CLI detects trtllm:// URL scheme
2. SystemController selects InProcessServiceManager (blocks worker spawning)
3. TimingManager.__init__ detects in-engine mode:
   a. Creates InProcessCreditRouter
   b. Creates Worker in-process (same event loop)
   c. Attaches worker to router
   d. Worker initializes InferenceClient → TRTLLMTransport
4. TRTLLMTransport._start_engine():
   a. Loads LLM with model path, TP config, scheduler config
   b. Applies low-latency preset if latency_optimized=True
   c. Runs warmup iterations
   d. Resets engine stats
   e. Starts telemetry loop
5. DatasetManager generates synthetic prompts (pre-tokenized if configured)
6. PhaseOrchestrator starts PROFILING phase:
   a. CreditIssuer acquires session/prefill slots
   b. CreditIssuer calls router.send_credit(credit)
   c. InProcessCreditRouter calls worker.receive_credit(credit)  # NO ZMQ
7. Worker._process_credit():
   a. Fetches conversation from DatasetManager
   b. InferenceClient formats payload via TRTLLMGenerateEndpoint
   c. TRTLLMTransport._generate():
      - engine.generate_async(prompt, sampling_params)
      - await output.aresult()
      - Captures decode_iterations, max_draft_len
   d. FirstToken callback fires → PhaseOrchestrator releases prefill slot
   e. CreditReturn flows back → PhaseOrchestrator releases session slot
8. Worker pushes RequestRecord via ZMQ PUSH → RecordProcessor
9. RecordProcessor computes metrics:
   - Standard: latency, throughput, TTFT, ITL
   - Spec decode: draft tokens, acceptance rate, acceptance length
   - Pushes MetricRecordsMessage → RecordsManager
10. RecordsManager aggregates all records:
    - Computes derived metrics (avg concurrent, per-GPU throughput)
    - Runs EnergyEfficiencyProcessor if telemetry available
    - Exports results
```
