---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Command Line Options
---

# Command Line Options

## `aiperf` Commands

### [`--install-completion`](#aiperf---install-completion)

Install shell completion for this application.

### [`analyze-trace`](#aiperf-analyze-trace)

Analyze a mooncake trace file for ISL/OSL distributions and cache hit rates.

### [`profile`](#aiperf-profile)

Run the Profile subcommand.

[Endpoint](#endpoint) • [Input](#input) • [Fixed Schedule](#fixed-schedule) • [Goodput](#goodput) • [Output](#output) • [HTTP Trace](#http-trace) • [Tokenizer](#tokenizer) • [Load Generator](#load-generator) • [Warmup](#warmup) • [User-Centric Rate](#user-centric-rate) • [Request Cancellation](#request-cancellation) • [Conversation Input](#conversation-input) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Prompt](#prompt) • [Prefix Prompt](#prefix-prompt) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Service](#service) • [Server Metrics](#server-metrics) • [GPU Telemetry](#gpu-telemetry) • [UI](#ui) • [Workers](#workers) • [ZMQ Communication](#zmq-communication) • [Multi-Run](#multi-run) • [Accuracy](#accuracy)

### [`plot`](#aiperf-plot)

Generate visualizations from AIPerf profiling data.

### [`plugins`](#aiperf-plugins)

Explore AIPerf plugins: aiperf plugins [category] [type]

### [`service`](#aiperf-service)

Run an AIPerf service in a single process.

### [`kube attach`](#aiperf-kube-attach)

Attach to a running AIPerf benchmark and stream progress.

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube cancel`](#aiperf-kube-cancel)

Cancel a running AIPerf benchmark.

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube debug`](#aiperf-kube-debug)

Run diagnostic analysis on a benchmark deployment.

### [`kube cleanup`](#aiperf-kube-cleanup)

Remove stale benchmark namespaces older than max-age seconds.

### [`kube shutdown`](#aiperf-kube-shutdown)

Shut down the controller pod's API service after a benchmark finishes.

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube delete`](#aiperf-kube-delete)

Delete an AIPerf benchmark job and clean up resources.

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube generate`](#aiperf-kube-generate)

Generate Kubernetes YAML manifests for an AIPerf benchmark.

[Endpoint](#endpoint) • [Input](#input) • [Fixed Schedule](#fixed-schedule) • [Goodput](#goodput) • [Output](#output) • [HTTP Trace](#http-trace) • [Tokenizer](#tokenizer) • [Load Generator](#load-generator) • [Warmup](#warmup) • [User-Centric Rate](#user-centric-rate) • [Request Cancellation](#request-cancellation) • [Conversation Input](#conversation-input) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Prompt](#prompt) • [Prefix Prompt](#prefix-prompt) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Service](#service) • [Server Metrics](#server-metrics) • [GPU Telemetry](#gpu-telemetry) • [UI](#ui) • [Workers](#workers) • [ZMQ Communication](#zmq-communication) • [Multi-Run](#multi-run) • [Accuracy](#accuracy) • [Kubernetes](#kubernetes) • [Kubernetes Node Placement](#kubernetes-node-placement) • [Kubernetes Scheduling](#kubernetes-scheduling) • [Kubernetes Metadata](#kubernetes-metadata) • [Kubernetes Secrets](#kubernetes-secrets)

### [`kube init`](#aiperf-kube-init)

Generate a starter configuration template for Kubernetes benchmarks.

### [`kube logs`](#aiperf-kube-logs)

Get logs from AIPerf benchmark pods.

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube preflight`](#aiperf-kube-preflight)

Run comprehensive pre-flight checks for Kubernetes deployment.

[Kubernetes](#kubernetes) • [Parameters](#parameters)

### [`kube profile`](#aiperf-kube-profile)

Run a benchmark in Kubernetes.

[Endpoint](#endpoint) • [Input](#input) • [Fixed Schedule](#fixed-schedule) • [Goodput](#goodput) • [Output](#output) • [HTTP Trace](#http-trace) • [Tokenizer](#tokenizer) • [Load Generator](#load-generator) • [Warmup](#warmup) • [User-Centric Rate](#user-centric-rate) • [Request Cancellation](#request-cancellation) • [Conversation Input](#conversation-input) • [Input Sequence Length (ISL)](#input-sequence-length-isl) • [Output Sequence Length (OSL)](#output-sequence-length-osl) • [Prompt](#prompt) • [Prefix Prompt](#prefix-prompt) • [Rankings](#rankings) • [Synthesis](#synthesis) • [Audio Input](#audio-input) • [Image Input](#image-input) • [Video Input](#video-input) • [Service](#service) • [Server Metrics](#server-metrics) • [GPU Telemetry](#gpu-telemetry) • [UI](#ui) • [Workers](#workers) • [ZMQ Communication](#zmq-communication) • [Multi-Run](#multi-run) • [Accuracy](#accuracy) • [Kubernetes](#kubernetes) • [Kubernetes Node Placement](#kubernetes-node-placement) • [Kubernetes Scheduling](#kubernetes-scheduling) • [Kubernetes Metadata](#kubernetes-metadata) • [Kubernetes Secrets](#kubernetes-secrets) • [Parameters](#parameters)

### [`kube results`](#aiperf-kube-results)

Retrieve results from an AIPerf benchmark.

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube list`](#aiperf-kube-list)

List AIPerf benchmark jobs and their status.

[Parameters](#parameters) • [Kubernetes](#kubernetes)

### [`kube validate`](#aiperf-kube-validate)

Validate AIPerfJob YAML files against the CRD schema and AIPerfConfig model.

### [`kube watch`](#aiperf-kube-watch)

Watch live status of benchmark pods, events, and resource usage.

<hr/>

## `aiperf --install-completion`

Install shell completion for this application.

This command generates and installs the completion script to the appropriate location for your shell. After installation, you may need to restart your shell or source your shell configuration file.

#### `--shell` `<str>`

Shell type for completion. If not specified, attempts to auto-detect current shell.

#### `-o`, `--output` `<str>`

Output path for the completion script. If not specified, uses shell-specific default.

<hr/>

## `aiperf analyze-trace`

Analyze a mooncake trace file for ISL/OSL distributions and cache hit rates.

#### `--input-file` `<str>` _(Required)_

Path to input mooncake trace JSONL file.

#### `--block-size` `<int>`

KV cache block size for analysis (default: 512).
<br/>_Default: `512`_

#### `--output-file` `<str>`

Optional output path for analysis report (JSON).

<hr/>

## `aiperf profile`

Run the Profile subcommand.

Benchmark generative AI models and measure performance metrics including throughput, latency, token statistics, and resource utilization.

**Examples:**

```bash
# Basic profiling with streaming
aiperf profile --model Qwen/Qwen3-0.6B --url localhost:8000 --endpoint-type chat --streaming

# Concurrency-based benchmarking
aiperf profile --model your_model --url localhost:8000 --concurrency 10 --request-count 100

# Request rate benchmarking (Poisson distribution)
aiperf profile --model your_model --url localhost:8000 --request-rate 5.0 --benchmark-duration 60

# Time-based benchmarking with grace period
aiperf profile --model your_model --url localhost:8000 --benchmark-duration 300 --benchmark-grace-period 30

# Custom dataset with fixed schedule replay
aiperf profile --model your_model --url localhost:8000 --input-file trace.jsonl --fixed-schedule

# Multi-turn conversations with ShareGPT dataset
aiperf profile --model your_model --url localhost:8000 --public-dataset sharegpt --num-sessions 50

# Goodput measurement with SLOs
aiperf profile --model your_model --url localhost:8000 --goodput "request_latency:250 inter_token_latency:10"
```

### Endpoint

#### `-m`, `--model-names`, `--model` `<list>`

List of model configurations. At least one model required.

#### `--model-selection-strategy` `<str>`

Strategy for selecting models when multiple are configured. round_robin cycles through models, random selects randomly, weighted uses configured weights, modality_aware routes by input type.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select models based on configured weights. Each model's weight determines its selection probability. |

#### `-u`, `--url` `<list>`

List of server URLs to benchmark. Requests distributed according to url_strategy. Example: ['http://localhost:8000/v1/chat/completions'].
<br/>_Default: `['localhost:8000']`_

#### `--url-strategy` `<str>`

Strategy for distributing requests across multiple URLs. round_robin cycles through URLs in order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--endpoint-type` `<str>`

API endpoint type determining request/response format. chat: OpenAI chat completions, completions: OpenAI completions, embeddings: vector embeddings, rankings: reranking, template: custom format.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`

Enable streaming (Server-Sent Events) responses. Required for accurate TTFT (time to first token) measurement. Server must support streaming for this to work.
<br/>_Flag (no value required)_

#### `--custom-endpoint`, `--endpoint` `<str>`

Override default endpoint path. Use for servers with non-standard API paths. Example: '/custom/v2/generate'.

#### `--api-key` `<str>`

API authentication key. Supports environment variable substitution: ${OPENAI_API_KEY}. Can also use ${VAR:default} syntax for defaults.

#### `--request-timeout-seconds` `<float>`

Request timeout in seconds. Requests exceeding this duration are marked as failed. Should exceed expected max response time.
<br/>_Default: `600.0`_

#### `--transport`, `--transport-type` `<str>`

HTTP transport protocol (http/https). Auto-detected from URL scheme if not specified. Explicit setting overrides auto-detection.
<br/>_Choices: [`http`]_

#### `--use-legacy-max-tokens`

Use 'max_tokens' field instead of 'max_completion_tokens'. Enable for compatibility with older OpenAI API versions.
<br/>_Flag (no value required)_

#### `--use-server-token-count`

Use server-reported token counts from response usage field. When true, trusts usage.prompt_tokens and usage.completion_tokens. When false, counts tokens locally using configured tokenizer.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy` `<str>`

HTTP connection management strategy. pooled: shared connection pool (fastest), never: new connection per request (includes TCP overhead), sticky_sessions: dedicated connection per session.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--download-video-content`

For video generation endpoints, download the video content after generation completes. When enabled, request latency includes the video download time. When disabled, only generation time is measured.
<br/>_Flag (no value required)_

#### `--extra-inputs` `<str>`

Additional fields to include in request body. Merged into every request. Common fields: temperature, top_p, top_k, stop.

#### `-H`, `--header` `<str>`

Custom HTTP headers to include in all requests. Useful for authentication, tracing, or routing. Values support environment variable substitution.

### Input

#### `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Can be absolute or relative. Supported formats depend on the format field: JSONL for single_turn/multi_turn, JSONL trace files for mooncake_trace, directories for random_pool.

#### `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking. AIPerf automatically downloads and parses these datasets.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `sharegpt` |  | ShareGPT dataset from HuggingFace. Multi-turn conversational dataset with user/assistant exchanges. |

#### `--custom-dataset-type` `<str>`

Dataset file format determining parsing logic and expected file structure. single_turn: JSONL with single prompt-response exchanges. multi_turn: JSONL with conversation history. mooncake_trace: timestamped trace files for replay. random_pool: directory of reusable prompts.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `single_turn` | _default_ | Simple prompt-response pairs. |
| `multi_turn` |  | Conversational data with multiple turns. |
| `mooncake_trace` |  | Mooncake production trace format. |
| `random_pool` |  | Treat file as a pool for random sampling. |

#### `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. sequential: iterate in order, wrapping to start after end. random: randomly sample with replacement (entries may repeat). shuffle: random permutation without replacement, re-shuffling after exhaustion.
<br/>_Choices: [`random`, `sequential`, `shuffle`]_
<br/>_Default: `sequential`_

#### `--random-seed` `<int>`

Global random seed for reproducibility. Can be overridden per-dataset. If not set, uses system entropy.

### Fixed Schedule

#### `--fixed-schedule`, `--no-fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for mooncake_trace datasets.

#### `--fixed-schedule-auto-offset`

Normalize trace timestamps to start at 0. Subtracts minimum timestamp from all entries. Only used with type='fixed_schedule'.
<br/>_Flag (no value required)_
<br/>_Default: `True`_

#### `--fixed-schedule-start-offset` `<int>`

Filter out trace requests before this timestamp in ms (must be >= 0). Only used with type='fixed_schedule'.

#### `--fixed-schedule-end-offset` `<int>`

Filter out trace requests after this timestamp in ms (must be >= 0). Only used with type='fixed_schedule'.

### Goodput

#### `--goodput` `<str>`

SLO (Service Level Objectives) configuration as a generic dict. Maps metric names to threshold values. A request is counted as good only if it meets ALL specified thresholds.

### Output

#### `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts. Created if it doesn't exist.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file` `<str>`

Filename prefix for all exported files. Example: 'my_run' produces 'my_run_summary.json', 'my_run_records.jsonl'.

#### `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. summary: Only aggregate metrics files. records: Includes per-request metrics. raw: Includes raw request/response data.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` | _default_ | Export only aggregated/summarized metrics (default, most compact) |
| `records` |  | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration` `<float>`

Time slice duration in seconds for trend analysis (must be > 0). Divides benchmark into windows for per-window statistics. Supports: 30, '30s', '5m', '2h'.

### HTTP Trace

#### `--export-http-trace`

Export HTTP trace data for debugging.
<br/>_Flag (no value required)_

#### `--show-trace-timing`

Display HTTP trace timing metrics in console output. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration.
<br/>_Flag (no value required)_

### Tokenizer

#### `--tokenizer` `<str>`

HuggingFace tokenizer identifier or local filesystem path. Should match the model's tokenizer for accurate token counts. Example: 'meta-llama/Llama-3.1-8B-Instruct'.

#### `--tokenizer-revision` `<str>`

Model revision to use: branch name, tag, or commit hash. Use for version pinning to ensure reproducibility.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`

Allow execution of custom tokenizer code from the repository. Required for some models but poses security risk. Only enable for trusted sources.
<br/>_Flag (no value required)_

### Load Generator

#### `--benchmark-duration` `<float>`

Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'.

#### `--benchmark-grace-period` `<float>`

Seconds to wait for in-flight requests at phase end (must be >= 0). Supports: 30, '30s', '2m'.

#### `--concurrency` `<int>`

Max concurrent in-flight requests (must be >= 1). For concurrency type: primary control. For rate types: acts as a cap.

#### `--prefill-concurrency` `<int>`

Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received.

#### `--request-rate` `<float>`

Target request rate in requests per second (must be > 0). Required for poisson/gamma/constant types. For user_centric: global rate shared across all users.

#### `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. constant: Generate requests at a fixed rate. poisson: Generate requests using a poisson distribution. gamma: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Gamma distribution shape parameter (must be > 0). Only used with type='gamma'. 1.0 = Poisson, &lt;1 = bursty, >1 = regular.

#### `--request-count`, `--num-requests` `<int>`

Stop after this many requests sent (must be >= 1).

#### `--concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--prefill-concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

### Warmup

#### `--request-rate-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--warmup-request-count`, `--num-warmup-requests` `<int>`

Warmup phase: Stop after this many requests sent (must be >= 1). If not set, uses the --request-count value.

#### `--warmup-duration` `<float>`

Warmup phase: Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'. If not set, uses the --benchmark-duration value.

#### `--num-warmup-sessions` `<int>`

Warmup phase: Stop after this many sessions completed (must be >= 1). If not set, uses the --conversation-num value.

#### `--warmup-concurrency` `<int>`

Warmup phase: Max concurrent in-flight requests (must be >= 1). For concurrency type: primary control. For rate types: acts as a cap. If not set, uses the --concurrency value.

#### `--warmup-prefill-concurrency` `<int>`

Warmup phase: Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received. If not set, uses the --prefill-concurrency value.

#### `--warmup-request-rate` `<float>`

Warmup phase: Target request rate in requests per second (must be > 0). Required for poisson/gamma/constant types. For user_centric: global rate shared across all users. If not set, uses the --request-rate value.

#### `--warmup-arrival-pattern` `<str>`

Warmup phase: Load generation type. concurrency: concurrency-controlled immediate dispatch, poisson/gamma/constant: rate-controlled with arrival distribution, user_centric: N users sharing global rate, fixed_schedule: replay from timestamps. If not set, uses the --arrival-pattern value.

#### `--warmup-grace-period` `<float>`

Warmup phase: Seconds to wait for in-flight requests at phase end (must be >= 0). Supports: 30, '30s', '2m'. If not set, uses the --benchmark-grace-period value.

#### `--warmup-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --concurrency-ramp-duration value.

#### `--warmup-prefill-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --prefill-concurrency-ramp-duration value.

#### `--warmup-request-rate-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --request-rate-ramp-duration value.

### User-Centric Rate

#### `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.

#### `--num-users` `<int>`

Number of simulated concurrent users (must be >= 1). Required for user_centric type. Requests distributed across users to achieve global rate.

### Request Cancellation

#### `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after --request-cancellation-delay seconds.

#### `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means send the full request, then immediately disconnect. Requires --request-cancellation-rate to be set.
<br/>_Default: `0.0`_

### Conversation Input

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time. Set to 0 for back-to-back turns.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays. Values &lt; 1 speed up, > 1 slow down.
<br/>_Default: `1.0`_

#### `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Set to 1 for single-turn.
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation.
<br/>_Default: `0`_

#### `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

Stop after this many sessions completed (must be >= 1).

#### `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents a unique prompt with sampled ISL/OSL. Entries are reused across conversations and turns according to the sampling strategy. Higher values provide more diversity.
<br/>_Default: `100`_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in mooncake_trace datasets. When hash_ids are provided in trace entries, prompts are divided into blocks of this size. Each hash_id maps to a cached block, enabling simulation of KV-cache sharing patterns from production workloads. Total prompt length = (num_hash_ids - 1) * block_size + final_block_size.

#### `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: ISL,OSL:prob;ISL,OSL:prob (probabilities 0-100 summing to 100).

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

### Prompt

#### `-b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by embeddings and rankings endpoint types where models can process multiple inputs simultaneously. Set to 1 for single-input requests. Not applicable to chat or completions endpoints.
<br/>_Default: `1`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for KV cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes are randomly selected from pool per request. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

Token length for each prefix prompt in the pool. Only used when pool_size is set. Note: due to prefix and user prompts being concatenated, the final prompt token count may be off by one. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. First part of a two-part prefix structure with high cache hit rate expected. Mutually exclusive with pool_size/length.

#### `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Second part of two-part prefix structure with lower cache hit rate expected. Mutually exclusive with pool_size/length.

### Rankings

#### `--rankings-passages-mean` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `10`_

#### `--rankings-passages-stddev` `<int>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `128`_

#### `--rankings-passages-prompt-token-stddev` `<int>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `32`_

#### `--rankings-query-prompt-token-stddev` `<int>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces. 1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in the radix tree. 1.5 means prefix branches are 50%% longer.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across. Higher values increase prefix diversity.
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths. 2.0 means prompts are 2x longer.
<br/>_Default: `1.0`_

#### `--synthesis-max-isl` `<int>`

Maximum input sequence length filter. Traces with input_length > max_isl are skipped entirely.

#### `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to this value (not filtered).

### Audio Input

#### `--audio-length-mean` `<float>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `10.0`_

#### `--audio-length-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

#### `--audio-batch-size`, `--batch-size-audio` `<int>`

Number of audio inputs to include in each multimodal request. Supported with chat endpoint type for multimodal models. Set to 0 to disable audio inputs.
<br/>_Default: `0`_

#### `--audio-format` `<str>`

File format for generated audio files. wav: uncompressed PCM (larger files). mp3: compressed (smaller files). Format affects file size in multimodal requests but not audio characteristics.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from. Each audio file is assigned a random depth from this list. Common values: 8 (low quality), 16 (CD quality), 24 (professional), 32 (high-end). Specify multiple values for mixed-quality testing.

#### `--audio-sample-rates` `<list>`

List of audio sample rates in kHz to randomly select from. Common values: 8.0 (telephony), 16.0 (speech), 44.1 (CD quality), 48.0 (professional). Specify multiple values for mixed-quality testing.

#### `--audio-num-channels` `<int>`

Number of audio channels. 1 = mono (single channel), 2 = stereo (left/right channels). Stereo doubles file size. Most speech models use mono.
<br/>_Default: `1`_

### Image Input

#### `--image-height-mean` `<float>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `512.0`_

#### `--image-height-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

#### `--image-width-mean` `<float>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `512.0`_

#### `--image-width-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with chat endpoint type for vision-language models. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Default: `0`_

#### `--image-format` `<str>`

Image file format for generated images. png: lossless compression (larger files, best quality). jpeg: lossy compression (smaller files, good quality). random: randomly select between PNG and JPEG per image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` |  | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` | _default_ | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

### Video Input

#### `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with chat endpoint type for video understanding models. Set to 0 to disable video inputs. Higher batch sizes significantly increase request payload size.
<br/>_Default: `0`_

#### `--video-duration` `<float>`

Duration in seconds for each generated video clip. Combined with fps, determines total frame count (frames = duration * fps). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing.
<br/>_Default: `1.0`_

#### `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: 4 (minimal, recommended for Cosmos models), 24 (cinematic), 30 (standard), 60 (high frame rate). Total frames = duration * fps.
<br/>_Default: `4`_

#### `--video-width` `<int>`

Video frame width in pixels. Determines video resolution and file size. Common values: 640 (SD), 1280 (HD), 1920 (Full HD). If not specified, uses codec/format defaults.

#### `--video-height` `<int>`

Video frame height in pixels. Combined with width determines aspect ratio and total pixel count per frame. Common values: 480 (SD), 720 (HD), 1080 (Full HD). If not specified, uses codec/format defaults.

#### `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format` `<str>`

Container format for generated video files. webm: VP9 codec, BSD-licensed, recommended for open-source workflows. mp4: H.264/H.265, widely compatible. avi: legacy, larger files. mkv: Matroska, flexible container. Format affects compatibility, file size, and encoding options.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec` `<str>`

Video codec for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL, widely compatible), libx265 (CPU, GPL, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

### Service

#### `--log-level` `<str>`

Global logging verbosity level. trace: most verbose, error: least verbose.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `TRACE` |  | Most verbose. Logs all operations including ZMQ messages and internal state changes. |
| `DEBUG` |  | Detailed debugging information. Logs function calls and important state transitions. |
| `INFO` | _default_ | General informational messages. Default level showing benchmark progress and results. |
| `NOTICE` |  | Important informational messages that are more significant than INFO but not warnings. |
| `WARNING` |  | Warning messages for potentially problematic situations that don't prevent execution. |
| `SUCCESS` |  | Success messages for completed operations and milestones. |
| `ERROR` |  | Error messages for failures that prevent specific operations but allow continued execution. |
| `CRITICAL` |  | Critical errors that may cause the benchmark to fail or produce invalid results. |

#### `-v`, `--verbose`

Equivalent to --log-level DEBUG. Enables detailed logging and switches UI to simple mode.
<br/>_Flag (no value required)_

#### `-vv`, `--extra-verbose`

Equivalent to --log-level TRACE. Most verbose logging including ZMQ messages. Switches UI to simple mode.
<br/>_Flag (no value required)_

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of parallel record processors. null = auto-detect based on CPU cores.

### Server Metrics

#### `--server-metrics` `<list>`

Server metrics collection (ENABLED BY DEFAULT). Optionally specify additional Prometheus endpoint URLs. Use --no-server-metrics to disable.

#### `--no-server-metrics`

Disable server metrics collection entirely.

#### `--server-metrics-formats` `<list>`

Export formats for scraped metrics. Options: json, csv, parquet, jsonl.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `json` |  | Export aggregated statistics in JSON hybrid format with metrics keyed by name. Best for: Programmatic access, CI/CD pipelines, automated analysis. |
| `csv` |  | Export aggregated statistics in CSV tabular format organized by metric type. Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames. |
| `jsonl` |  | Export raw time-series records in line-delimited JSON format. Best for: Time-series analysis, debugging, visualizing metric evolution. Warning: Can generate very large files for long-running benchmarks. |
| `parquet` |  | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry and optionally specify: 'dashboard' for realtime mode, custom DCGM URLs, or a metrics CSV file.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### UI

#### `--ui-type`, `--ui` `<str>`

User interface mode. dashboard: rich interactive UI, simple: text progress, none: silent operation.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

### Workers

#### `--workers-max`, `--max-workers` `<int>`

Maximum worker processes. null = auto-detect based on CPU cores.

### ZMQ Communication

#### `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services.

#### `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC socket files for local inter-process communication.

### Multi-Run

#### `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. When 1, runs a single benchmark. When >1, computes aggregate statistics across runs.
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Allows the system to stabilize between runs.
<br/>_Default: `0.0`_

#### `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Common values: 0.90 (90%%), 0.95 (95%%), 0.99 (99%%).
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for runs after the first. When true, only the first run includes warmup for steady-state measurement.
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--no-set-consistent-seed`

Automatically set random seed for consistent workloads across runs. When true, sets random_seed=42 if not specified, ensuring identical workloads for valid statistical comparison.
<br/>_Default: `True`_

### Accuracy

#### `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode.

#### `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). If not set, all tasks are included.

#### `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation. Maximum 8.
<br/>_Default: `0`_

#### `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation.
<br/>_Flag (no value required)_

#### `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution).

#### `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

<hr/>

## `aiperf plot`

Generate visualizations from AIPerf profiling data.

On first run, automatically creates ~/.aiperf/plot_config.yaml which you can edit to customize plots, including experiment classification (baseline vs treatment runs). Use --config to specify a different config file.

_**Note:** PNG export requires Chrome or Chromium to be installed on your system, as it is used by kaleido to render Plotly figures to static images._

_**Note:** The plot command expects default export filenames (e.g., `profile_export.jsonl`). Runs created with `--profile-export-file` or custom `--profile-export-prefix` use different filenames and will not be detected by the plot command._

**Examples:**

```bash
# Generate plots (auto-creates ~/.aiperf/plot_config.yaml on first run)
aiperf plot

# Use custom config
aiperf plot --config my_plots.yaml

# Show detailed error tracebacks
aiperf plot --verbose
```

#### `--paths`, `--empty-paths` `<list>`

Paths to profiling run directories. Defaults to ./artifacts if not specified.

#### `--output` `<str>`

Directory to save generated plots. Defaults to &lt;first_path>/plots if not specified.

#### `--theme` `<str>`

Plot theme to use: 'light' (white background) or 'dark' (dark background). Defaults to 'light'.
<br/>_Default: `light`_

#### `--config` `<str>`

Path to custom plot configuration YAML file. If not specified, auto-creates and uses ~/.aiperf/plot_config.yaml.

#### `--verbose`, `--no-verbose`

Show detailed error tracebacks in console (errors are always logged to ~/.aiperf/plot.log).

#### `--dashboard`, `--no-dashboard`

Launch interactive dashboard server instead of generating static PNGs.

#### `--host` `<str>`

Host for dashboard server (only used with --dashboard). Defaults to 127.0.0.1.
<br/>_Default: `127.0.0.1`_

#### `--port` `<int>`

Port for dashboard server (only used with --dashboard). Defaults to 8050.
<br/>_Default: `8050`_

<hr/>

## `aiperf plugins`

Explore AIPerf plugins: aiperf plugins [category] [type]

#### `--category` `<str>`

Category to explore.
<br/>_Choices: [`accuracy_benchmark`, `accuracy_grader`, `api_router`, `arrival_pattern`, `communication`, `communication_client`, `console_exporter`, `custom_dataset_loader`, `data_exporter`, `dataset_backing_store`, `dataset_client_store`, `dataset_composer`, `dataset_sampler`, `endpoint`, `gpu_telemetry_collector`, `plot`, `ramp`, `record_processor`, `results_processor`, `service`, `service_manager`, `timing_strategy`, `transport`, `ui`, `url_selection_strategy`, `zmq_proxy`]_

#### `--name` `<str>`

Type name for details.

#### `-a`, `--all`, `--no-all`

Show all categories and plugins.

#### `-v`, `--validate`, `--no-validate`

Validate plugins.yaml.

<hr/>

## `aiperf service`

Run an AIPerf service in a single process.

_Advanced use only — intended for developers and Kubernetes/distributed deployments where services run in separate containers or nodes._

For standard single-node benchmarking, use the `aiperf profile` command instead.

#### `--type` `<str>` _(Required)_

Service type to run.
<br/>_Choices: [`api`, `dataset_manager`, `gpu_telemetry_manager`, `record_processor`, `records_manager`, `server_metrics_manager`, `system_controller`, `timing_manager`, `worker`, `worker_manager`, `worker_pod_manager`]_

#### `--user-config-file` `<str>`

Path to the user configuration file (JSON or YAML). Falls back to AIPERF_CONFIG_USER_FILE environment variable.

#### `--service-config-file` `<str>`

Path to the service configuration file (JSON or YAML). Falls back to AIPERF_CONFIG_SERVICE_FILE environment variable, then to default ServiceConfig if neither is set.

#### `--service-id` `<str>`

Unique identifier for the service instance. Useful when running multiple instances of the same service type.

#### `--health-host` `<str>`

Host to bind the health server to. Falls back to AIPERF_SERVICE_HEALTH_HOST environment variable.

#### `--health-port` `<int>`

HTTP port for health endpoints (/healthz, /readyz). Required for Kubernetes liveness and readiness probes. Falls back to AIPERF_SERVICE_HEALTH_PORT environment variable.

#### `--api-port` `<int>`

HTTP port for API endpoints (e.g., /api/dataset, /api/progress). Only used by services that expose HTTP APIs.

<hr/>

## `aiperf kube attach`

Attach to a running AIPerf benchmark and stream progress.

Connects to a running benchmark via port-forward and streams real-time progress updates via WebSocket. Press Ctrl+C to disconnect.

If no job_id is specified, uses the last deployed benchmark.

**Examples:**

```bash
# Attach to the last deployed benchmark
aiperf kube attach

# Attach to a specific job
aiperf kube attach abc123

# Attach to a job in a specific namespace
aiperf kube attach abc123 --namespace aiperf-bench

# Use a different local port
aiperf kube attach abc123 --port 9091
```

### Parameters

#### `--job-id` `<str>`

The AIPerf job ID to attach to (default: last deployed job).

#### `-p`, `--port` `<int>`

Local port for port-forward (default: 0 = ephemeral).
<br/>_Default: `0`_

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace.

<hr/>

## `aiperf kube cancel`

Cancel a running AIPerf benchmark.

Deletes the JobSet and all associated resources. Results are lost once cancelled. Run 'aiperf kube results' first to save partial results.

Use 'aiperf kube delete' instead to clean up already-completed jobs, optionally removing the auto-generated namespace.

If no job_id is specified, uses the last deployed benchmark.

**Examples:**

```bash
# Cancel the last deployed job (with confirmation)
aiperf kube cancel

# Cancel a specific job
aiperf kube cancel abc123

# Force cancel without confirmation
aiperf kube cancel --force
```

### Parameters

#### `--job-id` `<str>`

The AIPerf job ID to cancel (default: last deployed job).

#### `-f`, `--force`, `--no-force`

Skip confirmation prompt.

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace.

<hr/>

## `aiperf kube debug`

Run diagnostic analysis on a benchmark deployment.

Inspects pod states, events, node resources, and container logs to identify problems. Outputs a structured report with suggestions.

**Examples:**

```bash
aiperf kube debug -n my-benchmark
aiperf kube debug --job-id abc123 -v
aiperf kube debug -A
```

#### `-n`, `--namespace` `<str>`

.

#### `-j`, `--job-id` `<str>`

.

#### `--kubeconfig` `<str>`

.

#### `--context` `<str>`

.

#### `-v`, `--verbose`, `--no-verbose`

.

#### `-A`, `--all-namespaces`, `--no-all-namespaces`

.

<hr/>

## `aiperf kube cleanup`

Remove stale benchmark namespaces older than max-age seconds.

Finds namespaces with aiperf labels that are older than the specified max-age and removes them. By default, only shows what would be deleted (dry-run mode). Use --force to actually delete.

**Examples:**

```bash
aiperf kube cleanup                    # Dry-run, show stale namespaces
aiperf kube cleanup --force            # Actually delete stale namespaces
aiperf kube cleanup --max-age 7200     # 2 hour threshold
aiperf kube cleanup --dry-run          # Preview what would be deleted
```

#### `--max-age` `<int>`

.
<br/>_Default: `3600`_

#### `--dry-run`, `--no-dry-run`

.

#### `-f`, `--force`, `--no-force`

.

#### `-l`, `--label` `<str>`

.
<br/>_Default: `aiperf/job-id`_

#### `--kubeconfig` `<str>`

.

#### `--context` `<str>`

.

<hr/>

## `aiperf kube shutdown`

Shut down the controller pod's API service after a benchmark finishes.

After a benchmark completes, the controller pod keeps its API service running so you can retrieve results. This command signals the API to shut down gracefully, allowing the controller pod to exit and release resources. The request is rejected if the benchmark is still running.

Equivalent to 'aiperf kube results --shutdown' without downloading results.

If no job_id is specified, uses the last deployed benchmark.

**Examples:**

```bash
# Shut down the API for the last deployed benchmark
aiperf kube shutdown

# Shut down the API for a specific benchmark
aiperf kube shutdown abc123
```

### Parameters

#### `--job-id` `<str>`

The AIPerf job ID (default: last deployed job).

#### `--port` `<int>`

Local port for API port-forward (default: 0 = ephemeral).
<br/>_Default: `0`_

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace.

<hr/>

## `aiperf kube delete`

Delete an AIPerf benchmark job and clean up resources.

Removes the JobSet, ConfigMap, RBAC resources, and pods. Unlike 'cancel', this is intended for cleaning up completed or failed jobs. Optionally deletes the namespace if it was auto-generated (aiperf-{job_id} format).

If no job_id is specified, uses the last deployed benchmark.

**Examples:**

```bash
# Delete the last deployed job (with confirmation)
aiperf kube delete

# Delete a specific job
aiperf kube delete abc123

# Force delete without confirmation
aiperf kube delete --force

# Also delete the auto-generated namespace
aiperf kube delete --delete-namespace
```

### Parameters

#### `--job-id` `<str>`

The AIPerf job ID to delete (default: last deployed job).

#### `-f`, `--force`, `--no-force`

Skip confirmation prompt.

#### `--delete-namespace`, `--no-delete-namespace`

Also delete the namespace if it was auto-generated.

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace.

<hr/>

## `aiperf kube generate`

Generate Kubernetes YAML manifests for an AIPerf benchmark.

Outputs YAML manifests to stdout for GitOps workflows or manual review. Use 'aiperf kube profile' to deploy directly to a cluster.

**Examples:**

```bash
# Generate manifests with CLI options
aiperf kube generate --model Qwen/Qwen3-0.6B --url localhost:8000 --image aiperf:latest

# Generate and save to file
aiperf kube generate --model Qwen/Qwen3-0.6B --url localhost:8000 --image aiperf:latest > k8s-deployment.yaml
```

### Endpoint

#### `-m`, `--model-names`, `--model` `<list>`

List of model configurations. At least one model required.

#### `--model-selection-strategy` `<str>`

Strategy for selecting models when multiple are configured. round_robin cycles through models, random selects randomly, weighted uses configured weights, modality_aware routes by input type.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select models based on configured weights. Each model's weight determines its selection probability. |

#### `-u`, `--url` `<list>`

List of server URLs to benchmark. Requests distributed according to url_strategy. Example: ['http://localhost:8000/v1/chat/completions'].
<br/>_Default: `['localhost:8000']`_

#### `--url-strategy` `<str>`

Strategy for distributing requests across multiple URLs. round_robin cycles through URLs in order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--endpoint-type` `<str>`

API endpoint type determining request/response format. chat: OpenAI chat completions, completions: OpenAI completions, embeddings: vector embeddings, rankings: reranking, template: custom format.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`

Enable streaming (Server-Sent Events) responses. Required for accurate TTFT (time to first token) measurement. Server must support streaming for this to work.
<br/>_Flag (no value required)_

#### `--custom-endpoint`, `--endpoint` `<str>`

Override default endpoint path. Use for servers with non-standard API paths. Example: '/custom/v2/generate'.

#### `--api-key` `<str>`

API authentication key. Supports environment variable substitution: ${OPENAI_API_KEY}. Can also use ${VAR:default} syntax for defaults.

#### `--request-timeout-seconds` `<float>`

Request timeout in seconds. Requests exceeding this duration are marked as failed. Should exceed expected max response time.
<br/>_Default: `600.0`_

#### `--transport`, `--transport-type` `<str>`

HTTP transport protocol (http/https). Auto-detected from URL scheme if not specified. Explicit setting overrides auto-detection.
<br/>_Choices: [`http`]_

#### `--use-legacy-max-tokens`

Use 'max_tokens' field instead of 'max_completion_tokens'. Enable for compatibility with older OpenAI API versions.
<br/>_Flag (no value required)_

#### `--use-server-token-count`

Use server-reported token counts from response usage field. When true, trusts usage.prompt_tokens and usage.completion_tokens. When false, counts tokens locally using configured tokenizer.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy` `<str>`

HTTP connection management strategy. pooled: shared connection pool (fastest), never: new connection per request (includes TCP overhead), sticky_sessions: dedicated connection per session.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--download-video-content`

For video generation endpoints, download the video content after generation completes. When enabled, request latency includes the video download time. When disabled, only generation time is measured.
<br/>_Flag (no value required)_

#### `--extra-inputs` `<str>`

Additional fields to include in request body. Merged into every request. Common fields: temperature, top_p, top_k, stop.

#### `-H`, `--header` `<str>`

Custom HTTP headers to include in all requests. Useful for authentication, tracing, or routing. Values support environment variable substitution.

### Input

#### `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Can be absolute or relative. Supported formats depend on the format field: JSONL for single_turn/multi_turn, JSONL trace files for mooncake_trace, directories for random_pool.

#### `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking. AIPerf automatically downloads and parses these datasets.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `sharegpt` |  | ShareGPT dataset from HuggingFace. Multi-turn conversational dataset with user/assistant exchanges. |

#### `--custom-dataset-type` `<str>`

Dataset file format determining parsing logic and expected file structure. single_turn: JSONL with single prompt-response exchanges. multi_turn: JSONL with conversation history. mooncake_trace: timestamped trace files for replay. random_pool: directory of reusable prompts.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `single_turn` | _default_ | Simple prompt-response pairs. |
| `multi_turn` |  | Conversational data with multiple turns. |
| `mooncake_trace` |  | Mooncake production trace format. |
| `random_pool` |  | Treat file as a pool for random sampling. |

#### `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. sequential: iterate in order, wrapping to start after end. random: randomly sample with replacement (entries may repeat). shuffle: random permutation without replacement, re-shuffling after exhaustion.
<br/>_Choices: [`random`, `sequential`, `shuffle`]_
<br/>_Default: `sequential`_

#### `--random-seed` `<int>`

Global random seed for reproducibility. Can be overridden per-dataset. If not set, uses system entropy.

### Fixed Schedule

#### `--fixed-schedule`, `--no-fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for mooncake_trace datasets.

#### `--fixed-schedule-auto-offset`

Normalize trace timestamps to start at 0. Subtracts minimum timestamp from all entries. Only used with type='fixed_schedule'.
<br/>_Flag (no value required)_
<br/>_Default: `True`_

#### `--fixed-schedule-start-offset` `<int>`

Filter out trace requests before this timestamp in ms (must be >= 0). Only used with type='fixed_schedule'.

#### `--fixed-schedule-end-offset` `<int>`

Filter out trace requests after this timestamp in ms (must be >= 0). Only used with type='fixed_schedule'.

### Goodput

#### `--goodput` `<str>`

SLO (Service Level Objectives) configuration as a generic dict. Maps metric names to threshold values. A request is counted as good only if it meets ALL specified thresholds.

### Output

#### `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts. Created if it doesn't exist.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file` `<str>`

Filename prefix for all exported files. Example: 'my_run' produces 'my_run_summary.json', 'my_run_records.jsonl'.

#### `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. summary: Only aggregate metrics files. records: Includes per-request metrics. raw: Includes raw request/response data.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` | _default_ | Export only aggregated/summarized metrics (default, most compact) |
| `records` |  | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration` `<float>`

Time slice duration in seconds for trend analysis (must be > 0). Divides benchmark into windows for per-window statistics. Supports: 30, '30s', '5m', '2h'.

### HTTP Trace

#### `--export-http-trace`

Export HTTP trace data for debugging.
<br/>_Flag (no value required)_

#### `--show-trace-timing`

Display HTTP trace timing metrics in console output. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration.
<br/>_Flag (no value required)_

### Tokenizer

#### `--tokenizer` `<str>`

HuggingFace tokenizer identifier or local filesystem path. Should match the model's tokenizer for accurate token counts. Example: 'meta-llama/Llama-3.1-8B-Instruct'.

#### `--tokenizer-revision` `<str>`

Model revision to use: branch name, tag, or commit hash. Use for version pinning to ensure reproducibility.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`

Allow execution of custom tokenizer code from the repository. Required for some models but poses security risk. Only enable for trusted sources.
<br/>_Flag (no value required)_

### Load Generator

#### `--benchmark-duration` `<float>`

Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'.

#### `--benchmark-grace-period` `<float>`

Seconds to wait for in-flight requests at phase end (must be >= 0). Supports: 30, '30s', '2m'.

#### `--concurrency` `<int>`

Max concurrent in-flight requests (must be >= 1). For concurrency type: primary control. For rate types: acts as a cap.

#### `--prefill-concurrency` `<int>`

Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received.

#### `--request-rate` `<float>`

Target request rate in requests per second (must be > 0). Required for poisson/gamma/constant types. For user_centric: global rate shared across all users.

#### `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. constant: Generate requests at a fixed rate. poisson: Generate requests using a poisson distribution. gamma: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Gamma distribution shape parameter (must be > 0). Only used with type='gamma'. 1.0 = Poisson, &lt;1 = bursty, >1 = regular.

#### `--request-count`, `--num-requests` `<int>`

Stop after this many requests sent (must be >= 1).

#### `--concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--prefill-concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

### Warmup

#### `--request-rate-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--warmup-request-count`, `--num-warmup-requests` `<int>`

Warmup phase: Stop after this many requests sent (must be >= 1). If not set, uses the --request-count value.

#### `--warmup-duration` `<float>`

Warmup phase: Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'. If not set, uses the --benchmark-duration value.

#### `--num-warmup-sessions` `<int>`

Warmup phase: Stop after this many sessions completed (must be >= 1). If not set, uses the --conversation-num value.

#### `--warmup-concurrency` `<int>`

Warmup phase: Max concurrent in-flight requests (must be >= 1). For concurrency type: primary control. For rate types: acts as a cap. If not set, uses the --concurrency value.

#### `--warmup-prefill-concurrency` `<int>`

Warmup phase: Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received. If not set, uses the --prefill-concurrency value.

#### `--warmup-request-rate` `<float>`

Warmup phase: Target request rate in requests per second (must be > 0). Required for poisson/gamma/constant types. For user_centric: global rate shared across all users. If not set, uses the --request-rate value.

#### `--warmup-arrival-pattern` `<str>`

Warmup phase: Load generation type. concurrency: concurrency-controlled immediate dispatch, poisson/gamma/constant: rate-controlled with arrival distribution, user_centric: N users sharing global rate, fixed_schedule: replay from timestamps. If not set, uses the --arrival-pattern value.

#### `--warmup-grace-period` `<float>`

Warmup phase: Seconds to wait for in-flight requests at phase end (must be >= 0). Supports: 30, '30s', '2m'. If not set, uses the --benchmark-grace-period value.

#### `--warmup-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --concurrency-ramp-duration value.

#### `--warmup-prefill-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --prefill-concurrency-ramp-duration value.

#### `--warmup-request-rate-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --request-rate-ramp-duration value.

### User-Centric Rate

#### `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.

#### `--num-users` `<int>`

Number of simulated concurrent users (must be >= 1). Required for user_centric type. Requests distributed across users to achieve global rate.

### Request Cancellation

#### `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after --request-cancellation-delay seconds.

#### `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means send the full request, then immediately disconnect. Requires --request-cancellation-rate to be set.
<br/>_Default: `0.0`_

### Conversation Input

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time. Set to 0 for back-to-back turns.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays. Values &lt; 1 speed up, > 1 slow down.
<br/>_Default: `1.0`_

#### `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Set to 1 for single-turn.
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation.
<br/>_Default: `0`_

#### `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

Stop after this many sessions completed (must be >= 1).

#### `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents a unique prompt with sampled ISL/OSL. Entries are reused across conversations and turns according to the sampling strategy. Higher values provide more diversity.
<br/>_Default: `100`_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in mooncake_trace datasets. When hash_ids are provided in trace entries, prompts are divided into blocks of this size. Each hash_id maps to a cached block, enabling simulation of KV-cache sharing patterns from production workloads. Total prompt length = (num_hash_ids - 1) * block_size + final_block_size.

#### `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: ISL,OSL:prob;ISL,OSL:prob (probabilities 0-100 summing to 100).

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

### Prompt

#### `-b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by embeddings and rankings endpoint types where models can process multiple inputs simultaneously. Set to 1 for single-input requests. Not applicable to chat or completions endpoints.
<br/>_Default: `1`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for KV cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes are randomly selected from pool per request. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

Token length for each prefix prompt in the pool. Only used when pool_size is set. Note: due to prefix and user prompts being concatenated, the final prompt token count may be off by one. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. First part of a two-part prefix structure with high cache hit rate expected. Mutually exclusive with pool_size/length.

#### `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Second part of two-part prefix structure with lower cache hit rate expected. Mutually exclusive with pool_size/length.

### Rankings

#### `--rankings-passages-mean` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `10`_

#### `--rankings-passages-stddev` `<int>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `128`_

#### `--rankings-passages-prompt-token-stddev` `<int>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `32`_

#### `--rankings-query-prompt-token-stddev` `<int>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces. 1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in the radix tree. 1.5 means prefix branches are 50%% longer.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across. Higher values increase prefix diversity.
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths. 2.0 means prompts are 2x longer.
<br/>_Default: `1.0`_

#### `--synthesis-max-isl` `<int>`

Maximum input sequence length filter. Traces with input_length > max_isl are skipped entirely.

#### `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to this value (not filtered).

### Audio Input

#### `--audio-length-mean` `<float>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `10.0`_

#### `--audio-length-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

#### `--audio-batch-size`, `--batch-size-audio` `<int>`

Number of audio inputs to include in each multimodal request. Supported with chat endpoint type for multimodal models. Set to 0 to disable audio inputs.
<br/>_Default: `0`_

#### `--audio-format` `<str>`

File format for generated audio files. wav: uncompressed PCM (larger files). mp3: compressed (smaller files). Format affects file size in multimodal requests but not audio characteristics.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from. Each audio file is assigned a random depth from this list. Common values: 8 (low quality), 16 (CD quality), 24 (professional), 32 (high-end). Specify multiple values for mixed-quality testing.

#### `--audio-sample-rates` `<list>`

List of audio sample rates in kHz to randomly select from. Common values: 8.0 (telephony), 16.0 (speech), 44.1 (CD quality), 48.0 (professional). Specify multiple values for mixed-quality testing.

#### `--audio-num-channels` `<int>`

Number of audio channels. 1 = mono (single channel), 2 = stereo (left/right channels). Stereo doubles file size. Most speech models use mono.
<br/>_Default: `1`_

### Image Input

#### `--image-height-mean` `<float>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `512.0`_

#### `--image-height-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

#### `--image-width-mean` `<float>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `512.0`_

#### `--image-width-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with chat endpoint type for vision-language models. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Default: `0`_

#### `--image-format` `<str>`

Image file format for generated images. png: lossless compression (larger files, best quality). jpeg: lossy compression (smaller files, good quality). random: randomly select between PNG and JPEG per image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` |  | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` | _default_ | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

### Video Input

#### `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with chat endpoint type for video understanding models. Set to 0 to disable video inputs. Higher batch sizes significantly increase request payload size.
<br/>_Default: `0`_

#### `--video-duration` `<float>`

Duration in seconds for each generated video clip. Combined with fps, determines total frame count (frames = duration * fps). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing.
<br/>_Default: `1.0`_

#### `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: 4 (minimal, recommended for Cosmos models), 24 (cinematic), 30 (standard), 60 (high frame rate). Total frames = duration * fps.
<br/>_Default: `4`_

#### `--video-width` `<int>`

Video frame width in pixels. Determines video resolution and file size. Common values: 640 (SD), 1280 (HD), 1920 (Full HD). If not specified, uses codec/format defaults.

#### `--video-height` `<int>`

Video frame height in pixels. Combined with width determines aspect ratio and total pixel count per frame. Common values: 480 (SD), 720 (HD), 1080 (Full HD). If not specified, uses codec/format defaults.

#### `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format` `<str>`

Container format for generated video files. webm: VP9 codec, BSD-licensed, recommended for open-source workflows. mp4: H.264/H.265, widely compatible. avi: legacy, larger files. mkv: Matroska, flexible container. Format affects compatibility, file size, and encoding options.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec` `<str>`

Video codec for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL, widely compatible), libx265 (CPU, GPL, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

### Service

#### `--log-level` `<str>`

Global logging verbosity level. trace: most verbose, error: least verbose.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `TRACE` |  | Most verbose. Logs all operations including ZMQ messages and internal state changes. |
| `DEBUG` |  | Detailed debugging information. Logs function calls and important state transitions. |
| `INFO` | _default_ | General informational messages. Default level showing benchmark progress and results. |
| `NOTICE` |  | Important informational messages that are more significant than INFO but not warnings. |
| `WARNING` |  | Warning messages for potentially problematic situations that don't prevent execution. |
| `SUCCESS` |  | Success messages for completed operations and milestones. |
| `ERROR` |  | Error messages for failures that prevent specific operations but allow continued execution. |
| `CRITICAL` |  | Critical errors that may cause the benchmark to fail or produce invalid results. |

#### `-v`, `--verbose`

Equivalent to --log-level DEBUG. Enables detailed logging and switches UI to simple mode.
<br/>_Flag (no value required)_

#### `-vv`, `--extra-verbose`

Equivalent to --log-level TRACE. Most verbose logging including ZMQ messages. Switches UI to simple mode.
<br/>_Flag (no value required)_

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of parallel record processors. null = auto-detect based on CPU cores.

### Server Metrics

#### `--server-metrics` `<list>`

Server metrics collection (ENABLED BY DEFAULT). Optionally specify additional Prometheus endpoint URLs. Use --no-server-metrics to disable.

#### `--no-server-metrics`

Disable server metrics collection entirely.

#### `--server-metrics-formats` `<list>`

Export formats for scraped metrics. Options: json, csv, parquet, jsonl.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `json` |  | Export aggregated statistics in JSON hybrid format with metrics keyed by name. Best for: Programmatic access, CI/CD pipelines, automated analysis. |
| `csv` |  | Export aggregated statistics in CSV tabular format organized by metric type. Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames. |
| `jsonl` |  | Export raw time-series records in line-delimited JSON format. Best for: Time-series analysis, debugging, visualizing metric evolution. Warning: Can generate very large files for long-running benchmarks. |
| `parquet` |  | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry and optionally specify: 'dashboard' for realtime mode, custom DCGM URLs, or a metrics CSV file.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### UI

#### `--ui-type`, `--ui` `<str>`

User interface mode. dashboard: rich interactive UI, simple: text progress, none: silent operation.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

### Workers

#### `--workers-max`, `--max-workers` `<int>`

Maximum worker processes. null = auto-detect based on CPU cores.

### ZMQ Communication

#### `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services.

#### `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC socket files for local inter-process communication.

### Multi-Run

#### `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. When 1, runs a single benchmark. When >1, computes aggregate statistics across runs.
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Allows the system to stabilize between runs.
<br/>_Default: `0.0`_

#### `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Common values: 0.90 (90%%), 0.95 (95%%), 0.99 (99%%).
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for runs after the first. When true, only the first run includes warmup for steady-state measurement.
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--no-set-consistent-seed`

Automatically set random seed for consistent workloads across runs. When true, sets random_seed=42 if not specified, ensuring identical workloads for valid statistical comparison.
<br/>_Default: `True`_

### Accuracy

#### `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode.

#### `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). If not set, all tasks are included.

#### `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation. Maximum 8.
<br/>_Default: `0`_

#### `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation.
<br/>_Flag (no value required)_

#### `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution).

#### `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace.

#### `--name` `<str>`

Human-readable name for the benchmark job (DNS label, max 40 chars).

#### `--image` `<str>` _(Required)_

AIPerf container image to use for Kubernetes deployment.

#### `--image-pull-policy` `<str>`

Image pull policy (Always, IfNotPresent, Never). Use 'Never' for minikube (or local clusters) with locally loaded images.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `Always` |  | Every time the kubelet launches a container, it queries the registry to resolve the name to a digest. Uses cached image if digest matches, otherwise pulls the image. |
| `Never` |  | The kubelet does not try fetching the image. Startup fails if the image is not already present locally. |
| `IfNotPresent` |  | The image is pulled only if it is not already present locally. |

#### `--workers-max` `<int>`

Total number of workers. Automatically distributed across pods based on --workers-per-pod (default 10). E.g., --workers-max 50 = 5 pods × 10 workers.
<br/>_Default: `10`_

#### `--ttl-seconds` `<int>`

Seconds to keep pods after completion (None to disable TTL).
<br/>_Default: `300`_

### Kubernetes Node Placement

#### `--node-selector` `<str>`

Node selector labels (e.g., {'gpu': 'true'}).
<br/>_Default: `{}`_

#### `--tolerations` `<list>`

Pod tolerations for scheduling on tainted nodes.
<br/>_Default: `[]`_

### Kubernetes Scheduling

#### `--queue-name` `<str>`

Kueue LocalQueue name for gang-scheduling. When set, the JobSet is submitted to Kueue for quota-managed admission.

#### `--priority-class` `<str>`

Kueue WorkloadPriorityClass name for scheduling priority.

### Kubernetes Metadata

#### `--annotations` `<str>`

Additional pod annotations.
<br/>_Default: `{}`_

#### `--labels` `<str>`

Additional pod labels.
<br/>_Default: `{}`_

### Kubernetes Secrets

#### `--image-pull-secrets` `<list>`

Image pull secret names.
<br/>_Default: `[]`_

#### `--env-vars` `<str>`

Extra environment variables (key: value).
<br/>_Default: `{}`_

#### `--env-from-secrets` `<str>`

Environment variables from secrets (ENV_NAME: secret_name/key).
<br/>_Default: `{}`_

#### `--secret-mounts` `<list>`

Secret volume mounts.
<br/>_Default: `[]`_

#### `--service-account` `<str>`

Service account name for pods.

<hr/>

## `aiperf kube init`

Generate a starter configuration template for Kubernetes benchmarks.

Outputs a commented YAML template with common configuration sections. Without --output, prints to stdout (suitable for piping). With --output, writes to a file.

**Examples:**

```bash
# Print template to stdout
aiperf kube init

# Pipe to a file
aiperf kube init > benchmark.yaml

# Write to a specific file
aiperf kube init --output benchmark.yaml
```

#### `-o`, `--output` `<str>`

Output file path. If not specified, prints to stdout.

<hr/>

## `aiperf kube logs`

Get logs from AIPerf benchmark pods.

Shows logs from all pods and containers associated with the job. If no job_id is specified, uses the last deployed benchmark.

Use --output to save logs to a directory instead of printing to stdout. Each pod's logs are saved as {pod-name}.log.

**Examples:**

```bash
# Get logs from last deployed job
aiperf kube logs

# Get logs from a specific job
aiperf kube logs abc123

# Get logs from a specific container
aiperf kube logs --container control-plane

# Follow logs in real-time
aiperf kube logs -f

# Get last 100 lines
aiperf kube logs --tail 100

# Save logs to a directory
aiperf kube logs --output ./my-logs
```

### Parameters

#### `--job-id` `<str>`

The AIPerf job ID to get logs from (default: last deployed job).

#### `--container` `<str>`

Specific container name to get logs from.

#### `-f`, `--follow`, `--no-follow`

Follow log output in real-time.

#### `--tail` `<int>`

Number of lines from the end to show.

#### `-o`, `--output` `<str>`

Directory to save log files (one per pod). Prints to stdout if not set.

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace.

<hr/>

## `aiperf kube preflight`

Run comprehensive pre-flight checks for Kubernetes deployment.

Validates cluster connectivity, permissions, resources, and configuration before deploying a benchmark. Useful for debugging cluster setup issues.

Checks performed: - Cluster connectivity and Kubernetes version - Namespace availability - RBAC permissions for all required operations - JobSet CRD installation and controller health - Resource quotas and node capacity - Secret existence (if specified) - Image pull information - Network policies - DNS resolution (CoreDNS) - Endpoint connectivity (if specified)

**Examples:**

```bash
# Run basic pre-flight checks
aiperf kube preflight

# Check a specific namespace with resource estimation
aiperf kube preflight --namespace aiperf-bench --workers-max 10

# Verify all deployment requirements
aiperf kube preflight \
    --image myregistry.io/aiperf:latest \
    --image-pull-secret my-registry-creds \
    --secret api-key \
    --endpoint http://my-llm:8000/v1
```

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace.

### Parameters

#### `--image` `<str>`

Container image to verify pull access for.

#### `--image-pull-secret`, `--empty-image-pull-secret` `<list>`

Image pull secret names to verify exist.

#### `--secret`, `--empty-secret` `<list>`

Secret names to verify exist (for env vars or mounts).

#### `--endpoint` `<str>`

LLM endpoint URL to verify connectivity.

#### `--workers-max` `<int>`

Planned number of workers for resource estimation.
<br/>_Default: `1`_

<hr/>

## `aiperf kube profile`

Run a benchmark in Kubernetes.

By default, blocks and streams controller logs until completion. Use --detach for fire-and-forget deployment.

Before deploying, validates that the LLM endpoint is reachable. Use --skip-endpoint-check to bypass this validation.

**Examples:**

```bash
# Stream controller logs (default)
aiperf kube profile --model Qwen/Qwen3-0.6B \
    --url http://server:8000 --image aiperf:latest --workers-max 10

# CI/CD: deploy and exit immediately
aiperf kube profile --model Qwen/Qwen3-0.6B \
    --url http://server:8000 --image aiperf:latest --detach
```

### Endpoint

#### `-m`, `--model-names`, `--model` `<list>`

List of model configurations. At least one model required.

#### `--model-selection-strategy` `<str>`

Strategy for selecting models when multiple are configured. round_robin cycles through models, random selects randomly, weighted uses configured weights, modality_aware routes by input type.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `round_robin` | _default_ | Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models). |
| `random` |  | Randomly select a model for each prompt using uniform distribution. |
| `weighted` |  | Select models based on configured weights. Each model's weight determines its selection probability. |

#### `-u`, `--url` `<list>`

List of server URLs to benchmark. Requests distributed according to url_strategy. Example: ['http://localhost:8000/v1/chat/completions'].
<br/>_Default: `['localhost:8000']`_

#### `--url-strategy` `<str>`

Strategy for distributing requests across multiple URLs. round_robin cycles through URLs in order.
<br/>_Choices: [`round_robin`]_
<br/>_Default: `round_robin`_

#### `--endpoint-type` `<str>`

API endpoint type determining request/response format. chat: OpenAI chat completions, completions: OpenAI completions, embeddings: vector embeddings, rankings: reranking, template: custom format.
<br/>_Choices: [`chat`, `cohere_rankings`, `completions`, `chat_embeddings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `video_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `solido_rag`, `template`]_
<br/>_Default: `chat`_

#### `--streaming`

Enable streaming (Server-Sent Events) responses. Required for accurate TTFT (time to first token) measurement. Server must support streaming for this to work.
<br/>_Flag (no value required)_

#### `--custom-endpoint`, `--endpoint` `<str>`

Override default endpoint path. Use for servers with non-standard API paths. Example: '/custom/v2/generate'.

#### `--api-key` `<str>`

API authentication key. Supports environment variable substitution: ${OPENAI_API_KEY}. Can also use ${VAR:default} syntax for defaults.

#### `--request-timeout-seconds` `<float>`

Request timeout in seconds. Requests exceeding this duration are marked as failed. Should exceed expected max response time.
<br/>_Default: `600.0`_

#### `--transport`, `--transport-type` `<str>`

HTTP transport protocol (http/https). Auto-detected from URL scheme if not specified. Explicit setting overrides auto-detection.
<br/>_Choices: [`http`]_

#### `--use-legacy-max-tokens`

Use 'max_tokens' field instead of 'max_completion_tokens'. Enable for compatibility with older OpenAI API versions.
<br/>_Flag (no value required)_

#### `--use-server-token-count`

Use server-reported token counts from response usage field. When true, trusts usage.prompt_tokens and usage.completion_tokens. When false, counts tokens locally using configured tokenizer.
<br/>_Flag (no value required)_

#### `--connection-reuse-strategy` `<str>`

HTTP connection management strategy. pooled: shared connection pool (fastest), never: new connection per request (includes TCP overhead), sticky_sessions: dedicated connection per session.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `pooled` | _default_ | Connections are pooled and reused across all requests |
| `never` |  | New connection for each request, closed after response |
| `sticky-user-sessions` |  | Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing) |

#### `--download-video-content`

For video generation endpoints, download the video content after generation completes. When enabled, request latency includes the video download time. When disabled, only generation time is measured.
<br/>_Flag (no value required)_

#### `--extra-inputs` `<str>`

Additional fields to include in request body. Merged into every request. Common fields: temperature, top_p, top_k, stop.

#### `-H`, `--header` `<str>`

Custom HTTP headers to include in all requests. Useful for authentication, tracing, or routing. Values support environment variable substitution.

### Input

#### `--input-file` `<str>`

Path to file or directory containing benchmark dataset. Can be absolute or relative. Supported formats depend on the format field: JSONL for single_turn/multi_turn, JSONL trace files for mooncake_trace, directories for random_pool.

#### `--public-dataset` `<str>`

Pre-configured public dataset to download and use for benchmarking. AIPerf automatically downloads and parses these datasets.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `sharegpt` |  | ShareGPT dataset from HuggingFace. Multi-turn conversational dataset with user/assistant exchanges. |

#### `--custom-dataset-type` `<str>`

Dataset file format determining parsing logic and expected file structure. single_turn: JSONL with single prompt-response exchanges. multi_turn: JSONL with conversation history. mooncake_trace: timestamped trace files for replay. random_pool: directory of reusable prompts.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `single_turn` | _default_ | Simple prompt-response pairs. |
| `multi_turn` |  | Conversational data with multiple turns. |
| `mooncake_trace` |  | Mooncake production trace format. |
| `random_pool` |  | Treat file as a pool for random sampling. |

#### `--dataset-sampling-strategy` `<str>`

Strategy for selecting entries from dataset during benchmarking. sequential: iterate in order, wrapping to start after end. random: randomly sample with replacement (entries may repeat). shuffle: random permutation without replacement, re-shuffling after exhaustion.
<br/>_Choices: [`random`, `sequential`, `shuffle`]_
<br/>_Default: `sequential`_

#### `--random-seed` `<int>`

Global random seed for reproducibility. Can be overridden per-dataset. If not set, uses system entropy.

### Fixed Schedule

#### `--fixed-schedule`, `--no-fixed-schedule`

Run requests according to timestamps specified in the input dataset. When enabled, AIPerf replays the exact timing pattern from the dataset. This mode is automatically enabled for mooncake_trace datasets.

#### `--fixed-schedule-auto-offset`

Normalize trace timestamps to start at 0. Subtracts minimum timestamp from all entries. Only used with type='fixed_schedule'.
<br/>_Flag (no value required)_
<br/>_Default: `True`_

#### `--fixed-schedule-start-offset` `<int>`

Filter out trace requests before this timestamp in ms (must be >= 0). Only used with type='fixed_schedule'.

#### `--fixed-schedule-end-offset` `<int>`

Filter out trace requests after this timestamp in ms (must be >= 0). Only used with type='fixed_schedule'.

### Goodput

#### `--goodput` `<str>`

SLO (Service Level Objectives) configuration as a generic dict. Maps metric names to threshold values. A request is counted as good only if it meets ALL specified thresholds.

### Output

#### `--output-artifact-dir`, `--artifact-dir` `<str>`

Output directory for all benchmark artifacts. Created if it doesn't exist.
<br/>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file` `<str>`

Filename prefix for all exported files. Example: 'my_run' produces 'my_run_summary.json', 'my_run_records.jsonl'.

#### `--export-level`, `--profile-export-level` `<str>`

Controls which output files are generated. summary: Only aggregate metrics files. records: Includes per-request metrics. raw: Includes raw request/response data.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `summary` | _default_ | Export only aggregated/summarized metrics (default, most compact) |
| `records` |  | Export per-record metrics after aggregation with display unit conversion |
| `raw` |  | Export raw parsed records with full request/response data (most detailed) |

#### `--slice-duration` `<float>`

Time slice duration in seconds for trend analysis (must be > 0). Divides benchmark into windows for per-window statistics. Supports: 30, '30s', '5m', '2h'.

### HTTP Trace

#### `--export-http-trace`

Export HTTP trace data for debugging.
<br/>_Flag (no value required)_

#### `--show-trace-timing`

Display HTTP trace timing metrics in console output. Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, and total duration.
<br/>_Flag (no value required)_

### Tokenizer

#### `--tokenizer` `<str>`

HuggingFace tokenizer identifier or local filesystem path. Should match the model's tokenizer for accurate token counts. Example: 'meta-llama/Llama-3.1-8B-Instruct'.

#### `--tokenizer-revision` `<str>`

Model revision to use: branch name, tag, or commit hash. Use for version pinning to ensure reproducibility.
<br/>_Default: `main`_

#### `--tokenizer-trust-remote-code`

Allow execution of custom tokenizer code from the repository. Required for some models but poses security risk. Only enable for trusted sources.
<br/>_Flag (no value required)_

### Load Generator

#### `--benchmark-duration` `<float>`

Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'.

#### `--benchmark-grace-period` `<float>`

Seconds to wait for in-flight requests at phase end (must be >= 0). Supports: 30, '30s', '2m'.

#### `--concurrency` `<int>`

Max concurrent in-flight requests (must be >= 1). For concurrency type: primary control. For rate types: acts as a cap.

#### `--prefill-concurrency` `<int>`

Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received.

#### `--request-rate` `<float>`

Target request rate in requests per second (must be > 0). Required for poisson/gamma/constant types. For user_centric: global rate shared across all users.

#### `--arrival-pattern`, `--request-rate-mode` `<str>`

Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma. constant: Generate requests at a fixed rate. poisson: Generate requests using a poisson distribution. gamma: Generate requests using a gamma distribution with tunable smoothness.
<br/>_Choices: [`concurrency_burst`, `constant`, `gamma`, `poisson`]_
<br/>_Default: `poisson`_

#### `--arrival-smoothness`, `--vllm-burstiness` `<float>`

Gamma distribution shape parameter (must be > 0). Only used with type='gamma'. 1.0 = Poisson, &lt;1 = bursty, >1 = regular.

#### `--request-count`, `--num-requests` `<int>`

Stop after this many requests sent (must be >= 1).

#### `--concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--prefill-concurrency-ramp-duration` `<float>`

Seconds to ramp from start to target value.

### Warmup

#### `--request-rate-ramp-duration` `<float>`

Seconds to ramp from start to target value.

#### `--warmup-request-count`, `--num-warmup-requests` `<int>`

Warmup phase: Stop after this many requests sent (must be >= 1). If not set, uses the --request-count value.

#### `--warmup-duration` `<float>`

Warmup phase: Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'. If not set, uses the --benchmark-duration value.

#### `--num-warmup-sessions` `<int>`

Warmup phase: Stop after this many sessions completed (must be >= 1). If not set, uses the --conversation-num value.

#### `--warmup-concurrency` `<int>`

Warmup phase: Max concurrent in-flight requests (must be >= 1). For concurrency type: primary control. For rate types: acts as a cap. If not set, uses the --concurrency value.

#### `--warmup-prefill-concurrency` `<int>`

Warmup phase: Max concurrent requests in prefill stage (must be >= 1). Limits requests before first token received. If not set, uses the --prefill-concurrency value.

#### `--warmup-request-rate` `<float>`

Warmup phase: Target request rate in requests per second (must be > 0). Required for poisson/gamma/constant types. For user_centric: global rate shared across all users. If not set, uses the --request-rate value.

#### `--warmup-arrival-pattern` `<str>`

Warmup phase: Load generation type. concurrency: concurrency-controlled immediate dispatch, poisson/gamma/constant: rate-controlled with arrival distribution, user_centric: N users sharing global rate, fixed_schedule: replay from timestamps. If not set, uses the --arrival-pattern value.

#### `--warmup-grace-period` `<float>`

Warmup phase: Seconds to wait for in-flight requests at phase end (must be >= 0). Supports: 30, '30s', '2m'. If not set, uses the --benchmark-grace-period value.

#### `--warmup-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --concurrency-ramp-duration value.

#### `--warmup-prefill-concurrency-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --prefill-concurrency-ramp-duration value.

#### `--warmup-request-rate-ramp-duration` `<float>`

Warmup phase: Seconds to ramp from start to target value. If not set, uses the --request-rate-ramp-duration value.

### User-Centric Rate

#### `--user-centric-rate` `<float>`

Enable user-centric rate limiting mode with the specified request rate (QPS). Each user has a gap = num_users / qps between turns. Designed for KV cache benchmarking with realistic multi-user patterns. Requires --num-users to be set.

#### `--num-users` `<int>`

Number of simulated concurrent users (must be >= 1). Required for user_centric type. Requests distributed across users to achieve global rate.

### Request Cancellation

#### `--request-cancellation-rate` `<float>`

Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally but aborted after --request-cancellation-delay seconds.

#### `--request-cancellation-delay` `<float>`

Seconds to wait after the request is fully sent before cancelling. A delay of 0 means send the full request, then immediately disconnect. Requires --request-cancellation-rate to be set.
<br/>_Default: `0.0`_

### Conversation Input

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time. Set to 0 for back-to-back turns.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

Standard deviation for turn delays in milliseconds.
<br/>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

Multiplier for scaling all turn delays. Values &lt; 1 speed up, > 1 slow down.
<br/>_Default: `1.0`_

#### `--conversation-turn-mean`, `--session-turns-mean` `<int>`

Mean number of request-response turns per conversation. Set to 1 for single-turn.
<br/>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

Standard deviation for number of turns per conversation.
<br/>_Default: `0`_

#### `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

Stop after this many sessions completed (must be >= 1).

#### `--num-dataset-entries`, `--num-prompts` `<int>`

Total number of unique entries to generate for the dataset. Each entry represents a unique prompt with sampled ISL/OSL. Entries are reused across conversations and turns according to the sampling strategy. Higher values provide more diversity.
<br/>_Default: `100`_

### Input Sequence Length (ISL)

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

Token block size for hash-based prompt caching in mooncake_trace datasets. When hash_ids are provided in trace entries, prompts are divided into blocks of this size. Each hash_id maps to a cached block, enabling simulation of KV-cache sharing patterns from production workloads. Total prompt length = (num_hash_ids - 1) * block_size + final_block_size.

#### `--seq-dist`, `--sequence-distribution` `<str>`

Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. Format: ISL,OSL:prob;ISL,OSL:prob (probabilities 0-100 summing to 100).

### Output Sequence Length (OSL)

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

### Prompt

#### `-b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

Number of text inputs to include in each request for batch processing endpoints. Supported by embeddings and rankings endpoint types where models can process multiple inputs simultaneously. Set to 1 for single-input requests. Not applicable to chat or completions endpoints.
<br/>_Default: `1`_

### Prefix Prompt

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

Number of distinct prefix prompts to generate for KV cache testing. Each prefix is prepended to user prompts, simulating cached context scenarios. Prefixes are randomly selected from pool per request. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

Token length for each prefix prompt in the pool. Only used when pool_size is set. Note: due to prefix and user prompts being concatenated, the final prompt token count may be off by one. Mutually exclusive with shared_system_length/user_context_length.
<br/>_Default: `0`_

#### `--shared-system-prompt-length` `<int>`

Length of shared system prompt in tokens. This prompt is identical across all sessions and appears as a system message. First part of a two-part prefix structure with high cache hit rate expected. Mutually exclusive with pool_size/length.

#### `--user-context-prompt-length` `<int>`

Length of per-session user context prompt in tokens. Each dataset entry gets a unique user context prompt. Second part of two-part prefix structure with lower cache hit rate expected. Mutually exclusive with pool_size/length.

### Rankings

#### `--rankings-passages-mean` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `10`_

#### `--rankings-passages-stddev` `<int>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0`_

#### `--rankings-passages-prompt-token-mean` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `128`_

#### `--rankings-passages-prompt-token-stddev` `<int>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0`_

#### `--rankings-query-prompt-token-mean` `<int>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `32`_

#### `--rankings-query-prompt-token-stddev` `<int>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0`_

### Synthesis

#### `--synthesis-speedup-ratio` `<float>`

Multiplier for timestamp scaling in synthesized traces. 1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-len-multiplier` `<float>`

Multiplier for core prefix branch lengths in the radix tree. 1.5 means prefix branches are 50%% longer.
<br/>_Default: `1.0`_

#### `--synthesis-prefix-root-multiplier` `<int>`

Number of independent radix trees to distribute traces across. Higher values increase prefix diversity.
<br/>_Default: `1`_

#### `--synthesis-prompt-len-multiplier` `<float>`

Multiplier for leaf path (unique prompt) lengths. 2.0 means prompts are 2x longer.
<br/>_Default: `1.0`_

#### `--synthesis-max-isl` `<int>`

Maximum input sequence length filter. Traces with input_length > max_isl are skipped entirely.

#### `--synthesis-max-osl` `<int>`

Maximum output sequence length cap. Traces with output_length > max_osl are capped to this value (not filtered).

### Audio Input

#### `--audio-length-mean` `<float>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `10.0`_

#### `--audio-length-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

#### `--audio-batch-size`, `--batch-size-audio` `<int>`

Number of audio inputs to include in each multimodal request. Supported with chat endpoint type for multimodal models. Set to 0 to disable audio inputs.
<br/>_Default: `0`_

#### `--audio-format` `<str>`

File format for generated audio files. wav: uncompressed PCM (larger files). mp3: compressed (smaller files). Format affects file size in multimodal requests but not audio characteristics.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `wav` | _default_ | WAV format. Uncompressed audio, larger file sizes, best quality. |
| `mp3` |  | MP3 format. Compressed audio, smaller file sizes, good quality. |

#### `--audio-depths` `<list>`

List of audio bit depths in bits to randomly select from. Each audio file is assigned a random depth from this list. Common values: 8 (low quality), 16 (CD quality), 24 (professional), 32 (high-end). Specify multiple values for mixed-quality testing.

#### `--audio-sample-rates` `<list>`

List of audio sample rates in kHz to randomly select from. Common values: 8.0 (telephony), 16.0 (speech), 44.1 (CD quality), 48.0 (professional). Specify multiple values for mixed-quality testing.

#### `--audio-num-channels` `<int>`

Number of audio channels. 1 = mono (single channel), 2 = stereo (left/right channels). Stereo doubles file size. Most speech models use mono.
<br/>_Default: `1`_

### Image Input

#### `--image-height-mean` `<float>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `512.0`_

#### `--image-height-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

#### `--image-width-mean` `<float>`

The mean (average) value of the distribution. For token counts, represents the target number of tokens.
<br/>_Default: `512.0`_

#### `--image-width-stddev` `<float>`

The standard deviation of the distribution. A value of 0 means deterministic (no variation). Higher values produce more spread in sampled values.
<br/>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image` `<int>`

Number of images to include in each multimodal request. Supported with chat endpoint type for vision-language models. Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.
<br/>_Default: `0`_

#### `--image-format` `<str>`

Image file format for generated images. png: lossless compression (larger files, best quality). jpeg: lossy compression (smaller files, good quality). random: randomly select between PNG and JPEG per image. Format affects file size in multimodal requests and encoding overhead.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `png` |  | PNG format. Lossless compression, larger file sizes, best quality. |
| `jpeg` | _default_ | JPEG format. Lossy compression, smaller file sizes, good for photos. |
| `random` |  | Randomly select PNG or JPEG for each image. |

### Video Input

#### `--video-batch-size`, `--batch-size-video` `<int>`

Number of video files to include in each multimodal request. Supported with chat endpoint type for video understanding models. Set to 0 to disable video inputs. Higher batch sizes significantly increase request payload size.
<br/>_Default: `0`_

#### `--video-duration` `<float>`

Duration in seconds for each generated video clip. Combined with fps, determines total frame count (frames = duration * fps). Longer durations increase file size and processing time. Typical values: 1-10 seconds for testing.
<br/>_Default: `1.0`_

#### `--video-fps` `<int>`

Frames per second for generated video. Higher FPS creates smoother video but increases frame count and file size. Common values: 4 (minimal, recommended for Cosmos models), 24 (cinematic), 30 (standard), 60 (high frame rate). Total frames = duration * fps.
<br/>_Default: `4`_

#### `--video-width` `<int>`

Video frame width in pixels. Determines video resolution and file size. Common values: 640 (SD), 1280 (HD), 1920 (Full HD). If not specified, uses codec/format defaults.

#### `--video-height` `<int>`

Video frame height in pixels. Combined with width determines aspect ratio and total pixel count per frame. Common values: 480 (SD), 720 (HD), 1080 (Full HD). If not specified, uses codec/format defaults.

#### `--video-synth-type` `<str>`

Algorithm for generating synthetic video content. Different types produce different visual patterns for testing. Content doesn't affect semantic meaning but may impact encoding efficiency and file size.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `moving_shapes` | _default_ | Generate videos with animated geometric shapes moving across the frame |
| `grid_clock` |  | Generate videos with a grid pattern and frame number overlay for frame-accurate verification |
| `noise` |  | Generate videos with random noise frames |

#### `--video-format` `<str>`

Container format for generated video files. webm: VP9 codec, BSD-licensed, recommended for open-source workflows. mp4: H.264/H.265, widely compatible. avi: legacy, larger files. mkv: Matroska, flexible container. Format affects compatibility, file size, and encoding options.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `mp4` |  | MP4 container. Widely compatible, good for H.264/H.265 codecs. |
| `webm` | _default_ | WebM container. Open format, optimized for web, good for VP9 codec. |

#### `--video-codec` `<str>`

Video codec for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL, widely compatible), libx265 (CPU, GPL, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br/>_Default: `libvpx-vp9`_

### Service

#### `--log-level` `<str>`

Global logging verbosity level. trace: most verbose, error: least verbose.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `TRACE` |  | Most verbose. Logs all operations including ZMQ messages and internal state changes. |
| `DEBUG` |  | Detailed debugging information. Logs function calls and important state transitions. |
| `INFO` | _default_ | General informational messages. Default level showing benchmark progress and results. |
| `NOTICE` |  | Important informational messages that are more significant than INFO but not warnings. |
| `WARNING` |  | Warning messages for potentially problematic situations that don't prevent execution. |
| `SUCCESS` |  | Success messages for completed operations and milestones. |
| `ERROR` |  | Error messages for failures that prevent specific operations but allow continued execution. |
| `CRITICAL` |  | Critical errors that may cause the benchmark to fail or produce invalid results. |

#### `-v`, `--verbose`

Equivalent to --log-level DEBUG. Enables detailed logging and switches UI to simple mode.
<br/>_Flag (no value required)_

#### `-vv`, `--extra-verbose`

Equivalent to --log-level TRACE. Most verbose logging including ZMQ messages. Switches UI to simple mode.
<br/>_Flag (no value required)_

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of parallel record processors. null = auto-detect based on CPU cores.

### Server Metrics

#### `--server-metrics` `<list>`

Server metrics collection (ENABLED BY DEFAULT). Optionally specify additional Prometheus endpoint URLs. Use --no-server-metrics to disable.

#### `--no-server-metrics`

Disable server metrics collection entirely.

#### `--server-metrics-formats` `<list>`

Export formats for scraped metrics. Options: json, csv, parquet, jsonl.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `json` |  | Export aggregated statistics in JSON hybrid format with metrics keyed by name. Best for: Programmatic access, CI/CD pipelines, automated analysis. |
| `csv` |  | Export aggregated statistics in CSV tabular format organized by metric type. Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames. |
| `jsonl` |  | Export raw time-series records in line-delimited JSON format. Best for: Time-series analysis, debugging, visualizing metric evolution. Warning: Can generate very large files for long-running benchmarks. |
| `parquet` |  | Export raw time-series data with delta calculations in Parquet columnar format. Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries. Includes cumulative deltas from reference point for counters and histograms. |

### GPU Telemetry

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry and optionally specify: 'dashboard' for realtime mode, custom DCGM URLs, or a metrics CSV file.

#### `--no-gpu-telemetry`

Disable GPU telemetry collection entirely.

### UI

#### `--ui-type`, `--ui` `<str>`

User interface mode. dashboard: rich interactive UI, simple: text progress, none: silent operation.
<br/>_Choices: [`dashboard`, `none`, `simple`]_
<br/>_Default: `dashboard`_

### Workers

#### `--workers-max`, `--max-workers` `<int>`

Maximum worker processes. null = auto-detect based on CPU cores.

### ZMQ Communication

#### `--zmq-host` `<str>`

Host address for internal ZMQ TCP communication between AIPerf services.

#### `--zmq-ipc-path` `<str>`

Directory path for ZMQ IPC socket files for local inter-process communication.

### Multi-Run

#### `--num-profile-runs` `<int>`

Number of profile runs to execute for confidence reporting. When 1, runs a single benchmark. When >1, computes aggregate statistics across runs.
<br/>_Default: `1`_

#### `--profile-run-cooldown-seconds` `<float>`

Cooldown duration in seconds between profile runs. Allows the system to stabilize between runs.
<br/>_Default: `0.0`_

#### `--confidence-level` `<float>`

Confidence level for computing confidence intervals (0-1). Common values: 0.90 (90%%), 0.95 (95%%), 0.99 (99%%).
<br/>_Default: `0.95`_

#### `--profile-run-disable-warmup-after-first`, `--no-profile-run-disable-warmup-after-first`

Disable warmup for runs after the first. When true, only the first run includes warmup for steady-state measurement.
<br/>_Default: `True`_

#### `--set-consistent-seed`, `--no-set-consistent-seed`

Automatically set random seed for consistent workloads across runs. When true, sets random_seed=42 if not specified, ensuring identical workloads for valid statistical comparison.
<br/>_Default: `True`_

### Accuracy

#### `--accuracy-benchmark` `<str>`

Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). When set, enables accuracy benchmarking mode.

#### `--accuracy-tasks` `<list>`

Specific tasks or subtasks within the benchmark to evaluate (e.g., specific MMLU subjects). If not set, all tasks are included.

#### `--accuracy-n-shots` `<int>`

Number of few-shot examples to include in the prompt. 0 means zero-shot evaluation. Maximum 8.
<br/>_Default: `0`_

#### `--accuracy-enable-cot`

Enable chain-of-thought prompting for accuracy evaluation.
<br/>_Flag (no value required)_

#### `--accuracy-grader` `<str>`

Override the default grader for the selected benchmark (e.g., exact_match, math, multiple_choice, code_execution).

#### `--accuracy-system-prompt` `<str>`

Custom system prompt to use for accuracy evaluation. Overrides any benchmark-specific system prompt.

#### `--accuracy-verbose`

Enable verbose output for accuracy evaluation, showing per-problem grading details.
<br/>_Flag (no value required)_

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace.

#### `--name` `<str>`

Human-readable name for the benchmark job (DNS label, max 40 chars).

#### `--image` `<str>` _(Required)_

AIPerf container image to use for Kubernetes deployment.

#### `--image-pull-policy` `<str>`

Image pull policy (Always, IfNotPresent, Never). Use 'Never' for minikube (or local clusters) with locally loaded images.

**Choices:**

| | | |
|-------|:-------:|-------------|
| `Always` |  | Every time the kubelet launches a container, it queries the registry to resolve the name to a digest. Uses cached image if digest matches, otherwise pulls the image. |
| `Never` |  | The kubelet does not try fetching the image. Startup fails if the image is not already present locally. |
| `IfNotPresent` |  | The image is pulled only if it is not already present locally. |

#### `--workers-max` `<int>`

Total number of workers. Automatically distributed across pods based on --workers-per-pod (default 10). E.g., --workers-max 50 = 5 pods × 10 workers.
<br/>_Default: `10`_

#### `--ttl-seconds` `<int>`

Seconds to keep pods after completion (None to disable TTL).
<br/>_Default: `300`_

### Kubernetes Node Placement

#### `--node-selector` `<str>`

Node selector labels (e.g., {'gpu': 'true'}).
<br/>_Default: `{}`_

#### `--tolerations` `<list>`

Pod tolerations for scheduling on tainted nodes.
<br/>_Default: `[]`_

### Kubernetes Scheduling

#### `--queue-name` `<str>`

Kueue LocalQueue name for gang-scheduling. When set, the JobSet is submitted to Kueue for quota-managed admission.

#### `--priority-class` `<str>`

Kueue WorkloadPriorityClass name for scheduling priority.

### Kubernetes Metadata

#### `--annotations` `<str>`

Additional pod annotations.
<br/>_Default: `{}`_

#### `--labels` `<str>`

Additional pod labels.
<br/>_Default: `{}`_

### Kubernetes Secrets

#### `--image-pull-secrets` `<list>`

Image pull secret names.
<br/>_Default: `[]`_

#### `--env-vars` `<str>`

Extra environment variables (key: value).
<br/>_Default: `{}`_

#### `--env-from-secrets` `<str>`

Environment variables from secrets (ENV_NAME: secret_name/key).
<br/>_Default: `{}`_

#### `--secret-mounts` `<list>`

Secret volume mounts.
<br/>_Default: `[]`_

#### `--service-account` `<str>`

Service account name for pods.

### Parameters

#### `-d`, `--detach`, `--no-detach`

Exit immediately after deploying (don't wait for completion). Automatically enabled in non-interactive environments (pipes, CI/CD).

#### `--no-wait`

Don't wait for pods to be ready before attaching (advanced).

#### `--attach-port` `<int>`

Local port for API port-forward (default: 0 = ephemeral).
<br/>_Default: `0`_

#### `--skip-endpoint-check`

Skip endpoint health validation before deploying.
<br/>_Flag (no value required)_

#### `--skip-preflight`

Skip automatic pre-flight checks before deploying. For comprehensive checks, run 'aiperf kube preflight' separately.
<br/>_Flag (no value required)_

<hr/>

## `aiperf kube results`

Retrieve results from an AIPerf benchmark.

By default, downloads all artifacts (per-record metrics, detailed exports, and all files in the /results directory) via the controller pod's API service. Falls back to kubectl cp if the API is unavailable.

Use --summary-only to download only the summary results instead of all artifacts. Use --from-pod to copy directly from the pod's /results directory using kubectl cp.

Use --shutdown to shut down the API service after downloading results, allowing the controller pod to exit cleanly.

If no job_id is specified, uses the last deployed benchmark.

**Examples:**

```bash
# Get results for last deployed job (downloads all artifacts)
aiperf kube results

# Get results for a specific job
aiperf kube results abc123

# Save results to a specific directory
aiperf kube results --output ./my-results

# Download only summary results
aiperf kube results --summary-only

# Get results directly from pod using kubectl cp
aiperf kube results --from-pod

# Download results and shut down the API service
aiperf kube results --shutdown
```

### Parameters

#### `--job-id` `<str>`

The AIPerf job ID to get results from (default: last deployed job).

#### `--output` `<str>`

Output directory for results (default: ./artifacts/{name}).

#### `--from-pod`, `--no-from-pod`

Copy results directly from pod using kubectl cp instead of API.

#### `-a`, `--all`, `--summary-only`

Download all artifacts via API (default: True). Use --summary-only to download only summary results.
<br/>_Flag (no value required)_
<br/>_Default: `True`_

#### `--shutdown`, `--no-shutdown`

Shut down the API service after downloading results. Only takes effect when results are retrieved via API (not --from-pod).

#### `--port` `<int>`

Local port for API port-forward (default: 0 = ephemeral).
<br/>_Default: `0`_

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace.

<hr/>

## `aiperf kube list`

List AIPerf benchmark jobs and their status.

Lists running and completed AIPerf JobSets with their status. By default searches all namespaces. Use --namespace to limit scope.

**Examples:**

```bash
# List all AIPerf jobs (all namespaces)
aiperf kube list

# List only running jobs
aiperf kube list --running

# List jobs in a specific namespace
aiperf kube list --namespace aiperf-bench

# Get status of a specific job
aiperf kube list abc123
```

### Parameters

#### `--job-id` `<str>`

Specific job ID to check (optional).

#### `-A`, `--all-namespaces`, `--no-all-namespaces`

Search in all namespaces (default: True).
<br/>_Default: `True`_

#### `--running`, `--no-running`

Show only running jobs.

#### `--completed`, `--no-completed`

Show only completed jobs.

#### `--failed`, `--no-failed`

Show only failed jobs.

#### `-w`, `--wide`, `--no-wide`

Show additional columns (job-id, custom-name, endpoint).

### Kubernetes

#### `--kubeconfig` `<str>`

Path to kubeconfig file (defaults to ~/.kube/config or KUBECONFIG env).

#### `--kube-context` `<str>`

Kubernetes context to use (defaults to current context in kubeconfig).

#### `--namespace` `<str>`

Kubernetes namespace.

<hr/>

## `aiperf kube validate`

Validate AIPerfJob YAML files against the CRD schema and AIPerfConfig model.

Performs comprehensive validation including: - YAML parsing and structure verification - Required fields: apiVersion, kind, metadata.name, spec (with AIPerfConfig fields) - Kubernetes resource name validation (RFC 1123) - AIPerfConfig validation via AIPerfJobSpecConverter - PodCustomization extraction validation - Worker count calculation (>= 1) - Unknown spec field detection (warning or error with --strict)

**Examples:**

```bash
aiperf kube validate aiperfjob.yaml
aiperf kube validate recipes/llama/*.yaml recipes/qwen/*.yaml
aiperf kube validate --strict aiperfjob.yaml
```

#### `--files`, `--empty-files` `<list>` _(Required)_

One or more AIPerfJob YAML file paths to validate.

#### `-s`, `--strict`, `--no-strict`

Fail on warnings such as unknown spec fields.

<hr/>

## `aiperf kube watch`

Watch live status of benchmark pods, events, and resource usage.

Monitors a benchmark deployment in real-time, showing pod status transitions, Kubernetes events, and detected problems. Press Ctrl+C to stop.

**Examples:**

```bash
aiperf kube watch -n my-benchmark
aiperf kube watch --job-id abc123
aiperf kube watch -A --interval 5
```

#### `-n`, `--namespace` `<str>`

Kubernetes namespace to watch.

#### `-j`, `--job-id` `<str>`

AIPerf job ID to watch.

#### `-i`, `--interval` `<int>`

Polling interval in seconds.
<br/>_Default: `10`_

#### `-t`, `--timeout` `<int>`

Maximum watch duration in seconds (0 for unlimited).
<br/>_Default: `0`_

#### `--kubeconfig` `<str>`

Path to kubeconfig file.

#### `--context` `<str>`

Kubernetes context name.

#### `-A`, `--all-namespaces`, `--no-all-namespaces`

Watch all AIPerf namespaces.
