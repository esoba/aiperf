<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Configuration v2.0 Examples

This directory contains example YAML configuration files for common benchmarking scenarios using the AIPerf config v2.0 system.

## Quick Start

```bash
# Run any example with:
aiperf --config examples_v2/basic_throughput.yaml
```

## Example Categories

### Basic Benchmarks

| Example | Description |
|---------|-------------|
| [basic_throughput.yaml](basic_throughput.yaml) | Maximum throughput test with concurrency mode |
| [latency_test.yaml](latency_test.yaml) | Latency measurement at controlled QPS |
| [warmup_profiling.yaml](warmup_profiling.yaml) | Two-phase benchmark with warmup |

### Load Testing

| Example | Description |
|---------|-------------|
| [goodput_slo.yaml](goodput_slo.yaml) | Measure goodput with SLO thresholds |
| [multi_url_load_balancing.yaml](multi_url_load_balancing.yaml) | Distribute load across multiple servers |

### Conversation & Multi-Turn

| Example | Description |
|---------|-------------|
| [multi_turn_conversation.yaml](multi_turn_conversation.yaml) | Multi-turn chatbot benchmarking |
| [public_dataset.yaml](public_dataset.yaml) | Use ShareGPT public dataset |

### Statistical & Quality

| Example | Description |
|---------|-------------|
| [multi_run.yaml](multi_run.yaml) | Multi-run benchmarking with confidence intervals |
| [accuracy.yaml](accuracy.yaml) | Accuracy evaluation (MMLU, AIME, etc.) |

### Advanced Features

| Example | Description |
|---------|-------------|
| [kv_cache_test.yaml](kv_cache_test.yaml) | KV cache / prefix caching testing |
| [long_context.yaml](long_context.yaml) | Long context (32K+) benchmarking |
| [trace_replay.yaml](trace_replay.yaml) | Replay production traces |
| [request_cancellation.yaml](request_cancellation.yaml) | Test request cancellation handling |

### Multimodal

| Example | Description |
|---------|-------------|
| [multimodal_vision.yaml](multimodal_vision.yaml) | Vision-language model benchmarking |
| [audio_multimodal.yaml](audio_multimodal.yaml) | Audio/speech model benchmarking |

### Specialized Endpoints

| Example | Description |
|---------|-------------|
| [embeddings.yaml](embeddings.yaml) | Embedding model benchmarking |
| [composed_dataset.yaml](composed_dataset.yaml) | File + synthetic augmentation |

## Configuration Structure

All config v2.0 files follow this structure:

```yaml
# Required sections
models: [...]        # Model(s) to benchmark
endpoint: {...}      # Server connection settings
datasets: {...}      # Named data sources
load: {...}          # Benchmark load configuration

# Optional sections
artifacts: {...}       # Export and console settings
slos: {...}            # SLO thresholds
tokenizer: {...}       # Token counting settings
gpu_telemetry: {...}   # GPU metrics collection
server_metrics: {...}  # Server metrics
runtime: {...}         # Worker settings
logging: {...}         # Logging/debug
multi_run: {...}       # Multi-run statistical benchmarking
accuracy: {...}        # Accuracy evaluation
random_seed: N         # Reproducibility
```

## Key Concepts

### Datasets

Datasets can be:
- **Synthetic**: Generated prompts with ISL/OSL distributions
- **File**: Load from JSONL files
- **Public**: Use datasets like ShareGPT
- **Composed**: File + synthetic augmentation

### Phases

Phases support:
- **Stop conditions**: `requests`, `duration`, or `sessions`
- **Load control**: `concurrency`, `rate`, `prefill_concurrency`
- **Arrival patterns**: `concurrency`, `poisson`, `gamma`, `constant`
- **Transitions**: `seamless` for smooth phase changes
- **Ramping**: `concurrency_ramp`, `rate_ramp`

### Timing Modes

- **Concurrency-based**: Send as fast as possible up to concurrency limit
- **Rate-based**: Send at target QPS with arrival pattern
- **User-centric**: Fixed per-user rate for KV cache testing
- **Fixed-schedule**: Replay exact timestamps from traces

## Customizing Examples

1. Copy an example that matches your use case
2. Update `endpoint.urls` with your server URL(s)
3. Update `models` with your model name
4. Adjust `load` for your test parameters
5. Configure `output.dir` for results location

## Environment Variables

Use `${VAR}` or `${VAR:default}` syntax for sensitive values:

```yaml
endpoint:
  api_key: ${OPENAI_API_KEY}
  urls:
    - ${SERVER_URL:http://localhost:8000/v1/chat/completions}
```
