<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dataset Loaders

AIPerf provides a unified dataset loader system for getting data into your benchmarks. Whether you have a custom JSONL file, want to use a public dataset, or need synthetic data generation, loaders handle it all.

## Overview

There are three data paths:

| Path | When to Use | CLI Flag |
|------|-------------|----------|
| **File-based** | You have custom data in JSONL files or a directory | `--input-file` |
| **Public dataset** | You want a well-known community dataset (e.g., ShareGPT) | `--public-dataset` |
| **Synthetic** | No dataset needed; generate prompts automatically (default) | *(none, or `--dataset-type`)* |

## How to Choose a Loader

| I want to... | Loader | CLI |
|--------------|--------|-----|
| Send custom text prompts | `single_turn` | `--input-file prompts.jsonl` |
| Send multi-turn conversations | `multi_turn` | `--input-file conversations.jsonl` |
| Sample from pools of queries/passages | `random_pool` | `--input-file data/ --dataset-type random_pool` |
| Replay a Mooncake trace | `mooncake_trace` | `--input-file trace.jsonl --dataset-type mooncake_trace` |
| Use the ShareGPT public dataset | `sharegpt` | `--public-dataset sharegpt` |
| Generate synthetic prompts | `synthetic_multimodal` | *(default)* |
| Benchmark a ranking endpoint | `synthetic_rankings` | *(auto-selected for ranking endpoints)* |

## Quick Reference

| Loader | Source | Multi-Modal | Multi-Turn | Timestamps | Auto-Detect | Default Sampling |
|--------|--------|:-----------:|:----------:|:----------:|:-----------:|:----------------:|
| `single_turn` | JSONL file | Yes | No | Yes | Yes | Sequential |
| `multi_turn` | JSONL file | Yes | Yes | Yes | Yes | Sequential |
| `random_pool` | JSONL file or directory | Yes | No | No | Explicit only | Shuffle |
| `mooncake_trace` | JSONL file | No | Yes | Yes | Yes | Sequential |
| `sharegpt` | Public dataset (JSON) | No | No | No | No | Sequential |
| `synthetic_multimodal` | Generated | Yes | Yes | No | N/A | Shuffle |
| `synthetic_rankings` | Generated | No | No | No | N/A | Shuffle |

## Three Load Paths

### Public Dataset

Download and benchmark with a well-known dataset:

```bash
aiperf --public-dataset sharegpt --endpoint-type chat --model my-model
```

### File Dataset

Load your own data from a JSONL file:

```bash
aiperf --input-file my_prompts.jsonl --endpoint-type chat --model my-model
```

### Synthetic (Default)

Generate synthetic data automatically:

```bash
aiperf --endpoint-type chat --model my-model --isl 128 --osl 64
```

## Auto-Detection

When you provide `--input-file` without `--dataset-type`, AIPerf auto-detects the format:

1. Read the first non-empty line of the JSONL file
2. If the line has a `type` field, use it directly (e.g., `{"type": "random_pool", ...}`)
3. Otherwise, try each registered loader's `can_load()` method
4. If exactly one loader matches, use it; if multiple match, raise an error

**When to use explicit `--dataset-type`:**

- `random_pool` format overlaps with `single_turn` (both have modality fields), so `random_pool` requires either an explicit `type` field in the data or `--dataset-type random_pool`
- `sharegpt` is not auto-detected from JSONL data (it uses JSON format via `--public-dataset`)
- Directories always resolve to `random_pool`

## File-Based Loaders

### single_turn

Single-turn dataset loader supporting multi-modal data and client-side batching. Each line is one independent request.

| Capability | Supported |
|------------|-----------|
| Multi-modal (text, image, audio, video) | Yes |
| Client-side batching | Yes |
| Multi-turn / sessions | No |
| Timestamps / delays | Yes |
| Named fields | Yes |

#### Schema

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"single_turn"` | Optional. Explicit type discriminator. |
| `text` | `string` | Simple text string content |
| `texts` | `string[]` or `Text[]` | List of text strings or Text objects |
| `image` | `string` | Simple image string content (URL, path, or base64) |
| `images` | `string[]` or `Image[]` | List of image strings or Image objects |
| `audio` | `string` | Simple audio string content |
| `audios` | `string[]` or `Audio[]` | List of audio strings or Audio objects |
| `video` | `string` | Simple video string content (URL, path, or base64) |
| `videos` | `string[]` or `Video[]` | List of video strings or Video objects |
| `timestamp` | `number` | Timestamp in milliseconds (mutually exclusive with `delay`) |
| `delay` | `number` | Delay in milliseconds before sending (mutually exclusive with `timestamp`) |
| `role` | `string` | Role of the turn |

At least one modality field must be provided. Singular and plural forms of the same modality are mutually exclusive (e.g., `text` and `texts` cannot both be set).

#### Examples

**Text only:**
```json
{"text": "What is deep learning?"}
```

**Multi-modal:**
```json
{"text": "What is in the image?", "image": "/path/to/image.png"}
```

**Batched multi-modal:**
```json
{"texts": ["Who are you?", "Hello world"], "images": ["/path/to/image.png", "/path/to/image2.png"]}
```

**Fixed schedule (timestamps):**
```json
{"timestamp": 0, "text": "What is deep learning?"}
{"timestamp": 1000, "text": "Who are you?"}
{"timestamp": 2000, "text": "What is AI?"}
```

**Delayed:**
```json
{"delay": 0, "text": "What is deep learning?"}
{"delay": 1234, "text": "Who are you?"}
```

**Named fields (multi-batch, multi-modal):**
```json
{
    "texts": [
        {"name": "text_field_A", "contents": ["Hello", "World"]},
        {"name": "text_field_B", "contents": ["Hi there"]}
    ],
    "images": [
        {"name": "image_field_A", "contents": ["/path/1.png", "/path/2.png"]},
        {"name": "image_field_B", "contents": ["/path/3.png"]}
    ]
}
```

**CLI:**
```bash
aiperf --input-file prompts.jsonl --endpoint-type chat --model my-model
```

**Default sampling strategy:** Sequential

See also: [Custom Prompt Benchmarking](tutorials/custom-prompt-benchmarking.md), [Fixed Schedule](tutorials/fixed-schedule.md), [Vision](tutorials/vision.md)

---

### multi_turn

Multi-turn dataset loader supporting conversation sessions with turn delays and multi-modal content. Each line is a complete conversation with multiple turns.

| Capability | Supported |
|------------|-----------|
| Multi-modal (text, image, audio, video) | Yes |
| Client-side batching | Yes |
| Multi-turn / sessions | Yes |
| Timestamps / delays | Yes |
| Named fields | Yes |

#### Schema

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"multi_turn"` | Optional. Explicit type discriminator. |
| `session_id` | `string` | Optional. Unique session identifier. Auto-generated if omitted. |
| `turns` | `SingleTurn[]` | **Required.** List of turns (each follows the SingleTurn schema above). |

Multiple entries with the same `session_id` are grouped into a single conversation.

#### Examples

**Simple multi-turn:**
```json
{
    "session_id": "session_123",
    "turns": [
        {"text": "Hello", "image": "url", "delay": 0},
        {"text": "Hi there", "delay": 1000}
    ]
}
```

**Batched:**
```json
{
    "session_id": "session_123",
    "turns": [
        {"texts": ["Who are you?", "Hello world"], "images": ["/path/1.png", "/path/2.png"]},
        {"texts": ["What is in the image?", "What is AI?"], "images": ["/path/3.png", "/path/4.png"]}
    ]
}
```

**Fixed schedule:**
```json
{
    "session_id": "session_123",
    "turns": [
        {"timestamp": 0, "text": "What is deep learning?"},
        {"timestamp": 1000, "text": "Who are you?"}
    ]
}
```

**Delayed:**
```json
{
    "session_id": "session_123",
    "turns": [
        {"delay": 0, "text": "What is deep learning?"},
        {"delay": 1000, "text": "Who are you?"}
    ]
}
```

**Full-featured (named fields, batched, multi-modal):**
```json
{
    "session_id": "session_123",
    "turns": [
        {
            "timestamp": 1234,
            "texts": [
                {"name": "text_field_a", "contents": ["hello", "world"]},
                {"name": "text_field_b", "contents": ["hi there"]}
            ],
            "images": [
                {"name": "image_field_a", "contents": ["/path/1.png", "/path/2.png"]},
                {"name": "image_field_b", "contents": ["/path/3.png"]}
            ]
        }
    ]
}
```

**CLI:**
```bash
aiperf --input-file conversations.jsonl --endpoint-type chat --model my-model
```

**Default sampling strategy:** Sequential

See also: [Multi-Turn Conversations](tutorials/multi-turn.md)

---

### random_pool

Random pool dataset loader that creates conversations by randomly sampling from predefined pools of data. Supports single files or directories of files, where each file becomes a separate pool.

| Capability | Supported |
|------------|-----------|
| Multi-modal (text, image, audio, video) | Yes |
| Client-side batching | Yes |
| Multi-turn / sessions | No |
| Timestamps / delays | No |
| Named fields | Yes |
| Directory input | Yes |

#### Schema

The `RandomPool` model shares the same modality fields as `SingleTurn` (text/texts, image/images, audio/audios, video/videos) but does not support `timestamp`, `delay`, or `role`.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"random_pool"` | Optional. Explicit type discriminator (recommended). |
| `text` | `string` | Simple text string content |
| `texts` | `string[]` or `Text[]` | List of text strings or Text objects |
| `image` | `string` | Simple image string content |
| `images` | `string[]` or `Image[]` | List of image strings or Image objects |
| `audio` | `string` | Simple audio string content |
| `audios` | `string[]` or `Audio[]` | List of audio strings or Audio objects |
| `video` | `string` | Simple video string content |
| `videos` | `string[]` or `Video[]` | List of video strings or Video objects |

!!! note
    Because `random_pool` and `single_turn` have overlapping schemas, auto-detection cannot distinguish them. Use an explicit `type` field in your data or pass `--dataset-type random_pool`.

#### Examples

**Single file:**
```json
{"text": "Who are you?", "image": "/path/to/image1.png"}
{"text": "Explain what is the meaning of life.", "image": "/path/to/image2.png"}
```

**Directory with named fields:**

`data/queries.jsonl`:
```json
{"texts": [{"name": "query", "contents": ["Who are you?"]}]}
{"texts": [{"name": "query", "contents": ["What is the meaning of life?"]}]}
```

`data/passages.jsonl`:
```json
{"texts": [{"name": "passage", "contents": ["I am a cat."]}]}
{"texts": [{"name": "passage", "contents": ["I am a dog."]}]}
```

Each file becomes a separate pool. The loader randomly samples from each pool and merges the results into conversations.

**CLI (file):**
```bash
aiperf --input-file pool.jsonl --dataset-type random_pool --endpoint-type chat --model my-model
```

**CLI (directory):**
```bash
aiperf --input-file data/ --endpoint-type chat --model my-model
```

**Default sampling strategy:** Shuffle

See also: [Rankings](tutorials/rankings.md)

---

### mooncake_trace

Mooncake trace dataset loader for replaying [Alibaba Mooncake](https://github.com/kvcache-ai/Mooncake) traces with timestamp-based scheduling. Designed for `fixed_schedule` timing mode.

| Capability | Supported |
|------------|-----------|
| Multi-modal | No (text only) |
| Client-side batching | No |
| Multi-turn / sessions | Yes |
| Timestamps / delays | Yes |
| Hash-based prefix caching | Yes |
| Synthesis transforms | Yes |

#### Schema

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"mooncake_trace"` | Optional. Explicit type discriminator. |
| `input_length` | `int` | Input sequence length. Required if `text_input` is not provided. |
| `text_input` | `string` | Actual text input. Required if `input_length` is not provided. |
| `output_length` | `int` | Optional. Output sequence length. |
| `hash_ids` | `int[]` | Optional. Hash IDs for reproducible prefix caching. Only with `input_length`. |
| `timestamp` | `number` | Optional. Timestamp in milliseconds for fixed schedule replay. |
| `delay` | `number` | Optional. Delay in milliseconds before sending the request. |
| `session_id` | `string` | Optional. Session ID for multi-turn grouping. |

Exactly one of `input_length` or `text_input` must be provided. `hash_ids` is only valid with `input_length`.

#### Examples

**Fixed schedule with hash IDs:**
```json
{"timestamp": 1000, "input_length": 300, "output_length": 40, "hash_ids": [123, 456]}
```

**Multi-turn:**
```json
{"session_id": "abc-123", "input_length": 300, "output_length": 40}
{"session_id": "abc-123", "delay": 2, "input_length": 150, "output_length": 20}
```

**Minimal:**
```json
{"input_length": 10, "hash_ids": [123]}
```

**With text input:**
```json
{"text_input": "Hello world", "output_length": 4}
```

**CLI:**
```bash
aiperf --input-file trace.jsonl --dataset-type mooncake_trace --timing-mode fixed_schedule --model my-model
```

**Default sampling strategy:** Sequential

See also: [Trace Replay](benchmark_modes/trace_replay.md)

---

### sharegpt

ShareGPT dataset loader for the [ShareGPT community dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered). Loads a JSON file containing multi-turn conversations and extracts prompt/completion pairs.

| Capability | Supported |
|------------|-----------|
| Multi-modal | No (text only) |
| Client-side batching | No |
| Multi-turn / sessions | No (uses first prompt/completion pair) |
| Timestamps / delays | No |

The ShareGPT loader:

- Requires at least 2 turns per conversation (prompt + completion)
- Filters by sequence length (min 4 tokens, max 1024 prompt, max 2048 total)
- Is not auto-detected from JSONL; use `--public-dataset sharegpt`

**CLI:**
```bash
aiperf --public-dataset sharegpt --endpoint-type chat --model my-model
```

**Default sampling strategy:** Sequential

## Synthetic Loaders

### synthetic_multimodal

The default loader when no input file or public dataset is specified. Generates multi-turn conversations with synthetic text, image, audio, and video payloads.

Key CLI parameters:

| Parameter | Description |
|-----------|-------------|
| `--isl <mean>` | Input sequence length (token count) |
| `--isl-stddev <stddev>` | Standard deviation for ISL |
| `--osl <mean>` | Output sequence length (token count) |
| `--osl-stddev <stddev>` | Standard deviation for OSL |
| `--num-dataset-entries <n>` | Number of conversations to generate |
| `--conversation-turn-mean <n>` | Average turns per conversation |
| `--image-width-mean <px>`, `--image-height-mean <px>` | Enable synthetic images (both must be > 0) |
| `--audio-length-mean <s>` | Enable synthetic audio (set length > 0 seconds) |

```bash
aiperf --endpoint-type chat --model my-model --isl 256 --osl 128 --num-dataset-entries 100
```

**Default sampling strategy:** Shuffle

### synthetic_rankings

Auto-selected when the endpoint type contains "rankings" (e.g., `nim_rankings`, `cohere_rankings`, `hf_tei_rankings`). Generates ranking tasks with one query and multiple passages.

Key CLI parameters:

| Parameter | Description |
|-----------|-------------|
| `--rankings-query-prompt-token-mean <n>` | Query token count |
| `--rankings-passages-mean <n>` | Number of passages per query |
| `--rankings-passages-prompt-token-mean <n>` | Passage token count |

```bash
aiperf --endpoint-type nim_rankings --model my-model
```

**Default sampling strategy:** Shuffle

## Common Features

### Model Selection

When benchmarking multiple models (`--model model-a model-b`), the `model_selection_strategy` controls how models are assigned to turns:

| Strategy | Behavior |
|----------|----------|
| `round_robin` (default) | Cycles through models in order |
| `random` | Randomly selects a model for each turn |
| `shuffle` | Shuffles models without replacement, reshuffles when exhausted |

```bash
aiperf --model model-a model-b --model-selection-strategy shuffle
```

### Output Token Sampling

When `--osl` is configured, each turn's `max_tokens` is sampled from a normal distribution with the configured mean and standard deviation (`--osl-stddev`). If `--osl` is not set, `max_tokens` is not added to requests (the server decides).

### Context Prompt Injection

Shared system prompts and per-session user context prompts can be injected into conversations:

| Parameter | Description |
|-----------|-------------|
| `--shared-system-prompt-length <n>` | Inject a shared system prompt (same for all conversations) |
| `--user-context-prompt-length <n>` | Inject a per-session user context prompt |

### Dataset Sampling Strategy

Controls how conversations are selected during the benchmark:

| Strategy | Behavior |
|----------|----------|
| `sequential` | Iterates through conversations in order, wrapping at the end |
| `shuffle` | Samples without replacement, then reshuffles |
| `random` | Samples with replacement (may repeat before seeing all) |

Each loader has a preferred default (see the [Quick Reference](#quick-reference) table). Override with `--dataset-sampling-strategy`.

## Troubleshooting

**"No loader can handle the data format"**

- Check that your JSONL file has valid JSON on each line
- Add `--dataset-type <type>` to specify the loader explicitly
- For `random_pool`, add a `"type": "random_pool"` field to your data or use `--dataset-type`

**"Multiple loaders can handle the data format"**

- Your data matches more than one loader. Add `--dataset-type <type>` to disambiguate

**"text and texts cannot be set together"**

- Use either the singular form (`text`) or the plural form (`texts`), not both in the same line

**"timestamp and delay cannot be set together"**

- Use either `timestamp` (absolute) or `delay` (relative), not both

**"Either 'input_length' or 'text_input' must be provided"**

- Mooncake trace entries need exactly one of `input_length` or `text_input`

**"Synthesis options are only supported with mooncake_trace datasets"**

- `--synthesis-speedup-ratio` and related flags only work with `--dataset-type mooncake_trace`
