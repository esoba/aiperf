<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Raw Endpoint Guide

Benchmark any HTTP API by sending pre-built request payloads verbatim, with no formatting or transformation.

## Overview

The `raw` endpoint type sends request payloads directly to the server without any modification. Unlike the `chat` or `completions` endpoints, which construct payloads from structured dataset fields (text, images, model name, etc.), the raw endpoint treats your input data as the complete API request body and passes it through to the transport layer unchanged.

**Use the raw endpoint when:**

- You have captured API traffic and want to replay it exactly
- You are benchmarking a non-OpenAI-compatible API
- You need full control over the request payload structure
- You want to replay `inputs.json` files exported from previous AIPerf runs

**Key characteristics:**

| Feature | Value |
|---------|-------|
| Endpoint type | `raw` |
| Default endpoint path | None (requires `--custom-endpoint`) |
| Streaming | Supported |
| Input tokenization | Disabled (server-side only) |
| Response parsing | Auto-detection with optional JMESPath extraction |
| Compatible dataset types | `raw_payload`, `inputs_json` (auto-detected) |

---

## How It Differs from Other Endpoints

Standard endpoints like `chat` build the API payload for you. Given a dataset turn with `{"text": "Hello"}`, the `chat` endpoint constructs:

```json
{
  "model": "my-model",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_completion_tokens": 256,
  "stream": true
}
```

The `raw` endpoint does none of this. It expects the dataset to contain the complete request payload. If a turn has a `raw_payload` field, that dict is sent directly to the transport. The `format_payload` method is intentionally disabled and raises `NotImplementedError` if called, which means dataset types that rely on endpoint formatting (like `single_turn` or `multi_turn`) are not compatible with the raw endpoint.

---

## Dataset Types

The raw endpoint works with two dataset types that produce `raw_payload` turns. Both are auto-detected from the file structure, so `--custom-dataset-type` is typically not required.

### raw_payload (JSONL)

Each line in a JSONL file is a complete API request body. The loader auto-detects files where each line contains a `messages` key.

**Single-file mode** -- each line becomes a single-turn conversation:

```jsonl
{"model": "my-model", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 100}
{"model": "my-model", "messages": [{"role": "user", "content": "Explain gravity."}], "max_tokens": 200}
{"model": "my-model", "messages": [{"role": "user", "content": "Write a haiku."}], "max_tokens": 50}
```

**Directory mode** -- each `.jsonl` file in the directory becomes a multi-turn conversation, with lines as sequential turns:

```
conversations/
  session_001.jsonl   # 3 lines = 3 turns in one conversation
  session_002.jsonl   # 2 lines = 2 turns in another conversation
```

### inputs_json (JSON)

The `inputs_json` loader reads AIPerf's `InputsFile` format -- a JSON file with a top-level `data` array containing sessions with pre-formatted payloads:

```json
{
  "data": [
    {
      "session_id": "sess_1",
      "payloads": [
        {"model": "my-model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100},
        {"model": "my-model", "messages": [{"role": "user", "content": "Follow up"}], "max_tokens": 100}
      ]
    },
    {
      "session_id": "sess_2",
      "payloads": [
        {"model": "my-model", "messages": [{"role": "user", "content": "Single request"}], "max_tokens": 50}
      ]
    }
  ]
}
```

Each session maps to a multi-turn conversation. Each payload in the `payloads` list becomes a turn with `raw_payload` set.

---

## Response Parsing

The raw endpoint parses responses using a two-step process:

1. **JMESPath extraction** (optional) -- if `response_field` is configured via `--extra-inputs`, a compiled JMESPath query extracts data from the JSON response
2. **Auto-detection fallback** -- if no JMESPath query is configured, or the query returns no match, the endpoint auto-detects the response type

### Auto-Detection

The auto-detection logic tries three response types in order (first match wins):

| Response Type | Detection Pattern (checked in priority order) |
|---------------|------------------------------------------------|
| **Embeddings** | `data[].embedding` (OpenAI format with `object: "embedding"`), `embeddings`, `embedding` |
| **Rankings** | `rankings`, `results` (any list) |
| **Text** | `text`, `content`, `response`, `output`, `result` (simple fields first, including list-of-strings), then `choices[0].text`, `choices[0].message.content`, `choices[0].delta.content` |

If the response is not valid JSON, plain text is returned as a text response.

### JMESPath Extraction

For APIs with non-standard response formats, use a JMESPath expression to extract the relevant field:

```bash
aiperf profile \
    --endpoint-type raw \
    --custom-endpoint /v1/my-api \
    --extra-inputs response_field:"data[0].text" \
    --input-file payloads.jsonl \
    --model my-model \
    --url localhost:8000
```

The JMESPath query `data[0].text` would extract `"extracted"` from a response like:

```json
{"data": [{"text": "extracted"}]}
```

If the JMESPath query fails to match, the endpoint falls back to auto-detection rather than returning an error.

---

## Usage Examples

### Basic Single-File Replay

```bash
cat > payloads.jsonl << 'EOF'
{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 100}
{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Explain neural networks."}], "max_tokens": 150}
{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "What are transformers?"}], "max_tokens": 200}
EOF

aiperf profile \
    --endpoint-type raw \
    --custom-endpoint /v1/chat/completions \
    --input-file payloads.jsonl \
    --model Qwen/Qwen3-0.6B \
    --streaming \
    --url localhost:8000 \
    --concurrency 2
```

Since the raw endpoint has no default endpoint path, you must provide `--custom-endpoint` to specify the API path.

### Streaming with Server Token Counts

Since the raw endpoint does not tokenize inputs client-side (`tokenizes_input: false`), you can use `--use-server-token-count` to rely on the server for token counting:

```bash
aiperf profile \
    --endpoint-type raw \
    --custom-endpoint /v1/chat/completions \
    --input-file payloads.jsonl \
    --model Qwen/Qwen3-0.6B \
    --streaming \
    --use-server-token-count \
    --url localhost:8000 \
    --concurrency 4
```

### Multi-Turn Directory Replay

```bash
mkdir -p conversations

cat > conversations/session_a.jsonl << 'EOF'
{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}
{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "How are you?"}], "max_tokens": 100}
EOF

cat > conversations/session_b.jsonl << 'EOF'
{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "What is AI?"}], "max_tokens": 200}
EOF

aiperf profile \
    --endpoint-type raw \
    --custom-endpoint /v1/chat/completions \
    --input-file conversations/ \
    --model Qwen/Qwen3-0.6B \
    --streaming \
    --url localhost:8000 \
    --concurrency 2
```

### Replaying an inputs.json File

If you have an `inputs.json` file from a previous AIPerf export or from GenAI-Perf:

```bash
aiperf profile \
    --endpoint-type raw \
    --custom-endpoint /v1/chat/completions \
    --input-file inputs.json \
    --model Qwen/Qwen3-0.6B \
    --url localhost:8000 \
    --concurrency 4
```

The `inputs_json` loader is auto-detected when the file contains a `data` array with `payloads` entries.

### Non-OpenAI API with JMESPath

For a custom API that returns responses in a non-standard format:

```bash
cat > custom_payloads.jsonl << 'EOF'
{"prompt": "Translate to French: Hello world", "parameters": {"max_new_tokens": 50}}
{"prompt": "Translate to French: Good morning", "parameters": {"max_new_tokens": 50}}
EOF

aiperf profile \
    --endpoint-type raw \
    --custom-endpoint /api/generate \
    --extra-inputs response_field:"generated_text" \
    --input-file custom_payloads.jsonl \
    --model my-model \
    --url localhost:8080 \
    --concurrency 2
```

---

## Tips

- **Always provide `--custom-endpoint`**: The raw endpoint has no default API path (`endpoint_path: null`). You must specify the target path explicitly.
- **Dataset type is auto-detected**: You do not need `--custom-dataset-type` in most cases. The loader examines the file structure (presence of `messages` key for raw_payload, or `data[].payloads` for inputs_json) and selects the correct loader automatically.
- **Payloads are sent as-is**: Nothing is added or removed from your payload. If you need `stream: true` in the request body, include it in your JSONL data. The `--streaming` flag controls how AIPerf reads the response (SSE parsing), but does not modify the payload.
- **Sampling strategy defaults to sequential**: Both `raw_payload` and `inputs_json` loaders default to sequential sampling, meaning conversations are replayed in the order they appear in the file.
- **Combine with `--extra-inputs`**: While the payload itself is sent verbatim, `--extra-inputs response_field:"..."` configures JMESPath response extraction on the endpoint side. Other `--extra-inputs` key-value pairs are available to the endpoint but do not modify raw payloads.
- **Input tokens are not counted client-side**: The raw endpoint sets `tokenizes_input: false`, so input sequence length metrics rely on server-reported values. Use `--use-server-token-count` for accurate token metrics.
