<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile NVIDIA Riva NLP with AIPerf

AIPerf supports benchmarking [NVIDIA Riva](https://developer.nvidia.com/riva) Natural Language Processing (NLP) services over gRPC. Seven endpoint types cover text classification, token classification, text transformation, punctuation, question answering, intent analysis, and entity recognition.

## Endpoint Types

| Endpoint Type | gRPC Method | Use Case |
|---|---|---|
| `riva_text_classify` | `/nvidia.riva.nlp.RivaLanguageUnderstanding/ClassifyText` | Sentiment analysis, topic classification |
| `riva_token_classify` | `/nvidia.riva.nlp.RivaLanguageUnderstanding/ClassifyTokens` | Per-token labeling (NER-like) |
| `riva_transform_text` | `/nvidia.riva.nlp.RivaLanguageUnderstanding/TransformText` | Text translation and transformation |
| `riva_punctuate_text` | `/nvidia.riva.nlp.RivaLanguageUnderstanding/PunctuateText` | Punctuation and capitalization |
| `riva_natural_query` | `/nvidia.riva.nlp.RivaLanguageUnderstanding/NaturalQuery` | Question answering over a context document |
| `riva_analyze_intent` | `/nvidia.riva.nlp.RivaLanguageUnderstanding/AnalyzeIntent` | Intent classification with slot filling |
| `riva_analyze_entities` | `/nvidia.riva.nlp.RivaLanguageUnderstanding/AnalyzeEntities` | Named entity recognition |

All NLP endpoints accept text input, return JSON responses, and operate over gRPC (no streaming). The URL must use the `grpc://` or `grpcs://` scheme.

---

## Section 1. Start a Riva NLP Server

Deploy a Riva NLP server using the NVIDIA NGC container. Refer to the [Riva Quick Start Guide](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html) for full setup instructions.

The default Riva gRPC port is `50051`.

---

## Section 2. Text Classification

Classify text into categories (e.g., sentiment, topic). Returns classification labels with confidence scores.

```bash
aiperf profile \
    --model riva-nlp \
    --url grpc://localhost:50051 \
    --endpoint-type riva_text_classify \
    --synthetic-input-tokens-mean 50 \
    --request-count 100 \
    --concurrency 10
```

### With Custom Input

```bash
cat <<EOF > classify_inputs.jsonl
{"texts": ["This product is amazing and works perfectly."]}
{"texts": ["The service was terrible and I want a refund."]}
{"texts": ["The weather today is partly cloudy."]}
EOF

aiperf profile \
    --model riva-nlp \
    --url grpc://localhost:50051 \
    --endpoint-type riva_text_classify \
    --input-file classify_inputs.jsonl \
    --custom-dataset-type single_turn \
    --request-count 3
```

---

## Section 3. Token Classification

Classify individual tokens within text (e.g., named entity recognition at the token level). Returns per-token labels.

```bash
aiperf profile \
    --model riva-nlp \
    --url grpc://localhost:50051 \
    --endpoint-type riva_token_classify \
    --synthetic-input-tokens-mean 50 \
    --request-count 100 \
    --concurrency 10
```

---

## Section 4. Text Transformation

Transform or translate text. Returns the transformed text output.

```bash
aiperf profile \
    --model riva-nlp \
    --url grpc://localhost:50051 \
    --endpoint-type riva_transform_text \
    --synthetic-input-tokens-mean 50 \
    --request-count 100 \
    --concurrency 10
```

---

## Section 5. Punctuation

Add punctuation and capitalization to unpunctuated text. Returns the punctuated text output.

```bash
aiperf profile \
    --model riva-nlp \
    --url grpc://localhost:50051 \
    --endpoint-type riva_punctuate_text \
    --synthetic-input-tokens-mean 50 \
    --request-count 100 \
    --concurrency 10
```

---

## Section 6. Natural Query (Question Answering)

Answer questions over a provided context document. Returns the top answer as text.

The `context` parameter is required -- it provides the document that Riva searches to find answers.

```bash
aiperf profile \
    --model riva-nlp \
    --url grpc://localhost:50051 \
    --endpoint-type riva_natural_query \
    --extra-inputs context:"NVIDIA Riva is a GPU-accelerated SDK for building speech AI applications. It includes automatic speech recognition, text-to-speech, and natural language processing services. Riva is optimized for real-time inference using NVIDIA GPUs." \
    --extra-inputs top_n:1 \
    --synthetic-input-tokens-mean 20 \
    --request-count 100 \
    --concurrency 10
```

### With Custom Questions

```bash
cat <<EOF > qa_inputs.jsonl
{"texts": ["What is NVIDIA Riva?"]}
{"texts": ["What services does Riva include?"]}
{"texts": ["What hardware is Riva optimized for?"]}
EOF

aiperf profile \
    --model riva-nlp \
    --url grpc://localhost:50051 \
    --endpoint-type riva_natural_query \
    --extra-inputs context:"NVIDIA Riva is a GPU-accelerated SDK for building speech AI applications. It includes automatic speech recognition, text-to-speech, and natural language processing services." \
    --input-file qa_inputs.jsonl \
    --custom-dataset-type single_turn \
    --request-count 3
```

| Option | Default | Description |
|---|---|---|
| `context` | `""` | The document text to search for answers |
| `top_n` | `1` | Number of answers to return |

---

## Section 7. Intent Analysis

Classify user intent and extract slots (entities) from text. Returns the detected intent and per-token slot labels.

```bash
aiperf profile \
    --model riva-nlp \
    --url grpc://localhost:50051 \
    --endpoint-type riva_analyze_intent \
    --synthetic-input-tokens-mean 20 \
    --request-count 100 \
    --concurrency 10
```

### With Domain Specification

```bash
aiperf profile \
    --model riva-nlp \
    --url grpc://localhost:50051 \
    --endpoint-type riva_analyze_intent \
    --extra-inputs domain:weather \
    --synthetic-input-tokens-mean 20 \
    --request-count 100 \
    --concurrency 10
```

| Option | Default | Description |
|---|---|---|
| `domain` | `""` (auto-detect) | Intent domain for slot filling |

---

## Section 8. Entity Analysis

Extract named entities from text. Returns entities with their labels (uses token classification response format).

```bash
aiperf profile \
    --model riva-nlp \
    --url grpc://localhost:50051 \
    --endpoint-type riva_analyze_entities \
    --synthetic-input-tokens-mean 50 \
    --request-count 100 \
    --concurrency 10
```

---

## Section 9. Common Configuration

### Language Code

All NLP endpoints that accept a text list also accept a `language_code` parameter:

```bash
aiperf profile \
    --model riva-nlp \
    --url grpc://localhost:50051 \
    --endpoint-type riva_text_classify \
    --extra-inputs language_code:en-US \
    --synthetic-input-tokens-mean 50 \
    --request-count 100
```

| Option | Default | Applies To |
|---|---|---|
| `language_code` | `en-US` | `riva_text_classify`, `riva_token_classify`, `riva_transform_text`, `riva_punctuate_text` |

### TLS-Encrypted Connections

Use `grpcs://` for TLS-encrypted gRPC channels:

```bash
aiperf profile \
    --model riva-nlp \
    --url grpcs://riva-server:50051 \
    --endpoint-type riva_text_classify \
    --synthetic-input-tokens-mean 50 \
    --request-count 50
```

---

## Metrics

All Riva NLP endpoints set `produces_tokens: false` and return JSON responses serialized as text. AIPerf reports request-level metrics:

- **Request Latency** -- end-to-end time from request send to response received
- **Request Throughput** -- requests completed per second
- **Request Count** -- total successful requests

---

## See Also

- [gRPC Transport Guide](grpc-transport.md) -- gRPC transport configuration and architecture
- [Riva ASR Tutorial](riva-asr.md) -- benchmarking Riva speech recognition
- [Riva TTS Tutorial](riva-tts.md) -- benchmarking Riva text-to-speech
