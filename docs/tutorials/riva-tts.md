<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile NVIDIA Riva TTS with AIPerf

AIPerf supports benchmarking [NVIDIA Riva](https://developer.nvidia.com/riva) Text-to-Speech (TTS) services over gRPC. Two endpoint types are available: batch synthesis and server-streaming synthesis.

## Endpoint Types

| Endpoint Type | gRPC Method | Streaming | Use Case |
|---|---|---|---|
| `riva_tts` | `/nvidia.riva.tts.RivaSpeechSynthesis/Synthesize` | No | Batch synthesis, returns complete audio |
| `riva_tts_streaming` | `/nvidia.riva.tts.RivaSpeechSynthesis/SynthesizeOnline` | Yes (server) | Streaming synthesis, returns audio chunks |

Both endpoints accept text input and return synthesized audio. Since Riva uses gRPC, the URL must use the `grpc://` or `grpcs://` scheme.

---

## Section 1. Start a Riva TTS Server

Deploy a Riva TTS server using the NVIDIA NGC container. Refer to the [Riva Quick Start Guide](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html) for full setup instructions.

The default Riva gRPC port is `50051`.

---

## Section 2. Batch TTS (Synthesize)

Send text to Riva and receive the complete synthesized audio. Use `riva_tts` when you want to measure end-to-end synthesis latency for complete utterances.

### Profile with Synthetic Text

```bash
aiperf profile \
    --model riva-tts \
    --url grpc://localhost:50051 \
    --endpoint-type riva_tts \
    --synthetic-input-tokens-mean 50 \
    --synthetic-input-tokens-stddev 10 \
    --request-count 100 \
    --concurrency 10
```

### Profile with Custom Text Input

Provide text input via a JSONL file:

```bash
cat <<EOF > tts_inputs.jsonl
{"texts": ["The quick brown fox jumps over the lazy dog."]}
{"texts": ["NVIDIA Riva provides world-class speech AI capabilities."]}
{"texts": ["Benchmarking text-to-speech synthesis with AIPerf."]}
EOF

aiperf profile \
    --model riva-tts \
    --url grpc://localhost:50051 \
    --endpoint-type riva_tts \
    --input-file tts_inputs.jsonl \
    --custom-dataset-type single_turn \
    --request-count 3
```

### Configuration Options

Pass TTS-specific options via `--extra-inputs`:

```bash
aiperf profile \
    --model riva-tts \
    --url grpc://localhost:50051 \
    --endpoint-type riva_tts \
    --extra-inputs voice_name:English-US.Female-1 \
    --extra-inputs language_code:en-US \
    --extra-inputs encoding:LINEAR_PCM \
    --extra-inputs sample_rate_hz:22050 \
    --synthetic-input-tokens-mean 50 \
    --request-count 100 \
    --concurrency 8
```

| Option | Default | Description |
|---|---|---|
| `voice_name` | `""` (server default) | Voice identifier string |
| `language_code` | `en-US` | BCP-47 language tag |
| `encoding` | `LINEAR_PCM` | Audio output encoding format |
| `sample_rate_hz` | `22050` | Output audio sample rate in Hz |

**Supported audio encodings:** `LINEAR_PCM`, `FLAC`, `MULAW`, `OGGOPUS`, `ALAW`

---

## Section 3. Streaming TTS (SynthesizeOnline)

Stream synthesized audio chunks as they are generated. Use `riva_tts_streaming` to measure time-to-first-audio-chunk and streaming throughput.

The streaming endpoint uses gRPC server-side streaming (`SynthesizeOnline`). Riva begins returning audio data before the entire utterance is synthesized.

```bash
aiperf profile \
    --model riva-tts \
    --url grpc://localhost:50051 \
    --endpoint-type riva_tts_streaming \
    --streaming \
    --synthetic-input-tokens-mean 100 \
    --request-count 100 \
    --concurrency 10
```

### With Voice Selection

```bash
aiperf profile \
    --model riva-tts \
    --url grpc://localhost:50051 \
    --endpoint-type riva_tts_streaming \
    --streaming \
    --extra-inputs voice_name:English-US.Male-1 \
    --extra-inputs sample_rate_hz:44100 \
    --synthetic-input-tokens-mean 100 \
    --request-count 50 \
    --concurrency 8
```

Streaming TTS accepts the same `--extra-inputs` options as batch TTS (see configuration table above).

---

## Section 4. TLS-Encrypted Connections

Use the `grpcs://` scheme for TLS-encrypted gRPC channels:

```bash
aiperf profile \
    --model riva-tts \
    --url grpcs://riva-server:50051 \
    --endpoint-type riva_tts \
    --synthetic-input-tokens-mean 50 \
    --request-count 50
```

---

## Section 5. Scaling Concurrency

Test how Riva TTS handles increasing load:

```bash
# Low concurrency baseline
aiperf profile \
    --model riva-tts \
    --url grpc://localhost:50051 \
    --endpoint-type riva_tts \
    --synthetic-input-tokens-mean 50 \
    --request-count 200 \
    --concurrency 1

# High concurrency stress test
aiperf profile \
    --model riva-tts \
    --url grpc://localhost:50051 \
    --endpoint-type riva_tts \
    --synthetic-input-tokens-mean 50 \
    --request-count 200 \
    --concurrency 32
```

---

## Metrics

Riva TTS endpoints produce `AudioResponseData` with:

- **audio_bytes** -- raw synthesized audio
- **sample_rate_hz** -- output sample rate
- **encoding** -- audio encoding format
- **duration_ms** -- calculated audio duration (LINEAR_PCM only)

Both TTS endpoints set `produces_tokens: false`, `produces_audio: true`, and `tokenizes_input: true`. This means input token counts are tracked but output token metrics are not available. AIPerf reports request-level metrics:

- **Request Latency** -- end-to-end time from request send to response received
- **Request Throughput** -- requests completed per second
- **Request Count** -- total successful requests

For streaming TTS, each audio chunk is parsed into its own `AudioResponseData`, enabling measurement of time-to-first-audio-chunk latency.

---

## See Also

- [gRPC Transport Guide](grpc-transport.md) -- gRPC transport configuration and architecture
- [Riva ASR Tutorial](riva-asr.md) -- benchmarking Riva speech recognition
- [Riva NLP Tutorial](riva-nlp.md) -- benchmarking Riva NLP services
