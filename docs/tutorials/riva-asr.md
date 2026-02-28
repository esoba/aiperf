<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile NVIDIA Riva ASR with AIPerf

AIPerf supports benchmarking [NVIDIA Riva](https://developer.nvidia.com/riva) Automatic Speech Recognition (ASR) services over gRPC. Two endpoint types are available: offline (batch) recognition and bidirectional streaming recognition.

## Endpoint Types

| Endpoint Type | gRPC Method | Streaming | Use Case |
|---|---|---|---|
| `riva_asr_offline` | `/nvidia.riva.asr.RivaSpeechRecognition/Recognize` | No | Batch transcription of complete audio |
| `riva_asr_streaming` | `/nvidia.riva.asr.RivaSpeechRecognition/StreamingRecognize` | Yes (bidi) | Real-time transcription with interim results |

Both endpoints accept audio input and return transcript text. Since Riva uses gRPC, the URL must use the `grpc://` or `grpcs://` scheme.

---

## Section 1. Start a Riva ASR Server

Deploy a Riva ASR server using the NVIDIA NGC container. Refer to the [Riva Quick Start Guide](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html) for full setup instructions.

Verify the server is ready:

```bash
# Check Riva health endpoint
curl -s http://localhost:8000/v1/health/ready
```

The default Riva gRPC port is `50051`.

---

## Section 2. Offline ASR (Batch Recognition)

Send complete audio files to Riva and receive the full transcript. Use `riva_asr_offline` when latency of individual chunks is not a concern.

### Profile with Synthetic Audio

```bash
aiperf profile \
    --model riva-asr \
    --url grpc://localhost:50051 \
    --endpoint-type riva_asr_offline \
    --audio-length-mean 5.0 \
    --audio-format wav \
    --audio-sample-rates 16 \
    --request-count 100 \
    --concurrency 10
```

### Profile with Custom Audio Files

Provide audio files via a JSONL input file:

```bash
cat <<EOF > asr_inputs.jsonl
{"audios": ["/path/to/audio1.wav"]}
{"audios": ["/path/to/audio2.wav"]}
{"audios": ["/path/to/audio3.wav"]}
EOF

aiperf profile \
    --model riva-asr \
    --url grpc://localhost:50051 \
    --endpoint-type riva_asr_offline \
    --input-file asr_inputs.jsonl \
    --custom-dataset-type single_turn \
    --request-count 3
```

AIPerf automatically loads audio files from disk and base64-encodes them for transport.

### Configuration Options

Pass ASR-specific options via `--extra-inputs`:

```bash
aiperf profile \
    --model riva-asr \
    --url grpc://localhost:50051 \
    --endpoint-type riva_asr_offline \
    --extra-inputs language_code:en-US \
    --extra-inputs sample_rate_hertz:16000 \
    --extra-inputs encoding:LINEAR_PCM \
    --audio-length-mean 5.0 \
    --audio-format wav \
    --audio-sample-rates 16 \
    --request-count 50 \
    --concurrency 8
```

| Option | Default | Description |
|---|---|---|
| `language_code` | `en-US` | BCP-47 language tag for the audio |
| `sample_rate_hertz` | `16000` | Sample rate of the input audio in Hz |
| `encoding` | `LINEAR_PCM` | Audio encoding format (see supported encodings below) |

**Supported audio encodings:** `LINEAR_PCM`, `FLAC`, `MULAW`, `OGGOPUS`, `ALAW`

---

## Section 3. Streaming ASR (Bidirectional Streaming)

Stream audio chunks to Riva and receive transcript results as they become available. Use `riva_asr_streaming` for real-time transcription benchmarks where time-to-first-result matters.

The streaming endpoint uses gRPC bidirectional streaming (`StreamingRecognize`). Audio is split into chunks and sent as a sequence of stream messages. Interim results are always enabled.

### Profile with Synthetic Audio

```bash
aiperf profile \
    --model riva-asr \
    --url grpc://localhost:50051 \
    --endpoint-type riva_asr_streaming \
    --streaming \
    --audio-length-mean 5.0 \
    --audio-format wav \
    --audio-sample-rates 16 \
    --request-count 100 \
    --concurrency 10
```

### Configuration Options

Streaming ASR accepts all the same options as offline, plus `chunk_size`:

```bash
aiperf profile \
    --model riva-asr \
    --url grpc://localhost:50051 \
    --endpoint-type riva_asr_streaming \
    --streaming \
    --extra-inputs language_code:en-US \
    --extra-inputs sample_rate_hertz:16000 \
    --extra-inputs encoding:LINEAR_PCM \
    --extra-inputs chunk_size:8000 \
    --audio-length-mean 10.0 \
    --audio-format wav \
    --audio-sample-rates 16 \
    --request-count 50 \
    --concurrency 8
```

| Option | Default | Description |
|---|---|---|
| `language_code` | `en-US` | BCP-47 language tag for the audio |
| `sample_rate_hertz` | `16000` | Sample rate of the input audio in Hz |
| `encoding` | `LINEAR_PCM` | Audio encoding format |
| `chunk_size` | `8000` | Audio chunk size in bytes (~0.5s at 16kHz 16-bit) |

Smaller chunk sizes produce more frequent interim results but increase gRPC overhead. The default of 8000 bytes corresponds to approximately 0.5 seconds of 16kHz 16-bit mono audio.

---

## Section 4. TLS-Encrypted Connections

Use the `grpcs://` scheme for TLS-encrypted gRPC channels:

```bash
aiperf profile \
    --model riva-asr \
    --url grpcs://riva-server:50051 \
    --endpoint-type riva_asr_offline \
    --audio-length-mean 5.0 \
    --audio-format wav \
    --audio-sample-rates 16 \
    --request-count 50
```

---

## Section 5. Scaling Concurrency

Test how Riva ASR handles increasing load:

```bash
# Low concurrency baseline
aiperf profile \
    --model riva-asr \
    --url grpc://localhost:50051 \
    --endpoint-type riva_asr_offline \
    --audio-length-mean 5.0 \
    --audio-format wav \
    --audio-sample-rates 16 \
    --request-count 200 \
    --concurrency 1

# High concurrency stress test
aiperf profile \
    --model riva-asr \
    --url grpc://localhost:50051 \
    --endpoint-type riva_asr_offline \
    --audio-length-mean 5.0 \
    --audio-format wav \
    --audio-sample-rates 16 \
    --request-count 200 \
    --concurrency 32
```

---

## Metrics

Since Riva ASR endpoints do not produce tokens (`produces_tokens: false`), AIPerf reports request-level metrics:

- **Request Latency** -- end-to-end time from request send to response received
- **Request Throughput** -- requests completed per second
- **Request Count** -- total successful requests

For streaming ASR, the streaming response chunks allow measurement of time-to-first-result latency.

---

## See Also

- [gRPC Transport Guide](grpc-transport.md) -- gRPC transport configuration and architecture
- [Riva TTS Tutorial](riva-tts.md) -- benchmarking Riva text-to-speech
- [Riva NLP Tutorial](riva-nlp.md) -- benchmarking Riva NLP services
- [Audio Tutorial](audio.md) -- audio input options for AIPerf
