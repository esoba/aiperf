<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Serve Multimodal Content via HTTP with the Content Server

By default, AIPerf embeds synthetic images and videos as base64 data URIs directly in API request payloads. This works, but produces multi-megabyte strings per image or video that inflate memory-mapped dataset files and network traffic.

The **content server** is an optional built-in HTTP file server that changes this: synthetic generators write files to disk and store lightweight HTTP URLs instead. The LLM inference server then fetches content on demand, the same way it would in production with real image/video URLs.

---

## Why Use the Content Server?

| Without Content Server | With Content Server |
|---|---|
| Each image/video is base64-encoded inline (~MBs per item) | Each image/video is a short URL string (~60 bytes) |
| Memory-mapped dataset files are large | Memory-mapped dataset files are small |
| Payloads sent over ZMQ are large | Payloads sent over ZMQ are small |
| LLM server receives base64 data | LLM server fetches files by URL (native `image_url` / `video_url` support) |

The content server also tracks every file request with detailed timing metrics (latency, time-to-first-byte, transfer duration), giving visibility into content delivery during benchmarks.

---

## Prerequisites

- An LLM inference server that supports `image_url` and/or `video_url` content types with HTTP URLs (e.g., vLLM, SGLang)
- The LLM server must be able to reach the content server over the network
- FFmpeg is required if using synthetic video (see [Synthetic Video](synthetic-video.md))

---

## Quick Start: Vision Benchmarking with Content Server

### 1. Start a vLLM Server

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2-VL-2B-Instruct
```

### 2. Run AIPerf with the Content Server Enabled

```bash
mkdir -p /tmp/aiperf-content

AIPERF_CONTENT_SERVER_ENABLED=true \
AIPERF_CONTENT_SERVER_CONTENT_DIR=/tmp/aiperf-content \
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --image-width-mean 512 \
    --image-height-mean 512 \
    --synthetic-input-tokens-mean 100 \
    --streaming \
    --url localhost:8000 \
    --request-count 20 \
    --concurrency 4
```

That's it. AIPerf will:
1. Start the content server on `http://0.0.0.0:8090`
2. Generate synthetic images and write them to `/tmp/aiperf-content/images/`
3. Store HTTP URLs (e.g., `http://0.0.0.0:8090/content/images/img_000001.png`) in the dataset instead of base64
4. Send requests to vLLM with `image_url` pointing to the content server
5. vLLM fetches each image from the content server as it processes the request

### Content Directory After Generation

```
/tmp/aiperf-content/
└── images/
    ├── img_000001.png
    ├── img_000002.png
    ├── img_000003.png
    └── ...
```

---

## Synthetic Video with Content Server

The content server is especially impactful for video, where base64 payloads can be tens of megabytes each.

```bash
AIPERF_CONTENT_SERVER_ENABLED=true \
AIPERF_CONTENT_SERVER_CONTENT_DIR=/tmp/aiperf-content \
aiperf profile \
    --model your-video-model \
    --endpoint-type chat \
    --video-width 640 \
    --video-height 480 \
    --video-fps 4 \
    --video-duration 5.0 \
    --synthetic-input-tokens-mean 50 \
    --streaming \
    --url localhost:8000 \
    --request-count 10
```

```
/tmp/aiperf-content/
└── video/
    ├── vid_000001.webm
    ├── vid_000002.webm
    └── ...
```

---

## Combined Multimodal: Image + Video + Text

You can enable images, video, and text simultaneously. Each modality that supports file writing will use the content server; audio stays inline as base64.

```bash
AIPERF_CONTENT_SERVER_ENABLED=true \
AIPERF_CONTENT_SERVER_CONTENT_DIR=/tmp/aiperf-content \
aiperf profile \
    --model your-multimodal-model \
    --endpoint-type chat \
    --image-width-mean 512 \
    --image-height-mean 512 \
    --video-width 640 \
    --video-height 480 \
    --video-fps 4 \
    --video-duration 5.0 \
    --audio-length-mean 3.0 \
    --synthetic-input-tokens-mean 100 \
    --streaming \
    --url localhost:8000 \
    --request-count 20
```

```
/tmp/aiperf-content/
├── images/
│   ├── img_000001.png
│   └── ...
└── video/
    ├── vid_000001.webm
    └── ...
```

> [!NOTE]
> **Audio is always inline.** The OpenAI `input_audio` API requires base64 data (no URL support), so audio content is never written to the content server regardless of configuration.

---

## Configuration Reference

The content server is configured entirely through environment variables:

| Environment Variable | Default | Description |
|---|---|---|
| `AIPERF_CONTENT_SERVER_ENABLED` | `false` | Enable the content server |
| `AIPERF_CONTENT_SERVER_HOST` | `0.0.0.0` | Host to bind the HTTP server to |
| `AIPERF_CONTENT_SERVER_PORT` | `8090` | Port for the HTTP server |
| `AIPERF_CONTENT_SERVER_CONTENT_DIR` | `""` (empty) | Directory for generated files. **Must be set to a non-empty path** for generators to write files and produce URLs instead of base64 |
| `AIPERF_CONTENT_SERVER_MAX_TRACKED_RECORDS` | `10000` | Maximum request records in the tracking buffer |

### When Does File Writing Activate?

File writing (URLs instead of base64) activates only when **both** conditions are met:
1. `AIPERF_CONTENT_SERVER_ENABLED=true`
2. `AIPERF_CONTENT_SERVER_CONTENT_DIR` is set to a non-empty path

If either condition is not met, generators fall back to the default base64 encoding behavior.

---

## How It Works

```
                          ┌─────────────────┐
                          │   AIPerf Host    │
                          │                  │
  ┌──────────────┐  write │ ┌──────────────┐ │  HTTP GET   ┌──────────────┐
  │   Generator  │───────►│ │ content_dir/ │◄├────────────│  LLM Server  │
  │ (image/video)│  files │ │  images/     │ │  fetch by   │  (vLLM, etc) │
  └──────────────┘        │ │  video/      │ │  URL        └──────────────┘
         │                │ └──────────────┘ │                    ▲
         │ URL            │                  │                    │
         ▼                │ ┌──────────────┐ │                    │
  ┌──────────────┐        │ │   Content    │ │                    │
  │    Dataset   │────────┼─│   Server     │─┘        API request │
  │  (mmap file) │  tiny  │ │  :8090       │     with image_url / │
  └──────────────┘  URLs  │ └──────────────┘     video_url fields │
         │                │                  │                    │
         ▼                │ ┌──────────────┐ │                    │
  ┌──────────────┐        │ │   Worker     │─┼────────────────────┘
  │    Worker    │────────┼─│  (endpoint)  │ │
  └──────────────┘        │ └──────────────┘ │
                          └─────────────────┘
```

1. **Dataset generation**: Image and video generators write files to `content_dir/` and store HTTP URLs in the dataset
2. **Content server**: Serves files over HTTP on the configured host and port
3. **Workers**: Read URLs from the mmap dataset and include them in API payloads as `image_url` or `video_url`
4. **LLM server**: Fetches content from the content server URLs when processing each request

---

## Network Considerations

The LLM inference server must be able to reach the content server's host and port. Common scenarios:

### Same Machine (Default)

When AIPerf and the LLM server run on the same machine, the default `0.0.0.0:8090` binding works out of the box.

### Docker Containers

If the LLM server is in a Docker container with `--network host`, it can reach the content server at `localhost:8090`. Otherwise, use the host machine's IP:

```bash
AIPERF_CONTENT_SERVER_HOST=0.0.0.0 \
AIPERF_CONTENT_SERVER_PORT=8090 \
AIPERF_CONTENT_SERVER_ENABLED=true \
AIPERF_CONTENT_SERVER_CONTENT_DIR=/tmp/aiperf-content \
aiperf profile ...
```

> [!IMPORTANT]
> The URLs stored in the dataset use the configured `HOST` and `PORT` values (e.g., `http://0.0.0.0:8090/content/...`). Ensure the LLM server can resolve and connect to this address. If the LLM server is on a different machine, set `HOST` to a routable IP address instead of `0.0.0.0`.

### Kubernetes

In Kubernetes, set `AIPERF_CONTENT_SERVER_HOST` to the pod's IP or service name so that the inference server pod can reach it.

---

## HTTP Endpoints

The content server exposes two endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/healthz` | GET | Health check (returns `200 OK`) |
| `/content/{file_path}` | GET | Serve a file from the content directory |

The `/content/` endpoint automatically detects MIME types and includes path traversal protection (attempts to escape the content directory return `403 Forbidden`).

---

## Scope and Limitations

| Supported | Not Supported |
|---|---|
| Synthetic images (PNG, JPEG) | Audio (requires inline base64 per OpenAI API) |
| Synthetic video (MP4, WebM) | Custom dataset media conversion (URLs in custom datasets already work as-is) |
| OpenAI-compatible `image_url` / `video_url` endpoints | Non-OpenAI endpoint formats |

---

## Troubleshooting

### LLM Server Can't Fetch Images

**Symptom**: Requests fail with connection errors or timeouts when the LLM server tries to fetch URLs.

**Check**:
1. Verify the content server is running: `curl http://localhost:8090/healthz`
2. Verify a file is accessible: `curl -I http://localhost:8090/content/images/img_000001.png`
3. Ensure the LLM server can reach the content server host/port (especially in Docker or Kubernetes)

### Content Directory Is Empty

**Symptom**: The content directory has no files after running a benchmark.

**Check**:
- Ensure both `AIPERF_CONTENT_SERVER_ENABLED=true` **and** `AIPERF_CONTENT_SERVER_CONTENT_DIR` is set to a non-empty path
- Verify image or video generation is actually enabled (non-zero `--image-width-mean`/`--image-height-mean` or `--video-width`/`--video-height`)

### Port Already In Use

**Symptom**: `Address already in use` error on startup.

**Fix**: Change the port: `AIPERF_CONTENT_SERVER_PORT=9090`
