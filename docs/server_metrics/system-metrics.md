<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# System Metrics Collector

The system metrics collector provides host-level resource monitoring using `psutil` and optionally `pynvml`. It runs **alongside** Prometheus scraping, giving you both server-side inference metrics and system-wide resource utilization in a single benchmark run.

## Quick Reference

| Feature | Description |
|---------|-------------|
| **Enable** | `--server-metrics system` |
| **Combine with Prometheus** | `--server-metrics system` (auto-discovered endpoints still scraped) |
| **Combine with custom URLs** | `--server-metrics system http://node1:8081/metrics` |
| **Dependencies** | `psutil` (required, bundled), `pynvml` (optional, for GPU process memory) |
| **Source identifier** | `system://localhost` (appears as `endpoint_url` in exports) |

## When to Use

- **Local inference server** without a Prometheus `/metrics` endpoint
- **System-wide visibility** into CPU, memory, and per-process GPU VRAM alongside Prometheus metrics
- **Triton or custom servers** where you want process-level GPU memory tracking
- **Bare-metal debugging** to correlate system resource pressure with inference latency

## Quick Start

### System metrics + Prometheus (recommended)

Add `system` to enable host-level monitoring alongside the auto-discovered Prometheus endpoint:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --server-metrics system \
    --concurrency 4 \
    --request-count 100
```

AIPerf will:
1. Auto-discover and scrape the Prometheus endpoint at `localhost:8000/metrics` (default behavior)
2. Additionally collect system-wide CPU, memory, and per-process GPU VRAM
3. Merge both data sources into the same export files

If the Prometheus endpoint is unreachable (e.g., the server doesn't expose `/metrics`), the system collector still runs independently.

### System metrics + custom Prometheus endpoints

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --server-metrics system http://node1:9090/metrics http://node2:9090/metrics \
    --concurrency 4 \
    --request-count 100
```

## Collected Metrics

All system metrics are emitted as **gauge** type, meaning they represent point-in-time values. They appear in exports with `endpoint_url: "system://localhost"`.

### CPU

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `system_cpu_utilization_percent` | gauge | percent | System-wide CPU utilization across all cores |

**Interpretation:**
- `stats.avg` -- typical CPU load during the benchmark
- `stats.max` -- peak CPU utilization (100% = all cores fully utilized)
- Useful for detecting client-side bottlenecks when the benchmark driver itself is CPU-bound

### Memory

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `system_memory_used_bytes` | gauge | bytes | Physical memory currently in use |
| `system_memory_total_bytes` | gauge | bytes | Total physical memory installed |
| `system_memory_available_bytes` | gauge | bytes | Memory available for new allocations (includes cache/buffers) |

**Interpretation:**
- Memory pressure: `used / total` approaching 1.0 indicates swap risk
- `available` is more useful than `total - used` because it accounts for OS caches that can be reclaimed

### GPU Process Memory (requires pynvml)

| Metric | Type | Unit | Labels | Description |
|--------|------|------|--------|-------------|
| `gpu_process_memory_used_bytes` | gauge | bytes | `pid`, `process_name`, `gpu_index` | GPU VRAM allocated per process |

**Labels:**
- `pid` -- Process ID
- `process_name` -- Process name with Triton-aware identification (`tritonserver`, `triton_python_backend_stub`, or the process name from `/proc`)
- `gpu_index` -- GPU device index (`0`, `1`, ...)

**Interpretation:**
- Identifies which processes are consuming GPU memory
- Useful for multi-model or multi-tenant deployments
- If `pynvml` is not installed or no NVIDIA GPUs are present, this metric is simply omitted

## Output Format

System metrics flow through the same pipeline as Prometheus metrics. They appear in all export formats with `endpoint_url` set to `system://localhost`:

### JSON export

```json
{
  "metrics": {
    "system_cpu_utilization_percent": {
      "type": "gauge",
      "description": "System-wide CPU utilization percentage",
      "unit": "percent",
      "series": [{
        "endpoint_url": "system://localhost",
        "labels": null,
        "stats": {
          "avg": 42.3, "min": 12.1, "max": 89.7, "std": 18.5,
          "p50": 40.2, "p90": 72.4, "p95": 81.3, "p99": 88.1
        }
      }]
    },
    "gpu_process_memory_used_bytes": {
      "type": "gauge",
      "description": "GPU memory used per process (bytes)",
      "unit": "bytes",
      "series": [{
        "endpoint_url": "system://localhost",
        "labels": { "pid": "12345", "process_name": "tritonserver", "gpu_index": "0" },
        "stats": {
          "avg": 4294967296.0, "min": 4294967296.0, "max": 4831838208.0
        }
      }]
    }
  }
}
```

### Parquet export

System metrics rows look identical to Prometheus rows, just with a different `endpoint_url`:

```
endpoint_url       | metric_name                       | metric_type | value      | pid   | process_name  | gpu_index
-------------------|-----------------------------------|-------------|------------|-------|---------------|----------
system://localhost | system_cpu_utilization_percent    | gauge       | 42.3       | null  | null          | null
system://localhost | system_memory_used_bytes          | gauge       | 1.7e10     | null  | null          | null
system://localhost | gpu_process_memory_used_bytes     | gauge       | 4.3e9      | 12345 | tritonserver  | 0
```

### Querying mixed sources

When both system and Prometheus collectors are active, filter by `endpoint_url` to separate them:

```python
import json

with open("server_metrics_export.json") as f:
    data = json.load(f)

# System metrics
for name, metric in data["metrics"].items():
    for series in metric["series"]:
        if series["endpoint_url"] == "system://localhost":
            print(f"[system] {name}: avg={series['stats'].get('avg')}")

# Prometheus metrics
for name, metric in data["metrics"].items():
    for series in metric["series"]:
        if series["endpoint_url"].startswith("http"):
            print(f"[prometheus] {name}: avg={series['stats'].get('avg')}")
```

```sql
-- DuckDB: System vs Prometheus
SELECT endpoint_url, metric_name, AVG(value) as avg_value
FROM 'server_metrics_export.parquet'
WHERE metric_type = 'gauge'
GROUP BY endpoint_url, metric_name
ORDER BY endpoint_url, metric_name;
```

## Configuration

### CLI options

| Option | Description |
|--------|-------------|
| `--server-metrics system` | Enable system collector alongside auto-discovered Prometheus |
| `--server-metrics system URL [URL...]` | Enable system collector + additional Prometheus endpoints |
| `--no-server-metrics` | Disable **all** collection (both system and Prometheus) |

### Environment variables

The system collector uses the same environment variables as Prometheus collectors:

| Variable | Default | Description |
|----------|---------|-------------|
| `AIPERF_SERVER_METRICS_COLLECTION_INTERVAL` | 0.333s | Collection frequency |

### Remote server warning

When `system` is specified but the `--url` points to a remote host, AIPerf logs a warning:

```
Using system collector with non-localhost server URL(s): ['http://remote-server:8000'].
System metrics are collected from the local machine only.
If the inference server is running remotely, the system metrics will reflect the client machine, not the server.
```

This is expected and intentional when you want client-side resource visibility. If you need server-side metrics, use Prometheus scraping (which is always active alongside `system`).

## Architecture

The system collector is implemented as a plugin (`server_metrics_collector: system` in `plugins.yaml`) that satisfies the `ServerMetricsCollectorProtocol`. It extends `AIPerfLifecycleMixin` for lifecycle management and background task scheduling.

```
ServerMetricsManager
├── PrometheusCollector (endpoint_url: "http://localhost:8000/metrics")
├── PrometheusCollector (endpoint_url: "http://node1:9090/metrics")
└── SystemMetricsCollector (endpoint_url: "system://localhost")
    ├── psutil → CPU, memory gauges
    └── NvmlHandleManager → pynvml lifecycle (shared utility)
        └── pynvml → per-process GPU VRAM gauges (optional)
```

All collectors produce `ServerMetricsRecord` objects and share the same callback pipeline. The `ServerMetricsManager` manages their lifecycle (initialize, start, stop) uniformly regardless of type.

### Shared NVML management

Both `SystemMetricsCollector` (server metrics) and `PyNVMLTelemetryCollector` (GPU telemetry) use pynvml for GPU access but serve different pipelines with incompatible data models. They share `NvmlHandleManager` (`aiperf.common.nvml_handle_manager`) — a thread-safe utility that centralizes NVML initialization, shutdown, and device handle enumeration. Each collector creates its own `NvmlHandleManager` instance (they run in separate OS processes) but avoids duplicating the boilerplate.

## Troubleshooting

| Problem | Check | Solution |
|---------|-------|----------|
| No GPU memory metrics | `python -c "import pynvml"` | Install `pynvml` package: `uv add pynvml` |
| CPU always 0% | First collection sample | Expected -- psutil needs two samples to compute delta. First real value appears at the second collection interval. |
| Warning about remote URLs | CLI output | Informational only. System metrics are local; use Prometheus for remote servers. |
| `system://localhost` in exports | Export files | Normal -- this is the source identifier for system metrics. |

---

**Related documentation:**
- [Server Metrics Tutorial](server-metrics.md) -- Overview of all server metrics collection
- [Server Metrics Reference](server_metrics_reference.md) -- Metric definitions by backend
- [JSON Schema Reference](server_metrics_json_schema.md) -- JSON export format
- [Parquet Schema Reference](server_metrics_parquet_schema.md) -- Parquet export format
