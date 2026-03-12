<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Kubernetes CLI Options

Options for `aiperf kube deploy` and `aiperf kube generate`.

## Core Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--image` | `str` | AIPerf container image to use (required) | - |
| `--namespace` | `str` | Kubernetes namespace | `aiperf-{job_id}` (auto) |
| `--workers` | `int` | Number of worker pod replicas | `1` |
| `--ttl-seconds` | `int` | Seconds to keep pods after completion | `300` |

## Node Placement

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--node-selector` | `dict` | Node selector labels (e.g., `{'gpu': 'true'}`) | `{}` |
| `--tolerations` | `list` | Pod tolerations for scheduling on tainted nodes | `[]` |

## Secrets & Environment

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--image-pull-secrets` | `list` | Image pull secret names | `[]` |
| `--env-vars` | `dict` | Extra environment variables (`key: value`) | `{}` |
| `--env-from-secrets` | `dict` | Env vars from secrets (`ENV_NAME: secret_name/key`) | `{}` |
| `--secret-mounts` | `list` | Secret volume mounts | `[]` |
| `--service-account` | `str` | Service account name for pods | - |

## Metadata

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--annotations` | `dict` | Additional pod annotations | `{}` |
| `--labels` | `dict` | Additional pod labels | `{}` |

## Example

```bash
aiperf kube deploy llama-3.1-8b my-registry/aiperf:v1 \
  --url http://my-inference.svc:8000 \
  --namespace my-benchmarks \
  --workers 4 \
  --node-selector '{"nvidia.com/gpu": "A100"}' \
  --env-from-secrets 'API_KEY:my-secret/api-key' \
  --ttl-seconds 600
```
