# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config template generator for Kubernetes deployments.

Provides a starter configuration template that users can customize
for their benchmark setup.
"""

KUBE_INIT_TEMPLATE = """\
# AIPerf Kubernetes Benchmark Configuration
#
# Usage:
#   aiperf kube profile --user-config {filename} --image <your-image>
#
# This file configures the benchmark workload. Kubernetes-specific options
# (image, workers, namespace, etc.) are passed as CLI flags.

# === Endpoint Configuration ===
# The LLM inference server to benchmark.
endpoint:
  # Model name(s) served by the endpoint
  model_names:
    - "your-model-name"

  # Endpoint URL(s) - supports multiple for load balancing
  urls:
    - "http://your-server:8000/v1"

# === Load Generator Configuration ===
# Controls how requests are sent to the endpoint.
# loadgen:
#   # Number of requests to send (use for fixed-count benchmarks)
#   request_count: 1000
#
#   # Duration in seconds (use for time-based benchmarks)
#   # benchmark_duration: 60
#
#   # Requests per second (0 = max throughput)
#   # request_rate: 0

# === Input Configuration ===
# Controls the prompt/token sizes for requests.
# input:
#   prompt:
#     input_tokens:
#       mean: 128
#     output_tokens:
#       mean: 128

# === Kubernetes CLI Flags (not set in this file) ===
# These are passed on the command line:
#
#   --image <image>           Container image (required)
#   --workers-max <n>         Total workers (default: 10)
#   --namespace <ns>          Kubernetes namespace (auto-generated if omitted)
#   --name <name>             Human-readable job name
#   --node-selector k=v       Node placement labels
#   --tolerations k=v:effect  Pod tolerations
#   --skip-preflight          Skip pre-flight checks
#   --detach                  Deploy and exit immediately
"""


def generate_init_template(filename: str) -> str:
    """Generate a config template with the given filename in usage instructions.

    Args:
        filename: The filename to substitute into usage instructions.

    Returns:
        The template string with filename substituted.
    """
    return KUBE_INIT_TEMPLATE.format(filename=filename)
