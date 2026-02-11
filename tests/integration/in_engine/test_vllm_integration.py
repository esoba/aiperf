# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real vLLM in-engine integration tests.

Runs the full aiperf CLI with vllm:// URLs to exercise the complete
endpoint -> transport -> engine pipeline on a real GPU.

Run with: uv run pytest tests/integration/in_engine/ -m integration -v --timeout=300
"""

from __future__ import annotations

import pytest

from tests.harness.utils import AIPerfCLI

MODEL = "Qwen/Qwen3-0.6B"
REQUEST_COUNT = 5
CONCURRENCY = 1
WORKERS_MAX = 1
UI = "simple"
# Keep memory footprint low for CI/dev GPUs
ENGINE_PARAMS = "gpu_memory_utilization:0.3 max_model_len:2048 enforce_eager:True"


def _base_cmd(
    extra: str = "",
    *,
    concurrency: int = CONCURRENCY,
    request_count: int = REQUEST_COUNT,
) -> str:
    """Build the common aiperf profile command."""
    parts = [
        "aiperf profile",
        f"--model {MODEL}",
        f"--url vllm://{MODEL}",
        f"--request-count {request_count}",
        f"--concurrency {concurrency}",
        f"--workers-max {WORKERS_MAX}",
        "--output-tokens-mean 16",
        "--prompt-input-tokens-mean 128",
        f"--ui {UI}",
        f"--tokenizer {MODEL}",
    ]
    for ep in ENGINE_PARAMS.split():
        parts.append(f"--engine-params {ep}")
    if extra:
        parts.append(extra)
    return " ".join(parts)


@pytest.mark.integration
@pytest.mark.asyncio
class TestVLLMInEngine:
    """End-to-end tests with a real vLLM engine via aiperf CLI."""

    async def test_basic_generate(self, cli: AIPerfCLI) -> None:
        """Basic vLLM in-engine generation via the full aiperf pipeline."""
        result = await cli.run(_base_cmd(), timeout=300.0)
        assert result.request_count == REQUEST_COUNT

    async def test_with_extra_inputs(self, cli: AIPerfCLI) -> None:
        """Extra inputs (temperature, top_p) flow through to vLLM SamplingParams."""
        result = await cli.run(
            _base_cmd("--extra-inputs temperature:0.5 --extra-inputs top_p:0.9"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_with_engine_params(self, cli: AIPerfCLI) -> None:
        """Engine params (gpu_memory_utilization) are forwarded to vLLM LLM constructor."""
        result = await cli.run(
            _base_cmd("--engine-params gpu_memory_utilization:0.3"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_ngram_speculative_decoding(self, cli: AIPerfCLI) -> None:
        """NGRAM speculative decoding works end-to-end (no draft model needed)."""
        result = await cli.run(
            _base_cmd(
                "--engine-params speculative_method:ngram "
                "--engine-params num_speculative_tokens:3 "
                "--engine-params prompt_lookup_max:5",
                request_count=10,
            ),
            timeout=300.0,
        )
        assert result.request_count == 10

    async def test_concurrent_requests(self, cli: AIPerfCLI) -> None:
        """AsyncLLMEngine handles concurrent requests (concurrency > 1)."""
        result = await cli.run(
            _base_cmd(concurrency=4, request_count=20),
            timeout=300.0,
        )
        assert result.request_count == 20
