# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real SGLang in-engine integration tests.

Runs the full aiperf CLI with sglang:// URLs to exercise the complete
endpoint -> transport -> engine pipeline on a real GPU.

Run with: uv run pytest tests/integration/in_engine/ -m integration -v --timeout=300
"""

from __future__ import annotations

import pytest

from tests.harness.utils import AIPerfCLI
from tests.integration.in_engine.conftest import InEngineWatchdog

MODEL = "Qwen/Qwen3-0.6B"
REQUEST_COUNT = 5
CONCURRENCY = 1
WORKERS_MAX = 1
UI = "simple"
# Keep memory footprint low for CI/dev GPUs
# SGLang needs more memory than vLLM due to different allocation strategy.
# disable_cuda_graph avoids CUDA graph capture (needs nvcc).
# context_length limits KV cache to reduce memory.
ENGINE_PARAMS = "mem_fraction_static:0.3 dtype:float16 trust_remote_code:True disable_cuda_graph:True attention_backend:triton sampling_backend:pytorch"


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
        f"--url sglang://{MODEL}",
        "--endpoint-type sglang_generate",
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
class TestSGLangInEngine:
    """End-to-end tests with a real SGLang engine via aiperf CLI."""

    async def test_basic_generate(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Basic SGLang in-engine generation via the full aiperf pipeline."""
        result = await cli.run(_base_cmd(), timeout=300.0)
        assert result.request_count == REQUEST_COUNT

    async def test_with_extra_inputs(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Extra inputs (temperature, top_p) flow through to SGLang sampling params."""
        result = await cli.run(
            _base_cmd("--extra-inputs temperature:0.5 --extra-inputs top_p:0.9"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_concurrent_requests(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """SGLang engine handles concurrent requests (concurrency > 1)."""
        result = await cli.run(
            _base_cmd(concurrency=4, request_count=20),
            timeout=300.0,
        )
        assert result.request_count == 20

    async def test_streaming_mode(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Streaming generation captures TTFT via SGLang async_generate(stream=True)."""
        result = await cli.run(
            _base_cmd("--streaming"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_pre_tokenized_dataset(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Pre-tokenized dataset bypasses chat template and sends input_ids directly."""
        result = await cli.run(
            _base_cmd("--pre-tokenized"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_warmup_iterations(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Warmup iterations run before profiling to stabilize latency measurements."""
        extra = (
            "--engine-params warmup_iterations:3 "
            "--engine-params warmup_input_tokens:64 "
            "--engine-params warmup_output_tokens:4"
        )
        result = await cli.run(
            _base_cmd(extra),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_engine_telemetry(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Engine telemetry polls stats at the configured interval during profiling."""
        extra = (
            "--engine-params telemetry:true --engine-params telemetry_interval_ms:100"
        )
        result = await cli.run(
            _base_cmd(extra),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_token_id_preservation(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """preserve_token_ids stores output token IDs in the response metadata."""
        result = await cli.run(
            _base_cmd("--engine-params preserve_token_ids:true"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_uniform_isl_distribution(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Uniform ISL distribution generates prompts with token counts in [min, max]."""
        extra = "--isl-distribution uniform --isl-min 50 --isl-max 200"
        result = await cli.run(
            _base_cmd(extra),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_world_size_metric(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """World size parameter is accepted and profiling completes with TP=1."""
        result = await cli.run(
            _base_cmd("--world-size 1"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_streaming_concurrent(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Streaming with concurrency > 1 exercises parallel TTFT capture."""
        result = await cli.run(
            _base_cmd("--streaming", concurrency=4, request_count=20),
            timeout=300.0,
        )
        assert result.request_count == 20
