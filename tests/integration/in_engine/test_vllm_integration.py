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
from tests.integration.in_engine.conftest import InEngineWatchdog

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
        "--endpoint-type vllm_generate",
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

    async def test_basic_generate(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Basic vLLM in-engine generation via the full aiperf pipeline."""
        result = await cli.run(_base_cmd(), timeout=300.0)
        assert result.request_count == REQUEST_COUNT

    async def test_with_extra_inputs(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Extra inputs (temperature, top_p) flow through to vLLM SamplingParams."""
        result = await cli.run(
            _base_cmd("--extra-inputs temperature:0.5 --extra-inputs top_p:0.9"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_with_engine_params(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Engine params (gpu_memory_utilization) are forwarded to vLLM LLM constructor."""
        result = await cli.run(
            _base_cmd("--engine-params gpu_memory_utilization:0.3"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_ngram_speculative_decoding(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
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

    async def test_concurrent_requests(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """AsyncLLMEngine handles concurrent requests (concurrency > 1)."""
        result = await cli.run(
            _base_cmd(concurrency=4, request_count=20),
            timeout=300.0,
        )
        assert result.request_count == 20

    async def test_streaming_mode(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Streaming generation uses DELTA output mode and captures TTFT."""
        result = await cli.run(
            _base_cmd("--streaming"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT
        assert result.has_streaming_metrics

    async def test_pre_tokenized_dataset(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Pre-tokenized mode sends token IDs directly, bypassing chat template."""
        result = await cli.run(
            _base_cmd("--pre-tokenized"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_warmup_iterations(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Warmup iterations run before measurement to stabilize CUDA graphs and KV cache."""
        result = await cli.run(
            _base_cmd(
                "--engine-params warmup_iterations:3 "
                "--engine-params warmup_input_tokens:64 "
                "--engine-params warmup_output_tokens:4"
            ),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_engine_telemetry(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Engine telemetry polls engine stats at configured interval during benchmark."""
        result = await cli.run(
            _base_cmd(
                "--engine-params telemetry:true "
                "--engine-params telemetry_interval_ms:100"
            ),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_token_id_preservation(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """preserve_token_ids stores output token IDs in the response record."""
        result = await cli.run(
            _base_cmd("--engine-params preserve_token_ids:true"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_uniform_isl_distribution(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Uniform ISL distribution generates prompts with lengths between min and max."""
        result = await cli.run(
            _base_cmd("--isl-distribution uniform --isl-min 50 --isl-max 200"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_world_size_metric(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """world_size parameter is passed through and benchmark completes with it set."""
        result = await cli.run(
            _base_cmd("--world-size 1"),
            timeout=300.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_streaming_concurrent(
        self, cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Streaming with concurrency > 1 handles multiple DELTA streams concurrently."""
        result = await cli.run(
            _base_cmd("--streaming", concurrency=4, request_count=20),
            timeout=300.0,
        )
        assert result.request_count == 20
        assert result.has_streaming_metrics
