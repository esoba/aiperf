# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real TRT-LLM in-engine integration tests.

Runs the full aiperf CLI with trtllm:// URLs to exercise the complete
endpoint -> transport -> engine pipeline on a real GPU.

IMPORTANT: TRT-LLM requires a specific container environment that is
incompatible with the host venv. These tests use the `trtllm_cli` fixture
which runs aiperf inside a Docker container automatically.

Run with: uv run pytest tests/integration/in_engine/test_trtllm_integration.py -m integration -v -s --timeout=600
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
# TRT-LLM specific engine params: dtype, max_batch_size, max_seq_len
ENGINE_PARAMS = "dtype:float16 max_batch_size:4 max_seq_len:512"


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
        f"--url trtllm://{MODEL}",
        "--endpoint-type trtllm_generate",
        f"--request-count {request_count}",
        f"--concurrency {concurrency}",
        f"--workers-max {WORKERS_MAX}",
        "--output-tokens-mean 16",
        "--prompt-input-tokens-mean 128",
        f"--ui {UI}",
        f"--tokenizer {MODEL}",
        "--no-server-metrics",  # No Prometheus for in-engine; prevents IPC socket cleanup bug
        "--no-gpu-telemetry",  # No DCGM in Docker; prevents total_energy_joules export error
    ]
    for ep in ENGINE_PARAMS.split():
        parts.append(f"--engine-params {ep}")
    if extra:
        parts.append(extra)
    return " ".join(parts)


@pytest.mark.integration
@pytest.mark.asyncio
class TestTRTLLMInEngine:
    """End-to-end tests with a real TRT-LLM engine via aiperf CLI.

    Uses `trtllm_cli` fixture which runs aiperf inside the TRT-LLM
    Docker container (nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7).
    """

    async def test_basic_generate(
        self, trtllm_cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Basic TRT-LLM in-engine generation via the full aiperf pipeline."""
        result = await trtllm_cli.run(_base_cmd(), timeout=600.0)
        assert result.request_count == REQUEST_COUNT

    async def test_with_extra_inputs(
        self, trtllm_cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Extra inputs (temperature, top_p) flow through to TRT-LLM sampling params."""
        result = await trtllm_cli.run(
            _base_cmd("--extra-inputs temperature:0.5 --extra-inputs top_p:0.9"),
            timeout=600.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_concurrent_requests(
        self, trtllm_cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """TRT-LLM engine handles concurrent requests (concurrency > 1)."""
        result = await trtllm_cli.run(
            _base_cmd(concurrency=4, request_count=20),
            timeout=600.0,
        )
        assert result.request_count == 20

    async def test_streaming_mode(
        self, trtllm_cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Streaming generation captures TTFT via cumulative text delta detection.

        TRT-LLM yields cumulative text (not deltas), so the transport tracks
        prev_text_len to detect new content for first-token timing.
        """
        result = await trtllm_cli.run(
            _base_cmd("--streaming"),
            timeout=600.0,
        )
        assert result.request_count == REQUEST_COUNT
        assert result.has_streaming_metrics

    async def test_pre_tokenized_dataset(
        self, trtllm_cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Pre-tokenized input bypasses chat template and passes token IDs directly.

        TRT-LLM accepts list[int] as prompt input via the pre-tokenized path,
        skipping _messages_to_prompt and apply_chat_template.
        """
        result = await trtllm_cli.run(
            _base_cmd("--pre-tokenized"),
            timeout=600.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_warmup_iterations(
        self, trtllm_cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Warmup iterations run before profiling to stabilize GPU performance.

        Exercises _run_warmup -> _warmup_single with configurable input/output
        token counts matching the endpoint's streaming mode.
        """
        result = await trtllm_cli.run(
            _base_cmd(
                "--engine-params warmup_iterations:3 "
                "--engine-params warmup_input_tokens:64 "
                "--engine-params warmup_output_tokens:4"
            ),
            timeout=600.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_engine_telemetry(
        self, trtllm_cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Engine telemetry polls TRT-LLM get_stats_async at configured interval.

        Enables the background telemetry loop that collects EngineIterationStats
        (batch_size, num_tokens, queue_depth) from the engine's stats API.
        """
        result = await trtllm_cli.run(
            _base_cmd(
                "--engine-params telemetry:true "
                "--engine-params telemetry_interval_ms:100"
            ),
            timeout=600.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_token_id_preservation(
        self, trtllm_cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Token ID preservation captures output_token_ids in InEngineResponse.

        When preserve_token_ids:true, the transport stores completion.token_ids
        from TRT-LLM output on each request for downstream analysis.
        """
        result = await trtllm_cli.run(
            _base_cmd("--engine-params preserve_token_ids:true"),
            timeout=600.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_latency_optimized_preset(
        self, trtllm_cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Latency-optimized preset configures TRT-LLM for single-request perf.

        Sets GUARANTEED_NO_EVICT scheduler, cuda graphs, PDL env vars,
        max_batch_size=1, chunked_prefill=false, and 2 warmup iterations
        matching trtllm-bench's low_latency preset.
        """
        result = await trtllm_cli.run(
            _base_cmd("--engine-params latency_optimized:true"),
            timeout=600.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_uniform_isl_distribution(
        self, trtllm_cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Uniform ISL distribution generates prompts with lengths in [min, max] range.

        Uses --isl-distribution uniform with --isl-min and --isl-max to control
        the input sequence length distribution for the synthetic dataset.
        """
        result = await trtllm_cli.run(
            _base_cmd("--isl-distribution uniform --isl-min 50 --isl-max 200"),
            timeout=600.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_world_size_metric(
        self, trtllm_cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """World size metric is reported for per-GPU throughput calculations.

        The --world-size flag sets the world_size used by WorldSizeMetric and
        PerGPUOutputThroughputMetric for multi-GPU normalization.
        """
        result = await trtllm_cli.run(
            _base_cmd("--world-size 1"),
            timeout=600.0,
        )
        assert result.request_count == REQUEST_COUNT

    async def test_streaming_concurrent(
        self, trtllm_cli: AIPerfCLI, watchdog: InEngineWatchdog
    ) -> None:
        """Streaming mode with concurrency exercises parallel async generators.

        Combines --streaming with concurrency=4 to verify that multiple
        concurrent generate_async(streaming=True) calls are handled correctly
        by TRT-LLM's inflight batching scheduler.
        """
        result = await trtllm_cli.run(
            _base_cmd("--streaming", concurrency=4, request_count=20),
            timeout=600.0,
        )
        assert result.request_count == 20
        assert result.has_streaming_metrics
