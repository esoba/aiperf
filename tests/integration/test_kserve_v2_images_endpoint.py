# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kserve_v2_images endpoint against HTTP mock server."""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestKServeV2ImagesEndpoint:
    """Tests for KServe V2 image generation endpoint with real HTTP mock server."""

    async def test_kserve_v2_images_no_streaming_metrics(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """V2 image generation has no TTFT/ITL metrics but has request latency."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model v2-sdxl \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type kserve_v2_images \
                --synthetic-input-tokens-mean 150 \
                --synthetic-input-tokens-stddev 30 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

        # Image generation should not have token-based streaming metrics
        assert result.json.time_to_first_token is None
        assert result.json.inter_token_latency is None

        # But should have basic request metrics
        assert result.json.request_latency is not None
        assert result.json.request_throughput is not None
