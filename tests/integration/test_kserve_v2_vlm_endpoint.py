# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kserve_v2_vlm endpoint against HTTP mock server."""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestKServeV2VLMEndpoint:
    """Tests for KServe V2 VLM endpoint with real HTTP mock server."""

    async def test_kserve_v2_vlm_produces_request_metrics(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """V2 VLM endpoint produces request latency metrics."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model v2-vlm-model \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type kserve_v2_vlm \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.json.request_latency is not None
