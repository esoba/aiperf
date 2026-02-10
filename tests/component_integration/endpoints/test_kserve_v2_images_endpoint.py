# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kserve_v2_images endpoint."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestKServeV2ImagesEndpoint:
    """Tests for KServe V2 image generation endpoint."""

    def test_kserve_v2_images_basic(self, cli: AIPerfCLI):
        """V2 image generation with synthetic prompt input."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model v2-sdxl \
                --tokenizer gpt2 \
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
        # Image generation doesn't produce tokens, so no TTFT
        assert (
            not hasattr(result.json, "time_to_first_token")
            or result.json.time_to_first_token is None
        )

    def test_kserve_v2_images_with_diffusion_params(self, cli: AIPerfCLI):
        """V2 image generation with extra diffusion parameters."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model v2-sdxl \
                --tokenizer gpt2 \
                --endpoint-type kserve_v2_images \
                --extra-inputs num_inference_steps:20 guidance_scale:7.5 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert (
            not hasattr(result.json, "time_to_first_token")
            or result.json.time_to_first_token is None
        )
