# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kserve_v2_vlm endpoint."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestKServeV2VLMEndpoint:
    """Tests for KServe V2 VLM (Vision-Language Model) endpoint."""

    def test_kserve_v2_vlm_synthetic(self, cli: AIPerfCLI):
        """V2 VLM request with synthetic text input."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model v2-vlm-model \
                --tokenizer gpt2 \
                --endpoint-type kserve_v2_vlm \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
