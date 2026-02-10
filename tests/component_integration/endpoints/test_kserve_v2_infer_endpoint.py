# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kserve_v2_infer endpoint."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestKServeV2InferEndpoint:
    """Tests for KServe V2 Open Inference Protocol infer endpoint."""

    def test_kserve_v2_infer_basic(self, cli: AIPerfCLI):
        """Basic V2 infer request with synthetic text input."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model v2-model \
                --tokenizer gpt2 \
                --endpoint-type kserve_v2_infer \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
