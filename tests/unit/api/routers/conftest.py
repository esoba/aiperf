# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for router component tests."""

from __future__ import annotations

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig


@pytest.fixture
def component_service_config() -> ServiceConfig:
    """ServiceConfig for component testing."""
    return ServiceConfig(api_port=9999, api_host="127.0.0.1")


@pytest.fixture
def component_user_config() -> UserConfig:
    """UserConfig for component testing."""
    return UserConfig(
        benchmark_id="test-bench",
        endpoint=EndpointConfig(model_names=["test-model"]),
    )
