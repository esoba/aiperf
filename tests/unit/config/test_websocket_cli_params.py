# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for API CLI parameter validation."""

import pytest
from pydantic import ValidationError

from aiperf.common.config import ServiceConfig


class TestAPICLIParams:
    """Test API CLI parameter validation."""

    def test_no_api_params(self) -> None:
        """Test ServiceConfig without API parameters."""
        config = ServiceConfig()
        assert config.api_port is None
        assert config.api_host is None

    def test_port_only_leaves_host_none(self) -> None:
        """Test that setting port only leaves host as None (resolved at runtime from env)."""
        config = ServiceConfig(api_port=9999)
        assert config.api_port == 9999
        assert config.api_host is None

    def test_port_and_host(self) -> None:
        """Test setting both port and host."""
        config = ServiceConfig(api_port=8080, api_host="0.0.0.0")
        assert config.api_port == 8080
        assert config.api_host == "0.0.0.0"

    def test_host_without_port_raises_error(self) -> None:
        """Test that setting host without port raises validation error."""
        with pytest.raises(ValidationError, match="--api-host requires"):
            ServiceConfig(api_host="0.0.0.0")
