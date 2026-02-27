# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for server metrics configuration and additive collector behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.messages import ProfileConfigureCommand
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.manager import ServerMetricsManager

# =============================================================================
# UserConfig parsing tests
# =============================================================================


class TestServerMetricsConfig:
    """Test UserConfig parsing of --server-metrics."""

    def test_default_system_metrics_disabled(self):
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["m"],
                type=EndpointType.CHAT,
                urls=["http://localhost:8000"],
            ),
        )
        assert config.system_metrics_enabled is False
        assert config.server_metrics_urls == []

    def test_system_flag_enables_system_metrics(self):
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["m"],
                type=EndpointType.CHAT,
                urls=["http://localhost:8000"],
            ),
            server_metrics=["system"],
        )
        assert config.system_metrics_enabled is True
        assert config.server_metrics_urls == []

    def test_system_flag_case_insensitive(self):
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["m"],
                type=EndpointType.CHAT,
                urls=["http://localhost:8000"],
            ),
            server_metrics=["System"],
        )
        assert config.system_metrics_enabled is True

    def test_urls_only(self):
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["m"],
                type=EndpointType.CHAT,
                urls=["http://localhost:8000"],
            ),
            server_metrics=["http://node1:9090/metrics"],
        )
        assert config.system_metrics_enabled is False
        assert len(config.server_metrics_urls) == 1

    def test_system_with_urls_both_enabled(self):
        """system + URLs is additive: both are configured."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["m"],
                type=EndpointType.CHAT,
                urls=["http://localhost:8000"],
            ),
            server_metrics=["system", "http://node1:9090/metrics"],
        )
        assert config.system_metrics_enabled is True
        assert len(config.server_metrics_urls) == 1
        assert "http://node1:9090/metrics" in config.server_metrics_urls

    def test_system_with_multiple_urls(self):
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["m"],
                type=EndpointType.CHAT,
                urls=["http://localhost:8000"],
            ),
            server_metrics=[
                "system",
                "http://node1:9090/metrics",
                "http://node2:9090/metrics",
            ],
        )
        assert config.system_metrics_enabled is True
        assert len(config.server_metrics_urls) == 2

    def test_invalid_server_metrics_item_raises_error(self):
        with pytest.raises(ValueError, match="Invalid server metrics item"):
            UserConfig(
                endpoint=EndpointConfig(
                    model_names=["m"],
                    type=EndpointType.CHAT,
                    urls=["http://localhost:8000"],
                ),
                server_metrics=["foobar"],
            )

    def test_system_with_remote_server_warns(self, caplog):
        """System collector with remote inference URL logs a warning."""
        UserConfig(
            endpoint=EndpointConfig(
                model_names=["m"],
                type=EndpointType.CHAT,
                urls=["http://remote-server:8000"],
            ),
            server_metrics=["system"],
        )
        assert any("client machine" in r.message for r in caplog.records)

    def test_system_with_localhost_no_warning(self, caplog):
        UserConfig(
            endpoint=EndpointConfig(
                model_names=["m"],
                type=EndpointType.CHAT,
                urls=["http://localhost:8000"],
            ),
            server_metrics=["system"],
        )
        assert not any("client machine" in r.message for r in caplog.records)


# =============================================================================
# Manager additive collector tests
# =============================================================================


@pytest.fixture
def user_config_system_only() -> UserConfig:
    """UserConfig with system collector enabled, no extra URLs."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["m"], type=EndpointType.CHAT, urls=["http://localhost:8000"]
        ),
        server_metrics=["system"],
    )


@pytest.fixture
def user_config_prometheus_only() -> UserConfig:
    """UserConfig with default (prometheus-only) configuration."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["m"], type=EndpointType.CHAT, urls=["http://localhost:8000"]
        ),
    )


@pytest.fixture
def user_config_both() -> UserConfig:
    """UserConfig with both system and extra Prometheus URLs."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["m"], type=EndpointType.CHAT, urls=["http://localhost:8000"]
        ),
        server_metrics=["system", "http://node1:9090/metrics"],
    )


class TestServerMetricsManagerAdditiveCollectors:
    """Test that ServerMetricsManager configures collectors additively."""

    def test_manager_system_enabled_flag(
        self,
        service_config: ServiceConfig,
        user_config_system_only: UserConfig,
    ):
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_system_only,
        )
        assert manager._system_metrics_enabled is True

    def test_manager_system_disabled_by_default(
        self,
        service_config: ServiceConfig,
        user_config_prometheus_only: UserConfig,
    ):
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_prometheus_only,
        )
        assert manager._system_metrics_enabled is False

    @pytest.mark.asyncio
    async def test_configure_prometheus_only(
        self,
        service_config: ServiceConfig,
        user_config_prometheus_only: UserConfig,
    ):
        """Default config: Prometheus called, system NOT called."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_prometheus_only,
        )
        manager._configure_system_collector = AsyncMock()
        manager._configure_prometheus_collectors = AsyncMock()
        manager.publish = AsyncMock()

        message = ProfileConfigureCommand(service_id="test", config={})
        await manager._profile_configure_command(message)

        manager._configure_prometheus_collectors.assert_called_once()
        manager._configure_system_collector.assert_not_called()

    @pytest.mark.asyncio
    async def test_configure_system_only(
        self,
        service_config: ServiceConfig,
        user_config_system_only: UserConfig,
    ):
        """system flag: both Prometheus AND system called."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_system_only,
        )
        manager._configure_system_collector = AsyncMock()
        manager._configure_prometheus_collectors = AsyncMock()
        manager.publish = AsyncMock()

        message = ProfileConfigureCommand(service_id="test", config={})
        await manager._profile_configure_command(message)

        manager._configure_prometheus_collectors.assert_called_once()
        manager._configure_system_collector.assert_called_once()

    @pytest.mark.asyncio
    async def test_configure_both(
        self,
        service_config: ServiceConfig,
        user_config_both: UserConfig,
    ):
        """system + URLs: both Prometheus AND system called."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_both,
        )
        manager._configure_system_collector = AsyncMock()
        manager._configure_prometheus_collectors = AsyncMock()
        manager.publish = AsyncMock()

        message = ProfileConfigureCommand(service_id="test", config={})
        await manager._profile_configure_command(message)

        manager._configure_prometheus_collectors.assert_called_once()
        manager._configure_system_collector.assert_called_once()

    @pytest.mark.asyncio
    async def test_configure_system_collector_creates_collector(
        self,
        service_config: ServiceConfig,
        user_config_system_only: UserConfig,
    ):
        """Verify that _configure_system_collector registers the collector."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_system_only,
        )
        manager.publish = AsyncMock()

        mock_collector = MagicMock()
        mock_collector.is_url_reachable = AsyncMock(return_value=True)
        mock_collector.initialize = AsyncMock()
        mock_collector.collect_and_process_metrics = AsyncMock()
        mock_collector.endpoint_url = "system://localhost"

        mock_cls = MagicMock(return_value=mock_collector)

        with patch(
            "aiperf.server_metrics.manager.plugins.get_class", return_value=mock_cls
        ):
            await manager._configure_system_collector()

        assert "system://localhost" in manager._collectors

    @pytest.mark.asyncio
    async def test_configure_system_collector_not_reachable_no_failure(
        self,
        service_config: ServiceConfig,
        user_config_system_only: UserConfig,
    ):
        """Unreachable system collector doesn't prevent overall success."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_system_only,
        )
        manager.publish = AsyncMock()

        mock_collector = MagicMock()
        mock_collector.is_url_reachable = AsyncMock(return_value=False)
        mock_collector.endpoint_url = "system://localhost"

        mock_cls = MagicMock(return_value=mock_collector)

        with patch(
            "aiperf.server_metrics.manager.plugins.get_class", return_value=mock_cls
        ):
            await manager._configure_system_collector()

        assert len(manager._collectors) == 0
