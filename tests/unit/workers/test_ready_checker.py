# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for endpoint readiness checker."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aiperf.workers.ready_checker import (
    _CANNED_PAYLOADS,
    _DEFAULT_PATHS,
    wait_for_endpoint,
)


class TestWaitForEndpoint:
    @pytest.mark.asyncio
    async def test_skips_when_timeout_zero(self) -> None:
        with patch("aiperf.workers.ready_checker.aiohttp") as mock_aiohttp:
            await wait_for_endpoint("http://localhost:8000", timeout=0)
            mock_aiohttp.ClientSession.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_timeout_negative(self) -> None:
        with patch("aiperf.workers.ready_checker.aiohttp") as mock_aiohttp:
            await wait_for_endpoint("http://localhost:8000", timeout=-1)
            mock_aiohttp.ClientSession.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_immediately_on_success(self) -> None:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.workers.ready_checker.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch("aiperf.workers.ready_checker.create_tcp_connector"),
        ):
            await wait_for_endpoint("http://localhost:8000", timeout=10)

        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "/v1/chat/completions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_retries_on_500(self) -> None:
        call_count = 0

        mock_resp_fail = AsyncMock()
        mock_resp_fail.status = 503
        mock_resp_fail.__aenter__ = AsyncMock(return_value=mock_resp_fail)
        mock_resp_fail.__aexit__ = AsyncMock(return_value=False)

        mock_resp_ok = AsyncMock()
        mock_resp_ok.status = 200
        mock_resp_ok.__aenter__ = AsyncMock(return_value=mock_resp_ok)
        mock_resp_ok.__aexit__ = AsyncMock(return_value=False)

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_resp_fail if call_count == 1 else mock_resp_ok

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=side_effect)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.workers.ready_checker.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch("aiperf.workers.ready_checker.create_tcp_connector"),
            patch("aiperf.workers.ready_checker.asyncio.sleep", new_callable=AsyncMock),
        ):
            await wait_for_endpoint("http://localhost:8000", timeout=30)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self) -> None:
        call_count = 0

        mock_resp_ok = AsyncMock()
        mock_resp_ok.status = 200
        mock_resp_ok.__aenter__ = AsyncMock(return_value=mock_resp_ok)
        mock_resp_ok.__aexit__ = AsyncMock(return_value=False)

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aiohttp.ClientConnectorError(
                    connection_key=MagicMock(), os_error=OSError("refused")
                )
            return mock_resp_ok

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=side_effect)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.workers.ready_checker.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch("aiperf.workers.ready_checker.create_tcp_connector"),
            patch("aiperf.workers.ready_checker.asyncio.sleep", new_callable=AsyncMock),
        ):
            await wait_for_endpoint("http://localhost:8000", timeout=30)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_raises(self) -> None:
        mock_resp = AsyncMock()
        mock_resp.status = 503
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.workers.ready_checker.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch("aiperf.workers.ready_checker.create_tcp_connector"),
            patch("aiperf.workers.ready_checker.asyncio.sleep", new_callable=AsyncMock),
            patch(
                "aiperf.workers.ready_checker.time.perf_counter",
                side_effect=[0, 0, 100],
            ),
            pytest.raises(TimeoutError, match="not ready after"),
        ):
            await wait_for_endpoint("http://localhost:8000", timeout=10)

    @pytest.mark.asyncio
    async def test_uses_completions_path(self) -> None:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.workers.ready_checker.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch("aiperf.workers.ready_checker.create_tcp_connector"),
        ):
            await wait_for_endpoint(
                "http://localhost:8000",
                endpoint_type="completions",
                timeout=10,
            )

        call_url = mock_session.post.call_args[0][0]
        assert "/v1/completions" in call_url

    @pytest.mark.asyncio
    async def test_custom_path_override(self) -> None:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.workers.ready_checker.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch("aiperf.workers.ready_checker.create_tcp_connector"),
        ):
            await wait_for_endpoint(
                "http://localhost:8000",
                path="/custom/generate",
                timeout=10,
            )

        call_url = mock_session.post.call_args[0][0]
        assert call_url == "http://localhost:8000/custom/generate"

    @pytest.mark.asyncio
    async def test_model_name_injected(self) -> None:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.workers.ready_checker.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch("aiperf.workers.ready_checker.create_tcp_connector"),
        ):
            await wait_for_endpoint(
                "http://localhost:8000",
                model="llama-3-8b",
                timeout=10,
            )

        payload = mock_session.post.call_args[1]["json"]
        assert payload["model"] == "llama-3-8b"

    @pytest.mark.asyncio
    async def test_api_key_sent(self) -> None:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.workers.ready_checker.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch("aiperf.workers.ready_checker.create_tcp_connector"),
        ):
            await wait_for_endpoint(
                "http://localhost:8000",
                api_key="sk-test-key",
                timeout=10,
            )

        headers = mock_session.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-test-key"

    @pytest.mark.asyncio
    async def test_400_counts_as_ready(self) -> None:
        """4xx means server is up and model loaded, just bad request params."""
        mock_resp = AsyncMock()
        mock_resp.status = 400
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "aiperf.workers.ready_checker.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch("aiperf.workers.ready_checker.create_tcp_connector"),
        ):
            await wait_for_endpoint("http://localhost:8000", timeout=10)


class TestCannedPayloads:
    def test_chat_payload_has_messages(self) -> None:
        payload = _CANNED_PAYLOADS["chat"]
        assert "messages" in payload
        assert payload["max_tokens"] == 1

    def test_completions_payload_has_prompt(self) -> None:
        payload = _CANNED_PAYLOADS["completions"]
        assert "prompt" in payload
        assert payload["max_tokens"] == 1

    def test_embeddings_payload_has_input(self) -> None:
        payload = _CANNED_PAYLOADS["embeddings"]
        assert "input" in payload

    def test_default_paths_cover_main_types(self) -> None:
        assert "chat" in _DEFAULT_PATHS
        assert "completions" in _DEFAULT_PATHS
        assert "embeddings" in _DEFAULT_PATHS
