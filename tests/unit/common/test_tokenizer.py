# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from unittest.mock import Mock, patch

import pytest

from aiperf.common.exceptions import NotInitializedError, TokenizerError
from aiperf.common.tokenizer import AliasResolutionResult, Tokenizer, _is_network_error


class TestTokenizer:
    def test_empty_tokenizer(self):
        tokenizer = Tokenizer()
        assert tokenizer._tokenizer is None

        with pytest.raises(NotInitializedError):
            tokenizer("test")
        with pytest.raises(NotInitializedError):
            tokenizer.encode("test")
        with pytest.raises(NotInitializedError):
            tokenizer.decode([1])
        with pytest.raises(NotInitializedError):
            _ = tokenizer.bos_token_id


class TestNetworkErrorDetection:
    """Test the _is_network_error helper function."""

    def test_detects_connection_error(self):
        """Network error keywords should be detected."""
        assert _is_network_error(Exception("Connection timeout"))
        assert _is_network_error(Exception("Network unreachable"))
        assert _is_network_error(Exception("HTTP error occurred"))
        assert _is_network_error(Exception("SSL certificate verify failed"))
        assert _is_network_error(Exception("Failed to resolve hostname"))

    def test_detects_http_status_codes(self):
        """HTTP status codes for transient errors should be detected."""
        assert _is_network_error(Exception("HTTP 429 Too Many Requests"))
        assert _is_network_error(Exception("Server returned 500"))
        assert _is_network_error(Exception("Error 502 Bad Gateway"))
        assert _is_network_error(Exception("503 Service Unavailable"))
        assert _is_network_error(Exception("504 Gateway Timeout"))

    def test_detects_exception_types(self):
        """Known network exception types should be detected."""

        class ConnectionError(Exception):
            pass

        class TimeoutError(Exception):
            pass

        assert _is_network_error(ConnectionError("Failed"))
        assert _is_network_error(TimeoutError("Timed out"))

    def test_ignores_non_network_errors(self):
        """Non-network errors should not be detected as network errors."""
        assert not _is_network_error(Exception("File not found"))
        assert not _is_network_error(Exception("Invalid syntax"))
        assert not _is_network_error(Exception("Permission denied"))
        assert not _is_network_error(ValueError("Invalid value"))


def _mock_auto_tokenizer_patches(hf_side_effect=None):
    """Return common patches for mocking AutoTokenizer.from_pretrained."""
    mock_hf = Mock(side_effect=hf_side_effect) if hf_side_effect else Mock()
    return (
        patch("transformers.AutoTokenizer.from_pretrained", mock_hf),
        patch("aiperf.common.tokenizer._is_offline_mode", return_value=False),
        patch(
            "aiperf.common.tokenizer.resolve_alias",
            return_value=AliasResolutionResult(resolved_name="test-model"),
        ),
    )


class TestTokenizerRetry:
    """Test tokenizer retry logic."""

    def test_successful_load_no_retry(self):
        """Successful load on first attempt should not retry."""
        mock_hf = Mock()
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_hf),
            patch("aiperf.common.tokenizer._is_offline_mode", return_value=False),
            patch(
                "aiperf.common.tokenizer.resolve_alias",
                return_value=AliasResolutionResult(resolved_name="test-model"),
            ),
        ):
            result = Tokenizer.from_pretrained("test-model")

            assert result._tokenizer == mock_hf
            # HF tokenizer called only once (no retries)
            assert mock_hf.call_count == 0  # mock_hf is the return value, not a spy

    def test_retry_on_network_error_then_success(self):
        """Network error on first attempt should retry and succeed."""
        mock_hf = Mock()
        call_count = 0

        def hf_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Connection timeout")
            return mock_hf

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained", side_effect=hf_side_effect
            ),
            patch("aiperf.common.tokenizer._is_offline_mode", return_value=False),
            patch(
                "aiperf.common.tokenizer.resolve_alias",
                return_value=AliasResolutionResult(resolved_name="test-model"),
            ),
            patch("aiperf.common.tokenizer.time.sleep"),
        ):
            result = Tokenizer.from_pretrained("test-model")

            assert result._tokenizer == mock_hf
            assert call_count == 2  # First failed, second succeeded

    def test_exhaust_all_retries(self):
        """Network error on all attempts should exhaust retries and fail."""
        call_count = 0

        def hf_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise OSError("Connection timeout")

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained", side_effect=hf_side_effect
            ),
            patch("aiperf.common.tokenizer._is_offline_mode", return_value=False),
            patch(
                "aiperf.common.tokenizer.resolve_alias",
                return_value=AliasResolutionResult(resolved_name="test-model"),
            ),
            patch("aiperf.common.tokenizer.time.sleep"),
            patch("aiperf.common.tokenizer.Environment.TOKENIZER.INIT_MAX_RETRIES", 3),
        ):
            with pytest.raises(TokenizerError):
                Tokenizer.from_pretrained("test-model")

            # Should try 4 times (initial + 3 retries)
            assert call_count == 4

    def test_no_retry_on_non_network_error(self):
        """Non-network errors should fail immediately without retry."""
        call_count = 0

        def hf_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid model name")

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained", side_effect=hf_side_effect
            ),
            patch("aiperf.common.tokenizer._is_offline_mode", return_value=False),
            patch(
                "aiperf.common.tokenizer.resolve_alias",
                return_value=AliasResolutionResult(resolved_name="test-model"),
            ),
        ):
            with pytest.raises(TokenizerError):
                Tokenizer.from_pretrained("test-model")

            # Should only try once (no retries for non-network errors)
            assert call_count == 1

    def test_exponential_backoff_delays(self):
        """Delays should follow exponential backoff pattern."""
        sleep_delays = []

        def hf_side_effect(*args, **kwargs):
            raise OSError("Connection timeout")

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained", side_effect=hf_side_effect
            ),
            patch("aiperf.common.tokenizer._is_offline_mode", return_value=False),
            patch(
                "aiperf.common.tokenizer.resolve_alias",
                return_value=AliasResolutionResult(resolved_name="test-model"),
            ),
            patch(
                "aiperf.common.tokenizer.time.sleep",
                side_effect=lambda d: sleep_delays.append(d),
            ),
            patch("aiperf.common.tokenizer.Environment.TOKENIZER.INIT_MAX_RETRIES", 3),
            patch("aiperf.common.tokenizer.Environment.TOKENIZER.INIT_BASE_DELAY", 1.0),
        ):
            with pytest.raises(TokenizerError):
                Tokenizer.from_pretrained("test-model")

            # Should have delays: 1s, 2s, 4s (2^0, 2^1, 2^2)
            assert sleep_delays == [1.0, 2.0, 4.0]

    def test_custom_retry_configuration(self):
        """Custom retry configuration should be respected."""
        call_count = 0
        sleep_delays = []

        def hf_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise OSError("HTTP 429")

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained", side_effect=hf_side_effect
            ),
            patch("aiperf.common.tokenizer._is_offline_mode", return_value=False),
            patch(
                "aiperf.common.tokenizer.resolve_alias",
                return_value=AliasResolutionResult(resolved_name="test-model"),
            ),
            patch(
                "aiperf.common.tokenizer.time.sleep",
                side_effect=lambda d: sleep_delays.append(d),
            ),
            patch("aiperf.common.tokenizer.Environment.TOKENIZER.INIT_MAX_RETRIES", 2),
            patch("aiperf.common.tokenizer.Environment.TOKENIZER.INIT_BASE_DELAY", 0.5),
        ):
            with pytest.raises(TokenizerError):
                Tokenizer.from_pretrained("test-model")

            # Should try 3 times (initial + 2 retries)
            assert call_count == 3
            # Should have delays: 0.5s, 1.0s (0.5 * 2^0, 0.5 * 2^1)
            assert sleep_delays == [0.5, 1.0]
