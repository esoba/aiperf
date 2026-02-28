# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared Riva endpoint helpers."""

from __future__ import annotations

import pytest

from aiperf.endpoints.riva_helpers import get_extra
from aiperf.endpoints.riva_nlp import RivaTextClassifyEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
)


class TestGetExtra:
    """Tests for get_extra helper."""

    def test_returns_dict_from_extra_list(self) -> None:
        """Extra params should be converted to a dict."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_TEXT_CLASSIFY,
            extra=[("language_code", "de-DE"), ("top_n", 5)],
        )
        endpoint = create_endpoint_with_mock_transport(
            RivaTextClassifyEndpoint, model_endpoint
        )
        result = get_extra(endpoint)

        assert result == {"language_code": "de-DE", "top_n": 5}

    def test_returns_empty_dict_when_no_extra(self) -> None:
        """Endpoint with no extra config should return empty dict."""
        model_endpoint = create_model_endpoint(EndpointType.RIVA_TEXT_CLASSIFY)
        endpoint = create_endpoint_with_mock_transport(
            RivaTextClassifyEndpoint, model_endpoint
        )
        result = get_extra(endpoint)

        assert result == {}

    def test_returns_copy_not_reference(self) -> None:
        """get_extra should return a new dict, not a reference to the original."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_TEXT_CLASSIFY,
            extra=[("key", "value")],
        )
        endpoint = create_endpoint_with_mock_transport(
            RivaTextClassifyEndpoint, model_endpoint
        )
        result1 = get_extra(endpoint)
        result2 = get_extra(endpoint)

        assert result1 == result2
        assert result1 is not result2

    @pytest.mark.parametrize(
        ("extra", "expected_key", "expected_value"),
        [
            ([("language_code", "en-US")], "language_code", "en-US"),
            ([("sample_rate_hertz", 16000)], "sample_rate_hertz", 16000),
            ([("encoding", "FLAC")], "encoding", "FLAC"),
        ],
    )
    def test_extracts_various_types(
        self, extra: list, expected_key: str, expected_value: object
    ) -> None:
        """get_extra should handle various value types."""
        model_endpoint = create_model_endpoint(
            EndpointType.RIVA_TEXT_CLASSIFY, extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            RivaTextClassifyEndpoint, model_endpoint
        )
        result = get_extra(endpoint)

        assert result[expected_key] == expected_value
