# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestRecord
from aiperf.plugin.enums import EndpointType, TransportType
from aiperf.plugin.schema.schemas import EndpointMetadata
from aiperf.workers.inference_client import (
    InferenceClient,
    detect_transport_from_url,
    validate_endpoint_transport_compatibility,
)


@pytest.fixture
def mock_http_transport_entry():
    """Create a mock transport entry with http/https url_schemes."""
    entry = MagicMock()
    entry.name = TransportType.HTTP.value
    entry.metadata = {"url_schemes": ["http", "https"]}
    return entry


def _make_endpoint_metadata(**overrides) -> EndpointMetadata:
    """Create an EndpointMetadata with sensible defaults."""
    defaults = {
        "metrics_title": "Test Metrics",
        "endpoint_path": "/v1/test",
        "supports_streaming": False,
        "tokenizes_input": True,
        "produces_tokens": True,
    }
    return EndpointMetadata(**(defaults | overrides))


class TestDetectTransportFromUrl:
    """Tests for detect_transport_from_url function."""

    @pytest.fixture(autouse=True)
    def mock_transport_entries(self, mock_http_transport_entry):
        """Mock plugins.list_entries to return http transport with url_schemes."""
        with patch(
            "aiperf.workers.inference_client.plugins.list_entries",
            return_value=[mock_http_transport_entry],
        ):
            yield

    @pytest.mark.parametrize(
        "url,expected_transport",
        [
            param("http://api.example.com:8000", TransportType.HTTP.value, id="http_with_port"),
            param("https://api.example.com:8443", TransportType.HTTP.value, id="https_with_port"),
            param("http://localhost:8000", TransportType.HTTP.value, id="http_localhost"),
            param("http://127.0.0.1:8000", TransportType.HTTP.value, id="http_localhost_ip"),
            param("http://[::1]:8000", TransportType.HTTP.value, id="http_ipv6"),
            param("http://api.example.com", TransportType.HTTP.value, id="http_no_port"),
            param("https://api.example.com", TransportType.HTTP.value, id="https_no_port"),
            param("http://localhost:8000/api/v1/chat", TransportType.HTTP.value, id="with_path"),
            param("http://api.example.com?model=gpt-4&key=value", TransportType.HTTP.value, id="with_query"),
            param("http://user:password@api.example.com:8000", TransportType.HTTP.value, id="with_credentials"),
            param("http://api.example.com#section", TransportType.HTTP.value, id="with_fragment"),
            param("http://api.example.com/path/with%20spaces", TransportType.HTTP.value, id="with_encoded_spaces"),
            param("https://api.openai.com/v1/chat/completions", TransportType.HTTP.value, id="openai_api"),
        ],
    )  # fmt: skip
    def test_http_https_detection(self, url, expected_transport):
        """Test detection of HTTP/HTTPS URLs with various components."""
        result = detect_transport_from_url(url)
        assert result == expected_transport

    @pytest.mark.parametrize(
        "url",
        [
            param("HTTP://api.example.com", id="uppercase_scheme"),
            param("Http://api.example.com", id="mixed_case_scheme"),
            param("hTTp://api.example.com", id="random_case_scheme"),
        ],
    )
    def test_scheme_case_insensitive(self, url):
        """Test that scheme detection is case-insensitive."""
        assert detect_transport_from_url(url) == TransportType.HTTP.value

    @pytest.mark.parametrize(
        "url",
        [
            param("", id="empty_string"),
            param("http://", id="scheme_only"),
            param("api.example.com:8000", id="no_scheme_with_port"),
            param("api.example.com", id="no_scheme_no_port"),
            param("localhost", id="localhost_no_scheme"),
            param("/path/to/file.sock", id="file_path"),
        ],
    )
    def test_edge_cases_default_to_http_or_raise(self, url):
        """Test edge cases return HTTP or raise ValueError."""
        with contextlib.suppress(ValueError):
            assert detect_transport_from_url(url) == TransportType.HTTP.value

    @pytest.mark.parametrize(
        "url",
        [
            param("unknown://api.example.com", id="unknown_scheme"),
            param("ftp://files.example.com", id="ftp_scheme"),
            param("grpc://localhost:50051", id="grpc_scheme"),
        ],
    )
    def test_unregistered_schemes_raise_error(self, url):
        """Test that unregistered schemes raise ValueError."""
        with pytest.raises(ValueError):
            detect_transport_from_url(url)


class TestInferenceClient:
    """Tests for InferenceClient functionality."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo."""
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000/v1/test",
            ),
        )

    @pytest.fixture
    def inference_client(self, model_endpoint, mock_http_transport_entry):
        """Create an InferenceClient instance."""
        mock_transport = MagicMock()
        mock_endpoint = MagicMock()
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}
        mock_endpoint.format_payload.return_value = {}

        def mock_get_class(protocol, name):
            if protocol == "endpoint":
                return lambda **kwargs: mock_endpoint
            if protocol == "transport":
                return lambda **kwargs: mock_transport
            raise ValueError(f"Unknown protocol: {protocol}")

        with (
            patch(
                "aiperf.workers.inference_client.plugins.get_class",
                side_effect=mock_get_class,
            ),
            patch(
                "aiperf.workers.inference_client.plugins.list_entries",
                return_value=[mock_http_transport_entry],
            ),
            patch(
                "aiperf.workers.inference_client.plugins.get_endpoint_metadata",
                return_value=_make_endpoint_metadata(supported_transports=["http"]),
            ),
        ):
            return InferenceClient(
                model_endpoint=model_endpoint, service_id="test-service-id"
            )

    @pytest.mark.asyncio
    async def test_send_request_sets_endpoint_headers(
        self, inference_client, model_endpoint, sample_request_info
    ):
        """Test that send_request sets endpoint_headers on request_info."""
        model_endpoint.endpoint.api_key = "test-key"
        model_endpoint.endpoint.headers = [("X-Custom", "value")]

        request_info = sample_request_info

        expected_headers = {
            "Authorization": "Bearer test-key",
            "X-Custom": "value",
        }
        inference_client.endpoint.get_endpoint_headers.return_value = expected_headers

        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord(request_info=sample_request_info)
        )

        await inference_client.send_request(request_info)

        assert "Authorization" in request_info.endpoint_headers
        assert request_info.endpoint_headers["Authorization"] == "Bearer test-key"
        assert request_info.endpoint_headers["X-Custom"] == "value"

    @pytest.mark.asyncio
    async def test_send_request_sets_endpoint_params(
        self, inference_client, model_endpoint, sample_request_info
    ):
        """Test that send_request sets endpoint_params on request_info."""
        model_endpoint.endpoint.url_params = {"api-version": "v1", "timeout": "30"}

        request_info = sample_request_info

        expected_params = {"api-version": "v1", "timeout": "30"}
        inference_client.endpoint.get_endpoint_params.return_value = expected_params

        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord(request_info=sample_request_info)
        )

        await inference_client.send_request(request_info)

        assert request_info.endpoint_params["api-version"] == "v1"
        assert request_info.endpoint_params["timeout"] == "30"

    @pytest.mark.asyncio
    async def test_configure_delegates_to_transport(self, inference_client):
        """Test that configure() delegates to transport.configure()."""
        inference_client.transport.configure = AsyncMock()
        await inference_client.configure()
        inference_client.transport.configure.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_request_calls_transport(
        self,
        inference_client,
        model_endpoint,
        sample_request_info,
        sample_request_record,
    ):
        """Test that send_request delegates to transport."""
        request_info = sample_request_info
        expected_record = sample_request_record

        inference_client.transport.send_request = AsyncMock(
            return_value=expected_record
        )

        record = await inference_client.send_request(request_info)

        inference_client.transport.send_request.assert_called_once()
        call_args = inference_client.transport.send_request.call_args
        assert call_args[0][0] == request_info
        assert record == expected_record


class TestValidateEndpointTransportCompatibility:
    """Tests for validate_endpoint_transport_compatibility function."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo with chat endpoint and http URL."""
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000",
            ),
        )

    @pytest.mark.parametrize(
        "endpoint_type,transport,supported_transports",
        [
            param("chat", "http", ["http"], id="chat_http"),
            param("completions", "http", ["http"], id="completions_http"),
            param("vllm_generate", "vllm", ["vllm"], id="vllm_generate_vllm"),
            param("sglang_generate", "sglang", ["sglang"], id="sglang_generate_sglang"),
            param("trtllm_generate", "trtllm", ["trtllm"], id="trtllm_generate_trtllm"),
        ],
    )
    def test_compatible_endpoint_transport_passes(
        self,
        model_endpoint,
        endpoint_type,
        transport,
        supported_transports,
    ):
        """Test that compatible endpoint-transport combos pass validation."""
        model_endpoint.endpoint.type = endpoint_type
        model_endpoint.transport = transport

        with patch(
            "aiperf.workers.inference_client.plugins.get_endpoint_metadata",
            return_value=_make_endpoint_metadata(
                supported_transports=supported_transports
            ),
        ):
            validate_endpoint_transport_compatibility(model_endpoint)

    def test_incompatible_endpoint_transport_raises(self, model_endpoint):
        """Test that incompatible endpoint-transport combos raise ValueError."""
        model_endpoint.endpoint.type = "chat"
        model_endpoint.transport = "vllm"

        mock_vllm_entry = MagicMock()
        mock_vllm_entry.name = "vllm_generate"
        mock_vllm_entry.get_typed_metadata.return_value = _make_endpoint_metadata(
            supported_transports=["vllm"]
        )

        with (
            patch(
                "aiperf.workers.inference_client.plugins.get_endpoint_metadata",
                return_value=_make_endpoint_metadata(supported_transports=["http"]),
            ),
            patch(
                "aiperf.workers.inference_client.plugins.list_entries",
                return_value=[mock_vllm_entry],
            ),
            pytest.raises(ValueError, match="does not support transport 'vllm'"),
        ):
            validate_endpoint_transport_compatibility(model_endpoint)

    def test_incompatible_error_suggests_compatible_endpoints(self, model_endpoint):
        """Test that the error message includes compatible endpoints."""
        model_endpoint.endpoint.type = "chat"
        model_endpoint.transport = "vllm"

        mock_vllm_entry = MagicMock()
        mock_vllm_entry.name = "vllm_generate"
        mock_vllm_entry.get_typed_metadata.return_value = _make_endpoint_metadata(
            supported_transports=["vllm"]
        )

        with (
            patch(
                "aiperf.workers.inference_client.plugins.get_endpoint_metadata",
                return_value=_make_endpoint_metadata(supported_transports=["http"]),
            ),
            patch(
                "aiperf.workers.inference_client.plugins.list_entries",
                return_value=[mock_vllm_entry],
            ),
            pytest.raises(ValueError, match="vllm_generate"),
        ):
            validate_endpoint_transport_compatibility(model_endpoint)

    def test_auto_detects_transport_from_url(
        self, model_endpoint, mock_http_transport_entry
    ):
        """Test that transport is auto-detected from URL when not set."""
        model_endpoint.transport = None

        with (
            patch(
                "aiperf.workers.inference_client.plugins.list_entries",
                return_value=[mock_http_transport_entry],
            ),
            patch(
                "aiperf.workers.inference_client.plugins.get_endpoint_metadata",
                return_value=_make_endpoint_metadata(supported_transports=["http"]),
            ),
        ):
            validate_endpoint_transport_compatibility(model_endpoint)

        assert model_endpoint.transport == "http"

    def test_infers_transport_from_endpoint_when_url_incompatible(
        self, model_endpoint, mock_http_transport_entry
    ):
        """Test transport inferred from endpoint metadata when URL gives incompatible transport.

        Simulates: --endpoint-type vllm_generate (no --url, no --transport).
        URL defaults to http://localhost:8000 -> detected as 'http', but vllm_generate
        only supports 'vllm'. Since transport was not explicit and endpoint has exactly
        one supported transport, it should be inferred as 'vllm'.
        """
        model_endpoint.endpoint.type = "vllm_generate"
        model_endpoint.transport = None

        with (
            patch(
                "aiperf.workers.inference_client.plugins.list_entries",
                return_value=[mock_http_transport_entry],
            ),
            patch(
                "aiperf.workers.inference_client.plugins.get_endpoint_metadata",
                return_value=_make_endpoint_metadata(supported_transports=["vllm"]),
            ),
        ):
            validate_endpoint_transport_compatibility(model_endpoint)

        assert model_endpoint.transport == "vllm"

    def test_explicit_incompatible_transport_raises(self, model_endpoint):
        """Test that explicit --transport that's incompatible still raises ValueError."""
        model_endpoint.endpoint.type = "chat"
        model_endpoint.transport = "vllm"  # explicitly set

        mock_vllm_entry = MagicMock()
        mock_vllm_entry.name = "vllm_generate"
        mock_vllm_entry.get_typed_metadata.return_value = _make_endpoint_metadata(
            supported_transports=["vllm"]
        )

        with (
            patch(
                "aiperf.workers.inference_client.plugins.get_endpoint_metadata",
                return_value=_make_endpoint_metadata(supported_transports=["http"]),
            ),
            patch(
                "aiperf.workers.inference_client.plugins.list_entries",
                return_value=[mock_vllm_entry],
            ),
            pytest.raises(ValueError, match="does not support transport 'vllm'"),
        ):
            validate_endpoint_transport_compatibility(model_endpoint)
