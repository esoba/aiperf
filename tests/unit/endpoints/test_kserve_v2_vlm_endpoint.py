# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import Image, Text, Turn
from aiperf.endpoints.kserve_v2_vlm import KServeV2VLMEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestKServeV2VLMFormatPayload:
    """Tests for KServeV2VLMEndpoint.format_payload."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(EndpointType.KSERVE_V2_VLM, model_name="vlm-model")

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(KServeV2VLMEndpoint, model_endpoint)

    def test_format_payload_text_and_image(self, endpoint, model_endpoint):
        """Test basic text + image tensor construction."""
        turn = Turn(
            texts=[Text(contents=["Describe this image"])],
            images=[Image(contents=["data:image/png;base64,iVBOR..."])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "inputs" in payload
        inputs = payload["inputs"]
        assert len(inputs) == 2

        assert inputs[0]["name"] == "text_input"
        assert inputs[0]["shape"] == [1]
        assert inputs[0]["datatype"] == "BYTES"
        assert inputs[0]["data"] == ["Describe this image"]

        assert inputs[1]["name"] == "image"
        assert inputs[1]["shape"] == [1]
        assert inputs[1]["datatype"] == "BYTES"
        assert inputs[1]["data"] == ["data:image/png;base64,iVBOR..."]

    def test_format_payload_text_only(self, endpoint, model_endpoint):
        """Test graceful handling when no images are provided."""
        turn = Turn(texts=[Text(contents=["Hello world"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        inputs = payload["inputs"]
        assert len(inputs) == 1
        assert inputs[0]["name"] == "text_input"
        assert inputs[0]["data"] == ["Hello world"]

    def test_format_payload_multiple_images(self, endpoint, model_endpoint):
        """Test batch images with shape=[N]."""
        turn = Turn(
            texts=[Text(contents=["Compare these images"])],
            images=[Image(contents=["img1_b64", "img2_b64", "img3_b64"])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        inputs = payload["inputs"]
        assert len(inputs) == 2
        image_input = inputs[1]
        assert image_input["name"] == "image"
        assert image_input["shape"] == [3]
        assert image_input["datatype"] == "BYTES"
        assert image_input["data"] == ["img1_b64", "img2_b64", "img3_b64"]

    def test_format_payload_with_max_tokens(self, endpoint, model_endpoint):
        """Test that max_tokens adds an INT32 tensor."""
        turn = Turn(
            texts=[Text(contents=["prompt"])],
            images=[Image(contents=["img_b64"])],
            max_tokens=512,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["inputs"]) == 3
        max_tokens_input = payload["inputs"][2]
        assert max_tokens_input["name"] == "max_tokens"
        assert max_tokens_input["shape"] == [1]
        assert max_tokens_input["datatype"] == "INT32"
        assert max_tokens_input["data"] == [512]

    def test_format_payload_custom_tensor_names(self):
        """Test v2_text_name and v2_image_name via extras."""
        extra = [
            ("v2_text_name", "INPUT_TEXT"),
            ("v2_image_name", "INPUT_IMAGE"),
        ]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_VLM, model_name="vlm-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2VLMEndpoint, model_endpoint
        )

        turn = Turn(
            texts=[Text(contents=["prompt"])],
            images=[Image(contents=["img_b64"])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"][0]["name"] == "INPUT_TEXT"
        assert payload["inputs"][1]["name"] == "INPUT_IMAGE"

    def test_format_payload_with_extra_params(self):
        """Test that remaining extras become parameters."""
        extra = [
            ("temperature", 0.7),
            ("top_k", 40),
        ]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_VLM, model_name="vlm-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2VLMEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["prompt"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["parameters"] == {"temperature": 0.7, "top_k": 40}

    def test_format_payload_empty_turns_raises(self, endpoint, model_endpoint):
        """Test that empty turns raises ValueError."""
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)


class TestKServeV2VLMParseResponse:
    """Tests for KServeV2VLMEndpoint.parse_response."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(EndpointType.KSERVE_V2_VLM, model_name="vlm-model")

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(KServeV2VLMEndpoint, model_endpoint)

    def test_parse_response_basic(self, endpoint):
        """Test parsing a standard V2 text response."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "text_output",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": ["The image shows a cat"],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "The image shows a cat"

    def test_parse_response_custom_output_name(self):
        """Test parsing with custom output tensor name."""
        extra = [("v2_output_name", "OUTPUT_TEXT")]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_VLM, model_name="vlm-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2VLMEndpoint, model_endpoint
        )

        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "OUTPUT_TEXT",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": ["Response text"],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Response text"

    def test_parse_response_fallback_to_first_output(self, endpoint):
        """Test fallback to first output when named output not found."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "other_output",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": ["Fallback text"],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Fallback text"

    def test_parse_response_no_json(self, endpoint):
        """Test parsing response with no JSON returns None."""
        response = create_mock_response(json_data=None)

        parsed = endpoint.parse_response(response)

        assert parsed is None

    def test_parse_response_no_outputs(self, endpoint):
        """Test parsing response with no outputs field returns None."""
        response = create_mock_response(json_data={"id": "123"})

        parsed = endpoint.parse_response(response)

        assert parsed is None
