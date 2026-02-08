# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.kserve_v2_images import KServeV2ImagesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestKServeV2ImagesFormatPayload:
    """Tests for KServeV2ImagesEndpoint.format_payload."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.KSERVE_V2_IMAGES, model_name="sdxl-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            KServeV2ImagesEndpoint, model_endpoint
        )

    def test_format_payload_basic(self, endpoint, model_endpoint):
        """Test basic prompt as BYTES tensor."""
        turn = Turn(texts=[Text(contents=["A beautiful sunset"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "inputs" in payload
        inputs = payload["inputs"]
        assert len(inputs) == 1
        assert inputs[0]["name"] == "prompt"
        assert inputs[0]["shape"] == [1]
        assert inputs[0]["datatype"] == "BYTES"
        assert inputs[0]["data"] == ["A beautiful sunset"]

    def test_format_payload_with_negative_prompt(self):
        """Test negative_prompt as extra BYTES tensor."""
        extra = [("negative_prompt", "blurry, low quality")]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_IMAGES, model_name="sdxl-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2ImagesEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["A cat"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        inputs = payload["inputs"]
        assert len(inputs) == 2
        neg_prompt = inputs[1]
        assert neg_prompt["name"] == "negative_prompt"
        assert neg_prompt["datatype"] == "BYTES"
        assert neg_prompt["data"] == ["blurry, low quality"]

    def test_format_payload_with_diffusion_params(self):
        """Test INT32/FP32/INT64 typed tensors for diffusion parameters."""
        extra = [
            ("num_inference_steps", "30"),
            ("guidance_scale", "7.5"),
            ("seed", "42"),
        ]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_IMAGES, model_name="sdxl-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2ImagesEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["A cat"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        inputs = payload["inputs"]
        # prompt + 3 diffusion params
        assert len(inputs) == 4

        param_map = {inp["name"]: inp for inp in inputs[1:]}

        assert param_map["num_inference_steps"]["datatype"] == "INT32"
        assert param_map["num_inference_steps"]["data"] == [30]

        assert param_map["guidance_scale"]["datatype"] == "FP32"
        assert param_map["guidance_scale"]["data"] == [7.5]

        assert param_map["seed"]["datatype"] == "INT64"
        assert param_map["seed"]["data"] == [42]

    def test_format_payload_custom_tensor_names(self):
        """Test custom prompt and output tensor names via extras."""
        extra = [
            ("v2_prompt_name", "INPUT_PROMPT"),
            ("v2_output_name", "OUTPUT_IMAGE"),
        ]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_IMAGES, model_name="sdxl-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2ImagesEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["prompt"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"][0]["name"] == "INPUT_PROMPT"

    def test_format_payload_with_extra_params(self):
        """Test that non-tensor extras become parameters."""
        extra = [
            ("custom_param", "value"),
            ("another", 123),
        ]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_IMAGES, model_name="sdxl-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2ImagesEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["prompt"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["parameters"] == {"custom_param": "value", "another": 123}

    def test_format_payload_empty_turns_raises(self, endpoint, model_endpoint):
        """Test that empty turns raises ValueError."""
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_max_tokens_warning(self, endpoint, model_endpoint):
        """Test that max_tokens logs a warning."""
        turn = Turn(texts=[Text(contents=["prompt"])], max_tokens=256)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        # Should not raise, just warn
        payload = endpoint.format_payload(request_info)

        # max_tokens should not appear in inputs
        input_names = [inp["name"] for inp in payload["inputs"]]
        assert "max_tokens" not in input_names


class TestKServeV2ImagesParseResponse:
    """Tests for KServeV2ImagesEndpoint.parse_response."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.KSERVE_V2_IMAGES, model_name="sdxl-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            KServeV2ImagesEndpoint, model_endpoint
        )

    def test_parse_response_single_image(self, endpoint):
        """Test BYTES output to ImageResponseData with single image."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "generated_image",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": ["iVBORw0KGgoAAAANSUhEUg..."],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert len(parsed.data.images) == 1
        assert parsed.data.images[0].b64_json == "iVBORw0KGgoAAAANSUhEUg..."

    def test_parse_response_multiple_images(self, endpoint):
        """Test batch output with multiple images."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "generated_image",
                        "shape": [2],
                        "datatype": "BYTES",
                        "data": ["img1_b64", "img2_b64"],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert len(parsed.data.images) == 2
        assert parsed.data.images[0].b64_json == "img1_b64"
        assert parsed.data.images[1].b64_json == "img2_b64"

    def test_parse_response_custom_output_name(self):
        """Test parsing with custom output tensor name."""
        extra = [("v2_output_name", "OUTPUT_IMAGE")]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_IMAGES, model_name="sdxl-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2ImagesEndpoint, model_endpoint
        )

        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "OUTPUT_IMAGE",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": ["img_b64"],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.images[0].b64_json == "img_b64"

    def test_parse_response_fallback_to_first_output(self, endpoint):
        """Test fallback to first output when named output not found."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "other_output",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": ["fallback_img_b64"],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.images[0].b64_json == "fallback_img_b64"

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

    def test_parse_response_empty_data(self, endpoint):
        """Test parsing response with empty data array returns None."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "generated_image",
                        "shape": [0],
                        "datatype": "BYTES",
                        "data": [],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is None
