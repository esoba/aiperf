# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.kserve_v2_infer import parse_v2_text_response


class KServeV2VLMEndpoint(BaseEndpoint):
    """KServe V2 Open Inference Protocol endpoint for vision-language models.

    Wraps text and images as BYTES tensors per the V2 inference protocol spec.
    Tensor names are configurable via --extra v2_text_name:X --extra v2_image_name:Y
    --extra v2_output_name:Z.
    """

    DEFAULT_TEXT_NAME = "text_input"
    DEFAULT_IMAGE_NAME = "image"
    DEFAULT_OUTPUT_NAME = "text_output"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = (
            dict(self.model_endpoint.endpoint.extra)
            if self.model_endpoint.endpoint.extra
            else {}
        )
        self._text_name: str = extra.pop("v2_text_name", self.DEFAULT_TEXT_NAME)
        self._image_name: str = extra.pop("v2_image_name", self.DEFAULT_IMAGE_NAME)
        self._output_name: str = extra.pop("v2_output_name", self.DEFAULT_OUTPUT_NAME)
        self._extra_params: dict[str, Any] = extra

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format V2 inference request with text and image BYTES tensor inputs.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            V2 Open Inference Protocol payload with text and optional image tensors
        """
        if not request_info.turns:
            raise ValueError("KServe V2 VLM endpoint requires at least one turn.")

        turn = request_info.turns[0]

        prompts = [
            content for text in turn.texts for content in text.contents if content
        ]
        prompt = " ".join(prompts) if prompts else ""

        inputs: list[dict[str, Any]] = [
            {
                "name": self._text_name,
                "shape": [1],
                "datatype": "BYTES",
                "data": [prompt],
            }
        ]

        image_data = [
            content for image in turn.images for content in image.contents if content
        ]
        if image_data:
            inputs.append(
                {
                    "name": self._image_name,
                    "shape": [len(image_data)],
                    "datatype": "BYTES",
                    "data": image_data,
                }
            )

        if turn.max_tokens is not None:
            inputs.append(
                {
                    "name": "max_tokens",
                    "shape": [1],
                    "datatype": "INT32",
                    "data": [turn.max_tokens],
                }
            )

        payload: dict[str, Any] = {"inputs": inputs}

        if self._extra_params:
            payload["parameters"] = dict(self._extra_params)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse V2 inference response, extracting text from BYTES output tensor.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text content, or None if no content
        """
        return parse_v2_text_response(self, response, self._output_name)
