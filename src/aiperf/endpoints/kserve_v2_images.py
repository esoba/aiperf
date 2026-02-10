# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    ImageDataItem,
    ImageResponseData,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint

# Diffusion parameters that map to typed tensors instead of the parameters dict
_TYPED_TENSOR_PARAMS: dict[str, tuple[str, type]] = {
    "negative_prompt": ("BYTES", str),
    "num_inference_steps": ("INT32", int),
    "guidance_scale": ("FP32", float),
    "seed": ("INT64", int),
}


class KServeV2ImagesEndpoint(BaseEndpoint):
    """KServe V2 Open Inference Protocol endpoint for image generation models.

    Wraps prompt as BYTES tensor and supports diffusion parameters as typed tensors.
    Tensor names are configurable via --extra v2_prompt_name:X --extra v2_output_name:Y.
    """

    DEFAULT_PROMPT_NAME = "prompt"
    DEFAULT_OUTPUT_NAME = "generated_image"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = (
            dict(self.model_endpoint.endpoint.extra)
            if self.model_endpoint.endpoint.extra
            else {}
        )
        self._prompt_name: str = extra.pop("v2_prompt_name", self.DEFAULT_PROMPT_NAME)
        self._output_name: str = extra.pop("v2_output_name", self.DEFAULT_OUTPUT_NAME)
        self._extra_params: dict[str, Any] = extra

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format V2 inference request with BYTES tensor input for image generation.

        Supports diffusion parameters as typed tensors:
        - negative_prompt (BYTES), num_inference_steps (INT32),
          guidance_scale (FP32), seed (INT64)

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            V2 Open Inference Protocol payload for image generation
        """
        if not request_info.turns:
            raise ValueError("KServe V2 images endpoint requires at least one turn.")

        turn = request_info.turns[0]

        if turn.max_tokens is not None:
            self.warning(
                "max_tokens is not supported for image generation and will be ignored."
            )

        prompts = [
            content for text in turn.texts for content in text.contents if content
        ]
        prompt = " ".join(prompts) if prompts else ""

        inputs: list[dict[str, Any]] = [
            {
                "name": self._prompt_name,
                "shape": [1],
                "datatype": "BYTES",
                "data": [prompt],
            }
        ]

        # Build remaining extras, separating typed tensor params from generic parameters
        remaining_extras: dict[str, Any] = {}
        for key, value in self._extra_params.items():
            if key in _TYPED_TENSOR_PARAMS:
                datatype, cast_fn = _TYPED_TENSOR_PARAMS[key]
                inputs.append(
                    {
                        "name": key,
                        "shape": [1],
                        "datatype": datatype,
                        "data": [cast_fn(value)],
                    }
                )
            else:
                remaining_extras[key] = value

        payload: dict[str, Any] = {"inputs": inputs}

        if remaining_extras:
            payload["parameters"] = remaining_extras

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse V2 inference response, extracting base64 image data from BYTES tensor.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with ImageResponseData, or None if no content
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        outputs = json_obj.get("outputs")
        if not outputs:
            return None

        # Find the output tensor with the matching name
        output = next(
            (o for o in outputs if o.get("name") == self._output_name), outputs[0]
        )

        data = output.get("data")
        if not isinstance(data, list) or not data:
            return None

        images = [
            ImageDataItem(b64_json=str(item)) for item in data if item is not None
        ]

        if not images:
            return None

        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=ImageResponseData(images=images),
        )
