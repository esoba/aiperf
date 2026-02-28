# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for KServeV2GrpcSerializer."""

from __future__ import annotations

from aiperf.transports.grpc.grpc_transport import GrpcSerializerProtocol
from aiperf.transports.grpc.kserve_v2_serializers import KServeV2GrpcSerializer
from aiperf.transports.grpc.proto.kserve import grpc_predict_v2_pb2 as pb2
from aiperf.transports.grpc.stream_chunk import StreamChunk


def _make_infer_response(text: str = "Generated text") -> pb2.ModelInferResponse:
    """Create a ModelInferResponse protobuf with BYTES output."""
    response = pb2.ModelInferResponse()
    response.model_name = "test-model"
    output = response.outputs.add()
    output.name = "text_output"
    output.datatype = "BYTES"
    output.shape.append(1)
    output.contents.bytes_contents.append(text.encode("utf-8"))
    return response


class TestKServeV2GrpcSerializerProtocol:
    """Tests for protocol conformance."""

    def test_implements_protocol(self) -> None:
        """KServeV2GrpcSerializer should satisfy GrpcSerializerProtocol."""
        assert isinstance(KServeV2GrpcSerializer(), GrpcSerializerProtocol)


class TestSerializeRequest:
    """Tests for KServeV2GrpcSerializer.serialize_request."""

    def test_roundtrip_bytes(self) -> None:
        """Serialized bytes should parse back to a valid ModelInferRequest."""
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Hello world"],
                }
            ]
        }
        serializer = KServeV2GrpcSerializer()
        data = serializer.serialize_request(
            payload, model_name="my-model", request_id="r1"
        )
        assert isinstance(data, bytes)
        assert len(data) > 0

        parsed = pb2.ModelInferRequest()
        parsed.ParseFromString(data)
        assert parsed.model_name == "my-model"
        assert parsed.id == "r1"
        assert len(parsed.inputs) == 1
        assert parsed.inputs[0].name == "text_input"

    def test_with_parameters(self) -> None:
        """Request-level parameters should survive serialization."""
        payload = {
            "inputs": [{"name": "t", "shape": [1], "datatype": "BYTES", "data": ["p"]}],
            "parameters": {"stream": True, "max_tokens": 100},
        }
        serializer = KServeV2GrpcSerializer()
        data = serializer.serialize_request(payload, model_name="m")
        parsed = pb2.ModelInferRequest()
        parsed.ParseFromString(data)
        assert parsed.parameters["stream"].bool_param is True
        assert parsed.parameters["max_tokens"].int64_param == 100


class TestDeserializeResponse:
    """Tests for KServeV2GrpcSerializer.deserialize_response."""

    def test_response_dict_and_size(self) -> None:
        """Should return correct dict and wire size."""
        proto = _make_infer_response("Hello")
        data = proto.SerializeToString()

        serializer = KServeV2GrpcSerializer()
        result_dict, size = serializer.deserialize_response(data)

        assert size == len(data)
        assert result_dict["model_name"] == "test-model"
        assert len(result_dict["outputs"]) == 1
        assert result_dict["outputs"][0]["name"] == "text_output"
        assert result_dict["outputs"][0]["data"] == ["Hello"]


class TestDeserializeStreamResponse:
    """Tests for KServeV2GrpcSerializer.deserialize_stream_response."""

    def test_success_chunk(self) -> None:
        """Normal streaming chunk should produce StreamChunk with response_dict."""
        stream_resp = pb2.ModelStreamInferResponse()
        stream_resp.infer_response.CopyFrom(_make_infer_response("token"))
        data = stream_resp.SerializeToString()

        serializer = KServeV2GrpcSerializer()
        chunk = serializer.deserialize_stream_response(data)

        assert isinstance(chunk, StreamChunk)
        assert chunk.error_message is None
        assert chunk.response_dict is not None
        assert chunk.response_dict["outputs"][0]["data"] == ["token"]
        assert chunk.response_size == len(data)

    def test_error_chunk(self) -> None:
        """Error streaming chunk should produce StreamChunk with error_message."""
        stream_resp = pb2.ModelStreamInferResponse()
        stream_resp.error_message = "Model crashed"
        data = stream_resp.SerializeToString()

        serializer = KServeV2GrpcSerializer()
        chunk = serializer.deserialize_stream_response(data)

        assert isinstance(chunk, StreamChunk)
        assert chunk.error_message == "Model crashed"
        assert chunk.response_dict is None
        assert chunk.response_size == len(data)


class TestDeserializeResponseRawOutputContents:
    """Tests for deserialize_response with raw_output_contents."""

    def test_raw_output_bytes_tensor(self) -> None:
        """Should correctly deserialize raw_output_contents BYTES tensor."""
        import struct

        response = pb2.ModelInferResponse()
        response.model_name = "m"
        output = response.outputs.add()
        output.name = "text_output"
        output.datatype = "BYTES"
        output.shape.append(1)
        text = b"Hello from raw"
        response.raw_output_contents.append(struct.pack("<I", len(text)) + text)
        data = response.SerializeToString()

        serializer = KServeV2GrpcSerializer()
        result_dict, size = serializer.deserialize_response(data)

        assert size == len(data)
        assert result_dict["outputs"][0]["data"] == ["Hello from raw"]

    def test_raw_output_int32_tensor(self) -> None:
        """Should correctly deserialize raw_output_contents INT32 tensor."""
        import struct

        response = pb2.ModelInferResponse()
        output = response.outputs.add()
        output.name = "count"
        output.datatype = "INT32"
        output.shape.append(2)
        response.raw_output_contents.append(struct.pack("<ii", 10, 20))
        data = response.SerializeToString()

        serializer = KServeV2GrpcSerializer()
        result_dict, _ = serializer.deserialize_response(data)

        assert result_dict["outputs"][0]["data"] == [10, 20]


class TestDeserializeStreamResponseEdgeCases:
    """Tests for edge cases in deserialize_stream_response."""

    def test_empty_infer_response_returns_empty_outputs(self) -> None:
        """Stream response with empty infer_response should return empty outputs."""
        stream_resp = pb2.ModelStreamInferResponse()
        # Set infer_response but with no outputs
        stream_resp.infer_response.model_name = "m"
        data = stream_resp.SerializeToString()

        serializer = KServeV2GrpcSerializer()
        chunk = serializer.deserialize_stream_response(data)

        assert chunk.error_message is None
        assert chunk.response_dict is not None
        assert chunk.response_dict["outputs"] == []


class TestSerializeRequestWithInputParameters:
    """Tests for serialize_request with input-level parameters."""

    def test_input_parameters_survive_roundtrip(self) -> None:
        """Input-level parameters should be preserved through serialization."""
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Hello"],
                    "parameters": {"binary_data_size": 512},
                }
            ],
        }
        serializer = KServeV2GrpcSerializer()
        data = serializer.serialize_request(payload, model_name="m")

        parsed = pb2.ModelInferRequest()
        parsed.ParseFromString(data)
        assert parsed.inputs[0].parameters["binary_data_size"].int64_param == 512


class TestRoundtripMultipleTypedTensors:
    """Tests for round-trip serialization of multiple tensor types."""

    def test_roundtrip_multiple_typed_tensors(self) -> None:
        """BYTES + INT32 + FP32 + INT64 tensors survive serialize -> deserialize."""
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["test prompt"],
                },
                {
                    "name": "max_tokens",
                    "shape": [1],
                    "datatype": "INT32",
                    "data": [128],
                },
                {
                    "name": "guidance_scale",
                    "shape": [1],
                    "datatype": "FP32",
                    "data": [7.5],
                },
                {
                    "name": "seed",
                    "shape": [1],
                    "datatype": "INT64",
                    "data": [42],
                },
            ]
        }
        serializer = KServeV2GrpcSerializer()
        data = serializer.serialize_request(payload, model_name="multi-type")

        parsed = pb2.ModelInferRequest()
        parsed.ParseFromString(data)

        assert parsed.model_name == "multi-type"
        assert len(parsed.inputs) == 4

        assert parsed.inputs[0].name == "text_input"
        assert parsed.inputs[0].datatype == "BYTES"
        assert parsed.inputs[0].contents.bytes_contents[0] == b"test prompt"

        assert parsed.inputs[1].name == "max_tokens"
        assert parsed.inputs[1].datatype == "INT32"
        assert parsed.inputs[1].contents.int_contents[0] == 128

        assert parsed.inputs[2].name == "guidance_scale"
        assert parsed.inputs[2].datatype == "FP32"
        assert abs(parsed.inputs[2].contents.fp32_contents[0] - 7.5) < 0.01

        assert parsed.inputs[3].name == "seed"
        assert parsed.inputs[3].datatype == "INT64"
        assert parsed.inputs[3].contents.int64_contents[0] == 42

    def test_roundtrip_fp32_tensor(self) -> None:
        """FP32 tensor value survives serialize -> response deserialize round-trip."""
        # Serialize a request with FP32
        payload = {
            "inputs": [
                {
                    "name": "embedding",
                    "shape": [3],
                    "datatype": "FP32",
                    "data": [0.1, 0.2, 0.3],
                }
            ]
        }
        serializer = KServeV2GrpcSerializer()
        data = serializer.serialize_request(payload, model_name="fp32-test")

        parsed = pb2.ModelInferRequest()
        parsed.ParseFromString(data)

        # Verify FP32 values are preserved
        assert len(parsed.inputs[0].contents.fp32_contents) == 3
        assert abs(parsed.inputs[0].contents.fp32_contents[0] - 0.1) < 0.001
        assert abs(parsed.inputs[0].contents.fp32_contents[1] - 0.2) < 0.001
        assert abs(parsed.inputs[0].contents.fp32_contents[2] - 0.3) < 0.001

    def test_roundtrip_int64_tensor(self) -> None:
        """INT64 tensor value survives serialize -> response deserialize round-trip."""
        # Build a response with INT64 output to test deserialization
        response = pb2.ModelInferResponse()
        response.model_name = "int64-test"
        output = response.outputs.add()
        output.name = "token_ids"
        output.datatype = "INT64"
        output.shape.append(3)
        output.contents.int64_contents.extend([100, 200, 300])
        data = response.SerializeToString()

        serializer = KServeV2GrpcSerializer()
        result_dict, size = serializer.deserialize_response(data)

        assert size == len(data)
        assert result_dict["outputs"][0]["name"] == "token_ids"
        assert result_dict["outputs"][0]["datatype"] == "INT64"
        assert result_dict["outputs"][0]["data"] == [100, 200, 300]
