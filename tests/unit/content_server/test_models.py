# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for content server data models."""

from aiperf.content_server.models import (
    ContentRequestRecord,
    ContentServerStatus,
    RequestTrackerSnapshot,
)


class TestContentRequestRecord:
    def test_create_with_all_fields(self) -> None:
        record = ContentRequestRecord(
            timestamp_ns=1_000_000,
            method="GET",
            path="/content/images/test.png",
            query_string="w=100",
            http_version="1.1",
            client_host="10.0.0.5",
            client_port=9999,
            request_headers={"host": "localhost:8090", "user-agent": "test/1.0"},
            status_code=200,
            content_type="image/png",
            response_headers={"content-length": "1024", "etag": '"abc"'},
            body_bytes=1024,
            body_chunk_count=1,
            latency_ns=5000,
            time_to_first_byte_ns=1000,
            time_to_first_body_byte_ns=2000,
            transfer_duration_ns=3000,
            error=None,
        )
        assert record.timestamp_ns == 1_000_000
        assert record.method == "GET"
        assert record.path == "/content/images/test.png"
        assert record.query_string == "w=100"
        assert record.http_version == "1.1"
        assert record.client_host == "10.0.0.5"
        assert record.client_port == 9999
        assert record.request_headers["user-agent"] == "test/1.0"
        assert record.status_code == 200
        assert record.content_type == "image/png"
        assert record.response_headers["etag"] == '"abc"'
        assert record.body_bytes == 1024
        assert record.body_chunk_count == 1
        assert record.latency_ns == 5000
        assert record.time_to_first_byte_ns == 1000
        assert record.time_to_first_body_byte_ns == 2000
        assert record.transfer_duration_ns == 3000
        assert record.error is None

    def test_defaults(self) -> None:
        record = ContentRequestRecord(
            timestamp_ns=1,
            method="GET",
            path="/",
            status_code=404,
        )
        assert record.query_string == ""
        assert record.http_version == "1.1"
        assert record.client_host == ""
        assert record.client_port == 0
        assert record.request_headers == {}
        assert record.content_type == "application/octet-stream"
        assert record.response_headers == {}
        assert record.body_bytes == 0
        assert record.body_chunk_count == 0
        assert record.latency_ns == 0
        assert record.time_to_first_byte_ns == 0
        assert record.time_to_first_body_byte_ns == 0
        assert record.transfer_duration_ns == 0
        assert record.error is None

    def test_serialization_roundtrip(self) -> None:
        record = ContentRequestRecord(
            timestamp_ns=1_000,
            method="GET",
            path="test.txt",
            status_code=200,
            body_bytes=42,
            request_headers={"host": "localhost"},
            response_headers={"content-type": "text/plain"},
        )
        data = record.model_dump()
        restored = ContentRequestRecord.model_validate(data)
        assert restored.timestamp_ns == record.timestamp_ns
        assert restored.body_bytes == record.body_bytes
        assert restored.request_headers == record.request_headers
        assert restored.response_headers == record.response_headers

    def test_error_field(self) -> None:
        record = ContentRequestRecord(
            timestamp_ns=1,
            method="GET",
            path="/fail",
            status_code=500,
            error="RuntimeError: boom",
        )
        assert record.error == "RuntimeError: boom"


class TestContentServerStatus:
    def test_create_enabled(self) -> None:
        status = ContentServerStatus(
            enabled=True,
            base_url="http://0.0.0.0:8090",
            content_dir="/tmp/content",
        )
        assert status.enabled is True
        assert status.base_url == "http://0.0.0.0:8090"
        assert status.reason is None

    def test_create_disabled_with_reason(self) -> None:
        status = ContentServerStatus(
            enabled=False,
            reason="not initialized",
        )
        assert status.enabled is False
        assert status.reason == "not initialized"
        assert status.base_url == ""


class TestRequestTrackerSnapshot:
    def test_create_empty(self) -> None:
        snapshot = RequestTrackerSnapshot()
        assert snapshot.total_requests == 0
        assert snapshot.total_bytes_served == 0
        assert snapshot.records == []

    def test_create_with_records(self) -> None:
        records = [
            ContentRequestRecord(
                timestamp_ns=i,
                method="GET",
                path=f"file_{i}",
                status_code=200,
                body_bytes=100,
            )
            for i in range(3)
        ]
        snapshot = RequestTrackerSnapshot(
            total_requests=10,
            total_bytes_served=5000,
            records=records,
        )
        assert snapshot.total_requests == 10
        assert len(snapshot.records) == 3
