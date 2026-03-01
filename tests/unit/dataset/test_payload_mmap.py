# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for payload mmap zero-serialization verbatim replay."""

import orjson
import pytest

from aiperf.common.models import Conversation
from aiperf.common.models.dataset_models import Turn
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetClient,
    MemoryMapDatasetClientStore,
    PayloadIndex,
    PayloadOffset,
)


def _make_payload(model: str = "test", content: str = "hello") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": content}]}


def _make_conversation(
    session_id: str, payloads: list[dict | None] | None = None
) -> Conversation:
    """Create a conversation with optional raw_payload on each turn."""
    turns = []
    for payload in payloads or [None]:
        turns.append(
            Turn(
                role="user",
                raw_payload=payload,
            )
        )
    return Conversation(session_id=session_id, turns=turns)


class TestPayloadModels:
    """Test PayloadOffset and PayloadIndex models."""

    def test_payload_offset_creation(self) -> None:
        offset = PayloadOffset(offset=0, size=100)
        assert offset.offset == 0
        assert offset.size == 100

    def test_payload_offset_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            PayloadOffset(offset=-1, size=100)

    def test_payload_index_roundtrip(self) -> None:
        index = PayloadIndex(
            entries={
                "conv-1": {"0": PayloadOffset(offset=0, size=50)},
                "conv-2": {"1": PayloadOffset(offset=50, size=30)},
            },
            total_size=80,
        )
        json_bytes = index.model_dump_json().encode("utf-8")
        restored = PayloadIndex.model_validate_json(json_bytes)
        assert restored.entries["conv-1"]["0"].offset == 0
        assert restored.entries["conv-2"]["1"].size == 30
        assert restored.total_size == 80

    def test_payload_index_empty(self) -> None:
        index = PayloadIndex()
        assert index.entries == {}
        assert index.total_size == 0


class TestBackingStorePayloadWrite:
    """Test payload extraction and writing during add_conversation."""

    @pytest.mark.asyncio
    async def test_payload_extracted_and_written(self, tmp_path, monkeypatch) -> None:
        """Payloads are extracted from turns and written to payload mmap."""
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="payload-test")
        await store.initialize()

        payload = _make_payload()
        conv = _make_conversation("sess-1", [payload])
        await store.add_conversation("sess-1", conv)
        await store.finalize()

        assert store._has_payloads is True
        metadata = store.get_client_metadata()
        assert metadata.payload_data_file_path is not None
        assert metadata.payload_index_file_path is not None
        assert metadata.payload_data_file_path.exists()
        assert metadata.payload_index_file_path.exists()

    @pytest.mark.asyncio
    async def test_raw_payload_nulled_after_extraction(
        self, tmp_path, monkeypatch
    ) -> None:
        """raw_payload is set to None on the turn after extraction."""
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="null-test")
        await store.initialize()

        payload = _make_payload()
        conv = _make_conversation("sess-1", [payload])
        await store.add_conversation("sess-1", conv)

        assert conv.turns[0].raw_payload is None

    @pytest.mark.asyncio
    async def test_no_payload_files_without_raw_payload(
        self, tmp_path, monkeypatch
    ) -> None:
        """No payload files created when conversations have no raw_payload."""
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="no-payload")
        await store.initialize()

        conv = _make_conversation("sess-1", [None])
        await store.add_conversation("sess-1", conv)
        await store.finalize()

        assert store._has_payloads is False
        metadata = store.get_client_metadata()
        assert metadata.payload_data_file_path is None
        assert metadata.payload_index_file_path is None


class TestPayloadMmapRoundTrip:
    """Test full write-read cycle for payload mmap."""

    @pytest.mark.asyncio
    async def test_single_conversation_roundtrip(self, tmp_path, monkeypatch) -> None:
        """Write and read back a single payload."""
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="rt-single")
        await store.initialize()

        payload = _make_payload(content="roundtrip test")
        conv = _make_conversation("sess-1", [payload])
        await store.add_conversation("sess-1", conv)
        await store.finalize()

        metadata = store.get_client_metadata()
        client = MemoryMapDatasetClient(
            metadata.data_file_path,
            metadata.index_file_path,
            payload_data_file_path=metadata.payload_data_file_path,
            payload_index_file_path=metadata.payload_index_file_path,
        )

        result = client.get_payload_bytes("sess-1", 0)
        assert result is not None
        assert orjson.loads(result) == payload
        client.close()

    @pytest.mark.asyncio
    async def test_multi_turn_roundtrip(self, tmp_path, monkeypatch) -> None:
        """Write and read back payloads across multiple turns."""
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="rt-multi")
        await store.initialize()

        payloads = [
            _make_payload(content="turn 0"),
            _make_payload(content="turn 1"),
        ]
        conv = _make_conversation("sess-1", payloads)
        await store.add_conversation("sess-1", conv)
        await store.finalize()

        metadata = store.get_client_metadata()
        client = MemoryMapDatasetClient(
            metadata.data_file_path,
            metadata.index_file_path,
            payload_data_file_path=metadata.payload_data_file_path,
            payload_index_file_path=metadata.payload_index_file_path,
        )

        for i, expected in enumerate(payloads):
            result = client.get_payload_bytes("sess-1", i)
            assert result is not None
            assert orjson.loads(result) == expected

        client.close()

    @pytest.mark.asyncio
    async def test_multi_conversation_roundtrip(self, tmp_path, monkeypatch) -> None:
        """Write and read back payloads across multiple conversations."""
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="rt-multi-conv")
        await store.initialize()

        p1 = _make_payload(content="conv1")
        p2 = _make_payload(content="conv2")
        await store.add_conversation("c1", _make_conversation("c1", [p1]))
        await store.add_conversation("c2", _make_conversation("c2", [p2]))
        await store.finalize()

        metadata = store.get_client_metadata()
        client = MemoryMapDatasetClient(
            metadata.data_file_path,
            metadata.index_file_path,
            payload_data_file_path=metadata.payload_data_file_path,
            payload_index_file_path=metadata.payload_index_file_path,
        )

        assert orjson.loads(client.get_payload_bytes("c1", 0)) == p1
        assert orjson.loads(client.get_payload_bytes("c2", 0)) == p2
        client.close()

    @pytest.mark.asyncio
    async def test_missing_payload_returns_none(self, tmp_path, monkeypatch) -> None:
        """get_payload_bytes returns None for turns without payloads."""
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="rt-none")
        await store.initialize()

        conv = _make_conversation("sess-1", [_make_payload()])
        await store.add_conversation("sess-1", conv)
        await store.finalize()

        metadata = store.get_client_metadata()
        client = MemoryMapDatasetClient(
            metadata.data_file_path,
            metadata.index_file_path,
            payload_data_file_path=metadata.payload_data_file_path,
            payload_index_file_path=metadata.payload_index_file_path,
        )

        assert client.get_payload_bytes("sess-1", 99) is None
        assert client.get_payload_bytes("nonexistent", 0) is None
        client.close()

    @pytest.mark.asyncio
    async def test_no_payload_mmap_returns_none(self, tmp_path, monkeypatch) -> None:
        """Client without payload mmap returns None."""
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="rt-no-payload")
        await store.initialize()

        conv = _make_conversation("sess-1", [None])
        await store.add_conversation("sess-1", conv)
        await store.finalize()

        metadata = store.get_client_metadata()
        client = MemoryMapDatasetClient(
            metadata.data_file_path,
            metadata.index_file_path,
        )

        assert client.get_payload_bytes("sess-1", 0) is None
        client.close()

    @pytest.mark.asyncio
    async def test_mixed_payload_and_no_payload_turns(
        self, tmp_path, monkeypatch
    ) -> None:
        """Only turns with raw_payload are stored; others return None."""
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="rt-mixed")
        await store.initialize()

        payload = _make_payload(content="only this one")
        conv = _make_conversation("sess-1", [payload, None])
        await store.add_conversation("sess-1", conv)
        await store.finalize()

        metadata = store.get_client_metadata()
        client = MemoryMapDatasetClient(
            metadata.data_file_path,
            metadata.index_file_path,
            payload_data_file_path=metadata.payload_data_file_path,
            payload_index_file_path=metadata.payload_index_file_path,
        )

        assert orjson.loads(client.get_payload_bytes("sess-1", 0)) == payload
        assert client.get_payload_bytes("sess-1", 1) is None
        client.close()


class TestClientStorePayloadBytes:
    """Test async wrapper for get_payload_bytes."""

    @pytest.mark.asyncio
    async def test_get_payload_bytes_async(self, tmp_path, monkeypatch) -> None:
        """MemoryMapDatasetClientStore.get_payload_bytes works end-to-end."""
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="async-test")
        await store.initialize()

        payload = _make_payload(content="async roundtrip")
        conv = _make_conversation("sess-1", [payload])
        await store.add_conversation("sess-1", conv)
        await store.finalize()

        metadata = store.get_client_metadata()
        client_store = MemoryMapDatasetClientStore(client_metadata=metadata)
        await client_store.initialize()

        result = await client_store.get_payload_bytes("sess-1", 0)
        assert result is not None
        assert orjson.loads(result) == payload

        assert await client_store.get_payload_bytes("sess-1", 99) is None
        await client_store.stop()

    @pytest.mark.asyncio
    async def test_get_payload_bytes_returns_none_when_not_initialized(self) -> None:
        """Returns None when client is not initialized."""
        from pathlib import Path

        from aiperf.common.models import MemoryMapClientMetadata

        metadata = MemoryMapClientMetadata(
            data_file_path=Path("/nonexistent/data.dat"),
            index_file_path=Path("/nonexistent/index.dat"),
        )
        client_store = MemoryMapDatasetClientStore(client_metadata=metadata)

        result = await client_store.get_payload_bytes("sess-1", 0)
        assert result is None
