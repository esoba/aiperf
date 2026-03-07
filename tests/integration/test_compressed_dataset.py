# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for compressed dataset support."""

import gzip
import lzma
import tarfile
import zipfile
from pathlib import Path

import orjson
import pytest
import zstandard

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults

SINGLE_TURN_PROMPTS = [
    {"text": "What is machine learning?"},
    {"text": "Explain neural networks."},
    {"text": "How does backpropagation work?"},
    {"text": "What are transformers?"},
    {"text": "Define reinforcement learning."},
]


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    with open(path, "wb") as f:
        for entry in entries:
            f.write(orjson.dumps(entry) + b"\n")


def _create_jsonl_gz(tmp_path: Path) -> Path:
    archive = tmp_path / "prompts.jsonl.gz"
    with gzip.open(archive, "wb") as f:
        for entry in SINGLE_TURN_PROMPTS:
            f.write(orjson.dumps(entry) + b"\n")
    return archive


def _create_jsonl_zst(tmp_path: Path) -> Path:
    archive = tmp_path / "prompts.jsonl.zst"
    compressor = zstandard.ZstdCompressor()
    content = b"".join(orjson.dumps(entry) + b"\n" for entry in SINGLE_TURN_PROMPTS)
    with open(archive, "wb") as f:
        f.write(compressor.compress(content))
    return archive


def _create_jsonl_xz(tmp_path: Path) -> Path:
    archive = tmp_path / "prompts.jsonl.xz"
    with lzma.open(archive, "wb") as f:
        for entry in SINGLE_TURN_PROMPTS:
            f.write(orjson.dumps(entry) + b"\n")
    return archive


def _create_zip(tmp_path: Path) -> Path:
    jsonl_file = tmp_path / "prompts.jsonl"
    _write_jsonl(jsonl_file, SINGLE_TURN_PROMPTS)
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(jsonl_file, arcname="prompts.jsonl")
    return archive


def _create_tar_gz(tmp_path: Path) -> Path:
    jsonl_file = tmp_path / "prompts.jsonl"
    _write_jsonl(jsonl_file, SINGLE_TURN_PROMPTS)
    archive = tmp_path / "dataset.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(jsonl_file, arcname="data/prompts.jsonl")
    return archive


def _create_tar_zst(tmp_path: Path) -> Path:
    jsonl_file = tmp_path / "prompts.jsonl"
    _write_jsonl(jsonl_file, SINGLE_TURN_PROMPTS)
    tar_path = tmp_path / "dataset.tar"
    with tarfile.open(tar_path, "w") as tf:
        tf.add(jsonl_file, arcname="prompts.jsonl")
    compressor = zstandard.ZstdCompressor()
    archive = tmp_path / "dataset.tar.zst"
    with open(tar_path, "rb") as src, open(archive, "wb") as dst:
        compressor.copy_stream(src, dst)
    return archive


def _create_zip_random_pool(tmp_path: Path) -> Path:
    pool_a = tmp_path / "pool_a.jsonl"
    pool_b = tmp_path / "pool_b.jsonl"
    _write_jsonl(pool_a, SINGLE_TURN_PROMPTS[:3])
    _write_jsonl(pool_b, SINGLE_TURN_PROMPTS[3:])
    archive = tmp_path / "pools.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(pool_a, arcname="pool_a.jsonl")
        zf.write(pool_b, arcname="pool_b.jsonl")
    return archive


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompressedDatasetSingleFile:
    """Tests for single-file compressed datasets (.gz, .zst, .xz)."""

    @pytest.mark.parametrize(
        "create_archive",
        [_create_jsonl_gz, _create_jsonl_zst, _create_jsonl_xz],
        ids=["gzip", "zstd", "xz"],
    )
    async def test_single_file_compression(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
        create_archive,
    ):
        """Test loading datasets from single-file compressed formats."""
        archive_path = create_archive(tmp_path)
        request_count = len(SINGLE_TURN_PROMPTS)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {archive_path} \
                --custom-dataset-type single_turn \
                --request-count {request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == request_count
        assert result.has_all_outputs


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompressedDatasetArchive:
    """Tests for multi-file archive datasets (.zip, .tar.gz, .tar.zst)."""

    async def test_zip_with_subpath(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test loading a dataset from a zip archive with --input-file-subpath."""
        archive_path = _create_zip(tmp_path)
        request_count = len(SINGLE_TURN_PROMPTS)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {archive_path} \
                --input-file-subpath prompts.jsonl \
                --custom-dataset-type single_turn \
                --request-count {request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == request_count
        assert result.has_all_outputs

    async def test_tar_gz_with_subpath(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test loading a dataset from a tar.gz archive with nested subpath."""
        archive_path = _create_tar_gz(tmp_path)
        request_count = len(SINGLE_TURN_PROMPTS)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {archive_path} \
                --input-file-subpath data/prompts.jsonl \
                --custom-dataset-type single_turn \
                --request-count {request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == request_count
        assert result.has_all_outputs

    async def test_tar_zst_with_subpath(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test loading a dataset from a tar.zst archive."""
        archive_path = _create_tar_zst(tmp_path)
        request_count = len(SINGLE_TURN_PROMPTS)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {archive_path} \
                --input-file-subpath prompts.jsonl \
                --custom-dataset-type single_turn \
                --request-count {request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == request_count
        assert result.has_all_outputs


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompressedDatasetRandomPool:
    """Tests for archive extraction as a directory for random_pool datasets."""

    async def test_zip_as_random_pool(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test extracting a zip archive as a directory for random_pool."""
        archive_path = _create_zip_random_pool(tmp_path)
        request_count = 10

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {archive_path} \
                --custom-dataset-type random_pool \
                --num-conversations {request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.has_all_outputs


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompressedDatasetAutoDetect:
    """Tests for auto-detection of dataset type from compressed files."""

    async def test_auto_detect_from_compressed_file(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test that dataset type is auto-detected after decompression."""
        archive_path = _create_jsonl_zst(tmp_path)
        request_count = len(SINGLE_TURN_PROMPTS)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {archive_path} \
                --request-count {request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == request_count
        assert result.has_all_outputs
