# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import lzma
import tarfile
import zipfile
from pathlib import Path

import pytest
import zstandard

from aiperf.dataset.archive import (
    _get_compression_type,
    _strip_compression_extension,
    cleanup_temp_dir,
    extract_compressed_file,
    is_compressed_file,
)

SAMPLE_CONTENT = b'{"text": "Hello world"}\n{"text": "Second line"}\n'


# ============================================================================
# is_compressed_file tests
# ============================================================================


class TestIsCompressedFile:
    @pytest.mark.parametrize(
        "filename",
        [
            "data.jsonl.gz",
            "data.jsonl.zst",
            "data.jsonl.xz",
            "data.zip",
            "data.tar",
            "data.tar.gz",
            "data.tgz",
            "data.tar.zst",
            "data.tar.xz",
        ],
    )
    def test_compressed_extensions_detected(self, filename: str) -> None:
        assert is_compressed_file(Path(filename)) is True

    @pytest.mark.parametrize(
        "filename",
        ["data.jsonl", "data.txt", "data.csv", "data.json"],
    )
    def test_non_compressed_extensions_not_detected(self, filename: str) -> None:
        assert is_compressed_file(Path(filename)) is False

    @pytest.mark.parametrize(
        "filename",
        ["DATA.JSONL.GZ", "Data.Tar.Gz", "ARCHIVE.ZIP", "data.JSONL.ZST"],
    )
    def test_case_insensitive_detection(self, filename: str) -> None:
        assert is_compressed_file(Path(filename)) is True


# ============================================================================
# _get_compression_type tests
# ============================================================================


class TestGetCompressionType:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("data.jsonl.gz", ".gz"),
            ("data.jsonl.zst", ".zst"),
            ("data.jsonl.xz", ".xz"),
            ("data.zip", ".zip"),
            ("data.tar", ".tar"),
            ("data.tar.gz", ".tar.gz"),
            ("data.tgz", ".tgz"),
            ("data.tar.zst", ".tar.zst"),
            ("data.tar.xz", ".tar.xz"),
        ],
    )
    def test_returns_correct_type(self, filename: str, expected: str) -> None:
        assert _get_compression_type(Path(filename)) == expected

    def test_tar_gz_matches_before_gz(self) -> None:
        assert _get_compression_type(Path("data.tar.gz")) == ".tar.gz"

    def test_returns_none_for_unrecognized(self) -> None:
        assert _get_compression_type(Path("data.jsonl")) is None


# ============================================================================
# _strip_compression_extension tests
# ============================================================================


class TestStripCompressionExtension:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("data.jsonl.gz", "data.jsonl"),
            ("data.jsonl.zst", "data.jsonl"),
            ("data.jsonl.xz", "data.jsonl"),
            ("archive.tar.gz", "archive"),
            ("archive.tgz", "archive"),
            ("archive.tar.zst", "archive"),
            ("archive.tar.xz", "archive"),
            ("archive.tar", "archive"),
            ("archive.zip", "archive"),
            ("data.jsonl", "data.jsonl"),
            ("no_extension", "no_extension"),
        ],
    )
    def test_strips_extension(self, filename: str, expected: str) -> None:
        assert _strip_compression_extension(filename) == expected

    def test_case_insensitive_stripping(self) -> None:
        assert _strip_compression_extension("DATA.JSONL.GZ") == "DATA.JSONL"


# ============================================================================
# extract_compressed_file tests - single file compression
# ============================================================================


class TestExtractSingleFileCompression:
    def test_extract_gzip(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.jsonl.gz"
        with gzip.open(archive, "wb") as f:
            f.write(SAMPLE_CONTENT)

        extracted, temp_dir = extract_compressed_file(archive)
        try:
            assert extracted.name == "data.jsonl"
            assert extracted.read_bytes() == SAMPLE_CONTENT
        finally:
            cleanup_temp_dir(temp_dir)

    def test_extract_zstd(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.jsonl.zst"
        cctx = zstandard.ZstdCompressor()
        with open(archive, "wb") as f:
            f.write(cctx.compress(SAMPLE_CONTENT))

        extracted, temp_dir = extract_compressed_file(archive)
        try:
            assert extracted.name == "data.jsonl"
            assert extracted.read_bytes() == SAMPLE_CONTENT
        finally:
            cleanup_temp_dir(temp_dir)

    def test_extract_xz(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.jsonl.xz"
        with lzma.open(archive, "wb") as f:
            f.write(SAMPLE_CONTENT)

        extracted, temp_dir = extract_compressed_file(archive)
        try:
            assert extracted.name == "data.jsonl"
            assert extracted.read_bytes() == SAMPLE_CONTENT
        finally:
            cleanup_temp_dir(temp_dir)

    def test_extract_single_file_with_inner_path(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.jsonl.gz"
        with gzip.open(archive, "wb") as f:
            f.write(SAMPLE_CONTENT)

        extracted, temp_dir = extract_compressed_file(archive, subpath="custom.jsonl")
        try:
            assert extracted.name == "custom.jsonl"
            assert extracted.read_bytes() == SAMPLE_CONTENT
        finally:
            cleanup_temp_dir(temp_dir)


# ============================================================================
# extract_compressed_file tests - multi-file archives
# ============================================================================


class TestExtractArchive:
    def test_extract_zip(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("prompts.jsonl", SAMPLE_CONTENT)

        extracted, temp_dir = extract_compressed_file(archive, subpath="prompts.jsonl")
        try:
            assert extracted.name == "prompts.jsonl"
            assert extracted.read_bytes() == SAMPLE_CONTENT
        finally:
            cleanup_temp_dir(temp_dir)

    def test_extract_zip_returns_dir_without_inner_path(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("prompts.jsonl", SAMPLE_CONTENT)

        extracted, temp_dir = extract_compressed_file(archive)
        try:
            assert extracted.is_dir()
            assert (extracted / "prompts.jsonl").read_bytes() == SAMPLE_CONTENT
        finally:
            cleanup_temp_dir(temp_dir)

    def test_extract_tar_gz(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.tar.gz"
        inner_file = tmp_path / "prompts.jsonl"
        inner_file.write_bytes(SAMPLE_CONTENT)

        with tarfile.open(archive, "w:gz") as tf:
            tf.add(inner_file, arcname="prompts.jsonl")

        extracted, temp_dir = extract_compressed_file(archive, subpath="prompts.jsonl")
        try:
            assert extracted.read_bytes() == SAMPLE_CONTENT
        finally:
            cleanup_temp_dir(temp_dir)

    def test_extract_tar_zst(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.tar.zst"
        inner_file = tmp_path / "prompts.jsonl"
        inner_file.write_bytes(SAMPLE_CONTENT)

        # Create tar, then compress with zstd
        tar_path = tmp_path / "data.tar"
        with tarfile.open(tar_path, "w") as tf:
            tf.add(inner_file, arcname="prompts.jsonl")

        cctx = zstandard.ZstdCompressor()
        with open(tar_path, "rb") as f_in, open(archive, "wb") as f_out:
            cctx.copy_stream(f_in, f_out)

        extracted, temp_dir = extract_compressed_file(archive, subpath="prompts.jsonl")
        try:
            assert extracted.read_bytes() == SAMPLE_CONTENT
        finally:
            cleanup_temp_dir(temp_dir)

    def test_extract_tar_xz(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.tar.xz"
        inner_file = tmp_path / "prompts.jsonl"
        inner_file.write_bytes(SAMPLE_CONTENT)

        with tarfile.open(archive, "w:xz") as tf:
            tf.add(inner_file, arcname="prompts.jsonl")

        extracted, temp_dir = extract_compressed_file(archive, subpath="prompts.jsonl")
        try:
            assert extracted.read_bytes() == SAMPLE_CONTENT
        finally:
            cleanup_temp_dir(temp_dir)

    def test_extract_plain_tar(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.tar"
        inner_file = tmp_path / "prompts.jsonl"
        inner_file.write_bytes(SAMPLE_CONTENT)

        with tarfile.open(archive, "w") as tf:
            tf.add(inner_file, arcname="prompts.jsonl")

        extracted, temp_dir = extract_compressed_file(archive, subpath="prompts.jsonl")
        try:
            assert extracted.read_bytes() == SAMPLE_CONTENT
        finally:
            cleanup_temp_dir(temp_dir)

    def test_extract_tgz(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.tgz"
        inner_file = tmp_path / "prompts.jsonl"
        inner_file.write_bytes(SAMPLE_CONTENT)

        with tarfile.open(archive, "w:gz") as tf:
            tf.add(inner_file, arcname="prompts.jsonl")

        extracted, temp_dir = extract_compressed_file(archive, subpath="prompts.jsonl")
        try:
            assert extracted.read_bytes() == SAMPLE_CONTENT
        finally:
            cleanup_temp_dir(temp_dir)


# ============================================================================
# Error handling tests
# ============================================================================


class TestErrorHandling:
    def test_extract_nonexistent_file_raises_error(self) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            extract_compressed_file(Path("/nonexistent/data.jsonl.gz"))

    def test_extract_unrecognized_format_raises_error(self, tmp_path: Path) -> None:
        plain_file = tmp_path / "data.jsonl"
        plain_file.write_bytes(SAMPLE_CONTENT)

        with pytest.raises(ValueError, match="Unrecognized compression format"):
            extract_compressed_file(plain_file)

    def test_extract_zip_path_traversal_raises_error(self, tmp_path: Path) -> None:
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../../../etc/passwd", b"malicious")

        with pytest.raises(ValueError, match="outside target directory"):
            extract_compressed_file(archive)

    def test_extract_cleans_up_temp_dir_on_failure(self, tmp_path: Path) -> None:
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../../../etc/passwd", b"malicious")

        with pytest.raises(ValueError):
            extract_compressed_file(archive)

        # The temp dir created during extraction should have been cleaned up.
        # We verify by checking no aiperf_dataset_ dirs remain in the system temp.
        import tempfile

        temp_root = Path(tempfile.gettempdir())
        leftover = [
            d
            for d in temp_root.iterdir()
            if d.name.startswith("aiperf_dataset_") and d.is_dir()
        ]
        assert len(leftover) == 0

    def test_extract_archive_missing_inner_path_raises_error(
        self, tmp_path: Path
    ) -> None:
        archive = tmp_path / "data.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("prompts.jsonl", SAMPLE_CONTENT)

        with pytest.raises(FileNotFoundError, match="Subpath.*not found"):
            extract_compressed_file(archive, subpath="nonexistent.jsonl")


# ============================================================================
# cleanup_temp_dir tests
# ============================================================================


class TestCleanupTempDir:
    def test_cleanup_removes_directory(self, tmp_path: Path) -> None:
        temp_dir = tmp_path / "to_clean"
        temp_dir.mkdir()
        (temp_dir / "file.txt").write_text("test")

        cleanup_temp_dir(temp_dir)
        assert not temp_dir.exists()

    def test_cleanup_nonexistent_dir_does_not_raise(self) -> None:
        cleanup_temp_dir(Path("/nonexistent/dir"))
