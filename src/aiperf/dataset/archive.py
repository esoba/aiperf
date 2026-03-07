# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import lzma
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import zstandard

SINGLE_FILE_EXTENSIONS = {".gz", ".zst", ".xz"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tar.gz", ".tgz", ".tar.zst", ".tar.xz"}
ALL_COMPRESSED_EXTENSIONS = SINGLE_FILE_EXTENSIONS | ARCHIVE_EXTENSIONS

# Pre-sorted longest-first to ensure .tar.gz matches before .gz
_ARCHIVE_EXTS_SORTED = sorted(ARCHIVE_EXTENSIONS, key=len, reverse=True)
_ALL_EXTS_SORTED = sorted(ALL_COMPRESSED_EXTENSIONS, key=len, reverse=True)


def is_compressed_file(path: Path) -> bool:
    """Check if the path has a recognized compressed file extension."""
    return _get_compression_type(path) is not None


def _get_compression_type(path: Path) -> str | None:
    """Return the compression type string for a path, or None if not compressed."""
    filename = path.name.lower()
    for extension in _ARCHIVE_EXTS_SORTED:
        if filename.endswith(extension):
            return extension
    for extension in SINGLE_FILE_EXTENSIONS:
        if filename.endswith(extension):
            return extension
    return None


def _strip_compression_extension(filename: str) -> str:
    """Strip the compression extension from a filename to get the inner name."""
    filename_lower = filename.lower()
    for extension in _ALL_EXTS_SORTED:
        if filename_lower.endswith(extension):
            return filename[: -len(extension)]
    return filename


def extract_compressed_file(
    source_path: Path, subpath: str | None = None
) -> tuple[Path, Path]:
    """Extract a compressed file to a temporary directory.

    For single-file compression (.gz, .zst, .xz), the file is decompressed
    with the compression extension stripped from the name.

    For multi-file archives (.zip, .tar, .tar.gz, .tgz, .tar.zst, .tar.xz),
    files are extracted to a temp directory. If subpath is specified,
    returns the path to that specific file within the extracted contents.
    If subpath is not specified for a multi-file archive, returns the
    temp directory itself (useful for random_pool datasets).

    Args:
        source_path: Path to the compressed file.
        subpath: For archives, the relative path to the desired file inside
            the archive. Required for multi-file archives when a single file
            is needed.

    Returns:
        A tuple of (extracted_path, output_dir) where extracted_path is the path
        to the extracted file or directory, and output_dir is the temporary
        directory that should be cleaned up after use.

    Raises:
        FileNotFoundError: If the archive or subpath doesn't exist.
        ValueError: If the compression format is not recognized.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"The file '{source_path}' does not exist.")

    compression_type = _get_compression_type(source_path)
    if compression_type is None:
        raise ValueError(f"Unrecognized compression format for '{source_path}'")

    output_dir = Path(tempfile.mkdtemp(prefix="aiperf_dataset_"))

    try:
        if compression_type in SINGLE_FILE_EXTENSIONS:
            return _decompress_single_file(
                source_path, compression_type, output_dir, subpath
            )
        return _extract_archive(source_path, compression_type, output_dir, subpath)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def _decompress_single_file(
    source_path: Path,
    compression_type: str,
    output_dir: Path,
    subpath: str | None,
) -> tuple[Path, Path]:
    """Decompress a single compressed file (.gz, .zst, .xz)."""
    output_name = subpath or _strip_compression_extension(source_path.name)
    output_path = output_dir / output_name

    if compression_type == ".gz":
        with gzip.open(source_path, "rb") as src, open(output_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    elif compression_type == ".zst":
        decompressor = zstandard.ZstdDecompressor()
        with open(source_path, "rb") as src, open(output_path, "wb") as dst:
            decompressor.copy_stream(src, dst)
    elif compression_type == ".xz":
        with lzma.open(source_path, "rb") as src, open(output_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    return output_path, output_dir


def _extract_archive(
    source_path: Path,
    compression_type: str,
    output_dir: Path,
    subpath: str | None,
) -> tuple[Path, Path]:
    """Extract a multi-file archive (.zip, .tar, .tar.gz, .tgz, .tar.zst, .tar.xz)."""
    if compression_type == ".zip":
        _extract_zip(source_path, output_dir)
    elif compression_type == ".tar":
        _extract_tar(source_path, output_dir)
    elif compression_type in (".tar.gz", ".tgz"):
        _extract_tar(source_path, output_dir, mode="r:gz")
    elif compression_type == ".tar.xz":
        _extract_tar(source_path, output_dir, mode="r:xz")
    elif compression_type == ".tar.zst":
        _extract_tar_zst(source_path, output_dir)

    if subpath:
        extracted_path = output_dir / subpath
        if not extracted_path.exists():
            raise FileNotFoundError(
                f"Subpath '{subpath}' not found in archive '{source_path}'. "
                f"Use --input-file-subpath to specify the correct path."
            )
        return extracted_path, output_dir

    return output_dir, output_dir


def _validate_zip_entry(output_dir: Path, member_name: str) -> None:
    """Validate a zip entry path to prevent path traversal (zip slip)."""
    resolved_target = (output_dir / member_name).resolve()
    if not resolved_target.is_relative_to(output_dir.resolve()):
        raise ValueError(
            f"Zip entry '{member_name}' would extract outside target directory"
        )


def _extract_zip(source_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(source_path, "r") as archive:
        for member_name in archive.namelist():
            _validate_zip_entry(output_dir, member_name)
        archive.extractall(output_dir)


def _safe_tar_extractall(tar: tarfile.TarFile, output_dir: Path) -> None:
    """Extract all members from a tarfile safely."""
    try:
        # Python 3.12+ added the filter argument to prevent path traversal and unsafe metadata.
        # Python 3.14+ requires the filter argument.
        tar.extractall(output_dir, filter="data")
    except TypeError:
        # Fall back for Python 3.10/3.11 that don't have the filter argument.
        tar.extractall(output_dir)


def _extract_tar(source_path: Path, output_dir: Path, mode: str = "r") -> None:
    with tarfile.open(source_path, mode) as tar:
        _safe_tar_extractall(tar, output_dir)


def _extract_tar_zst(source_path: Path, output_dir: Path) -> None:
    decompressor = zstandard.ZstdDecompressor()
    with (
        open(source_path, "rb") as compressed_stream,
        decompressor.stream_reader(compressed_stream) as tar_stream,
        tarfile.open(fileobj=tar_stream, mode="r|") as tar,
    ):
        _safe_tar_extractall(tar, output_dir)


def cleanup_temp_dir(temp_dir: Path) -> None:
    """Remove a temporary directory created during extraction."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
