# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for content server tests."""

from pathlib import Path

import pytest


@pytest.fixture
def content_dir(tmp_path: Path) -> Path:
    """Create a temporary content directory with sample files."""
    # Create subdirectory structure
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Create sample files
    (images_dir / "test.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    (images_dir / "photo.jpeg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
    (audio_dir / "clip.wav").write_bytes(b"RIFF" + b"\x00" * 80)
    (tmp_path / "readme.txt").write_text("test content")

    return tmp_path
