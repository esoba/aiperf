# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path

from aiperf.common.mixins import AIPerfLoggerMixin


class BaseGenerator(AIPerfLoggerMixin, ABC):
    """Abstract base class for all data generators.

    Provides a consistent interface for generating synthetic data while allowing
    each generator type to use its own specific configuration and runtime parameters.

    Each class should create its own unique RNG in its __init__ method.

    When ``content_dir`` is provided, generators that support it will write files
    to disk and return HTTP URLs (via ``base_url``) instead of base64 data URIs.
    """

    def __init__(
        self,
        *,
        content_dir: Path | None = None,
        base_url: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._content_dir = content_dir
        self._base_url = base_url
        self._file_counter = 0

    @property
    def _writes_files(self) -> bool:
        """Whether this generator writes files to disk instead of encoding inline."""
        return self._content_dir is not None

    def _next_file_path(
        self, subdir: str, prefix: str, extension: str
    ) -> tuple[Path, str]:
        """Return ``(file_path, url)`` for the next file to write.

        Creates the parent directory if it does not exist.

        Args:
            subdir: Subdirectory under ``content_dir`` (e.g. ``"images"``).
            prefix: Filename prefix (e.g. ``"img"``).
            extension: File extension without dot (e.g. ``"png"``).

        Returns:
            A tuple of the local file path and the corresponding HTTP URL.
        """
        self._file_counter += 1
        filename = f"{prefix}_{self._file_counter:06d}.{extension}"
        path = self._content_dir / subdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self._base_url}/content/{subdir}/{filename}"
        return path, url

    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        """Generate synthetic data.

        Args:
            *args: Variable length argument list (subclass-specific)
            **kwargs: Arbitrary keyword arguments (subclass-specific)

        Returns:
            Generated data as a string (could be text, base64 encoded media, or URL).
        """
        pass
