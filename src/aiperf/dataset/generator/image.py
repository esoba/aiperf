# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
from pathlib import Path

from PIL import Image

from aiperf.common import random_generator as rng
from aiperf.common.config import ImageConfig
from aiperf.common.enums import ImageFormat
from aiperf.dataset import utils
from aiperf.dataset.generator.base import BaseGenerator
from aiperf.dataset.utils import IMAGE_SAVE_KWARGS


class ImageGenerator(BaseGenerator):
    """A class that generates images from source images.

    This class provides methods to create synthetic images by resizing
    source images (located in the 'assets/source_images' directory)
    to specified dimensions and converting them to a chosen image format (e.g., PNG, JPEG).
    The dimensions can be randomized based on mean and standard deviation values.

    When ``content_dir`` is set, images are saved to disk and HTTP URLs are
    returned instead of base64 data URIs.
    """

    def __init__(self, config: ImageConfig, **kwargs) -> None:
        super().__init__(**kwargs)

        # Separate RNGs for independent concerns
        self._dimensions_rng = rng.derive("dataset.image.dimensions")
        self._format_rng = rng.derive("dataset.image.format")
        self._source_rng = rng.derive("dataset.image.source")

        self.config = config

        # Pre-load source images into memory for fast sampling
        source_images_dir = Path(__file__).parent.resolve() / "assets" / "source_images"
        image_paths = sorted(glob.glob(str(source_images_dir / "*")))
        if not image_paths:
            raise ValueError(
                f"No source images found in '{source_images_dir}'. "
                "Please ensure the source_images directory contains at least one image file."
            )

        self._source_images = []
        for path in image_paths:
            with Image.open(path) as img:
                self._source_images.append(img.copy())
        self.debug(
            lambda: f"Pre-loaded {len(self._source_images)} source images into memory"
        )

    def generate(self, *args, **kwargs) -> str:
        """Generate an image with the configured parameters.

        Returns:
            An HTTP URL (when content server is enabled) or a base64 data URI.
        """
        image_format = self.config.format
        if image_format == ImageFormat.RANDOM:
            formats = [f for f in ImageFormat if f != ImageFormat.RANDOM]
            image_format = self._format_rng.choice(formats)

        width = self._dimensions_rng.sample_positive_normal_integer(
            self.config.width.mean, self.config.width.stddev
        )
        height = self._dimensions_rng.sample_positive_normal_integer(
            self.config.height.mean, self.config.height.stddev
        )

        self.logger.debug(
            "Generating image with width=%d, height=%d",
            width,
            height,
        )

        image = self._sample_source_image()
        image = image.resize(size=(width, height))

        if self._writes_files:
            return self._save_image_to_file(image, image_format)

        base64_image = utils.encode_image(image, image_format)
        return f"data:image/{image_format.name.lower()};base64,{base64_image}"

    def _save_image_to_file(self, image: Image, image_format: ImageFormat) -> str:
        """Save an image to the content directory and return its URL.

        Args:
            image: The PIL Image to save.
            image_format: Target image format.

        Returns:
            The HTTP URL where the content server will serve the file.
        """
        fmt_name = image_format.name.lower()
        path, url = self._next_file_path("images", "img", fmt_name)

        if image_format == ImageFormat.JPEG and image.mode != "RGB":
            image = image.convert("RGB")

        save_kwargs = IMAGE_SAVE_KWARGS.get(image_format.name, {})
        image.save(str(path), format=image_format.name, **save_kwargs)
        return url

    def _sample_source_image(self):
        """Sample one image among the pre-loaded source images.

        Returns:
            A PIL Image object randomly selected from the source images.
            Returns a copy to prevent accidental mutation of cached images.
        """
        return self._source_rng.choice(self._source_images).copy()
