# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for content server integration with generators, composers, and DatasetManager.

When the content server is enabled, image and video generators write files to disk
and return HTTP URLs instead of base64 data URIs. Audio is excluded (requires inline data).
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from aiperf.common.config import (
    ImageConfig,
    ImageHeightConfig,
    ImageWidthConfig,
    VideoConfig,
)
from aiperf.common.enums import ImageFormat, VideoFormat, VideoSynthType
from aiperf.dataset.generator.base import BaseGenerator
from aiperf.dataset.generator.image import ImageGenerator
from aiperf.dataset.generator.video import VideoGenerator

# ============================================================================
# BaseGenerator
# ============================================================================


class ConcreteGenerator(BaseGenerator):
    """Minimal concrete implementation for testing BaseGenerator."""

    def generate(self, *args, **kwargs) -> str:
        return ""


class TestBaseGeneratorContentDir:
    def test_writes_files_false_by_default(self) -> None:
        gen = ConcreteGenerator()
        assert gen._writes_files is False

    def test_writes_files_true_with_content_dir(self, tmp_path: Path) -> None:
        gen = ConcreteGenerator(content_dir=tmp_path, base_url="http://localhost:8090")
        assert gen._writes_files is True

    def test_next_file_path_creates_directory_and_increments(
        self, tmp_path: Path
    ) -> None:
        gen = ConcreteGenerator(content_dir=tmp_path, base_url="http://host:9000")

        path1, url1 = gen._next_file_path("images", "img", "png")
        path2, url2 = gen._next_file_path("images", "img", "jpeg")

        assert path1 == tmp_path / "images" / "img_000001.png"
        assert url1 == "http://host:9000/content/images/img_000001.png"
        assert path2 == tmp_path / "images" / "img_000002.jpeg"
        assert url2 == "http://host:9000/content/images/img_000002.jpeg"
        assert (tmp_path / "images").is_dir()

    def test_next_file_path_different_subdirs(self, tmp_path: Path) -> None:
        gen = ConcreteGenerator(content_dir=tmp_path, base_url="http://h:80")

        path, url = gen._next_file_path("video", "vid", "mp4")
        assert path == tmp_path / "video" / "vid_000001.mp4"
        assert url == "http://h:80/content/video/vid_000001.mp4"
        assert (tmp_path / "video").is_dir()


# ============================================================================
# ImageGenerator — file-writing mode
# ============================================================================


class TestImageGeneratorFileWriting:
    @pytest.fixture
    def config(self) -> ImageConfig:
        return ImageConfig(
            width=ImageWidthConfig(mean=10, stddev=0),
            height=ImageHeightConfig(mean=10, stddev=0),
            format=ImageFormat.PNG,
        )

    @pytest.fixture
    def test_image(self) -> Image.Image:
        return Image.new("RGB", (5, 5), color="blue")

    @patch.object(ImageGenerator, "_sample_source_image")
    def test_generate_writes_file_and_returns_url(
        self, mock_sample, config: ImageConfig, test_image: Image.Image, tmp_path: Path
    ) -> None:
        mock_sample.return_value = test_image

        gen = ImageGenerator(
            config, content_dir=tmp_path, base_url="http://localhost:8090"
        )
        result = gen.generate()

        assert result == "http://localhost:8090/content/images/img_000001.png"
        written = tmp_path / "images" / "img_000001.png"
        assert written.exists()
        # Verify it's a valid image
        img = Image.open(written)
        assert img.size == (10, 10)

    @patch.object(ImageGenerator, "_sample_source_image")
    def test_generate_without_content_dir_returns_base64(
        self, mock_sample, config: ImageConfig, test_image: Image.Image
    ) -> None:
        mock_sample.return_value = test_image

        gen = ImageGenerator(config)
        result = gen.generate()

        assert result.startswith("data:image/png;base64,")

    @patch.object(ImageGenerator, "_sample_source_image")
    def test_generate_increments_counter(
        self, mock_sample, config: ImageConfig, test_image: Image.Image, tmp_path: Path
    ) -> None:
        mock_sample.return_value = test_image

        gen = ImageGenerator(config, content_dir=tmp_path, base_url="http://host:80")
        url1 = gen.generate()
        url2 = gen.generate()

        assert "img_000001" in url1
        assert "img_000002" in url2
        assert (tmp_path / "images" / "img_000001.png").exists()
        assert (tmp_path / "images" / "img_000002.png").exists()

    @pytest.mark.parametrize(
        "image_format,ext",
        [
            (ImageFormat.PNG, "png"),
            (ImageFormat.JPEG, "jpeg"),
        ],
    )
    @patch.object(ImageGenerator, "_sample_source_image")
    def test_generate_file_format_extension(
        self,
        mock_sample,
        image_format: ImageFormat,
        ext: str,
        test_image: Image.Image,
        tmp_path: Path,
    ) -> None:
        mock_sample.return_value = test_image

        config = ImageConfig(
            width=ImageWidthConfig(mean=10, stddev=0),
            height=ImageHeightConfig(mean=10, stddev=0),
            format=image_format,
        )
        gen = ImageGenerator(config, content_dir=tmp_path, base_url="http://host:80")
        url = gen.generate()

        assert url.endswith(f".{ext}")
        assert (tmp_path / "images" / f"img_000001.{ext}").exists()


# ============================================================================
# VideoGenerator — file-writing mode
# ============================================================================


class TestVideoGeneratorFileWriting:
    @pytest.fixture
    def config(self) -> VideoConfig:
        return VideoConfig(
            width=32,
            height=32,
            duration=0.5,
            fps=2,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=VideoSynthType.MOVING_SHAPES,
        )

    def test_generate_disabled_returns_empty_even_with_content_dir(
        self, tmp_path: Path
    ) -> None:
        config = VideoConfig(
            width=None,
            height=None,
            duration=1.0,
            fps=4,
            format=VideoFormat.WEBM,
            codec="libvpx-vp9",
            synth_type=VideoSynthType.MOVING_SHAPES,
        )
        gen = VideoGenerator(config, content_dir=tmp_path, base_url="http://host:80")
        assert gen.generate() == ""

    @patch("aiperf.dataset.generator.video.ffmpeg")
    @patch.object(VideoGenerator, "_check_ffmpeg_availability", return_value=True)
    def test_write_video_to_file_calls_ffmpeg(
        self, mock_ffmpeg_check, mock_ffmpeg, config: VideoConfig, tmp_path: Path
    ) -> None:
        gen = VideoGenerator(
            config, content_dir=tmp_path, base_url="http://localhost:8090"
        )
        frames = [Image.new("RGB", (32, 32), (255, 0, 0))]

        # Setup mock ffmpeg chain
        mock_input = Mock()
        mock_output = Mock()
        mock_ffmpeg.input.return_value = mock_input
        mock_input.output.return_value = mock_output
        mock_output.overwrite_output.return_value = mock_output
        mock_output.run.return_value = (b"", b"")

        url = gen._write_video_to_file(frames)

        assert url == "http://localhost:8090/content/video/vid_000001.webm"
        # Verify ffmpeg was called with file path (not pipe:)
        output_call = mock_input.output.call_args
        assert str(tmp_path / "video" / "vid_000001.webm") == output_call[0][0]

    @patch.object(VideoGenerator, "_check_ffmpeg_availability", return_value=False)
    def test_write_video_to_file_raises_when_no_ffmpeg(
        self, mock_check, config: VideoConfig, tmp_path: Path
    ) -> None:
        gen = VideoGenerator(config, content_dir=tmp_path, base_url="http://host:80")
        frames = [Image.new("RGB", (32, 32), (0, 0, 0))]

        with pytest.raises(RuntimeError, match="FFmpeg binary not found"):
            gen._write_video_to_file(frames)

    @patch("aiperf.dataset.generator.video.ffmpeg")
    @patch.object(VideoGenerator, "_check_ffmpeg_availability", return_value=True)
    def test_generate_uses_write_video_when_content_dir_set(
        self, mock_ffmpeg_check, mock_ffmpeg, config: VideoConfig, tmp_path: Path
    ) -> None:
        gen = VideoGenerator(config, content_dir=tmp_path, base_url="http://host:80")

        mock_input = Mock()
        mock_output = Mock()
        mock_ffmpeg.input.return_value = mock_input
        mock_input.output.return_value = mock_output
        mock_output.overwrite_output.return_value = mock_output
        mock_output.run.return_value = (b"", b"")

        result = gen.generate()

        assert result.startswith("http://host:80/content/video/vid_")

    def test_generate_without_content_dir_returns_base64(
        self, config: VideoConfig
    ) -> None:
        gen = VideoGenerator(config)
        # Mock the encoding path to avoid needing ffmpeg
        with patch.object(
            gen, "_encode_frames_to_base64", return_value="data:video/webm;base64,abc"
        ):
            result = gen.generate()
        assert result.startswith("data:video/")


# ============================================================================
# Composer forwarding
# ============================================================================


class TestComposerContentDirForwarding:
    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def base_config(self) -> dict:
        return {
            "endpoint": {"model_names": ["test-model"]},
            "input": {
                "conversation": {"num": 1, "turn": {"mean": 1}},
                "prompt": {"input_tokens": {"mean": 100}},
            },
        }

    def test_generators_receive_content_kwargs(
        self, base_config: dict, mock_tokenizer: MagicMock, tmp_path: Path
    ) -> None:
        from aiperf.common.config import UserConfig
        from aiperf.dataset.composer.base import BaseDatasetComposer

        class TestComposer(BaseDatasetComposer):
            def create_dataset(self):
                return []

        config = UserConfig(**base_config)
        composer = TestComposer(
            config, mock_tokenizer, content_dir=tmp_path, base_url="http://h:80"
        )

        assert composer.image_generator._content_dir == tmp_path
        assert composer.image_generator._base_url == "http://h:80"
        assert composer.video_generator._content_dir == tmp_path
        assert composer.video_generator._base_url == "http://h:80"
        # Audio should NOT have content_dir
        assert composer.audio_generator._content_dir is None

    def test_generators_default_no_content_dir(
        self, base_config: dict, mock_tokenizer: MagicMock
    ) -> None:
        from aiperf.common.config import UserConfig
        from aiperf.dataset.composer.base import BaseDatasetComposer

        class TestComposer(BaseDatasetComposer):
            def create_dataset(self):
                return []

        config = UserConfig(**base_config)
        composer = TestComposer(config, mock_tokenizer)

        assert composer.image_generator._content_dir is None
        assert composer.video_generator._content_dir is None


# ============================================================================
# DatasetManager._get_content_server_kwargs
# ============================================================================


class TestDatasetManagerContentServerKwargs:
    @pytest.fixture
    def dataset_manager(self):
        from aiperf.common.config import ServiceConfig, UserConfig
        from aiperf.dataset.dataset_manager import DatasetManager

        user_config = UserConfig(
            endpoint={"model_names": ["test-model"]},
            input={},
        )
        return DatasetManager(ServiceConfig(), user_config)

    def test_returns_empty_when_disabled(self, dataset_manager) -> None:
        with patch(
            "aiperf.dataset.dataset_manager.Environment.CONTENT_SERVER"
        ) as mock_settings:
            mock_settings.ENABLED = False
            mock_settings.CONTENT_DIR = "/some/dir"
            result = dataset_manager._get_content_server_kwargs()
        assert result == {}

    def test_returns_empty_when_content_dir_empty(self, dataset_manager) -> None:
        with patch(
            "aiperf.dataset.dataset_manager.Environment.CONTENT_SERVER"
        ) as mock_settings:
            mock_settings.ENABLED = True
            mock_settings.CONTENT_DIR = ""
            result = dataset_manager._get_content_server_kwargs()
        assert result == {}

    def test_returns_kwargs_when_enabled(self, dataset_manager) -> None:
        with patch(
            "aiperf.dataset.dataset_manager.Environment.CONTENT_SERVER"
        ) as mock_settings:
            mock_settings.ENABLED = True
            mock_settings.CONTENT_DIR = "/tmp/content"
            mock_settings.HOST = "0.0.0.0"
            mock_settings.PORT = 8090
            result = dataset_manager._get_content_server_kwargs()

        assert result == {
            "content_dir": Path("/tmp/content"),
            "base_url": "http://0.0.0.0:8090",
        }
