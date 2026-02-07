# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.models import Audio, Image, Text, Turn, Video
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.audio import AudioGenerator
from aiperf.dataset.generator.image import ImageGenerator
from aiperf.dataset.generator.video import VideoGenerator
from aiperf.dataset.loader.base import BaseDatasetLoader


class BaseSyntheticLoader(BaseDatasetLoader):
    """Base class for synthetic dataset loaders.

    Absorbs generator management, ISL/OSL pairing, and multi-modal payload
    generation from the old SyntheticDatasetComposer.

    Args:
        config: User configuration.
        tokenizer: Tokenizer instance.
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer, **kwargs: Any) -> None:
        super().__init__(config=config, tokenizer=tokenizer, **kwargs)

        self.image_generator = ImageGenerator(config.input.image)
        self.audio_generator = AudioGenerator(config.input.audio)
        self.video_generator = VideoGenerator(config.input.video)

        self._turn_sampler_rng = rng.derive("composer.conversation.turn_count")
        self._delay_sampler_rng = rng.derive("composer.conversation.turn_delay")

    @classmethod
    def can_load(
        cls,
        data: dict[str, Any] | None = None,
        filename: str | Path | None = None,
    ) -> bool:
        """Synthetic loaders don't load from files."""
        return False

    # ---- Synthetic payload generation helpers ----

    def _generate_text_payloads(self, turn: Turn, is_first: bool) -> Text:
        """Generate text payloads for a single turn.

        Args:
            turn: The turn object (used for ISL/OSL caching).
            is_first: Whether this is the first turn in a conversation.

        Returns:
            Text payload object.
        """
        text = Text(name="text")

        turn_id = id(turn)
        isl, _ = self._get_turn_sequence_lengths(turn_id)

        stddev = (
            0
            if self._seq_distribution is not None
            else self.config.input.prompt.input_tokens.stddev
        )

        for _ in range(self.config.input.prompt.batch_size):
            content = self.prompt_generator.generate(mean=isl, stddev=stddev)

            if is_first and self.prefix_prompt_enabled:
                prefix = self.prompt_generator.get_random_prefix_prompt()
                content = f"{prefix} {content}"

            text.contents.append(content)

        return text

    def _generate_image_payloads(self) -> Image:
        """Generate synthetic image payloads.

        Returns:
            Image payload object.
        """
        image = Image(name="image_url")
        for _ in range(self.config.input.image.batch_size):
            data = self.image_generator.generate()
            image.contents.append(data)
        return image

    def _generate_audio_payloads(self) -> Audio:
        """Generate synthetic audio payloads.

        Returns:
            Audio payload object.
        """
        audio = Audio(name="input_audio")
        for _ in range(self.config.input.audio.batch_size):
            data = self.audio_generator.generate()
            audio.contents.append(data)
        return audio

    def _generate_video_payloads(self) -> Video:
        """Generate synthetic video payloads.

        Returns:
            Video payload object.
        """
        video = Video(name="video_url")
        for _ in range(self.config.input.video.batch_size):
            data = self.video_generator.generate()
            if data:
                video.contents.append(data)
        return video

    @property
    def include_prompt(self) -> bool:
        """Whether prompt generation is enabled."""
        return self.config.input.prompt.input_tokens.mean > 0

    @property
    def include_image(self) -> bool:
        """Whether image generation is enabled."""
        return (
            self.config.input.image.width.mean > 0
            and self.config.input.image.height.mean > 0
        )

    @property
    def include_audio(self) -> bool:
        """Whether audio generation is enabled."""
        return self.config.input.audio.length.mean > 0

    @property
    def include_video(self) -> bool:
        """Whether video generation is enabled."""
        return bool(self.config.input.video.width and self.config.input.video.height)
