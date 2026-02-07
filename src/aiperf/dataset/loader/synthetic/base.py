# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.common import random_generator as rng
from aiperf.common.models import Audio, Image, Text, Turn, Video
from aiperf.dataset.generator.audio import AudioGenerator
from aiperf.dataset.generator.image import ImageGenerator
from aiperf.dataset.generator.video import VideoGenerator
from aiperf.dataset.loader.base import BaseDatasetLoader

if TYPE_CHECKING:
    from aiperf.dataset.loader.context import LoaderContext


class BaseSyntheticLoader(BaseDatasetLoader):
    """Base class for synthetic dataset loaders.

    Manages media generators, ISL/OSL pairing, and multi-modal payload
    generation. Shared state and finalization are accessed via ctx.

    Args:
        ctx: Shared loader context with dependencies and finalization.
    """

    def __init__(self, ctx: LoaderContext, **kwargs: Any) -> None:
        super().__init__(ctx=ctx, **kwargs)

        self.image_generator = ImageGenerator(ctx.config.input.image)
        self.audio_generator = AudioGenerator(ctx.config.input.audio)
        self.video_generator = VideoGenerator(ctx.config.input.video)

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
        isl, _ = self.ctx.get_turn_sequence_lengths(turn_id)

        stddev = (
            0
            if self.ctx.seq_distribution is not None
            else self.ctx.config.input.prompt.input_tokens.stddev
        )

        for _ in range(self.ctx.config.input.prompt.batch_size):
            content = self.ctx.prompt_generator.generate(mean=isl, stddev=stddev)

            if is_first and self.ctx.prefix_prompt_enabled:
                prefix = self.ctx.prompt_generator.get_random_prefix_prompt()
                content = f"{prefix} {content}"

            text.contents.append(content)

        return text

    def _generate_image_payloads(self) -> Image:
        """Generate synthetic image payloads.

        Returns:
            Image payload object.
        """
        image = Image(name="image_url")
        for _ in range(self.ctx.config.input.image.batch_size):
            data = self.image_generator.generate()
            image.contents.append(data)
        return image

    def _generate_audio_payloads(self) -> Audio:
        """Generate synthetic audio payloads.

        Returns:
            Audio payload object.
        """
        audio = Audio(name="input_audio")
        for _ in range(self.ctx.config.input.audio.batch_size):
            data = self.audio_generator.generate()
            audio.contents.append(data)
        return audio

    def _generate_video_payloads(self) -> Video:
        """Generate synthetic video payloads.

        Returns:
            Video payload object.
        """
        video = Video(name="video_url")
        for _ in range(self.ctx.config.input.video.batch_size):
            data = self.video_generator.generate()
            if data:
                video.contents.append(data)
        return video

    @property
    def include_prompt(self) -> bool:
        """Whether prompt generation is enabled."""
        return self.ctx.config.input.prompt.input_tokens.mean > 0

    @property
    def include_image(self) -> bool:
        """Whether image generation is enabled."""
        return (
            self.ctx.config.input.image.width.mean > 0
            and self.ctx.config.input.image.height.mean > 0
        )

    @property
    def include_audio(self) -> bool:
        """Whether audio generation is enabled."""
        return self.ctx.config.input.audio.length.mean > 0

    @property
    def include_video(self) -> bool:
        """Whether video generation is enabled."""
        return bool(
            self.ctx.config.input.video.width and self.ctx.config.input.video.height
        )
