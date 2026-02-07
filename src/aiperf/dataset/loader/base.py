# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.dataset.loader.context import LoaderContext


class BaseDatasetLoader(AIPerfLoggerMixin, ABC):
    """Root base class for all dataset loaders.

    Loaders handle format-specific parsing and generation. Shared state
    (prompt_generator, tokenizer, session IDs) and finalization behavior
    (model selection, max_tokens, context prompts) are accessed via ctx.

    Args:
        ctx: Shared loader context with dependencies and finalization.
    """

    def __init__(self, ctx: LoaderContext, **kwargs: Any) -> None:
        self.ctx = ctx
        super().__init__(**kwargs)

    @abstractmethod
    async def load(self) -> AsyncIterator[Conversation]:
        """Yield finalized conversations as an async iterator.

        Returns:
            AsyncIterator of Conversation objects, each fully finalized.
        """
        ...

    @classmethod
    @abstractmethod
    def can_load(
        cls,
        data: dict[str, Any] | None = None,
        filename: str | Path | None = None,
    ) -> bool:
        """Check if this loader can handle the given data format.

        Args:
            data: Optional dictionary from a single JSONL line.
            filename: Optional file/directory path.

        Returns:
            True if this loader can handle the input.
        """
        ...

    @classmethod
    @abstractmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred sampling strategy for this loader.

        Returns:
            The preferred DatasetSamplingStrategy.
        """
        ...
