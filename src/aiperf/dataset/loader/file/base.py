# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.common.config import UserConfig
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader.base import BaseDatasetLoader
from aiperf.dataset.loader.models import CustomDatasetT

if TYPE_CHECKING:
    from aiperf.common.models import Conversation


class BaseFileLoader(BaseDatasetLoader):
    """Base class for file-based dataset loaders.

    Implements a two-stage pipeline: parse_and_validate() -> convert_to_conversations().
    Turn finalization and conversation finalization are applied automatically.

    Args:
        filename: Path to the file or directory to load.
        config: User configuration.
        tokenizer: Tokenizer instance.
    """

    def __init__(
        self, *, filename: str, config: UserConfig, tokenizer: Tokenizer, **kwargs: Any
    ) -> None:
        super().__init__(config=config, tokenizer=tokenizer, **kwargs)
        self.filename = filename

    async def load(self) -> AsyncIterator[Conversation]:
        """Load dataset via two-stage pipeline, yielding finalized conversations.

        1. parse_and_validate() -- parse file(s) into typed model objects
        2. convert_to_conversations() -- convert to Conversation objects
        3. Finalize each conversation (turns, context prompts) and yield it

        Returns:
            AsyncIterator of finalized Conversation objects.
        """
        data = self.parse_and_validate()
        conversations = self.convert_to_conversations(data)

        for idx, conversation in enumerate(conversations):
            for turn in conversation.turns:
                self._finalize_turn(turn)
            self._finalize_conversation(conversation, idx)
            yield conversation

    @abstractmethod
    def parse_and_validate(self) -> dict[str, list[CustomDatasetT]]:
        """Parse and validate the input file(s).

        Returns:
            Parsed data keyed by session/filename.
        """
        ...

    @abstractmethod
    def convert_to_conversations(
        self, data: dict[str, list[CustomDatasetT]]
    ) -> list[Conversation]:
        """Convert parsed data to conversation objects.

        Args:
            data: Parsed data from parse_and_validate().

        Returns:
            List of Conversation objects (not yet finalized).
        """
        ...

    @classmethod
    def can_load(
        cls,
        data: dict[str, Any] | None = None,
        filename: str | Path | None = None,
    ) -> bool:
        """Check if this loader can handle the given data format.

        Dispatches to can_load_file or can_load_directory based on path type.

        Args:
            data: Optional dictionary from a JSONL line.
            filename: Optional file/directory path.

        Returns:
            True if this loader can handle the input.
        """
        if filename is not None:
            path = Path(filename) if isinstance(filename, str) else filename
            if path.is_dir():
                return cls.can_load_directory(path)
            elif data is not None:
                return cls.can_load_file(data, filename)
        elif data is not None:
            return cls.can_load_file(data, filename)
        return False

    @classmethod
    def can_load_file(
        cls,
        data: dict[str, Any],
        filename: str | Path | None = None,
    ) -> bool:
        """Check if this loader can handle a file with the given data sample.

        Args:
            data: Dictionary from first JSONL line.
            filename: Optional file path.

        Returns:
            True if this loader can handle the file.
        """
        return False

    @classmethod
    def can_load_directory(cls, path: Path) -> bool:
        """Check if this loader can handle the given directory.

        Args:
            path: Directory path.

        Returns:
            True if this loader can handle the directory.
        """
        return False
