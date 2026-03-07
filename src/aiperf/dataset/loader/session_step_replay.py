# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import orjson

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.models import SessionStepReplaySubstep
from aiperf.plugin.enums import DatasetSamplingStrategy


class SessionStepReplayDatasetLoader(BaseFileLoader):
    """A dataset loader for session replay data.

    Loads a JSON file containing captured agent sessions where each session
    has sequential substeps with candidate prompts. One candidate is randomly
    selected per substep at load time. Designed for use with the completions
    endpoint.

    The input format is a JSON object keyed by session ID, where each value
    is a list of substep objects:

    ```json
    {
        "session_1": [
            {
                "candidate_prompts": ["prompt A", "prompt B"],
                "expected_output_tokens": 512
            },
            {
                "candidate_prompts": ["prompt C"],
                "expected_output_tokens": 256
            }
        ],
        "session_2": [...]
    }
    ```

    Each substep becomes a turn in a multi-turn conversation. Substeps are
    independent (no context accumulation) and execute sequentially within
    each session.

    Usage:
        aiperf profile \\
            --input-file sessions.json \\
            --custom-dataset-type session_step_replay \\
            --endpoint-type completions \\
            --concurrency 100
    """

    def __init__(self, *, filename: str, **kwargs: Any) -> None:
        super().__init__(filename=filename, **kwargs)

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Detect session replay format by .json extension and structure.

        The format is a single JSON object (not JSONL), so line-based data
        detection won't work. Instead, detect by loading the file and checking
        that it contains a dict of session_id -> list of substep objects.
        """
        if filename is None:
            return False

        path = Path(filename)
        if path.suffix != ".json" or not path.is_file():
            return False

        try:
            with open(path, "rb") as f:
                raw = orjson.loads(f.read())
        except (OSError, orjson.JSONDecodeError):
            return False

        if not isinstance(raw, dict) or not raw:
            return False

        # Check that at least the first value looks like a list of substeps
        first_value = next(iter(raw.values()))
        if not isinstance(first_value, list) or not first_value:
            return False

        first_substep = first_value[0]
        return isinstance(first_substep, dict) and "candidate_prompts" in first_substep

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode | None:
        return ConversationContextMode.STANDALONE

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SHUFFLE

    def load_dataset(self) -> dict[str, list[SessionStepReplaySubstep]]:
        """Load session step replay data from a JSON file.

        Returns:
            A dictionary mapping session_id to list of substeps.
        """
        with open(self.filename, "rb") as f:
            raw: dict[str, list[dict[str, Any]]] = orjson.loads(f.read())

        data: dict[str, list[SessionStepReplaySubstep]] = {}
        for session_id, substeps in raw.items():
            if not isinstance(substeps, list):
                continue
            data[session_id] = [
                SessionStepReplaySubstep.model_validate(s) for s in substeps
            ]

        self.info(
            f"Loaded {sum(len(v) for v in data.values()):,} substeps "
            f"across {len(data):,} sessions"
        )
        return data

    def convert_to_conversations(
        self, data: dict[str, list[SessionStepReplaySubstep]]
    ) -> list[Conversation]:
        """Convert session step replay data to conversations.

        Stores all candidate prompts on each turn via prompt_candidates.
        The InferenceClient randomly selects one at request time, so
        replays of the same conversation can pick different candidates.
        """
        conversations: list[Conversation] = []

        for session_id, substeps in data.items():
            conversation = Conversation(session_id=session_id)
            for substep in substeps:
                conversation.turns.append(
                    Turn(
                        prompt_candidates=substep.candidate_prompts,
                        max_tokens=substep.expected_output_tokens,
                    )
                )
            conversations.append(conversation)

        return conversations
