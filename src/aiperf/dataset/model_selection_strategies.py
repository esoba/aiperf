# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, runtime_checkable

from aiperf.common import random_generator as rng
from aiperf.common.models import Turn


@runtime_checkable
class ModelSelectionStrategyProtocol(Protocol):
    """Protocol for model selection strategies."""

    def select(self, turn: Turn) -> str:
        """Select a model name for the given turn.

        Args:
            turn: The turn to select a model for.

        Returns:
            The selected model name.
        """
        ...


class RoundRobinModelSelectionStrategy:
    """Cycles through models in order.

    The nth call returns model at index (n mod number_of_models).
    """

    def __init__(self, model_names: list[str]) -> None:
        self._model_names = model_names
        self._counter = 0

    def select(self, turn: Turn) -> str:
        """Select the next model in round-robin order."""
        model_name = self._model_names[self._counter % len(self._model_names)]
        self._counter += 1
        return model_name


class RandomModelSelectionStrategy:
    """Randomly selects a model using uniform distribution."""

    def __init__(self, model_names: list[str]) -> None:
        self._model_names = model_names
        self._rng = rng.derive("model_selection.random")

    def select(self, turn: Turn) -> str:
        """Select a random model."""
        return self._rng.choice(self._model_names)


class ShuffleModelSelectionStrategy:
    """Shuffles without replacement, reshuffles when exhausted.

    Ensures all models are used before any model is reused.
    """

    def __init__(self, model_names: list[str]) -> None:
        self._model_names = list(model_names)
        self._rng = rng.derive("model_selection.shuffle")
        self._shuffled: list[str] = []
        self._index = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        """Reshuffle the model list."""
        self._shuffled = list(self._model_names)
        self._rng.shuffle(self._shuffled)
        self._index = 0

    def select(self, turn: Turn) -> str:
        """Select the next model from the shuffled list."""
        if self._index >= len(self._shuffled):
            self._reshuffle()
        model_name = self._shuffled[self._index]
        self._index += 1
        return model_name
