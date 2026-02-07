# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig


class OutputTokensSampler:
    """Samples max_tokens values from output token configuration.

    Uses a normal distribution (clamped to minimum 1) when mean is configured,
    or returns None if no output tokens configuration is set.

    Args:
        config: User configuration containing output token distribution parameters.
    """

    def __init__(self, config: UserConfig) -> None:
        self._output_tokens_config = config.input.prompt.output_tokens
        self._rng = rng.derive("dataset.output_tokens")

    def sample(self) -> int | None:
        """Sample a max_tokens value from the configured distribution.

        Returns:
            Sampled max_tokens (minimum 1), or None if no output_tokens.mean is configured.
        """
        if self._output_tokens_config.mean is None:
            return None
        return self._rng.sample_positive_normal_integer(
            self._output_tokens_config.mean,
            self._output_tokens_config.stddev,
        )
