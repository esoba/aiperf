# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for execution strategies."""

from pathlib import Path

import pytest

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import AIPerfConfig
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import FixedTrialsStrategy

_DEFAULT_ENDPOINT = {"urls": ["http://localhost:8000/v1/chat/completions"]}
_DEFAULT_DATASETS = {
    "main": {"type": "synthetic", "entries": 100, "prompts": {"isl": 128}}
}
_DEFAULT_LOAD = {"type": "concurrency", "requests": 10, "concurrency": 1}


def _make_config(**overrides) -> AIPerfConfig:
    return AIPerfConfig(
        models=overrides.pop("models", ["test-model"]),
        endpoint=overrides.pop("endpoint", _DEFAULT_ENDPOINT),
        datasets=overrides.pop("datasets", _DEFAULT_DATASETS),
        load=overrides.pop("load", _DEFAULT_LOAD),
        **overrides,
    )


class TestFixedTrialsStrategy:
    """Tests for FixedTrialsStrategy."""

    @pytest.mark.parametrize(
        "results,expected",
        [
            ([], True),  # No results yet
            (
                [
                    RunResult(
                        label="run_0001",
                        success=True,
                        summary_metrics={
                            "ttft_avg": JsonMetricResult(unit="ms", avg=100.0)
                        },
                        artifacts_path=Path("/tmp/run_0001"),
                    )
                ],
                True,
            ),  # Partial results
            (
                [
                    RunResult(
                        label="run_0001",
                        success=True,
                        summary_metrics={
                            "ttft": JsonMetricResult(unit="ms", avg=100.0)
                        },
                        artifacts_path=Path("/tmp/run_0001"),
                    ),
                    RunResult(
                        label="run_0002",
                        success=True,
                        summary_metrics={
                            "ttft": JsonMetricResult(unit="ms", avg=105.0)
                        },
                        artifacts_path=Path("/tmp/run_0002"),
                    ),
                ],
                False,
            ),  # Complete results (num_trials=2)
        ],
    )
    def test_should_continue_returns_expected(self, results, expected):
        """Test should_continue returns expected value based on results count."""
        num_trials = 2 if len(results) == 2 else 3
        strategy = FixedTrialsStrategy(num_trials=num_trials)
        assert strategy.should_continue(results) is expected

    @pytest.mark.parametrize(
        "run_index,num_trials,expected",
        [
            (0, 10, "run_0001"),
            (1, 10, "run_0002"),
            (9, 10, "run_0010"),
            (0, 5, "run_0001"),
            (4, 5, "run_0005"),
        ],
    )
    def test_get_run_label_zero_padding_returns_expected(
        self, run_index, num_trials, expected
    ):
        """Test get_run_label returns zero-padded labels with correct padding."""
        strategy = FixedTrialsStrategy(num_trials=num_trials)
        assert strategy.get_run_label(run_index) == expected

    def test_get_cooldown_seconds_configured_returns_value(self):
        """Test get_cooldown_seconds returns configured value."""
        strategy = FixedTrialsStrategy(num_trials=3, cooldown_seconds=5.0)
        assert strategy.get_cooldown_seconds() == 5.0

    def test_get_cooldown_seconds_default_returns_zero(self):
        """Test get_cooldown_seconds returns default value of zero."""
        strategy = FixedTrialsStrategy(num_trials=3)
        assert strategy.get_cooldown_seconds() == 0.0

    def test_auto_set_seed_on_first_run(self):
        """Test auto_set_seed sets random_seed on first run when None."""
        strategy = FixedTrialsStrategy(num_trials=3, auto_set_seed=True)

        config = _make_config(random_seed=None)

        new_config = strategy.get_next_config(config, [])

        assert new_config.input.random_seed == FixedTrialsStrategy.DEFAULT_SEED
        assert config.random_seed is None

    def test_auto_set_seed_preserves_user_seed(self):
        """Test auto_set_seed preserves user-specified random_seed."""
        strategy = FixedTrialsStrategy(num_trials=3, auto_set_seed=True)

        config = _make_config(random_seed=999)

        new_config = strategy.get_next_config(config, [])

        assert new_config.input.random_seed == 999

    def test_auto_set_seed_disabled(self):
        """Test that auto_set_seed=False doesn't modify config."""
        strategy = FixedTrialsStrategy(num_trials=3, auto_set_seed=False)

        config = _make_config(random_seed=None)

        new_config = strategy.get_next_config(config, [])

        assert new_config.input.random_seed is None

    def test_get_next_config_returns_base_config_after_first_run(self):
        """Test get_next_config returns modified config after first run when warmup disabled."""
        strategy = FixedTrialsStrategy(
            num_trials=3, auto_set_seed=True, disable_warmup_after_first=True
        )

        config = _make_config(
            random_seed=42,
            load={
                "warmup": {
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                    "exclude": True,
                },
                "default": {
                    "type": "concurrency",
                    "requests": 100,
                    "concurrency": 1,
                },
            },
        )

        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/run_0001"),
            )
        ]

        new_config = strategy.get_next_config(config, results)

        assert new_config is not config
        assert new_config.loadgen.warmup_request_count is None
        # Original config should be unchanged
        assert config.loadgen.warmup_request_count == 10

    def test_ensure_random_seed_creates_deep_copy(self):
        """Test that _ensure_random_seed creates a deep copy."""
        strategy = FixedTrialsStrategy(num_trials=3, auto_set_seed=True)

        config = _make_config(random_seed=None)

        new_config = strategy.get_next_config(config, [])

        assert new_config is not config
        assert new_config.input.random_seed == FixedTrialsStrategy.DEFAULT_SEED
        assert config.random_seed is None

    def test_invalid_cooldown_seconds(self):
        """Test that negative cooldown raises ValueError."""
        with pytest.raises(ValueError, match="Invalid cooldown_seconds"):
            FixedTrialsStrategy(num_trials=5, cooldown_seconds=-1.0)

    def test_label_sanitization(self):
        """Test that labels are sanitized to prevent path traversal."""
        strategy = FixedTrialsStrategy(num_trials=5)

        assert strategy.get_run_label(0) == "run_0001"
        assert strategy.get_run_label(99) == "run_0100"

    def test_disable_warmup_after_first_enabled(self):
        """Test that warmup is disabled after first run when disable_warmup_after_first=True."""
        strategy = FixedTrialsStrategy(num_trials=3, disable_warmup_after_first=True)

        config = _make_config(
            load={
                "warmup": {
                    "type": "concurrency",
                    "requests": 10,
                    "duration": 30.0,
                    "concurrency": 2,
                    "exclude": True,
                },
                "default": {
                    "type": "concurrency",
                    "requests": 100,
                    "concurrency": 1,
                },
            },
        )

        first_config = strategy.get_next_config(config, [])
        assert first_config.loadgen.warmup_request_count == 10
        assert first_config.loadgen.warmup_duration == 30.0
        assert first_config.loadgen.warmup_concurrency == 2

        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/run_0001"),
            )
        ]

        second_config = strategy.get_next_config(config, results)
        assert second_config.loadgen.warmup_request_count is None
        assert second_config.loadgen.warmup_duration is None
        assert second_config.loadgen.warmup_concurrency is None

        assert config.loadgen.warmup_request_count == 10

    def test_disable_warmup_after_first_disabled(self):
        """Test that warmup is preserved for all runs when disable_warmup_after_first=False."""
        strategy = FixedTrialsStrategy(num_trials=3, disable_warmup_after_first=False)

        config = _make_config(
            load={
                "warmup": {
                    "type": "concurrency",
                    "requests": 10,
                    "duration": 30.0,
                    "concurrency": 2,
                    "exclude": True,
                },
                "default": {
                    "type": "concurrency",
                    "requests": 100,
                    "concurrency": 1,
                },
            },
        )

        first_config = strategy.get_next_config(config, [])
        assert first_config.loadgen.warmup_request_count == 10
        assert first_config.loadgen.warmup_duration == 30.0
        assert first_config.loadgen.warmup_concurrency == 2

        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/run_0001"),
            )
        ]

        second_config = strategy.get_next_config(config, results)
        assert second_config.loadgen.warmup_request_count == 10
        assert second_config.loadgen.warmup_duration == 30.0
        assert second_config.loadgen.warmup_concurrency == 2

    def test_disable_warmup_creates_deep_copy(self):
        """Test that disabling warmup creates a deep copy and doesn't modify original."""
        strategy = FixedTrialsStrategy(num_trials=3, disable_warmup_after_first=True)

        config = _make_config(
            load={
                "warmup": {
                    "type": "concurrency",
                    "requests": 10,
                    "duration": 30.0,
                    "concurrency": 1,
                    "exclude": True,
                },
                "default": {
                    "type": "concurrency",
                    "requests": 100,
                    "concurrency": 1,
                },
            },
        )

        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/run_0001"),
            )
        ]

        second_config = strategy.get_next_config(config, results)

        assert second_config is not config
        assert second_config.loadgen.warmup_request_count is None

        assert config.loadgen.warmup_request_count == 10
        assert config.loadgen.warmup_duration == 30.0

    def test_get_run_path(self):
        """Test get_run_path returns correct path structure."""
        from pathlib import Path

        strategy = FixedTrialsStrategy(num_trials=3)
        base_dir = Path("/tmp/artifacts")

        path = strategy.get_run_path(base_dir, 0)
        assert path == Path("/tmp/artifacts/profile_runs/run_0001")

        path = strategy.get_run_path(base_dir, 1)
        assert path == Path("/tmp/artifacts/profile_runs/run_0002")

        path = strategy.get_run_path(base_dir, 9)
        assert path == Path("/tmp/artifacts/profile_runs/run_0010")

    def test_get_aggregate_path(self):
        """Test get_aggregate_path returns correct path."""
        from pathlib import Path

        strategy = FixedTrialsStrategy(num_trials=3)
        base_dir = Path("/tmp/artifacts")

        path = strategy.get_aggregate_path(base_dir)
        assert path == Path("/tmp/artifacts/aggregate")

    def test_path_building_consistency(self):
        """Test that path building is consistent with label generation."""
        from pathlib import Path

        strategy = FixedTrialsStrategy(num_trials=5)
        base_dir = Path("/tmp/artifacts")

        for run_index in range(5):
            label = strategy.get_run_label(run_index)
            path = strategy.get_run_path(base_dir, run_index)

            assert path.name == label
            assert str(path).endswith(f"profile_runs/{label}")
