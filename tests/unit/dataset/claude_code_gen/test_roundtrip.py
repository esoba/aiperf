# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end roundtrip test: config -> synthesize -> write -> validate."""

from __future__ import annotations

import math
from pathlib import Path

import orjson

from aiperf.dataset.claude_code_gen.config_loader import load_config
from aiperf.dataset.claude_code_gen.models import SessionDistributionConfig
from aiperf.dataset.claude_code_gen.session_synthesizer import SessionSynthesizer
from aiperf.dataset.claude_code_gen.writer import write_dataset
from aiperf.dataset.loader.models import MooncakeTrace


class TestRoundtrip:
    def test_config_to_mooncake_roundtrip(self, tmp_path: Path) -> None:
        config = SessionDistributionConfig()
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(50)

        run_dir = tmp_path / "run"
        jsonl_path, manifest_path, quality_path = write_dataset(
            sessions, run_dir, config, seed=42, config_name="default"
        )

        # Validate every row parses as MooncakeTrace
        line_count = 0
        with jsonl_path.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = orjson.loads(line)
                MooncakeTrace(**data)
                line_count += 1

        total_turns = sum(len(s.turns) for s in sessions)
        assert line_count == total_turns

        # Manifest is valid JSON
        manifest = orjson.loads(manifest_path.read_bytes())
        assert manifest["seed"] == 42
        assert manifest["num_sessions"] == 50

        # Quality report is valid JSON with new structure
        quality = orjson.loads(quality_path.read_bytes())
        assert "observed_vs_target" in quality
        assert "config_summary" in quality
        assert "session_end_stats" in quality

    def test_reproducibility_across_runs(self, tmp_path: Path) -> None:
        config = SessionDistributionConfig()

        synth1 = SessionSynthesizer(config, seed=123)
        sessions1 = synth1.synthesize_sessions(10)
        run1 = tmp_path / "run1"
        write_dataset(sessions1, run1, config, seed=123)

        synth2 = SessionSynthesizer(config, seed=123)
        sessions2 = synth2.synthesize_sessions(10)
        run2 = tmp_path / "run2"
        write_dataset(sessions2, run2, config, seed=123)

        assert (run1 / "dataset.jsonl").read_bytes() == (
            run2 / "dataset.jsonl"
        ).read_bytes()

    def test_stress_1k_sessions_no_errors(self, tmp_path: Path) -> None:
        config = SessionDistributionConfig()
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(1000)

        run_dir = tmp_path / "stress"
        write_dataset(sessions, run_dir, config, seed=42)

        # Spot-check: all session_ids unique
        session_ids = {s.session_id for s in sessions}
        assert len(session_ids) == 1000

        # Spot-check: file is non-empty and has correct line count
        total_turns = sum(len(s.turns) for s in sessions)
        jsonl = run_dir / "dataset.jsonl"
        with jsonl.open("rb") as f:
            line_count = sum(1 for line in f if line.strip())
        assert line_count == total_turns

    def test_cache_invariants_at_scale(self, tmp_path: Path) -> None:
        config = SessionDistributionConfig()
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(20)
        l1_blocks = synth.allocator.l1_blocks
        canonical_l1 = list(range(l1_blocks))
        block_size = synth.allocator.block_size

        for session in sessions:
            for i, turn in enumerate(session.turns):
                # Block count matches ISL
                expected_blocks = math.ceil(turn.input_length / block_size)
                assert len(turn.hash_ids) == expected_blocks, (
                    f"hash_ids count {len(turn.hash_ids)} != "
                    f"ceil({turn.input_length}/{block_size}) = {expected_blocks}"
                )

                # L1 consistency: used L1 IDs are a prefix of canonical range
                l1_used = min(l1_blocks, len(turn.hash_ids))
                assert turn.hash_ids[:l1_used] == canonical_l1[:l1_used]

                # Prefix property
                if i > 0:
                    prev = session.turns[i - 1].hash_ids
                    assert turn.hash_ids[: len(prev)] == prev, (
                        "Turn N must be prefix of Turn N+1"
                    )

    def test_manifest_can_be_used_as_config(self, tmp_path: Path) -> None:
        """Verify that a manifest.json from a run can be loaded as config."""
        config = SessionDistributionConfig()
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(10)

        run_dir = tmp_path / "run"
        _, manifest_path, _ = write_dataset(
            sessions, run_dir, config, seed=42, config_name="default"
        )

        reloaded = load_config(str(manifest_path))
        assert reloaded.system_prompt_tokens == config.system_prompt_tokens
        assert reloaded.max_prompt_tokens == config.max_prompt_tokens
