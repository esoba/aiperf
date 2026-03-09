# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end roundtrip test: config -> synthesize -> write -> validate."""

from __future__ import annotations

import math
from pathlib import Path

import orjson

from aiperf.dataset.claude_code_gen.config_loader import load_config
from aiperf.dataset.claude_code_gen.distributions import lognormal_from_mean_median
from aiperf.dataset.claude_code_gen.models import (
    CacheLayerConfig,
    GroupConfig,
    ResetConfig,
    SessionDistributionConfig,
)
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

    def test_jsonl_incremental_input_length_matches_new_tokens(
        self, tmp_path: Path
    ) -> None:
        """Verify every JSONL row's input_length equals SynthesizedTurn.new_tokens."""
        config = SessionDistributionConfig()
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(20)

        run_dir = tmp_path / "run"
        jsonl_path, _, _ = write_dataset(
            sessions, run_dir, config, seed=42, config_name="default"
        )

        expected = []
        for session in sessions:
            for turn in session.turns:
                expected.append((session.session_id, turn.new_tokens))

        actual = []
        with jsonl_path.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = orjson.loads(line)
                actual.append((data["session_id"], data["input_length"]))

        assert actual == expected

    def test_jsonl_rows_reconstruct_cumulative_isl(self, tmp_path: Path) -> None:
        """Verify that accumulating incremental JSONL rows reproduces the
        synthesizer's cumulative input_length, and that each row's hash_ids
        and input_length are consistent for parallel_convert."""
        config = SessionDistributionConfig(
            system_prompt_tokens=100,
            new_tokens_per_turn=lognormal_from_mean_median(mean=200, median=100),
            generation_length=lognormal_from_mean_median(mean=50, median=30),
            reset=ResetConfig(base_probability=0.0, context_scaling=1.0),
            max_prompt_tokens=3_000,
            cache=CacheLayerConfig(
                layer1_tokens=100,
                layer1_5_tokens=50,
                layer2=lognormal_from_mean_median(mean=200, median=150),
                block_size=64,
            ),
            group=GroupConfig(num_groups=5, zipf_alpha=1.2),
        )
        synth = SessionSynthesizer(config, seed=99)
        sessions = synth.synthesize_sessions(1)
        session = sessions[0]
        assert len(session.turns) >= 3, f"Expected >= 3 turns, got {len(session.turns)}"

        run_dir = tmp_path / "run"
        jsonl_path, _, _ = write_dataset(
            sessions, run_dir, config, seed=99, config_name="test"
        )

        rows = []
        with jsonl_path.open("rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(orjson.loads(line))

        block_size = config.cache.block_size
        cumulative_isl = 0
        for i, (row, turn) in enumerate(zip(rows, session.turns, strict=False)):
            cumulative_isl += row["input_length"]

            # Reconstructed cumulative ISL matches synthesizer
            assert cumulative_isl == turn.input_length, (
                f"turn {i}: reconstructed ISL {cumulative_isl} != "
                f"synthesizer {turn.input_length}"
            )

            # hash_ids count matches ceil(input_length / block_size)
            expected_blocks = math.ceil(row["input_length"] / block_size)
            assert len(row["hash_ids"]) == expected_blocks, (
                f"turn {i}: {len(row['hash_ids'])} blocks != "
                f"ceil({row['input_length']}/{block_size}) = {expected_blocks}"
            )

            # final_block_size is valid for parallel_convert
            n_blocks = len(row["hash_ids"])
            final_block = row["input_length"] - (n_blocks - 1) * block_size
            assert 1 <= final_block <= block_size, (
                f"turn {i}: final_block_size {final_block} out of range "
                f"[1, {block_size}]"
            )

            # Add previous output for next turn's cumulative
            cumulative_isl += row["output_length"]

    def test_jsonl_group_id_roundtrip(self, tmp_path: Path) -> None:
        """group_id survives synthesize -> write -> reload cycle."""
        config = SessionDistributionConfig()
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(20)

        run_dir = tmp_path / "run"
        jsonl_path, _, _ = write_dataset(
            sessions, run_dir, config, seed=42, config_name="default"
        )

        expected = {s.session_id: s.group_id for s in sessions}
        found = {}
        with jsonl_path.open("rb") as f:
            for line in f:
                data = orjson.loads(line.strip())
                if "group_id" in data:
                    found[data["session_id"]] = data["group_id"]

        assert found == expected

    def test_l15_sharing_in_written_dataset(self, tmp_path: Path) -> None:
        """Sessions in the same group must share L1.5 hash IDs in the JSONL output."""
        config = SessionDistributionConfig()
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(50)
        alloc = synth.allocator
        l1 = alloc.l1_blocks
        l15 = alloc.l15_blocks

        run_dir = tmp_path / "run"
        jsonl_path, _, _ = write_dataset(
            sessions, run_dir, config, seed=42, config_name="default"
        )

        # Collect turn-0 hash_ids and group_id per session from JSONL
        turn0_data: dict[str, dict] = {}
        with jsonl_path.open("rb") as f:
            for line in f:
                data = orjson.loads(line.strip())
                if "group_id" in data:
                    turn0_data[data["session_id"]] = data

        # Group by group_id and verify L1.5 blocks match within group
        by_group: dict[int, list[list[int]]] = {}
        for info in turn0_data.values():
            gid = info["group_id"]
            ids = info["hash_ids"]
            if len(ids) > l1 + l15:
                by_group.setdefault(gid, []).append(ids[l1 : l1 + l15])

        for gid, l15_lists in by_group.items():
            if len(l15_lists) < 2:
                continue
            for other in l15_lists[1:]:
                assert other == l15_lists[0], f"L1.5 mismatch in group {gid}"

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
