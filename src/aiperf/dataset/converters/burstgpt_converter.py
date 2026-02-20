# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BurstGPT CSV to mooncake JSONL converter."""

from __future__ import annotations

import math
import random
from typing import Any

import pandas as pd
from tqdm import tqdm

from aiperf.dataset.converters.burstgpt_config import BurstGptConfig
from aiperf.dataset.synthesis.rolling_hasher import RollingHasher


class BurstGptConverter:
    """Convert BurstGPT CSV traces to mooncake-style JSONL.

    Applies optional model/log-type filtering, timestamp speed adjustment,
    offset-to-zero normalization, and row skip/limit before converting each
    row to a mooncake record with random hash IDs.
    """

    def __init__(self, config: BurstGptConfig) -> None:
        self._config = config

    def convert(self) -> list[dict[str, Any]]:
        """Load, filter, and convert BurstGPT CSV to mooncake records."""
        c = self._config
        df = pd.read_csv(c.input_file)
        print(f"Loaded {c.input_file} ({df.shape[0]} rows, {df.shape[1]} columns)")

        df = self._apply_filters(df)
        df = self._apply_speed_ratio(df)
        df = self._offset_timestamps_to_zero(df)
        records = self._to_mooncake(df)
        if c.verbose:
            self._print_statistics(records)
        return records

    def default_output_filename(self) -> str:
        """Derive output filename from input filename."""
        return self._config.input_file.stem + ".jsonl"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self._config
        if c.model is not None:
            df = df[df["Model"] == c.model]
            print(f"After model filter ({c.model}): {len(df)} rows")
        if c.log_type is not None:
            df = df[df["Log Type"] == c.log_type]
            print(f"After log type filter ({c.log_type}): {len(df)} rows")
        if c.skip_num_prompt > 0:
            df = df.iloc[c.skip_num_prompt :]
            print(f"After skip ({c.skip_num_prompt}): {len(df)} rows")
        if c.num_prompt is not None:
            df = df.head(c.num_prompt)
            print(f"After limit ({c.num_prompt}): {len(df)} rows")
        return df.reset_index(drop=True)

    def _apply_speed_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._config.speed_ratio == 1.0:
            return df
        df = df.copy()
        df["Timestamp"] = df["Timestamp"] / self._config.speed_ratio
        print(f"Applied speed ratio: {self._config.speed_ratio}")
        return df

    @staticmethod
    def _offset_timestamps_to_zero(df: pd.DataFrame) -> pd.DataFrame:
        if "Timestamp" not in df.columns or len(df) == 0:
            return df
        min_ts = df["Timestamp"].min()
        if pd.isna(min_ts) or min_ts == 0:
            return df
        df = df.copy()
        df["Timestamp"] = df["Timestamp"] - float(min_ts)
        print(f"Offset timestamps to zero (subtracted {min_ts:.6f}s)")
        return df

    def _to_mooncake(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        c = self._config
        hasher = RollingHasher()
        records: list[dict[str, Any]] = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
            input_length = int(row["Request tokens"])
            output_length = int(row["Response tokens"])
            hash_array_length = math.ceil(input_length / c.block_size)

            random.seed(idx)
            content_blocks = [
                (random.randint(0, c.num_hash_blocks),)
                for _ in range(hash_array_length)
            ]

            records.append(
                {
                    "timestamp": int(row["Timestamp"] * 1000),
                    "input_length": input_length,
                    "output_length": output_length,
                    "hash_ids": hasher.hash_token_blocks(content_blocks),
                }
            )

        print(f"Converted {len(records)} rows to mooncake format")
        return records

    def _print_statistics(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        input_lengths = [r["input_length"] for r in records]
        output_lengths = [r["output_length"] for r in records]
        timestamps_ms = [r["timestamp"] for r in records]
        n = len(records)

        def _avg_std(vals: list[int]) -> tuple[float, float]:
            avg = sum(vals) / len(vals)
            std = (
                (sum((x - avg) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
                if len(vals) > 1
                else 0.0
            )
            return avg, std

        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)

        isl_avg, isl_std = _avg_std(input_lengths)
        print("\nInput Length (ISL):")
        print(f"  Min: {min(input_lengths)}")
        print(f"  Max: {max(input_lengths)}")
        print(f"  Avg: {isl_avg:.2f}")
        print(f"  Std: {isl_std:.2f}")

        osl_avg, osl_std = _avg_std(output_lengths)
        print("\nOutput Length (OSL):")
        print(f"  Min: {min(output_lengths)}")
        print(f"  Max: {max(output_lengths)}")
        print(f"  Avg: {osl_avg:.2f}")
        print(f"  Std: {osl_std:.2f}")

        max_seq = max(i + o for i, o in zip(input_lengths, output_lengths, strict=True))
        print("\nSequence Length (ISL + OSL):")
        print(f"  Max: {max_seq}")

        if n > 1:
            min_ts_s = min(timestamps_ms) / 1000.0
            max_ts_s = max(timestamps_ms) / 1000.0
            duration_s = max_ts_s - min_ts_s

            if duration_s > 0:
                avg_rps = n / duration_s
                print("\nRequest Rate:")
                print(f"  Total requests: {n}")
                print(f"  Duration: {duration_s:.2f} seconds")
                print(f"  Average RPS: {avg_rps:.2f}")

                plot_width = 60
                target_bins = 20
                bin_size_s = max(1.0, duration_s / target_bins)
                num_bins = max(1, math.ceil(duration_s / bin_size_s))

                counts = [0] * num_bins
                for ts_ms in timestamps_ms:
                    rel_s = (ts_ms / 1000.0) - min_ts_s
                    idx = max(0, min(int(rel_s / bin_size_s), num_bins - 1))
                    counts[idx] += 1

                rates = [c / bin_size_s for c in counts]
                peak_rps = max(rates) if rates else 0.0

                print("\nRequest rate vs time:")
                print(f"  Bin: {bin_size_s:.2f}s, Peak RPS: {peak_rps:.2f}")
                if peak_rps > 0:
                    digits = max(1, len(str(int(math.ceil(duration_s)))))
                    label_width = (2 * digits) + 2
                    bar_width = max(1, plot_width - label_width - 3)

                    for i, rps in enumerate(rates):
                        start_s = i * bin_size_s
                        end_s = min((i + 1) * bin_size_s, duration_s)
                        bar_len = int(round((rps / peak_rps) * bar_width))
                        bar = "#" * max(0, min(bar_width, bar_len))
                        label = f"{start_s:>{digits}.0f}-{end_s:>{digits}.0f}s"
                        line = f"{label} | {bar}"
                        print(line[:plot_width])
            else:
                print("\nRequest Rate:")
                print(f"  Total requests: {n}")
                print("  Duration: 0 seconds (all requests at same timestamp)")
                print("  Average RPS: N/A")
        else:
            print("\nRequest Rate:")
            print(f"  Total requests: {n}")
            print("  Average RPS: N/A (only 1 request)")

        print("=" * 60)
