#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert wall-clock timestamps in postmortem.md to relative (t=0, t=+Xm Ys)."""

import re
import sys
from pathlib import Path

BENCHMARK_START = "21:24:42"


def hms_to_seconds(ts: str) -> int:
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = int(parts[0]), int(parts[1]), 0
    else:
        raise ValueError(f"Bad timestamp: {ts}")
    return h * 3600 + m * 60 + s


def format_relative(delta_s: int) -> str:
    if delta_s == 0:
        return "t=0"
    sign = "+" if delta_s > 0 else "-"
    abs_s = abs(delta_s)
    minutes, seconds = divmod(abs_s, 60)
    if minutes == 0:
        return f"t={sign}{seconds}s"
    if seconds == 0:
        return f"t={sign}{minutes}m"
    return f"t={sign}{minutes}m{seconds:02d}s"


def convert(ts: str) -> str:
    base = hms_to_seconds(BENCHMARK_START)
    return format_relative(hms_to_seconds(ts) - base)


def main() -> None:
    path = Path(sys.argv[1])
    text = path.read_text()

    # Collect all unique timestamps found (for audit)
    found: dict[str, str] = {}

    def replace_ts(m: re.Match) -> str:
        ts = m.group(0)
        rel = convert(ts)
        found[ts] = rel
        return rel

    # HH:MM:SS first (greedy), then HH:MM (only where not already converted)
    result = re.sub(r"\b\d{1,2}:\d{2}:\d{2}\b", replace_ts, text)
    result = re.sub(r"\b(\d{1,2}:\d{2})\b", replace_ts, result)

    print("Conversions applied:")
    for wall, rel in sorted(found.items(), key=lambda kv: hms_to_seconds(kv[0])):
        print(f"  {wall:>8s} -> {rel}")

    path.write_text(result)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
