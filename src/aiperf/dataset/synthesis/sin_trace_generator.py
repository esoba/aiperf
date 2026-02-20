# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sinusoidal trace generator plugin."""

from __future__ import annotations

import math

import orjson
from tqdm import tqdm

from aiperf.common import random_generator as rng
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.random_generator import RandomGenerator
from aiperf.dataset.synthesis.rolling_hasher import RollingHasher
from aiperf.dataset.synthesis.sin_trace_config import SinTraceConfig

_logger = AIPerfLogger(__name__)


def _sinusoidal(t: float, min_val: float, max_val: float, period: float) -> float:
    """Compute sinusoidal value with phase shift starting from minimum at t=0."""
    mid = (min_val + max_val) / 2
    amp = (max_val - min_val) / 2
    return mid + amp * math.sin(2 * math.pi / period * t - math.pi / 2)


class SinTraceGenerator:
    """Trace generator with sinusoidal request rate and ISL/OSL ratio.

    Request rate and ISL/OSL ratio follow sinusoidal patterns with configurable
    min/max/period. Both use a phase shift of -pi/2 to start from minimum at t=0.
    """

    def __init__(self, config: SinTraceConfig) -> None:
        self._config = config
        self._rng: RandomGenerator = rng.derive("gen_trace.sin")

    def generate(self) -> list[bytes]:
        """Generate trace records with sinusoidal request rate and ISL/OSL ratio.

        Returns:
            List of orjson-serialized JSONL lines.
        """
        c = self._config
        rolling_hasher = RollingHasher()
        output_data: list[bytes] = []

        for t in tqdm(range(0, c.time_duration, c.process_interval)):
            t_end = min(t + c.process_interval, c.time_duration)
            request_rate = _sinusoidal(
                t, c.request_rate_min, c.request_rate_max, c.request_rate_period
            )
            num_requests = int(self._rng.poisson(request_rate * (t_end - t)))
            _logger.info(f"request_rate at {t:.2f}: {request_rate:.2f}")

            for req_idx in range(num_requests):
                t_req = t + (t_end - t) * req_idx / num_requests
                isl_osl_ratio = _sinusoidal(
                    t_req,
                    c.isl_osl_ratio_min,
                    c.isl_osl_ratio_max,
                    c.isl_osl_ratio_period,
                )
                _logger.info(f"isl_osl_ratio at {t_req:.2f}: {isl_osl_ratio:.2f}")

                if self._rng.random() < isl_osl_ratio:
                    isl, osl = c.isl1, c.osl1
                else:
                    isl, osl = c.isl2, c.osl2

                hash_block_ids = [
                    (self._rng.randrange(c.total_blocks),)
                    for _ in range(math.ceil(isl / c.block_size))
                ]
                rolling_hash_ids = rolling_hasher.hash_token_blocks(hash_block_ids)

                output_data.append(
                    orjson.dumps(
                        {
                            "timestamp": int(t_req * 1000),
                            "input_length": isl,
                            "output_length": osl,
                            "hash_ids": rolling_hash_ids,
                        }
                    )
                )

        return output_data

    def default_output_filename(self) -> str:
        """Generate default output filename from parameters."""
        c = self._config
        return (
            f"sin_b{c.block_size}_t{c.time_duration}"
            f"_rr{c.request_rate_min}-{c.request_rate_max}-{c.request_rate_period}"
            f"_io{c.isl1}{c.osl1}-{c.isl2}{c.osl2}"
            f"-{c.isl_osl_ratio_min}-{c.isl_osl_ratio_max}-{c.isl_osl_ratio_period}.jsonl"
        )
