# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path

from aiperf.common.enums import PublicDatasetType
from aiperf.common.environment import Environment
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import RequestRecord
from aiperf.transports.aiohttp_client import AioHttpClient

AIPERF_DATASET_CACHE_DIR = Path(".cache/aiperf/datasets")

_REGISTRY: dict[PublicDatasetType, "PublicDataset"] = {}

_logger = AIPerfLoggerMixin()


@dataclass(frozen=True)
class PublicDataset:
    """Lightweight metadata describing a public dataset.

    Attributes:
        dataset_type: The enum key for this public dataset.
        name: Human-readable name for logging/display.
        url: URL to download the dataset from.
        loader_type: Plugin name of the dataset_loader to use after download.
        remote_filename: Filename to use when caching locally.
    """

    dataset_type: PublicDatasetType
    name: str
    url: str
    loader_type: str
    remote_filename: str
    _auto_register: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        if self._auto_register:
            _REGISTRY[self.dataset_type] = self

    def get_cache_filename(self) -> str:
        """Get the local cache filename for this dataset."""
        return self.remote_filename


def get_public_dataset(dataset_type: PublicDatasetType) -> PublicDataset:
    """Look up a PublicDataset by type.

    Args:
        dataset_type: The public dataset type to look up.

    Returns:
        The PublicDataset metadata.

    Raises:
        ValueError: If the dataset type is not registered.
    """
    if dataset_type not in _REGISTRY:
        raise ValueError(f"Unknown public dataset type: {dataset_type}")
    return _REGISTRY[dataset_type]


async def download_public_dataset(dataset: PublicDataset) -> Path:
    """Download a public dataset (or return cached path).

    Args:
        dataset: The PublicDataset metadata.

    Returns:
        Local file path to the downloaded/cached dataset.
    """
    cache_filepath = AIPERF_DATASET_CACHE_DIR / dataset.get_cache_filename()

    if cache_filepath.exists():
        _logger.info(
            f"Loading {dataset.name} dataset from local cache {cache_filepath}"
        )
        return cache_filepath

    _logger.info(f"No local dataset cache found, downloading from {dataset.url}")

    http_client = AioHttpClient(timeout=Environment.DATASET.PUBLIC_DATASET_TIMEOUT)
    try:
        record: RequestRecord = await http_client.get_request(
            dataset.url, headers={"Accept": "application/json"}
        )
        content = record.responses[0].text
    finally:
        await http_client.close()

    _logger.info(f"Saving {dataset.name} dataset to local cache {cache_filepath}")
    try:
        cache_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_filepath, "w") as f:
            f.write(content)
    except Exception as e:
        raise DatasetLoaderError(f"Error saving dataset to local cache: {e}") from e

    return cache_filepath


# ---- Public dataset definitions (auto-register on import) ----

SHAREGPT = PublicDataset(
    dataset_type=PublicDatasetType.SHAREGPT,
    name="ShareGPT",
    url="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
    loader_type="sharegpt",
    remote_filename="ShareGPT_V3_unfiltered_cleaned_split.json",
)
