# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.common.environment import Environment
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation, RequestRecord
from aiperf.dataset.loader.base import BaseDatasetLoader
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.transports.aiohttp_client import AioHttpClient

if TYPE_CHECKING:
    from aiperf.dataset.loader.context import LoaderContext
    from aiperf.plugin.enums import DatasetSamplingStrategy

AIPERF_DATASET_CACHE_DIR = Path(".cache/aiperf/datasets")


class PublicDatasetLoader(BaseDatasetLoader):
    """Generic loader that downloads a public dataset and delegates to a file loader.

    Looks up its own metadata from the plugin registry to determine the download
    URL, cache filename, and which file loader to delegate to.
    """

    def __init__(
        self, *, public_dataset_type: str, ctx: LoaderContext, **kwargs: Any
    ) -> None:
        super().__init__(ctx=ctx, **kwargs)
        self._metadata = plugins.get_public_dataset_metadata(public_dataset_type)
        self._delegate: BaseDatasetLoader | None = None

    async def _get_delegate(self) -> BaseDatasetLoader:
        """Download the dataset and create the delegate file loader."""
        if self._delegate is None:
            local_path = await _download(
                self._metadata.name, self._metadata.url, self._metadata.remote_filename
            )
            LoaderClass = plugins.get_class(
                PluginType.DATASET_LOADER, self._metadata.file_loader
            )
            self._delegate = LoaderClass(filename=str(local_path), ctx=self.ctx)
        return self._delegate

    async def load(self) -> AsyncIterator[Conversation]:
        delegate = await self._get_delegate()
        async for conversation in delegate.load():
            yield conversation

    @classmethod
    def can_load(
        cls,
        data: dict[str, Any] | None = None,
        filename: str | Path | None = None,
    ) -> bool:
        return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        raise NotImplementedError(
            "Use get_preferred_sampling_strategy() on the delegate loader instance instead."
        )

    def get_delegate_preferred_sampling_strategy(self) -> DatasetSamplingStrategy:
        """Get the preferred sampling strategy from the delegate file loader class."""
        LoaderClass = plugins.get_class(
            PluginType.DATASET_LOADER, self._metadata.file_loader
        )
        return LoaderClass.get_preferred_sampling_strategy()


async def _download(name: str, url: str, remote_filename: str) -> Path:
    """Download a public dataset (or return cached path)."""
    cache_filepath = AIPERF_DATASET_CACHE_DIR / remote_filename

    if cache_filepath.exists():
        return cache_filepath

    http_client = AioHttpClient(timeout=Environment.DATASET.PUBLIC_DATASET_TIMEOUT)
    try:
        record: RequestRecord = await http_client.get_request(
            url, headers={"Accept": "application/json"}
        )
        content = record.responses[0].text
    finally:
        await http_client.close()

    try:
        cache_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_filepath, "w") as f:
            f.write(content)
    except Exception as e:
        raise DatasetLoaderError(f"Error saving dataset to local cache: {e}") from e

    return cache_filepath
