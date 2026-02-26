# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset router component -- owns dataset metadata and /api/dataset endpoints."""

from __future__ import annotations

import asyncio
import pathlib
from typing import Annotated, Any

import aiofiles.os as aio_os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.compression import (
    CompressionEncoding,
    select_encoding,
    stream_file_compressed,
)
from aiperf.common.enums import MessageType
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_message
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import MemoryMapClientMetadata

DatasetDep = Annotated["DatasetRouter", component_dependency("dataset")]

dataset_router = APIRouter(tags=["Dataset"], include_in_schema=False)


class DatasetRouter(MessageBusClientMixin, BaseRouter):
    """Owns dataset metadata and exposes /api/dataset endpoints."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._dataset_client_metadata: MemoryMapClientMetadata | None = None
        self._dataset_configured = asyncio.Event()

    def get_router(self) -> APIRouter:
        return dataset_router

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(
        self, message: DatasetConfiguredNotification
    ) -> None:
        """Store dataset file paths from DatasetManager."""
        if isinstance(message.client_metadata, MemoryMapClientMetadata):
            self._dataset_client_metadata = message.client_metadata
            self._dataset_configured.set()
            self.info(
                f"Dataset configured: {message.client_metadata.conversation_count} conversations, "
                f"compressed={message.client_metadata.compressed_data_file_path is not None}"
            )
        else:
            self.warning(
                f"Received dataset metadata with unsupported type: {type(message.client_metadata)}"
            )

    @property
    def dataset_client_metadata(self) -> MemoryMapClientMetadata | None:
        return self._dataset_client_metadata

    @property
    def dataset_configured(self) -> asyncio.Event:
        return self._dataset_configured


async def _stream_dataset_file(
    file_path: pathlib.Path,
    precompressed_path: pathlib.Path | None,
    accept_encoding: str | None,
    file_type: str,
) -> StreamingResponse:
    """Stream a dataset file with compression support."""
    file_exists = await aio_os.path.exists(file_path)
    precompressed_exists = precompressed_path is not None and await aio_os.path.exists(
        precompressed_path
    )

    compress_only_mode = precompressed_exists and not file_exists

    if not file_exists and not precompressed_exists:
        raise HTTPException(
            status_code=404, detail=f"{file_type} file not found: {file_path.name}"
        )

    if compress_only_mode:
        if "zstd" not in (accept_encoding or "").lower():
            raise HTTPException(
                status_code=406,
                detail=f"{file_type} is pre-compressed with zstd. Client must accept zstd encoding.",
            )
        stream = stream_file_compressed(
            precompressed_path, CompressionEncoding.IDENTITY
        )
        encoding = CompressionEncoding.ZSTD
    else:
        encoding = select_encoding(accept_encoding)
        if encoding == CompressionEncoding.ZSTD and precompressed_exists:
            stream = stream_file_compressed(
                precompressed_path, CompressionEncoding.IDENTITY
            )
        else:
            stream = stream_file_compressed(file_path, encoding)

    return StreamingResponse(
        stream,
        media_type="application/octet-stream",
        headers={
            "Content-Encoding": encoding,
            "Content-Disposition": f'attachment; filename="{file_path.name}"',
        },
    )


async def _wait_for_dataset_metadata(component: DatasetRouter) -> None:
    """Wait for dataset configuration with timeout."""
    if not component.dataset_configured.is_set():
        try:
            component.info("Waiting for dataset configuration...")
            await asyncio.wait_for(
                component.dataset_configured.wait(),
                timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="Dataset not yet configured. DatasetManager has not sent configuration.",
            ) from None

    if component.dataset_client_metadata is None:
        raise HTTPException(status_code=503, detail="Dataset metadata not available")


@dataset_router.get("/api/dataset/data")
async def get_dataset_data(
    component: DatasetDep, request: Request
) -> StreamingResponse:
    """Stream the dataset.dat file for Kubernetes workers (compressed)."""
    await _wait_for_dataset_metadata(component)
    metadata = component.dataset_client_metadata
    return await _stream_dataset_file(
        metadata.data_file_path,
        metadata.compressed_data_file_path,
        request.headers.get("accept-encoding"),
        "Dataset",
    )


@dataset_router.get("/api/dataset/index")
async def get_dataset_index(
    component: DatasetDep, request: Request
) -> StreamingResponse:
    """Stream the index.dat file for Kubernetes workers (compressed)."""
    await _wait_for_dataset_metadata(component)
    metadata = component.dataset_client_metadata
    return await _stream_dataset_file(
        metadata.index_file_path,
        metadata.compressed_index_file_path,
        request.headers.get("accept-encoding"),
        "Index",
    )
