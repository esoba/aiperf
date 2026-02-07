# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import OutputDefaults, ServiceConfig, UserConfig
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    MessageType,
)
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_command, on_request, on_stop
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    ConversationTurnRequestMessage,
    ConversationTurnResponseMessage,
    DatasetConfiguredNotification,
    ProfileConfigureCommand,
)
from aiperf.common.mixins import ReplyClientMixin
from aiperf.common.models import (
    DatasetMetadata,
    InputsFile,
    ModelEndpointInfo,
    RequestInfo,
    SessionPayloads,
)
from aiperf.common.models.dataset_models import ConversationMetadata
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.utils import load_json_str
from aiperf.dataset.public_datasets import (
    download_public_dataset,
    get_public_dataset,
)
from aiperf.dataset.utils import check_file_exists
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    DatasetBackingStoreType,
    DatasetLoaderType,
    PluginType,
)

if TYPE_CHECKING:
    from aiperf.dataset.loader.base import BaseDatasetLoader
    from aiperf.dataset.protocols import (
        DatasetBackingStoreProtocol,
        DatasetClientStoreProtocol,
    )
    from aiperf.endpoints.protocols import EndpointProtocol


class DatasetManager(ReplyClientMixin, BaseComponentService):
    """Manages dataset generation/acquisition and provides mmap access for workers.

    Primary responsibilities:
    - Generate synthetic prompts or load datasets from files/public sources
    - Stream conversations directly to memory-mapped files via backing store
    - Publish DatasetConfiguredNotification with mmap paths for worker access

    Workers access conversations directly via mmap (zero-copy), eliminating the
    need for ZMQ request-response communication with DatasetManager at runtime.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            reply_client_address=CommAddress.DATASET_MANAGER_PROXY_BACKEND,
            reply_client_bind=False,
        )
        self.user_config = user_config
        self.tokenizer: Tokenizer | None = None
        self.dataset_metadata: DatasetMetadata | None = None
        self.dataset_configured = asyncio.Event()

        BackingStoreClass = plugins.get_class(
            PluginType.DATASET_BACKING_STORE, DatasetBackingStoreType.MEMORY_MAP
        )
        self._backing_store: DatasetBackingStoreProtocol = BackingStoreClass(
            benchmark_id=user_config.benchmark_id,
        )
        self._dataset_client: DatasetClientStoreProtocol | None = None

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the dataset."""

        self.info("Configuring tokenizer(s) for dataset manager")
        begin = time.perf_counter()
        await self._configure_tokenizer()
        duration = time.perf_counter() - begin
        self.info(lambda: f"Tokenizer(s) configured in {duration:.2f} seconds")

        self.info(lambda: f"Configuring dataset for {self.service_id}")
        begin = time.perf_counter()
        await self._configure_dataset()
        await self._configure_dataset_client()
        await self._generate_inputs_json_file()

        duration = time.perf_counter() - begin
        self.info(lambda: f"Dataset configured in {duration:.2f} seconds")

    async def _configure_dataset_client(self) -> None:
        """Configure the dataset client for serving fallback requests."""
        client_metadata = self._backing_store.get_client_metadata()
        ClientStoreClass = plugins.get_class(
            PluginType.DATASET_CLIENT_STORE, client_metadata.client_type
        )
        self._dataset_client = ClientStoreClass(client_metadata=client_metadata)
        await self._dataset_client.initialize()
        self.dataset_configured.set()
        self.info("Dataset client initialized")

    async def _configure_tokenizer(self) -> None:
        """Configure the tokenizer for the dataset manager."""
        model_name = self.user_config.endpoint.model_names[0]
        tokenizer_config = self.user_config.tokenizer
        tokenizer_name = tokenizer_config.get_tokenizer_name_for_model(model_name)

        # Let exceptions propagate - controller_utils will display the error panel
        self.tokenizer = await asyncio.to_thread(
            Tokenizer.from_pretrained,
            tokenizer_name,
            trust_remote_code=tokenizer_config.trust_remote_code,
            revision=tokenizer_config.revision,
            resolve_alias=tokenizer_config.should_resolve_alias,
        )

    async def _generate_input_payloads(
        self,
        model_endpoint: ModelEndpointInfo,
    ) -> InputsFile:
        """Generate input payloads from the dataset for use in the inputs.json file."""
        inputs = InputsFile()

        EndpointClass = plugins.get_class(
            PluginType.ENDPOINT, model_endpoint.endpoint.type
        )
        endpoint: EndpointProtocol = EndpointClass(model_endpoint=model_endpoint)
        self.debug(
            lambda: f"Created endpoint protocol for {model_endpoint.endpoint.type}, "
            f"class: {endpoint.__class__.__name__}",
        )
        session_payloads_map: dict[str, list] = {}
        for conv_meta in self.dataset_metadata.conversations:
            conversation = await self._dataset_client.get_conversation(
                conv_meta.conversation_id
            )
            session_id = conversation.session_id
            if session_id not in session_payloads_map:
                session_payloads_map[session_id] = []

            for i, turn in enumerate(conversation.turns):
                request_info = RequestInfo(
                    model_endpoint=model_endpoint,
                    turns=[turn],
                    turn_index=i,
                    credit_num=i,
                    credit_phase=CreditPhase.PROFILING,
                    x_request_id="",
                    x_correlation_id="",
                    conversation_id=conversation.session_id,
                )
                request_info.endpoint_headers = endpoint.get_endpoint_headers(
                    request_info
                )
                request_info.endpoint_params = endpoint.get_endpoint_params(
                    request_info
                )
                payload = endpoint.format_payload(request_info)
                session_payloads_map[session_id].append(payload)

        for session_id, payloads in session_payloads_map.items():
            inputs.data.append(
                SessionPayloads(session_id=session_id, payloads=payloads)
            )
        return inputs

    async def _generate_inputs_json_file(self) -> None:
        """Generate inputs.json file in the artifact directory."""
        file_path = (
            self.user_config.output.artifact_directory / OutputDefaults.INPUTS_JSON_FILE
        )
        temp_file_path = file_path.with_suffix(".tmp")
        self.info(f"Generating inputs.json file at {file_path.resolve()}")

        try:
            start_time = time.perf_counter()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            model_endpoint = ModelEndpointInfo.from_user_config(self.user_config)
            inputs = await self._generate_input_payloads(model_endpoint)

            temp_file_path.write_bytes(
                orjson.dumps(
                    inputs.model_dump(exclude_none=True, mode="json"),
                    option=orjson.OPT_INDENT_2,
                )
            )
            temp_file_path.replace(file_path)

            duration = time.perf_counter() - start_time
            self.info(f"inputs.json file generated in {duration:.2f} seconds")

        except OSError as e:
            self.exception(
                f"Error generating inputs.json file at {file_path.resolve()}: {e!r}"
            )
            # NOTE: We don't raise an error here for OS related errors like writing to a file,
            # as this won't affect the benchmark execution.
        except Exception as e:
            # This is a fatal error, as later in the benchmark, errors will occur while trying to convert the payloads
            # on the worker side.
            self.exception(
                f"Error generating inputs.json file at {file_path.resolve()}: {e!r}"
            )
            raise
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()

    async def _create_public_dataset_loader(self) -> BaseDatasetLoader:
        """Download a public dataset and return its loader (without calling load)."""
        public_dataset_type = self.user_config.input.public_dataset
        dataset = get_public_dataset(public_dataset_type)
        local_path = await download_public_dataset(dataset)

        LoaderClass = plugins.get_class(PluginType.DATASET_LOADER, dataset.loader_type)
        loader = LoaderClass(
            filename=str(local_path),
            config=self.user_config,
            tokenizer=self.tokenizer,
        )

        # Set sampling strategy from loader if not user-specified
        if "dataset_sampling_strategy" not in self.user_config.input.model_fields_set:
            self.user_config.input.dataset_sampling_strategy = (
                loader.get_preferred_sampling_strategy()
            )

        return loader

    def _create_file_loader(self) -> BaseDatasetLoader:
        """Create a file-based loader (without calling load).

        Auto-detects loader type if not explicitly specified.
        """
        check_file_exists(self.user_config.input.file)

        dataset_type = self.user_config.input.dataset_type
        if dataset_type is None:
            dataset_type = self._infer_dataset_type(self.user_config.input.file)
            self.info(f"Auto-detected dataset type: {dataset_type}")

        # Validate synthesis options are only used with mooncake_trace
        self._validate_synthesis_config(dataset_type)

        LoaderClass = plugins.get_class(PluginType.DATASET_LOADER, dataset_type)
        loader = LoaderClass(
            filename=self.user_config.input.file,
            config=self.user_config,
            tokenizer=self.tokenizer,
        )

        # Set sampling strategy from loader if not user-specified
        if self.user_config.input.dataset_sampling_strategy is None:
            preferred_strategy = loader.get_preferred_sampling_strategy()
            self.user_config.input.dataset_sampling_strategy = preferred_strategy
            self.info(
                f"Using preferred sampling strategy for {dataset_type}: {preferred_strategy}"
            )

        return loader

    def _create_synthetic_loader(self) -> BaseDatasetLoader:
        """Create a synthetic loader (without calling load).

        Selects the appropriate synthetic loader based on endpoint type.
        """
        endpoint_type = self.user_config.endpoint.type

        if self._is_rankings_endpoint(endpoint_type):
            loader_type = DatasetLoaderType.SYNTHETIC_RANKINGS
        else:
            loader_type = DatasetLoaderType.SYNTHETIC_MULTIMODAL

        LoaderClass = plugins.get_class(PluginType.DATASET_LOADER, loader_type)
        loader = LoaderClass(
            config=self.user_config,
            tokenizer=self.tokenizer,
        )

        # Set default sampling strategy for synthetic datasets if not explicitly set
        from aiperf.common.config.config_defaults import InputDefaults

        if self.user_config.input.dataset_sampling_strategy is None:
            self.user_config.input.dataset_sampling_strategy = (
                InputDefaults.DATASET_SAMPLING_STRATEGY
            )
            self.info(
                f"Using default sampling strategy for synthetic dataset: {InputDefaults.DATASET_SAMPLING_STRATEGY}"
            )

        return loader

    async def _create_loader(self) -> BaseDatasetLoader:
        """Create the appropriate loader based on user config."""
        if self.user_config.input.public_dataset is not None:
            return await self._create_public_dataset_loader()
        elif self.user_config.input.file is not None:
            return self._create_file_loader()
        else:
            return self._create_synthetic_loader()

    def _infer_dataset_type(self, file_path: str) -> DatasetLoaderType:
        """Infer the dataset type from the input file.

        Queries all registered loaders to check if they can handle the data format.

        Args:
            file_path: Path to the JSONL file or directory

        Returns:
            DatasetLoaderType if successfully inferred

        Raises:
            ValueError: If no loader can handle the data format
        """
        path = Path(file_path)

        # If it's a directory, use path-based detection only
        if path.is_dir():
            return self._infer_type(data=None, filename=file_path)

        # For files, read first non-empty line and use both content and path detection
        try:
            with open(file_path) as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    data = load_json_str(line)
                    return self._infer_type(data=data, filename=file_path)
        except ValueError as e:
            self.exception(
                f"Error inferring dataset type from file: {file_path}: {e!r}"
            )
            raise

        raise ValueError(f"Empty file: {file_path}. Cannot infer dataset type.")

    def _infer_type(
        self, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> DatasetLoaderType:
        """Infer the dataset type from data and/or filename.

        Args:
            data: Optional dictionary representing a single line from the JSONL file.
            filename: Optional path to the input file/directory

        Returns:
            DatasetLoaderType if successfully inferred

        Raises:
            ValueError: If the type field is invalid or no loader can handle the data format
        """
        # Check for explicit type field first (most efficient)
        if data is not None and "type" in data:
            try:
                explicit_type = DatasetLoaderType(data["type"])
                LoaderClass = plugins.get_class(
                    PluginType.DATASET_LOADER, explicit_type
                )
                if not LoaderClass.can_load(data, filename):
                    raise ValueError(
                        f"Explicit type field {explicit_type} specified, but loader {LoaderClass.__name__} "
                        "cannot handle the data format. Please specify --dataset-type explicitly."
                    )
                self.info(f"Using explicit type field: {explicit_type}")
                return explicit_type
            except (ValueError, KeyError) as e:
                raise ValueError(
                    f"Invalid type field value: {data['type']}. Please specify --dataset-type explicitly."
                ) from e

        detected_type = None
        for entry, LoaderClass in plugins.iter_all(PluginType.DATASET_LOADER):
            if LoaderClass.can_load(data, filename):
                self.info(
                    f"Loader {LoaderClass.__name__} can handle the input file data format."
                )
                dataset_type = DatasetLoaderType(entry.name)
                if detected_type is not None:
                    raise ValueError(
                        f"Multiple loaders can handle the data format: {detected_type} and {dataset_type}. "
                        "Please specify --dataset-type explicitly."
                    )
                detected_type = dataset_type

        if detected_type is None:
            raise ValueError(
                "No loader can handle the data format. Please specify --dataset-type explicitly."
            )

        return detected_type

    def _validate_synthesis_config(self, dataset_type: DatasetLoaderType) -> None:
        """Validate that synthesis options are only used with mooncake_trace.

        Args:
            dataset_type: The determined dataset type.

        Raises:
            ValueError: If synthesis options are set but dataset type is not mooncake_trace.
        """
        if (
            self.user_config.input.synthesis.should_synthesize()
            and dataset_type != DatasetLoaderType.MOONCAKE_TRACE
        ):
            raise ValueError(
                f"Synthesis options (--synthesis-speedup-ratio, --synthesis-prefix-len-multiplier, "
                f"--synthesis-prefix-root-multiplier, --synthesis-prompt-len-multiplier) "
                f"are only supported with mooncake_trace datasets, but got {dataset_type}. "
                f"Either remove synthesis options or use --dataset-type mooncake_trace."
            )

    def _is_rankings_endpoint(self, endpoint_type: str) -> bool:
        return "rankings" in endpoint_type.lower()

    async def _configure_dataset(self) -> None:
        if self.user_config is None:
            raise self._service_error("User config is required for dataset manager")

        self.dataset_configured.clear()

        loader = await self._create_loader()

        await self._backing_store.initialize()

        conversation_metadata: list[ConversationMetadata] = []
        async for conversation in loader.load():
            await self._backing_store.add_conversation(
                conversation.session_id, conversation
            )
            conversation_metadata.append(conversation.metadata())

        await self._backing_store.finalize()
        client_metadata = self._backing_store.get_client_metadata()
        self.info(f"Backing store finalized: {client_metadata}")

        self.dataset_metadata = DatasetMetadata(
            conversations=conversation_metadata,
            sampling_strategy=self.user_config.input.dataset_sampling_strategy,
        )
        self.info(
            f"sampling strategy: {self.dataset_metadata.sampling_strategy}, "
            f"unique conversations: {len(self.dataset_metadata.conversations)}, "
            f"unique turn count: {self.dataset_metadata.total_turn_count}"
        )
        # Note: dataset_configured event is set in _profile_configure_command after
        # the dataset client is initialized, to avoid a race condition where fallback
        # requests arrive before the client is ready.
        await self.publish(
            DatasetConfiguredNotification(
                service_id=self.service_id,
                metadata=self.dataset_metadata,
                client_metadata=client_metadata,
            )
        )

    @on_request(MessageType.CONVERSATION_REQUEST)
    async def _handle_conversation_request(
        self, message: ConversationRequestMessage
    ) -> ConversationResponseMessage:
        """Handle a conversation request using the dataset client."""
        self.debug(lambda: f"Handling conversation request: {message}")

        await self._wait_for_dataset_configuration()

        if self._dataset_client is None:
            raise self._service_error(
                "Dataset client is not initialized. Dataset must be configured before handling requests.",
            )

        try:
            conversation = await self._dataset_client.get_conversation(
                message.conversation_id
            )
        except KeyError:
            raise self._service_error(
                f"Conversation {message.conversation_id} not found in dataset.",
            ) from None

        self.trace_or_debug(
            lambda: f"Sending conversation response: {conversation}",
            lambda: f"Sending conversation response with id: {conversation.session_id}",
        )
        return ConversationResponseMessage(
            service_id=self.service_id,
            request_id=message.request_id,
            conversation=conversation,
        )

    @on_request(MessageType.CONVERSATION_TURN_REQUEST)
    async def _handle_conversation_turn_request(
        self, message: ConversationTurnRequestMessage
    ) -> ConversationTurnResponseMessage:
        """Handle a turn request using the dataset client."""
        self.debug(lambda: f"Handling turn request: {message}")

        await self._wait_for_dataset_configuration()

        if self._dataset_client is None:
            raise self._service_error(
                "Dataset client is not initialized. Dataset must be configured before handling requests.",
            )

        try:
            conversation = await self._dataset_client.get_conversation(
                message.conversation_id
            )
        except KeyError as e:
            raise self._service_error(
                f"Conversation {message.conversation_id} not found in dataset.",
            ) from e

        if message.turn_index >= len(conversation.turns):
            raise self._service_error(
                f"Turn index {message.turn_index} is out of range for conversation {message.conversation_id}.",
            )

        turn = conversation.turns[message.turn_index]

        self.trace_or_debug(
            lambda: f"Sending turn response: {turn}",
            "Sending turn response",
        )
        return ConversationTurnResponseMessage(
            service_id=self.service_id,
            request_id=message.request_id,
            turn=turn,
        )

    async def _wait_for_dataset_configuration(self) -> None:
        """Wait for the dataset to be configured if it is not already."""
        if not self.dataset_configured.is_set():
            self.debug(
                "Dataset not configured. Waiting for dataset to be configured..."
            )
            await asyncio.wait_for(
                self.dataset_configured.wait(),
                timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
            )

    @on_stop
    async def _cleanup(self) -> None:
        """Clean up the backing store, dataset client, and associated mmap files."""
        if self._dataset_client is not None:
            await self._dataset_client.stop()
            self.debug("Dataset client cleanup complete")
        if self._backing_store is not None:
            await self._backing_store.stop()
            self.debug("Backing store cleanup complete")


def main() -> None:
    """Main entry point for the dataset manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.DATASET_MANAGER)


if __name__ == "__main__":
    main()
