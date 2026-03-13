# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    Conversation,
    ParsedResponse,
    ProcessHealth,
    ReasoningResponseData,
    RequestRecord,
    SSEMessage,
    TextResponseData,
)
from aiperf.credit.structs import Credit, CreditContext
from aiperf.plugin.enums import ServiceRunType
from aiperf.workers.worker import Worker
from tests.harness.fake_communication import FakeCommunication as FakeCommunication
from tests.harness.fake_service_manager import FakeServiceManager as FakeServiceManager
from tests.harness.fake_tokenizer import FakeTokenizer
from tests.harness.fake_transport import FakeTransport as FakeTransport

_STUB_PROCESS_HEALTH = ProcessHealth(
    create_time=0.0, uptime=1.0, cpu_usage=0.0, memory_usage=0
)


@pytest.fixture
async def mock_worker(
    aiperf_config,
    fake_tokenizer: FakeTokenizer,
    skip_service_registration,
    mock_psutil_process,
):
    """Create a fully initialized and started MockWorker (no SystemController needed).

    Patches psutil.Process so ProcessHealthMixin.__init__ never reads /proc,
    and stubs get_process_health / get_pss_memory so the @background_task
    health check never blocks on real syscalls.
    """
    worker = Worker(
        config=aiperf_config,
        service_id="mock-service-id",
    )
    worker._measure_baseline_rtt = AsyncMock()
    worker.get_process_health = Mock(return_value=_STUB_PROCESS_HEALTH)
    worker.get_pss_memory = Mock(return_value=None)
    await worker.initialize()
    await worker.start()
    yield worker
    await worker.stop()


@pytest.mark.asyncio
class TestWorker:
    async def test_process_response(
        self, monkeypatch, mock_worker, sample_request_record
    ):
        """Ensure process_response extracts text correctly from RequestRecord."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text="Hello, world!"),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(sample_request_record)
        assert turn.texts[0].contents == ["Hello, world!"]

    async def test_process_response_empty(
        self, monkeypatch, mock_worker, sample_request_record
    ):
        """Ensure process_response handles empty responses correctly."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text=""),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(sample_request_record)
        assert turn is None

    async def test_process_response_reasoning_extracts_content(
        self, monkeypatch, mock_worker
    ):
        """Ensure process_response extracts content from reasoning responses."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=ReasoningResponseData(
                reasoning="Let me think...",
                content="The answer is 42.",
            ),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(RequestRecord())
        assert turn.texts[0].contents == ["The answer is 42."]

    async def test_process_response_reasoning_only_returns_none(
        self, monkeypatch, mock_worker
    ):
        """Ensure process_response returns None for reasoning-only responses (no content)."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=ReasoningResponseData(
                reasoning="Let me think about this...",
                content=None,
            ),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(RequestRecord())
        assert turn is None

    async def test_process_response_mixed_reasoning_and_text_combines_content(
        self, monkeypatch, mock_worker
    ):
        """Ensure process_response combines text and reasoning content."""
        mock_parsed_responses = [
            ParsedResponse(
                perf_ns=0,
                data=TextResponseData(text="Hello"),
            ),
            ParsedResponse(
                perf_ns=1,
                data=ReasoningResponseData(
                    reasoning="Thinking...",
                    content="World",
                ),
            ),
        ]
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=mock_parsed_responses)
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(RequestRecord())
        assert turn.texts[0].contents == ["HelloWorld"]


# --- FirstToken Callback Test Helpers ---


def create_first_token_callback(worker: Worker):
    """Create a first token callback that mirrors Worker implementation.

    This callback uses endpoint.parse_response to check if an SSE message
    contains meaningful content.

    Returns:
        Async callback function (ttft_ns, message) -> bool
    """

    async def first_token_callback(ttft_ns: int, message: SSEMessage) -> bool:
        parsed = worker.inference_client.endpoint.parse_response(message)
        return parsed is not None and parsed.data is not None

    return first_token_callback


def setup_mock_endpoint(worker: Worker, monkeypatch, parse_response_return):
    """Setup mock endpoint with specified parse_response return value.

    Args:
        worker: MockWorker instance
        monkeypatch: pytest monkeypatch fixture
        parse_response_return: Return value or side_effect for parse_response
    """
    mock_endpoint = Mock()
    if isinstance(parse_response_return, list):
        mock_endpoint.parse_response = Mock(side_effect=parse_response_return)
    else:
        mock_endpoint.parse_response = Mock(return_value=parse_response_return)
    mock_endpoint.extract_response_data = Mock()  # Should NOT be called
    monkeypatch.setattr(worker.inference_client, "endpoint", mock_endpoint)
    return mock_endpoint


@pytest.mark.asyncio
class TestWorkerFirstTokenCallback:
    """Test suite for Worker's first_token_callback logic."""

    @pytest.mark.parametrize(
        "parse_return,expected_result,description",
        [
            # Meaningful content - should return True
            pytest.param(
                ParsedResponse(
                    perf_ns=100_000_000, data=TextResponseData(text="Hello")
                ),
                True,
                "meaningful text content",
                id="meaningful_content",
            ),
            # None response - should return False
            pytest.param(
                None,
                False,
                "parse_response returns None",
                id="none_response",
            ),
            # ParsedResponse with data=None (usage only) - should return False
            pytest.param(
                ParsedResponse(
                    perf_ns=100_000_000,
                    data=None,
                    usage={"prompt_tokens": 10, "completion_tokens": 0},
                ),
                False,
                "usage-only response with data=None",
                id="none_data",
            ),
        ],
    )
    async def test_callback_return_value(
        self, monkeypatch, mock_worker, parse_return, expected_result, description
    ):
        """Test callback returns correct bool based on parse_response result."""
        setup_mock_endpoint(mock_worker, monkeypatch, parse_return)
        callback = create_first_token_callback(mock_worker)

        test_message = SSEMessage(perf_ns=100_000_000)
        result = await callback(50_000_000, test_message)

        assert result is expected_result, f"Failed for: {description}"

    async def test_callback_finds_first_meaningful_content_after_junk(
        self, monkeypatch, mock_worker
    ):
        """Test callback correctly identifies first meaningful content after junk messages."""
        parse_returns = [
            None,  # First: junk
            ParsedResponse(perf_ns=200_000_000, data=None),  # Second: usage only
            ParsedResponse(  # Third: actual content
                perf_ns=300_000_000,
                data=TextResponseData(text="Finally some content!"),
            ),
        ]

        setup_mock_endpoint(mock_worker, monkeypatch, parse_returns)
        callback = create_first_token_callback(mock_worker)

        messages = [SSEMessage(perf_ns=i * 100_000_000) for i in range(1, 4)]
        results = [await callback(msg.perf_ns, msg) for msg in messages]

        assert results == [False, False, True]


# --- Fixture for CreditContext ---


@pytest.fixture
def sample_credit_context() -> CreditContext:
    """Create a sample CreditContext for testing."""
    return CreditContext(
        credit=Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="test-conv-123",
            x_correlation_id="test-correlation-id",
            turn_index=0,
            num_turns=1,
            issued_at_ns=1000000,
        ),
        drop_perf_ns=2000000,
    )


# --- RetrieveConversation Tests ---


@pytest.mark.asyncio
class TestRetrieveConversation:
    """Test suite for Worker's _retrieve_conversation method."""

    async def test_returns_from_dataset_client_when_available(
        self, mock_worker, sample_credit_context
    ):
        """When _dataset_client is set, should return conversation from it."""
        expected_conversation = Conversation(session_id="test-conv-123", turns=[])
        mock_client = AsyncMock()
        mock_client.get_conversation = AsyncMock(return_value=expected_conversation)
        mock_worker._dataset_client = mock_client

        result = await mock_worker._retrieve_conversation(
            conversation_id="test-conv-123",
            credit_context=sample_credit_context,
        )

        assert result == expected_conversation
        mock_client.get_conversation.assert_called_once_with("test-conv-123")

    async def test_raises_cancelled_error_when_stop_requested_and_no_client(
        self, mock_worker, sample_credit_context
    ):
        """When _dataset_client is None and stop_requested, should raise CancelledError."""
        mock_worker._dataset_client = None
        mock_worker.stop_requested = True

        with pytest.raises(asyncio.CancelledError, match="Stop requested"):
            await mock_worker._retrieve_conversation(
                conversation_id="test-conv-123",
                credit_context=sample_credit_context,
            )

    async def test_falls_back_to_dataset_manager_when_no_client_and_not_stopping(
        self, monkeypatch, mock_worker, sample_credit_context
    ):
        """When _dataset_client is None and not stopping, should request from DatasetManager."""
        mock_worker._dataset_client = None
        expected_conversation = Conversation(session_id="test-conv-123", turns=[])
        mock_fallback = AsyncMock(return_value=expected_conversation)
        monkeypatch.setattr(
            mock_worker, "_request_conversation_from_dataset_manager", mock_fallback
        )

        result = await mock_worker._retrieve_conversation(
            conversation_id="test-conv-123",
            credit_context=sample_credit_context,
        )

        assert result == expected_conversation
        mock_fallback.assert_called_once_with("test-conv-123", sample_credit_context)


class TestKubernetesMode:
    """Test Kubernetes-specific behavior in Worker."""

    @pytest.fixture
    async def k8s_worker(
        self,
        aiperf_config,
        fake_tokenizer: FakeTokenizer,
        skip_service_registration,
    ) -> Worker:
        """Create a Worker in Kubernetes mode."""
        aiperf_config.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            config=aiperf_config,
            service_id="k8s-worker",
        )
        worker._measure_baseline_rtt = AsyncMock()
        await worker.initialize()
        await worker.start()
        yield worker
        await worker.stop()

    @pytest.fixture
    async def local_worker(
        self,
        aiperf_config,
        fake_tokenizer: FakeTokenizer,
        skip_service_registration,
    ) -> Worker:
        """Create a Worker in local (multiprocessing) mode."""
        aiperf_config.service_run_type = ServiceRunType.MULTIPROCESSING
        worker = Worker(
            config=aiperf_config,
            service_id="local-worker",
        )
        worker._measure_baseline_rtt = AsyncMock()
        await worker.initialize()
        await worker.start()
        yield worker
        await worker.stop()

    @pytest.mark.parametrize(
        "run_type,expected",
        [
            param(ServiceRunType.KUBERNETES, True, id="kubernetes"),
            param(ServiceRunType.MULTIPROCESSING, False, id="multiprocessing"),
        ],
    )  # fmt: skip
    def test_is_kubernetes_mode(
        self,
        aiperf_config,
        run_type: str,
        expected: bool,
    ) -> None:
        """_is_kubernetes_mode should return True only for KUBERNETES run type."""
        aiperf_config.service_run_type = run_type
        worker = Worker(
            config=aiperf_config,
            service_id="test-worker",
        )
        assert worker._is_kubernetes_mode() is expected

    @pytest.mark.asyncio
    async def test_dataset_configured_deferred_in_kubernetes_mode(
        self, k8s_worker: Worker
    ) -> None:
        """In K8s mode, dataset config should be stored as pending, not processed immediately."""
        mock_msg = MagicMock()
        mock_msg.client_metadata = MagicMock()
        mock_msg.metadata = MagicMock()

        # Patch _initialize_dataset_client to verify it's NOT called
        k8s_worker._initialize_dataset_client = AsyncMock()

        await k8s_worker._on_dataset_configured(mock_msg)

        # Should have stored as pending
        assert k8s_worker._pending_dataset_config is mock_msg
        # Should NOT have initialized client
        k8s_worker._initialize_dataset_client.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dataset_configured_immediate_in_local_mode(
        self, local_worker: Worker
    ) -> None:
        """In local mode, dataset config should initialize the client immediately."""
        mock_msg = MagicMock()
        mock_msg.client_metadata = MagicMock()
        mock_msg.metadata = MagicMock()

        # Patch _initialize_dataset_client to verify it IS called
        local_worker._initialize_dataset_client = AsyncMock()

        await local_worker._on_dataset_configured(mock_msg)

        # Should NOT have stored as pending
        assert local_worker._pending_dataset_config is None
        # Should have initialized client immediately
        local_worker._initialize_dataset_client.assert_awaited_once_with(
            mock_msg.client_metadata
        )
