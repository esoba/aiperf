# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    Conversation,
    ParsedResponse,
    ReasoningResponseData,
    RequestInfo,
    RequestRecord,
    SSEMessage,
    Text,
    TextResponseData,
    Turn,
)
from aiperf.credit.structs import Credit, CreditContext
from aiperf.workers.worker import Worker
from tests.harness.fake_tokenizer import TOKEN, FakeTokenizer


@pytest.fixture
async def mock_worker(
    user_config: UserConfig,
    service_config: ServiceConfig,
    fake_tokenizer: FakeTokenizer,
    skip_service_registration,
):
    """Create a fully initialized and started MockWorker (no SystemController needed)."""
    worker = Worker(
        service_config=service_config,
        user_config=user_config,
        service_id="mock-service-id",
    )
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

    async def test_process_response_truncates_multi_turn_history_to_requested_osl(
        self, monkeypatch, mock_worker
    ):
        """Ensure multi-turn assistant history is clipped to the requested OSL."""
        long_text = "abcdefghijklmnop"
        parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text=long_text),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)

        request_info = RequestInfo(
            model_endpoint=mock_worker.model_endpoint,
            turns=[
                Turn(
                    role="user",
                    model="test-model",
                    max_tokens=2,
                    texts=[Text(contents=["turn 1"])],
                )
            ],
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="truncate-test",
            x_correlation_id="truncate-session",
            conversation_id="truncate-conversation",
            is_final_turn=False,
        )
        record = RequestRecord(request_info=request_info, model_name="test-model")

        turn = await mock_worker._process_response(record)

        assert turn is not None
        assert turn.texts[0].contents == [TOKEN * 2]
        assert parsed_response.data.text == long_text

    async def test_process_response_pads_multi_turn_history_to_requested_osl(
        self, monkeypatch, mock_worker
    ):
        """Ensure multi-turn assistant history is padded to the requested OSL."""
        response_text = TOKEN
        parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text=response_text),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)

        request_info = RequestInfo(
            model_endpoint=mock_worker.model_endpoint,
            turns=[
                Turn(
                    role="user",
                    model="test-model",
                    max_tokens=4,
                    texts=[Text(contents=["turn 1"])],
                )
            ],
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="pad-test",
            x_correlation_id="pad-session",
            conversation_id="pad-conversation",
            is_final_turn=False,
        )
        record = RequestRecord(request_info=request_info, model_name="test-model")

        turn = await mock_worker._process_response(record)

        assert turn is not None
        assert turn.texts[0].contents == [TOKEN * 4]
        assert parsed_response.data.text == response_text

    async def test_process_response_empty_output_uses_synthetic_padding(
        self, monkeypatch, mock_worker
    ):
        """Ensure empty assistant output still grows history to requested OSL."""
        parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text=""),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)

        request_info = RequestInfo(
            model_endpoint=mock_worker.model_endpoint,
            turns=[
                Turn(
                    role="user",
                    model="test-model",
                    max_tokens=3,
                    texts=[Text(contents=["turn 1"])],
                )
            ],
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="empty-pad-test",
            x_correlation_id="empty-pad-session",
            conversation_id="empty-pad-conversation",
            is_final_turn=False,
        )
        record = RequestRecord(request_info=request_info, model_name="test-model")

        turn = await mock_worker._process_response(record)

        assert turn is not None
        assert turn.texts[0].contents == [TOKEN * 3]

    async def test_process_response_leaves_text_unchanged_without_requested_osl(
        self, monkeypatch, mock_worker
    ):
        """Ensure clipping is skipped when no output token budget was requested."""
        response_text = "abcdefghijklmnop"
        parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text=response_text),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)

        request_info = RequestInfo(
            model_endpoint=mock_worker.model_endpoint,
            turns=[
                Turn(
                    role="user",
                    model="test-model",
                    texts=[Text(contents=["turn 1"])],
                )
            ],
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="no-osl-test",
            x_correlation_id="no-osl-session",
            conversation_id="no-osl-conversation",
            is_final_turn=False,
        )
        record = RequestRecord(request_info=request_info, model_name="test-model")

        turn = await mock_worker._process_response(record)

        assert turn is not None
        assert turn.texts[0].contents == [response_text]

    async def test_normalized_history_is_used_in_follow_up_payload(
        self, monkeypatch, mock_worker
    ):
        """Ensure the next turn payload uses normalized assistant history."""
        short_text = TOKEN
        parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text=short_text),
        )
        mock_endpoint = Mock(wraps=mock_worker.inference_client.endpoint)
        mock_endpoint.extract_response_data = Mock(return_value=[parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)

        conversation = Conversation(
            session_id="payload-session",
            turns=[
                Turn(
                    role="user",
                    model="test-model",
                    max_tokens=2,
                    texts=[Text(contents=["first turn"])],
                ),
                Turn(
                    role="user",
                    model="test-model",
                    max_tokens=4,
                    texts=[Text(contents=["second turn"])],
                ),
            ],
        )
        session = mock_worker.session_manager.create_and_store(
            "payload-session",
            conversation,
            num_turns=2,
        )
        session.advance_turn(0)

        first_request = RequestInfo(
            model_endpoint=mock_worker.model_endpoint,
            turns=session.turn_list,
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="payload-request-1",
            x_correlation_id="payload-session",
            conversation_id="payload-session",
            is_final_turn=False,
        )
        first_record = RequestRecord(
            request_info=first_request,
            model_name="test-model",
        )

        response_turn = await mock_worker._process_response(first_record)
        assert response_turn is not None
        session.store_response(response_turn)
        session.advance_turn(1)

        second_credit_context = CreditContext(
            credit=Credit(
                id=2,
                phase=CreditPhase.PROFILING,
                conversation_id="payload-session",
                x_correlation_id="payload-session",
                turn_index=1,
                num_turns=2,
                issued_at_ns=1_000_000,
            ),
            drop_perf_ns=2_000_000,
        )
        second_request = mock_worker._create_request_info(
            session=session,
            credit_context=second_credit_context,
            x_request_id="payload-request-2",
        )

        payload = mock_worker.inference_client.endpoint.format_payload(second_request)

        assert payload["messages"][0]["content"] == "first turn"
        assert payload["messages"][1]["content"] == TOKEN * 2
        assert payload["messages"][2]["content"] == "second turn"


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
