# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the pre-tokenized dataset path across in-engine transports.

Focuses on:
- Text model token_ids field: serialization, deserialization, backward compat
- PromptGenerator.generate_token_ids: basic path, hash_ids path, EOS filtering edge cases
- SyntheticDatasetComposer: pre_tokenized=True produces token_ids, False does not
- EngineGenerateEndpoint: format_payload includes input_ids when token_ids set
- Per-transport: input_ids bypasses _messages_to_prompt (non-streaming and streaming)
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.engine_generate import EngineGenerateEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)

# ============================================================
# Shared helpers for transport mocking
# ============================================================


def _make_model_endpoint(scheme: str, streaming: bool = False):
    """Build a ModelEndpointInfo for a given URL scheme."""
    from aiperf.common.enums import ModelSelectionStrategy
    from aiperf.common.models.model_endpoint_info import (
        EndpointInfo,
        ModelEndpointInfo,
        ModelInfo,
        ModelListInfo,
    )

    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_urls=[f"{scheme}://test-model"],
            streaming=streaming,
        ),
    )


def _mock_vllm_modules():
    """Return patched sys.modules dict for vLLM imports."""
    mock_vllm = types.ModuleType("vllm")
    mock_sp_mod = types.ModuleType("vllm.sampling_params")

    class FakeSP:
        def __init__(self, **kwargs):
            pass

    class FakeROK:
        FINAL_ONLY = 2
        DELTA = 1

    mock_vllm.SamplingParams = FakeSP
    mock_vllm.AsyncLLMEngine = MagicMock
    mock_vllm.AsyncEngineArgs = MagicMock
    mock_sp_mod.RequestOutputKind = FakeROK
    return {"vllm": mock_vllm, "vllm.sampling_params": mock_sp_mod}


def _mock_trtllm_modules():
    """Return patched sys.modules dict for TRT-LLM imports."""
    mock_trtllm = types.ModuleType("tensorrt_llm")

    class FakeSP:
        def __init__(self, **kwargs):
            pass

    mock_trtllm.SamplingParams = FakeSP
    return {"tensorrt_llm": mock_trtllm}


# ============================================================
# Text model: token_ids field
# ============================================================


class TestTextTokenIds:
    """Verify the Text model's token_ids field."""

    def test_default_is_none(self) -> None:
        text = Text(contents=["hello"])
        assert text.token_ids is None

    def test_stores_token_ids(self) -> None:
        ids = [10, 20, 30, 40, 50]
        text = Text(contents=["placeholder"], token_ids=ids)
        assert text.token_ids == ids

    def test_serializes_with_token_ids(self) -> None:
        ids = [1, 2, 3]
        text = Text(contents=["x"], token_ids=ids)
        data = text.model_dump()
        assert data["token_ids"] == [1, 2, 3]

    def test_serializes_without_token_ids(self) -> None:
        text = Text(contents=["x"])
        data = text.model_dump()
        assert data["token_ids"] is None

    def test_turn_copy_with_stripped_media_preserves_token_ids(self) -> None:
        ids = [100, 200, 300]
        turn = Turn(texts=[Text(contents=["hello"], token_ids=ids)])
        copied = turn.copy_with_stripped_media()
        assert copied.texts[0].token_ids == ids

    def test_backward_compat_deserialize_without_token_ids(self) -> None:
        """Text serialized before token_ids was added should deserialize with None."""
        legacy_data = {"name": "text", "contents": ["hello world"]}
        text = Text(**legacy_data)
        assert text.token_ids is None
        assert text.contents == ["hello world"]

    def test_round_trip_with_token_ids(self) -> None:
        """Serialize then deserialize preserves token_ids."""
        original = Text(contents=["placeholder"], token_ids=[42, 99, 7])
        data = original.model_dump()
        restored = Text(**data)
        assert restored.token_ids == [42, 99, 7]
        assert restored.contents == ["placeholder"]

    def test_round_trip_without_token_ids(self) -> None:
        """Serialize then deserialize preserves None for token_ids."""
        original = Text(contents=["real text"])
        data = original.model_dump()
        restored = Text(**data)
        assert restored.token_ids is None


# ============================================================
# PromptGenerator: generate_token_ids and EOS filtering
# ============================================================

MOCK_CORPUS = "To be or not to be that is the question\n"


@patch("builtins.open", mock_open(read_data=MOCK_CORPUS))
class TestGenerateTokenIds:
    """Verify generate_token_ids returns list[int] and filters EOS."""

    @pytest.fixture
    def mock_tokenizer(self, mock_tokenizer_cls):
        return mock_tokenizer_cls.from_pretrained("gpt2")

    @pytest.fixture
    def generator(self, mock_tokenizer):
        from aiperf.common.config import (
            InputTokensConfig,
            PrefixPromptConfig,
            PromptConfig,
        )
        from aiperf.dataset.generator.prompt import PromptGenerator

        config = PromptConfig(
            mean=10,
            stddev=0,
            input_tokens=InputTokensConfig(block_size=512),
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        )
        return PromptGenerator(config, mock_tokenizer)

    def test_returns_list_of_ints(self, generator) -> None:
        result = generator.generate_token_ids(mean=5, stddev=0)
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)

    def test_returns_correct_length(self, generator) -> None:
        result = generator.generate_token_ids(mean=5, stddev=0)
        assert len(result) == 5

    def test_eos_tokens_are_replaced(self, generator) -> None:
        """EOS token (id=2 in mock) should be replaced."""
        eos_id = generator.tokenizer.eos_token_id  # 2 in mock
        generator._sample_tokens = MagicMock(return_value=[1, eos_id, 3, eos_id, 5])
        generator.tokenizer._tokenizer.vocab_size = 100

        result = generator.generate_token_ids(mean=5, stddev=0)

        replacement = (eos_id + 1) % 100  # 3
        assert eos_id not in result
        assert result[1] == replacement
        assert result[3] == replacement

    def test_no_eos_tokens_unchanged(self, generator) -> None:
        """When no EOS tokens present, tokens pass through unchanged."""
        original = [10, 11, 12, 13, 14]
        generator._sample_tokens = MagicMock(return_value=list(original))
        generator.tokenizer._tokenizer.vocab_size = 100

        result = generator.generate_token_ids(mean=5, stddev=0)
        assert result == original

    def test_filter_eos_with_none_eos_id(self, generator) -> None:
        """When eos_token_id is None, tokens pass through unchanged."""
        original = [10, 11, 12]
        generator.tokenizer._tokenizer.eos_token_id = None
        result = generator._filter_eos_tokens(original)
        assert result == original

    # ---- EOS filtering edge cases ----

    def test_filter_eos_all_eos_input(self, generator) -> None:
        """All tokens are EOS -- every one should be replaced."""
        eos_id = generator.tokenizer.eos_token_id  # 2
        generator.tokenizer._tokenizer.vocab_size = 100
        all_eos = [eos_id, eos_id, eos_id]
        result = generator._filter_eos_tokens(all_eos)

        replacement = (eos_id + 1) % 100
        assert result == [replacement, replacement, replacement]
        assert eos_id not in result

    def test_filter_eos_single_token_eos(self, generator) -> None:
        """Single-element list containing EOS is replaced."""
        eos_id = generator.tokenizer.eos_token_id
        generator.tokenizer._tokenizer.vocab_size = 50
        result = generator._filter_eos_tokens([eos_id])
        assert result == [(eos_id + 1) % 50]

    def test_filter_eos_single_token_non_eos(self, generator) -> None:
        """Single-element list not containing EOS is unchanged."""
        generator.tokenizer._tokenizer.vocab_size = 50
        result = generator._filter_eos_tokens([7])
        assert result == [7]

    def test_filter_eos_empty_list(self, generator) -> None:
        """Empty input produces empty output."""
        generator.tokenizer._tokenizer.vocab_size = 100
        result = generator._filter_eos_tokens([])
        assert result == []

    def test_filter_eos_vocab_size_one_wraps_around(self, generator) -> None:
        """With vocab_size=1, replacement wraps to 0 via modulo."""
        eos_id = 0
        generator.tokenizer._tokenizer.eos_token_id = eos_id
        generator.tokenizer._tokenizer.vocab_size = 1
        result = generator._filter_eos_tokens([eos_id, eos_id])
        # (0 + 1) % 1 == 0, which is still eos_id -- degenerate but correct behavior
        assert result == [0, 0]

    def test_filter_eos_vocab_size_two(self, generator) -> None:
        """With vocab_size=2 and eos_id=1, replacement is 0."""
        generator.tokenizer._tokenizer.eos_token_id = 1
        generator.tokenizer._tokenizer.vocab_size = 2
        result = generator._filter_eos_tokens([0, 1, 0, 1])
        assert result == [0, 0, 0, 0]

    # ---- hash_ids path ----

    def test_generate_token_ids_with_hash_ids(self, generator) -> None:
        """hash_ids path uses _build_token_sequence and filters EOS."""
        generator.tokenizer._tokenizer.vocab_size = 100
        # 2 blocks with default block_size=512, need mean >= 513 for valid final block
        result = generator.generate_token_ids(mean=1024, hash_ids=[0, 1])
        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(t, int) for t in result)

        # EOS (id=2) must not appear
        eos_id = generator.tokenizer.eos_token_id
        if eos_id is not None:
            assert eos_id not in result

    def test_generate_token_ids_with_hash_ids_caches_blocks(self, generator) -> None:
        """Same hash_id should reuse the same token block."""
        generator.tokenizer._tokenizer.vocab_size = 100
        # First call populates cache for hash_id=42
        result1 = generator.generate_token_ids(mean=5, hash_ids=[42])
        # Second call should reuse from cache
        result2 = generator.generate_token_ids(mean=5, hash_ids=[42])
        assert result1 == result2


# ============================================================
# SyntheticDatasetComposer: pre_tokenized behavior
# ============================================================


class TestSyntheticComposerPreTokenized:
    """Verify pre_tokenized flag controls token_ids generation on Text objects."""

    @pytest.fixture
    def mock_tokenizer(self, mock_tokenizer_cls):
        return mock_tokenizer_cls.from_pretrained("test-model")

    @pytest.fixture
    def _make_config(self):
        """Factory for UserConfig with configurable pre_tokenized flag."""
        from aiperf.common.config import (
            ConversationConfig,
            EndpointConfig,
            InputConfig,
            InputTokensConfig,
            PrefixPromptConfig,
            PromptConfig,
            UserConfig,
        )

        def _factory(pre_tokenized: bool) -> UserConfig:
            return UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    pre_tokenized=pre_tokenized,
                    conversation=ConversationConfig(num_dataset_entries=2),
                    prompt=PromptConfig(
                        input_tokens=InputTokensConfig(mean=8, stddev=0),
                        prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
                    ),
                ),
            )

        return _factory

    @patch("aiperf.dataset.composer.base.ImageGenerator", MagicMock())
    @patch("builtins.open", mock_open(read_data=MOCK_CORPUS))
    def test_pre_tokenized_true_produces_token_ids(
        self, _make_config, mock_tokenizer
    ) -> None:
        """With pre_tokenized=True, Text objects should have token_ids set."""
        from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer

        config = _make_config(pre_tokenized=True)
        mock_tokenizer._tokenizer.vocab_size = 100
        composer = SyntheticDatasetComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        assert len(conversations) == 2
        for conv in conversations:
            for turn in conv.turns:
                for text in turn.texts:
                    assert text.token_ids is not None, "token_ids should be set"
                    assert isinstance(text.token_ids, list)
                    assert len(text.token_ids) > 0
                    assert all(isinstance(t, int) for t in text.token_ids)
                    # Placeholder content should be present
                    assert text.contents == ["<pre-tokenized>"]

    @patch("aiperf.dataset.composer.base.ImageGenerator", MagicMock())
    @patch("builtins.open", mock_open(read_data=MOCK_CORPUS))
    def test_pre_tokenized_false_does_not_produce_token_ids(
        self, _make_config, mock_tokenizer
    ) -> None:
        """With pre_tokenized=False, Text objects should NOT have token_ids."""
        from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer

        config = _make_config(pre_tokenized=False)
        composer = SyntheticDatasetComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        assert len(conversations) == 2
        for conv in conversations:
            for turn in conv.turns:
                for text in turn.texts:
                    assert text.token_ids is None, "token_ids should be None"
                    # Real text content should be generated
                    assert text.contents[0] != "<pre-tokenized>"
                    assert len(text.contents[0]) > 0

    @patch("aiperf.dataset.composer.base.ImageGenerator", MagicMock())
    @patch("builtins.open", mock_open(read_data=MOCK_CORPUS))
    def test_pre_tokenized_token_ids_have_no_eos(
        self, _make_config, mock_tokenizer
    ) -> None:
        """Pre-tokenized token IDs should have EOS tokens filtered out."""
        from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer

        config = _make_config(pre_tokenized=True)
        mock_tokenizer._tokenizer.vocab_size = 100
        composer = SyntheticDatasetComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        eos_id = mock_tokenizer.eos_token_id
        for conv in conversations:
            for turn in conv.turns:
                for text in turn.texts:
                    if eos_id is not None:
                        assert eos_id not in text.token_ids


# ============================================================
# EngineGenerateEndpoint: format_payload with input_ids
# ============================================================


class TestFormatPayloadInputIds:
    """Verify format_payload includes input_ids when token_ids is set."""

    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT, base_url="vllm://org/model"
        )
        return create_endpoint_with_mock_transport(
            EngineGenerateEndpoint, model_endpoint
        ), model_endpoint

    def test_includes_input_ids_when_token_ids_set(self, endpoint) -> None:
        ep, model_endpoint = endpoint
        ids = [10, 20, 30]
        turn = Turn(
            texts=[Text(contents=["placeholder"], token_ids=ids)],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = ep.format_payload(request_info)

        assert "input_ids" in payload
        assert payload["input_ids"] == ids

    def test_no_input_ids_when_token_ids_none(self, endpoint) -> None:
        ep, model_endpoint = endpoint
        turn = Turn(
            texts=[Text(contents=["real text"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = ep.format_payload(request_info)

        assert "input_ids" not in payload

    def test_no_input_ids_when_no_texts(self, endpoint) -> None:
        ep, model_endpoint = endpoint
        turn = Turn(texts=[], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = ep.format_payload(request_info)

        assert "input_ids" not in payload

    def test_uses_last_turn_text(self, endpoint) -> None:
        """When multiple turns, input_ids come from the last turn's last text."""
        ep, model_endpoint = endpoint
        turn1 = Turn(
            texts=[Text(contents=["first"], token_ids=[1, 2])],
            model="test-model",
        )
        turn2 = Turn(
            texts=[Text(contents=["second"], token_ids=[3, 4, 5])],
            model="test-model",
        )
        request_info = create_request_info(
            model_endpoint=model_endpoint, turns=[turn1, turn2]
        )

        payload = ep.format_payload(request_info)

        assert payload["input_ids"] == [3, 4, 5]


# ============================================================
# BaseInEngineTransport: input_ids threading
# ============================================================


class TestBaseTransportInputIds:
    """Verify send_request threads input_ids to _generate."""

    @pytest.fixture
    def transport_and_info(self):
        from aiperf.common.enums import CreditPhase
        from aiperf.common.models import RequestInfo
        from aiperf.plugin.schema.schemas import TransportMetadata
        from aiperf.transports.base_transports import FirstTokenCallback
        from aiperf.transports.in_engine.base_in_engine_transport import (
            BaseInEngineTransport,
        )

        class SpyTransport(BaseInEngineTransport):
            """Captures _generate kwargs for assertion."""

            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.last_generate_kwargs: dict[str, Any] = {}

            @classmethod
            def metadata(cls) -> TransportMetadata:
                return TransportMetadata(transport_type="spy", url_schemes=["spy"])

            async def _init_engine(self) -> None:
                pass

            async def _start_engine(self) -> None:
                pass

            async def _stop_engine(self) -> None:
                pass

            async def _warmup_single(
                self, prompt: str, max_tokens: int, *, streaming: bool
            ) -> None:
                pass

            async def _generate(
                self,
                *,
                messages: list[dict[str, Any]],
                sampling_params: Any,
                request_id: str,
                first_token_callback: FirstTokenCallback | None = None,
                input_ids: list[int] | None = None,
            ) -> tuple[str, int, int, str]:
                self.last_generate_kwargs = {
                    "messages": messages,
                    "sampling_params": sampling_params,
                    "request_id": request_id,
                    "input_ids": input_ids,
                }
                return ("output", 10, 5, "stop")

        model_endpoint = _make_model_endpoint("spy")
        transport = SpyTransport(model_endpoint=model_endpoint)
        request_info = RequestInfo(
            model_endpoint=model_endpoint,
            turns=[],
            endpoint_headers={},
            endpoint_params={},
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="req-001",
            x_correlation_id="corr-001",
            conversation_id="conv-001",
        )
        return transport, request_info

    @pytest.mark.asyncio
    async def test_input_ids_passed_to_generate(self, transport_and_info) -> None:
        transport, request_info = transport_and_info
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
            "input_ids": [10, 20, 30],
        }

        await transport.send_request(request_info, payload)

        assert transport.last_generate_kwargs["input_ids"] == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_input_ids_none_when_not_in_payload(self, transport_and_info) -> None:
        transport, request_info = transport_and_info
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "sampling_params": {},
        }

        await transport.send_request(request_info, payload)

        assert transport.last_generate_kwargs["input_ids"] is None


# ============================================================
# Per-transport: input_ids bypasses _messages_to_prompt
# ============================================================


class TestVLLMTransportInputIds:
    """Verify vLLM transport passes prompt_token_ids when input_ids is set."""

    @pytest.fixture(params=[False, True], ids=["non-streaming", "streaming"])
    def transport(self, request):
        from aiperf.transports.in_engine.vllm_transport import VLLMTransport

        model_endpoint = _make_model_endpoint("vllm", streaming=request.param)
        return VLLMTransport(model_endpoint=model_endpoint)

    @pytest.mark.asyncio
    async def test_input_ids_creates_prompt_dict(self, transport) -> None:
        """When input_ids is provided, prompt should be a dict with prompt_token_ids."""
        captured_prompt: dict[str, Any] = {}

        async def mock_generate(prompt, sampling_params, request_id):
            captured_prompt["value"] = prompt

            class Output:
                def __init__(self):
                    self.prompt_token_ids = [1, 2, 3]
                    self.outputs = [
                        type(
                            "C",
                            (),
                            {"text": "hi", "token_ids": [10], "finish_reason": "stop"},
                        )()
                    ]
                    self.finished = True

            yield Output()

        transport._engine = MagicMock()
        transport._engine.generate = mock_generate

        with patch.dict("sys.modules", _mock_vllm_modules()):
            await transport._generate(
                messages=[{"role": "user", "content": "test"}],
                sampling_params={},
                request_id="test-001",
                input_ids=[100, 200, 300],
            )

        assert captured_prompt["value"] == {"prompt_token_ids": [100, 200, 300]}

    @pytest.mark.asyncio
    async def test_none_input_ids_uses_text_prompt(self, transport) -> None:
        """When input_ids is None, prompt should be a string from _messages_to_prompt."""
        captured_prompt: dict[str, Any] = {}

        async def mock_generate(prompt, sampling_params, request_id):
            captured_prompt["value"] = prompt

            class Output:
                def __init__(self):
                    self.prompt_token_ids = [1, 2]
                    self.outputs = [
                        type(
                            "C",
                            (),
                            {"text": "hi", "token_ids": [10], "finish_reason": "stop"},
                        )()
                    ]
                    self.finished = True

            yield Output()

        transport._engine = MagicMock()
        transport._engine.generate = mock_generate
        transport._engine.get_tokenizer.return_value.apply_chat_template.return_value = "<|user|>test<|assistant|>"

        with patch.dict("sys.modules", _mock_vllm_modules()):
            await transport._generate(
                messages=[{"role": "user", "content": "test"}],
                sampling_params={},
                request_id="test-002",
                input_ids=None,
            )

        assert isinstance(captured_prompt["value"], str)

    @pytest.mark.asyncio
    async def test_messages_to_prompt_not_called_with_input_ids(
        self, transport
    ) -> None:
        """_messages_to_prompt must NOT be called when input_ids are provided."""
        call_tracker = {"called": False}
        original_m2p = transport._messages_to_prompt

        def tracking_m2p(messages):
            call_tracker["called"] = True
            return original_m2p(messages)

        transport._messages_to_prompt = tracking_m2p

        async def mock_generate(prompt, sampling_params, request_id):
            class Output:
                def __init__(self):
                    self.prompt_token_ids = [1, 2, 3]
                    self.outputs = [
                        type(
                            "C",
                            (),
                            {"text": "hi", "token_ids": [10], "finish_reason": "stop"},
                        )()
                    ]

            yield Output()

        transport._engine = MagicMock()
        transport._engine.generate = mock_generate

        with patch.dict("sys.modules", _mock_vllm_modules()):
            await transport._generate(
                messages=[{"role": "user", "content": "test"}],
                sampling_params={},
                request_id="test-003",
                input_ids=[1, 2, 3],
            )

        assert not call_tracker["called"], "_messages_to_prompt should not be called"


class TestSGLangTransportInputIds:
    """Verify SGLang transport passes input_ids kwarg when provided."""

    @pytest.fixture(params=[False, True], ids=["non-streaming", "streaming"])
    def transport(self, request):
        from aiperf.transports.in_engine.sglang_transport import SGLangTransport

        model_endpoint = _make_model_endpoint("sglang", streaming=request.param)
        return SGLangTransport(model_endpoint=model_endpoint)

    @pytest.mark.asyncio
    async def test_input_ids_passed_to_async_generate(self, transport) -> None:
        """When input_ids is provided, async_generate should receive input_ids kwarg."""
        captured_kwargs: dict[str, Any] = {}
        streaming = transport.model_endpoint.endpoint.streaming

        if streaming:

            async def mock_async_generate(**kwargs):
                captured_kwargs.update(kwargs)

                async def _stream():
                    yield {
                        "text": "output",
                        "meta_info": {
                            "prompt_tokens": 3,
                            "completion_tokens": 5,
                            "finish_reason": {"type": "stop"},
                        },
                    }

                return _stream()
        else:

            async def mock_async_generate(**kwargs):
                captured_kwargs.update(kwargs)
                return {
                    "text": "output",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 5,
                        "finish_reason": {"type": "stop"},
                    },
                }

        transport._engine = MagicMock()
        transport._engine.async_generate = mock_async_generate

        await transport._generate(
            messages=[{"role": "user", "content": "test"}],
            sampling_params={},
            request_id="test-001",
            input_ids=[10, 20, 30],
        )

        assert captured_kwargs.get("input_ids") == [10, 20, 30]
        assert "prompt" not in captured_kwargs

    @pytest.mark.asyncio
    async def test_none_input_ids_uses_prompt(self, transport) -> None:
        """When input_ids is None, async_generate should receive prompt string."""
        captured_kwargs: dict[str, Any] = {}
        streaming = transport.model_endpoint.endpoint.streaming

        if streaming:

            async def mock_async_generate(**kwargs):
                captured_kwargs.update(kwargs)

                async def _stream():
                    yield {
                        "text": "output",
                        "meta_info": {
                            "prompt_tokens": 3,
                            "completion_tokens": 5,
                            "finish_reason": {"type": "stop"},
                        },
                    }

                return _stream()
        else:

            async def mock_async_generate(**kwargs):
                captured_kwargs.update(kwargs)
                return {
                    "text": "output",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 5,
                        "finish_reason": {"type": "stop"},
                    },
                }

        transport._engine = MagicMock()
        transport._engine.async_generate = mock_async_generate
        transport._engine.tokenizer_manager.tokenizer.apply_chat_template.return_value = "<|user|>test<|assistant|>"

        await transport._generate(
            messages=[{"role": "user", "content": "test"}],
            sampling_params={},
            request_id="test-002",
            input_ids=None,
        )

        assert "prompt" in captured_kwargs
        assert isinstance(captured_kwargs["prompt"], str)
        assert "input_ids" not in captured_kwargs

    @pytest.mark.asyncio
    async def test_messages_to_prompt_not_called_with_input_ids(
        self, transport
    ) -> None:
        """_messages_to_prompt must NOT be called when input_ids are provided."""
        call_tracker = {"called": False}
        original_m2p = transport._messages_to_prompt

        def tracking_m2p(messages):
            call_tracker["called"] = True
            return original_m2p(messages)

        transport._messages_to_prompt = tracking_m2p
        streaming = transport.model_endpoint.endpoint.streaming

        if streaming:

            async def mock_async_generate(**kwargs):
                async def _stream():
                    yield {
                        "text": "output",
                        "meta_info": {
                            "prompt_tokens": 3,
                            "completion_tokens": 5,
                            "finish_reason": {"type": "stop"},
                        },
                    }

                return _stream()
        else:

            async def mock_async_generate(**kwargs):
                return {
                    "text": "output",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 5,
                        "finish_reason": {"type": "stop"},
                    },
                }

        transport._engine = MagicMock()
        transport._engine.async_generate = mock_async_generate

        await transport._generate(
            messages=[{"role": "user", "content": "test"}],
            sampling_params={},
            request_id="test-003",
            input_ids=[1, 2, 3],
        )

        assert not call_tracker["called"], "_messages_to_prompt should not be called"


class TestTRTLLMTransportInputIds:
    """Verify TRT-LLM transport passes list[int] directly when input_ids is set."""

    @pytest.fixture(params=[False, True], ids=["non-streaming", "streaming"])
    def transport(self, request):
        from aiperf.transports.in_engine.trtllm_transport import TRTLLMTransport

        model_endpoint = _make_model_endpoint("trtllm", streaming=request.param)
        return TRTLLMTransport(model_endpoint=model_endpoint)

    def _make_mock_output(self, prompt_token_ids=None):
        """Create a mock TRT-LLM output object."""

        class MockCompletion:
            text = "output"
            token_ids = [100, 101]
            finish_reason = "stop"

        class MockOutput:
            def __init__(self):
                self.prompt_token_ids = prompt_token_ids or [10, 20, 30]
                self.outputs = [MockCompletion()]
                self.decoding_iter = None

            async def aresult(self):
                return self

        return MockOutput

    @pytest.mark.asyncio
    async def test_input_ids_passed_as_prompt(self, transport) -> None:
        """When input_ids is provided, generate_async receives list[int] as prompt."""
        captured_prompt: dict[str, Any] = {}
        streaming = transport.model_endpoint.endpoint.streaming
        MockOutput = self._make_mock_output()

        if streaming:

            async def mock_generate_async(prompt, params, streaming=False):
                captured_prompt["value"] = prompt
                yield MockOutput()
        else:

            def mock_generate_async(prompt, params, streaming=False):
                captured_prompt["value"] = prompt
                return MockOutput()

        transport._engine = MagicMock()
        transport._engine.generate_async = mock_generate_async

        with patch.dict("sys.modules", _mock_trtllm_modules()):
            await transport._generate(
                messages=[{"role": "user", "content": "test"}],
                sampling_params={},
                request_id="test-001",
                input_ids=[10, 20, 30],
            )

        assert captured_prompt["value"] == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_none_input_ids_uses_string_prompt(self, transport) -> None:
        """When input_ids is None, generate_async receives a string prompt."""
        captured_prompt: dict[str, Any] = {}
        streaming = transport.model_endpoint.endpoint.streaming
        MockOutput = self._make_mock_output(prompt_token_ids=[1, 2])

        if streaming:

            async def mock_generate_async(prompt, params, streaming=False):
                captured_prompt["value"] = prompt
                yield MockOutput()
        else:

            def mock_generate_async(prompt, params, streaming=False):
                captured_prompt["value"] = prompt
                return MockOutput()

        transport._engine = MagicMock()
        transport._engine.generate_async = mock_generate_async
        transport._engine.tokenizer.apply_chat_template.return_value = (
            "<|user|>test<|assistant|>"
        )

        with patch.dict("sys.modules", _mock_trtllm_modules()):
            await transport._generate(
                messages=[{"role": "user", "content": "test"}],
                sampling_params={},
                request_id="test-002",
                input_ids=None,
            )

        assert isinstance(captured_prompt["value"], str)

    @pytest.mark.asyncio
    async def test_messages_to_prompt_not_called_with_input_ids(
        self, transport
    ) -> None:
        """_messages_to_prompt must NOT be called when input_ids are provided."""
        call_tracker = {"called": False}
        original_m2p = transport._messages_to_prompt

        def tracking_m2p(messages):
            call_tracker["called"] = True
            return original_m2p(messages)

        transport._messages_to_prompt = tracking_m2p
        streaming = transport.model_endpoint.endpoint.streaming
        MockOutput = self._make_mock_output()

        if streaming:

            async def mock_generate_async(prompt, params, streaming=False):
                yield MockOutput()
        else:

            def mock_generate_async(prompt, params, streaming=False):
                return MockOutput()

        transport._engine = MagicMock()
        transport._engine.generate_async = mock_generate_async

        with patch.dict("sys.modules", _mock_trtllm_modules()):
            await transport._generate(
                messages=[{"role": "user", "content": "test"}],
                sampling_params={},
                request_id="test-003",
                input_ids=[1, 2, 3],
            )

        assert not call_tracker["called"], "_messages_to_prompt should not be called"
