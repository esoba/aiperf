# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    SynthesisConfig,
    UserConfig,
)
from aiperf.dataset.loader.models import CodingTrace, CodingTraceRequest
from aiperf.plugin.enums import CustomDatasetType


class TestCodingTraceRequest:
    def test_create_request(self):
        req = CodingTraceRequest.model_validate(
            {"t": 0.0, "type": "s", "in": 1000, "out": 500, "hash_ids": [1, 2, 3]}
        )
        assert req.t == 0.0
        assert req.type == "s"
        assert req.input_tokens == 1000
        assert req.output_tokens == 500
        assert req.hash_ids == [1, 2, 3]

    def test_alias_fields(self):
        req = CodingTraceRequest.model_validate(
            {"t": 1.0, "type": "n", "in": 500, "out": 200}
        )
        assert req.input_tokens == 500
        assert req.output_tokens == 200

    def test_nested_requests(self):
        data = {
            "t": 0.0,
            "type": "s",
            "in": 1000,
            "out": 500,
            "requests": [
                {"t": 1.0, "type": "n", "in": 200, "out": 100},
                {"t": 2.0, "type": "n", "in": 300, "out": 150},
            ],
        }
        req = CodingTraceRequest.model_validate(data)
        assert len(req.requests) == 2
        assert req.requests[0].input_tokens == 200

    def test_optional_fields_default(self):
        req = CodingTraceRequest.model_validate(
            {"t": 0.0, "type": "s", "in": 100, "out": 50}
        )
        assert req.hash_ids == []
        assert req.input_types == []
        assert req.output_types == []
        assert req.stop is None
        assert req.model is None
        assert req.requests == []

    def test_subagent_container_no_tokens(self):
        """Subagent wrapper entries have no in/out fields - defaults to 0."""
        req = CodingTraceRequest.model_validate(
            {
                "t": 0.0,
                "type": "subagent",
                "requests": [
                    {"t": 0.1, "type": "s", "in": 100, "out": 50},
                ],
            }
        )
        assert req.input_tokens == 0
        assert req.output_tokens == 0
        assert len(req.requests) == 1


class TestCodingTrace:
    def test_create_trace(self):
        trace = CodingTrace.model_validate(
            {
                "id": "trace_001",
                "models": ["claude-sonnet"],
                "block_size": 64,
                "tool_tokens": 5000,
                "system_tokens": 3000,
                "requests": [
                    {"t": 0.0, "type": "s", "in": 1000, "out": 500},
                ],
            }
        )
        assert trace.id == "trace_001"
        assert trace.models == ["claude-sonnet"]
        assert trace.block_size == 64
        assert trace.tool_tokens == 5000
        assert trace.system_tokens == 3000
        assert len(trace.requests) == 1
        assert trace.type == CustomDatasetType.CODING_TRACE

    def test_empty_requests_raises(self):
        with pytest.raises(ValidationError, match="at least one request"):
            CodingTrace.model_validate(
                {
                    "id": "trace_001",
                    "requests": [],
                }
            )

    def test_default_values(self):
        trace = CodingTrace.model_validate(
            {
                "id": "trace_001",
                "requests": [{"t": 0.0, "type": "s", "in": 100, "out": 50}],
            }
        )
        assert trace.models == []
        assert trace.block_size == 64
        assert trace.tool_tokens == 0
        assert trace.system_tokens == 0


class TestCodingTraceLoader:
    @pytest.fixture
    def trace_dir(self, tmp_path):
        """Create a temporary directory with sample trace files.

        Uses growing input_tokens to exercise delta computation:
        turn 0: in=1000, out=500  -> delta=1000
        turn 1: in=3000, out=1000 -> delta=max(1, 3000-1000-500)=1500
        turn 2: in=8000, out=2000 -> delta=max(1, 8000-3000-1000)=4000
        """
        for i in range(3):
            trace = {
                "id": f"trace_{i:04d}",
                "models": ["test-model"],
                "block_size": 64,
                "tool_tokens": 5000,
                "system_tokens": 3000,
                "requests": [
                    {"t": 0.0, "type": "s", "in": 1000, "out": 500},
                    {"t": 15.0, "type": "n", "in": 3000, "out": 1000},
                    {"t": 30.0, "type": "n", "in": 8000, "out": 2000},
                ],
            }
            (tmp_path / f"trace_{i:04d}.json").write_text(json.dumps(trace))
        return tmp_path

    @pytest.fixture
    def single_trace_file(self, tmp_path):
        """Create a single trace JSON file."""
        trace = {
            "id": "single_trace",
            "block_size": 64,
            "tool_tokens": 2000,
            "system_tokens": 1000,
            "requests": [
                {"t": 0.0, "type": "s", "in": 500, "out": 200},
                {"t": 10.0, "type": "n", "in": 800, "out": 300},
            ],
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace))
        return path

    @pytest.fixture
    def mock_prompt_generator(self):
        gen = MagicMock()
        gen.tokenizer.resolved_name = "test-tokenizer"
        gen.generate = MagicMock(return_value="synthetic text prompt")
        gen.generate_prompt = MagicMock(return_value="x" * 400)
        return gen

    @pytest.fixture
    def default_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                warm_prefix_pct=0.0,
            ),
        )

    def test_can_load_directory(self, trace_dir):
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        assert CodingTraceLoader.can_load(filename=str(trace_dir))

    def test_can_load_data_dict(self):
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        data = {
            "id": "trace_001",
            "requests": [{"t": 0, "type": "s", "in": 100, "out": 50}],
        }
        assert CodingTraceLoader.can_load(data=data)

    def test_can_load_rejects_non_trace(self):
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        assert not CodingTraceLoader.can_load(data={"text": "hello"})
        assert not CodingTraceLoader.can_load(data=None, filename=None)

    def test_load_directory(
        self, trace_dir, mock_prompt_generator, default_user_config
    ):
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        loader = CodingTraceLoader(
            filename=str(trace_dir),
            prompt_generator=mock_prompt_generator,
            user_config=default_user_config,
        )
        data = loader.load_dataset()
        assert len(data) == 3
        for _conv_id, traces in data.items():
            assert len(traces) == 1
            assert len(traces[0].requests) == 3

    def test_load_single_file(
        self, single_trace_file, mock_prompt_generator, default_user_config
    ):
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        loader = CodingTraceLoader(
            filename=str(single_trace_file),
            prompt_generator=mock_prompt_generator,
            user_config=default_user_config,
        )
        data = loader.load_dataset()
        assert len(data) == 1
        assert "single_trace" in data

    def test_convert_to_conversations(
        self, trace_dir, mock_prompt_generator, default_user_config
    ):
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        loader = CodingTraceLoader(
            filename=str(trace_dir),
            prompt_generator=mock_prompt_generator,
            user_config=default_user_config,
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 3
        for conv in conversations:
            assert len(conv.turns) == 3
            # First turn has no delay
            assert conv.turns[0].delay is None
            # Subsequent turns have delays computed from timestamps
            assert conv.turns[1].delay == 15.0 * 1000  # 15s -> 15000ms
            assert conv.turns[2].delay == 15.0 * 1000  # 30-15=15s -> 15000ms

    def test_convert_prompt_truncation(
        self, trace_dir, mock_prompt_generator, default_user_config
    ):
        """Prompts are truncated by character ratio from a single base prompt.

        With trace_dir requests in=[1000, 3000, 8000], out=[500, 1000, 2000]:
        deltas are [1000, 1500, 4000], so max delta = 4000.
        generate_prompt is called once with max_delta.
        """
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        mock_prompt_generator.generate_prompt.return_value = "a" * 600

        loader = CodingTraceLoader(
            filename=str(trace_dir),
            prompt_generator=mock_prompt_generator,
            user_config=default_user_config,
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        # generate_prompt called once with the max delta (4000)
        mock_prompt_generator.generate_prompt.assert_called_once_with(4000)
        # All turns should have non-empty prompts
        for conv in conversations:
            for turn in conv.turns:
                assert len(turn.texts[0].contents[0]) > 0

    def test_warm_prefix_generation(self, trace_dir, mock_prompt_generator):
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        mock_prompt_generator.generate.return_value = "warm prefix text"

        input_config = InputConfig.model_construct(
            prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
            warm_prefix_pct=0.5,
            file=str(trace_dir),
        )
        config = UserConfig.model_construct(
            endpoint=EndpointConfig.model_construct(model_names=["test-model"]),
            input=input_config,
        )

        loader = CodingTraceLoader(
            filename=str(trace_dir),
            prompt_generator=mock_prompt_generator,
            user_config=config,
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            assert conv.system_message == "warm prefix text"

    def test_flatten_requests(self, mock_prompt_generator, default_user_config):
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        loader = CodingTraceLoader(
            filename="/tmp/fake",
            prompt_generator=mock_prompt_generator,
            user_config=default_user_config,
        )

        nested = [
            CodingTraceRequest.model_validate(
                {
                    "t": 0.0,
                    "type": "s",
                    "in": 1000,
                    "out": 500,
                    "requests": [
                        {"t": 1.0, "type": "n", "in": 200, "out": 100},
                        {
                            "t": 2.0,
                            "type": "n",
                            "in": 300,
                            "out": 150,
                            "requests": [
                                {"t": 3.0, "type": "n", "in": 50, "out": 25},
                            ],
                        },
                    ],
                }
            )
        ]

        flat = loader._flatten_requests(nested)
        assert len(flat) == 4
        assert flat[0].input_tokens == 1000
        assert flat[1].input_tokens == 200
        assert flat[2].input_tokens == 300
        assert flat[3].input_tokens == 50

    def test_flatten_skips_container_entries(
        self, mock_prompt_generator, default_user_config
    ):
        """Subagent containers with input_tokens=0 are skipped."""
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        loader = CodingTraceLoader(
            filename="/tmp/fake",
            prompt_generator=mock_prompt_generator,
            user_config=default_user_config,
        )

        nested = [
            CodingTraceRequest.model_validate(
                {
                    "t": 0.0,
                    "type": "subagent",
                    "requests": [
                        {"t": 0.1, "type": "s", "in": 100, "out": 50},
                    ],
                }
            )
        ]

        flat = loader._flatten_requests(nested)
        assert len(flat) == 1
        assert flat[0].input_tokens == 100

    def test_max_isl_truncation(self, tmp_path, mock_prompt_generator):
        """Max ISL truncates at first exceeding request, not per-request filter.

        Trace: in=[100, 5000, 200]. With max_isl=1000, the conversation is
        truncated at in=5000, so only the first request (in=100) remains.
        The request with in=200 after the exceeding one is NOT kept.
        """
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        trace = {
            "id": "trace_isl",
            "block_size": 64,
            "requests": [
                {"t": 0.0, "type": "s", "in": 100, "out": 50},
                {"t": 1.0, "type": "n", "in": 5000, "out": 200},
                {"t": 2.0, "type": "n", "in": 200, "out": 100},
            ],
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace))

        config = UserConfig.model_construct(
            endpoint=EndpointConfig.model_construct(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                warm_prefix_pct=0.0,
                synthesis=SynthesisConfig(max_isl=1000),
                custom_dataset_type=CustomDatasetType.CODING_TRACE,
                file=str(path),
            ),
        )

        loader = CodingTraceLoader(
            filename=str(path),
            prompt_generator=mock_prompt_generator,
            user_config=config,
        )
        data = loader.load_dataset()
        # Only 1 request remains (in=100), which is < 2, so trace is skipped
        assert len(data) == 0

    def test_max_isl_truncation_keeps_prefix(self, tmp_path, mock_prompt_generator):
        """Truncation keeps all requests before the first exceeding one."""
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        trace = {
            "id": "trace_isl",
            "block_size": 64,
            "requests": [
                {"t": 0.0, "type": "s", "in": 100, "out": 50},
                {"t": 1.0, "type": "n", "in": 500, "out": 200},
                {"t": 2.0, "type": "n", "in": 900, "out": 300},
                {"t": 3.0, "type": "n", "in": 5000, "out": 200},
                {"t": 4.0, "type": "n", "in": 200, "out": 100},
            ],
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace))

        config = UserConfig.model_construct(
            endpoint=EndpointConfig.model_construct(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                warm_prefix_pct=0.0,
                synthesis=SynthesisConfig(max_isl=1000),
                custom_dataset_type=CustomDatasetType.CODING_TRACE,
                file=str(path),
            ),
        )

        loader = CodingTraceLoader(
            filename=str(path),
            prompt_generator=mock_prompt_generator,
            user_config=config,
        )
        data = loader.load_dataset()
        assert len(data) == 1
        trace_loaded = list(data.values())[0][0]
        # Requests with in=[100, 500, 900] kept, in=5000 and in=200 truncated
        assert len(trace_loaded.requests) == 3
        assert trace_loaded.requests[-1].input_tokens == 900

    def test_delta_prompt_sizing(self, tmp_path, mock_prompt_generator):
        """Verify delta computation produces correctly sized prompts.

        Trace: in=[1000, 3000, 8000], out=[500, 1000, 2000]
        Expected deltas: [1000, max(1, 3000-1000-500)=1500, max(1, 8000-3000-1000)=4000]

        With base prompt of 4000 chars for max_delta=4000 (1 char/token),
        prompts should be sized proportionally to deltas.
        """
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        trace = {
            "id": "trace_delta",
            "block_size": 64,
            "requests": [
                {"t": 0.0, "type": "s", "in": 1000, "out": 500},
                {"t": 10.0, "type": "n", "in": 3000, "out": 1000},
                {"t": 20.0, "type": "n", "in": 8000, "out": 2000},
            ],
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace))

        # 1 char per token for easy math
        mock_prompt_generator.generate_prompt.return_value = "x" * 4000

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                warm_prefix_pct=0.0,
            ),
        )

        loader = CodingTraceLoader(
            filename=str(path),
            prompt_generator=mock_prompt_generator,
            user_config=config,
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conv = conversations[0]

        # generate_prompt called with max_delta=4000
        mock_prompt_generator.generate_prompt.assert_called_once_with(4000)

        # Prompt lengths should match deltas (1 char/token ratio)
        assert len(conv.turns[0].texts[0].contents[0]) == 1000  # delta=1000
        assert len(conv.turns[1].texts[0].contents[0]) == 1500  # delta=1500
        assert len(conv.turns[2].texts[0].contents[0]) == 4000  # delta=4000

    def test_delta_floor_at_one(self, tmp_path, mock_prompt_generator):
        """Delta floors at 1 when model output covers the growth.

        If input_tokens barely grows but prev output was large, delta would
        be negative. We floor at 1 to avoid empty prompts.

        Trace: in=[1000, 1200], out=[500, 300]
        Turn 1 delta: max(1, 1200 - 1000 - 500) = max(1, -300) = 1
        """
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        trace = {
            "id": "trace_floor",
            "block_size": 64,
            "requests": [
                {"t": 0.0, "type": "s", "in": 1000, "out": 500},
                {"t": 10.0, "type": "n", "in": 1200, "out": 300},
            ],
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace))

        # 1 char per token
        mock_prompt_generator.generate_prompt.return_value = "x" * 1000

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                warm_prefix_pct=0.0,
            ),
        )

        loader = CodingTraceLoader(
            filename=str(path),
            prompt_generator=mock_prompt_generator,
            user_config=config,
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        conv = conversations[0]
        # Turn 0: delta=1000, turn 1: delta=1 (floored)
        assert len(conv.turns[0].texts[0].contents[0]) == 1000
        # delta=1 -> char_count = max(1, int(1 * 1.0)) = 1
        assert len(conv.turns[1].texts[0].contents[0]) == 1

    def test_max_isl_truncation_mid_conversation(self, tmp_path, mock_prompt_generator):
        """Longer trace showing truncation preserves conversation prefix.

        7 requests with growing context. max_isl=6000 truncates at request 5
        (in=7000), keeping the first 4 requests.
        """
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        trace = {
            "id": "trace_long",
            "block_size": 64,
            "requests": [
                {"t": 0.0, "type": "s", "in": 500, "out": 200},
                {"t": 5.0, "type": "n", "in": 1000, "out": 400},
                {"t": 10.0, "type": "n", "in": 2000, "out": 800},
                {"t": 15.0, "type": "n", "in": 4000, "out": 1500},
                {"t": 20.0, "type": "n", "in": 7000, "out": 2000},
                {"t": 25.0, "type": "n", "in": 10000, "out": 3000},
                {"t": 30.0, "type": "n", "in": 15000, "out": 4000},
            ],
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace))

        config = UserConfig.model_construct(
            endpoint=EndpointConfig.model_construct(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                warm_prefix_pct=0.0,
                synthesis=SynthesisConfig(max_isl=6000),
                custom_dataset_type=CustomDatasetType.CODING_TRACE,
                file=str(path),
            ),
        )

        loader = CodingTraceLoader(
            filename=str(path),
            prompt_generator=mock_prompt_generator,
            user_config=config,
        )
        data = loader.load_dataset()
        assert len(data) == 1
        trace_loaded = list(data.values())[0][0]
        # Requests with in=[500, 1000, 2000, 4000] kept (all <= 6000)
        # in=7000 exceeds, so truncated along with everything after
        assert len(trace_loaded.requests) == 4
        assert [r.input_tokens for r in trace_loaded.requests] == [
            500,
            1000,
            2000,
            4000,
        ]

    def test_preferred_sampling_strategy(self):
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader
        from aiperf.plugin.enums import DatasetSamplingStrategy

        assert (
            CodingTraceLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    def test_configurable_min_requests(self, tmp_path, mock_prompt_generator):
        """Traces with fewer than min_requests are skipped after flattening."""
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        trace = {
            "id": "trace_short",
            "block_size": 64,
            "requests": [
                {"t": 0.0, "type": "s", "in": 100, "out": 50},
                {"t": 1.0, "type": "n", "in": 200, "out": 100},
                {"t": 2.0, "type": "n", "in": 300, "out": 150},
            ],
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace))

        config = UserConfig.model_construct(
            endpoint=EndpointConfig.model_construct(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                warm_prefix_pct=0.0,
                synthesis=SynthesisConfig(min_requests=5),
                custom_dataset_type=CustomDatasetType.CODING_TRACE,
                file=str(path),
            ),
        )

        loader = CodingTraceLoader(
            filename=str(path),
            prompt_generator=mock_prompt_generator,
            user_config=config,
        )
        data = loader.load_dataset()
        # Trace has 3 requests but min_requests=5, so it's skipped
        assert len(data) == 0

    def test_configurable_min_requests_passes(self, tmp_path, mock_prompt_generator):
        """Traces meeting min_requests threshold are kept."""
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        trace = {
            "id": "trace_ok",
            "block_size": 64,
            "requests": [
                {"t": 0.0, "type": "s", "in": 100, "out": 50},
                {"t": 1.0, "type": "n", "in": 200, "out": 100},
                {"t": 2.0, "type": "n", "in": 300, "out": 150},
            ],
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace))

        config = UserConfig.model_construct(
            endpoint=EndpointConfig.model_construct(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                warm_prefix_pct=0.0,
                synthesis=SynthesisConfig(min_requests=3),
                custom_dataset_type=CustomDatasetType.CODING_TRACE,
                file=str(path),
            ),
        )

        loader = CodingTraceLoader(
            filename=str(path),
            prompt_generator=mock_prompt_generator,
            user_config=config,
        )
        data = loader.load_dataset()
        assert len(data) == 1

    def test_warm_prefix_reduces_first_turn_delta(
        self, tmp_path, mock_prompt_generator
    ):
        """First-turn delta is reduced by prefix_tokens when warm prefix is enabled.

        Trace: in=[1000, 3000], out=[500, 1000]
        tool_tokens=5000, system_tokens=3000 -> context=8000
        warm_prefix_pct=0.5 -> prefix_tokens=4000

        Turn 0 delta: max(1, 1000 - 4000) = 1  (prefix covers first turn)
        Turn 1 delta: max(1, 3000 - 1000 - 500) = 1500  (no prefix effect)
        """
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        trace = {
            "id": "trace_prefix",
            "block_size": 64,
            "tool_tokens": 5000,
            "system_tokens": 3000,
            "requests": [
                {"t": 0.0, "type": "s", "in": 1000, "out": 500},
                {"t": 10.0, "type": "n", "in": 3000, "out": 1000},
            ],
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace))

        # 1 char per token
        mock_prompt_generator.generate_prompt.return_value = "x" * 1500
        mock_prompt_generator.generate.return_value = "warm prefix text"

        config = UserConfig.model_construct(
            endpoint=EndpointConfig.model_construct(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                warm_prefix_pct=0.5,
                file=str(path),
            ),
        )

        loader = CodingTraceLoader(
            filename=str(path),
            prompt_generator=mock_prompt_generator,
            user_config=config,
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        conv = conversations[0]
        # Turn 0: delta = max(1, 1000 - 4000) = 1 (floored)
        assert len(conv.turns[0].texts[0].contents[0]) == 1
        # Turn 1: delta = max(1, 3000 - 1000 - 500) = 1500 (unaffected by prefix)
        assert len(conv.turns[1].texts[0].contents[0]) == 1500

    def test_warm_prefix_smaller_than_first_turn(self, tmp_path, mock_prompt_generator):
        """First-turn delta is partially reduced when prefix < input_tokens.

        Trace: in=[1000, 3000], out=[500, 1000]
        tool_tokens=1000, system_tokens=1000 -> context=2000
        warm_prefix_pct=0.25 -> prefix_tokens=500

        Turn 0 delta: max(1, 1000 - 500) = 500
        Turn 1 delta: max(1, 3000 - 1000 - 500) = 1500
        max_delta=1500, chars_per_token = 1500/1500 = 1.0
        """
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        trace = {
            "id": "trace_small_prefix",
            "block_size": 64,
            "tool_tokens": 1000,
            "system_tokens": 1000,
            "requests": [
                {"t": 0.0, "type": "s", "in": 1000, "out": 500},
                {"t": 10.0, "type": "n", "in": 3000, "out": 1000},
            ],
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace))

        # 1 char per token
        mock_prompt_generator.generate_prompt.return_value = "x" * 1500
        mock_prompt_generator.generate.return_value = "warm prefix text"

        config = UserConfig.model_construct(
            endpoint=EndpointConfig.model_construct(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                warm_prefix_pct=0.25,
                file=str(path),
            ),
        )

        loader = CodingTraceLoader(
            filename=str(path),
            prompt_generator=mock_prompt_generator,
            user_config=config,
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        conv = conversations[0]
        # Turn 0: delta = max(1, 1000 - 500) = 500
        assert len(conv.turns[0].texts[0].contents[0]) == 500
        # Turn 1: delta = max(1, 3000 - 1000 - 500) = 1500
        assert len(conv.turns[1].texts[0].contents[0]) == 1500

    def test_flatten_converts_subagent_timestamps_to_absolute(
        self, mock_prompt_generator, default_user_config
    ):
        """Subagent timestamps are converted from relative to absolute time.

        Main: t=10.0
          Subagent container: t=5.0 (abs: 10+5=15)
            Child: t=1.0 (abs: 15+1=16)
            Child: t=3.0 (abs: 15+3=18)
        """
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        loader = CodingTraceLoader(
            filename="/tmp/fake",
            prompt_generator=mock_prompt_generator,
            user_config=default_user_config,
        )

        nested = [
            CodingTraceRequest.model_validate(
                {
                    "t": 10.0,
                    "type": "s",
                    "in": 1000,
                    "out": 500,
                    "requests": [
                        {
                            "t": 5.0,
                            "type": "subagent",
                            "requests": [
                                {"t": 1.0, "type": "n", "in": 200, "out": 100},
                                {"t": 3.0, "type": "n", "in": 300, "out": 150},
                            ],
                        },
                    ],
                }
            )
        ]

        flat = loader._flatten_requests(nested)
        assert len(flat) == 3
        # Results sorted by absolute time
        assert flat[0].t == 10.0  # main: 0 + 10
        assert flat[0].input_tokens == 1000
        assert flat[1].t == 16.0  # 0 + 10 + 5 + 1
        assert flat[1].input_tokens == 200
        assert flat[2].t == 18.0  # 0 + 10 + 5 + 3
        assert flat[2].input_tokens == 300

    def test_flatten_sorts_by_absolute_time(
        self, mock_prompt_generator, default_user_config
    ):
        """Flattened requests are sorted chronologically by absolute time."""
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        loader = CodingTraceLoader(
            filename="/tmp/fake",
            prompt_generator=mock_prompt_generator,
            user_config=default_user_config,
        )

        # Two top-level requests where the second appears earlier in absolute time
        # than nested children of the first
        nested = [
            CodingTraceRequest.model_validate(
                {
                    "t": 0.0,
                    "type": "s",
                    "in": 100,
                    "out": 50,
                    "requests": [
                        {"t": 10.0, "type": "n", "in": 200, "out": 100},
                    ],
                }
            ),
            CodingTraceRequest.model_validate(
                {"t": 5.0, "type": "n", "in": 300, "out": 150}
            ),
        ]

        flat = loader._flatten_requests(nested)
        assert len(flat) == 3
        # Should be sorted: t=0, t=5, t=10
        assert flat[0].t == 0.0
        assert flat[1].t == 5.0
        assert flat[2].t == 10.0

    def test_compute_trace_statistics(self, mock_prompt_generator, default_user_config):
        """Verify trace statistics computation."""
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        trace = CodingTrace.model_validate(
            {
                "id": "trace_stats",
                "block_size": 64,
                "requests": [
                    {
                        "t": 0.0,
                        "type": "s",
                        "in": 1000,
                        "out": 500,
                        "hash_ids": [1, 2, 3, 4, 5],
                    },
                    {
                        "t": 1.0,
                        "type": "n",
                        "in": 2000,
                        "out": 800,
                        "hash_ids": [3, 4, 5, 6, 7],
                    },
                ],
            }
        )

        stats = CodingTraceLoader._compute_trace_statistics(trace)
        assert stats.total_input_tokens == 3000
        assert stats.total_output_tokens == 1300
        assert stats.num_requests == 2
        assert stats.max_input_tokens == 2000
        # total_blocks = 5 (first) + 5 (second) = 10, hits = 3 overlap
        assert stats.estimated_cache_hit_ratio == pytest.approx(0.3)

    def test_compute_trace_statistics_no_hash_ids(
        self, mock_prompt_generator, default_user_config
    ):
        """Cache hit ratio is 0 when no hash_ids are present."""
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        trace = CodingTrace.model_validate(
            {
                "id": "trace_no_hash",
                "block_size": 64,
                "requests": [
                    {"t": 0.0, "type": "s", "in": 100, "out": 50},
                    {"t": 1.0, "type": "n", "in": 200, "out": 100},
                ],
            }
        )

        stats = CodingTraceLoader._compute_trace_statistics(trace)
        assert stats.estimated_cache_hit_ratio == 0.0

    def test_detect_request_pairs(self, mock_prompt_generator, default_user_config):
        """Consecutive requests with identical hash_ids and type='n' are marked as pairs."""
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        requests = [
            CodingTraceRequest.model_validate(
                {"t": 0.0, "type": "s", "in": 1000, "out": 500, "hash_ids": [1, 2, 3]}
            ),
            CodingTraceRequest.model_validate(
                {"t": 1.0, "type": "n", "in": 1000, "out": 500, "hash_ids": [1, 2, 3]}
            ),
            CodingTraceRequest.model_validate(
                {"t": 2.0, "type": "s", "in": 2000, "out": 800, "hash_ids": [4, 5, 6]}
            ),
        ]

        pairs = CodingTraceLoader._detect_request_pairs(requests)
        assert pairs == 1
        assert requests[0].is_pair_repeat is False
        assert requests[1].is_pair_repeat is True
        assert requests[2].is_pair_repeat is False

    def test_detect_request_pairs_no_match(
        self, mock_prompt_generator, default_user_config
    ):
        """Consecutive requests with different hash_ids are not paired."""
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        requests = [
            CodingTraceRequest.model_validate(
                {"t": 0.0, "type": "s", "in": 1000, "out": 500, "hash_ids": [1, 2, 3]}
            ),
            CodingTraceRequest.model_validate(
                {"t": 1.0, "type": "n", "in": 2000, "out": 800, "hash_ids": [4, 5, 6]}
            ),
        ]

        pairs = CodingTraceLoader._detect_request_pairs(requests)
        assert pairs == 0
        assert requests[0].is_pair_repeat is False
        assert requests[1].is_pair_repeat is False

    def test_detect_request_pairs_requires_type_n(
        self, mock_prompt_generator, default_user_config
    ):
        """Pair detection requires the second request to be type='n'."""
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        requests = [
            CodingTraceRequest.model_validate(
                {"t": 0.0, "type": "s", "in": 1000, "out": 500, "hash_ids": [1, 2, 3]}
            ),
            CodingTraceRequest.model_validate(
                {"t": 1.0, "type": "s", "in": 1000, "out": 500, "hash_ids": [1, 2, 3]}
            ),
        ]

        pairs = CodingTraceLoader._detect_request_pairs(requests)
        assert pairs == 0

    def test_pair_repeat_gets_minimal_delta(self, tmp_path, mock_prompt_generator):
        """Paired requests get delta=1 (minimal content) in conversion."""
        from aiperf.dataset.loader.coding_trace import CodingTraceLoader

        trace = {
            "id": "trace_pair",
            "block_size": 64,
            "requests": [
                {"t": 0.0, "type": "s", "in": 1000, "out": 500, "hash_ids": [1, 2, 3]},
                {"t": 1.0, "type": "n", "in": 1000, "out": 500, "hash_ids": [1, 2, 3]},
                {"t": 2.0, "type": "s", "in": 3000, "out": 1000, "hash_ids": [4, 5, 6]},
            ],
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace))

        # 1 char per token
        mock_prompt_generator.generate_prompt.return_value = "x" * 1500

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                warm_prefix_pct=0.0,
            ),
        )

        loader = CodingTraceLoader(
            filename=str(path),
            prompt_generator=mock_prompt_generator,
            user_config=config,
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        conv = conversations[0]
        # Turn 0: delta=1000
        assert len(conv.turns[0].texts[0].contents[0]) == 1000
        # Turn 1 (pair repeat): delta=1, minimal prompt
        assert len(conv.turns[1].texts[0].contents[0]) == 1
        # Turn 2: delta = max(1, 3000 - 1000 - 500) = 1500
        assert len(conv.turns[2].texts[0].contents[0]) == 1500
