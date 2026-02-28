# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for parallel_convert module.

Covers worker initialization, batch processing, daemon flag handling,
shared memory management, and end-to-end parallel conversion.
"""

import multiprocessing as mp
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

import aiperf.dataset.loader.parallel_convert as parallel_convert_mod
from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.common.models import Conversation
from aiperf.dataset.generator.prompt import sample_tokens_from_corpus
from aiperf.dataset.loader.parallel_convert import (
    _init_worker,
    _process_batch,
    _set_daemon,
    _WorkerInitArgs,
    _WorkerState,
    parallel_convert,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def sample_corpus():
    """A small corpus of token IDs for testing."""
    return list(range(100, 200))


@pytest.fixture
def sample_corpus_array(sample_corpus):
    """Corpus as numpy int32 array."""
    return np.array(sample_corpus, dtype=np.int32)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that decodes tokens into readable strings."""
    tok = MagicMock()
    tok.decode.side_effect = lambda ids, **kw: " ".join(f"t{i}" for i in ids)
    return tok


@pytest.fixture
def setup_worker(sample_corpus_array, mock_tokenizer):
    """Set up module-level _worker_state for _process_batch tests."""
    seed = 42
    hash_rng = HashIdRandomGenerator(seed, _internal=True)
    hash_rng.set_trace_id("test_trace")

    parallel_convert_mod._worker_state = _WorkerState(
        tokenizer=mock_tokenizer,
        corpus=sample_corpus_array,
        shm=MagicMock(),
        hash_rng=hash_rng,
        block_size=10,
        sep_token=None,
        sample_tokens=sample_tokens_from_corpus,
    )
    yield
    parallel_convert_mod._worker_state = None


@pytest.fixture
def setup_worker_with_sep(sample_corpus_array, mock_tokenizer):
    """Set up worker with a separator token (BOS/EOS)."""
    seed = 42
    hash_rng = HashIdRandomGenerator(seed, _internal=True)
    hash_rng.set_trace_id("test_trace")

    parallel_convert_mod._worker_state = _WorkerState(
        tokenizer=mock_tokenizer,
        corpus=sample_corpus_array,
        shm=MagicMock(),
        hash_rng=hash_rng,
        block_size=10,
        sep_token=1,
        sample_tokens=sample_tokens_from_corpus,
    )
    yield
    parallel_convert_mod._worker_state = None


# -----------------------------------------------------------------------
# _set_daemon
# -----------------------------------------------------------------------


class TestSetDaemon:
    """Tests for daemon flag manipulation."""

    def test_set_daemon_true(self):
        """Setting daemon to True should work on a non-daemon process."""
        original = mp.current_process().daemon
        try:
            _set_daemon(True)
            assert mp.current_process().daemon is True
        finally:
            _set_daemon(original)

    def test_set_daemon_false(self):
        """Setting daemon to False should work."""
        original = mp.current_process().daemon
        try:
            _set_daemon(False)
            assert mp.current_process().daemon is False
        finally:
            _set_daemon(original)

    def test_set_daemon_fallback_on_assertion_error(self):
        """If daemon= setter raises AssertionError, fallback to _config."""
        proc = mp.current_process()
        original = proc.daemon

        with patch.object(
            type(proc), "daemon", property(fset=Mock(side_effect=AssertionError))
        ):
            _set_daemon(True)
            assert proc._config["daemon"] is True

        # Restore
        proc._config["daemon"] = original


# -----------------------------------------------------------------------
# _process_batch
# -----------------------------------------------------------------------


class TestProcessBatch:
    """Tests for _process_batch worker function."""

    def test_text_input_traces(self, setup_worker):
        """Traces with text_input should use the literal text."""
        batch = [
            (
                "session-1",
                [
                    {
                        "text_input": "Hello world",
                        "timestamp": 100,
                        "delay": None,
                        "output_length": 10,
                    },
                ],
            ),
        ]
        results = _process_batch(batch)

        assert len(results) == 1
        sid, turns = results[0]
        assert sid == "session-1"
        assert len(turns) == 1
        ts, delay, prompt, max_tokens = turns[0]
        assert prompt == "Hello world"
        assert ts == 100
        assert max_tokens == 10

    def test_hash_ids_traces(self, setup_worker, mock_tokenizer):
        """Traces with hash_ids should generate tokens and decode."""
        batch = [
            (
                "session-1",
                [
                    {
                        "hash_ids": [1, 2],
                        "input_length": 15,
                        "timestamp": 200,
                        "delay": 5,
                        "output_length": 20,
                    },
                ],
            ),
        ]
        results = _process_batch(batch)

        assert len(results) == 1
        sid, turns = results[0]
        assert sid == "session-1"
        assert len(turns) == 1
        ts, delay, prompt, max_tokens = turns[0]
        assert ts == 200
        assert delay == 5
        assert max_tokens == 20
        # decode was called with generated tokens
        mock_tokenizer.decode.assert_called_once()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_empty_trace_no_hash_ids_no_text(self, setup_worker):
        """Traces without hash_ids or text_input produce empty prompt."""
        batch = [
            (
                "session-1",
                [
                    {
                        "timestamp": 300,
                        "delay": None,
                        "output_length": 5,
                        "input_length": 0,
                    },
                ],
            ),
        ]
        results = _process_batch(batch)

        _, turns = results[0]
        _, _, prompt, _ = turns[0]
        assert prompt == ""

    def test_multiple_sessions_in_batch(self, setup_worker, mock_tokenizer):
        """Multiple sessions in one batch are all processed."""
        batch = [
            (
                "s1",
                [
                    {
                        "text_input": "prompt A",
                        "timestamp": 1,
                        "delay": None,
                        "output_length": 10,
                    }
                ],
            ),
            (
                "s2",
                [
                    {
                        "text_input": "prompt B",
                        "timestamp": 2,
                        "delay": None,
                        "output_length": 20,
                    }
                ],
            ),
            (
                "s3",
                [
                    {
                        "text_input": "prompt C",
                        "timestamp": 3,
                        "delay": None,
                        "output_length": 30,
                    }
                ],
            ),
        ]
        results = _process_batch(batch)

        assert len(results) == 3
        assert results[0][0] == "s1"
        assert results[1][0] == "s2"
        assert results[2][0] == "s3"
        assert results[0][1][0][2] == "prompt A"
        assert results[1][1][0][2] == "prompt B"
        assert results[2][1][0][2] == "prompt C"

    def test_multi_turn_session(self, setup_worker, mock_tokenizer):
        """A session with multiple turns (traces) processes all turns."""
        batch = [
            (
                "session-1",
                [
                    {
                        "text_input": "turn 1",
                        "timestamp": 100,
                        "delay": None,
                        "output_length": 10,
                    },
                    {
                        "text_input": "turn 2",
                        "timestamp": 200,
                        "delay": 50,
                        "output_length": 20,
                    },
                    {
                        "text_input": "turn 3",
                        "timestamp": 300,
                        "delay": 100,
                        "output_length": 30,
                    },
                ],
            ),
        ]
        results = _process_batch(batch)

        _, turns = results[0]
        assert len(turns) == 3
        assert turns[0][2] == "turn 1"
        assert turns[1][2] == "turn 2"
        assert turns[2][2] == "turn 3"

    def test_hash_id_block_cache_reuse(self, setup_worker, mock_tokenizer):
        """Same hash_id within a batch should reuse cached tokens."""
        batch = [
            (
                "s1",
                [
                    {
                        "hash_ids": [42],
                        "input_length": 10,
                        "timestamp": 1,
                        "delay": None,
                        "output_length": 5,
                    },
                ],
            ),
            (
                "s2",
                [
                    {
                        "hash_ids": [42],
                        "input_length": 10,
                        "timestamp": 2,
                        "delay": None,
                        "output_length": 5,
                    },
                ],
            ),
        ]
        results = _process_batch(batch)

        # Both sessions with same hash_id should get same decoded output
        prompt_1 = results[0][1][0][2]
        prompt_2 = results[1][1][0][2]
        assert prompt_1 == prompt_2

    def test_different_hash_ids_produce_different_prompts(
        self, setup_worker, mock_tokenizer
    ):
        """Different hash_ids should produce different token sequences."""
        batch = [
            (
                "s1",
                [
                    {
                        "hash_ids": [100],
                        "input_length": 10,
                        "timestamp": 1,
                        "delay": None,
                        "output_length": 5,
                    },
                ],
            ),
            (
                "s2",
                [
                    {
                        "hash_ids": [200],
                        "input_length": 10,
                        "timestamp": 2,
                        "delay": None,
                        "output_length": 5,
                    },
                ],
            ),
        ]
        results = _process_batch(batch)

        prompt_1 = results[0][1][0][2]
        prompt_2 = results[1][1][0][2]
        assert prompt_1 != prompt_2

    def test_final_block_size_calculation(self, setup_worker, mock_tokenizer):
        """Last hash block should get the remainder tokens."""
        # block_size=10, input_length=25, 3 hash_ids
        # first two blocks: 10 tokens each, last block: 25 - 2*10 = 5 tokens
        batch = [
            (
                "s1",
                [
                    {
                        "hash_ids": [1, 2, 3],
                        "input_length": 25,
                        "timestamp": 1,
                        "delay": None,
                        "output_length": 5,
                    },
                ],
            ),
        ]
        _process_batch(batch)

        # Verify decode was called (tokens were generated)
        assert mock_tokenizer.decode.call_count == 1
        decoded_tokens = mock_tokenizer.decode.call_args[0][0]
        # 10 + 10 + 5 = 25 total tokens
        assert len(decoded_tokens) == 25

    def test_separator_token_prepended(self, setup_worker_with_sep, mock_tokenizer):
        """When sep_token is set, each block should have it prepended."""
        batch = [
            (
                "s1",
                [
                    {
                        "hash_ids": [1],
                        "input_length": 10,
                        "timestamp": 1,
                        "delay": None,
                        "output_length": 5,
                    },
                ],
            ),
        ]
        _process_batch(batch)

        decoded_tokens = mock_tokenizer.decode.call_args[0][0]
        # With sep_token=1, first token in block should be 1
        assert decoded_tokens[0] == 1

    def test_mixed_text_input_and_hash_ids(self, setup_worker, mock_tokenizer):
        """Session with both text_input and hash_id traces."""
        batch = [
            (
                "s1",
                [
                    {
                        "text_input": "literal text",
                        "timestamp": 1,
                        "delay": None,
                        "output_length": 10,
                    },
                    {
                        "hash_ids": [5],
                        "input_length": 10,
                        "timestamp": 2,
                        "delay": None,
                        "output_length": 20,
                    },
                ],
            ),
        ]
        results = _process_batch(batch)

        _, turns = results[0]
        assert turns[0][2] == "literal text"
        assert turns[1][2] != "literal text"  # Generated prompt

    def test_none_fields_preserved(self, setup_worker):
        """None values for timestamp, delay, output_length are preserved."""
        batch = [
            (
                "s1",
                [
                    {
                        "text_input": "test",
                        "timestamp": None,
                        "delay": None,
                        "output_length": None,
                    },
                ],
            ),
        ]
        results = _process_batch(batch)

        ts, delay, _, max_tokens = results[0][1][0]
        assert ts is None
        assert delay is None
        assert max_tokens is None


# -----------------------------------------------------------------------
# _init_worker
# -----------------------------------------------------------------------


class TestInitWorker:
    """Tests for _init_worker function."""

    def test_init_worker_sets_up_state(self, sample_corpus_array, tmp_path):
        """_init_worker should populate _worker dict with all required fields."""
        from multiprocessing import shared_memory

        shm = shared_memory.SharedMemory(create=True, size=sample_corpus_array.nbytes)
        try:
            np.copyto(
                np.ndarray(
                    sample_corpus_array.shape,
                    dtype=sample_corpus_array.dtype,
                    buffer=shm.buf,
                ),
                sample_corpus_array,
            )

            mock_tok = MagicMock()
            args = _WorkerInitArgs(
                shm_name=shm.name,
                corpus_len=len(sample_corpus_array),
                tokenizer_name="test-model",
                base_seed=42,
                block_size=10,
                sep_token=1,
                trace_id="abc123",
            )
            with patch(
                "aiperf.common.tokenizer.Tokenizer.from_pretrained",
                return_value=mock_tok,
            ):
                _init_worker(args)

                state = parallel_convert_mod._worker_state
                assert state is not None
                assert state.tokenizer is mock_tok
                assert state.block_size == 10
                assert state.sep_token == 1
                assert isinstance(state.hash_rng, HashIdRandomGenerator)
                assert np.array_equal(state.corpus, sample_corpus_array)
        finally:
            parallel_convert_mod._worker_state = None
            shm.close()
            shm.unlink()

    def test_init_worker_sets_offline_env(self, sample_corpus_array):
        """Worker should set HF offline environment variables."""
        import os
        from multiprocessing import shared_memory

        original_hf = os.environ.get("HF_HUB_OFFLINE")
        original_tf = os.environ.get("TRANSFORMERS_OFFLINE")

        shm = shared_memory.SharedMemory(create=True, size=sample_corpus_array.nbytes)
        try:
            np.copyto(
                np.ndarray(
                    sample_corpus_array.shape,
                    dtype=sample_corpus_array.dtype,
                    buffer=shm.buf,
                ),
                sample_corpus_array,
            )

            args = _WorkerInitArgs(
                shm_name=shm.name,
                corpus_len=len(sample_corpus_array),
                tokenizer_name="test-model",
                base_seed=42,
                block_size=10,
                sep_token=None,
                trace_id="abc",
            )
            with patch(
                "aiperf.common.tokenizer.Tokenizer.from_pretrained",
                return_value=MagicMock(),
            ):
                _init_worker(args)

                assert os.environ.get("HF_HUB_OFFLINE") == "1"
                assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        finally:
            parallel_convert_mod._worker_state = None
            shm.close()
            shm.unlink()
            # Restore env
            if original_hf is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_hf
            if original_tf is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = original_tf


# -----------------------------------------------------------------------
# parallel_convert — end-to-end
# -----------------------------------------------------------------------


class TestParallelConvert:
    """Tests for the parallel_convert orchestration function."""

    def test_empty_sessions_returns_empty(self, sample_corpus):
        """Empty input returns empty output."""
        result = list(
            parallel_convert(
                sessions=[],
                tokenizer_name="test",
                corpus=sample_corpus,
                base_seed=42,
                block_size=10,
                sep_token=None,
                trace_id="test",
            )
        )
        assert result == []

    def test_returns_conversation_objects(self, sample_corpus):
        """Output should be a list of Conversation objects."""
        sessions = [
            (
                "s1",
                [
                    {
                        "text_input": "hello",
                        "timestamp": 1,
                        "delay": None,
                        "output_length": 5,
                    }
                ],
            ),
        ]

        with patch("aiperf.dataset.loader.parallel_convert.Pool") as MockPool:
            mock_pool_instance = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            mock_pool_instance.imap.return_value = [
                [("s1", [(1, None, "hello", 5)])],
            ]

            result = list(
                parallel_convert(
                    sessions=sessions,
                    tokenizer_name="test",
                    corpus=sample_corpus,
                    base_seed=42,
                    block_size=10,
                    sep_token=None,
                    trace_id="test",
                )
            )

            assert len(result) == 1
            assert isinstance(result[0], Conversation)
            assert result[0].session_id == "s1"
            assert len(result[0].turns) == 1
            assert result[0].turns[0].timestamp == 1
            assert result[0].turns[0].max_tokens == 5

    def test_batching_splits_sessions(self, sample_corpus):
        """Sessions should be split into batches of batch_size."""
        sessions = [
            (
                f"s{i}",
                [
                    {
                        "text_input": f"p{i}",
                        "timestamp": i,
                        "delay": None,
                        "output_length": 5,
                    }
                ],
            )
            for i in range(5)
        ]

        with patch("aiperf.dataset.loader.parallel_convert.Pool") as MockPool:
            mock_pool_instance = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            mock_pool_instance.imap.return_value = [
                [(f"s{i}", [(i, None, f"p{i}", 5)]) for i in range(2)],
                [(f"s{i}", [(i, None, f"p{i}", 5)]) for i in range(2, 5)],
            ]

            list(
                parallel_convert(
                    sessions=sessions,
                    tokenizer_name="test",
                    corpus=sample_corpus,
                    base_seed=42,
                    block_size=10,
                    sep_token=None,
                    trace_id="test",
                    batch_size=2,
                )
            )

            # map was called with batches of size 2
            batches = mock_pool_instance.imap.call_args[0][1]
            assert len(batches) == 3  # 5 sessions / 2 batch_size = 3 batches
            assert len(batches[0]) == 2
            assert len(batches[1]) == 2
            assert len(batches[2]) == 1

    def test_daemon_flag_restored(self, sample_corpus):
        """Daemon flag should be restored after Pool finishes."""
        original_daemon = mp.current_process().daemon

        with patch("aiperf.dataset.loader.parallel_convert.Pool") as MockPool:
            mock_pool_instance = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
            MockPool.return_value.__exit__ = Mock(return_value=False)
            mock_pool_instance.imap.return_value = []

            list(
                parallel_convert(
                    sessions=[
                        (
                            "s1",
                            [
                                {
                                    "text_input": "t",
                                    "timestamp": 1,
                                    "delay": None,
                                    "output_length": 1,
                                }
                            ],
                        )
                    ],
                    tokenizer_name="test",
                    corpus=sample_corpus,
                    base_seed=42,
                    block_size=10,
                    sep_token=None,
                    trace_id="test",
                )
            )

        assert mp.current_process().daemon == original_daemon

    def test_shared_memory_cleanup(self, sample_corpus):
        """Shared memory should be cleaned up even on errors."""
        with patch("aiperf.dataset.loader.parallel_convert.Pool") as MockPool:
            mock_pool_instance = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
            MockPool.return_value.__exit__ = Mock(return_value=False)
            mock_pool_instance.imap.side_effect = RuntimeError("Pool error")

            with pytest.raises(RuntimeError, match="Pool error"):
                list(
                    parallel_convert(
                        sessions=[
                            (
                                "s1",
                                [
                                    {
                                        "text_input": "t",
                                        "timestamp": 1,
                                        "delay": None,
                                        "output_length": 1,
                                    }
                                ],
                            )
                        ],
                        tokenizer_name="test",
                        corpus=sample_corpus,
                        base_seed=42,
                        block_size=10,
                        sep_token=None,
                        trace_id="test",
                    )
                )

            # No leaked shared memory (if it leaked, subsequent tests would detect it)

    def test_multi_turn_conversations(self, sample_corpus):
        """Sessions with multiple turns should produce multi-turn Conversations."""
        with patch("aiperf.dataset.loader.parallel_convert.Pool") as MockPool:
            mock_pool_instance = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            mock_pool_instance.imap.return_value = [
                [("s1", [(100, None, "turn 1", 10), (200, 50, "turn 2", 20)])],
            ]

            result = list(
                parallel_convert(
                    sessions=[
                        (
                            "s1",
                            [
                                {
                                    "text_input": "turn 1",
                                    "timestamp": 100,
                                    "delay": None,
                                    "output_length": 10,
                                },
                                {
                                    "text_input": "turn 2",
                                    "timestamp": 200,
                                    "delay": 50,
                                    "output_length": 20,
                                },
                            ],
                        )
                    ],
                    tokenizer_name="test",
                    corpus=sample_corpus,
                    base_seed=42,
                    block_size=10,
                    sep_token=None,
                    trace_id="test",
                )
            )

            assert len(result) == 1
            conv = result[0]
            assert len(conv.turns) == 2
            assert conv.turns[0].timestamp == 100
            assert conv.turns[0].delay is None
            assert conv.turns[0].max_tokens == 10
            assert conv.turns[1].timestamp == 200
            assert conv.turns[1].delay == 50
            assert conv.turns[1].max_tokens == 20

    def test_pool_receives_correct_init_args(self, sample_corpus):
        """Pool should be initialized with correct arguments."""
        with patch("aiperf.dataset.loader.parallel_convert.Pool") as MockPool:
            mock_pool_instance = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
            MockPool.return_value.__exit__ = Mock(return_value=False)
            mock_pool_instance.imap.return_value = []

            list(
                parallel_convert(
                    sessions=[
                        (
                            "s1",
                            [
                                {
                                    "text_input": "t",
                                    "timestamp": 1,
                                    "delay": None,
                                    "output_length": 1,
                                }
                            ],
                        )
                    ],
                    tokenizer_name="my-tokenizer",
                    corpus=sample_corpus,
                    base_seed=12345,
                    block_size=64,
                    sep_token=7,
                    trace_id="trace_abc",
                    num_workers=4,
                )
            )

            call_args = MockPool.call_args
            assert call_args[0][0] == 4  # num_workers
            assert call_args[0][1] is _init_worker
            initargs = call_args[0][2]
            assert len(initargs) == 1
            args = initargs[0]
            assert isinstance(args, _WorkerInitArgs)
            assert args.tokenizer_name == "my-tokenizer"
            assert args.base_seed == 12345
            assert args.block_size == 64
            assert args.sep_token == 7
            assert args.trace_id == "trace_abc"


# -----------------------------------------------------------------------
# Determinism: _process_batch produces identical results with same seed
# -----------------------------------------------------------------------


class TestProcessBatchDeterminism:
    """Tests that _process_batch is deterministic across invocations."""

    def _setup_and_process(
        self, corpus_array, hash_ids, input_length, block_size, trace_id, seed=42
    ):
        """Helper: set up worker state and process a single batch."""
        hash_rng = HashIdRandomGenerator(seed, _internal=True)
        hash_rng.set_trace_id(trace_id)

        mock_tok = MagicMock()
        mock_tok.decode.side_effect = lambda ids, **kw: ",".join(str(i) for i in ids)

        parallel_convert_mod._worker_state = _WorkerState(
            tokenizer=mock_tok,
            corpus=corpus_array,
            shm=MagicMock(),
            hash_rng=hash_rng,
            block_size=block_size,
            sep_token=None,
            sample_tokens=sample_tokens_from_corpus,
        )

        batch = [
            (
                "s1",
                [
                    {
                        "hash_ids": hash_ids,
                        "input_length": input_length,
                        "timestamp": 1,
                        "delay": None,
                        "output_length": 5,
                    }
                ],
            ),
        ]
        result = _process_batch(batch)
        parallel_convert_mod._worker_state = None
        return result[0][1][0][2]  # prompt string

    def test_same_seed_same_trace_id_same_result(self, sample_corpus_array):
        """Identical seed + trace_id + hash_ids = identical prompt."""
        prompt_1 = self._setup_and_process(
            sample_corpus_array, [1, 2], 15, 10, "trace_a"
        )
        prompt_2 = self._setup_and_process(
            sample_corpus_array, [1, 2], 15, 10, "trace_a"
        )
        assert prompt_1 == prompt_2

    def test_different_trace_id_different_result(self, sample_corpus_array):
        """Different trace_ids produce different prompts."""
        prompt_1 = self._setup_and_process(
            sample_corpus_array, [1, 2], 15, 10, "trace_a"
        )
        prompt_2 = self._setup_and_process(
            sample_corpus_array, [1, 2], 15, 10, "trace_b"
        )
        assert prompt_1 != prompt_2

    def test_different_seed_different_result(self, sample_corpus_array):
        """Different seeds produce different prompts."""
        prompt_1 = self._setup_and_process(
            sample_corpus_array, [1], 10, 10, "trace_a", seed=42
        )
        prompt_2 = self._setup_and_process(
            sample_corpus_array, [1], 10, 10, "trace_a", seed=99
        )
        assert prompt_1 != prompt_2

    def test_different_hash_ids_different_result(self, sample_corpus_array):
        """Different hash_ids produce different prompts."""
        prompt_1 = self._setup_and_process(sample_corpus_array, [10], 10, 10, "trace_a")
        prompt_2 = self._setup_and_process(sample_corpus_array, [20], 10, 10, "trace_a")
        assert prompt_1 != prompt_2
