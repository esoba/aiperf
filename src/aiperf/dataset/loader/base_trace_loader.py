# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from aiperf.common.config.config_defaults import InputTokensDefaults
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.generator.base import BaseGenerator
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.parallel_convert import parallel_convert
from aiperf.dataset.synthesis.models import SynthesisParams
from aiperf.dataset.synthesis.synthesizer import Synthesizer
from aiperf.plugin.enums import DatasetSamplingStrategy

TraceT = TypeVar("TraceT")

_MIN_TRACES_FOR_PARALLEL = 10


def _compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of file content (first 16 hex chars).

    Falls back to hashing the filepath string if the file cannot be read.
    """
    try:
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    except (OSError, TypeError):
        return hashlib.sha256(filepath.encode()).hexdigest()[:16]


class BaseTraceDatasetLoader(BaseFileLoader, Generic[TraceT]):
    """Base class for trace dataset loaders with hash_ids-based prompt generation.

    Provides common infrastructure for loading trace-format datasets
    (Mooncake, Bailian, etc.) including shared initialization, timestamp
    filtering, parallel prompt generation with deterministic per-hash_id
    re-seeding, and synthesis integration.

    Subclasses must implement:
    - `can_load`: data format detection
    - `load_dataset`: JSONL parsing and session grouping
    - `_synthesis_exclude_fields`: fields to strip before synthesis
    - `_reconstruct_traces`: rebuild typed traces from synthesized dicts
    """

    def __init__(
        self,
        *,
        filename: str,
        prompt_generator: BaseGenerator,
        user_config: UserConfig,
        default_block_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(filename=filename, user_config=user_config, **kwargs)
        self.prompt_generator = prompt_generator
        self._skipped_traces = 0
        self._skipped_max_isl = 0
        self._capped_max_osl = 0
        self._start_offset = user_config.input.fixed_schedule_start_offset
        self._end_offset = user_config.input.fixed_schedule_end_offset
        self._max_isl = user_config.input.synthesis.max_isl
        self._max_osl = user_config.input.synthesis.max_osl
        self._trace_id: str = ""
        self._trust_remote_code = user_config.tokenizer.trust_remote_code
        self._revision = user_config.tokenizer.revision

        # Precedence: user CLI --isl-block-size > plugin metadata default > hardcoded fallback
        user_block_size = user_config.input.prompt.input_tokens.block_size
        if user_block_size is not None:
            self._block_size = user_block_size
        elif default_block_size is not None:
            self._block_size = default_block_size
        else:
            self._block_size = InputTokensDefaults.BLOCK_SIZE

    # ------------------------------------------------------------------
    # Shared class methods
    # ------------------------------------------------------------------

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Trace datasets use sequential sampling to preserve timestamp order."""
        return DatasetSamplingStrategy.SEQUENTIAL

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _parse_trace(self, line: str) -> TraceT:
        """Parse a single JSONL line into a typed trace object."""
        ...

    def _preprocess_trace(self, trace: TraceT) -> None:
        """Optional hook for per-trace pre-processing (e.g. unit conversion).

        Called after parsing but before filtering. Default is a no-op.
        """
        pass

    @abstractmethod
    def _group_traces(self, items: list[TraceT]) -> dict[str, list[TraceT]]:
        """Group flat trace entries into sessions keyed by session ID."""
        ...

    # ------------------------------------------------------------------
    # Timestamp / filtering helpers
    # ------------------------------------------------------------------

    def _timestamp_within_offsets(self, timestamp: int | float) -> bool:
        """Check if a timestamp falls within configured offsets."""
        return (self._start_offset is None or timestamp >= self._start_offset) and (
            self._end_offset is None or timestamp <= self._end_offset
        )

    def _filter_and_cap_trace(self, trace: TraceT) -> bool:
        """Apply timestamp-window, max_isl, and max_osl filters.

        Returns `True` if the trace should be kept, `False` to skip.
        """
        timestamp = getattr(trace, "timestamp", None)
        if timestamp is not None and not self._timestamp_within_offsets(timestamp):
            self._skipped_traces += 1
            return False

        input_length = getattr(trace, "input_length", None)
        if (
            self._max_isl is not None
            and input_length is not None
            and input_length > self._max_isl
        ):
            self._skipped_max_isl += 1
            return False

        output_length = getattr(trace, "output_length", None)
        if (
            self._max_osl is not None
            and output_length is not None
            and output_length > self._max_osl
        ):
            self._capped_max_osl += 1
            trace.output_length = self._max_osl  # type: ignore[attr-defined]

        return True

    def _log_filtering_summary(self) -> None:
        """Emit info-level messages for any skipped or capped traces."""
        if self._skipped_traces > 0:
            self.info(
                f"Skipped {self._skipped_traces:,} traces because they were "
                f"before the start offset of {self._start_offset} or "
                f"after the end offset of {self._end_offset}"
            )
        if self._skipped_max_isl > 0:
            self.info(
                f"Skipped {self._skipped_max_isl:,} traces because input_length "
                f"exceeded max_isl of {self._max_isl}"
            )
        if self._capped_max_osl > 0:
            self.info(
                f"{self._capped_max_osl:,} traces exceeded max_osl of "
                f"{self._max_osl} and were capped to {self._max_osl}"
            )

    # ------------------------------------------------------------------
    # load_dataset — template method
    # ------------------------------------------------------------------

    def load_dataset(self) -> dict[str, list[TraceT]]:
        """Load, filter, group, and optionally synthesize trace data.

        Template method that delegates format-specific work to subclass hooks:
        :meth:`_parse_trace`, :meth:`_preprocess_trace`, and
        :meth:`_group_traces`.
        """
        self._skipped_traces = 0
        self._skipped_max_isl = 0
        self._capped_max_osl = 0

        self._trace_id = _compute_file_hash(self.filename)
        self.prompt_generator._hash_id_corpus_rng.set_trace_id(self._trace_id)
        self.debug(lambda: f"Trace ID: {self._trace_id} for {self.filename}")
        items: list[TraceT] = []

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue

                trace = self._parse_trace(line)
                self._preprocess_trace(trace)

                if not self._filter_and_cap_trace(trace):
                    continue

                items.append(trace)

        self._log_filtering_summary()

        data = self._group_traces(items)
        self.debug(
            lambda: (
                f"Loaded {sum(len(v) for v in data.values()):,} traces "
                f"across {len(data):,} sessions from {self.filename}"
            )
        )

        if self.user_config.input.synthesis.should_synthesize():
            data = self._apply_synthesis(data)

        return data

    # ------------------------------------------------------------------
    # convert_to_conversations
    # ------------------------------------------------------------------

    def _get_text_input(self, trace: TraceT) -> str | None:
        """Return pre-existing text input, or `None` to use hash_ids generation.

        Override for traces that carry literal prompts (e.g. `MooncakeTrace.text_input`).
        Default: checks for a `text_input` attribute via getattr.
        """
        return getattr(trace, "text_input", None)

    def _infer_context_mode(
        self, traces: list[TraceT]
    ) -> ConversationContextMode | None:
        """Infer context_mode from trace data when not explicitly set.

        Override in subclasses to auto-detect based on trace content.
        Default returns None (falls through to global DELTAS_WITHOUT_RESPONSES default).
        """
        return None

    def _build_turn(self, trace: TraceT, prompt: str) -> Turn:
        """Build a :class:`Turn` from trace data and a generated prompt.

        Default implementation extracts `timestamp`, `delay`, `output_length`
        via getattr, which works for both Mooncake and Bailian traces.
        """
        return Turn(
            timestamp=getattr(trace, "timestamp", None),
            delay=getattr(trace, "delay", None),
            texts=[Text(name="text", contents=[prompt])],
            max_tokens=getattr(trace, "output_length", None),
        )

    def convert_to_conversations(
        self,
        data: dict[str, list[TraceT]],
        num_workers: int | None = None,
        batch_size: int = 100,
    ) -> Iterator[Conversation]:
        """Convert trace sessions to conversations using parallel workers.

        Uses multiprocessing Pool with shared memory for the token corpus.
        Each worker gets its own HashIdRandomGenerator to produce deterministic
        token sequences per hash_id regardless of worker count or order.

        Falls back to single-threaded conversion for small datasets.

        Yields:
            Conversation objects in session order.
        """
        sessions = list(data.items())
        if not sessions:
            return

        total_traces = sum(len(traces) for _, traces in sessions)
        if total_traces < _MIN_TRACES_FOR_PARALLEL:
            yield from self._convert_single_threaded(sessions)
            return

        pg = self.prompt_generator
        serialized = [
            (sid, [t.model_dump() for t in traces])  # type: ignore[union-attr]
            for sid, traces in sessions
        ]

        yield from parallel_convert(
            sessions=serialized,
            tokenizer_name=pg.tokenizer.resolved_name,
            corpus=pg._tokenized_corpus,
            base_seed=pg._hash_id_corpus_rng.seed,
            block_size=self._block_size,
            sep_token=pg.tokenizer.block_separation_token_id,
            trace_id=self._trace_id,
            trust_remote_code=self._trust_remote_code,
            revision=self._revision,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    def _convert_single_threaded(
        self, sessions: list[tuple[str, list[TraceT]]]
    ) -> Iterator[Conversation]:
        """Fallback single-threaded conversion for small datasets."""
        for session_id, traces in sessions:
            context_mode = self._infer_context_mode(traces)
            conversation = Conversation(
                session_id=session_id, context_mode=context_mode
            )
            for trace in traces:
                text_input = self._get_text_input(trace)
                if text_input is not None:
                    prompt = text_input
                else:
                    hash_ids: list[int] = getattr(trace, "hash_ids", None) or []
                    input_length: int = getattr(trace, "input_length", 0)
                    prompt = self.prompt_generator.generate(
                        mean=input_length,
                        stddev=0,
                        hash_ids=hash_ids,
                        block_size=self._block_size,
                    )
                conversation.turns.append(self._build_turn(trace, prompt))
            yield conversation

    # ------------------------------------------------------------------
    # Synthesis — shared orchestration with subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _synthesis_exclude_fields(self) -> frozenset[str]:
        """Fields to exclude when serializing traces for the Synthesizer."""
        ...

    def _synthesis_dump_kwargs(self) -> dict[str, Any]:
        """Extra kwargs for `model_dump` during synthesis serialization.

        Override to add e.g. `by_alias=True` for aliased fields.
        """
        return {}

    @abstractmethod
    def _reconstruct_traces(
        self, originals: list[TraceT], synth_dicts: list[dict[str, Any]]
    ) -> list[TraceT]:
        """Rebuild typed trace objects from synthesized dicts.

        Args:
            originals: The original traces for this session (for metadata recovery).
            synth_dicts: The synthesized dicts from the Synthesizer.
        """
        ...

    def _apply_synthesis(
        self, data: dict[str, list[TraceT]]
    ) -> dict[str, list[TraceT]]:
        """Apply synthesis transformations to traces in-memory."""
        params = SynthesisParams.from_synthesis_config(
            self.user_config.input.synthesis, block_size=self._block_size
        )

        exclude = self._synthesis_exclude_fields()
        dump_kwargs = self._synthesis_dump_kwargs()
        dict_data = {
            sid: [
                t.model_dump(exclude=exclude, exclude_none=True, **dump_kwargs)  # type: ignore[union-attr]
                for t in traces
            ]
            for sid, traces in data.items()
        }

        synthesized = Synthesizer(params=params).synthesize_grouped_traces(dict_data)

        return {
            sid: self._reconstruct_traces(data.get(sid, []), synth_traces)
            for sid, synth_traces in synthesized.items()
        }
