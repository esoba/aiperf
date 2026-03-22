# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic coding generator shaped like Anthropic-style agent proxy traces.

Content patterns are inspired by real Conflux / Claude Code-style traffic (mixed
``thinking`` narration, ``tool_use`` JSON lines, and tool-result-shaped blobs with
Read-skewed tool mix), but **everything is template-generated** — no export
files are read at runtime.

Compared to :class:`CodingContentGenerator`, pools emphasize serialized tool
payloads and multi-segment turns rather than a broad generic workload mix.
"""

from __future__ import annotations

from typing import Any

import orjson

from aiperf.common.config import PromptConfig
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.coding_content import (
    _ASSISTANT_PREAMBLES,
    _TEXT_POOL_BLOCKS,
    CodingContentGenerator,
)

# Tool mix approximates aggregate counts from a Conflux unified-split sample
# (Read-dominant, then Bash / Grep / Glob / Agent-like work).
_PROXY_TOOL_WEIGHTS = (
    ("Read", 50),
    ("Bash", 22),
    ("Grep", 9),
    ("Glob", 4),
    ("Edit", 5),
    ("Write", 3),
    ("TaskUpdate", 2),
    ("Agent", 3),
    ("TaskCreate", 2),
)

_THINKING_SNIPPETS = (
    "I should inspect {module} first — the stack trace points at {class_name}.{method}(). "
    "I'll read the surrounding file and check how {var} is passed.",
    "The failure looks like a stale assumption in {method}(): {var} can be empty when "
    "the batch is drained. I'll grep for callers and confirm the contract.",
    "Before editing, I want to see the full diff context. I'll read the file with an "
    "offset near the handler and then run the focused test target.",
    "This might be a regression from the recent {module} refactor. I'll search for "
    "{class_name} usage and list related paths under src/.",
    "I'll run the unit tests for {module} to reproduce, then narrow down with a grep "
    "for the error substring.",
)

_PROXY_USER_TEMPLATES = (
    "Fix the failing tests in {module} — start by reading the traceback and the "
    "implementation of {class_name}.{method}().",
    "Refactor {module} so {method}() is async-safe when {var} is shared across tasks.",
    "Add validation in {class_name} for {var} before we call into the {module} client.",
    "Investigate high latency in {module}: profile {method}() and suggest concrete changes.",
    "The CI job is red on main. Reproduce locally, then patch {class_name}.{method}().",
)

_PROXY_TOOL_POOL_BLOCK_COUNTS: dict[str, int] = {
    "_gen_proxy_assistant_flat": 200,
    "_gen_proxy_user_results_flat": 190,
    "_gen_proxy_user_plain": 55,
    "_gen_proxy_exchange": 130,
    "_gen_proxy_system_stub": 12,
    "_gen_python_code": 35,
    "_gen_error_traceback": 25,
    "_gen_git_diff": 20,
    "_gen_test_output": 25,
    "_gen_json_response": 20,
}


class ProxyInspiredCodingContentGenerator(CodingContentGenerator):
    """Template-only generator with proxy-trace-shaped token pools."""

    _TOOL_WEIGHTS = _PROXY_TOOL_WEIGHTS

    def __init__(
        self,
        config: PromptConfig,
        tokenizer: Tokenizer,
        pool_tokens_target: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, tokenizer, pool_tokens_target, **kwargs)

    def _build_tool_pool(self) -> None:
        blocks: list[str] = []
        for gen_name, count in _PROXY_TOOL_POOL_BLOCK_COUNTS.items():
            gen_fn = getattr(self, gen_name)
            for _ in range(int(count * self._pool_scale)):
                blocks.append(gen_fn())
        self._template_rng.shuffle(blocks)
        text = "\n\n".join(blocks)
        self._tool_pool = self.tokenizer.encode(text)
        self.debug(
            lambda: f"Built proxy-inspired tool pool with {len(self._tool_pool)} tokens "
            f"from {len(blocks)} blocks"
        )

    def _build_text_pool(self) -> None:
        blocks: list[str] = []
        n = int(_TEXT_POOL_BLOCKS * self._pool_scale)
        for i in range(n):
            if i % 10 < 7:
                blocks.append(self._gen_user_prompt())
            else:
                blocks.append(self._gen_proxy_user_plain())
        self._template_rng.shuffle(blocks)
        text = "\n\n".join(blocks)
        self._text_pool = self.tokenizer.encode(text)
        self.debug(
            lambda: f"Built proxy-inspired text pool with {len(self._text_pool)} tokens "
            f"from {len(blocks)} blocks"
        )

    def _gen_thinking_snippet(self) -> str:
        r = self._template_rng
        return r.choice(_THINKING_SNIPPETS).format(**self._template_fills())

    def _synth_output_for_tool(self, tool_name: str) -> str:
        r = self._template_rng
        match tool_name:
            case "Read":
                body = self._gen_python_code()
                return body[: r.randint(400, min(4500, len(body) or 400))]
            case "Bash":
                return r.choice(
                    (
                        self._gen_bash_output(),
                        self._gen_test_output(),
                        self._gen_cicd_output(),
                    )
                )
            case "Grep":
                return self._gen_tool_search()
            case "Glob":
                return self._gen_bash_file_explore()
            case "Edit" | "Write":
                return self._gen_git_diff()
            case "Agent":
                return self._gen_markdown_doc()
            case "TaskUpdate" | "TaskCreate":
                return self._gen_json_object()
            case _:
                return self._gen_bash_output()

    def _gen_proxy_assistant_flat(self) -> str:
        r = self._template_rng
        parts: list[str] = []
        if r.randint(1, 100) <= 42:
            parts.append(self._gen_thinking_snippet())
        if r.randint(1, 100) <= 28:
            parts.append(
                r.choice(_ASSISTANT_PREAMBLES).format(**self._template_fills())
            )
        num_tools = 1
        if r.randint(1, 100) <= 24:
            num_tools = r.randint(2, 4)
        for _ in range(num_tools):
            name, inp = self._make_tool_call()
            parts.append(orjson.dumps({"name": name, "input": inp}).decode())
        return "\n".join(parts)

    def _gen_proxy_user_results_flat(self) -> str:
        r = self._template_rng
        n = r.randint(1, 3)
        bodies: list[str] = []
        for _ in range(n):
            name, _ = self._make_tool_call()
            bodies.append(self._synth_output_for_tool(name))
        return "\n\n".join(bodies)

    def _gen_proxy_user_plain(self) -> str:
        r = self._template_rng
        return r.choice(_PROXY_USER_TEMPLATES).format(**self._template_fills())

    def _gen_proxy_exchange(self) -> str:
        return (
            self._gen_proxy_assistant_flat()
            + "\n\n---\n\n"
            + self._gen_proxy_user_results_flat()
        )

    def _gen_proxy_system_stub(self) -> str:
        r = self._template_rng
        stub = (
            "[agent-system] Interactive coding assistant. Tools: Read, Bash, Grep, Glob, "
            "Edit, Write. Work in the current repo; prefer small verified steps.\n"
        )
        extra = r.choice(_THINKING_SNIPPETS).format(**self._template_fills())
        return stub + extra
