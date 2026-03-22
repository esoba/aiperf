#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pre-generated responses demo script."""

from __future__ import annotations

import sys
from typing import Any

import orjson
import pytest

import demo_pregenerated_responses as demo


@pytest.fixture
def tokenizer(mock_tokenizer_cls: type) -> object:
    return mock_tokenizer_cls.from_pretrained("gpt2")


def test_print_conversation_full_shows_complete_payloads(
    capsys: pytest.CaptureFixture[str], tokenizer: object
) -> None:
    messages = [
        {
            "role": "assistant",
            "content": "assistant full text payload",
            "tool_calls": [
                {
                    "id": "toolu_01abcdef",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": orjson.dumps(
                            {
                                "file_path": "src/aiperf/dataset/generator/coding_content.py",
                                "offset": 123,
                                "limit": 456,
                            }
                        ).decode(),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_01abcdef",
            "content": "def full_tool_result():\n    return 'all content'\n",
        },
        {"role": "assistant", "content": "assistant final summary"},
    ]

    demo.print_conversation(messages, tokenizer, full=True)

    output = capsys.readouterr().out
    assert "assistant full text payload" in output
    assert "src/aiperf/dataset/generator/coding_content.py" in output
    assert '"offset":123' in output
    assert '"limit":456' in output
    assert "def full_tool_result():" in output
    assert "assistant final summary" in output


def test_print_raw_messages_emits_exact_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    messages = [
        {
            "role": "assistant",
            "content": "raw assistant payload",
            "tool_calls": [
                {
                    "id": "toolu_01raw",
                    "type": "function",
                    "function": {
                        "name": "Glob",
                        "arguments": '{"pattern":"**/*.py"}',
                    },
                }
            ],
        }
    ]

    demo.print_raw_messages(messages)

    output = capsys.readouterr().out
    assert orjson.loads(output) == messages


def test_build_demo_messages_can_prepend_user_prompt() -> None:
    class _StubGenerator:
        def generate_response(self, budget: int) -> list[dict[str, Any]]:
            assert budget == 300
            return [{"role": "assistant", "content": "assistant response"}]

        def _gen_user_prompt(self) -> str:
            return "user prompt first"

    messages = demo.build_demo_messages(_StubGenerator(), 300, include_user=True)

    assert messages == [
        {"role": "user", "content": "user prompt first"},
        {"role": "assistant", "content": "assistant response"},
    ]


@pytest.mark.parametrize(
    ("argv", "expected_full", "expected_raw", "expected_include_user"),
    [
        (["demo_pregenerated_responses.py", "--raw"], False, True, False),
        (
            ["demo_pregenerated_responses.py", "--full", "--raw", "--include-user"],
            True,
            True,
            True,
        ),
    ],
)
def test_main_passes_full_and_raw_flags(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_full: bool,
    expected_raw: bool,
    expected_include_user: bool,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_demo(
        token_budgets: list[int],
        seeds: int,
        full: bool,
        raw: bool,
        include_user: bool,
    ) -> None:
        captured["token_budgets"] = token_budgets
        captured["seeds"] = seeds
        captured["full"] = full
        captured["raw"] = raw
        captured["include_user"] = include_user

    monkeypatch.setattr(demo, "run_demo", fake_run_demo)
    monkeypatch.setattr(sys, "argv", argv)

    demo.main()

    assert captured["token_budgets"] == [100, 300, 600, 1000]
    assert captured["seeds"] == 3
    assert captured["full"] is expected_full
    assert captured["raw"] is expected_raw
    assert captured["include_user"] is expected_include_user
