# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Conflux proxy capture dataset loader."""

from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


def create_conflux_file(
    tmp_path: Path,
    records: list[dict],
    filename: str = "conflux_session.json",
) -> Path:
    """Create a pretty-printed Conflux JSON file for testing."""
    path = tmp_path / filename
    path.write_bytes(orjson.dumps(records, option=orjson.OPT_INDENT_2))
    return path


def make_conflux_records() -> list[dict]:
    """Generate a realistic multi-agent Conflux capture with staggered timestamps."""
    return [
        {
            "session_id": "sess-1",
            "agent_id": "planner",
            "timestamp": 0.0,
            "duration_ms": 800,
            "messages": [
                {"role": "system", "content": "You are a planning assistant."},
                {"role": "user", "content": "Plan a web scraper for news articles."},
            ],
            "tokens": {"input": 50, "output": 120},
            "hyperparameters": {"temperature": 0.7},
        },
        {
            "session_id": "sess-1",
            "agent_id": "coder",
            "timestamp": 1000.0,
            "duration_ms": 1200,
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Implement the scraper using BeautifulSoup."},
            ],
            "tokens": {"input": 80, "output": 250},
            "hyperparameters": {"temperature": 0.3},
        },
        {
            "session_id": "sess-1",
            "agent_id": "planner",
            "timestamp": 3000.0,
            "duration_ms": 600,
            "messages": [
                {"role": "system", "content": "You are a planning assistant."},
                {"role": "user", "content": "Plan a web scraper for news articles."},
                {"role": "assistant", "content": "Here is the plan..."},
                {"role": "user", "content": "Now add error handling to the plan."},
            ],
            "tokens": {"input": 150, "output": 100, "output_reasoning": 20},
        },
        {
            "session_id": "sess-1",
            "agent_id": "coder",
            "timestamp": 5000.0,
            "duration_ms": 1500,
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Implement the scraper using BeautifulSoup."},
                {"role": "assistant", "content": "Here is the code..."},
                {"role": "user", "content": "Add retry logic and rate limiting."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write content to a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        },
                    },
                }
            ],
            "tokens": {"input": 200, "output": 300},
            "hyperparameters": {"temperature": 0.2, "top_p": 0.95},
        },
        {
            "session_id": "sess-1",
            "agent_id": "reviewer",
            "timestamp": 8000.0,
            "duration_ms": 900,
            "messages": [
                {"role": "user", "content": "Review the scraper code for best practices."},
            ],
            "tokens": {"input": 100, "output": 180},
        },
    ]  # fmt: skip


@pytest.mark.integration
@pytest.mark.asyncio
class TestConfluxLoaderIntegration:
    """Integration tests for Conflux proxy capture dataset loader."""

    async def test_auto_detect_and_replay_with_speedup(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Auto-detect a pretty-printed Conflux file and replay with --fixed-schedule-speedup 10."""
        records = make_conflux_records()
        conflux_file = create_conflux_file(tmp_path, records)
        request_count = len(records)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {conflux_file} \
                --request-count {request_count} \
                --fixed-schedule \
                --fixed-schedule-speedup 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == request_count
        assert result.has_all_outputs

    async def test_explicit_type_with_speedup(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Explicit --custom-dataset-type conflux with speedup."""
        records = make_conflux_records()
        conflux_file = create_conflux_file(tmp_path, records)
        request_count = len(records)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {conflux_file} \
                --custom-dataset-type conflux \
                --request-count {request_count} \
                --fixed-schedule \
                --fixed-schedule-speedup 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == request_count
        assert result.has_all_outputs

    async def test_directory_of_conflux_files(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Load a directory of Conflux JSON files with auto-detection."""
        input_dir = tmp_path / "sessions"
        input_dir.mkdir()

        records = make_conflux_records()
        # Split into two files by agent
        planner_records = [r for r in records if r.get("agent_id") == "planner"]
        other_records = [r for r in records if r.get("agent_id") != "planner"]
        create_conflux_file(input_dir, planner_records, "session_planner.json")
        create_conflux_file(input_dir, other_records, "session_others.json")
        request_count = len(records)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {input_dir} \
                --request-count {request_count} \
                --fixed-schedule \
                --fixed-schedule-speedup 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == request_count
        assert result.has_all_outputs

    async def test_conflux_with_extra_params_in_payload(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Verify hyperparameters from Conflux records propagate into request payloads."""
        records = [
            {
                "session_id": "sess-1",
                "agent_id": "agent-A",
                "timestamp": 0.0,
                "duration_ms": 500,
                "messages": [{"role": "user", "content": "Hello"}],
                "tokens": {"input": 10, "output": 20},
                "hyperparameters": {"temperature": 0.42, "top_p": 0.88},
            },
        ]
        conflux_file = create_conflux_file(tmp_path, records)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {conflux_file} \
                --custom-dataset-type conflux \
                --request-count 1 \
                --fixed-schedule \
                --fixed-schedule-speedup 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == 1
        assert result.has_all_outputs

        # Verify hyperparameters made it into the actual payloads
        assert result.inputs is not None
        payloads = [p for session in result.inputs.data for p in session.payloads]
        assert len(payloads) >= 1
        payload = payloads[0]
        assert payload["temperature"] == 0.42
        assert payload["top_p"] == 0.88
