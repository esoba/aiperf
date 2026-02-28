# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for app module."""

import orjson


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    def test_root_endpoint(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "AIPerf Mock Server"
        assert data["version"] == "2.0.0"

    def test_health_endpoint(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "config" in data

    def test_chat_completions_endpoint(self, test_client):
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data

    def test_completions_endpoint(self, test_client):
        response = test_client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "Hello"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert "usage" in data

    def test_embeddings_endpoint(self, test_client):
        response = test_client.post(
            "/v1/embeddings",
            json={"model": "test-model", "input": "test text"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1

    def test_rankings_endpoint(self, test_client):
        response = test_client.post(
            "/v1/ranking",
            json={
                "model": "test-model",
                "query": {"text": "test query"},
                "passages": [{"text": "passage 1"}, {"text": "passage 2"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "rankings"
        assert len(data["rankings"]) == 2

    def test_dcgm_metrics_invalid_instance(self, test_client):
        response = test_client.get("/dcgm3/metrics")
        assert response.status_code == 404

    def test_image_generation_endpoint(self, test_client):
        response = test_client.post(
            "/v1/images/generations",
            json={
                "model": "black-forest-labs/FLUX.1-dev",
                "prompt": "A beautiful sunset over mountains",
                "n": 1,
                "response_format": "b64_json",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 1
        assert "b64_json" in data["data"][0]
        assert "usage" in data

    def test_image_generation_multiple_images(self, test_client):
        response = test_client.post(
            "/v1/images/generations",
            json={
                "model": "black-forest-labs/FLUX.1-dev",
                "prompt": "Test prompt",
                "n": 3,
                "size": "512x512",
                "quality": "standard",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        assert data["size"] == "512x512"
        assert data["quality"] == "standard"

    def test_solido_rag_endpoint(self, test_client):
        response = test_client.post(
            "/rag/api/prompt",
            json={
                "query": ["What is SOLIDO?"],
                "filters": {"family": "Solido", "tool": "SDE"},
                "inference_model": "test-model",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert data["inference_model"] == "test-model"
        assert data["filters"] == {"family": "Solido", "tool": "SDE"}

    def test_solido_rag_with_multiple_queries(self, test_client):
        response = test_client.post(
            "/rag/api/prompt",
            json={
                "query": ["Query 1", "Query 2", "Query 3"],
                "filters": {"family": "Test"},
                "inference_model": "rag-model",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "sources" in data
        # Should generate sources based on queries
        assert len(data["sources"]) == 3

    def test_anthropic_messages_endpoint(self, test_client):
        response = test_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert isinstance(data["content"], list)
        assert any(b["type"] == "text" for b in data["content"])
        assert "usage" in data
        assert "input_tokens" in data["usage"]
        assert "output_tokens" in data["usage"]

    def test_anthropic_messages_with_system(self, test_client):
        response = test_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "What is AI?"}],
                "max_tokens": 100,
                "system": "You are a helpful assistant.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"

    def test_anthropic_messages_streaming(self, test_client):
        response = test_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = _parse_anthropic_sse_events(response.text)

        event_types = [e["event"] for e in events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

        # Verify text deltas
        text_deltas = [
            e
            for e in events
            if e["event"] == "content_block_delta"
            and e["data"].get("delta", {}).get("type") == "text_delta"
        ]
        assert len(text_deltas) > 0

    def test_anthropic_messages_streaming_with_thinking(self, test_client):
        response = test_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "messages": [
                    {"role": "user", "content": "Solve this math problem step by step"}
                ],
                "max_tokens": 500,
                "stream": True,
                "thinking": {"type": "enabled", "budget_tokens": 100},
            },
        )
        assert response.status_code == 200

        events = _parse_anthropic_sse_events(response.text)

        # Should have thinking block before text block
        block_starts = [e for e in events if e["event"] == "content_block_start"]
        assert len(block_starts) >= 2
        assert block_starts[0]["data"]["content_block"]["type"] == "thinking"
        assert block_starts[1]["data"]["content_block"]["type"] == "text"

        # Should have thinking deltas
        thinking_deltas = [
            e
            for e in events
            if e["event"] == "content_block_delta"
            and e["data"].get("delta", {}).get("type") == "thinking_delta"
        ]
        assert len(thinking_deltas) > 0


def _parse_anthropic_sse_events(text: str) -> list[dict]:
    """Parse Anthropic SSE events from response text."""
    events = []
    current_event = None
    current_data = None

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: "):
            current_data = line[6:]
        elif line == "" and current_event is not None and current_data is not None:
            events.append({"event": current_event, "data": orjson.loads(current_data)})
            current_event = None
            current_data = None

    return events
