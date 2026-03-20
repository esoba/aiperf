# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.preflight_utils module.

Covers:
- check_rbac_access(): SelfSubjectAccessReview with allowed/denied, group inclusion, error propagation
- parse_image_ref(): registry/repo/tag extraction for various image reference formats
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.kubernetes.preflight_utils import check_rbac_access, parse_image_ref

# =============================================================================
# Helpers
# =============================================================================


def _mock_call_api_response(json_body: dict[str, Any], status_code: int = 200):
    """Return an async context-manager mock for api.call_api."""
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.status_code = status_code

    @asynccontextmanager
    async def _ctx(*args: Any, **kwargs: Any) -> Any:
        yield resp

    return _ctx


def _mock_call_api_raises(exc: Exception):
    """Return an async context-manager mock for api.call_api that raises."""

    @asynccontextmanager
    async def _ctx(*args: Any, **kwargs: Any) -> Any:
        raise exc
        yield  # noqa: F841, RET503

    return _ctx


# =============================================================================
# check_rbac_access
# =============================================================================


class TestCheckRbacAccess:
    """Tests for the check_rbac_access utility function."""

    @pytest.mark.asyncio
    async def test_returns_true_when_allowed(self) -> None:
        """Verify True is returned when the SelfSubjectAccessReview says allowed."""
        api = MagicMock()
        api.call_api = _mock_call_api_response({"status": {"allowed": True}})

        result = await check_rbac_access(api, "create", "pods", "", "test-ns")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_allowed(self) -> None:
        """Verify False is returned when the SelfSubjectAccessReview says not allowed."""
        api = MagicMock()
        api.call_api = _mock_call_api_response({"status": {"allowed": False}})

        result = await check_rbac_access(api, "delete", "pods", "", "test-ns")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_status_missing(self) -> None:
        """Verify False is returned when the response has no status field."""
        api = MagicMock()
        api.call_api = _mock_call_api_response({})

        result = await check_rbac_access(api, "get", "pods", "", "test-ns")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_allowed_missing(self) -> None:
        """Verify False is returned when status has no allowed field."""
        api = MagicMock()
        api.call_api = _mock_call_api_response({"status": {}})

        result = await check_rbac_access(api, "get", "pods", "", "test-ns")

        assert result is False

    @pytest.mark.asyncio
    async def test_propagates_exception_from_api(self) -> None:
        """Verify exceptions from api.call_api bubble up to the caller."""
        api = MagicMock()
        api.call_api = _mock_call_api_raises(RuntimeError("network failure"))

        with pytest.raises(RuntimeError, match="network failure"):
            await check_rbac_access(api, "create", "pods", "", "test-ns")

    @pytest.mark.asyncio
    async def test_includes_group_when_nonempty(self) -> None:
        """Verify the group field is included in resourceAttributes only when non-empty."""
        captured_kwargs: dict[str, Any] = {}
        resp = MagicMock()
        resp.json.return_value = {"status": {"allowed": True}}

        @asynccontextmanager
        async def _capture(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            yield resp

        api = MagicMock()
        api.call_api = _capture

        await check_rbac_access(api, "create", "jobsets", "jobset.x-k8s.io", "test-ns")

        body = captured_kwargs["json"]
        attrs = body["spec"]["resourceAttributes"]
        assert attrs["group"] == "jobset.x-k8s.io"

    @pytest.mark.asyncio
    async def test_excludes_group_when_empty(self) -> None:
        """Verify the group field is NOT in resourceAttributes when group is empty."""
        captured_kwargs: dict[str, Any] = {}
        resp = MagicMock()
        resp.json.return_value = {"status": {"allowed": True}}

        @asynccontextmanager
        async def _capture(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            yield resp

        api = MagicMock()
        api.call_api = _capture

        await check_rbac_access(api, "get", "pods", "", "test-ns")

        body = captured_kwargs["json"]
        attrs = body["spec"]["resourceAttributes"]
        assert "group" not in attrs

    @pytest.mark.asyncio
    async def test_namespace_passed_correctly(self) -> None:
        """Verify the namespace is included in the resource attributes."""
        captured_kwargs: dict[str, Any] = {}
        resp = MagicMock()
        resp.json.return_value = {"status": {"allowed": True}}

        @asynccontextmanager
        async def _capture(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            yield resp

        api = MagicMock()
        api.call_api = _capture

        await check_rbac_access(api, "list", "configmaps", "", "my-namespace")

        body = captured_kwargs["json"]
        attrs = body["spec"]["resourceAttributes"]
        assert attrs["namespace"] == "my-namespace"

    @pytest.mark.asyncio
    async def test_verb_and_resource_passed_correctly(self) -> None:
        """Verify verb and resource are included in the resource attributes."""
        captured_kwargs: dict[str, Any] = {}
        resp = MagicMock()
        resp.json.return_value = {"status": {"allowed": True}}

        @asynccontextmanager
        async def _capture(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            yield resp

        api = MagicMock()
        api.call_api = _capture

        await check_rbac_access(api, "delete", "secrets", "", "ns")

        body = captured_kwargs["json"]
        attrs = body["spec"]["resourceAttributes"]
        assert attrs["verb"] == "delete"
        assert attrs["resource"] == "secrets"


# =============================================================================
# parse_image_ref
# =============================================================================


class TestParseImageRef:
    """Tests for the parse_image_ref utility function."""

    @pytest.mark.parametrize(
        "image,expected",
        [
            param(
                "nvcr.io/nvidia/tritonserver:24.01",
                ("nvcr.io", "nvidia/tritonserver", "24.01"),
                id="full-image-ref",
            ),
            param(
                "python:3.10",
                ("docker.io", "python", "3.10"),
                id="short-docker-hub-name",
            ),
            param(
                "nvcr.io/nvidia/aiperf",
                ("nvcr.io", "nvidia/aiperf", ""),
                id="no-tag",
            ),
            param(
                "repo/img@sha256:abc123",
                ("docker.io", "repo/img", "sha256:abc123"),
                id="digest",
            ),
            param(
                "nginx",
                ("docker.io", "nginx", ""),
                id="single-word",
            ),
            param(
                "nginx:latest",
                ("docker.io", "nginx", "latest"),
                id="single-word-with-tag",
            ),
            param(
                "gcr.io/project/team/img:v1",
                ("gcr.io", "project/team/img", "v1"),
                id="multi-level-repo",
            ),
            param(
                "localhost:5000/myimage:v1",
                ("localhost:5000", "myimage", "v1"),
                id="port-in-registry",
            ),
        ],
    )  # fmt: skip
    def test_parse_image_ref(self, image: str, expected: tuple[str, str, str]) -> None:
        """Verify parse_image_ref correctly splits registry, repo, and tag."""
        assert parse_image_ref(image) == expected
