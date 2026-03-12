# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.kr8s_resources module.

Focuses on:
- AsyncJobSet class attributes match expected Kubernetes resource configuration
- JOBSET_VERSION constant correctness
- AsyncJobSet inherits from kr8s APIObject for async operations
"""

from kr8s._objects import APIObject

from aiperf.kubernetes.kr8s_resources import JOBSET_VERSION, AsyncJobSet

# ============================================================
# JOBSET_VERSION Constant
# ============================================================


class TestJobsetVersion:
    """Verify JOBSET_VERSION constant value."""

    def test_jobset_version_value(self) -> None:
        """JOBSET_VERSION matches the expected JobSet CRD API version."""
        assert JOBSET_VERSION == "jobset.x-k8s.io/v1alpha2"


# ============================================================
# AsyncJobSet Class Attributes
# ============================================================


class TestAsyncJobSetAttributes:
    """Verify AsyncJobSet is configured as an async, namespaced JobSet resource."""

    def test_kind_is_jobset(self) -> None:
        """AsyncJobSet.kind identifies the Kubernetes resource type."""
        assert AsyncJobSet.kind == "JobSet"

    def test_version_matches_constant(self) -> None:
        """AsyncJobSet.version is derived from JOBSET_VERSION."""
        assert AsyncJobSet.version == JOBSET_VERSION

    def test_namespaced_is_true(self) -> None:
        """AsyncJobSet is a namespaced resource (not cluster-scoped)."""
        assert AsyncJobSet.namespaced is True

    def test_asyncio_enabled(self) -> None:
        """AsyncJobSet uses the async kr8s interface."""
        assert AsyncJobSet._asyncio is True

    def test_endpoint_is_jobsets(self) -> None:
        """AsyncJobSet REST endpoint is the plural 'jobsets'."""
        assert AsyncJobSet.endpoint == "jobsets"

    def test_plural_is_jobsets(self) -> None:
        """AsyncJobSet plural form used for API discovery."""
        assert AsyncJobSet.plural == "jobsets"

    def test_singular_is_jobset(self) -> None:
        """AsyncJobSet singular form used for API discovery."""
        assert AsyncJobSet.singular == "jobset"

    def test_is_subclass_of_api_object(self) -> None:
        """AsyncJobSet inherits from kr8s APIObject for CRUD operations."""
        assert issubclass(AsyncJobSet, APIObject)

    def test_scalable_is_false(self) -> None:
        """AsyncJobSet is not a scalable resource by default."""
        assert AsyncJobSet.scalable is False
