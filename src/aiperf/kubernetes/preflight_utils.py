# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pre-flight check utilities.

Functions used by both the CLI ``PreflightChecker`` and the operator
``OperatorPreflightChecker``.
"""

from __future__ import annotations

from typing import Any


async def check_rbac_access(
    api: Any,
    verb: str,
    resource: str,
    group: str,
    namespace: str,
) -> bool:
    """Check if current user/service-account has a specific RBAC permission.

    Submits a ``SelfSubjectAccessReview`` to the API server.

    Args:
        api: kr8s async API client.
        verb: Kubernetes verb (e.g. "create", "get", "delete").
        resource: Resource type (e.g. "pods", "configmaps").
        group: API group (e.g. "rbac.authorization.k8s.io", or "" for core).
        namespace: Namespace to check the permission in.

    Returns:
        True if the access is allowed.
    """
    resource_attrs: dict[str, str] = {
        "verb": verb,
        "resource": resource,
        "namespace": namespace,
    }
    if group:
        resource_attrs["group"] = group

    body = {
        "apiVersion": "authorization.k8s.io/v1",
        "kind": "SelfSubjectAccessReview",
        "spec": {"resourceAttributes": resource_attrs},
    }

    async with api.call_api(
        "POST",
        base="/apis/authorization.k8s.io",
        version="v1",
        url="selfsubjectaccessreviews",
        json=body,
    ) as resp:
        result = resp.json()
        return result.get("status", {}).get("allowed", False)


def parse_image_ref(image: str) -> tuple[str, str, str]:
    """Parse a container image reference into (registry, repository, tag).

    Args:
        image: Image reference like "nvcr.io/nvidia/tritonserver:24.01".

    Returns:
        Tuple of (registry, repository, tag). Registry defaults to "docker.io"
        for short names, tag defaults to empty string if not specified.
    """
    tag = ""
    if "@" in image:
        image_no_tag, tag = image.rsplit("@", 1)
    elif ":" in image.split("/")[-1]:
        image_no_tag, tag = image.rsplit(":", 1)
    else:
        image_no_tag = image

    parts = image_no_tag.split("/")
    if len(parts) == 1 or ("." not in parts[0] and ":" not in parts[0]):
        return "docker.io", image_no_tag, tag

    return parts[0], "/".join(parts[1:]), tag
