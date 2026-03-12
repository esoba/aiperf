# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""kr8s custom resource class definitions.

Centralizes custom resource classes used across the kubernetes module.
Built-in kr8s classes (Pod, Namespace, ConfigMap, etc.) are imported
directly from kr8s.objects / kr8s.asyncio.objects where needed.
"""

from kr8s._objects import new_class

JOBSET_VERSION = "jobset.x-k8s.io/v1alpha2"
AsyncJobSet = new_class(
    kind="JobSet",
    version=JOBSET_VERSION,
    namespaced=True,
    asyncio=True,
)
