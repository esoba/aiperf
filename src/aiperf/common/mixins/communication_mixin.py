# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.protocols import CommunicationProtocol
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


class CommunicationMixin(AIPerfLifecycleMixin, ABC):
    """Mixin to provide access to a CommunicationProtocol instance. This mixin should be inherited
    by any mixin that needs access to the communication layer to create Communication clients.
    """

    def __init__(
        self, config=None, *, service_config: object | None = None, **kwargs
    ) -> None:
        if config is None:
            config = service_config
        super().__init__(config=config, **kwargs)
        self.service_config = config
        CommClass = plugins.get_class(
            PluginType.COMMUNICATION, self.service_config.comm_config.comm_backend
        )
        self.comms: CommunicationProtocol = CommClass(
            config=self.service_config.comm_config
        )
        self.attach_child_lifecycle(self.comms)
