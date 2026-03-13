# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC

from aiperf.common.hooks import (
    AIPerfHook,
    provides_hooks,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin


@provides_hooks(AIPerfHook.ON_COMMAND)
class CommandHandlerMixin(MessageBusClientMixin, ABC):
    """Mixin that declares @on_command hook support for services.

    Command dispatch is handled by the DEALER/ROUTER control channel:
    - BaseComponentService dispatches incoming Command structs to @on_command hooks
    - SystemController dispatches commands received on the ROUTER socket

    This mixin's role is to declare the ON_COMMAND hook type via @provides_hooks
    so that hook discovery works correctly in the class hierarchy.
    """

    def __init__(
        self,
        service_config: object,
        user_config: object,
        service_id: str,
        **kwargs,
    ) -> None:
        self.service_config = service_config
        self.user_config = user_config
        self.service_id = service_id

        super().__init__(
            service_config=self.service_config,
            user_config=self.user_config,
            **kwargs,
        )
