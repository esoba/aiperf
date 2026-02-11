# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Messages for the content server service."""

from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.types import MessageTypeT
from aiperf.content_server.models import ContentServerStatus


class ContentServerStatusMessage(BaseServiceMessage):
    """Message from ContentServer to SystemController indicating content server availability."""

    message_type: MessageTypeT = MessageType.CONTENT_SERVER_STATUS

    status: ContentServerStatus = Field(
        description="Content server status including enabled state, base URL, and content directory"
    )
