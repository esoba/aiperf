# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.auth.base_signer import SignedRequest
from aiperf.common.hooks import on_init
from aiperf.common.mixins import AIPerfLifecycleMixin

if TYPE_CHECKING:
    from aiperf.common.models.model_endpoint_info import ModelEndpointInfo


class SigV4RequestSigner(AIPerfLifecycleMixin):
    """AWS SigV4 request signer using botocore.

    Signs HTTP requests with AWS Signature Version 4 for authenticating
    against SageMaker, Bedrock, API Gateway, and other SigV4-protected endpoints.
    Uses botocore's credential chain for automatic credential discovery and refresh.
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs) -> None:
        super().__init__(**kwargs)
        self.region: str | None = model_endpoint.endpoint.aws_region
        self.service: str | None = model_endpoint.endpoint.aws_service
        self.profile: str | None = model_endpoint.endpoint.aws_profile
        self._credentials = None

    @on_init
    async def _init_credentials(self) -> None:
        """Initialize botocore session and resolve credentials."""
        try:
            import botocore.session
        except ImportError as e:
            raise ImportError(
                "SigV4 auth requires botocore. Install with: uv pip install aiperf[aws]"
            ) from e

        session = botocore.session.Session()
        if self.profile:
            session.set_config_variable("profile", self.profile)

        self._credentials = session.get_credentials()
        if self._credentials is None:
            raise ValueError(
                "No AWS credentials found. Configure via environment variables, "
                "~/.aws/credentials, or IAM role."
            )

        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials

        self._SigV4Auth = SigV4Auth
        self._AWSRequest = AWSRequest
        self._Credentials = Credentials

    async def sign(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> SignedRequest:
        """Sign an HTTP request with AWS SigV4.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full request URL
            headers: Current request headers (will be included in signature)
            body: Request body bytes (hashed for signature)

        Returns:
            SignedRequest with original + auth headers merged
        """
        frozen = self._credentials.get_frozen_credentials()
        credentials = self._Credentials(
            frozen.access_key, frozen.secret_key, frozen.token
        )

        request = self._AWSRequest(method=method, url=url, data=body, headers=headers)
        self._SigV4Auth(credentials, self.service, self.region).add_auth(request)
        return SignedRequest(headers=dict(request.headers))
