# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest

from aiperf.auth.base_signer import SignedRequest
from aiperf.auth.sigv4_signer import SigV4RequestSigner
from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.plugin.enums import EndpointType


def _make_model_endpoint(
    aws_region: str | None = "us-east-1",
    aws_service: str | None = "sagemaker",
    aws_profile: str | None = None,
) -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_urls=["https://endpoint.sagemaker.us-east-1.amazonaws.com"],
            aws_region=aws_region,
            aws_service=aws_service,
            aws_profile=aws_profile,
        ),
    )


class TestSigV4RequestSignerInit:
    def test_stores_config(self) -> None:
        signer = SigV4RequestSigner(
            model_endpoint=_make_model_endpoint(
                aws_region="eu-west-1",
                aws_service="bedrock-runtime",
                aws_profile="prod",
            )
        )
        assert signer.region == "eu-west-1"
        assert signer.service == "bedrock-runtime"
        assert signer.profile == "prod"


class TestSigV4RequestSignerInitCredentials:
    @pytest.mark.asyncio
    async def test_init_credentials_success(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        mock_creds = MagicMock()
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_creds

        with patch(
            "botocore.session.Session",
            return_value=mock_session,
        ):
            await signer._init_credentials()

        assert signer._credentials is mock_creds

    @pytest.mark.asyncio
    async def test_init_credentials_with_profile(self) -> None:
        signer = SigV4RequestSigner(
            model_endpoint=_make_model_endpoint(aws_profile="staging")
        )
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = MagicMock()

        with patch(
            "botocore.session.Session",
            return_value=mock_session,
        ):
            await signer._init_credentials()

        mock_session.set_config_variable.assert_called_once_with("profile", "staging")

    @pytest.mark.asyncio
    async def test_init_credentials_no_profile_skips_set(self) -> None:
        signer = SigV4RequestSigner(
            model_endpoint=_make_model_endpoint(aws_profile=None)
        )
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = MagicMock()

        with patch(
            "botocore.session.Session",
            return_value=mock_session,
        ):
            await signer._init_credentials()

        mock_session.set_config_variable.assert_not_called()

    @pytest.mark.asyncio
    async def test_init_credentials_none_raises(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None

        with (
            patch(
                "botocore.session.Session",
                return_value=mock_session,
            ),
            pytest.raises(ValueError, match="No AWS credentials found"),
        ):
            await signer._init_credentials()


def _setup_signer_for_sign(
    signer: SigV4RequestSigner,
    access_key: str = "AK",
    secret_key: str = "SK",
    token: str | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Set up a signer with mock credentials and botocore classes for sign() tests."""
    mock_frozen = MagicMock(access_key=access_key, secret_key=secret_key, token=token)
    mock_creds = MagicMock()
    mock_creds.get_frozen_credentials.return_value = mock_frozen
    signer._credentials = mock_creds

    mock_sigv4_cls = MagicMock()
    mock_sigv4_cls.return_value.add_auth.side_effect = lambda r: None

    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    signer._SigV4Auth = mock_sigv4_cls
    signer._AWSRequest = AWSRequest
    signer._Credentials = Credentials

    return mock_creds, mock_sigv4_cls


class TestSigV4RequestSignerSign:
    @pytest.mark.asyncio
    async def test_sign_adds_authorization_header(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        _, mock_sigv4_cls = _setup_signer_for_sign(
            signer,
            access_key="AKIAIOSFODNN7EXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )

        def add_auth_side_effect(request):
            request.headers["Authorization"] = "AWS4-HMAC-SHA256 Credential=..."
            request.headers["X-Amz-Date"] = "20260318T120000Z"

        mock_sigv4_cls.return_value.add_auth.side_effect = add_auth_side_effect

        headers = {"Content-Type": "application/json", "Host": "example.com"}
        body = b'{"prompt": "hello"}'
        result = await signer.sign("POST", "https://example.com/invoke", headers, body)

        assert isinstance(result, SignedRequest)
        assert "Authorization" in result.headers
        assert result.headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "X-Amz-Date" in result.headers
        assert result.url is None
        assert result.body is None

    @pytest.mark.asyncio
    async def test_sign_preserves_existing_headers(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        _setup_signer_for_sign(
            signer, access_key="AKID", secret_key="SECRET", token="TOKEN"
        )

        headers = {"Content-Type": "application/json", "X-Custom": "value"}
        result = await signer.sign("GET", "https://example.com", headers, None)

        assert result.headers["Content-Type"] == "application/json"
        assert result.headers["X-Custom"] == "value"

    @pytest.mark.asyncio
    async def test_sign_calls_get_frozen_credentials_each_time(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        mock_creds, _ = _setup_signer_for_sign(signer)

        await signer.sign("POST", "https://a.com", {}, b"")
        await signer.sign("POST", "https://b.com", {}, b"")

        assert mock_creds.get_frozen_credentials.call_count == 2

    @pytest.mark.asyncio
    async def test_sign_passes_correct_service_and_region(self) -> None:
        signer = SigV4RequestSigner(
            model_endpoint=_make_model_endpoint(
                aws_region="ap-southeast-1", aws_service="bedrock-runtime"
            )
        )
        _, mock_sigv4_cls = _setup_signer_for_sign(signer)

        await signer.sign("POST", "https://a.com", {}, b"")

        mock_sigv4_cls.assert_called_once()
        _, call_service, call_region = mock_sigv4_cls.call_args[0]
        assert call_service == "bedrock-runtime"
        assert call_region == "ap-southeast-1"
