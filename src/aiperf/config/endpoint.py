# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Endpoint - Server connection and API configuration
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

from aiperf.common.enums import (
    ConnectionReuseStrategy,
    ModelSelectionStrategy,
)
from aiperf.config.constraints import ConstraintsMixin, RequiredIf
from aiperf.plugin.enums import (
    EndpointType,
    TransportType,
    URLSelectionStrategy,
)

__all__ = [
    "EndpointConfig",
    "TemplateConfig",
]


class TemplateConfig(BaseModel):
    """
    Configuration for custom template-based endpoints.

    When endpoint type is "template", this configures how requests
    are formatted and responses are parsed.
    """

    model_config = ConfigDict(extra="forbid")

    body: Annotated[
        str,
        Field(
            description="Jinja2 template string for request body. "
            "Variables: {{prompt}}, {{max_tokens}}, {{model}}, {{messages}}, etc.",
        ),
    ]

    response_field: Annotated[
        str,
        Field(
            default="text",
            description="JSON path to extract response text from API response. "
            "Use dot notation for nested fields: 'choices.0.message.content'.",
        ),
    ]


class EndpointConfig(BaseModel, ConstraintsMixin):
    """
    Endpoint configuration for connecting to inference servers.

    This section configures how AIPerf connects to and communicates
    with the target inference server(s). It supports multiple URLs
    for load-balanced deployments and various API types.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    urls: Annotated[
        list[str],
        Field(
            min_length=1,
            description="List of server URLs to benchmark. "
            "Requests distributed according to url_strategy. "
            "Example: ['http://localhost:8000/v1/chat/completions']",
        ),
    ]

    url_strategy: Annotated[
        URLSelectionStrategy,
        Field(
            default=URLSelectionStrategy.ROUND_ROBIN,
            description="Strategy for distributing requests across multiple URLs. "
            "round_robin cycles through URLs in order.",
        ),
    ]

    type: Annotated[
        EndpointType,
        Field(
            default=EndpointType.CHAT,
            description="API endpoint type determining request/response format. "
            "chat: OpenAI chat completions, completions: OpenAI completions, "
            "embeddings: vector embeddings, rankings: reranking, "
            "template: custom format.",
        ),
    ]

    path: Annotated[
        str | None,
        Field(
            default=None,
            description="Override default endpoint path. "
            "Use for servers with non-standard API paths. "
            "Example: '/custom/v2/generate'",
        ),
    ]

    api_key: Annotated[
        str | None,
        Field(
            default=None,
            description="API authentication key. "
            "Supports environment variable substitution: ${OPENAI_API_KEY}. "
            "Can also use ${VAR:default} syntax for defaults.",
        ),
    ]

    timeout: Annotated[
        float,
        Field(
            default=600.0,
            gt=0.0,
            description="Request timeout in seconds. "
            "Requests exceeding this duration are marked as failed. "
            "Should exceed expected max response time.",
        ),
    ]

    streaming: Annotated[
        bool,
        Field(
            default=False,
            description="Enable streaming (Server-Sent Events) responses. "
            "Required for accurate TTFT (time to first token) measurement. "
            "Server must support streaming for this to work.",
        ),
    ]

    transport: Annotated[
        TransportType | None,
        Field(
            default=None,
            description="HTTP transport protocol (http/https). "
            "Auto-detected from URL scheme if not specified. "
            "Explicit setting overrides auto-detection.",
        ),
    ]

    connection_reuse: Annotated[
        ConnectionReuseStrategy,
        Field(
            default=ConnectionReuseStrategy.POOLED,
            description="HTTP connection management strategy. "
            "pooled: shared connection pool (fastest), "
            "never: new connection per request (includes TCP overhead), "
            "sticky_sessions: dedicated connection per session.",
        ),
    ]

    use_legacy_max_tokens: Annotated[
        bool,
        Field(
            default=False,
            description="Use 'max_tokens' field instead of 'max_completion_tokens'. "
            "Enable for compatibility with older OpenAI API versions.",
        ),
    ]

    use_server_token_count: Annotated[
        bool,
        Field(
            default=False,
            description="Use server-reported token counts from response usage field. "
            "When true, trusts usage.prompt_tokens and usage.completion_tokens. "
            "When false, counts tokens locally using configured tokenizer.",
        ),
    ]

    template: Annotated[
        TemplateConfig | None,
        RequiredIf("type", EndpointType.TEMPLATE),
        Field(
            default=None,
            description="Custom template configuration for template endpoint type. "
            "Only used when type='template'. "
            "Defines request body format and response parsing.",
        ),
    ]

    headers: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            description="Custom HTTP headers to include in all requests. "
            "Useful for authentication, tracing, or routing. "
            "Values support environment variable substitution.",
        ),
    ]

    download_video_content: Annotated[
        bool,
        Field(
            default=False,
            description="For video generation endpoints, download the video content after "
            "generation completes. When enabled, request latency includes the video "
            "download time. When disabled, only generation time is measured.",
        ),
    ]

    extra: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="Additional fields to include in request body. "
            "Merged into every request. "
            "Common fields: temperature, top_p, top_k, stop.",
        ),
    ]

    # Private attributes populated by AIPerfConfig.inject_endpoint_compat
    _model_names: list[str] = PrivateAttr(default_factory=list)
    _model_selection_strategy: ModelSelectionStrategy = PrivateAttr(
        default=ModelSelectionStrategy.ROUND_ROBIN
    )

    # =========================================================================
    # BACKWARD COMPATIBILITY PROPERTIES
    # =========================================================================

    @property
    def model_names(self) -> list[str]:
        """Model names from parent config. Populated by AIPerfConfig."""
        return self._model_names

    @property
    def model_selection_strategy(self) -> ModelSelectionStrategy:
        """Model selection strategy from parent config. Populated by AIPerfConfig."""
        return self._model_selection_strategy

    @property
    def timeout_seconds(self) -> float:
        """Alias for timeout (legacy field name)."""
        return self.timeout

    @property
    def custom_endpoint(self) -> str | None:
        """Alias for path (legacy field name)."""
        return self.path

    @property
    def url_selection_strategy(self) -> URLSelectionStrategy:
        """Alias for url_strategy (legacy field name)."""
        return self.url_strategy

    @property
    def connection_reuse_strategy(self) -> ConnectionReuseStrategy:
        """Alias for connection_reuse (legacy field name)."""
        return self.connection_reuse

    @property
    def url(self) -> str:
        """Return the first URL."""
        return self.urls[0]

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @model_validator(mode="before")
    @classmethod
    def normalize_before_validation(cls, data: Any) -> Any:
        """Normalize endpoint config before validation.

        Handles:
            - url → urls (singular to plural, wrapped in list)
            - Auto-set type to 'template' when template field is provided
        """
        if not isinstance(data, dict):
            return data

        # url → urls (singular to plural)
        if "url" in data and "urls" not in data:
            url = data.pop("url")
            data["urls"] = [url] if isinstance(url, str) else url

        # Auto-detect template type
        if "template" in data and data["template"] is not None and "type" not in data:
            data["type"] = EndpointType.TEMPLATE

        return data

    @model_validator(mode="after")
    def validate_streaming_support(self) -> EndpointConfig:
        """Validate streaming is supported for the endpoint type."""
        if not self.streaming:
            return self

        # Lazy import to avoid circular dependency
        try:
            from aiperf.plugin import plugins

            metadata = plugins.get_endpoint_metadata(self.type)
            if not metadata.supports_streaming:
                import warnings

                warnings.warn(
                    f"Streaming is not supported for endpoint type '{self.type}'. "
                    "Streaming will be disabled.",
                    UserWarning,
                    stacklevel=2,
                )
                # Note: We don't modify self.streaming here as Pydantic models
                # should be treated as immutable after validation. The runtime
                # code should check streaming support.
        except ImportError:
            # Plugin system not available during validation
            pass

        return self
