# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coding content generator for realistic coding trace replay.

Generates structurally plausible coding content (code, bash output, JSON,
errors, git diffs, CI output, configs, markdown, test output, user prompts)
using template-based generation with random identifiers.

Unlike PromptGenerator which uses Shakespeare as its corpus, this generator
builds two token pools from structural templates:
- text_pool: user prompts (natural language coding requests)
- tool_pool: mixed technical content (code, errors, diffs, configs, etc.)

Generation uses window slicing from pre-built token pools, same as PromptGenerator.
"""

from __future__ import annotations

from aiperf.common import random_generator as rng
from aiperf.common.config import PromptConfig
from aiperf.common.exceptions import ConfigurationError, NotInitializedError
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.base import BaseGenerator

# fmt: off
# -- Vocabulary tuples for template fills --

_MODULES = (
    "auth", "cache", "config", "database", "events", "handler", "logger",
    "metrics", "middleware", "pipeline", "processor", "registry", "router",
    "scheduler", "serializer", "service", "storage", "transport", "validator",
    "worker", "adapter", "broker", "collector", "dispatcher", "encoder",
    "factory", "gateway", "indexer", "manager", "monitor", "notifier",
    "observer", "parser", "provider", "queue", "resolver", "scanner",
    "session", "sink", "source", "stream", "transformer", "uploader",
    # web / HTTP
    "api", "webhook", "cors", "oauth", "graphql", "grpc", "websocket",
    "rate_limiter", "proxy", "load_balancer", "reverse_proxy",
    # database / data
    "migration", "schema", "repository", "connection_pool", "query_builder",
    "data_loader", "orm", "replication", "sharding", "backup",
    # ML / data science
    "inference", "tokenizer", "embedding", "feature_store", "model_registry",
    "trainer", "evaluator", "dataset", "sampler", "checkpoint",
    # DevOps / infra
    "deployer", "provisioner", "orchestrator", "health_check", "autoscaler",
    "dns_resolver", "cert_manager", "secret_store", "telemetry", "alerter",
    # security
    "firewall", "encryptor", "key_manager", "audit", "compliance",
)

_CLASSES = (
    "RequestHandler", "DataProcessor", "EventEmitter", "CacheManager",
    "ConnectionPool", "TaskScheduler", "MessageBroker", "StateManager",
    "ConfigLoader", "MetricsCollector", "RateLimiter", "CircuitBreaker",
    "RetryPolicy", "BatchProcessor", "StreamReader", "TokenValidator",
    "SessionStore", "PermissionChecker", "ResourceAllocator", "HealthMonitor",
    "LoadBalancer", "QueueConsumer", "IndexBuilder", "SchemaValidator",
    "PipelineStage", "WorkerPool", "ContextManager", "PluginLoader",
    "TemplateEngine", "SignalHandler", "ProtocolAdapter", "BufferManager",
    "ThrottleController", "RegistryClient", "LockManager", "SnapshotStore",
    "AuditLogger", "FeatureToggle", "MigrationRunner", "DeploymentManager",
    # HTTP layer
    "HttpClient", "RouteResolver", "CorsMiddleware", "AuthMiddleware",
    "ResponseSerializer", "RequestParser", "WebSocketManager", "ApiGateway",
    # data layer
    "QueryExecutor", "TransactionManager", "MigrationEngine", "PoolManager",
    "ReplicaSelector", "ShardRouter", "CursorIterator", "ChangeStream",
    # error types
    "RetryableError", "ValidationError", "TimeoutError", "QuotaExceeded",
    "ConflictError", "NotFoundError", "AuthorizationError", "RateLimitError",
    # ML / inference
    "ModelLoader", "TokenEncoder", "EmbeddingStore", "FeatureExtractor",
    "InferenceEngine", "BatchScheduler", "GradientAccumulator", "Checkpoint",
    # infra / orchestration
    "ServiceMesh", "HealthProbe", "AutoScaler", "SecretProvider",
    "CertRotator", "DnsCache", "TelemetryExporter", "AlertDispatcher",
)

_METHODS = (
    "process", "handle", "validate", "transform", "execute", "initialize",
    "configure", "dispatch", "resolve", "serialize", "deserialize", "encode",
    "decode", "publish", "subscribe", "notify", "aggregate", "partition",
    "schedule", "allocate", "release", "acquire", "flush", "compress",
    "decompress", "authenticate", "authorize", "revoke", "checkpoint",
    "rollback", "migrate", "replicate", "synchronize", "reconcile",
    "invalidate", "prefetch", "evict", "rebalance", "throttle", "retry",
    "render", "persist", "hydrate", "prune", "drain", "backfill",
    "enqueue", "dequeue", "broadcast", "handshake", "negotiate", "probe",
    "rotate", "shard", "merge", "split", "compact", "snapshot",
    "finalize", "abort", "resume", "suspend", "escalate", "demote",
    "promote", "quarantine", "scrub", "warm_up", "cool_down", "heal",
    "reclaim", "tombstone", "seal", "unseal", "bootstrap", "teardown",
)

_TYPES = (
    "str", "int", "float", "bool", "bytes", "dict", "list", "tuple", "set",
    "None", "Any", "Optional", "Sequence", "Mapping", "Iterator", "Callable",
    "Awaitable", "Coroutine", "AsyncIterator", "Generator", "TypeVar",
    "Protocol", "ClassVar", "Final", "Literal", "Union", "Type",
    "NamedTuple", "TypedDict", "Annotated", "ParamSpec", "Self",
)

_VARS = (
    "result", "data", "config", "context", "payload", "response", "request",
    "buffer", "cursor", "offset", "count", "total", "index", "batch",
    "chunk", "token", "record", "entry", "item", "value", "key", "state",
    "status", "event", "message", "signal", "metric", "timestamp", "duration",
    "timeout", "retries", "threshold", "capacity", "interval", "priority",
    "sequence", "channel", "endpoint", "header", "session", "connection",
    "pipeline", "schema", "trace_id", "tenant_id", "batch_size", "page_size",
    "shard_key", "replica_id", "worker_id", "partition_key", "ttl",
    "max_retries", "backoff", "jitter", "watermark", "checkpoint_id",
    "correlation_id", "span_id", "parent_id", "depth", "fanout",
    "concurrency", "rate", "window", "lag", "drift", "skew",
    "epoch", "generation", "version", "revision", "digest", "nonce",
)

_FILE_PATHS = (
    "src/main.py", "src/config.py", "src/models.py", "src/routes.py",
    "src/utils.py", "src/middleware.py", "src/database.py", "src/auth.py",
    "tests/test_main.py", "tests/test_models.py", "tests/conftest.py",
    "lib/core.go", "lib/handler.go", "lib/service.go", "lib/types.go",
    "pkg/api/server.go", "pkg/api/client.go", "pkg/store/store.go",
    "cmd/server/main.go", "internal/config/config.go",
    "src/lib.rs", "src/main.rs", "src/config.rs", "src/error.rs",
    "src/handler.rs", "src/models.rs", "src/routes.rs",
    "src/index.ts", "src/app.ts", "src/types.ts", "src/api.ts",
    "src/components/App.tsx", "src/components/Form.tsx",
    "Dockerfile", "Makefile", "docker-compose.yml", "pyproject.toml",
    ".github/workflows/ci.yml", "kubernetes/deployment.yaml",
)

_HTTP_ROUTES = (
    "/api/v1/users", "/api/v1/items", "/api/v1/orders", "/api/v1/auth/login",
    "/api/v1/auth/refresh", "/api/v2/search", "/api/v2/analytics",
    "/health", "/ready", "/metrics", "/api/v1/webhooks", "/api/v1/uploads",
    "/api/v1/notifications", "/api/v1/settings", "/api/v1/billing",
    "/api/v1/teams/{team_id}/members", "/api/v1/projects/{project_id}/runs",
    "/api/v1/tenants/{tenant_id}/quota", "/internal/gc", "/internal/debug/pprof",
)

_DB_TABLES = (
    "users", "orders", "items", "sessions", "audit_log", "migrations",
    "api_keys", "rate_limits", "notifications", "webhooks", "tenants",
    "permissions", "invitations", "uploads", "billing_events",
    "job_queue", "dead_letter", "feature_flags", "schema_versions", "locks",
)

_STATUS_CODES = (
    "200 OK", "201 Created", "204 No Content", "301 Moved Permanently",
    "400 Bad Request", "401 Unauthorized", "403 Forbidden", "404 Not Found",
    "409 Conflict", "429 Too Many Requests", "500 Internal Server Error",
    "502 Bad Gateway", "503 Service Unavailable", "504 Gateway Timeout",
)

_LANG_FILE_PATHS: dict[str, tuple[str, ...]] = {
    "python": (
        "src/main.py", "src/config.py", "src/models.py", "src/routes.py",
        "src/utils.py", "src/middleware.py", "src/database.py", "src/auth.py",
        "tests/test_main.py", "tests/test_models.py", "tests/conftest.py",
        "pyproject.toml", "Dockerfile", "Makefile",
        "src/api/v1/endpoints.py", "src/api/v1/schemas.py", "src/api/deps.py",
        "src/core/security.py", "src/core/events.py", "src/services/worker.py",
        "src/repositories/base.py", "tests/integration/test_api.py",
    ),
    "go": (
        "lib/core.go", "lib/handler.go", "lib/service.go", "lib/types.go",
        "pkg/api/server.go", "pkg/api/client.go", "pkg/store/store.go",
        "cmd/server/main.go", "internal/config/config.go",
        "go.mod", "go.sum", "Makefile",
        "internal/middleware/auth.go", "internal/middleware/ratelimit.go",
        "internal/repository/postgres.go", "internal/service/worker.go",
        "pkg/api/middleware.go", "pkg/api/routes.go",
        "internal/telemetry/tracing.go", "internal/health/probe.go",
    ),
    "rust": (
        "src/lib.rs", "src/main.rs", "src/config.rs", "src/error.rs",
        "src/handler.rs", "src/models.rs", "src/routes.rs",
        "Cargo.toml", "Cargo.lock",
        "src/middleware/auth.rs", "src/middleware/tracing.rs",
        "src/repository/mod.rs", "src/repository/postgres.rs",
        "src/service/mod.rs", "src/service/worker.rs",
        "tests/integration/api_test.rs", "benches/throughput.rs",
    ),
    "typescript": (
        "src/index.ts", "src/app.ts", "src/types.ts", "src/api.ts",
        "src/components/App.tsx", "src/components/Form.tsx",
        "src/utils.ts", "src/middleware.ts", "src/routes.ts",
        "package.json", "tsconfig.json", "Dockerfile",
        "src/services/auth.service.ts", "src/services/worker.service.ts",
        "src/middleware/rate-limiter.ts", "src/middleware/error-handler.ts",
        "src/models/user.model.ts", "src/models/order.model.ts",
        "src/repositories/base.repository.ts", "tests/integration/api.test.ts",
    ),
}

_ERROR_MESSAGES = (
    "connection refused", "timeout exceeded", "permission denied",
    "resource not found", "invalid argument", "out of memory",
    "deadlock detected", "rate limit exceeded", "authentication failed",
    "schema validation error", "serialization error", "buffer overflow",
    "index out of range", "null pointer dereference", "type mismatch",
    "missing required field", "duplicate key", "constraint violation",
    "circular dependency detected", "maximum recursion depth exceeded",
    "transaction aborted", "lock timeout after 30s", "quota exceeded",
    "connection pool exhausted", "certificate expired", "DNS resolution failed",
    "checksum mismatch", "payload too large", "stale read",
    "leader election in progress", "shard unavailable", "replica lag exceeded",
    "write conflict detected", "token revoked", "session expired",
    "circuit breaker open", "backpressure applied", "partition offline",
    "consensus timeout", "snapshot corrupted", "migration in progress",
)

_CLI_COMMANDS = (
    "git status", "git diff HEAD~1", "git log --oneline -10",
    "docker build -t app .", "docker compose up -d",
    "kubectl get pods -n default", "kubectl apply -f deployment.yaml",
    "cargo build --release", "cargo test -- --nocapture",
    "go build ./...", "go test -v ./...", "go vet ./...",
    "npm run build", "npm test", "npx tsc --noEmit",
    "pytest -xvs tests/", "ruff check .", "mypy src/",
    "make build", "make test", "make lint",
    "curl -s http://localhost:8080/health",
    "ps aux | grep python", "top -bn1 | head -20",
    # k8s / infra
    "kubectl describe pod app-7d4b8f-xz9k", "kubectl logs -f deploy/api --tail=100",
    "kubectl rollout status deploy/worker", "kubectl top nodes",
    "helm upgrade --install app ./chart -f values.yaml",
    "terraform plan -out=tfplan", "terraform apply tfplan",
    # redis / data stores
    "redis-cli INFO memory", "redis-cli --latency-history -i 1",
    "pg_dump -Fc mydb > backup.dump", "mongosh --eval 'db.stats()'",
    # perf / profiling
    "perf stat -e cache-misses,cache-references ./bin/server",
    "strace -c -p $(pgrep server)", "valgrind --tool=memcheck ./bin/app",
    "pprof -http=:6060 http://localhost:6060/debug/pprof/heap",
    # load testing
    "wrk -t12 -c400 -d30s http://localhost:8080/api/v1/items",
    "hey -n 10000 -c 100 http://localhost:8080/health",
    "ab -n 5000 -c 50 http://localhost:8080/",
    # misc dev
    "find . -name '*.py' | xargs wc -l | tail -1",
    "du -sh node_modules/ target/ dist/",
    "lsof -i :8080", "ss -tlnp | grep 8080",
    "journalctl -u myapp --since '1 hour ago'",
)

_GO_PACKAGES = (
    "fmt", "os", "io", "net", "http", "context", "sync", "time",
    "strings", "strconv", "encoding/json", "log", "errors", "math",
    "sort", "bytes", "crypto", "regexp", "path/filepath", "database/sql",
)

_RUST_CRATES = (
    "std::io", "std::fs", "std::collections", "std::sync", "std::fmt",
    "serde", "serde_json", "tokio", "anyhow", "thiserror", "tracing",
    "clap", "reqwest", "axum", "sqlx", "uuid", "chrono", "regex",
)

_TS_IMPORTS = (
    "express", "axios", "lodash", "zod", "prisma", "next",
    "react", "react-dom", "typescript", "jest", "vitest",
    "node:fs", "node:path", "node:http", "node:crypto",
)

_DECORATORS = (
    "@staticmethod", "@classmethod", "@property", "@abstractmethod",
    "@override", "@cached_property", "@dataclass", "@lru_cache",
    "@pytest.mark.asyncio", "@pytest.mark.parametrize",
    "@app.route", "@app.get", "@app.post", "@router.get",
)

_USER_REQUESTS = (
    # simple one-liners (original)
    "Fix the failing test in {module} — it returns {error}",
    "Add retry logic to {cls}.{method}() with exponential backoff",
    "Refactor the {method} function to use async/await instead of callbacks",
    "The {cls} class is throwing {error} when {var} is None",
    "Add input validation for the {var} parameter in {method}()",
    "Write unit tests for {cls}.{method}() covering edge cases",
    "Optimize the {method} query — it's taking too long with large datasets",
    "Add logging to {cls} so we can debug {error} in production",
    "Move the {method} logic from {module} to a shared utility",
    "Implement caching for {cls}.{method}() to reduce database load",
    "Update the {module} config to support environment variable overrides",
    "Add a health check endpoint that verifies {cls} connectivity",
    "The CI is failing because {module} import is broken after the refactor",
    "Create a migration script for the {var} schema change",
    "Add rate limiting to the {method} endpoint — we're getting hammered",
    "Debug why {cls}.{method}() returns stale data after {method}()",
    "Add pagination support to the {method}() response",
    "Implement graceful shutdown for the {cls} worker pool",
    "The {module} integration test is flaky — fix the race condition",
    "Add type hints to all public methods in {cls}",
    "Refactor {module} to use dependency injection instead of globals",
    "Add metrics collection for {method}() latency and error rates",
    "Fix the memory leak in {cls} — it's not releasing {var} properly",
    "Implement {method} fallback when the primary {module} is unavailable",
    "Add request/response logging middleware for the {module} API",
    "Write a load test for {cls}.{method}() with concurrent connections",
    "Add circuit breaker pattern to {cls} for external API calls",
    "The {cls}.{method}() docstring is wrong — update it to match the code",
    "Implement batch processing for {method}() to handle bulk {var} updates",
    "Add WebSocket support to {cls} for real-time {var} updates",
    # multi-step tasks
    "Migrate {cls}.{method}() from sync to async — it's called in 3 places across {module} and needs backward compat",
    "Split the {cls} class into two: one for {method} and one for the {var} lifecycle management",
    "We need to add {method}() to {cls}, then wire it into the {module} pipeline and add an integration test",
    "Extract the {method} logic from {cls} into a standalone service, update all callers, and add a deprecation warning to the old path",
    "Rewrite the {module} retry logic: replace the sleep loop with a proper backoff strategy using {cls}",
    # error context prompts
    "Getting {error} after upgrading {module} to the latest version — only happens under load",
    "The {cls}.{method}() call started returning {error} after we merged the {var} migration PR",
    "Users are reporting {error} intermittently — the {module} logs show {var} is sometimes null",
    "After deploying the {method} change, we see {error} on about 5%% of requests to {cls}",
    "The staging environment throws {error} but prod is fine — suspect it's the {var} config difference",
    # file path references
    "Look at {module}/{cls}.{method}() — the {var} parameter is never validated before being passed to the database layer",
    "In the {module} service, the {method}() function at line ~200 has a subtle bug with {var} boundary handling",
    "The {cls} constructor in {module} initializes {var} too early — move it to the {method}() call site",
    # constraint-carrying
    "Add {method}() to {cls} without breaking the existing API contract — we have downstream consumers",
    "Optimize {cls}.{method}() for the case where {var} has over 10K entries, but keep the simple path fast too",
    "Fix the {error} in {module} — but don't change the public interface, we're in a code freeze for other modules",
    "Add telemetry to {cls}.{method}() without adding any new dependencies to the {module} package",
    # multi-sentence with background
    "We profiled the {method} endpoint and {var} is growing unbounded in {cls}. We need to add eviction or cap the size. The 99th percentile latency spiked 3x last week.",
    "The {cls} pool keeps hitting {error} during peak hours. We scaled horizontally but the issue persists. I think {method}() is holding a lock too long.",
    "After the last {module} refactor, {cls}.{method}() no longer returns deterministic results. The old tests still pass but the integration tests are flaky. Might be a race condition on {var}.",
    "We're moving from REST to gRPC for the {module} service. Start by converting {cls}.{method}() — it's the most latency-sensitive endpoint. Keep the REST handler as a thin adapter for backward compat.",
    # review / debugging style
    "Can you review the {cls}.{method}() implementation? I think the error handling around {var} is wrong",
    "Why does {cls} create a new {var} on every call to {method}()? Seems wasteful",
    "Walk me through the {method}() flow in {module} — I need to understand where {var} gets validated",
    "Is there a reason {cls}.{method}() catches Exception instead of the specific {error}?",
    # infra / DevOps
    "Add a Dockerfile for the {module} service that runs {cls} on port 8080 with health checks",
    "The k8s deployment for {module} keeps OOMKilling — add memory limits and check if {cls} leaks during {method}()",
    "Set up a GitHub Action that runs the {module} tests, lints with ruff, and blocks merge on failure",
    "Add Prometheus metrics for {cls}.{method}() — we need p50/p95/p99 latency and error rate by status code",
    # data / schema
    "Add a new {var} column to the {module} table with a default value and backfill script",
    "The {cls} serializer is dropping {var} fields when they're empty lists — should preserve them as []",
    "Normalize the {var} schema in {module}: split the nested object into its own table with a foreign key",
)

_TEXT_POOL_BLOCKS = 200
_BASELINE_POOL_TOKENS = 200_000

# Block counts per generator, weighted to match real trace data distribution:
# ~35% code, ~25% bash, ~15% JSON, ~10% errors, ~15% other (diffs, CI, configs, docs, tests)
_TOOL_POOL_BLOCK_COUNTS: dict[str, int] = {
    "_gen_python_code": 53,
    "_gen_go_code": 53,
    "_gen_rust_code": 53,
    "_gen_typescript_code": 53,
    "_gen_bash_output": 150,
    "_gen_json_response": 90,
    "_gen_error_traceback": 60,
    "_gen_tool_use_block": 45,
    "_gen_git_diff": 18,
    "_gen_cicd_output": 18,
    "_gen_config_file": 18,
    "_gen_markdown_doc": 18,
    "_gen_test_output": 18,
}

# Language-specific pool: single code generator replaces all 4, rest stays proportional.
# ~35% code, ~25% bash, ~15% JSON, ~10% errors/tests/configs, ~15% other
_LANGUAGE_POOL_BLOCK_COUNTS: dict[str, int] = {
    "code": 160,
    "_gen_bash_output": 115,
    "_gen_json_response": 70,
    "_gen_error_traceback": 45,
    "_gen_tool_use_block": 30,
    "_gen_test_output": 25,
    "_gen_config_file": 20,
    "_gen_git_diff": 15,
    "_gen_cicd_output": 15,
    "_gen_markdown_doc": 10,
}

_LANGUAGE_AGNOSTIC_GENERATORS = (
    "_gen_bash_output", "_gen_json_response", "_gen_git_diff",
    "_gen_tool_use_block",
)

_LANGUAGE_GENERATORS: dict[str, tuple[str, ...]] = {
    "python": ("_gen_python_code",),
    "go": ("_gen_go_code",),
    "rust": ("_gen_rust_code",),
    "typescript": ("_gen_typescript_code",),
}
# fmt: on


class CodingContentGenerator(BaseGenerator):
    """Generator for structurally plausible coding content.

    Builds two pre-tokenized pools from template-based content:
    - text_pool: natural language coding requests (~100K tokens)
    - tool_pool: mixed technical content — code, errors, diffs, etc. (~500K tokens)

    Supports both PromptGenerator-compatible interface and typed generation
    that selects the appropriate pool based on content type.
    """

    supports_typed_prompt = True

    def __init__(
        self,
        config: PromptConfig,
        tokenizer: Tokenizer,
        pool_tokens_target: int | None = None,
        **kwargs,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self._pool_scale = max(
            1.0, (pool_tokens_target or _BASELINE_POOL_TOKENS) / _BASELINE_POOL_TOKENS
        )

        self._template_rng = rng.derive("dataset.coding_content.template")
        self._corpus_rng = rng.derive("dataset.coding_content.corpus")
        self._length_rng = rng.derive("dataset.coding_content.length")

        super().__init__(config=config, tokenizer=tokenizer, **kwargs)

        self._text_pool: list[int] = []
        self._tool_pool: list[int] = []
        self._language_pools: dict[str, list[int]] = {}
        self._cache: dict[int, list[int]] = {}
        self._decoded_cache: dict[tuple[tuple[int, ...], int, int], str] = {}

        self._build_text_pool()
        self._build_tool_pool()

    # -- BaseGenerator interface --

    def generate(
        self,
        mean: int | None = None,
        stddev: int | None = None,
        hash_ids: list[int] | None = None,
    ) -> str:
        if hash_ids:
            return self._generate_cached_prompt(
                mean, hash_ids, self.config.input_tokens.block_size
            )
        num_tokens = self.calculate_num_tokens(mean, stddev)
        return self.generate_prompt(num_tokens)

    # -- PromptGenerator-compatible interface --

    def generate_prompt(self, num_tokens: int) -> str:
        tokens = self._sample_tokens(num_tokens, self._tool_pool)
        return self.tokenizer.decode(tokens)

    def generate_typed_prompt(self, num_tokens: int, content_type: str) -> str:
        """Generate a prompt from the pool matching the content type.

        Args:
            num_tokens: Number of tokens to generate.
            content_type: Either "text" (user prompts) or "tool_result" (code/technical).
        """
        pool = self._text_pool if content_type == "text" else self._tool_pool
        tokens = self._sample_tokens(num_tokens, pool)
        return self.tokenizer.decode(tokens)

    def generate_language_prompt(
        self, num_tokens: int, content_type: str, language: str
    ) -> str:
        """Generate a prompt from a language-specific pool.

        Args:
            num_tokens: Number of tokens to generate.
            content_type: Either "text" (user prompts) or "tool_result" (code/technical).
            language: Programming language for tool pool selection.
        """
        if content_type == "text":
            pool = self._text_pool
        else:
            pool = self.build_language_pool(language)
        tokens = self._sample_tokens(num_tokens, pool)
        return self.tokenizer.decode(tokens)

    def build_language_pool(self, language: str) -> list[int]:
        """Build or return a cached language-specific tool pool.

        Args:
            language: One of "python", "go", "rust", "typescript".

        Returns:
            Token pool for the given language.
        """
        if language in self._language_pools:
            return self._language_pools[language]

        lang_gens = _LANGUAGE_GENERATORS.get(language, ())
        code_count = int(_LANGUAGE_POOL_BLOCK_COUNTS["code"] * self._pool_scale)

        blocks: list[str] = []

        # Language-specific code blocks
        for gen_name in lang_gens:
            gen_fn = getattr(self, gen_name)
            for _ in range(code_count):
                blocks.append(gen_fn())

        # Language-agnostic generators
        for gen_name in _LANGUAGE_AGNOSTIC_GENERATORS:
            gen_fn = getattr(self, gen_name)
            count = int(_LANGUAGE_POOL_BLOCK_COUNTS[gen_name] * self._pool_scale)
            for _ in range(count):
                blocks.append(gen_fn())

        # Language-specific mixed generators (errors, tests, configs, CI/CD, markdown)
        for _ in range(
            int(_LANGUAGE_POOL_BLOCK_COUNTS["_gen_error_traceback"] * self._pool_scale)
        ):
            blocks.append(self._gen_error_traceback(language=language))
        for _ in range(
            int(_LANGUAGE_POOL_BLOCK_COUNTS["_gen_test_output"] * self._pool_scale)
        ):
            blocks.append(self._gen_test_output(language=language))
        for _ in range(
            int(_LANGUAGE_POOL_BLOCK_COUNTS["_gen_config_file"] * self._pool_scale)
        ):
            blocks.append(self._gen_config_file(language=language))
        for _ in range(
            int(_LANGUAGE_POOL_BLOCK_COUNTS["_gen_cicd_output"] * self._pool_scale)
        ):
            blocks.append(self._gen_cicd_output(language=language))
        for _ in range(
            int(_LANGUAGE_POOL_BLOCK_COUNTS["_gen_markdown_doc"] * self._pool_scale)
        ):
            blocks.append(self._gen_markdown_doc(language=language))

        # Shuffle so any window slice gets diverse content types
        self._template_rng.shuffle(blocks)

        text = "\n\n".join(blocks)
        pool = self.tokenizer.encode(text)
        self._language_pools[language] = pool
        self.debug(
            lambda: f"Built {language} pool with {len(pool)} tokens "
            f"from {len(blocks)} blocks"
        )
        return pool

    def calculate_num_tokens(
        self,
        mean: int | None = None,
        stddev: int | None = None,
    ) -> int:
        return self._length_rng.sample_positive_normal_integer(mean, stddev)

    # -- Pool building --

    def _build_text_pool(self) -> None:
        blocks: list[str] = []
        for _ in range(int(_TEXT_POOL_BLOCKS * self._pool_scale)):
            blocks.append(self._gen_user_prompt())
        text = "\n\n".join(blocks)
        self._text_pool = self.tokenizer.encode(text)
        self.debug(
            lambda: f"Built text pool with {len(self._text_pool)} tokens "
            f"from {len(blocks)} blocks"
        )

    def _build_tool_pool(self) -> None:
        blocks: list[str] = []
        for gen_name, count in _TOOL_POOL_BLOCK_COUNTS.items():
            gen_fn = getattr(self, gen_name)
            for _ in range(int(count * self._pool_scale)):
                blocks.append(gen_fn())
        self._template_rng.shuffle(blocks)
        text = "\n\n".join(blocks)
        self._tool_pool = self.tokenizer.encode(text)
        self.debug(
            lambda: f"Built tool pool with {len(self._tool_pool)} tokens "
            f"from {len(blocks)} blocks"
        )

    def _sample_tokens(self, num_tokens: int, pool: list[int]) -> list[int]:
        if not pool:
            raise NotInitializedError("Token pool is not initialized.")
        pool_size = len(pool)
        if num_tokens <= 0:
            return []
        start_idx = self._corpus_rng.randrange(pool_size)
        end_idx = start_idx + num_tokens
        tokens = pool[start_idx:end_idx]
        if end_idx > pool_size:
            tokens += pool[: end_idx - pool_size]
        return tokens

    # -- Cache support (same pattern as PromptGenerator) --

    def _generate_cached_prompt(
        self,
        num_tokens: int,
        hash_ids: list[int],
        block_size: int,
    ) -> str:
        cache_key = (tuple(hash_ids), num_tokens, block_size)
        if cache_key in self._decoded_cache:
            return self._decoded_cache[cache_key]

        final_prompt = self._build_token_sequence(num_tokens, hash_ids, block_size)
        decoded = self.tokenizer.decode(final_prompt, skip_special_tokens=False)
        self._decoded_cache[cache_key] = decoded
        return decoded

    def _build_token_sequence(
        self,
        num_tokens: int,
        hash_ids: list[int],
        block_size: int,
    ) -> list[int]:
        final_prompt: list[int] = []
        current_block_size = block_size

        final_block_size = num_tokens - ((len(hash_ids) - 1) * block_size)
        if final_block_size <= 0 or block_size < final_block_size:
            raise ConfigurationError(
                f"Input length: {num_tokens}, Hash IDs: {hash_ids}, Block size: {block_size} "
                f"are not compatible. The final hash block size: {final_block_size} must be "
                f"greater than 0 and less than or equal to {block_size}."
            )

        for index, hash_id in enumerate(hash_ids):
            if index == len(hash_ids) - 1:
                current_block_size = final_block_size

            if hash_id not in self._cache:
                prompt_tokens: list[int] = []
                if self.tokenizer.block_separation_token_id is not None:
                    prompt_tokens += [self.tokenizer.block_separation_token_id]
                    prompt_tokens += self._sample_tokens(
                        current_block_size - 1, self._tool_pool
                    )
                else:
                    prompt_tokens += self._sample_tokens(
                        current_block_size, self._tool_pool
                    )
                self._cache[hash_id] = prompt_tokens

            final_prompt.extend(self._cache[hash_id])

        return final_prompt

    # -- Template generators --

    def _gen_python_code(self) -> str:
        return self._template_rng.choice(
            [
                self._gen_python_class,
                self._gen_python_functions,
                self._gen_python_test,
                self._gen_python_http_handler,
                self._gen_python_data_model,
            ]
        )()

    def _gen_python_class(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        mod = r.choice(_MODULES)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        t1, t2 = r.sample(_TYPES, 2)
        dec = r.choice(_DECORATORS)
        imp_mod = r.choice(_MODULES)
        imp_cls = r.choice(_CLASSES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
import {mod}
from {mod}.{imp_mod} import {imp_cls}


class {cls}:
    \"\"\"Handles {m1} operations for {mod}.\"\"\"

    _default_{v3} = 64

    def __init__(self, {v1}: {t1}, {v2}: {t2} = None):
        self._{v1} = {v1}
        self._{v2} = {v2}
        self._{v3} = self._default_{v3}
        self._initialized = False

    {dec}
    async def {m1}(self, {v1}: {t1}) -> {t2}:
        if not self._initialized:
            raise RuntimeError("{cls} not initialized")
        {v2} = await self._{m2}({v1})
        return {v2}

    async def _{m2}(self, {v1}: {t1}) -> {t2}:
        try:
            {v2} = {mod}.{m2}({v1})
            return {v2}
        except Exception as e:
            raise ValueError("{err}") from e

    def {m3}(self) -> None:
        self._initialized = True
        self._{v3} = 0
"""

    def _gen_python_functions(self) -> str:
        r = self._template_rng
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        t1, t2, t3 = r.sample(_TYPES, 3)
        mod = r.choice(_MODULES)
        imp_mod = r.choice(_MODULES)
        cls = r.choice(_CLASSES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from {mod}.{imp_mod} import {cls}

logger = logging.getLogger(__name__)


async def {m1}({v1}: {t1}, {v2}: {t2} | None = None) -> {t3}:
    async with _acquire_{v3}({v1}) as {v3}:
        {v2} = await {cls}().{m2}({v3})
        return [{v2} for _ in range(10) if {v2} is not None]


@asynccontextmanager
async def _acquire_{v3}({v1}: {t1}) -> AsyncIterator[{t2}]:
    {v3} = {mod}.{m3}({v1})
    try:
        yield {v3}
    finally:
        await {v3}.close()


def {m2}_sync({v1}: {t1}, *, max_retries: int = 3) -> {t2}:
    for attempt in range(max_retries):
        try:
            return {mod}.{m2}({v1})
        except RuntimeError:
            if attempt == max_retries - 1:
                raise
            logger.warning("{err}, attempt %d", attempt + 1)
    raise AssertionError("unreachable")
"""

    def _gen_python_test(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        mod = r.choice(_MODULES)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2 = r.sample(_VARS, 2)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
import pytest
from unittest.mock import AsyncMock, patch

from {mod} import {cls}


class Test{cls}:
    @pytest.fixture
    def instance(self):
        return {cls}({v1}="test_value")

    @pytest.mark.asyncio
    async def test_{m1}_returns_expected(self, instance):
        instance._{m2} = AsyncMock(return_value=42)
        result = await instance.{m1}()
        assert result == 42
        instance._{m2}.assert_awaited_once()

    @pytest.mark.parametrize("{v1}", ["alpha", "beta", "gamma"])
    def test_{m2}_with_values(self, instance, {v1}):
        instance._{v1} = {v1}
        result = instance.{m2}()
        assert result is not None

    @pytest.mark.asyncio
    async def test_{m3}_raises_on_{v2}(self, instance):
        with pytest.raises(ValueError, match="{err}"):
            await instance.{m3}(None)

    @pytest.mark.asyncio
    async def test_{m1}_with_mock_dependency(self, instance):
        with patch("{mod}.{m2}") as mock:
            mock.return_value = {{{{"key": "{v2}"}}}}\n            result = await instance.{m1}()
            assert "{v2}" in str(result)
"""

    def _gen_python_http_handler(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        mod = r.choice(_MODULES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2, v3 = r.sample(_VARS, 3)
        route = r.choice(_HTTP_ROUTES)
        table = r.choice(_DB_TABLES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from {mod}.{cls.lower()} import {cls}

router = APIRouter(prefix="{route}", tags=["{mod}"])


class {cls}Request(BaseModel):
    {v1}: str = Field(description="Primary {v1} identifier")
    {v2}: int = Field(default=10, ge=1, le=100, description="Page size")
    {v3}: str | None = Field(default=None, description="Optional filter")


class {cls}Response(BaseModel):
    items: list[dict] = Field(description="Result items from {table}")
    total: int = Field(description="Total count")
    page: int = Field(description="Current page number")


@router.post("/", response_model={cls}Response, status_code=201)
async def {m1}(
    body: {cls}Request,
    svc: {cls} = Depends(),
) -> {cls}Response:
    try:
        items = await svc.{m1}(body.{v1}, page_size=body.{v2})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {cls}Response(items=items, total=len(items), page=1)


@router.get("/{{{{{v1}}}}}")
async def {m2}({v1}: str, svc: {cls} = Depends()) -> dict:
    result = await svc.{m2}({v1})
    if result is None:
        raise HTTPException(status_code=404, detail="{err}")
    return {{"status": "ok", "data": result}}
"""

    def _gen_python_data_model(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        v1, v2, v3, v4 = r.sample(_VARS, 4)
        m1 = r.choice(_METHODS)
        table = r.choice(_DB_TABLES)

        return f"""\
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class {cls}Status(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class {cls}Config(BaseModel):
    {v1}: str = Field(description="{cls} {v1} identifier")
    {v2}: int = Field(default=0, ge=0, description="Current {v2} count")
    {v3}: float = Field(default=1.0, gt=0, description="Rate limit for {m1}")
    status: {cls}Status = Field(default={cls}Status.PENDING, description="Lifecycle status")
    {v4}: dict[str, str] = Field(default_factory=dict, description="Arbitrary {v4}")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    source_table: str = Field(default="{table}", description="Backing store table")

    @field_validator("{v1}")
    @classmethod
    def _validate_{v1}(cls, v: str) -> str:
        if not v or len(v) > 256:
            raise ValueError("{v1} must be 1-256 characters")
        return v.strip()

    @field_validator("{v3}")
    @classmethod
    def _validate_{v3}(cls, v: float) -> float:
        if v > 10_000:
            raise ValueError("{v3} exceeds max rate")
        return v

    def {m1}(self) -> bool:
        return self.status == {cls}Status.ACTIVE and self.{v2} > 0
"""

    def _gen_go_code(self) -> str:
        return self._template_rng.choice(
            [
                self._gen_go_struct,
                self._gen_go_http_handler,
                self._gen_go_errors,
                self._gen_go_test,
            ]
        )()

    def _gen_go_struct(self) -> str:
        r = self._template_rng
        pkg1, pkg2 = r.sample(list(_GO_PACKAGES), 2)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2, v3 = r.sample(_VARS, 3)
        pkg_name = r.choice(_MODULES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
package {pkg_name}

import (
    "{pkg1}"
    "{pkg2}"
)

type {cls} struct {{{{
    {v1} string `json:"{v1}"`
    {v2} int    `json:"{v2},omitempty"`
    {v3} bool   `json:"-"`
    mu  sync.RWMutex
}}}}

func New{cls}({v1} string) *{cls} {{{{
    return &{cls}{{{{{v1}: {v1}}}}}
}}}}

func (s *{cls}) {m1.title()}(ctx context.Context) error {{{{
    s.mu.Lock()
    defer s.mu.Unlock()
    if s.{v1} == "" {{{{
        return {pkg1}.Errorf("{err}")
    }}}}
    s.{v2}++
    return nil
}}}}

func (s *{cls}) {m2.title()}() (string, error) {{{{
    s.mu.RLock()
    defer s.mu.RUnlock()
    if !s.{v3} {{{{
        return "", {pkg1}.Errorf("%w: not initialized", Err{cls})
    }}}}
    return {pkg2}.Sprintf("%s:%d", s.{v1}, s.{v2}), nil
}}}}
"""

    def _gen_go_http_handler(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        pkg_name = r.choice(_MODULES)
        table = r.choice(_DB_TABLES)
        err = r.choice(_ERROR_MESSAGES)
        status_code = r.choice(
            ["http.StatusOK", "http.StatusCreated", "http.StatusAccepted"]
        )

        return f"""\
package {pkg_name}

import (
    "encoding/json"
    "net/http"
    "log/slog"
)

type {m1.title()}Request struct {{{{
    {v1.title()} string `json:"{v1}" binding:"required"`
    {v2.title()} int    `json:"{v2}" binding:"gte=0"`
}}}}

type {m1.title()}Response struct {{{{
    Items []map[string]any `json:"items"`
    Total int              `json:"total"`
}}}}

func (h *{cls}) {m1.title()}Handler(w http.ResponseWriter, r *http.Request) {{{{
    var req {m1.title()}Request
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {{{{
        slog.Error("{err}", "handler", "{m1}")
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }}}}

    items, err := h.svc.{m2.title()}(r.Context(), req.{v1.title()})
    if err != nil {{{{
        slog.Error("{err}", "table", "{table}")
        http.Error(w, "{err}", http.StatusInternalServerError)
        return
    }}}}

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader({status_code})
    json.NewEncoder(w).Encode({m1.title()}Response{{{{Items: items, Total: len(items)}}}})
}}}}
"""

    def _gen_go_errors(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        pkg_name = r.choice(_MODULES)
        e1, e2, e3 = r.sample(_ERROR_MESSAGES, 3)
        m1 = r.choice(_METHODS)
        v1 = r.choice(_VARS)

        return f"""\
package {pkg_name}

import (
    "errors"
    "fmt"
)

var (
    Err{cls}         = errors.New("{e1}")
    ErrNot{m1.title()} = errors.New("{e2}")
    ErrInvalid{v1.title()} = errors.New("{e3}")
)

type {cls}Error struct {{{{
    Op      string
    {v1.title()} string
    Err     error
}}}}

func (e *{cls}Error) Error() string {{{{
    return fmt.Sprintf("%s %s: %v", e.Op, e.{v1.title()}, e.Err)
}}}}

func (e *{cls}Error) Unwrap() error {{{{
    return e.Err
}}}}

func Wrap{cls}Error(op, {v1} string, err error) error {{{{
    return &{cls}Error{{{{Op: op, {v1.title()}: {v1}, Err: err}}}}
}}}}
"""

    def _gen_go_test(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        pkg_name = r.choice(_MODULES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)

        return f"""\
package {pkg_name}_test

import (
    "context"
    "testing"
)

func Test{cls}_{m1.title()}(t *testing.T) {{{{
    tests := []struct {{{{
        name    string
        {v1}    string
        want    int
        wantErr bool
    }}}}{{{{
        {{{{"valid {v1}", "test_value", 42, false}}}},
        {{{{"empty {v1}", "", 0, true}}}},
        {{{{"long {v1}", "a]very_long_value_that_exceeds_limit", 0, true}}}},
    }}}}

    for _, tt := range tests {{{{
        t.Run(tt.name, func(t *testing.T) {{{{
            s := New{cls}(tt.{v1})
            got, err := s.{m1.title()}(context.Background())
            if (err != nil) != tt.wantErr {{{{
                t.Errorf("{m1.title()}() error = %v, wantErr %v", err, tt.wantErr)
                return
            }}}}
            if got != tt.want {{{{
                t.Errorf("{m1.title()}() = %v, want %v", got, tt.want)
            }}}}
        }}}})
    }}}}
}}}}

func Test{cls}_{m2.title()}_Concurrent(t *testing.T) {{{{
    s := New{cls}("{v2}")
    ctx := context.Background()
    errs := make(chan error, 10)
    for i := 0; i < 10; i++ {{{{
        go func() {{{{ errs <- s.{m2.title()}(ctx) }}}}()
    }}}}
    for i := 0; i < 10; i++ {{{{
        if err := <-errs; err != nil {{{{
            t.Errorf("concurrent {m2}: %v", err)
        }}}}
    }}}}
}}}}
"""

    def _gen_rust_code(self) -> str:
        return self._template_rng.choice(
            [
                self._gen_rust_struct,
                self._gen_rust_http_handler,
                self._gen_rust_errors,
                self._gen_rust_test,
            ]
        )()

    def _gen_rust_struct(self) -> str:
        r = self._template_rng
        cr1, cr2 = r.sample(list(_RUST_CRATES), 2)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2, v3 = r.sample(_VARS, 3)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
use {cr1};
use {cr2};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct {cls} {{{{
    {v1}: String,
    {v2}: Vec<u8>,
    #[serde(default)]
    {v3}: Option<u64>,
    initialized: bool,
}}}}

impl {cls} {{{{
    pub fn new({v1}: impl Into<String>) -> Self {{{{
        Self {{{{
            {v1}: {v1}.into(),
            {v2}: Vec::new(),
            {v3}: None,
            initialized: false,
        }}}}
    }}}}

    pub async fn {m1}(&mut self) -> Result<(), anyhow::Error> {{{{
        if !self.initialized {{{{
            anyhow::bail!("{err}");
        }}}}
        self.{m2}().await
    }}}}

    async fn {m2}(&self) -> Result<(), anyhow::Error> {{{{
        let _{v2} = self.{v1}.as_bytes();
        tracing::debug!("{m2} completed for {{}}", self.{v1});
        Ok(())
    }}}}
}}}}
"""

    def _gen_rust_http_handler(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        mod = r.choice(_MODULES)

        return f"""\
use axum::{{extract::{{Path, State}}, http::StatusCode, Json}};
use serde::{{Deserialize, Serialize}};
use std::sync::Arc;

use crate::{mod}::{cls};

#[derive(Debug, Deserialize)]
pub struct {m1.title()}Request {{{{
    {v1}: String,
    {v2}: Option<i64>,
}}}}

#[derive(Debug, Serialize)]
pub struct {m1.title()}Response {{{{
    id: String,
    {v1}: String,
    created: bool,
}}}}

pub async fn {m1}_handler(
    State(svc): State<Arc<{cls}>>,
    Json(body): Json<{m1.title()}Request>,
) -> Result<Json<{m1.title()}Response>, StatusCode> {{{{
    let result = svc
        .{m1}(&body.{v1})
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json({m1.title()}Response {{{{
        id: result.id.to_string(),
        {v1}: body.{v1},
        created: true,
    }}}}))
}}}}

pub async fn {m2}_handler(
    State(svc): State<Arc<{cls}>>,
    Path({v1}): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {{{{
    svc.{m2}(&{v1})
        .await
        .map(|v| Json(serde_json::json!({{{{"status": "ok", "data": v}}}})))
        .map_err(|_| StatusCode::NOT_FOUND)
}}}}
"""

    def _gen_rust_errors(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        e1, e2, e3 = r.sample(_ERROR_MESSAGES, 3)
        v1 = r.choice(_VARS)
        mod = r.choice(_MODULES)

        return f"""\
use thiserror::Error;

#[derive(Debug, Error)]
pub enum {cls}Error {{{{
    #[error("{e1}")]
    NotInitialized,

    #[error("{e2}: {{{{{v1}}}}}")]
    InvalidInput {{{{ {v1}: String }}}},

    #[error("{e3}")]
    Internal(#[from] anyhow::Error),

    #[error("io error in {mod}")]
    Io(#[from] std::io::Error),

    #[error("serialization failed")]
    Serde(#[from] serde_json::Error),
}}}}

impl {cls}Error {{{{
    pub fn is_retryable(&self) -> bool {{{{
        matches!(self, Self::Internal(_) | Self::Io(_))
    }}}}

    pub fn status_code(&self) -> u16 {{{{
        match self {{{{
            Self::NotInitialized => 503,
            Self::InvalidInput {{{{ .. }}}} => 400,
            Self::Internal(_) => 500,
            Self::Io(_) => 502,
            Self::Serde(_) => 422,
        }}}}
    }}}}
}}}}
"""

    def _gen_rust_test(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        err = r.choice(_ERROR_MESSAGES)
        cr = r.choice(_RUST_CRATES)

        return f"""\
use {cr};

#[cfg(test)]
mod tests {{{{
    use super::*;

    fn make_{cls.lower()}() -> {cls} {{{{
        {cls}::new("{v1}_test")
    }}}}

    #[tokio::test]
    async fn test_{m1}_success() {{{{
        let mut svc = make_{cls.lower()}();
        svc.initialized = true;
        let result = svc.{m1}().await;
        assert!(result.is_ok(), "expected Ok, got {{:?}}", result);
    }}}}

    #[tokio::test]
    async fn test_{m1}_not_initialized() {{{{
        let mut svc = make_{cls.lower()}();
        let err = svc.{m1}().await.unwrap_err();
        assert!(err.to_string().contains("{err}"));
    }}}}

    #[test]
    fn test_{m2}_returns_bytes() {{{{
        let svc = make_{cls.lower()}();
        let {v2} = svc.{v1}.as_bytes();
        assert!(!{v2}.is_empty());
    }}}}

    #[tokio::test]
    async fn test_{m1}_concurrent() {{{{
        let svc = std::sync::Arc::new(tokio::sync::Mutex::new(make_{cls.lower()}()));
        let mut handles = vec![];
        for _ in 0..5 {{{{
            let svc = svc.clone();
            handles.push(tokio::spawn(async move {{{{
                svc.lock().await.{m1}().await
            }}}}));
        }}}}
        for h in handles {{{{
            let _ = h.await.unwrap();
        }}}}
    }}}}
}}}}
"""

    def _gen_typescript_code(self) -> str:
        return self._template_rng.choice(
            [
                self._gen_typescript_class,
                self._gen_typescript_http_handler,
                self._gen_typescript_types,
                self._gen_typescript_test,
            ]
        )()

    def _gen_typescript_class(self) -> str:
        r = self._template_rng
        imp = r.choice(_TS_IMPORTS)
        imp_cls = r.choice(_CLASSES)
        cls = r.choice(_CLASSES)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
import {{{{ {imp_cls} }}}} from '{imp}';

interface {cls}Config {{{{
  {v1}: string;
  {v2}?: number;
  timeout: number;
}}}}

export class {cls} {{{{
  #{v1}: string;
  #{v2}: number;
  readonly {v3}: string;

  constructor(config: {cls}Config) {{{{
    this.#{v1} = config.{v1};
    this.#{v2} = config.{v2} ?? 0;
    this.{v3} = crypto.randomUUID();
  }}}}

  async {m1}({v1}: string): Promise<void> {{{{
    try {{{{
      const {v2} = await this.{m2}({v1});
      console.log(`${{{{this.#{v1}}}}}: ${{{{{v2}}}}}`);
    }}}} catch (err) {{{{
      throw new Error(`{err}`);
    }}}}
  }}}}

  async {m3}(): Promise<boolean> {{{{
    return this.#{v2} > 0;
  }}}}

  private async {m2}({v1}: string): Promise<number> {{{{
    return this.#{v2};
  }}}}
}}}}
"""

    def _gen_typescript_http_handler(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        route = r.choice(_HTTP_ROUTES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
import {{ Hono }} from 'hono';
import {{ z }} from 'zod';
import {{ {cls} }} from './{cls.lower()}';

const {m1}Schema = z.object({{{{
  {v1}: z.string().min(1).max(256),
  {v2}: z.number().int().positive().optional(),
}}}});

type {m1.title()}Input = z.infer<typeof {m1}Schema>;

const app = new Hono();

app.post('{route}', async (c) => {{{{
  const body = {m1}Schema.safeParse(await c.req.json());
  if (!body.success) {{{{
    return c.json({{{{ error: body.error.flatten() }}}}, 400);
  }}}}

  const svc = new {cls}();
  try {{{{
    const result = await svc.{m1}(body.data.{v1});
    return c.json({{{{ status: 'ok', data: result }}}}, 201);
  }}}} catch (err) {{{{
    return c.json({{{{ error: '{err}' }}}}, 500);
  }}}}
}}}});

app.get('{route}/:id', async (c) => {{{{
  const id = c.req.param('id');
  const svc = new {cls}();
  const item = await svc.{m2}(id);
  if (!item) return c.json({{{{ error: 'not found' }}}}, 404);
  return c.json({{{{ status: 'ok', data: item }}}});
}}}});

export default app;
"""

    def _gen_typescript_types(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        v1, v2, v3 = r.sample(_VARS, 3)
        m1, m2 = r.sample(_METHODS, 2)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
export type {cls}Status = 'pending' | 'active' | 'failed' | 'completed';

export interface {cls}Event {{{{
  kind: '{m1}' | '{m2}' | 'error';
  {v1}: string;
  timestamp: number;
}}}}

export type {m1.title()}Event = Extract<{cls}Event, {{{{ kind: '{m1}' }}}}>;
export type ErrorEvent = Extract<{cls}Event, {{{{ kind: 'error' }}}}>;

export interface {cls}Config {{{{
  readonly {v1}: string;
  readonly {v2}: number;
  readonly {v3}?: Record<string, unknown>;
}}}}

export type Partial{cls} = Partial<{cls}Config> & Pick<{cls}Config, '{v1}'>;

export function is{cls}Event(e: unknown): e is {cls}Event {{{{
  return (
    typeof e === 'object' &&
    e !== null &&
    'kind' in e &&
    typeof (e as {cls}Event).{v1} === 'string'
  );
}}}}

export function assert{cls}Status(s: string): asserts s is {cls}Status {{{{
  const valid: {cls}Status[] = ['pending', 'active', 'failed', 'completed'];
  if (!valid.includes(s as {cls}Status)) {{{{
    throw new Error(`{err}: ${{{{s}}}}`);
  }}}}
}}}}
"""

    def _gen_typescript_test(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2 = r.sample(_VARS, 2)
        err = r.choice(_ERROR_MESSAGES)
        mod = r.choice(_MODULES)

        return f"""\
import {{ describe, it, expect, beforeEach, vi }} from 'vitest';
import {{ {cls} }} from '../{mod}';

describe('{cls}', () => {{{{
  let instance: {cls};

  beforeEach(() => {{{{
    instance = new {cls}({{{{ {v1}: 'test', timeout: 5000 }}}});
    vi.clearAllMocks();
  }}}});

  describe('{m1}', () => {{{{
    it('should return expected value', async () => {{{{
      const result = await instance.{m1}('{v2}');
      expect(result).toBeDefined();
      expect(typeof result).toBe('object');
    }}}});

    it('should throw on invalid input', async () => {{{{
      await expect(instance.{m1}('')).rejects.toThrow('{err}');
    }}}});
  }}}});

  describe('{m2}', () => {{{{
    it('should call dependency', async () => {{{{
      const spy = vi.spyOn(instance as any, '{m3}');
      await instance.{m2}('{v1}');
      expect(spy).toHaveBeenCalledOnce();
    }}}});
  }}}});

  it('should handle concurrent calls', async () => {{{{
    const promises = Array.from({{{{ length: 5 }}}}, () => instance.{m1}('{v1}'));
    const results = await Promise.all(promises);
    expect(results).toHaveLength(5);
  }}}});
}}}});
"""

    def _file_pool(self, language: str | None) -> tuple[str, ...]:
        if language:
            return _LANG_FILE_PATHS.get(language, _FILE_PATHS)
        return _FILE_PATHS

    def _gen_tool_use_block(self, language: str | None = None) -> str:
        r = self._template_rng
        return r.choice(
            [
                lambda: self._gen_tool_read(language=language),
                lambda: self._gen_tool_edit(language=language),
                lambda: self._gen_tool_search(language=language),
                lambda: self._gen_tool_bash(language=language),
            ]
        )()

    def _gen_tool_read(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        f = r.choice(file_pool)
        start_line = r.randint(1, 200)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        mod = r.choice(_MODULES)
        err = r.choice(_ERROR_MESSAGES)

        lang_lines: dict[str | None, list[str]] = {
            "python": [
                f"def {m1}(self, {v1}):",
                f"self._{v1} = {v1}",
                f"{v2} = {mod}.{m2}({v1})",
                f"if {v1} is None:",
                f'    raise ValueError("{err}")',
                f"return {v2}",
                f'logger.debug(f"{cls}.{m1}: {{{{{v1}}}}}")',
                "",
            ],
            "go": [
                f"func (s *{cls}) {m1.title()}(ctx context.Context) error {{",
                f"s.{v1} = {v1}",
                f"{v2}, err := s.{m2.title()}(ctx)",
                "if err != nil {",
                f'return fmt.Errorf("{err}: %w", err)',
                "}",
                "return nil",
                "",
            ],
            "rust": [
                f"pub async fn {m1}(&mut self) -> Result<()> {{",
                f"let {v1} = self.{v2}.clone();",
                f"let {v2} = self.{m2}(&{v1}).await?;",
                f"if {v2}.is_empty() {{",
                f'anyhow::bail!("{err}");',
                "}",
                "Ok(())",
                "",
            ],
            "typescript": [
                f"async {m1}({v1}: string): Promise<void> {{",
                f"this.{v1} = {v1};",
                f"const {v2} = await this.{m2}({v1});",
                f"if (!{v2}) {{",
                f"  throw new Error('{err}');",
                "}",
                f"console.log(`{cls}.{m1}: ${{{{{v2}}}}}`);",
                "",
            ],
        }
        code_lines = lang_lines.get(language, lang_lines["python"])

        lines = []
        for i in range(start_line, start_line + r.randint(15, 30)):
            indent = "    " if r.random() > 0.3 else "        "
            line_content = r.choice(code_lines)
            lines.append(f"{i:>6}\t{indent}{line_content}")

        content = "\n".join(lines)
        return f"""\
<tool_name>read</tool_name>
<parameter name="file_path">{f}</parameter>
<result>
{content}
</result>
"""

    def _gen_tool_edit(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        f = r.choice(file_pool)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        cls = r.choice(_CLASSES)
        err = r.choice(_ERROR_MESSAGES)

        edits: dict[str | None, tuple[str, str]] = {
            "python": (
                f"    def {m1}(self, {v1}):\n        return self._{m2}({v1})",
                f"    async def {m1}(self, {v1}: str) -> dict:\n"
                f"        try:\n"
                f"            {v2} = await self._{m2}({v1})\n"
                f"            if {v2} is None:\n"
                f'                raise ValueError("{err}")\n'
                f'            return {{{{"status": "ok", "data": {v2}}}}}\n'
                f"        except Exception as exc:\n"
                f'            logger.error("{cls}.{m1} failed: %s", exc)\n'
                f"            raise",
            ),
            "go": (
                f"func (s *{cls}) {m1.title()}() error {{{{\n    return nil\n}}}}",
                f"func (s *{cls}) {m1.title()}(ctx context.Context) error {{{{\n"
                f"    {v2}, err := s.{m2.title()}(ctx)\n"
                f"    if err != nil {{{{\n"
                f'        return fmt.Errorf("{err}: %w", err)\n'
                f"    }}}}\n"
                f"    s.{v1} = {v2}\n"
                f"    return nil\n"
                f"}}}}",
            ),
            "rust": (
                f"fn {m1}(&self) -> Result<()> {{{{\n    Ok(())\n}}}}",
                f"async fn {m1}(&mut self) -> Result<()> {{{{\n"
                f"    let {v2} = self.{m2}().await?;\n"
                f'    anyhow::ensure!(!{v2}.is_empty(), "{err}");\n'
                f"    self.{v1} = {v2};\n"
                f"    Ok(())\n"
                f"}}}}",
            ),
            "typescript": (
                f"{m1}({v1}: string) {{{{\n    return this.{m2}({v1});\n}}}}",
                f"async {m1}({v1}: string): Promise<Record<string, unknown>> {{{{\n"
                f"    const {v2} = await this.{m2}({v1});\n"
                f"    if (!{v2}) throw new Error('{err}');\n"
                f"    return {{ status: 'ok', data: {v2} }};\n"
                f"}}}}",
            ),
        }
        old_str, new_str = edits.get(language, edits["python"])

        return f"""\
<tool_name>edit</tool_name>
<parameter name="file_path">{f}</parameter>
<parameter name="old_string">{old_str}</parameter>
<parameter name="new_string">{new_str}</parameter>
<result>
The file {f} has been updated successfully.
</result>
"""

    def _gen_tool_search(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)

        lang_patterns: dict[str | None, list[str]] = {
            "python": [
                f"class {r.choice(_CLASSES)}",
                f"def {r.choice(_METHODS)}",
                f"import {r.choice(_MODULES)}",
                f"async def {r.choice(_METHODS)}",
            ],
            "go": [
                f"func {r.choice(_METHODS).title()}",
                f"type {r.choice(_CLASSES)} struct",
                f'"{r.choice(list(_GO_PACKAGES))}"',
                f"func New{r.choice(_CLASSES)}",
            ],
            "rust": [
                f"fn {r.choice(_METHODS)}",
                f"pub struct {r.choice(_CLASSES)}",
                f"use {r.choice(list(_RUST_CRATES))}",
                f"impl {r.choice(_CLASSES)}",
            ],
            "typescript": [
                f"class {r.choice(_CLASSES)}",
                f"export function {r.choice(_METHODS)}",
                f"import {{ {r.choice(_CLASSES)} }}",
                f"interface {r.choice(_CLASSES)}",
            ],
        }
        patterns = lang_patterns.get(language, lang_patterns["python"])
        pattern = r.choice([*patterns, r.choice(_ERROR_MESSAGES)])

        files = r.sample(list(file_pool), min(r.randint(3, 6), len(file_pool)))
        matches = []
        for f in files:
            line_num = r.randint(1, 400)
            ctx = r.choice(_VARS)
            matches.append(f"{f}:{line_num}:    {pattern}({ctx})")

        content = "\n".join(matches)
        return f"""\
<tool_name>search</tool_name>
<parameter name="pattern">{pattern}</parameter>
<result>
{content}
</result>
"""

    def _gen_tool_bash(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        cls = r.choice(_CLASSES)
        methods = r.sample(list(_METHODS), 4)
        n_pass = r.randint(10, 80)
        n_fail = r.randint(0, 3)
        dur = r.uniform(0.5, 30.0)

        lang_cmds: dict[str | None, str] = {
            "python": "pytest -xvs tests/",
            "go": "go test -v ./...",
            "rust": "cargo test",
            "typescript": "npx vitest run",
        }
        cmd = lang_cmds.get(language, r.choice(_CLI_COMMANDS))

        test_lines = []
        for m in methods:
            passed = r.random() > 0.2
            if language == "go":
                status = "ok" if passed else "FAIL"
                test_lines.append(
                    f"--- {status}: Test{m.title()} ({r.uniform(0.001, 2.0):.3f}s)"
                )
            elif language == "rust":
                status = "ok" if passed else "FAILED"
                test_lines.append(f"test {mod}::{cls.lower()}::test_{m} ... {status}")
            elif language == "typescript":
                mark = "\u2713" if passed else "\u2717"
                test_lines.append(f"  {mark} {cls} > {m} ({r.randint(1, 500)} ms)")
            else:
                status = "PASSED" if passed else "FAILED"
                test_lines.append(f"tests/test_{mod}.py::Test{cls}::test_{m} {status}")
        test_output = "\n".join(test_lines)

        return f"""\
<tool_name>bash</tool_name>
<parameter name="command">{cmd}</parameter>
<result>
{test_output}

{n_pass} passed, {n_fail} failed in {dur:.2f}s
</result>
"""

    def _gen_bash_output(self, language: str | None = None) -> str:
        r = self._template_rng
        return r.choice(
            [
                lambda: self._gen_bash_file_explore(language=language),
                lambda: self._gen_bash_build_test(language=language),
                lambda: self._gen_bash_git_workflow(language=language),
            ]
        )()

    def _gen_bash_file_explore(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        ext_cmds: dict[str | None, tuple[str, str]] = {
            "python": ("find . -name '*.py'", "src/**/*.py"),
            "go": ("find . -name '*.go'", "**/*.go"),
            "rust": ("find . -name '*.rs'", "src/**/*.rs"),
            "typescript": ("find . -name '*.ts'", "src/**/*.ts"),
        }
        find_cmd, glob_pat = ext_cmds.get(language, ext_cmds["python"])
        cmd = r.choice(("ls -la", find_cmd, "tree src/", "wc -l"))
        files = r.sample(list(file_pool), min(r.randint(4, 8), len(file_pool)))
        file_listing = "\n".join(
            f"  {f:<42} {r.randint(1, 500):>4} lines  {r.randint(1, 50):>3}K"
            for f in files
        )
        total_lines = r.randint(500, 15000)

        return f"""\
$ {cmd}
{file_listing}
$ wc -l {glob_pat} | tail -1
  {total_lines} total
$ du -sh .
  {r.randint(1, 500)}M\t.
"""

    def _gen_bash_build_test(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        n_pkgs = r.randint(10, 200)
        build_time = r.uniform(0.5, 30.0)
        n_pass = r.randint(20, 150)
        n_fail = r.randint(0, 5)
        test_time = r.uniform(1.0, 60.0)

        lang_build: dict[str | None, tuple[str, str]] = {
            "python": (
                "pip install -e '.[dev]'",
                f"pytest tests/ -x\n  {n_pass} passed, {n_fail} failed in {test_time:.1f}s",
            ),
            "go": (
                f"go build ./cmd/{mod}\n  Compiled {n_pkgs} packages in {build_time:.1f}s",
                f"go test -v -race ./...\n  {n_pass} passed, {n_fail} failed in {test_time:.1f}s",
            ),
            "rust": (
                f"cargo build --release\n  Compiling {n_pkgs} crates\n  Finished in {build_time:.1f}s",
                f"cargo test\n  {n_pass} passed, {n_fail} failed in {test_time:.1f}s",
            ),
            "typescript": (
                f"npm ci && npm run build\n  Resolved {n_pkgs} packages in {build_time:.1f}s",
                f"npx vitest run\n  {n_pass} passed, {n_fail} failed in {test_time:.1f}s",
            ),
        }
        build_cmd, test_cmd = lang_build.get(language, lang_build["python"])

        return f"""\
$ {build_cmd}
$ {test_cmd}
$ echo $?
{"0" if n_fail == 0 else "1"}
"""

    def _gen_bash_git_workflow(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        branch = f"{r.choice(_MODULES)}/{r.choice(_METHODS)}-{r.choice(_VARS)}"
        mod = r.choice(_MODULES)
        files = r.sample(list(file_pool), min(3, len(file_pool)))
        changed = "\n".join(f"  M {f}" for f in files)
        hash1 = f"{r.randint(1000000, 9999999):07x}"
        hash2 = f"{r.randint(1000000, 9999999):07x}"

        return f"""\
$ git checkout -b {branch}
Switched to a new branch '{branch}'
$ git status
On branch {branch}
Changes not staged for commit:
{changed}
$ git add -A && git commit -m "feat: {r.choice(_METHODS)} {r.choice(_VARS)} in {mod}"
[{branch} {hash1}] feat: {r.choice(_METHODS)} {r.choice(_VARS)} in {mod}
 {len(files)} files changed, {r.randint(10, 200)} insertions(+), {r.randint(1, 50)} deletions(-)
$ git log --oneline -3
{hash1} feat: {r.choice(_METHODS)} {r.choice(_VARS)} in {mod}
{hash2} fix: {r.choice(_ERROR_MESSAGES)}
"""

    def _gen_json_response(self, language: str | None = None) -> str:
        return self._template_rng.choice(
            [
                self._gen_json_object,
                self._gen_json_paginated,
                self._gen_json_error,
            ]
        )()

    def _gen_json_object(self) -> str:
        r = self._template_rng
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2, v3 = r.sample(_VARS, 3)
        cls = r.choice(_CLASSES)
        id_suffix = r.randint(1000, 9999)
        num_val = r.randint(0, 1000)
        float_val = r.uniform(0, 1)
        ts = r.randint(1700000000, 1800000000)
        items = [
            f'      {{{{"id": {r.randint(1, 999)}, "name": "{r.choice(_VARS)}"}}}}'
            for _ in range(3)
        ]
        items_str = ",\n".join(items)

        return f"""\
{{{{
  "status": "ok",
  "data": {{{{
    "{v1}": "{cls.lower()}_{id_suffix}",
    "{v2}": {num_val},
    "{v3}": {float_val:.4f},
    "metadata": {{{{
      "action": "{m1}",
      "source": "{m2}",
      "timestamp": "{ts}"
    }}}},
    "items": [
{items_str}
    ]
  }}}}
}}}}
"""

    def _gen_json_paginated(self) -> str:
        r = self._template_rng
        v1, v2 = r.sample(_VARS, 2)
        cls = r.choice(_CLASSES)
        total = r.randint(50, 5000)
        page = r.randint(1, 20)
        per_page = r.choice([10, 20, 50, 100])
        items = [
            f'    {{{{"id": "{cls.lower()}_{r.randint(1000, 9999)}", "{v1}": "{r.choice(_MODULES)}", "{v2}": {r.randint(0, 100)}}}}}'
            for _ in range(min(per_page, 5))
        ]
        items_str = ",\n".join(items)

        return f"""\
{{{{
  "data": [
{items_str}
  ],
  "pagination": {{{{
    "page": {page},
    "per_page": {per_page},
    "total": {total},
    "total_pages": {(total + per_page - 1) // per_page},
    "has_next": {str(page * per_page < total).lower()},
    "has_prev": {str(page > 1).lower()}
  }}}}
}}}}
"""

    def _gen_json_error(self) -> str:
        r = self._template_rng
        err = r.choice(_ERROR_MESSAGES)
        status = r.choice(_STATUS_CODES)
        code = status.split()[0]
        trace_id = f"{r.randint(100000, 999999):06x}-{r.randint(100000, 999999):06x}"
        v1 = r.choice(_VARS)
        cls = r.choice(_CLASSES)

        return f"""\
{{{{
  "error": {{{{
    "code": {code},
    "status": "{status}",
    "message": "{err}",
    "details": [
      {{{{
        "field": "{v1}",
        "reason": "{err}",
        "type": "{cls}"
      }}}}
    ],
    "trace_id": "{trace_id}",
    "documentation_url": "https://docs.example.com/errors/{code}"
  }}}}
}}}}
"""

    def _gen_error_traceback(self, language: str | None = None) -> str:
        r = self._template_rng
        err = r.choice(_ERROR_MESSAGES)
        cls = r.choice(_CLASSES)
        m1, m2, m3, m4 = r.sample(_METHODS, 4)

        lang_to_kind = {
            "python": "python",
            "go": "go",
            "rust": "rust",
            "typescript": "node",
        }
        kind = (
            lang_to_kind[language]
            if language in lang_to_kind
            else r.choice(["python", "go", "rust", "node"])
        )
        file_pool = (
            _LANG_FILE_PATHS.get(language, _FILE_PATHS) if language else _FILE_PATHS
        )
        f1, f2, f3, f4 = r.sample(list(file_pool), 4)
        if kind == "python":
            v = r.choice(_VARS)
            mod = r.choice(_MODULES)
            err2 = r.choice(_ERROR_MESSAGES)
            cls2 = r.choice(_CLASSES)
            return f"""\
Traceback (most recent call last):
  File "{f1}", line {r.randint(10, 500)}, in {m1}
    result = self.{m2}(data)
  File "{f2}", line {r.randint(10, 300)}, in {m2}
    {v} = await self._{m3}()
  File "{f3}", line {r.randint(10, 200)}, in _{m3}
    return {mod}.{m4}({v})
  File "{f4}", line {r.randint(1, 200)}, in {m4}
    raise ValueError("{err}")
ValueError: {err}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "{f1}", line {r.randint(10, 500)}, in {m1}
    self._{v} = {mod}.{m1}()
  File "{f2}", line {r.randint(10, 300)}, in __init__
    raise RuntimeError("{err2}")
RuntimeError: {cls}.{m1}() failed: {err2}

The above exception was the direct cause of the following exception:

{cls2}Error: {cls}.{m1}() aborted after {err}: {err2}
"""
        elif kind == "go":
            g1 = r.randint(1, 100)
            g2 = r.randint(101, 200)
            cls2 = r.choice(_CLASSES)
            return f"""\
goroutine {g1} [running]:
runtime/debug.Stack()
    /usr/local/go/src/runtime/debug/stack.go:{r.randint(10, 50)}
main.{cls}.{m1.title()}(...)
    {f1}:{r.randint(10, 300)}
main.{cls}.{m2.title()}(0xc000{r.randint(10000, 99999):05x})
    {f2}:{r.randint(10, 300)}
main.{cls}.{m3.title()}(0xc000{r.randint(10000, 99999):05x}, 0x{r.randint(100, 999):x})
    {f3}:{r.randint(10, 300)}
panic: {err}

goroutine {g2} [select]:
main.{cls2}.{m4.title()}(0xc000{r.randint(10000, 99999):05x})
    {f4}:{r.randint(10, 300)} +0x{r.randint(100, 999):x}
created by main.New{cls2}
    {f4}:{r.randint(10, 100)}
"""
        elif kind == "rust":
            mod1, mod2, mod3 = r.sample(list(_MODULES), 3)
            return f"""\
thread 'main' panicked at '{err}', {f1}:{r.randint(10, 300)}
stack backtrace:
   0: std::panicking::begin_panic
   1: {mod1}::{cls}::{m1}
             at {f1}:{r.randint(10, 300)}
   2: {mod2}::{cls}::{m2}
             at {f2}:{r.randint(10, 300)}
   3: {mod3}::{cls}::{m3}
             at {f3}:{r.randint(10, 300)}
   4: {mod1}::main
             at {f4}:{r.randint(10, 300)}
   5: std::rt::lang_start::{{{{closure}}}}
             at /rustc/src/rt.rs:{r.randint(50, 200)}
   6: std::rt::lang_start
             at /rustc/src/rt.rs:{r.randint(50, 200)}
note: run with `RUST_BACKTRACE=1` for a full backtrace
"""
        else:
            async_cls = r.choice(_CLASSES)
            async_method = r.choice(_METHODS)
            cls2 = r.choice(_CLASSES)
            return f"""\
Error: {err}
    at {cls}.{m1} ({f1}:{r.randint(10, 300)}:{r.randint(1, 40)})
    at {cls}.{m2} ({f2}:{r.randint(10, 300)}:{r.randint(1, 40)})
    at {cls2}.{m3} ({f3}:{r.randint(10, 300)}:{r.randint(1, 40)})
    at processTicksAndRejections (node:internal/process/task_queues:{r.randint(50, 100)})
    at async {async_cls}.{async_method} ({f4}:{r.randint(10, 300)})
Caused by: {r.choice(_ERROR_MESSAGES)}
    at {cls2}.{m4} ({f3}:{r.randint(10, 300)}:{r.randint(1, 40)})
"""

    def _gen_git_diff(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        f1, f2, f3 = r.sample(list(file_pool), 3)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        cls = r.choice(_CLASSES)
        ln = r.randint(10, 200)
        ln2 = r.randint(50, 300)
        err = r.choice(_ERROR_MESSAGES)
        mod = r.choice(_MODULES)
        idx = lambda: f"{r.randint(1000000, 9999999):07x}"  # noqa: E731
        hunk_old, hunk_new = r.randint(1, 50), r.randint(1, 50)
        commit_hash = f"{r.randint(1000000, 9999999):07x}"

        lang_hunks: dict[str | None, tuple[str, str, str]] = {
            "python": (
                f"""\
@@ -{ln},8 +{ln},14 @@ class {cls}:
     def {m1}(self):
-        {v1} = self._{m2}()
-        return {v1}
+        try:
+            {v1} = await self._{m2}()
+            if {v1} is None:
+                raise ValueError("{err}")
+            return {v1}
+        except Exception as e:
+            logger.error(f"{cls}.{m1} failed: {{{{e}}}}")
+            raise""",
                f"""\
@@ -{ln2},5 +{ln2},9 @@ def {m2}({v1}):
     {v2} = {mod}.{m3}({v1})
-    return {v2}
+    if not {v2}:
+        raise RuntimeError("{err}")
+    logger.info("{m2} completed: %s", {v2})
+    return {{{{"{v1}": {v2}, "status": "ok"}}}}""",
                f"""\
@@ -{hunk_old},3 +{hunk_new},7 @@
+import logging
+from {mod} import {cls}
+
+logger = logging.getLogger(__name__)""",
            ),
            "go": (
                f"""\
@@ -{ln},6 +{ln},12 @@ func (s *{cls}) {m1.title()}() error {{{{
-    return nil
+    {v1}, err := s.{m2.title()}(ctx)
+    if err != nil {{{{
+        return fmt.Errorf("{err}: %w", err)
+    }}}}
+    s.{v2} = {v1}
+    return nil""",
                f"""\
@@ -{ln2},4 +{ln2},8 @@ func (s *{cls}) {m2.title()}() (string, error) {{{{
     s.mu.RLock()
     defer s.mu.RUnlock()
-    return s.{v1}, nil
+    if s.{v1} == "" {{{{
+        return "", fmt.Errorf("{err}")
+    }}}}
+    return fmt.Sprintf("%s:%d", s.{v1}, s.{v2}), nil""",
                f"""\
@@ -{hunk_old},3 +{hunk_new},7 @@
+import (
+    "fmt"
+    "log/slog"
+)""",
            ),
            "rust": (
                f"""\
@@ -{ln},5 +{ln},11 @@ impl {cls} {{{{
     pub fn {m1}(&self) -> Result<()> {{{{
-        Ok(())
+        let {v1} = self.{m2}()?;
+        if {v1}.is_empty() {{{{
+            anyhow::bail!("{err}");
+        }}}}
+        tracing::info!("{m1} completed: {{}}", {v1});
+        Ok(())""",
                f"""\
@@ -{ln2},4 +{ln2},7 @@ impl {cls} {{{{
     fn {m2}(&self) -> Result<String> {{{{
-        Ok(self.{v1}.clone())
+        let {v2} = &self.{v1};
+        anyhow::ensure!(!{v2}.is_empty(), "{err}");
+        Ok({v2}.clone())""",
                f"""\
@@ -{hunk_old},3 +{hunk_new},6 @@
+use anyhow::Result;
+use tracing;
+use {mod}::{cls};""",
            ),
            "typescript": (
                f"""\
@@ -{ln},6 +{ln},12 @@ export class {cls} {{{{
   {m1}({v1}: string) {{{{
-    return this.{m2}({v1});
+    try {{{{
+      const {v2} = await this.{m2}({v1});
+      if (!{v2}) throw new Error('{err}');
+      return {{ status: 'ok', data: {v2} }};
+    }}}} catch (err) {{{{
+      console.error(`{cls}.{m1} failed: ${{{{err}}}}`);
+      throw err;
+    }}}}""",
                f"""\
@@ -{ln2},4 +{ln2},7 @@ export class {cls} {{{{
   private {m2}({v1}: string): {v2} {{{{
-    return this.#{v1};
+    if (!this.#{v1}) {{{{
+      throw new Error('{err}');
+    }}}}
+    return this.#{v1};""",
                f"""\
@@ -{hunk_old},3 +{hunk_new},6 @@
+import {{ {cls} }} from './{mod}';
+import type {{ {v3.title()} }} from './types';
+""",
            ),
        }
        hunk1, hunk2, hunk3 = lang_hunks.get(language, lang_hunks["python"])

        return f"""\
commit {commit_hash}
Author: dev <dev@example.com>
Date:   Mon Jan 15 14:32:00 2025 +0000

    feat({mod}): add async {m1} with error handling

diff --git a/{f1} b/{f1}
index {idx()}..{idx()} 100644
--- a/{f1}
+++ b/{f1}
{hunk1}
diff --git a/{f2} b/{f2}
index {idx()}..{idx()} 100644
--- a/{f2}
+++ b/{f2}
{hunk2}
diff --git a/{f3} b/{f3}
index {idx()}..{idx()} 100644
--- a/{f3}
+++ b/{f3}
{hunk3}
"""

    def _gen_cicd_output(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        n_pass = r.randint(20, 200)
        n_fail = r.randint(0, 5)
        n_skip = r.randint(0, 10)
        n_pkgs = r.randint(50, 300)
        install_time = r.uniform(0.5, 10)
        n_lint_files = r.randint(10, 100)
        n_type_mods = r.randint(100, 500)
        coverage = r.uniform(70, 99)
        ver = f"{r.randint(1, 9)}.{r.randint(0, 99)}.{r.randint(0, 99)}"
        artifact_size = r.uniform(0.1, 50)
        status = "PASSED" if n_fail == 0 else "FAILED"
        elapsed = r.randint(30, 600)

        lang_toolchain = {
            "python": {
                "install": f"pip install -r requirements.txt\n  Resolved {n_pkgs} packages in {install_time:.1f}s",
                "lint": f"ruff check . && ruff format --check .\n  All checks passed ({n_lint_files} files)",
                "typecheck": f"mypy src/\n  Success: {n_type_mods} modules checked",
                "test": f"pytest tests/ -v\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n  Coverage: {coverage:.1f}%",
                "build": f"python -m build\n  Built {mod}-{ver}.tar.gz ({artifact_size:.1f} MB)",
            },
            "go": {
                "install": f"go mod download\n  Resolved {n_pkgs} packages in {install_time:.1f}s",
                "lint": f"golangci-lint run ./...\n  All checks passed ({n_lint_files} files)",
                "typecheck": f"go vet ./...\n  Success: {n_type_mods} packages checked",
                "test": f"go test -v -race -coverprofile=coverage.out ./...\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n  Coverage: {coverage:.1f}%",
                "build": f"go build -o bin/{mod} ./cmd/{mod}\n  Built bin/{mod} ({artifact_size:.1f} MB)",
            },
            "rust": {
                "install": f"cargo fetch\n  Resolved {n_pkgs} crates in {install_time:.1f}s",
                "lint": f"cargo clippy -- -D warnings\n  All checks passed ({n_lint_files} files)",
                "typecheck": f"cargo check\n  Checked {n_type_mods} crates",
                "test": f"cargo test\n  {n_pass} passed, {n_fail} failed, {n_skip} ignored\n  Coverage: {coverage:.1f}%",
                "build": f"cargo build --release\n  Built target/release/{mod} ({artifact_size:.1f} MB)",
            },
            "typescript": {
                "install": f"npm ci\n  Resolved {n_pkgs} packages in {install_time:.1f}s",
                "lint": f"eslint src/ && prettier --check src/\n  All checks passed ({n_lint_files} files)",
                "typecheck": f"tsc --noEmit\n  Success: {n_type_mods} modules checked",
                "test": f"vitest run\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n  Coverage: {coverage:.1f}%",
                "build": f"npm run build\n  Built dist/{mod}-{ver}.tgz ({artifact_size:.1f} MB)",
            },
        }
        toolchain = lang_toolchain.get(
            language, r.choice(list(lang_toolchain.values()))
        )

        return f"""\
=== CI Pipeline: {mod} ===
Step 1/5: Installing dependencies...
  {toolchain["install"]}
Step 2/5: Linting...
  {toolchain["lint"]}
Step 3/5: Type checking...
  {toolchain["typecheck"]}
Step 4/5: Running tests...
  {toolchain["test"]}
Step 5/5: Building artifacts...
  {toolchain["build"]}
Pipeline {status} in {elapsed}s
"""

    def _gen_config_file(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        v1, v2, v3 = r.sample(_VARS, 3)

        lang_to_kinds: dict[str, list[str]] = {
            "python": ["yaml", "toml", "dockerfile"],
            "go": ["yaml", "makefile"],
            "rust": ["toml"],
            "typescript": ["yaml", "dockerfile"],
        }
        choices = (
            lang_to_kinds.get(language, ["yaml", "toml", "dockerfile", "makefile"])
            if language
            else ["yaml", "toml", "dockerfile", "makefile"]
        )
        kind = r.choice(choices)
        if kind == "yaml":
            port = r.randint(3000, 9999)
            workers = r.randint(1, 16)
            v2_val = r.randint(1, 1000)
            v3_val = r.choice(_MODULES)
            db_port = r.choice([5432, 3306, 27017, 6379])
            pool = r.randint(5, 50)
            return f"""\
# {mod} configuration
service:
  name: {mod}
  port: {port}
  workers: {workers}
  {v1}:
    enabled: true
    {v2}: {v2_val}
    {v3}: "{v3_val}"
  logging:
    level: info
    format: json
  database:
    host: localhost
    port: {db_port}
    pool_size: {pool}
"""
        elif kind == "toml":
            ver = f"{r.randint(0, 9)}.{r.randint(0, 99)}.{r.randint(0, 99)}"
            desc_cls = r.choice(_CLASSES)
            desc_method = r.choice(_METHODS)
            dep1, dep2 = r.choice(_MODULES), r.choice(_MODULES)
            dep1_ver = f"{r.randint(1, 5)}.{r.randint(0, 20)}"
            dep2_ver = f"{r.randint(0, 3)}.{r.randint(0, 40)}"
            tool_mod = r.choice(_MODULES)
            v1_val = r.randint(1, 100)
            v2_val = r.choice(_MODULES)
            return f"""\
[project]
name = "{mod}"
version = "{ver}"
description = "{desc_cls} {desc_method} service"

[dependencies]
{dep1} = "{dep1_ver}"
{dep2} = "{dep2_ver}"

[tool.{tool_mod}]
{v1} = {v1_val}
{v2} = "{v2_val}"
{v3} = true
"""
        elif kind == "dockerfile":
            env1_val = r.randint(1, 100)
            env2_val = r.choice(_MODULES)
            port = r.randint(3000, 9999)
            docker_lang = language or "python"
            if docker_lang == "python":
                py_ver = r.randint(10, 13)
                base_image = f"python:3.{py_ver}-slim"
                install_cmd = "COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt"
                run_cmd = f'CMD ["python", "-m", "{mod}"]'
            elif docker_lang == "go":
                go_ver = f"1.{r.randint(21, 23)}"
                base_image = f"golang:{go_ver}-alpine"
                install_cmd = "COPY go.mod go.sum ./\nRUN go mod download"
                run_cmd = f'CMD ["./bin/{mod}"]'
            elif docker_lang == "rust":
                base_image = "rust:1-slim"
                install_cmd = "COPY Cargo.toml Cargo.lock ./\nRUN cargo fetch"
                run_cmd = f'CMD ["./target/release/{mod}"]'
            else:
                node_ver = r.randint(18, 22)
                base_image = f"node:{node_ver}-alpine"
                install_cmd = "COPY package.json package-lock.json ./\nRUN npm ci"
                run_cmd = f'CMD ["node", "dist/{mod}/index.js"]'
            return f"""\
FROM {base_image}

WORKDIR /app

{install_cmd}

COPY src/ ./src/

ENV {v1.upper()}={env1_val}
ENV {v2.upper()}={env2_val}

EXPOSE {port}

{run_cmd}
"""
        else:
            return f"""\
.PHONY: build test lint clean

build:
\t@echo "Building {mod}..."
\tgo build -o bin/{mod} ./cmd/{mod}

test:
\t@echo "Testing {mod}..."
\tgo test -v -race ./...

lint:
\tgolangci-lint run ./...

clean:
\trm -rf bin/ dist/ *.egg-info
"""

    def _gen_markdown_doc(self, language: str | None = None) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        mod = r.choice(_MODULES)
        v1 = r.choice(_VARS)
        err = r.choice(_ERROR_MESSAGES)

        lang_examples = {
            "python": {
                "fence": "python",
                "code": f'from {mod} import {cls}\n\ninstance = {cls}({v1}="value")\nresult = await instance.{m1}()',
                "param_type": r.choice(
                    ("str", "int", "float", "bool", "dict", "list", "Any", "Optional")
                ),
                "return_type": r.choice(
                    ("str", "int", "bool", "dict", "list", "None", "Any")
                ),
            },
            "go": {
                "fence": "go",
                "code": f'import "{mod}"\n\nc := {mod}.New{cls}("{v1}")\nerr := c.{m1.title()}(ctx)',
                "param_type": r.choice(
                    (
                        "string",
                        "int",
                        "int64",
                        "bool",
                        "[]byte",
                        "error",
                        "context.Context",
                    )
                ),
                "return_type": r.choice(("string", "int", "bool", "error", f"*{cls}")),
            },
            "rust": {
                "fence": "rust",
                "code": f'use {mod}::{cls};\n\nlet mut c = {cls}::new("{v1}");\nc.{m1}().await?;',
                "param_type": r.choice(
                    (
                        "&str",
                        "String",
                        "i64",
                        "bool",
                        "Vec<u8>",
                        "&[u8]",
                        "Option<String>",
                    )
                ),
                "return_type": r.choice(
                    ("Result<()>", "Result<String>", "bool", "Option<String>", "&str")
                ),
            },
            "typescript": {
                "fence": "typescript",
                "code": f"import {{ {cls} }} from './{mod}';\n\nconst c = new {cls}({{ {v1}: 'value' }});\nawait c.{m1}();",
                "param_type": r.choice(
                    (
                        "string",
                        "number",
                        "boolean",
                        "Record<string, unknown>",
                        "unknown[]",
                    )
                ),
                "return_type": r.choice(
                    ("string", "number", "boolean", "void", "Promise<void>")
                ),
            },
        }
        example = lang_examples.get(language, r.choice(list(lang_examples.values())))

        v2, v3 = r.sample(_VARS, 2)
        err2 = r.choice(_ERROR_MESSAGES)
        env_var = f"AIPERF_{mod.upper()}_{v1.upper()}"

        return f"""\
# {cls}

## Overview

The `{cls}` class provides {m1} and {m2} operations for the `{mod}` module.

## Usage

```{example["fence"]}
{example["code"]}
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `{v1}` | {example["param_type"]} | required | Primary {v1} identifier |
| `{v2}` | {example["param_type"]} | `None` | Optional {v2} override |
| `{v3}` | int | `10` | Maximum {v3} per batch |
| `timeout` | float | `30.0` | Operation timeout in seconds |

Environment variable override: `{env_var}`

## API Reference

### `{m1}({v1})`

Performs the {m1} operation.

**Parameters:**
- `{v1}` ({example["param_type"]}): The input {v1}.

**Returns:** {example["return_type"]}

### `{m2}()`

Performs the {m2} operation.

**Raises:** `ValueError` if {err}.

## Errors

| Error | Condition | Recovery |
|-------|-----------|----------|
| `ValueError` | {err} | Check {v1} parameter |
| `RuntimeError` | {err2} | Retry with backoff |
| `TimeoutError` | Operation exceeds timeout | Increase timeout or reduce {v3} |
"""

    def _gen_test_output(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        cls = r.choice(_CLASSES)
        methods = r.sample(list(_METHODS), 5)

        lang_to_kind = {
            "python": "pytest",
            "go": "go",
            "rust": "cargo",
            "typescript": "jest",
        }
        kind = (
            lang_to_kind[language]
            if language in lang_to_kind
            else r.choice(["pytest", "go", "cargo"])
        )
        if kind == "pytest":
            lines = [
                "============================= test session starts ============================="
            ]
            lines.append(f"collected {r.randint(10, 100)} items\n")
            for m in methods:
                status = r.choice(["PASSED", "PASSED", "PASSED", "FAILED"])
                lines.append(f"tests/test_{mod}.py::Test{cls}::test_{m} {status}")
            n_pass = sum(1 for line in lines if "PASSED" in line)
            n_fail = len(methods) - n_pass
            dur = r.uniform(0.5, 30.0)
            lines.append(f"\n{'=' * 70}")
            lines.append(f"{n_pass} passed, {n_fail} failed in {dur:.2f}s")
            return "\n".join(lines) + "\n"
        elif kind == "jest":
            runner = r.choice(["JEST", "VITEST"])
            lines = [
                f" {runner}  v{r.randint(28, 30)}.{r.randint(0, 9)}.{r.randint(0, 9)}"
            ]
            lines.append("")
            results: list[str] = []
            for m in methods:
                passed = r.choice([True, True, True, False])
                mark = "\u2713" if passed else "\u2717"
                dur_ms = r.randint(1, 500)
                results.append(f"  {mark} {cls} > {m} ({dur_ms} ms)")
                lines.append(results[-1])
            n_pass = sum(1 for res in results if "\u2713" in res)
            n_fail = len(methods) - n_pass
            dur = r.uniform(0.5, 15.0)
            lines.append("")
            lines.append(
                f"Tests:       {n_fail} failed, {n_pass} passed, {len(methods)} total"
            )
            lines.append(f"Time:        {dur:.3f} s")
            lines.append(f"Ran all test suites matching /src/{mod}.test.ts/i.")
            return "\n".join(lines) + "\n"
        elif kind == "go":
            lines = []
            for m in methods:
                status = r.choice(["ok", "ok", "ok", "FAIL"])
                dur = r.uniform(0.001, 2.0)
                lines.append(f"--- {status}: Test{m.title()} ({dur:.3f}s)")
            lines.append(
                f"{status}  \t{mod}/{r.choice(_MODULES)}\t{r.uniform(0.1, 5.0):.3f}s"
            )
            return "\n".join(lines) + "\n"
        else:
            lines = [f"   Compiling {mod} v0.{r.randint(1, 99)}.{r.randint(0, 9)}"]
            lines.append(f"    Finished test target(s) in {r.uniform(1, 30):.2f}s")
            lines.append("     Running unittests src/lib.rs\n")
            for m in methods:
                status = r.choice(["ok", "ok", "ok", "FAILED"])
                lines.append(f"test {mod}::{cls.lower()}::test_{m} ... {status}")
            n_pass = sum(1 for line in lines if "... ok" in line)
            n_fail = len(methods) - n_pass
            lines.append(
                f"\ntest result: {'ok' if n_fail == 0 else 'FAILED'}. "
                f"{n_pass} passed; {n_fail} failed; 0 ignored"
            )
            return "\n".join(lines) + "\n"

    def _gen_user_prompt(self) -> str:
        r = self._template_rng
        template = r.choice(_USER_REQUESTS)
        base = template.format(
            module=r.choice(_MODULES),
            cls=r.choice(_CLASSES),
            method=r.choice(_METHODS),
            var=r.choice(_VARS),
            error=r.choice(_ERROR_MESSAGES),
            type=r.choice(_TYPES),
        )

        if r.random() < 0.3:
            base += "\n\n" + self._gen_prompt_context()
        return base

    def _gen_prompt_context(self) -> str:
        r = self._template_rng
        kind = r.choice(["snippet", "error_output", "constraint"])
        if kind == "snippet":
            cls = r.choice(_CLASSES)
            m1 = r.choice(_METHODS)
            v1, v2 = r.sample(_VARS, 2)
            f = r.choice(_FILE_PATHS)
            return (
                f"Here's the relevant code from `{f}`:\n\n"
                f"```\n"
                f"class {cls}:\n"
                f"    def {m1}(self, {v1}):\n"
                f"        {v2} = self._{v1}\n"
                f"        return {v2}\n"
                f"```"
            )
        elif kind == "error_output":
            err = r.choice(_ERROR_MESSAGES)
            cls = r.choice(_CLASSES)
            m1 = r.choice(_METHODS)
            f = r.choice(_FILE_PATHS)
            return (
                f"Error output:\n\n"
                f"```\n"
                f'  File "{f}", line {r.randint(10, 300)}, in {m1}\n'
                f'    raise RuntimeError("{err}")\n'
                f"RuntimeError: {err}\n"
                f"```"
            )
        else:
            return r.choice(
                (
                    "Constraint: no new dependencies allowed in this PR.",
                    "This is on the hot path — keep allocations minimal.",
                    "Must remain backward-compatible with the v1 API.",
                    f"The {r.choice(_MODULES)} service is frozen — only touch {r.choice(_MODULES)}.",
                    f"Target is under {r.randint(5, 50)}ms p99 latency.",
                    "We need this for the release on Friday — keep it simple.",
                )
            )
