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
)

_METHODS = (
    "process", "handle", "validate", "transform", "execute", "initialize",
    "configure", "dispatch", "resolve", "serialize", "deserialize", "encode",
    "decode", "publish", "subscribe", "notify", "aggregate", "partition",
    "schedule", "allocate", "release", "acquire", "flush", "compress",
    "decompress", "authenticate", "authorize", "revoke", "checkpoint",
    "rollback", "migrate", "replicate", "synchronize", "reconcile",
    "invalidate", "prefetch", "evict", "rebalance", "throttle", "retry",
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

_ERROR_MESSAGES = (
    "connection refused", "timeout exceeded", "permission denied",
    "resource not found", "invalid argument", "out of memory",
    "deadlock detected", "rate limit exceeded", "authentication failed",
    "schema validation error", "serialization error", "buffer overflow",
    "index out of range", "null pointer dereference", "type mismatch",
    "missing required field", "duplicate key", "constraint violation",
    "circular dependency detected", "maximum recursion depth exceeded",
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
)

_TEXT_POOL_BLOCKS = 200
_TOOL_POOL_BLOCKS_PER_TYPE = 50
_TOOL_GENERATOR_TYPES = (
    "_gen_python_code", "_gen_go_code", "_gen_rust_code",
    "_gen_typescript_code", "_gen_bash_output", "_gen_json_response",
    "_gen_error_traceback", "_gen_git_diff", "_gen_cicd_output",
    "_gen_config_file", "_gen_markdown_doc", "_gen_test_output",
)

_LANGUAGE_AGNOSTIC_GENERATORS = (
    "_gen_bash_output", "_gen_json_response", "_gen_git_diff",
    "_gen_cicd_output", "_gen_markdown_doc",
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

    def __init__(self, config: PromptConfig, tokenizer: Tokenizer, **kwargs):
        self.config = config
        self.tokenizer = tokenizer

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
        gen_names = lang_gens + _LANGUAGE_AGNOSTIC_GENERATORS

        blocks: list[str] = []
        for gen_name in gen_names:
            gen_fn = getattr(self, gen_name)
            for _ in range(_TOOL_POOL_BLOCKS_PER_TYPE):
                blocks.append(gen_fn())

        # Language-specific mixed generators
        for _ in range(_TOOL_POOL_BLOCKS_PER_TYPE):
            blocks.append(self._gen_error_traceback(language=language))
        for _ in range(_TOOL_POOL_BLOCKS_PER_TYPE):
            blocks.append(self._gen_test_output(language=language))
        for _ in range(_TOOL_POOL_BLOCKS_PER_TYPE):
            blocks.append(self._gen_config_file(language=language))

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
        for _ in range(_TEXT_POOL_BLOCKS):
            blocks.append(self._gen_user_prompt())
        text = "\n\n".join(blocks)
        self._text_pool = self.tokenizer.encode(text)
        self.debug(
            lambda: f"Built text pool with {len(self._text_pool)} tokens "
            f"from {len(blocks)} blocks"
        )

    def _build_tool_pool(self) -> None:
        blocks: list[str] = []
        for gen_name in _TOOL_GENERATOR_TYPES:
            gen_fn = getattr(self, gen_name)
            for _ in range(_TOOL_POOL_BLOCKS_PER_TYPE):
                blocks.append(gen_fn())
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
        r = self._template_rng
        cls = r.choice(_CLASSES)
        mod = r.choice(_MODULES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
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

    def __init__(self, {v1}: {t1}, {v2}: {t2} = None):
        self._{v1} = {v1}
        self._{v2} = {v2}
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
"""

    def _gen_go_code(self) -> str:
        r = self._template_rng
        pkg1, pkg2 = r.sample(list(_GO_PACKAGES), 2)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        pkg_name = r.choice(_MODULES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
package {pkg_name}

import (
    "{pkg1}"
    "{pkg2}"
)

type {cls} struct {{{{
    {v1} string
    {v2} int
    mu  sync.RWMutex
}}}}

func New{cls}({v1} string) *{cls} {{{{
    return &{cls}{{{{{{v1}}: {v1}}}}}
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
    return {pkg2}.Sprintf("%s:%d", s.{v1}, s.{v2}), nil
}}}}
"""

    def _gen_rust_code(self) -> str:
        r = self._template_rng
        cr1, cr2 = r.sample(list(_RUST_CRATES), 2)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
use {cr1};
use {cr2};

pub struct {cls} {{{{
    {v1}: String,
    {v2}: Vec<u8>,
    initialized: bool,
}}}}

impl {cls} {{{{
    pub fn new({v1}: impl Into<String>) -> Self {{{{
        Self {{{{
            {v1}: {v1}.into(),
            {v2}: Vec::new(),
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
        let {v2} = self.{v1}.as_bytes();
        Ok(())
    }}}}
}}}}
"""

    def _gen_typescript_code(self) -> str:
        r = self._template_rng
        imp = r.choice(_TS_IMPORTS)
        imp_cls = r.choice(_CLASSES)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
import {{{{ {imp_cls} }}}} from '{imp}';

interface {cls}Config {{{{
  {v1}: string;
  {v2}?: number;
  timeout: number;
}}}}

export class {cls} {{{{
  private {v1}: string;
  private {v2}: number;

  constructor(config: {cls}Config) {{{{
    this.{v1} = config.{v1};
    this.{v2} = config.{v2} ?? 0;
  }}}}

  async {m1}({v1}: string): Promise<void> {{{{
    try {{{{
      const {v2} = await this.{m2}({v1});
      console.log(`${{{{this.{v1}}}}}: ${{{{{v2}}}}}`);
    }}}} catch (err) {{{{
      throw new Error(`{err}`);
    }}}}
  }}}}

  private async {m2}({v1}: string): Promise<number> {{{{
    return this.{v2};
  }}}}
}}}}
"""

    def _gen_bash_output(self) -> str:
        r = self._template_rng
        cmd = r.choice(_CLI_COMMANDS)
        files = r.sample(list(_FILE_PATHS), 5)
        mod = r.choice(_MODULES)
        file_listing = "\n".join(
            f"  {f:<42} {r.randint(1, 500):>4} lines  {r.randint(1, 50):>3}K"
            for f in files
        )
        n_pkgs = r.randint(10, 200)
        build_time = r.uniform(0.5, 30.0)

        return f"""\
$ {cmd}
{file_listing}
$ cd {mod} && make build
Building {mod}...
Compiling {n_pkgs} packages
  Finished release target(s) in {build_time:.1f}s
$ echo $?
0
"""

    def _gen_json_response(self) -> str:
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

    def _gen_error_traceback(self, language: str | None = None) -> str:
        r = self._template_rng
        err = r.choice(_ERROR_MESSAGES)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        f1, f2, f3 = r.sample(list(_FILE_PATHS), 3)

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
        if kind == "python":
            v = r.choice(_VARS)
            mod = r.choice(_MODULES)
            return f"""\
Traceback (most recent call last):
  File "{f1}", line {r.randint(10, 500)}, in {m1}
    result = self.{m2}(data)
  File "{f2}", line {r.randint(10, 300)}, in {m2}
    raise ValueError("{err}")
  File "{f3}", line {r.randint(1, 200)}, in __init__
    self._{v} = {mod}.{m1}()
ValueError: {err}

During handling of the above exception, another exception occurred:

RuntimeError: {cls}.{m1}() failed: {err}
"""
        elif kind == "go":
            return f"""\
goroutine {r.randint(1, 100)} [running]:
runtime/debug.Stack()
    /usr/local/go/src/runtime/debug/stack.go:{r.randint(10, 50)}
main.{cls}.{m1.title()}(...)
    {f1}:{r.randint(10, 300)}
main.{cls}.{m2.title()}(0xc000{r.randint(10000, 99999):05x})
    {f2}:{r.randint(10, 300)}
panic: {err}
"""
        elif kind == "rust":
            mod1, mod2 = r.sample(list(_MODULES), 2)
            return f"""\
thread 'main' panicked at '{err}', {f1}:{r.randint(10, 300)}
stack backtrace:
   0: std::panicking::begin_panic
   1: {mod1}::{cls}::{m1}
             at {f1}:{r.randint(10, 300)}
   2: {mod2}::{cls}::{m2}
             at {f2}:{r.randint(10, 300)}
   3: std::rt::lang_start
             at /rustc/src/rt.rs:{r.randint(50, 200)}
note: run with `RUST_BACKTRACE=1` for a full backtrace
"""
        else:
            async_cls = r.choice(_CLASSES)
            async_method = r.choice(_METHODS)
            return f"""\
Error: {err}
    at {cls}.{m1} ({f1}:{r.randint(10, 300)}:{r.randint(1, 40)})
    at {cls}.{m2} ({f2}:{r.randint(10, 300)}:{r.randint(1, 40)})
    at processTicksAndRejections (node:internal/process/task_queues:{r.randint(50, 100)})
    at async {async_cls}.{async_method} ({f3}:{r.randint(10, 300)})
"""

    def _gen_git_diff(self) -> str:
        r = self._template_rng
        f1, f2 = r.sample(list(_FILE_PATHS), 2)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        cls = r.choice(_CLASSES)
        ln = r.randint(10, 200)
        err = r.choice(_ERROR_MESSAGES)
        idx1a, idx1b = r.randint(1000000, 9999999), r.randint(1000000, 9999999)
        idx2a, idx2b = r.randint(1000000, 9999999), r.randint(1000000, 9999999)
        hunk2_old, hunk2_new = r.randint(1, 50), r.randint(1, 50)

        return f"""\
diff --git a/{f1} b/{f1}
index {idx1a:07x}..{idx1b:07x} 100644
--- a/{f1}
+++ b/{f1}
@@ -{ln},8 +{ln},12 @@ class {cls}:
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
+            raise

     def {m2}(self):
         return self._{v2}
diff --git a/{f2} b/{f2}
index {idx2a:07x}..{idx2b:07x} 100644
--- a/{f2}
+++ b/{f2}
@@ -{hunk2_old},3 +{hunk2_new},5 @@
+import logging
+
+logger = logging.getLogger(__name__)
"""

    def _gen_cicd_output(self) -> str:
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

        return f"""\
=== CI Pipeline: {mod} ===
Step 1/5: Installing dependencies...
  Resolved {n_pkgs} packages in {install_time:.1f}s
Step 2/5: Linting...
  All checks passed ({n_lint_files} files)
Step 3/5: Type checking...
  Success: {n_type_mods} modules checked
Step 4/5: Running tests...
  {n_pass} passed, {n_fail} failed, {n_skip} skipped
  Coverage: {coverage:.1f}%
Step 5/5: Building artifacts...
  Built {mod}-{ver}.tar.gz ({artifact_size:.1f} MB)
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
            py_ver = r.randint(10, 13)
            copy_file = r.choice(_FILE_PATHS)
            env1_val = r.randint(1, 100)
            env2_val = r.choice(_MODULES)
            port = r.randint(3000, 9999)
            return f"""\
FROM python:3.{py_ver}-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY {copy_file} .

ENV {v1.upper()}={env1_val}
ENV {v2.upper()}={env2_val}

EXPOSE {port}

CMD ["python", "-m", "{mod}"]
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

    def _gen_markdown_doc(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        mod = r.choice(_MODULES)
        v1 = r.choice(_VARS)
        param_type = r.choice(_TYPES)
        return_type = r.choice(_TYPES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
# {cls}

## Overview

The `{cls}` class provides {m1} and {m2} operations for the `{mod}` module.

## Usage

```python
from {mod} import {cls}

instance = {cls}({v1}="value")
result = await instance.{m1}()
```

## API Reference

### `{m1}({v1})`

Performs the {m1} operation.

**Parameters:**
- `{v1}` ({param_type}): The input {v1}.

**Returns:** {return_type}

### `{m2}()`

Performs the {m2} operation.

**Raises:** `ValueError` if {err}.
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
            "typescript": "pytest",
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
        return template.format(
            module=r.choice(_MODULES),
            cls=r.choice(_CLASSES),
            method=r.choice(_METHODS),
            var=r.choice(_VARS),
            error=r.choice(_ERROR_MESSAGES),
            type=r.choice(_TYPES),
        )
