# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tokenizer_validator: alias resolution, cache warming, and the public entry point."""

import os
from unittest.mock import MagicMock, patch

import pytest
from pytest import param

from aiperf.common.exceptions import TokenizerError
from aiperf.common.tokenizer import AliasResolutionResult
from aiperf.common.tokenizer_validator import (
    _cache_tokenizer,
    _prefetch_tokenizers,
    _resolve_aliases,
    validate_tokenizer_early,
)

# ---------------------------------------------------------------------------
# _cache_tokenizer (subprocess target)
# ---------------------------------------------------------------------------


class TestCacheTokenizer:
    """Tests for _cache_tokenizer subprocess function."""

    def test_returns_name_and_positive_elapsed(self):
        with patch("aiperf.common.tokenizer.Tokenizer.from_pretrained") as mock_fp:
            mock_fp.return_value = MagicMock()
            name, elapsed = _cache_tokenizer("gpt2", False, "main")

        assert name == "gpt2"
        assert elapsed >= 0
        mock_fp.assert_called_once_with(
            "gpt2", trust_remote_code=False, revision="main", resolve_alias=False
        )

    def test_passes_trust_remote_code_and_revision(self):
        with patch("aiperf.common.tokenizer.Tokenizer.from_pretrained") as mock_fp:
            mock_fp.return_value = MagicMock()
            _cache_tokenizer("custom/model", True, "v2.0")

        mock_fp.assert_called_once_with(
            "custom/model", trust_remote_code=True, revision="v2.0", resolve_alias=False
        )

    def test_propagates_tokenizer_error(self):
        with patch("aiperf.common.tokenizer.Tokenizer.from_pretrained") as mock_fp:
            mock_fp.side_effect = TokenizerError("bad", tokenizer_name="gpt2")
            with pytest.raises(TokenizerError, match="bad"):
                _cache_tokenizer("gpt2", False, "main")


# ---------------------------------------------------------------------------
# _prefetch_tokenizers
# ---------------------------------------------------------------------------


class TestPrefetchTokenizers:
    """Tests for _prefetch_tokenizers concurrent caching."""

    @pytest.fixture
    def logger_and_messages(self):
        messages: list[str] = []
        logger = MagicMock()
        logger.info = MagicMock(side_effect=lambda msg: messages.append(msg))
        return logger, messages

    @pytest.fixture
    def console(self):
        from rich.console import Console

        return Console(force_terminal=False)

    def _make_pool(self, name_to_future: dict[str, MagicMock]) -> MagicMock:
        """Build a mock ProcessPoolExecutor whose submit returns per-name futures."""
        pool = MagicMock()
        pool.submit.side_effect = lambda fn, n, *a, **kw: name_to_future[n]
        pool.__enter__ = MagicMock(return_value=pool)
        pool.__exit__ = MagicMock(return_value=False)
        return pool

    def _patch_pool_and_completed(self, pool, futures_dict):
        """Context manager that patches ProcessPoolExecutor and as_completed."""
        return (
            patch("concurrent.futures.ProcessPoolExecutor", return_value=pool),
            patch(
                "concurrent.futures.as_completed",
                return_value=iter(futures_dict.values()),
            ),
        )

    def _run_prefetch(self, names, logger, console, name_to_future):
        """Helper to run _prefetch_tokenizers with properly mocked executor."""
        pool = self._make_pool(name_to_future)
        p1, p2 = self._patch_pool_and_completed(pool, name_to_future)
        with p1, p2:
            _prefetch_tokenizers(names, False, "main", logger, console)

    def test_logs_cached_on_success(self, logger_and_messages, console):
        logger, messages = logger_and_messages
        future = MagicMock()
        future.result.return_value = ("gpt2", 1.23)

        self._run_prefetch({"gpt2"}, logger, console, {"gpt2": future})

        assert any("Cached gpt2" in m for m in messages)
        assert any("1.23s" in m for m in messages)

    def test_logs_summary_with_total_time(self, logger_and_messages, console):
        logger, messages = logger_and_messages
        future = MagicMock()
        future.result.return_value = ("gpt2", 0.5)

        self._run_prefetch({"gpt2"}, logger, console, {"gpt2": future})

        assert any("1 tokenizer cached" in m for m in messages)
        # Summary includes total elapsed time
        assert any("cached" in m and "s" in m for m in messages)

    def test_forwards_trust_remote_code_and_revision(
        self, logger_and_messages, console
    ):
        logger, _ = logger_and_messages
        future = MagicMock()
        future.result.return_value = ("model", 0.1)
        pool = self._make_pool({"model": future})
        p1, p2 = self._patch_pool_and_completed(pool, {"model": future})
        with p1, p2:
            _prefetch_tokenizers({"model"}, True, "v2.0", logger, console)

        pool.submit.assert_called_once_with(_cache_tokenizer, "model", True, "v2.0")

    def test_caches_multiple_tokenizers(self, logger_and_messages, console):
        logger, messages = logger_and_messages
        future_a = MagicMock()
        future_a.result.return_value = ("modelA", 0.5)
        future_b = MagicMock()
        future_b.result.return_value = ("modelB", 0.8)

        self._run_prefetch(
            {"modelA", "modelB"},
            logger,
            console,
            {"modelA": future_a, "modelB": future_b},
        )

        cached = [m for m in messages if m.strip().startswith("Cached ")]
        assert len(cached) == 2

    def test_exits_on_tokenizer_error(self, logger_and_messages, console):
        logger, _ = logger_and_messages
        future = MagicMock()
        future.result.side_effect = TokenizerError("gated repo", tokenizer_name="llama")
        pool = self._make_pool({"llama": future})
        p1, p2 = self._patch_pool_and_completed(pool, {"llama": future})

        with (
            p1,
            p2,
            patch(
                "aiperf.common.tokenizer_display.display_tokenizer_validation_error"
            ) as mock_display,
            pytest.raises(SystemExit),
        ):
            _prefetch_tokenizers({"llama"}, False, "main", logger, console)

        mock_display.assert_called_once()
        kwargs = mock_display.call_args.kwargs
        assert kwargs["console"] is console
        assert kwargs["cause_chain"] is not None
        assert kwargs["error_message"] is not None

    def test_error_display_uses_tokenizer_name_attr(self, logger_and_messages, console):
        logger, _ = logger_and_messages
        future = MagicMock()
        future.result.side_effect = TokenizerError("fail", tokenizer_name="my-model")
        pool = self._make_pool({"my-model": future})
        p1, p2 = self._patch_pool_and_completed(pool, {"my-model": future})

        with (
            p1,
            p2,
            patch(
                "aiperf.common.tokenizer_display.display_tokenizer_validation_error"
            ) as mock_display,
            pytest.raises(SystemExit),
        ):
            _prefetch_tokenizers({"my-model"}, False, "main", logger, console)

        assert mock_display.call_args[0][0] == "my-model"

    def test_error_display_falls_back_to_future_name(
        self, logger_and_messages, console
    ):
        logger, _ = logger_and_messages
        future = MagicMock()
        future.result.side_effect = RuntimeError("unknown error")
        pool = self._make_pool({"fallback-name": future})
        p1, p2 = self._patch_pool_and_completed(pool, {"fallback-name": future})

        with (
            p1,
            p2,
            patch(
                "aiperf.common.tokenizer_display.display_tokenizer_validation_error"
            ) as mock_display,
            pytest.raises(SystemExit),
        ):
            _prefetch_tokenizers({"fallback-name"}, False, "main", logger, console)

        assert mock_display.call_args[0][0] == "fallback-name"


# ---------------------------------------------------------------------------
# _resolve_aliases
# ---------------------------------------------------------------------------


class TestResolveAliases:
    """Tests for _resolve_aliases."""

    @pytest.fixture
    def logger_and_messages(self):
        messages: list[str] = []
        logger = MagicMock()
        logger.info = MagicMock(side_effect=lambda msg: messages.append(msg))
        return logger, messages

    @pytest.fixture
    def console(self):
        from rich.console import Console

        return Console(force_terminal=False)

    def test_resolves_single_name(self, logger_and_messages, console):
        logger, _ = logger_and_messages
        result = AliasResolutionResult(resolved_name="openai-community/gpt2")
        with patch(
            "aiperf.common.tokenizer.Tokenizer.resolve_alias", return_value=result
        ):
            resolved = _resolve_aliases(["gpt2"], logger, console)

        assert resolved == {"gpt2": "openai-community/gpt2"}

    def test_resolves_multiple_names(self, logger_and_messages, console):
        logger, _ = logger_and_messages
        results = {
            "gpt2": AliasResolutionResult(resolved_name="openai-community/gpt2"),
            "bert": AliasResolutionResult(
                resolved_name="google-bert/bert-base-uncased"
            ),
        }
        with patch(
            "aiperf.common.tokenizer.Tokenizer.resolve_alias",
            side_effect=lambda n: results[n],
        ):
            resolved = _resolve_aliases(["gpt2", "bert"], logger, console)

        assert resolved == {
            "gpt2": "openai-community/gpt2",
            "bert": "google-bert/bert-base-uncased",
        }

    def test_canonical_name_maps_to_itself(self, logger_and_messages, console):
        logger, _ = logger_and_messages
        result = AliasResolutionResult(resolved_name="meta-llama/Llama-3.1-8B")
        with patch(
            "aiperf.common.tokenizer.Tokenizer.resolve_alias", return_value=result
        ):
            resolved = _resolve_aliases(["meta-llama/Llama-3.1-8B"], logger, console)

        assert resolved == {"meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B"}

    def test_exits_on_resolution_failure(self, logger_and_messages, console):
        logger, _ = logger_and_messages
        with (
            patch(
                "aiperf.common.tokenizer.Tokenizer.resolve_alias",
                side_effect=RuntimeError("network down"),
            ),
            pytest.raises(SystemExit),
        ):
            _resolve_aliases(["bad-model"], logger, console)

        logger.error.assert_called_once()
        assert "bad-model" in logger.error.call_args[0][0]

    def test_exits_on_ambiguous_name(self, logger_and_messages, console):
        logger, _ = logger_and_messages
        result = AliasResolutionResult(
            resolved_name="llama",
            suggestions=[("meta-llama/Llama-3.1-8B", 1_000_000)],
        )
        with (
            patch(
                "aiperf.common.tokenizer.Tokenizer.resolve_alias", return_value=result
            ),
            patch(
                "aiperf.common.tokenizer_display.display_tokenizer_ambiguous_name"
            ) as mock_display,
            pytest.raises(SystemExit),
        ):
            _resolve_aliases(["llama"], logger, console)

        mock_display.assert_called_once()

    def test_builds_display_entries_with_correct_was_resolved(
        self, logger_and_messages, console
    ):
        logger, _ = logger_and_messages
        results = {
            "gpt2": AliasResolutionResult(resolved_name="openai-community/gpt2"),
            "meta-llama/Llama-3.1-8B": AliasResolutionResult(
                resolved_name="meta-llama/Llama-3.1-8B"
            ),
        }
        with (
            patch(
                "aiperf.common.tokenizer.Tokenizer.resolve_alias",
                side_effect=lambda n: results[n],
            ),
            patch(
                "aiperf.common.tokenizer_display.log_tokenizer_validation_results"
            ) as mock_log,
        ):
            _resolve_aliases(["gpt2", "meta-llama/Llama-3.1-8B"], logger, console)

        entries = mock_log.call_args[0][0]
        assert len(entries) == 2
        assert entries[0].was_resolved is True
        assert entries[0].original_name == "gpt2"
        assert entries[1].was_resolved is False
        assert entries[1].original_name == "meta-llama/Llama-3.1-8B"

    def test_logs_validation_results(self, logger_and_messages, console):
        logger, messages = logger_and_messages
        result = AliasResolutionResult(resolved_name="gpt2")
        with patch(
            "aiperf.common.tokenizer.Tokenizer.resolve_alias", return_value=result
        ):
            _resolve_aliases(["gpt2"], logger, console)

        assert any("validated" in m for m in messages)


# ---------------------------------------------------------------------------
# validate_tokenizer_early
# ---------------------------------------------------------------------------


class TestValidateTokenizerEarly:
    """Tests for the public entry point."""

    @pytest.fixture
    def logger(self):
        logger = MagicMock()
        logger.info = MagicMock()
        logger.debug = MagicMock()
        return logger

    @pytest.fixture
    def base_user_config(self):
        """Minimal config mock with fields needed by validate_tokenizer_early."""
        from aiperf.common.enums import DatasetType

        config = MagicMock()
        config.endpoint.use_server_token_count = False
        config.get_model_names.return_value = ["model-a"]
        # Default dataset is synthetic
        default_ds = MagicMock()
        default_ds.type = DatasetType.SYNTHETIC
        config.get_default_dataset.return_value = default_ds
        config.tokenizer.name = None
        config.tokenizer.trust_remote_code = False
        config.tokenizer.revision = "main"
        return config

    @pytest.fixture
    def mock_plugins(self):
        meta = MagicMock()
        meta.produces_tokens = True
        meta.tokenizes_input = True
        with patch(
            "aiperf.plugin.plugins.get_endpoint_metadata", return_value=meta
        ) as mock_get:
            yield mock_get

    def test_skips_when_server_token_count_with_real_data(
        self, logger, base_user_config
    ):
        from aiperf.common.enums import DatasetType

        base_user_config.endpoint.use_server_token_count = True
        file_ds = MagicMock()
        file_ds.type = DatasetType.FILE
        base_user_config.get_default_dataset.return_value = file_ds

        with patch("aiperf.plugin.plugins.get_endpoint_metadata") as mock_get:
            meta = MagicMock()
            meta.produces_tokens = True
            meta.tokenizes_input = True
            mock_get.return_value = meta
            result = validate_tokenizer_early(base_user_config, logger)

        assert result is None
        logger.debug.assert_called()

    def test_skips_when_endpoint_needs_no_tokenizer(self, logger, base_user_config):
        with patch("aiperf.plugin.plugins.get_endpoint_metadata") as mock_get:
            meta = MagicMock()
            meta.produces_tokens = False
            meta.tokenizes_input = False
            mock_get.return_value = meta
            result = validate_tokenizer_early(base_user_config, logger)

        assert result is None

    def test_does_not_skip_for_synthetic_with_server_token_count(
        self, logger, base_user_config, mock_plugins
    ):
        base_user_config.endpoint.use_server_token_count = True
        # All dataset fields None => synthetic => should NOT skip

        with (
            patch(
                "aiperf.common.tokenizer_validator._resolve_aliases",
                return_value={"model-a": "resolved-a"},
            ),
            patch("aiperf.common.tokenizer_validator._prefetch_tokenizers"),
        ):
            result = validate_tokenizer_early(base_user_config, logger)

        assert result is not None

    def test_calls_resolve_and_prefetch(self, logger, base_user_config, mock_plugins):
        with (
            patch(
                "aiperf.common.tokenizer_validator._resolve_aliases",
                return_value={"model-a": "resolved-a"},
            ) as mock_resolve,
            patch(
                "aiperf.common.tokenizer_validator._prefetch_tokenizers"
            ) as mock_prefetch,
        ):
            validate_tokenizer_early(base_user_config, logger)

        mock_resolve.assert_called_once()
        mock_prefetch.assert_called_once()
        assert mock_prefetch.call_args[0][0] == {"resolved-a"}

    def test_returns_mapping_from_model_names(
        self, logger, base_user_config, mock_plugins
    ):
        base_user_config.get_model_names.return_value = ["model-a", "model-b"]

        with (
            patch(
                "aiperf.common.tokenizer_validator._resolve_aliases",
                return_value={"model-a": "resolved-a", "model-b": "resolved-b"},
            ),
            patch("aiperf.common.tokenizer_validator._prefetch_tokenizers"),
        ):
            result = validate_tokenizer_early(base_user_config, logger)

        assert result == {"model-a": "resolved-a", "model-b": "resolved-b"}

    def test_explicit_tokenizer_name_maps_all_models(
        self, logger, base_user_config, mock_plugins
    ):
        base_user_config.tokenizer.name = "custom-tok"
        base_user_config.get_model_names.return_value = ["model-a", "model-b"]

        with (
            patch(
                "aiperf.common.tokenizer_validator._resolve_aliases",
                return_value={"custom-tok": "resolved-tok"},
            ),
            patch("aiperf.common.tokenizer_validator._prefetch_tokenizers"),
        ):
            result = validate_tokenizer_early(base_user_config, logger)

        assert result == {"model-a": "resolved-tok", "model-b": "resolved-tok"}

    def test_deduplicates_tokenizers_for_prefetch(
        self, logger, base_user_config, mock_plugins
    ):
        base_user_config.get_model_names.return_value = ["model-a", "model-b"]

        with (
            patch(
                "aiperf.common.tokenizer_validator._resolve_aliases",
                return_value={"model-a": "same-tok", "model-b": "same-tok"},
            ),
            patch(
                "aiperf.common.tokenizer_validator._prefetch_tokenizers"
            ) as mock_prefetch,
        ):
            validate_tokenizer_early(base_user_config, logger)

        # Only one unique tokenizer passed to prefetch
        assert mock_prefetch.call_args[0][0] == {"same-tok"}

    def test_skips_prefetch_when_offline(
        self, logger, base_user_config, mock_plugins, monkeypatch
    ):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

        with (
            patch(
                "aiperf.common.tokenizer_validator._resolve_aliases",
                return_value={"model-a": "resolved-a"},
            ),
            patch(
                "aiperf.common.tokenizer_validator._prefetch_tokenizers"
            ) as mock_prefetch,
        ):
            validate_tokenizer_early(base_user_config, logger)

        mock_prefetch.assert_not_called()

    @pytest.mark.parametrize(
        ("hf_offline", "transformers_offline"),
        [
            param("1", "", id="only_hf_hub"),
            param("", "1", id="only_transformers"),
        ],
    )  # fmt: skip
    def test_prefetches_when_only_one_offline_var_set(
        self,
        logger,
        base_user_config,
        mock_plugins,
        monkeypatch,
        hf_offline,
        transformers_offline,
    ):
        if hf_offline:
            monkeypatch.setenv("HF_HUB_OFFLINE", hf_offline)
        else:
            monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        if transformers_offline:
            monkeypatch.setenv("TRANSFORMERS_OFFLINE", transformers_offline)
        else:
            monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        with (
            patch(
                "aiperf.common.tokenizer_validator._resolve_aliases",
                return_value={"model-a": "resolved-a"},
            ),
            patch(
                "aiperf.common.tokenizer_validator._prefetch_tokenizers"
            ) as mock_prefetch,
        ):
            validate_tokenizer_early(base_user_config, logger)

        mock_prefetch.assert_called_once()


# ---------------------------------------------------------------------------
# _enable_hf_offline_mode (bootstrap.py)
# ---------------------------------------------------------------------------


class TestEnableHfOfflineMode:
    """Tests for _enable_hf_offline_mode in bootstrap.py."""

    def test_sets_both_env_vars(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        from aiperf.common.bootstrap import _enable_hf_offline_mode

        _enable_hf_offline_mode()

        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    def test_overwrites_existing_values(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

        from aiperf.common.bootstrap import _enable_hf_offline_mode

        _enable_hf_offline_mode()

        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


class TestBootstrapOfflineMode:
    """Tests that bootstrap calls _enable_hf_offline_mode for child processes."""

    @pytest.fixture(autouse=True)
    def setup_bootstrap_mocks(
        self,
        mock_psutil_process,
        mock_setup_child_process_logging,
        register_dummy_services,
    ):
        pass

    def test_offline_mode_enabled_in_child_process(
        self,
        service_config_no_uvloop,
        mock_log_queue,
        monkeypatch,
    ):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        # Simulate being a child process
        with patch("multiprocessing.parent_process", return_value=MagicMock()):
            from aiperf.common.bootstrap import bootstrap_and_run_service

            bootstrap_and_run_service(
                "test_dummy",
                run=service_config_no_uvloop,
                log_queue=mock_log_queue,
                service_id="test_dummy",
            )

        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    def test_offline_mode_not_set_in_main_process(
        self,
        service_config_no_uvloop,
        mock_log_queue,
        monkeypatch,
    ):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        # Simulate being the main process
        with patch("multiprocessing.parent_process", return_value=None):
            from aiperf.common.bootstrap import bootstrap_and_run_service

            bootstrap_and_run_service(
                "test_dummy",
                run=service_config_no_uvloop,
                log_queue=mock_log_queue,
                service_id="test_dummy",
            )

        assert os.environ.get("HF_HUB_OFFLINE") is None
        assert os.environ.get("TRANSFORMERS_OFFLINE") is None
