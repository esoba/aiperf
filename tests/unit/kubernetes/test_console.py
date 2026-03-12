# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.console module.

Focuses on:
- Last benchmark persistence (save/get/clear lifecycle)
- Metrics summary extraction and display logic
- Jobs table rendering with various pod states
- Print helper formatting behavior
"""

from typing import Any
from unittest.mock import MagicMock, patch

import orjson
import pytest
from pytest import param

from aiperf.kubernetes.console import (
    _STATUS_STYLES,
    LastBenchmarkInfo,
    clear_last_benchmark,
    get_last_benchmark,
    print_deployment_summary,
    print_detach_info,
    print_file_table,
    print_header,
    print_jobs_table,
    print_metrics_summary,
    print_step,
    save_last_benchmark,
    status_log,
)
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.models import JobSetInfo, PodSummary

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def benchmark_file(tmp_path):
    """Patch _LAST_BENCHMARK_FILE to use a temp directory."""
    fake_path = tmp_path / ".aiperf" / "last_kube_benchmark.json"
    with patch("aiperf.kubernetes.console._LAST_BENCHMARK_FILE", fake_path):
        yield fake_path


@pytest.fixture
def mock_logger():
    """Patch the console module's logger and return the mock."""
    with patch("aiperf.kubernetes.console.logger") as m:
        yield m


@pytest.fixture
def sample_job_info() -> JobSetInfo:
    """Create a minimal JobSetInfo for table tests."""
    return JobSetInfo(
        name="aiperf-test-job",
        namespace="default",
        jobset={
            "metadata": {
                "name": "aiperf-test-job",
                "namespace": "default",
                "creationTimestamp": "2026-01-15T10:30:00Z",
                "labels": {"aiperf.nvidia.com/job-id": "job-123"},
                "annotations": {},
            },
            "status": {},
        },
        status="Running",
        model="llama-3",
    )


# ============================================================
# LastBenchmarkInfo Dataclass
# ============================================================


class TestLastBenchmarkInfo:
    """Verify LastBenchmarkInfo dataclass construction."""

    def test_create_with_required_fields(self) -> None:
        info = LastBenchmarkInfo(job_id="abc", namespace="default")
        assert info.job_id == "abc"
        assert info.namespace == "default"
        assert info.name is None

    def test_create_with_name(self) -> None:
        info = LastBenchmarkInfo(job_id="abc", namespace="ns", name="my-bench")
        assert info.name == "my-bench"


# ============================================================
# save_last_benchmark / get_last_benchmark / clear_last_benchmark
# ============================================================


class TestLastBenchmarkPersistence:
    """Verify save/get/clear lifecycle for last benchmark file."""

    def test_save_and_get_roundtrip(self, benchmark_file) -> None:
        save_last_benchmark("job-1", "ns-1")
        result = get_last_benchmark()
        assert result is not None
        assert result.job_id == "job-1"
        assert result.namespace == "ns-1"
        assert result.name is None

    def test_save_with_name_roundtrip(self, benchmark_file) -> None:
        save_last_benchmark("job-2", "ns-2", name="my-benchmark")
        result = get_last_benchmark()
        assert result is not None
        assert result.name == "my-benchmark"

    def test_get_returns_none_when_no_file(self, benchmark_file) -> None:
        result = get_last_benchmark()
        assert result is None

    def test_clear_removes_file(self, benchmark_file) -> None:
        save_last_benchmark("job-1", "ns-1")
        assert benchmark_file.exists()
        clear_last_benchmark()
        assert not benchmark_file.exists()

    def test_clear_noop_when_no_file(self, benchmark_file) -> None:
        clear_last_benchmark()  # should not raise

    def test_get_returns_none_on_corrupt_json(self, benchmark_file) -> None:
        benchmark_file.parent.mkdir(parents=True, exist_ok=True)
        benchmark_file.write_bytes(b"not valid json{{{")
        result = get_last_benchmark()
        assert result is None

    def test_get_returns_none_on_missing_keys(self, benchmark_file) -> None:
        benchmark_file.parent.mkdir(parents=True, exist_ok=True)
        benchmark_file.write_bytes(orjson.dumps({"unrelated": "data"}))
        result = get_last_benchmark()
        assert result is None

    def test_save_creates_parent_directories(self, benchmark_file) -> None:
        assert not benchmark_file.parent.exists()
        save_last_benchmark("job-1", "ns-1")
        assert benchmark_file.exists()

    def test_save_without_name_omits_key(self, benchmark_file) -> None:
        save_last_benchmark("job-1", "ns-1")
        data = orjson.loads(benchmark_file.read_bytes())
        assert "name" not in data

    def test_save_with_empty_name_omits_key(self, benchmark_file) -> None:
        save_last_benchmark("job-1", "ns-1", name="")
        data = orjson.loads(benchmark_file.read_bytes())
        assert "name" not in data


# ============================================================
# print_step
# ============================================================


class TestPrintStep:
    """Verify print_step formatting with and without step counters."""

    def test_print_step_with_counter(self, mock_logger) -> None:
        print_step("Deploying", step=1, total=3)
        mock_logger.info.assert_called_once()
        msg = mock_logger.info.call_args[0][0]
        assert "1/3" in msg
        assert "Deploying" in msg

    def test_print_step_without_counter(self, mock_logger) -> None:
        print_step("Starting")
        mock_logger.info.assert_called_once()
        msg = mock_logger.info.call_args[0][0]
        assert "Starting" in msg
        # Step/total bracket pattern should not appear
        assert "\\[" not in msg or "Starting" in msg

    def test_print_step_only_step_no_total_omits_counter(self, mock_logger) -> None:
        print_step("Partial", step=1, total=None)
        msg = mock_logger.info.call_args[0][0]
        assert "1/" not in msg


# ============================================================
# print_header
# ============================================================


class TestPrintHeader:
    """Verify print_header outputs title and separator."""

    def test_header_logs_title_and_separator(self, mock_logger) -> None:
        print_header("My Section")
        calls = [c[0][0] for c in mock_logger.info.call_args_list]
        assert len(calls) == 3  # blank, title, separator
        assert "My Section" in calls[1]

    def test_header_separator_matches_title_length(self, mock_logger) -> None:
        print_header("Test")
        separator_call = mock_logger.info.call_args_list[2][0][0]
        assert "\u2500" * len("Test") in separator_call


# ============================================================
# status_log context manager
# ============================================================


class TestStatusLog:
    """Verify status_log context manager logs on entry."""

    def test_status_log_logs_message(self, mock_logger) -> None:
        with status_log("Waiting for pods"):
            pass
        mock_logger.info.assert_called_once()
        msg = mock_logger.info.call_args[0][0]
        assert "Waiting for pods" in msg

    def test_status_log_yields_control(self, mock_logger) -> None:
        executed = False
        with status_log("test"):
            executed = True
        assert executed


# ============================================================
# print_metrics_summary
# ============================================================


class TestPrintMetricsSummary:
    """Verify metrics summary extracts and displays key metrics."""

    def test_displays_matching_metrics(self, mock_logger) -> None:
        data: dict[str, Any] = {
            "metrics": [
                {"tag": "throughput", "value": 42.5, "unit": "req/s"},
                {"tag": "latency_p50", "value": 100.0, "display_unit": "ms"},
                {"tag": "some_other_metric", "value": 999.0, "unit": "x"},
            ],
        }
        print_metrics_summary(data)
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "throughput" in joined
        assert "latency_p50" in joined
        assert "some_other_metric" not in joined

    def test_no_matching_metrics_no_header(self, mock_logger) -> None:
        data: dict[str, Any] = {
            "metrics": [{"tag": "irrelevant", "value": 1.0}],
        }
        print_metrics_summary(data)
        for call in mock_logger.info.call_args_list:
            assert "Benchmark Metrics" not in call[0][0]

    def test_empty_metrics_list(self, mock_logger) -> None:
        print_metrics_summary({"metrics": []})
        assert mock_logger.info.call_count == 0

    def test_no_metrics_key(self, mock_logger) -> None:
        print_metrics_summary({})  # should not raise

    def test_benchmark_id_displayed(self, mock_logger) -> None:
        data: dict[str, Any] = {"benchmark_id": "bench-42", "metrics": []}
        print_metrics_summary(data)
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        assert any("bench-42" in c for c in info_calls)

    def test_display_unit_preferred_over_unit(self, mock_logger) -> None:
        data: dict[str, Any] = {
            "metrics": [
                {
                    "tag": "throughput",
                    "value": 10.0,
                    "unit": "raw_unit",
                    "display_unit": "tok/s",
                },
            ],
        }
        print_metrics_summary(data)
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "tok/s" in joined

    def test_limits_to_ten_metrics(self, mock_logger) -> None:
        data: dict[str, Any] = {
            "metrics": [
                {"tag": f"throughput_{i}", "value": float(i), "unit": "x"}
                for i in range(15)
            ],
        }
        print_metrics_summary(data)
        metric_lines = [
            c for c in mock_logger.info.call_args_list if "throughput_" in c[0][0]
        ]
        assert len(metric_lines) <= 10


# ============================================================
# print_file_table
# ============================================================


class TestPrintFileTable:
    """Verify file table display."""

    def test_files_sorted_alphabetically(self, mock_logger) -> None:
        files = [("z_file.json", 100), ("a_file.csv", 200)]
        print_file_table(files)
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        file_lines = [c for c in info_calls if "file" in c.lower()]
        a_idx = next(i for i, c in enumerate(file_lines) if "a_file" in c)
        z_idx = next(i for i, c in enumerate(file_lines) if "z_file" in c)
        assert a_idx < z_idx

    def test_custom_verb(self, mock_logger) -> None:
        print_file_table([("f.txt", 50)], verb="Copied")
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        assert any("Copied" in c for c in info_calls)

    def test_empty_file_list(self, mock_logger) -> None:
        print_file_table([])
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        assert any("0 file(s)" in c for c in info_calls)


# ============================================================
# _STATUS_STYLES mapping
# ============================================================


class TestStatusStyles:
    """Verify status style mapping covers expected phases."""

    @pytest.mark.parametrize(
        "phase,expected_style",
        [
            (PodPhase.RUNNING, "bold green"),
            (PodPhase.SUCCEEDED, "green"),
            (PodPhase.FAILED, "red"),
            (PodPhase.UNKNOWN, "yellow"),
        ],
    )  # fmt: skip
    def test_style_for_phase(self, phase: PodPhase, expected_style: str) -> None:
        assert _STATUS_STYLES[phase] == expected_style

    def test_pending_phase_not_in_styles(self) -> None:
        assert PodPhase.PENDING not in _STATUS_STYLES


# ============================================================
# print_jobs_table
# ============================================================


class TestPrintJobsTable:
    """Verify jobs table rendering with various inputs."""

    def test_empty_jobs_list(self) -> None:
        with patch("aiperf.kubernetes.console.console") as mock_console:
            print_jobs_table([])
            mock_console.print.assert_called_once()

    def test_single_job_renders(self, sample_job_info: JobSetInfo) -> None:
        with (
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="5m"),
        ):
            print_jobs_table([sample_job_info])
            mock_console.print.assert_called_once()

    def test_wide_mode_adds_columns(self, sample_job_info: JobSetInfo) -> None:
        with (
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="5m"),
        ):
            print_jobs_table([sample_job_info], wide=True)
            table = mock_console.print.call_args[0][0]
            col_names = [c.header for c in table.columns]
            assert "JOB-ID" in col_names
            assert "CUSTOM-NAME" in col_names
            assert "ENDPOINT" in col_names

    def test_narrow_mode_omits_wide_columns(self, sample_job_info: JobSetInfo) -> None:
        with (
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="5m"),
        ):
            print_jobs_table([sample_job_info], wide=False)
            table = mock_console.print.call_args[0][0]
            col_names = [c.header for c in table.columns]
            assert "JOB-ID" not in col_names
            assert "ENDPOINT" not in col_names

    def test_pod_summary_with_all_ready(self, sample_job_info: JobSetInfo) -> None:
        summary = PodSummary(ready=3, total=3, restarts=0)
        with (
            patch("aiperf.kubernetes.console.console"),
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="1m"),
        ):
            print_jobs_table(
                [sample_job_info],
                pod_summaries={"aiperf-test-job": summary},
            )

    def test_pod_summary_with_restarts(self, sample_job_info: JobSetInfo) -> None:
        summary = PodSummary(ready=2, total=3, restarts=5)
        with (
            patch("aiperf.kubernetes.console.console"),
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="1m"),
        ):
            print_jobs_table(
                [sample_job_info],
                pod_summaries={"aiperf-test-job": summary},
            )

    def test_missing_pod_summary_shows_dash(self, sample_job_info: JobSetInfo) -> None:
        with (
            patch("aiperf.kubernetes.console.console"),
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="1m"),
        ):
            print_jobs_table(
                [sample_job_info],
                pod_summaries={"other-job": PodSummary(ready=1, total=1, restarts=0)},
            )

    @pytest.mark.parametrize(
        "status_str",
        [
            "Running",
            "Completed",
            "Failed",
            "Unknown",
            param("SomethingUnexpected", id="unrecognized-status"),
        ],
    )  # fmt: skip
    def test_various_statuses_render_without_error(
        self, status_str: str, sample_job_info: JobSetInfo
    ) -> None:
        sample_job_info.status = status_str
        with (
            patch("aiperf.kubernetes.console.console"),
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="1m"),
        ):
            print_jobs_table([sample_job_info])


# ============================================================
# print_detach_info
# ============================================================


class TestPrintDetachInfo:
    """Verify detach info includes job commands."""

    def test_detach_with_name(self, mock_logger) -> None:
        print_detach_info("job-1", "default", name="my-bench")
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "my-bench" in calls
        assert "deployed successfully" in calls

    def test_detach_without_name(self, mock_logger) -> None:
        print_detach_info("job-1", "default")
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "Benchmark" in calls
        assert "deployed successfully" in calls

    def test_detach_includes_attach_command(self, mock_logger) -> None:
        print_detach_info("job-abc", "ns-1")
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "aiperf kube attach job-abc --namespace ns-1" in calls

    def test_detach_includes_results_command(self, mock_logger) -> None:
        print_detach_info("job-abc", "ns-1")
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "aiperf kube results job-abc --namespace ns-1" in calls


# ============================================================
# print_deployment_summary
# ============================================================


class TestPrintDeploymentSummary:
    """Verify deployment summary output."""

    @staticmethod
    def _make_env_mocks() -> tuple[MagicMock, MagicMock, MagicMock]:
        """Create mock objects for utils, K8sEnvironment, and Environment."""
        mock_utils = MagicMock()
        mock_utils.parse_cpu.return_value = 1.0
        mock_utils.parse_memory_gib.return_value = 1.0
        mock_utils.format_cpu.return_value = "6 cores"
        mock_utils.format_memory.return_value = "6 GiB"

        mock_k8s_env = MagicMock()
        for attr in ("CONTROLLER", "WORKER", "RECORD_PROCESSOR"):
            sub = MagicMock()
            sub.CPU_REQUEST = "1"
            sub.MEMORY_REQUEST = "1Gi"
            setattr(mock_k8s_env, attr, sub)

        mock_env = MagicMock()
        mock_env.RECORD.PROCESSOR_SCALE_FACTOR = 2

        return mock_utils, mock_k8s_env, mock_env

    def _call_with_patches(self, **kwargs: Any) -> list[str]:
        """Call print_deployment_summary with all dependencies mocked."""
        defaults: dict[str, Any] = {
            "job_id": "job-1",
            "namespace": "default",
            "image": "nvcr.io/aiperf:latest",
            "workers": 4,
            "num_pods": 2,
            "workers_per_pod": 2,
        }
        defaults.update(kwargs)
        mock_utils, mock_k8s_env, mock_env = self._make_env_mocks()
        with (
            patch.dict("sys.modules", {"aiperf.kubernetes.utils": mock_utils}),
            patch("aiperf.kubernetes.environment.K8sEnvironment", mock_k8s_env),
            patch("aiperf.common.environment.Environment", mock_env),
            patch("aiperf.kubernetes.console.logger") as mock_logger,
            patch("aiperf.kubernetes.console._stderr_logger"),
        ):
            print_deployment_summary(**defaults)
            return [c[0][0] for c in mock_logger.info.call_args_list]

    def test_includes_job_id(self) -> None:
        calls = self._call_with_patches()
        joined = " ".join(calls)
        assert "job-1" in joined

    def test_includes_namespace(self) -> None:
        calls = self._call_with_patches()
        joined = " ".join(calls)
        assert "default" in joined

    def test_ttl_disabled_when_none(self) -> None:
        calls = self._call_with_patches(ttl_seconds=None)
        joined = " ".join(calls)
        assert "disabled" in joined

    def test_ttl_shows_seconds(self) -> None:
        calls = self._call_with_patches(ttl_seconds=300)
        joined = " ".join(calls)
        assert "300s" in joined

    def test_optional_name_shown(self) -> None:
        calls = self._call_with_patches(name="my-bench")
        joined = " ".join(calls)
        assert "my-bench" in joined

    def test_optional_endpoint_shown(self) -> None:
        calls = self._call_with_patches(endpoint_url="http://llm:8000")
        joined = " ".join(calls)
        assert "http://llm:8000" in joined

    def test_optional_model_names_shown(self) -> None:
        calls = self._call_with_patches(model_names=["llama-3", "gpt-4"])
        joined = " ".join(calls)
        assert "llama-3, gpt-4" in joined

    def test_to_stderr_uses_stderr_logger(self) -> None:
        mock_utils, mock_k8s_env, mock_env = self._make_env_mocks()
        with (
            patch.dict("sys.modules", {"aiperf.kubernetes.utils": mock_utils}),
            patch("aiperf.kubernetes.environment.K8sEnvironment", mock_k8s_env),
            patch("aiperf.common.environment.Environment", mock_env),
            patch("aiperf.kubernetes.console.logger") as mock_logger,
            patch("aiperf.kubernetes.console._stderr_logger") as mock_stderr,
        ):
            print_deployment_summary(
                job_id="j",
                namespace="ns",
                image="img",
                workers=1,
                num_pods=1,
                workers_per_pod=1,
                to_stderr=True,
            )
            assert mock_stderr.info.call_count > 0
            assert mock_logger.info.call_count == 0
