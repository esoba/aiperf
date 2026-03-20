#!/usr/bin/env bash
# Kubernetes E2E test runner with per-file output logging and timeouts.
# Usage: ./run_k8s_tests.sh [test_file_pattern]
# Example: ./run_k8s_tests.sh test_benchmark
#          ./run_k8s_tests.sh           # runs all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/k8s_test_logs"
CLUSTER_NAME="${K8S_TEST_CLUSTER:-aiperf-26d8cbe8}"
TIMEOUT_SECS="${K8S_TEST_TIMEOUT:-600}"  # 10 min default per test file

mkdir -p "$LOG_DIR"

# Test files in recommended order (dependencies first)
TEST_FILES=(
    test_deployment.py
    test_benchmark.py
    test_metrics.py
    test_edge_cases.py
    test_scaling.py
    test_cli_commands.py
    test_kube_profile.py
    test_operator.py
    test_helm.py
)

# Filter if pattern provided
if [[ -n "${1:-}" ]]; then
    FILTERED=()
    for f in "${TEST_FILES[@]}"; do
        if [[ "$f" == *"$1"* ]]; then
            FILTERED+=("$f")
        fi
    done
    TEST_FILES=("${FILTERED[@]}")
fi

echo "============================================"
echo "K8s E2E Test Runner"
echo "Cluster: $CLUSTER_NAME"
echo "Timeout: ${TIMEOUT_SECS}s per file"
echo "Log dir: $LOG_DIR"
echo "Tests:   ${#TEST_FILES[@]} files"
echo "============================================"

PASS=0
FAIL=0
SKIP=0
RESULTS=()

for test_file in "${TEST_FILES[@]}"; do
    name="${test_file%.py}"
    log_file="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo ">>> Running $test_file (timeout: ${TIMEOUT_SECS}s)"
    echo ">>> Log: $log_file"

    start_time=$(date +%s)

    set +e
    timeout "$TIMEOUT_SECS" uv run pytest \
        "tests/kubernetes/${test_file}" \
        -v -s \
        --tb=long \
        --k8s-reuse-cluster \
        --k8s-skip-build \
        --k8s-skip-load \
        --k8s-skip-cleanup \
        --k8s-skip-preflight \
        --k8s-cluster "$CLUSTER_NAME" \
        --k8s-benchmark-timeout 300 \
        -m "k8s and not k8s_slow" \
        2>&1 | tee "$log_file"
    exit_code=$?
    set -e

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    if [[ $exit_code -eq 0 ]]; then
        status="PASS"
        PASS=$((PASS + 1))
    elif [[ $exit_code -eq 124 ]]; then
        status="TIMEOUT"
        FAIL=$((FAIL + 1))
    elif [[ $exit_code -eq 5 ]]; then
        status="NO_TESTS"
        SKIP=$((SKIP + 1))
    else
        status="FAIL"
        FAIL=$((FAIL + 1))
    fi

    RESULTS+=("$status ${elapsed}s $test_file")
    echo ">>> $test_file: $status (${elapsed}s)"
done

echo ""
echo "============================================"
echo "SUMMARY"
echo "============================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "PASS: $PASS  FAIL: $FAIL  SKIP: $SKIP"
echo "============================================"
