#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end test for AIPerf Kubernetes deployment using minikube
#
# Usage:
#   ./deploy/test-k8s-e2e.sh              # Full test (build, deploy, monitor, cleanup)
#   ./deploy/test-k8s-e2e.sh --no-cleanup # Keep cluster after test
#   ./deploy/test-k8s-e2e.sh --skip-build # Skip image building (use existing images)
#   ./deploy/test-k8s-e2e.sh --help       # Show help

set -euo pipefail

# Configuration
CLUSTER_NAME="${CLUSTER_NAME:-aiperf-test}"
AIPERF_IMAGE="${AIPERF_IMAGE:-aiperf:local}"
MOCK_SERVER_IMAGE="${MOCK_SERVER_IMAGE:-aiperf-mock-server:latest}"
CONFIG_FILE="${CONFIG_FILE:-dev/deploy/test-benchmark-config.yaml}"
JOBSET_VERSION="${JOBSET_VERSION:-v0.8.0}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
TIMEOUT="${TIMEOUT:-300}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
CLEANUP=true
SKIP_BUILD=false
VERBOSE=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

#######################################
# Logging functions
#######################################
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_step() {
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  $*${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

#######################################
# Show usage
#######################################
show_help() {
    cat << EOF
AIPerf Kubernetes End-to-End Test Script

Usage: $(basename "$0") [OPTIONS]

Options:
    --no-cleanup     Keep the minikube cluster after test completes
    --skip-build     Skip Docker image building (use existing images)
    --verbose        Show verbose output
    --cluster NAME   Set cluster name (default: aiperf-test)
    --timeout SECS   Set timeout in seconds (default: 300)
    --help           Show this help message

Environment Variables:
    CLUSTER_NAME       Minikube profile name (default: aiperf-test)
    AIPERF_IMAGE       AIPerf image name (default: aiperf:local)
    MOCK_SERVER_IMAGE  Mock server image (default: aiperf-mock-server:latest)
    CONFIG_FILE        Benchmark config file (default: dev/deploy/test-benchmark-config.yaml)
    JOBSET_VERSION     JobSet controller version (default: v0.8.0)
    POLL_INTERVAL      Polling interval in seconds (default: 5)
    TIMEOUT            Timeout in seconds (default: 300)

Examples:
    # Run full test
    ./deploy/test-k8s-e2e.sh

    # Run without cleanup for debugging
    ./deploy/test-k8s-e2e.sh --no-cleanup

    # Use existing images
    ./deploy/test-k8s-e2e.sh --skip-build

    # Custom cluster name
    ./deploy/test-k8s-e2e.sh --cluster my-test-cluster
EOF
}

#######################################
# Parse arguments
#######################################
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-cleanup)
                CLEANUP=false
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --cluster)
                CLUSTER_NAME="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

#######################################
# Check prerequisites
#######################################
check_prerequisites() {
    log_step "Checking prerequisites"

    local missing=()

    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    fi

    if ! command -v minikube &> /dev/null; then
        missing+=("minikube")
    fi

    if ! command -v kubectl &> /dev/null; then
        missing+=("kubectl")
    fi

    if ! command -v uv &> /dev/null; then
        missing+=("uv")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Please install them before running this script"
        exit 1
    fi

    if [[ ! -f "${PROJECT_ROOT}/${CONFIG_FILE}" ]]; then
        log_error "Config file not found: ${PROJECT_ROOT}/${CONFIG_FILE}"
        exit 1
    fi

    log_success "All prerequisites met"
}

#######################################
# Build Docker images
#######################################
build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_step "Skipping image build (--skip-build)"
        return
    fi

    log_step "Building Docker images"

    cd "${PROJECT_ROOT}"

    log_info "Building ${AIPERF_IMAGE}..."
    docker build -t "${AIPERF_IMAGE}" .
    log_success "Built ${AIPERF_IMAGE}"

    log_info "Building ${MOCK_SERVER_IMAGE}..."
    docker build -f dev/deploy/Dockerfile.mock-server -t "${MOCK_SERVER_IMAGE}" .
    log_success "Built ${MOCK_SERVER_IMAGE}"
}

#######################################
# Create minikube cluster
#######################################
create_cluster() {
    log_step "Creating minikube cluster: ${CLUSTER_NAME}"

    if minikube status -p "${CLUSTER_NAME}" &>/dev/null; then
        log_warn "Profile ${CLUSTER_NAME} already exists, deleting..."
        minikube delete -p "${CLUSTER_NAME}"
    fi

    minikube start -p "${CLUSTER_NAME}" --driver=docker --wait=all
    log_success "Cluster ${CLUSTER_NAME} created"

    kubectl config use-context "${CLUSTER_NAME}" > /dev/null
    kubectl cluster-info --context "${CLUSTER_NAME}" > /dev/null
    log_success "kubectl configured for ${CLUSTER_NAME}"
}

#######################################
# Load images into minikube
#######################################
load_images() {
    log_step "Loading images into minikube cluster"

    log_info "Loading ${AIPERF_IMAGE}..."
    minikube image load "${AIPERF_IMAGE}" -p "${CLUSTER_NAME}"

    log_info "Loading ${MOCK_SERVER_IMAGE}..."
    minikube image load "${MOCK_SERVER_IMAGE}" -p "${CLUSTER_NAME}"

    log_success "Images loaded into cluster"
}

#######################################
# Install JobSet controller
#######################################
install_jobset() {
    log_step "Installing JobSet controller ${JOBSET_VERSION}"

    kubectl apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/${JOBSET_VERSION}/manifests.yaml"

    log_info "Waiting for JobSet controller to be ready..."
    kubectl wait --for=condition=available --timeout=60s \
        deployment/jobset-controller-manager -n jobset-system || true

    log_success "JobSet controller installed"
}

#######################################
# Deploy mock server
#######################################
deploy_mock_server() {
    log_step "Deploying mock server"

    kubectl apply -f "${PROJECT_ROOT}/dev/deploy/mock-server.yaml"

    log_info "Waiting for mock server rollout..."
    kubectl rollout status deployment/aiperf-mock-server --timeout=120s

    log_success "Mock server deployed and ready"
}

#######################################
# Deploy benchmark
#######################################
deploy_benchmark() {
    log_step "Deploying benchmark"

    cd "${PROJECT_ROOT}"

    # Generate manifest with imagePullPolicy: Never for minikube (images loaded locally)
    local manifest
    manifest=$(uv run aiperf kube deploy --user-config "${CONFIG_FILE}" --image "${AIPERF_IMAGE}" --dry-run 2>/dev/null)
    manifest=$(echo "$manifest" | sed "s/image: ${AIPERF_IMAGE}/image: ${AIPERF_IMAGE}\n              imagePullPolicy: Never/g")

    # Apply and capture the namespace
    local output
    output=$(echo "$manifest" | kubectl apply -f - 2>&1)
    echo "$output"

    # Extract namespace from output
    NAMESPACE=$(echo "$output" | grep "^namespace/" | head -1 | sed 's/namespace\///' | sed 's/ .*//')

    if [[ -z "$NAMESPACE" ]]; then
        log_error "Failed to extract namespace from deployment output"
        exit 1
    fi

    log_success "Benchmark deployed in namespace: ${NAMESPACE}"
}

#######################################
# Wait for JobSet completion
#######################################
wait_for_completion() {
    log_step "Monitoring benchmark (namespace: ${NAMESPACE})"

    local start_time
    start_time=$(date +%s)
    local jobset_name
    jobset_name=$(kubectl get jobset -n "${NAMESPACE}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [[ -z "$jobset_name" ]]; then
        log_error "No JobSet found in namespace ${NAMESPACE}"
        exit 1
    fi

    log_info "JobSet: ${jobset_name}"
    log_info "Waiting for completion (timeout: ${TIMEOUT}s)..."

    while true; do
        local elapsed
        elapsed=$(($(date +%s) - start_time))

        if [[ $elapsed -gt $TIMEOUT ]]; then
            log_error "Timeout waiting for JobSet completion"
            show_debug_info
            exit 1
        fi

        # Get JobSet status
        local terminal_state
        terminal_state=$(kubectl get jobset "${jobset_name}" -n "${NAMESPACE}" -o jsonpath='{.status.terminalState}' 2>/dev/null || echo "")

        # Get pod status summary
        local pod_status
        pod_status=$(kubectl get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | awk '{print $3}' | sort | uniq -c | tr '\n' ' ')

        printf "\r[%3ds] JobSet: %-12s | Pods: %s" "$elapsed" "${terminal_state:-Running}" "${pod_status}"

        if [[ "$terminal_state" == "Completed" ]]; then
            echo ""
            log_success "JobSet completed successfully!"
            return 0
        elif [[ "$terminal_state" == "Failed" ]]; then
            echo ""
            log_error "JobSet failed!"
            show_debug_info
            exit 1
        fi

        sleep "${POLL_INTERVAL}"
    done
}

#######################################
# Show debug info on failure
#######################################
show_debug_info() {
    log_warn "Debug information:"
    echo ""
    echo "=== Pods ==="
    kubectl get pods -n "${NAMESPACE}" -o wide
    echo ""
    echo "=== Events ==="
    kubectl get events -n "${NAMESPACE}" --sort-by='.lastTimestamp' | tail -20
    echo ""
    echo "=== Container statuses ==="
    kubectl get pods -n "${NAMESPACE}" -o jsonpath='{range .items[*]}{.metadata.name}:{"\n"}{range .status.containerStatuses[*]}  {.name}: {.state}{"\n"}{end}{end}'
}

#######################################
# Extract and display metrics
#######################################
show_metrics() {
    log_step "Extracting benchmark metrics"

    local controller_pod
    controller_pod=$(kubectl get pods -n "${NAMESPACE}" -l job-name -o name 2>/dev/null | grep controller | head -1 | sed 's/pod\///')

    if [[ -z "$controller_pod" ]]; then
        log_error "Could not find controller pod"
        return 1
    fi

    log_info "Reading metrics from ${controller_pod}"

    # Extract the metrics table from logs
    local logs
    logs=$(kubectl logs -n "${NAMESPACE}" "${controller_pod}" -c system-controller 2>&1)

    # Extract clean metrics (strip ANSI codes and find the table)
    local metrics
    metrics=$(echo "$logs" | sed 's/\x1b\[[0-9;]*m//g' | grep -A 20 "NVIDIA AIPerf | LLM Metrics" | head -25)

    if [[ -n "$metrics" ]]; then
        echo ""
        echo "$metrics"
        echo ""
    fi

    # Extract key metrics for summary (using │ as field separator from the table)
    local clean_logs
    clean_logs=$(echo "$logs" | sed 's/\x1b\[[0-9;]*m//g')

    # Extract values from the metrics table - values are after the metric name and │
    local request_throughput output_throughput request_count latency_avg
    request_throughput=$(echo "$clean_logs" | grep "Request Throughput" | grep -v "Output" | head -1 | sed 's/.*│[[:space:]]*//' | awk '{print $1}')
    output_throughput=$(echo "$clean_logs" | grep "Output Token Throughput" | head -1 | sed 's/.*│[[:space:]]*//' | awk '{print $1}')
    request_count=$(echo "$clean_logs" | grep "Request Count" | head -1 | sed 's/.*│[[:space:]]*//' | awk '{print $1}')
    latency_avg=$(echo "$clean_logs" | grep "Request Latency" | head -1 | sed 's/.*│[[:space:]]*//' | awk '{print $1}')

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Request Throughput:      ${request_throughput:-N/A} req/s"
    echo "  Output Token Throughput: ${output_throughput:-N/A} tokens/s"
    echo "  Request Count:           ${request_count:-N/A} requests"
    echo "  Avg Request Latency:     ${latency_avg:-N/A} ms"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    log_success "Metrics extracted successfully"
}

#######################################
# Cleanup
#######################################
cleanup() {
    if [[ "$CLEANUP" == "true" ]]; then
        log_step "Cleaning up"
        minikube delete -p "${CLUSTER_NAME}" 2>/dev/null || true
        log_success "Cluster ${CLUSTER_NAME} deleted"
    else
        log_warn "Skipping cleanup (--no-cleanup specified)"
        log_info "To delete cluster manually: minikube delete -p ${CLUSTER_NAME}"
    fi
}

#######################################
# Main
#######################################
main() {
    parse_args "$@"

    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║          AIPerf Kubernetes End-to-End Test                 ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    # Set up cleanup trap
    trap cleanup EXIT

    check_prerequisites
    build_images
    create_cluster
    load_images
    install_jobset
    deploy_mock_server
    deploy_benchmark
    wait_for_completion
    show_metrics

    echo ""
    log_success "═══════════════════════════════════════════════════════════"
    log_success "  All tests passed! AIPerf Kubernetes deployment working."
    log_success "═══════════════════════════════════════════════════════════"
    echo ""
}

main "$@"
