#!/bin/bash
# test_baselines.sh -- Quick smoke test for all four baselines.
#
# Usage:
#   bash scripts/test_baselines.sh
#   N=50 bash scripts/test_baselines.sh
#   NODE2=sophia-gpu-09 bash scripts/test_baselines.sh

set -euo pipefail

N=${N:-20}
CONCURRENCY=${CONCURRENCY:-10}
ROUTER_PORT=${ROUTER_PORT:-8080}
NODE1=${NODE1:-localhost}
NODE2=${NODE2:-""}
DATASET=${DATASET:-"datasets/hf_1500.json"}
PRIORS=${PRIORS:-"results/priors_new.json"}
ROUTER_URL="http://${NODE1}:${ROUTER_PORT}"
VLLM_SR_URL="http://${NODE1}:8888"
SKIP_VLLM_SR=${SKIP_VLLM_SR:-0}

PASS=0
FAIL=0
SKIP=0

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/test_baselines_${TS}"
mkdir -p "$OUT_DIR"

ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
skip() { echo "  [SKIP] $*"; SKIP=$((SKIP+1)); }
sep()  { echo ""; echo "-- $* ----------------------------------------"; }

check_csv() {
    local file=$1 min_rows=${2:-2}
    if [ ! -f "$file" ]; then
        fail "output file not created: $file"; return 1
    fi
    local rows
    rows=$(tail -n +2 "$file" | wc -l | tr -d ' ')
    if [ "$rows" -lt "$min_rows" ]; then
        fail "$file has only $rows rows (expected >= $min_rows)"; return 1
    fi
    ok "$file  ($rows rows)"
    return 0
}

echo ""
echo "=================================================================="
echo "  Baseline Smoke Test  $(date)"
echo "  N requests per baseline : $N"
echo "  Dataset                 : $DATASET"
echo "  Priors                  : $PRIORS"
echo "  Output dir              : $OUT_DIR/"
echo "=================================================================="

sep "Pre-flight checks"

if [ ! -f "$DATASET" ]; then
    echo "  Dataset not found: $DATASET"
    echo "    Run: python tests/build_dataset.py"
    exit 1
fi
echo "  Dataset : $DATASET"

if curl --noproxy '*' -sf "$ROUTER_URL/router/health" > /dev/null 2>&1; then
    ok "Router reachable at $ROUTER_URL"
else
    fail "Router NOT reachable at $ROUTER_URL"
    exit 1
fi

for port in 8000 8001 8002 8003 8004; do
    if curl --noproxy '*' -sf "http://${NODE1}:${port}/health" > /dev/null 2>&1; then
        ok "Backend port $port reachable"
    else
        fail "Backend port $port NOT reachable on $NODE1"
    fi
done

if [ -n "$NODE2" ]; then
    if curl --noproxy '*' -sf "http://${NODE2}:8005/health" > /dev/null 2>&1; then
        ok "llama4-scout (port 8005) reachable on $NODE2"
    else
        fail "llama4-scout NOT reachable on $NODE2:8005"
    fi
fi

VLLM_SR_UP=0
if [ "$SKIP_VLLM_SR" -eq 0 ]; then
    if curl --noproxy '*' -sf "$VLLM_SR_URL/health" > /dev/null 2>&1 || \
       curl --noproxy '*' -sf "$VLLM_SR_URL/v1/models" > /dev/null 2>&1; then
        ok "vllm-sr reachable at $VLLM_SR_URL"
        VLLM_SR_UP=1
    else
        skip "vllm-sr NOT running at $VLLM_SR_URL (set SKIP_VLLM_SR=1 to suppress)"
    fi
else
    skip "vllm-sr check skipped (SKIP_VLLM_SR=1)"
fi

sep "Baseline 1 -- TTCA single-shot (no retry)"
echo "  $N requests -> $ROUTER_URL  [mode=ttca --no-retry]"
python tests/load_test.py \
    --dataset     "$DATASET" \
    --router      "$ROUTER_URL" \
    --mode        ttca \
    --no-retry \
    --requests    "$N" \
    --concurrency "$CONCURRENCY" \
    --output      "$OUT_DIR/ttca_no_retry.csv" 2>&1 | tail -5
check_csv "$OUT_DIR/ttca_no_retry.csv" "$N"

sep "Baseline 2 -- Complexity-tier routing"
echo "  $N requests, deterministic (domain,complexity) -> model"
NODE2_FLAG=""
[ -n "$NODE2" ] && NODE2_FLAG="--node2-host $NODE2"
python tests/baseline_complexity_tier.py \
    --dataset     "$DATASET" \
    --concurrency "$CONCURRENCY" \
    $NODE2_FLAG \
    --output      "$OUT_DIR/baseline_tier.csv" 2>&1 | tail -5
check_csv "$OUT_DIR/baseline_tier.csv" "$N"

sep "Baseline 3 -- Cascade / RouteLLM-style"
if [ ! -f "$PRIORS" ]; then
    skip "Priors file not found: $PRIORS -- cascade baseline skipped"
else
    echo "  $N requests, weak=qwen-7b / strong=deepseek-r1-14b, threshold=0.80"
    python tests/baseline_cascade.py \
        --dataset     "$DATASET" \
        --priors      "$PRIORS" \
        --threshold   0.80 \
        --concurrency "$CONCURRENCY" \
        --output      "$OUT_DIR/baseline_cascade.csv" 2>&1 | tail -5
    check_csv "$OUT_DIR/baseline_cascade.csv" "$N"
fi

sep "Baseline 4 -- vLLM Semantic Router"
if [ "$VLLM_SR_UP" -eq 1 ]; then
    echo "  $N requests -> $VLLM_SR_URL  [model=MoM]"
    python tests/baseline_vllm_router.py \
        --dataset     "$DATASET" \
        --endpoint    "$VLLM_SR_URL" \
        --concurrency "$CONCURRENCY" \
        --output      "$OUT_DIR/baseline_vllm_sr.csv" 2>&1 | tail -5
    check_csv "$OUT_DIR/baseline_vllm_sr.csv" "$N"
else
    skip "vllm-sr not running -- baseline 4 skipped"
fi

echo ""
echo "=================================================================="
echo "  Smoke Test Complete  $(date)"
echo "  PASS: $PASS   FAIL: $FAIL   SKIP: $SKIP"
echo "  Results: $OUT_DIR/"
echo "=================================================================="

if [ "$FAIL" -gt 0 ]; then
    echo "  $FAIL check(s) failed."
    exit 1
fi
echo ""
echo "  All checks passed. Submit with:"
echo "    bash scripts/submit.sh"
echo "    bash scripts/submit_baselines.sh"
