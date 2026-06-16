#!/bin/bash
# run_experiment.sh -- Full two-node experiment on Sophia ALCF.
#
# Usage (inside a PBS interactive job):
#   cd ~/semantic-llm-router
#   bash scripts/run_experiment.sh
#
# Optional env overrides:
#   ROUTER_PORT=8080 N_REQUESTS=1000 CONCURRENCY=50 bash scripts/run_experiment.sh

set -euo pipefail

ROUTER_PORT=${ROUTER_PORT:-8080}
PRIORS_FILE=${PRIORS_FILE:-"results/priors_all5.json"}
DATASET=${DATASET:-"datasets/hf_1500.json"}
N_REQUESTS=${N_REQUESTS:-1500}
CONCURRENCY=${CONCURRENCY:-50}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-30}
TTCA_ALPHA=${TTCA_ALPHA:-1.0}
TTCA_COST_BETA=${TTCA_COST_BETA:-0.0}
EXPERIMENT_MODE=${EXPERIMENT_MODE:-static}
LOG_DIR="$HOME/vllm_logs"
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/experiment_${TS}_alpha${TTCA_ALPHA}_beta${TTCA_COST_BETA}_${EXPERIMENT_MODE}"
mkdir -p "$RESULTS_DIR"

if [ -z "${PBS_NODEFILE:-}" ]; then
    echo "ERROR: PBS_NODEFILE not set. Run inside a PBS job."
    exit 1
fi

NODES=($(sort -u "$PBS_NODEFILE"))
NODE1=${NODES[0]}
NODE2=${NODES[1]}
ROUTER_URL="http://${NODE1}:${ROUTER_PORT}"

echo "=================================================================="
echo "  Two-Node Experiment  $(date)"
echo "  NODE1 : $NODE1"
echo "  NODE2 : $NODE2"
echo "  Router: $ROUTER_URL"
echo "  TTCA_ALPHA: $TTCA_ALPHA  TTCA_COST_BETA: $TTCA_COST_BETA"
echo "  MODE  : $EXPERIMENT_MODE"
echo "  Output: $RESULTS_DIR"
echo "=================================================================="

echo "  Setting TTCA_ALPHA=$TTCA_ALPHA TTCA_COST_BETA=$TTCA_COST_BETA in semantic_router/config.py..."
sed -i "s/^TTCA_ALPHA: float = .*/TTCA_ALPHA: float = ${TTCA_ALPHA}/" semantic_router/config.py
sed -i "s/^TTCA_COST_BETA: float = .*/TTCA_COST_BETA: float = ${TTCA_COST_BETA}/" semantic_router/config.py
grep "TTCA_" semantic_router/config.py

wait_router() {
    echo "  Waiting for router at $ROUTER_URL..."
    for i in $(seq 1 60); do
        if curl --noproxy '*' -sf "$ROUTER_URL/router/health" > /dev/null 2>&1; then
            echo "  Router ready!"; return 0
        fi
        sleep 5
    done
    echo "ERROR: Router not ready after 5 minutes"; exit 1
}

wait_models() {
    local EXPECTED=$1
    echo "  Waiting for $EXPECTED models to register..."
    for i in $(seq 1 240); do
        local N
        N=$(curl --noproxy '*' -sf "$ROUTER_URL/v1/models" \
            | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo 0)
        echo "  [$((i*15))s] Registered: $N / $EXPECTED"
        if [ "$N" -ge "$EXPECTED" ]; then
            echo "  All $EXPECTED models ready!"; return 0
        fi
        sleep 15
    done
    echo "WARNING: Only $N/$EXPECTED models registered after 60 min -- continuing anyway"
}

echo ""
echo "[1/8] Starting router on $NODE1..."
nohup uvicorn semantic_router.main:app \
    --host 0.0.0.0 --port "$ROUTER_PORT" \
    > "$LOG_DIR/router.log" 2>&1 &
ROUTER_PID=$!
echo "  PID: $ROUTER_PID  log: $LOG_DIR/router.log"
sleep 5
wait_router

echo ""
if [ "$EXPERIMENT_MODE" = "dynamic" ]; then
    echo "[2/8] Starting provisioner on $NODE1 (dynamic mode)..."
    PROV_FLAGS="--router-mode ttca"
    PROV_INITIAL="qwen-7b,deepseek-r1-7b"
else
    echo "[2/8] Starting provisioner on $NODE1 (static mode)..."
    PROV_FLAGS="--router-mode ttca --static"
    PROV_INITIAL="qwen-7b,deepseek-r1-7b,qwen3-coder-30b,gemma-3-27b,deepseek-r1-14b"
fi
nohup python provisioner/dynamic_provisioner.py \
    --router-url   "$ROUTER_URL" \
    --node-host    "$NODE1" \
    $PROV_FLAGS \
    --priors-path  "$PRIORS_FILE" \
    --initial-models "$PROV_INITIAL" \
    > "$LOG_DIR/provisioner_node1.log" 2>&1 &
PROV1_PID=$!
echo "  PID: $PROV1_PID"

echo ""
echo "[3/8] Starting provisioner on $NODE2 (llama4-scout, 8 GPUs)..."
# shellcheck disable=SC2029
ssh "$NODE2" "
    cd ~/semantic-llm-router
    export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
    python provisioner/dynamic_provisioner.py \
        --router-url   '$ROUTER_URL' \
        --node-host    '$NODE2' \
        --router-mode  ttca \
        --static \
        --priors-path  '$PRIORS_FILE' \
        --initial-models llama4-scout \
        </dev/null>>'$LOG_DIR/provisioner_node2.log' 2>&1 &
    BGPID=\$!
    disown \$BGPID
    echo PID:\$BGPID
" < /dev/null

echo ""
if [ "$EXPERIMENT_MODE" = "dynamic" ]; then
    echo "[4/8] Waiting for seed models..."
    wait_models 3
else
    echo "[4/8] Waiting for all 6 models (static mode)..."
    wait_models 6
fi

echo "  Registering node-1 models with priors..."
python tests/register_with_priors.py \
    --priors     "$PRIORS_FILE" \
    --router-url "$ROUTER_URL"

echo "  Registering llama4-scout (node 2)..."
curl --noproxy '*' -sf -X POST "$ROUTER_URL/router/register" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_id\": \"llama4-scout\",
    \"model_name\": \"meta-llama/Llama-4-Scout-17B-16E-Instruct\",
    \"backend\": \"vllm\",
    \"base_url\": \"http://$NODE2:8005\",
    \"domains\": [\"factual\",\"reasoning\",\"creative\",\"math\",\"code\"],
    \"min_accuracy_capability\": {\"_default\": 0.88},
    \"accuracy_priors\": {
      \"factual:easy\": 0.97, \"factual:medium\": 0.96, \"factual:hard\": 0.94,
      \"reasoning:easy\": 0.98, \"reasoning:hard\": 0.96,
      \"math:easy\": 0.95, \"math:medium\": 0.93, \"math:hard\": 0.88,
      \"code:easy\": 0.96, \"code:medium\": 0.93, \"code:hard\": 0.88,
      \"creative:easy\": 0.95, \"creative:medium\": 0.93, \"creative:hard\": 0.91
    },
    \"decode_tokens_per_sec\": 500,
    \"skip_calibration\": true
  }" > /dev/null

echo "  Final model list:"
curl --noproxy '*' -sf "$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; [print(f'    {m[\"id\"]}') for m in json.load(sys.stdin)['data']]"

echo ""
echo "[5/10] Building eval matrix (6 models x dataset, ~30-45 min)..."
python tests/eval_all_models.py \
    --dataset     "$DATASET" \
    --output      "$RESULTS_DIR/eval_matrix.csv" \
    --concurrency "$EVAL_CONCURRENCY" \
    --node2-host  "$NODE2"

echo ""
echo "[6/10] Extracting real accuracy priors from eval matrix..."
python tests/extract_priors.py \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --output      "$RESULTS_DIR/priors_new.json"

echo ""
echo "[7/10] Re-registering all models with real accuracy priors..."
python tests/register_with_priors.py \
    --priors     "$RESULTS_DIR/priors_new.json" \
    --router-url "$ROUTER_URL" \
    --node2-host "$NODE2"

echo ""
echo "[8/10] TTCA router load test..."
python tests/load_test.py \
    --dataset     "$DATASET" \
    --router      "$ROUTER_URL" \
    --mode        ttca \
    --requests    "$N_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --output      "$RESULTS_DIR/router_ttca.csv"

echo ""
echo "[9/10] Accuracy router load test..."
python tests/load_test.py \
    --dataset     "$DATASET" \
    --router      "$ROUTER_URL" \
    --mode        accuracy \
    --requests    "$N_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --output      "$RESULTS_DIR/router_accuracy.csv"

echo ""
echo "[10/10] Round-robin baseline..."
python tests/round_robin_test.py \
    --dataset     "$DATASET" \
    --output      "$RESULTS_DIR/rr_baseline.csv" \
    --concurrency "$CONCURRENCY" \
    --node2-host  "$NODE2"

# Baseline comparisons
echo ""
echo "[Baseline 1/4] TTCA single-shot (no retry)..."
python tests/load_test.py \
    --dataset     "$DATASET" \
    --router      "$ROUTER_URL" \
    --mode        ttca \
    --no-retry \
    --requests    "$N_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --output      "$RESULTS_DIR/ttca_no_retry.csv"

echo ""
echo "[Baseline 2/4] Complexity-tier routing..."
python tests/baseline_complexity_tier.py \
    --dataset     "$DATASET" \
    --concurrency "$CONCURRENCY" \
    --node2-host  "$NODE2" \
    --output      "$RESULTS_DIR/baseline_tier.csv"

echo ""
echo "[Baseline 3/4] Cascade routing / RouteLLM-style..."
python tests/baseline_cascade.py \
    --dataset     "$DATASET" \
    --priors      "$RESULTS_DIR/priors_new.json" \
    --threshold   0.80 \
    --concurrency "$CONCURRENCY" \
    --output      "$RESULTS_DIR/baseline_cascade.csv"

echo ""
echo "[Baseline 4/4] vLLM Semantic Router (if running on port 8888)..."
if curl --noproxy '*' -sf "http://${NODE1}:8888/health" > /dev/null 2>&1 || \
   curl --noproxy '*' -sf "http://${NODE1}:8888/v1/models" > /dev/null 2>&1; then
    python tests/baseline_vllm_router.py \
        --dataset     "$DATASET" \
        --endpoint    "http://${NODE1}:8888" \
        --concurrency "$CONCURRENCY" \
        --output      "$RESULTS_DIR/baseline_vllm_sr.csv"
else
    echo "  vllm-sr not running on port 8888 -- skipping."
fi

# Comparison reports
echo ""
echo "=== TTCA (with retry) vs TTCA (no retry) ==="
python tests/compare_ttca.py \
    --router      "$RESULTS_DIR/router_ttca.csv" \
    --baseline    "$RESULTS_DIR/ttca_no_retry.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    | tee "$RESULTS_DIR/compare_retry_benefit.txt"

echo ""
echo "=== TTCA (no retry) vs Complexity-Tier ==="
python tests/compare_categories.py \
    --router      "$RESULTS_DIR/ttca_no_retry.csv" \
    --baseline    "$RESULTS_DIR/baseline_tier.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --output      "$RESULTS_DIR/compare_vs_tier.csv" \
    | tee "$RESULTS_DIR/compare_vs_tier.txt"

echo ""
echo "=== TTCA (no retry) vs Cascade/RouteLLM ==="
python tests/compare_categories.py \
    --router      "$RESULTS_DIR/ttca_no_retry.csv" \
    --baseline    "$RESULTS_DIR/baseline_cascade.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --output      "$RESULTS_DIR/compare_vs_cascade.csv" \
    | tee "$RESULTS_DIR/compare_vs_cascade.txt"

if [ -f "$RESULTS_DIR/baseline_vllm_sr.csv" ]; then
    echo ""
    echo "=== TTCA (no retry) vs vLLM Semantic Router ==="
    python tests/compare_categories.py \
        --router      "$RESULTS_DIR/ttca_no_retry.csv" \
        --baseline    "$RESULTS_DIR/baseline_vllm_sr.csv" \
        --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
        --output      "$RESULTS_DIR/compare_vs_vllm_sr.csv" \
        | tee "$RESULTS_DIR/compare_vs_vllm_sr.txt"
fi

echo ""
echo "=================================================================="
echo "  Experiment complete!  $(date)"
echo "  Results: $RESULTS_DIR/"
echo "=========================================================="
