#!/bin/bash
# run_experiment.sh — Full two-node experiment on Sophia ALCF.
#
# Reads NODE1 and NODE2 from $PBS_NODEFILE automatically, starts all
# services, runs all load tests and baselines, then generates comparison
# reports. Results are saved to results/experiment_<timestamp>/.
#
# Usage (inside a PBS interactive job):
#   qsub -I -l select=2:ngpus=8:ncpus=64 -l walltime=08:00:00 \
#        -l filesystems=home:eagle -A UIC-HPC -q by-node
#   cd ~/semantic-llm-router
#   bash scripts/run_experiment.sh
#
# Optional env overrides:
#   ROUTER_PORT=8080 N_REQUESTS=1000 CONCURRENCY=50 bash scripts/run_experiment.sh

set -euo pipefail

# -- Configuration ------------------------------------------------------------
ROUTER_PORT=${ROUTER_PORT:-8080}
PRIORS_FILE=${PRIORS_FILE:-"results/priors_all5.json"}
DATASET=${DATASET:-"datasets/hf_1000.json"}
N_REQUESTS=${N_REQUESTS:-1000}
CONCURRENCY=${CONCURRENCY:-50}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-30}
LOG_DIR="$HOME/vllm_logs"
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/experiment_${TS}"
mkdir -p "$RESULTS_DIR"

# -- Discover nodes from PBS --------------------------------------------------
if [ -z "${PBS_NODEFILE:-}" ]; then
    echo "ERROR: PBS_NODEFILE not set. Run inside a PBS job:"
    echo "  qsub -I -l select=2:ngpus=8:ncpus=64 -l walltime=08:00:00 \\"
    echo "       -l filesystems=home:eagle -A UIC-HPC -q by-node"
    exit 1
fi

NODES=($(sort -u "$PBS_NODEFILE"))
NODE1=${NODES[0]}
NODE2=${NODES[1]}
ROUTER_URL="http://${NODE1}:${ROUTER_PORT}"

echo "=================================================================="
echo "  Two-Node Experiment  $(date)"
echo "  NODE1 : $NODE1  (qwen-7b / deepseek-r1-7b / coder-32b / gemma-3-27b / deepseek-r1-14b)"
echo "  NODE2 : $NODE2  (llama4-scout, 8 GPUs)"
echo "  Router: $ROUTER_URL"
echo "  Output: $RESULTS_DIR"
echo "=================================================================="

# -- Helper: wait for router --------------------------------------------------
wait_router() {
    echo "  Waiting for router at $ROUTER_URL..."
    for i in $(seq 1 60); do
        if curl --noproxy '*' -sf "$ROUTER_URL/router/health" > /dev/null 2>&1; then
            echo "  Router ready!"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Router not ready after 5 minutes"
    exit 1
}

# -- Helper: wait for N registered models ------------------------------------
wait_models() {
    local EXPECTED=$1
    echo "  Waiting for $EXPECTED models to register..."
    for i in $(seq 1 240); do
        local N
        N=$(curl --noproxy '*' -sf "$ROUTER_URL/v1/models" \
            | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo 0)
        echo "  [$((i*15))s] Registered: $N / $EXPECTED"
        if [ "$N" -ge "$EXPECTED" ]; then
            echo "  All $EXPECTED models ready!"
            return 0
        fi
        sleep 15
    done
    echo "WARNING: Only $N/$EXPECTED models after 60 min -- continuing anyway"
}

# -- [1/10] Router ------------------------------------------------------------
echo ""
echo "[1/10] Starting router on $NODE1..."
nohup uvicorn semantic_router.main:app \
    --host 0.0.0.0 --port "$ROUTER_PORT" \
    > "$LOG_DIR/router.log" 2>&1 &
echo "  PID: $!  log: $LOG_DIR/router.log"
sleep 5
wait_router

# -- [2/10] Node 1 provisioner ------------------------------------------------
echo ""
echo "[2/10] Starting provisioner on $NODE1 (5 models)..."
nohup python provisioner/dynamic_provisioner.py \
    --router-url   "$ROUTER_URL" \
    --node-host    "$NODE1" \
    --router-mode  ttca \
    --static \
    --priors-path  "$PRIORS_FILE" \
    --initial-models qwen-7b,deepseek-r1-7b,coder-32b,gemma-3-27b,deepseek-r1-14b \
    > "$LOG_DIR/provisioner_node1.log" 2>&1 &
echo "  PID: $!  log: $LOG_DIR/provisioner_node1.log"

# -- [3/10] Node 2 provisioner (llama4-scout, 8 GPUs) ------------------------
echo ""
echo "[3/10] Starting provisioner on $NODE2 (llama4-scout, 8 GPUs)..."
# shellcheck disable=SC2029
ssh "$NODE2" "cd ~/semantic-llm-router && \
    export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache && \
    nohup python provisioner/dynamic_provisioner.py \
        --router-url   '$ROUTER_URL' \
        --node-host    '$NODE2' \
        --router-mode  ttca \
        --static \
        --priors-path  '$PRIORS_FILE' \
        --initial-models llama4-scout \
        > '$LOG_DIR/provisioner_node2.log' 2>&1 & echo PID:\$!"
echo "  log: $LOG_DIR/provisioner_node2.log (on $NODE2)"

# -- [4/10] Wait for all 6 models + seed with placeholder priors -------------
echo ""
echo "[4/10] Waiting for all 6 models..."
wait_models 6

echo "  Seeding models with placeholder priors (will be replaced after eval)..."
python tests/register_with_priors.py \
    --priors     "$PRIORS_FILE" \
    --router-url "$ROUTER_URL" \
    --node2-host "$NODE2"

echo "  Model list:"
curl --noproxy '*' -sf "$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; [print(f'    {m[\"id\"]}') for m in json.load(sys.stdin)['data']]"

# -- [5/10] Eval matrix: send all queries to all 6 models --------------------
# This step collects ground truth correctness for every (query, model) pair.
# Ground truth comes from datasets/hf_1000.json (HuggingFace benchmarks).
echo ""
echo "[5/10] Building eval matrix (6 models x 1000 queries, ~30-45 min)..."
echo "  Ground truth is in $DATASET — no manual labelling needed."
python tests/eval_all_models.py \
    --dataset     "$DATASET" \
    --output      "$RESULTS_DIR/eval_matrix.csv" \
    --concurrency "$EVAL_CONCURRENCY" \
    --node2-host  "$NODE2"

# -- [6/10] Extract real accuracy priors from eval matrix --------------------
echo ""
echo "[6/10] Extracting real accuracy priors from eval results..."
python tests/extract_priors.py \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --output      "$RESULTS_DIR/priors_new.json"

echo "  Measured accuracy per model:"
python3 -c "
import json
p = json.load(open('$RESULTS_DIR/priors_new.json'))
for model, priors in sorted(p.items()):
    if not priors: continue
    keys = sorted(priors.keys())
    avg  = sum(priors.values()) / len(priors)
    print(f'    {model:<24} {len(priors):2} keys  avg={avg:.3f}')
"

# -- [7/10] Re-register all models with real measured priors -----------------
echo ""
echo "[7/10] Re-registering all models with real accuracy priors..."
python tests/register_with_priors.py \
    --priors     "$RESULTS_DIR/priors_new.json" \
    --router-url "$ROUTER_URL" \
    --node2-host "$NODE2"
echo "  Router now uses measured accuracy — routing decisions are calibrated."

# -- [8/10] TTCA load test ----------------------------------------------------
echo ""
echo "[8/10] TTCA router load test (${N_REQUESTS} requests, real priors active)..."
python tests/load_test.py \
    --dataset     "$DATASET" \
    --router      "$ROUTER_URL" \
    --mode        ttca \
    --requests    "$N_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --output      "$RESULTS_DIR/router_ttca.csv"

# -- [9/10] Accuracy load test ------------------------------------------------
echo ""
echo "[9/10] Accuracy router load test (${N_REQUESTS} requests)..."
python tests/load_test.py \
    --dataset     "$DATASET" \
    --router      "$ROUTER_URL" \
    --mode        accuracy \
    --requests    "$N_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --output      "$RESULTS_DIR/router_accuracy.csv"

# -- [10/10] Round-robin baseline + comparisons -------------------------------
echo ""
echo "[10/10] Round-robin baseline (${N_REQUESTS} requests)..."
python tests/round_robin_test.py \
    --dataset     "$DATASET" \
    --output      "$RESULTS_DIR/rr_baseline.csv" \
    --concurrency "$CONCURRENCY" \
    --node2-host  "$NODE2"

echo ""
echo "=== TTCA router vs Round-Robin ===" | tee "$RESULTS_DIR/compare_ttca_vs_rr.txt"
python tests/compare_ttca.py \
    --router      "$RESULTS_DIR/router_ttca.csv" \
    --baseline    "$RESULTS_DIR/rr_baseline.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    | tee -a "$RESULTS_DIR/compare_ttca_vs_rr.txt"

echo ""
echo "=== Accuracy router vs Round-Robin ===" | tee "$RESULTS_DIR/compare_accuracy_vs_rr.txt"
python tests/compare_ttca.py \
    --router      "$RESULTS_DIR/router_accuracy.csv" \
    --baseline    "$RESULTS_DIR/rr_baseline.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    | tee -a "$RESULTS_DIR/compare_accuracy_vs_rr.txt"

echo ""
echo "=== TTCA vs Accuracy router ===" | tee "$RESULTS_DIR/compare_ttca_vs_accuracy.txt"
python tests/compare_ttca.py \
    --router      "$RESULTS_DIR/router_ttca.csv" \
    --baseline    "$RESULTS_DIR/router_accuracy.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    | tee -a "$RESULTS_DIR/compare_ttca_vs_accuracy.txt"

# -- Done ---------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "  Experiment complete!  $(date)"
echo "  Results in: $RESULTS_DIR/"
echo "    eval_matrix.csv              (ground truth correctness)"
echo "    priors_new.json              (real measured accuracy priors)"
echo "    router_ttca.csv              (TTCA router results)"
echo "    router_accuracy.csv          (accuracy router results)"
echo "    rr_baseline.csv              (round-robin baseline)"
echo "    compare_ttca_vs_rr.txt       (TTCA router vs baseline)"
echo "    compare_accuracy_vs_rr.txt   (accuracy router vs baseline)"
echo "    compare_ttca_vs_accuracy.txt (TTCA vs accuracy)"
echo "=================================================================="
