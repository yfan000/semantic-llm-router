#!/bin/bash
# run_experiment_polaris.sh — Full four-node experiment on Polaris (ALCF).
#
# Polaris has 4 A100 GPUs per node (vs Sophia's 8). Four nodes are allocated:
#   NODE1      — router + provisioner + qwen-7b / deepseek-r1-7b / qwen3-coder-30b (4 GPUs)
#   NODE2      — gemma-3-27b (GPU 0-1, port 8003) + deepseek-r1-14b (GPU 2-3, port 8004)
#   SCOUT_NODE1 — llama4-scout Ray head (GPU 0-3)
#   SCOUT_NODE2 — llama4-scout Ray worker (GPU 0-3); Ray+vLLM TP=8 spans both SCOUT nodes
#
# Port layout (same logical roles as Sophia, port numbers preserved):
#   NODE1:8000  qwen-7b          (TP=1)
#   NODE1:8001  deepseek-r1-7b   (TP=1)
#   NODE1:8002  qwen3-coder-30b  (TP=2)
#   NODE2:8003  gemma-3-27b      (TP=2)
#   NODE2:8004  deepseek-r1-14b  (TP=2)
#   SCOUT_NODE1:8005  llama4-scout (TP=8 via Ray)
#
# Usage (inside a PBS interactive job):
#   qsub -I -l select=4:ngpus=4:ncpus=32 -l walltime=08:00:00 \
#        -l filesystems=home:eagle -A UIC-HPC -q prod
#   cd ~/semantic-llm-router
#   bash scripts/run_experiment_polaris.sh
#
# Optional env overrides:
#   ROUTER_PORT=8080 N_REQUESTS=1000 CONCURRENCY=50 bash scripts/run_experiment_polaris.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
ROUTER_PORT=${ROUTER_PORT:-8080}
PRIORS_FILE=${PRIORS_FILE:-"results/priors_all5.json"}
DATASET=${DATASET:-"datasets/hf_1500.json"}
N_REQUESTS=${N_REQUESTS:-1500}
CONCURRENCY=${CONCURRENCY:-50}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-30}
TTCA_ALPHA=${TTCA_ALPHA:-1.0}
TTCA_COST_BETA=${TTCA_COST_BETA:-0.0}
TTCA_ALPHA_FACTUAL=${TTCA_ALPHA_FACTUAL:-0.3}
TTCA_ALPHA_MATH=${TTCA_ALPHA_MATH:-0.7}
TTCA_ALPHA_CODE=${TTCA_ALPHA_CODE:-1.0}
TTCA_ALPHA_REASONING=${TTCA_ALPHA_REASONING:-0.7}
EXPERIMENT_MODE=${EXPERIMENT_MODE:-static}
LOG_DIR="$HOME/vllm_logs"
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/experiment_${TS}_alpha${TTCA_ALPHA}_beta${TTCA_COST_BETA}_${EXPERIMENT_MODE}"
mkdir -p "$RESULTS_DIR"

# ── Discover nodes from PBS ───────────────────────────────────────────────────
if [ -z "${PBS_NODEFILE:-}" ]; then
    echo "ERROR: PBS_NODEFILE not set. Run inside a PBS job:"
    echo "  qsub -I -l select=4:ngpus=4:ncpus=32 -l walltime=08:00:00 \\"
    echo "       -l filesystems=home:eagle -A UIC-HPC -q prod"
    exit 1
fi

NODES=($(sort -u "$PBS_NODEFILE"))
if [ "${#NODES[@]}" -lt 4 ]; then
    echo "ERROR: Expected 4 nodes, got ${#NODES[@]}: ${NODES[*]}"
    echo "  Submit with: select=4:ngpus=4:ncpus=32"
    exit 1
fi
NODE1=${NODES[0]}        # router + provisioner + 3 light models
NODE2=${NODES[1]}        # gemma-3-27b + deepseek-r1-14b (SSH-started)
SCOUT_NODE1=${NODES[2]}  # llama4-scout Ray head
SCOUT_NODE2=${NODES[3]}  # llama4-scout Ray worker
ROUTER_URL="http://${NODE1}:${ROUTER_PORT}"

echo "=================================================================="
echo "  Four-Node Experiment (Polaris)  $(date)"
echo "  NODE1       : $NODE1  (router + qwen-7b / deepseek-r1-7b / qwen3-coder-30b)"
echo "  NODE2       : $NODE2  (gemma-3-27b / deepseek-r1-14b)"
echo "  SCOUT_NODE1 : $SCOUT_NODE1  (llama4-scout Ray head, 4+4 GPUs via TP=8)"
echo "  SCOUT_NODE2 : $SCOUT_NODE2  (llama4-scout Ray worker)"
echo "  Router      : $ROUTER_URL"
echo "  TTCA_ALPHA  : $TTCA_ALPHA  TTCA_COST_BETA: $TTCA_COST_BETA"
echo "  MODE        : $EXPERIMENT_MODE  (static=all-preloaded | dynamic=cold-spinup)"
echo "  Output      : $RESULTS_DIR"
echo "=================================================================="

# Patch TTCA_ALPHA and TTCA_COST_BETA into config.py so the router uses the requested values.
echo "  Setting TTCA parameters in semantic_router/config.py..."
sed -i "s/^TTCA_ALPHA: float = .*/TTCA_ALPHA: float = ${TTCA_ALPHA}/" semantic_router/config.py
sed -i "s/^TTCA_COST_BETA: float = .*/TTCA_COST_BETA: float = ${TTCA_COST_BETA}/" semantic_router/config.py
sed -i "s/^TTCA_ALPHA_FACTUAL: float = .*/TTCA_ALPHA_FACTUAL: float = ${TTCA_ALPHA_FACTUAL}/" semantic_router/config.py
sed -i "s/^TTCA_ALPHA_MATH: float = .*/TTCA_ALPHA_MATH: float = ${TTCA_ALPHA_MATH}/" semantic_router/config.py
sed -i "s/^TTCA_ALPHA_CODE: float = .*/TTCA_ALPHA_CODE: float = ${TTCA_ALPHA_CODE}/" semantic_router/config.py
sed -i "s/^TTCA_ALPHA_REASONING: float = .*/TTCA_ALPHA_REASONING: float = ${TTCA_ALPHA_REASONING}/" semantic_router/config.py
grep "TTCA_" semantic_router/config.py

# ── Helper: wait for router ───────────────────────────────────────────────────
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

# ── Helper: wait for N models to register with the router ─────────────────────
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
    echo "WARNING: Only $N/$EXPECTED models registered after 60 min — continuing anyway"
}

# ── Helper: wait for a vLLM service /health endpoint ─────────────────────────
wait_vllm_health() {
    local HOST_PORT="$1"
    local LABEL="${2:-$HOST_PORT}"
    echo "  Waiting for $LABEL at $HOST_PORT..."
    for i in $(seq 1 120); do
        if curl --noproxy '*' -sf "http://${HOST_PORT}/health" > /dev/null 2>&1; then
            echo "  $LABEL ready! (after $((i*30))s)"
            return 0
        fi
        if [ $((i % 4)) -eq 0 ]; then
            echo "  [$((i*30))s] Still waiting for $LABEL..."
        fi
        sleep 30
    done
    echo "WARNING: $LABEL not ready after 60 min — attempting to continue"
}

# ── Cleanup trap: stop Ray cluster on exit ────────────────────────────────────
cleanup_ray() {
    [ -n "${SCOUT_NODE1:-}" ] || return 0
    [ -n "${SCOUT_NODE2:-}" ] || return 0
    echo "  Stopping Ray cluster on $SCOUT_NODE1 and $SCOUT_NODE2..."
    ssh "$SCOUT_NODE2" "
        VLLM_ENV=\"\$HOME/.conda/envs/2026-06-08/vllm_env\"
        export PATH=\"\${VLLM_ENV}/bin:\$PATH\"
        ray stop --force </dev/null >/dev/null 2>&1 || true
    " < /dev/null 2>/dev/null || true
    ssh "$SCOUT_NODE1" "
        VLLM_ENV=\"\$HOME/.conda/envs/2026-06-08/vllm_env\"
        export PATH=\"\${VLLM_ENV}/bin:\$PATH\"
        ray stop --force </dev/null >/dev/null 2>&1 || true
    " < /dev/null 2>/dev/null || true
    echo "  Ray cluster stopped."
}
trap cleanup_ray EXIT ERR

# ── Step 1: Start router ──────────────────────────────────────────────────────
echo ""
echo "[1/8] Starting router on $NODE1..."
nohup uvicorn semantic_router.main:app \
    --host 0.0.0.0 --port "$ROUTER_PORT" \
    > "$LOG_DIR/router.log" 2>&1 &
ROUTER_PID=$!
echo "  PID: $ROUTER_PID  log: $LOG_DIR/router.log"
sleep 5
wait_router

# ── Step 2: Start NODE1 provisioner ──────────────────────────────────────────
echo ""
if [ "$EXPERIMENT_MODE" = "dynamic" ]; then
    echo "[2/8] Starting provisioner on $NODE1 (dynamic mode — 2 seed models, cold-spinup enabled)..."
    PROV_FLAGS="--router-mode ttca"
    # NODE1 has 4 GPUs: seed on GPU 0-1, qwen3-coder-30b cold on GPU 2-3
    PROV_INITIAL="qwen-7b,deepseek-r1-7b"
    echo "  Seed models: $PROV_INITIAL (2 GPUs)"
    echo "  Cold model (spun up on demand): qwen3-coder-30b (GPU 2-3)"
    echo "  Note: gemma-3-27b and deepseek-r1-14b run permanently on $NODE2 (SSH-started)"
else
    echo "[2/8] Starting provisioner on $NODE1 (static mode — 3 models, all 4 GPUs)..."
    PROV_FLAGS="--router-mode ttca --static"
    # NODE1 has 4 GPUs: qwen-7b(1) + deepseek-7b(1) + qwen3-coder-30b(2)
    PROV_INITIAL="qwen-7b,deepseek-r1-7b,qwen3-coder-30b"
fi
nohup python provisioner/dynamic_provisioner.py \
    --router-url   "$ROUTER_URL" \
    --node-host    "$NODE1" \
    $PROV_FLAGS \
    --priors-path  "$PRIORS_FILE" \
    --initial-models "$PROV_INITIAL" \
    > "$LOG_DIR/provisioner_node1.log" 2>&1 &
PROV1_PID=$!
echo "  PID: $PROV1_PID  log: $LOG_DIR/provisioner_node1.log"

# ── Step 3: Start NODE2 overflow models + Ray cluster + llama4-scout ─────────
echo ""
echo "[3/8] Starting overflow models on $NODE2 and llama4-scout Ray cluster..."

# ---- NODE2: gemma-3-27b on GPU 0-1 (port 8003) ----
echo "  Starting gemma-3-27b on $NODE2 (port 8003, GPU 0-1)..."
# shellcheck disable=SC2029
ssh "$NODE2" "
    VLLM_ENV=\"\$HOME/.conda/envs/2026-06-08/vllm_env\"
    export PATH=\"\${VLLM_ENV}/bin:\$PATH\"
    export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
    export CUDA_VISIBLE_DEVICES=0,1
    nohup vllm serve google/gemma-3-27b-it \
        --host 0.0.0.0 --port 8003 \
        --tensor-parallel-size 2 \
        --max-model-len 8192 \
        --trust-remote-code \
        >'${LOG_DIR}/gemma_27b_node2.log' 2>&1 &
    BGPID=\$!
    disown \$BGPID
    echo PID:\$BGPID
" < /dev/null
echo "  log: $LOG_DIR/gemma_27b_node2.log (on $NODE2)"

# ---- NODE2: deepseek-r1-14b on GPU 2-3 (port 8004) ----
echo "  Starting deepseek-r1-14b on $NODE2 (port 8004, GPU 2-3)..."
# shellcheck disable=SC2029
ssh "$NODE2" "
    VLLM_ENV=\"\$HOME/.conda/envs/2026-06-08/vllm_env\"
    export PATH=\"\${VLLM_ENV}/bin:\$PATH\"
    export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
    export CUDA_VISIBLE_DEVICES=2,3
    nohup vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --host 0.0.0.0 --port 8004 \
        --tensor-parallel-size 2 \
        --max-model-len 8192 \
        --trust-remote-code \
        >'${LOG_DIR}/deepseek_14b_node2.log' 2>&1 &
    BGPID=\$!
    disown \$BGPID
    echo PID:\$BGPID
" < /dev/null
echo "  log: $LOG_DIR/deepseek_14b_node2.log (on $NODE2)"

# ---- SCOUT: Ray head on SCOUT_NODE1 ----
echo "  Starting Ray head on $SCOUT_NODE1 (port 6379)..."
# shellcheck disable=SC2029
ssh "$SCOUT_NODE1" "
    VLLM_ENV=\"\$HOME/.conda/envs/2026-06-08/vllm_env\"
    export PATH=\"\${VLLM_ENV}/bin:\$PATH\"
    ray start --head --port=6379 --num-cpus=32 --num-gpus=4 \
        </dev/null >'${LOG_DIR}/ray_head.log' 2>&1 &
    BGPID=\$!
    disown \$BGPID
    echo PID:\$BGPID
" < /dev/null
echo "  Waiting 20s for Ray head to initialize... (log: $LOG_DIR/ray_head.log)"
sleep 20

# ---- SCOUT: Ray worker on SCOUT_NODE2 ----
echo "  Starting Ray worker on $SCOUT_NODE2 (joining $SCOUT_NODE1:6379)..."
# shellcheck disable=SC2029
ssh "$SCOUT_NODE2" "
    VLLM_ENV=\"\$HOME/.conda/envs/2026-06-08/vllm_env\"
    export PATH=\"\${VLLM_ENV}/bin:\$PATH\"
    ray start --address='${SCOUT_NODE1}:6379' --num-cpus=32 --num-gpus=4 \
        </dev/null >'${LOG_DIR}/ray_worker.log' 2>&1 &
    BGPID=\$!
    disown \$BGPID
    echo PID:\$BGPID
" < /dev/null
echo "  Waiting 20s for Ray worker to join cluster... (log: $LOG_DIR/ray_worker.log)"
sleep 20

# ---- SCOUT: vLLM llama4-scout (Ray distributes TP=8 across SCOUT_NODE1 + SCOUT_NODE2) ----
echo "  Starting vLLM llama4-scout on $SCOUT_NODE1 (port 8005, TP=8 via Ray)..."
# Ray head is on SCOUT_NODE1 and sees 4+4=8 GPUs via the worker on SCOUT_NODE2.
# vLLM's --tensor-parallel-size 8 uses the Ray cluster to span both nodes automatically.
# shellcheck disable=SC2029
ssh "$SCOUT_NODE1" "
    VLLM_ENV=\"\$HOME/.conda/envs/2026-06-08/vllm_env\"
    export PATH=\"\${VLLM_ENV}/bin:\$PATH\"
    export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    nohup vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
        --host 0.0.0.0 --port 8005 \
        --tensor-parallel-size 8 \
        --max-model-len 8192 \
        --trust-remote-code \
        >'${LOG_DIR}/llama4_scout.log' 2>&1 &
    BGPID=\$!
    disown \$BGPID
    echo PID:\$BGPID
" < /dev/null
echo "  log: $LOG_DIR/llama4_scout.log (on $SCOUT_NODE1)"

# ── Step 4: Wait for all models and register ──────────────────────────────────
echo ""
if [ "$EXPERIMENT_MODE" = "dynamic" ]; then
    echo "[4/8] Waiting for seed models on $NODE1 (dynamic mode: 2 seed models)..."
    wait_models 2   # qwen-7b + deepseek-r1-7b (provisioner auto-registers)
else
    echo "[4/8] Waiting for $NODE1 provisioner models (static mode: 3 models)..."
    wait_models 3   # qwen-7b + deepseek-r1-7b + qwen3-coder-30b
fi

# Wait for NODE2 models and register them manually (no provisioner on NODE2)
echo "  Waiting for gemma-3-27b on $NODE2:8003..."
wait_vllm_health "$NODE2:8003" "gemma-3-27b"

echo "  Waiting for deepseek-r1-14b on $NODE2:8004..."
wait_vllm_health "$NODE2:8004" "deepseek-r1-14b"

echo "  Registering gemma-3-27b (node 2, port 8003) with estimated priors..."
curl --noproxy '*' -sf -X POST "$ROUTER_URL/router/register" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_id\": \"gemma-3-27b\",
    \"model_name\": \"google/gemma-3-27b-it\",
    \"backend\": \"vllm\",
    \"base_url\": \"http://$NODE2:8003\",
    \"domains\": [\"factual\",\"reasoning\",\"creative\",\"math\",\"code\"],
    \"min_accuracy_capability\": {\"_default\": 0.82},
    \"accuracy_priors\": {
      \"factual:easy\": 0.94, \"factual:medium\": 0.91, \"factual:hard\": 0.86,
      \"reasoning:easy\": 0.92, \"reasoning:medium\": 0.88, \"reasoning:hard\": 0.82,
      \"math:easy\": 0.88, \"math:medium\": 0.82, \"math:hard\": 0.72,
      \"code:easy\": 0.91, \"code:medium\": 0.86, \"code:hard\": 0.78,
      \"creative:easy\": 0.94, \"creative:medium\": 0.91, \"creative:hard\": 0.88
    },
    \"decode_tokens_per_sec\": 100,
    \"skip_calibration\": true
  }" > /dev/null

echo "  Registering deepseek-r1-14b (node 2, port 8004) with estimated priors..."
curl --noproxy '*' -sf -X POST "$ROUTER_URL/router/register" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_id\": \"deepseek-r1-14b\",
    \"model_name\": \"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B\",
    \"backend\": \"vllm\",
    \"base_url\": \"http://$NODE2:8004\",
    \"domains\": [\"factual\",\"reasoning\",\"creative\",\"math\",\"code\"],
    \"min_accuracy_capability\": {\"_default\": 0.80},
    \"accuracy_priors\": {
      \"factual:easy\": 0.91, \"factual:medium\": 0.87, \"factual:hard\": 0.82,
      \"reasoning:easy\": 0.94, \"reasoning:medium\": 0.90, \"reasoning:hard\": 0.84,
      \"math:easy\": 0.92, \"math:medium\": 0.87, \"math:hard\": 0.80,
      \"code:easy\": 0.90, \"code:medium\": 0.85, \"code:hard\": 0.78,
      \"creative:easy\": 0.90, \"creative:medium\": 0.87, \"creative:hard\": 0.83
    },
    \"decode_tokens_per_sec\": 150,
    \"skip_calibration\": true
  }" > /dev/null

# Wait for llama4-scout (may take 10-20 min to shard 109B weights across 8 GPUs)
echo "  Waiting for llama4-scout on $SCOUT_NODE1:8005 (may take 10-20 min for 109B model)..."
wait_vllm_health "$SCOUT_NODE1:8005" "llama4-scout"

echo "  Registering llama4-scout (scout cluster) with estimated priors..."
curl --noproxy '*' -sf -X POST "$ROUTER_URL/router/register" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_id\": \"llama4-scout\",
    \"model_name\": \"meta-llama/Llama-4-Scout-17B-16E-Instruct\",
    \"backend\": \"vllm\",
    \"base_url\": \"http://$SCOUT_NODE1:8005\",
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

# Register NODE1 models with priors from the priors file
echo "  Registering node-1 models with priors..."
python tests/register_with_priors.py \
    --priors     "$PRIORS_FILE" \
    --router-url "$ROUTER_URL"

echo "  Final model list:"
curl --noproxy '*' -sf "$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; [print(f'    {m[\"id\"]}') for m in json.load(sys.stdin)['data']]"

# ── Step 5: Eval matrix ───────────────────────────────────────────────────────
echo ""
echo "[5/10] Building eval matrix (6 models x 1000 queries, ~30-45 min)..."
python tests/eval_all_models.py \
    --dataset     "$DATASET" \
    --output      "$RESULTS_DIR/eval_matrix.csv" \
    --concurrency "$EVAL_CONCURRENCY" \
    --node2-host  "$SCOUT_NODE1"

# ── Step 6: Extract real accuracy priors from eval matrix ─────────────────────
echo ""
echo "[6/10] Extracting real accuracy priors from eval matrix..."
python tests/extract_priors.py \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --output      "$RESULTS_DIR/priors_new.json"

echo "  Priors extracted:"
python3 -c "
import json
p = json.load(open('$RESULTS_DIR/priors_new.json'))
for model, priors in sorted(p.items()):
    keys = sorted(priors.keys())
    vals = [priors[k] for k in keys[:3]]
    print(f'    {model:<22} {len(priors)} keys  e.g. {keys[0]}={vals[0]:.3f}')
"

# ── Step 6b: Build optimal tier maps from eval matrix ────────────────────────
echo ""
echo "[6b/10] Building data-driven optimal tier maps..."
python tests/build_optimal_tier.py \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --output      "$RESULTS_DIR/optimal_tier_maps.json" \
    --alpha       "$TTCA_ALPHA" \
    --beta        "$TTCA_COST_BETA"

# ── Step 7: Re-register all models with real priors ────────────────────────────
echo ""
echo "[7/10] Re-registering all models with real accuracy priors..."
python tests/register_with_priors.py \
    --priors     "$RESULTS_DIR/priors_new.json" \
    --router-url "$ROUTER_URL" \
    --node2-host "$SCOUT_NODE1"

echo "  Models registered with real priors:"
curl --noproxy '*' -sf "$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; [print(f'    {m[\"id\"]}') for m in json.load(sys.stdin)['data']]"

# ── Step 8: TTCA load test ────────────────────────────────────────────────────
echo ""
echo "[8/10] TTCA router load test (${N_REQUESTS} requests, real priors active)..."
python tests/load_test.py \
    --dataset     "$DATASET" \
    --router      "$ROUTER_URL" \
    --mode        ttca \
    --requests    "$N_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --output      "$RESULTS_DIR/router_ttca.csv"

# ── Step 9: Accuracy load test ────────────────────────────────────────────────
echo ""
echo "[9/10] Accuracy router load test (${N_REQUESTS} requests)..."
python tests/load_test.py \
    --dataset     "$DATASET" \
    --router      "$ROUTER_URL" \
    --mode        accuracy \
    --requests    "$N_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --output      "$RESULTS_DIR/router_accuracy.csv"

# ── Step 10: Round-robin baseline + comparisons ───────────────────────────────
echo ""
echo "[10/10] Round-robin baseline (${N_REQUESTS} requests)..."
python tests/round_robin_test.py \
    --dataset     "$DATASET" \
    --output      "$RESULTS_DIR/rr_baseline.csv" \
    --concurrency "$CONCURRENCY" \
    --node2-host  "$SCOUT_NODE1"

echo ""
echo "=== Per-category breakdown: TTCA router vs Round-Robin ===" | tee "$RESULTS_DIR/compare_categories_ttca.txt"
python tests/compare_categories.py \
    --router      "$RESULTS_DIR/router_ttca.csv" \
    --baseline    "$RESULTS_DIR/rr_baseline.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --output      "$RESULTS_DIR/compare_categories_ttca.csv" \
    | tee -a "$RESULTS_DIR/compare_categories_ttca.txt"

echo ""
echo "=== Per-category breakdown: Accuracy router vs Round-Robin ===" | tee "$RESULTS_DIR/compare_categories_accuracy.txt"
python tests/compare_categories.py \
    --router      "$RESULTS_DIR/router_accuracy.csv" \
    --baseline    "$RESULTS_DIR/rr_baseline.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --output      "$RESULTS_DIR/compare_categories_accuracy.csv" \
    | tee -a "$RESULTS_DIR/compare_categories_accuracy.txt"

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

# ── Baseline comparisons ──────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  Baseline comparisons"
echo "=================================================================="

echo ""
echo "[Baseline 1/4] TTCA single-shot (no retry, ${N_REQUESTS} requests)..."
python tests/load_test.py \
    --dataset     "$DATASET" \
    --router      "$ROUTER_URL" \
    --mode        ttca \
    --no-retry \
    --requests    "$N_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --output      "$RESULTS_DIR/ttca_no_retry.csv"

echo ""
echo "[Baseline 2/4] Complexity-tier routing (${N_REQUESTS} requests)..."
python tests/baseline_complexity_tier.py \
    --dataset     "$DATASET" \
    --concurrency "$CONCURRENCY" \
    --node2-host  "$SCOUT_NODE1" \
    --output      "$RESULTS_DIR/baseline_tier.csv"

echo ""
echo "[Baseline 3/4] Cascade routing / RouteLLM-style (${N_REQUESTS} requests)..."
python tests/baseline_cascade.py \
    --dataset     "$DATASET" \
    --priors      "$RESULTS_DIR/priors_new.json" \
    --threshold   0.80 \
    --concurrency "$CONCURRENCY" \
    --output      "$RESULTS_DIR/baseline_cascade.csv"

echo ""
echo "[Baseline 5/6] Accuracy-optimal tier (data-driven, ${N_REQUESTS} requests)..."
python tests/baseline_complexity_tier.py \
    --dataset     "$DATASET" \
    --concurrency "$CONCURRENCY" \
    --node2-host  "$SCOUT_NODE1" \
    --tier-map    "$RESULTS_DIR/optimal_tier_maps.json" \
    --tier-variant accuracy_optimal \
    --output      "$RESULTS_DIR/baseline_tier_optimal_acc.csv"

echo ""
echo "[Baseline 6/6] TTCA-optimal tier (data-driven, ${N_REQUESTS} requests)..."
python tests/baseline_complexity_tier.py \
    --dataset     "$DATASET" \
    --concurrency "$CONCURRENCY" \
    --node2-host  "$SCOUT_NODE1" \
    --tier-map    "$RESULTS_DIR/optimal_tier_maps.json" \
    --tier-variant ttca_optimal \
    --output      "$RESULTS_DIR/baseline_tier_optimal_ttca.csv"

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
    echo "  vllm-sr not running on port 8888 — skipping."
    echo "  To run it: pip install vllm-sr && vllm-sr serve --config configs/vllm_sr_config.yaml"
fi

# ── Baseline comparison reports ───────────────────────────────────────────────
echo ""
echo "=== TTCA (with retry) vs TTCA (no retry) ===" | tee "$RESULTS_DIR/compare_retry_benefit.txt"
python tests/compare_ttca.py \
    --router      "$RESULTS_DIR/router_ttca.csv" \
    --baseline    "$RESULTS_DIR/ttca_no_retry.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    | tee -a "$RESULTS_DIR/compare_retry_benefit.txt"

echo ""
echo "=== TTCA (no retry) vs Complexity-Tier ===" | tee "$RESULTS_DIR/compare_vs_tier.txt"
python tests/compare_categories.py \
    --router      "$RESULTS_DIR/ttca_no_retry.csv" \
    --baseline    "$RESULTS_DIR/baseline_tier.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --output      "$RESULTS_DIR/compare_vs_tier.csv" \
    | tee -a "$RESULTS_DIR/compare_vs_tier.txt"

echo ""
echo "=== TTCA (with retry) vs Cascade/RouteLLM ===" | tee "$RESULTS_DIR/compare_ttca_vs_cascade.txt"
python tests/compare_ttca.py \
    --router      "$RESULTS_DIR/router_ttca.csv" \
    --baseline    "$RESULTS_DIR/baseline_cascade.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    | tee -a "$RESULTS_DIR/compare_ttca_vs_cascade.txt"
echo ""
python tests/compare_categories.py \
    --router      "$RESULTS_DIR/router_ttca.csv" \
    --baseline    "$RESULTS_DIR/baseline_cascade.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --output      "$RESULTS_DIR/compare_ttca_vs_cascade.csv" \
    | tee -a "$RESULTS_DIR/compare_ttca_vs_cascade.txt"

echo ""
echo "=== TTCA (no retry) vs Accuracy-Optimal Tier ===" | tee "$RESULTS_DIR/compare_vs_tier_optimal_acc.txt"
python tests/compare_categories.py \
    --router      "$RESULTS_DIR/ttca_no_retry.csv" \
    --baseline    "$RESULTS_DIR/baseline_tier_optimal_acc.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --output      "$RESULTS_DIR/compare_vs_tier_optimal_acc.csv" \
    | tee -a "$RESULTS_DIR/compare_vs_tier_optimal_acc.txt"

echo ""
echo "=== TTCA (no retry) vs TTCA-Optimal Tier ===" | tee "$RESULTS_DIR/compare_vs_tier_optimal_ttca.txt"
python tests/compare_categories.py \
    --router      "$RESULTS_DIR/ttca_no_retry.csv" \
    --baseline    "$RESULTS_DIR/baseline_tier_optimal_ttca.csv" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --output      "$RESULTS_DIR/compare_vs_tier_optimal_ttca.csv" \
    | tee -a "$RESULTS_DIR/compare_vs_tier_optimal_ttca.txt"

if [ -f "$RESULTS_DIR/baseline_vllm_sr.csv" ]; then
    echo ""
    echo "=== TTCA (no retry) vs vLLM Semantic Router ===" | tee "$RESULTS_DIR/compare_vs_vllm_sr.txt"
    python tests/compare_categories.py \
        --router      "$RESULTS_DIR/ttca_no_retry.csv" \
        --baseline    "$RESULTS_DIR/baseline_vllm_sr.csv" \
        --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
        --output      "$RESULTS_DIR/compare_vs_vllm_sr.csv" \
        | tee -a "$RESULTS_DIR/compare_vs_vllm_sr.txt"

    echo ""
    echo "=== TTCA (with retry) vs vLLM Semantic Router ===" | tee "$RESULTS_DIR/compare_ttca_retry_vs_vllm_sr.txt"
    python tests/compare_categories.py \
        --router      "$RESULTS_DIR/router_ttca.csv" \
        --baseline    "$RESULTS_DIR/baseline_vllm_sr.csv" \
        --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
        --output      "$RESULTS_DIR/compare_ttca_retry_vs_vllm_sr.csv" \
        | tee -a "$RESULTS_DIR/compare_ttca_retry_vs_vllm_sr.txt"
fi

# ── Unified all-systems comparison ───────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  Unified all-systems comparison"
echo "=================================================================="

SYS_ARGS=(
    "--system" "TTCA+retry:$RESULTS_DIR/router_ttca.csv"
    "--system" "Cascade/RouteLLM:$RESULTS_DIR/baseline_cascade.csv"
    "--system" "Round-Robin:$RESULTS_DIR/rr_baseline.csv"
    "--system" "TTCA-no-retry:$RESULTS_DIR/ttca_no_retry.csv"
    "--system" "Tier-optimal-acc:$RESULTS_DIR/baseline_tier_optimal_acc.csv"
    "--system" "Tier-optimal-ttca:$RESULTS_DIR/baseline_tier_optimal_ttca.csv"
    "--system" "Complexity-tier:$RESULTS_DIR/baseline_tier.csv"
)
[ -f "$RESULTS_DIR/baseline_vllm_sr.csv" ] && \
    SYS_ARGS+=("--system" "vLLM-SR:$RESULTS_DIR/baseline_vllm_sr.csv")

python tests/compare_all.py \
    "${SYS_ARGS[@]}" \
    --eval-matrix "$RESULTS_DIR/eval_matrix.csv" \
    --ref         "TTCA+retry" \
    --output      "$RESULTS_DIR/compare_all.csv" \
    | tee "$RESULTS_DIR/compare_all.txt"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  Experiment complete!  $(date)"
echo "  Results: $RESULTS_DIR/"
echo "    eval_matrix.csv"
echo "    router_ttca.csv"
echo "    router_accuracy.csv"
echo "    rr_baseline.csv"
echo "    compare_all.txt / compare_all.csv  ALL systems side-by-side (start here)"
echo "  — PRIMARY comparisons (TTCA+retry vs full systems) —"
echo "    compare_ttca_vs_rr.txt              TTCA+retry vs round-robin"
echo "    compare_ttca_vs_cascade.csv         TTCA+retry vs cascade/RouteLLM"
echo "    compare_ttca_retry_vs_vllm_sr.csv   TTCA+retry vs vLLM-SR (if run)"
echo "  — DIAGNOSTIC —"
echo "    compare_retry_benefit.txt           TTCA+retry vs TTCA no-retry (retry value)"
echo "    compare_vs_tier.csv                 TTCA no-retry vs complexity-tier"
echo "    compare_vs_tier_optimal_acc.csv     TTCA no-retry vs accuracy-optimal tier"
echo "    compare_vs_tier_optimal_ttca.csv    TTCA no-retry vs TTCA-optimal tier"
echo "  — Baselines produced —"
echo "    baseline_cascade.csv   baseline_tier.csv   baseline_vllm_sr.csv"
echo "    baseline_tier_optimal_acc.csv   baseline_tier_optimal_ttca.csv"
echo "  Ray cluster teardown runs automatically on exit."
echo "=================================================================="
