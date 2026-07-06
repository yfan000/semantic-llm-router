#!/bin/bash
# submit_static_vs_dynamic.sh — Compare static vs dynamic provisioning modes.
#
# Runs the SAME workload through both modes and reports side-by-side:
#   - Accuracy   (% correct answers)
#   - Latency    (P50 / P95 per request)
#   - Energy     (joules per request — static wastes energy on idle GPUs)
#   - Cost       (USD per request)
#
# Static mode:  all 6 models pre-loaded before the workload starts.
#               No cold start; all capacity available immediately.
#               GPU cost: all 8 GPUs on node1 busy the whole time.
#
# Dynamic mode: only qwen-7b + deepseek-r1-7b start; others spin up as load
#               increases. Cold start penalty for first hard requests.
#               GPU cost: proportional to actual demand.
#
# Both modes use the same random seed and dataset sample for fair comparison.
#
# Usage:
#   bash scripts/submit_static_vs_dynamic.sh
#   N_REQUESTS=500 RATE=10 bash scripts/submit_static_vs_dynamic.sh
#
# Parameters:
#   N_REQUESTS   total requests per mode (default 300)
#   CONCURRENCY  max simultaneous in-flight requests (default 50)
#   RATE         arrival rate in req/s, open-loop (default: empty = closed-loop)
#                Example: RATE=10 sends exactly 10 req/s
#   SEED         random seed for workload sampling (default 42)

set -euo pipefail

N_REQUESTS=${N_REQUESTS:-300}
CONCURRENCY=${CONCURRENCY:-50}
RATE=${RATE:-""}          # empty = closed-loop; set e.g. RATE=10 for 10 req/s
RATE_FLAG=""
[ -n "$RATE" ] && RATE_FLAG="--rate ${RATE}"
PROJECT=${PROJECT:-UIC-HPC}
QUEUE=${QUEUE:-by-node}
DATASET=${DATASET:-"datasets/hf_3000.json"}
PRIORS=${PRIORS:-"results/priors_all5.json"}
SEED=${SEED:-42}

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HOME/vllm_logs/static_vs_dynamic_${TS}"
mkdir -p "$LOG_DIR"

PBSSCRIPT=$(mktemp /tmp/svd_XXXXXX.pbs)

cat > "$PBSSCRIPT" << PBSEOF
#!/bin/bash
#PBS -l select=2:ngpus=8:ncpus=64
#PBS -l walltime=05:00:00
#PBS -l filesystems=home:eagle
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -N static_vs_dynamic
#PBS -o ${LOG_DIR}/job.out
#PBS -e ${LOG_DIR}/job.err

echo "PBS script started at \$(date) on \$(hostname)"
VLLM_ENV="\$HOME/.conda/envs/2026-06-08/vllm_env"
export PATH="\${VLLM_ENV}/bin:\$PATH"
echo "  Python: \$(which python 2>/dev/null || echo NOT FOUND)"
export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache

cd ~/semantic-llm-router
git pull --quiet

NODES=(\$(sort -u \$PBS_NODEFILE))
NODE1=\${NODES[0]}
NODE2=\${NODES[1]}
ROUTER_URL="http://\${NODE1}:8080"
RESULTS_DIR="results/static_vs_dynamic_${TS}"
mkdir -p "\$RESULTS_DIR"

echo "=================================================================="
echo "  Static vs Dynamic Mode Comparison   \$(date)"
echo "  NODE1 : \$NODE1   NODE2 : \$NODE2"
RATE_STR="${RATE:-closed-loop}"
echo "  N_REQUESTS  : ${N_REQUESTS}"
echo "  CONCURRENCY : ${CONCURRENCY}  (max in-flight)"
echo "  RATE        : \${RATE_STR} req/s"
echo "  Dataset     : ${DATASET}  (seed=${SEED})"
echo "  Metrics     : accuracy, latency, energy/req, cost/req"
echo "=================================================================="

# ── Generate fixed workload (same for both modes) ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[0] Generating fixed workload (N=${N_REQUESTS}, seed=${SEED})..."
python3 -c "
import json, random
random.seed(${SEED})
data = json.load(open('${DATASET}'))
# Balanced mix: easy/medium/hard across all domains
sample = random.sample(data, min(${N_REQUESTS}, len(data)))
json.dump(sample, open('/tmp/svd_workload.json', 'w'))
from collections import Counter
by_domain = Counter(x.get('domain') for x in sample)
by_complex = Counter(x.get('complexity') for x in sample)
print(f'  Workload: {len(sample)} requests')
print(f'  By domain    : {dict(sorted(by_domain.items()))}')
print(f'  By complexity: {dict(sorted(by_complex.items()))}')
"

# ── Helper ─────────────────────────────────────────────────────────────────────────────────────────────
wait_router() {
    for i in \$(seq 1 60); do
        curl --noproxy '*' -sf "\$ROUTER_URL/router/health" > /dev/null 2>&1 && return 0
        sleep 5
    done
    echo "ERROR: Router not ready"; exit 1
}
wait_models() {
    local N=\$1
    for i in \$(seq 1 240); do
        local cnt
        cnt=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
            | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" \
            2>/dev/null || echo 0)
        echo "  [\$((i*15))s] \$cnt/\$N models ready"
        [ "\$cnt" -ge "\$N" ] && return 0
        sleep 15
    done
    echo "WARNING: only \$cnt/\$N models after timeout"
}
kill_all_models() {
    echo "  Stopping all vLLM processes..."
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 10
    # Deregister from router
    curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
        | python3 -c "
import sys, json, urllib.request
for m in json.load(sys.stdin).get('data', []):
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                f'http://\${NODE1}:8080/router/{m[\"id\"]}',
                method='DELETE'), timeout=5)
    except: pass
" 2>/dev/null || true
    echo "  All models stopped."
}

# ══════════════════════════════════════════════════════════════════
# MODE 1: STATIC — all 6 models pre-loaded
# ══════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  MODE 1: STATIC (all 6 models pre-loaded)"
echo "══════════════════════════════════════════════════════════════════"

echo ""
echo "[S1] Starting router..."
nohup uvicorn semantic_router.main:app \
    --host 0.0.0.0 --port 8080 \
    > ~/vllm_logs/router_svd_static.log 2>&1 &
sleep 8; wait_router; echo "  Router ready."

echo ""
echo "[S2] Starting all 5 models on \$NODE1 (static)..."
STATIC_PROV_START=\$(date +%s)
nohup python provisioner/dynamic_provisioner.py \
    --router-url  "\$ROUTER_URL" \
    --node-host   "\$NODE1" \
    --router-mode ttca \
    --static \
    --priors-path "${PRIORS}" \
    --initial-models "qwen-7b,deepseek-r1-7b,qwen3-coder-30b,gemma-3-27b,deepseek-r1-14b" \
    > ~/vllm_logs/prov_svd_static_node1.log 2>&1 &

echo "[S2b] Starting llama4-scout on \$NODE2..."
ssh "\$NODE2" "
    cd ~/semantic-llm-router
    export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
    nohup python provisioner/dynamic_provisioner.py \
        --router-url '\$ROUTER_URL' --node-host '\$NODE2' \
        --router-mode ttca --static --priors-path '${PRIORS}' \
        --initial-models llama4-scout \
        </dev/null >>~/vllm_logs/prov_svd_static_node2.log 2>&1 &
    disown \$!
" </dev/null

echo ""
echo "[S3] Waiting for all 6 models..."
wait_models 6

# Pre-evaluate all models on the workload (runs while static models are loaded)
echo ""
echo "[S3b] Pre-evaluating all 6 models on workload → eval_matrix.csv"
echo "      (Enables Tier-Optimal-Acc and Tier-Optimal-Cost oracle baselines)"
echo "      This takes ~20-40 min while models are warm. Running concurrently."
python tests/eval_all_models.py \
    --dataset     /tmp/svd_workload.json \
    --output      "\$RESULTS_DIR/eval_matrix.csv" \
    --concurrency 20
echo "  eval_matrix.csv done."

echo ""
echo "[S3c] Building optimal tier maps from eval_matrix..."
python tests/build_optimal_tier.py \
    --eval-matrix "\$RESULTS_DIR/eval_matrix.csv" \
    --output      "\$RESULTS_DIR/optimal_tier_maps.json" \
    --alpha 1.0 --beta 0.0
echo "  optimal_tier_maps.json done."

echo ""
echo "[S3d] Building Tier-Optimal-Cost oracle CSV..."
python tests/baseline_cost_optimal.py \
    --eval-matrix "\$RESULTS_DIR/eval_matrix.csv" \
    --output      "\$RESULTS_DIR/baseline_cost_optimal.csv"

echo ""
echo "[S3e] Running Tier-Optimal-Acc oracle (complexity-tier with accuracy_optimal map)..."
python tests/baseline_complexity_tier.py \
    --dataset     /tmp/svd_workload.json \
    --tier-map    "\$RESULTS_DIR/optimal_tier_maps.json" \
    --tier-variant accuracy_optimal \
    --concurrency ${CONCURRENCY} \
    --output      "\$RESULTS_DIR/baseline_tier_acc_optimal.csv"

echo ""
echo "[S4] Running workload in STATIC mode (${N_REQUESTS} requests, rate=${RATE:-closed-loop})..."
STATIC_START=\$(date +%s)
python tests/load_test.py \
    --dataset     /tmp/svd_workload.json \
    --router      "\$ROUTER_URL" \
    --mode        ttca \
    --requests    ${N_REQUESTS} \
    --concurrency ${CONCURRENCY} \
    ${RATE_FLAG} \
    --output      "\$RESULTS_DIR/static_results.csv"
STATIC_END=\$(date +%s)
STATIC_WALL=\$((STATIC_END - STATIC_PROV_START))
echo "  Static total experiment time: \${STATIC_WALL}s (provisioner start → load test end)"

# Count models that were registered during the run
STATIC_MODELS=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; ids=[m['id'] for m in json.load(sys.stdin)['data']]; print(len(ids), ','.join(sorted(ids)))" 2>/dev/null || echo "? unknown")
echo "  Models loaded during static run: \$STATIC_MODELS"

echo ""
echo "[S5] Tearing down static mode..."
kill_all_models
sleep 5

# ══════════════════════════════════════════════════════════════════
# MODE 2: DYNAMIC — start with 2 seed models, others spin up
# ══════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  MODE 2: DYNAMIC (qwen-7b + deepseek-r1-7b seeds)"
echo "══════════════════════════════════════════════════════════════════"

echo ""
echo "[D1] Starting router..."
nohup uvicorn semantic_router.main:app \
    --host 0.0.0.0 --port 8080 \
    > ~/vllm_logs/router_svd_dynamic.log 2>&1 &
sleep 8; wait_router; echo "  Router ready."

echo ""
echo "[D2] Starting seed models (dynamic mode)..."
DYNAMIC_PROV_START=\$(date +%s)
nohup python provisioner/dynamic_provisioner.py \
    --router-url  "\$ROUTER_URL" \
    --node-host   "\$NODE1" \
    --router-mode ttca \
    --priors-path "${PRIORS}" \
    --initial-models "qwen-7b,deepseek-r1-7b,qwen3-coder-30b" \
    > ~/vllm_logs/prov_svd_dynamic_node1.log 2>&1 &

ssh "\$NODE2" "
    cd ~/semantic-llm-router
    export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
    nohup python provisioner/dynamic_provisioner.py \
        --router-url '\$ROUTER_URL' --node-host '\$NODE2' \
        --router-mode ttca --static --priors-path '${PRIORS}' \
        --initial-models llama4-scout \
        </dev/null >>~/vllm_logs/prov_svd_dynamic_node2.log 2>&1 &
    disown \$!
" </dev/null

echo ""
echo "[D3] Waiting for seed models (4 — qwen-7b + deepseek-r1-7b + qwen3-coder-30b + llama4-scout)..."
wait_models 4

# ── Warm-up phase: populate reputation tracker with real latency data ────────────────────────
echo ""
echo "[D3b] Warm-up: 50 easy requests to measure real model latencies..."
echo "  (Without warm-up, TTCA uses catalog estimates which can be wildly off)"
python3 -c "
import json, random
random.seed(99)
data = json.load(open('/tmp/svd_workload.json'))
easy = [x for x in data if x.get('complexity') == 'easy']
sample = random.sample(easy, min(50, len(easy)))
json.dump(sample, open('/tmp/svd_warmup.json', 'w'))
print(f'  Warm-up: {len(sample)} easy requests')
"
python tests/load_test.py \
    --dataset     /tmp/svd_warmup.json \
    --router      "\$ROUTER_URL" \
    --mode        ttca \
    --requests    50 \
    --concurrency 10 \
    --output      /dev/null
echo "  Warm-up done. Reputation tracker has real latency measurements."

echo ""
echo "[D4] Running workload in DYNAMIC mode (${N_REQUESTS} requests, rate=${RATE:-closed-loop})..."
DYNAMIC_START=\$(date +%s)
python tests/load_test.py \
    --dataset     /tmp/svd_workload.json \
    --router      "\$ROUTER_URL" \
    --mode        ttca \
    --requests    ${N_REQUESTS} \
    --concurrency ${CONCURRENCY} \
    ${RATE_FLAG} \
    --output      "\$RESULTS_DIR/dynamic_results.csv"
DYNAMIC_END=\$(date +%s)
DYNAMIC_WALL=\$((DYNAMIC_END - DYNAMIC_PROV_START))
echo "  Dynamic total experiment time: \${DYNAMIC_WALL}s (provisioner start → load test end)"

# Show what models were spun up during dynamic run
DYNAMIC_SPINUPS=\$(grep "SPIN UP" ~/vllm_logs/prov_svd_dynamic_node1.log 2>/dev/null \
    | grep -v "reason=initial" | awk '{print \$3}' | sort -u | tr '\n' ',' | sed 's/,$//')
echo "  Models dynamically spun up: \${DYNAMIC_SPINUPS:-none}"

# ── Baselines: Cascade + RouteLLM ────────────────────────────────────────────────────────────────────────────────────────────
# NOTE: run BEFORE kill_all_models — baselines hit live vLLM endpoints
echo ""
echo "=================================================================="
echo "  BASELINES: Cascade and RouteLLM (run on same workload)"
echo "=================================================================="

echo ""
echo "  [Baseline 1/2] Cascade (threshold=0.80)..."
python tests/baseline_cascade.py \
    --dataset     /tmp/svd_workload.json \
    --priors      "${PRIORS}" \
    --threshold   0.80 \
    --concurrency ${CONCURRENCY} \
    --output      "\$RESULTS_DIR/baseline_cascade.csv"

echo ""
echo "  [Baseline 2/2] RouteLLM MF router..."
echo "  (Calibrating threshold to match cascade strong-model usage rate)"
# First calibrate to find the threshold that gives ~30% strong-model usage
# (matching the cascade baseline for fair comparison)
python tests/baseline_routellm.py \
    --dataset     /tmp/svd_workload.json \
    --calibrate \
    2>&1 | tee "\$RESULTS_DIR/routellm_calibration.txt" || true

# Run with default threshold=0.5 (routes ~20-30% to strong).
# Adjust ROUTELLM_THRESHOLD based on calibration output above.
ROUTELLM_THRESHOLD=\${ROUTELLM_THRESHOLD:-0.5}
echo "  Using threshold=\${ROUTELLM_THRESHOLD} (set ROUTELLM_THRESHOLD env var to override)"
python tests/baseline_routellm.py \
    --dataset     /tmp/svd_workload.json \
    --threshold   \${ROUTELLM_THRESHOLD} \
    --concurrency ${CONCURRENCY} \
    --output      "\$RESULTS_DIR/baseline_routellm.csv" \
    || echo "  WARNING: RouteLLM baseline failed (pip install routellm to enable)"

echo ""
echo "[D5] Tearing down dynamic mode..."
kill_all_models

# ══════════════════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  COMPARISON: All systems"
echo "══════════════════════════════════════════════════════════════════"

echo ""
echo "  Total experiment time (provisioner start → load test end):"
echo "    Static : \${STATIC_WALL}s  (includes ~10-20min model loading + workload)"
echo "    Dynamic: \${DYNAMIC_WALL}s  (includes seed model loading + workload)"
LOAD_TEST_WALL_STATIC=\$((STATIC_END - STATIC_START))
LOAD_TEST_WALL_DYNAMIC=\$((DYNAMIC_END - DYNAMIC_START))
echo "  Load test only:"
echo "    Static : \${LOAD_TEST_WALL_STATIC}s"
echo "    Dynamic: \${LOAD_TEST_WALL_DYNAMIC}s"

# Build --system list dynamically (include RouteLLM and oracle baselines only if files exist)
COMPARE_SYSTEMS=""
COMPARE_SYSTEMS="\$COMPARE_SYSTEMS --system \"Round-Robin:results/rr_baseline.csv\""
COMPARE_SYSTEMS="\$COMPARE_SYSTEMS --system \"Cascade:\$RESULTS_DIR/baseline_cascade.csv\""
[ -f "\$RESULTS_DIR/baseline_routellm.csv" ] && \
  COMPARE_SYSTEMS="\$COMPARE_SYSTEMS --system \"RouteLLM:\$RESULTS_DIR/baseline_routellm.csv\""
COMPARE_SYSTEMS="\$COMPARE_SYSTEMS --system \"Static (TTCA):\$RESULTS_DIR/static_results.csv\""
COMPARE_SYSTEMS="\$COMPARE_SYSTEMS --system \"Dynamic (TTCA):\$RESULTS_DIR/dynamic_results.csv\""
[ -f "\$RESULTS_DIR/baseline_tier_acc_optimal.csv" ] && \
  COMPARE_SYSTEMS="\$COMPARE_SYSTEMS --system \"Tier-Opt-Acc:\$RESULTS_DIR/baseline_tier_acc_optimal.csv\""
[ -f "\$RESULTS_DIR/baseline_cost_optimal.csv" ] && \
  COMPARE_SYSTEMS="\$COMPARE_SYSTEMS --system \"Tier-Opt-Cost:\$RESULTS_DIR/baseline_cost_optimal.csv\""

echo ""
eval python tests/compare_all.py \
    \$COMPARE_SYSTEMS \
    --ref    "Static (TTCA)" \
    --eval-matrix "\$RESULTS_DIR/eval_matrix.csv" \
    --output "\$RESULTS_DIR/compare_all_systems.csv" \
    | tee "\$RESULTS_DIR/compare_all_systems.txt"

# ── Per-source breakdown ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
echo ""
echo "  Per-source energy/cost breakdown:"
python3 -c "
import csv, json
from collections import defaultdict

def summarize(path, label):
    rows = [r for r in csv.DictReader(open(path)) if r.get('status') == '200']
    energy = [float(r['energy_j']) for r in rows if r.get('energy_j')]
    cost   = [float(r['charged_usd']) for r in rows if r.get('charged_usd')]
    lats   = [float(r.get('actual_latency_ms') or r.get('wall_ms',0)) for r in rows
              if r.get('actual_latency_ms') or r.get('wall_ms')]
    winners = {}
    for r in rows:
        m = r.get('model_winner','?')
        winners[m] = winners.get(m, 0) + 1
    print(f'  {label}:')
    print(f'    Requests      : {len(rows)}')
    if energy:
        print(f'    Energy total  : {sum(energy):.1f}J   mean={sum(energy)/len(energy):.2f}J/req')
    if cost:
        print(f'    Cost total    : \${sum(cost):.6f}   mean=\${sum(cost)/len(cost):.8f}/req')
    if lats:
        slats = sorted(lats)
        p50 = slats[len(slats)//2]
        p95 = slats[int(len(slats)*0.95)]
        print(f'    Latency       : P50={p50/1000:.2f}s  P95={p95/1000:.2f}s')
    print(f'    Model routing : {dict(sorted(winners.items(), key=lambda x:-x[1]))}')
    print()

summarize('\$RESULTS_DIR/static_results.csv',  'Static ')
summarize('\$RESULTS_DIR/dynamic_results.csv', 'Dynamic')
"

# ── GPU energy comparison (idle + serving, from provisioner logs) ────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  GPU ENERGY COMPARISON (includes idle GPU power)"
echo "  Static:  all GPUs always on → idle models still burn power"
echo "  Dynamic: GPUs power down when not needed"
echo "=================================================================="
python3 tests/compute_gpu_energy.py \
    --static-log          ~/vllm_logs/prov_svd_static_node1.log \
    --static-wall         \$STATIC_WALL \
    --static-start-epoch  \$STATIC_PROV_START \
    --dynamic-log         ~/vllm_logs/prov_svd_dynamic_node1.log \
    --dynamic-wall        \$DYNAMIC_WALL \
    --dynamic-start-epoch \$DYNAMIC_PROV_START \
    | tee "\$RESULTS_DIR/gpu_energy_comparison.txt"

# ── Done ─────────────────────────────────────────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "  Comparison complete!  \$(date)"
echo "  Results: \$RESULTS_DIR/"
echo "    static_results.csv            Static TTCA load test"
echo "    dynamic_results.csv           Dynamic TTCA load test"
echo "    baseline_cascade.csv          Cascade baseline (threshold=0.80)"
echo "    baseline_routellm.csv         RouteLLM MF router baseline"
echo "    eval_matrix.csv               All models × all requests (ground truth)"
echo "    optimal_tier_maps.json        Accuracy/TTCA/Cost optimal tier maps"
echo "    baseline_tier_acc_optimal.csv Tier-Optimal-Acc oracle (upper bound)"
echo "    baseline_cost_optimal.csv     Tier-Optimal-Cost oracle (lower bound)"
echo "    compare_all_systems.csv       All systems side-by-side (CSV)"
echo "    compare_all_systems.txt       All systems side-by-side (human readable)"
echo "    gpu_energy_comparison.txt     GPU-hours including idle energy"
echo ""
echo "  Routing strategy comparison:"
echo "    Cascade           : prior-threshold binary routing (weak vs strong)"
echo "    RouteLLM          : learned binary routing (MF router, GPT-4/Haiku trained)"
echo "    Static (TTCA)     : domain-aware TTCA, full fleet always loaded"
echo "    Dynamic (TTCA)    : domain-aware TTCA, fleet scales with demand"
echo "    Tier-Opt-Acc      : oracle upper bound on accuracy (best model per cell)"
echo "    Tier-Opt-Cost     : oracle lower bound on cost (cheapest correct per request)"
echo "=================================================================="
PBSEOF

echo "Submitting static vs dynamic comparison..."
echo "  N_REQUESTS  : $N_REQUESTS"
echo "  CONCURRENCY : $CONCURRENCY"
echo "  RATE        : ${RATE:-closed-loop} req/s"
echo "  Dataset     : $DATASET  (seed=$SEED)"
echo "  Walltime    : 05:00:00  (static ~2h + dynamic ~2h + buffer)"
echo "  Log dir     : $LOG_DIR/"
echo ""

JOBID=$(qsub "$PBSSCRIPT")
rm -f "$PBSSCRIPT"

echo "Submitted: $JOBID"
echo ""
echo "Monitor:"
echo "  qstat -u yuping"
echo "  tail -f $LOG_DIR/job.out"
