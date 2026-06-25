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
#                Example: RATE=10 sends exactly 10 req/s regardless of model speed
#                Leave empty for closed-loop (saturated queue, triggers provisioner faster)
#   SEED         random seed for workload sampling (default 42, reproducible)

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

RATE_STR="${RATE:-closed-loop}"
echo "=================================================================="
echo "  Static vs Dynamic Mode Comparison   \$(date)"
echo "  NODE1 : \$NODE1   NODE2 : \$NODE2"
echo "  N_REQUESTS  : ${N_REQUESTS}"
echo "  CONCURRENCY : ${CONCURRENCY}  (max in-flight)"
echo "  RATE        : \${RATE_STR} req/s"
echo "  Dataset     : ${DATASET}  (seed=${SEED})"
echo "  Metrics     : accuracy, latency, energy/req, cost/req"
echo "=================================================================="

# ── Generate fixed workload (same for both modes) ─────────────────────────────
echo ""
echo "[0] Generating fixed workload (N=${N_REQUESTS}, seed=${SEED})..."
python3 -c "
import json, random
random.seed(${SEED})
data = json.load(open('${DATASET}'))
sample = random.sample(data, min(${N_REQUESTS}, len(data)))
json.dump(sample, open('/tmp/svd_workload.json', 'w'))
from collections import Counter
by_domain = Counter(x.get('domain') for x in sample)
by_complex = Counter(x.get('complexity') for x in sample)
print(f'  Workload: {len(sample)} requests')
print(f'  By domain    : {dict(sorted(by_domain.items()))}')
print(f'  By complexity: {dict(sorted(by_complex.items()))}')
"

# ── Helpers ───────────────────────────────────────────────────────────────────
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
    curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
        | python3 -c "
import sys, json, urllib.request
for m in json.load(sys.stdin).get('data', []):
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                'http://${NODE1}:8080/router/' + m['id'],
                method='DELETE'), timeout=5)
    except: pass
" 2>/dev/null || true
    echo "  All models stopped."
}

# ════════════════════════════════════════════════════════════════════
# MODE 1: STATIC — all 6 models pre-loaded
# ════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  MODE 1: STATIC (all 6 models pre-loaded — no cold start)"
echo "══════════════════════════════════════════════════════════════════"

nohup uvicorn semantic_router.main:app \
    --host 0.0.0.0 --port 8080 \
    > ~/vllm_logs/router_svd_static.log 2>&1 &
sleep 8; wait_router; echo "  Router ready."

nohup python provisioner/dynamic_provisioner.py \
    --router-url "\$ROUTER_URL" --node-host "\$NODE1" \
    --router-mode ttca --static --priors-path "${PRIORS}" \
    --initial-models "qwen-7b,deepseek-r1-7b,qwen3-coder-30b,gemma-3-27b,deepseek-r1-14b" \
    > ~/vllm_logs/prov_svd_static_node1.log 2>&1 &

ssh "\$NODE2" "
    cd ~/semantic-llm-router; export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
    nohup python provisioner/dynamic_provisioner.py \
        --router-url '\$ROUTER_URL' --node-host '\$NODE2' \
        --router-mode ttca --static --priors-path '${PRIORS}' \
        --initial-models llama4-scout \
        </dev/null >>~/vllm_logs/prov_svd_static_node2.log 2>&1 &
    disown \$!
" </dev/null

echo "Waiting for all 6 models..."
wait_models 6

echo ""
echo "[S4] Running workload — STATIC (${N_REQUESTS} requests, rate=${RATE:-closed-loop})..."
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
STATIC_WALL=\$((STATIC_END - STATIC_START))
echo "  Static wall time: \${STATIC_WALL}s"

echo "Tearing down static mode..."
kill_all_models; sleep 5

# ════════════════════════════════════════════════════════════════════
# MODE 2: DYNAMIC — only seed models, others spin up on demand
# ════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  MODE 2: DYNAMIC (qwen-7b + deepseek-r1-7b seeds, cold start)"
echo "══════════════════════════════════════════════════════════════════"

nohup uvicorn semantic_router.main:app \
    --host 0.0.0.0 --port 8080 \
    > ~/vllm_logs/router_svd_dynamic.log 2>&1 &
sleep 8; wait_router; echo "  Router ready."

nohup python provisioner/dynamic_provisioner.py \
    --router-url "\$ROUTER_URL" --node-host "\$NODE1" \
    --router-mode ttca --priors-path "${PRIORS}" \
    --initial-models "qwen-7b,deepseek-r1-7b" \
    > ~/vllm_logs/prov_svd_dynamic_node1.log 2>&1 &

ssh "\$NODE2" "
    cd ~/semantic-llm-router; export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
    nohup python provisioner/dynamic_provisioner.py \
        --router-url '\$ROUTER_URL' --node-host '\$NODE2' \
        --router-mode ttca --static --priors-path '${PRIORS}' \
        --initial-models llama4-scout \
        </dev/null >>~/vllm_logs/prov_svd_dynamic_node2.log 2>&1 &
    disown \$!
" </dev/null

echo "Waiting for seed models (3)..."
wait_models 3

echo ""
echo "[D4] Running workload — DYNAMIC (${N_REQUESTS} requests, rate=${RATE:-closed-loop})..."
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
DYNAMIC_WALL=\$((DYNAMIC_END - DYNAMIC_START))
echo "  Dynamic wall time: \${DYNAMIC_WALL}s"

DYNAMIC_SPINUPS=\$(grep "SPIN UP" ~/vllm_logs/prov_svd_dynamic_node1.log 2>/dev/null \
    | grep -v "reason=initial" | awk '{print \$3}' | sort -u | tr '\n' ',' | sed 's/,$//')
echo "  Models dynamically spun up: \${DYNAMIC_SPINUPS:-none}"

echo "Tearing down dynamic mode..."
kill_all_models

# ════════════════════════════════════════════════════════════════════
# COMPARISON
# ════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  COMPARISON: Static vs Dynamic"
echo "══════════════════════════════════════════════════════════════════"

WALL_DIFF=\$((DYNAMIC_WALL - STATIC_WALL))
echo ""
echo "  Wall time:"
echo "    Static : \${STATIC_WALL}s  (all models pre-loaded)"
echo "    Dynamic: \${DYNAMIC_WALL}s  (cold start for first hard requests)"
[ "\$WALL_DIFF" -ge 0 ] && \
    echo "    Cold start overhead: +\${WALL_DIFF}s" || \
    echo "    Dynamic faster by: \${WALL_DIFF#-}s"

echo ""
python tests/compare_all.py \
    --system "Static:\$RESULTS_DIR/static_results.csv" \
    --system "Dynamic:\$RESULTS_DIR/dynamic_results.csv" \
    --ref    "Static" \
    --output "\$RESULTS_DIR/static_vs_dynamic.csv" \
    | tee "\$RESULTS_DIR/static_vs_dynamic.txt"

echo ""
echo "  Per-mode energy and cost totals:"
python3 -c "
import csv
from statistics import mean

def summarize(path, label):
    rows = [r for r in csv.DictReader(open(path)) if r.get('status') == '200']
    energy = [float(r['energy_j'])    for r in rows if r.get('energy_j')]
    cost   = [float(r['charged_usd']) for r in rows if r.get('charged_usd')]
    lats   = [float(r.get('actual_latency_ms') or r.get('wall_ms', 0))
              for r in rows if r.get('actual_latency_ms') or r.get('wall_ms')]
    winners = {}
    for r in rows:
        m = r.get('model_winner', '?')
        winners[m] = winners.get(m, 0) + 1
    slats = sorted(lats)
    p50 = slats[len(slats)//2] / 1000 if slats else 0
    p95 = slats[int(len(slats)*0.95)] / 1000 if slats else 0
    print(f'  {label}:')
    print(f'    Requests   : {len(rows)}')
    print(f'    Latency    : P50={p50:.2f}s  P95={p95:.2f}s')
    if energy:
        print(f'    Energy     : total={sum(energy):.1f}J  mean={mean(energy):.2f}J/req')
    if cost:
        print(f'    Cost       : total=\${sum(cost):.6f}  mean=\${mean(cost):.8f}/req')
    print(f'    Model mix  : {dict(sorted(winners.items(), key=lambda x:-x[1]))}')
    print()

summarize('\$RESULTS_DIR/static_results.csv',  'Static ')
summarize('\$RESULTS_DIR/dynamic_results.csv', 'Dynamic')
"

echo "=================================================================="
echo "  Done!  \$(date)"
echo "  Results: \$RESULTS_DIR/"
echo "    static_results.csv        Static mode  (all 6 models from second 1)"
echo "    dynamic_results.csv       Dynamic mode (cold start, spin-up on demand)"
echo "    static_vs_dynamic.csv     Side-by-side accuracy/latency/energy/cost"
echo "    static_vs_dynamic.txt     Human-readable summary"
echo ""
echo "  Key insight: Dynamic saves energy by only running models that are needed."
echo "  Trade-off: cold start latency for first hard requests in each domain."
echo "=================================================================="
PBSEOF

echo "Submitting static vs dynamic comparison..."
echo "  N_REQUESTS  : $N_REQUESTS"
echo "  CONCURRENCY : $CONCURRENCY"
echo "  RATE        : ${RATE:-closed-loop} req/s"
echo "  Dataset     : $DATASET  (seed=$SEED)"
echo "  Walltime    : 05:00:00"
echo "  Log dir     : $LOG_DIR/"
echo ""

JOBID=$(qsub "$PBSSCRIPT")
rm -f "$PBSSCRIPT"

echo "Submitted: $JOBID"
echo ""
echo "Monitor:"
echo "  qstat -u yuping"
echo "  tail -f $LOG_DIR/job.out"
