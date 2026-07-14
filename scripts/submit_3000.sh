#!/bin/bash
# submit_3000.sh — Same as submit_static_vs_dynamic.sh but runs all 3000 requests.
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
#   bash scripts/submit_3000.sh
#   RATE=10 bash scripts/submit_3000.sh
#
# Parameters:
#   N_REQUESTS   total requests per mode (default 3000)
#   CONCURRENCY  max simultaneous in-flight requests (default 50)
#   RATE         arrival rate in req/s, open-loop (default: empty = closed-loop)
#   SEED         random seed for workload sampling (default 42)

set -euo pipefail

N_REQUESTS=${N_REQUESTS:-3000}
CONCURRENCY=${CONCURRENCY:-50}
RATE=${RATE:-""}          # empty = closed-loop; set e.g. RATE=10 for 10 req/s
RATE_FLAG=""
[ -n "$RATE" ] && RATE_FLAG="--rate ${RATE}"
PROJECT=${PROJECT:-UIC-HPC}
QUEUE=${QUEUE:-by-node}
DATASET=${DATASET:-"datasets/hf_3000.json"}
PRIORS=${PRIORS:-"results/priors_all5.json"}
SEED=${SEED:-42}
DOMAIN_FILTER=${DOMAIN_FILTER:-""}
DOMAIN_LABEL="${DOMAIN_FILTER:-all}"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HOME/vllm_logs/static_vs_dynamic_${TS}"
mkdir -p "$LOG_DIR"

PBSSCRIPT=$(mktemp /tmp/svd_XXXXXX.pbs)

cat > "$PBSSCRIPT" << PBSEOF
#!/bin/bash
#PBS -l select=2:ngpus=8:ncpus=64
#PBS -l walltime=10:00:00
#PBS -l filesystems=home:eagle
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -N static_vs_dynamic_3000
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

# ── Generate fixed workload (same for both modes) ─────────────────────────────
echo ""
echo "[0] Generating fixed workload (N=${N_REQUESTS}, seed=${SEED}, domain=${DOMAIN_LABEL})..."
python3 -c "
import json, random
random.seed(${SEED})
data = json.load(open('${DATASET}'))
domain_filter = '${DOMAIN_FILTER}'
if domain_filter:
    data = [x for x in data if x.get('domain') == domain_filter]
    print(f'  Domain filter: {domain_filter} ({len(data)} items available)')
sample = random.sample(data, min(${N_REQUESTS}, len(data)))
json.dump(sample, open('/tmp/svd_workload.json', 'w'))
from collections import Counter
by_domain = Counter(x.get('domain') for x in sample)
by_complex = Counter(x.get('complexity') for x in sample)
DOMAINS = ['factual', 'math', 'code', 'reasoning']
COMPLEXITIES = ['easy', 'medium', 'hard']
by_cell = Counter((x.get('domain'), x.get('complexity')) for x in sample)
print(f'  Workload: {len(sample)} requests')
print(f'  By domain    : {dict(sorted(by_domain.items()))}')
print(f'  By complexity: {dict(sorted(by_complex.items()))}')
print(f'  By domain x complexity:')
print(f"    {\"\":12} {\"easy\":>6} {\"medium\":>8} {\"hard\":>6} {\"total\":>7}")
for d in DOMAINS:
    row = [by_cell.get((d, c), 0) for c in COMPLEXITIES]
    pcts = [f\"{v/len(sample)*100:.0f}%\" for v in row]
    print(f"    {d:12} {row[0]:>3}({pcts[0]:>3}) {row[1]:>3}({pcts[1]:>3}) {row[2]:>3}({pcts[2]:>3}) {sum(row):>6}")
col_tots = [sum(by_cell.get((d,c),0) for d in DOMAINS) for c in COMPLEXITIES]
print(f"    {\"total\":12} {col_tots[0]:>6} {col_tots[1]:>8} {col_tots[2]:>6} {len(sample):>7}")
"


# ── Helper ────────────────────────────────────────────────────────────────────
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
    ssh "\$NODE2" "pkill -f 'vllm serve' 2>/dev/null || true" </dev/null 2>/dev/null || true
    sleep 10
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
wait_node2_gpus_free() {
    echo "  Sending kill signal to node2 vLLM processes..."
    ssh "\$NODE2" "pkill -f 'vllm serve' 2>/dev/null; true" </dev/null 2>/dev/null || true
    echo "  Waiting 60s for node2 GPU memory to free..."
    sleep 60
    echo "  Node2 teardown wait complete."
}
wait_llama4_scout() {
    echo "  Waiting for llama4-scout on \$NODE2:8005 (timeout 90 min)..."
    for i in \$(seq 1 360); do
        if curl --noproxy '*' -sf "http://\$NODE2:8005/health" > /dev/null 2>&1; then
            echo "  llama4-scout ready! (\$((i*15))s elapsed)"
            return 0
        fi
        [ \$((i % 8)) -eq 0 ] && echo "  [\$((i*15))s] Still waiting for llama4-scout..."
        sleep 15
    done
    echo "WARNING: llama4-scout not ready after 90 min — continuing anyway"
}

# ════════════════════════════════════════════════════════════════════
# MODE 1: STATIC — all 6 models pre-loaded
# ════════════════════════════════════════════════════════════════════
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

SCOUT_CNT=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(sum(1 for m in d if 'llama4' in m['id']))" 2>/dev/null || echo 0)
if [ "\$SCOUT_CNT" -eq 0 ]; then
    echo "  llama4-scout not yet registered — waiting for it on \$NODE2:8005..."
    wait_llama4_scout
    if curl --noproxy '*' -sf "http://\$NODE2:8005/health" > /dev/null 2>&1; then
        curl --noproxy '*' -sf -X POST "\$ROUTER_URL/router/register" \
            -H "Content-Type: application/json" \
            -d "{\"model_id\":\"llama4-scout\",\"model_name\":\"meta-llama/Llama-4-Scout-17B-16E-Instruct\",\"backend\":\"vllm\",\"base_url\":\"http://\${NODE2}:8005\",\"domains\":[\"code\",\"factual\",\"math\",\"reasoning\"],\"efficiency_tokens_per_joule\":3.0,\"input_rate_usd_per_token\":0.0000001,\"output_rate_usd_per_token\":0.0000003,\"skip_calibration\":true}" \
            2>/dev/null && echo "  Manually registered llama4-scout with router." \
            || echo "  WARNING: manual registration attempt failed"
    else
        echo "  WARNING: llama4-scout not responding on \$NODE2:8005 — proceeding with 5 models"
    fi
fi

echo ""
echo "[S3b] Pre-evaluating all 6 models on workload → eval_matrix.csv"
echo "      (Enables Tier-Optimal-Acc and Tier-Optimal-Cost oracle baselines)"
echo "      This takes ~20-40 min while models are warm. Running concurrently."
python tests/eval_all_models.py \
    --dataset     /tmp/svd_workload.json \
    --output      "\$RESULTS_DIR/eval_matrix.csv" \
    --concurrency 20 \
    --node2-host  "\$NODE2"
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
echo "[S3e] Running Tier-Optimal-Acc oracle..."
python tests/baseline_complexity_tier.py \
    --dataset     /tmp/svd_workload.json \
    --tier-map    "\$RESULTS_DIR/optimal_tier_maps.json" \
    --tier-variant accuracy_optimal \
    --concurrency ${CONCURRENCY} \
    --output      "\$RESULTS_DIR/baseline_tier_acc_optimal.csv"

echo ""
echo "[S3f] CARROT baseline (mu=0.3)..."
python tests/baseline_carrot.py \
    --dataset     /tmp/svd_workload.json \
    --eval-matrix "\$RESULTS_DIR/eval_matrix.csv" \
    --mu          0.3 \
    --concurrency ${CONCURRENCY} \
    --node2-host  "\$NODE2" \
    --output      "\$RESULTS_DIR/baseline_carrot.csv"

echo ""
echo "[S3g] OmniRouter baseline (alpha=0.75)..."
python tests/baseline_omni_router.py \
    --dataset     /tmp/svd_workload.json \
    --eval-matrix "\$RESULTS_DIR/eval_matrix.csv" \
    --alpha       0.75 \
    --concurrency ${CONCURRENCY} \
    --node2-host  "\$NODE2" \
    --output      "\$RESULTS_DIR/baseline_omni_router.csv"

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

STATIC_MODELS=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; ids=[m['id'] for m in json.load(sys.stdin)['data']]; print(len(ids), ','.join(sorted(ids)))" 2>/dev/null || echo "? unknown")
echo "  Models loaded during static run: \$STATIC_MODELS"

echo ""
echo "=================================================================="
echo "  BASELINES: Cascade and RouteLLM (all 6 models alive)"
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
OPENAI_API_KEY=dummy python tests/baseline_routellm.py \
    --dataset     /tmp/svd_workload.json \
    --calibrate \
    2>&1 | tee "\$RESULTS_DIR/routellm_calibration.txt" || true

ROUTELLM_THRESHOLD=\${ROUTELLM_THRESHOLD:-0.5}
echo "  Using threshold=\${ROUTELLM_THRESHOLD}"
OPENAI_API_KEY=dummy python tests/baseline_routellm.py \
    --dataset     /tmp/svd_workload.json \
    --threshold   \${ROUTELLM_THRESHOLD} \
    --concurrency ${CONCURRENCY} \
    --output      "\$RESULTS_DIR/baseline_routellm.csv" \
    || echo "  WARNING: RouteLLM baseline failed"

echo ""
echo "[S5] Tearing down static mode..."
kill_all_models
wait_node2_gpus_free
sleep 5

# ════════════════════════════════════════════════════════════════════
# MODE 2: DYNAMIC — start with seed models, others spin up
# ════════════════════════════════════════════════════════════════════
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
echo "[D3] Waiting for node1 seed models (3)..."
wait_models 3
echo ""
echo "[D3b] Waiting for llama4-scout on \$NODE2..."
wait_llama4_scout

echo ""
echo "[D3c] Warm-up: 50 easy requests to measure real model latencies..."
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
echo "  Warm-up done."

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

DYNAMIC_SPINUPS=\$(grep "SPIN UP" ~/vllm_logs/prov_svd_dynamic_node1.log 2>/dev/null \
    | grep -v "reason=initial" | awk '{print \$3}' | sort -u | tr '\n' ',' | sed 's/,$//')
echo "  Models dynamically spun up: \${DYNAMIC_SPINUPS:-none}"

echo ""
echo "[D5] Tearing down dynamic mode..."
kill_all_models

# ════════════════════════════════════════════════════════════════════
# COMPARISON
# ════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  COMPARISON: All systems"
echo "══════════════════════════════════════════════════════════════════"

echo ""
echo "  Total experiment time (provisioner start → load test end):"
echo "    Static : \${STATIC_WALL}s"
echo "    Dynamic: \${DYNAMIC_WALL}s"
LOAD_TEST_WALL_STATIC=\$((STATIC_END - STATIC_START))
LOAD_TEST_WALL_DYNAMIC=\$((DYNAMIC_END - DYNAMIC_START))
echo "  Load test only:"
echo "    Static : \${LOAD_TEST_WALL_STATIC}s"
echo "    Dynamic: \${LOAD_TEST_WALL_DYNAMIC}s"

COMPARE_ARGS=(tests/compare_all.py)
COMPARE_ARGS+=(--system "Round-Robin:results/rr_baseline.csv")
COMPARE_ARGS+=(--system "Cascade:\$RESULTS_DIR/baseline_cascade.csv")
[ -f "\$RESULTS_DIR/baseline_routellm.csv" ] && \
  COMPARE_ARGS+=(--system "RouteLLM:\$RESULTS_DIR/baseline_routellm.csv")
[ -f "\$RESULTS_DIR/baseline_carrot.csv" ] && \
  COMPARE_ARGS+=(--system "CARROT:\$RESULTS_DIR/baseline_carrot.csv")
[ -f "\$RESULTS_DIR/baseline_omni_router.csv" ] && \
  COMPARE_ARGS+=(--system "OmniRouter:\$RESULTS_DIR/baseline_omni_router.csv")
COMPARE_ARGS+=(--system "Static (TTCA):\$RESULTS_DIR/static_results.csv")
COMPARE_ARGS+=(--system "Dynamic (TTCA):\$RESULTS_DIR/dynamic_results.csv")
[ -f "\$RESULTS_DIR/baseline_tier_acc_optimal.csv" ] && \
  COMPARE_ARGS+=(--system "Tier-Opt-Acc:\$RESULTS_DIR/baseline_tier_acc_optimal.csv")
[ -f "\$RESULTS_DIR/baseline_cost_optimal.csv" ] && \
  COMPARE_ARGS+=(--system "Tier-Opt-Cost:\$RESULTS_DIR/baseline_cost_optimal.csv")
COMPARE_ARGS+=(--ref "Static (TTCA)")
COMPARE_ARGS+=(--eval-matrix "\$RESULTS_DIR/eval_matrix.csv")
COMPARE_ARGS+=(--output "\$RESULTS_DIR/compare_all_systems.csv")

echo ""
echo "  [compare_all.py] Ranking all systems..."
python "\${COMPARE_ARGS[@]}" 2>&1 | tee "\$RESULTS_DIR/compare_all_systems.txt"
echo "  [compare_all.py] Done."

echo ""
echo "=================================================================="
echo "  STATIC vs DYNAMIC — Full Metric Comparison"
echo "=================================================================="
export RESULTS_DIR_PY=\$RESULTS_DIR
python3 << PYEOF
import csv, sys, os
from statistics import mean

rd = os.environ['RESULTS_DIR_PY']

def load(path, label):
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print('  WARNING: ' + path + ' not found')
        return None
    ok = [r for r in rows if r.get('status') == '200']
    scored = [r for r in ok if r.get('correct') in ('true','false','True','False','0','1')]
    correct = [r for r in scored if r.get('correct') in ('true','True','1')]
    lats = [float(r.get('actual_latency_ms') or r.get('wall_ms', 0))
            for r in ok if r.get('actual_latency_ms') or r.get('wall_ms')]
    costs = [float(r['charged_usd']) for r in ok if r.get('charged_usd')]
    energy = [float(r['energy_j']) for r in ok if r.get('energy_j')]
    slo_rows = [r for r in rows if r.get('slo_violated') in ('true','false','True','False')]
    slo_viol = [r for r in slo_rows if r.get('slo_violated') in ('true','True')]
    retries = [int(r['retries']) for r in ok if r.get('retries','') not in ('','0',None)]
    slats = sorted(lats)
    ttca_lats = sorted([
        float(r.get('actual_latency_ms') or r.get('wall_ms', 0))
        for r in ok
        if (r.get('gt_correct') in ('true','True') or r.get('correct') in ('true','True','1'))
        and (r.get('actual_latency_ms') or r.get('wall_ms'))
    ])
    def _pct(arr, p): return arr[int(len(arr)*p)] if arr else None
    return {
        'label':         label,
        'n_total':       len(rows),
        'n_ok':          len(ok),
        'accuracy':      len(correct)/len(scored)*100 if scored else None,
        'lat_mean':      mean(lats)/1000              if lats   else None,
        'lat_p50':       slats[len(slats)//2]/1000    if lats   else None,
        'lat_p95':       slats[int(len(slats)*0.95)]/1000 if lats else None,
        'ttca_mean':     mean(ttca_lats)/1000         if ttca_lats else None,
        'ttca_p50':      _pct(ttca_lats, 0.50)/1000  if ttca_lats else None,
        'ttca_p90':      _pct(ttca_lats, 0.90)/1000  if ttca_lats else None,
        'ttca_p95':      _pct(ttca_lats, 0.95)/1000  if ttca_lats else None,
        'cost_mean':     mean(costs)                  if costs  else None,
        'energy_mean':   mean(energy)                 if energy else None,
        'slo_viol_n':    len(slo_viol),
        'slo_total':     len(slo_rows),
        'slo_pct':       len(slo_viol)/len(slo_rows)*100 if slo_rows else None,
        'retries_total': sum(int(r.get('retries',0)) for r in ok),
        'retries_mean':  mean(retries) if retries else 0.0,
    }

s = load(rd + '/static_results.csv',  'Static')
d = load(rd + '/dynamic_results.csv', 'Dynamic')
if not s or not d:
    sys.exit(0)

def fmt(v, spec, suffix=''):
    return ((spec % v) + suffix) if v is not None else '-'
def usd(v):
    return ('$' + '%.8f' % v) if v is not None else '-'
def delta(sv, dv, spec, suffix=''):
    if sv is None or dv is None: return '-'
    diff = dv - sv
    sign = '+' if diff >= 0 else ''
    return sign + (spec % diff) + suffix

W1, W2, W3, W4 = 28, 14, 14, 14
sep = '-' * (W1 + W2 + W3 + W4 + 3)
print('\n  ' + ('Metric').ljust(W1) + ('Static').rjust(W2) + ' ' + ('Dynamic').rjust(W3) + ' ' + ('Delta (D-S)').rjust(W4))
print('  ' + sep)
print('  ' + ('Requests (200 OK)').ljust(W1) + str(s['n_ok']).rjust(W2) + ' ' + str(d['n_ok']).rjust(W3))
print('  ' + ('Accuracy').ljust(W1) + fmt(s['accuracy'],'%.1f','%').rjust(W2) + ' ' + fmt(d['accuracy'],'%.1f','%').rjust(W3) + ' ' + delta(s['accuracy'],d['accuracy'],'%.1f','pp').rjust(W4))
print('  ' + ('Lat mean').ljust(W1) + fmt(s['lat_mean'],'%.2f','s').rjust(W2) + ' ' + fmt(d['lat_mean'],'%.2f','s').rjust(W3) + ' ' + delta(s['lat_mean'],d['lat_mean'],'%.2f','s').rjust(W4))
print('  ' + ('Lat P50').ljust(W1) + fmt(s['lat_p50'],'%.2f','s').rjust(W2) + ' ' + fmt(d['lat_p50'],'%.2f','s').rjust(W3) + ' ' + delta(s['lat_p50'],d['lat_p50'],'%.2f','s').rjust(W4))
print('  ' + ('Lat P95').ljust(W1) + fmt(s['lat_p95'],'%.2f','s').rjust(W2) + ' ' + fmt(d['lat_p95'],'%.2f','s').rjust(W3) + ' ' + delta(s['lat_p95'],d['lat_p95'],'%.2f','s').rjust(W4))
print('  ' + ('TTCA mean (correct only)').ljust(W1) + fmt(s['ttca_mean'],'%.2f','s').rjust(W2) + ' ' + fmt(d['ttca_mean'],'%.2f','s').rjust(W3) + ' ' + delta(s['ttca_mean'],d['ttca_mean'],'%.2f','s').rjust(W4))
print('  ' + ('TTCA P50').ljust(W1) + fmt(s['ttca_p50'],'%.2f','s').rjust(W2) + ' ' + fmt(d['ttca_p50'],'%.2f','s').rjust(W3) + ' ' + delta(s['ttca_p50'],d['ttca_p50'],'%.2f','s').rjust(W4))
print('  ' + ('TTCA P90').ljust(W1) + fmt(s['ttca_p90'],'%.2f','s').rjust(W2) + ' ' + fmt(d['ttca_p90'],'%.2f','s').rjust(W3) + ' ' + delta(s['ttca_p90'],d['ttca_p90'],'%.2f','s').rjust(W4))
print('  ' + ('TTCA P95').ljust(W1) + fmt(s['ttca_p95'],'%.2f','s').rjust(W2) + ' ' + fmt(d['ttca_p95'],'%.2f','s').rjust(W3) + ' ' + delta(s['ttca_p95'],d['ttca_p95'],'%.2f','s').rjust(W4))
print('  ' + ('Cost/req').ljust(W1) + usd(s['cost_mean']).rjust(W2) + ' ' + usd(d['cost_mean']).rjust(W3))
print('  ' + ('SLO violations').ljust(W1) + (str(s['slo_viol_n'])+'/'+str(s['slo_total'])).rjust(W2) + ' ' + (str(d['slo_viol_n'])+'/'+str(d['slo_total'])).rjust(W3))
print('  ' + ('SLO violation rate').ljust(W1) + fmt(s['slo_pct'],'%.1f','%').rjust(W2) + ' ' + fmt(d['slo_pct'],'%.1f','%').rjust(W3) + ' ' + delta(s['slo_pct'],d['slo_pct'],'%.1f','pp').rjust(W4))
print('  ' + ('Total retries').ljust(W1) + str(s['retries_total']).rjust(W2) + ' ' + str(d['retries_total']).rjust(W3) + ' ' + delta(s['retries_total'],d['retries_total'],'%d').rjust(W4))
print('  ' + ('Avg retries/req').ljust(W1) + fmt(s['retries_mean'],'%.3f').rjust(W2) + ' ' + fmt(d['retries_mean'],'%.3f').rjust(W3))

# ── TTCA by domain x complexity ──────────────────────────────────
DOMAINS_LIST = ['factual', 'math', 'code', 'reasoning']
COMPLEXITIES_LIST = ['easy', 'medium', 'hard']

def _load_rows(path):
    try:
        return list(csv.DictReader(open(path)))
    except FileNotFoundError:
        return []

def _ttca_cells(rows):
    cells = {}
    for r in rows:
        if r.get('status') != '200':
            continue
        if not (r.get('gt_correct') in ('true','True') or r.get('correct') in ('true','True','1')):
            continue
        lat = r.get('actual_latency_ms') or r.get('wall_ms')
        if not lat:
            continue
        key = (r.get('domain',''), r.get('complexity',''))
        cells.setdefault(key, []).append(float(lat)/1000)
    return cells

s_cells = _ttca_cells(_load_rows(rd + '/static_results.csv'))
d_cells = _ttca_cells(_load_rows(rd + '/dynamic_results.csv'))

print('\n  TTCA MEAN BY DOMAIN x COMPLEXITY  (correct answers only, seconds)')
W_cat, W_col = 20, 12
print('  ' + 'Category'.ljust(W_cat) + 'Static'.rjust(W_col) + ' ' + 'Dynamic'.rjust(W_col) + ' ' + 'Delta(D-S)'.rjust(W_col))
print('  ' + '-' * (W_cat + W_col * 3 + 2))
prev_d = None
for domain in DOMAINS_LIST:
    for complexity in COMPLEXITIES_LIST:
        if prev_d and domain != prev_d:
            print()
        prev_d = domain
        sv = mean(s_cells[(domain, complexity)]) if s_cells.get((domain, complexity)) else None
        dv = mean(d_cells[(domain, complexity)]) if d_cells.get((domain, complexity)) else None
        s_str = ('%.2fs' % sv) if sv is not None else '-'
        d_str = ('%.2fs' % dv) if dv is not None else '-'
        dl_str = (('+' if dv - sv >= 0 else '') + '%.2fs' % (dv - sv)) if sv is not None and dv is not None else '-'
        print('  ' + (domain + ':' + complexity).ljust(W_cat) + s_str.rjust(W_col) + ' ' + d_str.rjust(W_col) + ' ' + dl_str.rjust(W_col))
print()
PYEOF

echo ""
echo "=================================================================="
echo "  GPU ENERGY COMPARISON (includes idle GPU power)"
echo "=================================================================="
python3 tests/compute_gpu_energy.py \
    --static-log          ~/vllm_logs/prov_svd_static_node1.log \
    --static-wall         \$STATIC_WALL \
    --static-start-epoch  \$STATIC_PROV_START \
    --dynamic-log         ~/vllm_logs/prov_svd_dynamic_node1.log \
    --dynamic-wall        \$DYNAMIC_WALL \
    --dynamic-start-epoch \$DYNAMIC_PROV_START \
    | tee "\$RESULTS_DIR/gpu_energy_comparison.txt"

echo "=================================================================="
echo "  Comparison complete!  \$(date)"
echo "  Results: \$RESULTS_DIR/"
echo "=================================================================="
PBSEOF

echo "Submitting 3000-request comparison..."
echo "  N_REQUESTS  : $N_REQUESTS"
echo "  CONCURRENCY : $CONCURRENCY"
echo "  RATE        : ${RATE:-closed-loop} req/s"
echo "  Dataset     : $DATASET  (seed=$SEED)"
echo "  Walltime    : 10:00:00"
echo "  Log dir     : $LOG_DIR/"
echo ""

JOBID=$(qsub "$PBSSCRIPT")
rm -f "$PBSSCRIPT"

echo "Submitted: $JOBID"
echo ""
echo "Monitor:"
echo "  qstat -u yuping"
echo "  tail -f $LOG_DIR/job.out"
