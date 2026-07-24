#!/bin/bash
# submit_alpha_sweep.sh — Sweep TTCA latency exponent α to map the accuracy-latency tradeoff.
#
# All four per-domain α values are set uniformly to the same value.
# β is fixed at 0.0 so cost plays no role in the sweep.
#
# How α works:
#   score(m,q) = p̂ / (L̂^α × ĉ^β)   [β=0 here]
#   α=0.0  — latency ignored; router becomes accuracy-only → selects strongest (slowest) model
#   α=0.5  — gentle latency discount
#   α=1.0  — default; 10× latency penalty costs 10× more utility to overcome
#   α=1.5  — aggressive; large models need strong accuracy gain to beat fast smaller models
#   α=2.0  — very aggressive; latency dominates routing
#
# Usage:
#   bash scripts/submit_alpha_sweep.sh
#   ALPHAS="0.0 0.5 1.0 2.0" N_REQUESTS=1000 bash scripts/submit_alpha_sweep.sh
#
# Parameters:
#   ALPHAS       space-separated α values to test (default: "0.0 0.3 0.5 1.0 1.5 2.0")
#   N_REQUESTS   requests per alpha run (default 300)
#   CONCURRENCY  max simultaneous in-flight requests (default 50)
#   RATE         arrival rate req/s; empty = closed-loop (default: empty)
#   SEED         random seed for workload sampling (default 42)

set -euo pipefail

ALPHAS_RAW=${ALPHAS:-"0.0 0.3 0.5 1.0 1.5 2.0"}
N_REQUESTS=${N_REQUESTS:-300}
CONCURRENCY=${CONCURRENCY:-50}
RATE=${RATE:-""}
RATE_FLAG=""
[ -n "$RATE" ] && RATE_FLAG="--rate ${RATE}"
PROJECT=${PROJECT:-UIC-HPC}
QUEUE=${QUEUE:-by-node}
DATASET=${DATASET:-"datasets/hf_3000.json"}
PRIORS=${PRIORS:-"results/priors_all5.json"}
SEED=${SEED:-42}

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HOME/vllm_logs/alpha_sweep_${TS}"
mkdir -p "$LOG_DIR"

PBSSCRIPT=$(mktemp /tmp/alpha_sweep_XXXXXX.pbs)

cat > "$PBSSCRIPT" << PBSEOF
#!/bin/bash
#PBS -l select=2:ngpus=8:ncpus=64
#PBS -l walltime=14:00:00
#PBS -l filesystems=home:eagle
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -N alpha_sweep
#PBS -o ${LOG_DIR}/job.out
#PBS -e ${LOG_DIR}/job.err

echo "Alpha sweep started at \$(date) on \$(hostname)"
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
RESULTS_DIR="results/alpha_sweep_${TS}"
mkdir -p "\$RESULTS_DIR"
ALPHAS="${ALPHAS_RAW}"

echo "=================================================================="
echo "  TTCA Alpha Sweep   \$(date)"
echo "  NODE1       : \$NODE1   NODE2 : \$NODE2"
echo "  ALPHAS      : \$ALPHAS"
echo "  BETA (fixed): 0.0  (cost-blind)"
echo "  N_REQUESTS  : ${N_REQUESTS}  (per alpha run)"
echo "  CONCURRENCY : ${CONCURRENCY}"
echo "  Dataset     : ${DATASET}  (seed=${SEED})"
echo "  Results dir : \$RESULTS_DIR/"
echo "=================================================================="

# ── Helpers (identical to submit_beta_sweep.sh) ───────────────────────────────
wait_router() {
    for i in \$(seq 1 60); do
        curl --noproxy '*' -sf "\$ROUTER_URL/router/health" > /dev/null 2>&1 && return 0
        sleep 5
    done
    echo "ERROR: Router not ready after 5 min"; exit 1
}
wait_models() {
    local N=\$1
    for i in \$(seq 1 240); do
        local cnt
        cnt=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
            | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" \
            2>/dev/null || echo 0)
        [ \$((i % 4)) -eq 0 ] && echo "  [\$((i*15))s] \$cnt/\$N models ready"
        [ "\$cnt" -ge "\$N" ] && return 0
        sleep 15
    done
    echo "WARNING: only \$cnt/\$N models after timeout — continuing"
}
wait_llama4_scout() {
    for i in \$(seq 1 360); do
        if curl --noproxy '*' -sf "http://\$NODE2:8005/health" > /dev/null 2>&1; then
            echo "  llama4-scout ready (\$((i*15))s elapsed)"
            return 0
        fi
        [ \$((i % 8)) -eq 0 ] && echo "  [\$((i*15))s] Waiting for llama4-scout on \$NODE2:8005..."
        sleep 15
    done
    echo "WARNING: llama4-scout not ready after 90 min — continuing anyway"
}
kill_all_models() {
    echo "  Stopping vLLM processes..."
    pkill -f "vllm serve" 2>/dev/null || true
    ssh "\$NODE2" "pkill -f 'vllm serve' 2>/dev/null || true" </dev/null 2>/dev/null || true
    sleep 10
    curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
        | python3 -c "
import sys, json, urllib.request
for m in json.load(sys.stdin).get('data', []):
    try:
        urllib.request.urlopen(urllib.request.Request(
            f'http://\${NODE1}:8080/router/{m[\"id\"]}', method='DELETE'), timeout=5)
    except: pass
" 2>/dev/null || true
    echo "  All models stopped."
}
start_static_fleet() {
    local LABEL=\$1
    echo "  Starting router (label=\${LABEL})..."
    nohup uvicorn semantic_router.main:app \
        --host 0.0.0.0 --port 8080 \
        > ~/vllm_logs/router_\${LABEL}.log 2>&1 &
    sleep 8; wait_router

    echo "  Starting node1 models (static)..."
    nohup python provisioner/dynamic_provisioner.py \
        --router-url  "\$ROUTER_URL" \
        --node-host   "\$NODE1" \
        --router-mode ttca \
        --static \
        --priors-path "${PRIORS}" \
        --initial-models "qwen-7b,deepseek-r1-7b,qwen3-coder-30b,gemma-3-27b,deepseek-r1-14b" \
        > ~/vllm_logs/prov_\${LABEL}_node1.log 2>&1 &

    echo "  Starting llama4-scout on \$NODE2..."
    ssh "\$NODE2" "
        cd ~/semantic-llm-router
        export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
        nohup python provisioner/dynamic_provisioner.py \
            --router-url '\$ROUTER_URL' --node-host '\$NODE2' \
            --router-mode ttca --static --priors-path '${PRIORS}' \
            --initial-models llama4-scout \
            </dev/null >>~/vllm_logs/prov_\${LABEL}_node2.log 2>&1 &
        disown \$!
    " </dev/null

    wait_models 5

    SCOUT_CNT=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
        | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(sum(1 for m in d if 'llama4' in m['id']))" \
        2>/dev/null || echo 0)
    if [ "\$SCOUT_CNT" -eq 0 ]; then
        wait_llama4_scout
        if curl --noproxy '*' -sf "http://\$NODE2:8005/health" > /dev/null 2>&1; then
            curl --noproxy '*' -sf -X POST "\$ROUTER_URL/router/register" \
                -H "Content-Type: application/json" \
                -d "{\"model_id\":\"llama4-scout\",\"model_name\":\"meta-llama/Llama-4-Scout-17B-16E-Instruct\",\"backend\":\"vllm\",\"base_url\":\"http://\${NODE2}:8005\",\"domains\":[\"code\",\"factual\",\"math\",\"reasoning\"],\"efficiency_tokens_per_joule\":3.0,\"input_rate_usd_per_token\":0.0000001,\"output_rate_usd_per_token\":0.0000003,\"skip_calibration\":true}" \
                2>/dev/null && echo "  Manually registered llama4-scout" || true
        fi
    fi
    wait_models 6
}
warmup_priors() {
    curl --noproxy '*' -sf -X POST "\$ROUTER_URL/router/warmup" \
        -H "Content-Type: application/json" \
        -d "{\"eval_matrix_path\": \"\$RESULTS_DIR/eval_matrix.csv\"}" \
        | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  Seeded {r[\"cells_seeded\"]} (model,domain,complexity) cells')" \
        2>/dev/null || echo "  WARNING: warmup failed"
}
print_quick_stats() {
    local CSV=\$1
    local LABEL=\$2
    [ -f "\$CSV" ] || { echo "  (no CSV to summarize)"; return; }
    python3 -c "
import csv
from statistics import mean, quantiles
rows = [r for r in csv.DictReader(open('\$CSV')) if r.get('status') == '200']
scored  = [r for r in rows if r.get('gt_scored') == 'true']
correct = [r for r in scored if r.get('gt_correct') == 'true']
costs   = [float(r['charged_usd']) for r in rows if r.get('charged_usd')]
lats    = [float(r.get('actual_latency_ms') or r.get('wall_ms') or 0)
           for r in rows if r.get('actual_latency_ms') or r.get('wall_ms')]
ttca_lats = [float(r.get('actual_latency_ms') or r.get('wall_ms') or 0)
             for r in correct if r.get('actual_latency_ms') or r.get('wall_ms')]
n = len(rows)
acc       = len(correct)/len(scored)*100 if scored else 0
cost      = mean(costs) if costs else 0
lat       = mean(lats)/1000 if lats else 0
ttca_mean = mean(ttca_lats)/1000 if ttca_lats else 0
ttca_p95  = quantiles(ttca_lats, n=20)[18]/1000 if len(ttca_lats) >= 20 else (max(ttca_lats)/1000 if ttca_lats else 0)
ctca_costs = [float(r['charged_usd']) for r in correct if r.get('charged_usd')]
ctca = mean(ctca_costs) if ctca_costs else 0
print(f'  Quick stats for \$LABEL:  n={n}  acc={acc:.1f}%  lat={lat:.2f}s  ttca_mean={ttca_mean:.2f}s  ttca_p95={ttca_p95:.2f}s  cost/req=\${cost:.6f}  ctca=\${ctca:.6f}')
" 2>/dev/null || echo "  (stats unavailable)"
}

# ── [0] Generate workload ─────────────────────────────────────────────────────
echo ""
echo "[0] Generating workload (N=${N_REQUESTS}, seed=${SEED})..."
python3 -c "
import json, random
random.seed(${SEED})
data = json.load(open('${DATASET}'))
sample = random.sample(data, min(${N_REQUESTS}, len(data)))
json.dump(sample, open('\$RESULTS_DIR/workload.json', 'w'))
from collections import Counter
by_cell = Counter((x.get('domain'), x.get('complexity')) for x in sample)
DOMAINS = ['factual', 'math', 'code', 'reasoning']
COMPLEXITIES = ['easy', 'medium', 'hard']
print(f'  Workload: {len(sample)} requests')
print(f\"    {'':12} {'easy':>6} {'medium':>8} {'hard':>6} {'total':>7}\")
for d in DOMAINS:
    row = [by_cell.get((d, c), 0) for c in COMPLEXITIES]
    print(f\"    {d:12} {row[0]:>6} {row[1]:>8} {row[2]:>6} {sum(row):>7}\")
col_tots = [sum(by_cell.get((d,c),0) for d in DOMAINS) for c in COMPLEXITIES]
print(f\"    {'total':12} {col_tots[0]:>6} {col_tots[1]:>8} {col_tots[2]:>6} {len(sample):>7}\")
"

# ── [1] Baselines phase (computed once, shared across all alpha runs) ──────────
echo ""
echo "=================================================================="
echo "  [1] Baselines phase (eval_matrix + all baselines, computed once)"
echo "=================================================================="

# Ensure config is at repo defaults before starting
git checkout semantic_router/config.py

start_static_fleet "baselines"

echo ""
echo "[1a] Pre-evaluating all 6 models → eval_matrix.csv..."
python tests/eval_all_models.py \
    --dataset     "\$RESULTS_DIR/workload.json" \
    --output      "\$RESULTS_DIR/eval_matrix.csv" \
    --concurrency 20 \
    --node2-host  "\$NODE2"
echo "  eval_matrix.csv done."

echo ""
echo "[1b] Building optimal tier maps..."
python tests/build_optimal_tier.py \
    --eval-matrix "\$RESULTS_DIR/eval_matrix.csv" \
    --output      "\$RESULTS_DIR/optimal_tier_maps.json" \
    --alpha 1.0 --beta 0.0

echo ""
echo "[1c] Tier-Optimal-Cost oracle..."
python tests/baseline_cost_optimal.py \
    --eval-matrix "\$RESULTS_DIR/eval_matrix.csv" \
    --output      "\$RESULTS_DIR/baseline_cost_optimal.csv"

echo ""
echo "[1d] Tier-Optimal-Acc oracle..."
python tests/baseline_complexity_tier.py \
    --dataset     "\$RESULTS_DIR/workload.json" \
    --tier-map    "\$RESULTS_DIR/optimal_tier_maps.json" \
    --tier-variant accuracy_optimal \
    --concurrency ${CONCURRENCY} \
    --output      "\$RESULTS_DIR/baseline_tier_acc_optimal.csv"

echo ""
echo "[1e] CARROT baseline (mu=0.3)..."
python tests/baseline_carrot.py \
    --dataset     "\$RESULTS_DIR/workload.json" \
    --eval-matrix "\$RESULTS_DIR/eval_matrix.csv" \
    --mu          0.3 \
    --concurrency ${CONCURRENCY} \
    --node2-host  "\$NODE2" \
    --output      "\$RESULTS_DIR/baseline_carrot.csv"

echo ""
echo "[1f] OmniRouter baseline (alpha=0.75)..."
python tests/baseline_omni_router.py \
    --dataset     "\$RESULTS_DIR/workload.json" \
    --eval-matrix "\$RESULTS_DIR/eval_matrix.csv" \
    --alpha       0.75 \
    --concurrency ${CONCURRENCY} \
    --node2-host  "\$NODE2" \
    --output      "\$RESULTS_DIR/baseline_omni_router.csv"

echo ""
echo "[1g] Cascade baseline (threshold=0.80)..."
python tests/baseline_cascade.py \
    --dataset     "\$RESULTS_DIR/workload.json" \
    --priors      "${PRIORS}" \
    --threshold   0.80 \
    --concurrency ${CONCURRENCY} \
    --output      "\$RESULTS_DIR/baseline_cascade.csv"

echo ""
echo "[1h] Round-Robin baseline..."
python tests/round_robin_test.py \
    --dataset     "\$RESULTS_DIR/workload.json" \
    --requests    ${N_REQUESTS} \
    --concurrency ${CONCURRENCY} \
    --node2-host  "\$NODE2" \
    --output      "\$RESULTS_DIR/rr_baseline.csv"

echo ""
echo "[1i] Teardown baselines fleet..."
kill_all_models
echo "  Waiting 30s for GPU memory to free..."
sleep 30

# ── [2] Alpha sweep ───────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  [2] TTCA Alpha Sweep  (β fixed at 0.0)"
echo "=================================================================="

ALPHA_NUM=0
for ALPHA in \$ALPHAS; do
    ALPHA_NUM=\$((ALPHA_NUM + 1))
    ALPHA_LABEL=\$(echo "\$ALPHA" | tr '.' '_')
    OUT_CSV="\$RESULTS_DIR/alpha_\${ALPHA_LABEL}_results.csv"

    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    echo "  Alpha run \$ALPHA_NUM: α = \$ALPHA  (all domains uniform, β=0)"
    echo "══════════════════════════════════════════════════════════════════"

    # Patch all four per-domain alpha values to the same ALPHA
    echo "  Patching semantic_router/config.py (α_uniform=\$ALPHA, β=0.0)..."
    sed -i "s/^TTCA_ALPHA_FACTUAL: float = .*/TTCA_ALPHA_FACTUAL: float = \${ALPHA}/"   semantic_router/config.py
    sed -i "s/^TTCA_ALPHA_MATH: float = .*/TTCA_ALPHA_MATH: float = \${ALPHA}/"         semantic_router/config.py
    sed -i "s/^TTCA_ALPHA_CODE: float = .*/TTCA_ALPHA_CODE: float = \${ALPHA}/"         semantic_router/config.py
    sed -i "s/^TTCA_ALPHA_REASONING: float = .*/TTCA_ALPHA_REASONING: float = \${ALPHA}/" semantic_router/config.py
    # Ensure β=0 (cost-blind) throughout the alpha sweep
    sed -i "s/^TTCA_COST_BETA: float = .*/TTCA_COST_BETA: float = 0.0/"                semantic_router/config.py
    grep -E "TTCA_ALPHA|TTCA_COST_BETA" semantic_router/config.py | head -5 | sed 's/^/    /'

    start_static_fleet "alpha_\${ALPHA_LABEL}"

    warmup_priors

    echo ""
    echo "  Running load test (TTCA, α=\$ALPHA, N=${N_REQUESTS})..."
    START_TS=\$(date +%s)
    python tests/load_test.py \
        --dataset     "\$RESULTS_DIR/workload.json" \
        --router      "\$ROUTER_URL" \
        --mode        ttca \
        --requests    ${N_REQUESTS} \
        --concurrency ${CONCURRENCY} \
        ${RATE_FLAG} \
        --output      "\$OUT_CSV" \
        2>&1 | tee "\$RESULTS_DIR/alpha_\${ALPHA_LABEL}_load.log"
    END_TS=\$(date +%s)
    [ -f "\$OUT_CSV" ] || echo "  WARNING: no output CSV — check alpha_\${ALPHA_LABEL}_load.log"
    echo "  Load test wall time: \$((END_TS - START_TS))s"

    print_quick_stats "\$OUT_CSV" "α=\$ALPHA"

    echo ""
    echo "  Teardown α=\$ALPHA fleet..."
    kill_all_models
    echo "  Waiting 30s for GPU memory to free..."
    sleep 30
done

# Restore config.py to repo defaults
git checkout semantic_router/config.py
echo "  config.py restored to repo version."

# ── [3] Compare all alphas + baselines ───────────────────────────────────────
echo ""
echo "=================================================================="
echo "  [3] Comparing all alphas + baselines"
echo "=================================================================="

COMPARE_ARGS=(tests/compare_all.py)
[ -f "\$RESULTS_DIR/rr_baseline.csv" ] && \
    COMPARE_ARGS+=(--system "Round-Robin:\$RESULTS_DIR/rr_baseline.csv")
[ -f "\$RESULTS_DIR/baseline_cascade.csv" ] && \
    COMPARE_ARGS+=(--system "Cascade:\$RESULTS_DIR/baseline_cascade.csv")
[ -f "\$RESULTS_DIR/baseline_carrot.csv" ] && \
    COMPARE_ARGS+=(--system "CARROT:\$RESULTS_DIR/baseline_carrot.csv")
[ -f "\$RESULTS_DIR/baseline_omni_router.csv" ] && \
    COMPARE_ARGS+=(--system "OmniRouter:\$RESULTS_DIR/baseline_omni_router.csv")

for ALPHA in \$ALPHAS; do
    ALPHA_LABEL=\$(echo "\$ALPHA" | tr '.' '_')
    [ -f "\$RESULTS_DIR/alpha_\${ALPHA_LABEL}_results.csv" ] && \
        COMPARE_ARGS+=(--system "TTCA a=\${ALPHA}:\$RESULTS_DIR/alpha_\${ALPHA_LABEL}_results.csv")
done

[ -f "\$RESULTS_DIR/baseline_tier_acc_optimal.csv" ] && \
    COMPARE_ARGS+=(--system "Tier-Opt-Acc:\$RESULTS_DIR/baseline_tier_acc_optimal.csv")
[ -f "\$RESULTS_DIR/baseline_cost_optimal.csv" ] && \
    COMPARE_ARGS+=(--system "Tier-Opt-Cost:\$RESULTS_DIR/baseline_cost_optimal.csv")

# Reference = α=1.0 (paper default)
if [ -f "\$RESULTS_DIR/alpha_1_0_results.csv" ]; then
    COMPARE_ARGS+=(--ref "TTCA a=1.0")
fi

COMPARE_ARGS+=(--eval-matrix "\$RESULTS_DIR/eval_matrix.csv")
COMPARE_ARGS+=(--output "\$RESULTS_DIR/compare_alpha_sweep.csv")

echo "  Running compare_all.py..."
python "\${COMPARE_ARGS[@]}" 2>&1 | tee "\$RESULTS_DIR/compare_alpha_sweep.txt"

# ── [4] Summary table ─────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  Alpha sweep summary: accuracy vs TTCA latency vs CTCA"
echo "=================================================================="
python3 -c "
import csv, os
from statistics import mean, quantiles
rd = '\$RESULTS_DIR'
alphas = '\$ALPHAS'.split()
rows_data = []
for alpha in alphas:
    label = alpha.replace('.', '_')
    path = os.path.join(rd, f'alpha_{label}_results.csv')
    if not os.path.exists(path):
        continue
    rows = [r for r in csv.DictReader(open(path)) if r.get('status') == '200']
    scored   = [r for r in rows if r.get('gt_scored') == 'true']
    correct  = [r for r in scored if r.get('gt_correct') == 'true']
    lats     = [float(r.get('actual_latency_ms') or r.get('wall_ms') or 0)
                for r in rows if r.get('actual_latency_ms') or r.get('wall_ms')]
    ttca_lats = [float(r.get('actual_latency_ms') or r.get('wall_ms') or 0)
                 for r in correct if r.get('actual_latency_ms') or r.get('wall_ms')]
    costs    = [float(r['charged_usd']) for r in rows if r.get('charged_usd')]
    acc       = len(correct)/len(scored)*100 if scored else 0
    lat_mean  = mean(lats)/1000 if lats else 0
    lat_p95   = quantiles(lats, n=20)[18]/1000 if len(lats) >= 20 else 0
    ttca_mean = mean(ttca_lats)/1000 if ttca_lats else 0
    ttca_p95  = quantiles(ttca_lats, n=20)[18]/1000 if len(ttca_lats) >= 20 else (max(ttca_lats)/1000 if ttca_lats else 0)
    cost      = mean(costs) if costs else 0
    ctca_costs = [float(r['charged_usd']) for r in correct if r.get('charged_usd')]
    ctca      = mean(ctca_costs) if ctca_costs else 0
    rows_data.append((alpha, len(rows), acc, lat_mean, lat_p95, ttca_mean, ttca_p95, cost, ctca))

print(f\"  {'Alpha':>6}  {'N':>5}  {'Accuracy':>9}  {'Lat Mean':>9}  {'Lat P95':>8}  {'TTCA Mean':>10}  {'TTCA P95':>9}  {'Cost/req':>12}  {'CTCA':>12}\")
print(f\"  {'-'*98}\")
for alpha, n, acc, lat_mean, lat_p95, ttca_mean, ttca_p95, cost, ctca in rows_data:
    print(f\"  {alpha:>6}  {n:>5}  {acc:>8.1f}%  {lat_mean:>8.2f}s  {lat_p95:>7.2f}s  {ttca_mean:>9.2f}s  {ttca_p95:>8.2f}s  \${cost:.8f}  \${ctca:.8f}\")
if len(rows_data) >= 2:
    ref = next((r for r in rows_data if r[0] == '1.0'), rows_data[0])
    print()
    print(f\"  vs α={ref[0]} (paper default):  acc delta   ttca_mean delta   ttca_p95 delta   ctca delta\")
    for alpha, n, acc, lat_mean, lat_p95, ttca_mean, ttca_p95, cost, ctca in rows_data:
        if alpha == ref[0]: continue
        da   = acc - ref[2]
        dtm  = ttca_mean - ref[5]
        dtp  = ttca_p95  - ref[6]
        dctca = ctca - ref[8]
        sign_a  = '+' if da   >= 0 else ''
        sign_tm = '+' if dtm  >= 0 else ''
        sign_tp = '+' if dtp  >= 0 else ''
        sign_c  = '+' if dctca >= 0 else ''
        print(f\"  α={alpha:>5}          {sign_a}{da:.1f}pp        {sign_tm}{dtm:.2f}s           {sign_tp}{dtp:.2f}s         {sign_c}\${dctca:.8f}\")
" 2>/dev/null || echo "  (summary unavailable)"

echo ""
echo "=================================================================="
echo "  Alpha sweep complete!  \$(date)"
echo "  Results: \$RESULTS_DIR/"
echo ""
echo "  To visualize (after pulling updated code locally):"
echo "    python tests/plot_sensitivity.py \\"
echo "      --alpha-dir \$RESULTS_DIR \\"
echo "      --beta-dir  results/beta_sweep_<TIMESTAMP>"
echo ""
echo "  To re-run compare_all.py:"
echo "    RDIR=\$RESULTS_DIR"
echo "    python tests/compare_all.py \\"
for ALPHA in \$ALPHAS; do
    ALPHA_LABEL=\$(echo "\$ALPHA" | tr '.' '_')
    echo "      --system \"TTCA a=\${ALPHA}:\\\$RDIR/alpha_\${ALPHA_LABEL}_results.csv\" \\"
done
echo "      --system \"CARROT:\$RESULTS_DIR/baseline_carrot.csv\" \\"
echo "      --eval-matrix \$RESULTS_DIR/eval_matrix.csv"
echo "=================================================================="
PBSEOF

N_ALPHAS=$(echo "$ALPHAS_RAW" | wc -w | tr -d ' ')
echo "Submitting TTCA alpha sweep..."
echo "  ALPHAS      : $ALPHAS_RAW  ($N_ALPHAS values)"
echo "  BETA (fixed): 0.0  (cost-blind)"
echo "  N_REQUESTS  : $N_REQUESTS  (per alpha run)"
echo "  CONCURRENCY : $CONCURRENCY"
echo "  RATE        : ${RATE:-closed-loop} req/s"
echo "  Dataset     : $DATASET  (seed=$SEED)"
echo "  Walltime    : 14:00:00  (baselines ~2h + ${N_ALPHAS} alphas × ~45min each)"
echo "  Log dir     : $LOG_DIR/"
echo ""

JOBID=$(qsub "$PBSSCRIPT")
rm -f "$PBSSCRIPT"

echo "Submitted: $JOBID"
echo ""
echo "Monitor:"
echo "  qstat -u yuping"
echo "  tail -f $LOG_DIR/job.out"
