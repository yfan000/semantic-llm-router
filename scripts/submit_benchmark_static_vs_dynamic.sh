#!/bin/bash
# submit_benchmark_static_vs_dynamic.sh — Static vs dynamic comparison on the benchmark dataset.
#
# Mirrors submit_static_vs_dynamic.sh but uses benchmark_1000.json
# (GPQA Diamond / MMLU-Pro / GSM1K / OlympiadBench / LiveCodeBench) and
# run_benchmark_eval.py, which scores responses inline (mcq/numeric/expression/code)
# so no separate eval_all_models.py pass is needed.
#
# Static mode:  all 6 models pre-loaded before workload starts.
# Dynamic mode: qwen-7b + deepseek-r1-7b + qwen3-coder-30b seed; others spin up.
#
# Usage:
#   bash scripts/submit_benchmark_static_vs_dynamic.sh
#   N_REQUESTS=500 bash scripts/submit_benchmark_static_vs_dynamic.sh
#
# Parameters:
#   N_REQUESTS   total requests per mode (default 300; max 1000 = full dataset)
#   CONCURRENCY  max simultaneous in-flight requests (default 50)
#   SEED         random seed for workload sampling (default 42)
#   DATASET      path to benchmark JSON (default datasets/benchmark_1000.json)
#   PRIORS       path to priors file for Cascade baseline (default results/priors_all5.json)

set -euo pipefail

N_REQUESTS=${N_REQUESTS:-300}
CONCURRENCY=${CONCURRENCY:-50}
SEED=${SEED:-42}
PROJECT=${PROJECT:-UIC-HPC}
QUEUE=${QUEUE:-by-node}
DATASET=${DATASET:-"datasets/benchmark_1000.json"}
PRIORS=${PRIORS:-"results/priors_all5.json"}

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HOME/vllm_logs/benchmark_svd_${TS}"
mkdir -p "$LOG_DIR"

PBSSCRIPT=$(mktemp /tmp/bsvd_XXXXXX.pbs)

cat > "$PBSSCRIPT" << PBSEOF
#!/bin/bash
#PBS -l select=2:ngpus=8:ncpus=64
#PBS -l walltime=05:00:00
#PBS -l filesystems=home:eagle
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -N benchmark_svd
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
RESULTS_DIR="results/benchmark_svd_${TS}"
mkdir -p "\$RESULTS_DIR"

echo "=================================================================="
echo "  Benchmark Static vs Dynamic Comparison   \$(date)"
echo "  NODE1       : \$NODE1   NODE2 : \$NODE2"
echo "  N_REQUESTS  : ${N_REQUESTS}"
echo "  CONCURRENCY : ${CONCURRENCY}"
echo "  Dataset     : ${DATASET}  (seed=${SEED})"
echo "  Scoring     : inline (mcq / numeric / expression / code)"
echo "  Results     : \$RESULTS_DIR/"
echo "=================================================================="

# ── Build dataset if missing ──────────────────────────────────────────────────
if [ ! -f "${DATASET}" ]; then
    echo ""
    echo "[0] Building benchmark dataset..."
    python tests/build_benchmark_dataset.py --total 1000 --output "${DATASET}"
fi

NITEMS=\$(python3 -c "import json; print(len(json.load(open('${DATASET}'))))" 2>/dev/null || echo "?")
echo "[0] Dataset: ${DATASET}  (\$NITEMS items)"

# ── Generate fixed workload (same for both modes) ─────────────────────────────
echo ""
echo "[0b] Generating fixed workload (N=${N_REQUESTS}, seed=${SEED})..."
python3 -c "
import json, random
random.seed(${SEED})
data = json.load(open('${DATASET}'))
n = ${N_REQUESTS}
if n <= len(data):
    sample = random.sample(data, n)
else:
    # Oversample: full dataset once + random remainder (no item repeated more than twice)
    extra  = random.sample(data, n - len(data))
    sample = data + extra
    random.shuffle(sample)
# Re-assign req_ids to match position in workload
for i, item in enumerate(sample):
    item = dict(item); item['req_id'] = i; sample[i] = item
json.dump(sample, open('\$RESULTS_DIR/workload.json', 'w'), indent=2)
from collections import Counter
by_src    = Counter(x.get('source','?') for x in sample)
by_domain = Counter(x.get('domain','?') for x in sample)
by_cpx    = Counter(x.get('complexity','?') for x in sample)
by_atype  = Counter(x.get('answer_type','?') for x in sample)
print(f'  Workload: {len(sample)} requests')
print(f'  By source     : {dict(sorted(by_src.items()))}')
print(f'  By domain     : {dict(sorted(by_domain.items()))}')
print(f'  By complexity : {dict(sorted(by_cpx.items()))}')
print(f'  By answer_type: {dict(sorted(by_atype.items()))}')
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
    ssh "\$NODE2" "pkill -f 'vllm serve' 2>/dev/null; true" </dev/null 2>/dev/null || true
    echo "  Waiting 60s for node2 GPU memory to free..."
    sleep 60
}
wait_llama4_scout() {
    echo "  Waiting for llama4-scout on \$NODE2:8005 (timeout 90 min)..."
    for i in \$(seq 1 360); do
        if curl --noproxy '*' -sf "http://\$NODE2:8005/health" > /dev/null 2>&1; then
            echo "  llama4-scout ready! (\$((i*15))s elapsed)"; return 0
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
    > ~/vllm_logs/router_bsvd_static.log 2>&1 &
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
    > ~/vllm_logs/prov_bsvd_static_node1.log 2>&1 &

echo "[S2b] Starting llama4-scout on \$NODE2..."
ssh "\$NODE2" "
    cd ~/semantic-llm-router
    export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
    nohup python provisioner/dynamic_provisioner.py \
        --router-url '\$ROUTER_URL' --node-host '\$NODE2' \
        --router-mode ttca --static --priors-path '${PRIORS}' \
        --initial-models llama4-scout \
        </dev/null >>~/vllm_logs/prov_bsvd_static_node2.log 2>&1 &
    disown \$!
" </dev/null

echo ""
echo "[S3] Waiting for all 6 models..."
wait_models 6

# Check llama4-scout registered; register manually if needed
SCOUT_CNT=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(sum(1 for m in d if 'llama4' in m['id']))" 2>/dev/null || echo 0)
if [ "\$SCOUT_CNT" -eq 0 ]; then
    echo "  llama4-scout not registered — waiting for it on \$NODE2:8005..."
    wait_llama4_scout
    if curl --noproxy '*' -sf "http://\$NODE2:8005/health" > /dev/null 2>&1; then
        curl --noproxy '*' -sf -X POST "\$ROUTER_URL/router/register" \
            -H "Content-Type: application/json" \
            -d "{\"model_id\":\"llama4-scout\",\"model_name\":\"meta-llama/Llama-4-Scout-17B-16E-Instruct\",\"backend\":\"vllm\",\"base_url\":\"http://\${NODE2}:8005\",\"domains\":[\"code\",\"factual\",\"math\",\"reasoning\"],\"efficiency_tokens_per_joule\":3.0,\"input_rate_usd_per_token\":0.0000001,\"output_rate_usd_per_token\":0.0000003,\"skip_calibration\":true}" \
            2>/dev/null && echo "  Manually registered llama4-scout." \
            || echo "  WARNING: manual registration failed"
    fi
fi

echo ""
echo "[S3b] Warm-starting router priors from ${PRIORS}..."
curl --noproxy '*' -sf -X POST "\$ROUTER_URL/router/warmup" \
    -H "Content-Type: application/json" \
    -d "{\"eval_matrix_path\": \"${PRIORS}\"}" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  Seeded {r.get(\"cells_seeded\",\"?\")} cells')" \
    2>/dev/null || echo "  (warmup skipped — endpoint not available)"

echo ""
echo "[S4] STATIC workload — run_benchmark_eval.py (${N_REQUESTS} requests, inline scoring)..."
STATIC_START=\$(date +%s)
python tests/run_benchmark_eval.py \
    --dataset     "\$RESULTS_DIR/workload.json" \
    --router      "\$ROUTER_URL" \
    --mode        ttca \
    --concurrency ${CONCURRENCY} \
    --output      "\$RESULTS_DIR/static_results.csv" \
    2>&1 | tee "\$RESULTS_DIR/static_load.log"
STATIC_END=\$(date +%s)
STATIC_WALL=\$((STATIC_END - STATIC_PROV_START))
echo "  Static experiment time: \${STATIC_WALL}s"

# ── Baselines (all 6 models still live) ──────────────────────────────────────
echo ""
echo "=================================================================="
echo "  BASELINES (all 6 models alive)"
echo "=================================================================="

echo ""
echo "  [B0] Building eval_matrix (all models × all queries)..."
python tests/eval_all_models.py \
    --dataset     "\$RESULTS_DIR/workload.json" \
    --output      "\$RESULTS_DIR/eval_matrix.csv" \
    --concurrency 20 \
    --node2-host  "\$NODE2" \
    2>&1 | tee "\$RESULTS_DIR/eval_matrix.log" || echo "  WARNING: eval_matrix generation failed"
echo "  eval_matrix.csv done."

EVAL_MATRIX_FLAG=""
[ -f "\$RESULTS_DIR/eval_matrix.csv" ] && EVAL_MATRIX_FLAG="--eval-matrix \$RESULTS_DIR/eval_matrix.csv"

echo ""
echo "  [B1] CARROT baseline..."
python tests/baseline_carrot.py \
    --dataset     "\$RESULTS_DIR/workload.json" \
    \$EVAL_MATRIX_FLAG \
    --concurrency ${CONCURRENCY} \
    --output      "\$RESULTS_DIR/baseline_carrot.csv" \
    2>&1 | tee "\$RESULTS_DIR/carrot.log" || echo "  WARNING: CARROT failed"

echo ""
echo "  [B2] OmniRouter baseline (alpha=0.75)..."
python tests/baseline_omni_router.py \
    --dataset     "\$RESULTS_DIR/workload.json" \
    \$EVAL_MATRIX_FLAG \
    --alpha       0.75 \
    --concurrency ${CONCURRENCY} \
    --output      "\$RESULTS_DIR/baseline_omni_router.csv" \
    2>&1 | tee "\$RESULTS_DIR/omni_router.log" || echo "  WARNING: OmniRouter failed"

echo ""
echo "  [B3] Cascade baseline (threshold=0.80)..."
if [ -f "${PRIORS}" ]; then
    python tests/baseline_cascade.py \
        --dataset     "\$RESULTS_DIR/workload.json" \
        --priors      "${PRIORS}" \
        --threshold   0.80 \
        --concurrency ${CONCURRENCY} \
        --output      "\$RESULTS_DIR/baseline_cascade.csv" \
        2>&1 | tee "\$RESULTS_DIR/cascade.log" || echo "  WARNING: Cascade failed"
else
    echo "  Cascade skipped — priors not found: ${PRIORS}"
    echo "  (Generate with: python tests/extract_priors.py --eval-matrix <eval_matrix.csv>)"
fi

echo ""
echo "  [B4] Round-Robin baseline..."
python tests/round_robin_test.py \
    --dataset     "\$RESULTS_DIR/workload.json" \
    --requests    ${N_REQUESTS} \
    --concurrency ${CONCURRENCY} \
    --output      "\$RESULTS_DIR/rr_baseline.csv" \
    2>&1 | tee "\$RESULTS_DIR/rr.log" || echo "  WARNING: Round-Robin failed"

echo ""
echo "[S5] Tearing down static mode..."
kill_all_models
wait_node2_gpus_free
sleep 5

# ════════════════════════════════════════════════════════════════════
# MODE 2: DYNAMIC — seed models only; others spin up on demand
# ════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  MODE 2: DYNAMIC (qwen-7b + deepseek-r1-7b + qwen3-coder-30b seeds)"
echo "══════════════════════════════════════════════════════════════════"

echo ""
echo "[D1] Starting router..."
nohup uvicorn semantic_router.main:app \
    --host 0.0.0.0 --port 8080 \
    > ~/vllm_logs/router_bsvd_dynamic.log 2>&1 &
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
    > ~/vllm_logs/prov_bsvd_dynamic_node1.log 2>&1 &

ssh "\$NODE2" "
    cd ~/semantic-llm-router
    export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
    nohup python provisioner/dynamic_provisioner.py \
        --router-url '\$ROUTER_URL' --node-host '\$NODE2' \
        --router-mode ttca --static --priors-path '${PRIORS}' \
        --initial-models llama4-scout \
        </dev/null >>~/vllm_logs/prov_bsvd_dynamic_node2.log 2>&1 &
    disown \$!
" </dev/null

echo ""
echo "[D3] Waiting for 3 seed models..."
wait_models 3
echo "[D3b] Waiting for llama4-scout on \$NODE2..."
wait_llama4_scout

echo ""
echo "[D3c] Warm-up: 50 easy requests to seed reputation tracker..."
python3 -c "
import json, random
random.seed(99)
data = json.load(open('\$RESULTS_DIR/workload.json'))
easy = [x for x in data if x.get('complexity') == 'easy']
sample = random.sample(easy, min(50, len(easy)))
json.dump(sample, open('/tmp/bsvd_warmup.json', 'w'))
print(f'  Warm-up: {len(sample)} easy requests')
"
python tests/run_benchmark_eval.py \
    --dataset     /tmp/bsvd_warmup.json \
    --router      "\$ROUTER_URL" \
    --mode        ttca \
    --concurrency 10 \
    --output      /dev/null
echo "  Warm-up done."

echo ""
echo "[D4] DYNAMIC workload — run_benchmark_eval.py (${N_REQUESTS} requests, inline scoring)..."
DYNAMIC_START=\$(date +%s)
python tests/run_benchmark_eval.py \
    --dataset     "\$RESULTS_DIR/workload.json" \
    --router      "\$ROUTER_URL" \
    --mode        ttca \
    --concurrency ${CONCURRENCY} \
    --output      "\$RESULTS_DIR/dynamic_results.csv" \
    2>&1 | tee "\$RESULTS_DIR/dynamic_load.log"
DYNAMIC_END=\$(date +%s)
DYNAMIC_WALL=\$((DYNAMIC_END - DYNAMIC_PROV_START))
echo "  Dynamic experiment time: \${DYNAMIC_WALL}s"

DYNAMIC_SPINUPS=\$(grep "SPIN UP" ~/vllm_logs/prov_bsvd_dynamic_node1.log 2>/dev/null \
    | grep -v "reason=initial" | awk '{print \$3}' | sort -u | tr '\n' ',' | sed 's/,\$//')
echo "  Models dynamically spun up: \${DYNAMIC_SPINUPS:-none}"

echo ""
echo "[D5] Tearing down dynamic mode..."
kill_all_models

# ════════════════════════════════════════════════════════════════════
# COMPARISON
# ════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  COMPARISON"
echo "══════════════════════════════════════════════════════════════════"

echo ""
echo "  Experiment wall times:"
echo "    Static  total: \${STATIC_WALL}s  (provisioner start → load test end)"
echo "    Dynamic total: \${DYNAMIC_WALL}s"
echo "    Static  load test only: \$((STATIC_END  - STATIC_START))s"
echo "    Dynamic load test only: \$((DYNAMIC_END - DYNAMIC_START))s"

# Build --system args for compare_all.py
COMPARE_ARGS=(tests/compare_all.py)
[ -f "\$RESULTS_DIR/rr_baseline.csv" ]            && COMPARE_ARGS+=(--system "Round-Robin:\$RESULTS_DIR/rr_baseline.csv")
[ -f "\$RESULTS_DIR/baseline_cascade.csv" ]       && COMPARE_ARGS+=(--system "Cascade:\$RESULTS_DIR/baseline_cascade.csv")
[ -f "\$RESULTS_DIR/baseline_carrot.csv" ]        && COMPARE_ARGS+=(--system "CARROT:\$RESULTS_DIR/baseline_carrot.csv")
[ -f "\$RESULTS_DIR/baseline_omni_router.csv" ]   && COMPARE_ARGS+=(--system "OmniRouter:\$RESULTS_DIR/baseline_omni_router.csv")
[ -f "\$RESULTS_DIR/static_results.csv" ]         && COMPARE_ARGS+=(--system "Static (TTCA):\$RESULTS_DIR/static_results.csv")
[ -f "\$RESULTS_DIR/dynamic_results.csv" ]        && COMPARE_ARGS+=(--system "Dynamic (TTCA):\$RESULTS_DIR/dynamic_results.csv")
COMPARE_ARGS+=(--ref "Static (TTCA)")
COMPARE_ARGS+=(--output "\$RESULTS_DIR/compare_all_systems.csv")

echo ""
echo "  [compare_all.py] Ranking all systems..."
python "\${COMPARE_ARGS[@]}" 2>&1 | tee "\$RESULTS_DIR/compare_all_systems.txt"

# ── Per-benchmark accuracy table ──────────────────────────────────────────────
echo ""
echo "  [Per-benchmark accuracy] Static vs Dynamic..."
python3 << PYEOF
import csv, os
from collections import defaultdict

rd = "\$RESULTS_DIR"

def load_scored(path):
    try:
        return [r for r in csv.DictReader(open(path)) if r.get('is_correct') in ('true','false')]
    except FileNotFoundError:
        return []

static  = load_scored(rd + '/static_results.csv')
dynamic = load_scored(rd + '/dynamic_results.csv')

sources = sorted({r.get('source','?') for r in static + dynamic})
W = 26

print()
print(f"  {'Source':<{W}} {'Static Acc':>12} {'Dynamic Acc':>12} {'Delta':>8}")
print(f"  {'-'*(W+35)}")
for src in sources:
    s_rows = [r for r in static  if r.get('source') == src]
    d_rows = [r for r in dynamic if r.get('source') == src]
    s_acc  = sum(1 for r in s_rows if r['is_correct']=='true') / len(s_rows) * 100 if s_rows else None
    d_acc  = sum(1 for r in d_rows if r['is_correct']=='true') / len(d_rows) * 100 if d_rows else None
    s_s    = f"{s_acc:.1f}% (n={len(s_rows)})" if s_acc is not None else '—'
    d_s    = f"{d_acc:.1f}% (n={len(d_rows)})" if d_acc is not None else '—'
    dl_s   = f"{d_acc-s_acc:+.1f}pp" if s_acc is not None and d_acc is not None else '—'
    print(f"  {src:<{W}} {s_s:>12} {d_s:>12} {dl_s:>8}")

# Overall
s_acc  = sum(1 for r in static  if r['is_correct']=='true') / len(static)  * 100 if static  else None
d_acc  = sum(1 for r in dynamic if r['is_correct']=='true') / len(dynamic) * 100 if dynamic else None
s_s    = f"{s_acc:.1f}% (n={len(static)})"   if s_acc is not None else '—'
d_s    = f"{d_acc:.1f}% (n={len(dynamic)})"  if d_acc is not None else '—'
dl_s   = f"{d_acc-s_acc:+.1f}pp" if s_acc is not None and d_acc is not None else '—'
print(f"  {'OVERALL':<{W}} {s_s:>12} {d_s:>12} {dl_s:>8}")
print()
PYEOF

# ── GPU energy comparison ─────────────────────────────────────────────────────
echo ""
echo "  [GPU energy comparison]"
python3 tests/compute_gpu_energy.py \
    --static-log          ~/vllm_logs/prov_bsvd_static_node1.log \
    --static-wall         \$STATIC_WALL \
    --static-start-epoch  \$STATIC_PROV_START \
    --dynamic-log         ~/vllm_logs/prov_bsvd_dynamic_node1.log \
    --dynamic-wall        \$DYNAMIC_WALL \
    --dynamic-start-epoch \$DYNAMIC_PROV_START \
    2>/dev/null | tee "\$RESULTS_DIR/gpu_energy_comparison.txt" \
    || echo "  (compute_gpu_energy.py not available — skipping energy comparison)"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  Done!  \$(date)"
echo "  Results: \$RESULTS_DIR/"
echo ""
echo "  Files:"
echo "    static_results.csv         Static TTCA (inline scored)"
echo "    dynamic_results.csv        Dynamic TTCA (inline scored)"
echo "    baseline_carrot.csv        CARROT baseline"
echo "    baseline_omni_router.csv   OmniRouter baseline"
echo "    baseline_cascade.csv       Cascade baseline"
echo "    rr_baseline.csv            Round-Robin baseline"
echo "    compare_all_systems.txt    Side-by-side comparison (human readable)"
echo "    compare_all_systems.csv    Side-by-side comparison (CSV)"
echo "    gpu_energy_comparison.txt  GPU energy (idle + serving)"
echo ""
echo "  Re-run comparison only (no traffic):"
echo "    RESULTS_DIR=\$RESULTS_DIR COMPARE_ONLY=1 bash scripts/run_benchmark_comparison.sh"
echo "=================================================================="
PBSEOF

echo "Submitting benchmark static vs dynamic comparison..."
echo "  N_REQUESTS  : $N_REQUESTS"
echo "  CONCURRENCY : $CONCURRENCY"
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
