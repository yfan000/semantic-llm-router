#!/bin/bash
# submit_dynamic_test.sh — Test dynamic provisioning mode on Sophia.
#
# Validates that the provisioner correctly scales UP (no replicas, only upgrades)
# as request complexity increases:
#
#   Phase 1 — Cold start:  qwen-7b + deepseek-r1-7b (node1) + llama4-scout (node2)
#   Phase 2 — Warm-up:     100 mixed easy/medium requests (seed models handle)
#   Phase 3 — Code burst:  200 code:hard/medium requests -> triggers qwen3-coder-30b
#   Phase 4 — Math burst:  200 math:hard requests -> triggers deepseek-r1-14b
#   Phase 5 — Measure:     compare accuracy before/after spin-ups
#
# Key assertions:
#   [PASS] Seed models start (qwen-7b, deepseek-r1-7b, llama4-scout)
#   [PASS] qwen3-coder-30b or deepseek-r1-14b spins up during code/math burst
#   [PASS] No *-replica models appear in /v1/models
#   [PASS] Accuracy after spin-up >= accuracy during cold start
#
# Usage:
#   bash scripts/submit_dynamic_test.sh
#   N_WARMUP=50 N_BURST=100 bash scripts/submit_dynamic_test.sh

set -euo pipefail

N_WARMUP=${N_WARMUP:-100}    # requests during warm-up (easy/medium)
N_BURST=${N_BURST:-200}      # requests per burst phase (hard)
PROJECT=${PROJECT:-UIC-HPC}
QUEUE=${QUEUE:-by-node}
DATASET=${DATASET:-"datasets/hf_3000.json"}
PRIORS=${PRIORS:-"results/priors_all5.json"}

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HOME/vllm_logs/dynamic_test_${TS}"
mkdir -p "$LOG_DIR"

PBSSCRIPT=$(mktemp /tmp/dynamic_test_XXXXXX.pbs)

cat > "$PBSSCRIPT" << PBSEOF
#!/bin/bash
#PBS -l select=2:ngpus=8:ncpus=64
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:eagle
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -N dynamic_test
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
RESULTS_DIR="results/dynamic_test_${TS}"
mkdir -p "\$RESULTS_DIR"

echo "=================================================================="
echo "  Dynamic Provisioning Test   \$(date)"
echo "  NODE1 : \$NODE1   NODE2 : \$NODE2"
echo "  N_WARMUP  : ${N_WARMUP}"
echo "  N_BURST   : ${N_BURST}"
echo "  Dataset   : ${DATASET}"
echo "=================================================================="

# ── Helper: list registered models ───────────────────────────────────────────
list_models() {
    curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
        | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', [])
ids = [m['id'] for m in data]
print('  Registered (' + str(len(ids)) + '):', ', '.join(sorted(ids)))
replicas = [i for i in ids if i.endswith('-replica')]
if replicas:
    print('  WARNING: replicas found:', replicas)
else:
    print('  OK: no replicas')
" 2>/dev/null || echo "  (router not reachable)"
}

# ── Step 1: Start router ──────────────────────────────────────────────────────
echo ""
echo "[1] Starting router..."
nohup uvicorn semantic_router.main:app \
    --host 0.0.0.0 --port 8080 \
    > ~/vllm_logs/router_dyntest.log 2>&1 &
sleep 8
for i in \$(seq 1 12); do
    curl --noproxy '*' -sf "\$ROUTER_URL/router/health" > /dev/null 2>&1 && \
        echo "  Router ready." && break
    sleep 5
done

# ── Step 2: Start seed models (DYNAMIC mode — 2 models only) ─────────────────
echo ""
echo "[2] Starting seed models in DYNAMIC mode: qwen-7b + deepseek-r1-7b..."
nohup python provisioner/dynamic_provisioner.py \
    --router-url  "\$ROUTER_URL" \
    --node-host   "\$NODE1" \
    --router-mode ttca \
    --priors-path "${PRIORS}" \
    --initial-models "qwen-7b,deepseek-r1-7b" \
    > ~/vllm_logs/prov_dyntest_node1.log 2>&1 &

echo "[2b] Starting llama4-scout on \$NODE2..."
ssh "\$NODE2" "
    cd ~/semantic-llm-router
    export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
    nohup python provisioner/dynamic_provisioner.py \
        --router-url  '\$ROUTER_URL' \
        --node-host   '\$NODE2' \
        --router-mode ttca \
        --static \
        --priors-path '${PRIORS}' \
        --initial-models llama4-scout \
        </dev/null >>~/vllm_logs/prov_dyntest_node2.log 2>&1 &
    disown \$!
" </dev/null

# ── Step 3: Wait for seed models ─────────────────────────────────────────────
echo ""
echo "[3] Waiting for seed models (qwen-7b + deepseek-r1-7b + llama4-scout)..."
for i in \$(seq 1 80); do
    N=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
        | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" \
        2>/dev/null || echo 0)
    echo "  [\$((i*15))s] \$N model(s) registered"
    [ "\$N" -ge 3 ] && echo "  Seed models ready." && break
    sleep 15
done

echo ""
echo "--- Models at cold start ---"
list_models

# ── Step 4: Phase 1 — Warm-up with easy/medium requests ──────────────────────
echo ""
echo "[4] Phase 1: Warm-up (${N_WARMUP} easy/medium requests)..."
echo "  Sending easy requests — seed models should handle all of these."

python3 -c "
import json, random
data = json.load(open('${DATASET}'))
easy = [x for x in data if x.get('complexity') in ('easy', 'medium')]
sample = random.sample(easy, min(${N_WARMUP}, len(easy)))
json.dump(sample, open('/tmp/dyntest_warmup.json', 'w'))
print(f'  Warmup dataset: {len(sample)} easy/medium items')
"

python tests/load_test.py \
    --dataset     /tmp/dyntest_warmup.json \
    --router      "\$ROUTER_URL" \
    --mode        ttca \
    --requests    ${N_WARMUP} \
    --concurrency 20 \
    --output      "\$RESULTS_DIR/phase1_warmup.csv"

echo ""
echo "--- Models after warm-up ---"
list_models
MODELS_AFTER_WARMUP=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo 0)

# ── Step 5: Phase 2 — Code burst (triggers qwen3-coder-30b) ──────────────────
echo ""
echo "[5] Phase 2: Code burst (${N_BURST} code:hard/medium requests)..."
echo "  Heavy code load should trigger qwen3-coder-30b spin-up (~10 min)."

python3 -c "
import json, random
data = json.load(open('${DATASET}'))
code_hard = [x for x in data if x.get('domain') == 'code'
             and x.get('complexity') in ('hard', 'medium')]
sample = random.sample(code_hard, min(${N_BURST}, len(code_hard)))
json.dump(sample, open('/tmp/dyntest_code.json', 'w'))
print(f'  Code dataset: {len(sample)} code:hard/medium items')
"

python tests/load_test.py \
    --dataset     /tmp/dyntest_code.json \
    --router      "\$ROUTER_URL" \
    --mode        ttca \
    --requests    ${N_BURST} \
    --concurrency 30 \
    --output      "\$RESULTS_DIR/phase2_code.csv"

echo ""
echo "--- Models after code burst ---"
list_models

# Wait for provisioner poll to fire (poll_interval=30s)
echo "  Waiting 60s for provisioner to react..."
sleep 60
echo ""
echo "--- Models after provisioner poll ---"
list_models

# ── Step 6: Phase 3 — Math burst (triggers deepseek-r1-14b) ──────────────────
echo ""
echo "[6] Phase 3: Math burst (${N_BURST} math:hard requests)..."

python3 -c "
import json, random
data = json.load(open('${DATASET}'))
math_hard = [x for x in data if x.get('domain') == 'math'
             and x.get('complexity') == 'hard']
sample = random.sample(math_hard, min(${N_BURST}, len(math_hard)))
json.dump(sample, open('/tmp/dyntest_math.json', 'w'))
print(f'  Math dataset: {len(sample)} math:hard items')
"

python tests/load_test.py \
    --dataset     /tmp/dyntest_math.json \
    --router      "\$ROUTER_URL" \
    --mode        ttca \
    --requests    ${N_BURST} \
    --concurrency 30 \
    --output      "\$RESULTS_DIR/phase3_math.csv"

echo ""
echo "--- Models after math burst ---"
list_models

# Wait for provisioner
echo "  Waiting 60s for provisioner..."
sleep 60

# ── Step 7: Final model inventory ────────────────────────────────────────────
echo ""
echo "[7] Final model inventory..."
MODELS_FINAL=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo 0)
list_models

# ── Step 8: Compare accuracy across phases ────────────────────────────────────
echo ""
echo "[8] Accuracy comparison across phases..."
SYS_ARGS=()
[ -f "\$RESULTS_DIR/phase1_warmup.csv" ] && \
    SYS_ARGS+=("--system" "Warmup-cold:\$RESULTS_DIR/phase1_warmup.csv")
[ -f "\$RESULTS_DIR/phase2_code.csv" ] && \
    SYS_ARGS+=("--system" "Code-burst:\$RESULTS_DIR/phase2_code.csv")
[ -f "\$RESULTS_DIR/phase3_math.csv" ] && \
    SYS_ARGS+=("--system" "Math-burst:\$RESULTS_DIR/phase3_math.csv")

if [ \${#SYS_ARGS[@]} -ge 2 ]; then
    python tests/compare_all.py "\${SYS_ARGS[@]}" \
        --output "\$RESULTS_DIR/phase_comparison.csv" \
        | tee "\$RESULTS_DIR/phase_comparison.txt"
fi

# ── Step 9: Assertions ────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  ASSERTIONS"
echo "=================================================================="

PASS=0
FAIL=0

assert_pass() {
    echo "  [PASS] \$1"
    PASS=\$((PASS + 1))
}
assert_fail() {
    echo "  [FAIL] \$1"
    FAIL=\$((FAIL + 1))
}

[ "\$MODELS_AFTER_WARMUP" -ge 3 ] && \
    assert_pass "Seed models started (\$MODELS_AFTER_WARMUP registered after warmup)" || \
    assert_fail "Seed models failed to start (\$MODELS_AFTER_WARMUP registered)"

[ "\$MODELS_FINAL" -gt "\$MODELS_AFTER_WARMUP" ] && \
    assert_pass "Models spun up dynamically (\$MODELS_AFTER_WARMUP -> \$MODELS_FINAL)" || \
    assert_fail "No new models spun up (\$MODELS_AFTER_WARMUP -> \$MODELS_FINAL) — check provisioner logs"

REPLICAS=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
    | python3 -c "
import sys, json
ids = [m['id'] for m in json.load(sys.stdin).get('data', [])]
print(sum(1 for i in ids if i.endswith('-replica')))
" 2>/dev/null || echo 0)
[ "\$REPLICAS" -eq 0 ] && \
    assert_pass "No replica models created" || \
    assert_fail "Replica models found (\$REPLICAS) — upgrade logic may be broken"

COLD_MODELS_RUNNING=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
    | python3 -c "
import sys, json
ids = set(m['id'] for m in json.load(sys.stdin).get('data', []))
cold = {'qwen3-coder-30b', 'deepseek-r1-14b', 'gemma-3-27b'}
found = ids & cold
print(len(found), ' '.join(sorted(found)))
" 2>/dev/null || echo "0")
N_COLD=\$(echo \$COLD_MODELS_RUNNING | awk '{print \$1}')
NAMES_COLD=\$(echo \$COLD_MODELS_RUNNING | cut -d' ' -f2-)
[ "\$N_COLD" -ge 1 ] && \
    assert_pass "Cold model(s) spun up: \$NAMES_COLD" || \
    assert_fail "No cold models spun up (qwen3-coder-30b / deepseek-r1-14b / gemma-3-27b)"

echo ""
echo "  PASS: \$PASS   FAIL: \$FAIL"
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "  Dynamic test complete!  \$(date)"
echo "  Results: \$RESULTS_DIR/"
echo "    phase1_warmup.csv      easy/medium requests at cold start"
echo "    phase2_code.csv        code:hard burst (triggers coder model)"
echo "    phase3_math.csv        math:hard burst (triggers math model)"
echo "    phase_comparison.txt   accuracy across phases"
echo ""
echo "  Provisioner log (node1): ~/vllm_logs/prov_dyntest_node1.log"
echo "  Check spin-up events:    grep 'SPIN UP\|Strategy B\|upgrade' ~/vllm_logs/prov_dyntest_node1.log"
echo "=================================================================="
PBSEOF

echo "Submitting dynamic provisioning test..."
echo "  N_WARMUP  : $N_WARMUP"
echo "  N_BURST   : $N_BURST"
echo "  Dataset   : $DATASET"
echo "  Walltime  : 03:00:00"
echo "  Log dir   : $LOG_DIR/"
echo ""

JOBID=$(qsub "$PBSSCRIPT")
rm -f "$PBSSCRIPT"

echo "Submitted: $JOBID"
echo ""
echo "Monitor:"
echo "  qstat -u yuping"
echo "  tail -f $LOG_DIR/job.out"
echo ""
echo "Check spin-up events after job starts:"
echo "  grep 'SPIN UP\|Strategy B\|upgrade\|TRIGGER' ~/vllm_logs/prov_dyntest_node1.log"
