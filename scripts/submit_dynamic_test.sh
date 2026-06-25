#!/bin/bash
# submit_dynamic_test.sh — Test dynamic provisioning mode on Sophia.
#
# Validates that the provisioner correctly scales UP (no replicas, only upgrades)
# as request complexity increases:
#
#   Phase 1 — Cold start:  qwen-7b + deepseek-r1-7b (node1) + llama4-scout (node2)
#   Phase 2 — Warm-up:     100 easy/medium requests at low concurrency (seed models handle)
#   Phase 3 — Code burst:  300 code:hard/medium at high concurrency -> triggers qwen3-coder-30b
#   Phase 4 — Math burst:  300 math:hard at high concurrency -> triggers deepseek-r1-14b
#   Phase 5 — Measure:     compare accuracy before/after spin-ups
#
# Key assertions:
#   [PASS] Seed models start (qwen-7b, deepseek-r1-7b, llama4-scout)
#   [PASS] At least 1 non-seed model was spun up (checked via log, not live count)
#   [PASS] No *-replica models appear in /v1/models
#   [PASS] A cold model (qwen3-coder-30b / deepseek-r1-14b) appeared
#
# Note on assertion design: intermediate models can be spun UP then DOWN (idle timeout)
# before assertions run, so the live model count (MODELS_FINAL) may equal MODELS_AFTER_WARMUP
# even when upgrades worked correctly. The log-based SPINUP_COUNT is the true indicator.
#
# Why high concurrency for bursts:
#   Provisioner triggers on: num_requests_running > 20 OR kv_cache > 70%.
#   At concurrency=80, deepseek-r1-7b (1 GPU) saturates on hard requests,
#   pushing num_requests_running above RUNNING_THRESHOLD=20 -> provisioner fires.
#
# Usage:
#   bash scripts/submit_dynamic_test.sh
#   N_BURST=500 CONCURRENCY_BURST=100 bash scripts/submit_dynamic_test.sh

set -euo pipefail

N_WARMUP=${N_WARMUP:-100}    # requests during warm-up (easy/medium)
N_BURST=${N_BURST:-300}      # requests per burst phase (hard)
CONCURRENCY_WARMUP=${CONCURRENCY_WARMUP:-20}   # low — seed models handle easily
CONCURRENCY_BURST=${CONCURRENCY_BURST:-80}     # high — must saturate vLLM to trigger provisioner
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
echo "  N_WARMUP          : ${N_WARMUP}  (concurrency=${CONCURRENCY_WARMUP})"
echo "  N_BURST           : ${N_BURST}  (concurrency=${CONCURRENCY_BURST})"
echo "  Dataset           : ${DATASET}"
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

# ── Step 4: Phase 1 — Warm-up (low concurrency, seed models handle) ──────────
echo ""
echo "[4] Phase 1: Warm-up (${N_WARMUP} easy/medium, concurrency=${CONCURRENCY_WARMUP})..."
echo "  Low concurrency — seed models handle without triggering provisioner."

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
    --concurrency ${CONCURRENCY_WARMUP} \
    --output      "\$RESULTS_DIR/phase1_warmup.csv"

echo ""
echo "--- Models after warm-up (should still be 3) ---"
list_models
MODELS_AFTER_WARMUP=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo 0)

# ── Step 5: Phase 2 — Code burst (high concurrency -> triggers spin-up) ──────
echo ""
echo "[5] Phase 2: Code burst (${N_BURST} code:hard/medium, concurrency=${CONCURRENCY_BURST})..."
echo "  High concurrency saturates deepseek-r1-7b -> provisioner triggers upgrade."

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
    --concurrency ${CONCURRENCY_BURST} \
    --output      "\$RESULTS_DIR/phase2_code.csv"

echo ""
echo "--- Models immediately after code burst ---"
list_models
echo "  Waiting 90s for provisioner to detect overload and trigger spin-up..."
sleep 90
echo ""
echo "--- Models after provisioner poll ---"
list_models

# ── Step 6: Phase 3 — Math burst ─────────────────────────────────────────────
echo ""
echo "[6] Phase 3: Math burst (${N_BURST} math:hard, concurrency=${CONCURRENCY_BURST})..."

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
    --concurrency ${CONCURRENCY_BURST} \
    --output      "\$RESULTS_DIR/phase3_math.csv"

echo ""
echo "--- Models after math burst ---"
list_models
echo "  Waiting 90s for provisioner..."
sleep 90

# ── Step 7: Final model inventory ────────────────────────────────────────────
echo ""
echo "[7] Final model inventory..."
MODELS_FINAL=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo 0)
list_models

# Count distinct non-seed model IDs that appeared in SPIN UP log lines.
# This includes models that were later spun down due to idle timeout — the
# true indicator of whether dynamic upgrades worked.
SPINUP_COUNT=\$(grep "SPIN UP" ~/vllm_logs/prov_dyntest_node1.log 2>/dev/null \
    | grep -v "reason=initial" | awk '{print \$3}' | sort -u | wc -l | tr -d ' ')
echo "  Distinct non-seed models spun up during test: \$SPINUP_COUNT"

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

assert_pass() { echo "  [PASS] \$1"; PASS=\$((PASS + 1)); }
assert_fail() { echo "  [FAIL] \$1"; FAIL=\$((FAIL + 1)); }

[ "\$MODELS_AFTER_WARMUP" -ge 3 ] && \
    assert_pass "Seed models started (\$MODELS_AFTER_WARMUP registered after warmup)" || \
    assert_fail "Seed models failed to start (\$MODELS_AFTER_WARMUP registered)"

# Use log-based count: intermediate models may be spun down before assertions run,
# making MODELS_FINAL == MODELS_AFTER_WARMUP even when upgrades worked correctly.
[ "\$SPINUP_COUNT" -ge 1 ] && \
    assert_pass "Models spun up dynamically (\$SPINUP_COUNT non-seed model(s) via upgrade)" || \
    assert_fail "No non-seed models were spun up — check provisioner logs"

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
# Also check log — cold model may have been spun down already
LOG_COLD=\$(grep "SPIN UP.*qwen3-coder-30b\|SPIN UP.*deepseek-r1-14b\|SPIN UP.*gemma-3-27b" \
    ~/vllm_logs/prov_dyntest_node1.log 2>/dev/null | wc -l | tr -d ' ')
[ "\$N_COLD" -ge 1 ] && \
    assert_pass "Cold model(s) currently running: \$NAMES_COLD" || \
    { [ "\$LOG_COLD" -ge 1 ] && \
      assert_pass "Cold model(s) spun up (then idle-down): confirmed in log (\$LOG_COLD SPIN UP events)" || \
      assert_fail "No cold models spun up (qwen3-coder-30b / deepseek-r1-14b / gemma-3-27b)"; }

# Informational: quality-upgrade ceiling warnings
CEILING_COUNT=\$(grep "Case 3 ceiling" ~/vllm_logs/prov_dyntest_node1.log 2>/dev/null | wc -l | tr -d ' ')
if [ "\$CEILING_COUNT" -gt 0 ]; then
    echo "  [INFO] 'Ceiling reached' logged \$CEILING_COUNT time(s) — quality-upgrade trigger"
    echo "         could not find a better model via priors. Upgrades still happened via"
    echo "         overload trigger (UPGRADE_PATH). Fix: run submit.sh once to generate"
    echo "         priors for all 6 models, then use that priors file for dynamic tests."
fi

echo ""
echo "  PASS: \$PASS   FAIL: \$FAIL"
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "  Dynamic test complete!  \$(date)"
echo "  Results: \$RESULTS_DIR/"
echo "    phase1_warmup.csv      easy/medium at cold start (concurrency=${CONCURRENCY_WARMUP})"
echo "    phase2_code.csv        code:hard burst           (concurrency=${CONCURRENCY_BURST})"
echo "    phase3_math.csv        math:hard burst           (concurrency=${CONCURRENCY_BURST})"
echo "    phase_comparison.txt   accuracy across phases"
echo ""
echo "  Provisioner log (node1): ~/vllm_logs/prov_dyntest_node1.log"
echo "  Check spin-up events:"
echo "    grep 'SPIN UP\|Strategy B\|upgrade\|TRIGGER' ~/vllm_logs/prov_dyntest_node1.log"
echo "=================================================================="
PBSEOF

echo "Submitting dynamic provisioning test..."
echo "  N_WARMUP          : $N_WARMUP  (concurrency=$CONCURRENCY_WARMUP)"
echo "  N_BURST           : $N_BURST  (concurrency=$CONCURRENCY_BURST)"
echo "  Dataset           : $DATASET"
echo "  Walltime          : 03:00:00"
echo "  Log dir           : $LOG_DIR/"
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
