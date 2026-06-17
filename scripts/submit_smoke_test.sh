#!/bin/bash
# submit_smoke_test.sh — Run the baseline smoke test as a short PBS batch job.
#
# Starts 2 seed models (qwen-7b + qwen3-coder-30b) on node1 and llama4-scout
# on node2, runs 20 requests through each baseline, then reports pass/fail.
# Faster than the full experiment: total ~30-40 min vs 5-6 h.
#
# Usage:
#   bash scripts/submit_smoke_test.sh
#   N=50 bash scripts/submit_smoke_test.sh   # more requests per baseline

set -euo pipefail

N=${N:-20}
PROJECT=${PROJECT:-UIC-HPC}
QUEUE=${QUEUE:-by-node}
DATASET=${DATASET:-"datasets/hf_1500.json"}
PRIORS=${PRIORS:-"results/priors_all5.json"}

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HOME/vllm_logs/smoke_${TS}"
mkdir -p "$LOG_DIR"

PBSSCRIPT=$(mktemp /tmp/smoke_XXXXXX.pbs)

cat > "$PBSSCRIPT" << PBSEOF
#!/bin/bash
#PBS -l select=2:ngpus=8:ncpus=64
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -N smoke_test
#PBS -o ${LOG_DIR}/job.out
#PBS -e ${LOG_DIR}/job.err

set -euo pipefail
source /soft/anaconda3/etc/profile.d/conda.sh
conda activate 2026-06-08/vllm_env
export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache

cd ~/semantic-llm-router
git pull --quiet

NODES=(\$(sort -u \$PBS_NODEFILE))
NODE1=\${NODES[0]}
NODE2=\${NODES[1]}

echo "=================================================================="
echo "  Smoke Test   \$(date)"
echo "  NODE1: \$NODE1   NODE2: \$NODE2"
echo "  N requests per baseline: ${N}"
echo "  Dataset: ${DATASET}"
echo "=================================================================="

mkdir -p ~/vllm_logs

# ── Start router ──────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Starting router on \$NODE1..."
nohup uvicorn semantic_router.main:app \
    --host 0.0.0.0 --port 8080 \
    > ~/vllm_logs/router_smoke.log 2>&1 &
sleep 8

for i in \$(seq 1 12); do
    curl --noproxy '*' -sf "http://\${NODE1}:8080/router/health" > /dev/null 2>&1 && break
    sleep 5
done
echo "  Router ready."

# ── Start 2 seed models on node1 (fast to load, 1 GPU each) ──────────────────
echo ""
echo "[2/4] Starting qwen-7b + qwen3-coder-30b on \$NODE1..."
nohup python provisioner/dynamic_provisioner.py \
    --router-url  "http://\${NODE1}:8080" \
    --node-host   "\$NODE1" \
    --router-mode ttca \
    --static \
    --priors-path "${PRIORS}" \
    --initial-models "qwen-7b,qwen3-coder-30b" \
    > ~/vllm_logs/prov1_smoke.log 2>&1 &

# ── Start llama4-scout on node2 ───────────────────────────────────────────────
echo ""
echo "[3/4] Starting llama4-scout on \$NODE2..."
ssh "\$NODE2" "
    cd ~/semantic-llm-router
    export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
    nohup python provisioner/dynamic_provisioner.py \
        --router-url  'http://\${NODE1}:8080' \
        --node-host   '\$NODE2' \
        --router-mode ttca \
        --static \
        --priors-path '${PRIORS}' \
        --initial-models llama4-scout \
        </dev/null >>~/vllm_logs/prov2_smoke.log 2>&1 &
    disown \$!
" </dev/null

# ── Wait for at least 2 models ────────────────────────────────────────────────
echo ""
echo "  Waiting for models (up to 20 min)..."
for i in \$(seq 1 80); do
    N_READY=\$(curl --noproxy '*' -sf "http://\${NODE1}:8080/v1/models" \
        | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" \
        2>/dev/null || echo 0)
    echo "  [\$((i*15))s] \$N_READY model(s) ready"
    [ "\$N_READY" -ge 2 ] && echo "  Enough models ready — starting test." && break
    sleep 15
done

# Register with priors
python tests/register_with_priors.py \
    --priors     "${PRIORS}" \
    --router-url "http://\${NODE1}:8080" \
    --node2-host "\$NODE2"

# ── Run smoke test ────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Running smoke test (${N} requests per baseline)..."
N=${N} \
NODE1=\$NODE1 \
NODE2=\$NODE2 \
ROUTER_PORT=8080 \
DATASET="${DATASET}" \
PRIORS="${PRIORS}" \
SKIP_VLLM_SR=1 \
bash scripts/test_baselines.sh

echo ""
echo "=================================================================="
echo "  Smoke test complete.  \$(date)"
if grep -q "FAIL: 0" "${LOG_DIR}/job.out" 2>/dev/null; then
    echo "  STATUS: ALL PASSED — safe to submit full experiment:"
    echo "    bash scripts/submit.sh"
else
    echo "  STATUS: CHECK FAILURES above before submitting full job"
fi
echo "=================================================================="
PBSEOF

echo "Submitting smoke test batch job..."
echo "  N requests per baseline : $N"
echo "  Dataset                 : $DATASET"
echo "  Walltime                : 01:00:00"
echo "  Log dir                 : $LOG_DIR/"
echo ""

JOBID=$(qsub "$PBSSCRIPT")
rm -f "$PBSSCRIPT"

echo "Submitted: $JOBID"
echo ""
echo "Monitor:"
echo "  qstat -u yuping"
echo "  tail -f $LOG_DIR/job.out"
echo ""
echo "If all pass, submit the full experiment:"
echo "  bash scripts/submit.sh"
echo "  TTCA_COST_BETA=1.0 bash scripts/submit.sh"
