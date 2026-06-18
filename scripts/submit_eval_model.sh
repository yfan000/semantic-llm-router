#!/bin/bash
# submit_eval_model.sh — Evaluate a single model and merge its priors.
#
# Runs eval_all_models.py for ONE model only (much faster than full 6-model eval),
# extracts accuracy priors, and merges them into an existing priors JSON file.
#
# Usage:
#   bash scripts/submit_eval_model.sh                        # default: qwen3-coder-30b
#   MODEL=deepseek-r1-14b bash scripts/submit_eval_model.sh  # any model
#   PRIORS_FILE=results/priors_new.json bash scripts/submit_eval_model.sh
#
# Time: ~30-45 min (1 model × 1500 queries, concurrency=30)

set -euo pipefail

MODEL=${MODEL:-"qwen3-coder-30b"}
DATASET=${DATASET:-"datasets/hf_1500.json"}
PRIORS_FILE=${PRIORS_FILE:-"results/priors_all5.json"}
CONCURRENCY=${CONCURRENCY:-30}
PROJECT=${PROJECT:-UIC-HPC}
QUEUE=${QUEUE:-by-node}

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HOME/vllm_logs/eval_${MODEL}_${TS}"
mkdir -p "$LOG_DIR"

PBSSCRIPT=$(mktemp /tmp/eval_model_XXXXXX.pbs)

cat > "$PBSSCRIPT" << PBSEOF
#!/bin/bash
#PBS -l select=1:ngpus=8:ncpus=64
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:eagle
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -N eval_${MODEL}
#PBS -o ${LOG_DIR}/job.out
#PBS -e ${LOG_DIR}/job.err

echo "PBS script started at \$(date) on \$(hostname)"

VLLM_ENV="\$HOME/.conda/envs/2026-06-08/vllm_env"
export PATH="\${VLLM_ENV}/bin:\$PATH"
echo "  Python: \$(which python 2>/dev/null || echo NOT FOUND)"

export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
cd ~/semantic-llm-router
git pull --quiet

NODE1=\$(hostname)
ROUTER_PORT=8080
ROUTER_URL="http://\${NODE1}:\${ROUTER_PORT}"

echo "=================================================================="
echo "  Single-Model Eval: ${MODEL}"
echo "  Dataset  : ${DATASET}"
echo "  Priors   : ${PRIORS_FILE}"
echo "  Node     : \$NODE1"
echo "  Time     : \$(date)"
echo "=================================================================="

# ── Start router ──────────────────────────────────────────────────────────────
echo ""
echo "[1/5] Starting router..."
nohup uvicorn semantic_router.main:app \\
    --host 0.0.0.0 --port \$ROUTER_PORT \\
    > ~/vllm_logs/router_eval.log 2>&1 &
sleep 8
for i in \$(seq 1 12); do
    curl --noproxy '*' -sf "\$ROUTER_URL/router/health" > /dev/null 2>&1 && echo "  Router ready." && break
    sleep 5
done

# ── Start the target model ────────────────────────────────────────────────────
echo ""
echo "[2/5] Starting ${MODEL}..."
nohup python provisioner/dynamic_provisioner.py \\
    --router-url   "\$ROUTER_URL" \\
    --node-host    "\$NODE1" \\
    --router-mode  ttca \\
    --static \\
    --priors-path  "${PRIORS_FILE}" \\
    --initial-models "${MODEL}" \\
    > ~/vllm_logs/prov_eval.log 2>&1 &

# Wait for model to be ready
echo "  Waiting for ${MODEL} to register (up to 15 min)..."
for i in \$(seq 1 60); do
    N=\$(curl --noproxy '*' -sf "\$ROUTER_URL/v1/models" \\
        | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" \\
        2>/dev/null || echo 0)
    echo "  [\$((i*15))s] \$N model(s) ready"
    [ "\$N" -ge 1 ] && echo "  ${MODEL} ready." && break
    sleep 15
done

# ── Run eval for this model only ──────────────────────────────────────────────
EVAL_OUT="results/eval_${MODEL}_${TS}.csv"
echo ""
echo "[3/5] Running eval (1500 queries against ${MODEL}, concurrency=${CONCURRENCY})..."
python tests/eval_all_models.py \\
    --dataset     "${DATASET}" \\
    --output      "\$EVAL_OUT" \\
    --concurrency "${CONCURRENCY}" \\
    --model       "${MODEL}"

echo "  Eval complete: \$EVAL_OUT"

# ── Extract priors ────────────────────────────────────────────────────────────
PRIORS_OUT="results/priors_${MODEL}_${TS}.json"
echo ""
echo "[4/5] Extracting priors from eval matrix..."
python tests/extract_priors.py \\
    --eval-matrix "\$EVAL_OUT" \\
    --output      "\$PRIORS_OUT"

echo "  Priors extracted:"
python3 -c "
import json
p = json.load(open('\$PRIORS_OUT'))
for model, priors in sorted(p.items()):
    keys = sorted(priors)
    print(f'  {model}: {len(priors)} keys')
    for k in keys:
        print(f'    {k:<30} {priors[k]:.4f}')
"

# ── Merge into existing priors file ──────────────────────────────────────────
echo ""
echo "[5/5] Merging into ${PRIORS_FILE}..."
python3 -c "
import json, shutil, os

base_path = '${PRIORS_FILE}'
new_path  = '\$PRIORS_OUT'

# Backup original
backup = base_path + '.bak'
shutil.copy2(base_path, backup)
print(f'  Backed up: {backup}')

base = json.load(open(base_path))
new  = json.load(open(new_path))

before = {m: len(v) for m, v in base.items()}
base.update(new)
after  = {m: len(v) for m, v in base.items()}

json.dump(base, open(base_path, 'w'), indent=2)

print(f'  Merged priors for: {list(new.keys())}')
for m in new:
    b = before.get(m, 0)
    a = after.get(m, 0)
    print(f'    {m}: {b} → {a} keys')
print(f'  Saved: {base_path}')
"

echo ""
echo "=================================================================="
echo "  Done!  \$(date)"
echo "  Eval matrix : \$EVAL_OUT"
echo "  New priors  : \$PRIORS_OUT"
echo "  Updated     : ${PRIORS_FILE}"
echo ""
echo "  Verify with:"
echo "    python3 -c \"import json; p=json.load(open('${PRIORS_FILE}')); print(p.get('${MODEL}', 'NOT FOUND'))\""
echo "=================================================================="
PBSEOF

echo "Submitting single-model eval job..."
echo "  Model      : $MODEL"
echo "  Dataset    : $DATASET"
echo "  Priors file: $PRIORS_FILE (will be updated)"
echo "  Log dir    : $LOG_DIR/"
echo ""

JOBID=$(qsub "$PBSSCRIPT")
rm -f "$PBSSCRIPT"

echo "Submitted: $JOBID"
echo ""
echo "Monitor:"
echo "  qstat -u yuping"
echo "  tail -f $LOG_DIR/job.out"
echo ""
echo "After job completes, verify priors:"
echo "  python3 -c \"import json; p=json.load(open('$PRIORS_FILE')); print(p.get('$MODEL', 'NOT FOUND'))\""
