#!/bin/bash
# submit_baselines.sh — Submit baseline-only comparison as a PBS batch job.
#
# Runs the 4 baseline comparisons (TTCA no-retry, complexity-tier, cascade,
# vLLM-SR) WITHOUT re-running the expensive eval_matrix step (6 models ×
# 1500 queries ≈ 3 hours). If you provide an existing RESULTS_DIR the job
# re-uses its eval_matrix.csv and priors_new.json, cutting total time to
# ~1-2 hours.
#
# Usage:
#   # First time — no existing results, builds everything:
#   bash scripts/submit_baselines.sh
#
#   # Re-use prior eval_matrix (fastest):
#   RESULTS_DIR=results/experiment_20241201_120000_alpha1.0_beta0.0_static \
#       bash scripts/submit_baselines.sh
#
#   # Custom threshold for cascade baseline:
#   CASCADE_THRESHOLD=0.75 bash scripts/submit_baselines.sh
#
# Monitor:
#   qstat -u yuping
#   tail -f ~/vllm_logs/baselines_<ts>/job.out

set -euo pipefail

# ── Parameters ────────────────────────────────────────────────────────────────
WALLTIME=${WALLTIME:-03:00:00}         # baselines only need ~1-2 h
TTCA_ALPHA=${TTCA_ALPHA:-1.0}
TTCA_COST_BETA=${TTCA_COST_BETA:-0.0}
N_REQUESTS=${N_REQUESTS:-1500}
CONCURRENCY=${CONCURRENCY:-50}
CASCADE_THRESHOLD=${CASCADE_THRESHOLD:-0.80}
PROJECT=${PROJECT:-UIC-HPC}
QUEUE=${QUEUE:-by-node}
DATASET=${DATASET:-"datasets/hf_1500.json"}
# Point to an existing experiment directory to skip eval_matrix/priors steps.
# Leave empty to run steps 1-7 (router + models + eval_matrix + priors) first.
RESULTS_DIR=${RESULTS_DIR:-""}

# ── Log dir ────────────────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HOME/vllm_logs/baselines_${TS}"
mkdir -p "$LOG_DIR"

JOB_NAME="baselines_a${TTCA_ALPHA}_b${TTCA_COST_BETA}"

# ── Generate PBS script ────────────────────────────────────────────────────────
PBSSCRIPT=$(mktemp /tmp/baselines_XXXXXX.pbs)

cat > "$PBSSCRIPT" << PBSEOF
#!/bin/bash
#PBS -l select=2:ngpus=8:ncpus=64
#PBS -l walltime=${WALLTIME}
#PBS -l filesystems=home:eagle
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -N ${JOB_NAME}
#PBS -o ${LOG_DIR}/job.out
#PBS -e ${LOG_DIR}/job.err

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
echo "PBS script started at \$(date) on \$(hostname)"
VLLM_ENV="\$HOME/.conda/envs/2026-06-08/vllm_env"
export PATH="\${VLLM_ENV}/bin:\$PATH"
echo "  Python: \$(which python 2>/dev/null || echo NOT FOUND)  (\$(python --version 2>&1 || echo N/A))"
export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache

cd ~/semantic-llm-router
git pull --quiet

# ── Discover nodes ────────────────────────────────────────────────────────────
NODES=(\$(sort -u \$PBS_NODEFILE))
NODE1=\${NODES[0]}
NODE2=\${NODES[1]}
ROUTER_PORT=${TTCA_ALPHA//./}_8080   # just use 8080
ROUTER_PORT=8080
ROUTER_URL="http://\${NODE1}:\${ROUTER_PORT}"
LOG_DIR="${LOG_DIR}"

echo "=================================================================="
echo "  Baselines Batch Job   \$(date)"
echo "  Job ID         : \$PBS_JOBID"
echo "  NODE1          : \$NODE1"
echo "  NODE2          : \$NODE2"
echo "  TTCA_ALPHA     : ${TTCA_ALPHA}   TTCA_COST_BETA: ${TTCA_COST_BETA}"
echo "  N_REQUESTS     : ${N_REQUESTS}"
echo "  RESULTS_DIR    : ${RESULTS_DIR:-'(new — will run steps 1-7 first)'}"
echo "=================================================================="

# Patch TTCA params into config
sed -i "s/^TTCA_ALPHA: float = .*/TTCA_ALPHA: float = ${TTCA_ALPHA}/" semantic_router/config.py
sed -i "s/^TTCA_COST_BETA: float = .*/TTCA_COST_BETA: float = ${TTCA_COST_BETA}/" semantic_router/config.py

# ── Wait-for-router helper ────────────────────────────────────────────────────
wait_router() {
    for i in \$(seq 1 60); do
        if curl --noproxy '*' -sf "\$ROUTER_URL/router/health" > /dev/null 2>&1; then
            echo "  Router ready."; return 0
        fi
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
        echo "  [\$((i*15))s] Registered: \$cnt / \$N"
        [ "\$cnt" -ge "\$N" ] && echo "  Models ready." && return 0
        sleep 15
    done
    echo "WARNING: Only \$cnt/\$N models ready — continuing anyway"
}

# ── Determine results directory ───────────────────────────────────────────────
EXISTING_DIR="${RESULTS_DIR}"

if [ -n "\$EXISTING_DIR" ] && [ -f "\$EXISTING_DIR/eval_matrix.csv" ]; then
    # ── Fast path: re-use existing eval_matrix and priors ────────────────────
    echo ""
    echo "Re-using existing results from: \$EXISTING_DIR"
    RESULTS_DIR="\$EXISTING_DIR/baselines_\$(date +%Y%m%d_%H%M%S)"
    mkdir -p "\$RESULTS_DIR"

    # Copy eval_matrix and priors into new subdir for this run
    cp "\$EXISTING_DIR/eval_matrix.csv" "\$RESULTS_DIR/"
    PRIORS_FILE="\$EXISTING_DIR/priors_new.json"
    [ ! -f "\$PRIORS_FILE" ] && PRIORS_FILE="${PRIORS_FILE:-results/priors_all5.json}"

    # Start router
    echo ""
    echo "[1/3] Starting router on \$NODE1..."
    nohup uvicorn semantic_router.main:app \
        --host 0.0.0.0 --port "\$ROUTER_PORT" \
        > "\$LOG_DIR/router.log" 2>&1 &
    sleep 5; wait_router

    # Start models — static mode, all 5 on node1 + llama4-scout on node2
    echo ""
    echo "[2/3] Starting models (static)..."
    nohup python provisioner/dynamic_provisioner.py \
        --router-url "\$ROUTER_URL" --node-host "\$NODE1" \
        --router-mode ttca --static \
        --priors-path "\$PRIORS_FILE" \
        --initial-models "qwen-7b,deepseek-r1-7b,qwen3-coder-30b,gemma-3-27b,deepseek-r1-14b" \
        > "\$LOG_DIR/provisioner_node1.log" 2>&1 &

    ssh "\$NODE2" "
        cd ~/semantic-llm-router
        export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache
        python provisioner/dynamic_provisioner.py \
            --router-url '\$ROUTER_URL' --node-host '\$NODE2' \
            --router-mode ttca --static \
            --priors-path '\$PRIORS_FILE' \
            --initial-models llama4-scout \
            </dev/null >>'\$LOG_DIR/provisioner_node2.log' 2>&1 &
        disown \$!; echo PID:\$!
    " < /dev/null

    echo ""
    echo "[3/3] Waiting for 6 models..."
    wait_models 6

    # Re-register with the priors from the existing run
    python tests/register_with_priors.py \
        --priors "\$PRIORS_FILE" --router-url "\$ROUTER_URL"

else
    # ── Full path: run steps 1-7 of run_experiment.sh first ─────────────────
    echo ""
    echo "No existing results dir — running full steps 1-7 first."
    echo "(Set RESULTS_DIR=<path> to skip this ~3h step)"
    echo ""

    export TTCA_ALPHA=${TTCA_ALPHA}
    export TTCA_COST_BETA=${TTCA_COST_BETA}
    export N_REQUESTS=${N_REQUESTS}
    export CONCURRENCY=${CONCURRENCY}
    export EVAL_CONCURRENCY=30
    export DATASET="${DATASET}"
    export EXPERIMENT_MODE=static

    # Run the first 7 steps of run_experiment.sh inline
    # (reuse the script but stop after priors re-registration)
    bash scripts/run_experiment.sh &
    FULL_PID=\$!

    # Wait for step 7 to complete by polling for priors_new.json
    RESULTS_DIR=\$(ls -dt results/experiment_* 2>/dev/null | head -1)
    PRIORS_FILE="\$RESULTS_DIR/priors_new.json"
    echo "Waiting for eval_matrix + priors in \$RESULTS_DIR ..."
    for i in \$(seq 1 720); do  # up to 3h
        [ -f "\$PRIORS_FILE" ] && echo "  Priors ready at \${i}min." && break
        sleep 15
    done

    # Kill the background experiment (we'll run baselines ourselves)
    kill \$FULL_PID 2>/dev/null || true
fi

# ── Run baselines ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  Running baselines in: \$RESULTS_DIR"
echo "=================================================================="

DATASET="${DATASET}"
CONCURRENCY="${CONCURRENCY}"
N_REQUESTS="${N_REQUESTS}"
ROUTER_URL_LOCAL="\$ROUTER_URL"

# Baseline 1: TTCA single-shot (no retry)
echo ""
echo "[Baseline 1/4] TTCA single-shot — no retry (\$N_REQUESTS requests)..."
python tests/load_test.py \
    --dataset "\$DATASET" --router "\$ROUTER_URL_LOCAL" \
    --mode ttca --no-retry \
    --requests "\$N_REQUESTS" --concurrency "\$CONCURRENCY" \
    --output "\$RESULTS_DIR/ttca_no_retry.csv"

# Baseline 2: Complexity-tier
echo ""
echo "[Baseline 2/4] Complexity-tier routing (\$N_REQUESTS requests)..."
python tests/baseline_complexity_tier.py \
    --dataset "\$DATASET" --concurrency "\$CONCURRENCY" \
    --node2-host "\$NODE2" \
    --output "\$RESULTS_DIR/baseline_tier.csv"

# Baseline 3: Cascade (RouteLLM-style)
PRIORS_FILE=\${PRIORS_FILE:-"\$RESULTS_DIR/priors_new.json"}
echo ""
echo "[Baseline 3/4] Cascade / RouteLLM-style (threshold=${CASCADE_THRESHOLD})..."
python tests/baseline_cascade.py \
    --dataset "\$DATASET" --priors "\$PRIORS_FILE" \
    --threshold ${CASCADE_THRESHOLD} --concurrency "\$CONCURRENCY" \
    --output "\$RESULTS_DIR/baseline_cascade.csv"

# Baseline 4: vLLM Semantic Router (optional)
echo ""
echo "[Baseline 4/4] vLLM Semantic Router (checking port 8888)..."
if curl --noproxy '*' -sf "http://\${NODE1}:8888/health" > /dev/null 2>&1 || \
   curl --noproxy '*' -sf "http://\${NODE1}:8888/v1/models" > /dev/null 2>&1; then
    python tests/baseline_vllm_router.py \
        --dataset "\$DATASET" --endpoint "http://\${NODE1}:8888" \
        --concurrency "\$CONCURRENCY" \
        --output "\$RESULTS_DIR/baseline_vllm_sr.csv"
else
    echo "  vllm-sr not running — skipping. Start with: vllm-sr serve"
fi

# ── Comparison reports ────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  Generating comparison reports"
echo "=================================================================="

RR="\$RESULTS_DIR/rr_baseline.csv"
# Generate round-robin baseline if not present in this results dir
if [ ! -f "\$RR" ]; then
    echo "  Generating round-robin baseline..."
    python tests/round_robin_test.py \
        --dataset "\$DATASET" --concurrency "\$CONCURRENCY" \
        --node2-host "\$NODE2" --output "\$RR"
fi

EVAL="\$RESULTS_DIR/eval_matrix.csv"

echo ""
echo "=== TTCA (with retry) vs TTCA (no retry) — retry benefit ===" \
    | tee "\$RESULTS_DIR/compare_retry_benefit.txt"
python tests/compare_ttca.py \
    --router "\$RESULTS_DIR/ttca_no_retry.csv" \
    --baseline "\$RR" \
    --eval-matrix "\$EVAL" \
    | tee -a "\$RESULTS_DIR/compare_retry_benefit.txt"

echo ""
echo "=== TTCA (no retry) vs Complexity-Tier ===" \
    | tee "\$RESULTS_DIR/compare_vs_tier.txt"
python tests/compare_categories.py \
    --router "\$RESULTS_DIR/ttca_no_retry.csv" \
    --baseline "\$RESULTS_DIR/baseline_tier.csv" \
    --eval-matrix "\$EVAL" \
    --output "\$RESULTS_DIR/compare_vs_tier.csv" \
    | tee -a "\$RESULTS_DIR/compare_vs_tier.txt"

echo ""
echo "=== TTCA (no retry) vs Cascade / RouteLLM-style ===" \
    | tee "\$RESULTS_DIR/compare_vs_cascade.txt"
python tests/compare_categories.py \
    --router "\$RESULTS_DIR/ttca_no_retry.csv" \
    --baseline "\$RESULTS_DIR/baseline_cascade.csv" \
    --eval-matrix "\$EVAL" \
    --output "\$RESULTS_DIR/compare_vs_cascade.csv" \
    | tee -a "\$RESULTS_DIR/compare_vs_cascade.txt"

if [ -f "\$RESULTS_DIR/baseline_vllm_sr.csv" ]; then
    echo ""
    echo "=== TTCA (no retry) vs vLLM Semantic Router ===" \
        | tee "\$RESULTS_DIR/compare_vs_vllm_sr.txt"
    python tests/compare_categories.py \
        --router "\$RESULTS_DIR/ttca_no_retry.csv" \
        --baseline "\$RESULTS_DIR/baseline_vllm_sr.csv" \
        --eval-matrix "\$EVAL" \
        --output "\$RESULTS_DIR/compare_vs_vllm_sr.csv" \
        | tee -a "\$RESULTS_DIR/compare_vs_vllm_sr.txt"
fi

echo ""
echo "=================================================================="
echo "  Baselines complete!  \$(date)"
echo "  Results: \$RESULTS_DIR/"
echo "    ttca_no_retry.csv          TTCA single-shot (no retry)"
echo "    baseline_tier.csv          Complexity-tier / RouterDC"
echo "    baseline_cascade.csv       2-model cascade / RouteLLM"
echo "    baseline_vllm_sr.csv       vLLM Semantic Router (if run)"
echo "    compare_retry_benefit.txt  Value of cascade retry"
echo "    compare_vs_tier.csv        vs complexity-tier"
echo "    compare_vs_cascade.csv     vs RouteLLM-style"
echo "    compare_vs_vllm_sr.csv     vs vLLM-SR (if run)"
echo "=================================================================="
PBSEOF

# ── Submit ─────────────────────────────────────────────────────────────────────
echo "Submitting baselines batch job..."
echo "  TTCA_ALPHA        : $TTCA_ALPHA"
echo "  TTCA_COST_BETA    : $TTCA_COST_BETA"
echo "  N_REQUESTS        : $N_REQUESTS"
echo "  CASCADE_THRESHOLD : $CASCADE_THRESHOLD"
echo "  WALLTIME          : $WALLTIME"
echo "  RESULTS_DIR       : ${RESULTS_DIR:-'(new — will generate eval_matrix first)'}"
echo "  Log dir           : $LOG_DIR/"
echo ""

JOBID=$(qsub "$PBSSCRIPT")
rm -f "$PBSSCRIPT"

echo "Submitted: $JOBID"
echo ""
echo "Monitor:"
echo "  qstat -u yuping"
echo "  tail -f $LOG_DIR/job.out"
echo "  tail -f $LOG_DIR/job.err"
echo ""
echo "Cancel if needed:"
echo "  qdel $JOBID"
