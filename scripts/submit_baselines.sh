#!/bin/bash
# submit_baselines.sh -- Submit baseline-only comparison as a PBS batch job.
#
# Usage:
#   bash scripts/submit_baselines.sh
#   RESULTS_DIR=results/experiment_20241201_120000_alpha1.0_beta0.0_static \
#       bash scripts/submit_baselines.sh

set -euo pipefail

WALLTIME=${WALLTIME:-03:00:00}
TTCA_ALPHA=${TTCA_ALPHA:-1.0}
TTCA_COST_BETA=${TTCA_COST_BETA:-0.0}
N_REQUESTS=${N_REQUESTS:-1500}
CONCURRENCY=${CONCURRENCY:-50}
CASCADE_THRESHOLD=${CASCADE_THRESHOLD:-0.80}
PROJECT=${PROJECT:-UIC-HPC}
QUEUE=${QUEUE:-by-node}
DATASET=${DATASET:-"datasets/hf_1500.json"}
RESULTS_DIR=${RESULTS_DIR:-""}

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HOME/vllm_logs/baselines_${TS}"
mkdir -p "$LOG_DIR"

JOB_NAME="baselines_a${TTCA_ALPHA}_b${TTCA_COST_BETA}"
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
source /soft/anaconda3/etc/profile.d/conda.sh
conda activate 2024-08-08/vllm_env
export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache

cd ~/semantic-llm-router
git pull --quiet

NODES=(\$(sort -u \$PBS_NODEFILE))
NODE1=\${NODES[0]}
NODE2=\${NODES[1]}
ROUTER_PORT=8080
ROUTER_URL="http://\${NODE1}:\${ROUTER_PORT}"
LOG_DIR="${LOG_DIR}"

sed -i "s/^TTCA_ALPHA: float = .*/TTCA_ALPHA: float = ${TTCA_ALPHA}/" semantic_router/config.py
sed -i "s/^TTCA_COST_BETA: float = .*/TTCA_COST_BETA: float = ${TTCA_COST_BETA}/" semantic_router/config.py

wait_router() {
    for i in \$(seq 1 60); do
        if curl --noproxy '*' -sf "\$ROUTER_URL/router/health" > /dev/null 2>&1; then
            echo "  Router ready."; return 0
        fi
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
        echo "  [\$((i*15))s] Registered: \$cnt / \$N"
        [ "\$cnt" -ge "\$N" ] && echo "  Models ready." && return 0
        sleep 15
    done
    echo "WARNING: Only \$cnt/\$N models ready -- continuing anyway"
}

EXISTING_DIR="${RESULTS_DIR}"

if [ -n "\$EXISTING_DIR" ] && [ -f "\$EXISTING_DIR/eval_matrix.csv" ]; then
    echo "Re-using existing results from: \$EXISTING_DIR"
    RESULTS_DIR="\$EXISTING_DIR/baselines_\$(date +%Y%m%d_%H%M%S)"
    mkdir -p "\$RESULTS_DIR"
    cp "\$EXISTING_DIR/eval_matrix.csv" "\$RESULTS_DIR/"
    PRIORS_FILE="\$EXISTING_DIR/priors_new.json"
    [ ! -f "\$PRIORS_FILE" ] && PRIORS_FILE="results/priors_all5.json"

    nohup uvicorn semantic_router.main:app \
        --host 0.0.0.0 --port "\$ROUTER_PORT" \
        > "\$LOG_DIR/router.log" 2>&1 &
    sleep 5; wait_router

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
            </dev/null>>'\$LOG_DIR/provisioner_node2.log' 2>&1 &
        disown \$!; echo PID:\$!
    " < /dev/null

    wait_models 6
    python tests/register_with_priors.py --priors "\$PRIORS_FILE" --router-url "\$ROUTER_URL"
else
    echo "No existing results dir -- running full experiment first."
    export TTCA_ALPHA=${TTCA_ALPHA}
    export TTCA_COST_BETA=${TTCA_COST_BETA}
    export N_REQUESTS=${N_REQUESTS}
    export CONCURRENCY=${CONCURRENCY}
    export EVAL_CONCURRENCY=30
    export DATASET="${DATASET}"
    export EXPERIMENT_MODE=static
    bash scripts/run_experiment.sh &
    FULL_PID=\$!
    RESULTS_DIR=\$(ls -dt results/experiment_* 2>/dev/null | head -1)
    PRIORS_FILE="\$RESULTS_DIR/priors_new.json"
    echo "Waiting for priors in \$RESULTS_DIR ..."
    for i in \$(seq 1 720); do
        [ -f "\$PRIORS_FILE" ] && echo "  Priors ready at \${i}min." && break
        sleep 15
    done
    kill \$FULL_PID 2>/dev/null || true
fi

DATASET="${DATASET}"
CONCURRENCY="${CONCURRENCY}"
N_REQUESTS="${N_REQUESTS}"

echo "[Baseline 1/4] TTCA single-shot -- no retry..."
python tests/load_test.py \
    --dataset "\$DATASET" --router "\$ROUTER_URL" \
    --mode ttca --no-retry \
    --requests "\$N_REQUESTS" --concurrency "\$CONCURRENCY" \
    --output "\$RESULTS_DIR/ttca_no_retry.csv"

echo "[Baseline 2/4] Complexity-tier routing..."
python tests/baseline_complexity_tier.py \
    --dataset "\$DATASET" --concurrency "\$CONCURRENCY" \
    --node2-host "\$NODE2" \
    --output "\$RESULTS_DIR/baseline_tier.csv"

PRIORS_FILE=\${PRIORS_FILE:-"\$RESULTS_DIR/priors_new.json"}
echo "[Baseline 3/4] Cascade / RouteLLM-style..."
python tests/baseline_cascade.py \
    --dataset "\$DATASET" --priors "\$PRIORS_FILE" \
    --threshold ${CASCADE_THRESHOLD} --concurrency "\$CONCURRENCY" \
    --output "\$RESULTS_DIR/baseline_cascade.csv"

echo "[Baseline 4/4] vLLM Semantic Router..."
if curl --noproxy '*' -sf "http://\${NODE1}:8888/health" > /dev/null 2>&1 || \
   curl --noproxy '*' -sf "http://\${NODE1}:8888/v1/models" > /dev/null 2>&1; then
    python tests/baseline_vllm_router.py \
        --dataset "\$DATASET" --endpoint "http://\${NODE1}:8888" \
        --concurrency "\$CONCURRENCY" \
        --output "\$RESULTS_DIR/baseline_vllm_sr.csv"
else
    echo "  vllm-sr not running -- skipping."
fi

RR="\$RESULTS_DIR/rr_baseline.csv"
if [ ! -f "\$RR" ]; then
    python tests/round_robin_test.py \
        --dataset "\$DATASET" --concurrency "\$CONCURRENCY" \
        --node2-host "\$NODE2" --output "\$RR"
fi

EVAL="\$RESULTS_DIR/eval_matrix.csv"
python tests/compare_categories.py \
    --router "\$RESULTS_DIR/ttca_no_retry.csv" \
    --baseline "\$RESULTS_DIR/baseline_tier.csv" \
    --eval-matrix "\$EVAL" \
    --output "\$RESULTS_DIR/compare_vs_tier.csv" \
    | tee "\$RESULTS_DIR/compare_vs_tier.txt"

python tests/compare_categories.py \
    --router "\$RESULTS_DIR/ttca_no_retry.csv" \
    --baseline "\$RESULTS_DIR/baseline_cascade.csv" \
    --eval-matrix "\$EVAL" \
    --output "\$RESULTS_DIR/compare_vs_cascade.csv" \
    | tee "\$RESULTS_DIR/compare_vs_cascade.txt"

if [ -f "\$RESULTS_DIR/baseline_vllm_sr.csv" ]; then
    python tests/compare_categories.py \
        --router "\$RESULTS_DIR/ttca_no_retry.csv" \
        --baseline "\$RESULTS_DIR/baseline_vllm_sr.csv" \
        --eval-matrix "\$EVAL" \
        --output "\$RESULTS_DIR/compare_vs_vllm_sr.csv" \
        | tee "\$RESULTS_DIR/compare_vs_vllm_sr.txt"
fi

echo ""
echo "=================================================================="
echo "  Baselines complete!  \$(date)"
echo "  Results: \$RESULTS_DIR/"
echo "=========================================================="
PBSEOF

echo "Submitting baselines batch job..."
echo "  TTCA_ALPHA        : $TTCA_ALPHA"
echo "  TTCA_COST_BETA    : $TTCA_COST_BETA"
echo "  N_REQUESTS        : $N_REQUESTS"
echo "  CASCADE_THRESHOLD : $CASCADE_THRESHOLD"
echo "  WALLTIME          : $WALLTIME"
echo "  RESULTS_DIR       : ${RESULTS_DIR:-'(new)'}"
echo "  Log dir           : $LOG_DIR/"
echo ""

JOBID=$(qsub "$PBSSCRIPT")
rm -f "$PBSSCRIPT"

echo "Submitted: $JOBID"
echo "  qstat -u yuping"
echo "  tail -f $LOG_DIR/job.out"
