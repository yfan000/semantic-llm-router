#!/bin/bash
# submit_polaris.sh — Submit run_experiment_polaris.sh as a PBS batch job on Polaris (ALCF).
#
# Polaris differs from Sophia: 4 GPUs/node (A100 40GB) vs Sophia's 8 GPUs/node.
# This script requests 4 nodes (same total GPU count: 16) and uses the prod queue.
#
# Usage:
#   bash scripts/submit_polaris.sh
#   TTCA_ALPHA=0.5 bash scripts/submit_polaris.sh
#   EXPERIMENT_MODE=dynamic bash scripts/submit_polaris.sh
#   EXPERIMENT_MODE=dynamic TTCA_ALPHA=0.5 WALLTIME=10:00:00 bash scripts/submit_polaris.sh
#
# Monitor after submission:
#   qstat -u yuping
#   tail -f ~/vllm_logs/batch_<ts>/job.out

set -euo pipefail

# ── Job parameters (override via env vars) ────────────────────────────────────
WALLTIME=${WALLTIME:-08:00:00}
TTCA_ALPHA=${TTCA_ALPHA:-1.0}
TTCA_COST_BETA=${TTCA_COST_BETA:-0.0}
EXPERIMENT_MODE=${EXPERIMENT_MODE:-static}
N_REQUESTS=${N_REQUESTS:-3000}
CONCURRENCY=${CONCURRENCY:-50}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-30}
PROJECT=${PROJECT:-UIC-HPC}
QUEUE=${QUEUE:-prod}

# ── Log directory (persisted on eagle) ────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HOME/vllm_logs/batch_${TS}_alpha${TTCA_ALPHA}_beta${TTCA_COST_BETA}_${EXPERIMENT_MODE}"
mkdir -p "$LOG_DIR"

JOB_NAME="ttca_${EXPERIMENT_MODE}_a${TTCA_ALPHA}_b${TTCA_COST_BETA}"

# ── Generate PBS script ───────────────────────────────────────────────────────
PBSSCRIPT=$(mktemp /tmp/experiment_XXXXXX.pbs)

cat > "$PBSSCRIPT" << PBSEOF
#!/bin/bash
#PBS -l select=4:ngpus=4:ncpus=32
#PBS -l walltime=${WALLTIME}
#PBS -l filesystems=home:eagle
#PBS -A ${PROJECT}
#PBS -q ${QUEUE}
#PBS -N ${JOB_NAME}
#PBS -o ${LOG_DIR}/job.out
#PBS -e ${LOG_DIR}/job.err

# ── Environment setup ─────────────────────────────────────────────────────────
echo "PBS script started at \$(date) on \$(hostname)"
VLLM_ENV="\$HOME/.conda/envs/2026-06-08/vllm_env"
export PATH="\${VLLM_ENV}/bin:\$PATH"
echo "  Python: \$(which python 2>/dev/null || echo NOT FOUND)  (\$(python --version 2>&1 || echo N/A))"

export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache

# ── Change to project directory ───────────────────────────────────────────────
cd ~/semantic-llm-router
git pull --quiet   # pick up any last-minute code changes

# ── Experiment settings ───────────────────────────────────────────────────────
export TTCA_ALPHA=${TTCA_ALPHA}
export TTCA_COST_BETA=${TTCA_COST_BETA}
export EXPERIMENT_MODE=${EXPERIMENT_MODE}
export N_REQUESTS=${N_REQUESTS}
export CONCURRENCY=${CONCURRENCY}
export EVAL_CONCURRENCY=${EVAL_CONCURRENCY}

echo "=================================================================="
echo "  PBS Batch Job (Polaris)"
echo "  Job ID    : \$PBS_JOBID"
echo "  Nodes     : \$(sort -u \$PBS_NODEFILE | tr '\n' ' ')"
echo "  TTCA_ALPHA    : ${TTCA_ALPHA}"
echo "  TTCA_COST_BETA: ${TTCA_COST_BETA}"
echo "  MODE          : ${EXPERIMENT_MODE}"
echo "  Log dir       : ${LOG_DIR}/"
echo "=================================================================="

# ── Run experiment ────────────────────────────────────────────────────────────
bash scripts/run_experiment_polaris.sh

echo "Batch job complete."
PBSEOF

# ── Submit ────────────────────────────────────────────────────────────────────
echo "Submitting Polaris experiment batch job..."
echo "  TTCA_ALPHA      : $TTCA_ALPHA"
echo "  TTCA_COST_BETA  : $TTCA_COST_BETA"
echo "  EXPERIMENT_MODE : $EXPERIMENT_MODE"
echo "  WALLTIME        : $WALLTIME"
echo "  N_REQUESTS      : $N_REQUESTS"
echo "  Log dir         : $LOG_DIR/"
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
