#!/bin/bash
# submit.sh — Submit run_experiment.sh as a PBS batch job on Sophia ALCF.
#
# Creates a temporary PBS script with all env vars baked in and submits it
# with qsub. The job runs unattended — no interactive session needed.
#
# Usage:
#   bash scripts/submit.sh
#   TTCA_ALPHA=0.5 bash scripts/submit.sh
#   EXPERIMENT_MODE=dynamic bash scripts/submit.sh
#   EXPERIMENT_MODE=dynamic TTCA_ALPHA=0.5 WALLTIME=10:00:00 bash scripts/submit.sh
#
# Monitor after submission:
#   qstat -u yuping
#   tail -f ~/vllm_logs/batch_<ts>/job.out

set -euo pipefail

# ── Job parameters (override via env vars) ──────────────────────────────────────────────────
WALLTIME=${WALLTIME:-08:00:00}
TTCA_ALPHA=${TTCA_ALPHA:-1.0}
EXPERIMENT_MODE=${EXPERIMENT_MODE:-static}
N_REQUESTS=${N_REQUESTS:-1000}
CONCURRENCY=${CONCURRENCY:-50}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-30}
PROJECT=${PROJECT:-UIC-HPC}
QUEUE=${QUEUE:-by-node}

# ── Log directory (persisted on eagle) ──────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HOME/vllm_logs/batch_${TS}_alpha${TTCA_ALPHA}_${EXPERIMENT_MODE}"
mkdir -p "$LOG_DIR"

JOB_NAME="ttca_${EXPERIMENT_MODE}_a${TTCA_ALPHA}"

# ── Generate PBS script ───────────────────────────────────────────────────────────────
PBSSCRIPT=$(mktemp /tmp/experiment_XXXXXX.pbs)

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

# ── Environment setup ─────────────────────────────────────────────────────────────────
source /soft/anaconda3/etc/profile.d/conda.sh
conda activate 2024-08-08/vllm_env

export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache

# ── Change to project directory ─────────────────────────────────────────────────────
cd ~/semantic-llm-router
git pull --quiet   # pick up any last-minute code changes

# ── Experiment settings ───────────────────────────────────────────────────────────────
export TTCA_ALPHA=${TTCA_ALPHA}
export EXPERIMENT_MODE=${EXPERIMENT_MODE}
export N_REQUESTS=${N_REQUESTS}
export CONCURRENCY=${CONCURRENCY}
export EVAL_CONCURRENCY=${EVAL_CONCURRENCY}

echo "=================================================================="
echo "  PBS Batch Job"
echo "  Job ID    : \$PBS_JOBID"
echo "  Nodes     : \$(sort -u \$PBS_NODEFILE | tr '\n' ' ')"
echo "  TTCA_ALPHA: ${TTCA_ALPHA}"
echo "  MODE      : ${EXPERIMENT_MODE}"
echo "  Log dir   : ${LOG_DIR}/"
echo "=================================================================="

# ── Run experiment ──────────────────────────────────────────────────────────────────
bash scripts/run_experiment.sh

echo "Batch job complete."
PBSEOF

# ── Submit ──────────────────────────────────────────────────────────────────────────────
echo "Submitting experiment batch job..."
echo "  TTCA_ALPHA      : $TTCA_ALPHA"
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
