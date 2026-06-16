#!/bin/bash
# submit.sh -- Submit run_experiment.sh as a PBS batch job on Sophia ALCF.
#
# Usage:
#   bash scripts/submit.sh
#   TTCA_ALPHA=0.5 bash scripts/submit.sh
#   TTCA_COST_BETA=0.5 EXPERIMENT_MODE=dynamic bash scripts/submit.sh

set -euo pipefail

WALLTIME=${WALLTIME:-08:00:00}
TTCA_ALPHA=${TTCA_ALPHA:-1.0}
TTCA_COST_BETA=${TTCA_COST_BETA:-0.0}
EXPERIMENT_MODE=${EXPERIMENT_MODE:-static}
N_REQUESTS=${N_REQUESTS:-1000}
CONCURRENCY=${CONCURRENCY:-50}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-30}
PROJECT=${PROJECT:-UIC-HPC}
QUEUE=${QUEUE:-by-node}

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$HOME/vllm_logs/batch_${TS}_alpha${TTCA_ALPHA}_beta${TTCA_COST_BETA}_${EXPERIMENT_MODE}"
mkdir -p "$LOG_DIR"

JOB_NAME="ttca_${EXPERIMENT_MODE}_a${TTCA_ALPHA}_b${TTCA_COST_BETA}"

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

source /soft/anaconda3/etc/profile.d/conda.sh
conda activate 2024-08-08/vllm_env
export HF_HOME=/eagle/UIC-HPC/yuping/hf_cache

cd ~/semantic-llm-router
git pull --quiet

export TTCA_ALPHA=${TTCA_ALPHA}
export TTCA_COST_BETA=${TTCA_COST_BETA}
export EXPERIMENT_MODE=${EXPERIMENT_MODE}
export N_REQUESTS=${N_REQUESTS}
export CONCURRENCY=${CONCURRENCY}
export EVAL_CONCURRENCY=${EVAL_CONCURRENCY}

echo "=================================================================="
echo "  PBS Batch Job"
echo "  Job ID        : \$PBS_JOBID"
echo "  Nodes         : \$(sort -u \$PBS_NODEFILE | tr '\n' ' ')"
echo "  TTCA_ALPHA    : ${TTCA_ALPHA}"
echo "  TTCA_COST_BETA: ${TTCA_COST_BETA}"
echo "  MODE          : ${EXPERIMENT_MODE}"
echo "  Log dir       : ${LOG_DIR}/"
echo "=================================================================="

bash scripts/run_experiment.sh

echo "Batch job complete."
PBSEOF

echo "Submitting experiment batch job..."
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
echo "  qdel $JOBID  # to cancel"
