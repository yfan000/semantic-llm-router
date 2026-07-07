#!/bin/bash
# submit_sensitivity.sh — Sweep TTCA α and β parameters for sensitivity study.
#
# Submits one PBS job per parameter value via scripts/submit.sh.
# Results land in results/experiment_<ts>_alpha<v>_beta<v>_static/
#
# Usage:
#   bash scripts/submit_sensitivity.sh --alpha-sweep
#   bash scripts/submit_sensitivity.sh --beta-sweep
#
# Optional overrides:
#   N_REQUESTS=300 CONCURRENCY=50 WALLTIME=04:00:00 bash scripts/submit_sensitivity.sh --alpha-sweep
#
# After all jobs finish, aggregate results:
#   python tests/analyze_sensitivity.py --sweep-param alpha \
#       --results-glob "results/experiment_*_beta0.0_static"
#   python tests/analyze_sensitivity.py --sweep-param beta \
#       --results-glob "results/experiment_*_alpha1.0_beta*_static"

set -euo pipefail

# Both sweeps use the same value set for symmetry.
# Note: β effects are subtler at equal numeric values because cost is in USD (~1e-8)
# while latency is in ms (~1000–60000). β needs to be large before it visibly changes routing.
SWEEP_VALUES=(0.0 0.3 0.5 1.0 1.5 2.0)

N_REQUESTS=${N_REQUESTS:-300}
CONCURRENCY=${CONCURRENCY:-50}
WALLTIME=${WALLTIME:-04:00:00}   # shorter than full experiment (no large eval matrix needed)
EXPERIMENT_MODE=static            # sensitivity uses static mode for reproducibility

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 --alpha-sweep | --beta-sweep"
    exit 1
fi

MODE="$1"

if [[ "$MODE" == "--alpha-sweep" ]]; then
    echo "========================================================"
    echo "  α sensitivity sweep: [${SWEEP_VALUES[*]}]"
    echo "  TTCA_COST_BETA fixed at 0.0 (cost excluded)"
    echo "  All per-domain alphas set uniformly to sweep value"
    echo "  N_REQUESTS=$N_REQUESTS  CONCURRENCY=$CONCURRENCY"
    echo "========================================================"
    echo ""

    for alpha in "${SWEEP_VALUES[@]}"; do
        echo "  Submitting α=$alpha ..."
        TTCA_ALPHA="$alpha" \
        TTCA_ALPHA_FACTUAL="$alpha" \
        TTCA_ALPHA_MATH="$alpha" \
        TTCA_ALPHA_CODE="$alpha" \
        TTCA_ALPHA_REASONING="$alpha" \
        TTCA_COST_BETA=0.0 \
        EXPERIMENT_MODE="$EXPERIMENT_MODE" \
        N_REQUESTS="$N_REQUESTS" \
        CONCURRENCY="$CONCURRENCY" \
        WALLTIME="$WALLTIME" \
        bash scripts/submit.sh
        sleep 3   # avoid PBS submission flood
        echo ""
    done

    echo "========================================================"
    echo "  All α sweep jobs submitted (${#SWEEP_VALUES[@]} jobs)."
    echo ""
    echo "  After jobs finish, analyze with:"
    echo "    python tests/analyze_sensitivity.py \\"
    echo "        --sweep-param alpha \\"
    echo "        --results-glob 'results/experiment_*_beta0.0_static'"
    echo "========================================================"

elif [[ "$MODE" == "--beta-sweep" ]]; then
    echo "========================================================"
    echo "  β sensitivity sweep: [${SWEEP_VALUES[*]}]"
    echo "  Per-domain α at paper defaults (factual=0.3, math=0.7, code=1.0, reasoning=0.7)"
    echo "  N_REQUESTS=$N_REQUESTS  CONCURRENCY=$CONCURRENCY"
    echo "========================================================"
    echo ""

    for beta in "${SWEEP_VALUES[@]}"; do
        echo "  Submitting β=$beta ..."
        # Do NOT override per-domain alphas — let run_experiment.sh use its defaults
        TTCA_COST_BETA="$beta" \
        EXPERIMENT_MODE="$EXPERIMENT_MODE" \
        N_REQUESTS="$N_REQUESTS" \
        CONCURRENCY="$CONCURRENCY" \
        WALLTIME="$WALLTIME" \
        bash scripts/submit.sh
        sleep 3
        echo ""
    done

    echo "========================================================"
    echo "  All β sweep jobs submitted (${#SWEEP_VALUES[@]} jobs)."
    echo ""
    echo "  After jobs finish, analyze with:"
    echo "    python tests/analyze_sensitivity.py \\"
    echo "        --sweep-param beta \\"
    echo "        --results-glob 'results/experiment_*_alpha1.0_beta*_static'"
    echo "========================================================"

else
    echo "ERROR: Unknown mode '$MODE'"
    echo "Usage: $0 --alpha-sweep | --beta-sweep"
    exit 1
fi
