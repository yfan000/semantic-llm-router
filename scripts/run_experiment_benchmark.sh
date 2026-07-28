#!/bin/bash
# run_experiment_benchmark.sh -- Full pipeline: start servers -> eval matrix -> run all methods -> compare.
#
# Steps:
#   1. Start vLLM backends + semantic router (via start_vllm.sh)
#   2. Build eval matrix -- all models x all dataset items (optional, ~45 min)
#   3. Run all routing methods: TTCA, Accuracy, CARROT, OmniRouter, Cascade, Round-Robin
#   4. Compare all results with compare_all.py
#
# Usage:
#   bash scripts/run_experiment_benchmark.sh
#
# Environment overrides:
#   DATASET=datasets/benchmark_1000.json   # which dataset to use
#   N=500                                   # limit requests per method (0 = all)
#   ROUTER_PORT=8080                        # router port
#   CONCURRENCY=16                          # parallel request workers
#   VLLM_SCRIPT=~/start_vllm.sh            # path to start_vllm.sh
#   SKIP_START=1                            # skip step 1 (servers already running)
#   SKIP_EVAL_MATRIX=1                      # skip step 2 (faster, less accurate scoring)
#   SKIP_OMNI=1                            # skip OmniRouter (slow LLM-judge baseline)
#   SKIP_CASCADE=1                          # skip Cascade (needs priors file)
#   SKIP_ACCURACY=1                         # skip accuracy router mode
#   SKIP_CARROT=1                           # skip CARROT baseline
#   SKIP_RR=1                               # skip round-robin baseline
#
# Examples:
#   # Quick run -- skip slow baselines, only 200 requests each
#   N=200 SKIP_OMNI=1 SKIP_CASCADE=1 bash scripts/run_experiment_benchmark.sh
#
#   # Servers already running, skip startup
#   SKIP_START=1 DATASET=datasets/hf_3000.json bash scripts/run_experiment_benchmark.sh

set -euo pipefail

# -- Configuration -------------------------------------------------------------
DATASET=${DATASET:-"datasets/benchmark_1000.json"}
N=${N:-0}                                 # 0 = full dataset
ROUTER_PORT=${ROUTER_PORT:-8080}
ROUTER_URL="http://localhost:${ROUTER_PORT}"
CONCURRENCY=${CONCURRENCY:-16}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-16}
VLLM_SCRIPT=${VLLM_SCRIPT:-"$HOME/start_vllm.sh"}
PRIORS=${PRIORS:-"results/priors_all5.json"}
REF_SYSTEM=${REF_SYSTEM:-"TTCA"}

SKIP_START=${SKIP_START:-0}
SKIP_EVAL_MATRIX=${SKIP_EVAL_MATRIX:-0}
SKIP_ACCURACY=${SKIP_ACCURACY:-0}
SKIP_CARROT=${SKIP_CARROT:-0}
SKIP_OMNI=${SKIP_OMNI:-0}
SKIP_CASCADE=${SKIP_CASCADE:-0}
SKIP_RR=${SKIP_RR:-0}

# -- Output directory ----------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/benchmark_experiment_${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/pipeline.log"

# -- Helpers -------------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*" | tee -a "$LOG"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"              | tee -a "$LOG"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"               | tee -a "$LOG"; exit 1; }
sep()  { echo "" | tee -a "$LOG"
         echo "========================================================" | tee -a "$LOG"
         echo "  $*" | tee -a "$LOG"
         echo "========================================================" | tee -a "$LOG"; }

N_FLAG=""
[ "$N" -gt 0 ] && N_FLAG="--requests $N"

declare -a SYS_ARGS=()

collect() {
    local name=$1 csv=$2
    if [ -f "$csv" ]; then
        rows=$(tail -n +2 "$csv" | wc -l | tr -d ' ')
        log "  + $name  ($rows rows)"
        SYS_ARGS+=("--system" "${name}:${csv}")
    else
        warn "  - $name -- CSV not found, skipping from comparison"
    fi
}

# -- Banner --------------------------------------------------------------------
echo "" | tee "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  Benchmark Experiment Pipeline  $(date)"                     | tee -a "$LOG"
echo "  Dataset      : $DATASET"                                    | tee -a "$LOG"
echo "  N per method : ${N:-all}"                                   | tee -a "$LOG"
echo "  Router       : $ROUTER_URL"                                 | tee -a "$LOG"
echo "  Concurrency  : $CONCURRENCY"                                | tee -a "$LOG"
echo "  Results dir  : $RESULTS_DIR"                                | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Sanity-check dataset
[ -f "$DATASET" ] || err "Dataset not found: $DATASET
  Build it first:
    python tests/build_benchmark_dataset.py --total 1000   # new benchmark
    python tests/build_dataset.py --count 3000             # original HF dataset"

NITEMS=$(python3 -c "import json; print(len(json.load(open('$DATASET'))))" 2>/dev/null || echo "?")
log "Dataset: $DATASET  ($NITEMS items)"

# ============================================================
# STEP 1 -- Start vLLM backends + router
# ============================================================
sep "Step 1 -- Start vLLM backends + router"

if [ "$SKIP_START" -eq 1 ]; then
    warn "SKIP_START=1 -- assuming servers are already running"
    if ! curl --noproxy '*' -sf "$ROUTER_URL/router/health" > /dev/null 2>&1; then
        err "Router not reachable at $ROUTER_URL  (set SKIP_START=0 to start it)"
    fi
    log "Router confirmed at $ROUTER_URL"
else
    [ -f "$VLLM_SCRIPT" ] || err "start_vllm.sh not found: $VLLM_SCRIPT
  Set VLLM_SCRIPT=/path/to/start_vllm.sh"

    log "Running $VLLM_SCRIPT start ..."
    bash "$VLLM_SCRIPT" start 2>&1 | tee -a "$LOG"
    log "All vLLM backends and router are ready."
fi

# Show registered models
log "Registered models:"
curl --noproxy '*' -sf "$ROUTER_URL/v1/models" \
    | python3 -c "import sys,json; [print('    -', m['id']) for m in json.load(sys.stdin)['data']]" \
    2>/dev/null | tee -a "$LOG" || warn "  (could not list models)"

# ============================================================
# STEP 2 -- Build eval matrix (optional)
# ============================================================
sep "Step 2 -- Eval matrix (all models x all items)"
EVAL_MATRIX_FLAG=""

if [ "$SKIP_EVAL_MATRIX" -eq 1 ]; then
    warn "SKIP_EVAL_MATRIX=1 -- compare_all will use inline gt_correct from each CSV"
else
    log "Running eval_all_models.py (this takes ~30-60 min for 1000 items x 4 models)..."
    python tests/eval_all_models.py \
        --dataset     "$DATASET" \
        --output      "$RESULTS_DIR/eval_matrix.csv" \
        --concurrency "$EVAL_CONCURRENCY" \
        2>&1 | tee -a "$LOG"

    if [ -f "$RESULTS_DIR/eval_matrix.csv" ]; then
        rows=$(tail -n +2 "$RESULTS_DIR/eval_matrix.csv" | wc -l | tr -d ' ')
        log "Eval matrix: $rows rows -> $RESULTS_DIR/eval_matrix.csv"
        EVAL_MATRIX_FLAG="--eval-matrix $RESULTS_DIR/eval_matrix.csv"
    else
        warn "Eval matrix not produced -- continuing without it"
    fi
fi

# ============================================================
# STEP 3 -- Run all routing methods
# ============================================================
sep "Step 3 -- Run all routing methods"

# -- 3a: TTCA -----------------------------------------------------------------
log "[3a] TTCA  (semantic router, mode=ttca)"
python tests/load_test.py \
    --dataset     "$DATASET" \
    --router      "$ROUTER_URL" \
    --mode        ttca \
    --concurrency "$CONCURRENCY" \
    $N_FLAG \
    --output      "$RESULTS_DIR/ttca.csv" \
    2>&1 | tee -a "$LOG"
collect "TTCA" "$RESULTS_DIR/ttca.csv"

# -- 3b: Accuracy mode --------------------------------------------------------
if [ "$SKIP_ACCURACY" -eq 0 ]; then
    log "[3b] Accuracy  (semantic router, mode=accuracy)"
    python tests/load_test.py \
        --dataset     "$DATASET" \
        --router      "$ROUTER_URL" \
        --mode        accuracy \
        --concurrency "$CONCURRENCY" \
        $N_FLAG \
        --output      "$RESULTS_DIR/accuracy.csv" \
        2>&1 | tee -a "$LOG"
    collect "Accuracy" "$RESULTS_DIR/accuracy.csv"
else
    warn "[3b] Accuracy mode skipped (SKIP_ACCURACY=1)"
fi

# -- 3c: CARROT ---------------------------------------------------------------
if [ "$SKIP_CARROT" -eq 0 ]; then
    log "[3c] CARROT baseline"
    python tests/baseline_carrot.py \
        --dataset     "$DATASET" \
        --concurrency "$CONCURRENCY" \
        $N_FLAG \
        --output      "$RESULTS_DIR/carrot.csv" \
        2>&1 | tee -a "$LOG"
    collect "CARROT" "$RESULTS_DIR/carrot.csv"
else
    warn "[3c] CARROT skipped (SKIP_CARROT=1)"
fi

# -- 3d: OmniRouter -----------------------------------------------------------
if [ "$SKIP_OMNI" -eq 0 ]; then
    log "[3d] OmniRouter baseline  (note: calls LLM judge per request -- slow)"
    python tests/baseline_omni_router.py \
        --dataset     "$DATASET" \
        --concurrency "$CONCURRENCY" \
        $N_FLAG \
        --output      "$RESULTS_DIR/omni_router.csv" \
        2>&1 | tee -a "$LOG"
    collect "OmniRouter" "$RESULTS_DIR/omni_router.csv"
else
    warn "[3d] OmniRouter skipped (SKIP_OMNI=1)"
fi

# -- 3e: Cascade --------------------------------------------------------------
if [ "$SKIP_CASCADE" -eq 0 ]; then
    if [ ! -f "$PRIORS" ]; then
        warn "[3e] Cascade skipped -- priors file not found: $PRIORS
       Generate it with: python tests/extract_priors.py --eval-matrix <eval_matrix.csv> --output $PRIORS"
    else
        log "[3e] Cascade baseline  (priors=$PRIORS, threshold=0.80)"
        python tests/baseline_cascade.py \
            --dataset     "$DATASET" \
            --priors      "$PRIORS" \
            --threshold   0.80 \
            --concurrency "$CONCURRENCY" \
            $N_FLAG \
            --output      "$RESULTS_DIR/cascade.csv" \
            2>&1 | tee -a "$LOG"
        collect "Cascade" "$RESULTS_DIR/cascade.csv"
    fi
else
    warn "[3e] Cascade skipped (SKIP_CASCADE=1)"
fi

# -- 3f: Round-Robin ----------------------------------------------------------
if [ "$SKIP_RR" -eq 0 ]; then
    log "[3f] Round-Robin baseline"
    python tests/round_robin_test.py \
        --dataset     "$DATASET" \
        --concurrency "$CONCURRENCY" \
        $N_FLAG \
        --output      "$RESULTS_DIR/round_robin.csv" \
        2>&1 | tee -a "$LOG"
    collect "Round-Robin" "$RESULTS_DIR/round_robin.csv"
else
    warn "[3f] Round-Robin skipped (SKIP_RR=1)"
fi

# ============================================================
# STEP 4 -- compare_all.py
# ============================================================
sep "Step 4 -- compare_all.py"

[ "${#SYS_ARGS[@]}" -gt 0 ] || err "No result CSVs produced -- nothing to compare."

log "Comparing systems, ref='$REF_SYSTEM' ..."

python tests/compare_all.py \
    "${SYS_ARGS[@]}" \
    $EVAL_MATRIX_FLAG \
    --ref    "$REF_SYSTEM" \
    --output "$RESULTS_DIR/compare_all.csv" \
    2>&1 | tee "$RESULTS_DIR/compare_all.txt"

# -- Summary ------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  Pipeline complete!  $(date)"                               | tee -a "$LOG"
echo "  Results: $RESULTS_DIR/"                                    | tee -a "$LOG"
echo ""                                                             | tee -a "$LOG"
echo "  Files produced:"                                           | tee -a "$LOG"
for csv in eval_matrix.csv ttca.csv accuracy.csv carrot.csv omni_router.csv cascade.csv round_robin.csv; do
    fp="$RESULTS_DIR/$csv"
    if [ -f "$fp" ]; then
        rows=$(tail -n +2 "$fp" | wc -l | tr -d ' ')
        printf "    %-32s %s rows\n" "$csv" "$rows" | tee -a "$LOG"
    fi
done
echo "" | tee -a "$LOG"
echo "  Main report  : $RESULTS_DIR/compare_all.txt"           | tee -a "$LOG"
echo "  CSV export   : $RESULTS_DIR/compare_all.csv"           | tee -a "$LOG"
echo "  Full log     : $LOG"                                    | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
