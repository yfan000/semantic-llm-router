#!/bin/bash
# run_benchmark_comparison.sh -- Run all routing methods against a benchmark dataset
# and produce a unified compare_all.py report.
#
# Methods run:
#   TTCA         -- semantic router, mode=ttca  (load_test.py)
#   Accuracy     -- semantic router, mode=accuracy  (load_test.py)
#   CARROT       -- ML-trained routing baseline  (baseline_carrot.py)
#   OmniRouter   -- LLM-judge routing baseline  (baseline_omni_router.py)
#   Cascade      -- RouteLLM-style weak->strong  (baseline_cascade.py)
#   Round-Robin  -- uniform distribution baseline  (round_robin_test.py)
#
# Usage:
#   bash scripts/run_benchmark_comparison.sh
#
#   # Custom dataset or request count
#   DATASET=datasets/hf_3000.json N=500 bash scripts/run_benchmark_comparison.sh
#
#   # Point at a remote router
#   ROUTER_URL=http://sophia-gpu-01:8080 bash scripts/run_benchmark_comparison.sh
#
#   # Skip slow methods
#   SKIP_OMNI=1 SKIP_CASCADE=1 bash scripts/run_benchmark_comparison.sh
#
#   # Re-run compare_all on an existing results dir (no new traffic)
#   RESULTS_DIR=results/benchmark_comparison_20260728_120000 COMPARE_ONLY=1 \
#       bash scripts/run_benchmark_comparison.sh

set -euo pipefail

# -- Configuration -------------------------------------------------------------
ROUTER_URL=${ROUTER_URL:-"http://localhost:8080"}
DATASET=${DATASET:-"datasets/benchmark_1000.json"}
N=${N:-0}                           # 0 = full dataset
CONCURRENCY=${CONCURRENCY:-16}
PRIORS=${PRIORS:-"results/priors_all5.json"}
REF_SYSTEM=${REF_SYSTEM:-"TTCA"}    # reference for delta columns in compare_all

# Skip flags -- set to 1 to skip a method
SKIP_ACCURACY=${SKIP_ACCURACY:-0}
SKIP_CARROT=${SKIP_CARROT:-0}
SKIP_OMNI=${SKIP_OMNI:-0}
SKIP_CASCADE=${SKIP_CASCADE:-0}
SKIP_RR=${SKIP_RR:-0}

# Set to 1 to skip traffic and re-run compare_all on an existing RESULTS_DIR
COMPARE_ONLY=${COMPARE_ONLY:-0}

# -- Output directory ----------------------------------------------------------
if [ -z "${RESULTS_DIR:-}" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="results/benchmark_comparison_${TS}"
fi
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/run.log"

# -- Helpers -------------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*" | tee -a "$LOG"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG"; exit 1; }
sep()  { echo "" | tee -a "$LOG"; echo "-- $* -----------------------------------------" | tee -a "$LOG"; }

N_FLAG=""
[ "$N" -gt 0 ] && N_FLAG="--requests $N"

# Track which CSVs were produced
declare -a SYS_ARGS=()

# -- Header --------------------------------------------------------------------
echo "" | tee "$LOG"
echo "===========================================================" | tee -a "$LOG"
echo "  Benchmark Comparison  $(date)" | tee -a "$LOG"
echo "  Dataset     : $DATASET" | tee -a "$LOG"
echo "  Router      : $ROUTER_URL" | tee -a "$LOG"
echo "  N requests  : ${N:-all}" | tee -a "$LOG"
echo "  Results dir : $RESULTS_DIR" | tee -a "$LOG"
echo "  Log         : $LOG" | tee -a "$LOG"
echo "===========================================================" | tee -a "$LOG"

# -- Pre-flight ----------------------------------------------------------------
if [ "$COMPARE_ONLY" -eq 0 ]; then
    sep "Pre-flight checks"

    [ -f "$DATASET" ] || err "Dataset not found: $DATASET
  Build it with:
    python tests/build_benchmark_dataset.py   # new benchmark (GPQA/MMLU-Pro/GSM1K/...)
    python tests/build_dataset.py             # original HF dataset"

    log "Dataset: $DATASET  ($(python3 -c "import json; d=json.load(open('$DATASET')); print(len(d),'items')" 2>/dev/null || echo '?'))"

    if curl --noproxy '*' -sf "$ROUTER_URL/router/health" > /dev/null 2>&1; then
        log "Router reachable at $ROUTER_URL"
    else
        err "Router not reachable at $ROUTER_URL
  Start it with:
    uvicorn semantic_router.main:app --host 0.0.0.0 --port 8080"
    fi

    # Show registered models
    MODELS=$(curl --noproxy '*' -sf "$ROUTER_URL/v1/models" \
        | python3 -c "import sys,json; [print('  ',m['id']) for m in json.load(sys.stdin)['data']]" 2>/dev/null || echo "  (could not list)")
    log "Registered models:"; echo "$MODELS" | tee -a "$LOG"
fi

# -- Helper: collect CSV into SYS_ARGS if it exists ---------------------------
collect() {
    local name=$1 csv=$2
    if [ -f "$csv" ]; then
        row_count=$(tail -n +2 "$csv" | wc -l | tr -d ' ')
        log "  Collected $name ($row_count rows) -> $csv"
        SYS_ARGS+=("--system" "${name}:${csv}")
    else
        warn "  Skipping $name -- CSV not found: $csv"
    fi
}

# -- If compare-only, collect all existing CSVs and jump to compare -----------
if [ "$COMPARE_ONLY" -eq 1 ]; then
    log "COMPARE_ONLY=1 -- skipping traffic, collecting existing CSVs"
    collect "TTCA"        "$RESULTS_DIR/ttca.csv"
    collect "Accuracy"    "$RESULTS_DIR/accuracy.csv"
    collect "CARROT"      "$RESULTS_DIR/carrot.csv"
    collect "OmniRouter"  "$RESULTS_DIR/omni_router.csv"
    collect "Cascade"     "$RESULTS_DIR/cascade.csv"
    collect "Round-Robin" "$RESULTS_DIR/round_robin.csv"
else

# -- Method 1: TTCA -----------------------------------------------------------
sep "Method 1 -- TTCA (semantic router, mode=ttca)"
log "Sending requests -> $ROUTER_URL  [mode=ttca]"
python tests/load_test.py \
    --dataset     "$DATASET" \
    --router      "$ROUTER_URL" \
    --mode        ttca \
    --concurrency "$CONCURRENCY" \
    $N_FLAG \
    --output      "$RESULTS_DIR/ttca.csv" 2>&1 | tee -a "$LOG"
collect "TTCA" "$RESULTS_DIR/ttca.csv"

# -- Method 2: Accuracy mode --------------------------------------------------
if [ "$SKIP_ACCURACY" -eq 0 ]; then
    sep "Method 2 -- Accuracy mode (semantic router, mode=accuracy)"
    log "Sending requests -> $ROUTER_URL  [mode=accuracy]"
    python tests/load_test.py \
        --dataset     "$DATASET" \
        --router      "$ROUTER_URL" \
        --mode        accuracy \
        --concurrency "$CONCURRENCY" \
        $N_FLAG \
        --output      "$RESULTS_DIR/accuracy.csv" 2>&1 | tee -a "$LOG"
    collect "Accuracy" "$RESULTS_DIR/accuracy.csv"
else
    warn "SKIP_ACCURACY=1 -- skipping accuracy mode"
fi

# -- Method 3: CARROT ---------------------------------------------------------
if [ "$SKIP_CARROT" -eq 0 ]; then
    sep "Method 3 -- CARROT baseline"
    log "Running CARROT baseline..."
    python tests/baseline_carrot.py \
        --dataset     "$DATASET" \
        --concurrency "$CONCURRENCY" \
        $N_FLAG \
        --output      "$RESULTS_DIR/carrot.csv" 2>&1 | tee -a "$LOG"
    collect "CARROT" "$RESULTS_DIR/carrot.csv"
else
    warn "SKIP_CARROT=1 -- skipping CARROT"
fi

# -- Method 4: OmniRouter -----------------------------------------------------
if [ "$SKIP_OMNI" -eq 0 ]; then
    sep "Method 4 -- OmniRouter baseline"
    log "Running OmniRouter baseline..."
    python tests/baseline_omni_router.py \
        --dataset     "$DATASET" \
        --concurrency "$CONCURRENCY" \
        $N_FLAG \
        --output      "$RESULTS_DIR/omni_router.csv" 2>&1 | tee -a "$LOG"
    collect "OmniRouter" "$RESULTS_DIR/omni_router.csv"
else
    warn "SKIP_OMNI=1 -- skipping OmniRouter"
fi

# -- Method 5: Cascade (RouteLLM-style) ---------------------------------------
if [ "$SKIP_CASCADE" -eq 0 ]; then
    sep "Method 5 -- Cascade / RouteLLM-style baseline"
    if [ ! -f "$PRIORS" ]; then
        warn "Priors file not found: $PRIORS -- skipping Cascade"
        warn "  Generate priors with: python tests/extract_priors.py --eval-matrix <eval_matrix.csv>"
    else
        log "Running Cascade baseline (priors=$PRIORS)..."
        python tests/baseline_cascade.py \
            --dataset     "$DATASET" \
            --priors      "$PRIORS" \
            --threshold   0.80 \
            --concurrency "$CONCURRENCY" \
            $N_FLAG \
            --output      "$RESULTS_DIR/cascade.csv" 2>&1 | tee -a "$LOG"
        collect "Cascade" "$RESULTS_DIR/cascade.csv"
    fi
else
    warn "SKIP_CASCADE=1 -- skipping Cascade"
fi

# -- Method 6: Round-Robin ----------------------------------------------------
if [ "$SKIP_RR" -eq 0 ]; then
    sep "Method 6 -- Round-Robin baseline"
    log "Running Round-Robin baseline..."
    python tests/round_robin_test.py \
        --dataset     "$DATASET" \
        --concurrency "$CONCURRENCY" \
        $N_FLAG \
        --output      "$RESULTS_DIR/round_robin.csv" 2>&1 | tee -a "$LOG"
    collect "Round-Robin" "$RESULTS_DIR/round_robin.csv"
else
    warn "SKIP_RR=1 -- skipping Round-Robin"
fi

fi  # end COMPARE_ONLY block

# -- compare_all.py -----------------------------------------------------------
sep "compare_all.py -- unified comparison"

if [ "${#SYS_ARGS[@]}" -eq 0 ]; then
    err "No CSVs collected -- nothing to compare."
fi

log "Comparing ${#SYS_ARGS[@]} systems with ref='$REF_SYSTEM'..."
log "Systems: ${SYS_ARGS[*]}"

# Build eval-matrix flag if one exists in the results dir
EVAL_MATRIX_FLAG=""
if [ -f "$RESULTS_DIR/eval_matrix.csv" ]; then
    EVAL_MATRIX_FLAG="--eval-matrix $RESULTS_DIR/eval_matrix.csv"
    log "Using eval matrix: $RESULTS_DIR/eval_matrix.csv"
fi

python tests/compare_all.py \
    "${SYS_ARGS[@]}" \
    $EVAL_MATRIX_FLAG \
    --ref    "$REF_SYSTEM" \
    --output "$RESULTS_DIR/compare_all.csv" \
    2>&1 | tee "$RESULTS_DIR/compare_all.txt"

# -- Done ---------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "===========================================================" | tee -a "$LOG"
echo "  Done!  $(date)" | tee -a "$LOG"
echo "  Results: $RESULTS_DIR/" | tee -a "$LOG"
echo "" | tee -a "$LOG"
for csv in ttca.csv accuracy.csv carrot.csv omni_router.csv cascade.csv round_robin.csv; do
    fp="$RESULTS_DIR/$csv"
    if [ -f "$fp" ]; then
        rows=$(tail -n +2 "$fp" | wc -l | tr -d ' ')
        printf "    %-28s %s rows\n" "$csv" "$rows" | tee -a "$LOG"
    fi
done
echo "" | tee -a "$LOG"
echo "  Primary output  : $RESULTS_DIR/compare_all.txt" | tee -a "$LOG"
echo "  CSV export      : $RESULTS_DIR/compare_all.csv" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "  Re-run compare only (no traffic):" | tee -a "$LOG"
echo "    RESULTS_DIR=$RESULTS_DIR COMPARE_ONLY=1 bash scripts/run_benchmark_comparison.sh" | tee -a "$LOG"
echo "===========================================================" | tee -a "$LOG"
