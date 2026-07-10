#!/bin/bash
# test_comparison.sh — Run just the comparison section locally against existing results.
#
# Usage (from ~/semantic-llm-router):
#   bash scripts/test_comparison.sh results/static_vs_dynamic_20260707_170456
#
# This mirrors the comparison block in submit_static_vs_dynamic.sh so you can
# verify the output before resubmitting a PBS job.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <results-dir>"
    echo "  e.g. $0 results/static_vs_dynamic_20260707_170456"
    exit 1
fi

RESULTS_DIR="$1"

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "ERROR: results directory not found: $RESULTS_DIR"
    exit 1
fi

echo "=================================================================="
echo "  Testing comparison section"
echo "  Results: $RESULTS_DIR"
echo "=================================================================="

# ── Build --system list (same logic as the PBS script) ────────────────────────
COMPARE_ARGS=(tests/compare_all.py)
[ -f "results/rr_baseline.csv" ] && \
  COMPARE_ARGS+=(--system "Round-Robin:results/rr_baseline.csv")
[ -f "$RESULTS_DIR/baseline_cascade.csv" ] && \
  COMPARE_ARGS+=(--system "Cascade:$RESULTS_DIR/baseline_cascade.csv")
[ -f "$RESULTS_DIR/baseline_routellm.csv" ] && \
  COMPARE_ARGS+=(--system "RouteLLM:$RESULTS_DIR/baseline_routellm.csv")
[ -f "$RESULTS_DIR/baseline_carrot.csv" ] && \
  COMPARE_ARGS+=(--system "CARROT:$RESULTS_DIR/baseline_carrot.csv")
[ -f "$RESULTS_DIR/baseline_omni_router.csv" ] && \
  COMPARE_ARGS+=(--system "OmniRouter:$RESULTS_DIR/baseline_omni_router.csv")
[ -f "$RESULTS_DIR/static_results.csv" ] && \
  COMPARE_ARGS+=(--system "Static (TTCA):$RESULTS_DIR/static_results.csv")
[ -f "$RESULTS_DIR/dynamic_results.csv" ] && \
  COMPARE_ARGS+=(--system "Dynamic (TTCA):$RESULTS_DIR/dynamic_results.csv")
[ -f "$RESULTS_DIR/baseline_tier_acc_optimal.csv" ] && \
  COMPARE_ARGS+=(--system "Tier-Opt-Acc:$RESULTS_DIR/baseline_tier_acc_optimal.csv")
[ -f "$RESULTS_DIR/baseline_cost_optimal.csv" ] && \
  COMPARE_ARGS+=(--system "Tier-Opt-Cost:$RESULTS_DIR/baseline_cost_optimal.csv")

if [ "${#COMPARE_ARGS[@]}" -eq 1 ]; then
    echo "ERROR: no result CSVs found in $RESULTS_DIR"
    exit 1
fi

COMPARE_ARGS+=(--ref "Static (TTCA)")
[ -f "$RESULTS_DIR/eval_matrix.csv" ] && \
  COMPARE_ARGS+=(--eval-matrix "$RESULTS_DIR/eval_matrix.csv")

# ── compare_all.py ranked table ───────────────────────────────────────────────
echo ""
echo "  [compare_all.py] Ranking all systems..."
python "${COMPARE_ARGS[@]}" 2>&1
echo "  [compare_all.py] Done."

# ── Static vs Dynamic: full metric comparison ─────────────────────────────────
echo ""
echo "=================================================================="
echo "  STATIC vs DYNAMIC — Full Metric Comparison"
echo "=================================================================="
export RESULTS_DIR_PY="$RESULTS_DIR"
python3 << PYEOF
import csv, sys, os
from statistics import mean

rd = os.environ['RESULTS_DIR_PY']

def load(path, label):
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print('  WARNING: ' + path + ' not found')
        return None
    ok = [r for r in rows if r.get('status') == '200']
    scored = [r for r in ok if r.get('correct') in ('true','false','True','False','0','1')]
    correct = [r for r in scored if r.get('correct') in ('true','True','1')]
    lats = [float(r.get('actual_latency_ms') or r.get('wall_ms', 0))
            for r in ok if r.get('actual_latency_ms') or r.get('wall_ms')]
    costs = [float(r['charged_usd']) for r in ok if r.get('charged_usd')]
    energy = [float(r['energy_j']) for r in ok if r.get('energy_j')]
    slo_rows = [r for r in rows if r.get('slo_violated') in ('true','false','True','False')]
    slo_viol = [r for r in slo_rows if r.get('slo_violated') in ('true','True')]
    retries = [int(r['retries']) for r in ok if r.get('retries','') not in ('','0',None)]
    slats = sorted(lats)
    return {
        'label':         label,
        'n_total':       len(rows),
        'n_ok':          len(ok),
        'accuracy':      len(correct)/len(scored)*100 if scored else None,
        'lat_mean':      mean(lats)/1000              if lats   else None,
        'lat_p50':       slats[len(slats)//2]/1000    if lats   else None,
        'lat_p95':       slats[int(len(slats)*0.95)]/1000 if lats else None,
        'cost_mean':     mean(costs)                  if costs  else None,
        'energy_mean':   mean(energy)                 if energy else None,
        'slo_viol_n':    len(slo_viol),
        'slo_total':     len(slo_rows),
        'slo_pct':       len(slo_viol)/len(slo_rows)*100 if slo_rows else None,
        'retries_total': sum(int(r.get('retries',0)) for r in ok),
        'retries_mean':  mean(retries) if retries else 0.0,
    }

s = load(rd + '/static_results.csv',  'Static')
d = load(rd + '/dynamic_results.csv', 'Dynamic')
if not s or not d:
    sys.exit(0)

def fmt(v, spec, suffix=''):
    return ((spec % v) + suffix) if v is not None else '-'
def usd(v):
    return ('$' + '%.8f' % v) if v is not None else '-'
def delta(sv, dv, spec, suffix=''):
    if sv is None or dv is None: return '-'
    diff = dv - sv
    sign = '+' if diff >= 0 else ''
    return sign + (spec % diff) + suffix

W1, W2, W3, W4 = 28, 14, 14, 14
sep = '-' * (W1 + W2 + W3 + W4 + 3)
print('\n  ' + ('Metric').ljust(W1) + ('Static').rjust(W2) + ' ' + ('Dynamic').rjust(W3) + ' ' + ('Delta (D-S)').rjust(W4))
print('  ' + sep)
print('  ' + ('Requests (200 OK)').ljust(W1) + str(s['n_ok']).rjust(W2) + ' ' + str(d['n_ok']).rjust(W3))
print('  ' + ('Accuracy').ljust(W1) + fmt(s['accuracy'],'%.1f','%').rjust(W2) + ' ' + fmt(d['accuracy'],'%.1f','%').rjust(W3) + ' ' + delta(s['accuracy'],d['accuracy'],'%.1f','pp').rjust(W4))
print('  ' + ('TTCA mean (lat mean)').ljust(W1) + fmt(s['lat_mean'],'%.2f','s').rjust(W2) + ' ' + fmt(d['lat_mean'],'%.2f','s').rjust(W3) + ' ' + delta(s['lat_mean'],d['lat_mean'],'%.2f','s').rjust(W4))
print('  ' + ('Lat P50').ljust(W1) + fmt(s['lat_p50'],'%.2f','s').rjust(W2) + ' ' + fmt(d['lat_p50'],'%.2f','s').rjust(W3) + ' ' + delta(s['lat_p50'],d['lat_p50'],'%.2f','s').rjust(W4))
print('  ' + ('Lat P95').ljust(W1) + fmt(s['lat_p95'],'%.2f','s').rjust(W2) + ' ' + fmt(d['lat_p95'],'%.2f','s').rjust(W3) + ' ' + delta(s['lat_p95'],d['lat_p95'],'%.2f','s').rjust(W4))
print('  ' + ('Cost/req').ljust(W1) + usd(s['cost_mean']).rjust(W2) + ' ' + usd(d['cost_mean']).rjust(W3))
print('  ' + ('SLO violations').ljust(W1) + (str(s['slo_viol_n'])+'/'+str(s['slo_total'])).rjust(W2) + ' ' + (str(d['slo_viol_n'])+'/'+str(d['slo_total'])).rjust(W3))
print('  ' + ('SLO violation rate').ljust(W1) + fmt(s['slo_pct'],'%.1f','%').rjust(W2) + ' ' + fmt(d['slo_pct'],'%.1f','%').rjust(W3) + ' ' + delta(s['slo_pct'],d['slo_pct'],'%.1f','pp').rjust(W4))
print('  ' + ('Total retries').ljust(W1) + str(s['retries_total']).rjust(W2) + ' ' + str(d['retries_total']).rjust(W3) + ' ' + delta(s['retries_total'],d['retries_total'],'%d').rjust(W4))
print('  ' + ('Avg retries/req').ljust(W1) + fmt(s['retries_mean'],'%.3f').rjust(W2) + ' ' + fmt(d['retries_mean'],'%.3f').rjust(W3))
PYEOF
