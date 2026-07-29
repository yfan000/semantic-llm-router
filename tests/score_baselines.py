"""
score_baselines.py — Post-hoc scoring for baseline CSVs.

Baseline scripts (CARROT, OmniRouter, Cascade, Round-Robin) write response_text
and ground_truth but no is_correct column.  This script adds:
  score        — float 0.0-1.0 from the benchmark-appropriate scorer
  is_correct   — "true" / "false" (score >= 0.9 threshold)
  answer_type  — copied from workload (mcq / numeric / expression / code)
  source       — copied from workload (gpqa_diamond / mmlu_pro / ...)

It updates each CSV file in-place (original is preserved as <file>.bak).

Usage:
    python tests/score_baselines.py \\
        --workload  results/benchmark_svd_20260728_192259/workload.json \\
        --csvs      results/benchmark_svd_20260728_192259/baseline_carrot.csv \\
                    results/benchmark_svd_20260728_192259/baseline_cascade.csv \\
                    results/benchmark_svd_20260728_192259/baseline_omni_router.csv \\
                    results/benchmark_svd_20260728_192259/rr_baseline.csv

    # Or score every baseline in a results dir at once:
    RDIR=results/benchmark_svd_20260728_192259
    python tests/score_baselines.py \\
        --workload $RDIR/workload.json \\
        --csvs $RDIR/baseline_carrot.csv $RDIR/baseline_cascade.csv \\
               $RDIR/baseline_omni_router.csv $RDIR/rr_baseline.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

# Import scorers from run_benchmark_eval (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from run_benchmark_eval import SCORERS, _DOMAIN_FALLBACK

CORRECT_THRESHOLD = 0.9


def score_row(response_text: str, ground_truth: str,
              answer_type: str, domain: str) -> tuple[float | None, str]:
    """Return (score, is_correct_str). is_correct_str is '' when un-scoreable."""
    scorer = SCORERS.get(answer_type) or _DOMAIN_FALLBACK.get(domain)
    if scorer is None or not response_text.strip():
        return None, ""
    sc = scorer(response_text, ground_truth)
    if sc is None:
        return None, ""
    is_correct = "true" if sc >= CORRECT_THRESHOLD else "false"
    return sc, is_correct


def process_csv(csv_path: str, workload_index: dict[str, dict]) -> None:
    """Add score / is_correct / answer_type / source to one CSV, in-place."""
    bak_path = csv_path + ".bak"
    shutil.copy2(csv_path, bak_path)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fields = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        print(f"  {csv_path}: empty, skipping")
        return

    # Determine extra fields to add (only if not already present)
    new_fields: list[str] = []
    for col in ("answer_type", "source", "score", "is_correct"):
        if col not in original_fields:
            new_fields.append(col)

    all_fields = original_fields + new_fields

    scored = skipped = already = 0
    for row in rows:
        # Skip if already scored
        if row.get("is_correct") in ("true", "false"):
            already += 1
            continue

        req_id = str(row.get("req_id", "")).strip()
        item   = workload_index.get(req_id)
        if item is None:
            skipped += 1
            for col in new_fields:
                row.setdefault(col, "")
            continue

        # Populate from workload if missing in the row
        answer_type = row.get("answer_type") or item.get("answer_type", "")
        source      = row.get("source")      or item.get("source",      "")
        domain      = row.get("domain")      or item.get("domain",      "")
        ground_truth = row.get("ground_truth") or item.get("ground_truth", "")
        response_text = row.get("response_text", "").strip()

        row["answer_type"] = answer_type
        row["source"]      = source

        sc, is_correct = score_row(response_text, ground_truth, answer_type, domain)
        row["score"]      = f"{sc:.4f}" if sc is not None else ""
        row["is_correct"] = is_correct
        scored += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    n_correct = sum(1 for r in rows if r.get("is_correct") == "true")
    n_scored  = sum(1 for r in rows if r.get("is_correct") in ("true", "false"))
    acc = n_correct / n_scored * 100 if n_scored else 0
    print(f"  {Path(csv_path).name:<40}  "
          f"scored={scored}  skipped={skipped}  already={already}  "
          f"accuracy={acc:.1f}%  ({n_correct}/{n_scored})  bak={Path(bak_path).name}")


def main() -> None:
    global CORRECT_THRESHOLD
    parser = argparse.ArgumentParser(
        description="Add is_correct/score to baseline CSVs using workload ground truth."
    )
    parser.add_argument("--workload", required=True,
                        help="workload.json produced by the submit script")
    parser.add_argument("--csvs", nargs="+", required=True,
                        help="One or more baseline CSV files to score in-place")
    parser.add_argument("--threshold", type=float, default=CORRECT_THRESHOLD,
                        help=f"Score threshold for is_correct=true (default {CORRECT_THRESHOLD})")
    args = parser.parse_args()

    CORRECT_THRESHOLD = args.threshold

    # Load workload, index by req_id (both int and str keys)
    with open(args.workload) as f:
        workload = json.load(f)
    workload_index: dict[str, dict] = {}
    for item in workload:
        rid = item.get("req_id")
        if rid is not None:
            workload_index[str(rid)] = item
    print(f"Workload: {len(workload_index)} items from {args.workload}")
    print()

    for csv_path in args.csvs:
        if not os.path.exists(csv_path):
            print(f"  WARNING: not found, skipping: {csv_path}")
            continue
        process_csv(csv_path, workload_index)

    print()
    print("Done. Re-run compare_all.py to see updated accuracy numbers.")
    print()
    print("Example:")
    rdir = str(Path(args.csvs[0]).parent)
    print(f"  python tests/compare_all.py \\")
    print(f'    --system "Round-Robin:{rdir}/rr_baseline.csv" \\')
    print(f'    --system "Cascade:{rdir}/baseline_cascade.csv" \\')
    print(f'    --system "CARROT:{rdir}/baseline_carrot.csv" \\')
    print(f'    --system "OmniRouter:{rdir}/baseline_omni_router.csv" \\')
    print(f'    --system "Static (TTCA):{rdir}/static_results.csv" \\')
    print(f'    --system "Dynamic (TTCA):{rdir}/dynamic_results.csv" \\')
    print(f'    --ref "Static (TTCA)" \\')
    print(f'    --output {rdir}/compare_all_scored.csv')


if __name__ == "__main__":
    main()
