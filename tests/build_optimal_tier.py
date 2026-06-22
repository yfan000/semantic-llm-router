"""
build_optimal_tier.py — Build data-driven optimal tier maps from eval matrix.

After running eval_all_models.py you have accuracy and latency for every
(model × domain × complexity) cell.  This script finds the best model per
cell under two objectives:

  accuracy_optimal  — highest mean accuracy per cell (ignores cost/latency)
  ttca_optimal      — highest mean TTCA score per cell:
                        score = accuracy / (latency^alpha × cost^beta)

The output JSON is consumed by baseline_complexity_tier.py via --tier-map.

Usage:
    python tests/build_optimal_tier.py \\
        --eval-matrix results/eval_matrix.csv \\
        --output      results/optimal_tier_maps.json \\
        --alpha 1.0 --beta 0.0
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# Cost rates (USD per token) and energy efficiency — must match baseline_complexity_tier.py
COST_RATES: dict[str, dict] = {
    "qwen-7b":         {"input": 3e-7, "output": 6e-7, "eff": 13.0},
    "deepseek-r1-7b":  {"input": 3e-7, "output": 6e-7, "eff": 13.0},
    "qwen3-coder-30b": {"input": 7e-7, "output": 14e-7, "eff": 12.0},
    "gemma-3-27b":     {"input": 8e-7, "output": 16e-7, "eff": 5.0},
    "deepseek-r1-14b": {"input": 5e-7, "output": 10e-7, "eff": 6.0},
    "llama4-scout":    {"input": 10e-7, "output": 20e-7, "eff": 3.0},
}


def _estimate_cost(model_id: str, query: str, response_text: str) -> float:
    rates = COST_RATES.get(model_id, {"input": 5e-7, "output": 10e-7})
    in_tok  = len(query.split()) * 1.3
    out_tok = len(response_text.split()) * 1.3 if response_text else 200
    return in_tok * rates["input"] + out_tok * rates["output"]


def build_maps(eval_matrix: str, alpha: float, beta: float) -> dict:
    # Accumulators: key=(domain, complexity, model_id)
    acc_sum:   dict[tuple, float] = defaultdict(float)
    ttca_sum:  dict[tuple, float] = defaultdict(float)
    counts:    dict[tuple, int]   = defaultdict(int)

    with open(eval_matrix, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "200":
                continue
            if row.get("is_correct") not in ("true", "false"):
                continue

            domain     = row["domain"]
            complexity = row["complexity"]
            model_id   = row["model_id"]
            key        = (domain, complexity, model_id)

            correct  = 1.0 if row["is_correct"] == "true" else 0.0
            wall_ms  = float(row["wall_ms"]) if row["wall_ms"] else 1000.0
            cost     = _estimate_cost(model_id, row.get("query", ""), row.get("response_text", ""))

            acc  = max(correct, 0.01)
            lat  = max(wall_ms, 1.0)
            cost = max(cost, 1e-9)

            ttca_score = acc / ((lat ** alpha) * (cost ** beta))

            acc_sum[key]  += correct
            ttca_sum[key] += ttca_score
            counts[key]   += 1

    # Compute per-cell means
    cells: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for (domain, complexity, model_id), n in counts.items():
        cell_key = (domain, complexity)
        cells[cell_key][model_id] = {
            "accuracy":   acc_sum[(domain, complexity, model_id)] / n,
            "ttca_score": ttca_sum[(domain, complexity, model_id)] / n,
            "n":          n,
        }

    # Pick best model per cell for each objective
    accuracy_optimal: dict[str, str] = {}
    ttca_optimal:     dict[str, str] = {}
    summary_rows: list[dict] = []

    for (domain, complexity), models in sorted(cells.items()):
        cell_str = f"{domain}:{complexity}"

        best_acc  = max(models, key=lambda m: models[m]["accuracy"])
        best_ttca = max(models, key=lambda m: models[m]["ttca_score"])

        accuracy_optimal[cell_str] = best_acc
        ttca_optimal[cell_str]     = best_ttca

        # Summary row
        for model_id, stats in sorted(models.items()):
            summary_rows.append({
                "domain":      domain,
                "complexity":  complexity,
                "model_id":    model_id,
                "accuracy":    round(stats["accuracy"], 4),
                "ttca_score":  round(stats["ttca_score"], 6),
                "n":           stats["n"],
                "best_acc":    model_id == best_acc,
                "best_ttca":   model_id == best_ttca,
            })

    return {
        "accuracy_optimal": accuracy_optimal,
        "ttca_optimal":     ttca_optimal,
        "alpha":            alpha,
        "beta":             beta,
        "summary":          summary_rows,
    }


def print_table(maps: dict) -> None:
    acc_map  = maps["accuracy_optimal"]
    ttca_map = maps["ttca_optimal"]
    summary  = {(r["domain"], r["complexity"], r["model_id"]): r for r in maps["summary"]}

    cells = sorted({tuple(k.split(":")) for k in acc_map})

    print(f"\n  {'Cell':<20} {'Accuracy-Optimal':<22} {'acc':>6}  {'TTCA-Optimal':<22} {'ttca':>10}")
    print(f"  {'-'*82}")
    for domain, complexity in cells:
        cell_str = f"{domain}:{complexity}"
        am = acc_map.get(cell_str, "-")
        tm = ttca_map.get(cell_str, "-")
        am_acc   = summary.get((domain, complexity, am),  {}).get("accuracy",   0)
        tm_ttca  = summary.get((domain, complexity, tm),  {}).get("ttca_score", 0)
        print(f"  {cell_str:<20} {am:<22} {am_acc:>6.3f}  {tm:<22} {tm_ttca:>10.4f}")

    print()
    if acc_map == ttca_map:
        print("  Note: accuracy-optimal and TTCA-optimal maps are identical for these settings.")
    else:
        diffs = [(k, acc_map[k], ttca_map[k]) for k in acc_map if acc_map[k] != ttca_map[k]]
        print(f"  Cells where optimal model differs: {len(diffs)}")
        for cell, am, tm in diffs:
            print(f"    {cell}: accuracy→{am}  ttca→{tm}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-matrix", required=True,
                        help="Path to eval_matrix.csv from eval_all_models.py")
    parser.add_argument("--output", default="results/optimal_tier_maps.json")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Latency exponent for TTCA score (default 1.0)")
    parser.add_argument("--beta",  type=float, default=0.0,
                        help="Cost exponent for TTCA score (default 0.0)")
    args = parser.parse_args()

    print(f"\n  Building optimal tier maps from: {args.eval_matrix}")
    print(f"  TTCA params: alpha={args.alpha}  beta={args.beta}")

    maps = build_maps(args.eval_matrix, args.alpha, args.beta)

    print_table(maps)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(maps, f, indent=2)
    print(f"  Saved: {args.output}")
    print(f"    accuracy_optimal: {len(maps['accuracy_optimal'])} cells")
    print(f"    ttca_optimal:     {len(maps['ttca_optimal'])} cells\n")


if __name__ == "__main__":
    main()
