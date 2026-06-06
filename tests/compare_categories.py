"""
compare_categories.py — Per-category comparison of router vs baseline.

Produces a detailed breakdown by domain×complexity showing:
  - Request count
  - Accuracy (% correct answers)
  - Latency (P50, P95, mean)
  - Energy cost (J)
  - Charged cost (USD)
  - Model routing distribution

Usage:
    python tests/compare_categories.py \\
        --router      results/router_ttca.csv \\
        --baseline    results/rr_baseline.csv \\
        --eval-matrix results/eval_matrix.csv \\
        --output      results/compare_categories.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from statistics import mean


CATEGORIES = [
    ("factual",   "easy"),   ("factual",   "medium"),  ("factual",   "hard"),
    ("math",      "easy"),   ("math",      "medium"),   ("math",      "hard"),
    ("code",      "easy"),   ("code",      "medium"),   ("code",      "hard"),
    ("reasoning", "easy"),   ("reasoning", "medium"),   ("reasoning", "hard"),
    ("creative",  "easy"),   ("creative",  "medium"),   ("creative",  "hard"),
]


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(v) -> float | None:
    try:
        return float(v) if v and str(v).strip() else None
    except (ValueError, TypeError):
        return None


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def build_eval_index(eval_matrix_path: str) -> dict:
    """Build lookup: (req_id, model_id) -> is_correct bool."""
    if not eval_matrix_path or not os.path.exists(eval_matrix_path):
        return {}
    index = {}
    with open(eval_matrix_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ic = row.get("is_correct", "")
            if ic in ("true", "false"):
                index[(row.get("req_id", ""), row.get("model_id", ""))] = (ic == "true")
    return index


def summarize(rows: list[dict], eval_index: dict) -> dict:
    ok = [r for r in rows if str(r.get("status")) == "200"]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in ok:
        key = (r.get("domain", "?"), r.get("complexity", "?"))
        groups[key].append(r)

    stats = {}
    for cat in CATEGORIES:
        g = groups.get(cat, [])
        lats  = [v for r in g if (v := safe_float(r.get("actual_latency_ms") or r.get("wall_ms"))) is not None]
        enrgs = [v for r in g if (v := safe_float(r.get("energy_j")))      is not None]
        costs = [v for r in g if (v := safe_float(r.get("charged_usd")))   is not None]

        model_counts: dict[str, int] = defaultdict(int)
        for r in g:
            model_counts[r.get("model_winner", "?") or "?"] += 1
        top_model = max(model_counts, key=model_counts.__getitem__) if model_counts else "-"

        # Accuracy: use gt_correct from router headers if available,
        # otherwise look up in eval_matrix by (req_id, model_winner)
        correct = scored = 0
        for r in g:
            if r.get("gt_scored") == "true":
                scored += 1
                if r.get("gt_correct") == "true":
                    correct += 1
            elif eval_index:
                ic = eval_index.get((r.get("req_id", ""), r.get("model_winner", "")))
                if ic is not None:
                    scored += 1
                    if ic:
                        correct += 1

        stats[cat] = {
            "count":        len(g),
            "accuracy":     correct / scored if scored > 0 else None,
            "correct":      correct,
            "scored":       scored,
            "lat_p50":      percentile(lats, 50)  if lats  else None,
            "lat_p95":      percentile(lats, 95)  if lats  else None,
            "lat_mean":     mean(lats)             if lats  else None,
            "energy_total": sum(enrgs)             if enrgs else None,
            "energy_mean":  mean(enrgs)            if enrgs else None,
            "cost_total":   sum(costs)             if costs else None,
            "cost_mean":    mean(costs)            if costs else None,
            "top_model":    top_model,
        }
    return stats


def fmt_pct(v) -> str:
    return f"{v*100:.1f}%" if v is not None else "-"

def fmt_ms(v) -> str:
    return f"{v/1000:.1f}s" if v is not None else "-"

def fmt_j(v) -> str:
    return f"{v:.1f}" if v is not None else "-"

def fmt_usd(v) -> str:
    return f"${v:.4f}" if v is not None else "-"


def print_table(rs: dict, bs: dict, rl: str, bl: str) -> None:
    W = 116
    print(f"\n{'='*W}")
    print(f"  CATEGORY COMPARISON:  {rl}  vs  {bl}")
    print(f"{'='*W}")

    # Accuracy
    print(f"\n  ACCURACY  (% correct answers)")
    print(f"  {'Category':<18} {'Router':>10}  {'Baseline':>10}  {'Diff':>9}  {'Count (R/B)':>12}  {'Top model (router)':>22}")
    print(f"  {'-'*(W-2)}")
    for cat in CATEGORIES:
        r, b = rs[cat], bs[cat]
        if r["count"] == 0 and b["count"] == 0:
            continue
        label = f"{cat[0]}:{cat[1]}"
        ra, ba = r["accuracy"], b["accuracy"]
        diff = f"+{(ra-ba)*100:.1f}pp" if (ra is not None and ba is not None) else "-"
        counts = f"{r['count']:4} / {b['count']:<4}"
        print(f"  {label:<18} {fmt_pct(ra):>10}  {fmt_pct(ba):>10}  {diff:>9}  {counts:>12}  {r['top_model']:>22}")

    # Latency
    print(f"\n  LATENCY  (P50 / P95 / mean)")
    print(f"  {'Category':<18} {'Router':>30}  {'Baseline':>30}  {'Count (R/B)':>12}")
    print(f"  {'-'*(W-2)}")
    for cat in CATEGORIES:
        r, b = rs[cat], bs[cat]
        if r["count"] == 0 and b["count"] == 0:
            continue
        label = f"{cat[0]}:{cat[1]}"
        rv = f"{fmt_ms(r['lat_p50'])} / {fmt_ms(r['lat_p95'])} / {fmt_ms(r['lat_mean'])}"
        bv = f"{fmt_ms(b['lat_p50'])} / {fmt_ms(b['lat_p95'])} / {fmt_ms(b['lat_mean'])}"
        counts = f"{r['count']:4} / {b['count']:<4}"
        print(f"  {label:<18} {rv:>30}  {bv:>30}  {counts:>12}")

    # Energy
    print(f"\n  ENERGY  (total J / mean J per request)")
    print(f"  {'Category':<18} {'Router':>22}  {'Baseline':>22}  {'Count (R/B)':>12}")
    print(f"  {'-'*(W-2)}")
    for cat in CATEGORIES:
        r, b = rs[cat], bs[cat]
        if r["count"] == 0 and b["count"] == 0:
            continue
        label = f"{cat[0]}:{cat[1]}"
        rv = f"{fmt_j(r['energy_total'])} / {fmt_j(r['energy_mean'])}"
        bv = f"{fmt_j(b['energy_total'])} / {fmt_j(b['energy_mean'])}"
        print(f"  {label:<18} {rv:>22}  {bv:>22}  {r['count']:4} / {b['count']:<4}")

    # Cost
    print(f"\n  COST  (total USD / mean USD per request)")
    print(f"  {'Category':<18} {'Router':>22}  {'Baseline':>22}  {'Count (R/B)':>12}")
    print(f"  {'-'*(W-2)}")
    for cat in CATEGORIES:
        r, b = rs[cat], bs[cat]
        if r["count"] == 0 and b["count"] == 0:
            continue
        label = f"{cat[0]}:{cat[1]}"
        rv = f"{fmt_usd(r['cost_total'])} / {fmt_usd(r['cost_mean'])}"
        bv = f"{fmt_usd(b['cost_total'])} / {fmt_usd(b['cost_mean'])}"
        print(f"  {label:<18} {rv:>22}  {bv:>22}  {r['count']:4} / {b['count']:<4}")

    print(f"\n{'='*W}\n")


def save_csv(rs: dict, bs: dict, output: str) -> None:
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    fields = [
        "category", "domain", "complexity",
        "router_count", "router_accuracy_pct", "router_correct", "router_scored",
        "router_lat_p50_ms", "router_lat_p95_ms", "router_lat_mean_ms",
        "router_energy_total_j", "router_energy_mean_j",
        "router_cost_total_usd", "router_cost_mean_usd", "router_top_model",
        "baseline_count", "baseline_accuracy_pct", "baseline_correct", "baseline_scored",
        "baseline_lat_p50_ms", "baseline_lat_p95_ms", "baseline_lat_mean_ms",
        "baseline_energy_total_j", "baseline_energy_mean_j",
        "baseline_cost_total_usd", "baseline_cost_mean_usd", "baseline_top_model",
        "accuracy_diff_pp", "lat_p50_improvement_pct",
    ]
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for cat in CATEGORIES:
            r, b = rs[cat], bs[cat]
            if r["count"] == 0 and b["count"] == 0:
                continue
            acc_diff = None
            if r["accuracy"] is not None and b["accuracy"] is not None:
                acc_diff = round((r["accuracy"] - b["accuracy"]) * 100, 2)
            lat_imp = None
            if r["lat_p50"] and b["lat_p50"] and b["lat_p50"] > 0:
                lat_imp = round((b["lat_p50"] - r["lat_p50"]) / b["lat_p50"] * 100, 1)
            w.writerow({
                "category":                f"{cat[0]}:{cat[1]}",
                "domain":                  cat[0],
                "complexity":              cat[1],
                "router_count":            r["count"],
                "router_accuracy_pct":     round(r["accuracy"] * 100, 2) if r["accuracy"] is not None else "",
                "router_correct":          r["correct"],
                "router_scored":           r["scored"],
                "router_lat_p50_ms":       round(r["lat_p50"], 0)  if r["lat_p50"]  else "",
                "router_lat_p95_ms":       round(r["lat_p95"], 0)  if r["lat_p95"]  else "",
                "router_lat_mean_ms":      round(r["lat_mean"], 0) if r["lat_mean"] else "",
                "router_energy_total_j":   round(r["energy_total"], 2) if r["energy_total"] else "",
                "router_energy_mean_j":    round(r["energy_mean"], 3)  if r["energy_mean"]  else "",
                "router_cost_total_usd":   f"{r['cost_total']:.6f}" if r["cost_total"] else "",
                "router_cost_mean_usd":    f"{r['cost_mean']:.8f}"  if r["cost_mean"]  else "",
                "router_top_model":        r["top_model"],
                "baseline_count":          b["count"],
                "baseline_accuracy_pct":   round(b["accuracy"] * 100, 2) if b["accuracy"] is not None else "",
                "baseline_correct":        b["correct"],
                "baseline_scored":         b["scored"],
                "baseline_lat_p50_ms":     round(b["lat_p50"], 0)  if b["lat_p50"]  else "",
                "baseline_lat_p95_ms":     round(b["lat_p95"], 0)  if b["lat_p95"]  else "",
                "baseline_lat_mean_ms":    round(b["lat_mean"], 0) if b["lat_mean"] else "",
                "baseline_energy_total_j": round(b["energy_total"], 2) if b["energy_total"] else "",
                "baseline_energy_mean_j":  round(b["energy_mean"], 3)  if b["energy_mean"]  else "",
                "baseline_cost_total_usd": f"{b['cost_total']:.6f}" if b["cost_total"] else "",
                "baseline_cost_mean_usd":  f"{b['cost_mean']:.8f}"  if b["cost_mean"]  else "",
                "baseline_top_model":      b["top_model"],
                "accuracy_diff_pp":        acc_diff if acc_diff is not None else "",
                "lat_p50_improvement_pct": lat_imp  if lat_imp  is not None else "",
            })
    print(f"  Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router",      required=True, help="Router results CSV")
    parser.add_argument("--baseline",    required=True, help="Baseline results CSV")
    parser.add_argument("--eval-matrix", default="",   help="eval_matrix.csv for baseline accuracy lookup")
    parser.add_argument("--output",      default="",   help="Output CSV path")
    args = parser.parse_args()

    eval_index = build_eval_index(args.eval_matrix)

    rl = os.path.splitext(os.path.basename(args.router))[0]
    bl = os.path.splitext(os.path.basename(args.baseline))[0]

    router_stats   = summarize(load_csv(args.router),   eval_index)
    baseline_stats = summarize(load_csv(args.baseline), eval_index)

    print_table(router_stats, baseline_stats, rl, bl)

    if args.output:
        save_csv(router_stats, baseline_stats, args.output)


if __name__ == "__main__":
    main()
