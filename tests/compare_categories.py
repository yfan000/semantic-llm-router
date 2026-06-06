"""
compare_categories.py — Per-category comparison of router vs baseline.

Produces a detailed breakdown by domain×complexity showing:
  - Request count
  - Latency (P50, P95, mean)
  - Energy cost (J)
  - Charged cost (USD)
  - Model routing distribution

Usage:
    python tests/compare_categories.py \\
        --router   results/router_ttca.csv \\
        --baseline results/rr_baseline.csv \\
        --output   results/compare_categories.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from statistics import mean, median


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


def safe_float(v: str) -> float | None:
    try:
        return float(v) if v and v.strip() else None
    except (ValueError, TypeError):
        return None


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = int(len(vals) * p / 100)
    return vals[min(idx, len(vals) - 1)]


def summarize(rows: list[dict], label: str) -> dict:
    ok = [r for r in rows if str(r.get("status")) == "200"]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in ok:
        key = (r.get("domain", "?"), r.get("complexity", "?"))
        groups[key].append(r)

    stats = {}
    for cat in CATEGORIES:
        g = groups.get(cat, [])
        lats  = [v for r in g if (v := safe_float(r.get("actual_latency_ms") or r.get("wall_ms"))) is not None]
        enrgs = [v for r in g if (v := safe_float(r.get("energy_j"))) is not None]
        costs = [v for r in g if (v := safe_float(r.get("charged_usd"))) is not None]
        models: dict[str, int] = defaultdict(int)
        for r in g:
            m = r.get("model_winner", "?") or "?"
            models[m] += 1
        top_model = max(models, key=models.__getitem__) if models else "-"

        stats[cat] = {
            "count":        len(g),
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


def fmt_ms(v: float | None) -> str:
    return f"{v/1000:.1f}s" if v is not None else "-"

def fmt_j(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "-"

def fmt_usd(v: float | None) -> str:
    return f"${v:.4f}" if v is not None else "-"


def print_table(router_stats: dict, baseline_stats: dict,
                router_label: str, baseline_label: str) -> None:
    W = 110
    print(f"\n{'='*W}")
    print(f"  CATEGORY COMPARISON: {router_label}  vs  {baseline_label}")
    print(f"{'='*W}")

    sections = [
        ("LATENCY  (P50 / P95 / mean)",     "lat_p50", "lat_p95", "lat_mean",      fmt_ms),
        ("ENERGY   (total J / mean J)",      "energy_total", "energy_mean", None,  fmt_j),
        ("COST     (total USD / mean USD)",   "cost_total", "cost_mean", None,      fmt_usd),
    ]

    for section_name, k1, k2, k3, fmt in sections:
        print(f"\n  {section_name}")
        hdr = f"  {'Category':<18} {'Router':>30}  {'Baseline':>30}  {'Count (R/B)':>12}  {'Top model (router)':>20}"
        print(hdr)
        print(f"  {'-'*(W-2)}")
        for cat in CATEGORIES:
            r = router_stats[cat]
            b = baseline_stats[cat]
            if r["count"] == 0 and b["count"] == 0:
                continue
            label = f"{cat[0]}:{cat[1]}"
            if k3:
                rv = f"{fmt(r[k1])} / {fmt(r[k2])} / {fmt(r[k3])}"
                bv = f"{fmt(b[k1])} / {fmt(b[k2])} / {fmt(b[k3])}"
            else:
                rv = f"{fmt(r[k1])} / {fmt(r[k2])}"
                bv = f"{fmt(b[k1])} / {fmt(b[k2])}"
            counts = f"{r['count']:4} / {b['count']:<4}"
            print(f"  {label:<18} {rv:>30}  {bv:>30}  {counts:>12}  {r['top_model']:>20}")

    print(f"\n{'='*W}\n")


def save_csv(router_stats: dict, baseline_stats: dict,
             router_label: str, baseline_label: str, output: str) -> None:
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    fields = [
        "category", "domain", "complexity",
        "router_count", "router_lat_p50_ms", "router_lat_p95_ms", "router_lat_mean_ms",
        "router_energy_total_j", "router_energy_mean_j",
        "router_cost_total_usd", "router_cost_mean_usd", "router_top_model",
        "baseline_count", "baseline_lat_p50_ms", "baseline_lat_p95_ms", "baseline_lat_mean_ms",
        "baseline_energy_total_j", "baseline_energy_mean_j",
        "baseline_cost_total_usd", "baseline_cost_mean_usd", "baseline_top_model",
        "lat_p50_improvement_pct",
    ]
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for cat in CATEGORIES:
            r = router_stats[cat]
            b = baseline_stats[cat]
            if r["count"] == 0 and b["count"] == 0:
                continue
            imp = None
            if r["lat_p50"] and b["lat_p50"] and b["lat_p50"] > 0:
                imp = round((b["lat_p50"] - r["lat_p50"]) / b["lat_p50"] * 100, 1)
            w.writerow({
                "category":                f"{cat[0]}:{cat[1]}",
                "domain":                  cat[0],
                "complexity":              cat[1],
                "router_count":            r["count"],
                "router_lat_p50_ms":       round(r["lat_p50"], 0)  if r["lat_p50"]  else "",
                "router_lat_p95_ms":       round(r["lat_p95"], 0)  if r["lat_p95"]  else "",
                "router_lat_mean_ms":      round(r["lat_mean"], 0) if r["lat_mean"] else "",
                "router_energy_total_j":   round(r["energy_total"], 2) if r["energy_total"] else "",
                "router_energy_mean_j":    round(r["energy_mean"], 3)  if r["energy_mean"]  else "",
                "router_cost_total_usd":   f"{r['cost_total']:.6f}" if r["cost_total"] else "",
                "router_cost_mean_usd":    f"{r['cost_mean']:.8f}"  if r["cost_mean"]  else "",
                "router_top_model":        r["top_model"],
                "baseline_count":          b["count"],
                "baseline_lat_p50_ms":     round(b["lat_p50"], 0)  if b["lat_p50"]  else "",
                "baseline_lat_p95_ms":     round(b["lat_p95"], 0)  if b["lat_p95"]  else "",
                "baseline_lat_mean_ms":    round(b["lat_mean"], 0) if b["lat_mean"] else "",
                "baseline_energy_total_j": round(b["energy_total"], 2) if b["energy_total"] else "",
                "baseline_energy_mean_j":  round(b["energy_mean"], 3)  if b["energy_mean"]  else "",
                "baseline_cost_total_usd": f"{b['cost_total']:.6f}" if b["cost_total"] else "",
                "baseline_cost_mean_usd":  f"{b['cost_mean']:.8f}"  if b["cost_mean"]  else "",
                "baseline_top_model":      b["top_model"],
                "lat_p50_improvement_pct": imp if imp is not None else "",
            })
    print(f"  Saved CSV: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router",   required=True, help="Router results CSV")
    parser.add_argument("--baseline", required=True, help="Baseline results CSV")
    parser.add_argument("--output",   default="",    help="Output CSV path")
    args = parser.parse_args()

    router_label   = os.path.splitext(os.path.basename(args.router))[0]
    baseline_label = os.path.splitext(os.path.basename(args.baseline))[0]

    router_rows   = load_csv(args.router)
    baseline_rows = load_csv(args.baseline)

    router_stats   = summarize(router_rows,   router_label)
    baseline_stats = summarize(baseline_rows, baseline_label)

    print_table(router_stats, baseline_stats, router_label, baseline_label)

    if args.output:
        save_csv(router_stats, baseline_stats, router_label, baseline_label, args.output)


if __name__ == "__main__":
    main()
