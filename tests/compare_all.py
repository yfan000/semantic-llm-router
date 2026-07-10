"""
compare_all.py — Side-by-side comparison of all routing approaches.

Produces a single unified table ranking every system by accuracy, latency,
and cost — both overall and broken down by (domain, complexity).

Usage:
    python tests/compare_all.py \\
        --system "TTCA+retry:results/router_ttca.csv" \\
        --system "Cascade:results/baseline_cascade.csv" \\
        --system "vLLM-SR:results/baseline_vllm_sr.csv" \\
        --system "Round-Robin:results/rr_baseline.csv" \\
        --eval-matrix results/eval_matrix.csv \\
        --output results/compare_all.csv

Each --system argument is NAME:PATH.  Order determines the column order in the
domain breakdown table.  Systems are ranked by overall accuracy in the summary.
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
]


def _safe_float(v) -> float | None:
    try:
        return float(v) if v and str(v).strip() else None
    except (ValueError, TypeError):
        return None


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def load_eval_index(path: str) -> dict[tuple[str, str], bool]:
    """Build (req_id, model_id) -> is_correct from eval_matrix.csv."""
    if not path or not os.path.exists(path):
        return {}
    index: dict[tuple[str, str], bool] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ic = row.get("is_correct", "")
            if ic in ("true", "false"):
                index[(row.get("req_id", ""), row.get("model_id", ""))] = (ic == "true")
    return index


def load_system(path: str, eval_index: dict) -> dict:
    """Load one system's result CSV and return aggregate stats."""
    rows_ok: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("status")) == "200":
                rows_ok.append(row)

    # ── Per-category stats ────────────────────────────────────────────────────
    by_cat: dict[tuple, dict] = {}
    for cat in CATEGORIES:
        g = [r for r in rows_ok
             if r.get("domain") == cat[0] and r.get("complexity") == cat[1]]

        lats   = [v for r in g
                  if (v := _safe_float(r.get("actual_latency_ms") or
                                       r.get("wall_ms"))) is not None]
        costs  = [v for r in g
                  if (v := _safe_float(r.get("charged_usd"))) is not None]
        energy = [v for r in g
                  if (v := _safe_float(r.get("energy_j"))) is not None]

        correct = scored = 0
        ttca_lats_cat: list[float] = []
        for r in g:
            lat = _safe_float(r.get("actual_latency_ms") or r.get("wall_ms"))
            is_correct = False
            if r.get("gt_scored") == "true":
                scored += 1
                if r.get("gt_correct") == "true":
                    correct += 1
                    is_correct = True
            elif eval_index:
                ic = eval_index.get((r.get("req_id", ""),
                                     r.get("model_winner", "")))
                if ic is not None:
                    scored += 1
                    if ic:
                        correct += 1
                        is_correct = True
            if is_correct and lat is not None:
                ttca_lats_cat.append(lat)

        slo_rows_g = [r for r in g if r.get("slo_violated") in ("true", "false")]
        slo_viol_g = sum(1 for r in slo_rows_g if r.get("slo_violated") == "true")

        attempts_g = [int(r["retries"]) + 1
                      for r in g if r.get("retries", "") not in ("", None)]

        by_cat[cat] = {
            "n":              len(g),
            "accuracy":       correct / scored if scored > 0 else None,
            "correct":        correct,
            "scored":         scored,
            "lat_p50":        _percentile(lats, 50) if lats else None,
            "lat_p95":        _percentile(lats, 95) if lats else None,
            "lat_mean":       mean(lats)             if lats else None,
            "ttca_mean":      mean(ttca_lats_cat)              if ttca_lats_cat else None,
            "ttca_p50":       _percentile(ttca_lats_cat, 50)   if ttca_lats_cat else None,
            "ttca_p90":       _percentile(ttca_lats_cat, 90)   if ttca_lats_cat else None,
            "ttca_p95":       _percentile(ttca_lats_cat, 95)   if ttca_lats_cat else None,
            "cost_mean":      mean(costs)            if costs else None,
            "cost_total":     sum(costs)             if costs else None,
            "energy_mean":    mean(energy)           if energy else None,
            "energy_total":   sum(energy)            if energy else None,
            "slo_viol_n":     slo_viol_g,
            "slo_total":      len(slo_rows_g),
            "slo_viol_rate":  slo_viol_g / len(slo_rows_g) if slo_rows_g else None,
            "attempts_mean":  mean(attempts_g) if attempts_g else None,
        }

    # ── Overall aggregates ────────────────────────────────────────────────────
    all_lats     = [v for r in rows_ok
                    if (v := _safe_float(r.get("actual_latency_ms") or
                                         r.get("wall_ms"))) is not None]
    all_costs    = [v for r in rows_ok
                    if (v := _safe_float(r.get("charged_usd"))) is not None]
    all_energy   = [v for r in rows_ok
                    if (v := _safe_float(r.get("energy_j"))) is not None]
    all_attempts = [int(r["retries"]) + 1
                    for r in rows_ok if r.get("retries", "") not in ("", None)]

    # TTCA lats: latency of requests that received a correct answer
    all_ttca_lats: list[float] = []
    for r in rows_ok:
        lat = _safe_float(r.get("actual_latency_ms") or r.get("wall_ms"))
        if lat is None:
            continue
        is_correct = False
        if r.get("gt_scored") == "true":
            is_correct = (r.get("gt_correct") == "true")
        elif eval_index:
            ic = eval_index.get((r.get("req_id", ""), r.get("model_winner", "")))
            is_correct = bool(ic)
        if is_correct:
            all_ttca_lats.append(lat)

    tot_correct  = sum(s["correct"]   for s in by_cat.values())
    tot_scored   = sum(s["scored"]    for s in by_cat.values())
    tot_slo_viol = sum(s["slo_viol_n"] for s in by_cat.values())
    tot_slo_rows = sum(s["slo_total"]  for s in by_cat.values())

    return {
        "n":              len(rows_ok),
        "accuracy":       tot_correct / tot_scored if tot_scored > 0 else None,
        "lat_p50":        _percentile(all_lats, 50) if all_lats else None,
        "lat_p95":        _percentile(all_lats, 95) if all_lats else None,
        "lat_mean":       mean(all_lats)             if all_lats else None,
        "ttca_mean":      mean(all_ttca_lats)              if all_ttca_lats else None,
        "ttca_p50":       _percentile(all_ttca_lats, 50)   if all_ttca_lats else None,
        "ttca_p90":       _percentile(all_ttca_lats, 90)   if all_ttca_lats else None,
        "ttca_p95":       _percentile(all_ttca_lats, 95)   if all_ttca_lats else None,
        "cost_mean":      mean(all_costs)            if all_costs else None,
        "cost_total":     sum(all_costs)             if all_costs else None,
        "energy_mean":    mean(all_energy)           if all_energy else None,
        "energy_total":   sum(all_energy)            if all_energy else None,
        "slo_viol_rate":  tot_slo_viol / tot_slo_rows if tot_slo_rows > 0 else None,
        "slo_viol_n":     tot_slo_viol,
        "slo_total":      tot_slo_rows,
        "attempts_mean":  mean(all_attempts) if all_attempts else None,
        "by_cat":         by_cat,
    }


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(v) -> str:
    return f"{v*100:.1f}%" if v is not None else "   -  "

def _ms(v) -> str:
    if v is None:
        return "    -   "
    return f"{v/1000:.2f}s" if v >= 1000 else f"{v:.0f}ms"

def _usd(v) -> str:
    return f"${v:.6f}" if v is not None else "     -    "

def _j(v) -> str:
    return f"{v:.1f}J" if v is not None else "    -   "

def _pp(a, b) -> str:
    if a is None or b is None:
        return "   -  "
    d = (a - b) * 100
    return f"{d:+.1f}pp"


def print_summary(systems: list[tuple[str, dict]], ref_name: str | None = None) -> None:
    """Print ranked summary table including accuracy, latency, energy and cost."""
    ranked = sorted(systems, key=lambda x: x[1]["accuracy"] or 0, reverse=True)
    ref = next((s for n, s in systems if n == ref_name), None) if ref_name else None

    # Check whether any system has SLO or attempts data
    any_slo      = any(s.get("slo_total", 0) > 0 for _, s in systems)
    any_attempts = any(s.get("attempts_mean") is not None for _, s in systems)

    W = 185 if any_slo else 173
    if any_attempts:
        W += 12
    print(f"\n{'='*W}")
    print(f"  ALL SYSTEMS — OVERALL COMPARISON  (ranked by accuracy)")
    print(f"{'='*W}")
    hdr_vs   = f"vs {ref_name}" if ref_name else ""
    slo_hdr  = f"  {'SLO Viol%':>10}" if any_slo else ""
    att_hdr  = f"  {'Avg Att.':>8}"   if any_attempts else ""
    print(f"\n  {'System':<22} {'Requests':>8} {'Accuracy':>9} {hdr_vs:>9}"
          f"  {'Lat Mean':>8}  {'Lat P50':>8}  {'Lat P95':>8}"
          f"  {'TTCA Mean':>10}  {'TTCA P50':>9}  {'TTCA P90':>9}  {'TTCA P95':>9}"
          f"  {'Energy/req':>11}  {'Cost/req':>11}"
          + slo_hdr + att_hdr)
    print(f"  {'-'*(W-2)}")

    for name, stats in ranked:
        vs = _pp(stats["accuracy"], ref["accuracy"]) if (ref and name != ref_name) else (
             "  [ref] " if (ref and name == ref_name) else "")
        # Aggregate energy_mean across all categories
        energy_vals = [s["energy_mean"] for s in stats["by_cat"].values()
                       if s.get("energy_mean") is not None]
        energy_mean = sum(energy_vals) / len(energy_vals) if energy_vals else None
        slo_rate = stats.get("slo_viol_rate")
        slo_col  = f"  {slo_rate*100:>9.1f}%" if (any_slo and slo_rate is not None) else (
                   f"  {'  -':>10}"            if any_slo else "")
        att_mean = stats.get("attempts_mean")
        att_col  = (f"  {att_mean:>8.2f}" if att_mean is not None else f"  {'1.00':>8}") if any_attempts else ""
        print(f"  {name:<22} {stats['n']:>8} {_pct(stats['accuracy']):>9} {vs:>9}"
              f"  {_ms(stats['lat_mean']):>8}  {_ms(stats['lat_p50']):>8}  {_ms(stats['lat_p95']):>8}"
              f"  {_ms(stats['ttca_mean']):>10}  {_ms(stats['ttca_p50']):>9}  {_ms(stats['ttca_p90']):>9}  {_ms(stats['ttca_p95']):>9}"
              f"  {_j(energy_mean):>11}  {_usd(stats['cost_mean']):>11}"
              + slo_col + att_col)

    if ref_name:
        print(f"\n  Reference (Δ column): {ref_name}")
    print()
    print()


def print_domain_breakdown(systems: list[tuple[str, dict]]) -> None:
    """Print per-(domain,complexity) accuracy breakdown."""
    names = [n for n, _ in systems]
    col_w = max(10, max(len(n) for n in names) + 2)

    W = 18 + col_w * len(names) + 4
    print(f"\n{'='*W}")
    print(f"  ACCURACY BY DOMAIN x COMPLEXITY")
    print(f"{'='*W}")
    print(f"  {'Category':<18}", end="")
    for name in names:
        print(f"  {name[:col_w-2]:>{col_w-2}}", end="")
    print()
    print(f"  {'-'*(W-2)}")

    prev_domain = None
    for cat in CATEGORIES:
        domain, complexity = cat
        # Print separator between domains
        if prev_domain and domain != prev_domain:
            print()
        prev_domain = domain

        label = f"{domain}:{complexity}"
        accs     = [(name, stats["by_cat"][cat]["accuracy"])     for name, stats in systems]
        slo_viol = [(name, stats["by_cat"][cat].get("slo_viol_rate")) for name, stats in systems]
        any_cat_slo = any(r is not None for _, r in slo_viol)

        # Bold (mark with *) the best accuracy per row
        best = max((a for _, a in accs if a is not None), default=None)

        print(f"  {label:<18}", end="")
        for name, acc in accs:
            cell = _pct(acc)
            marker = "*" if acc is not None and best is not None and abs(acc - best) < 0.001 else " "
            print(f"  {marker}{cell:>{col_w-2}}", end="")
        if any_cat_slo:
            print(f"  {'SLO%':>{col_w}}", end="")
            for _, rate in slo_viol:
                cell = f"{rate*100:.0f}%" if rate is not None else "-"
                print(f"  {cell:>{col_w}}", end="")
        print()

    print()


def save_csv(systems: list[tuple[str, dict]], output: str) -> None:
    """Save unified comparison to CSV for downstream analysis."""
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    fields = ["category", "domain", "complexity"]
    for name, _ in systems:
        safe = name.replace(" ", "_").replace("+", "plus").replace("/", "_")
        fields += [
            f"{safe}_n",
            f"{safe}_accuracy_pct",
            f"{safe}_lat_p50_ms",
            f"{safe}_lat_p95_ms",
            f"{safe}_ttca_mean_ms",
            f"{safe}_ttca_p50_ms",
            f"{safe}_ttca_p90_ms",
            f"{safe}_ttca_p95_ms",
            f"{safe}_cost_mean_usd",
            f"{safe}_slo_viol_pct",
            f"{safe}_attempts_mean",
        ]

    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()

        # Overall row
        row: dict = {"category": "OVERALL", "domain": "", "complexity": ""}
        for name, stats in systems:
            safe = name.replace(" ", "_").replace("+", "plus").replace("/", "_")
            row[f"{safe}_n"]              = stats["n"]
            row[f"{safe}_accuracy_pct"]   = round(stats["accuracy"] * 100, 2) if stats["accuracy"] is not None else ""
            row[f"{safe}_lat_p50_ms"]     = round(stats["lat_p50"],  0) if stats["lat_p50"]  else ""
            row[f"{safe}_lat_p95_ms"]     = round(stats["lat_p95"],  0) if stats["lat_p95"]  else ""
            row[f"{safe}_ttca_mean_ms"]   = round(stats["ttca_mean"], 0) if stats.get("ttca_mean") else ""
            row[f"{safe}_ttca_p50_ms"]    = round(stats["ttca_p50"],  0) if stats.get("ttca_p50")  else ""
            row[f"{safe}_ttca_p90_ms"]    = round(stats["ttca_p90"],  0) if stats.get("ttca_p90")  else ""
            row[f"{safe}_ttca_p95_ms"]    = round(stats["ttca_p95"],  0) if stats.get("ttca_p95")  else ""
            row[f"{safe}_cost_mean_usd"]  = f"{stats['cost_mean']:.8f}" if stats["cost_mean"] else ""
            slo_r = stats.get("slo_viol_rate")
            row[f"{safe}_slo_viol_pct"]   = round(slo_r * 100, 1) if slo_r is not None else ""
            att = stats.get("attempts_mean")
            row[f"{safe}_attempts_mean"]  = round(att, 3) if att is not None else 1.0
        w.writerow(row)

        # Per-category rows
        for cat in CATEGORIES:
            domain, complexity = cat
            row = {"category": f"{domain}:{complexity}", "domain": domain, "complexity": complexity}
            for name, stats in systems:
                safe = name.replace(" ", "_").replace("+", "plus").replace("/", "_")
                s = stats["by_cat"][cat]
                row[f"{safe}_n"]             = s["n"]
                row[f"{safe}_accuracy_pct"]  = round(s["accuracy"] * 100, 2) if s["accuracy"] is not None else ""
                row[f"{safe}_lat_p50_ms"]    = round(s["lat_p50"],  0) if s["lat_p50"]  else ""
                row[f"{safe}_lat_p95_ms"]    = round(s["lat_p95"],  0) if s["lat_p95"]  else ""
                row[f"{safe}_ttca_mean_ms"]  = round(s["ttca_mean"], 0) if s.get("ttca_mean") else ""
                row[f"{safe}_ttca_p50_ms"]   = round(s["ttca_p50"],  0) if s.get("ttca_p50")  else ""
                row[f"{safe}_ttca_p90_ms"]   = round(s["ttca_p90"],  0) if s.get("ttca_p90")  else ""
                row[f"{safe}_ttca_p95_ms"]   = round(s["ttca_p95"],  0) if s.get("ttca_p95")  else ""
                row[f"{safe}_cost_mean_usd"] = f"{s['cost_mean']:.8f}" if s["cost_mean"] else ""
                slo_cat = s.get("slo_viol_rate")
                row[f"{safe}_slo_viol_pct"]  = round(slo_cat * 100, 1) if slo_cat is not None else ""
                att_cat = s.get("attempts_mean")
                row[f"{safe}_attempts_mean"] = round(att_cat, 3) if att_cat is not None else 1.0
            w.writerow(row)

    print(f"  Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system",      action="append", default=[],
                        metavar="NAME:PATH",
                        help="System to compare. Repeat for each system. "
                             "Format: 'DisplayName:path/to/results.csv'")
    parser.add_argument("--eval-matrix", default="",
                        help="eval_matrix.csv for accuracy lookup (optional but recommended)")
    parser.add_argument("--output",      default="",
                        help="Path to save unified CSV (optional)")
    parser.add_argument("--ref",         default=None,
                        help="System name to use as reference for Δ accuracy column. "
                             "Defaults to the first --system.")
    args = parser.parse_args()

    if not args.system:
        parser.error("Provide at least two --system NAME:PATH arguments")

    # Parse NAME:PATH pairs
    parsed: list[tuple[str, str]] = []
    for spec in args.system:
        if ":" not in spec:
            parser.error(f"--system must be NAME:PATH, got: {spec!r}")
        name, path = spec.split(":", 1)
        if not os.path.exists(path):
            print(f"  WARNING: file not found, skipping: {path}")
            continue
        parsed.append((name.strip(), path.strip()))

    if len(parsed) < 2:
        parser.error("Need at least 2 valid --system files to compare")

    eval_index = load_eval_index(args.eval_matrix)
    if eval_index:
        print(f"  Loaded eval index: {len(eval_index)} (req_id, model_id) entries")

    print(f"  Loading {len(parsed)} systems...")
    systems: list[tuple[str, dict]] = []
    for name, path in parsed:
        stats = load_system(path, eval_index)
        systems.append((name, stats))
        print(f"    {name:<22} {stats['n']:5} requests  acc={_pct(stats['accuracy'])}")

    ref_name = args.ref or parsed[0][0]

    print_summary(systems, ref_name=ref_name)
    print_domain_breakdown(systems)

    if args.output:
        save_csv(systems, args.output)


if __name__ == "__main__":
    main()
