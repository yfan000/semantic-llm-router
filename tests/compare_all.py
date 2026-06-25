"""
compare_all.py — Side-by-side comparison of all routing approaches.

Produces a single unified table ranking every system by accuracy, latency,
energy and cost — both overall and broken down by (domain, complexity).

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

    # Per-category stats
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
        for r in g:
            if r.get("gt_scored") == "true":
                scored += 1
                if r.get("gt_correct") == "true":
                    correct += 1
            elif eval_index:
                ic = eval_index.get((r.get("req_id", ""),
                                     r.get("model_winner", "")))
                if ic is not None:
                    scored += 1
                    if ic:
                        correct += 1

        by_cat[cat] = {
            "n":            len(g),
            "accuracy":     correct / scored if scored > 0 else None,
            "correct":      correct,
            "scored":       scored,
            "lat_p50":      _percentile(lats, 50) if lats else None,
            "lat_p95":      _percentile(lats, 95) if lats else None,
            "lat_mean":     mean(lats)             if lats else None,
            "cost_mean":    mean(costs)            if costs else None,
            "cost_total":   sum(costs)             if costs else None,
            "energy_mean":  mean(energy)           if energy else None,
            "energy_total": sum(energy)            if energy else None,
        }

    # ── Overall aggregates ────────────────────────────────────────────────────
    all_lats   = [v for r in rows_ok
                  if (v := _safe_float(r.get("actual_latency_ms") or
                                       r.get("wall_ms"))) is not None]
    all_costs  = [v for r in rows_ok
                  if (v := _safe_float(r.get("charged_usd"))) is not None]
    all_energy = [v for r in rows_ok
                  if (v := _safe_float(r.get("energy_j"))) is not None]

    tot_correct = sum(s["correct"] for s in by_cat.values())
    tot_scored  = sum(s["scored"]  for s in by_cat.values())

    return {
        "n":            len(rows_ok),
        "accuracy":     tot_correct / tot_scored if tot_scored > 0 else None,
        "lat_p50":      _percentile(all_lats, 50) if all_lats else None,
        "lat_p95":      _percentile(all_lats, 95) if all_lats else None,
        "lat_mean":     mean(all_lats)             if all_lats else None,
        "cost_mean":    mean(all_costs)            if all_costs else None,
        "cost_total":   sum(all_costs)             if all_costs else None,
        "energy_mean":  mean(all_energy)           if all_energy else None,
        "energy_total": sum(all_energy)            if all_energy else None,
        "by_cat":       by_cat,
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

    W = 118
    print(f"\n{'='*W}")
    print(f"  ALL SYSTEMS — OVERALL COMPARISON  (ranked by accuracy)")
    print(f"{'='*W}")
    hdr_vs = f"vs {ref_name}" if ref_name else ""
    print(f"\n  {'System':<22} {'Requests':>8} {'Accuracy':>9} {hdr_vs:>9}"
          f"  {'Lat P50':>8}  {'Lat P95':>8}  {'Energy/req':>11}  {'Cost/req':>11}")
    print(f"  {'-'*(W-2)}")

    for name, stats in ranked:
        vs = _pp(stats["accuracy"], ref["accuracy"]) if (ref and name != ref_name) else (
             "  [ref] " if (ref and name == ref_name) else "")
        energy_mean = stats.get("energy_mean")
        print(f"  {name:<22} {stats['n']:>8} {_pct(stats['accuracy']):>9} {vs:>9}"
              f"  {_ms(stats['lat_p50']):>8}  {_ms(stats['lat_p95']):>8}"
              f"  {_j(energy_mean):>11}  {_usd(stats['cost_mean']):>11}")

    if ref_name:
        print(f"\n  Reference (Δ column): {ref_name}")
    print()


def print_domain_breakdown(systems: list[tuple[str, dict]]) -> None:
    """Print per-(domain, complexity) accuracy for all systems side by side."""
    names = [n for n, _ in systems]
    col_w = max(10, max(len(n) for n in names) + 1)

    W = 20 + col_w * len(names) + 4
    print(f"\n{'='*W}")
    print(f"  ACCURACY BY DOMAIN x COMPLEXITY  (* = best in row)")
    print(f"{'='*W}")
    print(f"  {'Category':<20}", end="")
    for name in names:
        print(f"  {name[:col_w]:>{col_w}}", end="")
    print()
    print(f"  {'-'*(W-2)}")

    prev_domain = None
    for cat in CATEGORIES:
        domain, complexity = cat
        if prev_domain and domain != prev_domain:
            print()
        prev_domain = domain

        label = f"{domain}:{complexity}"
        accs = [(name, stats["by_cat"][cat]["accuracy"]) for name, stats in systems]
        best = max((a for _, a in accs if a is not None), default=None)

        print(f"  {label:<20}", end="")
        for name, acc in accs:
            cell = _pct(acc)
            marker = "*" if (acc is not None and best is not None
                             and abs(acc - best) < 0.001) else " "
            print(f"  {marker}{cell:>{col_w-1}}", end="")
        print()
    print()


def save_csv(systems: list[tuple[str, dict]], output: str) -> None:
    """Save unified comparison to a CSV for downstream analysis / plotting."""
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    def _safe(name: str) -> str:
        return name.replace(" ", "_").replace("+", "plus").replace("/", "_")

    fields = ["category", "domain", "complexity"]
    for name, _ in systems:
        s = _safe(name)
        fields += [f"{s}_n", f"{s}_accuracy_pct",
                   f"{s}_lat_p50_ms", f"{s}_lat_p95_ms",
                   f"{s}_energy_mean_j", f"{s}_cost_mean_usd"]

    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()

        def _row(cat_label: str, domain: str, complexity: str, getter) -> dict:
            row: dict = {"category": cat_label, "domain": domain,
                         "complexity": complexity}
            for name, stats in systems:
                s = _safe(name)
                st = getter(stats)
                row[f"{s}_n"]              = st.get("n", "")
                acc = st.get("accuracy")
                row[f"{s}_accuracy_pct"]   = round(acc * 100, 2) if acc is not None else ""
                p50 = st.get("lat_p50")
                row[f"{s}_lat_p50_ms"]     = round(p50, 0) if p50 else ""
                p95 = st.get("lat_p95")
                row[f"{s}_lat_p95_ms"]     = round(p95, 0) if p95 else ""
                em  = st.get("energy_mean")
                row[f"{s}_energy_mean_j"]  = f"{em:.3f}" if em else ""
                cm  = st.get("cost_mean")
                row[f"{s}_cost_mean_usd"]  = f"{cm:.8f}" if cm else ""
            return row

        w.writerow(_row("OVERALL", "", "", lambda st: st))
        for cat in CATEGORIES:
            domain, complexity = cat
            w.writerow(_row(f"{domain}:{complexity}", domain, complexity,
                            lambda st, c=cat: st["by_cat"][c]))

    print(f"  Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system",      action="append", default=[],
                        metavar="NAME:PATH",
                        help="System to compare — repeat for each system. "
                             "Format: 'DisplayName:path/to/results.csv'")
    parser.add_argument("--eval-matrix", default="",
                        help="eval_matrix.csv for accuracy lookup (optional but recommended)")
    parser.add_argument("--output",      default="",
                        help="Path to save unified CSV (optional)")
    parser.add_argument("--ref",         default=None,
                        help="System name to use as reference for the Δ accuracy column. "
                             "Defaults to the first --system.")
    args = parser.parse_args()

    if not args.system:
        parser.error("Provide at least two --system NAME:PATH arguments")

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
        print(f"    {name:<24} {stats['n']:5} requests  acc={_pct(stats['accuracy'])}"
              f"  energy={_j(stats.get('energy_mean'))}  cost={_usd(stats.get('cost_mean'))}")

    ref_name = args.ref or parsed[0][0]

    print_summary(systems, ref_name=ref_name)
    print_domain_breakdown(systems)

    if args.output:
        save_csv(systems, args.output)


if __name__ == "__main__":
    main()
