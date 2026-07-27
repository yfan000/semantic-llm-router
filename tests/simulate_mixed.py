"""
simulate_mixed.py — Simulate the mixed router result from existing CSVs.

Constructs a virtual mixed-router CSV without re-running any experiments:
  - code:hard queries  → rows from TTCA CSV
  - all other queries  → rows from CARROT CSV (75%) or OmniRouter CSV (25%)

The resulting CSV can be passed to compare_all.py as --system "Mixed:...".

Usage:
    python tests/simulate_mixed.py \\
        --ttca      results/RUNDIR/beta_0_0_results.csv \\
        --carrot    results/RUNDIR/baseline_carrot.csv \\
        --omni      results/RUNDIR/baseline_omni_router.csv \\
        --carrot-frac 0.75 \\
        --seed      42 \\
        --output    results/RUNDIR/simulated_mixed.csv

    # Then compare:
    python tests/compare_all.py \\
        --system "CARROT:results/RUNDIR/baseline_carrot.csv" \\
        --system "OmniRouter:results/RUNDIR/baseline_omni_router.csv" \\
        --system "TTCA:results/RUNDIR/beta_0_0_results.csv" \\
        --system "Mixed:results/RUNDIR/simulated_mixed.csv" \\
        --eval-matrix results/RUNDIR/eval_matrix.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from statistics import mean

TTCA_CELL = ("code", "hard")


def load_csv(path: str) -> dict[str, dict]:
    """Load CSV into dict keyed by req_id."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["req_id"]: r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate mixed router from existing result CSVs")
    parser.add_argument("--ttca",        required=True, help="TTCA results CSV")
    parser.add_argument("--carrot",      required=True, help="CARROT results CSV")
    parser.add_argument("--omni",        required=True, help="OmniRouter results CSV")
    parser.add_argument("--carrot-frac", type=float, default=0.75,
                        help="Fraction of non-code:hard queries from CARROT (default 0.75)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output",      required=True, help="Output mixed CSV path")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"  Loading CSVs...")
    ttca_rows   = load_csv(args.ttca)
    carrot_rows = load_csv(args.carrot)
    omni_rows   = load_csv(args.omni)

    # Determine full req_id set (union of all three, in order)
    all_ids = sorted(
        set(ttca_rows) | set(carrot_rows) | set(omni_rows),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    print(f"  Total unique req_ids: {len(all_ids)}")

    fieldnames = list(next(iter(ttca_rows.values())).keys())

    mixed_rows = []
    source_log = Counter()
    missing    = []

    for req_id in all_ids:
        sample = (ttca_rows.get(req_id)
                  or carrot_rows.get(req_id)
                  or omni_rows.get(req_id))
        domain     = sample.get("domain", "")
        complexity = sample.get("complexity", "")

        if (domain, complexity) == TTCA_CELL:
            row    = ttca_rows.get(req_id)
            source = "ttca"
        else:
            if rng.random() < args.carrot_frac:
                row    = carrot_rows.get(req_id)
                source = "carrot"
            else:
                row    = omni_rows.get(req_id)
                source = "omni"

        if row is None:
            missing.append(req_id)
            continue

        row = dict(row)
        row["mode"] = f"mixed/{source}"
        mixed_rows.append(row)
        source_log[source] += 1

    print(f"  Routing breakdown:")
    print(f"    ttca   : {source_log['ttca']:4d}  (code:hard)")
    n_other = len(all_ids) - source_log["ttca"]
    print(f"    carrot : {source_log['carrot']:4d}  ({100*source_log['carrot']//max(n_other,1)}% of non-code:hard)")
    print(f"    omni   : {source_log['omni']:4d}  ({100*source_log['omni']//max(n_other,1)}% of non-code:hard)")
    if missing:
        print(f"  WARNING: {len(missing)} req_ids missing from chosen CSV — skipped")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(mixed_rows)

    ok    = [r for r in mixed_rows if str(r.get("status")) == "200"]
    costs = [float(r["charged_usd"]) for r in ok if r.get("charged_usd")]
    print(f"\n  Quick summary ({len(ok)} successful rows):")
    print(f"    cost/req  : ${mean(costs):.8f}" if costs else "    cost/req  : n/a")

    cell_src: dict[str, Counter] = {}
    for r in mixed_rows:
        cell = f"{r.get('domain','?')}:{r.get('complexity','?')}"
        if cell not in cell_src:
            cell_src[cell] = Counter()
        cell_src[cell][r["mode"].split("/")[-1]] += 1

    print(f"\n  Per-cell source:")
    for cell in sorted(cell_src):
        counts = cell_src[cell]
        total  = sum(counts.values())
        parts  = "  ".join(f"{src}={n}" for src, n in sorted(counts.items()))
        print(f"    {cell:<22}  n={total:4d}  {parts}")

    print(f"\n  Saved: {args.output}")
    print(f"\n  Next step:")
    print(f"    python tests/compare_all.py \\")
    print(f"      --system \"Mixed:{args.output}\" \\")
    print(f"      --system \"CARROT:{args.carrot}\" \\")
    print(f"      --system \"OmniRouter:{args.omni}\" \\")
    print(f"      --system \"TTCA:{args.ttca}\" \\")
    print(f"      --eval-matrix <your_eval_matrix.csv>")


if __name__ == "__main__":
    main()
