"""
baseline_cost_optimal.py — Tier-Optimal-Cost oracle baseline.

Computes a per-request cost lower bound: for each request, select the
cheapest model that actually answered it correctly (using the pre-computed
eval_matrix.csv as ground truth).

This is a *post-hoc oracle* — it cannot be deployed in production because it
requires knowing which models answer correctly before routing.  It serves as a
lower bound on cost at equal or better accuracy than any deployed system.

Comparison with Tier-Optimal-Accuracy oracle (baseline_complexity_tier.py
with --tier-map accuracy_optimal):
  Tier-Optimal-Acc  : per-cell best, routes every easy/factual to llama4-scout
                       if that cell's accuracy winner is llama4-scout — expensive
  Tier-Optimal-Cost : per-request cheapest correct model — the minimum you could
                       pay if you knew the answer in advance

No live vLLM endpoints are required.  The script reads eval_matrix.csv,
synthesizes an output CSV, and exits.

Usage:
    python tests/baseline_cost_optimal.py \\
        --eval-matrix results/eval_matrix.csv \\
        --output      results/baseline_cost_optimal.csv

compare_all.py compatibility:
    compare_all.py uses eval_index[(req_id, model_winner)] -> is_correct to
    score each row.  For every request where a correct model exists, this
    oracle sets model_winner to the cheapest correct model_id, so the eval
    index lookup always returns True — accuracy = 100% of solvable requests.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

# ---------------------------------------------------------------------------
# Market-rate cost tables (USD per token) — matches dynamic_provisioner.py
# ---------------------------------------------------------------------------

COST_RATES: dict[str, dict] = {
    "qwen-7b":         {"input": 5e-8,  "output": 1e-7},   # $0.05/$0.10 per 1M
    "deepseek-r1-7b":  {"input": 6e-8,  "output": 1.4e-7}, # $0.06/$0.14
    "qwen3-coder-30b": {"input": 1.5e-7,"output": 6e-7},   # $0.15/$0.60
    "gemma-3-27b":     {"input": 8e-8,  "output": 1.6e-7}, # $0.08/$0.16
    "deepseek-r1-14b": {"input": 1e-7,  "output": 2.5e-7}, # $0.10/$0.25
    "llama4-scout":    {"input": 1e-7,  "output": 3e-7},   # $0.10/$0.30
}

EFF_TOK_PER_J: dict[str, float] = {
    "qwen-7b":         13.0,
    "deepseek-r1-7b":  13.0,
    "qwen3-coder-30b": 12.0,
    "gemma-3-27b":      5.0,
    "deepseek-r1-14b":  6.0,
    "llama4-scout":     3.0,
}

FIELDNAMES = [
    "req_id", "domain", "complexity", "query", "ground_truth", "mode",
    "status", "model_winner", "bid_latency_ms", "actual_latency_ms",
    "ttft_ms", "output_tokens", "charged_usd", "energy_j", "load",
    "wall_ms", "slo_ms", "slo_violated", "response_text", "error",
]


def _estimate_cost(model_id: str, query: str, response_text: str) -> float:
    rates = COST_RATES.get(model_id, {"input": 1e-7, "output": 2e-7})
    in_tok  = len(query.split()) * 1.3
    out_tok = len(response_text.split()) * 1.3 if response_text else 200.0
    return in_tok * rates["input"] + out_tok * rates["output"]


def _estimate_energy(model_id: str, response_text: str) -> float:
    eff = EFF_TOK_PER_J.get(model_id, 6.0)
    out_tok = len(response_text.split()) * 1.3 if response_text else 200.0
    return out_tok / eff


def build_oracle(eval_matrix: str) -> list[dict]:
    """Read eval_matrix.csv, pick cheapest correct model per request.

    Returns a list of synthesized result rows in FIELDNAMES format.
    """
    # Group rows by req_id
    by_req: dict[str, list[dict]] = defaultdict(list)
    with open(eval_matrix, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "200":
                continue
            by_req[row["req_id"]].append(row)

    output_rows: list[dict] = []
    n_correct = 0
    n_no_correct = 0

    for req_id, candidates in sorted(by_req.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        # Attach estimated cost to each candidate
        for c in candidates:
            c["_cost"] = _estimate_cost(
                c["model_id"], c.get("query", ""), c.get("response_text", "")
            )

        # Prefer cheapest model that got it right
        correct_candidates = [c for c in candidates if c.get("is_correct") == "true"]

        if correct_candidates:
            chosen = min(correct_candidates, key=lambda c: c["_cost"])
            n_correct += 1
        else:
            # Fallback: cheapest model overall (will score as incorrect via eval_index)
            chosen = min(candidates, key=lambda c: c["_cost"])
            n_no_correct += 1

        model_id     = chosen["model_id"]
        query        = chosen.get("query", "")
        resp_text    = chosen.get("response_text", "")
        cost         = chosen["_cost"]
        wall_ms_val  = chosen.get("wall_ms", "")
        out_tok_est  = int(len(resp_text.split()) * 1.3) if resp_text else 200
        energy       = _estimate_energy(model_id, resp_text)

        output_rows.append({
            "req_id":            req_id,
            "domain":            chosen.get("domain", ""),
            "complexity":        chosen.get("complexity", ""),
            "query":             query[:100],
            "ground_truth":      str(chosen.get("ground_truth", "")),
            "mode":              "cost_optimal",
            "status":            "200",
            "model_winner":      model_id,
            "bid_latency_ms":    "",
            "actual_latency_ms": wall_ms_val,
            "ttft_ms":           wall_ms_val,
            "output_tokens":     out_tok_est,
            "charged_usd":       f"{cost:.8f}",
            "energy_j":          f"{energy:.3f}",
            "load":              "",
            "wall_ms":           wall_ms_val,
            "slo_ms":            "",
            "slo_violated":      "",
            "response_text":     resp_text[:500] if resp_text else "",
            "error":             "",
        })

    return output_rows, n_correct, n_no_correct


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tier-Optimal-Cost oracle — cheapest correct model per request."
    )
    parser.add_argument("--eval-matrix", required=True,
                        help="Path to eval_matrix.csv from eval_all_models.py")
    parser.add_argument("--output", default="results/baseline_cost_optimal.csv")
    args = parser.parse_args()

    if not os.path.exists(args.eval_matrix):
        raise SystemExit(f"ERROR: eval_matrix not found: {args.eval_matrix}")

    print(f"\n  [Cost-Optimal Oracle] Reading: {args.eval_matrix}")
    rows, n_correct, n_no_correct = build_oracle(args.eval_matrix)
    n = len(rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Requests      : {n}")
    print(f"  Solvable      : {n_correct} ({100*n_correct//max(n,1)}%) — cheapest correct model selected")
    print(f"  Unsolvable    : {n_no_correct} ({100*n_no_correct//max(n,1)}%) — fallback to cheapest model")

    # Distribution over selected models
    model_counts: dict[str, int] = {}
    for r in rows:
        m = r["model_winner"]
        model_counts[m] = model_counts.get(m, 0) + 1
    print(f"\n  Model selection distribution:")
    for m, c in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {m:<28} {c:4d} ({100*c//max(n,1)}%)")

    costs  = [float(r["charged_usd"]) for r in rows if r["charged_usd"]]
    energy = [float(r["energy_j"])    for r in rows if r["energy_j"]]
    if costs:
        print(f"\n  Cost:   total=${sum(costs):.6f}  avg=${mean(costs):.8f}/req")
    if energy:
        print(f"  Energy: total={sum(energy):.1f}J  avg={mean(energy):.2f}J/req")

    print(f"\n  Saved: {args.output}")
    print(f'  (Pass to compare_all.py as --system "Tier-Opt-Cost:{args.output}")\n')


if __name__ == "__main__":
    main()
