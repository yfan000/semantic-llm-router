"""
compare_ttca.py — Time-to-Correct-Answer (TTCA) analysis.

Two modes:

  Simulation mode (no --eval-matrix):
    Retry latency = median wall_ms of fallback model from the test CSV.
    Assumes retry succeeds (optimistic upper bound for the router benefit).

  Exact mode (--eval-matrix results/eval_matrix.csv):
    Uses actual per-request latency AND correctness of the fallback model
    from eval_all_models.py. No median estimates -- every number is real.
    Unresolved = BOTH models gave wrong answer for that specific request.

Usage:
    # Simulation (quick, no pre-evaluation needed)
    python tests/compare_ttca.py \\
        --router   results/router_accuracy.csv \\
        --baseline results/rr_baseline.csv

    # Exact (recommended -- run eval_all_models.py first)
    python tests/compare_ttca.py \\
        --router      results/router_accuracy.csv \\
        --baseline    results/rr_baseline.csv \\
        --eval-matrix results/eval_matrix.csv
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


# ---------------------------------------------------------------------------
# Accuracy scoring
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _score_math(response: str, gt: str):
    pred = _extract_numbers(response)
    true = _extract_numbers(str(gt))
    if not pred or not true:
        return None
    p, t = pred[-1], true[-1]
    if t == 0:
        return 1.0 if abs(p) < 0.01 else 0.0
    return 1.0 if (abs(p - t) / abs(t) < 0.01 or abs(p - t) < 0.01) else 0.0


def _score_code(response: str, gt: str):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    code = blocks[0].strip() if blocks else response.strip()
    try:
        ast.parse(code)
    except SyntaxError:
        return 0.0
    if "assert" in str(gt) or "==" in str(gt):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code + "\n" + str(gt))
            fname = f.name
        try:
            r = subprocess.run([sys.executable, fname], timeout=5, capture_output=True)
            return 1.0 if r.returncode == 0 else 0.0
        except Exception:
            return 0.0
    return 0.5


def _score_keyword(response: str, gt: str):
    if not gt or str(gt).strip() in ("", "None"):
        return None
    words = set(w.lower() for w in re.findall(r"\b\w{4,}\b", str(gt)))
    if not words:
        return None
    hits = sum(1 for w in words if w in response.lower())
    overlap = hits / len(words)
    return 1.0 if overlap >= 0.5 else overlap


_SCORERS = {
    "math":      _score_math,
    "code":      _score_code,
    "factual":   _score_keyword,
    "reasoning": _score_keyword,
    "creative":  lambda r, _: 1.0 if len(r.split()) >= 20 else 0.0,
}

CORRECT_THRESHOLD = 0.7


def score_row(row: dict):
    if str(row.get("status", "")) != "200":
        return None
    response = row.get("response_text", "")
    if not response:
        return None
    scorer = _SCORERS.get(row.get("domain", "").lower())
    return scorer(response, row.get("ground_truth", "")) if scorer else None


def is_correct(row: dict) -> bool | None:
    s = score_row(row)
    return None if s is None else s >= CORRECT_THRESHOLD


# ---------------------------------------------------------------------------
# Eval matrix (from eval_all_models.py)
# ---------------------------------------------------------------------------

def load_eval_matrix(path: str) -> dict[tuple[int, str], dict]:
    """
    Load eval_all_models.py output.
    Returns {(req_id, model_id): row} for O(1) per-request lookup.
    """
    matrix: dict[tuple[int, str], dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row["req_id"]), row["model_id"])
                matrix[key] = row
            except (KeyError, ValueError):
                pass
    return matrix


# ---------------------------------------------------------------------------
# TTCA computation
# ---------------------------------------------------------------------------

def _model_latencies(rows: list[dict]) -> dict[str, float]:
    by_model: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if str(r.get("status")) == "200":
            try:
                by_model[r["model_winner"]].append(float(r["wall_ms"]))
            except (KeyError, ValueError):
                pass
    return {m: median(v) for m, v in by_model.items()}


def _retry_info(
    req_id: int,
    retry_model: str,
    model_latencies: dict[str, float],
    eval_matrix: dict | None,
) -> tuple[float, bool]:
    """Return (retry_latency_ms, retry_is_correct)."""
    fallback_lat = model_latencies.get(retry_model,
                   median(model_latencies.values()) if model_latencies else 0.0)

    if eval_matrix is not None:
        fb = eval_matrix.get((req_id, retry_model))
        if fb and str(fb.get("status")) == "200":
            try:
                return float(fb["wall_ms"]), fb.get("is_correct") == "true"
            except (KeyError, ValueError):
                pass
        # Fallback model also failed in eval matrix
        return fallback_lat, False
    else:
        # Simulation: use median, optimistically assume success
        return fallback_lat, True


def compute_ttca(
    rows: list[dict],
    label: str,
    eval_matrix: dict | None = None,
) -> dict:
    ok              = [r for r in rows if str(r.get("status")) == "200"]
    model_latencies = _model_latencies(ok)
    all_models      = list(model_latencies.keys())
    exact           = eval_matrix is not None

    ttca_resolved: list[float] = []
    ttca_all:      list[float] = []
    first_try_ok = retry_ok = unresolved = unscorable = 0
    per_domain: dict[str, dict] = defaultdict(
        lambda: {"first_try": 0, "retry": 0, "unresolved": 0, "ttca": []}
    )

    for row in ok:
        correctness = is_correct(row)
        if correctness is None:
            unscorable += 1
            continue
        try:
            wall   = float(row["wall_ms"])
            req_id = int(row["req_id"])
        except (KeyError, ValueError):
            unscorable += 1
            continue

        winner = row.get("model_winner", "")
        domain = row.get("domain", "unknown")
        other  = [m for m in all_models if m != winner]

        if correctness:
            ttca_resolved.append(wall)
            ttca_all.append(wall)
            first_try_ok += 1
            per_domain[domain]["first_try"] += 1
            per_domain[domain]["ttca"].append(wall)

        elif not other:
            ttca_all.append(wall)
            unresolved += 1
            per_domain[domain]["unresolved"] += 1
            per_domain[domain]["ttca"].append(wall)

        else:
            retry_lat, retry_correct = _retry_info(
                req_id, other[0], model_latencies, eval_matrix
            )
            total = wall + retry_lat
            ttca_all.append(total)
            per_domain[domain]["ttca"].append(total)

            if retry_correct or not exact:
                ttca_resolved.append(total)
                retry_ok += 1
                per_domain[domain]["retry"] += 1
            else:
                unresolved += 1
                per_domain[domain]["unresolved"] += 1

    n_scorable = first_try_ok + retry_ok + unresolved

    def pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    domain_stats: dict[str, dict] = {}
    for d, v in per_domain.items():
        n   = v["first_try"] + v["retry"] + v["unresolved"]
        tvs = v["ttca"]
        domain_stats[d] = {
            "first_try_rate":  v["first_try"] / max(n, 1),
            "resolution_rate": (v["first_try"] + v["retry"]) / max(n, 1),
            "ttca_mean": mean(tvs) if tvs else 0.0,
            "ttca_p50":  sorted(tvs)[len(tvs) // 2] if tvs else 0.0,
            "n": n,
        }

    return {
        "label":              label,
        "exact":              exact,
        "n_total":            len(ok),
        "n_scorable":         n_scorable,
        "n_unscorable":       unscorable,
        "first_try_ok":       first_try_ok,
        "retry_ok":           retry_ok,
        "unresolved":         unresolved,
        "first_try_rate":     first_try_ok / max(n_scorable, 1),
        "resolution_rate":    (first_try_ok + retry_ok) / max(n_scorable, 1),
        "ttca_resolved_p50":  pct(ttca_resolved, 50),
        "ttca_resolved_p95":  pct(ttca_resolved, 95),
        "ttca_resolved_mean": mean(ttca_resolved) if ttca_resolved else 0.0,
        "ttca_all_p50":       pct(ttca_all, 50),
        "ttca_all_p95":       pct(ttca_all, 95),
        "ttca_all_mean":      mean(ttca_all) if ttca_all else 0.0,
        "model_latencies":    model_latencies,
        "by_domain":          domain_stats,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct_change(baseline: float, new: float) -> str:
    if baseline == 0:
        return "-"
    d = (new - baseline) / baseline * 100
    return f"{'up' if d > 0 else 'down'} {d:+.1f}%"


def print_report(results: list[dict]) -> None:
    W  = 76
    C1 = 30
    C2 = 20

    def row(label, *vals, good=None):
        mark = ""
        if good is not None and len(vals) >= 2:
            try:
                v0 = float(str(vals[0]).replace("%", "").replace(" ms", ""))
                v1 = float(str(vals[1]).replace("%", "").replace(" ms", ""))
                mark = "  <-" if (good and v0 > v1) or (not good and v0 < v1) else ""
            except Exception:
                pass
        cols = "".join(f"  {str(v):<{C2}}" for v in vals)
        print(f"  {label:<{C1}}{cols}{mark}")

    print(f"\n{'='*W}")
    print("  TIME-TO-CORRECT-ANSWER (TTCA) COMPARISON")
    print(f"{'='*W}")
    mode = ("EXACT -- per-request data from eval_all_models.py"
            if results[0]["exact"]
            else "SIMULATION -- median fallback latency (run eval_all_models.py for exact)")
    print(f"  Mode   : {mode}")
    print(f"  Rule   : wrong answer on first model -> TTCA += actual retry latency")
    print(f"  Correct: score >= {CORRECT_THRESHOLD:.0%}")
    print()

    for r in results:
        tag = "(exact)" if r["exact"] else "(simulated)"
        print(f"  [{r['label']}] {tag}  "
              f"n={r['n_total']}  scorable={r['n_scorable']}  unscorable={r['n_unscorable']}")
    print()

    for r in results:
        print(f"  [{r['label']}] model median latencies:")
        for m, lat in sorted(r["model_latencies"].items()):
            print(f"    {m:<28} {lat:.0f} ms")
    print()

    labels = [r["label"][:C2] for r in results]
    change_hdr = ["Change"] if len(results) == 2 else []
    row("METRIC", *labels, *change_hdr)
    print(f"  {'-'*72}")

    def mrow(label, key, pct=False, lower_better=False):
        vals  = [r[key] for r in results]
        fmted = [f"{v*100:.1f}%" if pct else f"{v:.0f} ms" for v in vals]
        extra = [_pct_change(vals[1], vals[0])] if len(results) == 2 else []
        row(label, *fmted, *extra, good=(not lower_better))

    mrow("First-try success rate",      "first_try_rate",     pct=True)
    mrow("Resolution rate (<=1 retry)", "resolution_rate",    pct=True)
    print(f"  {'-'*72}")
    print(f"  Counts: first-try / retried / unresolved")
    for r in results:
        print(f"    {r['label']}: {r['first_try_ok']} / {r['retry_ok']} / {r['unresolved']}")
    print(f"  {'-'*72}")
    print(f"  TTCA (resolved requests only)")
    mrow("  P50",  "ttca_resolved_p50",  lower_better=True)
    mrow("  P95",  "ttca_resolved_p95",  lower_better=True)
    mrow("  Mean", "ttca_resolved_mean", lower_better=True)
    print(f"  {'-'*72}")
    print(f"  TTCA (all scorable, unresolved penalised)")
    mrow("  P50",  "ttca_all_p50",  lower_better=True)
    mrow("  P95",  "ttca_all_p95",  lower_better=True)
    mrow("  Mean", "ttca_all_mean", lower_better=True)

    # Per-domain
    all_domains: set[str] = set()
    for r in results:
        all_domains.update(r["by_domain"].keys())

    print(f"\n  {'DOMAIN':<{C1}}", end="")
    for r in results:
        print(f"  {r['label'][:C2]:<{C2}}", end="")
    print()
    print(f"  {'-'*72}")

    for domain in ["math", "code", "factual", "reasoning"]:
        if domain not in all_domains:
            continue
        print(f"  {domain}")
        for key, label, is_pct, lb in [
            ("first_try_rate",  "    First-try rate", True,  False),
            ("ttca_mean",       "    TTCA mean",      False, True),
            ("ttca_p50",        "    TTCA P50",       False, True),
        ]:
            vals  = [r["by_domain"].get(domain, {}).get(key, 0.0) for r in results]
            fmted = [f"{v*100:.1f}%" if is_pct else f"{v:.0f} ms" for v in vals]
            extra = []
            if len(results) == 2 and vals[0] and vals[1]:
                extra = [_pct_change(vals[1], vals[0])]
            row(label, *fmted, *extra, good=(not lb))

    # Summary
    if len(results) == 2:
        r0, r1 = results[0], results[1]
        print(f"\n{'='*W}")
        print("  SUMMARY")
        print(f"{'='*W}")
        ftr_d  = (r0["first_try_rate"]  - r1["first_try_rate"])  * 100
        rr_d   = (r0["resolution_rate"] - r1["resolution_rate"]) * 100
        ttca_d = r0["ttca_all_mean"]    - r1["ttca_all_mean"]
        print(f"  First-try success : {r0['label']} {ftr_d:+.1f}pp vs {r1['label']}")
        print(f"  Resolution rate   : {r0['label']} {rr_d:+.1f}pp vs {r1['label']}")
        if ttca_d < 0:
            pct = abs(ttca_d) / max(r1["ttca_all_mean"], 1) * 100
            print(f"  TTCA improvement  : {r0['label']} is {abs(ttca_d):.0f} ms faster "
                  f"({pct:.1f}% reduction in time-to-correct-answer)")
        else:
            print(f"  TTCA              : {r1['label']} is {ttca_d:.0f} ms faster per request")
            print("                      (router trades first-request latency for fewer retries)")

    print(f"\n{'='*W}\n")
    print("  Notes:")
    print(f"  - Correct defined as score >= {CORRECT_THRESHOLD:.0%}")
    if results[0]["exact"]:
        print("  - Retry latency and correctness from actual eval_matrix data (exact)")
        print("  - Unresolved = both models gave the wrong answer for that specific request")
    else:
        print("  - Retry latency = median wall_ms of fallback model (simulation/estimate)")
        print("  - Run eval_all_models.py first, pass --eval-matrix for exact results")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-to-Correct-Answer comparison")
    parser.add_argument("--router",      required=True, help="Router results CSV")
    parser.add_argument("--baseline",    required=True, help="Round-robin baseline CSV")
    parser.add_argument("--eval-matrix", default=None,
                        help="eval_matrix.csv from eval_all_models.py (exact mode)")
    args = parser.parse_args()

    eval_matrix = None
    if args.eval_matrix:
        eval_matrix = load_eval_matrix(args.eval_matrix)
        print(f"\n  Loaded eval matrix: {len(eval_matrix)} (req_id, model_id) entries")

    results = [
        compute_ttca(load_csv(args.router),
                     label=f"Router ({Path(args.router).stem})",
                     eval_matrix=eval_matrix),
        compute_ttca(load_csv(args.baseline),
                     label=f"Round-Robin ({Path(args.baseline).stem})",
                     eval_matrix=eval_matrix),
    ]
    print_report(results)


if __name__ == "__main__":
    main()
