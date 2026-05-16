"""
compare_ttca.py — Time-to-Correct-Answer (TTCA) analysis.

Two modes:
  Simulation mode (no --eval-matrix):
    Retry latency = median wall_ms of the fallback model from the test CSV.
    Assumes retry always succeeds (optimistic).

  Exact mode (--eval-matrix results/eval_matrix.csv):
    Uses actual per-request latency AND correctness of the fallback model
    from eval_all_models.py output. No estimates needed.
    Unresolved = both models answered wrong for that specific request.

Usage:
    # Simulation mode
    python tests/compare_ttca.py \\
        --router   results/router_accuracy.csv \\
        --baseline results/rr_baseline.csv

    # Exact mode (recommended)
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


def _score_math(response: str, ground_truth: str):
    pred = _extract_numbers(response)
    true = _extract_numbers(str(ground_truth))
    if not pred or not true:
        return None
    p, t = pred[-1], true[-1]
    if t == 0:
        return 1.0 if abs(p) < 0.01 else 0.0
    return 1.0 if (abs(p - t) / abs(t) < 0.01 or abs(p - t) < 0.01) else 0.0


def _score_code(response: str, ground_truth: str):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    code = blocks[0].strip() if blocks else response.strip()
    try:
        ast.parse(code)
    except SyntaxError:
        return 0.0
    gt = str(ground_truth)
    if "assert" in gt or "==" in gt:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code + "\n" + gt)
            fname = f.name
        try:
            r = subprocess.run([sys.executable, fname], timeout=5, capture_output=True)
            return 1.0 if r.returncode == 0 else 0.0
        except Exception:
            return 0.0
    return 0.5


def _score_keyword(response: str, ground_truth: str):
    if not ground_truth or ground_truth.strip() in ("", "None"):
        return None
    gt_words = set(w.lower() for w in re.findall(r"\b\w{4,}\b", str(ground_truth)))
    if not gt_words:
        return None
    matches = sum(1 for w in gt_words if w in response.lower())
    overlap = matches / len(gt_words)
    return 1.0 if overlap >= 0.5 else overlap


_SCORERS = {
    "math":      _score_math,
    "code":      _score_code,
    "factual":   _score_keyword,
    "reasoning": _score_keyword,
    "creative":  lambda r, _: 1.0 if len(r.split()) >= 20 else 0.0,
}

CORRECT_THRESHOLD = 0.7

# When ALL models give wrong answer the user is stuck — we apply a penalty
# multiplier to TTCA so "no correct answer after N attempts" weighs more
# than "correct answer after N attempts". Set to 1.0 to disable.
UNRESOLVED_PENALTY = 3.0


def score_row(row: dict):
    if str(row.get("status", "")) != "200":
        return None
    response = row.get("response_text", "")
    if not response:
        return None
    scorer = _SCORERS.get(row.get("domain", "").lower())
    if scorer is None:
        return None
    return scorer(response, row.get("ground_truth", ""))


def is_correct(row: dict) -> bool | None:
    s = score_row(row)
    if s is None:
        return None
    return s >= CORRECT_THRESHOLD


# ---------------------------------------------------------------------------
# Eval matrix
# ---------------------------------------------------------------------------

def load_eval_matrix(path: str) -> dict[tuple[int, str], dict]:
    """
    Load eval_all_models.py output.
    Returns {(req_id, model_id): row} for O(1) lookup.
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

def compute_model_latencies(rows: list[dict]) -> dict[str, float]:
    by_model: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if str(r.get("status")) == "200":
            try:
                by_model[r["model_winner"]].append(float(r["wall_ms"]))
            except (KeyError, ValueError):
                pass
    return {m: median(vals) for m, vals in by_model.items()}


def compute_ttca(
    rows: list[dict],
    label: str,
    eval_matrix: dict | None = None,
) -> dict:
    ok             = [r for r in rows if str(r.get("status")) == "200"]
    model_latencies = compute_model_latencies(ok)
    all_models     = list(model_latencies.keys())
    exact_mode     = eval_matrix is not None

    ttca_resolved: list[float] = []
    ttca_all:      list[float] = []
    first_try_ok = retry_ok = unresolved = unscorable = 0

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

        if correctness:
            ttca_resolved.append(wall)
            ttca_all.append(wall)
            first_try_ok += 1
            continue

        # First model was wrong — need retry
        other_models = [m for m in all_models if m != winner]
        if not other_models:
            ttca_all.append(wall)
            unresolved += 1
            continue

        retry_model = other_models[0]

        if exact_mode:
            # Look up actual latency + correctness of fallback for this request
            fallback = eval_matrix.get((req_id, retry_model))
            if fallback and str(fallback.get("status")) == "200":
                try:
                    retry_lat = float(fallback["wall_ms"])
                except (KeyError, ValueError):
                    retry_lat = model_latencies.get(retry_model, median(model_latencies.values()))
                retry_correct = fallback.get("is_correct") == "true"
            else:
                retry_lat     = model_latencies.get(retry_model, median(model_latencies.values()))
                retry_correct = False
        else:
            # Simulation: median latency, assume success
            retry_lat     = model_latencies.get(retry_model, median(model_latencies.values()))
            retry_correct = True

        total = wall + retry_lat
        if retry_correct or not exact_mode:
            ttca_resolved.append(total)
            ttca_all.append(total)
            retry_ok += 1
        else:
            # All models wrong — apply penalty to signal user is completely stuck
            ttca_all.append(total * UNRESOLVED_PENALTY)
            unresolved += 1

    n_scorable = first_try_ok + retry_ok + unresolved
    failure_rate = unresolved / max(n_scorable, 1)

    def pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    return {
        "label":              label,
        "exact_mode":         exact_mode,
        "n_total":            len(ok),
        "n_scorable":         n_scorable,
        "n_unscorable":       unscorable,
        "first_try_ok":       first_try_ok,
        "retry_ok":           retry_ok,
        "unresolved":         unresolved,
        "failure_rate":       unresolved / max(n_scorable, 1),
        "first_try_rate":     first_try_ok / max(n_scorable, 1),
        "resolution_rate":    (first_try_ok + retry_ok) / max(n_scorable, 1),
        "ttca_resolved_p50":  pct(ttca_resolved, 50),
        "ttca_resolved_p95":  pct(ttca_resolved, 95),
        "ttca_resolved_mean": mean(ttca_resolved) if ttca_resolved else 0.0,
        "ttca_all_p50":       pct(ttca_all, 50),
        "ttca_all_p95":       pct(ttca_all, 95),
        "ttca_all_mean":      mean(ttca_all) if ttca_all else 0.0,
        "model_latencies":    model_latencies,
        "by_domain":          _by_domain(ok, model_latencies, eval_matrix),
    }


def _by_domain(
    ok: list[dict],
    model_latencies: dict[str, float],
    eval_matrix: dict | None,
) -> dict:
    all_models  = list(model_latencies.keys())
    exact_mode  = eval_matrix is not None
    by_domain: dict[str, dict] = defaultdict(lambda: {
        "first_try": 0, "retry": 0, "unresolved": 0, "ttca": []
    })

    for row in ok:
        correctness = is_correct(row)
        if correctness is None:
            continue
        try:
            wall   = float(row["wall_ms"])
            req_id = int(row["req_id"])
        except (KeyError, ValueError):
            continue

        d      = row.get("domain", "unknown")
        winner = row.get("model_winner", "")

        if correctness:
            by_domain[d]["first_try"] += 1
            by_domain[d]["ttca"].append(wall)
            continue

        other = [m for m in all_models if m != winner]
        if not other:
            by_domain[d]["unresolved"] += 1
            by_domain[d]["ttca"].append(wall)
            continue

        retry_model = other[0]
        if exact_mode:
            fallback = eval_matrix.get((req_id, retry_model))
            if fallback and str(fallback.get("status")) == "200":
                try:
                    retry_lat = float(fallback["wall_ms"])
                except (KeyError, ValueError):
                    retry_lat = model_latencies.get(retry_model, median(model_latencies.values()))
                retry_correct = fallback.get("is_correct") == "true"
            else:
                retry_lat, retry_correct = model_latencies.get(retry_model, 0.0), False
        else:
            retry_lat     = model_latencies.get(retry_model, median(model_latencies.values()))
            retry_correct = True

        total = wall + retry_lat
        if retry_correct or not exact_mode:
            by_domain[d]["retry"] += 1
            by_domain[d]["ttca"].append(total)
        else:
            by_domain[d]["unresolved"] += 1
            by_domain[d]["ttca"].append(total * UNRESOLVED_PENALTY)

    result = {}
    for d, v in by_domain.items():
        total  = v["first_try"] + v["retry"] + v["unresolved"]
        ttca_v = v["ttca"]
        result[d] = {
            "first_try_rate":  v["first_try"] / max(total, 1),
            "resolution_rate": (v["first_try"] + v["retry"]) / max(total, 1),
            "ttca_mean":       mean(ttca_v) if ttca_v else 0.0,
            "ttca_p50":        sorted(ttca_v)[len(ttca_v) // 2] if ttca_v else 0.0,
            "n": total,
        }
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct_change(baseline: float, new: float) -> str:
    if baseline == 0:
        return "-"
    delta = (new - baseline) / baseline * 100
    sign  = "+" if delta >= 0 else ""
    arrow = "up" if delta > 0 else "down"
    return f"{arrow} {sign}{delta:.1f}%"


def print_report(results: list[dict]) -> None:
    W  = 76
    C1 = 30
    C2 = 22

    def row(label, *vals, highlight=False):
        mark = "  <-" if highlight else ""
        cols = "".join(f"{str(v):<{C2}}" for v in vals)
        print(f"  {label:<{C1}} {cols}{mark}")

    print(f"\n{'='*W}")
    print("  TIME-TO-CORRECT-ANSWER (TTCA) COMPARISON")
    print(f"{'='*W}")
    mode_tag = "EXACT (eval matrix)" if results[0]["exact_mode"] else "SIMULATION (median latency)"
    print(f"  Mode: {mode_tag}")
    print(f"  If first model wrong: TTCA += actual retry model latency from eval matrix")
    print(f"  Correct threshold: >= {CORRECT_THRESHOLD:.0%}")
    print()

    for r in results:
        print(f"  [{r['label']}]  n={r['n_total']}  "
              f"scorable={r['n_scorable']}  unscorable={r['n_unscorable']}")
    print()

    for r in results:
        print(f"  [{r['label']}] model median latencies:")
        for m, lat in sorted(r["model_latencies"].items()):
            print(f"    {m:<28} {lat:.0f} ms")
    print()

    # Headers
    print(f"  {'METRIC':<{C1}}", end="")
    for r in results:
        print(f"  {r['label'][:20]:<{C2}}", end="")
    if len(results) == 2:
        print(f"  {'Change':<{C2}}", end="")
    print()
    print(f"  {'-'*74}")

    def mrow(label, key, fmt=str, higher_better=True):
        vals = [r[key] for r in results]
        fmted = [fmt(v) for v in vals]
        change = ""
        hi = False
        if len(results) == 2:
            change = pct_change(vals[1], vals[0])
            hi = (vals[0] > vals[1]) if higher_better else (vals[0] < vals[1])
        row(label, *fmted, *(([change]) if len(results) == 2 else []), highlight=hi)

    mrow("First-try success rate",      "first_try_rate",
         fmt=lambda x: f"{x*100:.1f}%", higher_better=True)
    mrow("Resolution rate (<=1 retry)", "resolution_rate",
         fmt=lambda x: f"{x*100:.1f}%", higher_better=True)
    mrow("Failure rate (all wrong)",    "failure_rate",
         fmt=lambda x: f"{x*100:.1f}%", higher_better=False)

    print(f"  {'-'*74}")
    print(f"  Counts: first-try correct / retried OK / ALL-WRONG (both models failed)")
    for r in results:
        unres_pct = r["unresolved"] / max(r["n_scorable"], 1) * 100
        tag = f"  [{unres_pct:.1f}% no correct answer]" if r["unresolved"] > 0 else ""
        print(f"    {r['label']}: {r['first_try_ok']} / {r['retry_ok']} / {r['unresolved']}{tag}")

    print(f"  {'-'*74}")
    print(f"  TTCA - resolved requests only (at least one model correct)")
    mrow("  P50",  "ttca_resolved_p50",  fmt=lambda x: f"{x:.0f} ms", higher_better=False)
    mrow("  P95",  "ttca_resolved_p95",  fmt=lambda x: f"{x:.0f} ms", higher_better=False)
    mrow("  Mean", "ttca_resolved_mean", fmt=lambda x: f"{x:.0f} ms", higher_better=False)
    print(f"  {'-'*74}")
    print(f"  TTCA - all scorable (unresolved penalised x{UNRESOLVED_PENALTY:.0f})")
    mrow("  P50",  "ttca_all_p50",  fmt=lambda x: f"{x:.0f} ms", higher_better=False)
    mrow("  P95",  "ttca_all_p95",  fmt=lambda x: f"{x:.0f} ms", higher_better=False)
    mrow("  Mean", "ttca_all_mean", fmt=lambda x: f"{x:.0f} ms", higher_better=False)

    # Per-domain
    DOMAINS = ["math", "code", "factual", "reasoning"]
    print(f"\n  {'DOMAIN BREAKDOWN':<{C1}}", end="")
    for r in results:
        print(f"  {r['label'][:20]:<{C2}}", end="")
    print()
    print(f"  {'-'*74}")

    all_domains = set()
    for r in results:
        all_domains.update(r["by_domain"].keys())

    for domain in DOMAINS:
        if domain not in all_domains:
            continue
        print(f"  {domain}")

        def drow(label, key, fmt=lambda x: f"{x:.0f} ms", higher_better=False):
            vals = [r["by_domain"].get(domain, {}).get(key, 0.0) for r in results]
            fmted = [fmt(v) for v in vals]
            change = ""
            hi = False
            if len(results) == 2 and vals[0] and vals[1]:
                change = pct_change(vals[1], vals[0])
                hi = (vals[0] < vals[1]) if not higher_better else (vals[0] > vals[1])
            row(f"    {label}", *fmted, *(([change]) if len(results) == 2 else []), highlight=hi)

        drow("First-try rate", "first_try_rate",
             fmt=lambda x: f"{x*100:.1f}%", higher_better=True)
        drow("TTCA mean", "ttca_mean", higher_better=False)
        drow("TTCA P50",  "ttca_p50",  higher_better=False)

    # Summary
    if len(results) == 2:
        r, b = results[0], results[1]
        print(f"\n{'='*W}")
        print("  SUMMARY")
        print(f"{'='*W}")
        ftr_d  = (r["first_try_rate"]  - b["first_try_rate"])  * 100
        rr_d   = (r["resolution_rate"] - b["resolution_rate"]) * 100
        ttca_d = r["ttca_all_mean"]    - b["ttca_all_mean"]
        print(f"  First-try success: {r['label']} {ftr_d:+.1f}pp vs {b['label']}")
        print(f"  Resolution rate:   {r['label']} {rr_d:+.1f}pp vs {b['label']}")
        if ttca_d < 0:
            pct = abs(ttca_d) / max(b["ttca_all_mean"], 1) * 100
            print(f"  TTCA improvement:  {r['label']} is {abs(ttca_d):.0f} ms faster "
                  f"({pct:.1f}% reduction in time-to-correct-answer)")
        else:
            print(f"  TTCA:              {b['label']} is {ttca_d:.0f} ms faster per request")
            print("                     Router trades single-request latency for higher "
                  "first-try accuracy")

    print(f"\n{'='*W}\n")
    print(f"  Notes:")
    print(f"  - Correct threshold: score >= {CORRECT_THRESHOLD:.0%}")
    if results[0]["exact_mode"]:
        print("  - Retry latency and correctness from actual eval_matrix per-request data")
        print("  - Unresolved = BOTH models gave wrong answer for that specific request")
    else:
        print("  - Retry latency = median wall_ms of fallback model (simulation)")
        print("  - Run eval_all_models.py and pass --eval-matrix for exact unresolved count")
    print(f"  - Unresolved TTCA penalised x{UNRESOLVED_PENALTY:.0f}: "
          f"user waited through all attempts and still got no correct answer")
    print("  - Use --penalty N to change the multiplier (1.0 = no penalty beyond wait time)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    global UNRESOLVED_PENALTY
    parser = argparse.ArgumentParser(description="Time-to-Correct-Answer comparison")
    parser.add_argument("--router",      required=True, help="Router results CSV")
    parser.add_argument("--baseline",    required=True, help="Round-robin baseline CSV")
    parser.add_argument("--eval-matrix", default=None,
                        help="eval_matrix.csv from eval_all_models.py for exact TTCA")
    parser.add_argument("--penalty",     type=float, default=UNRESOLVED_PENALTY,
                        help=f"TTCA multiplier when all models fail (default {UNRESOLVED_PENALTY})")
    args = parser.parse_args()

    UNRESOLVED_PENALTY = args.penalty
    eval_matrix = load_eval_matrix(args.eval_matrix) if args.eval_matrix else None
    if eval_matrix:
        print(f"  Loaded eval matrix: {len(eval_matrix)} (req_id, model_id) entries")

    results = [
        compute_ttca(load_csv(args.router),   label=f"Router ({Path(args.router).stem})",
                     eval_matrix=eval_matrix),
        compute_ttca(load_csv(args.baseline), label=f"Round-Robin ({Path(args.baseline).stem})",
                     eval_matrix=eval_matrix),
    ]
    print_report(results)


if __name__ == "__main__":
    main()
