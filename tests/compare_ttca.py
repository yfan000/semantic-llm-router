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

    # ttca_resolved: time until user gets correct answer (first-try OR after retry)
    # ttca_failed:   time user wasted when ALL models gave wrong answer (no resolution)
    ttca_resolved: list[float] = []
    ttca_failed:   list[float] = []
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
            # Correct on first try
            ttca_resolved.append(wall)
            first_try_ok += 1
            continue

        # First model wrong — retry with another model
        other_models = [m for m in all_models if m != winner]
        if not other_models:
            # Only one model registered, no retry possible
            ttca_failed.append(wall)
            unresolved += 1
            continue

        retry_model = other_models[0]

        if exact_mode:
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
            retry_lat     = model_latencies.get(retry_model, median(model_latencies.values()))
            retry_correct = True  # simulation: optimistically assume retry succeeds

        total = wall + retry_lat
        if retry_correct or not exact_mode:
            ttca_resolved.append(total)
            retry_ok += 1
        else:
            # Both models wrong — user never got a correct answer
            ttca_failed.append(total)
            unresolved += 1

    n_scorable = first_try_ok + retry_ok + unresolved

    def pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    return {
        "label":               label,
        "exact_mode":          exact_mode,
        "n_total":             len(ok),
        "n_scorable":          n_scorable,
        "n_unscorable":        unscorable,
        "first_try_ok":        first_try_ok,
        "retry_ok":            retry_ok,
        "unresolved":          unresolved,
        "failure_rate":        unresolved / max(n_scorable, 1),
        "first_try_rate":      first_try_ok / max(n_scorable, 1),
        "resolution_rate":     (first_try_ok + retry_ok) / max(n_scorable, 1),
        # TTCA-resolved: distribution of wait time when user DOES get correct answer
        "ttca_resolved_p50":   pct(ttca_resolved, 50),
        "ttca_resolved_p95":   pct(ttca_resolved, 95),
        "ttca_resolved_mean":  mean(ttca_resolved) if ttca_resolved else 0.0,
        # TTCA-failed: time wasted when user NEVER gets correct answer (all models wrong)
        "ttca_failed_p50":     pct(ttca_failed, 50),
        "ttca_failed_p95":     pct(ttca_failed, 95),
        "ttca_failed_mean":    mean(ttca_failed) if ttca_failed else 0.0,
        "n_failed":            len(ttca_failed),
        "model_latencies":     model_latencies,
        "by_domain":            _categorize(ok, model_latencies, eval_matrix,
                                             key_fn=lambda r: r.get("domain", "unknown")),
        "by_complexity":        _categorize(ok, model_latencies, eval_matrix,
                                             key_fn=lambda r: r.get("complexity", "unknown")),
        "by_domain_complexity": _categorize(ok, model_latencies, eval_matrix,
                                             key_fn=lambda r:
                                             f"{r.get('domain','?')}:{r.get('complexity','?')}"),
    }


def _categorize(
    ok: list[dict],
    model_latencies: dict[str, float],
    eval_matrix: dict | None,
    key_fn,          # callable(row) -> str grouping key
) -> dict:
    """Generic breakdown by any grouping key (domain, complexity, domain:complexity)."""
    all_models = list(model_latencies.keys())
    exact_mode = eval_matrix is not None
    buckets: dict[str, dict] = defaultdict(lambda: {
        "first_try": 0, "retry": 0, "unresolved": 0,
        "ttca_resolved": [], "ttca_failed": [],
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

        key    = key_fn(row)
        winner = row.get("model_winner", "")

        if correctness:
            buckets[key]["first_try"] += 1
            buckets[key]["ttca_resolved"].append(wall)
            continue

        other = [m for m in all_models if m != winner]
        if not other:
            buckets[key]["unresolved"] += 1
            buckets[key]["ttca_failed"].append(wall)
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
            buckets[key]["retry"] += 1
            buckets[key]["ttca_resolved"].append(total)
        else:
            buckets[key]["unresolved"] += 1
            buckets[key]["ttca_failed"].append(total)

    result = {}
    for k, v in buckets.items():
        n   = v["first_try"] + v["retry"] + v["unresolved"]
        res = v["ttca_resolved"]
        fal = v["ttca_failed"]
        result[k] = {
            "first_try_rate":     v["first_try"] / max(n, 1),
            "resolution_rate":    (v["first_try"] + v["retry"]) / max(n, 1),
            "failure_rate":       v["unresolved"] / max(n, 1),
            "ttca_resolved_mean": mean(res) if res else 0.0,
            "ttca_resolved_p50":  sorted(res)[len(res) // 2] if res else 0.0,
            "ttca_failed_mean":   mean(fal) if fal else 0.0,
            "n": n,
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

    labels     = [r["label"][:C2] for r in results]
    change_hdr = ["Change"] if len(results) == 2 else []

    def hdr(*cols):
        print(f"  {cols[0]:<{C1}}", end="")
        for c in cols[1:]:
            print(f"  {str(c):<{C2}}", end="")
        print()

    def mrow(label, key, fmt=str, higher_better=True):
        vals  = [r[key] for r in results]
        fmted = [fmt(v) for v in vals]
        change, hi = "", False
        if len(results) == 2:
            change = pct_change(vals[1], vals[0])
            hi = (vals[0] > vals[1]) if higher_better else (vals[0] < vals[1])
        row(label, *fmted, *(([change]) if len(results) == 2 else []), highlight=hi)

    # ── Resolution stats ──────────────────────────────────────────────────
    hdr("OUTCOME", *labels, *change_hdr)
    print(f"  {'-'*74}")
    mrow("First-try success rate",       "first_try_rate",
         fmt=lambda x: f"{x*100:.1f}%", higher_better=True)
    mrow("Resolution rate (<=1 retry)",  "resolution_rate",
         fmt=lambda x: f"{x*100:.1f}%", higher_better=True)
    mrow("Failure rate (all wrong)",     "failure_rate",
         fmt=lambda x: f"{x*100:.1f}%", higher_better=False)
    print(f"  {'-'*74}")
    print(f"  Counts: first-try correct / retried OK / ALL-WRONG (no correct answer)")
    for r in results:
        unres_pct = r["unresolved"] / max(r["n_scorable"], 1) * 100
        tag = f"  ({unres_pct:.1f}% of requests had no correct answer)" if r["unresolved"] > 0 else ""
        print(f"    {r['label']}: {r['first_try_ok']} / {r['retry_ok']} / {r['unresolved']}{tag}")

    # ── TTCA — got correct answer ─────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  TTCA — REQUESTS THAT GOT CORRECT ANSWER  (n first-try + n retried)")
    print(f"  Time from question to correct answer, including any retry wait")
    hdr("", *labels, *change_hdr)
    print(f"  {'-'*74}")
    mrow("  Count",
         "resolution_rate",
         fmt=lambda x: f"{int(x * r['n_scorable'])} / {r['n_scorable']}"
             if False else "",   # placeholder — print manually below
         higher_better=True)
    for r in results:
        n_res = r["first_try_ok"] + r["retry_ok"]
        print(f"    [{r['label']}]  {n_res} requests resolved")
    mrow("  P50",  "ttca_resolved_p50",  fmt=lambda x: f"{x:.0f} ms", higher_better=False)
    mrow("  P95",  "ttca_resolved_p95",  fmt=lambda x: f"{x:.0f} ms", higher_better=False)
    mrow("  Mean", "ttca_resolved_mean", fmt=lambda x: f"{x:.0f} ms", higher_better=False)

    # ── TTCA — never got correct answer ───────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  TTCA — REQUESTS THAT NEVER GOT CORRECT ANSWER  (all models wrong)")
    print(f"  Time wasted waiting through all failed attempts")
    hdr("", *labels, *change_hdr)
    print(f"  {'-'*74}")
    for r in results:
        print(f"    [{r['label']}]  {r['unresolved']} requests unresolved "
              f"({r['unresolved']/max(r['n_scorable'],1)*100:.1f}%)")
    has_failed = any(r["unresolved"] > 0 for r in results)
    if has_failed:
        mrow("  P50",  "ttca_failed_p50",  fmt=lambda x: f"{x:.0f} ms" if x else "n/a",
             higher_better=False)
        mrow("  P95",  "ttca_failed_p95",  fmt=lambda x: f"{x:.0f} ms" if x else "n/a",
             higher_better=False)
        mrow("  Mean", "ttca_failed_mean", fmt=lambda x: f"{x:.0f} ms" if x else "n/a",
             higher_better=False)
    else:
        print("  (no failures in simulation mode — run with --eval-matrix for exact count)")

    # helper: print one stats row from a by_X dict
    def cat_row(label, by_key, cat_key, stat, fmt, higher_better):
        vals  = [r[by_key].get(cat_key, {}).get(stat, 0.0) for r in results]
        fmted = [fmt(v) for v in vals]
        change, hi = "", False
        if len(results) == 2 and vals[0] and vals[1]:
            change = pct_change(vals[1], vals[0])
            hi = (vals[0] > vals[1]) if higher_better else (vals[0] < vals[1])
        row(f"  {label}", *fmted, *(([change]) if len(results) == 2 else []), highlight=hi)

    def pct_fmt(x):  return f"{x*100:.1f}%"
    def ms_fmt(x):   return f"{x:.0f} ms" if x else "n/a"

    # ── By domain ────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  BREAKDOWN BY DOMAIN")
    print(f"{'='*W}")
    all_domains: set[str] = set()
    for r in results:
        all_domains.update(r["by_domain"].keys())

    for domain in ["math", "code", "factual", "reasoning"]:
        if domain not in all_domains:
            continue
        n_vals = [r["by_domain"].get(domain, {}).get("n", 0) for r in results]
        print(f"\n  {domain.upper()}  (n={'/'.join(str(n) for n in n_vals)})")
        hdr("", *labels, *change_hdr)
        print(f"  {'-'*70}")
        cat_row("First-try rate",      "by_domain", domain, "first_try_rate",     pct_fmt, True)
        cat_row("Resolution rate",     "by_domain", domain, "resolution_rate",    pct_fmt, True)
        cat_row("Failure rate",        "by_domain", domain, "failure_rate",       pct_fmt, False)
        cat_row("TTCA-resolved mean",  "by_domain", domain, "ttca_resolved_mean", ms_fmt,  False)
        cat_row("TTCA-failed mean",    "by_domain", domain, "ttca_failed_mean",   ms_fmt,  False)

    # ── By complexity ────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  BREAKDOWN BY DIFFICULTY")
    print(f"{'='*W}")

    for cplx in ["easy", "medium", "hard"]:
        n_vals = [r["by_complexity"].get(cplx, {}).get("n", 0) for r in results]
        if not any(n_vals):
            continue
        print(f"\n  {cplx.upper()}  (n={'/'.join(str(n) for n in n_vals)})")
        hdr("", *labels, *change_hdr)
        print(f"  {'-'*70}")
        cat_row("First-try rate",      "by_complexity", cplx, "first_try_rate",     pct_fmt, True)
        cat_row("Resolution rate",     "by_complexity", cplx, "resolution_rate",    pct_fmt, True)
        cat_row("Failure rate",        "by_complexity", cplx, "failure_rate",       pct_fmt, False)
        cat_row("TTCA-resolved mean",  "by_complexity", cplx, "ttca_resolved_mean", ms_fmt,  False)
        cat_row("TTCA-failed mean",    "by_complexity", cplx, "ttca_failed_mean",   ms_fmt,  False)

    # ── By domain × complexity matrix ────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  DOMAIN × DIFFICULTY MATRIX  (first-try rate  |  TTCA-resolved mean)")
    print(f"{'='*W}")

    DOMAINS   = ["math", "code", "factual", "reasoning"]
    COMPLEXITIES = ["easy", "medium", "hard"]

    for r in results:
        print(f"\n  [{r['label']}]")
        # header row
        print(f"  {'':14}", end="")
        for cplx in COMPLEXITIES:
            print(f"  {cplx:>22}", end="")
        print()
        print(f"  {'-'*80}")
        for domain in DOMAINS:
            print(f"  {domain:<14}", end="")
            for cplx in COMPLEXITIES:
                key  = f"{domain}:{cplx}"
                stat = r["by_domain_complexity"].get(key, {})
                ftr  = stat.get("first_try_rate", 0.0)
                ttca = stat.get("ttca_resolved_mean", 0.0)
                n    = stat.get("n", 0)
                if n:
                    cell = f"{ftr*100:.0f}% / {ttca:.0f}ms"
                else:
                    cell = "-"
                print(f"  {cell:>22}", end="")
            print()

    # ── Summary ───────────────────────────────────────────────────────────
    if len(results) == 2:
        r0, r1 = results[0], results[1]
        print(f"\n{'='*W}")
        print("  SUMMARY")
        print(f"{'='*W}")
        ftr_d  = (r0["first_try_rate"]   - r1["first_try_rate"])   * 100
        rr_d   = (r0["resolution_rate"]  - r1["resolution_rate"])  * 100
        fail_d = (r0["failure_rate"]     - r1["failure_rate"])      * 100
        ttca_d = r0["ttca_resolved_mean"] - r1["ttca_resolved_mean"]
        print(f"  First-try success  : {r0['label']} {ftr_d:+.1f}pp vs {r1['label']}")
        print(f"  Resolution rate    : {r0['label']} {rr_d:+.1f}pp vs {r1['label']}")
        print(f"  Failure rate       : {r0['label']} {fail_d:+.1f}pp vs {r1['label']}")
        if ttca_d < 0:
            pct_imp = abs(ttca_d) / max(r1["ttca_resolved_mean"], 1) * 100
            print(f"  TTCA (resolved)    : {r0['label']} is {abs(ttca_d):.0f} ms faster "
                  f"({pct_imp:.1f}% faster time-to-correct-answer)")
        elif ttca_d > 0:
            print(f"  TTCA (resolved)    : {r1['label']} is {ttca_d:.0f} ms faster per resolved request")
            print("                       (router uses slower 7B model which needs fewer retries)")

    print(f"\n{'='*W}\n")
    print("  Notes:")
    print(f"  - Correct defined as score >= {CORRECT_THRESHOLD:.0%}")
    print("  - TTCA-resolved: time until user gets correct answer (first-try or after retry)")
    print("  - TTCA-failed:   time wasted when ALL models gave wrong answer (no correct answer)")
    if results[0]["exact_mode"]:
        print("  - Exact mode: per-request latency and correctness from eval_matrix")
    else:
        print("  - Simulation mode: retry latency = median of fallback model; retry assumed correct")
        print("  - Run eval_all_models.py + pass --eval-matrix for exact failure count")
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
                        help="eval_matrix.csv from eval_all_models.py for exact TTCA")
    args = parser.parse_args()

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
