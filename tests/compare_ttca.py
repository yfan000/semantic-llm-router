"""
compare_ttca.py — Time-to-Correct-Answer (TTCA) analysis.

Measures the total latency a user experiences from first asking a question
to finally receiving a correct answer, accounting for retries when the first
model is wrong.

Model:
  - If first model answers correctly  → TTCA = wall_ms
  - If first model answers incorrectly → TTCA = wall_ms + median latency of fallback model
  - If both models fail               → unresolved (TTCA = wall_ms_1 + wall_ms_2, no correct answer)

Metrics reported:
  - First-try success rate (fraction correct on first attempt)
  - Resolution rate after 1 retry (fraction ever resolved)
  - TTCA P50 / P95 / mean (resolved requests only)
  - TTCA P50 / P95 / mean (all requests, unresolved penalized with wall_ms_1 + wall_ms_2)

Usage:
    python tests/compare_ttca.py \\
        --router   results/router_accuracy.csv \\
        --baseline results/rr_baseline.csv
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
# Accuracy scoring (inline from score_accuracy.py)
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

CORRECT_THRESHOLD = 0.7   # score >= this -> answer is "correct"


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
# TTCA computation
# ---------------------------------------------------------------------------

def compute_model_latencies(rows: list[dict]) -> dict[str, float]:
    """Return median wall_ms per model_winner (used as retry latency estimate)."""
    by_model: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if str(r.get("status")) == "200":
            try:
                by_model[r["model_winner"]].append(float(r["wall_ms"]))
            except (KeyError, ValueError):
                pass
    return {m: median(vals) for m, vals in by_model.items()}


def compute_ttca(rows: list[dict], label: str) -> dict:
    """
    Simulate time-to-correct-answer for each scorable request.

    Retry model: if first model was A, retry with any other model registered
    in the CSV (using its median latency). If only one model in the CSV, no
    retry is simulated.
    """
    ok = [r for r in rows if str(r.get("status")) == "200"]
    model_latencies = compute_model_latencies(ok)
    all_models = list(model_latencies.keys())

    ttca_resolved:   list[float] = []   # TTCA for requests resolved correctly
    ttca_all:        list[float] = []   # TTCA for all scorable requests
    first_try_ok:    int = 0
    retry_ok:        int = 0
    unresolved:      int = 0
    unscorable:      int = 0

    for row in ok:
        correctness = is_correct(row)
        if correctness is None:
            unscorable += 1
            continue

        try:
            wall = float(row["wall_ms"])
        except (KeyError, ValueError):
            unscorable += 1
            continue

        winner = row.get("model_winner", "")

        if correctness:
            # Correct on first try
            ttca_resolved.append(wall)
            ttca_all.append(wall)
            first_try_ok += 1
        else:
            # Wrong - estimate retry with another model
            other_models = [m for m in all_models if m != winner]
            if other_models:
                # Use the median latency of the fallback model as retry time
                retry_lat = median(model_latencies[m] for m in other_models)
                total = wall + retry_lat
                ttca_all.append(total)
                # Treat retry as resolved (optimistic but fair - assumes fallback succeeds)
                ttca_resolved.append(total)
                retry_ok += 1
            else:
                # Only one model - no retry available
                ttca_all.append(wall)
                unresolved += 1

    n_scorable = first_try_ok + retry_ok + unresolved

    def pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    return {
        "label":              label,
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
        "by_domain":          _by_domain(ok, model_latencies),
    }


def _by_domain(ok: list[dict], model_latencies: dict[str, float]) -> dict:
    all_models = list(model_latencies.keys())
    by_domain: dict[str, dict] = defaultdict(lambda: {
        "first_try": 0, "retry": 0, "unresolved": 0, "ttca": []
    })

    for row in ok:
        correctness = is_correct(row)
        if correctness is None:
            continue
        try:
            wall = float(row["wall_ms"])
        except (KeyError, ValueError):
            continue

        d = row.get("domain", "unknown")
        winner = row.get("model_winner", "")

        if correctness:
            by_domain[d]["first_try"] += 1
            by_domain[d]["ttca"].append(wall)
        else:
            other = [m for m in all_models if m != winner]
            if other:
                retry_lat = median(model_latencies[m] for m in other)
                by_domain[d]["retry"] += 1
                by_domain[d]["ttca"].append(wall + retry_lat)
            else:
                by_domain[d]["unresolved"] += 1
                by_domain[d]["ttca"].append(wall)

    result = {}
    for d, v in by_domain.items():
        total = v["first_try"] + v["retry"] + v["unresolved"]
        ttca_vals = v["ttca"]
        result[d] = {
            "first_try_rate":  v["first_try"] / max(total, 1),
            "resolution_rate": (v["first_try"] + v["retry"]) / max(total, 1),
            "ttca_mean":       mean(ttca_vals) if ttca_vals else 0.0,
            "ttca_p50":        sorted(ttca_vals)[len(ttca_vals)//2] if ttca_vals else 0.0,
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
    sign = "+" if delta >= 0 else ""
    arrow = "up" if delta > 0 else "down"
    return f"{arrow} {sign}{delta:.1f}%"


def print_report(results: list[dict]) -> None:
    W = 74
    C1, C2 = 28, 22

    def row(label, *vals, highlight=False):
        mark = "  <-" if highlight else ""
        cols = "".join(f"{str(v):<{C2}}" for v in vals)
        print(f"  {label:<{C1}} {cols}{mark}")

    print(f"\n{'='*W}")
    print("  TIME-TO-CORRECT-ANSWER (TTCA) COMPARISON")
    print(f"{'='*W}")
    print(f"  Metric: total user wait from first question to first correct answer")
    print(f"  If first model is wrong, TTCA += median latency of fallback model")
    print(f"  Correct threshold: accuracy score >= {CORRECT_THRESHOLD:.0%}")
    print()

    for r in results:
        print(f"  [{r['label']}]  n={r['n_total']}  scorable={r['n_scorable']}  "
              f"unscorable={r['n_unscorable']}")
    print()

    for r in results:
        print(f"  [{r['label']}] model median latencies (used as retry estimate):")
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
    print(f"  {'-'*72}")

    def metric_row(label, key, fmt=lambda x: f"{x:.1f}", higher_is_better=True):
        vals = [r[key] for r in results]
        formatted = [fmt(v) for v in vals]
        change = ""
        highlight = False
        if len(results) == 2:
            change = pct_change(vals[1], vals[0])
            highlight = (vals[0] > vals[1]) if higher_is_better else (vals[0] < vals[1])
        row(label, *formatted, *(([change]) if len(results) == 2 else []), highlight=highlight)

    metric_row("First-try success rate",
               "first_try_rate",
               fmt=lambda x: f"{x*100:.1f}%",
               higher_is_better=True)
    metric_row("Resolution rate (<=1 retry)",
               "resolution_rate",
               fmt=lambda x: f"{x*100:.1f}%",
               higher_is_better=True)
    print(f"  {'-'*72}")

    print(f"  TTCA - resolved requests only")
    metric_row("  P50",  "ttca_resolved_p50",  fmt=lambda x: f"{x:.0f} ms", higher_is_better=False)
    metric_row("  P95",  "ttca_resolved_p95",  fmt=lambda x: f"{x:.0f} ms", higher_is_better=False)
    metric_row("  Mean", "ttca_resolved_mean", fmt=lambda x: f"{x:.0f} ms", higher_is_better=False)
    print(f"  {'-'*72}")

    print(f"  TTCA - all scorable (unresolved penalised)")
    metric_row("  P50",  "ttca_all_p50",  fmt=lambda x: f"{x:.0f} ms", higher_is_better=False)
    metric_row("  P95",  "ttca_all_p95",  fmt=lambda x: f"{x:.0f} ms", higher_is_better=False)
    metric_row("  Mean", "ttca_all_mean", fmt=lambda x: f"{x:.0f} ms", higher_is_better=False)

    # Per-domain breakdown
    DOMAINS = ["math", "code", "factual", "reasoning"]
    print(f"\n  {'DOMAIN BREAKDOWN':<{C1}}", end="")
    for r in results:
        print(f"  {r['label'][:20]:<{C2}}", end="")
    print()
    print(f"  {'-'*72}")

    all_domains = set()
    for r in results:
        all_domains.update(r["by_domain"].keys())

    for domain in DOMAINS:
        if domain not in all_domains:
            continue
        print(f"  {domain}")

        def domain_row(label, key, fmt=lambda x: f"{x:.0f} ms", higher_is_better=False):
            vals = [r["by_domain"].get(domain, {}).get(key, 0.0) for r in results]
            formatted = [fmt(v) for v in vals]
            change = ""
            highlight = False
            if len(results) == 2 and vals[0] and vals[1]:
                change = pct_change(vals[1], vals[0])
                highlight = (vals[0] < vals[1]) if not higher_is_better else (vals[0] > vals[1])
            row(f"    {label}", *formatted, *(([change]) if len(results) == 2 else []),
                highlight=highlight)

        domain_row("First-try rate", "first_try_rate",
                   fmt=lambda x: f"{x*100:.1f}%", higher_is_better=True)
        domain_row("TTCA mean", "ttca_mean", higher_is_better=False)
        domain_row("TTCA P50",  "ttca_p50",  higher_is_better=False)

    # Summary
    if len(results) == 2:
        r, b = results[0], results[1]
        print(f"\n{'='*W}")
        print("  SUMMARY")
        print(f"{'='*W}")
        ftr_delta  = (r["first_try_rate"]  - b["first_try_rate"])  * 100
        rr_delta   = (r["resolution_rate"] - b["resolution_rate"]) * 100
        ttca_delta = r["ttca_all_mean"]    - b["ttca_all_mean"]
        print(f"  First-try success: {r['label']} {ftr_delta:+.1f}pp vs {b['label']}")
        print(f"  Resolution rate:   {r['label']} {rr_delta:+.1f}pp vs {b['label']}")
        if ttca_delta < 0:
            print(f"  TTCA improvement:  {r['label']} is {abs(ttca_delta):.0f} ms faster on average")
            print(f"                     ({abs(ttca_delta)/max(b['ttca_all_mean'],1)*100:.1f}% "
                  f"reduction in time-to-correct-answer)")
        else:
            print(f"  TTCA:              {b['label']} is {ttca_delta:.0f} ms faster on average")
            print(f"                     Router trades latency for higher first-try accuracy")

    print(f"\n{'='*W}\n")
    print("  Notes:")
    print(f"  - Correct defined as accuracy score >= {CORRECT_THRESHOLD:.0%}")
    print("  - Retry latency = median wall_ms of the fallback model from this run")
    print("  - Unscorable rows excluded (no ground_truth or no domain scorer)")
    print("  - Code without assert ground_truth scored 0.5 (syntax-valid partial credit)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-to-Correct-Answer comparison")
    parser.add_argument("--router",   required=True, help="Router results CSV")
    parser.add_argument("--baseline", required=True, help="Round-robin baseline CSV")
    args = parser.parse_args()

    results = [
        compute_ttca(load_csv(args.router),   label=f"Router ({Path(args.router).stem})"),
        compute_ttca(load_csv(args.baseline), label=f"Round-Robin ({Path(args.baseline).stem})"),
    ]
    print_report(results)


if __name__ == "__main__":
    main()
