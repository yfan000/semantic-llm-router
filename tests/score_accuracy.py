"""
score_accuracy.py — Offline accuracy evaluation for router vs round-robin CSVs.

Reads one or more result CSVs (produced by load_test.py or round_robin_test.py),
scores each response against ground_truth using domain-specific deterministic
methods, and prints a side-by-side accuracy comparison.

Usage:
    python tests/score_accuracy.py \\
        --router   results/router_accuracy.csv \\
        --baseline results/round_robin.csv

    # Score a single CSV (no comparison)
    python tests/score_accuracy.py --router results/router_accuracy.csv

Scoring methods by domain:
    math       -- regex-extract the final numeric answer, compare to ground_truth
    code       -- Python syntax check + optional subprocess execution
    factual    -- answer-keyword overlap: ground_truth keywords found in response
    reasoning  -- expected-answer keyword presence in response
    creative   -- non-empty non-trivial response (always 1.0 if >= 20 words)
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
from statistics import mean


# ---------------------------------------------------------------------------
# Domain-specific scorers
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> list[float]:
    """Return all numbers found in text (handles negatives and decimals)."""
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text)]


def score_math(response: str, ground_truth: str) -> float:
    """
    Extract the last number from the response and compare to the last number
    in ground_truth. Tolerance: within 1% or absolute diff <= 0.01.
    Returns 1.0 (correct), 0.0 (wrong), or None (no number found -> skip).
    """
    pred_nums = _extract_numbers(response)
    true_nums = _extract_numbers(str(ground_truth))
    if not pred_nums or not true_nums:
        return None
    pred = pred_nums[-1]
    true = true_nums[-1]
    if true == 0:
        return 1.0 if abs(pred) < 0.01 else 0.0
    return 1.0 if (abs(pred - true) / abs(true) < 0.01 or abs(pred - true) < 0.01) else 0.0


def score_code(response: str, ground_truth: str) -> float:
    """
    Extract Python code block from response and:
      1. Run ast.parse() -- syntax check (0.0 on failure)
      2. If ground_truth contains assert statements, run them (1.0/0.0)
      3. Otherwise 0.5 for syntactically valid code (we can't fully verify)
    """
    # Extract code from markdown fences
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    code = code_blocks[0].strip() if code_blocks else response.strip()

    # Syntax check
    try:
        ast.parse(code)
    except SyntaxError:
        return 0.0

    # If ground_truth has assertions, run them
    gt = str(ground_truth)
    if "assert" in gt or "==" in gt:
        test_code = code + "\n" + gt
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(test_code)
            fname = f.name
        try:
            result = subprocess.run(
                [sys.executable, fname],
                timeout=5,
                capture_output=True,
            )
            return 1.0 if result.returncode == 0 else 0.0
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0

    # Syntax valid but no test cases -- partial credit
    return 0.5


def score_factual(response: str, ground_truth: str) -> float:
    """
    Keyword overlap: extract meaningful words (>=4 chars) from ground_truth,
    check what fraction appear in the response (case-insensitive).
    Threshold: >= 0.5 overlap -> 1.0, else proportional score.
    """
    if not ground_truth or ground_truth.strip() in ("", "None"):
        return None
    gt_words = set(w.lower() for w in re.findall(r"\b\w{4,}\b", str(ground_truth)))
    if not gt_words:
        return None
    resp_lower = response.lower()
    matches = sum(1 for w in gt_words if w in resp_lower)
    overlap = matches / len(gt_words)
    return 1.0 if overlap >= 0.5 else overlap


def score_reasoning(response: str, ground_truth: str) -> float:
    """
    Check if the expected answer/conclusion from ground_truth appears in the response.
    Uses the same keyword overlap as factual scoring.
    """
    return score_factual(response, ground_truth)


def score_creative(response: str, ground_truth: str) -> float:
    """
    Creative responses have no objective ground truth.
    Score 1.0 if response has >= 20 words, 0.0 otherwise (model refused or error).
    """
    words = response.split()
    return 1.0 if len(words) >= 20 else 0.0


SCORERS = {
    "math":      score_math,
    "code":      score_code,
    "factual":   score_factual,
    "reasoning": score_reasoning,
    "creative":  score_creative,
}


def score_row(row: dict) -> float | None:
    """Return accuracy score [0,1] or None if unscorable."""
    domain    = row.get("domain", "").lower()
    response  = row.get("response_text", "")
    gt        = row.get("ground_truth", "")
    status    = row.get("status", "")

    if str(status) != "200" or not response:
        return None

    scorer = SCORERS.get(domain)
    if scorer is None:
        return None

    return scorer(response, gt)


# ---------------------------------------------------------------------------
# CSV loading and aggregation
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict], label: str) -> dict:
    """Return {overall, per_domain, per_model} accuracy dicts."""
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_model:  dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []

    for row in rows:
        s = score_row(row)
        if s is None:
            continue
        domain = row.get("domain", "unknown")
        model  = row.get("model_winner", "unknown")
        by_domain[domain].append(s)
        by_model[model].append(s)
        all_scores.append(s)

    return {
        "label":      label,
        "n_scored":   len(all_scores),
        "n_total":    len(rows),
        "overall":    mean(all_scores) if all_scores else None,
        "by_domain":  {d: mean(v) for d, v in sorted(by_domain.items())},
        "by_model":   {m: (mean(v), len(v)) for m, v in sorted(by_model.items())},
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(results: list[dict]) -> None:
    W = 70
    DOMAINS = ["math", "code", "factual", "reasoning", "creative"]

    print(f"\n{'='*W}")
    print("  ACCURACY EVALUATION REPORT")
    print(f"{'='*W}")

    # Overall summary
    print(f"\n  {'Strategy':<26} {'Scored':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*56}")
    for r in results:
        acc = f"{r['overall']*100:.1f}%" if r["overall"] is not None else "N/A"
        print(f"  {r['label']:<26} {r['n_scored']:>8} {r['n_total']:>8} {acc:>10}")

    # Per-domain breakdown
    print(f"\n  {'Domain':<14}", end="")
    for r in results:
        print(f"  {r['label'][:20]:>22}", end="")
    print()
    print(f"  {'-'*(14 + 24*len(results))}")

    all_domains = set()
    for r in results:
        all_domains.update(r["by_domain"].keys())
    for domain in DOMAINS:
        if domain not in all_domains:
            continue
        print(f"  {domain:<14}", end="")
        for r in results:
            acc = r["by_domain"].get(domain)
            cell = f"{acc*100:.1f}%" if acc is not None else "-"
            print(f"  {cell:>22}", end="")
        print()

    # Per-model breakdown (within each strategy)
    for r in results:
        if not r["by_model"]:
            continue
        print(f"\n  [{r['label']}] per-model accuracy:")
        print(f"  {'Model':<26} {'Accuracy':>10} {'Requests':>10}")
        print(f"  {'-'*48}")
        for model, (acc, n) in sorted(r["by_model"].items(), key=lambda x: -x[1][0]):
            print(f"  {model:<26} {acc*100:>9.1f}% {n:>10}")

    # Delta (first vs rest)
    if len(results) >= 2:
        base = results[0]
        for r in results[1:]:
            print(f"\n  Delta accuracy ({base['label']} vs {r['label']}):")
            if base["overall"] is not None and r["overall"] is not None:
                delta = (base["overall"] - r["overall"]) * 100
                sign  = "+" if delta >= 0 else ""
                print(f"    Overall: {sign}{delta:.1f}pp  ", end="")
                if delta > 0:
                    print(f"(router is better by {delta:.1f} percentage points)")
                elif delta < 0:
                    print(f"(round-robin is better by {abs(delta):.1f} percentage points)")
                else:
                    print("(identical)")
            # Per domain delta
            all_d = set(base["by_domain"]) | set(r["by_domain"])
            for d in sorted(all_d):
                ba = base["by_domain"].get(d)
                ra = r["by_domain"].get(d)
                if ba is not None and ra is not None:
                    dd = (ba - ra) * 100
                    sign = "+" if dd >= 0 else ""
                    print(f"    {d:<14}: {sign}{dd:.1f}pp")

    print(f"\n{'='*W}\n")

    # Scoring method notes
    print("  Scoring methods:")
    print("    math      -- final numeric answer match (+-1% tolerance)")
    print("    code      -- Python syntax check + assertion execution if ground_truth has asserts")
    print("    factual   -- keyword overlap with ground_truth (>=50% -> correct)")
    print("    reasoning -- keyword overlap with expected answer")
    print("    creative  -- non-empty response with >=20 words (no objective ground truth)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Score router vs baseline accuracy")
    parser.add_argument("--router",   help="Router results CSV (load_test output)")
    parser.add_argument("--baseline", help="Baseline results CSV (round_robin output)")
    parser.add_argument("--csv",      help="Any single CSV to score (no comparison)")
    args = parser.parse_args()

    results = []

    if args.router:
        rows = load_csv(args.router)
        results.append(aggregate(rows, label=f"Router ({Path(args.router).stem})"))

    if args.baseline:
        rows = load_csv(args.baseline)
        results.append(aggregate(rows, label=f"Round-Robin ({Path(args.baseline).stem})"))

    if args.csv:
        rows = load_csv(args.csv)
        results.append(aggregate(rows, label=Path(args.csv).stem))

    if not results:
        parser.print_help()
        sys.exit(1)

    print_report(results)


if __name__ == "__main__":
    main()
