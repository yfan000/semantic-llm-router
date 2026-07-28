"""
run_benchmark_eval.py — End-to-end benchmark evaluation for the semantic router.

Loads a dataset built by build_benchmark_dataset.py, sends each query to the
router asynchronously, scores responses using benchmark-appropriate methods,
and reports accuracy by source, domain, complexity, and model.

Scoring by answer_type:
  mcq        — extract letter (A-J) from response, exact match vs GT letter
  numeric    — extract last number, compare ±1% tolerance (grade-school math)
  expression — LaTeX/symbolic; numeric fallback then normalized string match
  code       — stdin/stdout execution against LiveCodeBench test cases

Usage:
    # Build dataset then evaluate
    python tests/run_benchmark_eval.py \\
        --build --total 1000 \\
        --router http://localhost:8080 \\
        --mode ttca \\
        --output results/benchmark_eval.csv

    # Evaluate an existing dataset
    python tests/run_benchmark_eval.py \\
        --dataset datasets/benchmark_1000.json \\
        --router  http://localhost:8080 \\
        --mode    ttca \\
        --output  results/benchmark_eval.csv

    # Quick smoke test (first 50 items)
    python tests/run_benchmark_eval.py \\
        --dataset datasets/benchmark_1000.json \\
        --router  http://localhost:8080 \\
        --n 50

    # Compare two router modes side-by-side (score existing CSVs)
    python tests/run_benchmark_eval.py \\
        --score-only results/ttca_eval.csv results/carrot_eval.csv
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text)]


def score_mcq(response: str, ground_truth: str) -> Optional[float]:
    """Extract the letter choice from response text, exact match vs GT letter.

    Checks high-confidence patterns first (explicit "answer is A", bold **A**,
    "Answer: A") and falls back to scanning the response tail for a bare letter
    only when nothing explicit is found.
    Returns None when no letter can be extracted (e.g. refusal or empty).
    """
    gt = ground_truth.strip().upper()
    if not gt or gt[0] not in "ABCDEFGHIJ":
        return None
    gt = gt[0]

    resp_upper = response.upper()

    # High-confidence patterns — explicit answer markers
    hi_patterns = [
        r"(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s+IS\s*[:(]?\s*\*?\*?([A-J])\*?\*?",
        r"ANSWER:\s*\*?\*?([A-J])\*?\*?",
        r"\*\*([A-J])\*\*",
        r"^([A-J])[.):\s]",
        r"OPTION\s+([A-J])\b",
        r"CHOICE\s+([A-J])\b",
        r"\(([A-J])\)\s+IS\s+CORRECT",
        r"SELECT\s+([A-J])\b",
        r"RESPONSE:\s*([A-J])\b",
        r"MY\s+ANSWER\s+IS\s+([A-J])\b",
    ]

    found = []
    for pat in hi_patterns:
        for m in re.finditer(pat, resp_upper, re.MULTILINE):
            found.append(m.group(1))

    if found:
        from collections import Counter
        winner = Counter(found).most_common(1)[0][0]
        return 1.0 if winner == gt else 0.0

    # Fallback: look for a lone letter in the last 300 chars
    tail = resp_upper[-300:]
    for m in re.finditer(r"\b([A-J])\b", tail):
        return 1.0 if m.group(1) == gt else 0.0

    return None  # could not extract a letter


def score_numeric(response: str, ground_truth: str) -> Optional[float]:
    """Extract the last number from response, compare ±1% or absolute ≤0.01.

    Suitable for GSM1K and other grade-school numeric math benchmarks.
    Returns None when no number is found in either response or ground truth.
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


def score_expression(response: str, ground_truth: str) -> Optional[float]:
    """Score symbolic/LaTeX answers (OlympiadBench).

    Strategy:
    1. Extract \\boxed{} from the response (models are prompted to use it).
    2. Try numeric comparison with 2% tolerance.
    3. Normalize both sides (strip whitespace, LaTeX macros) and string-compare.
    4. Return 0.0 on definitive mismatch, None if the response has no answer signal.
    """
    # Extract \boxed{...} — models are instructed to put answers there
    boxed = re.search(r"\\boxed\{([^}]+)\}", response)
    pred_str = boxed.group(1).strip() if boxed else ""

    if not pred_str:
        # Try last math environment or final number
        last_env = re.search(r"\$([^$]+)\$", response)
        pred_str = last_env.group(1).strip() if last_env else ""

    if not pred_str:
        # No clear answer marker — fall back to last number in response
        nums = _extract_numbers(response)
        if not nums:
            return None
        pred_str = str(nums[-1])

    gt_str = str(ground_truth).strip()

    # Numeric comparison (2% tolerance for Olympiad problems)
    pred_nums = _extract_numbers(pred_str)
    true_nums = _extract_numbers(gt_str)
    if pred_nums and true_nums:
        pred = pred_nums[-1]
        true = true_nums[-1]
        if true == 0:
            return 1.0 if abs(pred) < 0.01 else 0.0
        if abs(pred - true) / abs(true) < 0.02:
            return 1.0

    # Normalized string comparison
    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"\s+", "", s)
        replacements = [
            ("\\cdot",  "*"), ("\\times", "*"), ("\\div", "/"),
            ("\\frac",  ""), ("\\sqrt",   "sqrt"),
            ("{", ""),  ("}", ""),  ("^", "**"), ("\\pm", "±"),
        ]
        for old, new in replacements:
            s = s.replace(old, new)
        return s

    if _norm(pred_str) and _norm(gt_str) and _norm(pred_str) == _norm(gt_str):
        return 1.0

    return 0.0


def score_code_exec(response: str, ground_truth: str) -> Optional[float]:
    """Score code via syntax check + stdin/stdout execution (LiveCodeBench).

    Runs up to 5 public test cases from the JSON ground_truth.
    Returns fraction of test cases passed (0.0–1.0).
    Returns 0.5 for syntactically valid code with no executable tests.
    """
    # Extract code from markdown fences
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    code = code_blocks[0].strip() if code_blocks else response.strip()

    # Syntax check
    try:
        ast.parse(code)
    except SyntaxError:
        return 0.0

    # Try to parse test cases as JSON [{input, output}, ...]
    try:
        test_cases = json.loads(ground_truth)
    except Exception:
        # Fallback: assert-style ground truth (HumanEval/MBPP)
        gt = str(ground_truth)
        if "assert" in gt:
            test_src = code + "\n" + gt
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(test_src)
                fname = f.name
            try:
                res = subprocess.run([sys.executable, fname],
                                     timeout=5, capture_output=True)
                return 1.0 if res.returncode == 0 else 0.0
            except Exception:
                return 0.0
        return 0.5  # no tests — syntax valid, partial credit

    if not test_cases:
        return 0.5

    # Stdin/stdout execution against each test case (cap at 5)
    passed = 0
    total  = min(len(test_cases), 5)
    for tc in test_cases[:total]:
        stdin_data = str(tc.get("input",  "")).strip()
        expected   = str(tc.get("output", "")).strip()

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            res = subprocess.run(
                [sys.executable, fname],
                input=stdin_data, text=True, capture_output=True, timeout=5,
            )
            got = res.stdout.strip()
            if got == expected:
                passed += 1
        except Exception:
            pass

    return passed / total


# Map answer_type → scorer function
SCORERS = {
    "mcq":        score_mcq,
    "numeric":    score_numeric,
    "expression": score_expression,
    "code":       score_code_exec,
}

# Fallback by domain when answer_type is missing
_DOMAIN_FALLBACK = {
    "math":      score_numeric,
    "reasoning": score_mcq,
    "factual":   score_mcq,
    "code":      score_code_exec,
}


def score_item(response: str, item: dict) -> Optional[float]:
    scorer = SCORERS.get(item.get("answer_type", ""))
    if scorer is None:
        scorer = _DOMAIN_FALLBACK.get(item.get("domain", ""))
    if scorer is None:
        return None
    return scorer(response, item.get("ground_truth", ""))


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "req_id", "source", "domain", "complexity", "answer_type",
    "query", "ground_truth",
    "status", "model_winner", "actual_latency_ms", "wall_ms",
    "charged_usd", "response_text",
    "score", "is_correct", "error",
]


# ---------------------------------------------------------------------------
# Async router client
# ---------------------------------------------------------------------------

async def _send_one(
    client,
    router_url: str,
    sem: asyncio.Semaphore,
    item: dict,
    mode: str,
    idx: int,
    total: int,
    counter: list,
) -> dict:
    async with sem:
        t0 = time.monotonic()
        row: dict = {
            "req_id":            item.get("req_id", idx),
            "source":            item.get("source", ""),
            "domain":            item.get("domain", ""),
            "complexity":        item.get("complexity", ""),
            "answer_type":       item.get("answer_type", ""),
            "query":             item.get("query", "")[:300],
            "ground_truth":      item.get("ground_truth", "")[:100],
            "status":            "",
            "model_winner":      "",
            "actual_latency_ms": "",
            "wall_ms":           "",
            "charged_usd":       "",
            "response_text":     "",
            "score":             "",
            "is_correct":        "",
            "error":             "",
        }
        try:
            payload = {
                "model":    "auto",
                "messages": [{"role": "user", "content": item["query"]}],
                "max_tokens": 1024,
                "extra_body": {"router": {"mode": mode}},
            }
            resp = await client.post(
                f"{router_url}/v1/chat/completions",
                json=payload,
                timeout=90.0,
            )
            wall_ms = (time.monotonic() - t0) * 1000
            row["status"]            = str(resp.status_code)
            row["wall_ms"]           = f"{wall_ms:.1f}"
            row["actual_latency_ms"] = resp.headers.get("X-Router-Actual-Latency-Ms", "")
            row["model_winner"]      = resp.headers.get("X-Router-Model-Winner", "")
            row["charged_usd"]       = resp.headers.get("X-Router-Charged-USD", "")

            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                row["response_text"] = text

                sc = score_item(text, item)
                if sc is not None:
                    row["score"]      = f"{sc:.4f}"
                    row["is_correct"] = "true" if sc >= 0.9 else "false"
        except Exception as exc:
            row["status"] = "error"
            row["error"]  = str(exc)[:300]

        counter[0] += 1
        done = counter[0]
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total} requests done", flush=True)

        return row


async def run_eval(
    router_url: str,
    dataset: list[dict],
    mode: str,
    output: str,
    concurrency: int,
) -> list[dict]:
    try:
        import httpx
    except ImportError:
        print("httpx not installed. Run: pip install httpx")
        sys.exit(1)

    sem     = asyncio.Semaphore(concurrency)
    counter = [0]

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    results = []
    with open(output, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=FIELDNAMES)
        writer.writeheader()

        async with httpx.AsyncClient() as client:
            tasks = [
                _send_one(client, router_url, sem, item, mode, i, len(dataset), counter)
                for i, item in enumerate(dataset)
            ]
            for coro in asyncio.as_completed(tasks):
                row = await coro
                writer.writerow(row)
                csvf.flush()
                results.append(row)

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _acc(rows: list[dict]) -> Optional[float]:
    scored = [r for r in rows if r.get("is_correct") in ("true", "false")]
    if not scored:
        return None
    correct = sum(1 for r in scored if r["is_correct"] == "true")
    return correct / len(scored)


def _mean_lat(rows: list[dict]) -> Optional[float]:
    lats = []
    for r in rows:
        try:
            lats.append(float(r.get("wall_ms", "")))
        except (ValueError, TypeError):
            pass
    return sum(lats) / len(lats) if lats else None


def print_report(results: list[dict], label: str = "") -> None:
    W = 82
    title = f"BENCHMARK EVALUATION REPORT{' — ' + label if label else ''}"
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")

    n200   = sum(1 for r in results if r.get("status") == "200")
    scored = [r for r in results if r.get("is_correct") in ("true", "false")]
    acc    = _acc(scored)
    lat    = _mean_lat(results)

    print(f"\n  Requests total  : {len(results)}")
    print(f"  HTTP 200        : {n200}")
    print(f"  Scored          : {len(scored)}")
    if acc is not None:
        print(f"  Overall accuracy: {acc*100:.1f}%")
    if lat is not None:
        print(f"  Mean wall lat   : {lat:.0f} ms")

    # By source
    print(f"\n  {'Source':<22} {'N':>5} {'Scored':>7} {'Acc':>8} {'Lat(ms)':>9}")
    print(f"  {'-'*54}")
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_src[r.get("source", "?")].append(r)
    for src in sorted(by_src):
        rows  = by_src[src]
        sc    = [r for r in rows if r.get("is_correct") in ("true", "false")]
        a     = _acc(sc)
        l     = _mean_lat(rows)
        a_s   = f"{a*100:.1f}%" if a is not None else "—"
        l_s   = f"{l:.0f}"     if l is not None else "—"
        print(f"  {src:<22} {len(rows):>5} {len(sc):>7} {a_s:>8} {l_s:>9}")

    # By domain / complexity
    print(f"\n  {'Domain':<12} {'Complexity':<10} {'N':>5} {'Scored':>7} {'Acc':>8}")
    print(f"  {'-'*45}")
    by_dc: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        by_dc[(r.get("domain", "?"), r.get("complexity", "?"))].append(r)
    for (dom, cpx) in sorted(by_dc):
        rows = by_dc[(dom, cpx)]
        sc   = [r for r in rows if r.get("is_correct") in ("true", "false")]
        a    = _acc(sc)
        a_s  = f"{a*100:.1f}%" if a is not None else "—"
        print(f"  {dom:<12} {cpx:<10} {len(rows):>5} {len(sc):>7} {a_s:>8}")

    # By model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        m = r.get("model_winner", "").strip()
        if m:
            by_model[m].append(r)
    if by_model:
        print(f"\n  {'Model':<36} {'N':>5} {'Acc':>8}")
        print(f"  {'-'*52}")
        for model in sorted(by_model, key=lambda m: -len(by_model[m])):
            rows = by_model[model]
            sc   = [r for r in rows if r.get("is_correct") in ("true", "false")]
            a    = _acc(sc)
            a_s  = f"{a*100:.1f}%" if a is not None else "—"
            short = model.split("/")[-1][:36]
            print(f"  {short:<36} {len(rows):>5} {a_s:>8}")

    # Scoring method legend
    print(f"\n  Scoring methods:")
    print(f"    mcq        — letter extraction (A-J), exact match")
    print(f"    numeric    — last-number extraction, ±1% tolerance")
    print(f"    expression — \\boxed{{}} extraction, numeric + string normalization")
    print(f"    code       — stdin/stdout execution against test cases (pass rate)")
    print(f"\n{'='*W}\n")


def print_comparison(csv_paths: list[str]) -> None:
    """Load multiple scored CSVs and print a side-by-side accuracy table."""
    all_data = []
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        all_data.append((Path(path).stem, rows))

    W = 82
    print(f"\n{'='*W}")
    print("  BENCHMARK COMPARISON")
    print(f"{'='*W}\n")

    labels = [name for name, _ in all_data]

    # Header
    print(f"  {'Benchmark':<22}", end="")
    for lbl in labels:
        print(f"  {lbl[:18]:>18}", end="")
    print()
    print(f"  {'-'*(22 + 20*len(labels))}")

    # Gather all sources
    sources = []
    for _, rows in all_data:
        for r in rows:
            s = r.get("source", "?")
            if s not in sources:
                sources.append(s)

    for src in sources:
        print(f"  {src:<22}", end="")
        for _, rows in all_data:
            subset = [r for r in rows if r.get("source") == src]
            sc     = [r for r in subset if r.get("is_correct") in ("true", "false")]
            a      = _acc(sc)
            a_s    = f"{a*100:.1f}%" if a is not None else "—"
            print(f"  {a_s:>18}", end="")
        print()

    # Overall row
    print(f"  {'OVERALL':<22}", end="")
    for _, rows in all_data:
        sc = [r for r in rows if r.get("is_correct") in ("true", "false")]
        a  = _acc(sc)
        a_s = f"{a*100:.1f}%" if a is not None else "—"
        print(f"  {a_s:>18}", end="")
    print()
    print(f"\n{'='*W}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation for the semantic LLM router."
    )
    # Dataset
    parser.add_argument("--dataset",     default="datasets/benchmark_1000.json",
                        help="Path to benchmark JSON (build_benchmark_dataset.py output)")
    parser.add_argument("--build",       action="store_true",
                        help="Build the dataset before running (calls build_benchmark_dataset.py)")
    parser.add_argument("--total",       type=int, default=1000,
                        help="Dataset size when --build is used")
    parser.add_argument("--cutoff",      default="2024-06-01",
                        help="Contamination cutoff when --build is used")
    parser.add_argument("--seed",        type=int, default=42)
    # Router
    parser.add_argument("--router",      default="http://localhost:8080")
    parser.add_argument("--mode",        default="ttca",
                        choices=["ttca", "accuracy", "cost", "eco", "carrot"],
                        help="Router routing mode")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--n",           type=int, default=None,
                        help="Limit to first N items (smoke test)")
    # Output
    parser.add_argument("--output",      default="results/benchmark_eval.csv")
    # Comparison mode
    parser.add_argument("--score-only",  nargs="+", metavar="CSV",
                        help="Skip eval; compare already-scored CSVs side-by-side")
    args = parser.parse_args()

    # Comparison-only mode
    if args.score_only:
        print_comparison(args.score_only)
        return

    import random
    random.seed(args.seed)

    # Optionally build dataset
    if args.build:
        sys.path.insert(0, str(Path(__file__).parent))
        from build_benchmark_dataset import build
        build(args.total, args.dataset, cutoff=args.cutoff)

    # Load dataset
    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}")
        print("Run with --build to create it, or specify an existing --dataset path.")
        sys.exit(1)

    with open(args.dataset) as f:
        dataset = json.load(f)

    if args.n is not None:
        dataset = dataset[:args.n]

    # Print plan
    by_src: dict[str, int] = defaultdict(int)
    for item in dataset:
        by_src[item.get("source", "?")] += 1
    print(f"\nRunning benchmark eval:")
    print(f"  Dataset  : {args.dataset} ({len(dataset)} items)")
    for src, cnt in sorted(by_src.items()):
        print(f"    {src:<22} {cnt}")
    print(f"  Router   : {args.router}")
    print(f"  Mode     : {args.mode}")
    print(f"  Output   : {args.output}")
    print(f"  Workers  : {args.concurrency}\n")

    results = asyncio.run(run_eval(
        router_url=args.router,
        dataset=dataset,
        mode=args.mode,
        output=args.output,
        concurrency=args.concurrency,
    ))

    print_report(results, label=f"{args.mode} @ {args.router}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
