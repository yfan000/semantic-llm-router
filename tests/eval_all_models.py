"""
eval_all_models.py — Pre-evaluation matrix.

Runs every request in the dataset through ALL model backends before testing.
Produces eval_matrix.csv with one row per (request x model), containing:
  - actual wall_ms for each model on each request
  - accuracy score and is_correct flag

This matrix is then consumed by compare_ttca.py (--eval-matrix flag) for
exact Time-to-Correct-Answer computation with no median-latency simulation.

Scoring strategy by source:
  LiveCodeBench  : execution-scored (stdin/stdout), ground_truth is JSON array
  SWE-bench      : keyword overlap with patch symbols (source="swe_bench")
  HumanEval/MBPP : execution-scored via assert statements in ground_truth
  MMLU-Pro       : keyword overlap with correct answer text
  GSM8K/MATH     : numeric extraction and comparison
  LogiQA/ARC     : keyword overlap with correct answer text

Workflow:
    1. python tests/eval_all_models.py --dataset datasets/hf_3000.json
    2. python tests/load_test.py       --dataset datasets/hf_3000.json ...
    3. python tests/round_robin_test.py --dataset datasets/hf_3000.json ...
    4. python tests/compare_ttca.py --router ... --baseline ... --eval-matrix results/eval_matrix.csv

Usage:
    python tests/eval_all_models.py \\
        --dataset     datasets/hf_3000.json \\
        --output      results/eval_matrix.csv \\
        --concurrency 30
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import httpx


# ---------------------------------------------------------------------------
# Model backends — must match what is registered with the router
# ---------------------------------------------------------------------------

BACKENDS: list[dict] = [
    {
        "model_id":   "qwen-7b",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "base_url":   "http://localhost:8000",
    },
    {
        "model_id":   "deepseek-r1-7b",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "base_url":   "http://localhost:8001",
    },
    {
        "model_id":   "qwen3-coder-30b",
        "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "base_url":   "http://localhost:8002",
    },
    {
        "model_id":   "gemma-3-27b",
        "model_name": "google/gemma-3-27b-it",
        "base_url":   "http://localhost:8003",
    },
    {
        "model_id":   "deepseek-r1-14b",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "base_url":   "http://localhost:8004",
    },
    # llama4-scout added dynamically via --node2-host (lives on node 2, port 8005)
]

CORRECT_THRESHOLD = 0.7


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
    if "assert" not in str(gt) and "==" not in str(gt):
        return 0.5  # syntax ok but no test to run
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code + "\n" + str(gt))
        fname = f.name
    try:
        r = subprocess.run([sys.executable, fname], timeout=5, capture_output=True)
        return 1.0 if r.returncode == 0 else 0.0
    except Exception:
        return 0.0
    finally:
        try:
            import os as _os; _os.unlink(fname)
        except Exception:
            pass


def _score_livecodebench(response: str, test_cases_json: str) -> float:
    """Score a LiveCodeBench response by executing against public test cases.

    Strategy:
      1. Extract code block from model output.
      2. For each test case: run the code as a script with stdin=test_input,
         compare stdout to expected output (stdin/stdout style).
      3. Return fraction of passing test cases (0.0 - 1.0).
    """
    try:
        test_cases = json.loads(test_cases_json)
    except Exception:
        return _score_code(response, test_cases_json)

    if not test_cases:
        return _score_code(response, "")

    blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    code = blocks[0].strip() if blocks else response.strip()

    try:
        ast.parse(code)
    except SyntaxError:
        return 0.0

    passes = 0
    sample = test_cases[:3]  # cap at 3 test cases to limit eval time
    for tc in sample:
        inp      = str(tc.get("input",  "")).strip()
        expected = str(tc.get("output", "")).strip()
        fname = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(code)
                fname = f.name
            result = subprocess.run(
                [sys.executable, fname],
                input=inp, capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip() == expected:
                passes += 1
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        finally:
            if fname:
                try:
                    import os as _os; _os.unlink(fname)
                except Exception:
                    pass

    return passes / len(sample)


def _score_keyword(response: str, gt: str):
    if not gt or str(gt).strip() in ("", "None"):
        return None
    words = set(w.lower() for w in re.findall(r"\b\w{4,}\b", str(gt)))
    if not words:
        return None
    hits = sum(1 for w in words if w in response.lower())
    overlap = hits / len(words)
    return 1.0 if overlap >= 1.0 else overlap  # 100% keyword match required


_SCORERS = {
    "math":      _score_math,
    "code":      _score_code,
    "factual":   _score_keyword,
    "reasoning": _score_keyword,
}


def score_response(domain: str, response: str, gt: str, source: str = "") -> float | None:
    if not response:
        return None
    # LiveCodeBench: ground_truth is a JSON array of test cases -> execution scorer
    if domain.lower() == "code" and gt.lstrip().startswith("[{"):
        return _score_livecodebench(response, gt)
    # SWE-bench: ground_truth is patch symbol keywords -> keyword overlap scorer.
    # _score_code() would try ast.parse() on a natural-language response and
    # return 0.0 for everything, making all models look equally bad on SWE-bench.
    if source == "swe_bench":
        return _score_keyword(response, gt)
    scorer = _SCORERS.get(domain.lower())
    return scorer(response, gt) if scorer else None


# ---------------------------------------------------------------------------
# Call one model for one request
# ---------------------------------------------------------------------------

async def call_model(
    client: httpx.AsyncClient,
    req_id: int,
    item: dict,
    backend: dict,
) -> dict:
    result = {
        "req_id":        req_id,
        "domain":        item["domain"],
        "complexity":    item["complexity"],
        "query":         item["query"][:120],
        "ground_truth":  str(item.get("ground_truth", "")),
        "model_id":      backend["model_id"],
        "wall_ms":       "",
        "status":        "",
        "response_text": "",
        "score":         "",
        "is_correct":    "",
        "error":         "",
    }

    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{backend['base_url']}/v1/chat/completions",
            json={
                "model":    backend.get("model_name", backend["model_id"]),
                "messages": [{"role": "user", "content": item["query"]}],
            },
            timeout=120.0,
        )
        wall_ms = int((time.monotonic() - t0) * 1000)
        result["wall_ms"] = wall_ms
        result["status"]  = resp.status_code

        if resp.status_code == 200:
            body    = resp.json()
            choices = body.get("choices", [])
            text    = choices[0].get("message", {}).get("content", "") if choices else ""
            result["response_text"] = text

            s = score_response(item["domain"], text, item.get("ground_truth", ""),
                               item.get("source", ""))
            if s is not None:
                result["score"]      = f"{s:.4f}"
                result["is_correct"] = "true" if s >= CORRECT_THRESHOLD else "false"
        else:
            result["error"] = str(resp.status_code)

    except Exception as e:
        result["wall_ms"] = int((time.monotonic() - t0) * 1000)
        result["status"]  = "error"
        result["error"]   = str(e)[:200]

    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run(dataset_path: str, output: str, concurrency: int) -> None:
    with open(dataset_path) as f:
        items = json.load(f)

    n_req   = len(items)
    n_mod   = len(BACKENDS)
    n_total = n_req * n_mod

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # Show source breakdown
    from collections import Counter
    sources = Counter(item.get("source", "hf") for item in items)

    print(f"\n  Pre-evaluation: {n_req} requests x {n_mod} models = {n_total} calls")
    print(f"  Models      : {', '.join(b['model_id'] for b in BACKENDS)}")
    print(f"  Concurrency : {concurrency}  (total simultaneous calls)")
    print(f"  Sources     : {dict(sources)}")
    print(f"  Output      : {output}\n")

    fieldnames = [
        "req_id", "domain", "complexity", "query", "ground_truth",
        "model_id", "wall_ms", "status", "response_text", "score", "is_correct", "error",
    ]

    sem  = asyncio.Semaphore(concurrency)
    done = 0
    t0   = time.monotonic()

    async def bounded(req_id: int, item: dict, backend: dict, writer, f) -> None:
        nonlocal done
        async with sem:
            async with httpx.AsyncClient(trust_env=False) as client:
                r = await call_model(client, req_id, item, backend)
            writer.writerow(r)
            f.flush()
            done += 1
            if done % max(n_total // 20, 1) == 0:
                elapsed = time.monotonic() - t0
                bar = "=" * int(done / n_total * 40)
                print(f"\r  [{bar:<40}] {done}/{n_total}  {done/elapsed:.1f} calls/s",
                      end="", flush=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        await asyncio.gather(*[
            bounded(i, items[i], backend, writer, f)
            for i in range(n_req)
            for backend in BACKENDS
        ])

    elapsed = time.monotonic() - t0
    print(f"\r  [{'='*40}] {n_total}/{n_total}  ({elapsed:.1f}s total)\n")

    # --- Summary ---
    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok       = [r for r in rows if str(r.get("status")) == "200"]
    scorable = [r for r in ok   if r.get("is_correct") in ("true", "false")]

    print(f"  Calls       : {n_total}")
    print(f"  Successful  : {len(ok)}")
    print(f"  Scorable    : {len(scorable)}")

    # Per-model accuracy
    by_model: dict[str, list[bool]] = defaultdict(list)
    for r in scorable:
        by_model[r["model_id"]].append(r["is_correct"] == "true")

    print(f"\n  {'Model':<28} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*58}")
    for mid, correct_list in sorted(by_model.items()):
        n_c = sum(correct_list)
        n_t = len(correct_list)
        print(f"  {mid:<28} {n_c:>8} {n_t:>8} {n_c/max(n_t,1)*100:>9.1f}%")

    # Per-domain breakdown
    by_dm: dict[tuple, list[bool]] = defaultdict(list)
    for r in scorable:
        by_dm[(r["domain"], r["model_id"])].append(r["is_correct"] == "true")

    domains = sorted({r["domain"] for r in scorable})
    models  = sorted({r["model_id"] for r in scorable})

    print(f"\n  {'Domain':<14}", end="")
    for m in models:
        print(f"  {m[:22]:>24}", end="")
    print()
    print(f"  {'-'*(14 + 26 * len(models))}")
    for domain in domains:
        print(f"  {domain:<14}", end="")
        for m in models:
            lst = by_dm.get((domain, m), [])
            cell = f"{sum(lst)/len(lst)*100:.1f}%" if lst else "-"
            print(f"  {cell:>24}", end="")
        print()

    print(f"\n  Saved: {output}")
    print(f"\n  Next steps:")
    print(f"    python tests/load_test.py --dataset {dataset_path} --mode accuracy \\")
    print(f"        --output results/router_accuracy.csv")
    print(f"    python tests/round_robin_test.py --dataset {dataset_path} \\")
    print(f"        --output results/rr_baseline.csv")
    print(f"    python tests/compare_ttca.py \\")
    print(f"        --router results/router_accuracy.csv \\")
    print(f"        --baseline results/rr_baseline.csv \\")
    print(f"        --eval-matrix {output}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--output",      default="results/eval_matrix.csv")
    parser.add_argument("--concurrency", type=int, default=30,
                        help="Max simultaneous calls across all models")
    parser.add_argument("--node2-host",  default=None,
                        help="Hostname of node 2 running llama4-scout (e.g. sophia-gpu-09).")
    parser.add_argument("--model",       default=None,
                        help="Evaluate only this model_id (e.g. qwen3-coder-30b). "
                             "Useful for adding priors for a single new model without "
                             "re-running the full 6-model eval matrix.")
    args = parser.parse_args()

    if args.node2_host:
        BACKENDS.append({
            "model_id":   "llama4-scout",
            "model_name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "base_url":   f"http://{args.node2_host}:8005",
        })

    # Filter to a single model when --model is specified
    if args.model:
        filtered = [b for b in BACKENDS if b["model_id"] == args.model]
        if not filtered:
            valid = ", ".join(b["model_id"] for b in BACKENDS)
            print(f"ERROR: model '{args.model}' not found. Valid: {valid}")
            raise SystemExit(1)
        BACKENDS[:] = filtered
        print(f"  [--model] Evaluating only: {args.model}")

    asyncio.run(run(args.dataset, args.output, args.concurrency))


if __name__ == "__main__":
    main()
