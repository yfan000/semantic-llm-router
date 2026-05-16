"""
eval_all_models.py — Pre-evaluation matrix: run every request on every model.

For each request in the dataset, sends it to ALL configured model backends
concurrently and scores each response against ground_truth. Produces an
eval_matrix.csv with one row per (request × model) pair.

This matrix is then used by compare_ttca.py for exact Time-to-Correct-Answer
computation — no median-latency simulation needed.

Usage:
    # Configure BACKENDS below, then run:
    python tests/eval_all_models.py \
        --dataset datasets/hf_1000.json \
        --output  results/eval_matrix.csv \
        --concurrency 10

Output columns:
    req_id, domain, complexity, query, ground_truth,
    model_id, wall_ms, status, response_text, score, is_correct
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
from datetime import datetime
from statistics import mean
from pathlib import Path

import httpx


# ---------------------------------------------------------------------------
# Model backends — edit to match your deployment
# ---------------------------------------------------------------------------

BACKENDS: list[dict] = [
    {
        "model_id":   "qwen2.5-1.5b",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",  # name sent to vLLM
        "base_url":   "http://localhost:8001",
    },
    {
        "model_id":   "qwen2.5-7b",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "base_url":   "http://localhost:8002",
    },
]

CORRECT_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Accuracy scoring (same as score_accuracy.py / compare_ttca.py)
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


def score_response(domain: str, response: str, ground_truth: str):
    if not response:
        return None
    scorer = _SCORERS.get(domain.lower())
    if scorer is None:
        return None
    return scorer(response, ground_truth)


# ---------------------------------------------------------------------------
# Single inference call to one backend
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
            body = resp.json()
            choices = body.get("choices", [])
            response_text = choices[0].get("message", {}).get("content", "") if choices else ""
            result["response_text"] = response_text

            s = score_response(item["domain"], response_text, item.get("ground_truth", ""))
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
# Main runner
# ---------------------------------------------------------------------------

async def run(dataset_path: str, output: str, concurrency: int) -> None:
    with open(dataset_path) as f:
        items = json.load(f)

    n_requests  = len(items)
    n_models    = len(BACKENDS)
    n_total     = n_requests * n_models

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Eval matrix: {n_requests} requests × {n_models} models = {n_total} calls")
    print(f"  Models: {', '.join(b['model_id'] for b in BACKENDS)}")
    print(f"  Concurrency: {concurrency}  |  Output: {output}\n")

    fieldnames = [
        "req_id", "domain", "complexity", "query", "ground_truth",
        "model_id", "wall_ms", "status", "response_text", "score", "is_correct",
    ]

    sem    = asyncio.Semaphore(concurrency)
    done   = 0
    t0_all = time.monotonic()

    async def bounded_call(req_id: int, item: dict, backend: dict, writer, f) -> None:
        nonlocal done
        async with sem:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await call_model(client, req_id, item, backend)
            writer.writerow(r)
            f.flush()
            done += 1
            if done % max(n_total // 20, 1) == 0:
                elapsed = time.monotonic() - t0_all
                bar = "=" * int(done / n_total * 40)
                print(f"\r  [{bar:<40}] {done}/{n_total}  {done/elapsed:.1f} calls/s",
                      end="", flush=True)

    # Build all (req_id, item, backend) tasks
    tasks = []
    for i, item in enumerate(items):
        for backend in BACKENDS:
            tasks.append((i, item, backend))

    results: list[dict] = []

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        await asyncio.gather(*[
            bounded_call(req_id, item, backend, writer, f)
            for req_id, item, backend in tasks
        ])

    elapsed = time.monotonic() - t0_all
    print(f"\r  [{'='*40}] {n_total}/{n_total}  ({elapsed:.1f}s)\n")

    # --- Summary ---
    with open(output, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    ok = [r for r in all_rows if str(r.get("status")) == "200"]
    scorable = [r for r in ok if r.get("is_correct") in ("true", "false")]

    print(f"  Total calls : {n_total}")
    print(f"  Successful  : {len(ok)}")
    print(f"  Scorable    : {len(scorable)}")
    print()

    # Per-model accuracy
    by_model: dict[str, list] = defaultdict(list)
    for r in scorable:
        by_model[r["model_id"]].append(r["is_correct"] == "true")

    print(f"  {'Model':<26} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*56}")
    for model_id, correct_list in sorted(by_model.items()):
        n_correct = sum(correct_list)
        n_tot     = len(correct_list)
        print(f"  {model_id:<26} {n_correct:>8} {n_tot:>8} {n_correct/max(n_tot,1)*100:>9.1f}%")

    # Per-domain accuracy per model
    by_domain_model: dict[tuple, list] = defaultdict(list)
    for r in scorable:
        by_domain_model[(r["domain"], r["model_id"])].append(r["is_correct"] == "true")

    domains = sorted({r["domain"] for r in scorable})
    models  = sorted({r["model_id"] for r in scorable})

    print(f"\n  {'Domain':<14}", end="")
    for m in models:
        print(f"  {m[:20]:>22}", end="")
    print()
    print(f"  {'-'*(14 + 24*len(models))}")
    for domain in domains:
        print(f"  {domain:<14}", end="")
        for m in models:
            lst = by_domain_model.get((domain, m), [])
            if lst:
                acc = sum(lst) / len(lst) * 100
                print(f"  {acc:>21.1f}%", end="")
            else:
                print(f"  {'—':>22}", end="")
        print()

    print(f"\n  Eval matrix saved to: {output}")
    print(f"  Use with: python tests/compare_ttca.py --router results/router.csv "
          f"--baseline results/rr.csv --eval-matrix {output}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-evaluate all requests on all models")
    parser.add_argument("--dataset",     required=True, help="Dataset JSON from build_dataset.py")
    parser.add_argument("--output",      default="results/eval_matrix.csv")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Concurrent calls per model (total = concurrency × n_models)")
    args = parser.parse_args()
    asyncio.run(run(args.dataset, args.output, args.concurrency))


if __name__ == "__main__":
    main()
