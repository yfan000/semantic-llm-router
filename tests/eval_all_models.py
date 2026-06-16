"""eval_all_models.py -- Pre-evaluation matrix.

Runs every request through ALL model backends and produces eval_matrix.csv.

Usage:
    python tests/eval_all_models.py --dataset datasets/hf_1000.json
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


BACKENDS: list[dict] = [
    {"model_id": "qwen-7b",          "model_name": "Qwen/Qwen2.5-7B-Instruct",                  "base_url": "http://localhost:8000"},
    {"model_id": "deepseek-r1-7b",   "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   "base_url": "http://localhost:8001"},
    {"model_id": "qwen3-coder-30b",  "model_name": "Qwen/Qwen3-Coder-30B-A3B",                  "base_url": "http://localhost:8002"},
    {"model_id": "gemma-3-27b",      "model_name": "google/gemma-3-27b-it",                      "base_url": "http://localhost:8003"},
    {"model_id": "deepseek-r1-14b",  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  "base_url": "http://localhost:8004"},
]

CORRECT_THRESHOLD = 0.7


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
        return 0.5
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
    sample = test_cases[:3]
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
    return 1.0 if overlap >= 1.0 else overlap


_SCORERS = {
    "math":      _score_math,
    "code":      _score_code,
    "factual":   _score_keyword,
    "reasoning": _score_keyword,
}


def score_response(domain: str, response: str, gt: str):
    if not response:
        return None
    if domain.lower() == "code" and gt.lstrip().startswith("[{"):
        return _score_livecodebench(response, gt)
    scorer = _SCORERS.get(domain.lower())
    return scorer(response, gt) if scorer else None


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
            s = score_response(item["domain"], text, item.get("ground_truth", ""))
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


async def run(dataset_path: str, output: str, concurrency: int) -> None:
    with open(dataset_path) as f:
        items = json.load(f)
    n_req   = len(items)
    n_mod   = len(BACKENDS)
    n_total = n_req * n_mod
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  Pre-evaluation: {n_req} requests x {n_mod} models = {n_total} calls")
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
    print(f"  Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--output",      default="results/eval_matrix.csv")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--node2-host",  default=None)
    args = parser.parse_args()
    if args.node2_host:
        BACKENDS.append({
            "model_id":   "llama4-scout",
            "model_name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "base_url":   f"http://{args.node2_host}:8005",
        })
    asyncio.run(run(args.dataset, args.output, args.concurrency))


if __name__ == "__main__":
    main()
