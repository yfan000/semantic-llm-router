"""
round_robin_test.py — Baseline load test using round-robin model selection.

Sends requests alternately to two vLLM backends, bypassing the semantic router.
Produces a CSV in the same format as load_test.py so compare_results.py
can compare them side by side.

Usage:
    python tests/round_robin_test.py --requests 1000 --concurrency 50 \
        --output results/round_robin.csv

    python tests/round_robin_test.py --dataset datasets/hf_1000.json \
        --concurrency 50 --output results/round_robin.csv
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import itertools
import json
import os
import random
import time
from datetime import datetime
from statistics import mean

import httpx

BACKENDS = [
    {
        "model_id":                    "qwen2.5-1.5b",
        "base_url":                    "http://localhost:8001",
        "input_rate_usd_per_token":    0.0000001,
        "output_rate_usd_per_token":   0.0000002,
        "efficiency_tokens_per_joule": 26.7,
    },
    {
        "model_id":                    "phi-3.5-mini",
        "base_url":                    "http://localhost:8002",
        "input_rate_usd_per_token":    0.0000002,
        "output_rate_usd_per_token":   0.0000004,
        "efficiency_tokens_per_joule": 20.0,
    },
]

OUTPUT_TOKENS = {
    ("factual","easy"):80,   ("factual","medium"):200,   ("factual","hard"):350,
    ("math","easy"):120,     ("math","medium"):280,       ("math","hard"):450,
    ("code","easy"):150,     ("code","medium"):350,       ("code","hard"):650,
    ("creative","easy"):250, ("creative","medium"):500,   ("creative","hard"):800,
    ("reasoning","easy"):180,("reasoning","medium"):380,  ("reasoning","hard"):600,
}

QUERIES = [
    ("factual","easy","What is the capital of France?"),
    ("factual","easy","Who invented the telephone?"),
    ("factual","easy","What is the chemical symbol for gold?"),
    ("factual","medium","Explain the difference between DNA and RNA."),
    ("factual","medium","What caused the fall of the Roman Empire?"),
    ("factual","hard","Explain the geopolitical implications of the Bretton Woods collapse."),
    ("math","easy","What is 15% of 240?"),
    ("math","easy","What is the square root of 144?"),
    ("math","medium","Solve the quadratic equation x squared minus 5x plus 6 equals 0."),
    ("math","medium","What is the probability of rolling two sixes in a row?"),
    ("math","hard","Prove that there are infinitely many prime numbers."),
    ("math","hard","Solve the differential equation dy/dx = 2xy with y(0) = 1."),
    ("code","easy","Write a Python function to reverse a string."),
    ("code","easy","Write a Python function to check if a number is even."),
    ("code","medium","Implement a binary search algorithm in Python."),
    ("code","medium","Write a Python decorator that measures execution time."),
    ("code","hard","Implement a thread-safe singleton pattern in Python."),
    ("reasoning","easy","Alice is taller than Bob. Bob is taller than Carol. Who is shortest?"),
    ("reasoning","easy","All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?"),
    ("reasoning","medium","Compare and contrast SQL and NoSQL databases for a high-traffic web app."),
    ("reasoning","medium","What are the trade-offs between microservices and monolithic architecture?"),
    ("reasoning","hard","Analyse the game-theoretic implications of the prisoner's dilemma for climate agreements."),
    ("creative","easy","Write a haiku about the ocean."),
    ("creative","medium","Write a short story about a robot learning to paint."),
    ("creative","hard","Write a philosophical dialogue between Socrates and a modern AI."),
]


def build_request_list(n: int, dataset_path: str | None) -> list[dict]:
    if dataset_path:
        with open(dataset_path) as f:
            items = json.load(f)
        random.shuffle(items)
        if n > len(items):
            items += random.choices(items, k=n - len(items))
        return items[:n]
    pool = [{"domain": d, "complexity": c, "query": q} for d, c, q in QUERIES]
    random.shuffle(pool)
    if n > len(pool):
        pool += random.choices(pool, k=n - len(pool))
    return pool[:n]


async def send_request(client: httpx.AsyncClient, req_id: int, item: dict, backend: dict) -> dict:
    input_tokens = len(item["query"].split()) * 1.3
    output_est   = OUTPUT_TOKENS.get((item["domain"], item["complexity"]), 300)

    result = {
        "req_id": req_id, "domain": item["domain"], "complexity": item["complexity"],
        "query": item["query"][:100], "mode": "round_robin", "status": "",
        "model_winner": backend["model_id"], "bid_latency_ms": "", "actual_latency_ms": "",
        "ttft_ms": "", "output_tokens": "",
        "charged_usd": f"{input_tokens*backend['input_rate_usd_per_token']+output_est*backend['output_rate_usd_per_token']:.8f}",
        "energy_j": "", "load": "", "wall_ms": "", "error": "",
    }

    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{backend['base_url']}/v1/chat/completions",
            json={"model": backend["model_id"],
                  "messages": [{"role": "user", "content": item["query"]}]},
        )
        wall_ms = int((time.monotonic() - t0) * 1000)
        result.update({"wall_ms": wall_ms, "actual_latency_ms": wall_ms,
                        "ttft_ms": wall_ms, "status": resp.status_code})
        if resp.status_code == 200:
            body = resp.json()
            out  = body.get("usage", {}).get("completion_tokens", output_est)
            cost = input_tokens * backend["input_rate_usd_per_token"] + out * backend["output_rate_usd_per_token"]
            result.update({
                "output_tokens": out,
                "energy_j":      f"{out / backend['efficiency_tokens_per_joule']:.3f}",
                "charged_usd":   f"{cost:.8f}",
            })
        else:
            result["error"] = str(resp.status_code)
    except Exception as e:
        result.update({"wall_ms": int((time.monotonic()-t0)*1000), "status": "error", "error": str(e)})
    return result


async def run(n_requests: int, concurrency: int, output: str, dataset_path: str | None) -> None:
    items       = build_request_list(n_requests, dataset_path)
    assignments = list(itertools.islice(itertools.cycle(BACKENDS), n_requests))
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    fieldnames = ["req_id","domain","complexity","query","mode","status","model_winner",
                  "bid_latency_ms","actual_latency_ms","ttft_ms","output_tokens",
                  "charged_usd","energy_j","load","wall_ms","error"]

    results: list[dict] = []
    sem = asyncio.Semaphore(concurrency)
    done = 0
    t0 = time.monotonic()

    print(f"\n  [Round-Robin] {n_requests} requests, concurrency={concurrency}")
    print(f"  Alternating: {' | '.join(b['model_id'] for b in BACKENDS)}")
    print(f"  Output: {output}\n")

    async def bounded(req_id, item, backend, writer, f):
        nonlocal done
        async with sem:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await send_request(client, req_id, item, backend)
            results.append(r); writer.writerow(r); f.flush()
            done += 1
            if done % max(n_requests // 20, 1) == 0:
                elapsed = time.monotonic() - t0
                bar = "=" * int(done / n_requests * 40)
                print(f"\r  [{bar:<40}] {done}/{n_requests}  {done/elapsed:.1f} req/s", end="", flush=True)

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        await asyncio.gather(*[bounded(i, items[i], assignments[i], writer, f) for i in range(n_requests)])

    elapsed = time.monotonic() - t0
    ok = [r for r in results if r["status"] == 200]
    print(f"\r  [{'='*40}] {n_requests}/{n_requests}  ({elapsed:.1f}s, {n_requests/elapsed:.1f} req/s)\n")
    print(f"  Successful: {len(ok)}/{n_requests}")
    mc: dict[str, int] = {}
    for r in ok: mc[r["model_winner"]] = mc.get(r["model_winner"], 0) + 1
    for m, c in sorted(mc.items(), key=lambda x: -x[1]):
        print(f"  {m:<22} {c} requests ({100*c//len(ok)}%)")
    costs = [float(r["charged_usd"]) for r in ok if r["charged_usd"]]
    if costs: print(f"  Total cost: ${sum(costs):.6f}  avg=${mean(costs):.8f}")
    print(f"\n  Saved: {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requests",    type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--output",      default="")
    p.add_argument("--dataset",     default=None)
    args = p.parse_args()
    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/round_robin_{ts}.csv"
    asyncio.run(run(args.requests, args.concurrency, args.output, args.dataset))

if __name__ == "__main__":
    main()
