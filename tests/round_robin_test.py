"""
round_robin_test.py — Baseline load test: round-robin or single-model.

Sends requests to vLLM backends directly, bypassing the semantic router.
Produces a CSV in the same format as load_test.py so compare_results.py
can compare side by side.

Usage:
    # Round-robin across all models
    python tests/round_robin_test.py \
        --dataset datasets/hf_1000.json \
        --output results/round_robin.csv \
        --concurrency 50

    # All requests to one specific model
    python tests/round_robin_test.py \
        --dataset datasets/hf_1000.json \
        --model qwen-7b \
        --output results/single_qwen7b.csv \
        --concurrency 50
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

LATENCY_SLO_MS: dict[tuple[str, str], int] = {
    ("factual",   "easy"):   1000,
    ("factual",   "medium"): 2000,
    ("factual",   "hard"):   4000,
    ("math",      "easy"):   1000,
    ("math",      "medium"): 3000,
    ("math",      "hard"):   6000,
    ("code",      "easy"):   1500,
    ("code",      "medium"): 4000,
    ("code",      "hard"):   8000,
    ("reasoning", "easy"):   1000,
    ("reasoning", "medium"): 3000,
    ("reasoning", "hard"):   6000,
    ("creative",  "easy"):   1500,
    ("creative",  "medium"): 5000,
    ("creative",  "hard"):   8000,
}

# ---------------------------------------------------------------------------
# Backend configuration -- Sophia 6-model setup (2 nodes)
# ---------------------------------------------------------------------------

BACKENDS = [
    {
        "model_id":                    "qwen-7b",
        "model_name":                  "Qwen/Qwen2.5-7B-Instruct",
        "base_url":                    "http://localhost:8000",
        "input_rate_usd_per_token":    0.0000003,
        "output_rate_usd_per_token":   0.0000006,
        "efficiency_tokens_per_joule": 13.0,
    },
    {
        "model_id":                    "qwen-14b",
        "model_name":                  "Qwen/Qwen2.5-14B-Instruct",
        "base_url":                    "http://localhost:8001",
        "input_rate_usd_per_token":    0.0000005,
        "output_rate_usd_per_token":   0.0000010,
        "efficiency_tokens_per_joule": 8.0,
    },
    {
        "model_id":                    "deepseek-r1-7b",
        "model_name":                  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "base_url":                    "http://localhost:8002",
        "input_rate_usd_per_token":    0.0000003,
        "output_rate_usd_per_token":   0.0000006,
        "efficiency_tokens_per_joule": 13.0,
    },
    {
        "model_id":                    "coder-32b",
        "model_name":                  "Qwen/Qwen2.5-Coder-32B-Instruct",
        "base_url":                    "http://localhost:8003",
        "input_rate_usd_per_token":    0.0000010,
        "output_rate_usd_per_token":   0.0000020,
        "efficiency_tokens_per_joule": 4.0,
    },
    {
        "model_id":                    "deepseek-v2-lite",
        "model_name":                  "deepseek-ai/DeepSeek-V2-Lite",
        "base_url":                    "http://localhost:8005",
        "input_rate_usd_per_token":    0.0000003,
        "output_rate_usd_per_token":   0.0000006,
        "efficiency_tokens_per_joule": 11.0,
    },
    {
        "model_id":                    "qwen-32b",
        "model_name":                  "Qwen/Qwen2.5-32B-Instruct",
        "base_url":                    "http://sophia-gpu-07:8004",
        "input_rate_usd_per_token":    0.0000010,
        "output_rate_usd_per_token":   0.0000020,
        "efficiency_tokens_per_joule": 4.0,
    },
]

# Set by --model flag; None means round-robin over all backends
_single_model: str | None = None


def build_request_list(n: int, dataset_path: str | None) -> list[dict]:
    if dataset_path:
        with open(dataset_path) as f:
            items = json.load(f)
        if n > len(items):
            items += random.choices(items, k=n - len(items))
        return items[:n]

    pool = [{"domain": d, "complexity": c, "query": q} for d, c, q in [
        ("factual",   "easy",   "What is the capital of France?"),
        ("factual",   "medium", "Explain the difference between DNA and RNA."),
        ("factual",   "hard",   "Explain the geopolitical implications of the Bretton Woods collapse."),
        ("math",      "easy",   "What is 15% of 240?"),
        ("math",      "medium", "Solve the quadratic equation x^2 - 5x + 6 = 0."),
        ("math",      "hard",   "Prove that there are infinitely many prime numbers."),
        ("code",      "easy",   "Write a Python function to reverse a string."),
        ("code",      "medium", "Implement a binary search algorithm in Python."),
        ("code",      "hard",   "Implement a thread-safe singleton pattern in Python."),
        ("reasoning", "easy",   "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?"),
        ("reasoning", "medium", "Compare SQL and NoSQL databases for a high-traffic web app."),
        ("reasoning", "hard",   "Analyse the game-theoretic implications of the prisoner's dilemma for climate agreements."),
    ]]
    random.shuffle(pool)
    if n > len(pool):
        pool += random.choices(pool, k=n - len(pool))
    return pool[:n]


async def send_request(
    client: httpx.AsyncClient,
    req_id: int,
    item: dict,
    backend: dict,
) -> dict:
    input_tokens = sum(len(m.split()) * 1.3 for m in [item["query"]])
    OUTPUT_TOKENS = {
        ("factual","easy"):80,   ("factual","medium"):200,   ("factual","hard"):350,
        ("math","easy"):120,     ("math","medium"):280,       ("math","hard"):450,
        ("code","easy"):150,     ("code","medium"):350,       ("code","hard"):650,
        ("creative","easy"):250, ("creative","medium"):500,   ("creative","hard"):800,
        ("reasoning","easy"):180,("reasoning","medium"):380,  ("reasoning","hard"):600,
    }
    output_est    = OUTPUT_TOKENS.get((item["domain"], item["complexity"]), 300)
    estimated_cost = (
        input_tokens * backend["input_rate_usd_per_token"]
        + output_est * backend["output_rate_usd_per_token"]
    )
    slo_ms = LATENCY_SLO_MS.get((item["domain"], item["complexity"]), None)

    result = {
        "req_id":            req_id,
        "domain":            item["domain"],
        "complexity":        item["complexity"],
        "query":             item["query"][:100],
        "ground_truth":      str(item.get("ground_truth", "")),
        "mode":              f"single:{backend['model_id']}" if _single_model else "round_robin",
        "status":            "",
        "model_winner":      backend["model_id"],
        "bid_latency_ms":    "",
        "actual_latency_ms": "",
        "ttft_ms":           "",
        "output_tokens":     "",
        "charged_usd":       f"{estimated_cost:.8f}",
        "energy_j":          "",
        "load":              "",
        "wall_ms":           "",
        "slo_ms":            str(slo_ms) if slo_ms else "",
        "slo_violated":      "",
        "response_text":     "",
        "error":             "",
    }

    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{backend['base_url']}/v1/chat/completions",
            json={
                "model":    backend.get("model_name", backend["model_id"]),
                "messages": [{"role": "user", "content": item["query"]}],
            },
        )
        wall_ms = int((time.monotonic() - t0) * 1000)
        result["wall_ms"]           = wall_ms
        result["actual_latency_ms"] = wall_ms
        result["ttft_ms"]           = wall_ms
        result["status"]            = resp.status_code
        if slo_ms is not None:
            result["slo_violated"] = "true" if wall_ms > slo_ms else "false"

        if resp.status_code == 200:
            body       = resp.json()
            out_tokens = body.get("usage", {}).get("completion_tokens", output_est)
            result["output_tokens"] = out_tokens
            result["energy_j"]      = f"{out_tokens / backend['efficiency_tokens_per_joule']:.3f}"
            actual_cost = (
                input_tokens * backend["input_rate_usd_per_token"]
                + out_tokens * backend["output_rate_usd_per_token"]
            )
            result["charged_usd"] = f"{actual_cost:.8f}"
            choices = body.get("choices", [])
            if choices:
                result["response_text"] = choices[0].get("message", {}).get("content", "")
        else:
            result["error"] = str(resp.status_code)

    except Exception as e:
        result["wall_ms"] = int((time.monotonic() - t0) * 1000)
        result["status"]  = "error"
        result["error"]   = str(e)

    return result


async def run(
    n_requests: int,
    concurrency: int,
    output: str,
    dataset_path: str | None,
    single_model: str | None = None,
) -> None:
    items = build_request_list(n_requests, dataset_path)
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    if single_model is not None:
        backend_map = {b["model_id"]: b for b in BACKENDS}
        if single_model not in backend_map:
            valid = ", ".join(backend_map.keys())
            raise ValueError(f"Unknown model '{single_model}'. Valid: {valid}")
        chosen      = backend_map[single_model]
        assignments = [chosen] * n_requests
        print(f"\n  [Single-Model] {n_requests} requests -> {single_model}, concurrency={concurrency}")
    else:
        rr          = itertools.cycle(BACKENDS)
        assignments = [next(rr) for _ in range(n_requests)]
        print(f"\n  [Round-Robin] {n_requests} requests, concurrency={concurrency}")
        print(f"  Backends: {' | '.join(b['model_id'] for b in BACKENDS)}")

    print(f"  Output: {output}\n")

    fieldnames = [
        "req_id", "domain", "complexity", "query", "ground_truth", "mode", "status",
        "model_winner", "bid_latency_ms", "actual_latency_ms",
        "ttft_ms", "output_tokens", "charged_usd", "energy_j", "load", "wall_ms",
        "slo_ms", "slo_violated", "response_text", "error",
    ]

    results: list[dict] = []
    sem     = asyncio.Semaphore(concurrency)
    done    = 0
    t0_wall = time.monotonic()

    async def bounded(req_id: int, item: dict, backend: dict, writer, f) -> None:
        nonlocal done
        async with sem:
            # trust_env=False bypasses system proxy settings that can block
            # cross-node requests (e.g. sophia-gpu-07) on HPC clusters.
            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                r = await send_request(client, req_id, item, backend)
            results.append(r)
            writer.writerow(r)
            f.flush()
            done += 1
            if done % max(n_requests // 20, 1) == 0:
                elapsed = time.monotonic() - t0_wall
                bar = "=" * int(done / n_requests * 40)
                print(f"\r  [{bar:<40}] {done}/{n_requests}  {done/elapsed:.1f} req/s", end="", flush=True)

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        await asyncio.gather(*[
            bounded(i, items[i], assignments[i], writer, f)
            for i in range(n_requests)
        ])

    elapsed = time.monotonic() - t0_wall
    ok      = [r for r in results if r["status"] == 200]
    print(f"\r  [{'='*40}] {n_requests}/{n_requests}  ({elapsed:.1f}s, {n_requests/elapsed:.1f} req/s)\n")

    print(f"  Successful : {len(ok)}/{n_requests}")
    mc: dict[str, int] = {}
    for r in ok:
        mc[r["model_winner"]] = mc.get(r["model_winner"], 0) + 1
    for m, c in sorted(mc.items(), key=lambda x: -x[1]):
        print(f"  {m:<22} {c} requests ({100*c//max(len(ok),1)}%)")

    costs = [float(r["charged_usd"]) for r in ok if r["charged_usd"]]
    if costs:
        print(f"  Total cost : ${sum(costs):.6f}  avg=${mean(costs):.8f}")
    print(f"\n  Saved: {output}")


def main() -> None:
    global _single_model
    valid_models = [b["model_id"] for b in BACKENDS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests",    type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--output",      default="")
    parser.add_argument("--dataset",     default=None)
    parser.add_argument("--model",       default=None, choices=valid_models,
                        help="Send ALL requests to one model (default: round-robin over all)")
    args = parser.parse_args()

    if args.model:
        _single_model = args.model
        print(f"  [Model] All {args.requests} requests -> {args.model}")

    if not args.output:
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = (
            f"results/single_{args.model}_{ts}.csv" if args.model
            else f"results/round_robin_{ts}.csv"
        )
    asyncio.run(run(args.requests, args.concurrency, args.output, args.dataset, args.model))


if __name__ == "__main__":
    main()
