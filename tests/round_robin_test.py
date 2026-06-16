"""round_robin_test.py -- Baseline load test using round-robin model selection.

Usage:
    python tests/round_robin_test.py --dataset datasets/hf_1000.json \\
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

LATENCY_SLO_MS: dict[tuple[str, str], int] = {
    ("factual",   "easy"):   1000, ("factual",   "medium"): 2000, ("factual",   "hard"):   4000,
    ("math",      "easy"):   1500, ("math",      "medium"): 3000, ("math",      "hard"):   6000,
    ("code",      "easy"):   2000, ("code",      "medium"): 4000, ("code",      "hard"):   8000,
    ("reasoning", "easy"):   1500, ("reasoning", "medium"): 3000, ("reasoning", "hard"):   6000,
    ("creative",  "easy"):   2000, ("creative",  "medium"): 5000, ("creative",  "hard"):   8000,
}

BACKENDS = [
    {"model_id": "qwen-7b",         "model_name": "Qwen/Qwen2.5-7B-Instruct",                  "base_url": "http://localhost:8000", "input_rate_usd_per_token": 0.0000003, "output_rate_usd_per_token": 0.0000006, "efficiency_tokens_per_joule": 13.0},
    {"model_id": "deepseek-r1-7b",  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   "base_url": "http://localhost:8001", "input_rate_usd_per_token": 0.0000003, "output_rate_usd_per_token": 0.0000006, "efficiency_tokens_per_joule": 13.0},
    {"model_id": "qwen3-coder-30b", "model_name": "Qwen/Qwen3-Coder-30B-A3B",                  "base_url": "http://localhost:8002", "input_rate_usd_per_token": 0.0000007, "output_rate_usd_per_token": 0.0000014, "efficiency_tokens_per_joule": 12.0},
    {"model_id": "gemma-3-27b",     "model_name": "google/gemma-3-27b-it",                      "base_url": "http://localhost:8003", "input_rate_usd_per_token": 0.0000008, "output_rate_usd_per_token": 0.0000016, "efficiency_tokens_per_joule": 5.0},
    {"model_id": "deepseek-r1-14b", "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  "base_url": "http://localhost:8004", "input_rate_usd_per_token": 0.0000005, "output_rate_usd_per_token": 0.0000010, "efficiency_tokens_per_joule": 6.0},
]

_single_model: str | None = None


def build_request_list(n: int, dataset_path: str | None) -> list[dict]:
    if dataset_path:
        with open(dataset_path) as f:
            items = json.load(f)
        if n > len(items):
            items += random.choices(items, k=n - len(items))
        return items[:n]
    pool = [{"domain": "factual", "complexity": "easy", "query": "What is the capital of France?"}] * n
    return pool[:n]


async def send_request(
    client: httpx.AsyncClient,
    req_id: int,
    item: dict,
    backend: dict,
) -> dict:
    input_tokens = len(item["query"].split()) * 1.3
    OUTPUT_TOKENS = {
        ("factual","easy"):80, ("factual","medium"):200, ("factual","hard"):350,
        ("math","easy"):120,   ("math","medium"):280,    ("math","hard"):450,
        ("code","easy"):150,   ("code","medium"):350,    ("code","hard"):650,
        ("creative","easy"):250,("creative","medium"):500,("creative","hard"):800,
        ("reasoning","easy"):180,("reasoning","medium"):380,("reasoning","hard"):600,
    }
    output_est = OUTPUT_TOKENS.get((item["domain"], item["complexity"]), 300)
    estimated_cost = (
        input_tokens  * backend["input_rate_usd_per_token"]
        + output_est  * backend["output_rate_usd_per_token"]
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
                "model":      backend.get("model_name", backend["model_id"]),
                "messages":   [{"role": "user", "content": item["query"]}],
                "max_tokens": 512,
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
            body = resp.json()
            out_tokens = body.get("usage", {}).get("completion_tokens", output_est)
            result["output_tokens"] = out_tokens
            result["energy_j"] = f"{out_tokens / backend['efficiency_tokens_per_joule']:.3f}"
            actual_cost = (
                input_tokens  * backend["input_rate_usd_per_token"]
                + out_tokens  * backend["output_rate_usd_per_token"]
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
            raise ValueError(f"Unknown model '{single_model}'")
        chosen = backend_map[single_model]
        assignments = [chosen] * n_requests
        print(f"\n  [Single-Model] {n_requests} requests -> {single_model}, concurrency={concurrency}")
    else:
        rr = itertools.cycle(BACKENDS)
        assignments = [next(rr) for _ in range(n_requests)]
        print(f"\n  [Round-Robin] {n_requests} requests, concurrency={concurrency}")

    fieldnames = [
        "req_id", "domain", "complexity", "query", "ground_truth", "mode", "status",
        "model_winner", "bid_latency_ms", "actual_latency_ms",
        "ttft_ms", "output_tokens", "charged_usd", "energy_j", "load", "wall_ms",
        "slo_ms", "slo_violated", "response_text", "error",
    ]

    results: list[dict] = []
    sem = asyncio.Semaphore(concurrency)
    done = 0
    t0_wall = time.monotonic()
    print(f"  Output: {output}\n")

    async def bounded(req_id: int, item: dict, backend: dict, writer, f) -> None:
        nonlocal done
        async with sem:
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
    ok = [r for r in results if r["status"] == 200]
    print(f"\r  [{'='*40}] {n_requests}/{n_requests}  ({elapsed:.1f}s, {n_requests/elapsed:.1f} req/s)\n")
    print(f"  Successful : {len(ok)}/{n_requests}")
    mc: dict[str, int] = {}
    for r in ok:
        mc[r["model_winner"]] = mc.get(r["model_winner"], 0) + 1
    for m, c in sorted(mc.items(), key=lambda x: -x[1]):
        print(f"  {m:<22} {c} requests ({100*c//len(ok)}%)")
    costs = [float(r["charged_usd"]) for r in ok if r["charged_usd"]]
    if costs:
        print(f"  Total cost : ${sum(costs):.6f}  avg=${mean(costs):.8f}")
    print(f"\n  Saved: {output}")


def main() -> None:
    global _single_model
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests",    type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--output",      default="")
    parser.add_argument("--dataset",     default=None)
    parser.add_argument("--node2-host",  default=None)
    parser.add_argument("--model",       default=None)
    args = parser.parse_args()

    if args.node2_host:
        BACKENDS.append({
            "model_id": "llama4-scout", "model_name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "base_url": f"http://{args.node2_host}:8005",
            "input_rate_usd_per_token": 0.0000010, "output_rate_usd_per_token": 0.0000020,
            "efficiency_tokens_per_joule": 3.0,
        })

    if args.model and args.model not in [b["model_id"] for b in BACKENDS]:
        print(f"Unknown model '{args.model}'.")
        return

    if args.model:
        _single_model = args.model

    if not args.output:
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{args.model}" if args.model else ""
        args.output = f"results/single{suffix}_{ts}.csv" if args.model else f"results/round_robin_{ts}.csv"
    asyncio.run(run(args.requests, args.concurrency, args.output, args.dataset, args.model))


if __name__ == "__main__":
    main()
