"""baseline_vllm_router.py -- vLLM Semantic Router baseline.

Sends requests through a running vllm-sr instance (Envoy proxy on port 8888).
Single-shot only -- vllm-sr has no retry mechanism.

Usage:
    python tests/baseline_vllm_router.py \\
        --dataset datasets/hf_1500.json --endpoint http://localhost:8888
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import json
import os
import time
from datetime import datetime
from statistics import mean

import httpx

FIELDNAMES = [
    "req_id", "domain", "complexity", "query", "ground_truth", "mode",
    "status", "model_winner", "bid_latency_ms", "actual_latency_ms",
    "ttft_ms", "output_tokens", "charged_usd", "energy_j", "load",
    "wall_ms", "slo_ms", "slo_violated", "response_text", "error",
]

VLLM_SR_MODEL = "MoM"
AVG_INPUT_RATE  = 0.0000006
AVG_OUTPUT_RATE = 0.0000012
AVG_EFF_TOK_J   = 8.0


async def send_one(client: httpx.AsyncClient, endpoint: str, req_id: int, item: dict) -> dict:
    domain     = item["domain"]
    complexity = item["complexity"]
    input_tokens = len(item["query"].split()) * 1.3
    out_est      = 300
    cost_est     = input_tokens * AVG_INPUT_RATE + out_est * AVG_OUTPUT_RATE
    result = {
        "req_id": req_id, "domain": domain, "complexity": complexity,
        "query": item["query"][:100], "ground_truth": str(item.get("ground_truth", "")),
        "mode": "vllm_sr", "status": "", "model_winner": "vllm-sr",
        "bid_latency_ms": "", "actual_latency_ms": "", "ttft_ms": "",
        "output_tokens": "", "charged_usd": f"{cost_est:.8f}",
        "energy_j": "", "load": "", "wall_ms": "",
        "slo_ms": "", "slo_violated": "", "response_text": "", "error": "",
    }
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{endpoint}/v1/chat/completions",
            json={"model": VLLM_SR_MODEL, "messages": [{"role": "user", "content": item["query"]}], "max_tokens": 512},
        )
        wall_ms = int((time.monotonic() - t0) * 1000)
        result["wall_ms"] = result["actual_latency_ms"] = result["ttft_ms"] = wall_ms
        result["status"]  = resp.status_code
        if resp.status_code == 200:
            body = resp.json()
            chosen = (
                resp.headers.get("x-router-model")
                or resp.headers.get("x-selected-model")
                or body.get("model", "vllm-sr")
            )
            result["model_winner"] = chosen
            out_tokens = body.get("usage", {}).get("completion_tokens", out_est)
            result["output_tokens"] = out_tokens
            actual_cost = input_tokens * AVG_INPUT_RATE + out_tokens * AVG_OUTPUT_RATE
            result["charged_usd"] = f"{actual_cost:.8f}"
            result["energy_j"]    = f"{out_tokens / AVG_EFF_TOK_J:.3f}"
            choices = body.get("choices", [])
            if choices:
                result["response_text"] = choices[0].get("message", {}).get("content", "")
        else:
            try:
                result["error"] = resp.json().get("detail", str(resp.status_code))
            except Exception:
                result["error"] = resp.text[:200]
    except Exception as e:
        result["wall_ms"] = int((time.monotonic() - t0) * 1000)
        result["status"]  = "error"
        result["error"]   = str(e)[:200]
    return result


async def check_vllm_sr(endpoint: str) -> bool:
    for path in ["/health", "/v1/models"]:
        try:
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
                r = await client.get(f"{endpoint}{path}")
                if r.status_code == 200:
                    return True
        except Exception:
            pass
    return False


async def run(dataset_path: str, endpoint: str, output: str, concurrency: int) -> None:
    with open(dataset_path) as f:
        items = json.load(f)
    n = len(items)
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    print(f"\n  [vLLM-SR] {n} requests, endpoint={endpoint}, concurrency={concurrency}")
    print(f"  Output: {output}\n")
    ok_flag = await check_vllm_sr(endpoint)
    if not ok_flag:
        print(f"  WARNING: vllm-sr not reachable at {endpoint}\n")
    else:
        print(f"  vllm-sr reachable at {endpoint}\n")
    sem  = asyncio.Semaphore(concurrency)
    done = 0
    t0   = time.monotonic()

    async def bounded(req_id: int, item: dict, writer, f) -> None:
        nonlocal done
        async with sem:
            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                r = await send_one(client, endpoint, req_id, item)
            writer.writerow(r)
            f.flush()
            done += 1
            if done % max(n // 20, 1) == 0:
                elapsed = time.monotonic() - t0
                bar = "=" * int(done / n * 40)
                print(f"\r  [{bar:<40}] {done}/{n}  {done/elapsed:.1f} req/s", end="", flush=True)

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        await asyncio.gather(*[bounded(i, items[i], writer, f) for i in range(n)])

    elapsed = time.monotonic() - t0
    print(f"\r  [{'='*40}] {n}/{n}  ({elapsed:.1f}s, {n/elapsed:.1f} req/s)\n")
    with open(output, newline="") as f:
        rows = list(csv.DictReader(f))
    ok_rows = [r for r in rows if str(r["status"]) == "200"]
    print(f"  Successful : {len(ok_rows)}/{n}")
    lats = [float(r["wall_ms"]) for r in ok_rows if r["wall_ms"]]
    if lats:
        lats.sort()
        print(f"  Latency P50: {lats[len(lats)//2]:.0f}ms  P95: {lats[int(len(lats)*.95)]:.0f}ms  mean: {mean(lats):.0f}ms")
    costs = [float(r["charged_usd"]) for r in ok_rows if r["charged_usd"]]
    if costs:
        print(f"  Total cost : ${sum(costs):.6f}  avg=${mean(costs):.8f}")
    print(f"\n  Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--endpoint",    default="http://localhost:8888")
    parser.add_argument("--output",      default="")
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()
    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/baseline_vllm_sr_{ts}.csv"
    asyncio.run(run(args.dataset, args.endpoint, args.output, args.concurrency))


if __name__ == "__main__":
    main()
