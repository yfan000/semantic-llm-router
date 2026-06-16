"""baseline_cascade.py -- RouteLLM-style two-model cascade baseline.

WEAK=qwen-7b, STRONG=deepseek-r1-14b. Route to WEAK if its accuracy
prior >= threshold, else to STRONG.

Usage:
    python tests/baseline_cascade.py \\
        --dataset datasets/hf_1500.json --priors results/priors_new.json
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

WEAK_MODEL   = "qwen-7b"
STRONG_MODEL = "deepseek-r1-14b"

BACKENDS: dict[str, dict] = {
    "qwen-7b": {
        "model_name":    "Qwen/Qwen2.5-7B-Instruct",
        "base_url":      "http://localhost:8000",
        "input_rate":    0.0000003, "output_rate": 0.0000006,
        "eff_tok_per_j": 13.0,
    },
    "deepseek-r1-14b": {
        "model_name":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "base_url":      "http://localhost:8004",
        "input_rate":    0.0000005, "output_rate": 0.0000010,
        "eff_tok_per_j": 6.0,
    },
}

OUTPUT_TOKENS: dict[tuple[str, str], int] = {
    ("factual","easy"):80,    ("factual","medium"):200,   ("factual","hard"):350,
    ("math","easy"):120,      ("math","medium"):280,      ("math","hard"):450,
    ("code","easy"):150,      ("code","medium"):350,      ("code","hard"):650,
    ("reasoning","easy"):180, ("reasoning","medium"):380, ("reasoning","hard"):600,
}

FIELDNAMES = [
    "req_id", "domain", "complexity", "query", "ground_truth", "mode",
    "status", "model_winner", "bid_latency_ms", "actual_latency_ms",
    "ttft_ms", "output_tokens", "charged_usd", "energy_j", "load",
    "wall_ms", "slo_ms", "slo_violated", "response_text", "error",
]


def load_priors(path: str) -> dict[str, dict[str, float]]:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def pick_model(domain: str, complexity: str, weak_priors: dict[str, float], threshold: float) -> str:
    key = f"{domain}:{complexity}"
    weak_acc = weak_priors.get(key, weak_priors.get("_default", 0.0))
    return WEAK_MODEL if weak_acc >= threshold else STRONG_MODEL


async def send_one(client: httpx.AsyncClient, req_id: int, item: dict, model_id: str) -> dict:
    backend    = BACKENDS[model_id]
    domain     = item["domain"]
    complexity = item["complexity"]
    input_tokens = len(item["query"].split()) * 1.3
    out_est      = OUTPUT_TOKENS.get((domain, complexity), 300)
    cost_est     = input_tokens * backend["input_rate"] + out_est * backend["output_rate"]
    result = {
        "req_id": req_id, "domain": domain, "complexity": complexity,
        "query": item["query"][:100], "ground_truth": str(item.get("ground_truth", "")),
        "mode": "cascade", "status": "", "model_winner": model_id,
        "bid_latency_ms": "", "actual_latency_ms": "", "ttft_ms": "",
        "output_tokens": "", "charged_usd": f"{cost_est:.8f}",
        "energy_j": "", "load": "", "wall_ms": "",
        "slo_ms": "", "slo_violated": "", "response_text": "", "error": "",
    }
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{backend['base_url']}/v1/chat/completions",
            json={"model": backend["model_name"], "messages": [{"role": "user", "content": item["query"]}], "max_tokens": 512},
        )
        wall_ms = int((time.monotonic() - t0) * 1000)
        result["wall_ms"] = result["actual_latency_ms"] = result["ttft_ms"] = wall_ms
        result["status"]  = resp.status_code
        if resp.status_code == 200:
            body       = resp.json()
            out_tokens = body.get("usage", {}).get("completion_tokens", out_est)
            result["output_tokens"] = out_tokens
            result["energy_j"]      = f"{out_tokens / backend['eff_tok_per_j']:.3f}"
            actual_cost = input_tokens * backend["input_rate"] + out_tokens * backend["output_rate"]
            result["charged_usd"] = f"{actual_cost:.8f}"
            choices = body.get("choices", [])
            if choices:
                result["response_text"] = choices[0].get("message", {}).get("content", "")
        else:
            result["error"] = str(resp.status_code)
    except Exception as e:
        result["wall_ms"] = int((time.monotonic() - t0) * 1000)
        result["status"]  = "error"
        result["error"]   = str(e)[:200]
    return result


async def run(dataset_path: str, priors_path: str, output: str, threshold: float, concurrency: int) -> None:
    with open(dataset_path) as f:
        items = json.load(f)
    n = len(items)
    priors = load_priors(priors_path)
    weak_priors: dict[str, float] = priors.get(WEAK_MODEL, {})
    if not weak_priors:
        print(f"  WARNING: No priors for {WEAK_MODEL} -- all queries go to {STRONG_MODEL}")
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    print(f"\n  [Cascade] {n} requests, threshold={threshold}, concurrency={concurrency}")
    print(f"  Weak: {WEAK_MODEL}  Strong: {STRONG_MODEL}  Priors: {priors_path}")
    print(f"  Output: {output}\n")
    assignments = [
        pick_model(items[i]["domain"], items[i]["complexity"], weak_priors, threshold)
        for i in range(n)
    ]
    weak_count   = assignments.count(WEAK_MODEL)
    strong_count = assignments.count(STRONG_MODEL)
    print(f"  Routing: {WEAK_MODEL}={weak_count} ({100*weak_count//n}%)  {STRONG_MODEL}={strong_count} ({100*strong_count//n}%)\n")
    sem  = asyncio.Semaphore(concurrency)
    done = 0
    t0   = time.monotonic()

    async def bounded(req_id: int, item: dict, model_id: str, writer, f) -> None:
        nonlocal done
        async with sem:
            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                r = await send_one(client, req_id, item, model_id)
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
        await asyncio.gather(*[bounded(i, items[i], assignments[i], writer, f) for i in range(n)])

    elapsed = time.monotonic() - t0
    print(f"\r  [{'='*40}] {n}/{n}  ({elapsed:.1f}s, {n/elapsed:.1f} req/s)\n")
    with open(output, newline="") as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if str(r["status"]) == "200"]
    mc: dict[str, int] = {}
    for r in ok:
        mc[r["model_winner"]] = mc.get(r["model_winner"], 0) + 1
    print(f"  Successful: {len(ok)}/{n}")
    for m, c in sorted(mc.items(), key=lambda x: -x[1]):
        print(f"    {m:<24} {c} requests ({100*c//max(len(ok),1)}%)")
    costs = [float(r["charged_usd"]) for r in ok if r["charged_usd"]]
    if costs:
        print(f"  Total cost: ${sum(costs):.6f}  avg=${mean(costs):.8f}")
    print(f"\n  Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--priors",      default="results/priors_new.json")
    parser.add_argument("--output",      default="")
    parser.add_argument("--threshold",   type=float, default=0.80)
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()
    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/baseline_cascade_{ts}.csv"
    asyncio.run(run(args.dataset, args.priors, args.output, args.threshold, args.concurrency))


if __name__ == "__main__":
    main()
