"""
baseline_complexity_tier.py — Complexity-tier routing baseline.

Routes each query to a fixed model determined by (domain, complexity).
Captures the idea of RouterDC / difficulty-calibrated routing without any
learned classifier: easy→cheap-fast, hard→accurate-expensive.

Produces a CSV in the same format as load_test.py so compare_categories.py
can compare it against TTCA/round-robin/vllm-sr results.

Usage:
    python tests/baseline_complexity_tier.py \\
        --dataset   datasets/hf_1500.json \\
        --output    results/baseline_tier.csv \\
        --concurrency 50 \\
        --node2-host sophia-gpu-09
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

# ---------------------------------------------------------------------------
# Tier routing table — (domain, complexity) → model_id
#
# Design rationale:
#   easy   → qwen-7b       (fastest, cheapest; high accuracy on simple tasks)
#   medium → domain-specific mid-tier (reasoning models for math/code/reasoning,
#                                      general model for factual)
#   hard   → deepseek-r1-14b or llama4-scout (highest accuracy)
# ---------------------------------------------------------------------------

TIER_MAP: dict[tuple[str, str], str] = {
    ("factual",   "easy"):   "qwen-7b",
    ("factual",   "medium"): "gemma-3-27b",
    ("factual",   "hard"):   "llama4-scout",
    ("math",      "easy"):   "qwen-7b",
    ("math",      "medium"): "deepseek-r1-7b",
    ("math",      "hard"):   "deepseek-r1-14b",
    ("code",      "easy"):   "qwen3-coder-30b",
    ("code",      "medium"): "qwen3-coder-30b",
    ("code",      "hard"):   "deepseek-r1-14b",
    ("reasoning", "easy"):   "qwen-7b",
    ("reasoning", "medium"): "deepseek-r1-7b",
    ("reasoning", "hard"):   "deepseek-r1-14b",
}

# Default tiers for any domain not in the table above
_COMPLEXITY_DEFAULT = {"easy": "qwen-7b", "medium": "deepseek-r1-7b", "hard": "deepseek-r1-14b"}

BACKENDS: dict[str, dict] = {
    "qwen-7b": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "base_url":   "http://localhost:8000",
        "input_rate":  5e-8,   "output_rate": 1e-7,    # $0.05/$0.10 per 1M
        "eff_tok_per_j": 13.0,
    },
    "deepseek-r1-7b": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "base_url":   "http://localhost:8001",
        "input_rate":  6e-8,   "output_rate": 1.4e-7,  # $0.06/$0.14
        "eff_tok_per_j": 13.0,
    },
    "qwen3-coder-30b": {
        "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "base_url":   "http://localhost:8002",
        "input_rate":  1.5e-7, "output_rate": 6e-7,    # $0.15/$0.60
        "eff_tok_per_j": 12.0,
    },
    "gemma-3-27b": {
        "model_name": "google/gemma-3-27b-it",
        "base_url":   "http://localhost:8003",
        "input_rate":  8e-8,   "output_rate": 1.6e-7,  # $0.08/$0.16
        "eff_tok_per_j": 5.0,
    },
    "deepseek-r1-14b": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "base_url":   "http://localhost:8004",
        "input_rate":  1e-7,   "output_rate": 2.5e-7,  # $0.10/$0.25
        "eff_tok_per_j": 6.0,
    },
    "llama4-scout": {
        "model_name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "base_url":   "",  # filled in via --node2-host
        "input_rate":  1e-7,   "output_rate": 3e-7,    # $0.10/$0.30
        "eff_tok_per_j": 3.0,
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
    "wall_ms", "slo_ms", "slo_violated", "retries", "response_text", "error",
]


_DYNAMIC_TIER_MAP: dict[tuple[str, str], str] = {}


def pick_model(domain: str, complexity: str) -> str:
    if _DYNAMIC_TIER_MAP:
        key = f"{domain}:{complexity}"
        return _DYNAMIC_TIER_MAP.get(key, _COMPLEXITY_DEFAULT.get(complexity, "qwen-7b"))
    return TIER_MAP.get((domain, complexity),
                        _COMPLEXITY_DEFAULT.get(complexity, "qwen-7b"))


async def send_one(client: httpx.AsyncClient, req_id: int, item: dict) -> dict:
    domain     = item["domain"]
    complexity = item["complexity"]
    model_id   = pick_model(domain, complexity)
    backend    = BACKENDS[model_id]

    input_tokens = len(item["query"].split()) * 1.3
    out_est      = OUTPUT_TOKENS.get((domain, complexity), 300)
    cost_est     = input_tokens * backend["input_rate"] + out_est * backend["output_rate"]

    result = {
        "req_id":            req_id,
        "domain":            domain,
        "complexity":        complexity,
        "query":             item["query"][:100],
        "ground_truth":      str(item.get("ground_truth", "")),
        "mode":              "complexity_tier",
        "status":            "",
        "model_winner":      model_id,
        "bid_latency_ms":    "",
        "actual_latency_ms": "",
        "ttft_ms":           "",
        "output_tokens":     "",
        "charged_usd":       f"{cost_est:.8f}",
        "energy_j":          "",
        "load":              "",
        "wall_ms":           "",
        "slo_ms":            "",
        "slo_violated":      "",
        "retries":           "0",
        "response_text":     "",
        "error":             "",
    }

    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{backend['base_url']}/v1/chat/completions",
            json={
                "model":      backend["model_name"],
                "messages":   [{"role": "user", "content": item["query"]}],
                "max_tokens": 512,
            },
        )
        wall_ms = int((time.monotonic() - t0) * 1000)
        result["wall_ms"] = result["actual_latency_ms"] = result["ttft_ms"] = wall_ms
        result["status"]  = resp.status_code

        if resp.status_code == 200:
            body       = resp.json()
            out_tokens = body.get("usage", {}).get("completion_tokens", out_est)
            result["output_tokens"] = out_tokens
            result["energy_j"]      = f"{out_tokens / backend['eff_tok_per_j']:.3f}"
            actual_cost = (input_tokens * backend["input_rate"]
                           + out_tokens  * backend["output_rate"])
            result["charged_usd"]    = f"{actual_cost:.8f}"
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


async def run(dataset_path: str, output: str, concurrency: int) -> None:
    with open(dataset_path) as f:
        items = json.load(f)
    n = len(items)

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    print(f"\n  [Complexity-Tier] {n} requests, concurrency={concurrency}")
    print(f"  Output: {output}\n")

    # Show routing table
    for (d, c), m in sorted(TIER_MAP.items()):
        print(f"    {d:<12} {c:<8} → {m}")
    print()

    sem  = asyncio.Semaphore(concurrency)
    done = 0
    t0   = time.monotonic()

    async def bounded(req_id: int, item: dict, writer, f) -> None:
        nonlocal done
        async with sem:
            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                r = await send_one(client, req_id, item)
            writer.writerow(r)
            f.flush()
            done += 1
            if done % max(n // 20, 1) == 0:
                elapsed = time.monotonic() - t0
                bar = "=" * int(done / n * 40)
                print(f"\r  [{bar:<40}] {done}/{n}  {done/elapsed:.1f} req/s",
                      end="", flush=True)

    results = []
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        tasks = [bounded(i, items[i], writer, f) for i in range(n)]
        await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t0
    print(f"\r  [{'='*40}] {n}/{n}  ({elapsed:.1f}s, {n/elapsed:.1f} req/s)\n")

    # Quick summary
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
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--output",       default="")
    parser.add_argument("--concurrency",  type=int, default=50)
    parser.add_argument("--node2-host",   default=None,
                        help="Hostname of node 2 for llama4-scout (e.g. sophia-gpu-09)")
    parser.add_argument("--tier-map",     default=None,
                        help="JSON file from build_optimal_tier.py. Overrides hardcoded TIER_MAP.")
    parser.add_argument("--tier-variant", default="accuracy_optimal",
                        choices=["accuracy_optimal", "ttca_optimal"],
                        help="Which variant to use from --tier-map (default: accuracy_optimal)")
    args = parser.parse_args()

    if args.tier_map:
        import json as _json
        data = _json.load(open(args.tier_map))
        variant = data.get(args.tier_variant, {})
        if not variant:
            raise SystemExit(f"ERROR: variant '{args.tier_variant}' not found in {args.tier_map}")
        _DYNAMIC_TIER_MAP.update(variant)
        print(f"  [--tier-map] Using {args.tier_variant} from {args.tier_map} ({len(variant)} cells)")

    if args.node2_host:
        BACKENDS["llama4-scout"]["base_url"] = f"http://{args.node2_host}:8005"
    else:
        # If no node2-host, fallback hard-tier to deepseek-r1-14b instead of llama4-scout
        for key in list(TIER_MAP):
            if TIER_MAP[key] == "llama4-scout":
                TIER_MAP[key] = "deepseek-r1-14b"

    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/baseline_tier_{ts}.csv"

    asyncio.run(run(args.dataset, args.output, args.concurrency))


if __name__ == "__main__":
    main()
