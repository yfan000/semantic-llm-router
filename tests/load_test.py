"""load_test.py -- Load test against the running router.

Usage:
    python tests/load_test.py --router http://localhost:8000 --requests 1000
    python tests/load_test.py --mode ttca --no-retry  # single-shot baseline comparison
    python tests/load_test.py --rate 10               # open-loop 10 req/s arrival rate
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import os
import random
import time
from datetime import datetime
from statistics import mean, median

import httpx

QUERIES: dict[tuple[str, str], list[str]] = {
    ("factual", "easy"): [
        "What is the capital of France?", "What is the chemical symbol for gold?",
        "How many days are in a leap year?", "Who invented the telephone?",
        "What is the speed of light in km/s?", "What is the largest planet in our solar system?",
        "How many continents are there on Earth?", "What is the boiling point of water in Celsius?",
        "Who painted the Mona Lisa?", "What is the capital of Japan?",
    ],
    ("factual", "medium"): [
        "Explain the difference between DNA and RNA.",
        "What caused the fall of the Roman Empire?",
        "How does the immune system fight viruses?",
        "What is the significance of the Magna Carta?",
        "Explain how vaccines work.",
    ],
    ("factual", "hard"): [
        "Explain the geopolitical implications of the Bretton Woods collapse.",
        "Analyse the long-term economic consequences of hyperinflation in Weimar Germany.",
        "What are the epistemological foundations of the scientific method?",
    ],
    ("math", "easy"): [
        "What is 15% of 240?", "What is the square root of 144?",
        "If a rectangle is 8m wide and 5m tall, what is its area?",
        "What is 3/4 expressed as a decimal?", "What is 2 to the power of 8?",
    ],
    ("math", "medium"): [
        "Solve the quadratic equation x squared minus 5x plus 6 equals 0.",
        "Find the derivative of f(x) = 3x^3 - 2x^2 + x - 5.",
        "What is the probability of rolling two sixes in a row with a fair die?",
    ],
    ("math", "hard"): [
        "Prove by induction that the sum of the first n natural numbers is n(n+1)/2.",
        "Find the eigenvalues and eigenvectors of the matrix [[3,1],[1,3]].",
        "Prove that the square root of 2 is irrational.",
    ],
    ("code", "easy"): [
        "Write a Python function to reverse a string.",
        "Write a Python function to check if a number is even.",
        "Write a function to find the maximum of two numbers.",
    ],
    ("code", "medium"): [
        "Implement a binary search algorithm in Python.",
        "Write a Python class for a stack data structure.",
        "Implement merge sort in Python.",
    ],
    ("code", "hard"): [
        "Implement a thread-safe singleton pattern in Python.",
        "Write a Python implementation of a consistent hash ring.",
        "Implement an async connection pool in Python using asyncio.",
    ],
    ("reasoning", "easy"): [
        "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?",
        "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?",
    ],
    ("reasoning", "medium"): [
        "Compare and contrast SQL and NoSQL databases for a high-traffic web app.",
        "What are the trade-offs between microservices and monolithic architecture?",
    ],
    ("reasoning", "hard"): [
        "Analyse the game-theoretic implications of the prisoner's dilemma for international climate agreements.",
    ],
    ("creative", "easy"): [
        "Write a haiku about the ocean.", "Write a two-sentence story about a friendly dragon.",
    ],
    ("creative", "medium"): [
        "Write a short story about a robot learning to paint.",
        "Write a poem about the loneliness of space exploration.",
    ],
    ("creative", "hard"): [
        "Write a philosophical dialogue between Socrates and a modern AI.",
    ],
}


def build_request_list(n: int, dataset_path: str | None = None) -> list[dict]:
    if dataset_path:
        import json as _json
        with open(dataset_path) as f:
            items = _json.load(f)
        print(f"  Loaded {len(items)} items from {dataset_path}")
        if n > len(items):
            items += random.choices(items, k=n - len(items))
        return items[:n]

    all_queries = []
    for (domain, complexity), queries in QUERIES.items():
        for q in queries:
            all_queries.append({"domain": domain, "complexity": complexity, "query": q})
    random.shuffle(all_queries)
    if n > len(all_queries):
        all_queries.extend(random.choices(all_queries, k=n - len(all_queries)))
    return all_queries[:n]


MODES = ["cost", "eco", "accuracy", "custom"]
MODE_WEIGHTS = [0.35, 0.25, 0.25, 0.15]

_FORCED_MODE: str | None = None
_NO_RETRY: bool = False


def random_router_params(item: dict | None = None) -> dict:
    if _FORCED_MODE is not None:
        params: dict = {"mode": _FORCED_MODE}
    else:
        mode = random.choices(MODES, weights=MODE_WEIGHTS)[0]
        if mode == "custom":
            w = [random.random() for _ in range(4)]
            s = sum(w)
            params = {
                "mode": "custom",
                "cost_weight":     round(w[0]/s, 3),
                "latency_weight":  round(w[1]/s, 3),
                "accuracy_weight": round(w[2]/s, 3),
                "energy_weight":   round(w[3]/s, 3),
            }
        else:
            params = {"mode": mode}

    if item and item.get("domain"):
        params["domain"] = item["domain"]
    if item and item.get("complexity"):
        params["complexity"] = item["complexity"]

    if _NO_RETRY:
        params["no_retry"] = True

    return params


async def send_request(
    client: httpx.AsyncClient,
    router_url: str,
    req_id: int,
    item: dict,
) -> dict:
    params = random_router_params(item)
    payload = {
        "model":    "auto",
        "messages": [{"role": "user", "content": item["query"]}],
        "extra_body": {"router": params},
    }

    result = {
        "req_id":            req_id,
        "domain":            item["domain"],
        "complexity":        item["complexity"],
        "query":             item["query"][:100],
        "ground_truth":      str(item.get("ground_truth", "")),
        "mode":              params.get("mode", "custom"),
        "status":            "",
        "model_winner":      "",
        "bid_latency_ms":    "",
        "actual_latency_ms": "",
        "ttft_ms":           "",
        "itl_ms":            "",
        "output_tokens":     "",
        "charged_usd":       "",
        "energy_j":          "",
        "load":              "",
        "wall_ms":           "",
        "slo_ms":            "",
        "slo_violated":      "",
        "retries":           "",
        "gt_scored":         "",
        "gt_correct":        "",
        "response_text":     "",
        "error":             "",
    }

    t_start = time.monotonic()
    try:
        resp = await client.post(f"{router_url}/v1/chat/completions", json=payload)
        wall_ms = int((time.monotonic() - t_start) * 1000)
        result["wall_ms"] = wall_ms
        result["status"]  = resp.status_code

        h = resp.headers
        result["model_winner"]      = h.get("x-router-model", "")
        result["bid_latency_ms"]    = h.get("x-router-bid-latency-ms", "")
        result["actual_latency_ms"] = h.get("x-router-actual-latency-ms", "")
        result["charged_usd"]       = h.get("x-router-charged-usd", "")
        result["energy_j"]          = h.get("x-router-energy-j", "")
        result["load"]              = h.get("x-router-load", "")
        result["slo_ms"]       = h.get("x-router-slo-ms", "")
        result["slo_violated"] = h.get("x-router-slo-violated", "")
        attempt = h.get("x-router-attempt", "1")
        try:
            result["retries"] = int(attempt) - 1
        except (ValueError, TypeError):
            result["retries"] = 0
        result["gt_scored"]  = h.get("x-router-gt-scored", "")
        result["gt_correct"] = h.get("x-router-gt-correct", "")

        if resp.status_code == 200:
            body = resp.json()
            usage = body.get("usage", {})
            result["output_tokens"] = usage.get("completion_tokens", "")
            result["ttft_ms"] = h.get("x-router-actual-latency-ms", "")
            choices = body.get("choices", [])
            if choices:
                result["response_text"] = choices[0].get("message", {}).get("content", "")
        else:
            try:
                result["error"] = resp.json().get("detail", str(resp.status_code))
            except Exception:
                result["error"] = resp.text[:200]

    except Exception as e:
        result["wall_ms"] = int((time.monotonic() - t_start) * 1000)
        result["status"]  = "error"
        result["error"]   = str(e)

    return result


def analyse(rows: list[dict]) -> None:
    W = 62
    ok   = [r for r in rows if r["status"] == 200]
    errs = [r for r in rows if r["status"] != 200]

    def safe_float(v: str) -> float | None:
        try: return float(v)
        except: return None

    print(f"\n{'='*W}")
    print("  LOAD TEST RESULTS")
    print(f"{'='*W}")
    print(f"  Total requests : {len(rows)}")
    print(f"  Successful     : {len(ok)}  ({100*len(ok)//max(len(rows),1)}%)")
    print(f"  Errors         : {len(errs)}")

    if not ok:
        return

    model_counts: dict[str, int] = {}
    for r in ok:
        m = r["model_winner"] or "unknown"
        model_counts[m] = model_counts.get(m, 0) + 1
    print(f"\n{'='*W}")
    print("  ROUTING DISTRIBUTION")
    print(f"{'='*W}")
    for model, cnt in sorted(model_counts.items(), key=lambda x: -x[1]):
        bar = "X" * int(cnt / len(ok) * 30)
        pct = 100 * cnt // len(ok)
        print(f"  {model:<22} {bar:<30} {pct:3}%  ({cnt})")

    wall_vals = [safe_float(r["wall_ms"]) for r in ok if safe_float(r["wall_ms"])]
    if wall_vals:
        wall_vals.sort()
        print(f"\n  Wall P50={wall_vals[len(wall_vals)//2]:.0f}ms  P95={wall_vals[int(len(wall_vals)*.95)]:.0f}ms  mean={mean(wall_vals):.0f}ms")

    costs = [safe_float(r["charged_usd"]) for r in ok if safe_float(r["charged_usd"]) is not None]
    if costs:
        print(f"  Total cost: ${sum(costs):.6f}  avg=${mean(costs):.8f}")

    print(f"\n{'='*W}\n")


async def run(router_url: str, n_requests: int, concurrency: int, output: str,
              dataset_path: str | None = None,
              rate: float | None = None) -> None:
    """Run the load test.

    Args:
        rate: Optional target arrival rate in requests/second (open-loop).
              None = closed-loop (fire as fast as concurrency allows).
              When set, requests are dispatched at approximately this rate;
              concurrency still caps simultaneous in-flight requests.
    """
    requests_list = build_request_list(n_requests, dataset_path)
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    fieldnames = [
        "req_id", "domain", "complexity", "query", "ground_truth", "mode",
        "status", "model_winner", "bid_latency_ms", "actual_latency_ms",
        "ttft_ms", "itl_ms", "output_tokens",
        "charged_usd", "energy_j", "load", "wall_ms",
        "slo_ms", "slo_violated", "retries", "gt_scored", "gt_correct",
        "response_text", "error",
    ]

    results: list[dict] = []
    sem = asyncio.Semaphore(concurrency)
    done = 0
    t0 = time.monotonic()

    rate_str = f"{rate:.1f} req/s (open-loop)" if rate else f"closed-loop (concurrency={concurrency})"
    print(f"\n  Sending {n_requests} requests to {router_url}")
    print(f"  Rate       : {rate_str}")
    print(f"  Concurrency: {concurrency}  (max simultaneous in-flight)")
    if _NO_RETRY:
        print("  [No-Retry] cascade retry disabled -- single-shot mode")
    print(f"  Output     : {output}\n")

    async def bounded(req_id: int, item: dict, writer, f) -> None:
        nonlocal done
        async with sem:
            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                r = await send_request(client, router_url, req_id, item)
            results.append(r)
            writer.writerow(r)
            f.flush()
            done += 1
            if done % max(n_requests // 20, 1) == 0:
                elapsed = time.monotonic() - t0
                actual_rate = done / elapsed
                bar = "=" * int(done / n_requests * 40)
                print(f"\r  [{bar:<40}] {done}/{n_requests}  {actual_rate:.1f} req/s", end="", flush=True)

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if rate is None:
            # Closed-loop: dispatch all tasks immediately, semaphore limits concurrency
            await asyncio.gather(*[
                bounded(i, requests_list[i], writer, f)
                for i in range(n_requests)
            ])
        else:
            # Open-loop: dispatch at target rate, semaphore caps in-flight concurrency.
            # Request i fires at t0 + i/rate seconds. If the semaphore is saturated
            # (system overloaded), the request queues until a slot opens — this models
            # real queue build-up rather than silently dropping requests.
            interval = 1.0 / rate
            tasks = []
            for i in range(n_requests):
                target_time = t0 + i * interval
                delay = target_time - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                task = asyncio.create_task(bounded(i, requests_list[i], writer, f))
                tasks.append(task)
            await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t0
    actual_rate = n_requests / elapsed
    print(f"\r  [{'='*40}] {n_requests}/{n_requests}  ({elapsed:.1f}s total, {actual_rate:.1f} req/s)\n")

    analyse(results)
    print(f"  Full results saved to: {output}")


def main() -> None:
    global _FORCED_MODE, _NO_RETRY
    parser = argparse.ArgumentParser(description="Load test the semantic LLM router")
    parser.add_argument("--router",      default="http://localhost:8000")
    parser.add_argument("--requests",    type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max simultaneous in-flight requests (default 10)")
    parser.add_argument("--output",      default="")
    parser.add_argument("--dataset",     default=None,
                        help="Path to JSON dataset from build_dataset.py")
    parser.add_argument("--mode",        default=None,
                        choices=["accuracy", "cost", "eco", "custom", "ttca"],
                        help="Force ALL requests to use this router mode (default: mixed)")
    parser.add_argument("--no-retry",    action="store_true",
                        help="Disable cascade retry — accept first model's answer "
                             "regardless of correctness. Use for single-shot baseline comparison.")
    parser.add_argument("--rate",        type=float, default=None,
                        help="Target arrival rate in requests/second (open-loop). "
                             "Default: None = closed-loop (fire as fast as concurrency allows). "
                             "Example: --rate 10 dispatches one request every 0.1s. "
                             "Concurrency still caps simultaneous in-flight requests.")
    args = parser.parse_args()

    if args.mode:
        _FORCED_MODE = args.mode
        print(f"  [Mode] All {args.requests} requests forced to: {args.mode}")

    if args.no_retry:
        _NO_RETRY = True

    if not args.output:
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{args.mode}" if args.mode else ""
        args.output = f"results/load_test{suffix}_{ts}.csv"

    asyncio.run(run(args.router, args.requests, args.concurrency, args.output,
                    getattr(args, "dataset", None),
                    rate=args.rate))


if __name__ == "__main__":
    main()
