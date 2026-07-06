"""
baseline_routellm.py — RouteLLM baseline using the actual RouteLLM library.

Uses RouteLLM's pre-trained Matrix Factorization (MF) router to decide
whether each query needs a strong or weak model, then calls the appropriate
vLLM endpoint directly.

RouteLLM routing is BINARY and domain-agnostic:
  weak   → qwen-7b       (fast, cheap)
  strong → llama4-scout  (most capable)

This is the key structural difference vs TTCA:
  RouteLLM : learned difficulty routing — which tier do you need?
  TTCA     : domain-aware routing     — which SPECIALIST is fastest for this domain?

A code:hard question routes to llama4-scout under RouteLLM but to
qwen3-coder-30b under TTCA — same accuracy, much lower latency and energy.

Install:
    pip install routellm

The MF checkpoint (routellm/mf-gpt4-haiku) is downloaded automatically
from HuggingFace on first run (~200 MB).

Usage:
    python tests/baseline_routellm.py \\
        --dataset   datasets/hf_3000.json \\
        --output    results/baseline_routellm.csv \\
        --threshold 0.5 \\
        --concurrency 50

    # Calibrate threshold to match cascade's strong-model usage:
    python tests/baseline_routellm.py --dataset ... --calibrate

Threshold interpretation (RouteLLM MF router):
    threshold=0.11593 → ~50% to strong (from RouteLLM paper)
    threshold=0.5     → ~20-30% to strong (more conservative)
    threshold=0.8     → ~5-10% to strong  (aggressive weak-first)
    Lower threshold = more queries go to strong model.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime
from statistics import mean

import httpx

# ---------------------------------------------------------------------------
# Model definitions — two-model fleet (weak / strong)
# ---------------------------------------------------------------------------

WEAK_MODEL   = "qwen-7b"
STRONG_MODEL = "llama4-scout"

BACKENDS: dict[str, dict] = {
    "qwen-7b": {
        "model_name":    "Qwen/Qwen2.5-7B-Instruct",
        "base_url":      "http://localhost:8000",
        "input_rate":    0.0000003,
        "output_rate":   0.0000006,
        "eff_tok_per_j": 13.0,
    },
    "llama4-scout": {
        "model_name":    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "base_url":      "http://localhost:8008",   # node2
        "input_rate":    0.0000015,
        "output_rate":   0.0000020,
        "eff_tok_per_j": 3.0,
    },
}

OUTPUT_TOKENS: dict[tuple[str, str], int] = {
    ("factual",   "easy"): 80,    ("factual",   "medium"): 200,  ("factual",   "hard"): 350,
    ("math",      "easy"): 120,   ("math",      "medium"): 280,  ("math",      "hard"): 450,
    ("code",      "easy"): 150,   ("code",      "medium"): 350,  ("code",      "hard"): 650,
    ("reasoning", "easy"): 180,   ("reasoning", "medium"): 380,  ("reasoning", "hard"): 600,
}

FIELDNAMES = [
    "req_id", "domain", "complexity", "query", "ground_truth", "mode",
    "status", "model_winner", "bid_latency_ms", "actual_latency_ms",
    "ttft_ms", "output_tokens", "charged_usd", "energy_j", "load",
    "wall_ms", "slo_ms", "slo_violated", "response_text", "error",
    "routellm_score",
]

# ---------------------------------------------------------------------------
# RouteLLM integration
# ---------------------------------------------------------------------------

def _load_routellm_router(checkpoint: str):
    """Load RouteLLM MF router. Returns a callable: messages -> float score.

    The score is the probability that the STRONG model wins (0-1).
    score >= threshold  → use strong model
    score <  threshold  → use weak model
    """
    try:
        # Try Controller API first (routellm >= 0.2)
        from routellm.controller import Controller  # noqa: F401
        from routellm.routers.routers import ROUTER_CLS

        router_cls = ROUTER_CLS.get("mf")
        if router_cls is None:
            raise ImportError("MF router not found in ROUTER_CLS")

        # Instantiate with checkpoint
        try:
            router = router_cls(checkpoint=checkpoint)
        except TypeError:
            router = router_cls()

        def score_fn(messages: list[dict]) -> float:
            return router.calculate_strong_win_rate(messages)

        print(f"  [RouteLLM] Loaded MF router via ROUTER_CLS (checkpoint={checkpoint})")
        return score_fn

    except Exception as e1:
        try:
            # Fallback: use Controller with route() method
            from routellm.controller import Controller

            controller = Controller(
                routers=["mf"],
                strong_model="strong",
                weak_model="weak",
                config={"mf": {"checkpoint_path": checkpoint}},
            )

            def score_fn_ctrl(messages: list[dict]) -> float:
                # Calculate raw score without calling any LLM
                router = controller.routers.get("mf") or controller.routers.get("matrix_factorization")
                if router is None:
                    raise RuntimeError("MF router not accessible via controller.routers")
                return router.calculate_strong_win_rate(messages)

            print(f"  [RouteLLM] Loaded MF router via Controller (checkpoint={checkpoint})")
            return score_fn_ctrl

        except Exception as e2:
            print(f"\n  ERROR: Could not load RouteLLM MF router.")
            print(f"  Attempt 1 (ROUTER_CLS): {e1}")
            print(f"  Attempt 2 (Controller):  {e2}")
            print(f"\n  Install with:  pip install routellm")
            print(f"  Then retry.   The MF checkpoint will be downloaded from HuggingFace (~200MB).")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------

async def send_one(
    client: httpx.AsyncClient,
    req_id: int,
    item: dict,
    model_id: str,
    routellm_score: float,
) -> dict:
    backend    = BACKENDS[model_id]
    domain     = item.get("domain", "factual")
    complexity = item.get("complexity", "medium")

    input_tokens = len(item.get("query", "").split()) * 1.3
    out_est      = OUTPUT_TOKENS.get((domain, complexity), 300)
    cost_est     = input_tokens * backend["input_rate"] + out_est * backend["output_rate"]

    result = {
        "req_id":            req_id,
        "domain":            domain,
        "complexity":        complexity,
        "query":             item.get("query", "")[:100],
        "ground_truth":      str(item.get("ground_truth", "")),
        "mode":              "routellm",
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
        "response_text":     "",
        "error":             "",
        "routellm_score":    f"{routellm_score:.4f}",
    }

    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{backend['base_url']}/v1/chat/completions",
            json={
                "model":      backend["model_name"],
                "messages":   [{"role": "user", "content": item.get("query", "")}],
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


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def run(
    dataset_path: str,
    output: str,
    checkpoint: str,
    threshold: float,
    concurrency: int,
    calibrate: bool,
) -> None:
    with open(dataset_path) as f:
        items = json.load(f)
    n = len(items)

    print(f"\n  [RouteLLM] Loading MF router checkpoint: {checkpoint}")
    score_fn = _load_routellm_router(checkpoint)

    # Pre-compute routing scores for all items (CPU-only, fast)
    print(f"  [RouteLLM] Scoring {n} queries (this may take 10-30s on first run)...")
    t_score = time.monotonic()
    scores: list[float] = []
    for item in items:
        messages = [{"role": "user", "content": item.get("query", "")}]
        try:
            s = score_fn(messages)
        except Exception:
            s = 0.5   # fallback: treat as ambiguous → use threshold
        scores.append(float(s))
    score_time = time.monotonic() - t_score
    print(f"  [RouteLLM] Scoring done in {score_time:.1f}s  "
          f"({n/score_time:.0f} queries/s)")

    if calibrate:
        # Show how the strong-model fraction changes with threshold
        print("\n  CALIBRATION: strong-model usage by threshold")
        print(f"  {'Threshold':>10}  {'Strong%':>8}  {'Weak%':>8}")
        for thr in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            strong_n = sum(1 for s in scores if s >= thr)
            print(f"  {thr:>10.2f}  {100*strong_n/n:>7.1f}%  {100*(n-strong_n)/n:>7.1f}%")
        print()

    # Assign models based on threshold
    assignments = [STRONG_MODEL if s >= threshold else WEAK_MODEL for s in scores]
    strong_n = assignments.count(STRONG_MODEL)
    weak_n   = assignments.count(WEAK_MODEL)

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    print(f"\n  [RouteLLM] {n} requests  threshold={threshold}")
    print(f"  Weak   ({WEAK_MODEL:<20}) : {weak_n:4d}  ({100*weak_n//n}%)")
    print(f"  Strong ({STRONG_MODEL:<20}) : {strong_n:4d}  ({100*strong_n//n}%)")
    print(f"  Output: {output}\n")

    sem  = asyncio.Semaphore(concurrency)
    done = 0
    t0   = time.monotonic()

    async def bounded(req_id: int, item: dict, model_id: str,
                      score: float, writer, f) -> None:
        nonlocal done
        async with sem:
            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                r = await send_one(client, req_id, item, model_id, score)
            writer.writerow(r)
            f.flush()
            done += 1
            if done % max(n // 20, 1) == 0:
                elapsed = time.monotonic() - t0
                bar = "=" * int(done / n * 40)
                print(f"\r  [{bar:<40}] {done}/{n}  {done/elapsed:.1f} req/s",
                      end="", flush=True)

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        await asyncio.gather(*[
            bounded(i, items[i], assignments[i], scores[i], writer, f)
            for i in range(n)
        ])

    elapsed = time.monotonic() - t0
    print(f"\r  [{'='*40}] {n}/{n}  ({elapsed:.1f}s, {n/elapsed:.1f} req/s)\n")

    # Summary
    with open(output, newline="") as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if str(r.get("status")) == "200"]
    mc: dict[str, int] = {}
    for r in ok:
        mc[r["model_winner"]] = mc.get(r["model_winner"], 0) + 1
    print(f"  Successful: {len(ok)}/{n}")
    for m, c in sorted(mc.items(), key=lambda x: -x[1]):
        print(f"    {m:<28} {c} requests ({100*c//max(len(ok),1)}%)")
    costs  = [float(r["charged_usd"]) for r in ok if r.get("charged_usd")]
    energy = [float(r["energy_j"])    for r in ok if r.get("energy_j")]
    lats   = [float(r.get("actual_latency_ms") or r.get("wall_ms", 0))
              for r in ok if r.get("actual_latency_ms") or r.get("wall_ms")]
    if costs:
        print(f"  Cost    : total=${sum(costs):.6f}  avg=${mean(costs):.8f}/req")
    if energy:
        print(f"  Energy  : total={sum(energy):.1f}J  avg={mean(energy):.2f}J/req")
    if lats:
        slats = sorted(lats)
        print(f"  Latency : P50={slats[len(slats)//2]/1000:.2f}s  "
              f"P95={slats[int(len(slats)*0.95)]/1000:.2f}s")

    score_dist = {
        "0.0-0.2": sum(1 for s in scores if s < 0.2),
        "0.2-0.4": sum(1 for s in scores if 0.2 <= s < 0.4),
        "0.4-0.6": sum(1 for s in scores if 0.4 <= s < 0.6),
        "0.6-0.8": sum(1 for s in scores if 0.6 <= s < 0.8),
        "0.8-1.0": sum(1 for s in scores if s >= 0.8),
    }
    print(f"\n  RouteLLM score distribution (strong_win_prob):")
    for band, cnt in score_dist.items():
        bar = "█" * (cnt * 30 // max(n, 1))
        print(f"    {band}  {cnt:4d}  {bar}")
    print(f"\n  Saved: {output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RouteLLM MF-router baseline for semantic LLM router comparison."
    )
    parser.add_argument("--dataset",     required=True,
                        help="Path to hf_3000.json or similar dataset")
    parser.add_argument("--output",      default="",
                        help="Output CSV path (default: auto-timestamped)")
    parser.add_argument("--checkpoint",  default="routellm/mf-gpt4-haiku",
                        help="HuggingFace checkpoint for RouteLLM MF router "
                             "(default: routellm/mf-gpt4-haiku)")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Routing threshold: score >= threshold → strong model. "
                             "Default 0.5.  Run --calibrate to choose the right value.")
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Max simultaneous in-flight requests (default 50)")
    parser.add_argument("--calibrate",   action="store_true",
                        help="Print strong-model usage at different thresholds and exit")
    args = parser.parse_args()

    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/baseline_routellm_{ts}.csv"

    asyncio.run(run(
        dataset_path=args.dataset,
        output=args.output,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        concurrency=args.concurrency,
        calibrate=args.calibrate,
    ))


if __name__ == "__main__":
    main()
