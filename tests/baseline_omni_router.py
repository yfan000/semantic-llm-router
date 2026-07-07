"""baseline_omni_router.py — OmniRouter: Budget and Performance Controllable Multi-LLM Routing.
arXiv: 2502.20576  (Mei et al., 2025)  |  github.com/dongyuanjushi/OmniRouter

Adapted for our vLLM fleet (6 models, localhost endpoints).
Trains on eval_matrix.csv produced by eval_all_models.py (S3b step).

Core algorithm:
  1. Build KNN index from eval_matrix query embeddings + per-model accuracy/cost labels.
  2. For each query: retrieve K nearest neighbors → predict accuracy & cost per model.
  3. Lagrangian dual optimizer (over the full batch):
       Minimize total cost s.t.
         avg_accuracy >= alpha      (quality floor)
         queries_routed_to_j <= L_j (capacity per model)
       Dual variables λ1 (quality), λ2_j (capacity) updated by projected gradient ascent.
  4. Execute batch with pre-determined model assignments.

Key differences from the paper:
  - KNN uses numpy cosine similarity instead of FAISS (fast enough for <=3000 queries)
  - Target models are our 6 vLLM models, not the paper's Ollama/API pool

Usage:
    python tests/baseline_omni_router.py \\
        --dataset     /tmp/svd_workload.json \\
        --eval-matrix results/static_vs_dynamic_X/eval_matrix.csv \\
        --alpha       0.75 \\
        --concurrency 50 \\
        --output      results/static_vs_dynamic_X/baseline_omni_router.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from collections import Counter
from datetime import datetime
from statistics import mean

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Model backends — must match dynamic_provisioner.py MODEL_CATALOG
# ---------------------------------------------------------------------------

BACKENDS: dict[str, dict] = {
    "qwen-7b": {
        "model_name":    "Qwen/Qwen2.5-7B-Instruct",
        "base_url":      "http://localhost:8000",
        "input_rate":    5e-8,
        "output_rate":   1e-7,
        "eff_tok_per_j": 13.0,
    },
    "deepseek-r1-7b": {
        "model_name":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "base_url":      "http://localhost:8001",
        "input_rate":    6e-8,
        "output_rate":   1.4e-7,
        "eff_tok_per_j": 13.0,
    },
    "qwen3-coder-30b": {
        "model_name":    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "base_url":      "http://localhost:8002",
        "input_rate":    1.5e-7,
        "output_rate":   6e-7,
        "eff_tok_per_j": 12.0,
    },
    "gemma-3-27b": {
        "model_name":    "google/gemma-3-27b-it",
        "base_url":      "http://localhost:8003",
        "input_rate":    8e-8,
        "output_rate":   1.6e-7,
        "eff_tok_per_j": 5.0,
    },
    "deepseek-r1-14b": {
        "model_name":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "base_url":      "http://localhost:8004",
        "input_rate":    1e-7,
        "output_rate":   2.5e-7,
        "eff_tok_per_j": 6.0,
    },
    "llama4-scout": {
        "model_name":    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "base_url":      "",
        "input_rate":    1e-7,
        "output_rate":   3e-7,
        "eff_tok_per_j": 3.0,
    },
}

MODEL_IDS: list[str] = list(BACKENDS.keys())

OUTPUT_TOKENS_EST: dict[tuple[str, str], int] = {
    ("factual", "easy"): 80,     ("factual", "medium"): 200,  ("factual", "hard"): 350,
    ("math",    "easy"): 120,    ("math",    "medium"): 280,  ("math",    "hard"): 450,
    ("code",    "easy"): 150,    ("code",    "medium"): 350,  ("code",    "hard"): 650,
    ("reasoning","easy"):180,    ("reasoning","medium"):380,  ("reasoning","hard"):600,
}

FIELDNAMES = [
    "req_id", "domain", "complexity", "query", "ground_truth", "mode",
    "status", "model_winner", "bid_latency_ms", "actual_latency_ms",
    "ttft_ms", "output_tokens", "charged_usd", "energy_j", "load",
    "wall_ms", "slo_ms", "slo_violated", "retries", "response_text", "error",
]


# ---------------------------------------------------------------------------
# Phase 1: Build KNN index from eval_matrix
# ---------------------------------------------------------------------------

def load_eval_matrix(eval_matrix_path: str, dataset: list[dict]
                     ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        queries     — list of N_train query strings
        emb_train   — [N_train, D] float32 embeddings
        acc_train   — [N_train, M] float32 accuracy (0/1) per model
        cost_train  — [N_train, M] float32 dollar cost per model
    """
    from sentence_transformers import SentenceTransformer

    id_to_item = {str(idx): item for idx, item in enumerate(dataset)}

    from collections import defaultdict
    by_req: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(eval_matrix_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = row.get("req_id", "")
            mid = row.get("model_id", "")
            if rid in id_to_item and mid in BACKENDS:
                by_req[rid][mid] = row

    queries: list[str] = []
    acc_rows:  list[list[float]] = []
    cost_rows: list[list[float]] = []

    for rid, model_rows in sorted(by_req.items(), key=lambda x: int(x[0])):
        item = id_to_item[rid]
        queries.append(item["query"])
        acc_row  = []
        cost_row = []
        for mid in MODEL_IDS:
            row     = model_rows.get(mid, {})
            correct = float(row.get("is_correct", "false") == "true")
            out_tok = float(row.get("output_tokens", 0) or 0)
            in_tok  = len(item["query"].split()) * 1.3
            backend = BACKENDS[mid]
            cost    = in_tok * backend["input_rate"] + out_tok * backend["output_rate"]
            acc_row.append(correct)
            cost_row.append(cost)
        acc_rows.append(acc_row)
        cost_rows.append(cost_row)

    print(f"  [OmniRouter] Embedding {len(queries)} training queries...")
    encoder  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb      = encoder.encode(queries, batch_size=64, show_progress_bar=False,
                              normalize_embeddings=True).astype(np.float32)
    acc_mat  = np.array(acc_rows,  dtype=np.float32)   # [N_train, M]
    cost_mat = np.array(cost_rows, dtype=np.float32)   # [N_train, M]

    return queries, emb, acc_mat, cost_mat, encoder


# ---------------------------------------------------------------------------
# Phase 2: KNN prediction for test batch
# ---------------------------------------------------------------------------

def knn_predict(emb_train: np.ndarray,
                acc_train: np.ndarray,
                cost_train: np.ndarray,
                emb_test: np.ndarray,
                K: int = 16,
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Numpy cosine KNN (emb_train is already L2-normalized).
    Returns acc_pred [N_test, M] and cost_pred [N_test, M].
    """
    # emb_test may not be normalized yet — normalize it
    norms = np.linalg.norm(emb_test, axis=1, keepdims=True).clip(min=1e-12)
    emb_test_n = emb_test / norms

    # Similarity matrix [N_test, N_train]
    sim = emb_test_n @ emb_train.T  # cosine similarity (emb_train already normalized)

    N_test  = sim.shape[0]
    n_models = acc_train.shape[1]
    acc_pred  = np.zeros((N_test, n_models), dtype=np.float32)
    cost_pred = np.zeros((N_test, n_models), dtype=np.float32)

    # Process in chunks to avoid large memory spike
    chunk = 256
    for start in range(0, N_test, chunk):
        end       = min(start + chunk, N_test)
        sim_chunk = sim[start:end]                         # [chunk, N_train]
        top_k_idx = np.argpartition(sim_chunk, -K, axis=1)[:, -K:]  # [chunk, K]

        for i in range(end - start):
            idx    = top_k_idx[i]
            w      = np.maximum(sim_chunk[i, idx], 0)     # [K], clip negatives
            w_sum  = w.sum()
            if w_sum < 1e-12:
                w = np.ones(K, dtype=np.float32) / K
                w_sum = 1.0
            acc_pred[start + i]  = (w[:, None] * acc_train[idx]).sum(axis=0)  / w_sum
            cost_pred[start + i] = (w[:, None] * cost_train[idx]).sum(axis=0) / w_sum

    return acc_pred, cost_pred


# ---------------------------------------------------------------------------
# Phase 3: Lagrangian dual optimizer
# ---------------------------------------------------------------------------

def lagrangian_optimize(acc_pred: np.ndarray,
                        cost_pred: np.ndarray,
                        alpha: float = 0.75,
                        max_iter: int = 300,
                        lr: float = 0.05,
                        capacity_slack: float = 1.5,
                        ) -> np.ndarray:
    """
    Global batch optimizer (OmniRouter §3.2):
      minimize   Σ_i cost[i, assign[i]]
      subject to avg accuracy >= alpha
                 Σ_i 1{assign[i]==j} <= L_j  for all j

    Returns assign: [N_test] int array of model indices.
    """
    N, M = acc_pred.shape
    L    = max(1, int(N / M * capacity_slack))   # soft capacity per model

    lam1   = 0.0           # quality dual variable
    lam2   = np.zeros(M, dtype=np.float64)  # capacity dual per model

    assign = np.zeros(N, dtype=np.int32)

    for it in range(max_iter):
        # Assignment step: each query → argmin_j (cost - λ1*acc/N + λ2_j)
        adjusted = (cost_pred.astype(np.float64)
                    - lam1 * acc_pred.astype(np.float64) / N
                    + lam2[None, :])
        assign = adjusted.argmin(axis=1).astype(np.int32)

        # Dual update
        avg_acc = acc_pred[np.arange(N), assign].mean()
        counts  = np.bincount(assign, minlength=M).astype(np.float64)

        dlam1 = alpha - avg_acc
        dlam2 = counts - L

        lam1 = max(0.0, lam1 + lr * dlam1)
        lam2 = np.maximum(0.0, lam2 + lr * dlam2)

        # Convergence: small dual gradient
        if abs(dlam1) < 1e-4 and np.abs(dlam2).max() < 1e-4 * L:
            print(f"  [OmniRouter] Converged at iter {it+1}")
            break

    avg_acc_final = acc_pred[np.arange(N), assign].mean()
    total_cost    = cost_pred[np.arange(N), assign].sum()
    dist = Counter(MODEL_IDS[j] for j in assign)
    print(f"  [OmniRouter] Optimization done: avg_acc={avg_acc_final:.3f} "
          f"(target={alpha}), total_cost=${total_cost:.6f}")
    print("  [OmniRouter] Routing distribution:")
    for mid, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {mid:<24} {cnt:4d} ({100*cnt//N:2d}%)")

    return assign


# ---------------------------------------------------------------------------
# Phase 4: Execute requests (same async pattern as baseline_cascade.py)
# ---------------------------------------------------------------------------

async def send_one(client: httpx.AsyncClient, req_id: int,
                   item: dict, model_id: str) -> dict:
    backend    = BACKENDS[model_id]
    domain     = item.get("domain", "")
    complexity = item.get("complexity", "")
    in_tokens  = len(item["query"].split()) * 1.3
    out_est    = OUTPUT_TOKENS_EST.get((domain, complexity), 300)
    cost_est   = in_tokens * backend["input_rate"] + out_est * backend["output_rate"]

    result = {
        "req_id": str(req_id), "domain": domain, "complexity": complexity,
        "query": item["query"][:100], "ground_truth": str(item.get("ground_truth", "")),
        "mode": "omni_router", "status": "", "model_winner": model_id,
        "bid_latency_ms": "", "actual_latency_ms": "", "ttft_ms": "",
        "output_tokens": "", "charged_usd": f"{cost_est:.8f}", "energy_j": "",
        "load": "", "wall_ms": "", "slo_ms": "", "slo_violated": "",
        "retries": "0", "response_text": "", "error": "",
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
            actual_cost = (in_tokens * backend["input_rate"]
                           + out_tokens * backend["output_rate"])
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


async def run_requests(dataset: list[dict], model_assignments: list[str],
                       output: str, concurrency: int) -> None:
    n    = len(dataset)
    sem  = asyncio.Semaphore(concurrency)
    done = 0
    t0   = time.monotonic()

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    async def bounded(req_id: int, item: dict, model_id: str, writer, f) -> None:
        nonlocal done
        async with sem:
            async with httpx.AsyncClient(timeout=180.0, trust_env=False) as client:
                r = await send_one(client, req_id, item, model_id)
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
            bounded(i, dataset[i], model_assignments[i], writer, f)
            for i in range(n)
        ])

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
        print(f"    {m:<24} {c} ({100*c//max(len(ok),1)}%)")
    costs = [float(r["charged_usd"]) for r in ok if r.get("charged_usd")]
    if costs:
        print(f"  Total cost: ${sum(costs):.6f}  avg=${mean(costs):.8f}")
    print(f"\n  Saved: {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OmniRouter baseline (Mei et al. 2025)")
    parser.add_argument("--dataset",      required=True,
                        help="Workload JSON (list of {query, domain, complexity, ...})")
    parser.add_argument("--eval-matrix",  default=None,
                        help="eval_matrix.csv from eval_all_models.py")
    parser.add_argument("--alpha",        type=float, default=0.75,
                        help="Target average accuracy constraint (default 0.75)")
    parser.add_argument("--K",            type=int, default=16,
                        help="Number of KNN neighbors for retrieval (default 16)")
    parser.add_argument("--max-iter",     type=int, default=300,
                        help="Max Lagrangian optimizer iterations (default 300)")
    parser.add_argument("--capacity-slack", type=float, default=1.5,
                        help="L_j = N/M * slack (default 1.5 — 50%% above equal split)")
    parser.add_argument("--concurrency",  type=int, default=50)
    parser.add_argument("--output",       default="")
    parser.add_argument("--node2-host",   default=None,
                        help="Hostname of node2 for llama4-scout (e.g. sophia-gpu-09)")
    args = parser.parse_args()

    if args.node2_host:
        BACKENDS["llama4-scout"]["base_url"] = f"http://{args.node2_host}:8005"
    else:
        MODEL_IDS.remove("llama4-scout")
        del BACKENDS["llama4-scout"]
        print("  [OmniRouter] No --node2-host: llama4-scout excluded from routing pool.")

    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/baseline_omni_router_{ts}.csv"

    print(f"\n  [OmniRouter] Loading dataset: {args.dataset}")
    with open(args.dataset) as f:
        dataset: list[dict] = json.load(f)
    n = len(dataset)
    print(f"  [OmniRouter] {n} queries, alpha={args.alpha}, K={args.K}, "
          f"concurrency={args.concurrency}")

    # ── Phase 1: Build KNN index ─────────────────────────────────────────────
    if not (args.eval_matrix and os.path.exists(args.eval_matrix)):
        print(f"  [OmniRouter] WARNING: eval_matrix not found at {args.eval_matrix!r}")
        print("  [OmniRouter] Falling back to cheapest-model routing (qwen-7b).")
        assignments = ["qwen-7b"] * n
        asyncio.run(run_requests(dataset, assignments, args.output, args.concurrency))
        return

    _, emb_train, acc_train, cost_train, encoder = load_eval_matrix(
        args.eval_matrix, dataset)

    # ── Phase 2: Predict for test batch ────────────────────────────────────
    print(f"  [OmniRouter] Embedding {n} test queries...")
    test_queries = [item["query"] for item in dataset]
    emb_test = encoder.encode(test_queries, batch_size=64, show_progress_bar=False,
                              normalize_embeddings=True).astype(np.float32)

    print(f"  [OmniRouter] KNN retrieval (K={args.K})...")
    acc_pred, cost_pred = knn_predict(emb_train, acc_train, cost_train,
                                      emb_test, K=args.K)

    # ── Phase 3: Lagrangian optimization ──────────────────────────────────
    print(f"  [OmniRouter] Lagrangian optimization (alpha={args.alpha}, "
          f"max_iter={args.max_iter})...")
    assign_idx = lagrangian_optimize(
        acc_pred, cost_pred,
        alpha=args.alpha,
        max_iter=args.max_iter,
        capacity_slack=args.capacity_slack,
    )
    model_assignments = [MODEL_IDS[j] for j in assign_idx]

    # ── Phase 4: Execute requests ──────────────────────────────────────────
    print(f"\n  [OmniRouter] Executing {n} requests...")
    asyncio.run(run_requests(dataset, model_assignments, args.output, args.concurrency))


if __name__ == "__main__":
    main()
