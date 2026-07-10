"""
baseline_carrot.py — CARROT: Cost Aware Rate Optimal Router (Somerstep et al., 2025).
arXiv: 2502.03261  |  github.com/somerstep/CARROT

Adapted for our vLLM fleet (6 models, localhost endpoints).
Trains on eval_matrix.csv produced by eval_all_models.py (S3b step).

Core algorithm:
  1. Embed queries with sentence-transformers (all-MiniLM-L6-v2).
  2. Fit per-model LogisticRegression to predict P(correct | query).
  3. Fit per-model LinearRegression to predict E[output_tokens | query].
  4. Route each query to: argmax_j { (1-mu) * P̂_j(q) − mu * norm_cost_j(q) }
     where mu ∈ [0,1] traces the Pareto frontier (0=quality, 1=cost).

Usage:
    python tests/baseline_carrot.py \\
        --dataset     /tmp/svd_workload.json \\
        --eval-matrix results/static_vs_dynamic_X/eval_matrix.csv \\
        --mu          0.3 \\
        --concurrency 50 \\
        --output      results/static_vs_dynamic_X/baseline_carrot.csv
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
import numpy as np

# ---------------------------------------------------------------------------
# Model backends — must match dynamic_provisioner.py MODEL_CATALOG
# ---------------------------------------------------------------------------

BACKENDS: dict[str, dict] = {
    "qwen-7b": {
        "model_name":    "Qwen/Qwen2.5-7B-Instruct",
        "base_url":      "http://localhost:8000",
        "input_rate":    5e-8,    # $0.05/M input tokens
        "output_rate":   1e-7,    # $0.10/M output tokens
        "eff_tok_per_j": 13.0,
    },
    "deepseek-r1-7b": {
        "model_name":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "base_url":      "http://localhost:8001",
        "input_rate":    6e-8,    # $0.06/M
        "output_rate":   1.4e-7,  # $0.14/M
        "eff_tok_per_j": 13.0,
    },
    "qwen3-coder-30b": {
        "model_name":    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "base_url":      "http://localhost:8002",
        "input_rate":    1.5e-7,  # $0.15/M
        "output_rate":   6e-7,    # $0.60/M
        "eff_tok_per_j": 12.0,
    },
    "gemma-3-27b": {
        "model_name":    "google/gemma-3-27b-it",
        "base_url":      "http://localhost:8003",
        "input_rate":    8e-8,    # $0.08/M
        "output_rate":   1.6e-7,  # $0.16/M
        "eff_tok_per_j": 5.0,
    },
    "deepseek-r1-14b": {
        "model_name":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "base_url":      "http://localhost:8004",
        "input_rate":    1e-7,    # $0.10/M
        "output_rate":   2.5e-7,  # $0.25/M
        "eff_tok_per_j": 6.0,
    },
    "llama4-scout": {
        "model_name":    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "base_url":      "",      # filled in via --node2-host
        "input_rate":    1e-7,    # $0.10/M
        "output_rate":   3e-7,    # $0.30/M
        "eff_tok_per_j": 3.0,
    },
}

# Ordered list of model IDs (determines column indexing)
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
# Phase 1: Train predictors from eval_matrix.csv
# ---------------------------------------------------------------------------

def load_eval_matrix(eval_matrix_path: str, dataset: list[dict]
                     ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Returns:
        queries     — list of query strings (one per training row)
        correct_mat — [N, M] bool array: correct_mat[i, j] = is model j correct on query i
        tokens_mat  — [N, M] float array: output token counts
    """
    # Index dataset by req_id (= str(index))
    id_to_item = {str(idx): item for idx, item in enumerate(dataset)}

    # Read eval_matrix rows, group by req_id
    from collections import defaultdict
    by_req: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(eval_matrix_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = row.get("req_id", "")
            mid = row.get("model_id", "")
            if rid in id_to_item and mid in BACKENDS:
                by_req[rid][mid] = row

    # Build aligned arrays
    queries: list[str] = []
    correct_rows: list[list[bool]] = []
    tokens_rows:  list[list[float]] = []

    for rid, model_rows in sorted(by_req.items(), key=lambda x: int(x[0])):
        item = id_to_item[rid]
        queries.append(item["query"])
        correct_row = []
        tokens_row  = []
        for mid in MODEL_IDS:
            row = model_rows.get(mid, {})
            ic = row.get("is_correct", "false") == "true"
            ot = float(row.get("output_tokens", 0) or 0)
            correct_row.append(ic)
            tokens_row.append(ot)
        correct_rows.append(correct_row)
        tokens_rows.append(tokens_row)

    correct_mat = np.array(correct_rows, dtype=np.float32)  # [N, M]
    tokens_mat  = np.array(tokens_rows,  dtype=np.float32)  # [N, M]
    return queries, correct_mat, tokens_mat


def train_predictors(train_queries: list[str],
                     correct_mat: np.ndarray,
                     tokens_mat:  np.ndarray):
    """
    Embed queries, fit per-model classifiers and regressors.
    Returns (encoder, classifiers, regressors, train_embeddings).
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression, Ridge

    print("  [CARROT] Embedding training queries...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = encoder.encode(train_queries, batch_size=64, show_progress_bar=False,
                         normalize_embeddings=True)  # [N, D]

    n_models = len(MODEL_IDS)
    classifiers: list = []
    regressors:  list = []

    print(f"  [CARROT] Training {n_models} per-model classifiers + regressors...")
    for j, mid in enumerate(MODEL_IDS):
        y_clf = correct_mat[:, j]
        y_reg = tokens_mat[:, j]

        # Skip models with no training data
        if y_clf.sum() == 0 or (1 - y_clf).sum() == 0:
            print(f"    {mid}: degenerate labels ({y_clf.mean():.2f} accuracy) — using constant")
            from sklearn.dummy import DummyClassifier, DummyRegressor
            clf = DummyClassifier(strategy="most_frequent").fit(emb, y_clf)
            reg = DummyRegressor().fit(emb, y_reg)
        else:
            clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs").fit(emb, y_clf)
            reg = Ridge(alpha=1.0).fit(emb, y_reg)

        classifiers.append(clf)
        regressors.append(reg)
        acc = y_clf.mean()
        print(f"    {mid:<24} train_acc={acc:.3f}  mean_tokens={y_reg.mean():.0f}")

    return encoder, classifiers, regressors


# ---------------------------------------------------------------------------
# Phase 2: Routing decisions (per-query, vectorized)
# ---------------------------------------------------------------------------

def compute_assignments(test_queries: list[str],
                        encoder,
                        classifiers: list,
                        regressors:  list,
                        mu: float) -> list[str]:
    """
    For each test query, compute CARROT score per model and return the winner.
    score_j(q) = (1 - mu) * P̂_j(correct | q)  −  mu * normalized_cost_j(q)
    """
    print(f"  [CARROT] Embedding {len(test_queries)} test queries (mu={mu})...")
    emb = encoder.encode(test_queries, batch_size=64, show_progress_bar=False,
                         normalize_embeddings=True)  # [N_test, D]

    n_models = len(MODEL_IDS)
    n_test   = len(test_queries)

    # Accuracy predictions: [N_test, n_models]
    acc_preds = np.zeros((n_test, n_models), dtype=np.float32)
    for j, clf in enumerate(classifiers):
        proba = clf.predict_proba(emb)
        pos_class_idx = list(clf.classes_).index(1.0) if 1.0 in clf.classes_ else -1
        acc_preds[:, j] = proba[:, pos_class_idx] if pos_class_idx >= 0 else proba[:, -1]

    # Cost predictions (normalized to [0,1] across models): [N_test, n_models]
    cost_raw = np.zeros((n_test, n_models), dtype=np.float32)
    for j, (mid, reg) in enumerate(zip(MODEL_IDS, regressors)):
        backend    = BACKENDS[mid]
        out_tokens = np.maximum(reg.predict(emb), 0)            # predicted output tokens
        in_tokens  = np.array([len(q.split()) * 1.3 for q in test_queries])
        cost_raw[:, j] = (in_tokens * backend["input_rate"]
                          + out_tokens * backend["output_rate"])

    cost_max = cost_raw.max(axis=1, keepdims=True).clip(min=1e-12)
    cost_norm = cost_raw / cost_max                              # [N_test, n_models]

    # Exclude degenerate models (constant acc prediction — DummyClassifier)
    # They have acc≈0 and predicted cost=0, so score=0 beats all real models → spurious wins.
    valid_mask = np.ones(n_models, dtype=bool)
    for j in range(n_models):
        if n_test > 1 and np.std(acc_preds[:, j]) < 1e-6:
            valid_mask[j] = False
            print(f"    Excluding {MODEL_IDS[j]} from routing (degenerate predictions)")
    if not valid_mask.any():
        valid_mask[:] = True  # fallback: keep all if everything is degenerate

    # CARROT score (higher = better)
    scores = (1.0 - mu) * acc_preds - mu * cost_norm            # [N_test, n_models]
    scores[:, ~valid_mask] = -np.inf                            # mask degenerate models
    chosen_j = scores.argmax(axis=1)                            # [N_test]

    assignments = [MODEL_IDS[j] for j in chosen_j]

    # Distribution summary
    from collections import Counter
    dist = Counter(assignments)
    print("  [CARROT] Routing distribution:")
    for mid, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {mid:<24} {cnt:4d} ({100*cnt//len(test_queries):2d}%)")

    return assignments


# ---------------------------------------------------------------------------
# Phase 3: Execute requests (same async pattern as baseline_cascade.py)
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
        "mode": "carrot", "status": "", "model_winner": model_id,
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


async def run_requests(dataset: list[dict], assignments: list[str],
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
            bounded(i, dataset[i], assignments[i], writer, f)
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
        description="CARROT router baseline (Somerstep et al. 2025)")
    parser.add_argument("--dataset",      required=True,
                        help="Workload JSON (list of {query, domain, complexity, ...})")
    parser.add_argument("--eval-matrix",  default=None,
                        help="eval_matrix.csv from eval_all_models.py (required for training)")
    parser.add_argument("--mu",           type=float, default=0.3,
                        help="Cost/quality trade-off: 0=max quality, 1=min cost (default 0.3)")
    parser.add_argument("--concurrency",  type=int, default=50)
    parser.add_argument("--output",       default="")
    parser.add_argument("--node2-host",   default=None,
                        help="Hostname of node2 for llama4-scout (e.g. sophia-gpu-09)")
    args = parser.parse_args()

    if args.node2_host:
        BACKENDS["llama4-scout"]["base_url"] = f"http://{args.node2_host}:8005"
    else:
        # Without node2, remove llama4-scout from routing pool
        MODEL_IDS.remove("llama4-scout")
        del BACKENDS["llama4-scout"]
        print("  [CARROT] No --node2-host: llama4-scout excluded from routing pool.")

    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/baseline_carrot_{ts}.csv"

    print(f"\n  [CARROT] Loading dataset: {args.dataset}")
    with open(args.dataset) as f:
        dataset: list[dict] = json.load(f)
    n = len(dataset)
    print(f"  [CARROT] {n} queries, mu={args.mu}, concurrency={args.concurrency}")

    # ── Phase 1: Train ────────────────────────────────────────────────────────
    if args.eval_matrix and os.path.exists(args.eval_matrix):
        print(f"\n  [CARROT] Training on {args.eval_matrix}...")
        train_queries, correct_mat, tokens_mat = load_eval_matrix(
            args.eval_matrix, dataset)
        print(f"  [CARROT] Training rows: {len(train_queries)} queries × {len(MODEL_IDS)} models")
        encoder, classifiers, regressors = train_predictors(
            train_queries, correct_mat, tokens_mat)
    else:
        print(f"  [CARROT] WARNING: eval_matrix not found at {args.eval_matrix!r}")
        print("  [CARROT] Falling back to cheapest-model routing (qwen-7b).")
        assignments = ["qwen-7b"] * n
        if not args.output:
            args.output = f"results/baseline_carrot_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        asyncio.run(run_requests(dataset, assignments, args.output, args.concurrency))
        return

    # ── Phase 2: Compute routing assignments ──────────────────────────────────
    test_queries = [item["query"] for item in dataset]
    assignments  = compute_assignments(test_queries, encoder, classifiers,
                                       regressors, mu=args.mu)

    # ── Phase 3: Execute requests ─────────────────────────────────────────────
    print(f"\n  [CARROT] Executing {n} requests...")
    asyncio.run(run_requests(dataset, assignments, args.output, args.concurrency))


if __name__ == "__main__":
    main()
