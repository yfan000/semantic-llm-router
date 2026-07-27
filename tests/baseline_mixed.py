"""
baseline_mixed.py — Mixed routing: TTCA for code:hard, CARROT/OmniRouter ensemble elsewhere.

Strategy:
  - code:hard queries  → TTCA live router  (wins pen.$/ans by −15% vs best baseline)
  - all other queries  → CARROT (75%) or OmniRouter (25%), chosen randomly per query

Rationale: per-cell pen.$/ans analysis shows TTCA wins only in code:hard where its
accuracy advantage is large enough to offset higher cost. CARROT dominates most other
cells; OmniRouter adds diversity on factual queries.

Usage:
    python tests/baseline_mixed.py \\
        --dataset     results/RUNDIR/workload.json \\
        --eval-matrix results/RUNDIR/eval_matrix.csv \\
        --router      http://NODE1:8080 \\
        --mu          0.3 \\
        --alpha       0.75 \\
        --carrot-frac 0.75 \\
        --concurrency 50 \\
        --node2-host  NODE2 \\
        --seed        42 \\
        --output      results/RUNDIR/baseline_mixed.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import time
from collections import Counter
from datetime import datetime
from statistics import mean

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Model backends (shared by CARROT and OmniRouter routing phases)
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
        "base_url":      "",   # filled in via --node2-host
        "input_rate":    1e-7,
        "output_rate":   3e-7,
        "eff_tok_per_j": 3.0,
    },
}

MODEL_IDS: list[str] = list(BACKENDS.keys())

OUTPUT_TOKENS_EST: dict[tuple[str, str], int] = {
    ("factual",   "easy"):   80,  ("factual",   "medium"): 200, ("factual",   "hard"): 350,
    ("math",      "easy"):  120,  ("math",      "medium"): 280, ("math",      "hard"): 450,
    ("code",      "easy"):  150,  ("code",      "medium"): 350, ("code",      "hard"): 650,
    ("reasoning", "easy"):  180,  ("reasoning", "medium"): 380, ("reasoning", "hard"): 600,
}

FIELDNAMES = [
    "req_id", "domain", "complexity", "query", "ground_truth", "mode",
    "status", "model_winner", "bid_latency_ms", "actual_latency_ms",
    "ttft_ms", "output_tokens", "charged_usd", "energy_j", "load",
    "wall_ms", "slo_ms", "slo_violated", "retries", "response_text", "error",
]

TTCA_CELL = ("code", "hard")


# ---------------------------------------------------------------------------
# CARROT routing
# ---------------------------------------------------------------------------

def _load_eval_matrix_carrot(eval_matrix_path: str, dataset: list[dict]
                              ) -> tuple[list[str], np.ndarray, np.ndarray]:
    from collections import defaultdict
    id_to_item = {str(idx): item for idx, item in enumerate(dataset)}
    by_req: dict[str, dict] = defaultdict(dict)
    with open(eval_matrix_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid, mid = row.get("req_id", ""), row.get("model_id", "")
            if rid in id_to_item and mid in BACKENDS:
                by_req[rid][mid] = row

    queries, correct_rows, tokens_rows = [], [], []
    for rid, model_rows in sorted(by_req.items(), key=lambda x: int(x[0])):
        queries.append(id_to_item[rid]["query"])
        correct_rows.append([float(model_rows.get(m, {}).get("is_correct", "false") == "true")
                             for m in MODEL_IDS])
        tokens_rows.append([float(model_rows.get(m, {}).get("output_tokens", 0) or 0)
                            for m in MODEL_IDS])
    return queries, np.array(correct_rows, dtype=np.float32), np.array(tokens_rows, dtype=np.float32)


def _carrot_assignments(dataset: list[dict], eval_matrix: str,
                        mu: float, encoder) -> list[str]:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.dummy import DummyClassifier, DummyRegressor

    train_queries, correct_mat, tokens_mat = _load_eval_matrix_carrot(eval_matrix, dataset)
    print(f"  [CARROT] Training {len(MODEL_IDS)} classifiers on {len(train_queries)} rows...")
    emb_train = encoder.encode(train_queries, batch_size=64, show_progress_bar=False,
                               normalize_embeddings=True)
    classifiers, regressors = [], []
    for j, mid in enumerate(MODEL_IDS):
        y_clf, y_reg = correct_mat[:, j], tokens_mat[:, j]
        if y_clf.sum() == 0 or (1 - y_clf).sum() == 0:
            clf = DummyClassifier(strategy="most_frequent").fit(emb_train, y_clf)
            reg = DummyRegressor().fit(emb_train, y_reg)
        else:
            clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs").fit(emb_train, y_clf)
            reg = Ridge(alpha=1.0).fit(emb_train, y_reg)
        classifiers.append(clf)
        regressors.append(reg)

    test_queries = [item["query"] for item in dataset]
    print(f"  [CARROT] Embedding {len(test_queries)} test queries...")
    emb_test = encoder.encode(test_queries, batch_size=64, show_progress_bar=False,
                              normalize_embeddings=True)
    n_test = len(test_queries)
    acc_preds  = np.zeros((n_test, len(MODEL_IDS)), dtype=np.float32)
    cost_raw   = np.zeros((n_test, len(MODEL_IDS)), dtype=np.float32)
    for j, (mid, clf, reg) in enumerate(zip(MODEL_IDS, classifiers, regressors)):
        proba = clf.predict_proba(emb_test)
        pos   = list(clf.classes_).index(1.0) if 1.0 in clf.classes_ else -1
        acc_preds[:, j] = proba[:, pos] if pos >= 0 else proba[:, -1]
        out_tok = np.maximum(reg.predict(emb_test), 0)
        in_tok  = np.array([len(q.split()) * 1.3 for q in test_queries])
        cost_raw[:, j] = (in_tok * BACKENDS[mid]["input_rate"]
                          + out_tok * BACKENDS[mid]["output_rate"])

    cost_norm = cost_raw / cost_raw.max(axis=1, keepdims=True).clip(min=1e-12)
    valid = np.array([n_test <= 1 or np.std(acc_preds[:, j]) >= 1e-6
                      for j in range(len(MODEL_IDS))], dtype=bool)
    if not valid.any():
        valid[:] = True
    scores = (1.0 - mu) * acc_preds - mu * cost_norm
    scores[:, ~valid] = -np.inf
    return [MODEL_IDS[j] for j in scores.argmax(axis=1)]


# ---------------------------------------------------------------------------
# OmniRouter routing
# ---------------------------------------------------------------------------

def _load_eval_matrix_omni(eval_matrix_path: str, dataset: list[dict],
                            encoder) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from collections import defaultdict
    id_to_item = {str(idx): item for idx, item in enumerate(dataset)}
    by_req: dict[str, dict] = defaultdict(dict)
    with open(eval_matrix_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid, mid = row.get("req_id", ""), row.get("model_id", "")
            if rid in id_to_item and mid in BACKENDS:
                by_req[rid][mid] = row

    queries, acc_rows, cost_rows = [], [], []
    for rid, model_rows in sorted(by_req.items(), key=lambda x: int(x[0])):
        item = id_to_item[rid]
        queries.append(item["query"])
        acc_row, cost_row = [], []
        for mid in MODEL_IDS:
            row     = model_rows.get(mid, {})
            correct = float(row.get("is_correct", "false") == "true")
            out_tok = float(row.get("output_tokens", 0) or 0)
            in_tok  = len(item["query"].split()) * 1.3
            acc_row.append(correct)
            cost_row.append(in_tok * BACKENDS[mid]["input_rate"]
                            + out_tok * BACKENDS[mid]["output_rate"])
        acc_rows.append(acc_row)
        cost_rows.append(cost_row)

    print(f"  [OmniRouter] Embedding {len(queries)} training queries...")
    emb_train = encoder.encode(queries, batch_size=64, show_progress_bar=False,
                               normalize_embeddings=True).astype(np.float32)
    return emb_train, np.array(acc_rows, dtype=np.float32), np.array(cost_rows, dtype=np.float32)


def _knn_predict(emb_train, acc_train, cost_train, emb_test, K=16):
    norms      = np.linalg.norm(emb_test, axis=1, keepdims=True).clip(min=1e-12)
    emb_test_n = emb_test / norms
    sim        = emb_test_n @ emb_train.T
    N_test, n_models = emb_test.shape[0], acc_train.shape[1]
    acc_pred  = np.zeros((N_test, n_models), dtype=np.float32)
    cost_pred = np.zeros((N_test, n_models), dtype=np.float32)
    chunk = 256
    for s in range(0, N_test, chunk):
        e         = min(s + chunk, N_test)
        sim_chunk = sim[s:e]
        top_k     = np.argpartition(sim_chunk, -K, axis=1)[:, -K:]
        for i in range(e - s):
            idx   = top_k[i]
            w     = np.maximum(sim_chunk[i, idx], 0)
            w_sum = w.sum()
            if w_sum < 1e-12:
                w, w_sum = np.ones(K, dtype=np.float32) / K, 1.0
            acc_pred[s + i]  = (w[:, None] * acc_train[idx]).sum(0) / w_sum
            cost_pred[s + i] = (w[:, None] * cost_train[idx]).sum(0) / w_sum
    return acc_pred, cost_pred


def _lagrangian_optimize(acc_pred, cost_pred, alpha=0.75, max_iter=300,
                         lr=0.05, capacity_slack=1.5):
    N, M = acc_pred.shape
    L    = max(1, int(N / M * capacity_slack))
    lam1, lam2 = 0.0, np.zeros(M, dtype=np.float64)
    assign = np.zeros(N, dtype=np.int32)
    for it in range(max_iter):
        adj    = (cost_pred.astype(np.float64)
                  - lam1 * acc_pred.astype(np.float64) / N
                  + lam2[None, :])
        assign = adj.argmin(axis=1).astype(np.int32)
        avg_acc = acc_pred[np.arange(N), assign].mean()
        counts  = np.bincount(assign, minlength=M).astype(np.float64)
        dlam1 = alpha - avg_acc
        dlam2 = counts - L
        lam1  = max(0.0, lam1 + lr * dlam1)
        lam2  = np.maximum(0.0, lam2 + lr * dlam2)
        if abs(dlam1) < 1e-4 and np.abs(dlam2).max() < 1e-4 * L:
            break
    return assign


def _omni_assignments(dataset: list[dict], eval_matrix: str,
                      alpha: float, encoder) -> list[str]:
    emb_train, acc_train, cost_train = _load_eval_matrix_omni(eval_matrix, dataset, encoder)
    test_queries = [item["query"] for item in dataset]
    print(f"  [OmniRouter] Embedding {len(test_queries)} test queries...")
    emb_test = encoder.encode(test_queries, batch_size=64, show_progress_bar=False,
                              normalize_embeddings=True).astype(np.float32)
    print(f"  [OmniRouter] KNN + Lagrangian optimization (alpha={alpha})...")
    acc_pred, cost_pred = _knn_predict(emb_train, acc_train, cost_train, emb_test)
    assign_idx = _lagrangian_optimize(acc_pred, cost_pred, alpha=alpha)
    avg_acc  = acc_pred[np.arange(len(dataset)), assign_idx].mean()
    tot_cost = cost_pred[np.arange(len(dataset)), assign_idx].sum()
    print(f"  [OmniRouter] avg_acc={avg_acc:.3f}  total_cost=${tot_cost:.6f}")
    return [MODEL_IDS[j] for j in assign_idx]


# ---------------------------------------------------------------------------
# Execution: direct model call (CARROT / OmniRouter path)
# ---------------------------------------------------------------------------

async def _send_direct(client: httpx.AsyncClient, req_id: int,
                       item: dict, model_id: str, sub_mode: str) -> dict:
    backend    = BACKENDS[model_id]
    domain     = item.get("domain", "")
    complexity = item.get("complexity", "")
    in_tokens  = len(item["query"].split()) * 1.3
    out_est    = OUTPUT_TOKENS_EST.get((domain, complexity), 300)

    result = {
        "req_id": str(req_id), "domain": domain, "complexity": complexity,
        "query": item["query"][:100], "ground_truth": str(item.get("ground_truth", "")),
        "mode": f"mixed/{sub_mode}", "status": "", "model_winner": model_id,
        "bid_latency_ms": "", "actual_latency_ms": "", "ttft_ms": "",
        "output_tokens": "", "charged_usd": "",
        "energy_j": "", "load": "", "wall_ms": "", "slo_ms": "",
        "slo_violated": "", "retries": "0", "response_text": "", "error": "",
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
            body      = resp.json()
            out_tok   = body.get("usage", {}).get("completion_tokens", out_est)
            cost      = in_tokens * backend["input_rate"] + out_tok * backend["output_rate"]
            result["output_tokens"] = out_tok
            result["energy_j"]      = f"{out_tok / backend['eff_tok_per_j']:.3f}"
            result["charged_usd"]   = f"{cost:.8f}"
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
# Execution: TTCA router call (code:hard path)
# ---------------------------------------------------------------------------

async def _send_ttca(client: httpx.AsyncClient, req_id: int,
                     item: dict, router_url: str) -> dict:
    domain     = item.get("domain", "")
    complexity = item.get("complexity", "")

    result = {
        "req_id": str(req_id), "domain": domain, "complexity": complexity,
        "query": item["query"][:100], "ground_truth": str(item.get("ground_truth", "")),
        "mode": "mixed/ttca", "status": "", "model_winner": "",
        "bid_latency_ms": "", "actual_latency_ms": "", "ttft_ms": "",
        "output_tokens": "", "charged_usd": "",
        "energy_j": "", "load": "", "wall_ms": "", "slo_ms": "",
        "slo_violated": "", "retries": "0", "response_text": "", "error": "",
    }

    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{router_url}/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": item["query"]}],
                "max_tokens": 512,
                "extra_body": {
                    "router": {
                        "mode": "ttca",
                        "domain": domain,
                        "complexity": complexity,
                    }
                },
            },
        )
        wall_ms = int((time.monotonic() - t0) * 1000)
        result["wall_ms"] = result["actual_latency_ms"] = result["ttft_ms"] = wall_ms
        result["status"]  = resp.status_code
        if resp.status_code == 200:
            body = resp.json()
            result["model_winner"]   = resp.headers.get("x-router-model-winner", "")
            result["charged_usd"]    = resp.headers.get("x-router-charged-usd", "")
            result["bid_latency_ms"] = resp.headers.get("x-router-bid-latency-ms", "")
            result["retries"]        = resp.headers.get("x-router-retries", "0")
            result["load"]           = resp.headers.get("x-router-load", "")
            out_tok = body.get("usage", {}).get("completion_tokens", 0)
            result["output_tokens"]  = out_tok
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
# Main execution loop
# ---------------------------------------------------------------------------

async def run_mixed(dataset: list[dict],
                    assignments: list[str | None],
                    sub_modes: list[str],
                    router_url: str,
                    output: str,
                    concurrency: int) -> None:
    n    = len(dataset)
    sem  = asyncio.Semaphore(concurrency)
    done = 0
    t0   = time.monotonic()

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    async def bounded(req_id: int, item: dict, model_id: str | None,
                      sub_mode: str, writer, f) -> None:
        nonlocal done
        async with sem:
            async with httpx.AsyncClient(timeout=300.0, trust_env=False) as client:
                if sub_mode == "ttca":
                    r = await _send_ttca(client, req_id, item, router_url)
                else:
                    r = await _send_direct(client, req_id, item, model_id, sub_mode)
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
            bounded(i, dataset[i], assignments[i], sub_modes[i], writer, f)
            for i in range(n)
        ])

    elapsed = time.monotonic() - t0
    print(f"\r  [{'='*40}] {n}/{n}  ({elapsed:.1f}s, {n/elapsed:.1f} req/s)\n")

    with open(output, newline="") as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if str(r.get("status")) == "200"]
    mode_counts  = Counter(r.get("mode", "") for r in ok)
    model_counts = Counter(r.get("model_winner", "") for r in ok)
    costs = [float(r["charged_usd"]) for r in ok if r.get("charged_usd")]

    print(f"  Successful  : {len(ok)}/{n}")
    print(f"  By sub-mode :")
    for m, c in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"    {m:<20} {c:4d}  ({100*c//max(len(ok),1):2d}%)")
    print(f"  By model    :")
    for m, c in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {m:<24} {c:4d}  ({100*c//max(len(ok),1):2d}%)")
    if costs:
        print(f"  Total cost  : ${sum(costs):.6f}  avg=${mean(costs):.8f}")
    print(f"\n  Saved: {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed router: TTCA for code:hard, CARROT/OmniRouter elsewhere")
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--eval-matrix",  required=True)
    parser.add_argument("--router",       default="http://localhost:8080",
                        help="TTCA router URL (default: http://localhost:8080)")
    parser.add_argument("--mu",           type=float, default=0.3,
                        help="CARROT cost/quality tradeoff (default 0.3)")
    parser.add_argument("--alpha",        type=float, default=0.75,
                        help="OmniRouter accuracy floor (default 0.75)")
    parser.add_argument("--carrot-frac",  type=float, default=0.75,
                        help="Fraction of non-code:hard queries via CARROT (default 0.75)")
    parser.add_argument("--concurrency",  type=int,   default=50)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--node2-host",   default=None,
                        help="Hostname of node2 for llama4-scout")
    parser.add_argument("--output",       default="")
    args = parser.parse_args()

    if args.node2_host:
        BACKENDS["llama4-scout"]["base_url"] = f"http://{args.node2_host}:8005"
    else:
        MODEL_IDS.remove("llama4-scout")
        del BACKENDS["llama4-scout"]
        print("  [Mixed] No --node2-host: llama4-scout excluded from routing pool.")

    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/baseline_mixed_{ts}.csv"

    rng = random.Random(args.seed)

    print(f"\n  [Mixed] Loading dataset: {args.dataset}")
    with open(args.dataset) as f:
        dataset: list[dict] = json.load(f)
    n = len(dataset)

    ttca_idx  = [i for i, x in enumerate(dataset)
                 if (x.get("domain"), x.get("complexity")) == TTCA_CELL]
    other_idx = [i for i, x in enumerate(dataset)
                 if (x.get("domain"), x.get("complexity")) != TTCA_CELL]

    print(f"  [Mixed] {n} queries total")
    print(f"    code:hard  → TTCA router                          : {len(ttca_idx):4d} queries")
    print(f"    other      → CARROT ({args.carrot_frac:.0%}) / OmniRouter ({1-args.carrot_frac:.0%}): {len(other_idx):4d} queries")
    print(f"  [Mixed] Router URL : {args.router}")
    print(f"  [Mixed] CARROT mu={args.mu}  OmniRouter alpha={args.alpha}  seed={args.seed}")

    from sentence_transformers import SentenceTransformer
    print("\n  [Mixed] Loading sentence encoder (shared)...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("\n  [Mixed] Computing CARROT assignments...")
    carrot_assign = _carrot_assignments(dataset, args.eval_matrix, args.mu, encoder)

    print("\n  [Mixed] Computing OmniRouter assignments...")
    omni_assign   = _omni_assignments(dataset, args.eval_matrix, args.alpha, encoder)

    assignments: list[str | None] = [None] * n
    sub_modes:   list[str]        = [""] * n

    for i in ttca_idx:
        assignments[i] = None
        sub_modes[i]   = "ttca"

    carrot_n = omni_n = 0
    for i in other_idx:
        if rng.random() < args.carrot_frac:
            assignments[i] = carrot_assign[i]
            sub_modes[i]   = "carrot"
            carrot_n += 1
        else:
            assignments[i] = omni_assign[i]
            sub_modes[i]   = "omni"
            omni_n += 1

    print(f"\n  [Mixed] Final routing plan:")
    print(f"    ttca   : {len(ttca_idx):4d}  (code:hard)")
    print(f"    carrot : {carrot_n:4d}  ({100*carrot_n//max(len(other_idx),1)}% of non-code:hard)")
    print(f"    omni   : {omni_n:4d}  ({100*omni_n//max(len(other_idx),1)}% of non-code:hard)")

    print(f"\n  [Mixed] Executing {n} requests (concurrency={args.concurrency})...")
    asyncio.run(run_mixed(dataset, assignments, sub_modes, args.router,
                          args.output, args.concurrency))


if __name__ == "__main__":
    main()
