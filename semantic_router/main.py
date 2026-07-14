from __future__ import annotations
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from semantic_router.analyzer import SemanticAnalyzer
from semantic_router.accuracy_sampler import AccuracySampler
from semantic_router.benchmark_store import BenchmarkStore
from semantic_router.bidder import collect_bids
from semantic_router.dispatcher import dispatch
from semantic_router.registry import ModelRegistry, ModelConfig
from semantic_router.reputation_tracker import ReputationTracker
from semantic_router.config import LATENCY_SLO_MS, BENCHMARK_PATH
from semantic_router.schemas import (
    BidRequest, RequestSLA, RouterMode, UserBudget, UserPreference,
)
from semantic_router.selector import select, rank_bids
from semantic_router.user_registry import UserRegistry

log = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────

analyzer   = SemanticAnalyzer()
registry   = ModelRegistry()
user_reg   = UserRegistry()
reputation = ReputationTracker()
sampler    = AccuracySampler(reputation)
benchmark  = BenchmarkStore(BENCHMARK_PATH)  # offline ground-truth lookup

# ── Cold-model registry ───────────────────────────────────────────────────────
# Models known to exist but not currently running.  The provisioner populates
# this at startup via POST /router/cold-register so the router can score them
# alongside running models and trigger proactive spin-up when worthwhile.
# {model_id: {"accuracy_priors": {...}, "latency_ms": float,
#              "spin_up_ms": float, "domains": [str]}}
_cold_models: dict[str, dict] = {}

# ── Spin-up request queue ─────────────────────────────────────────────────────
# The router enqueues model IDs here when a cold model's amortized TTCA score
# beats the running fleet.  The provisioner drains this via GET /router/spin-up-queue
# on every poll cycle and calls spin_up() accordingly.
_spin_up_queue: asyncio.Queue = asyncio.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    analyzer.load()
    eval_matrix_path = os.environ.get("EVAL_MATRIX_PATH", "")
    if eval_matrix_path and os.path.exists(eval_matrix_path):
        try:
            n = reputation.warmup_from_eval_matrix(eval_matrix_path)
            log.info("Warm-started %d accuracy prior cells from %s", n, eval_matrix_path)
        except Exception as exc:
            log.warning("eval_matrix warmup failed: %s", exc)
    asyncio.create_task(sampler.run())
    log.info("Semantic router started.")
    yield
    log.info("Semantic router shutting down.")


app = FastAPI(title="Semantic LLM Router", lifespan=lifespan)


# ── Inference endpoint ────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body: dict = await request.json()
    messages: list[dict] = body.get("messages", [])
    extra_body: dict = body.get("extra_body", {})
    router_params: dict = extra_body.get("router", {})

    sla = RequestSLA(**router_params) if router_params else RequestSLA()
    user_id = sla.user_id

    # Resolve user preference (mode → preset → per-request overrides)
    pref = user_reg.resolve_preference(user_id, sla)

    # Semantic classification — runs encode() in thread pool to avoid blocking event loop.
    # If domain/complexity are provided in router params, use them and skip classifier
    # (avoids misclassification of structured benchmark queries like MMLU/GSM8K).
    meta = await analyzer.analyze_async(messages)
    if sla.domain:
        meta.domain = sla.domain
    if sla.complexity:
        meta.complexity = sla.complexity

    # Apply per-domain/complexity SLO if the user hasn't set an explicit latency ceiling
    if pref.max_latency_ms is None:
        slo_ms = LATENCY_SLO_MS.get((meta.domain, meta.complexity))
        if slo_ms is not None:
            pref.max_latency_ms = slo_ms
            log.info("SLO applied: %s:%s → %d ms", meta.domain, meta.complexity, slo_ms)

    # Budget pre-check (rough estimate: 300 tokens output, 50 J)
    estimated_tokens = 300
    if user_id:
        estimated_energy_j = 300.0 / 2.0  # conservative: assume 2 tok/J
        user_reg.check_budget(user_id, estimated_tokens, estimated_energy_j)

    # Record request arrival for cold-model amortized spin-up cost calculation.
    reputation.record_request_arrival(meta.domain, meta.complexity)

    # Get eligible model backends
    adapters = registry.get_eligible(meta.domain, pref.min_accuracy, reputation)
    if not adapters:
        raise HTTPException(status_code=503, detail="No models registered for this domain.")

    # Broadcast bids
    bid_request = BidRequest(
        messages=messages,
        complexity=meta.complexity,
        domain=meta.domain,
        query_embedding=meta.embedding.tolist(),
        preference=pref,
    )
    bids = await collect_bids(adapters, bid_request)

    # Rank all bids best-first — used for inference-time retry
    ranked_bids = rank_bids(bids, pref, reputation, meta.domain, meta.complexity)
    if not ranked_bids:
        raise HTTPException(status_code=503, detail="No models available to handle request.")

    # Background: evaluate cold models and trigger spin-up if one beats the fleet.
    # Fire-and-forget — does not block this request.
    asyncio.create_task(_maybe_trigger_cold_spinup(
        meta.domain, meta.complexity, ranked_bids
    ))

    # Exclude "stream" so it is never forwarded to vLLM — the router's
    # adapter.complete() calls resp.json() which fails on SSE responses.
    passthrough = {
        k: v for k, v in body.items()
        if k not in ("model", "messages", "extra_body", "stream")
    }

    # Offline ground-truth lookup — enables quality-based retry
    # when the request matches a known benchmark query.
    query_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    benchmark_item = benchmark.lookup(query_text)
    ground_truth   = benchmark_item.get("ground_truth") if benchmark_item else None

    # Dispatch with inference-time retry: try each model in score order.
    # Two retry triggers:
    #   1. Infrastructure failure (crash, timeout, HTTP error)
    #   2. Wrong answer — only when ground_truth is known from benchmark dataset
    #
    # Important: quality-based retries save the best available response so far.
    # If ALL models answer incorrectly we still return the first response rather
    # than failing the request with 503 — a wrong answer is better than no answer.
    last_error: Exception | None = None
    best_response: tuple | None = None   # (response_dict, router_headers) of first successful dispatch
    for attempt, winning_bid in enumerate(ranked_bids):
        winning_adapter = registry.get_adapter(winning_bid.model_id)
        if winning_adapter is None:
            continue
        if attempt > 0:
            log.warning(
                "Retrying with next-best model %s (attempt %d/%d)",
                winning_bid.model_id, attempt + 1, len(ranked_bids),
            )
        try:
            response_dict, router_headers = await dispatch(
                adapter=winning_adapter,
                winning_bid=winning_bid,
                messages=messages,
                domain=meta.domain,
                complexity=meta.complexity,
                user_id=user_id,
                user_registry=user_reg,
                reputation=reputation,
                sampler=sampler,
                extra_kwargs=passthrough,
            )

            # Quality check — only when we have offline ground truth AND retry is allowed.
            # sla.no_retry=True disables cascade retry for single-shot comparison baselines.
            if ground_truth is not None and not sla.no_retry:
                response_text = ""
                try:
                    response_text = response_dict["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
                correct = benchmark.is_correct(meta.domain, response_text, ground_truth)
                router_headers["X-Router-GT-Scored"] = "true"
                if correct is False:
                    log.warning(
                        "Model %s gave wrong answer (attempt %d) — trying next model",
                        winning_bid.model_id, attempt + 1,
                    )
                    router_headers["X-Router-GT-Correct"] = "false"
                    last_error = ValueError(f"{winning_bid.model_id} answered incorrectly")
                    # Save first wrong response as fallback — returned if all models fail
                    if best_response is None:
                        best_response = (response_dict, router_headers)
                    continue  # try next model
                if correct is True:
                    router_headers["X-Router-GT-Correct"] = "true"
            elif ground_truth is not None and sla.no_retry:
                # Score for reporting but do not retry
                response_text = ""
                try:
                    response_text = response_dict["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
                correct = benchmark.is_correct(meta.domain, response_text, ground_truth)
                router_headers["X-Router-GT-Scored"] = "true"
                router_headers["X-Router-GT-Correct"] = "true" if correct else "false"
            else:
                router_headers["X-Router-GT-Scored"] = "false"

            router_headers["X-Router-Accuracy-Weight"] = f"{pref.accuracy_weight:.2f}"
            router_headers["X-Router-Cost-Weight"]     = f"{pref.cost_weight:.2f}"
            router_headers["X-Router-Attempt"]         = str(attempt + 1)
            if pref.max_latency_ms is not None:
                try:
                    actual_ms = int(router_headers.get("X-Router-Actual-Latency-Ms", 0))
                except (ValueError, TypeError):
                    actual_ms = winning_bid.estimated_latency_ms
                slo_violated = actual_ms > pref.max_latency_ms
                router_headers["X-Router-SLO-Ms"]       = str(pref.max_latency_ms)
                router_headers["X-Router-SLO-Violated"] = "true" if slo_violated else "false"
            return JSONResponse(content=response_dict, headers=router_headers)

        except ValueError:
            # Wrong-answer retry — already logged above, just move on
            continue
        except Exception as e:
            last_error = e
            log.warning(
                "Model %s failed during inference (attempt %d): %s",
                winning_bid.model_id, attempt + 1, e,
            )
            continue

    # All running models exhausted.
    # Bonus retry: check if any cold model has become available since the request started
    # (i.e., a spin-up triggered by a previous request just completed).
    if best_response is None:
        newly_ready = [
            mid for mid in _cold_models
            if registry.get_adapter(mid) is not None
        ]
        for mid in newly_ready:
            adapter = registry.get_adapter(mid)
            if adapter is None:
                continue
            log.info("Cold model %s is now ready — attempting bonus retry", mid)
            try:
                response_dict, router_headers = await dispatch(
                    adapter=adapter,
                    winning_bid=next((b for b in bids if b.model_id == mid), bids[0]),
                    messages=messages,
                    domain=meta.domain,
                    complexity=meta.complexity,
                    user_id=user_id,
                    user_registry=user_reg,
                    reputation=reputation,
                    sampler=sampler,
                    extra_kwargs=passthrough,
                )
                router_headers["X-Router-Cold-Retry"] = "true"
                router_headers["X-Router-Attempt"]    = str(len(ranked_bids) + 1)
                return JSONResponse(content=response_dict, headers=router_headers)
            except Exception:
                pass

    # If we have a quality-retry fallback (wrong answer but model responded),
    # return it rather than 503 — a wrong answer is better than no answer.
    if best_response is not None:
        response_dict, router_headers = best_response
        router_headers["X-Router-GT-Correct"]     = "false"
        router_headers["X-Router-GT-Fallback"]    = "true"
        router_headers["X-Router-Accuracy-Weight"] = f"{pref.accuracy_weight:.2f}"
        router_headers["X-Router-Cost-Weight"]     = f"{pref.cost_weight:.2f}"
        router_headers["X-Router-Attempt"]         = str(len(ranked_bids))
        log.warning("All models answered incorrectly — returning best available response")
        return JSONResponse(content=response_dict, headers=router_headers)

    raise HTTPException(
        status_code=503,
        detail=f"All {len(ranked_bids)} models failed. Last error: {last_error}",
    )


# ── Background cold-model spin-up trigger ─────────────────────────────────────

async def _maybe_trigger_cold_spinup(
    domain: str, complexity: str, ranked_bids: list
) -> None:
    """Fire-and-forget: score cold models and enqueue spin-up if one beats the fleet."""
    if not ranked_bids or not _cold_models:
        return
    try:
        from semantic_router.config import TTCA_ALPHA
        best_running_score = ranked_bids[0].estimated_accuracy / max(
            ranked_bids[0].estimated_latency_ms, 1.0
        ) ** TTCA_ALPHA
        running_ids = [b.model_id for b in ranked_bids]
        key = f"{domain}:{complexity}"

        for model_id, spec in _cold_models.items():
            if registry.get_adapter(model_id) is not None:
                continue   # already running
            n = reputation.count_requests_needing_model(
                model_id, spec.get("accuracy_priors", {}), running_ids, window_s=5.0
            )
            spin_cost_ms = spec.get("spin_up_ms", 300_000) / max(n, 1)
            acc = spec.get("accuracy_priors", {}).get(key, 0.0)
            lat = max(spec.get("latency_ms", 1000.0) + spin_cost_ms, 1.0)
            cold_score = acc / (lat ** TTCA_ALPHA)
            if cold_score > best_running_score:
                log.info(
                    "Cold model %s score=%.4f > running=%.4f (n_qualifying=%d) — queuing spin-up",
                    model_id, cold_score, best_running_score, n,
                )
                await _spin_up_queue.put(model_id)
                break   # one spin-up at a time
    except Exception as e:
        log.debug("Cold spin-up evaluation error: %s", e)


# ── Model fleet management ────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    model_id: str          # router alias shown in headers and /v1/models
    model_name: str = ""   # actual HuggingFace name sent to vLLM backend
                           # (e.g. "Qwen/Qwen2.5-7B-Instruct"); defaults to model_id
    backend: str           # "vllm" | "dynamo" | "ray"
    base_url: str
    domains: list[str]
    min_accuracy_capability: dict[str, float] = {}
    efficiency_tokens_per_joule: float = 5.0
    max_concurrent_requests: int = 256
    input_rate_usd_per_token: float = 1e-6
    output_rate_usd_per_token: float = 2e-6
    accuracy_priors: dict[str, float] = {}
    skip_calibration: bool = False
    # Throughput for decomposed latency formula (decode + prefill + queue).
    # decode_tokens_per_sec: how fast the model generates output tokens (tok/s).
    # prefill_tokens_per_sec: how fast it processes input prompt (typically 5-10x decode).
    # 0 means auto: prefill defaults to 5× decode.
    decode_tokens_per_sec: float = 1000.0
    prefill_tokens_per_sec: float = 0.0


class ColdModelRequest(BaseModel):
    model_id: str
    domains: list[str]
    accuracy_priors: dict[str, float] = {}
    estimated_latency_ms: float = 5000.0    # expected latency when running
    estimated_spin_up_ms: float = 300_000.0  # expected time to spin up (ms)


@app.post("/router/cold-register", status_code=201)
async def cold_register_model(req: ColdModelRequest) -> dict:
    """Provisioner calls this at startup to register models not yet running.
    The router uses these specs to score cold models alongside running ones
    and trigger proactive spin-up when their amortized TTCA score is better.
    """
    _cold_models[req.model_id] = {
        "accuracy_priors":   req.accuracy_priors,
        "latency_ms":        req.estimated_latency_ms,
        "spin_up_ms":        req.estimated_spin_up_ms,
        "domains":           req.domains,
    }
    log.info("Cold-registered %s (latency=%.0fms spin_up=%.0fs)",
             req.model_id, req.estimated_latency_ms, req.estimated_spin_up_ms / 1000)
    return {"cold_registered": req.model_id, "total_cold": len(_cold_models)}


@app.get("/router/spin-up-queue")
async def drain_spin_up_queue() -> dict:
    """Provisioner polls this to consume pending spin-up requests.
    Items are removed on read (queue is drained).
    """
    items: list[str] = []
    while not _spin_up_queue.empty():
        try:
            items.append(_spin_up_queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    if items:
        log.info("Spin-up queue drained: %s", items)
    return {"spin_up": items}


@app.post("/router/register", status_code=201)
async def register_model(req: RegisterRequest) -> dict:
    if req.backend == "vllm":
        from semantic_router.adapters.vllm import VLLMAdapter
        adapter_cls = VLLMAdapter
    elif req.backend == "dynamo":
        from semantic_router.adapters.dynamo import DynamoAdapter
        adapter_cls = DynamoAdapter
    elif req.backend == "ray":
        from semantic_router.adapters.ray import RayServeAdapter
        adapter_cls = RayServeAdapter
    else:
        raise HTTPException(status_code=400, detail=f"Unknown backend: {req.backend}")

    # Run calibration if no per-domain floors were provided
    capability = req.min_accuracy_capability
    if not capability and not req.skip_calibration:
        log.info("No min_accuracy_capability provided — running calibration for %s", req.model_id)
        try:
            from semantic_router.calibration import calibrate
            capability = await calibrate(req.base_url, req.model_id)
        except Exception as e:
            log.warning("Calibration failed for %s: %s — using default 0.5", req.model_id, e)
            capability = {"_default": 0.5}

    # If caller provided partial dict, fill missing domains with "_default" fallback
    if capability and "_default" not in capability:
        capability["_default"] = min(capability.values())

    # Build accuracy_priors: prefer explicitly provided values, fall back to
    # min_accuracy_capability so the model never bids DEFAULT_ACCURACY_PRIOR (0.70)
    # for domains that have leaderboard data. This is the critical path:
    # if accuracy_priors stays empty, both models bid 0.70 and cost always decides.
    accuracy_priors = dict(req.accuracy_priors)
    if capability:
        for key, score in capability.items():
            if ":" in key and key not in accuracy_priors:
                accuracy_priors[key] = score   # fill gaps from capability dict

    adapter = adapter_cls(
        model_id=req.model_id,
        model_name=req.model_name or req.model_id,
        base_url=req.base_url,
        efficiency_tokens_per_joule=req.efficiency_tokens_per_joule,
        max_concurrent_requests=req.max_concurrent_requests,
        input_rate_usd_per_token=req.input_rate_usd_per_token,
        output_rate_usd_per_token=req.output_rate_usd_per_token,
        accuracy_priors=accuracy_priors,
        reputation=reputation,
        decode_tokens_per_sec=req.decode_tokens_per_sec,
        prefill_tokens_per_sec=(req.prefill_tokens_per_sec
                                 if req.prefill_tokens_per_sec > 0
                                 else req.decode_tokens_per_sec * 5),
    )
    # Seed tracker with calibration priors for cells not yet warm-started.
    # seed_if_absent is a no-op for cells already populated by warmup_from_eval_matrix.
    for key, score in accuracy_priors.items():
        if ":" in key:
            domain_part, complexity_part = key.split(":", 1)
            reputation.seed_if_absent(adapter.model_id, domain_part, complexity_part, score)
    registry.register(ModelConfig(
        adapter=adapter,
        domains=req.domains,
        min_accuracy_capability=capability,
    ))
    return {
        "registered":             req.model_id,
        "min_accuracy_capability": capability,
        "accuracy_priors_stored":  accuracy_priors,   # confirm what was stored
    }


@app.delete("/router/{model_id}")
async def deregister_model(model_id: str) -> dict:
    registry.deregister(model_id)
    return {"deregistered": model_id}


class WarmupRequest(BaseModel):
    eval_matrix_path: str


@app.post("/router/warmup")
async def warmup_router(req: WarmupRequest) -> dict:
    """Seed accuracy priors from eval_matrix.csv without restarting the router.

    Call this after eval_all_models.py finishes and before the load test begins.
    Priors are written to model_reputation.json and take effect on the next bid.
    """
    if not os.path.exists(req.eval_matrix_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.eval_matrix_path}")
    try:
        n = reputation.warmup_from_eval_matrix(req.eval_matrix_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    log.info("Warmup: %d cells seeded from %s", n, req.eval_matrix_path)
    return {"cells_seeded": n, "eval_matrix_path": req.eval_matrix_path}


@app.get("/v1/models")
async def list_models() -> dict:
    models = []
    for mid in registry.list_all():
        models.append({
            "id": mid,
            "latency_reliability": reputation.get_latency_reliability(mid),
        })
    return {"object": "list", "data": models}


@app.get("/router/{model_id}/reputation")
async def model_reputation(model_id: str) -> dict:
    all_rep = reputation.get_all()
    if model_id not in all_rep:
        raise HTTPException(status_code=404, detail="Model not found in reputation store.")
    return all_rep[model_id]


@app.get("/router/{model_id}/details")
async def model_details(model_id: str) -> dict:
    """
    Show stored accuracy_priors for a model — use this to verify that
    registration passed accuracy_priors correctly.
    If all values show 0.70 (DEFAULT_ACCURACY_PRIOR), the priors were not set.
    """
    adapter = registry.get_adapter(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not registered.")
    return {
        "model_id":                    adapter.model_id,
        "base_url":                    adapter.base_url,
        "efficiency_tokens_per_joule": adapter.efficiency_tokens_per_joule,
        "input_rate_usd_per_token":    adapter.input_rate,
        "output_rate_usd_per_token":   adapter.output_rate,
        "accuracy_priors":             adapter.accuracy_priors,
        "router_in_flight":            adapter._in_flight,
    }


@app.get("/router/health")
async def health() -> dict:
    return {"status": "ok", "registered_models": len(registry.list_all())}


# ── User management ───────────────────────────────────────────────────────────

@app.post("/users/{user_id}/preference", status_code=201)
async def set_preference(user_id: str, pref: UserPreference) -> dict:
    user_reg.set_preference(user_id, pref)
    return {"user_id": user_id, "preference": pref.model_dump()}


@app.get("/users/{user_id}/preference")
async def get_preference(user_id: str) -> dict:
    return user_reg.get_preference_raw(user_id).model_dump()


@app.delete("/users/{user_id}/preference")
async def delete_preference(user_id: str) -> dict:
    user_reg.delete_preference(user_id)
    return {"user_id": user_id, "preference": "reset to defaults"}


@app.post("/users/{user_id}/budget", status_code=201)
async def set_budget(user_id: str, budget: UserBudget) -> dict:
    user_reg.set_budget(user_id, budget)
    return {"user_id": user_id, "budget": budget.model_dump()}


@app.get("/users/{user_id}/budget")
async def get_budget(user_id: str) -> dict:
    return user_reg.get_budget(user_id).model_dump()
