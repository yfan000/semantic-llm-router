from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from semantic_router.analyzer import SemanticAnalyzer
from semantic_router.accuracy_sampler import AccuracySampler
from semantic_router.bidder import collect_bids
from semantic_router.dispatcher import dispatch
from semantic_router.registry import ModelRegistry, ModelConfig
from semantic_router.reputation_tracker import ReputationTracker
from semantic_router.config import LATENCY_SLO_MS
from semantic_router.schemas import (
    BidRequest, RequestSLA, RouterMode, UserBudget, UserPreference,
)
from semantic_router.selector import select
from semantic_router.user_registry import UserRegistry

log = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────

analyzer   = SemanticAnalyzer()
registry   = ModelRegistry()
user_reg   = UserRegistry()
reputation = ReputationTracker()
sampler    = AccuracySampler(reputation)


@asynccontextmanager
async def lifespan(app: FastAPI):
    analyzer.load()
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

    # Select winner
    winning_bid = select(bids, pref, reputation, meta.domain, meta.complexity)
    winning_adapter = registry.get_adapter(winning_bid.model_id)

    # Dispatch inference
    # Exclude "stream" so it is never forwarded to vLLM — the router's
    # adapter.complete() calls resp.json() which fails on SSE responses.
    passthrough = {
        k: v for k, v in body.items()
        if k not in ("model", "messages", "extra_body", "stream")
    }
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
    router_headers["X-Router-Accuracy-Weight"] = f"{pref.accuracy_weight:.2f}"
    router_headers["X-Router-Cost-Weight"]     = f"{pref.cost_weight:.2f}"
    if pref.max_latency_ms is not None:
        slo_violated = winning_bid.estimated_latency_ms > pref.max_latency_ms
        router_headers["X-Router-SLO-Ms"]       = str(pref.max_latency_ms)
        router_headers["X-Router-SLO-Violated"] = "true" if slo_violated else "false"

    return JSONResponse(content=response_dict, headers=router_headers)


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
    )
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
