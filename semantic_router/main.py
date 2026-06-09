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

# ── Singletons ────────────────────────────────────────────────────────────────────────────────

analyzer   = SemanticAnalyzer()
registry   = ModelRegistry()
user_reg   = UserRegistry()
reputation = ReputationTracker()
sampler    = AccuracySampler(reputation)
benchmark  = BenchmarkStore(BENCHMARK_PATH)  # offline ground-truth lookup


@asynccontextmanager
async def lifespan(app: FastAPI):
    analyzer.load()
    asyncio.create_task(sampler.run())
    log.info("Semantic router started.")
    yield
    log.info("Semantic router shutting down.")


app = FastAPI(title="Semantic LLM Router", lifespan=lifespan)


# ── Inference endpoint ────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body: dict = await request.json()
    messages: list[dict] = body.get("messages", [])
    extra_body: dict = body.get("extra_body", {})
    router_params: dict = extra_body.get("router", {})

    sla = RequestSLA(**router_params) if router_params else RequestSLA()
    user_id = sla.user_id

    pref = user_reg.resolve_preference(user_id, sla)

    meta = await analyzer.analyze_async(messages)
    if sla.domain:
        meta.domain = sla.domain
    if sla.complexity:
        meta.complexity = sla.complexity

    if pref.max_latency_ms is None:
        slo_ms = LATENCY_SLO_MS.get((meta.domain, meta.complexity))
        if slo_ms is not None:
            pref.max_latency_ms = slo_ms
            log.info("SLO applied: %s:%s -> %d ms", meta.domain, meta.complexity, slo_ms)

    estimated_tokens = 300
    if user_id:
        estimated_energy_j = 300.0 / 2.0
        user_reg.check_budget(user_id, estimated_tokens, estimated_energy_j)

    adapters = registry.get_eligible(meta.domain, pref.min_accuracy, reputation)
    if not adapters:
        raise HTTPException(status_code=503, detail="No models registered for this domain.")

    bid_request = BidRequest(
        messages=messages,
        complexity=meta.complexity,
        domain=meta.domain,
        query_embedding=meta.embedding.tolist(),
        preference=pref,
    )
    bids = await collect_bids(adapters, bid_request)

    ranked_bids = rank_bids(bids, pref, reputation, meta.domain, meta.complexity)
    if not ranked_bids:
        raise HTTPException(status_code=503, detail="No models available to handle request.")

    passthrough = {
        k: v for k, v in body.items()
        if k not in ("model", "messages", "extra_body", "stream")
    }

    query_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    benchmark_item = benchmark.lookup(query_text)
    ground_truth   = benchmark_item.get("ground_truth") if benchmark_item else None

    last_error: Exception | None = None
    best_response: tuple | None = None
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

            if ground_truth is not None:
                response_text = ""
                try:
                    response_text = response_dict["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
                correct = benchmark.is_correct(meta.domain, response_text, ground_truth)
                router_headers["X-Router-GT-Scored"] = "true"
                if correct is False:
                    log.warning(
                        "Model %s gave wrong answer (attempt %d) -- trying next model",
                        winning_bid.model_id, attempt + 1,
                    )
                    router_headers["X-Router-GT-Correct"] = "false"
                    last_error = ValueError(f"{winning_bid.model_id} answered incorrectly")
                    if best_response is None:
                        best_response = (response_dict, router_headers)
                    continue
                if correct is True:
                    router_headers["X-Router-GT-Correct"] = "true"
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
            continue
        except Exception as e:
            last_error = e
            log.warning(
                "Model %s failed during inference (attempt %d): %s",
                winning_bid.model_id, attempt + 1, e,
            )
            continue

    if best_response is not None:
        response_dict, router_headers = best_response
        router_headers["X-Router-GT-Correct"]      = "false"
        router_headers["X-Router-GT-Fallback"]     = "true"
        router_headers["X-Router-Accuracy-Weight"] = f"{pref.accuracy_weight:.2f}"
        router_headers["X-Router-Cost-Weight"]     = f"{pref.cost_weight:.2f}"
        router_headers["X-Router-Attempt"]         = str(len(ranked_bids))
        log.warning("All models answered incorrectly -- returning best available response")
        return JSONResponse(content=response_dict, headers=router_headers)

    raise HTTPException(
        status_code=503,
        detail=f"All {len(ranked_bids)} models failed. Last error: {last_error}",
    )


# ── Model fleet management ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    model_id: str          # router alias shown in headers and /v1/models
    model_name: str = ""   # actual HuggingFace name sent to vLLM backend
    backend: str
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
    # 0 means auto: prefill defaults to 5x decode.
    decode_tokens_per_sec: float = 1000.0
    prefill_tokens_per_sec: float = 0.0


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

    capability = req.min_accuracy_capability
    if not capability and not req.skip_calibration:
        log.info("No min_accuracy_capability provided -- running calibration for %s", req.model_id)
        try:
            from semantic_router.calibration import calibrate
            capability = await calibrate(req.base_url, req.model_id)
        except Exception as e:
            log.warning("Calibration failed for %s: %s -- using default 0.5", req.model_id, e)
            capability = {"_default": 0.5}

    if capability and "_default" not in capability:
        capability["_default"] = min(capability.values())

    accuracy_priors = dict(req.accuracy_priors)
    if capability:
        for key, score in capability.items():
            if ":" in key and key not in accuracy_priors:
                accuracy_priors[key] = score

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
    registry.register(ModelConfig(
        adapter=adapter,
        domains=req.domains,
        min_accuracy_capability=capability,
    ))
    return {
        "registered":              req.model_id,
        "min_accuracy_capability": capability,
        "accuracy_priors_stored":  accuracy_priors,
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


# ── User management ───────────────────────────────────────────────────────────────────

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
