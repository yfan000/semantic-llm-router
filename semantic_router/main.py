from __future__ import annotations
import asyncio, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from semantic_router.analyzer import SemanticAnalyzer
from semantic_router.accuracy_sampler import AccuracySampler
from semantic_router.bidder import collect_bids
from semantic_router.dispatcher import dispatch
from semantic_router.registry import ModelRegistry, ModelConfig
from semantic_router.reputation_tracker import ReputationTracker
from semantic_router.schemas import BidRequest, RequestSLA, UserBudget, UserPreference
from semantic_router.selector import select
from semantic_router.user_registry import UserRegistry

log = logging.getLogger(__name__)
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

app = FastAPI(title="Semantic LLM Router", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body: dict = await request.json()
    messages = body.get("messages", [])
    sla = RequestSLA(**body.get("extra_body", {}).get("router", {}))
    pref = user_reg.resolve_preference(sla.user_id, sla)
    meta = analyzer.analyze(messages)
    if sla.user_id:
        user_reg.check_budget(sla.user_id, 300, 300.0 / 2.0)
    adapters = registry.get_eligible(meta.domain, pref.min_accuracy)
    if not adapters:
        raise HTTPException(status_code=503, detail="No models registered for this domain.")
    bids = await collect_bids(adapters, BidRequest(messages=messages, complexity=meta.complexity,
        domain=meta.domain, query_embedding=meta.embedding.tolist(), preference=pref))
    winning_bid = select(bids, pref, reputation)
    response_dict, headers = await dispatch(
        adapter=registry.get_adapter(winning_bid.model_id), winning_bid=winning_bid,
        messages=messages, domain=meta.domain, complexity=meta.complexity,
        user_id=sla.user_id, user_registry=user_reg, reputation=reputation, sampler=sampler,
        extra_kwargs={k: v for k, v in body.items() if k not in ("model", "messages", "extra_body")},
    )
    return JSONResponse(content=response_dict, headers=headers)


class RegisterRequest(BaseModel):
    model_id: str; backend: str; base_url: str; domains: list[str]
    min_accuracy_capability: float = 0.5; efficiency_tokens_per_joule: float = 5.0
    max_concurrent_requests: int = 256; input_rate_usd_per_token: float = 1e-6
    output_rate_usd_per_token: float = 2e-6; accuracy_priors: dict[str, float] = {}


@app.post("/router/register", status_code=201)
async def register_model(req: RegisterRequest) -> dict:
    if req.backend == "vllm":   from semantic_router.adapters.vllm   import VLLMAdapter      as cls
    elif req.backend == "dynamo": from semantic_router.adapters.dynamo import DynamoAdapter    as cls
    elif req.backend == "ray":  from semantic_router.adapters.ray    import RayServeAdapter   as cls
    else: raise HTTPException(400, f"Unknown backend: {req.backend}")
    adapter = cls(model_id=req.model_id, base_url=req.base_url,
        efficiency_tokens_per_joule=req.efficiency_tokens_per_joule,
        max_concurrent_requests=req.max_concurrent_requests,
        input_rate_usd_per_token=req.input_rate_usd_per_token,
        output_rate_usd_per_token=req.output_rate_usd_per_token,
        accuracy_priors=req.accuracy_priors)
    registry.register(ModelConfig(adapter=adapter, domains=req.domains, min_accuracy_capability=req.min_accuracy_capability))
    return {"registered": req.model_id}


@app.delete("/router/{model_id}")
async def deregister_model(model_id: str) -> dict:
    registry.deregister(model_id); return {"deregistered": model_id}

@app.get("/v1/models")
async def list_models() -> dict:
    return {"object": "list", "data": [{"id": m, "latency_reliability": reputation.get_latency_reliability(m)} for m in registry.list_all()]}

@app.get("/router/{model_id}/reputation")
async def model_reputation(model_id: str) -> dict:
    all_rep = reputation.get_all()
    if model_id not in all_rep: raise HTTPException(404, "Model not found.")
    return all_rep[model_id]

@app.get("/router/health")
async def health() -> dict:
    return {"status": "ok", "registered_models": len(registry.list_all())}

@app.post("/users/{user_id}/preference", status_code=201)
async def set_preference(user_id: str, pref: UserPreference) -> dict:
    user_reg.set_preference(user_id, pref); return {"user_id": user_id, "preference": pref.model_dump()}

@app.get("/users/{user_id}/preference")
async def get_preference(user_id: str) -> dict:
    return user_reg.get_preference_raw(user_id).model_dump()

@app.delete("/users/{user_id}/preference")
async def delete_preference(user_id: str) -> dict:
    user_reg.delete_preference(user_id); return {"user_id": user_id, "preference": "reset to defaults"}

@app.post("/users/{user_id}/budget", status_code=201)
async def set_budget(user_id: str, budget: UserBudget) -> dict:
    user_reg.set_budget(user_id, budget); return {"user_id": user_id, "budget": budget.model_dump()}

@app.get("/users/{user_id}/budget")
async def get_budget(user_id: str) -> dict:
    return user_reg.get_budget(user_id).model_dump()
