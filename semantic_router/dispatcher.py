from __future__ import annotations
import time, logging
from semantic_router.adapters.base import ModelAdapter
from semantic_router.schemas import BidResponse
from semantic_router.reputation_tracker import ReputationTracker
from semantic_router.user_registry import UserRegistry
from semantic_router.accuracy_sampler import AccuracySampler, SampleItem

log = logging.getLogger(__name__)


async def dispatch(
    adapter: ModelAdapter, winning_bid: BidResponse, messages: list[dict],
    domain: str, complexity: str, user_id: str | None,
    user_registry: UserRegistry, reputation: ReputationTracker,
    sampler: AccuracySampler, extra_kwargs: dict | None = None,
) -> tuple[dict, dict]:
    t0 = time.monotonic()
    response = await adapter.complete(messages, **(extra_kwargs or {}))
    actual_latency_ms = int((time.monotonic() - t0) * 1000)

    usage = response.get("usage", {})
    actual_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    actual_output  = usage.get("completion_tokens", 0)
    actual_energy_j = actual_output / max(adapter.efficiency_tokens_per_joule, 1e-9)

    reputation.record_latency(winning_bid.model_id, winning_bid.estimated_latency_ms, actual_latency_ms)
    if user_id:
        try: user_registry.deduct_budget(user_id, actual_tokens, actual_energy_j)
        except Exception as e: log.warning("Budget deduction failed for %s: %s", user_id, e)

    query = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    try:    resp_text = response["choices"][0]["message"]["content"]
    except: resp_text = ""

    sampler.enqueue(SampleItem(model_id=winning_bid.model_id, domain=domain, complexity=complexity, query=query, response=resp_text))

    return response, {
        "X-Router-Model":             winning_bid.model_id,
        "X-Router-Charged-USD":       f"{winning_bid.estimated_cost_usd:.6f}",
        "X-Router-Energy-J":          f"{actual_energy_j:.3f}",
        "X-Router-Bid-Latency-Ms":    str(winning_bid.estimated_latency_ms),
        "X-Router-Actual-Latency-Ms": str(actual_latency_ms),
        "X-Router-Load":              f"{winning_bid.current_load:.2f}",
    }
