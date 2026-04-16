from __future__ import annotations
from fastapi import HTTPException
from semantic_router.schemas import BidResponse, UserPreference
from semantic_router.reputation_tracker import ReputationTracker


class NoEligibleModelError(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=503, detail="No model satisfied the SLA constraints.")


def select(bids: list[BidResponse], pref: UserPreference, reputation: ReputationTracker) -> BidResponse:
    if not bids:
        raise NoEligibleModelError()
    candidates = [
        b for b in bids
        if (pref.max_latency_ms is None or b.estimated_latency_ms <= pref.max_latency_ms)
        and (pref.min_accuracy is None or b.estimated_accuracy >= pref.min_accuracy)
    ]
    if not candidates:
        raise NoEligibleModelError()

    def ec(b: BidResponse) -> float:
        return b.estimated_cost_usd * reputation.get_penalty_multiplier(b.model_id)

    max_cost = max(ec(b) for b in candidates) or 1.0
    max_lat  = max(b.estimated_latency_ms for b in candidates) or 1.0
    max_eng  = max(b.estimated_energy_j   for b in candidates) or 1.0

    def score(b: BidResponse) -> float:
        return (pref.cost_weight * (ec(b) / max_cost)
              + pref.latency_weight  * (b.estimated_latency_ms / max_lat)
              + pref.accuracy_weight * (1.0 - b.estimated_accuracy)
              + pref.energy_weight   * (b.estimated_energy_j   / max_eng))

    return min(candidates, key=score)
