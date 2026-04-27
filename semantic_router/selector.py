from __future__ import annotations
from fastapi import HTTPException
from semantic_router.schemas import BidResponse, UserPreference
from semantic_router.reputation_tracker import ReputationTracker


class NoEligibleModelError(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=503,
            detail="No model satisfied the SLA constraints. Relax latency or accuracy requirements.",
        )


def select(
    bids: list[BidResponse],
    pref: UserPreference,
    reputation: ReputationTracker,
    domain: str,
    complexity: str,
) -> BidResponse:
    if not bids:
        raise NoEligibleModelError()

    # Stage 1 — hard filter
    candidates = [
        b for b in bids
        if (pref.max_latency_ms is None or b.estimated_latency_ms <= pref.max_latency_ms)
        and (pref.min_accuracy is None or b.estimated_accuracy >= pref.min_accuracy)
    ]
    # Fallback: if no model meets the hard floor, pick the most accurate one available
    if not candidates:
        candidates = sorted(bids, key=lambda b: b.estimated_accuracy, reverse=True)[:1]

    # Stage 2 — 4D weighted score (lower = better)
    def effective_cost(bid: BidResponse) -> float:
        return bid.estimated_cost_usd * reputation.get_penalty_multiplier(bid.model_id)

    def effective_accuracy(bid: BidResponse) -> float:
        discount = reputation.get_accuracy_discount(bid.model_id, domain, complexity)
        return bid.estimated_accuracy * discount

    max_cost   = max(effective_cost(b) for b in candidates) or 1.0
    max_lat    = max(b.estimated_latency_ms for b in candidates) or 1.0
    max_energy = max(b.estimated_energy_j   for b in candidates) or 1.0

    def score(bid: BidResponse) -> float:
        return (
            pref.cost_weight     * (effective_cost(bid)       / max_cost)
            + pref.latency_weight  * (bid.estimated_latency_ms  / max_lat)
            + pref.accuracy_weight * (1.0 - effective_accuracy(bid))
            + pref.energy_weight   * (bid.estimated_energy_j    / max_energy)
        )

    return min(candidates, key=score)
