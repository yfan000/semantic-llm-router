from __future__ import annotations
import logging
from fastapi import HTTPException
from semantic_router.schemas import BidResponse, UserPreference
from semantic_router.reputation_tracker import ReputationTracker

log = logging.getLogger(__name__)


class NoEligibleModelError(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=503,
            detail="No model satisfied the SLA constraints.",
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

    log.info("SELECT domain=%s complexity=%s n_bids=%d acc_weight=%.2f cost_weight=%.2f",
             domain, complexity, len(bids), pref.accuracy_weight, pref.cost_weight)

    candidates = [
        b for b in bids
        if (pref.max_latency_ms is None or b.estimated_latency_ms <= pref.max_latency_ms)
        and (pref.min_accuracy is None or b.estimated_accuracy >= pref.min_accuracy)
    ]
    if not candidates:
        candidates = sorted(bids, key=lambda b: b.estimated_accuracy, reverse=True)[:1]

    def effective_cost(bid: BidResponse) -> float:
        return bid.estimated_cost_usd * reputation.get_penalty_multiplier(bid.model_id)

    max_cost   = max(effective_cost(b) for b in candidates) or 1.0
    max_lat    = max(b.estimated_latency_ms for b in candidates) or 1.0
    max_energy = max(b.estimated_energy_j   for b in candidates) or 1.0

    scores = {}
    for bid in candidates:
        ec = effective_cost(bid)
        s = (
            pref.cost_weight     * (ec                       / max_cost)
            + pref.latency_weight  * (bid.estimated_latency_ms / max_lat)
            + pref.accuracy_weight * (1.0 - bid.estimated_accuracy)
            + pref.energy_weight   * (bid.estimated_energy_j   / max_energy)
        )
        scores[bid.model_id] = s
        log.info(
            "  %-22s acc=%.3f cost=%.6f lat=%dms energy=%.2fJ "
            "cost_norm=%.3f lat_norm=%.3f energy_norm=%.3f score=%.4f",
            bid.model_id, bid.estimated_accuracy, bid.estimated_cost_usd,
            bid.estimated_latency_ms, bid.estimated_energy_j,
            ec / max_cost, bid.estimated_latency_ms / max_lat,
            bid.estimated_energy_j / max_energy, s,
        )

    winner = min(candidates, key=lambda b: scores[b.model_id])
    log.info("  WINNER: %s (score=%.4f)", winner.model_id, scores[winner.model_id])
    return winner
