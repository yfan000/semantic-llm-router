from __future__ import annotations
import logging
from fastapi import HTTPException
from semantic_router.schemas import BidResponse, UserPreference, RouterMode
from semantic_router.reputation_tracker import ReputationTracker
from semantic_router.config import (
    TTCA_ALPHA, TTCA_COST_BETA,
    TTCA_ALPHA_FACTUAL, TTCA_ALPHA_MATH,
    TTCA_ALPHA_CODE, TTCA_ALPHA_REASONING,
)

# Domain-specific latency exponents for TTCA scoring.
# Built at import time from config constants so they pick up sed-patched values.
_DOMAIN_ALPHA: dict[str, float] = {
    "factual":   TTCA_ALPHA_FACTUAL,
    "math":      TTCA_ALPHA_MATH,
    "code":      TTCA_ALPHA_CODE,
    "reasoning": TTCA_ALPHA_REASONING,
}

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
        fastest = sorted(bids, key=lambda b: b.estimated_latency_ms)
        candidates = fastest[:1]
        log.warning(
            "SLO violated by all bids (slo=%s ms) -- falling back to fastest: %s (%d ms)",
            pref.max_latency_ms, candidates[0].model_id, candidates[0].estimated_latency_ms,
        )

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
            "  %-22s acc=%.3f cost=%.6f lat=%dms energy=%.2fJ score=%.4f",
            bid.model_id, bid.estimated_accuracy, bid.estimated_cost_usd,
            bid.estimated_latency_ms, bid.estimated_energy_j, s,
        )

    winner = min(candidates, key=lambda b: scores[b.model_id])
    log.info("  WINNER: %s (score=%.4f)", winner.model_id, scores[winner.model_id])
    return winner


def rank_bids(
    bids: list[BidResponse],
    pref: UserPreference,
    reputation: ReputationTracker,
    domain: str,
    complexity: str,
) -> list[BidResponse]:
    """Return all bids sorted best-first by weighted score.

    TTCA mode: score = acc / (lat^alpha x cost^beta)  -- higher is better.
    Uses domain-specific alpha: factual=0.3 (accuracy), code=1.0 (classic TTCA).
    beta=0 (default) reduces to classic TTCA acc/lat^alpha.
    """
    if not bids:
        return []

    def effective_cost(bid: BidResponse) -> float:
        return bid.estimated_cost_usd * reputation.get_penalty_multiplier(bid.model_id)

    candidates = [
        b for b in bids
        if (pref.max_latency_ms is None or b.estimated_latency_ms <= pref.max_latency_ms)
        and (pref.min_accuracy is None or b.estimated_accuracy >= pref.min_accuracy)
    ]
    if not candidates:
        candidates = sorted(bids, key=lambda b: b.estimated_latency_ms)

    if pref.mode == RouterMode.TTCA:
        # Use domain-specific alpha if configured; fall back to global TTCA_ALPHA.
        # Lower alpha = accuracy-focused (factual:0.3), higher = speed-focused (code:1.0).
        alpha = _DOMAIN_ALPHA.get(domain, TTCA_ALPHA)

        def ttca_score(bid: BidResponse) -> float:
            acc = max(bid.estimated_accuracy, 0.01)
            per_cat = reputation.get_avg_latency_ms_per_cat(bid.model_id, domain, complexity)
            global_  = reputation.get_avg_latency_ms(bid.model_id)
            lat = per_cat if per_cat is not None else (
                global_ if global_ is not None else bid.estimated_latency_ms
            )
            lat = max(lat, 1.0)   # guard against zero/negative
            cost = max(bid.estimated_cost_usd, 1e-9)
            # Domain-specific alpha: factual=0.3 (accuracy), code=1.0 (classic TTCA)
            return acc / ((lat ** alpha) * (cost ** TTCA_COST_BETA))

        ranked = sorted(candidates, key=ttca_score, reverse=True)   # highest score first
        for bid in ranked:
            per_cat = reputation.get_avg_latency_ms_per_cat(bid.model_id, domain, complexity)
            global_  = reputation.get_avg_latency_ms(bid.model_id)
            lat = per_cat if per_cat is not None else (
                global_ if global_ is not None else bid.estimated_latency_ms
            )
            lat = max(lat, 1.0)
            src = "per-cat" if per_cat is not None else (
                "global" if global_ is not None else "estimated"
            )
            cost = max(bid.estimated_cost_usd, 1e-9)
            score = bid.estimated_accuracy / ((lat ** alpha) * (cost ** TTCA_COST_BETA))
            log.info(
                "  TTCA %-22s lat=%dms(%s) acc=%.3f cost=%.6f alpha=%.1f(domain=%s) beta=%.1f score=%.4f",
                bid.model_id, lat, src,
                bid.estimated_accuracy, bid.estimated_cost_usd,
                alpha, domain, TTCA_COST_BETA, score,
            )
        return ranked

    max_cost   = max(effective_cost(b) for b in candidates) or 1.0
    max_lat    = max(b.estimated_latency_ms for b in candidates) or 1.0
    max_energy = max(b.estimated_energy_j   for b in candidates) or 1.0

    def score(bid: BidResponse) -> float:
        ec = effective_cost(bid)
        return (
            pref.cost_weight     * (ec                       / max_cost)
            + pref.latency_weight  * (bid.estimated_latency_ms / max_lat)
            + pref.accuracy_weight * (1.0 - bid.estimated_accuracy)
            + pref.energy_weight   * (bid.estimated_energy_j   / max_energy)
        )

    return sorted(candidates, key=score)
