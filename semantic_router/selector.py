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

    # Stage 1 — hard filter on latency SLO and accuracy floor
    candidates = [
        b for b in bids
        if (pref.max_latency_ms is None or b.estimated_latency_ms <= pref.max_latency_ms)
        and (pref.min_accuracy is None or b.estimated_accuracy >= pref.min_accuracy)
    ]
    if not candidates:
        # SLO/accuracy violated by all models — fall back to fastest available
        fastest = sorted(bids, key=lambda b: b.estimated_latency_ms)
        candidates = fastest[:1]
        log.warning(
            "SLO violated by all bids (slo=%s ms) — falling back to fastest: %s (%d ms)",
            pref.max_latency_ms, candidates[0].model_id, candidates[0].estimated_latency_ms,
        )

    # Stage 2 — 4D weighted score (lower = better)
    # bid.estimated_accuracy comes from adapter.accuracy_priors (leaderboard values)
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
            "cost_norm=%.3f lat_norm=%.3f energy_norm=%.3f → score=%.4f",
            bid.model_id, bid.estimated_accuracy, bid.estimated_cost_usd,
            bid.estimated_latency_ms, bid.estimated_energy_j,
            ec / max_cost, bid.estimated_latency_ms / max_lat,
            bid.estimated_energy_j / max_energy, s,
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

    Used by the inference retry loop — if the top model fails during
    dispatch, the caller tries the next-best model in this ranked list.
    Unlike select(), this never raises: an empty list is returned when
    there are no bids.

    TTCA mode uses latency/accuracy directly (not a weighted sum) because
    E[time to correct answer] = latency / accuracy. This correctly prefers
    fast models even at lower accuracy when retry is available:
      E[TTCA small-first] = p*L_s + (1-p)*(L_s+L_l)  (typically lower)
      E[TTCA large-first] = p*L_l + (1-p)*(L_l+L_s)  (typically higher)
    Optimal ordering: sort by L/p ascending.
    """
    if not bids:
        return []

    def effective_cost(bid: BidResponse) -> float:
        return bid.estimated_cost_usd * reputation.get_penalty_multiplier(bid.model_id)

    # Apply hard filters — SLO and accuracy floor
    candidates = [
        b for b in bids
        if (pref.max_latency_ms is None or b.estimated_latency_ms <= pref.max_latency_ms)
        and (pref.min_accuracy is None or b.estimated_accuracy >= pref.min_accuracy)
    ]
    if not candidates:
        # Fall back to all bids sorted by latency when SLO is violated
        candidates = sorted(bids, key=lambda b: b.estimated_latency_ms)

    # TTCA mode: sort by latency/accuracy (expected time to correct answer).
    # This is mathematically optimal when retry-on-wrong-answer is enabled.
    # A fast model with 50% accuracy (L/p=1000) beats a slow model with 95%
    # accuracy (L/p=2105) because trying fast-first reduces expected total time.
    #
    # Latency source (in priority order):
    #   1. Per-category EMA: actual latency for THIS domain:complexity (most accurate)
    #      — critical because reasoning models are 5x slower on math:hard vs factual:easy
    #   2. Global EMA: actual latency averaged over all categories (good general estimate)
    #   3. Bid estimate: output_tokens/throughput formula (cold start only)
    if pref.mode == RouterMode.TTCA:
        # Use domain-specific alpha if configured; fall back to global TTCA_ALPHA.
        # Lower alpha = accuracy-focused (factual:0.3), higher = speed-focused (code:1.0).
        alpha = _DOMAIN_ALPHA.get(domain, TTCA_ALPHA)

        # Normalise cost so the cheapest candidate = 1.0 and all others > 1.0.
        # Without this, USD costs < 1 (e.g. $0.0001) produce cost^beta < cost, which
        # shrinks the denominator and *boosts* expensive models — the opposite of
        # the intended penalty. Relative normalisation makes beta scale-invariant.
        min_cost_raw = min(max(b.estimated_cost_usd, 1e-9) for b in candidates)

        def ttca_score(bid: BidResponse) -> float:
            acc = max(bid.estimated_accuracy, 0.01)
            per_cat = reputation.get_avg_latency_ms_per_cat(bid.model_id, domain, complexity)
            global_  = reputation.get_avg_latency_ms(bid.model_id)
            lat = per_cat if per_cat is not None else (
                global_ if global_ is not None else bid.estimated_latency_ms
            )
            lat = max(lat, 1.0)   # guard against zero/negative
            cost_norm = max(bid.estimated_cost_usd, 1e-9) / min_cost_raw  # cheapest = 1.0
            # score = acc / (lat^alpha x cost_norm^beta)
            # Domain-specific alpha: factual=0.3 (accuracy), code=1.0 (classic TTCA)
            return acc / ((lat ** alpha) * (cost_norm ** TTCA_COST_BETA))

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
            cost_norm = max(bid.estimated_cost_usd, 1e-9) / min_cost_raw
            score = bid.estimated_accuracy / ((lat ** alpha) * (cost_norm ** TTCA_COST_BETA))
            log.info(
                "  TTCA %-22s lat=%dms(%s) acc=%.3f cost=%.6f cost_norm=%.3f α=%.1f(domain=%s) β=%.1f → score=%.4f",
                bid.model_id, lat, src,
                bid.estimated_accuracy, bid.estimated_cost_usd, cost_norm,
                alpha, domain, TTCA_COST_BETA, score,
            )
        return ranked

    # All other modes: 4D weighted score (lower = better)
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
