from __future__ import annotations
import json
import os
import tempfile
from semantic_router.config import (
    LATENCY_EMA_ALPHA, LATENCY_GRACE_RATIO,
    ACCURACY_EMA_ALPHA, DEFAULT_ACCURACY_PRIOR,
    ACCURACY_BID_EMA_ALPHA, ACCURACY_BID_GRACE_RATIO,
    MODEL_REPUTATION_PATH,
)


class ReputationTracker:
    def __init__(self, path: str = MODEL_REPUTATION_PATH) -> None:
        self._path = path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path) as f:
                self._data = json.load(f)

    def _save(self) -> None:
        dir_ = os.path.dirname(self._path) or "."
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
            json.dump(self._data, f, indent=2)
            tmp = f.name
        os.replace(tmp, self._path)

    def _ensure(self, model_id: str) -> dict:
        if model_id not in self._data:
            self._data[model_id] = {
                "latency_reliability": 1.0,
                "sample_count": 0,
                "accuracy_priors": {},
                # EMA of judge/bid ratio per domain:complexity
                # 1.0 = honest bidder, <1.0 = consistently overbids
                "accuracy_bid_reliability": {},
            }
        return self._data[model_id]

    # -- Latency reliability -------------------------------------------------

    def record_latency(self, model_id: str, bid_latency_ms: int, actual_latency_ms: int) -> None:
        rep = self._ensure(model_id)
        overrun = actual_latency_ms / max(bid_latency_ms, 1)
        ratio = 1.0 if overrun <= LATENCY_GRACE_RATIO else min(1.0 / overrun, 1.0)
        rep["latency_reliability"] = (
            (1 - LATENCY_EMA_ALPHA) * rep["latency_reliability"]
            + LATENCY_EMA_ALPHA * ratio
        )
        rep["sample_count"] += 1
        self._save()

    def get_penalty_multiplier(self, model_id: str) -> float:
        """Cost inflation factor for latency overbidders. 1.0 = honest, >1 = penalised."""
        reliability = self._data.get(model_id, {}).get("latency_reliability", 1.0)
        return 1.0 / max(reliability, 0.1)

    def get_latency_reliability(self, model_id: str) -> float:
        return self._data.get(model_id, {}).get("latency_reliability", 1.0)

    # -- Accuracy priors (eligibility floor) ---------------------------------

    def get_accuracy_prior(self, model_id: str, domain: str, complexity: str) -> float:
        key = f"{domain}:{complexity}"
        rep = self._data.get(model_id, {})
        return rep.get("accuracy_priors", {}).get(key, DEFAULT_ACCURACY_PRIOR)

    def update_accuracy_prior(
        self, model_id: str, domain: str, complexity: str, judge_score: float
    ) -> None:
        rep = self._ensure(model_id)
        key = f"{domain}:{complexity}"
        old = rep["accuracy_priors"].get(key, DEFAULT_ACCURACY_PRIOR)
        rep["accuracy_priors"][key] = (
            (1 - ACCURACY_EMA_ALPHA) * old + ACCURACY_EMA_ALPHA * judge_score
        )
        self._save()

    # -- Accuracy bid reliability (scoring penalty) ---------------------------

    def record_accuracy_bid(
        self,
        model_id: str,
        domain: str,
        complexity: str,
        bid_accuracy: float,
        judge_score: float,
    ) -> None:
        """
        Track how well the model's accuracy claim matched the judge score.
        ratio = min(judge_score / bid_accuracy, 1.0)
          1.0  -> delivered what was promised (or better)
          <1.0 -> overbid; claimed higher accuracy than judged
        Grace ratio: discrepancies within 10% are treated as honest.
        """
        rep = self._ensure(model_id)
        key = f"{domain}:{complexity}"
        ratio = min(judge_score / max(bid_accuracy, 1e-6), 1.0)
        ratio = 1.0 if ratio >= ACCURACY_BID_GRACE_RATIO else ratio
        old = rep["accuracy_bid_reliability"].get(key, 1.0)
        rep["accuracy_bid_reliability"][key] = (
            (1 - ACCURACY_BID_EMA_ALPHA) * old + ACCURACY_BID_EMA_ALPHA * ratio
        )
        self._save()

    def get_accuracy_discount(self, model_id: str, domain: str, complexity: str) -> float:
        """
        Discount factor [0.1, 1.0] applied to bid.estimated_accuracy in the selector.
        Honest bidders get 1.0 (no discount).
        Chronic overbidders get <1.0, making their accuracy appear lower.
        """
        rep = self._data.get(model_id, {})
        key = f"{domain}:{complexity}"
        reliability = rep.get("accuracy_bid_reliability", {}).get(key, 1.0)
        return max(reliability, 0.1)  # cap at 90% discount

    # -- Domain floor (live from production data) ----------------------------

    def get_domain_floor(self, model_id: str, domain: str) -> float | None:
        """
        Live production floor: min accuracy prior across all complexity levels.
        Returns None until production data exists; caller falls back to calibration.
        """
        rep = self._data.get(model_id, {})
        priors = rep.get("accuracy_priors", {})
        domain_scores = [v for k, v in priors.items() if k.startswith(f"{domain}:")]
        return min(domain_scores) if domain_scores else None

    def get_all(self) -> dict:
        return dict(self._data)
