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
                # Bid reliability per "domain:complexity" -- EMA of judge/bid ratio
                "accuracy_bid_reliability": {},
                # Global EMA of actual measured latency (ms)
                "avg_latency_ms": None,
                # Per-category EMA of latency -- more accurate for TTCA scoring
                # because latency varies significantly by domain:complexity.
                "avg_latency_ms_per_cat": {},
                # Per-category EMA of actual completion token count.
                # Reasoning models (deepseek-r1-*) generate 3-5x more tokens than
                # instruction models for the same prompt, so the static global
                # table underestimates their bid latency. This corrects that.
                "avg_output_tokens": {},
            }
        else:
            # Backfill new fields for models loaded from older JSON snapshots.
            rep = self._data[model_id]
            rep.setdefault("avg_latency_ms_per_cat", {})
            rep.setdefault("avg_output_tokens", {})
        return self._data[model_id]

    # -- Latency reliability --------------------------------------------------

    def record_latency(
        self, model_id: str, bid_latency_ms: int, actual_latency_ms: int
    ) -> None:
        rep = self._ensure(model_id)
        overrun = actual_latency_ms / max(bid_latency_ms, 1)
        ratio = 1.0 if overrun <= LATENCY_GRACE_RATIO else min(1.0 / overrun, 1.0)
        rep["latency_reliability"] = (
            (1 - LATENCY_EMA_ALPHA) * rep["latency_reliability"]
            + LATENCY_EMA_ALPHA * ratio
        )
        old = rep.get("avg_latency_ms")
        rep["avg_latency_ms"] = (
            float(actual_latency_ms) if old is None
            else (1 - LATENCY_EMA_ALPHA) * old + LATENCY_EMA_ALPHA * actual_latency_ms
        )
        rep["sample_count"] += 1
        self._save()

    def get_avg_latency_ms(self, model_id: str) -> float | None:
        """Return global EMA of actual measured latency, or None if no data yet."""
        return self._data.get(model_id, {}).get("avg_latency_ms")

    def record_latency_per_cat(
        self, model_id: str, domain: str, complexity: str, actual_latency_ms: int
    ) -> None:
        """Record per-category latency EMA (more accurate for TTCA than global EMA)."""
        rep = self._ensure(model_id)
        key = f"{domain}:{complexity}"
        old = rep["avg_latency_ms_per_cat"].get(key)
        rep["avg_latency_ms_per_cat"][key] = (
            float(actual_latency_ms) if old is None
            else (1 - LATENCY_EMA_ALPHA) * old + LATENCY_EMA_ALPHA * actual_latency_ms
        )
        self._save()

    def get_avg_latency_ms_per_cat(
        self, model_id: str, domain: str, complexity: str
    ) -> float | None:
        """Return per-category latency EMA, or None if no data yet for this category."""
        key = f"{domain}:{complexity}"
        return self._data.get(model_id, {}).get("avg_latency_ms_per_cat", {}).get(key)

    def record_output_tokens(
        self, model_id: str, domain: str, complexity: str, tokens: int
    ) -> None:
        """Record actual completion token count for a (model, domain:complexity) pair."""
        rep = self._ensure(model_id)
        key = f"{domain}:{complexity}"
        old = rep["avg_output_tokens"].get(key)
        rep["avg_output_tokens"][key] = (
            float(tokens) if old is None
            else (1 - LATENCY_EMA_ALPHA) * old + LATENCY_EMA_ALPHA * tokens
        )
        self._save()

    def get_avg_output_tokens(
        self, model_id: str, domain: str, complexity: str
    ) -> float | None:
        """Return observed avg output token count, or None if no data yet."""
        key = f"{domain}:{complexity}"
        return self._data.get(model_id, {}).get("avg_output_tokens", {}).get(key)

    def get_penalty_multiplier(self, model_id: str) -> float:
        rep = self._data.get(model_id, {})
        reliability = rep.get("latency_reliability", 1.0)
        return 1.0 / max(reliability, 0.1)

    def get_latency_reliability(self, model_id: str) -> float:
        return self._data.get(model_id, {}).get("latency_reliability", 1.0)

    # -- Accuracy priors -------------------------------------------------------

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

    def record_accuracy_bid(
        self,
        model_id: str,
        domain: str,
        complexity: str,
        bid_accuracy: float,
        judge_score: float,
    ) -> None:
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
        rep = self._data.get(model_id, {})
        key = f"{domain}:{complexity}"
        reliability = rep.get("accuracy_bid_reliability", {}).get(key, 1.0)
        return max(reliability, 0.1)

    def get_domain_floor(self, model_id: str, domain: str) -> float | None:
        rep = self._data.get(model_id, {})
        priors = rep.get("accuracy_priors", {})
        domain_scores = [v for k, v in priors.items() if k.startswith(f"{domain}:")]
        return min(domain_scores) if domain_scores else None

    def get_all(self) -> dict:
        return dict(self._data)
