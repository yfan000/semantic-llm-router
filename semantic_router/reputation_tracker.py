from __future__ import annotations
import json, os, tempfile
from semantic_router.config import LATENCY_EMA_ALPHA, LATENCY_GRACE_RATIO, ACCURACY_EMA_ALPHA, DEFAULT_ACCURACY_PRIOR, MODEL_REPUTATION_PATH


class ReputationTracker:
    def __init__(self, path: str = MODEL_REPUTATION_PATH) -> None:
        self._path = path
        self._data: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as f:
                self._data = json.load(f)

    def _save(self) -> None:
        dir_ = os.path.dirname(self._path) or "."
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
            json.dump(self._data, f, indent=2)
        os.replace(f.name, self._path)

    def _ensure(self, model_id: str) -> dict:
        if model_id not in self._data:
            self._data[model_id] = {"latency_reliability": 1.0, "sample_count": 0, "accuracy_priors": {}}
        return self._data[model_id]

    def record_latency(self, model_id: str, bid_ms: int, actual_ms: int) -> None:
        rep = self._ensure(model_id)
        overrun = actual_ms / max(bid_ms, 1)
        ratio = 1.0 if overrun <= LATENCY_GRACE_RATIO else min(1.0 / overrun, 1.0)
        rep["latency_reliability"] = (1 - LATENCY_EMA_ALPHA) * rep["latency_reliability"] + LATENCY_EMA_ALPHA * ratio
        rep["sample_count"] += 1
        self._save()

    def get_penalty_multiplier(self, model_id: str) -> float:
        return 1.0 / max(self._data.get(model_id, {}).get("latency_reliability", 1.0), 0.1)

    def get_latency_reliability(self, model_id: str) -> float:
        return self._data.get(model_id, {}).get("latency_reliability", 1.0)

    def update_accuracy_prior(self, model_id: str, domain: str, complexity: str, score: float) -> None:
        rep = self._ensure(model_id)
        key = f"{domain}:{complexity}"
        old = rep["accuracy_priors"].get(key, DEFAULT_ACCURACY_PRIOR)
        rep["accuracy_priors"][key] = (1 - ACCURACY_EMA_ALPHA) * old + ACCURACY_EMA_ALPHA * score
        self._save()

    def get_all(self) -> dict:
        return dict(self._data)
