from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from semantic_router.adapters.base import ModelAdapter

if TYPE_CHECKING:
    from semantic_router.reputation_tracker import ReputationTracker


@dataclass
class ModelConfig:
    adapter: ModelAdapter
    domains: list[str]
    # Per-domain accuracy floors from calibration, e.g. {"math": 0.45, "factual": 0.82}
    # Special key "_default" used as fallback when domain is not listed
    min_accuracy_capability: dict[str, float]


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}

    def register(self, config: ModelConfig) -> None:
        self._models[config.adapter.model_id] = config

    def deregister(self, model_id: str) -> None:
        self._models.pop(model_id, None)

    def get_eligible(
        self,
        domain: str,
        min_accuracy: float | None,
        reputation: "ReputationTracker | None" = None,
    ) -> list[ModelAdapter]:
        results = []
        for cfg in self._models.values():
            if domain not in cfg.domains and "general" not in cfg.domains:
                continue
            if min_accuracy is not None:
                # Prefer live production floor derived from judge-scored priors.
                # Falls back to static calibration floor until data accumulates.
                live_floor = (
                    reputation.get_domain_floor(cfg.adapter.model_id, domain)
                    if reputation is not None else None
                )
                floor = live_floor if live_floor is not None \
                    else cfg.min_accuracy_capability.get(domain) \
                    or cfg.min_accuracy_capability.get("_default", 0.0)
                if floor < min_accuracy:
                    continue
            results.append(cfg.adapter)
        return results

    def list_all(self) -> list[str]:
        return list(self._models.keys())

    def get_adapter(self, model_id: str) -> ModelAdapter | None:
        cfg = self._models.get(model_id)
        return cfg.adapter if cfg else None
