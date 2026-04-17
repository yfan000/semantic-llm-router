from __future__ import annotations
from dataclasses import dataclass
from semantic_router.adapters.base import ModelAdapter


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

    def get_eligible(self, domain: str, min_accuracy: float | None) -> list[ModelAdapter]:
        results = []
        for cfg in self._models.values():
            if domain not in cfg.domains and "general" not in cfg.domains:
                continue
            if min_accuracy is not None:
                # Use domain-specific floor, fall back to "_default", then 0.0
                capability = cfg.min_accuracy_capability.get(domain) \
                          or cfg.min_accuracy_capability.get("_default", 0.0)
                if capability < min_accuracy:
                    continue
            results.append(cfg.adapter)
        return results

    def list_all(self) -> list[str]:
        return list(self._models.keys())

    def get_adapter(self, model_id: str) -> ModelAdapter | None:
        cfg = self._models.get(model_id)
        return cfg.adapter if cfg else None
